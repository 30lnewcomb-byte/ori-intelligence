"""Ori's compact, native TensorFlow language model.

The language model provides learned language generation. Identity, memory,
tools, permissions, and orchestration stay outside the model in Ori Core.

Tokenizer note:
    Ori uses a SentencePiece BPE tokenizer trained from the same English +
    Ori corpus used by the language model. This replaces the old word-level
    tokenizer so unfamiliar words can be represented as subword pieces rather
    than collapsing immediately to <unk>.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path

import sentencepiece as spm
import tensorflow as tf


@dataclass(frozen=True)
class OriLMConfig:
    vocab_size: int = 512
    context_length: int = 256
    d_model: int = 192
    num_heads: int = 4
    num_layers: int = 4
    d_ff: int = 768
    dropout: float = 0.1
    pad_id: int = 0
    bos_id: int = 1
    eos_id: int = 2
    unk_id: int = 3


class OriTokenizer:
    """SentencePiece BPE tokenizer used by Ori's language model."""

    def __init__(self, processor: spm.SentencePieceProcessor):
        self.processor = processor

    @property
    def vocab_size(self) -> int:
        return int(self.processor.get_piece_size())

    @property
    def vocab(self) -> dict[str, int]:
        return {
            "<pad>": 0,
            "<bos>": 1,
            "<eos>": 2,
            "<unk>": 3,
        }

    def encode(
        self,
        text: str,
        add_bos: bool = True,
        add_eos: bool = True,
    ) -> list[int]:
        ids = list(self.processor.encode(text, out_type=int))
        if add_bos:
            ids.insert(0, self.processor.bos_id())
        if add_eos:
            ids.append(self.processor.eos_id())
        return ids

    def decode(self, ids: list[int]) -> str:
        return self.processor.decode(ids).strip()

    @classmethod
    def load(cls, path: str | Path) -> "OriTokenizer":
        processor = spm.SentencePieceProcessor(model_file=str(path))
        if processor.pad_id() != 0 or processor.bos_id() != 1 or processor.eos_id() != 2:
            raise ValueError("Ori tokenizer special-token IDs are incompatible with the model.")
        return cls(processor)


class TransformerBlock(tf.keras.layers.Layer):
    def __init__(self, config: OriLMConfig, **kwargs):
        super().__init__(**kwargs)
        self.norm1 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.attn = tf.keras.layers.MultiHeadAttention(
            num_heads=config.num_heads,
            key_dim=config.d_model // config.num_heads,
            dropout=config.dropout,
        )
        self.norm2 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.ffn = tf.keras.Sequential(
            [
                tf.keras.layers.Dense(config.d_ff, activation=tf.nn.gelu),
                tf.keras.layers.Dropout(config.dropout),
                tf.keras.layers.Dense(config.d_model),
            ],
            name="ffn",
        )
        self.dropout = tf.keras.layers.Dropout(config.dropout)

    def call(self, x, training=False):
        length = tf.shape(x)[1]
        causal_mask = tf.linalg.band_part(
            tf.ones((length, length), dtype=tf.bool), -1, 0
        )
        normalized = self.norm1(x)
        attention = self.attn(
            normalized,
            normalized,
            attention_mask=causal_mask,
            training=training,
        )
        x = x + self.dropout(attention, training=training)
        x = x + self.dropout(
            self.ffn(self.norm2(x), training=training),
            training=training,
        )
        return x


class OriLanguageModel(tf.keras.Model):
    """Compact decoder-only Transformer used by Ori's language path."""

    def __init__(self, config: OriLMConfig, **kwargs):
        super().__init__(**kwargs)
        self.config = config
        self.tokens = tf.keras.layers.Embedding(
            config.vocab_size, config.d_model, name="token_embedding"
        )
        self.positions = tf.keras.layers.Embedding(
            config.context_length, config.d_model, name="position_embedding"
        )
        self.dropout = tf.keras.layers.Dropout(config.dropout)
        self.blocks = [
            TransformerBlock(config, name=f"transformer_{i}")
            for i in range(config.num_layers)
        ]
        self.norm = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        # The output projection reuses the token embedding matrix. This
        # weight tying gives a compact model more effective parameters without
        # increasing the vocabulary projection size independently.
        self.lm_head = None

    def call(self, token_ids, training=False):
        length = tf.shape(token_ids)[1]
        positions = tf.range(length)[tf.newaxis, :]
        x = self.tokens(token_ids) + self.positions(positions)
        x = self.dropout(x, training=training)
        for block in self.blocks:
            x = block(x, training=training)
        x = self.norm(x)
        return tf.linalg.matmul(x, self.tokens.embeddings, transpose_b=True)

    def next_logits(self, token_ids):
        return self(token_ids, training=False)[:, -1, :]

    def generate(
        self,
        tokenizer: OriTokenizer,
        prompt: str,
        max_new_tokens: int = 96,
        temperature: float = 0.2,
        top_k: int = 20,
    ) -> tuple[str, str]:
        """Generate text from learned weights and return (text, finish_reason)."""
        if not prompt.strip():
            raise ValueError("Generation prompt is empty.")
        if max_new_tokens < 1 or max_new_tokens > 256:
            raise ValueError("max_new_tokens must be between 1 and 256.")
        if temperature < 0 or temperature > 2:
            raise ValueError("temperature must be between 0 and 2.")

        context = tokenizer.encode(prompt, add_bos=True, add_eos=False)
        context = context[-self.config.context_length :]
        generated: list[int] = []

        # A compact model benefits from a small repetition penalty during
        # sampling. This changes token probabilities rather than choosing a
        # scripted answer, helping it avoid getting trapped in short loops.
        repetition_penalty = 1.08
        min_generated_tokens = min(2, max_new_tokens)

        for step in range(max_new_tokens):
            logits = self.next_logits(
                tf.constant([context], dtype=tf.int32)
            )[0]

            recent_ids = set(generated[-24:])
            if recent_ids:
                recent_tensor = tf.constant(sorted(recent_ids), dtype=tf.int32)
                recent_logits = tf.gather(logits, recent_tensor)
                recent_logits = tf.where(
                    recent_logits > 0,
                    recent_logits / repetition_penalty,
                    recent_logits * repetition_penalty,
                )
                logits = tf.tensor_scatter_nd_update(
                    logits,
                    recent_tensor[:, tf.newaxis],
                    recent_logits,
                )

            blocked_ids = [
                tokenizer.vocab["<pad>"],
                tokenizer.vocab["<bos>"],
                tokenizer.vocab["<unk>"],
            ]
            if step < min_generated_tokens:
                blocked_ids.append(tokenizer.vocab["<eos>"])

            blocked = tf.reduce_any(
                tf.one_hot(
                    blocked_ids,
                    depth=self.config.vocab_size,
                    dtype=tf.bool,
                    on_value=True,
                    off_value=False,
                ),
                axis=0,
            )
            logits = tf.where(
                blocked,
                tf.fill(tf.shape(logits), tf.constant(-1e9)),
                logits,
            )

            if temperature <= 1e-6:
                next_id = int(tf.argmax(logits).numpy())
            else:
                scaled = logits / temperature
                if top_k > 0:
                    k = min(top_k, self.config.vocab_size)
                    values, indices = tf.math.top_k(scaled, k=k)
                    sampled = tf.random.categorical(
                        values[tf.newaxis, :], 1
                    )[0, 0]
                    next_id = int(indices[sampled].numpy())
                else:
                    sampled = tf.random.categorical(
                        scaled[tf.newaxis, :], 1
                    )[0, 0]
                    next_id = int(sampled.numpy())

            if next_id == tokenizer.vocab["<eos>"]:
                return tokenizer.decode(generated), "stop"

            generated.append(next_id)
            context.append(next_id)
            context = context[-self.config.context_length :]

        return tokenizer.decode(generated), "length"


def build_model(config: OriLMConfig | None = None) -> OriLanguageModel:
    config = config or OriLMConfig()
    model = OriLanguageModel(config, name="ori_language_model")
    model(tf.zeros((1, min(2, config.context_length)), dtype=tf.int32))
    return model

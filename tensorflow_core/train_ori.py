"""Train Ori's native TensorFlow language model in two learned stages.

Stage 1: pretrain on a bounded public-domain English corpus.
Stage 2: fine-tune those same weights on Ori-specific conversation/tool data.

This keeps the model's language ability broader than a tiny set of canned Ori
responses while still giving it an Ori-specific behavior layer.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import tempfile

import numpy as np
import sentencepiece as spm
import tensorflow as tf

from english_corpus import download_public_domain_english
from ori_model import OriLMConfig, OriLanguageModel, OriTokenizer
from train_specialists import coding_examples, cad_examples, train_worker


class MaskedCausalLoss(tf.keras.losses.Loss):
    def __init__(self, pad_id: int):
        super().__init__()
        self.pad_id = pad_id

    def call(self, y_true, y_pred):
        loss = tf.keras.losses.sparse_categorical_crossentropy(
            y_true, y_pred, from_logits=True
        )
        mask = tf.cast(tf.not_equal(y_true, self.pad_id), loss.dtype)
        return tf.reduce_sum(loss * mask) / tf.maximum(tf.reduce_sum(mask), 1.0)


def load_records(path: Path) -> list[str]:
    texts: list[str] = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(f"Invalid JSON on line {line_number}") from exc
        if "text" in row:
            texts.append(str(row["text"]))
        elif "input" in row and "response" in row:
            texts.append(f"User: {row['input']} Ori: {row['response']}")
        else:
            raise ValueError(f"Line {line_number} needs text or input+response")
    if not texts:
        raise ValueError("Training dataset is empty")
    return texts


def curated_conversation_examples() -> list[str]:
    """Add varied, natural one-turn dialogue to the learned fine-tuning set.

    These are training examples, not runtime scripts. They give the small model
    more conversational coverage without hard-coding any response in inference.
    """
    pairs = [
        ("hi Ori", ["Hey! What's up?", "Hi! What are you working on?", "Hey! What can we work on?"]),
        ("Hi, Ori", ["Hey! What's up?", "Hi! What can I help with?", "Hey! What are you thinking about?"]),
        ("hello Ori", ["Hey! What are you working on?", "Hi! What can I help with?", "Hey! What's on your mind?"]),
        ("Hello there", ["Hi! How can I help?", "Hey! What are you working on today?", "Hello! What can we figure out?"]),
        ("hey Ori", ["Hey! What's up?", "Hey! What are you working on?", "Hi! What should we tackle?"]),
        ("hey there", ["Hey! What can we work on?", "Hi! What's up?", "Hey! How can I help?"]),
        ("good morning Ori", ["Good morning! What are we working on?", "Good morning! What's on your mind?", "Good morning! What should we tackle?"]),
        ("good afternoon Ori", ["Good afternoon! What are you working on?", "Good afternoon! What can we figure out?", "Good afternoon! How can I help?"]),
        ("good evening Ori", ["Good evening! What can we work on?", "Good evening! What's on your mind?", "Good evening! What should we tackle?"]),
        ("are you there Ori", ["Yep. I'm here.", "Yes, I'm here.", "I'm here. What's up?"]),
        ("are you listening Ori", ["Yes. I'm following the conversation.", "Yep. I'm listening.", "Yes. Go ahead."]),
        ("how are you Ori", ["I'm doing well and ready to help.", "I'm doing well. What's up?", "Doing well. What are we working on?"]),
        ("what are you doing", ["I'm here and ready to help.", "I'm ready for whatever we're working on.", "I'm here. What should we tackle?"]),
        ("what are you up to", ["I'm here, ready to work with you.", "I'm ready to help with whatever you're working on.", "I'm here. What do you have in mind?"]),
        ("what's up", ["Not much. What are you working on?", "I'm here. What's on your mind?", "Hey! What should we work on?"]),
        ("I have a question", ["Go ahead.", "Sure. What's your question?", "Absolutely. Ask away."]),
        ("I need help", ["Sure. Tell me what you're trying to do.", "Absolutely. What do you need help with?", "Okay. Tell me where you're stuck."]),
        ("can you help me", ["Yes. Tell me what you need.", "Sure. What are you working on?", "Absolutely. What's the problem?"]),
        ("I need some help", ["Sure. Tell me what you're trying to do.", "Okay. What are you working on?", "Absolutely. Tell me what's going wrong."]),
        ("I'm stuck", ["That's okay. Tell me where you got stuck.", "No problem. Let's narrow it down.", "That's fine. What part is giving you trouble?"]),
        ("I don't understand", ["No problem. Tell me which part is confusing.", "That's okay. I can explain it another way.", "Sure. Which part should I explain?"]),
        ("can you explain that", ["Absolutely. I'll break it down.", "Sure. Let's go through it step by step.", "Yes. I'll explain it clearly."]),
        ("say that another way", ["Sure. I'll put it more simply.", "Absolutely. Here's another way to look at it.", "Yep. I'll explain it differently."]),
        ("what should we do next", ["Let's look at the goal and choose the next useful step.", "We can figure out the next step from what you've already done.", "Let's see what remains and pick the next step."]),
        ("what next", ["Tell me what you want to accomplish next.", "Let's look at what remains.", "We can take the next useful step from here."]),
        ("thanks Ori", ["You're welcome.", "You're welcome! Let's keep going.", "Anytime."]),
        ("thank you Ori", ["You're welcome!", "Glad I could help.", "You're welcome."]),
        ("nice job Ori", ["Thanks!", "Thanks! Let's keep going.", "Appreciate it."]),
        ("that helped", ["Glad it helped.", "Good to hear.", "Nice. Let's keep going."]),
        ("let's keep going", ["Sounds good.", "Absolutely. What's next?", "Let's do it."]),
        ("can we try again", ["Yes. We can try a different approach.", "Absolutely. Let's try again.", "Sure. Let's take another look."]),
        ("what happened", ["Let's look at what changed.", "We can trace what happened step by step.", "Let's check the evidence first."]),
        ("is this right", ["We can check it carefully.", "Let's verify it.", "I'll help you check it."]),
        ("can we make this better", ["Yes. Let's look for the changes with the biggest impact.", "Absolutely. Show me the current version.", "Yes. We can improve it without adding unnecessary complexity."]),
        ("can we improve the design", ["Yes. We can improve hierarchy, spacing, and consistency.", "Absolutely. Let's look at the current design.", "Yes. We can make it clearer and more polished."]),
        ("make this more professional", ["Sure. We can improve the hierarchy, wording, and details.", "Absolutely. Let's make it cleaner and more consistent.", "Yes. Show me what you have now."]),
        ("help me think this through", ["Sure. Tell me what you're considering.", "Absolutely. Let's break down the options.", "Okay. What's the part you're unsure about?"]),
        ("what do you think", ["Tell me what you're considering and I'll give you a reasoned take.", "I can help compare the options.", "Give me the details and I'll think it through with you."]),
        ("which one is better", ["Tell me the options and what matters most.", "I can compare the tradeoffs.", "Give me both choices and I'll help weigh them."]),
        ("is there an easier way", ["Probably. Show me the current approach and we'll simplify it.", "We can look for a version with fewer moving parts.", "Let's compare the simpler options."]),
    ]
    examples: list[str] = []
    for user_text, replies in pairs:
        for reply in replies:
            examples.append(f"User: {user_text} Ori: {reply}")
    return examples


def dedupe_texts(texts: list[str]) -> list[str]:
    seen: set[str] = set()
    unique: list[str] = []
    for text in texts:
        normalized = " ".join(text.split()).strip()
        key = normalized.casefold()
        if not normalized or key in seen:
            continue
        seen.add(key)
        unique.append(normalized)
    return unique


def train_tokenizer(
    texts: list[str],
    output: Path,
    vocab_size: int,
) -> OriTokenizer:
    corpus_path = output / "tokenizer-training.txt"
    corpus_path.write_text("\n".join(texts), encoding="utf-8")

    prefix = output / "ori_tokenizer"
    spm.SentencePieceTrainer.train(
        input=str(corpus_path),
        model_prefix=str(prefix),
        vocab_size=vocab_size,
        model_type="bpe",
        character_coverage=1.0,
        pad_id=0,
        bos_id=1,
        eos_id=2,
        unk_id=3,
        pad_piece="<pad>",
        bos_piece="<bos>",
        eos_piece="<eos>",
        unk_piece="<unk>",
        normalization_rule_name="nmt_nfkc",
        add_dummy_prefix=True,
        remove_extra_whitespaces=True,
        max_sentence_length=12000,
        shuffle_input_sentence=True,
        hard_vocab_limit=False,
        minloglevel=1,
    )

    model_path = output / "ori_tokenizer.model"
    if not model_path.exists():
        raise RuntimeError("SentencePiece tokenizer training did not produce a model.")

    return OriTokenizer.load(model_path)


def make_document_arrays(
    texts: list[str],
    tokenizer: OriTokenizer,
    context: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Pack normal English documents into contiguous causal-LM windows."""

    stream: list[int] = []
    for text in texts:
        stream.extend(tokenizer.encode(text, add_bos=False, add_eos=True))

    xs: list[list[int]] = []
    ys: list[list[int]] = []
    # Overlapping windows give each token multiple neighboring contexts.
    stride = max(64, context // 2)
    for start in range(0, max(0, len(stream) - 1), stride):
        chunk = stream[start : start + context + 1]
        if len(chunk) < 3:
            continue
        x = chunk[:-1]
        y = chunk[1:]
        if len(x) < context:
            pad = tokenizer.vocab["<pad>"]
            x = x + [pad] * (context - len(x))
            y = y + [pad] * (context - len(y))
        xs.append(x[:context])
        ys.append(y[:context])

    if not xs:
        raise ValueError("No usable English sequences were produced")
    return np.asarray(xs, dtype=np.int32), np.asarray(ys, dtype=np.int32)


def make_finetune_arrays(
    texts: list[str],
    tokenizer: OriTokenizer,
    context: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Create response-focused causal-LM examples for Ori fine-tuning.

    The user prompt supplies context, but only Ori's response tokens (plus
    EOS) contribute to the fine-tuning loss. This keeps the pretrained English
    model from spending its limited fine-tuning capacity memorizing prompts.
    """
    xs: list[list[int]] = []
    ys: list[list[int]] = []
    pad = tokenizer.vocab["<pad>"]

    for text in texts:
        marker = " Ori:"
        split_at = text.find(marker)
        if split_at < 0:
            continue
        prefix = text[: split_at + len(marker)]
        response = text[split_at + len(marker) :].strip()
        if not response:
            continue

        prefix_ids = tokenizer.encode(prefix, add_bos=True, add_eos=False)
        response_ids = tokenizer.encode(response, add_bos=False, add_eos=True)
        ids = (prefix_ids + response_ids)[: context + 1]
        if len(ids) < 3:
            continue

        x = ids[:-1]
        y = ids[1:]
        response_start = len(prefix_ids)
        for index in range(len(y)):
            # y[index] corresponds to ids[index + 1].
            if index + 1 < response_start:
                y[index] = pad

        x += [pad] * (context - len(x))
        y += [pad] * (context - len(y))
        xs.append(x[:context])
        ys.append(y[:context])

    if not xs:
        raise ValueError("No usable Ori fine-tuning sequences were produced")
    return np.asarray(xs, dtype=np.int32), np.asarray(ys, dtype=np.int32)
def compile_for_learning(
    model: OriLanguageModel,
    pad_id: int,
    learning_rate: float,
) -> None:
    model.compile(
        optimizer=tf.keras.optimizers.AdamW(
            learning_rate=learning_rate,
            weight_decay=1e-4,
            clipnorm=1.0,
        ),
        loss=MaskedCausalLoss(pad_id),
    )


def fit_stage(
    model: OriLanguageModel,
    x: np.ndarray,
    y: np.ndarray,
    *,
    epochs: int,
    batch_size: int,
    validation_split: float,
    checkpoint_dir: Path,
    prefix: str,
) -> dict[str, object]:
    callbacks: list[tf.keras.callbacks.Callback] = [
        tf.keras.callbacks.ModelCheckpoint(
            filepath=str(checkpoint_dir / f"{prefix}-epoch-{{epoch:02d}}.weights.h5"),
            save_weights_only=True,
            save_best_only=False,
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="loss",
            factor=0.5,
            patience=3,
            min_lr=1e-5,
            verbose=1,
        ),
    ]

    fit_kwargs: dict[str, object] = {
        "batch_size": batch_size,
        "epochs": epochs,
        "shuffle": True,
        "callbacks": callbacks,
    }

    has_validation = len(x) >= 10 and validation_split > 0
    if has_validation:
        fit_kwargs["validation_split"] = validation_split
        callbacks.append(
            tf.keras.callbacks.EarlyStopping(
                monitor="val_loss",
                patience=5,
                restore_best_weights=True,
            )
        )

    history = model.fit(x, y, **fit_kwargs)
    return {
        "epochs_completed": len(history.history["loss"]),
        "final_loss": float(history.history["loss"][-1]),
        "final_val_loss": (
            float(history.history["val_loss"][-1])
            if "val_loss" in history.history
            else None
        ),
        "validation_enabled": has_validation,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="data/ori_training_expanded.jsonl")
    parser.add_argument("--output", default="artifacts/ori-small")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--english-epochs", type=int, default=16)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--validation-split", type=float, default=0.0)
    parser.add_argument("--vocab-size", type=int, default=1024)
    parser.add_argument("--english-max-chars", type=int, default=1_000_000)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    if args.epochs < 1 or args.english_epochs < 1:
        raise ValueError("Both --epochs and --english-epochs must be at least 1")
    if not 0 <= args.validation_split < 1:
        raise ValueError("--validation-split must be between 0 and 1")

    tf.keras.utils.set_random_seed(args.seed)

    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    checkpoint_dir = output / "checkpoint"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    curated_examples = curated_conversation_examples()
    ori_texts = dedupe_texts(
        load_records(Path(args.data)) + curated_examples
    )
    english_texts = download_public_domain_english(
        max_chars=args.english_max_chars
    )

    # The tokenizer sees both distributions, so English pretraining and Ori
    # fine-tuning use exactly the same token IDs.
    tokenizer = train_tokenizer(
        english_texts + ori_texts,
        output,
        vocab_size=args.vocab_size,
    )
    tokenizer_path = output / "ori_tokenizer.model"
    if not tokenizer_path.exists():
        raise RuntimeError("Final tokenizer artifact is missing.")

    config = OriLMConfig(
        vocab_size=tokenizer.vocab_size,
        context_length=256,
        d_model=192,
        num_heads=4,
        num_layers=4,
        d_ff=768,
        dropout=0.1,
    )
    model = OriLanguageModel(config, name="ori_language_model")
    model(tf.zeros((1, config.context_length), dtype=tf.int32))

    english_x, english_y = make_document_arrays(
        english_texts,
        tokenizer,
        config.context_length,
    )
    ori_x, ori_y = make_finetune_arrays(
        ori_texts,
        tokenizer,
        config.context_length,
    )

    # Stage 1: teach the network general English syntax, spelling, punctuation,
    # and local language patterns before adding Ori-specific behavior.
    print(
        f"Stage 1/2: pretraining on {len(english_x)} English windows "
        f"for {args.english_epochs} epochs."
    )
    compile_for_learning(model, config.pad_id, learning_rate=3e-4)
    pretraining = fit_stage(
        model,
        english_x,
        english_y,
        epochs=args.english_epochs,
        batch_size=args.batch_size,
        validation_split=0.0,
        checkpoint_dir=checkpoint_dir,
        prefix="english",
    )

    model.save_weights(output / "english-pretrained.weights.h5")

    # Keep a small English replay set during fine-tuning so Ori-specific data
    # does not immediately erase the language foundation.
    replay_count = min(max(64, len(ori_x) * 3), len(english_x))
    replay_x = english_x[:replay_count]
    replay_y = english_y[:replay_count]
    finetune_x = np.concatenate([ori_x, replay_x], axis=0)
    finetune_y = np.concatenate([ori_y, replay_y], axis=0)

    print(
        f"Stage 2/2: fine-tuning on {len(ori_x)} Ori examples "
        f"+ {replay_count} English replay windows for {args.epochs} epochs."
    )
    compile_for_learning(model, config.pad_id, learning_rate=5e-5)
    finetuning = fit_stage(
        model,
        finetune_x,
        finetune_y,
        epochs=args.epochs,
        batch_size=args.batch_size,
        validation_split=args.validation_split,
        checkpoint_dir=checkpoint_dir,
        prefix="ori",
    )

    model.save_weights(output / "model.weights.h5")
    (output / "config.json").write_text(
        json.dumps(config.__dict__, indent=2),
        encoding="utf-8",
    )
    (output / "training_summary.json").write_text(
        json.dumps(
            {
                "model": "ori-small",
                "training_strategy": "english-pretraining-then-ori-finetuning",
                "english_sources": "Project Gutenberg public-domain texts",
                "english_windows": len(english_x),
                "ori_examples": len(ori_texts),
                "curated_conversation_examples": len(curated_examples),
                "vocabulary_size": tokenizer.vocab_size,
                "english_epochs_requested": args.english_epochs,
                "ori_epochs_requested": args.epochs,
                "pretraining": pretraining,
                "finetuning": finetuning,
                "replay_windows": replay_count,
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    # Remove the temporary SentencePiece training input from the deployable
    # artifact directory; only the tokenizer model is needed at runtime.
    try:
        (output / "tokenizer-training.txt").unlink()
        (output / "ori_tokenizer.vocab").unlink()
    except FileNotFoundError:
        pass

    # Emit a tiny build-time smoke sample. This is diagnostic only; inference
    # still happens through the runtime after the service starts.
    sample, _ = model.generate(
        tokenizer,
        "User: Hi Ori! Ori:",
        max_new_tokens=24,
        temperature=0.20,
        top_k=8,
    )
    print(f"Greeting smoke sample: {sample!r}")

    # The existing Render service invokes this script directly. Train the
    # internal coding and 3D specialists here so no dashboard build-command
    # change is required.
    print("Training internal specialist workers...")
    train_worker(
        "ori-coder",
        coding_examples(),
        tokenizer,
        Path("artifacts") / "ori-coder",
        epochs=10,
        d_model=128,
        num_layers=3,
        d_ff=512,
        learning_rate=8e-5,
    )
    train_worker(
        "ori-3d",
        cad_examples(),
        tokenizer,
        Path("artifacts") / "ori-3d",
        epochs=10,
        d_model=128,
        num_layers=3,
        d_ff=512,
        learning_rate=8e-5,
    )
    print(f"Ori model saved to {output}")


if __name__ == "__main__":
    main()

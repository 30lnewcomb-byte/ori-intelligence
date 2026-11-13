"""Generate a response from a trained Ori checkpoint."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import tensorflow as tf

from ori_model import OriLMConfig, OriLanguageModel, OriTokenizer


def load_model(artifact: Path) -> tuple[OriLanguageModel, OriTokenizer]:
    tokenizer = OriTokenizer.load(artifact / "vocab.json")
    config = OriLMConfig(**json.loads((artifact / "config.json").read_text(encoding="utf-8")))
    model = OriLanguageModel(config, name="ori_language_model")
    model(tf.zeros((1, 2), dtype=tf.int32))
    model.load_weights(artifact / "model.weights.h5")
    return model, tokenizer


def generate(model, tokenizer, prompt: str, max_new_tokens: int = 64) -> str:
    prefix = tokenizer.encode(prompt, add_bos=True, add_eos=False)
    ids = list(prefix)
    eos_id = tokenizer.vocab["<eos>"]
    for _ in range(max_new_tokens):
        context = ids[-model.config.context_length:]
        logits = model.next_logits(tf.constant([context], dtype=tf.int32))[0]
        token_id = int(tf.argmax(logits).numpy())
        ids.append(token_id)
        if token_id == eos_id:
            break
    return tokenizer.decode(ids[len(prefix):]).strip()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("prompt")
    parser.add_argument("--artifact", default="artifacts/ori-small")
    parser.add_argument("--max-new-tokens", type=int, default=64)
    args = parser.parse_args()
    model, tokenizer = load_model(Path(args.artifact))
    print(generate(model, tokenizer, args.prompt, args.max_new_tokens))


if __name__ == "__main__":
    main()

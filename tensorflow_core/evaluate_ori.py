"""Evaluate an Ori checkpoint against a held-out behavior set."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import tensorflow as tf

from ori_model import OriLMConfig, OriLanguageModel, OriTokenizer


def load_eval(path: Path) -> list[dict]:
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            rows.append(json.loads(line))
    if not rows:
        raise ValueError("Evaluation dataset is empty")
    return rows


def score_completion(model, tokenizer, prompt: str, expected: str) -> float:
    prefix = tokenizer.encode(prompt, add_bos=True, add_eos=False)
    target = tokenizer.encode(expected, add_bos=False, add_eos=True)
    ids = prefix[-model.config.context_length :]
    correct = 0
    total = 0

    for token in target:
        logits = model.next_logits(
            tf.constant([ids], dtype=tf.int32)
        )[0]
        prediction = int(tf.argmax(logits).numpy())
        correct += int(prediction == token)
        total += 1
        ids.append(token)
        ids = ids[-model.config.context_length :]

    return correct / total if total else 0.0


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="data/ori_eval.jsonl")
    parser.add_argument("--artifact", default="artifacts/ori-small")
    parser.add_argument(
        "--report",
        default="artifacts/evaluation/ori_eval_report.json",
    )
    args = parser.parse_args()

    artifact = Path(args.artifact)
    tokenizer = OriTokenizer.load(artifact / "vocab.json")
    config = OriLMConfig(
        **json.loads((artifact / "config.json").read_text(encoding="utf-8"))
    )
    model = OriLanguageModel(config, name="ori_language_model")
    model(tf.zeros((1, 2), dtype=tf.int32))
    model.load_weights(artifact / "model.weights.h5")

    rows = load_eval(Path(args.data))
    details = []
    for row in rows:
        score = score_completion(
            model,
            tokenizer,
            row["prompt"],
            row["expected"],
        )
        minimum = float(row.get("minimum_score", 0.0))
        details.append(
            {
                "prompt": row["prompt"],
                "score": float(score),
                "minimum_score": minimum,
                "passed": score >= minimum,
            }
        )

    scores = [item["score"] for item in details]
    passed = sum(item["passed"] for item in details)
    result = {
        "model": "ori-small",
        "examples": len(details),
        "mean_token_accuracy": float(np.mean(scores)),
        "median_token_accuracy": float(np.median(scores)),
        "passed": passed,
        "pass_rate": passed / len(details),
        "details": details,
    }

    report = Path(args.report)
    report.parent.mkdir(parents=True, exist_ok=True)
    report.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps({k: v for k, v in result.items() if k != "details"}, indent=2))


if __name__ == "__main__":
    main()

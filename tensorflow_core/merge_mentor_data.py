"""Merge Mentor corrections into training data with provenance and deduplication."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


def load_jsonl(path: Path) -> list[dict]:
    rows = []
    if not path.exists():
        return rows
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            rows.append(json.loads(line))
    return rows


def fingerprint(prompt: str, answer: str) -> str:
    value = f"{prompt}\n{answer}".encode("utf-8")
    return hashlib.sha256(value).hexdigest()[:16]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", default="data/ori_training.jsonl")
    parser.add_argument("--mentor", default="artifacts/mentor/mentor_candidates.jsonl")
    parser.add_argument("--output", default="data/ori_training_mentor_merged.jsonl")
    parser.add_argument("--min-rationale", action="store_true")
    args = parser.parse_args()

    base = load_jsonl(Path(args.base))
    mentor = load_jsonl(Path(args.mentor))
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)

    seen: set[str] = set()
    merged: list[dict] = []
    for row in base:
        prompt = str(row.get("input", row.get("prompt", ""))).strip()
        answer = str(row.get("response", row.get("answer", row.get("text", "")))).strip()
        if not prompt or not answer:
            continue
        key = fingerprint(prompt, answer)
        if key in seen:
            continue
        seen.add(key)
        merged.append({"input": prompt, "response": answer, "source": "base", "fingerprint": key})

    mentor_added = 0
    for row in mentor:
        prompt = str(row.get("prompt", "")).strip()
        answer = str(row.get("answer", "")).strip()
        rationale = str(row.get("mentor_rationale", "")).strip()
        if not prompt or not answer or (args.min_rationale and not rationale):
            continue
        key = fingerprint(prompt, answer)
        if key in seen:
            continue
        seen.add(key)
        merged.append({
            "input": prompt,
            "response": answer,
            "source": "qwen-mentor",
            "mentor_rationale": rationale,
            "fingerprint": key,
        })
        mentor_added += 1

    output.write_text(
        "\n".join(json.dumps(row, ensure_ascii=False) for row in merged) + "\n",
        encoding="utf-8",
    )
    summary = {
        "base_examples": len(base),
        "mentor_candidates": len(mentor),
        "mentor_added": mentor_added,
        "merged_examples": len(merged),
        "output": str(output),
    }
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()

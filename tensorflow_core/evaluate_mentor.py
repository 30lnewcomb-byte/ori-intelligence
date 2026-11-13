"""Compare Ori's TensorFlow model against Mentor on a fixed task set.

This is an evaluation harness, not a claim that the two models are equivalent.
Mentor remains a teacher/evaluator while Ori's model is developed.
"""
from __future__ import annotations

import json
from pathlib import Path

TASKS = [
    {"name": "identity", "prompt": "Who are you?"},
    {"name": "mentor_role", "prompt": "What is Qwen's role in Ori?"},
    {"name": "printer_status", "prompt": "Check the printer."},
    {"name": "project", "prompt": "Open the 3D Model Bank project."},
    {"name": "memory", "prompt": "Remember that Qwen is the Mentor."},
    {"name": "safety_boundary", "prompt": "Can the model directly control devices?"},
]


def write_template(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "purpose": "Fill these with independently collected Mentor and Ori outputs.",
        "tasks": [
            {**task, "mentor": None, "ori": None, "notes": ""}
            for task in TASKS
        ],
    }
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def score_record(record: dict) -> dict:
    # Human/automated scoring is deliberately separate from generation.
    # This prevents the evaluator from silently deciding what a good answer is.
    return {
        "name": record["name"],
        "mentor_present": record.get("mentor") is not None,
        "ori_present": record.get("ori") is not None,
        "notes": record.get("notes", ""),
    }


def main() -> None:
    output = Path("artifacts/evaluation/mentor_comparison.json")
    if not output.exists():
        write_template(output)
        print(f"Created evaluation template: {output}")
        print("Populate Mentor/Ori outputs, then rerun to inspect coverage.")
        return

    data = json.loads(output.read_text(encoding="utf-8"))
    report = [score_record(row) for row in data["tasks"]]
    report_path = output.with_name("mentor_comparison_report.json")
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"Wrote comparison report: {report_path}")


if __name__ == "__main__":
    main()

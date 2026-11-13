"""Train Ori's compact native TensorFlow intent classifier."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import tensorflow as tf

from model import LABELS, OriCoreModel


# Small but deliberately diverse bootstrap corpus.
TRAINING_DATA = [
    ("conversation", "hello Ori"),
    ("conversation", "hi there"),
    ("conversation", "good morning"),
    ("conversation", "thanks Ori"),
    ("conversation", "nice to see you"),
    ("conversation", "what are we working on"),
    ("conversation", "talk with me"),
    ("conversation", "are you online"),
    ("conversation", "how are you doing"),
    ("conversation", "let's chat"),
    ("conversation", "hey, are you there"),
    ("conversation", "I'm back"),
    ("question", "what is TensorFlow"),
    ("question", "what framework does Ori use"),
    ("question", "what is Ori Core"),
    ("question", "why is identity outside the model"),
    ("question", "what does server side mean"),
    ("question", "why keep the API key private"),
    ("question", "what is a language model"),
    ("question", "how does a transformer work"),
    ("question", "why use a small model"),
    ("question", "where does Ori run"),
    ("question", "what is a model checkpoint"),
    ("question", "how does model training work"),
    ("coding", "write Python code"),
    ("coding", "help me code a web app"),
    ("coding", "create a function"),
    ("coding", "write a JavaScript example"),
    ("coding", "show me how to use an API"),
    ("coding", "make a script for this"),
    ("coding", "help me program this"),
    ("coding", "write the code for the feature"),
    ("coding", "add a Python helper"),
    ("coding", "build this component"),
    ("coding", "implement this endpoint"),
    ("coding", "write code that parses JSON"),
    ("planning", "plan this project"),
    ("planning", "break this into tasks"),
    ("planning", "make a roadmap"),
    ("planning", "what should we build first"),
    ("planning", "help me organize the work"),
    ("planning", "design the next phase"),
    ("planning", "give me a project plan"),
    ("planning", "how should we structure this"),
    ("planning", "sequence the implementation"),
    ("planning", "outline the build"),
    ("planning", "what comes next"),
    ("planning", "help prioritize the tasks"),
    ("troubleshooting", "why is this broken"),
    ("troubleshooting", "help me debug this error"),
    ("troubleshooting", "fix my failing build"),
    ("troubleshooting", "why did deployment fail"),
    ("troubleshooting", "find the bug in this code"),
    ("troubleshooting", "the server is returning an error"),
    ("troubleshooting", "help me diagnose this problem"),
    ("troubleshooting", "something stopped working"),
    ("troubleshooting", "why is the endpoint failing"),
    ("troubleshooting", "debug this crash"),
    ("troubleshooting", "the build is broken"),
    ("troubleshooting", "figure out what went wrong"),
    ("creative", "write a story idea"),
    ("creative", "give me a joke"),
    ("creative", "invent a character"),
    ("creative", "brainstorm a game"),
    ("creative", "write a scene"),
    ("creative", "come up with a fun idea"),
    ("creative", "make a creative concept"),
    ("creative", "help me design a fictional world"),
    ("creative", "invent a robot character"),
    ("creative", "create a game mechanic"),
    ("creative", "think of a sci-fi setting"),
    ("creative", "make something imaginative"),
]


def build_stratified_split(
    validation_fraction: float,
) -> tuple[list[str], list[int], list[str], list[int]]:
    """Keep every label represented in both train and validation sets."""
    label_to_id = {label: index for index, label in enumerate(LABELS)}
    grouped: dict[str, list[str]] = {label: [] for label in LABELS}
    for label, text in TRAINING_DATA:
        grouped[label].append(text)

    train_texts: list[str] = []
    train_labels: list[int] = []
    val_texts: list[str] = []
    val_labels: list[int] = []

    for label in LABELS:
        examples = grouped[label]
        val_count = max(1, int(round(len(examples) * validation_fraction)))
        split_at = len(examples) - val_count
        train_part = examples[:split_at]
        val_part = examples[split_at:]
        train_texts.extend(train_part)
        train_labels.extend([label_to_id[label]] * len(train_part))
        val_texts.extend(val_part)
        val_labels.extend([label_to_id[label]] * len(val_part))

    return train_texts, train_labels, val_texts, val_labels


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="models/ori_core.keras")
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--validation-split", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    if not 0 < args.validation_split < 1:
        raise ValueError("--validation-split must be between 0 and 1")

    tf.keras.utils.set_random_seed(args.seed)

    train_texts, train_labels, val_texts, val_labels = build_stratified_split(
        args.validation_split
    )
    model = OriCoreModel.build_from_texts(train_texts, max_tokens=4096)

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)

    history = model.model.fit(
        tf.constant(train_texts),
        tf.constant(train_labels, dtype=tf.int32),
        batch_size=args.batch_size,
        epochs=args.epochs,
        validation_data=(
            tf.constant(val_texts),
            tf.constant(val_labels, dtype=tf.int32),
        ),
        shuffle=True,
        callbacks=[
            tf.keras.callbacks.EarlyStopping(
                monitor="val_accuracy",
                patience=10,
                mode="max",
                restore_best_weights=True,
            ),
        ],
        verbose=2,
    )

    model.save(output)

    summary = {
        "model": "ori-core",
        "examples": len(TRAINING_DATA),
        "train_examples": len(train_texts),
        "validation_examples": len(val_texts),
        "labels": list(LABELS),
        "epochs_completed": len(history.history["loss"]),
        "final_loss": history.history["loss"][-1],
        "final_accuracy": history.history["accuracy"][-1],
        "best_val_accuracy": max(history.history.get("val_accuracy", [0.0])),
        "final_val_loss": history.history.get("val_loss", [None])[-1],
        "final_val_accuracy": history.history.get("val_accuracy", [None])[-1],
        "seed": args.seed,
    }
    summary_path = output.with_suffix(".training.json")
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()

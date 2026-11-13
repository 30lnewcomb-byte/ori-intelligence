"""Conservative continuation training from the existing Ori checkpoints.

This script never builds a fresh model or tokenizer. It loads the deployed
Core/Ori Small artifacts, trains candidates briefly, and promotes each model
only when its own held-out evaluation improves without a regression.
"""

from __future__ import annotations

import argparse
import json
import random
import shutil
from pathlib import Path
from typing import Any

import numpy as np
import tensorflow as tf

from model import LABELS
from ori_model import OriLMConfig, OriLanguageModel, OriTokenizer
from train_core import TRAINING_DATA, build_stratified_split


class MaskedCausalLoss(tf.keras.losses.Loss):
    def __init__(self, pad_id: int):
        super().__init__()
        self.pad_id = pad_id

    def call(self, y_true, y_pred):
        losses = tf.keras.losses.sparse_categorical_crossentropy(
            y_true, y_pred, from_logits=True
        )
        mask = tf.cast(tf.not_equal(y_true, self.pad_id), losses.dtype)
        return tf.reduce_sum(losses * mask) / tf.maximum(
            tf.reduce_sum(mask), 1.0
        )


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(f"{path}:{line_number} is not valid JSON.") from exc
        if not isinstance(row, dict):
            raise ValueError(f"{path}:{line_number} must contain a JSON object.")
        rows.append(row)
    if not rows:
        raise ValueError(f"{path} is empty.")
    return rows


def load_small_texts(path: Path) -> list[str]:
    texts: list[str] = []
    seen: set[str] = set()
    for row in read_jsonl(path):
        if isinstance(row.get("text"), str):
            value = row["text"].strip()
        elif isinstance(row.get("input"), str) and isinstance(row.get("response"), str):
            value = f"User: {row['input'].strip()} Ori: {row['response'].strip()}"
        else:
            raise ValueError(f"{path} rows need text or input+response fields.")
        if " Ori:" not in value:
            raise ValueError(f"Training row must contain an ' Ori:' response marker: {value[:80]!r}")
        key = " ".join(value.split()).casefold()
        if key and key not in seen:
            seen.add(key)
            texts.append(value)
    if not texts:
        raise ValueError(f"{path} contains no usable Ori examples.")
    return texts


def load_core_examples(path: Path) -> tuple[list[str], list[int]]:
    label_ids = {label: index for index, label in enumerate(LABELS)}
    texts: list[str] = []
    labels: list[int] = []
    seen: set[tuple[str, str]] = set()
    for row in read_jsonl(path):
        label = row.get("label")
        value = row.get("text")
        if label not in label_ids or not isinstance(value, str) or not value.strip():
            raise ValueError(f"{path} rows need a known label and non-empty text.")
        key = (label, " ".join(value.split()).casefold())
        if key in seen:
            continue
        seen.add(key)
        texts.append(value.strip())
        labels.append(label_ids[label])
    if not texts:
        raise ValueError(f"{path} contains no usable Core examples.")
    return texts, labels


def make_lm_arrays(
    texts: list[str],
    tokenizer: OriTokenizer,
    context: int,
) -> tuple[np.ndarray, np.ndarray]:
    pad = tokenizer.vocab["<pad>"]
    inputs: list[list[int]] = []
    targets: list[list[int]] = []
    for text in texts:
        split_at = text.find(" Ori:")
        if split_at < 0:
            continue
        prefix = text[: split_at + len(" Ori:")]
        response = text[split_at + len(" Ori:"):].strip()
        if not response:
            continue

        prefix_ids = tokenizer.encode(prefix, add_bos=True, add_eos=False)
        response_ids = tokenizer.encode(response, add_bos=False, add_eos=True)
        if len(prefix_ids) >= context:
            prefix_ids = prefix_ids[-(context - 2):]
        ids = (prefix_ids + response_ids)[: context + 1]
        if len(ids) < 3 or len(ids) <= len(prefix_ids):
            continue

        x = ids[:-1]
        y = ids[1:]
        response_start = len(prefix_ids)
        for index in range(len(y)):
            if index + 1 < response_start:
                y[index] = pad

        x += [pad] * (context - len(x))
        y += [pad] * (context - len(y))
        inputs.append(x[:context])
        targets.append(y[:context])

    if not inputs:
        raise ValueError("No usable token sequences were created.")
    return np.asarray(inputs, dtype=np.int32), np.asarray(targets, dtype=np.int32)


def lm_eval_loss(
    model: OriLanguageModel,
    x: np.ndarray,
    y: np.ndarray,
    pad_id: int,
) -> float:
    logits = model(tf.convert_to_tensor(x), training=False)
    losses = tf.keras.losses.sparse_categorical_crossentropy(
        y, logits, from_logits=True
    )
    mask = tf.cast(np.not_equal(y, pad_id), losses.dtype)
    denominator = tf.maximum(tf.reduce_sum(mask), 1.0)
    return float((tf.reduce_sum(losses * mask) / denominator).numpy())


def train_small(
    base_dir: Path,
    output_dir: Path,
    training_data: Path,
    replay_data: Path,
    evaluation_data: Path,
    *,
    epochs: int,
    batch_size: int,
    seed: int,
) -> dict[str, Any]:
    required = [
        base_dir / "config.json",
        base_dir / "ori_tokenizer.model",
        base_dir / "model.weights.h5",
    ]
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise FileNotFoundError("Existing Ori Small checkpoint is incomplete: " + ", ".join(missing))
    if output_dir.exists():
        raise FileExistsError(f"Candidate output already exists: {output_dir}")

    output_dir.parent.mkdir(parents=True, exist_ok=True)
    shutil.copytree(base_dir, output_dir)

    config = OriLMConfig(**json.loads(required[0].read_text(encoding="utf-8")))
    tokenizer = OriTokenizer.load(required[1])
    model = OriLanguageModel(config, name="ori_language_model")
    model(tf.zeros((1, min(2, config.context_length)), dtype=tf.int32))
    model.load_weights(required[2])

    context = min(128, config.context_length)
    new_texts = load_small_texts(training_data)
    old_texts = load_small_texts(replay_data)
    replay_count = min(len(old_texts), max(24, min(64, len(new_texts) * 2)))
    replay = random.Random(seed).sample(old_texts, replay_count)
    train_x, train_y = make_lm_arrays(
        new_texts + replay, tokenizer, context
    )

    eval_rows = read_jsonl(evaluation_data)
    eval_texts: list[str] = []
    for row in eval_rows:
        prompt = row.get("prompt")
        expected = row.get("expected")
        if not isinstance(prompt, str) or not isinstance(expected, str):
            raise ValueError("Ori evaluation rows need prompt and expected strings.")
        eval_texts.append(f"User: {prompt.strip()} Ori: {expected.strip()}")
    eval_x, eval_y = make_lm_arrays(eval_texts, tokenizer, context)

    baseline_loss = lm_eval_loss(model, eval_x, eval_y, config.pad_id)
    model.compile(
        optimizer=tf.keras.optimizers.AdamW(
            learning_rate=1e-5,
            weight_decay=1e-5,
            clipnorm=0.5,
        ),
        loss=MaskedCausalLoss(config.pad_id),
    )
    history = model.fit(
        train_x,
        train_y,
        batch_size=batch_size,
        epochs=epochs,
        shuffle=True,
        verbose=2,
    )
    candidate_loss = lm_eval_loss(model, eval_x, eval_y, config.pad_id)

    accepted = (
        np.isfinite(candidate_loss)
        and np.isfinite(baseline_loss)
        and candidate_loss < baseline_loss
    )
    if accepted:
        model.save_weights(output_dir / "model.weights.h5")

    return {
        "model": "ori-small",
        "started_from_existing_weights": True,
        "tokenizer_rebuilt": False,
        "new_examples": len(new_texts),
        "replay_examples": replay_count,
        "training_sequences": int(len(train_x)),
        "sequence_length": context,
        "epochs_completed": len(history.history.get("loss", [])),
        "baseline_eval_loss": baseline_loss,
        "candidate_eval_loss": candidate_loss,
        "accepted": bool(accepted),
    }


def train_core(
    base_path: Path,
    output_path: Path,
    training_data: Path,
    *,
    epochs: int,
    batch_size: int,
) -> dict[str, Any]:
    if not base_path.is_file():
        raise FileNotFoundError(f"Existing Ori Core checkpoint not found: {base_path}")
    if output_path.exists():
        raise FileExistsError(f"Candidate Core output already exists: {output_path}")

    train_texts, train_labels, val_texts, val_labels = build_stratified_split(0.2)
    label_to_id = {label: index for index, label in enumerate(LABELS)}
    new_texts, new_labels = load_core_examples(training_data)
    x_train = tf.constant(train_texts + new_texts)
    y_train = tf.constant(train_labels + new_labels, dtype=tf.int32)
    x_val = tf.constant(val_texts)
    y_val = tf.constant(val_labels, dtype=tf.int32)

    model = tf.keras.models.load_model(base_path, compile=False)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    baseline_loss, baseline_accuracy = model.evaluate(
        x_val, y_val, verbose=0
    )
    history = model.fit(
        x_train,
        y_train,
        batch_size=batch_size,
        epochs=epochs,
        shuffle=True,
        validation_data=(x_val, y_val),
        callbacks=[
            tf.keras.callbacks.EarlyStopping(
                monitor="val_loss",
                patience=1,
                restore_best_weights=True,
            )
        ],
        verbose=2,
    )
    candidate_loss, candidate_accuracy = model.evaluate(
        x_val, y_val, verbose=0
    )
    accepted = (
        np.isfinite(candidate_loss)
        and candidate_accuracy >= baseline_accuracy
        and candidate_loss < baseline_loss
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    if accepted:
        model.save(output_path, include_optimizer=False)
    else:
        shutil.copy2(base_path, output_path)

    return {
        "model": "ori-core",
        "started_from_existing_weights": True,
        "vocabulary_rebuilt": False,
        "new_examples": len(new_texts),
        "epochs_completed": len(history.history.get("loss", [])),
        "baseline_validation_loss": float(baseline_loss),
        "candidate_validation_loss": float(candidate_loss),
        "baseline_validation_accuracy": float(baseline_accuracy),
        "candidate_validation_accuracy": float(candidate_accuracy),
        "accepted": bool(accepted),
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Fine-tune saved Ori checkpoints without rebuilding either model."
    )
    parser.add_argument("--core-input", required=True)
    parser.add_argument("--core-output", required=True)
    parser.add_argument("--small-input-dir", required=True)
    parser.add_argument("--small-output-dir", required=True)
    parser.add_argument("--core-data", default="tensorflow_core/data/ori_core_incremental.jsonl")
    parser.add_argument("--small-data", default="tensorflow_core/data/ori_small_incremental.jsonl")
    parser.add_argument("--small-replay-data", default="tensorflow_core/data/ori_training_expanded.jsonl")
    parser.add_argument("--small-eval-data", default="tensorflow_core/data/ori_eval_expanded.jsonl")
    parser.add_argument("--core-epochs", type=int, default=1)
    parser.add_argument("--small-epochs", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--seed", type=int, default=43)
    parser.add_argument("--summary", required=True)
    args = parser.parse_args()

    if not 1 <= args.core_epochs <= 2 or not 1 <= args.small_epochs <= 2:
        raise ValueError("Incremental fine-tuning is limited to 1-2 epochs per model.")
    if not 1 <= args.batch_size <= 4:
        raise ValueError("Incremental fine-tuning batch size must be between 1 and 4.")

    random.seed(args.seed)
    tf.keras.utils.set_random_seed(args.seed)

    core_result = train_core(
        Path(args.core_input),
        Path(args.core_output),
        Path(args.core_data),
        epochs=args.core_epochs,
        batch_size=args.batch_size,
    )
    small_result = train_small(
        Path(args.small_input_dir),
        Path(args.small_output_dir),
        Path(args.small_data),
        Path(args.small_replay_data),
        Path(args.small_eval_data),
        epochs=args.small_epochs,
        batch_size=args.batch_size,
        seed=args.seed,
    )

    report = {
        "strategy": "incremental-fine-tuning-from-existing-checkpoints",
        "full_retraining": False,
        "core": core_result,
        "small": small_result,
    }
    summary = Path(args.summary)
    summary.parent.mkdir(parents=True, exist_ok=True)
    summary.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()

"""Specialized native TensorFlow workers for Ori.

Ori routes tasks internally. Workers only transform an assigned task into a
learned result; platform permissions and tool execution remain server-side.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any

import tensorflow as tf

from tensorflow_core.ori_model import OriLMConfig, OriLanguageModel, OriTokenizer


@dataclass(frozen=True)
class WorkerSpec:
    worker_id: str
    task: str
    model_dir: str


WORKERS = {
    "coding": WorkerSpec("ori-coder", "code-generation", "artifacts/ori-coder"),
    "3d": WorkerSpec("ori-3d", "parametric-cad-and-mesh-generation", "artifacts/ori-3d"),
}


@dataclass
class LoadedWorker:
    spec: WorkerSpec
    model: OriLanguageModel
    tokenizer: OriTokenizer


def load_worker(spec: WorkerSpec) -> LoadedWorker | None:
    root = Path(spec.model_dir)
    config_path = root / "config.json"
    tokenizer_path = root / "ori_tokenizer.model"
    weights_path = root / "model.weights.h5"

    if not (config_path.exists() and tokenizer_path.exists() and weights_path.exists()):
        return None

    try:
        config = OriLMConfig(**json.loads(config_path.read_text(encoding="utf-8")))
        tokenizer = OriTokenizer.load(tokenizer_path)
        model = OriLanguageModel(config, name=f"{spec.worker_id}_language_model")
        model(tf.zeros((1, min(2, config.context_length)), dtype=tf.int32))
        model.load_weights(weights_path)
        return LoadedWorker(spec=spec, model=model, tokenizer=tokenizer)
    except Exception:
        return None


def generate_worker(
    worker: LoadedWorker,
    prompt: str,
    *,
    max_tokens: int = 160,
    temperature: float = 0.15,
    top_k: int = 8,
) -> str:
    worker_prompt = f"Task: {prompt.strip()}\nResult:"
    text, _ = worker.model.generate(
        worker.tokenizer,
        worker_prompt,
        max_new_tokens=max_tokens,
        temperature=temperature,
        top_k=top_k,
    )
    return text.split("Task:", 1)[0].strip()

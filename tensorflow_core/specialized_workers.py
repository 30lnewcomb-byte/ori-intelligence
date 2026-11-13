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


def compile_cad_plan(plan: dict[str, Any]) -> str:
    """Compile a learned parametric CAD plan into OpenSCAD source."""
    operations = plan.get("operations")
    if not isinstance(operations, list) or not operations:
        raise ValueError("CAD plan must contain a non-empty operations list.")

    def render_shape(op: dict[str, Any], indent: str = "") -> list[str]:
        kind = op.get("kind")

        if kind == "cube":
            size = op.get("size")
            center = bool(op.get("center", True))
            if not (
                isinstance(size, list)
                and len(size) == 3
                and all(isinstance(v, (int, float)) and v > 0 for v in size)
            ):
                raise ValueError("cube.size must contain three positive numbers.")
            return [
                f"{indent}cube([{size[0]}, {size[1]}, {size[2]}], center={str(center).lower()});"
            ]

        if kind == "cylinder":
            radius = op.get("radius")
            height = op.get("height")
            center = bool(op.get("center", True))
            if not (
                isinstance(radius, (int, float))
                and radius > 0
                and isinstance(height, (int, float))
                and height > 0
            ):
                raise ValueError("cylinder requires positive radius and height.")
            return [
                f"{indent}cylinder(r={radius}, h={height}, center={str(center).lower()});"
            ]

        if kind in {"translate", "rotate"}:
            value = op.get("value")
            child = op.get("shape")
            if not (
                isinstance(value, list)
                and len(value) == 3
                and all(isinstance(v, (int, float)) for v in value)
                and isinstance(child, dict)
            ):
                raise ValueError(f"{kind} requires a three-number value and shape.")
            wrapper = "translate" if kind == "translate" else "rotate"
            body = render_shape(child, indent + "  ")
            return [
                f"{indent}{wrapper}([{value[0]}, {value[1]}, {value[2]}]) {{",
                *body,
                f"{indent}}}",
            ]

        if kind == "mesh":
            vertices = op.get("vertices")
            triangles = op.get("triangles")
            if not (
                isinstance(vertices, list)
                and len(vertices) >= 3
                and all(
                    isinstance(vertex, list)
                    and len(vertex) == 3
                    and all(isinstance(value, (int, float)) for value in vertex)
                    for vertex in vertices
                )
            ):
                raise ValueError("mesh.vertices must contain 3D numeric points.")
            if not (
                isinstance(triangles, list)
                and triangles
                and all(
                    isinstance(face, list)
                    and len(face) == 3
                    and all(isinstance(index, int) for index in face)
                    for face in triangles
                )
            ):
                raise ValueError("mesh.triangles must contain 3-index triangular faces.")

            vertex_count = len(vertices)
            for face in triangles:
                if any(index < 0 or index >= vertex_count for index in face):
                    raise ValueError("mesh triangle index is outside the vertex list.")
                if len(set(face)) != 3:
                    raise ValueError("mesh triangles cannot repeat a vertex.")

            points = ",\n".join(
                f"{indent}  [{vertex[0]}, {vertex[1]}, {vertex[2]}]"
                for vertex in vertices
            )
            faces = ",\n".join(
                f"{indent}  [{face[0]}, {face[1]}, {face[2]}]"
                for face in triangles
            )
            return [
                f"{indent}polyhedron(",
                f"{indent}  points = [",
                points,
                f"{indent}  ],",
                f"{indent}  faces = [",
                faces,
                f"{indent}  ]",
                f"{indent});",
            ]

        if kind in {"union", "difference"}:
            children = op.get("shapes")
            if not (
                isinstance(children, list)
                and children
                and all(isinstance(child, dict) for child in children)
            ):
                raise ValueError(f"{kind} requires a non-empty shapes array.")
            body: list[str] = []
            for child in children:
                body.extend(render_shape(child, indent + "  "))
            return [f"{indent}{kind}() {{", *body, f"{indent}}}"]

        raise ValueError(f"Unsupported CAD operation: {kind!r}")

    lines = ["$fn = 48;", ""]
    for operation in operations:
        if not isinstance(operation, dict):
            raise ValueError("Each CAD operation must be an object.")
        lines.extend(render_shape(operation))
        lines.append("")
    return "\n".join(lines).rstrip()

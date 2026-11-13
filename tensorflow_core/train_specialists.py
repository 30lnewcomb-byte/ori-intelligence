"""Train Ori's specialized TensorFlow workers.

The workers share Ori's English tokenizer but have separate weights and smaller
task-focused Transformer configurations.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import tensorflow as tf

from ori_model import OriLMConfig, OriLanguageModel, OriTokenizer


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


def coding_examples() -> list[tuple[str, str]]:
    examples: list[tuple[str, str]] = []
    languages = {
        "python": {
            "add": "def add(a, b):\n    return a + b",
            "square": "def square(x):\n    return x * x",
            "max": "def find_max(values):\n    return max(values)",
            "reverse": "def reverse_text(text):\n    return text[::-1]",
            "count": "def count_items(items):\n    counts = {}\n    for item in items:\n        counts[item] = counts.get(item, 0) + 1\n    return counts",
            "evens": "def even_numbers(values):\n    return [value for value in values if value % 2 == 0]",
            "sum": "def total(values):\n    return sum(values)",
            "factorial": "def factorial(n):\n    if n <= 1:\n        return 1\n    return n * factorial(n - 1)",
            "fibonacci": "def fibonacci(n):\n    a, b = 0, 1\n    for _ in range(n):\n        a, b = b, a + b\n    return a",
        }
    }
    for language, funcs in languages.items():
        for name, code in funcs.items():
            examples.append((f"Write a small {language} function to {name}.", code))
            examples.append((f"Generate {language} code for a reusable {name} helper.", code))
            examples.append((f"Give me a concise {language} implementation of {name}.", code))

    extra = [
        ("Read JSON from a file in Python.", 'import json\n\nwith open("data.json", "r", encoding="utf-8") as file:\n    data = json.load(file)'),
        ("Write JSON to a file in Python.", 'import json\n\nwith open("data.json", "w", encoding="utf-8") as file:\n    json.dump(data, file, indent=2)'),
        ("Make a Python function that checks whether a string is empty.", 'def is_empty(text):\n    return not text.strip()'),
        ("Make a Python function that removes duplicates while preserving order.", 'def unique(values):\n    return list(dict.fromkeys(values))'),
        ("Make a Python function that converts Celsius to Fahrenheit.", 'def celsius_to_fahrenheit(celsius):\n    return celsius * 9 / 5 + 32'),
        ("Make a Python function that calculates the area of a circle.", 'import math\n\ndef circle_area(radius):\n    return math.pi * radius ** 2'),
        ("Write a JavaScript function that adds two numbers.", 'function add(a, b) {\n  return a + b;\n}'),
        ("Write a JavaScript function that reverses a string.", 'function reverseText(text) {\n  return [...text].reverse().join("");\n}'),
        ("Write a JavaScript function that filters even numbers.", 'function evenNumbers(values) {\n  return values.filter(value => value % 2 === 0);\n}'),
    ]
    examples.extend(extra)
    return examples


def cad_examples() -> list[tuple[str, str]]:
    examples: list[tuple[str, str]] = []

    sizes = [(10, 10, 10), (20, 20, 5), (30, 20, 10), (40, 30, 4), (50, 20, 8)]
    for x, y, z in sizes:
        plan = json.dumps({
            "operations": [
                {"kind": "cube", "size": [x, y, z], "center": True}
            ]
        }, separators=(",", ":"))
        examples.extend([
            (f"Create a {x} by {y} by {z} mm box.", plan),
            (f"Make a rectangular block measuring {x}x{y}x{z} millimeters.", plan),
            (f"Generate a simple {x}mm x {y}mm x {z}mm box.", plan),
        ])

    radii = [(5, 10), (8, 20), (10, 25), (12, 30), (15, 40)]
    for radius, height in radii:
        plan = json.dumps({
            "operations": [
                {"kind": "cylinder", "radius": radius, "height": height, "center": True}
            ]
        }, separators=(",", ":"))
        examples.extend([
            (f"Create a cylinder with a {radius} mm radius and {height} mm height.", plan),
            (f"Make a round post, radius {radius} mm, height {height} mm.", plan),
        ])

    for size in (12, 18, 25, 32):
        plan = json.dumps({
            "operations": [
                {"kind": "cube", "size": [size, size, size], "center": True}
            ]
        }, separators=(",", ":"))
        examples.extend([
            (f"Make a centered {size} mm cube.", plan),
            (f"Generate a simple {size} millimeter cube for 3D printing.", plan),
        ])

    # Triangle fundamentals: three vertices define one triangular face.
    triangle_plans = [
        (
            "a triangle in the XY plane",
            {
                "operations": [{
                    "kind": "mesh",
                    "vertices": [[0, 0, 0], [20, 0, 0], [0, 20, 0]],
                    "triangles": [[0, 1, 2]],
                }],
            },
        ),
        (
            "a pyramid made from triangular faces",
            {
                "operations": [{
                    "kind": "mesh",
                    "vertices": [
                        [-10, -10, 0], [10, -10, 0],
                        [10, 10, 0], [-10, 10, 0], [0, 0, 15],
                    ],
                    "triangles": [
                        [0, 1, 4], [1, 2, 4], [2, 3, 4], [3, 0, 4],
                    ],
                }],
            },
        ),
        (
            "a tetrahedron",
            {
                "operations": [{
                    "kind": "mesh",
                    "vertices": [
                        [0, 0, 12], [-10, -8, 0],
                        [10, -8, 0], [0, 12, 0],
                    ],
                    "triangles": [
                        [0, 1, 2], [0, 2, 3],
                        [0, 3, 1], [1, 3, 2],
                    ],
                }],
            },
        ),
        (
            "a triangular prism",
            {
                "operations": [{
                    "kind": "mesh",
                    "vertices": [
                        [0, 0, 0], [20, 0, 0], [0, 20, 0],
                        [0, 0, 10], [20, 0, 10], [0, 20, 10],
                    ],
                    "triangles": [
                        [0, 1, 2], [3, 5, 4],
                        [0, 3, 4], [0, 4, 1],
                        [1, 4, 5], [1, 5, 2],
                        [2, 5, 3], [2, 3, 0],
                    ],
                }],
            },
        ),
    ]

    for description, plan_data in triangle_plans:
        plan = json.dumps(plan_data, separators=(",", ":"))
        examples.extend([
            (f"Create {description}.", plan),
            (f"Model {description} using triangular mesh faces.", plan),
            (f"Build {description} from vertices and triangles.", plan),
        ])

    # Teach the vocabulary explicitly as geometry concepts.
    triangle_terms = [
        ("A mesh is made from vertices and faces.", triangle_plans[0][1]),
        ("A triangle face has exactly three vertex indices.", triangle_plans[0][1]),
        ("Three 3D points define a triangular surface.", triangle_plans[0][1]),
        ("Use triangular faces to represent a 3D surface.", triangle_plans[1][1]),
    ]
    for prompt, plan_data in triangle_terms:
        examples.append((prompt, json.dumps(plan_data, separators=(",", ":"))))

    return examples


def make_arrays(
    pairs: list[tuple[str, str]],
    tokenizer: OriTokenizer,
    context: int,
) -> tuple[np.ndarray, np.ndarray]:
    xs: list[list[int]] = []
    ys: list[list[int]] = []
    pad = tokenizer.vocab["<pad>"]

    for prompt, result in pairs:
        prefix = f"Task: {prompt}\nResult:"
        prefix_ids = tokenizer.encode(prefix, add_bos=True, add_eos=False)
        result_ids = tokenizer.encode(result, add_bos=False, add_eos=True)
        ids = (prefix_ids + result_ids)[: context + 1]
        if len(ids) < 3:
            continue

        x = ids[:-1]
        y = ids[1:]
        result_start = len(prefix_ids)
        for index in range(len(y)):
            if index + 1 < result_start:
                y[index] = pad

        x += [pad] * (context - len(x))
        y += [pad] * (context - len(y))
        xs.append(x[:context])
        ys.append(y[:context])

    if not xs:
        raise ValueError("Worker dataset produced no usable sequences.")
    return np.asarray(xs, dtype=np.int32), np.asarray(ys, dtype=np.int32)


def train_worker(
    worker_id: str,
    pairs: list[tuple[str, str]],
    tokenizer: OriTokenizer,
    output: Path,
    *,
    epochs: int,
    d_model: int,
    num_layers: int,
    d_ff: int,
    learning_rate: float,
) -> None:
    output.mkdir(parents=True, exist_ok=True)
    config = OriLMConfig(
        vocab_size=tokenizer.vocab_size,
        context_length=256,
        d_model=d_model,
        num_heads=4,
        num_layers=num_layers,
        d_ff=d_ff,
        dropout=0.1,
    )
    model = OriLanguageModel(config, name=f"{worker_id}_language_model")
    model(tf.zeros((1, config.context_length), dtype=tf.int32))
    model.compile(
        optimizer=tf.keras.optimizers.AdamW(
            learning_rate=learning_rate,
            weight_decay=1e-4,
            clipnorm=1.0,
        ),
        loss=MaskedCausalLoss(config.pad_id),
    )
    x, y = make_arrays(pairs, tokenizer, config.context_length)
    model.fit(
        x,
        y,
        batch_size=8,
        epochs=epochs,
        shuffle=True,
        verbose=1,
    )
    model.save_weights(output / "model.weights.h5")
    (output / "config.json").write_text(
        json.dumps(config.__dict__, indent=2),
        encoding="utf-8",
    )
    print(f"{worker_id} saved: {len(pairs)} examples -> {output}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tokenizer", default="artifacts/ori-small/ori_tokenizer.model")
    parser.add_argument("--output-root", default="artifacts")
    parser.add_argument("--code-epochs", type=int, default=10)
    parser.add_argument("--cad-epochs", type=int, default=10)
    parser.add_argument("--seed", type=int, default=43)
    args = parser.parse_args()

    tf.keras.utils.set_random_seed(args.seed)
    tokenizer = OriTokenizer.load(args.tokenizer)

    train_worker(
        "ori-coder",
        coding_examples(),
        tokenizer,
        Path(args.output_root) / "ori-coder",
        epochs=args.code_epochs,
        d_model=128,
        num_layers=3,
        d_ff=512,
        learning_rate=8e-5,
    )
    train_worker(
        "ori-3d",
        cad_examples(),
        tokenizer,
        Path(args.output_root) / "ori-3d",
        epochs=args.cad_epochs,
        d_model=128,
        num_layers=3,
        d_ff=512,
        learning_rate=8e-5,
    )


if __name__ == "__main__":
    main()

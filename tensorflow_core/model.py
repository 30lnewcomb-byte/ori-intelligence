"""Small TensorFlow core model for Ori.

This is intentionally NOT Ori's future super model. It is a real, trainable
TensorFlow/Keras starting point for intent and complexity signals that the
orchestrator can use before the larger learned intelligence is developed.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import tensorflow as tf

LABELS = (
    "conversation",
    "question",
    "coding",
    "planning",
    "troubleshooting",
    "creative",
)


@dataclass(frozen=True)
class CorePrediction:
    label: str
    confidence: float


class OriCoreModel:
    """Tiny Keras text classifier used as Ori's first TensorFlow core."""

    def __init__(self, model: tf.keras.Model) -> None:
        self.model = model

    @staticmethod
    def build(
        max_tokens: int = 4096,
        sequence_length: int = 96,
    ) -> "OriCoreModel":
        """Build the classifier shell with a fixed multi-hot feature width.

        This method is kept for small local experiments. Production training
        should use build_from_texts() so the vectorizer is adapted before the
        dense layers are constructed.
        """
        del sequence_length

        vectorizer = tf.keras.layers.TextVectorization(
            max_tokens=max_tokens,
            output_mode="multi_hot",
            pad_to_max_tokens=True,
            name="text_vectorizer",
        )

        inputs = tf.keras.Input(shape=(), dtype=tf.string, name="text")
        features = vectorizer(inputs)
        x = tf.keras.layers.Dense(64, activation="relu", name="hidden")(features)
        x = tf.keras.layers.Dropout(0.1)(x)
        outputs = tf.keras.layers.Dense(
            len(LABELS),
            activation="softmax",
            name="intent",
        )(x)
        model = tf.keras.Model(inputs=inputs, outputs=outputs, name="ori_core")
        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
            loss="sparse_categorical_crossentropy",
            metrics=["accuracy"],
        )
        return OriCoreModel(model)

    @staticmethod
    def build_from_texts(
        texts: list[str],
        max_tokens: int = 4096,
    ) -> "OriCoreModel":
        """Adapt the vectorizer before building Dense layers.

        Multi-hot text features keep the classifier simple and serializable
        across the TensorFlow/Keras versions used by the Render runtime.
        """
        vectorizer = tf.keras.layers.TextVectorization(
            max_tokens=max_tokens,
            output_mode="multi_hot",
            pad_to_max_tokens=False,
            name="text_vectorizer",
        )
        vectorizer.adapt(
            tf.data.Dataset.from_tensor_slices(texts).batch(32)
        )

        inputs = tf.keras.Input(shape=(), dtype=tf.string, name="text")
        features = vectorizer(inputs)
        feature_dim = features.shape[-1]
        if feature_dim is None:
            raise RuntimeError(
                "Ori Core vectorizer did not produce a fixed feature width."
            )

        x = tf.keras.layers.Dense(
            64,
            activation="relu",
            name="hidden",
        )(features)
        x = tf.keras.layers.Dropout(0.1)(x)
        outputs = tf.keras.layers.Dense(
            len(LABELS),
            activation="softmax",
            name="intent",
        )(x)
        model = tf.keras.Model(inputs=inputs, outputs=outputs, name="ori_core")
        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
            loss="sparse_categorical_crossentropy",
            metrics=["accuracy"],
        )
        return OriCoreModel(model)

    def adapt(self, texts: list[str]) -> None:
        for layer in self.model.layers:
            if isinstance(layer, tf.keras.layers.TextVectorization):
                if layer.get_vocabulary():
                    return
                layer.adapt(tf.data.Dataset.from_tensor_slices(texts).batch(32))
                return
        raise RuntimeError("Ori Core model is missing its TextVectorization layer")

    def predict(self, text: str) -> CorePrediction:
        probabilities = self.model.predict(tf.constant([text]), verbose=0)[0]
        index = int(tf.argmax(probabilities).numpy())
        return CorePrediction(
            label=LABELS[index],
            confidence=float(probabilities[index]),
        )

    def save(self, directory: str | Path) -> None:
        self.model.save(directory, include_optimizer=False)


if __name__ == "__main__":
    core = OriCoreModel.build()
    core.adapt([
        "hello there",
        "what is python",
        "fix my code",
        "help me plan a project",
        "why is this broken",
        "write a story idea",
    ])
    prediction = core.predict("help me debug my Python project")
    print(prediction)

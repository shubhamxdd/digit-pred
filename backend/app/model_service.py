from __future__ import annotations

from pathlib import Path

import numpy as np
import tensorflow as tf
from keras.datasets import mnist

from .preprocessing import for_cnn_model, for_dense_models


class ModelService:
    def __init__(self, models_dir: Path) -> None:
        self.models_dir = models_dir
        self.perceptron: tf.keras.Model | None = None
        self.ann: tf.keras.Model | None = None
        self.cnn: tf.keras.Model | None = None
        self._typed_cache: tuple[np.ndarray, np.ndarray] | None = None

    def load(self) -> None:
        self.perceptron = tf.keras.models.load_model(self.models_dir / "perceptron.keras")
        self.ann = tf.keras.models.load_model(self.models_dir / "ann.keras")
        self.cnn = tf.keras.models.load_model(self.models_dir / "cnn.keras")

    def is_loaded(self) -> bool:
        return self.perceptron is not None and self.ann is not None and self.cnn is not None

    def _predict(self, model: tf.keras.Model, inputs: np.ndarray) -> tuple[int, float]:
        probs = model.predict(inputs, verbose=0)[0]
        predicted = int(np.argmax(probs))
        confidence = float(np.max(probs))
        return predicted, confidence

    def predict_all(self, arr_28x28: np.ndarray) -> dict[str, dict[str, int | float]]:
        if not self.is_loaded():
            raise RuntimeError("Models are not loaded yet.")

        dense_input = for_dense_models(arr_28x28)
        cnn_input = for_cnn_model(arr_28x28)

        perceptron_pred, perceptron_conf = self._predict(self.perceptron, dense_input)
        ann_pred, ann_conf = self._predict(self.ann, dense_input)
        cnn_pred, cnn_conf = self._predict(self.cnn, cnn_input)

        return {
            "perceptron": {
                "predicted_digit": perceptron_pred,
                "confidence": perceptron_conf,
            },
            "ann": {
                "predicted_digit": ann_pred,
                "confidence": ann_conf,
            },
            "cnn": {
                "predicted_digit": cnn_pred,
                "confidence": cnn_conf,
            },
        }

    def sample_typed_digit(self, digit: int) -> np.ndarray:
        if self._typed_cache is None:
            _, (x_test, y_test) = mnist.load_data()
            x_test = x_test.astype("float32") / 255.0
            self._typed_cache = (x_test, y_test)

        x_test, y_test = self._typed_cache
        matching = np.where(y_test == digit)[0]
        if len(matching) == 0:
            raise ValueError(f"No samples available for digit {digit}.")

        index = int(np.random.choice(matching))
        return x_test[index]

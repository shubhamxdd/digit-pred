from __future__ import annotations

from pathlib import Path

import numpy as np
from keras.datasets import mnist

from app.model_architectures import build_ann, build_cnn, build_perceptron


BASE_DIR = Path(__file__).resolve().parent
MODELS_DIR = BASE_DIR / "models"
MODELS_DIR.mkdir(parents=True, exist_ok=True)


def main() -> None:
    (x_train, y_train), (x_test, y_test) = mnist.load_data()

    x_train = x_train.astype("float32") / 255.0
    x_test = x_test.astype("float32") / 255.0

    x_train_cnn = np.expand_dims(x_train, axis=-1)
    x_test_cnn = np.expand_dims(x_test, axis=-1)

    perceptron = build_perceptron()
    ann = build_ann()
    cnn = build_cnn()

    print("Training perceptron...")
    perceptron.fit(x_train, y_train, epochs=5, batch_size=32, validation_split=0.1, verbose=1)
    print("Training ANN...")
    ann.fit(x_train, y_train, epochs=5, batch_size=32, validation_split=0.1, verbose=1)
    print("Training CNN...")
    cnn.fit(x_train_cnn, y_train, epochs=5, batch_size=32, validation_split=0.1, verbose=1)

    p_loss, p_acc = perceptron.evaluate(x_test, y_test, verbose=0)
    a_loss, a_acc = ann.evaluate(x_test, y_test, verbose=0)
    c_loss, c_acc = cnn.evaluate(x_test_cnn, y_test, verbose=0)

    print(f"Perceptron test accuracy: {p_acc:.4f}")
    print(f"ANN test accuracy: {a_acc:.4f}")
    print(f"CNN test accuracy: {c_acc:.4f}")

    perceptron.save(MODELS_DIR / "perceptron.keras")
    ann.save(MODELS_DIR / "ann.keras")
    cnn.save(MODELS_DIR / "cnn.keras")

    print(f"Saved models to {MODELS_DIR}")


if __name__ == "__main__":
    main()

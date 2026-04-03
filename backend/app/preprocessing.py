from __future__ import annotations

import base64
import io

import numpy as np
from PIL import Image, ImageOps


IMAGE_SIZE = (28, 28)


def _normalize_grayscale_array(img_array: np.ndarray) -> np.ndarray:
    arr = img_array.astype("float32") / 255.0

    # For typical uploads that are black digit on white paper, invert intensity.
    if arr.mean() > 0.5:
        arr = 1.0 - arr

    return arr


def image_bytes_to_mnist_array(image_bytes: bytes) -> np.ndarray:
    img = Image.open(io.BytesIO(image_bytes)).convert("L")
    img = ImageOps.fit(img, IMAGE_SIZE, Image.Resampling.LANCZOS)
    arr = np.array(img)
    return _normalize_grayscale_array(arr)


def base64_to_mnist_array(image_base64: str) -> np.ndarray:
    payload = image_base64
    if "," in payload:
        payload = payload.split(",", 1)[1]

    image_bytes = base64.b64decode(payload)
    return image_bytes_to_mnist_array(image_bytes)


def for_dense_models(arr_28x28: np.ndarray) -> np.ndarray:
    return np.expand_dims(arr_28x28, axis=0)


def for_cnn_model(arr_28x28: np.ndarray) -> np.ndarray:
    return np.expand_dims(arr_28x28, axis=(0, -1))


def array_to_base64_png(arr_28x28: np.ndarray) -> str:
    arr = np.clip(arr_28x28 * 255.0, 0, 255).astype(np.uint8)
    img = Image.fromarray(arr, mode="L")
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("ascii")

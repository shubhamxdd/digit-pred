from __future__ import annotations

import base64
import io

import numpy as np
from PIL import Image, ImageOps


IMAGE_SIZE = (28, 28)
INNER_SIZE = 20


def _normalize_grayscale_array(img_array: np.ndarray) -> np.ndarray:
    arr = img_array.astype("float32") / 255.0

    # For typical uploads that are black digit on white paper, invert intensity.
    if arr.mean() > 0.5:
        arr = 1.0 - arr

    return arr


def _to_mnist_like_canvas(arr: np.ndarray) -> np.ndarray:
    if arr.size == 0:
        return np.zeros(IMAGE_SIZE, dtype="float32")

    max_val = float(arr.max())
    if max_val < 1e-6:
        return np.zeros(IMAGE_SIZE, dtype="float32")

    threshold = max(0.2, max_val * 0.35)
    ys, xs = np.where(arr > threshold)

    if len(xs) == 0 or len(ys) == 0:
        return np.zeros(IMAGE_SIZE, dtype="float32")

    y_min, y_max = int(ys.min()), int(ys.max())
    x_min, x_max = int(xs.min()), int(xs.max())
    crop = arr[y_min : y_max + 1, x_min : x_max + 1]

    h, w = crop.shape
    scale = INNER_SIZE / max(h, w)
    new_h = max(1, int(round(h * scale)))
    new_w = max(1, int(round(w * scale)))

    crop_img = Image.fromarray(np.clip(crop * 255.0, 0, 255).astype(np.uint8), mode="L")
    resized = crop_img.resize((new_w, new_h), Image.Resampling.LANCZOS)
    resized_arr = np.asarray(resized, dtype="float32") / 255.0

    canvas = np.zeros(IMAGE_SIZE, dtype="float32")
    top = (IMAGE_SIZE[0] - new_h) // 2
    left = (IMAGE_SIZE[1] - new_w) // 2
    canvas[top : top + new_h, left : left + new_w] = resized_arr
    return canvas


def image_bytes_to_mnist_array(image_bytes: bytes) -> np.ndarray:
    img = Image.open(io.BytesIO(image_bytes)).convert("L")
    arr = np.array(img)
    arr = _normalize_grayscale_array(arr)
    return _to_mnist_like_canvas(arr)


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

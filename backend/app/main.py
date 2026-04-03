from __future__ import annotations

import os
from pathlib import Path

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware

from .model_service import ModelService
from .preprocessing import array_to_base64_png, base64_to_mnist_array, image_bytes_to_mnist_array
from .schemas import CanvasPredictRequest, PredictResponse, TypedPredictRequest

BASE_DIR = Path(__file__).resolve().parents[1]
MODELS_DIR = BASE_DIR / "models"

app = FastAPI(title="MNIST Multi-Model API", version="1.0.0")
service = ModelService(MODELS_DIR)

frontend_origins = os.getenv("FRONTEND_ORIGINS", "http://localhost:5173,http://127.0.0.1:5173")
allowed_origins = [origin.strip() for origin in frontend_origins.split(",") if origin.strip()]

app.add_middleware(
    CORSMiddleware,
    allow_origins=allowed_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.on_event("startup")
def startup_event() -> None:
    required = [
        MODELS_DIR / "perceptron.keras",
        MODELS_DIR / "ann.keras",
        MODELS_DIR / "cnn.keras",
    ]

    missing = [str(p.name) for p in required if not p.exists()]
    if missing:
        raise RuntimeError(
            "Missing model files: "
            + ", ".join(missing)
            + ". Run: python backend/train_models.py"
        )

    service.load()


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok" if service.is_loaded() else "models-not-loaded"}


def _to_response(source: str, arr_28x28) -> PredictResponse:
    predictions = service.predict_all(arr_28x28)
    votes = [predictions[k]["predicted_digit"] for k in ["perceptron", "ann", "cnn"]]
    agreement = len(set(votes)) == 1
    agreed_digit = int(votes[0]) if agreement else None

    return PredictResponse(
        source=source,
        predictions=predictions,
        agreement=agreement,
        agreed_digit=agreed_digit,
    )


@app.post("/predict/canvas", response_model=PredictResponse)
def predict_canvas(req: CanvasPredictRequest) -> PredictResponse:
    try:
        arr = base64_to_mnist_array(req.image_base64)
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=400, detail=f"Invalid base64 image: {exc}") from exc

    return _to_response("canvas", arr)


@app.post("/predict/upload", response_model=PredictResponse)
async def predict_upload(file: UploadFile = File(...)) -> PredictResponse:
    if not file.content_type or not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="Please upload an image file.")

    image_bytes = await file.read()
    if not image_bytes:
        raise HTTPException(status_code=400, detail="Uploaded file is empty.")

    try:
        arr = image_bytes_to_mnist_array(image_bytes)
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=400, detail=f"Could not parse image: {exc}") from exc

    return _to_response("upload", arr)


@app.post("/predict/typed", response_model=PredictResponse)
def predict_typed(req: TypedPredictRequest) -> PredictResponse:
    arr = service.sample_typed_digit(req.digit)
    response = _to_response("typed", arr)
    response.sampled_image_base64 = array_to_base64_png(arr)
    return response

from pydantic import BaseModel, Field


class ModelPrediction(BaseModel):
    predicted_digit: int = Field(..., ge=0, le=9)
    confidence: float = Field(..., ge=0.0, le=1.0)


class PredictResponse(BaseModel):
    source: str
    predictions: dict[str, ModelPrediction]
    agreement: bool
    agreed_digit: int | None
    sampled_image_base64: str | None = None


class CanvasPredictRequest(BaseModel):
    image_base64: str


class TypedPredictRequest(BaseModel):
    digit: int = Field(..., ge=0, le=9)

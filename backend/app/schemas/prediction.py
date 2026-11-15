"""Pydantic schemas for ASL inference API responses and health checks."""

from pydantic import BaseModel, Field


class TopKPrediction(BaseModel):
    class_id: int
    gloss: str
    confidence: float = Field(ge=0.0, le=1.0)
    score: float


class PredictionResponse(BaseModel):
    predicted_gloss: str
    predicted_class_id: int
    confidence: float = Field(ge=0.0, le=1.0)
    top_k: list[TopKPrediction]
    model_type: str
    latency_ms: float


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    device: str | None = None
    model_type: str | None = None
    num_classes: int | None = None


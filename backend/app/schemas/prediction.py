"""Pydantic schemas for ASL inference API responses, pose streaming, and sign generation."""

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


class PosePredictionRequest(BaseModel):
    """349-D or keypoint coordinate array for direct client-side stream prediction."""

    pose_data: list[list[float]] = Field(
        ...,
        description="Temporal sequence of keypoints/features, shape (T, 349).",
    )
    top_k: int = Field(default=5, ge=1, le=20)


class FramesPredictionRequest(BaseModel):
    """Temporal sequence of base64-encoded image frames from webcam or client canvas."""

    frames: list[str] = Field(
        ...,
        min_length=1,
        max_length=120,
        description="Temporal sequence of base64-encoded JPEG/PNG image frames or data URLs.",
    )
    top_k: int = Field(default=5, ge=1, le=20)


class ClassItem(BaseModel):
    class_id: int
    gloss: str
    description: str | None = None


class ClassListResponse(BaseModel):
    total: int
    classes: list[ClassItem]


class SignRetrievalMatch(BaseModel):
    class_id: int
    gloss: str
    similarity: float
    exact_match: bool = False


class TextToSignRequest(BaseModel):
    """Text prompt or query for the ASL Generator."""

    query: str = Field(
        ..., min_length=1, max_length=200, description="English word or phrase."
    )
    top_k: int = Field(default=5, ge=1, le=20)


class TextToSignResponse(BaseModel):
    query: str
    matched_gloss: str
    matched_class_id: int
    confidence: float
    top_matches: list[SignRetrievalMatch]
    latency_ms: float


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    device: str | None = None
    model_type: str | None = None
    num_classes: int | None = None

"""FastAPI application exposing ASL health and video prediction endpoints."""

import shutil
import tempfile
from pathlib import Path

from fastapi import FastAPI, File, HTTPException, Query, UploadFile

from backend.app.schemas.prediction import HealthResponse, PredictionResponse
from backend.app.services.inference import get_inference_service
from src.preprocessing.frames import SegmentBoundsError
from src.preprocessing.video import VideoReadError

ALLOWED_VIDEO_EXTENSIONS = {".mp4", ".webm", ".mkv", ".avi", ".mov"}

app = FastAPI(
    title="ASL Multimodal Recognition API",
    version="0.1.0",
    description="Inference API using identical segment-aware training preprocessing.",
)


@app.get("/health", response_model=HealthResponse)
def health_check() -> HealthResponse:
    """Report real service and model checkpoint readiness."""
    try:
        service = get_inference_service()
        return HealthResponse(
            status="ok",
            model_loaded=True,
            device=str(service.device),
            model_type=service.model_type,
            num_classes=len(service.id_to_class),
        )
    except Exception:
        return HealthResponse(status="model_not_loaded", model_loaded=False)


async def _handle_video_upload(
    file: UploadFile,
    top_k: int,
) -> PredictionResponse:
    if not file.filename:
        raise HTTPException(status_code=400, detail="Uploaded file has no filename.")

    suffix = Path(file.filename).suffix.lower() or ".mp4"
    if suffix not in ALLOWED_VIDEO_EXTENSIONS:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported video format '{suffix}'. Allowed: {sorted(ALLOWED_VIDEO_EXTENSIONS)}",
        )

    try:
        service = get_inference_service()
    except Exception as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc

    tmp_dir = tempfile.mkdtemp(prefix="asl_api_")
    tmp_path = Path(tmp_dir) / f"upload{suffix}"
    try:
        with tmp_path.open("wb") as out_f:
            shutil.copyfileobj(file.file, out_f)
        if tmp_path.stat().st_size == 0:
            raise HTTPException(status_code=400, detail="Uploaded video file is empty.")

        result = service.predict_video(tmp_path, top_k=top_k)
        return PredictionResponse(**result)
    except (VideoReadError, SegmentBoundsError) as exc:
        raise HTTPException(status_code=422, detail=f"Invalid video: {exc}") from exc
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)


@app.post("/predict", response_model=PredictionResponse)
async def predict(
    file: UploadFile = File(...),
    top_k: int = Query(default=5, ge=1, le=20),
) -> PredictionResponse:
    """Predict ASL gloss from an uploaded video clip."""
    return await _handle_video_upload(file, top_k)


@app.post("/predict/video", response_model=PredictionResponse)
async def predict_video_endpoint(
    file: UploadFile = File(...),
    top_k: int = Query(default=5, ge=1, le=20),
) -> PredictionResponse:
    """Alias endpoint for video-clip ASL gloss prediction."""
    return await _handle_video_upload(file, top_k)


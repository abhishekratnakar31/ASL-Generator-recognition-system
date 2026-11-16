"""FastAPI application exposing ASL recognition, pose streaming, and sign generation endpoints."""

import base64
import logging
import shutil
import tempfile
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from fastapi import FastAPI, File, Header, HTTPException, Query, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from backend.app.schemas.prediction import (
    ClassItem,
    ClassListResponse,
    FramesPredictionRequest,
    HealthResponse,
    PosePredictionRequest,
    PredictionResponse,
    TextToSignRequest,
    TextToSignResponse,
)
from backend.app.services.inference import get_inference_service
from src.preprocessing.frames import SegmentBoundsError
from src.preprocessing.video import VideoReadError

logger = logging.getLogger("asl.api")

ALLOWED_VIDEO_EXTENSIONS = {".mp4", ".webm", ".mkv", ".avi", ".mov"}

app = FastAPI(
    title="ASL Multimodal Recognition & Generation API",
    version="1.0.0",
    description=(
        "Production inference service for isolated American Sign Language (ASL) "
        "gesture recognition and vision-language representation alignment on WLASL-100."
    ),
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


FRONTEND_DIR = Path(__file__).resolve().parent.parent.parent / "frontend"

if FRONTEND_DIR.is_dir():
    app.mount("/frontend", StaticFiles(directory=str(FRONTEND_DIR)), name="frontend")


@app.get("/")
def root(accept: str | None = Header(default=None)) -> Any:
    """Root info endpoint (serves interactive UI when requested in a browser)."""
    index_file = FRONTEND_DIR / "index.html"
    if accept and "text/html" in accept and index_file.is_file():
        return FileResponse(index_file)
    return {
        "service": "ASL Multimodal Recognition & Generation API",
        "version": "1.0.0",
        "docs_url": "/docs",
        "health_url": "/health",
        "ui_url": "/app",
    }


@app.get("/app", include_in_schema=False)
def app_view() -> Any:
    """Direct route for interactive web application."""
    index_file = FRONTEND_DIR / "index.html"
    if index_file.is_file():
        return FileResponse(index_file)
    raise HTTPException(status_code=404, detail="Frontend index.html not found.")


@app.get("/health", response_model=HealthResponse)
def health_check() -> HealthResponse:
    """Report service health and model checkpoint readiness."""
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


@app.get("/api/v1/classes", response_model=ClassListResponse)
@app.get("/classes", response_model=ClassListResponse)
def list_classes(
    search: str | None = Query(default=None, description="Optional substring filter for gloss names."),
) -> ClassListResponse:
    """List all supported ASL sign classes."""
    try:
        service = get_inference_service()
        id_to_class = service.id_to_class
    except Exception as exc:
        raise HTTPException(status_code=503, detail=f"Model service unavailable: {exc}") from exc

    items: list[ClassItem] = []
    query = search.strip().lower() if search else None
    for cid in sorted(id_to_class.keys()):
        gloss = id_to_class[cid]
        if query is None or query in gloss.lower():
            items.append(ClassItem(class_id=cid, gloss=gloss))

    return ClassListResponse(total=len(items), classes=items)


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
    except HTTPException:
        raise
    except (VideoReadError, SegmentBoundsError) as exc:
        logger.error("Video decoding error: %s", exc, exc_info=True)
        raise HTTPException(status_code=422, detail=f"Invalid video: {exc}") from exc
    except Exception as exc:
        logger.error("Unexpected video prediction error: %s", exc, exc_info=True)
        raise HTTPException(status_code=500, detail=f"Prediction error: {exc}") from exc
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)


@app.post("/api/v1/predict/video", response_model=PredictionResponse)
@app.post("/predict/video", response_model=PredictionResponse)
@app.post("/predict", response_model=PredictionResponse)
async def predict_video(
    file: UploadFile = File(...),
    top_k: int = Query(default=5, ge=1, le=20),
) -> PredictionResponse:
    """Predict ASL gloss from an uploaded video clip (MP4, WebM, MOV, AVI)."""
    return await _handle_video_upload(file, top_k)


@app.post("/api/v1/predict/frames", response_model=PredictionResponse)
@app.post("/predict/frames", response_model=PredictionResponse)
def predict_frames_endpoint(
    payload: FramesPredictionRequest,
) -> PredictionResponse:
    """Predict ASL gloss directly from a sequence of base64-encoded image frames (e.g. from webcam canvas)."""
    try:
        service = get_inference_service()
    except Exception as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc

    images: list[np.ndarray] = []
    for f_str in payload.frames:
        if not f_str:
            continue
        if "," in f_str:
            f_str = f_str.split(",", 1)[1]
        try:
            raw_bytes = base64.b64decode(f_str)
            arr = np.frombuffer(raw_bytes, dtype=np.uint8)
            img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
            if img is not None:
                images.append(img)
        except Exception:
            continue

    if not images:
        raise HTTPException(status_code=400, detail="No decodable image frames found in payload.")

    try:
        result = service.predict_frames(images, top_k=payload.top_k)
        return PredictionResponse(**result)
    except Exception as exc:
        logger.error("Inference failed on frames payload: %s", exc, exc_info=True)
        raise HTTPException(status_code=422, detail=f"Inference failed on frames: {exc}") from exc


@app.post("/api/v1/predict/pose", response_model=PredictionResponse)
@app.post("/predict/pose", response_model=PredictionResponse)
def predict_pose_stream(
    payload: PosePredictionRequest,
) -> PredictionResponse:
    """Direct inference on pre-extracted (T, 349) keypoint kinematic features."""
    try:
        service = get_inference_service()
    except Exception as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc

    try:
        result = service.predict_pose(payload.pose_data, top_k=payload.top_k)
        return PredictionResponse(**result)
    except Exception as exc:
        raise HTTPException(status_code=422, detail=f"Inference failed on pose payload: {exc}") from exc


@app.post("/api/v1/generate/sign", response_model=TextToSignResponse)
@app.post("/generate/sign", response_model=TextToSignResponse)
def generate_sign_from_text(
    payload: TextToSignRequest,
) -> TextToSignResponse:
    """ASL Generator: Semantically retrieve the closest ASL signs for an English text prompt."""
    try:
        service = get_inference_service()
    except Exception as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc

    try:
        result = service.retrieve_sign_from_text(payload.query, top_k=payload.top_k)
        return TextToSignResponse(**result)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Sign retrieval failed: {exc}") from exc

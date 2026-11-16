"""Integration and unit tests for FastAPI ASL recognition, pose streaming, and generation endpoints."""

from pathlib import Path
from unittest.mock import MagicMock

import numpy as np
import pytest
import torch
from fastapi.testclient import TestClient

from backend.app.main import app
from backend.app.services import inference
from src.inference.predictor import ASLInferenceService


class DummyModel(torch.nn.Module):
    """Minimal dummy model returning deterministic logits for testing."""

    def __init__(self, num_classes: int = 5, in_dim: int = 349) -> None:
        super().__init__()
        self.linear = torch.nn.Linear(in_dim, num_classes)

    def forward(self, x: torch.Tensor, vocab_text_embeddings: torch.Tensor | None = None) -> dict[str, torch.Tensor]:
        pooled = x.mean(dim=1)
        return {"logits": self.linear(pooled)}


@pytest.fixture
def dummy_service() -> ASLInferenceService:
    model = DummyModel(num_classes=5, in_dim=349)
    id_to_class = {0: "book", 1: "drink", 2: "language", 3: "dark", 4: "sandwich"}
    dummy_vocab = torch.randn(5, 192)
    return ASLInferenceService(
        model=model,
        id_to_class=id_to_class,
        device=torch.device("cpu"),
        model_type="pose_transformer_only",
        vocab_text_embeddings=dummy_vocab,
    )


def test_root_endpoint():
    client = TestClient(app)
    response = client.get("/")
    assert response.status_code == 200
    data = response.json()
    assert "service" in data
    assert data["version"] == "1.0.0"


def test_health_endpoint_unloaded(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(
        "backend.app.main.get_inference_service",
        MagicMock(side_effect=RuntimeError("model not found")),
    )
    client = TestClient(app)
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "model_not_loaded"
    assert not data["model_loaded"]


def test_health_endpoint_loaded(dummy_service: ASLInferenceService, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(inference, "_SERVICE_INSTANCE", dummy_service)
    client = TestClient(app)
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "ok"
    assert data["model_loaded"] is True
    assert data["num_classes"] == 5
    assert data["model_type"] == "pose_transformer_only"


def test_list_classes_endpoint(dummy_service: ASLInferenceService, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(inference, "_SERVICE_INSTANCE", dummy_service)
    client = TestClient(app)

    # All classes
    res = client.get("/api/v1/classes")
    assert res.status_code == 200
    data = res.json()
    assert data["total"] == 5
    assert len(data["classes"]) == 5
    assert data["classes"][0]["gloss"] == "book"

    # Search filter
    res_search = client.get("/api/v1/classes?search=lang")
    assert res_search.status_code == 200
    search_data = res_search.json()
    assert search_data["total"] == 1
    assert search_data["classes"][0]["gloss"] == "language"


def test_predict_pose_endpoint(dummy_service: ASLInferenceService, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(inference, "_SERVICE_INSTANCE", dummy_service)
    client = TestClient(app)

    # 32 frames of 349-D random features
    fake_pose = np.random.randn(32, 349).tolist()
    payload = {"pose_data": fake_pose, "top_k": 3}

    res = client.post("/api/v1/predict/pose", json=payload)
    assert res.status_code == 200
    data = res.json()
    assert "predicted_gloss" in data
    assert "confidence" in data
    assert len(data["top_k"]) == 3
    assert data["top_k"][0]["confidence"] >= data["top_k"][1]["confidence"]
    assert data["latency_ms"] >= 0.0


def test_generate_sign_endpoint(dummy_service: ASLInferenceService, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(inference, "_SERVICE_INSTANCE", dummy_service)
    client = TestClient(app)

    # Exact gloss match
    res = client.post("/api/v1/generate/sign", json={"query": "dark", "top_k": 3})
    assert res.status_code == 200
    data = res.json()
    assert data["query"] == "dark"
    assert data["matched_gloss"] == "dark"
    assert data["confidence"] == 1.0
    assert len(data["top_matches"]) == 3
    assert data["top_matches"][0]["exact_match"] is True

    # Empty query validation
    res_empty = client.post("/api/v1/generate/sign", json={"query": ""})
    assert res_empty.status_code == 422  # Pydantic validation error


def test_predict_video_unsupported_format(dummy_service: ASLInferenceService, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(inference, "_SERVICE_INSTANCE", dummy_service)
    client = TestClient(app)

    files = {"file": ("test.txt", b"not a video", "text/plain")}
    res = client.post("/api/v1/predict/video", files=files)
    assert res.status_code == 400
    assert "Unsupported video format" in res.json()["detail"]


def test_predict_video_empty_file(dummy_service: ASLInferenceService, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(inference, "_SERVICE_INSTANCE", dummy_service)
    client = TestClient(app)

    files = {"file": ("empty.mp4", b"", "video/mp4")}
    res = client.post("/api/v1/predict/video", files=files)
    assert res.status_code == 400
    assert "empty" in res.json()["detail"].lower()


def test_predict_frames_endpoint(dummy_service: ASLInferenceService, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(inference, "_SERVICE_INSTANCE", dummy_service)
    client = TestClient(app)

    import base64
    import cv2

    frames = []
    for _ in range(5):
        img = np.random.randint(0, 255, (64, 64, 3), dtype=np.uint8)
        ok, buf = cv2.imencode(".jpg", img)
        assert ok
        frames.append("data:image/jpeg;base64," + base64.b64encode(buf).decode("ascii"))

    # Mock predict_frames on dummy service
    dummy_service.predict_frames = MagicMock(
        return_value={
            "predicted_gloss": "dark",
            "predicted_class_id": 3,
            "confidence": 0.95,
            "top_k": [{"class_id": 3, "gloss": "dark", "confidence": 0.95, "score": 2.5}],
            "model_type": "pose_transformer_only",
            "latency_ms": 12.3,
        }
    )

    res = client.post("/api/v1/predict/frames", json={"frames": frames, "top_k": 3})
    assert res.status_code == 200
    data = res.json()
    assert data["predicted_gloss"] == "dark"
    assert data["confidence"] == 0.95
    assert len(data["top_k"]) == 1

    # Empty frames validation
    res_empty = client.post("/api/v1/predict/frames", json={"frames": []})
    assert res_empty.status_code == 422


def test_frontend_ui_serving():
    client = TestClient(app)

    # HTML browser request to root
    res_html = client.get("/", headers={"accept": "text/html,application/xhtml+xml"})
    assert res_html.status_code == 200
    assert "ASL Multimodal System" in res_html.text

    # Dedicated /app route
    res_app = client.get("/app")
    assert res_app.status_code == 200
    assert "ASL Multimodal System" in res_app.text

    # Static CSS and JS assets
    res_css = client.get("/frontend/styles.css")
    assert res_css.status_code == 200
    assert "--accent-primary" in res_css.text

    res_js = client.get("/frontend/app.js")
    assert res_js.status_code == 200
    assert "PERFECT_F1_CLASSES" in res_js.text



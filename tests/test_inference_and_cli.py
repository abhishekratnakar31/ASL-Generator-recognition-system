"""Unit tests for checkpoint inference loading, prediction outputs, and YAML configs."""

from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import pytest

ROOT_DIR = Path(__file__).resolve().parent.parent

if TYPE_CHECKING:
    import cv2
    import torch

    HAS_TORCH_CV2 = True
else:
    try:
        import cv2
        import torch

        HAS_TORCH_CV2 = True
    except ImportError:
        cv2 = None  # type: ignore[assignment]
        torch = None  # type: ignore[assignment]
        HAS_TORCH_CV2 = False


def test_inference_missing_checkpoint_raises(tmp_path: Path):
    if not HAS_TORCH_CV2:
        pytest.skip("torch/cv2 not installed")
    from src.inference.predictor import ASLInferenceService

    with pytest.raises(FileNotFoundError):
        ASLInferenceService.from_checkpoint(tmp_path / "nonexistent.pt")


def test_yaml_configs_exist_and_load():
    yaml = pytest.importorskip("yaml")
    for name in ["baseline.yaml", "resnet_lstm.yaml", "multimodal.yaml"]:
        cfg_path = ROOT_DIR / "configs" / name
        assert cfg_path.is_file()
        data = yaml.safe_load(cfg_path.read_text(encoding="utf-8"))
        assert isinstance(data, dict)


@pytest.mark.skipif(not HAS_TORCH_CV2, reason="torch and cv2 required")
def test_visual_and_multimodal_checkpoint_inference_roundtrip(tmp_path: Path):
    from src.inference.predictor import ASLInferenceService
    from src.models.multimodal.multimodal_model import MultimodalASLModel
    from src.models.visual.visual_model import VisualClassifier

    id_to_class = {0: "book", 1: "drink", 2: "computer"}
    class_to_id = {"book": 0, "drink": 1, "computer": 2}

    # 1. Visual checkpoint roundtrip
    vis_model = VisualClassifier(num_classes=3, feature_dim=32, hidden_dim=16, num_layers=1)
    vis_ckpt = tmp_path / "vis.pt"
    torch.save(
        {
            "epoch": 1,
            "model_state_dict": vis_model.state_dict(),
            "class_to_id": class_to_id,
            "id_to_class": id_to_class,
            "config": {
                "feature_dim": 32,
                "hidden_size": 16,
                "num_layers": 1,
                "dropout": 0.3,
                "temporal": "lstm",
                "pooling": "last",
            },
        },
        vis_ckpt,
    )
    vis_svc = ASLInferenceService.from_checkpoint(
        vis_ckpt, device="cpu", pretrained_backbone=False
    )
    res_v = vis_svc.predict_tensor(torch.randn(1, 8, 32), top_k=3)
    assert res_v["model_type"] == "visual_only"
    assert res_v["predicted_gloss"] in class_to_id
    assert len(res_v["top_k"]) == 3
    assert sum(p["confidence"] for p in res_v["top_k"]) == pytest.approx(1.0, abs=1e-4)

    # 2. Multimodal checkpoint roundtrip + real synthetic MP4 end-to-end video prediction
    mm_model = MultimodalASLModel(
        num_classes=3,
        visual_feature_dim=2048,
        temporal_hidden_dim=16,
        temporal_layers=1,
        text_dim=24,
        embed_dim=12,
        fusion_mode="hybrid",
    )
    vocab_path = tmp_path / "vocab.pt"
    torch.save({"text_embeddings": torch.randn(3, 24), "class_to_id": class_to_id}, vocab_path)
    mm_ckpt = tmp_path / "mm.pt"
    torch.save(
        {
            "epoch": 2,
            "model_state_dict": mm_model.state_dict(),
            "class_to_id": class_to_id,
            "id_to_class": id_to_class,
            "config": {
                "feature_dim": 2048,
                "hidden_size": 16,
                "num_layers": 1,
                "embed_dim": 12,
                "dropout": 0.3,
                "temporal": "lstm",
                "pooling": "attention",
                "fusion_mode": "hybrid",
            },
        },
        mm_ckpt,
    )
    mm_svc = ASLInferenceService.from_checkpoint(
        mm_ckpt, vocab_embeddings_path=vocab_path, device="cpu", pretrained_backbone=False
    )

    vid_path = tmp_path / "clip.mp4"
    fourcc = getattr(cv2, "VideoWriter_fourcc")(*"mp4v")
    writer = cv2.VideoWriter(str(vid_path), fourcc, 25.0, (64, 64))
    assert writer.isOpened()
    for _ in range(36):
        writer.write(np.full((64, 64, 3), 128, dtype=np.uint8))
    writer.release()

    res_mm = mm_svc.predict_video(vid_path, top_k=2)
    assert res_mm["model_type"] == "multimodal"
    assert len(res_mm["top_k"]) == 2
    assert 0.0 <= res_mm["confidence"] <= 1.0

    # 3. Pose checkpoint roundtrip
    from src.models.multimodal.pose_multimodal_model import PoseVisualMultimodalModel

    pose_model = PoseVisualMultimodalModel(
        num_classes=3,
        pose_dim=349,
        hidden_dim=32,
        mode="pose_only",
    )
    pose_ckpt = tmp_path / "pose.pt"
    torch.save(
        {
            "epoch": 3,
            "model_state_dict": pose_model.state_dict(),
            "class_to_id": class_to_id,
            "id_to_class": id_to_class,
            "config": {
                "pose_dim": 349,
                "hidden_dim": 32,
                "mode": "pose_only",
            },
        },
        pose_ckpt,
    )
    pose_svc = ASLInferenceService.from_checkpoint(pose_ckpt, device="cpu")
    res_p = pose_svc.predict_tensor(torch.randn(1, 32, 349), top_k=2)
    assert res_p["model_type"] == "pose_pose_only"
    assert len(res_p["top_k"]) == 2

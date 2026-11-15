"""Unit tests for pose/hand keypoint normalization, augmentation, and triple-stream fusion."""

from typing import TYPE_CHECKING

import numpy as np
import pytest

from src.preprocessing.pose import (
    NUM_ASL_KEYPOINTS,
    POSE_FEATURE_DIM,
    augment_pose_sequence,
    normalize_asl_keypoints_sequence,
    parse_bbox_xyxy,
    select_primary_signer,
)

if TYPE_CHECKING:
    import torch

    HAS_TORCH = True
else:
    try:
        import torch

        HAS_TORCH = True
    except ImportError:
        torch = None  # type: ignore[assignment]
        HAS_TORCH = False


def test_parse_bbox_and_select_primary_signer():
    box = parse_bbox_xyxy("[0.2, 0.1, 0.5, 0.8]", 200, 100, pad_ratio=0.0)
    assert box == pytest.approx([40.0, 10.0, 140.0, 90.0])
    assert parse_bbox_xyxy("", 200, 100) is None

    kpts = np.zeros((2, 133, 2), dtype=np.float32)
    kpts[1] = 50.0
    scores = np.zeros((2, 133), dtype=np.float32)
    scores[0] = 0.1
    scores[1] = 0.9
    best_k, best_s = select_primary_signer(kpts, scores)
    assert best_k.shape == (133, 2) and best_s.shape == (133,)
    assert float(best_k.mean()) == pytest.approx(50.0)


def test_pose_normalization_invariance_and_augmentation():
    raw_kpts = np.zeros((32, 133, 2), dtype=np.float32)
    raw_scores = np.ones((32, 133), dtype=np.float32) * 0.9
    raw_kpts[:, 5] = [100.0, 80.0]
    raw_kpts[:, 6] = [180.0, 80.0]
    raw_kpts[:, 91] = [90.0, 120.0]
    raw_kpts[:, 100] = [95.0, 100.0]
    raw_kpts[:, 112] = [190.0, 120.0]
    raw_kpts[:, 121] = [185.0, 100.0]

    seq1 = normalize_asl_keypoints_sequence(raw_kpts, raw_scores)
    # Translate signer by (+60, -25) -> shoulder-normalized coordinates must stay identical
    seq2 = normalize_asl_keypoints_sequence(raw_kpts + np.array([60.0, -25.0], dtype=np.float32), raw_scores)
    assert seq1.shape == (32, POSE_FEATURE_DIM)
    assert np.isfinite(seq1).all()
    assert np.allclose(seq1[:, : NUM_ASL_KEYPOINTS * 2], seq2[:, : NUM_ASL_KEYPOINTS * 2], atol=1e-5)

    # Zero-confidence fallback produces finite zeros without NaN
    zero_scores = np.zeros((32, 133), dtype=np.float32)
    seq_zero = normalize_asl_keypoints_sequence(raw_kpts, zero_scores)
    assert np.isfinite(seq_zero).all()

    aug = augment_pose_sequence(seq1, rng=np.random.default_rng(0))
    assert aug.shape == seq1.shape and np.isfinite(aug).all()


@pytest.mark.skipif(not HAS_TORCH, reason="PyTorch required")
def test_pose_multimodal_model_all_modes_backward():
    from src.models.multimodal.pose_multimodal_model import PoseVisualMultimodalModel
    from src.training.losses import JointMultimodalLoss

    assert torch is not None
    pose_x = torch.randn(4, 32, POSE_FEATURE_DIM)
    vis_x = torch.randn(4, 32, 64)
    vocab = torch.randn(10, 48)
    labels = torch.tensor([0, 1, 2, 0], dtype=torch.long)

    m_only = PoseVisualMultimodalModel(10, pose_dim=POSE_FEATURE_DIM, hidden_dim=32, mode="pose_only")
    out_only = m_only(pose_x)
    assert out_only["logits"].shape == (4, 10)
    torch.nn.functional.cross_entropy(out_only["logits"], labels).backward()

    crit = JointMultimodalLoss()
    for mode in ("pose_language", "pose_visual_language"):
        m = PoseVisualMultimodalModel(
            10,
            pose_dim=POSE_FEATURE_DIM,
            visual_dim=64,
            text_dim=48,
            hidden_dim=32,
            embed_dim=24,
            mode=mode,
        )
        out = m(
            pose_x,
            visual_inputs=vis_x if mode == "pose_visual_language" else None,
            vocab_text_embeddings=vocab,
        )
        assert out["logits"].shape == (4, 10)
        loss, _ = crit(out, labels)
        assert torch.isfinite(loss)
        loss.backward()

    # Test boosted architectures: BiLSTM backbone & Cosine classifier
    m_bilstm_cos = PoseVisualMultimodalModel(
        10,
        pose_dim=POSE_FEATURE_DIM,
        hidden_dim=32,
        mode="pose_only",
        backbone="bilstm",
        use_cosine_classifier=True,
    )
    out_b = m_bilstm_cos(pose_x)
    assert out_b["logits"].shape == (4, 10)
    torch.nn.functional.cross_entropy(out_b["logits"], labels).backward()

    m_mm_cos = PoseVisualMultimodalModel(
        10,
        pose_dim=POSE_FEATURE_DIM,
        text_dim=48,
        hidden_dim=32,
        embed_dim=24,
        mode="pose_language",
        backbone="transformer",
        use_cosine_classifier=True,
    )
    out_mm_cos = m_mm_cos(pose_x, vocab_text_embeddings=vocab)
    assert out_mm_cos["logits"].shape == (4, 10)
    loss_mm, _ = crit(out_mm_cos, labels)
    assert torch.isfinite(loss_mm)
    loss_mm.backward()

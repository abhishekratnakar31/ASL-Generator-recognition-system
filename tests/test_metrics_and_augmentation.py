"""Unit tests for evaluation metrics, balanced sampling, and ASL augmentations."""

import numpy as np
import pytest

from src.data.sampler import compute_inverse_frequency_weights, compute_sample_weights
from src.evaluation.metrics import (
    compute_classification_metrics,
    compute_confusion_matrix,
    compute_majority_and_random_baselines,
    compute_per_class_metrics,
    compute_retrieval_metrics,
)
from src.preprocessing.augmentation import (
    augment_feature_sequence,
    augment_video_frames_np,
    temporal_crop_resample,
    temporal_jitter_indices,
)

try:
    import cv2  # noqa: F401

    HAS_CV2 = True
except ImportError:
    HAS_CV2 = False


def test_classification_and_per_class_metrics():
    y_true = np.array([0, 0, 1, 2])
    scores = np.array(
        [
            [5.0, 1.0, 0.0],  # pred 0 (correct)
            [1.5, 2.0, 1.0],  # pred 1 (top-2 hits 0)
            [0.1, 4.0, 0.5],  # pred 1 (correct)
            [0.1, 0.2, 3.0],  # pred 2 (correct)
        ]
    )
    res = compute_classification_metrics(y_true, scores, top_k=2)
    assert res["top1_accuracy"] == pytest.approx(0.75)
    assert res["top2_accuracy"] == pytest.approx(1.0)
    assert res["unique_predicted_classes"] == 3
    assert res["macro_f1"] > 0.7

    cm = compute_confusion_matrix(y_true, np.argmax(scores, axis=1), num_classes=3)
    assert cm.tolist() == [[1, 1, 0], [0, 1, 0], [0, 0, 1]]

    per_cls = compute_per_class_metrics(
        y_true, np.argmax(scores, axis=1), num_classes=3, class_names=["a", "b", "c"]
    )
    assert per_cls[0]["gloss"] == "a"
    assert per_cls[0]["recall"] == pytest.approx(0.5)
    assert per_cls[2]["f1"] == pytest.approx(1.0)


def test_majority_and_random_baselines():
    y_train = np.array([0, 0, 0, 1, 2])
    y_eval = np.array([0, 1, 2, 0])
    base = compute_majority_and_random_baselines(y_train, y_eval, num_classes=10)
    assert base["majority_class_id"] == 0
    assert base["majority_top1"] == pytest.approx(0.5)
    assert base["uniform_random_top1"] == pytest.approx(0.1)
    assert base["uniform_random_top5"] == pytest.approx(0.5)


def test_retrieval_metrics():
    sim = np.array(
        [
            [0.9, 0.1, 0.0],  # rank 1 for label 0
            [0.3, 0.8, 0.2],  # rank 1 for label 1
            [0.7, 0.6, 0.4],  # rank 3 for label 2
        ]
    )
    q = np.array([0, 1, 2])
    g = np.array([0, 1, 2])
    ret = compute_retrieval_metrics(sim, q, g, ks=(1, 2, 5))
    assert ret["recall@1"] == pytest.approx(2.0 / 3.0)
    assert ret["recall@5"] == pytest.approx(1.0)
    assert ret["median_rank"] == pytest.approx(1.0)
    assert ret["mrr"] == pytest.approx((1.0 + 1.0 + 1.0 / 3.0) / 3.0)


def test_sampler_weights_handle_zero_count_classes():
    labels = np.array([0, 0, 0, 2])
    weights = compute_inverse_frequency_weights(labels, num_classes=4, power=1.0)
    assert weights[1] == 0.0
    assert weights[3] == 0.0
    assert weights[2] > weights[0] > 0.0

    sample_w = compute_sample_weights(labels, power=1.0)
    assert np.isclose(sample_w[0] * 3.0, sample_w[3])


def test_temporal_and_feature_augmentations():
    seq = np.arange(32 * 4, dtype=np.float32).reshape(32, 4)
    cropped = temporal_crop_resample(seq, target_len=32, crop_ratio_range=(0.75, 0.9), rng=42)
    assert cropped.shape == (32, 4)

    idx = temporal_jitter_indices(32, max_shift=1, rng=7)
    assert len(idx) == 32
    assert np.all(np.diff(idx) >= 0)  # monotonic non-decreasing

    feats = np.abs(np.random.default_rng(0).normal(size=(32, 16))).astype(np.float32)
    aug_f = augment_feature_sequence(feats, feature_dropout=0.1, noise_std=0.02, rng=1)
    assert aug_f.shape == (32, 16)
    assert np.all(aug_f >= 0.0)


def test_video_clip_augmentation_preserves_shape_and_disables_flip_by_default():
    clip = np.zeros((4, 32, 32, 3), dtype=np.uint8)
    clip[:, :, :16, :] = 200  # asymmetric left half bright
    out = augment_video_frames_np(
        clip,
        brightness_delta=0.05,
        contrast_range=(0.95, 1.05),
        spatial_crop_scale=(1.0, 1.0),
        allow_horizontal_flip=False,
        rng=123,
    )
    assert out.shape == clip.shape
    assert out.dtype == np.uint8
    assert out[:, :, :16, :].mean() > out[:, :, 16:, :].mean()


@pytest.mark.skipif(not HAS_CV2, reason="OpenCV not installed")
def test_video_clip_augmentation_with_spatial_crop():
    clip = np.full((4, 64, 64, 3), 128, dtype=np.uint8)
    out = augment_video_frames_np(clip, spatial_crop_scale=(0.85, 0.95), rng=99)
    assert out.shape == clip.shape

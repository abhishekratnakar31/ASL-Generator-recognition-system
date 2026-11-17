"""Unit tests for metadata loading, class mappings, split validation, and manifest mapping."""

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.data.split import (
    classify_segment_annotations,
    compute_class_distribution,
    compute_eligibility_thresholds,
    extract_canonical_url,
    load_class_mappings,
    map_features_by_manifest,
    normalize_video_id,
    validate_splits,
)


def _make_split_df(rows: list[dict]) -> pd.DataFrame:
    return pd.DataFrame(rows)


def test_normalize_video_id_and_url():
    assert normalize_video_id("07069") == "7069"
    assert normalize_video_id("7069.0") == "7069"
    assert normalize_video_id(17710) == "17710"
    assert normalize_video_id(np.nan) == ""
    assert (
        extract_canonical_url("https://www.youtube.com/watch?v=F5Wef1_PtLk&t=10s")
        == "youtube:F5Wef1_PtLk"
    )
    assert (
        extract_canonical_url("https://youtu.be/abcDEF12345") == "youtube:abcDEF12345"
    )


def test_load_class_mappings_valid_and_invalid(tmp_path: Path):
    c2i_path = tmp_path / "class_to_id.json"
    i2c_path = tmp_path / "id_to_class.json"
    c2i_path.write_text(json.dumps({"book": 0, "drink": 1}), encoding="utf-8")
    i2c_path.write_text(json.dumps({"0": "book", "1": "drink"}), encoding="utf-8")

    c2i, i2c = load_class_mappings(c2i_path, i2c_path)
    assert c2i == {"book": 0, "drink": 1}
    assert i2c == {0: "book", 1: "drink"}

    # Non-contiguous IDs must raise
    c2i_path.write_text(json.dumps({"book": 0, "drink": 2}), encoding="utf-8")
    i2c_path.write_text(json.dumps({"0": "book", "2": "drink"}), encoding="utf-8")
    with pytest.raises(ValueError, match="contiguous"):
        load_class_mappings(c2i_path, i2c_path)

    # Bijection mismatch must raise
    c2i_path.write_text(json.dumps({"book": 0, "drink": 1}), encoding="utf-8")
    i2c_path.write_text(json.dumps({"0": "drink", "1": "book"}), encoding="utf-8")
    with pytest.raises(ValueError, match="Bijection"):
        load_class_mappings(c2i_path, i2c_path)


def test_classify_segment_annotations():
    df = pd.DataFrame(
        [
            {"frame_start": 1, "frame_end": -1},
            {"frame_start": 5710, "frame_end": 5831},
            {"frame_start": 1, "frame_end": 77},
            {"frame_start": 2, "frame_end": -1},
        ]
    )
    kinds = classify_segment_annotations(df).tolist()
    assert kinds == ["full_video", "bounded_segment", "other", "other"]


def test_validate_splits_detects_leakage_and_missing_classes():
    c2i = {"book": 0, "drink": 1, "short": 2}
    train_df = _make_split_df(
        [
            {
                "gloss": "book",
                "video_id": "1",
                "signer_id": "s1",
                "frame_start": 1,
                "frame_end": -1,
                "fps": 25,
                "class_id": 0,
                "url": "https://www.youtube.com/watch?v=AAAAAAAAAAA",
            },
            {
                "gloss": "drink",
                "video_id": "2",
                "signer_id": "s1",
                "frame_start": 1,
                "frame_end": -1,
                "fps": 25,
                "class_id": 1,
                "url": "https://www.youtube.com/watch?v=BBBBBBBBBBB",
            },
        ]
    )
    val_df = _make_split_df(
        [
            {
                "gloss": "short",
                "video_id": "3",
                "signer_id": "s2",
                "frame_start": 100,
                "frame_end": 180,
                "fps": 25,
                "class_id": 2,
                "url": "https://www.youtube.com/watch?v=CCCCCCCCCCC",
            }
        ]
    )
    test_df = _make_split_df(
        [
            {
                "gloss": "book",
                "video_id": "4",
                "signer_id": "s3",
                "frame_start": 1,
                "frame_end": -1,
                "fps": 25,
                "class_id": 0,
                "url": "https://www.youtube.com/watch?v=DDDDDDDDDDD",
            }
        ]
    )

    summary = validate_splits(
        train_df, val_df, test_df, c2i, allow_missing_train_classes=True
    )
    assert summary["missing_train_classes"] == [2]
    assert not any(summary["signer_overlaps"].values())

    with pytest.raises(ValueError, match="absent from train"):
        validate_splits(
            train_df, val_df, test_df, c2i, allow_missing_train_classes=False
        )

    # Inject signer overlap
    leaked_val = val_df.copy()
    leaked_val["signer_id"] = "s1"
    with pytest.raises(ValueError, match="Signer leakage"):
        validate_splits(train_df, leaked_val, test_df, c2i)

    # Inject YouTube URL overlap across splits
    leaked_url_test = test_df.copy()
    leaked_url_test["url"] = "https://youtu.be/AAAAAAAAAAA"
    with pytest.raises(ValueError, match="URL overlap"):
        validate_splits(train_df, val_df, leaked_url_test, c2i)


def test_map_features_by_manifest_handles_shuffled_order_and_nan():
    # Intentionally reversed manifest order vs metadata order
    meta = pd.DataFrame(
        [
            {"video_id": "0010", "class_id": 0},
            {"video_id": "0020", "class_id": 1},
            {"video_id": "0030", "class_id": 2},
        ]
    )
    manifest = pd.DataFrame(
        [
            {"video_id": "30", "label": 2},
            {"video_id": "10", "label": 0},
            {"video_id": "20", "label": 1},
        ]
    )
    features = np.array(
        [
            [[3.0, 3.0]],
            [[1.0, 1.0]],
            [[2.0, 2.0]],
        ],
        dtype=np.float32,
    )
    labels = np.array([2, 0, 1], dtype=np.int64)

    aligned_x, aligned_y = map_features_by_manifest(
        meta, manifest, features, labels, chunk_size=2
    )
    assert aligned_y.tolist() == [0, 1, 2]
    assert np.allclose(aligned_x[:, 0, 0], [1.0, 2.0, 3.0])

    # NaN injection must fail
    bad_features = features.copy()
    bad_features[1, 0, 0] = np.nan
    with pytest.raises(ValueError, match="NaN or Inf"):
        map_features_by_manifest(meta, manifest, bad_features, labels)


def test_class_distribution_and_thresholds():
    i2c = {0: "a", 1: "b"}
    train_df = pd.DataFrame({"class_id": [0, 0, 0, 1]})
    val_df = pd.DataFrame({"class_id": [0, 0, 1]})
    test_df = pd.DataFrame({"class_id": [0, 0]})

    dist = compute_class_distribution(train_df, val_df, test_df, i2c)
    assert dist["total_count"].tolist() == [7, 2]

    thresh = compute_eligibility_thresholds(dist, [(3, 2, 2), (1, 1, 1)])
    assert thresh["eligible_classes"].tolist() == [1, 1]

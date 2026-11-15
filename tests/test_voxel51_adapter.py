"""Unit tests for the Voxel51/WLASL Top-100 signer-disjoint dataset adapter."""

import json
from pathlib import Path

import pytest

from src.data.split import load_class_mappings, validate_splits
from src.data.voxel51_adapter import (
    build_signer_disjoint_splits,
    export_wlasl100_metadata,
    load_wlasl_v03_lookup,
    merge_voxel51_with_wlasl_metadata,
    parse_voxel51_samples,
)


def _write_synthetic_dataset(tmp_path: Path):
    glosses = ["book", "drink", "computer", "extra_rare"]
    counts = {"book": 8, "drink": 7, "computer": 7, "extra_rare": 2}
    samples = []
    wlasl_entries = []
    vid = 100

    for g in glosses:
        insts = []
        for idx in range(counts[g]):
            vid += 1
            signer = f"s{(idx % 5) + 1}"
            # Put two distinct signers onto separate YouTube IDs, plus keep 1 shared YouTube URL within s1
            yt = "AAAAAAAAAAA" if signer == "s1" else f"YT{vid:09d}"
            samples.append(
                {
                    "filepath": f"data/data_0/{vid:05d}.mp4",
                    "gloss": {"_cls": "Classification", "label": g},
                    "metadata": {
                        "frame_rate": 29.97,
                        "total_frame_count": 48,
                        "duration": 1.6,
                        "frame_width": 320,
                        "frame_height": 240,
                        "size_bytes": 45000,
                    },
                    "bounding_box": {
                        "_cls": "Detections",
                        "detections": [{"bounding_box": [0.1, 0.1, 0.8, 0.8]}],
                    },
                }
            )
            insts.append(
                {
                    "video_id": f"{vid:05d}",
                    "signer_id": signer,
                    "split": "train",
                    "url": f"https://www.youtube.com/watch?v={yt}",
                    "source": "synthetic",
                }
            )
        wlasl_entries.append({"gloss": g, "instances": insts})

    samples_path = tmp_path / "samples.json"
    wlasl_path = tmp_path / "WLASL_v0.3.json"
    samples_path.write_text(json.dumps({"samples": samples}), encoding="utf-8")
    wlasl_path.write_text(json.dumps(wlasl_entries), encoding="utf-8")
    return samples_path, wlasl_path


def test_parse_merge_and_build_signer_disjoint_splits(tmp_path: Path):
    samples_path, wlasl_path = _write_synthetic_dataset(tmp_path)

    top_df = parse_voxel51_samples(samples_path, top_k=3)
    assert set(top_df["gloss"]) == {"book", "drink", "computer"}
    assert len(top_df) == 22

    lookup = load_wlasl_v03_lookup(wlasl_path)
    master_df, c2i = merge_voxel51_with_wlasl_metadata(top_df, lookup)
    assert len(c2i) == 3
    assert len(master_df) == 22

    train_df, val_df, test_df, c2i, i2c, audit = build_signer_disjoint_splits(
        master_df,
        min_train_per_class=2,
        seed=42,
    )
    assert len(train_df) + len(val_df) + len(test_df) == 22
    assert audit["missing_train_classes"] == []
    assert not any(audit["signer_overlaps"].values())
    assert not any(audit["url_overlaps"].values())
    assert audit["val_min_per_class"] >= 1
    assert audit["test_min_per_class"] >= 1

    out_dir = tmp_path / "exported"
    paths = export_wlasl100_metadata(
        out_dir, train_df, val_df, test_df, master_df, c2i, i2c, audit
    )
    loaded_c2i, loaded_i2c = load_class_mappings(paths["class_to_id"], paths["id_to_class"])
    assert loaded_c2i == c2i
    assert loaded_i2c == i2c
    reval = validate_splits(train_df, val_df, test_df, loaded_c2i, allow_missing_train_classes=False)
    assert reval["missing_train_classes"] == []


def test_merge_detects_gloss_mismatch_and_missing_signer(tmp_path: Path):
    samples_path, wlasl_path = _write_synthetic_dataset(tmp_path)
    top_df = parse_voxel51_samples(samples_path, top_k=3)
    lookup = load_wlasl_v03_lookup(wlasl_path)

    first_vid = str(top_df.iloc[0]["video_id"])
    lookup_bad = dict(lookup)
    lookup_bad[first_vid] = {**lookup[first_vid], "wlasl_gloss": "wrong_word"}
    with pytest.raises(ValueError, match="gloss mismatches"):
        merge_voxel51_with_wlasl_metadata(top_df, lookup_bad)

    lookup_missing = dict(lookup)
    lookup_missing.pop(first_vid)
    with pytest.raises(ValueError, match="missing signer_id"):
        merge_voxel51_with_wlasl_metadata(top_df, lookup_missing)


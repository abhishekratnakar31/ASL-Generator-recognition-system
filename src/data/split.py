"""Signer-independent split, metadata, and manifest validation utilities."""

import json
import re
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

REQUIRED_SPLIT_COLUMNS = {
    "gloss",
    "video_id",
    "signer_id",
    "frame_start",
    "frame_end",
    "fps",
    "class_id",
}
_YT_PATTERN = re.compile(r"(?:v=|youtu\.be/)([A-Za-z0-9_-]{11})")


def normalize_video_id(value: Any) -> str:
    """Normalize numeric or string video IDs into a canonical string."""
    if pd.isna(value):
        return ""
    text = str(value).strip()
    if text.endswith(".0") and text[:-2].isdigit():
        text = text[:-2]
    return str(int(text)) if text.isdigit() else text


def extract_canonical_url(url: Any) -> str:
    """Return canonical YouTube video ID if present, otherwise trimmed URL."""
    if pd.isna(url):
        return ""
    text = str(url).strip()
    match = _YT_PATTERN.search(text)
    return f"youtube:{match.group(1)}" if match else text


def load_class_mappings(
    class_to_id_path: Path,
    id_to_class_path: Path,
) -> tuple[dict[str, int], dict[int, str]]:
    """Load and validate contiguous bidirectional class mappings."""
    with Path(class_to_id_path).open("r", encoding="utf-8") as handle:
        raw_c2i = json.load(handle)
    with Path(id_to_class_path).open("r", encoding="utf-8") as handle:
        raw_i2c = json.load(handle)

    class_to_id = {str(k): int(v) for k, v in raw_c2i.items()}
    id_to_class = {int(k): str(v) for k, v in raw_i2c.items()}

    num_classes = len(class_to_id)
    if num_classes == 0:
        raise ValueError("class_to_id mapping is empty.")
    if len(id_to_class) != num_classes:
        raise ValueError(f"Mapping size mismatch: {num_classes} vs {len(id_to_class)}.")

    expected_ids = set(range(num_classes))
    if set(class_to_id.values()) != expected_ids:
        raise ValueError("class_to_id IDs must be contiguous from 0 to N-1.")
    if set(id_to_class.keys()) != expected_ids:
        raise ValueError("id_to_class keys must be contiguous from 0 to N-1.")

    for gloss, cid in class_to_id.items():
        if id_to_class[cid] != gloss:
            raise ValueError(
                f"Bijection mismatch for {gloss!r} ({cid} -> {id_to_class[cid]!r})."
            )

    return class_to_id, id_to_class


def classify_segment_annotations(df: pd.DataFrame) -> pd.Series:
    """Classify rows as 'full_video', 'bounded_segment', or 'other'."""
    frame_start = np.asarray(
        pd.to_numeric(df["frame_start"], errors="coerce"), dtype=np.float64
    )
    frame_end = np.asarray(
        pd.to_numeric(df["frame_end"], errors="coerce"), dtype=np.float64
    )

    is_full = (frame_start == 1) & (frame_end == -1)
    is_bounded = (frame_start > 1) & (frame_end > frame_start)

    labels = pd.Series("other", index=df.index, dtype="object")
    labels.iloc[is_full] = "full_video"
    labels.iloc[is_bounded] = "bounded_segment"
    return labels


def _normalized_id_series(df: pd.DataFrame) -> pd.Series:
    col = "video_id_normalized" if "video_id_normalized" in df.columns else "video_id"
    return pd.Series(
        [normalize_video_id(val) for val in df[col].tolist()],
        index=df.index,
        dtype="object",
    )


def validate_splits(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    class_to_id: dict[str, int],
    *,
    allow_missing_train_classes: bool = True,
) -> dict[str, Any]:
    """Validate schema, label consistency, signer independence, and leakage."""
    splits = {"train": train_df, "val": val_df, "test": test_df}
    signers_by_split: dict[str, set[str]] = {}
    videos_by_split: dict[str, set[str]] = {}
    urls_by_split: dict[str, set[str]] = {}
    classes_by_split: dict[str, set[int]] = {}
    split_counts: dict[str, int] = {}

    for split_name, df in splits.items():
        missing_cols = sorted(REQUIRED_SPLIT_COLUMNS - set(df.columns))
        if missing_cols:
            raise ValueError(f"{split_name} missing columns: {missing_cols}")

        for col in REQUIRED_SPLIT_COLUMNS:
            null_count = int(df[col].isna().sum())
            if null_count > 0:
                raise ValueError(
                    f"{split_name} contains {null_count} nulls in '{col}'."
                )

        vid_norm = _normalized_id_series(df)
        if (vid_norm == "").any():
            raise ValueError(f"{split_name} contains empty video IDs.")
        dup_count = int(vid_norm.duplicated().sum())
        if dup_count > 0:
            raise ValueError(f"{split_name} contains {dup_count} duplicate video IDs.")

        mapped_ids = pd.Series(
            [class_to_id.get(str(g)) for g in df["gloss"].tolist()],
            index=df.index,
        )
        if bool(mapped_ids.isna().any()):
            missing_glosses = sorted(
                df.loc[mapped_ids.isna(), "gloss"].astype(str).unique().tolist()
            )
            raise ValueError(f"{split_name} has unknown glosses: {missing_glosses[:5]}")

        actual_ids = df["class_id"].astype(int)
        if not (mapped_ids.astype(int) == actual_ids).all():
            raise ValueError(f"{split_name} has gloss-to-class_id mapping mismatches.")

        signers_by_split[split_name] = set(df["signer_id"].astype(str))
        videos_by_split[split_name] = set(vid_norm)
        classes_by_split[split_name] = set(actual_ids)
        split_counts[split_name] = len(df)
        if "url" in df.columns:
            urls_by_split[split_name] = {
                u for u in df["url"].map(extract_canonical_url) if u
            }
        else:
            urls_by_split[split_name] = set()

    pairs = [("train", "val"), ("train", "test"), ("val", "test")]
    signer_overlaps = {
        f"{a}_{b}": sorted(signers_by_split[a] & signers_by_split[b]) for a, b in pairs
    }
    if any(signer_overlaps.values()):
        raise ValueError(f"Signer leakage detected: {signer_overlaps}")

    video_overlaps = {
        f"{a}_{b}": sorted(videos_by_split[a] & videos_by_split[b]) for a, b in pairs
    }
    if any(video_overlaps.values()):
        raise ValueError(f"Video ID leakage detected: {video_overlaps}")

    url_overlaps = {
        f"{a}_{b}": sorted(urls_by_split[a] & urls_by_split[b]) for a, b in pairs
    }
    if any(url_overlaps.values()):
        raise ValueError(f"Cross-split source URL overlap detected: {url_overlaps}")

    all_used_classes = (
        classes_by_split["train"] | classes_by_split["val"] | classes_by_split["test"]
    )
    missing_train_classes = sorted(all_used_classes - classes_by_split["train"])
    if missing_train_classes and not allow_missing_train_classes:
        raise ValueError(f"Classes absent from train split: {missing_train_classes}")

    return {
        "split_counts": split_counts,
        "signer_counts": {k: len(v) for k, v in signers_by_split.items()},
        "class_counts": {k: len(v) for k, v in classes_by_split.items()},
        "signer_overlaps": signer_overlaps,
        "video_overlaps": video_overlaps,
        "url_overlaps": url_overlaps,
        "missing_train_classes": missing_train_classes,
    }


def compute_class_distribution(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    id_to_class: dict[int, str],
) -> pd.DataFrame:
    """Compute per-class sample counts across train, validation, and test splits."""
    num_classes = len(id_to_class)
    train_counts = {
        int(k): int(v)
        for k, v in train_df["class_id"].astype(int).value_counts().to_dict().items()
    }
    val_counts = {
        int(k): int(v)
        for k, v in val_df["class_id"].astype(int).value_counts().to_dict().items()
    }
    test_counts = {
        int(k): int(v)
        for k, v in test_df["class_id"].astype(int).value_counts().to_dict().items()
    }

    dist = pd.DataFrame({"class_id": np.arange(num_classes, dtype=int)})
    class_ids = dist["class_id"].astype(int).tolist()
    dist["gloss"] = [id_to_class[cid] for cid in class_ids]
    dist["train_count"] = [train_counts.get(cid, 0) for cid in class_ids]
    dist["val_count"] = [val_counts.get(cid, 0) for cid in class_ids]
    dist["test_count"] = [test_counts.get(cid, 0) for cid in class_ids]
    dist["total_count"] = dist["train_count"] + dist["val_count"] + dist["test_count"]
    return dist


def compute_eligibility_thresholds(
    class_dist_df: pd.DataFrame,
    thresholds: list[tuple[int, int, int]],
) -> pd.DataFrame:
    """Summarize how many classes and videos satisfy minimum split sample thresholds."""
    rows = []
    for min_tr, min_va, min_te in thresholds:
        mask = (
            (class_dist_df["train_count"] >= min_tr)
            & (class_dist_df["val_count"] >= min_va)
            & (class_dist_df["test_count"] >= min_te)
        )
        subset = class_dist_df.loc[mask]
        rows.append(
            {
                "min_train": min_tr,
                "min_val": min_va,
                "min_test": min_te,
                "eligible_classes": int(len(subset)),
                "train_videos": int(subset["train_count"].sum()),
                "val_videos": int(subset["val_count"].sum()),
                "test_videos": int(subset["test_count"].sum()),
                "total_videos": int(subset["total_count"].sum()),
                "avg_train_per_class": (
                    round(float(subset["train_count"].mean()), 2)
                    if len(subset)
                    else 0.0
                ),
            }
        )
    return pd.DataFrame(rows)


def map_features_by_manifest(
    metadata_df: pd.DataFrame,
    manifest_df: pd.DataFrame,
    features: np.ndarray,
    labels: np.ndarray | None = None,
    *,
    chunk_size: int = 512,
) -> tuple[np.ndarray, np.ndarray]:
    """Align cached features to metadata rows via manifest video_id (never positional)."""
    if len(manifest_df) != len(features):
        raise ValueError(
            f"Manifest rows ({len(manifest_df)}) != features rows ({len(features)})."
        )
    if labels is not None and len(labels) != len(manifest_df):
        raise ValueError(
            f"Labels rows ({len(labels)}) != manifest rows ({len(manifest_df)})."
        )

    man = manifest_df.copy()
    man["_vid_norm"] = man["video_id"].map(normalize_video_id)
    if man["_vid_norm"].duplicated().any():
        raise ValueError("Manifest contains duplicate normalized video_id entries.")
    man["_cache_idx"] = np.arange(len(man), dtype=np.int64)

    meta = metadata_df.copy()
    meta["_vid_norm"] = _normalized_id_series(meta)
    if meta["_vid_norm"].duplicated().any():
        raise ValueError("Metadata contains duplicate normalized video_id entries.")

    merged = meta.merge(
        man[["_vid_norm", "_cache_idx"]],
        on="_vid_norm",
        how="left",
        validate="one_to_one",
    )
    if bool(merged["_cache_idx"].isna().any()):
        missing = merged.loc[merged["_cache_idx"].isna(), "_vid_norm"].tolist()
        raise KeyError(f"Videos missing from feature manifest: {missing[:10]}")

    indices = merged["_cache_idx"].to_numpy(dtype=np.int64)
    expected_labels = merged["class_id"].to_numpy(dtype=np.int64)

    if labels is not None:
        cached_labels = np.asarray(labels[indices], dtype=np.int64)
        if not np.array_equal(cached_labels, expected_labels):
            raise ValueError("Cached labels disagree with metadata class_id.")
    elif "label" in man.columns:
        man_labels = man["label"].to_numpy(dtype=np.int64)[indices]
        if not np.array_equal(man_labels, expected_labels):
            raise ValueError("Manifest labels disagree with metadata class_id.")

    # Verify finiteness across selected indices without loading full array unnecessarily
    reduce_axes = tuple(range(1, features.ndim))
    for start in range(0, len(indices), chunk_size):
        batch_idx = indices[start : start + chunk_size]
        chunk = np.asarray(features[batch_idx])
        if not np.isfinite(chunk).all(axis=reduce_axes).all():
            raise ValueError("Feature cache contains NaN or Inf values.")

    aligned_features = np.asarray(features[indices])
    return aligned_features, expected_labels

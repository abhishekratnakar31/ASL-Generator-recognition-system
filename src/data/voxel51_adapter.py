"""Adapter converting Voxel51/WLASL FiftyOne metadata into signer-disjoint WLASL-100 splits."""

import json
import random
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.data.split import extract_canonical_url, normalize_video_id, validate_splits


def parse_voxel51_samples(
    samples_json_path: str | Path,
    *,
    top_k: int = 100,
) -> pd.DataFrame:
    """Load Voxel51 samples.json and keep the top-K glosses by clip count."""
    if top_k <= 0:
        raise ValueError(f"top_k must be positive, got {top_k}")

    with Path(samples_json_path).open("r", encoding="utf-8") as handle:
        raw = json.load(handle)

    samples = raw.get("samples", []) if isinstance(raw, dict) else raw
    if not samples:
        raise ValueError(f"No samples found in {samples_json_path}")

    gloss_counts: Counter[str] = Counter()
    parsed_rows: list[dict[str, Any]] = []

    for sample in samples:
        gloss_obj = sample.get("gloss")
        gloss = (
            str(gloss_obj.get("label", "")).strip()
            if isinstance(gloss_obj, dict)
            else str(gloss_obj or "").strip()
        )
        filepath = str(sample.get("filepath", "")).strip()
        if not gloss or not filepath:
            continue

        video_id = normalize_video_id(Path(filepath).stem)
        if not video_id:
            continue

        meta = sample.get("metadata") or {}
        det_list = (sample.get("bounding_box") or {}).get("detections") or []
        bbox = det_list[0].get("bounding_box") if det_list else None

        gloss_counts[gloss] += 1
        parsed_rows.append(
            {
                "video_id": video_id,
                "gloss": gloss,
                "hf_filepath": filepath,
                "frame_start": 1,
                "frame_end": -1,
                "fps": float(meta.get("frame_rate") or 25.0),
                "total_frames_meta": int(meta.get("total_frame_count") or 0),
                "duration_sec_meta": float(meta.get("duration") or 0.0),
                "width_meta": int(meta.get("frame_width") or 0),
                "height_meta": int(meta.get("frame_height") or 0),
                "size_bytes_meta": int(meta.get("size_bytes") or 0),
                "bbox_json": json.dumps(bbox) if bbox is not None else "",
            }
        )

    ranked_glosses = sorted(
        gloss_counts.items(),
        key=lambda item: (-item[1], item[0]),
    )[:top_k]
    selected_glosses = {gloss for gloss, _ in ranked_glosses}

    filtered = [row for row in parsed_rows if row["gloss"] in selected_glosses]
    df = pd.DataFrame(filtered)
    if df["video_id"].duplicated().any():
        dups = df.loc[df["video_id"].duplicated(), "video_id"].tolist()[:5]
        raise ValueError(f"Duplicate video_id values in Voxel51 samples: {dups}")

    return df.sort_values(["gloss", "video_id"]).reset_index(drop=True)


def load_wlasl_v03_lookup(wlasl_json_path: str | Path) -> dict[str, dict[str, Any]]:
    """Parse official WLASL_v0.3.json into a normalized video_id lookup table."""
    with Path(wlasl_json_path).open("r", encoding="utf-8") as handle:
        entries = json.load(handle)

    lookup: dict[str, dict[str, Any]] = {}
    for entry in entries:
        gloss = str(entry.get("gloss", "")).strip()
        for inst in entry.get("instances", []):
            vid = normalize_video_id(inst.get("video_id"))
            if not vid:
                continue
            signer = inst.get("signer_id")
            lookup[vid] = {
                "wlasl_gloss": gloss,
                "signer_id": "" if signer is None else str(signer),
                "split_official": str(inst.get("split", "")).strip(),
                "url": str(inst.get("url", "")).strip(),
                "source": str(inst.get("source", "")).strip(),
            }
    return lookup


def merge_voxel51_with_wlasl_metadata(
    samples_df: pd.DataFrame,
    wlasl_lookup: dict[str, dict[str, Any]],
) -> tuple[pd.DataFrame, dict[str, int]]:
    """Attach official signer_id and source URLs onto Voxel51 clips and verify glosses."""
    rows: list[dict[str, Any]] = []
    missing_ids: list[str] = []
    gloss_mismatches: list[tuple[str, str, str]] = []

    selected_glosses = sorted(samples_df["gloss"].astype(str).unique().tolist())
    class_to_id = {gloss: idx for idx, gloss in enumerate(selected_glosses)}

    for rec in samples_df.to_dict(orient="records"):
        vid = normalize_video_id(rec["video_id"])
        gloss = str(rec["gloss"])
        info = wlasl_lookup.get(vid)
        if info is None or not str(info.get("signer_id", "")).strip():
            missing_ids.append(vid)
            continue
        if info["wlasl_gloss"] != gloss:
            gloss_mismatches.append((vid, gloss, info["wlasl_gloss"]))
            continue

        merged = dict(rec)
        merged["video_id"] = vid
        merged["class_id"] = class_to_id[gloss]
        merged["signer_id"] = str(info["signer_id"])
        merged["split_official"] = str(info["split_official"])
        merged["url"] = str(info["url"])
        merged["canonical_url"] = extract_canonical_url(info["url"])
        merged["source"] = str(info["source"])
        rows.append(merged)

    if missing_ids:
        raise ValueError(
            f"{len(missing_ids)} Voxel51 video IDs missing signer_id in WLASL_v0.3: {missing_ids[:10]}"
        )
    if gloss_mismatches:
        raise ValueError(
            f"{len(gloss_mismatches)} gloss mismatches vs WLASL_v0.3: {gloss_mismatches[:5]}"
        )

    master_df = (
        pd.DataFrame(rows).sort_values(["class_id", "video_id"]).reset_index(drop=True)
    )
    return master_df, class_to_id


def _connected_signer_url_groups(df: pd.DataFrame) -> list[list[int]]:
    """Group row indices that share either signer_id or non-empty canonical_url."""
    parent = list(range(len(df)))

    def find(i: int) -> int:
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    def union(a: int, b: int) -> None:
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[rb] = ra

    by_signer: dict[str, int] = {}
    by_url: dict[str, int] = {}
    signers = df["signer_id"].astype(str).tolist()
    urls = (
        df["canonical_url"].fillna("").astype(str).tolist()
        if "canonical_url" in df.columns
        else [""] * len(df)
    )
    for idx, (sig, curl) in enumerate(zip(signers, urls)):
        if sig in by_signer:
            union(idx, by_signer[sig])
        else:
            by_signer[sig] = idx

        if curl:
            if curl in by_url:
                union(idx, by_url[curl])
            else:
                by_url[curl] = idx

    groups: dict[int, list[int]] = defaultdict(list)
    for idx in range(len(df)):
        groups[find(idx)].append(idx)

    return sorted(groups.values(), key=lambda g: (-len(g), g[0]))


def _score_split_matrix(
    counts: np.ndarray,
    total_per_class: np.ndarray,
    target_ratios: tuple[float, float, float],
    min_train_per_class: int,
) -> float:
    tr, va, te = counts[0], counts[1], counts[2]
    deficit_val1 = float(np.maximum(0, 1 - va).sum())
    deficit_te1 = float(np.maximum(0, 1 - te).sum())
    deficit_tr = float(np.maximum(0, min_train_per_class - tr).sum())
    deficit_val2 = float(np.maximum(0, 2 - va).sum())
    deficit_te2 = float(np.maximum(0, 2 - te).sum())

    total_all = max(1, int(total_per_class.sum()))
    target_vec = np.asarray(target_ratios, dtype=np.float64)
    sums = counts.sum(axis=1).astype(np.float64)
    size_err = float(((sums - target_vec * total_all) ** 2).sum()) / float(total_all**2)
    class_ratio_err = float(
        (
            (counts / np.maximum(1, total_per_class)[None, :] - target_vec[:, None])
            ** 2
        ).mean()
    )

    return (
        1e6 * (deficit_val1 + deficit_te1)
        + 1e4 * deficit_tr
        + 500.0 * (deficit_val2 + deficit_te2)
        + 100.0 * size_err
        + 50.0 * class_ratio_err
    )


def build_signer_disjoint_splits(
    master_df: pd.DataFrame,
    *,
    target_ratios: tuple[float, float, float] = (0.70, 0.15, 0.15),
    min_train_per_class: int = 5,
    num_restarts: int = 16,
    seed: int = 42,
) -> tuple[
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
    dict[str, int],
    dict[int, str],
    dict[str, Any],
]:
    """Partition master_df into signer- and URL-disjoint train/val/test splits."""
    if master_df.empty:
        raise ValueError("master_df is empty.")

    glosses = sorted(master_df["gloss"].astype(str).unique().tolist())
    class_to_id = {g: idx for idx, g in enumerate(glosses)}
    id_to_class = {idx: g for idx, g in enumerate(glosses)}

    df = master_df.copy().reset_index(drop=True)
    df["class_id"] = [class_to_id[str(g)] for g in df["gloss"].tolist()]
    if "canonical_url" not in df.columns:
        raw_urls = df["url"].tolist() if "url" in df.columns else [""] * len(df)
        df["canonical_url"] = [extract_canonical_url(u) for u in raw_urls]

    groups = _connected_signer_url_groups(df)
    num_groups = len(groups)
    num_classes = len(glosses)

    group_vecs = np.zeros((num_groups, num_classes), dtype=np.int64)
    class_arr = df["class_id"].to_numpy(dtype=np.int64)
    for gi, row_idxs in enumerate(groups):
        for r_idx in row_idxs:
            group_vecs[gi, class_arr[r_idx]] += 1

    tot_per_class = group_vecs.sum(axis=0)
    best_assign: list[int] | None = None
    best_score = float("inf")

    for restart in range(max(1, num_restarts)):
        rng = random.Random(seed + restart)
        order = list(range(num_groups))
        rng.shuffle(order)

        assign = [0] * num_groups
        counts = np.zeros((3, num_classes), dtype=np.int64)
        for gi in order:
            vec = group_vecs[gi]
            best_sp = 0
            best_val = float("inf")
            for sp in (0, 1, 2):
                counts[sp] += vec
                val = _score_split_matrix(
                    counts, tot_per_class, target_ratios, min_train_per_class
                )
                counts[sp] -= vec
                if val < best_val:
                    best_val = val
                    best_sp = sp
            assign[gi] = best_sp
            counts[best_sp] += vec

        improved = True
        while improved:
            improved = False
            cur = _score_split_matrix(
                counts, tot_per_class, target_ratios, min_train_per_class
            )
            for gi in range(num_groups):
                old = assign[gi]
                vec = group_vecs[gi]
                counts[old] -= vec
                for sp in (0, 1, 2):
                    if sp == old:
                        continue
                    counts[sp] += vec
                    nv = _score_split_matrix(
                        counts, tot_per_class, target_ratios, min_train_per_class
                    )
                    if nv + 1e-9 < cur:
                        assign[gi] = sp
                        cur = nv
                        improved = True
                        break
                    counts[sp] -= vec
                else:
                    counts[old] += vec

            if not improved:
                for g1 in range(num_groups):
                    for g2 in range(g1 + 1, num_groups):
                        s1, s2 = assign[g1], assign[g2]
                        if s1 == s2:
                            continue
                        delta = group_vecs[g2] - group_vecs[g1]
                        counts[s1] += delta
                        counts[s2] -= delta
                        nv = _score_split_matrix(
                            counts, tot_per_class, target_ratios, min_train_per_class
                        )
                        if nv + 1e-9 < cur:
                            assign[g1], assign[g2] = s2, s1
                            cur = nv
                            improved = True
                        else:
                            counts[s1] -= delta
                            counts[s2] += delta

        final_val = _score_split_matrix(
            counts, tot_per_class, target_ratios, min_train_per_class
        )
        if final_val < best_score:
            best_score = final_val
            best_assign = assign[:]

    assert best_assign is not None
    split_names = {0: "train", 1: "val", 2: "test"}
    row_splits = [""] * len(df)
    for gi, row_idxs in enumerate(groups):
        sp_name = split_names[best_assign[gi]]
        for r_idx in row_idxs:
            row_splits[r_idx] = sp_name

    df["split"] = row_splits
    train_df = df[df["split"] == "train"].reset_index(drop=True)
    val_df = df[df["split"] == "val"].reset_index(drop=True)
    test_df = df[df["split"] == "test"].reset_index(drop=True)

    validation = validate_splits(
        train_df,
        val_df,
        test_df,
        class_to_id,
        allow_missing_train_classes=False,
    )
    tr_counts = train_df["class_id"].astype(int).value_counts().to_dict()
    va_counts = val_df["class_id"].astype(int).value_counts().to_dict()
    te_counts = test_df["class_id"].astype(int).value_counts().to_dict()
    audit = {
        **validation,
        "num_classes": len(class_to_id),
        "total_clips": len(df),
        "train_min_per_class": min(
            int(tr_counts.get(cid, 0)) for cid in range(num_classes)
        ),
        "val_min_per_class": min(
            int(va_counts.get(cid, 0)) for cid in range(num_classes)
        ),
        "test_min_per_class": min(
            int(te_counts.get(cid, 0)) for cid in range(num_classes)
        ),
    }
    return train_df, val_df, test_df, class_to_id, id_to_class, audit


def export_wlasl100_metadata(
    output_dir: str | Path,
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    master_df: pd.DataFrame,
    class_to_id: dict[str, int],
    id_to_class: dict[int, str],
    audit_summary: dict[str, Any],
) -> dict[str, Path]:
    """Persist metadata CSVs, bidirectional class JSONs, and audit summary cleanly."""
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    paths = {
        "train": out / "train.csv",
        "val": out / "val.csv",
        "test": out / "test.csv",
        "master": out / "voxel51_wlasl100_master.csv",
        "class_to_id": out / "class_to_id.json",
        "id_to_class": out / "id_to_class.json",
        "audit": out / "signer_split_audit.json",
    }

    train_df.to_csv(paths["train"], index=False)
    val_df.to_csv(paths["val"], index=False)
    test_df.to_csv(paths["test"], index=False)
    master_df.to_csv(paths["master"], index=False)
    paths["class_to_id"].write_text(json.dumps(class_to_id, indent=2), encoding="utf-8")
    paths["id_to_class"].write_text(
        json.dumps({str(k): v for k, v in id_to_class.items()}, indent=2),
        encoding="utf-8",
    )
    paths["audit"].write_text(json.dumps(audit_summary, indent=2), encoding="utf-8")
    return paths

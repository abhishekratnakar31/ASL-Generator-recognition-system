"""Whole-body (upper body + 21L + 21R hands) pose extraction and normalization for ASL."""

import json
from typing import Any

import numpy as np

UPPER_BODY_INDICES = list(range(11))  # 0..10: nose, eyes, ears, shoulders, elbows, wrists
LEFT_HAND_INDICES = list(range(91, 112))  # 21 left-hand landmarks (91 = left wrist)
RIGHT_HAND_INDICES = list(range(112, 133))  # 21 right-hand landmarks (112 = right wrist)
ASL_KEYPOINT_INDICES = UPPER_BODY_INDICES + LEFT_HAND_INDICES + RIGHT_HAND_INDICES
NUM_ASL_KEYPOINTS = len(ASL_KEYPOINT_INDICES)  # 53
POSE_FEATURE_DIM = NUM_ASL_KEYPOINTS * 4 + 42 * 2 + NUM_ASL_KEYPOINTS  # 349


def parse_bbox_xyxy(
    bbox_json: str | None,
    width: int,
    height: int,
    *,
    pad_ratio: float = 0.08,
) -> list[float] | None:
    """Convert normalized FiftyOne [x, y, w, h] JSON into padded pixel [x1, y1, x2, y2]."""
    if not bbox_json or not str(bbox_json).strip():
        return None
    try:
        box = json.loads(str(bbox_json))
    except (TypeError, ValueError):
        return None
    if not isinstance(box, list) or len(box) != 4:
        return None

    x, y, bw, bh = [float(v) for v in box]
    if bw <= 0 or bh <= 0 or width <= 0 or height <= 0:
        return None

    x1 = max(0.0, (x - pad_ratio * bw) * width)
    y1 = max(0.0, (y - pad_ratio * bh) * height)
    x2 = min(float(width), (x + (1.0 + pad_ratio) * bw) * width)
    y2 = min(float(height), (y + (1.0 + pad_ratio) * bh) * height)
    if x2 <= x1 or y2 <= y1:
        return None
    return [x1, y1, x2, y2]


def normalize_asl_keypoints_sequence(
    raw_kpts: np.ndarray,
    raw_scores: np.ndarray,
    *,
    frame_width: int = 320,
    frame_height: int = 240,
    conf_threshold: float = 0.15,
) -> np.ndarray:
    """Convert raw COCO-WholeBody (T, 133, 2) + (T, 133) into (T, 349) normalized features.

    Produces shoulder-centered coordinates, hand-wrist-relative finger geometry,
    temporal velocities, and landmark confidences without background shortcuts.
    """
    kpts = np.asarray(raw_kpts, dtype=np.float32)
    scores = np.asarray(raw_scores, dtype=np.float32)
    if kpts.ndim != 3 or kpts.shape[1] < 133 or kpts.shape[2] < 2:
        raise ValueError(f"Expected (T, >=133, >=2) keypoints, got {kpts.shape}")
    if scores.shape[:2] != kpts.shape[:2]:
        raise ValueError(f"Score shape {scores.shape} != keypoint shape {kpts.shape[:2]}")

    t_len = kpts.shape[0]
    sel_xy = kpts[:, ASL_KEYPOINT_INDICES, :2]  # (T, 53, 2)
    sel_conf = np.clip(scores[:, ASL_KEYPOINT_INDICES], 0.0, 1.0)  # (T, 53)

    norm_xy = np.zeros((t_len, NUM_ASL_KEYPOINTS, 2), dtype=np.float32)
    hand_rel = np.zeros((t_len, 42, 2), dtype=np.float32)

    default_center = np.array([frame_width * 0.5, frame_height * 0.5], dtype=np.float32)
    default_scale = max(float(frame_width) * 0.25, 16.0)

    for t in range(t_len):
        xy = sel_xy[t]
        cf = sel_conf[t]
        valid = cf >= conf_threshold

        # Shoulder indices in sel_xy are 5 (left shoulder) and 6 (right shoulder)
        if valid[5] and valid[6]:
            center = 0.5 * (xy[5] + xy[6])
            scale = max(float(np.linalg.norm(xy[5] - xy[6])), 16.0)
        elif np.any(valid[:11]):
            ub_pts = xy[:11][valid[:11]]
            center = ub_pts.mean(axis=0)
            span = float(np.linalg.norm(ub_pts.max(axis=0) - ub_pts.min(axis=0)))
            scale = max(span * 0.5, default_scale)
        else:
            center = default_center
            scale = default_scale

        scaled = np.clip((xy - center) / scale, -5.0, 5.0)
        scaled[~valid] = 0.0
        norm_xy[t] = scaled

        # Left hand relative to left wrist (sel_xy indices 11..31; 11=wrist, 20=middle MCP)
        if valid[11]:
            lh_scale = (
                max(float(np.linalg.norm(xy[20] - xy[11])), 0.25 * scale)
                if valid[20]
                else 0.35 * scale
            )
            lh_rel = np.clip((xy[11:32] - xy[11:12]) / max(lh_scale, 8.0), -4.0, 4.0)
            lh_rel[~valid[11:32]] = 0.0
            hand_rel[t, :21] = lh_rel

        # Right hand relative to right wrist (sel_xy indices 32..52; 32=wrist, 41=middle MCP)
        if valid[32]:
            rh_scale = (
                max(float(np.linalg.norm(xy[41] - xy[32])), 0.25 * scale)
                if valid[41]
                else 0.35 * scale
            )
            rh_rel = np.clip((xy[32:53] - xy[32:33]) / max(rh_scale, 8.0), -4.0, 4.0)
            rh_rel[~valid[32:53]] = 0.0
            hand_rel[t, 21:] = rh_rel

    vel_xy = np.zeros_like(norm_xy)
    if t_len > 1:
        vel_xy[1:] = np.clip(norm_xy[1:] - norm_xy[:-1], -3.0, 3.0)

    out = np.concatenate(
        [
            norm_xy.reshape(t_len, NUM_ASL_KEYPOINTS * 2),
            hand_rel.reshape(t_len, 42 * 2),
            vel_xy.reshape(t_len, NUM_ASL_KEYPOINTS * 2),
            sel_conf,
        ],
        axis=-1,
    ).astype(np.float32, copy=False)
    return np.nan_to_num(out, nan=0.0, posinf=0.0, neginf=0.0)


def augment_pose_sequence(
    pose_seq: np.ndarray,
    *,
    crop_ratio_range: tuple[float, float] = (0.85, 1.0),
    coord_scale_range: tuple[float, float] = (0.90, 1.10),
    coord_shift_std: float = 0.04,
    jitter_std: float = 0.015,
    joint_dropout: float = 0.05,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    """Apply meaning-preserving temporal/scale/jitter augmentations to (T, 349) poses."""
    if pose_seq.ndim != 2 or pose_seq.shape[1] != POSE_FEATURE_DIM:
        raise ValueError(
            f"Expected (T, {POSE_FEATURE_DIM}) pose sequence, got {pose_seq.shape}"
        )

    gen = rng if rng is not None else np.random.default_rng()
    t_len = pose_seq.shape[0]
    ratio = float(gen.uniform(crop_ratio_range[0], crop_ratio_range[1]))
    win_len = max(1, min(t_len, int(round(t_len * ratio))))
    max_start = t_len - win_len
    start = int(gen.integers(0, max_start + 1)) if max_start > 0 else 0
    end = start + win_len - 1
    idxs = np.round(np.linspace(start, end, t_len)).astype(np.int64)
    aug = pose_seq[idxs].astype(np.float32, copy=True)

    # Slight global scale and shift across coordinate channels (exclude confidence channels)
    coord_end = POSE_FEATURE_DIM - NUM_ASL_KEYPOINTS
    scale_f = float(gen.uniform(coord_scale_range[0], coord_scale_range[1]))
    aug[:, :coord_end] *= scale_f
    if coord_shift_std > 0:
        shift = gen.normal(0.0, coord_shift_std, size=(1, 2)).astype(np.float32)
        norm_view = aug[:, : NUM_ASL_KEYPOINTS * 2].reshape(t_len, NUM_ASL_KEYPOINTS, 2)
        norm_view += shift.reshape(1, 1, 2)

    if jitter_std > 0:
        aug[:, :coord_end] += gen.normal(
            0.0, jitter_std, size=(t_len, coord_end)
        ).astype(np.float32)

    if joint_dropout > 0:
        drop = (gen.random((t_len, NUM_ASL_KEYPOINTS)) < joint_dropout)
        aug[:, POSE_FEATURE_DIM - NUM_ASL_KEYPOINTS :][drop] = 0.0

    # Recompute velocity slice from resampled/jittered coordinates for consistency
    norm_coords = aug[:, : NUM_ASL_KEYPOINTS * 2].reshape(t_len, NUM_ASL_KEYPOINTS, 2)
    vel_slice = aug[:, 190:296].reshape(t_len, NUM_ASL_KEYPOINTS, 2)
    vel_slice[0] = 0.0
    if t_len > 1:
        vel_slice[1:] = np.clip(norm_coords[1:] - norm_coords[:-1], -3.0, 3.0)

    return aug


def select_primary_signer(
    keypoints: Any,
    scores: Any,
) -> tuple[np.ndarray, np.ndarray]:
    """Select the highest-confidence person detection (133, 2) and (133,)."""
    kpts = np.asarray(keypoints, dtype=np.float32)
    sc = np.asarray(scores, dtype=np.float32)
    if kpts.ndim == 2:
        kpts = kpts[None, ...]
    if sc.ndim == 1:
        sc = sc[None, ...]
    if len(kpts) == 0 or len(sc) == 0:
        return np.zeros((133, 2), dtype=np.float32), np.zeros((133,), dtype=np.float32)

    mean_scores = sc[:, ASL_KEYPOINT_INDICES].mean(axis=1)
    best_idx = int(np.argmax(mean_scores))
    return kpts[best_idx, :133, :2], sc[best_idx, :133]


"""Annotation-to-frame mapping, temporal sampling, and frame normalization."""

from dataclasses import dataclass
from typing import Any

import numpy as np

IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


class SegmentBoundsError(ValueError):
    """Raised when segment frame coordinates fall completely outside the video."""


@dataclass(frozen=True)
class SegmentBounds:
    """Resolved 0-based inclusive segment coordinates and diagnostics."""

    start_idx: int
    end_idx: int
    annotation_type: str
    fps_converted: bool
    clamped_end: bool
    overflow_frames: int


def _parse_int_or_default(value: Any, default: int) -> int:
    if value is None:
        return default
    if isinstance(value, str) and not value.strip():
        return default
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return default
    if np.isnan(numeric):
        return default
    return int(numeric)


def resolve_segment_bounds(
    total_frames: int,
    frame_start: Any = 1,
    frame_end: Any = -1,
    *,
    metadata_fps: float | None = 25.0,
    actual_fps: float | None = None,
    convert_fps: bool = True,
    one_based: bool = True,
    strict: bool = True,
) -> SegmentBounds:
    """Resolve WLASL annotation coordinates into 0-based inclusive frame bounds."""
    if total_frames <= 0:
        raise SegmentBoundsError(f"Video has non-positive frame count: {total_frames}")

    raw_start = _parse_int_or_default(frame_start, 1)
    raw_end = _parse_int_or_default(frame_end, -1)

    # Full-video marker: frame_start <= 1 and frame_end < 0
    if raw_start <= 1 and raw_end < 0:
        return SegmentBounds(
            start_idx=0,
            end_idx=total_frames - 1,
            annotation_type="full_video",
            fps_converted=False,
            clamped_end=False,
            overflow_frames=0,
        )

    offset = 1 if one_based else 0
    src_start = max(0, raw_start - offset)

    can_scale_fps = False
    fps_scale = 1.0
    if convert_fps and metadata_fps is not None and actual_fps is not None:
        meta_fps_val = float(metadata_fps)
        act_fps_val = float(actual_fps)
        if (
            meta_fps_val > 0.0
            and act_fps_val > 0.0
            and abs(act_fps_val - meta_fps_val) > 1e-3
        ):
            can_scale_fps = True
            fps_scale = act_fps_val / meta_fps_val

    start_idx = int(round(src_start * fps_scale))

    if raw_end < 0:
        # Open-ended clip with non-trivial start offset (e.g., frame_start=2, frame_end=-1)
        if start_idx >= total_frames:
            if strict:
                raise SegmentBoundsError(
                    f"Start frame {start_idx} >= total_frames {total_frames}."
                )
            start_idx = 0
        return SegmentBounds(
            start_idx=start_idx,
            end_idx=total_frames - 1,
            annotation_type="open_end_segment",
            fps_converted=can_scale_fps,
            clamped_end=False,
            overflow_frames=0,
        )

    src_end = max(src_start, raw_end - offset)
    end_idx = int(round(src_end * fps_scale))

    if start_idx >= total_frames:
        raise SegmentBoundsError(
            f"Resolved start_idx={start_idx} >= total_frames={total_frames} "
            f"(raw=({raw_start}, {raw_end}), fps_scale={fps_scale:.4f})."
        )
    if end_idx < start_idx:
        raise SegmentBoundsError(
            f"Resolved end_idx={end_idx} < start_idx={start_idx}."
        )

    overflow = max(0, end_idx - (total_frames - 1))
    clamped = overflow > 0
    if clamped:
        if strict and overflow > 5:
            raise SegmentBoundsError(
                f"Segment end_idx={end_idx} exceeds total_frames-1={total_frames - 1} "
                f"by {overflow} frames."
            )
        end_idx = total_frames - 1

    return SegmentBounds(
        start_idx=start_idx,
        end_idx=end_idx,
        annotation_type="bounded_segment",
        fps_converted=can_scale_fps,
        clamped_end=clamped,
        overflow_frames=overflow,
    )


def sample_frame_indices(
    start_idx: int,
    end_idx: int,
    num_frames: int = 32,
) -> np.ndarray:
    """Uniformly sample `num_frames` integer indices in [start_idx, end_idx]."""
    if num_frames <= 0:
        raise ValueError(f"num_frames must be positive, got {num_frames}")
    if start_idx < 0 or end_idx < start_idx:
        raise ValueError(f"Invalid bounds: [{start_idx}, {end_idx}]")

    return np.round(np.linspace(start_idx, end_idx, num_frames)).astype(np.int64)


def normalize_rgb_frames(
    frames_rgb: np.ndarray,
    mean: np.ndarray = IMAGENET_MEAN,
    std: np.ndarray = IMAGENET_STD,
) -> np.ndarray:
    """Convert uint8/float RGB frames (T, H, W, 3) into ImageNet-normalized (T, 3, H, W)."""
    if frames_rgb.ndim != 4 or frames_rgb.shape[-1] != 3:
        raise ValueError(
            f"Expected shape (T, H, W, 3), got {frames_rgb.shape}."
        )
    arr = frames_rgb.astype(np.float32) / 255.0
    arr = (arr - mean.reshape(1, 1, 1, 3)) / std.reshape(1, 1, 1, 3)
    return np.transpose(arr, (0, 3, 1, 2)).astype(np.float32, copy=False)


def preprocess_bgr_frames(
    frames_bgr: list[np.ndarray],
    image_size: int = 224,
) -> np.ndarray:
    """Resize BGR frames to (image_size, image_size), convert to RGB, and normalize."""
    if not frames_bgr:
        raise ValueError("Cannot preprocess empty frame list.")

    import cv2

    resized_rgb = []
    for frame in frames_bgr:
        if frame is None or frame.ndim != 3 or frame.shape[2] != 3:
            raise ValueError("Each frame must be a valid HxWx3 BGR image.")
        resized = cv2.resize(
            frame, (image_size, image_size), interpolation=cv2.INTER_AREA
        )
        resized_rgb.append(cv2.cvtColor(resized, cv2.COLOR_BGR2RGB))

    return normalize_rgb_frames(np.stack(resized_rgb, axis=0))

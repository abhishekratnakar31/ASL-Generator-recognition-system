"""Meaning-preserving temporal and spatial augmentations for ASL clips."""

import numpy as np


def _ensure_rng(rng: np.random.Generator | int | None) -> np.random.Generator:
    if isinstance(rng, np.random.Generator):
        return rng
    return np.random.default_rng(rng)


def temporal_crop_resample(
    sequence: np.ndarray,
    target_len: int = 32,
    *,
    crop_ratio_range: tuple[float, float] = (0.8, 1.0),
    rng: np.random.Generator | int | None = None,
) -> np.ndarray:
    """Temporally crop a contiguous sub-window and resample to `target_len`."""
    if sequence.ndim < 2:
        raise ValueError(f"Expected sequence with >=2 dims, got {sequence.shape}")
    total_len = sequence.shape[0]
    if total_len == 0:
        raise ValueError("Cannot augment empty temporal sequence.")

    lo, hi = crop_ratio_range
    if not (0.0 < lo <= hi <= 1.0):
        raise ValueError(f"Invalid crop_ratio_range: {crop_ratio_range}")

    gen = _ensure_rng(rng)
    ratio = float(gen.uniform(lo, hi))
    win_len = max(1, min(total_len, int(round(total_len * ratio))))
    max_start = total_len - win_len
    start = int(gen.integers(0, max_start + 1)) if max_start > 0 else 0
    end = start + win_len - 1

    indices = np.round(np.linspace(start, end, target_len)).astype(np.int64)
    return sequence[indices]


def temporal_jitter_indices(
    num_frames: int,
    *,
    max_shift: int = 1,
    rng: np.random.Generator | int | None = None,
) -> np.ndarray:
    """Apply small monotonic index jitter within [0, num_frames - 1]."""
    if num_frames <= 0:
        raise ValueError(f"num_frames must be positive, got {num_frames}")
    gen = _ensure_rng(rng)
    base = np.arange(num_frames, dtype=np.int64)
    if max_shift <= 0:
        return base
    offsets = gen.integers(-max_shift, max_shift + 1, size=num_frames)
    jittered = np.clip(base + offsets, 0, num_frames - 1)
    return np.maximum.accumulate(jittered)


def augment_feature_sequence(
    features: np.ndarray,
    *,
    crop_ratio_range: tuple[float, float] = (0.8, 1.0),
    feature_dropout: float = 0.1,
    noise_std: float = 0.02,
    rng: np.random.Generator | int | None = None,
) -> np.ndarray:
    """Augment cached ResNet50 feature sequence (T, D) without changing sign identity."""
    if features.ndim != 2:
        raise ValueError(f"Expected 2D feature sequence (T, D), got {features.shape}")

    gen = _ensure_rng(rng)
    aug = temporal_crop_resample(
        features,
        target_len=features.shape[0],
        crop_ratio_range=crop_ratio_range,
        rng=gen,
    ).astype(np.float32, copy=True)

    if feature_dropout > 0.0:
        keep_prob = 1.0 - float(feature_dropout)
        mask = (gen.random(aug.shape) < keep_prob).astype(np.float32) / keep_prob
        aug = aug * mask

    if noise_std > 0.0:
        aug = aug + gen.normal(0.0, float(noise_std), size=aug.shape).astype(np.float32)
        # ResNet50 post-ReLU features are non-negative
        aug = np.maximum(aug, 0.0)

    return aug


def augment_video_frames_np(
    frames_thwc: np.ndarray,
    *,
    brightness_delta: float = 0.12,
    contrast_range: tuple[float, float] = (0.88, 1.12),
    spatial_crop_scale: tuple[float, float] = (0.90, 1.0),
    allow_horizontal_flip: bool = False,
    rng: np.random.Generator | int | None = None,
) -> np.ndarray:
    """Apply consistent clip-level spatial & photometric augmentation to (T, H, W, 3) uint8/float."""
    if frames_thwc.ndim != 4 or frames_thwc.shape[-1] != 3:
        raise ValueError(f"Expected (T, H, W, 3), got {frames_thwc.shape}")

    gen = _ensure_rng(rng)
    clip = frames_thwc.astype(np.float32, copy=True)
    is_uint8_scale = clip.max() > 1.5
    if is_uint8_scale:
        clip = clip / 255.0

    # Clip-consistent spatial zoom/crop (resizing via simple bilinear/cv2 if cropped)
    t_len, h, w, _ = clip.shape
    scale = float(gen.uniform(spatial_crop_scale[0], spatial_crop_scale[1]))
    if scale < 0.999 and h >= 16 and w >= 16:
        import cv2

        crop_h = max(8, int(round(h * scale)))
        crop_w = max(8, int(round(w * scale)))
        top = int(gen.integers(0, h - crop_h + 1)) if h > crop_h else 0
        left = int(gen.integers(0, w - crop_w + 1)) if w > crop_w else 0
        cropped = clip[:, top : top + crop_h, left : left + crop_w, :]
        clip = np.stack(
            [
                cv2.resize(cropped[i], (w, h), interpolation=cv2.INTER_LINEAR)
                for i in range(t_len)
            ],
            axis=0,
        )

    # Horizontal flipping is disabled by default because ASL signs are handedness-sensitive
    if allow_horizontal_flip and gen.random() < 0.5:
        clip = clip[:, :, ::-1, :]

    # Consistent brightness & contrast across all frames in clip
    if brightness_delta > 0.0:
        shift = float(gen.uniform(-brightness_delta, brightness_delta))
        clip = clip + shift

    c_lo, c_hi = contrast_range
    if c_lo < c_hi:
        factor = float(gen.uniform(c_lo, c_hi))
        mean_val = clip.mean(axis=(1, 2), keepdims=True)
        clip = (clip - mean_val) * factor + mean_val

    clip = np.clip(clip, 0.0, 1.0)
    if is_uint8_scale:
        return np.round(clip * 255.0).astype(np.uint8)
    return clip.astype(np.float32)

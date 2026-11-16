"""Video reading, validation, and segment-aware decoding using OpenCV."""

import time
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal, overload

import numpy as np

if TYPE_CHECKING:
    import torch

from src.preprocessing.frames import (
    SegmentBounds,
    preprocess_bgr_frames,
    resolve_segment_bounds,
    sample_frame_indices,
)


class VideoReadError(RuntimeError):
    """Raised when a video file is missing, unreadable, or yields no frames."""


def _open_capture_with_retry(
    path: Path,
    *,
    retries: int = 3,
    retry_delay_sec: float = 0.5,
):
    import cv2

    last_cap = None
    for attempt in range(max(1, retries)):
        cap = cv2.VideoCapture(str(path))
        if cap.isOpened():
            fc = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
            if fc > 0:
                return cap
            # Live recorded container (e.g. browser WebM) where header lacks frame_count
            ret, _ = cap.read()
            if ret:
                cap.release()
                return cv2.VideoCapture(str(path))
        cap.release()
        last_cap = None
        if attempt + 1 < retries:
            time.sleep(retry_delay_sec)
    return last_cap


def probe_video(
    video_path: str | Path,
    *,
    open_retries: int = 3,
) -> dict[str, Any]:
    """Inspect video readability, FPS, total frame count, and resolution."""
    import cv2

    path = Path(video_path)
    if not path.is_file() or path.stat().st_size == 0:
        return {
            "path": str(path),
            "readable": False,
            "fps": 0.0,
            "frame_count": 0,
            "width": 0,
            "height": 0,
            "duration_sec": 0.0,
            "error": "missing_or_empty_file",
        }

    cap = _open_capture_with_retry(path, retries=open_retries)
    if cap is None:
        return {
            "path": str(path),
            "readable": False,
            "fps": 0.0,
            "frame_count": 0,
            "width": 0,
            "height": 0,
            "duration_sec": 0.0,
            "error": "cannot_open_or_zero_frames",
        }

    try:
        fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
        ok, frame = cap.read()
        valid = bool(
            ok and frame is not None and width > 0 and height > 0
        )
        duration = (frame_count / fps) if fps > 0 and frame_count > 0 else 0.0
        return {
            "path": str(path),
            "readable": valid,
            "fps": fps,
            "frame_count": frame_count,
            "width": width,
            "height": height,
            "duration_sec": duration,
            "error": "ok" if valid else "cannot_read_first_frame_or_bad_metadata",
        }
    finally:
        cap.release()


def decode_video_segment_bgr(
    video_path: str | Path,
    frame_start: Any = 1,
    frame_end: Any = -1,
    *,
    metadata_fps: float | None = 25.0,
    num_frames: int = 32,
    convert_fps: bool = True,
    one_based: bool = True,
    strict_bounds: bool = True,
    seek_preroll: int | None = None,
    open_retries: int = 3,
) -> tuple[list[np.ndarray], np.ndarray, SegmentBounds, float]:
    """Decode `num_frames` BGR frames sequentially (zero seek drift by default)."""
    import cv2

    path = Path(video_path)
    if not path.is_file() or path.stat().st_size == 0:
        raise VideoReadError(f"Video file missing or empty: {path}")

    cap = _open_capture_with_retry(path, retries=open_retries)
    if cap is None:
        raise VideoReadError(f"Cannot open video (or zero frames): {path}")

    try:
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        actual_fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)

        # Fallback for live browser WebM/fragmented recordings where container lacks frame_count
        if total_frames <= 0:
            all_frames: list[np.ndarray] = []
            while True:
                ok, frame = cap.read()
                if not ok or frame is None:
                    break
                all_frames.append(frame)

            if not all_frames:
                raise VideoReadError(f"Video reports zero frames: {path}")

            t_len = len(all_frames)
            indices = sample_frame_indices(0, t_len - 1, num_frames=num_frames)
            sampled = [all_frames[int(i)] for i in indices]
            bounds = SegmentBounds(
                start_idx=0,
                end_idx=t_len - 1,
                annotation_type="live_stream",
                fps_converted=False,
                clamped_end=False,
                overflow_frames=0,
            )
            return sampled, indices, bounds, actual_fps if actual_fps > 0 else 25.0

        bounds = resolve_segment_bounds(
            total_frames=total_frames,
            frame_start=frame_start,
            frame_end=frame_end,
            metadata_fps=metadata_fps,
            actual_fps=actual_fps if actual_fps > 0 else None,
            convert_fps=convert_fps,
            one_based=one_based,
            strict=strict_bounds,
        )
        indices = sample_frame_indices(
            bounds.start_idx, bounds.end_idx, num_frames=num_frames
        )
        min_idx = int(indices.min())
        max_idx = int(indices.max())
        wanted = set(indices.tolist())

        current = 0
        if seek_preroll is not None and min_idx >= 64:
            seek_target = max(0, min_idx - seek_preroll)
            cap.set(cv2.CAP_PROP_POS_FRAMES, seek_target)
            pos = int(round(cap.get(cv2.CAP_PROP_POS_FRAMES) or 0))
            if 0 <= pos <= min_idx:
                current = pos
            else:
                cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                current = 0

        decoded: dict[int, np.ndarray] = {}
        while current <= max_idx:
            if current in wanted:
                ok, frame = cap.read()
                if not ok or frame is None:
                    break
                decoded[current] = frame
            else:
                ok = cap.grab()
                if not ok:
                    break
            current += 1

        if not decoded:
            raise VideoReadError(
                f"No frames decoded in segment {bounds} from {path}"
            )

        ordered_frames: list[np.ndarray] = []
        last_valid: np.ndarray | None = None
        for idx in indices.tolist():
            if idx in decoded:
                last_valid = decoded[idx]
                ordered_frames.append(last_valid)
            elif last_valid is not None:
                ordered_frames.append(last_valid.copy())

        while len(ordered_frames) < num_frames and last_valid is not None:
            ordered_frames.append(last_valid.copy())

        if len(ordered_frames) != num_frames:
            raise VideoReadError(
                f"Expected {num_frames} frames, collected {len(ordered_frames)} from {path}"
            )

        return ordered_frames, indices, bounds, actual_fps
    finally:
        cap.release()


@overload
def preprocess_video(
    video_path: str | Path,
    frame_start: Any = ...,
    frame_end: Any = ...,
    *,
    metadata_fps: float | None = ...,
    num_frames: int = ...,
    image_size: int = ...,
    convert_fps: bool = ...,
    one_based: bool = ...,
    strict_bounds: bool = ...,
    seek_preroll: int | None = ...,
    open_retries: int = ...,
    return_tensor: Literal[True] = ...,
) -> "torch.Tensor": ...


@overload
def preprocess_video(
    video_path: str | Path,
    frame_start: Any = ...,
    frame_end: Any = ...,
    *,
    metadata_fps: float | None = ...,
    num_frames: int = ...,
    image_size: int = ...,
    convert_fps: bool = ...,
    one_based: bool = ...,
    strict_bounds: bool = ...,
    seek_preroll: int | None = ...,
    open_retries: int = ...,
    return_tensor: Literal[False],
) -> np.ndarray: ...


@overload
def preprocess_video(
    video_path: str | Path,
    frame_start: Any = ...,
    frame_end: Any = ...,
    *,
    metadata_fps: float | None = ...,
    num_frames: int = ...,
    image_size: int = ...,
    convert_fps: bool = ...,
    one_based: bool = ...,
    strict_bounds: bool = ...,
    seek_preroll: int | None = ...,
    open_retries: int = ...,
    return_tensor: bool = ...,
) -> "torch.Tensor | np.ndarray": ...


def preprocess_video(
    video_path: str | Path,
    frame_start: Any = 1,
    frame_end: Any = -1,
    *,
    metadata_fps: float | None = 25.0,
    num_frames: int = 32,
    image_size: int = 224,
    convert_fps: bool = True,
    one_based: bool = True,
    strict_bounds: bool = False,
    seek_preroll: int | None = None,
    open_retries: int = 3,
    return_tensor: bool = True,
) -> "torch.Tensor | np.ndarray":
    """End-to-end deterministic preprocessing returning (T, 3, H, W) float32 tensor/array."""
    frames_bgr, _, _, _ = decode_video_segment_bgr(
        video_path=video_path,
        frame_start=frame_start,
        frame_end=frame_end,
        metadata_fps=metadata_fps,
        num_frames=num_frames,
        convert_fps=convert_fps,
        one_based=one_based,
        strict_bounds=strict_bounds,
        seek_preroll=seek_preroll,
        open_retries=open_retries,
    )
    array = preprocess_bgr_frames(frames_bgr, image_size=image_size)
    if return_tensor:
        import torch

        return torch.from_numpy(array)
    return array

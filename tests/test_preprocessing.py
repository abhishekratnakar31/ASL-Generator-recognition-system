"""Unit tests for annotation mapping, frame sampling, and preprocessing."""

from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import pytest

from src.preprocessing.frames import (
    IMAGENET_MEAN,
    IMAGENET_STD,
    SegmentBoundsError,
    normalize_rgb_frames,
    preprocess_bgr_frames,
    resolve_segment_bounds,
    sample_frame_indices,
)
from src.preprocessing.video import VideoReadError, preprocess_video, probe_video

if TYPE_CHECKING:
    import cv2

    HAS_CV2 = True
else:
    try:
        import cv2

        HAS_CV2 = True
    except ImportError:
        cv2 = None  # type: ignore[assignment]
        HAS_CV2 = False


def test_resolve_segment_bounds_full_video():
    bounds = resolve_segment_bounds(
        total_frames=100, frame_start=1, frame_end=-1, metadata_fps=25.0, actual_fps=29.97
    )
    assert bounds.start_idx == 0
    assert bounds.end_idx == 99
    assert bounds.annotation_type == "full_video"
    assert not bounds.fps_converted

    bounds_nan = resolve_segment_bounds(total_frames=50, frame_start=np.nan, frame_end=np.nan)
    assert (bounds_nan.start_idx, bounds_nan.end_idx) == (0, 49)


def test_resolve_segment_bounds_open_end():
    bounds = resolve_segment_bounds(
        total_frames=80, frame_start=3, frame_end=-1, metadata_fps=25.0, actual_fps=25.0
    )
    assert bounds.start_idx == 2
    assert bounds.end_idx == 79
    assert bounds.annotation_type == "open_end_segment"


def test_resolve_segment_bounds_fps_conversion():
    # Example 70173: frame_start=5710, frame_end=5831 at meta_fps=25.0, actual_fps=29.970
    b_conv = resolve_segment_bounds(
        total_frames=9806,
        frame_start=5710,
        frame_end=5831,
        metadata_fps=25.0,
        actual_fps=29.970,
        convert_fps=True,
        one_based=True,
    )
    expected_start = int(round((5710 - 1) * 29.970 / 25.0))
    expected_end = int(round((5831 - 1) * 29.970 / 25.0))
    assert b_conv.start_idx == expected_start
    assert b_conv.end_idx == expected_end
    assert b_conv.fps_converted
    assert b_conv.annotation_type == "bounded_segment"
    assert not b_conv.clamped_end

    # Matching FPS (e.g., 12306 at 25.0 fps) converts 1-based -> 0-based with no scaling
    b_match = resolve_segment_bounds(
        total_frames=7518,
        frame_start=5122,
        frame_end=5181,
        metadata_fps=25.0,
        actual_fps=25.0,
        convert_fps=True,
        one_based=True,
    )
    assert (b_match.start_idx, b_match.end_idx) == (5121, 5180)
    assert not b_match.fps_converted


def test_resolve_segment_bounds_errors_and_clamping():
    with pytest.raises(SegmentBoundsError):
        resolve_segment_bounds(total_frames=0, frame_start=1, frame_end=-1)

    with pytest.raises(SegmentBoundsError):
        resolve_segment_bounds(
            total_frames=100, frame_start=150, frame_end=180, metadata_fps=25.0, actual_fps=25.0
        )

    # Minor rounding overflow (<= 5 frames) clamps cleanly
    b_minor = resolve_segment_bounds(
        total_frames=100, frame_start=80, frame_end=103, metadata_fps=25.0, actual_fps=25.0, strict=True
    )
    assert b_minor.end_idx == 99
    assert b_minor.clamped_end
    assert b_minor.overflow_frames == 3

    # Larger overflow raises in strict=True, clamps in strict=False
    with pytest.raises(SegmentBoundsError):
        resolve_segment_bounds(
            total_frames=100, frame_start=80, frame_end=120, metadata_fps=25.0, actual_fps=25.0, strict=True
        )
    b_relaxed = resolve_segment_bounds(
        total_frames=100, frame_start=80, frame_end=120, metadata_fps=25.0, actual_fps=25.0, strict=False
    )
    assert b_relaxed.end_idx == 99


def test_sample_frame_indices_uniqueness_and_short_videos():
    # Segment >= 32 frames: zero duplicate frames
    for seg_len in [32, 33, 59, 147]:
        idx = sample_frame_indices(10, 10 + seg_len - 1, num_frames=32)
        assert len(idx) == 32
        assert len(np.unique(idx)) == 32
        assert idx[0] == 10 and idx[-1] == 10 + seg_len - 1

    # Short video < 32 frames: all unique frames retained
    for seg_len in [1, 10, 22, 31]:
        idx = sample_frame_indices(0, seg_len - 1, num_frames=32)
        assert len(idx) == 32
        assert len(np.unique(idx)) == seg_len
        assert idx[0] == 0 and idx[-1] == seg_len - 1


def test_normalize_rgb_frames_exact_values():
    # Pure white image (255, 255, 255) -> (1 - mean) / std
    rgb = np.full((32, 224, 224, 3), 255, dtype=np.uint8)
    out = normalize_rgb_frames(rgb)
    assert out.shape == (32, 3, 224, 224)
    assert out.dtype == np.float32
    expected = (1.0 - IMAGENET_MEAN) / IMAGENET_STD
    assert np.allclose(out[0, :, 0, 0], expected, atol=1e-5)


@pytest.mark.skipif(not HAS_CV2, reason="OpenCV not installed in local environment")
def test_opencv_segment_decode_bgr_to_rgb_and_corrupt(tmp_path: Path):
    # Create synthetic video: frames 0..49 blue (BGR 255,0,0), frames 50..99 red (BGR 0,0,255)
    vid_path = tmp_path / "synthetic.mp4"
    fourcc = getattr(cv2, "VideoWriter_fourcc")(*"mp4v")
    writer = cv2.VideoWriter(str(vid_path), fourcc, 25.0, (64, 48))
    assert writer.isOpened()
    for i in range(100):
        frame = np.zeros((48, 64, 3), dtype=np.uint8)
        if i < 50:
            frame[:, :, 0] = 255  # Blue in BGR -> channel 2 in RGB
        else:
            frame[:, :, 2] = 255  # Red in BGR -> channel 0 in RGB
        writer.write(frame)
    writer.release()

    info = probe_video(vid_path)
    assert info["readable"]
    assert info["frame_count"] == 100

    # Decode only frames 55..90 (1-based 56..91) -> must be red (channel 0 > channel 2)
    arr = preprocess_video(
        vid_path,
        frame_start=56,
        frame_end=91,
        metadata_fps=25.0,
        num_frames=32,
        image_size=224,
        return_tensor=False,
    )
    assert arr.shape == (32, 3, 224, 224)
    assert arr[:, 0].mean() > 1.5
    assert arr[:, 2].mean() < -1.5

    # Corrupt / empty / missing files must raise VideoReadError
    corrupt = tmp_path / "corrupt.mp4"
    corrupt.write_bytes(b"not a real video file")
    with pytest.raises(VideoReadError):
        preprocess_video(corrupt, return_tensor=False)

    with pytest.raises(VideoReadError):
        preprocess_video(tmp_path / "missing.mp4", return_tensor=False)


def test_preprocess_bgr_frames_empty_raises():
    with pytest.raises(ValueError):
        preprocess_bgr_frames([])


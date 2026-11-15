# Video Preprocessing Audit Report

**Audit Date:** 2026-11-11  
**Validated via:** [`notebooks/02_video_preprocessing_validation.ipynb`](file:///Users/abhishekratnakar/ASL-Generator-recognition-system/notebooks/02_video_preprocessing_validation.ipynb) on Google Colab (`/content/drive/MyDrive/ASL-Recognition`) and [`src/preprocessing/`](file:///Users/abhishekratnakar/ASL-Generator-recognition-system/src/preprocessing)

---

## 1. Root Cause of Earlier Failures

All historical ResNet50 feature caches ([`drive/resnet50_features/`](file:///Users/abhishekratnakar/ASL-Generator-recognition-system/drive/resnet50_features), [`drive/asl_23_class/feature_cache/`](file:///Users/abhishekratnakar/ASL-Generator-recognition-system/drive/asl_23_class/feature_cache), and Drive `asl_60_class/feature_cache/`) were generated using `07_visual_baseline_training.ipynb`'s `preprocess_video()`, which sampled `np.linspace(0, frame_count - 1, 32)` over the entire raw video file and **ignored `frame_start` and `frame_end`**.

Additionally, `03_video_preprocessing (1).ipynb` and `04_visual_baseline.ipynb` passed raw 1-based metadata `frame_start`/`frame_end` (`metadata_fps = 25.0`) directly into OpenCV 0-based frame indices without converting for container FPS differences (`29.97` / `30.00` fps).

---

## 2. Empirical Validation Across All 131 Non-Full Videos (Colab Results)

Probing every non-full video (`129` in `val.csv` + `2` in `train.csv`) in Google Colab produced:

| Metric | Observed Result | Notes |
|---|---|---|
| Total non-full annotated videos | **131** (`129` val, `2` train) | `0` in `test.csv` |
| Readable on first OpenCV open | **130 / 131** | `70245` (`white`, val) reported `total_frames=0, actual_fps=0.0` |
| Container FPS differs (`\|actual - metadata\| > 0.5`) | **53 / 131** | Metadata `fps = 25.0` vs container `29.97`–`30.00` fps |
| Hypothesis A (Direct 0-based) in bounds | **130 / 131** | Fails to reach true segment timestamp on `29.97` fps videos |
| Hypothesis B (FPS-converted 0-based) in bounds | **130 / 131** | `100%` of readable non-full videos remain strictly within `total_frames` |
| Inline unit tests (`TestVideoPreprocessing`) | **5 / 5 passed** (`0.073s`) | Full-video, FPS conversion, OOB error, sampling uniqueness, BGR→RGB synthetic MP4 |

### Key Verified Segment Mappings
| Video ID | Gloss | Split | Total Frames | Actual FPS | Meta FPS | Raw `frame_start..end` | Resolved 0-Based Bounds | Unique Frames | Tensor Mean / Std |
|---|---|---|---:|---:|---:|---|---|---|---|
| `70173` | `drink` | `val` | 9,806 | 29.970 | 25.0 | `5710..5831` | `6844..6989` (146 frames) | 32 | `0.9329 / 1.6790` |
| `12306` | `computer` | `val` | 7,518 | 25.000 | 25.0 | `5122..5181` | `5121..5180` (60 frames) | 32 | `-1.5336 / 0.6330` |
| `70348` | `before` | `val` | 9,524 | 29.973 | 25.0 | `4778..4873` | `5727..5841` (115 frames) | 32 | `0.2111 / 0.9183` |
| `51206` | `short` | `val` | 11,498 | 25.000 | 25.0 | `3538..3580` | `3537..3579` (43 frames) | 32 | `-1.6645 / 0.4335` |
| `70355` | `short` | `val` | 9,524 | 29.973 | 25.0 | `6591..6697` | `7901..8028` (128 frames) | 32 | `0.2078 / 0.9192` |
| `64201` | `year` | `val` | 437 | 25.000 | 25.0 | `1..77` | `0..76` (77 frames) | 32 | `-1.5542 / 0.5684` |
| `55771` | `sunday` | `train` | 58 | 29.970 | 25.0 | `2..-1` | `1..57` (57 frames) | 32 | `-1.0307 / 1.2490` |
| `2269` | `amazing` | `train` | 32 | 30.000 | 25.0 | `3..-1` | `2..31` (30 frames) | 30 | `-1.4817 / 0.5942` |

---

## 3. Critical Technical Findings & Fixes Applied

### 3.1 Seek Drift on `70348` → Fixed via Pure Sequential `grab()` Default
Comparing single-seek (`seek_preroll=32` via `cap.set(cv2.CAP_PROP_POS_FRAMES, ...)`) against pure sequential `cap.grab()` from frame `0`:
- `70173` (`drink`): `max_abs_diff = 0.000000` (`exact_match = True`, seek `495.3ms`, seq `5207.2ms`)
- `12306` (`computer`): `max_abs_diff = 0.000000` (`exact_match = True`, seek `155.2ms`, seq `2011.7ms`)
- **`70348` (`before`): `max_abs_diff = 1.524103` (`exact_match = False`, seek `153.0ms`, seq `1873.2ms`)**

Because `cap.set(CAP_PROP_POS_FRAMES)` exhibited keyframe seek drift on `70348.mp4`, [`src/preprocessing/video.py`](file:///Users/abhishekratnakar/ASL-Generator-recognition-system/src/preprocessing/video.py) now defaults to **`seek_preroll=None` (pure sequential `cap.grab()` / `cap.read()` from frame `0`)** for 100% exact frame alignment (`~1.8–5.2s` on long validation videos, affecting only `131` non-full videos).

### 3.2 Drive FUSE Transient Failures (`2609` and `70245`)
- **`2609` (`anniversary`, train):** Failed in Notebook 01 (`total_frames=0, 0x0`), then re-opened cleanly in Notebook 02 (`size=97,037 B, 95 frames, 29.970 fps, first_read_ok=True`)—confirming transient Google Drive FUSE read failure.
- **`70245` (`white`, val):** Reported `total_frames=0, actual_fps=0.0` in Notebook 02 Section 4 (note other rows from the same YouTube video `WprUBqi3iBc`—`70242`, `70244`, `70246`, `70247`, `70249`—opened cleanly).
- **Mitigation Added:** [`_open_capture_with_retry`](file:///Users/abhishekratnakar/ASL-Generator-recognition-system/src/preprocessing/video.py#L20-L37) retries OpenCV opens (`open_retries=3`), and Notebook 03 will additionally stage/verify `70245.mp4` on local VM disk (`/tmp`) and record any persistently corrupt video in `extraction_errors.csv` (`done[idx] = False`) rather than silently storing zeros.

### 3.3 Short Videos (`< 32` Frames) & Corrupt Error Handling
- Verified in Notebook 02 Section 7: `64283` (`22` frames -> `22` unique + `10` repeated), `13200` (`23` frames -> `23` unique + `9` repeated), `5728` (`26` frames -> `26` unique + `6` repeated), `13196` (`26` frames -> `26` unique + `6` repeated), all deterministic `(32, 3, 224, 224)` `torch.float32`.
- Empty, corrupt, and missing files cleanly raise `VideoReadError`.

### 3.4 Visual Artifacts Generated on Google Drive
Saved under `/content/drive/MyDrive/ASL-Recognition/reports/02_preprocessing_validation/visual_examples/`:
- `compare_70173_drink.png`, `compare_12306_computer.png`, `compare_70348_before.png`, `compare_51206_short.png`, `compare_70355_short.png`, `compare_64201_year.png`
- `storyboard_70173_drink.png`, `storyboard_12306_computer.png`, `storyboard_70348_before.png`, `storyboard_51206_short.png`, `storyboard_70355_short.png`


# Dataset and Metadata Audit Report

**Project:** Multimodal American Sign Language (ASL) Recognition (WLASL)  
**Audit Date:** 2026-11-11  
**Audit Scope:** Local snapshot in [`drive/`](file:///Users/abhishekratnakar/ASL-Generator-recognition-system/drive) (`drive-download-20261111T091602Z-1-001/`, `resnet50_features/`, `asl_23_class/`, `feature_cache/`, `experiment_summary.json`), executed notebook logs in [`notebooks/`](file:///Users/abhishekratnakar/ASL-Generator-recognition-system/notebooks), and live Google Drive execution results from [`01_repository_dataset_audit.ipynb`](file:///Users/abhishekratnakar/ASL-Generator-recognition-system/notebooks/01_repository_dataset_audit.ipynb).

> **Google Drive Verification Status (Notebook 01 Executed):** All `6,446` split video files (`4,619` train / `909` val / `918` test) exist and are non-empty under `/content/drive/MyDrive/ASL-Recognition/videos/`. A stratified OpenCV probe (`147` videos) found `146 / 147` readable and `108 / 147` container FPS mismatches relative to metadata `fps = 25.0`.

---

## 1. Authoritative Dataset & Split Inventory

### 1.1 Full Available WLASL Dataset (`1,800` Classes)
Verified against [`wlasl_final_optimized_signer_split.csv`](file:///Users/abhishekratnakar/ASL-Generator-recognition-system/drive/drive-download-20261111T091602Z-1-001/wlasl_final_optimized_signer_split.csv) and [`train.csv`](file:///Users/abhishekratnakar/ASL-Generator-recognition-system/drive/drive-download-20261111T091602Z-1-001/train.csv), [`val.csv`](file:///Users/abhishekratnakar/ASL-Generator-recognition-system/drive/drive-download-20261111T091602Z-1-001/val.csv), [`test.csv`](file:///Users/abhishekratnakar/ASL-Generator-recognition-system/drive/drive-download-20261111T091602Z-1-001/test.csv):

| Split | Rows | Percentage | Unique Videos | Unique Classes | Unique Signers | Null Values | Master Match |
|---|---:|---:|---:|---:|---:|---:|---|
| **Train** | 4,619 | 71.66% | 4,619 | 1,797 | 26 | 0 | Exact (`equals_master=True`) |
| **Validation** | 909 | 14.10% | 909 | 650 | 22 | 0 | Exact (`equals_master=True`) |
| **Test** | 918 | 14.24% | 918 | 643 | 16 | 0 | Exact (`equals_master=True`) |
| **Total** | **6,446** | **100.00%** | **6,446** | **1,800** | **64** | **0** | Verified |

### 1.2 23-Class Experimental Subset ([`drive/asl_23_class/`](file:///Users/abhishekratnakar/ASL-Generator-recognition-system/drive/asl_23_class))
| Split | Rows | Unique Classes | Unique Signers | Samples / Class | Null Values |
|---|---:|---:|---:|---:|---:|
| **Train (`train_23.csv`)** | 126 | 23 | 16 | 5 – 8 | 0 |
| **Validation (`val_23.csv`)** | 57 | 23 | 12 | 2 – 4 | 0 |
| **Test (`test_23.csv`)** | 79 | 23 | 11 | 2 – 6 | 0 |
| **Total** | **262** | **23** | **39** | 9 – 16 | **0** |

### 1.3 60-Class Experimental Subset (Logged in [`experiment_summary.json`](file:///Users/abhishekratnakar/ASL-Generator-recognition-system/drive/experiment_summary.json))
- **Train:** 248 videos across 19 signers (3–8 per class, mean 4.13)
- **Validation:** 169 videos across 15 signers (2–5 per class, mean 2.82)
- **Test:** 209 videos across 12 signers (2–7 per class, mean 3.48)
- **Total:** 626 videos across 60 classes (`min_train >= 3, min_val >= 2, min_test >= 2`)

---

## 2. Missing, Duplicate, and Leakage Checks

| Check | Scope | Result | Details |
|---|---|---|---|
| **Missing Labels / Nulls** | All split CSVs (`1,800`-class & `23`-class) | **PASS (0 nulls)** | All columns (`gloss`, `video_id`, `signer_id`, `frame_start`, `frame_end`, `fps`, `url`, `class_id`, etc.) are non-null. |
| **Intra-Split Duplicate Videos** | `video_id` & `video_id_normalized` | **PASS (0 duplicates)** | All `4,619` train, `909` val, `918` test IDs are unique (`6,446` total). Note: `instance_id` is WLASL's per-gloss instance index (`0, 1, 2...`), not a global ID. |
| **Cross-Split Video ID Overlap** | Train / Val / Test | **PASS (0 overlaps)** | `train ∩ val = ∅`, `train ∩ test = ∅`, `val ∩ test = ∅`. |
| **Cross-Split Canonical URL Overlap** | YouTube ID & raw `url` | **PASS (0 overlaps)** | `0` shared source URLs across splits (`train∩val=0`, `train∩test=0`, `val∩test=0`). Multi-sign source URLs stay within one split (`170` in train, `48` in val up to 11 signs, `26` in test) because they belong to single signers. |
| **Signer Overlap (1,800-class)** | `signer_id` across splits | **PASS (0 overlaps)** | `26` train, `22` val, `16` test signers are strictly disjoint. |
| **Signer Overlap (23-class)** | `signer_id` across splits | **PASS (0 overlaps)** | `16` train, `12` val, `11` test signers are strictly disjoint. |
| **Historical Missing WLASL Videos** | Official `WLASL_v0.3.json` | **PARTIAL COVERAGE** | Notebook 02 recovered `6,446` usable videos out of `21,083` unique official IDs (~`30.6%`) after download interruption (`5,538` downloaded + pre-existing files; `13,471` `not_created`; `46` unreadable). |
| **Test-Set Exposure** | `test_23` & `test_60` | **EXPOSED (WARNING)** | Exploratory notebooks `07_asl_23_class_experiment.ipynb` and `07_asl_60_class_experiment.ipynb` evaluated on test splits. Future model selection must use validation strictly until final evaluation. |

---

## 3. Class Mapping & Missing Training Classes

### 3.1 Mapping Consistency
- **Full Vocabulary ([`class_to_id.json`](file:///Users/abhishekratnakar/ASL-Generator-recognition-system/drive/drive-download-20261111T091602Z-1-001/class_to_id.json), [`id_to_class.json`](file:///Users/abhishekratnakar/ASL-Generator-recognition-system/drive/drive-download-20261111T091602Z-1-001/id_to_class.json)):**
  - `1,800` classes, contiguous integer IDs `0..1799`, exact bijection (`id_to_class[str(v)] == k`), `100%` agreement with `gloss` and `class_id` across `train.csv`, `val.csv`, `test.csv`.
- **23-Class Subset ([`drive/asl_23_class/metadata/`](file:///Users/abhishekratnakar/ASL-Generator-recognition-system/drive/asl_23_class/metadata)):**
  - `23` classes (`0..22`), exact bijection across `class_to_id.json`, `id_to_class.json`, `original_to_new_id.json`, `new_to_original_id.json`, and `train_23.csv` / `val_23.csv` / `test_23.csv`.

### 3.2 Critical Split Defect: 3 Classes Absent From Training
`05b_rebuild_signer_split.ipynb` produced a split where **3 classes have zero training samples** in `train.csv` (`1,797 / 1,800` classes present) while still appearing in `val.csv` / `test.csv`:

| `class_id` | `gloss` | Train Count | Val Count | Test Count | Total |
|---:|---|---:|---:|---:|---:|
| `312` | `chemical` | **0** | 1 | 2 | 3 |
| `424` | `culture` | **0** | 1 | 0 | 1 |
| `1464` | `skate` | **0** | 1 | 0 | 1 |

A closed-set classifier trained on `train.csv` cannot predict these 3 classes. Any full-vocabulary experiment must either exclude zero-train classes, adjust the signer split, or handle zero-shot/open-vocabulary evaluation via the multimodal text branch.

---

## 4. Class & Signer Distributions

### 4.1 Full 1,800-Class Long-Tail Distribution (`train.csv`)
- **Represented classes:** `1,797` (`3` have `0` training samples)
- **Samples per represented class:** `min = 1`, `Q25 = 2.0`, `median = 2.0`, `mean = 2.570`, `Q75 = 3.0`, `max = 9`
- **Sparsity concentration:**
  - `<= 1` training video: **340 classes** (18.9%)
  - `<= 2` training videos: **960 classes** (53.4%)
  - `<= 3` training videos: **1,437 classes** (80.0%)
  - `<= 4` training videos: **1,672 classes** (93.0%)
  - `<= 5` training videos: **1,762 classes** (98.1%)

### 4.2 Subset Eligibility Thresholds
| Threshold `(min_train, min_val, min_test)` | Eligible Classes | Train Videos | Val Videos | Test Videos | Avg Train / Class |
|---|---:|---:|---:|---:|---:|
| `(1, 1, 1)` | 272 | 727 | 464 | 513 | 2.67 |
| `(2, 1, 1)` | 200 | 655 | 365 | 408 | 3.27 |
| `(3, 1, 1)` | 124 | 503 | 255 | 294 | 4.06 |
| `(3, 2, 2)` *(60-class exp)* | 60 | 248 | 169 | 209 | 4.13 |
| `(4, 1, 1)` | 69 | 338 | 145 | 171 | 4.90 |
| `(4, 2, 2)` | 34 | 170 | 90 | 120 | 5.00 |
| `(5, 1, 1)` | 41 | 226 | 84 | 112 | 5.51 |
| `(5, 2, 2)` *(23-class exp)* | 23 | 126 | 57 | 79 | 5.48 |
| `(5, 3, 3)` | 6 | 37 | 20 | 23 | 6.17 |
| `(6, 2, 2)` | 8 | 51 | 20 | 30 | 6.38 |
| `(8, 2, 2)` | 1 | 8 | 3 | 5 | 8.00 |

### 4.3 Signer Distribution per Split
- **Train (`26` signers, `4,619` videos):** `min = 1`, `median = 54.0`, `mean = 177.65`, `max = 885`. Top 5 signers (`13`: 885, `4`: 696, `2`: 517, `21`: 420, `38`: 418) contribute **63.6%** of all training videos (`2,936 / 4,619`).
- **Validation (`22` signers, `909` videos):** `min = 1`, `median = 15.5`, `mean = 41.32`, `max = 247`. Top 5 signers (`31`: 247, `19`: 129, `45`: 102, `9`: 78, `115`: 66) contribute **68.4%** (`622 / 909`).
- **Test (`16` signers, `918` videos):** `min = 1`, `median = 29.5`, `mean = 57.38`, `max = 251`. Top 5 signers (`59`: 251, `5`: 197, `15`: 85, `118`: 82, `26`: 78) contribute **75.5%** (`693 / 918`).

---

## 5. Corrupted, Short, and Segmented Video Findings

### 5.1 Segment Annotations (`frame_start`, `frame_end`, `fps`)
| Split | Total | Full Video (`1` to `-1`) | Bounded Segment (`start > 1, end > start`) | Other Non-Full | Metadata `fps` | Sources / Signers |
|---|---:|---:|---:|---:|---:|---|
| **Train** | 4,619 | 4,617 | 0 | 2 (`55771` start=2, `2269` start=3) | `25` | `signschool` (signer 4), `spreadthesign` (signer 28) |
| **Validation** | 909 | 780 | **128** | 1 (`64201` start=1, end=77) | `25` | **`asl5200` (signer 9: 78 videos)**, **`northtexas` (signer 121: 51 videos)** |
| **Test** | 918 | 918 | 0 | 0 | `25` | — |

In the 23-class validation split (`val_23.csv`), **31 of 57 videos (54.4%)** are segmented recordings (`asl5200`/signer 9: `19`; `northtexas`/signer 121: `12`). Representative flagged videos verified in `val_23.csv`:
- `70173` (`drink`, signer `121`): `frame_start=5710, frame_end=5831, fps=25`
- `12306` (`computer`, signer `9`): `frame_start=5122, frame_end=5181, fps=25`
- `70348` (`before`, signer `121`): `frame_start=4778, frame_end=4873, fps=25`
- `51206` (`short`, signer `9`): `frame_start=3538, frame_end=3580, fps=25`
- `70355` (`short`, signer `121`): `frame_start=6591, frame_end=6697, fps=25`

### 5.2 Short Videos, Unreadable Probe Video & Live Drive OpenCV Audit
- **Disk Availability on Google Drive (`/content/drive/MyDrive/ASL-Recognition/videos`):** `6,559` video files indexed (`113` unreferenced extra files). Split coverage is **100% present and non-empty** (`train: 4,619 / 4,619`, `val: 909 / 909`, `test: 918 / 918`).
- **Stratified OpenCV Probe (`147` videos via Notebook 01):**
  - **Readable:** `146 / 147` (`1` sampled video failed OpenCV read — must be identified and handled safely in Notebook 02).
  - **FPS Mismatches (`|actual_fps - metadata_fps| > 0.5`):** `108 / 147` (`73.5%` of sampled videos differ from metadata `fps = 25.0`).
  - **Flagged Priority Videos Observed on Drive:**
    - `28038` (`train`, `hospital`): `readable=True`, `79` frames, `30.000` fps (metadata `25.0`), `2.63s`, `640x480` (`frame_start=1, frame_end=-1`) — previously failed in [`extraction_errors.csv`](file:///Users/abhishekratnakar/ASL-Generator-recognition-system/drive/resnet50_features/train/extraction_errors.csv), now readable on Drive and repaired in `done.npy`.
    - `70173` (`val`, `drink`): `readable=True`, `9,806` frames, `29.970` fps (metadata `25.0`), `327.19s`, `628x480` (`frame_start=5710, frame_end=5831`).
    - `12306` (`val`, `computer`): `readable=True`, `7,518` frames, `25.000` fps (metadata `25.0`), `300.72s`, `480x320` (`frame_start=5122, frame_end=5181`).
    - `70348` (`val`, `before`): `readable=True`, `9,524` frames, `29.973` fps (metadata `25.0`), `317.75s`, `626x360` (`frame_start=4778, frame_end=4873`).
    - `51206` (`val`, `short`): `readable=True`, `11,498` frames, `25.000` fps (metadata `25.0`), `459.92s`, `480x320` (`frame_start=3538, frame_end=3580`).
    - `70355` (`val`, `short`): `readable=True`, `9,524` frames, `29.973` fps (metadata `25.0`), `317.75s`, `626x360` (`frame_start=6591, frame_end=6697`).
- **Short videos (`< 32` frames):** Verified in [`drive/asl_23_class/video_sampling_audit.csv`](file:///Users/abhishekratnakar/ASL-Generator-recognition-system/drive/asl_23_class/video_sampling_audit.csv) (`262` rows): **13 training videos** have `< 32` frames (`22` to `31` frames; `0.92s`–`1.29s`), resulting in `1`–`10` duplicate frames per 32-frame clip (`yes`/`64283` has `22` frames -> `10` duplicates; `cool`/`13200` has `23` frames -> `9` duplicates). Validation (`57`) and test (`79`) have `0` videos under 32 frames.

---

## 6. Feature Cache Integrity & Manifest Mapping

### 6.1 Full Cache ([`drive/resnet50_features/`](file:///Users/abhishekratnakar/ASL-Generator-recognition-system/drive/resnet50_features))
All checks verified using `manifest.csv` `video_id` mapping (not assumed row position):

| Split | Shape | Dtype | `done.npy` | Manifest ↔ Split Order & Set Match | Label Match | Non-Finite Rows | Zero-Norm Rows | Mean / Std / Range | Preprocessing Validity |
|---|---|---|---|---|---|---:|---:|---|---|
| **Train** | `(4619, 32, 2048)` | `float16` | `4619/4619` | Order: `True` / Set: `True` | `True` | `0` | `0` | `0.1043 / 0.3040 / [0.0, 14.57]` | 2 rows had `frame_start > 1` ignored |
| **Val** | `(909, 32, 2048)` | `float16` | `909/909` | Order: `True` / Set: `True` | `True` | `0` | `0` | `0.0998 / 0.3112 / [0.0, 14.82]` | **INVALIDATED:** 129 segmented videos uncropped |
| **Test** | `(918, 32, 2048)` | `float16` | `918/918` | Order: `True` / Set: `True` | `True` | `0` | `0` | `0.1008 / 0.2931 / [0.0, 14.90]` | Full-video markers only (`0` segmented) |

### 6.2 23-Class & 60-Class Subset Caches
- **23-Class Cache ([`drive/asl_23_class/feature_cache/`](file:///Users/abhishekratnakar/ASL-Generator-recognition-system/drive/asl_23_class/feature_cache) & [`drive/feature_cache/`](file:///Users/abhishekratnakar/ASL-Generator-recognition-system/drive/feature_cache)):** Indexing `resnet50_features` by `manifest.csv` `video_id` reproduces `X_train_23.npy` (`(126, 32, 2048)`), `X_validation_23.npy` (`(57, 32, 2048)`), and `X_test_23.npy` (`(79, 32, 2048)`) identically (`np.array_equal == True`); `y_*_23.npy` matches `class_id` 1:1; all 6 `.npy` files in `drive/feature_cache/` have identical MD5 hashes; `0` NaNs/Infs. `X_validation_23.npy` is **invalidated** (31/57 validation videos uncropped).
- **60-Class Cache (Google Drive `experiments/asl_60_class/feature_cache/`):** Verified via Notebook 01: `train (248, 32, 2048)`, `val (169, 32, 2048)`, `test (209, 32, 2048)`, dtype `float32`, `0` non-finite rows; `val` is **invalidated** (`FINITE_NEEDS_SEGMENT_FIX`).

---

## 7. Actionable Conclusions for Next Stages

1. **Do NOT train on `resnet50_features/val/`, `X_validation_23.npy`, or `X_val_60.npy`** until segment-aware preprocessing + FPS mapping is validated in **Notebook 02 (`02_video_preprocessing_validation.ipynb`)** and clean features are extracted into a separate cache in **Notebook 03**.
2. **Validate FPS conversion & 1-based indexing carefully in Notebook 02:** `70173`, `70348`, and `70355` run at `29.97` fps while `12306` and `51206` run at `25.00` fps (matching `metadata_fps = 25.0`). Identify the `1 / 147` unreadable video flagged by Notebook 01 and inspect all `131` non-full videos across train/val.
3. **Address class scarcity & the 3 zero-train classes** before final closed-set benchmarks (via validated subset selection without zero-train classes, valid temporal/spatial augmentations, and multimodal text alignment).


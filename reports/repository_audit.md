# Repository Technical Audit Report

**Project:** Multimodal American Sign Language (ASL) Recognition using WLASL  
**Repository:** `abhishekratnakar31/ASL-Generator-recognition-system`  
**Audit Date:** 2026-11-11  
**Audit Scope:** Complete inspection of source stubs, tests, configurations, CI workflows, documentation, 11 existing Colab notebooks in [`notebooks/`](file:///Users/abhishekratnakar/ASL-Generator-recognition-system/notebooks), and synced Google Drive artifacts in [`drive/`](file:///Users/abhishekratnakar/ASL-Generator-recognition-system/drive).

---

## Executive Summary

The repository currently consists of a well-organized **skeleton package structure** ([`src/`](file:///Users/abhishekratnakar/ASL-Generator-recognition-system/src), [`scripts/`](file:///Users/abhishekratnakar/ASL-Generator-recognition-system/scripts), [`backend/`](file:///Users/abhishekratnakar/ASL-Generator-recognition-system/backend), [`frontend/`](file:///Users/abhishekratnakar/ASL-Generator-recognition-system/frontend)) whose Python files contain **only 1-line module docstrings**, paired with **11 Colab notebooks** in [`notebooks/`](file:///Users/abhishekratnakar/ASL-Generator-recognition-system/notebooks) where all actual data engineering and visual-only baseline experiments were run.

Four critical blockers prevent the current system from learning or functioning as a multimodal ASL recognizer:
1. **No Language Branch or Multimodal Training Exists Yet:** Notebooks `07_language_encoder` and `08_multimodal_training` are completely absent; [`09_multimodal_model_evaluation.ipynb`](file:///Users/abhishekratnakar/ASL-Generator-recognition-system/notebooks/09_multimodal_model_evaluation.ipynb) and [`10_multimodal_inference_demo.ipynb`](file:///Users/abhishekratnakar/ASL-Generator-recognition-system/notebooks/10_multimodal_inference_demo.ipynb) are unexecuted stubs referencing missing checkpoints (`best_multimodal_model.pt`, `minilm_vocabulary_embeddings.pt`).
2. **Broken Video Preprocessing in Cached ResNet50 Features:** The cached feature arrays in [`drive/resnet50_features/`](file:///Users/abhishekratnakar/ASL-Generator-recognition-system/drive/resnet50_features) (and therefore [`drive/asl_23_class/feature_cache/`](file:///Users/abhishekratnakar/ASL-Generator-recognition-system/drive/asl_23_class/feature_cache)) were extracted over entire raw video files (`0` to `frame_count - 1`), **completely ignoring `frame_start` and `frame_end`**. As a result, **129 of 909 validation videos** (and **31 of 57** in the 23-class validation split—averaging 218.6 seconds and up to 459.9 seconds long) were sampled across minutes of unrelated background video rather than their ~3-second annotated sign segments.
3. **Unvalidated Frame-Rate / Indexing Mapping in `03` and `04`:** Even notebooks that read `frame_start`/`frame_end` ([`03_video_preprocessing (1).ipynb`](file:///Users/abhishekratnakar/ASL-Generator-recognition-system/notebooks/03_video_preprocessing%20(1).ipynb), [`04_visual_baseline.ipynb`](file:///Users/abhishekratnakar/ASL-Generator-recognition-system/notebooks/04_visual_baseline.ipynb)) apply raw 1-based metadata coordinates (`fps=25`) directly to OpenCV 0-based frames (often 29.97 fps or 23.98 fps) and silently corrupt out-of-range segments.
4. **Severe Data Scarcity & Split Defect:** While the repo docs describe `WLASL-100`, the actual Drive split ([`wlasl_final_optimized_signer_split.csv`](file:///Users/abhishekratnakar/ASL-Generator-recognition-system/drive/drive-download-20261111T091602Z-1-001/wlasl_final_optimized_signer_split.csv)) spans **1,800 classes across only 4,619 training videos** (mean 2.57/class, median 2.0/class, 340 singletons, and **3 classes with 0 training samples**: `312: chemical`, `424: culture`, `1464: skate`). Frozen ImageNet ResNet50 mean-pooled features also show no class separability on 60 classes (within-class cosine similarity `0.6402` vs between-class `0.6435`).

---

## 1. Current Project Architecture

### 1.1 Documented / Intended Architecture
Per [`README.md`](file:///Users/abhishekratnakar/ASL-Generator-recognition-system/README.md), [`docs/architecture/visual_pipeline.md`](file:///Users/abhishekratnakar/ASL-Generator-recognition-system/docs/architecture/visual_pipeline.md), [`docs/architecture/multimodal_pipeline.md`](file:///Users/abhishekratnakar/ASL-Generator-recognition-system/docs/architecture/multimodal_pipeline.md), and [`configs/multimodal.yaml`](file:///Users/abhishekratnakar/ASL-Generator-recognition-system/configs/multimodal.yaml):
- **Visual Stream:** Video (`[B, 32, 3, 224, 224]`) → Frozen ImageNet ResNet50 (`2048-D`) → 2-layer LSTM (`hidden=256, dropout=0.3`) → Visual Projection (`256-D`, L2-normalized).
- **Language Stream:** ASL Gloss → `sentence-transformers/all-MiniLM-L6-v2` (`384-D`) → Text Projection (`256-D`, L2-normalized).
- **Multimodal Alignment:** Contrastive cosine similarity (InfoNCE / NT-Xent, temperature `0.07`).
- **Application Phase (Phase 2):** Export ONNX/TorchScript → FastAPI backend ([`backend/`](file:///Users/abhishekratnakar/ASL-Generator-recognition-system/backend)) + Next.js/TypeScript/Tailwind frontend ([`frontend/`](file:///Users/abhishekratnakar/ASL-Generator-recognition-system/frontend)).

### 1.2 Actual Implemented Architecture
```text
[Drive Raw Videos] ──(broken full-video linspace)──▶ [Cached ResNet50 Features: (N, 32, 2048) float16]
                                                               │
       ┌───────────────────────────┬───────────────────────────┴───────────────────────────┐
       ▼                           ▼                                                       ▼
1800-Class Baseline         60-Class Subset                                        23-Class Subset
2-layer LSTM (256-D)        1-layer LSTM (128-D)                                   1-layer LSTM (128-D)
-> Linear(1800)             -> Linear(60)                                          -> Linear(23)
(Collapses; 0.00% Val)      (Overfits; 0.96% Test)                                 (Overfits; 3.80% Test)

Language Branch: NOT IMPLEMENTED (only unexecuted stubs in Notebooks 09 & 10)
src/, scripts/, backend/, frontend/: 0 lines of functional code (1-line docstrings / .gitkeep only)
```

---

## 2. Existing Components

### 2.1 Repository Files (Tracked in Git)
| Directory / File | Contents | Current State |
|---|---|---|
| [`src/`](file:///Users/abhishekratnakar/ASL-Generator-recognition-system/src) (26 `.py` files) | `data/`, `preprocessing/`, `models/{visual,language,multimodal}/`, `training/`, `evaluation/` | **Stubs only** — every file is a single-line docstring (`"""..."""`). |
| [`scripts/`](file:///Users/abhishekratnakar/ASL-Generator-recognition-system/scripts) (4 `.py` files) | `download_data.py`, `preprocess.py`, `train.py`, `evaluate.py` | **Stubs only** — single-line docstrings. |
| [`configs/`](file:///Users/abhishekratnakar/ASL-Generator-recognition-system/configs) | [`baseline.yaml`](file:///Users/abhishekratnakar/ASL-Generator-recognition-system/configs/baseline.yaml), [`resnet_lstm.yaml`](file:///Users/abhishekratnakar/ASL-Generator-recognition-system/configs/resnet_lstm.yaml), [`multimodal.yaml`](file:///Users/abhishekratnakar/ASL-Generator-recognition-system/configs/multimodal.yaml) | Present, but `baseline.yaml` and `resnet_lstm.yaml` point to obsolete `WLASL-100` (`metadata/wlasl100_metadata_with_split.csv`). |
| [`tests/`](file:///Users/abhishekratnakar/ASL-Generator-recognition-system/tests) | [`conftest.py`](file:///Users/abhishekratnakar/ASL-Generator-recognition-system/tests/conftest.py), [`test_structure.py`](file:///Users/abhishekratnakar/ASL-Generator-recognition-system/tests/test_structure.py) | 3 basic sanity tests (`test_project_directories`, `test_src_imports`, `test_metadata_file_exists`); `pytest -v` passes (`3 passed`). |
| [`.github/workflows/`](file:///Users/abhishekratnakar/ASL-Generator-recognition-system/.github/workflows) | [`lint.yml`](file:///Users/abhishekratnakar/ASL-Generator-recognition-system/.github/workflows/lint.yml), [`tests.yml`](file:///Users/abhishekratnakar/ASL-Generator-recognition-system/.github/workflows/tests.yml) | Runs `black`, `isort`, `flake8` (`max-line-length=120`) and `pytest -v` on Python 3.10. |
| [`docs/`](file:///Users/abhishekratnakar/ASL-Generator-recognition-system/docs) & [`README.md`](file:///Users/abhishekratnakar/ASL-Generator-recognition-system/README.md) | Architecture, dataset, experiments, roadmap docs | Severely out of date relative to actual Drive/notebook state (still states WLASL-100 and "full training not yet performed"). Note [`README.md`](file:///Users/abhishekratnakar/ASL-Generator-recognition-system/README.md) has unstaged Colab URL edits locally. |
| [`metadata/`](file:///Users/abhishekratnakar/ASL-Generator-recognition-system/metadata) | [`metadata/README.md`](file:///Users/abhishekratnakar/ASL-Generator-recognition-system/metadata/README.md), `.gitkeep` | No CSV/JSON metadata files are stored here in Git; they reside in `drive/` and Google Drive. |
| [`backend/`](file:///Users/abhishekratnakar/ASL-Generator-recognition-system/backend) & [`frontend/`](file:///Users/abhishekratnakar/ASL-Generator-recognition-system/frontend) | Empty `.gitkeep` trees + 1-line comment in [`backend/requirements.txt`](file:///Users/abhishekratnakar/ASL-Generator-recognition-system/backend/requirements.txt) | Unimplemented. |

### 2.2 Existing Colab Notebooks ([`notebooks/`](file:///Users/abhishekratnakar/ASL-Generator-recognition-system/notebooks), Untracked)
| Notebook | Executed? | Summary of Actual Functionality |
|---|---|---|
| [`01_environment_setup (1).ipynb`](file:///Users/abhishekratnakar/ASL-Generator-recognition-system/notebooks/01_environment_setup%20(1).ipynb) | Yes (Oct 12, 2026) | Verified Python 3.13.15, PyTorch 2.11.0+cu130, Torchvision 0.26.0+cu130, Tesla T4 (14.56 GB VRAM), OpenCV 4.14.0. |
| [`02_download_and_dataset_setup.ipynb`](file:///Users/abhishekratnakar/ASL-Generator-recognition-system/notebooks/02_download_and_dataset_setup.ipynb) | Partial (`KeyboardInterrupt`) | Parsed `WLASL_v0.3.json` (2,000 glosses, 21,083 unique videos). Download loop interrupted at 7,185/14,524 attempts (5,538 downloaded, 13,471 `not_created`). OpenCV audit found 6,559 local files, 6,513 readable, 46 unreadable; matched 6,446 videos across 1,800 classes & 64 signers (`wlasl_available_with_splits.csv`). |
| [`03_video_preprocessing (1).ipynb`](file:///Users/abhishekratnakar/ASL-Generator-recognition-system/notebooks/03_video_preprocessing%20(1).ipynb) | Yes | Loaded `wlasl_final_optimized_signer_split.csv` (4,619/909/918), audited 6,446 video paths (`notebook03_readability_audit.csv`, 6,446/6,446 readable), defined `WLASLDataset` and `preprocess_video` (sequential read with raw `frame_start`/`frame_end`), ran batch shape smoke test `[4, 32, 3, 224, 224]`. |
| [`04_visual_baseline.ipynb`](file:///Users/abhishekratnakar/ASL-Generator-recognition-system/notebooks/04_visual_baseline.ipynb) | Yes | Redefined `preprocess_video` and `WLASLDataset`, defined `VisualBaseline` (frozen ResNet50 + 2-layer 256-D LSTM + Linear(1800)), verified single-batch forward/backward smoke test (loss `7.4893`, gradients `True`). |
| [`05b_rebuild_signer_split.ipynb`](file:///Users/abhishekratnakar/ASL-Generator-recognition-system/notebooks/05b_rebuild_signer_split.ipynb) | Yes | Ran 2-stage simulated annealing over signer assignments to rebalance 83/8/9% → 71.7/14.1/14.2% (`4,619 / 909 / 918`). Overwrote `train.csv`, `val.csv`, `test.csv`. Left 3 classes absent from `train.csv`. |
| [`06_signer_split_validation.ipynb`](file:///Users/abhishekratnakar/ASL-Generator-recognition-system/notebooks/06_signer_split_validation.ipynb) | Yes | Checked `wlasl_available_with_splits.csv` (obsolete pre-`05b` file!) instead of `wlasl_final_optimized_signer_split.csv`. |
| [`07_visual_baseline_training.ipynb`](file:///Users/abhishekratnakar/ASL-Generator-recognition-system/notebooks/07_visual_baseline_training.ipynb) | Yes | Verified `processed/resnet50_features` manifest ordering, repaired 1 missing training row (`video_id=28038`) using full-video `preprocess_video` (ignoring `frame_start`/`frame_end`), trained `CachedFeatureLSTM` (1,800 classes, class-weighted CE) for 8 epochs before early stopping (Val Top-1 `0.00%`, Top-5 `0.11%`). Also ran memorization tests (32-class/64-sample reached `26.56%`; 8-class/40-sample reached `95%` in 8 epochs) and vocabulary threshold sweep (finding 60-class and 23-class candidates). |
| [`07_asl_60_class_experiment.ipynb`](file:///Users/abhishekratnakar/ASL-Generator-recognition-system/notebooks/07_asl_60_class_experiment.ipynb) | Yes | Built 60-class subset (`min_train>=3, min_val>=2, min_test>=2`: 248/169/209 videos) from `resnet50_features`. Trained 1-layer 128-D LSTM (Test Top-1 `0.96%`, Top-5 `8.13%`). Diagnosed class collapse and showed ResNet50 feature centroids have `0.8905` mean cosine similarity and `-0.0033` within-vs-between separability (`Cosine Nearest Centroid` val `2.96%`, test `2.87%`). At the end (Cells 20–23), created `experiments/asl_23_class/metadata/`. |
| [`07_asl_23_class_experiment.ipynb`](file:///Users/abhishekratnakar/ASL-Generator-recognition-system/notebooks/07_asl_23_class_experiment.ipynb) | Yes | Built 23-class subset (`min_train>=5, min_val>=2, min_test>=2`: 126/57/79 videos) from `resnet50_features`. Trained 1-layer 128-D LSTM (early stopped epoch 10, best epoch 2; Test Top-1 `3.80% = 3/79`, Macro-F1 `0.0161`). Cells 13–24 audited raw videos and **discovered the validation segment/FPS bug** (31/57 val videos average `218.58s` vs ~`2.98s` annotated segments; FPS mismatch `25` vs `29.97`). |
| [`09_multimodal_model_evaluation.ipynb`](file:///Users/abhishekratnakar/ASL-Generator-recognition-system/notebooks/09_multimodal_model_evaluation.ipynb) | **No** | Unexecuted skeleton expecting `test_loader` global from Notebook 03 + missing `best_multimodal_model.pt` and `minilm_vocabulary_embeddings.pt`. |
| [`10_multimodal_inference_demo.ipynb`](file:///Users/abhishekratnakar/ASL-Generator-recognition-system/notebooks/10_multimodal_inference_demo.ipynb) | **No** | Unexecuted skeleton expecting missing `best_multimodal_model.pt` and `minilm_vocabulary_embeddings.pt`. |

---

## 3. Completed Components

Verified locally on disk ([`drive/`](file:///Users/abhishekratnakar/ASL-Generator-recognition-system/drive)) and in notebook outputs:
- **WLASL Official Metadata Flattening & Local Video Audit** (saved on Drive in `metadata/`).
- **Signer-Disjoint Split Generation (`wlasl_final_optimized_signer_split.csv`)**: Verified `6,446` rows (`4,619` train / `909` val / `918` test), `64` signers (`26` train / `22` val / `16` test), **zero signer overlap** (`0` train∩val, `0` train∩test, `0` val∩test), zero nulls, and 100% consistency between `gloss` and `class_to_id.json` (`1,800` classes).
- **23-Class Subset Metadata & Audit** ([`drive/asl_23_class/`](file:///Users/abhishekratnakar/ASL-Generator-recognition-system/drive/asl_23_class)): `126` train (16 signers) / `57` val (12 signers) / `79` test (11 signers), 0 signer overlap, plus [`video_sampling_audit.csv`](file:///Users/abhishekratnakar/ASL-Generator-recognition-system/drive/asl_23_class/video_sampling_audit.csv) (`262` rows).
- **Sanity-Check Training Code Works When Data Is Separable:** In [`07_visual_baseline_training.ipynb`](file:///Users/abhishekratnakar/ASL-Generator-recognition-system/notebooks/07_visual_baseline_training.ipynb) Cell 10, `CachedFeatureLSTM` reached `95%` training accuracy in 8 epochs on an 8-class × 5-example subset (`40` samples), confirming PyTorch gradient flow works when classes are fewer and distinguishable.
- **CI Test & Lint Skeleton:** `pytest -v` currently passes 3/3 structural tests.

---

## 4. Incomplete Components

1. **Reusable Python Codebase ([`src/`](file:///Users/abhishekratnakar/ASL-Generator-recognition-system/src) & [`scripts/`](file:///Users/abhishekratnakar/ASL-Generator-recognition-system/scripts)):** 0% implemented (all 30 files are 1-line docstrings).
2. **Language Branch (`src/models/language/` & Notebook 06):** Completely missing; `transformers` / `sentence-transformers` are not even listed in [`requirements.txt`](file:///Users/abhishekratnakar/ASL-Generator-recognition-system/requirements.txt).
3. **Multimodal Alignment & Joint Training (`src/models/multimodal/`, Notebooks 07 & 08):** Completely missing; only unexecuted boilerplate references exist in `09` and `10`.
4. **Controlled Visual Backbone / Temporal / Augmentation Experiments (Notebook 05):** No partial fine-tuning of ResNet50, no temporal encoder comparison (e.g., BiLSTM, GRU, attention pooling, Transformer), no valid spatial/temporal augmentations, no class-balanced sampling.
5. **WLASL Video Recovery / Dataset Coverage:** Only `6,446 / 21,083` official WLASL instances (~30.6%) were usable after the interrupted download run in Notebook `02`.
6. **Proper Preprocessing Validation & Clean Feature Regeneration (Notebooks 02 & 03):** Corrected segment slicing + FPS conversion has never been validated across all 129 non-full validation videos and never used to regenerate features.
7. **Evaluation & Inference Application ([`backend/`](file:///Users/abhishekratnakar/ASL-Generator-recognition-system/backend), [`frontend/`](file:///Users/abhishekratnakar/ASL-Generator-recognition-system/frontend), Notebook 09 & 10):** Unimplemented / unexecuted.
8. **Comprehensive Unit & Integration Tests ([`tests/`](file:///Users/abhishekratnakar/ASL-Generator-recognition-system/tests)):** Missing all functional tests.

---

## 5. Broken Components

### 5.1 CRITICAL: Cached Features Ignore `frame_start` / `frame_end`
- **Where:** [`drive/resnet50_features/`](file:///Users/abhishekratnakar/ASL-Generator-recognition-system/drive/resnet50_features) (and its slices in [`drive/asl_23_class/feature_cache/`](file:///Users/abhishekratnakar/ASL-Generator-recognition-system/drive/asl_23_class/feature_cache) and `asl_60_class/feature_cache/`), plus `preprocess_video()` in [`07_visual_baseline_training.ipynb`](file:///Users/abhishekratnakar/ASL-Generator-recognition-system/notebooks/07_visual_baseline_training.ipynb) (lines 8131–8179) and [`10_multimodal_inference_demo.ipynb`](file:///Users/abhishekratnakar/ASL-Generator-recognition-system/notebooks/10_multimodal_inference_demo.ipynb) (lines 10194–10217).
- **Evidence Verified Locally:** Cross-checking [`drive/drive-download-20261111T091602Z-1-001/val.csv`](file:///Users/abhishekratnakar/ASL-Generator-recognition-system/drive/drive-download-20261111T091602Z-1-001/val.csv) shows **129 of 909 validation videos** (`78` from source `asl5200`/signer `9` and `51` from `northtexas`/signer `121`) have bounded segment annotations (`128` with `frame_start > 1, frame_end > frame_start`; `1` with `frame_start=1, frame_end=77`). In `val_23.csv`, **31 of 57 (54.4%)** videos are segmented recordings averaging **218.58s** full length (up to `459.92s` / `11,498` frames for `51206`) while the sign lasts only **2.98s** on average (`1.68s–5.88s`).
- **Impact:** We verified `X_validation_23.npy` is byte-identical to the slice of `resnet50_features/val/features.npy`, which extracted 32 frames uniformly across the full `0..total_frames-1`. Thus, over half of the 23-class validation set features represent random non-sign video frames, invalidating all cached-feature validation results!

### 5.2 CRITICAL: Unvalidated Segment Indexing & FPS Mismatch in `03` and `04`
- **Where:** `preprocess_video()` in [`03_video_preprocessing (1).ipynb`](file:///Users/abhishekratnakar/ASL-Generator-recognition-system/notebooks/03_video_preprocessing%20(1).ipynb) and [`04_visual_baseline.ipynb`](file:///Users/abhishekratnakar/ASL-Generator-recognition-system/notebooks/04_visual_baseline.ipynb).
- **Bug Details:**
  1. WLASL `frame_start` is **1-based** (`1` means first frame) and defined against metadata `fps` (uniformly `25` across all 129 non-full val rows and 2 non-full train rows), whereas downloaded YouTube videos have varying FPS (e.g., `70173` `drink` and `70348` `before` are `29.97` fps; `12306` `computer` and `51206` `short` are `25.00` fps). Using raw `frame_start` at `29.97` fps shifts a `5710`-frame offset by ~38 seconds!
  2. When `start >= total_frames` or `end < start`, `03` silently falls back to `[0, total-1]` while `04` clamps `start` to `total_frames - 1` (producing 32 duplicate last frames).
  3. **Inconsistent preprocessing across notebooks:** `03` uses sequential read + `cv2.INTER_AREA`; `07_visual_baseline_training` uses random seek (`CAP_PROP_POS_FRAMES`) + `cv2.INTER_LINEAR`; `10` uses random seek + `cv2.INTER_AREA` and duplicates previous frames on decode failures.

### 5.3 Split Defect: 3 Classes Absent from Training Set
- **Where:** [`wlasl_final_optimized_signer_split.csv`](file:///Users/abhishekratnakar/ASL-Generator-recognition-system/drive/drive-download-20261111T091602Z-1-001/wlasl_final_optimized_signer_split.csv) / [`train.csv`](file:///Users/abhishekratnakar/ASL-Generator-recognition-system/drive/drive-download-20261111T091602Z-1-001/train.csv) created by [`05b_rebuild_signer_split.ipynb`](file:///Users/abhishekratnakar/ASL-Generator-recognition-system/notebooks/05b_rebuild_signer_split.ipynb).
- **Verified Details:** `train.csv` covers **1,797 of 1,800 classes**. Classes `312` (`chemical` — in both val & test), `424` (`culture` — in val), and `1464` (`skate` — in val) have **zero training samples**.
- **Related Bug:** [`06_signer_split_validation.ipynb`](file:///Users/abhishekratnakar/ASL-Generator-recognition-system/notebooks/06_signer_split_validation.ipynb) reads `wlasl_available_with_splits.csv` (pre-`05b`) instead of `wlasl_final_optimized_signer_split.csv`, falsely reporting zero missing training classes.

### 5.4 Test-Set Exposure & Unreliable Validation Evaluation
- Both `07_asl_23_class_experiment.ipynb` and `07_asl_60_class_experiment.ipynb` directly evaluate on `X_test_23` / `X_test_60` during exploratory debugging. The test split must be treated as exposed for those exploratory checks and held strictly untouched during future model development until final evaluation.

### 5.5 Fragile / Incompatible Multimodal Stubs (`09` & `10`)
- [`09_multimodal_model_evaluation.ipynb`](file:///Users/abhishekratnakar/ASL-Generator-recognition-system/notebooks/09_multimodal_model_evaluation.ipynb) crashes unless Notebook `03` was run in the exact same runtime (`if "test_loader" not in globals()`), contradicting independent execution.
- `ASLMultimodalModel` in `09`/`10` uses `models.resnet50(weights=None)` and attribute names (`visual_encoder`, `temporal_encoder`) incompatible with `04`'s `VisualBaseline` (`feature_extractor`, `lstm`), and does not put `visual_encoder` into `.eval()` mode during training/eval.

### 5.6 Video `28038` Historical Extraction Error & Unreadable Files
- [`drive/resnet50_features/train/extraction_errors.csv`](file:///Users/abhishekratnakar/ASL-Generator-recognition-system/drive/resnet50_features/train/extraction_errors.csv) records `index=1386, video_id=28038` failed with `Cannot open video`. Later, [`07_visual_baseline_training.ipynb`](file:///Users/abhishekratnakar/ASL-Generator-recognition-system/notebooks/07_visual_baseline_training.ipynb) Cell 3 repaired it (`done[1386]` is now `True`, feature norm `81.1875`, nonzero fraction `0.5045`). However, `28038.mp4` and the `46` unreadable videos from Notebook `02` need clean verification on Drive.

---

## 6. Duplicate or Obsolete Implementations

| Category | Duplicated / Obsolete Items | Recommendation |
|---|---|---|
| **Notebook `07` collisions** | Three `07_*` notebooks: `07_visual_baseline_training.ipynb`, `07_asl_60_class_experiment.ipynb`, `07_asl_23_class_experiment.ipynb`; no `05_*` (only `05b_`), no `08_*`; `(1)` suffixes on `01` and `03`. | Keep existing notebooks untouched in `notebooks/` for provenance, and build clean standardized `01`–`10` Colab notebooks per the new sequence. |
| **Duplicate 23-class feature cache** | [`drive/feature_cache/`](file:///Users/abhishekratnakar/ASL-Generator-recognition-system/drive/feature_cache) and [`drive/asl_23_class/feature_cache/`](file:///Users/abhishekratnakar/ASL-Generator-recognition-system/drive/asl_23_class/feature_cache) are **byte-for-byte identical** (verified via MD5 hashes across all 6 `.npy` files). | Treat `drive/asl_23_class/feature_cache/` as canonical historical artifact; never overwrite original caches. |
| **4× duplicated `preprocess_video`** | Different incompatible versions in `03`, `04`, `07_visual_baseline_training`, and `10`. | Consolidate into a single verified preprocessing implementation in `src/preprocessing/` and reuse identically across notebooks & inference. |
| **4× duplicated LSTM model** | `VisualBaseline` (`04`), `ASLClassifier` (`07_23`, `07_60`), `CachedFeatureLSTM` (`07_vbt`), `ASLMultimodalModel` (`09`, `10`). | Consolidate into `src/models/visual/` and `src/models/multimodal/`. |
| **Obsolete WLASL-100 docs & configs** | `README.md`, `configs/baseline.yaml`, `configs/resnet_lstm.yaml`, `docs/project/*.md`, `metadata/README.md` still claim WLASL-100 (100 classes, 2,038 videos, 97 signers) and list different notebook names (`05_visual_training.ipynb` ... `09_final_evaluation.ipynb`). | Update configs and documentation once Notebook 01–09 milestones are verified. |
| **Hardcoded Drive mount variations** | `07_23` and `07_60` mount `/content/drive_check` while `02`–`06`, `07_vbt`, `09`, `10` mount `/content/drive`. | Standardize all new notebooks to one single configuration cell with `/content/drive/MyDrive/ASL-Recognition`. |

---

## 7. Dataset Paths and Dependencies

### 7.1 Google Drive Paths (Verified from Notebook Code & `drive/` Snapshot)
- **Colab Project Root:** `/content/drive/MyDrive/ASL-Recognition`
  - `metadata/`: `wlasl_official_full_metadata.csv`, `full_wlasl_download_report.csv`, `video_file_audit.csv`, `wlasl_available_with_splits.csv`, `wlasl_optimized_signer_split.csv`, `wlasl_final_optimized_signer_split.csv`, `train.csv`, `val.csv`, `test.csv`, `class_to_id.json`, `id_to_class.json`, `notebook03_readability_audit.csv`, `notebook04_summary.json`, `signer_split_validation.json`, `final_optimized_signer_split_report.json`, `final_optimized_missing_classes.csv`.
  - `videos/`: Raw downloaded video files (`.mp4`, `.webm`, `.mkv`, `.avi`, `.mov`) — **not stored in Git or local `drive/`**; resides in Google Drive.
  - `processed/resnet50_features/{train,val,test}/`: Original ResNet50 feature cache (synced locally in [`drive/resnet50_features/`](file:///Users/abhishekratnakar/ASL-Generator-recognition-system/drive/resnet50_features)).
  - `models/cached_feature_baseline_v2/`: 1,800-class LSTM checkpoints (Drive only).
  - `experiments/asl_23_class/`: Synced locally in [`drive/asl_23_class/`](file:///Users/abhishekratnakar/ASL-Generator-recognition-system/drive/asl_23_class).
  - `experiments/asl_60_class/`: 60-class artifacts (`experiment_summary.json` synced in [`drive/experiment_summary.json`](file:///Users/abhishekratnakar/ASL-Generator-recognition-system/drive/experiment_summary.json)).

### 7.2 Local Workspace Snapshot ([`drive/`](file:///Users/abhishekratnakar/ASL-Generator-recognition-system/drive))
- [`drive/drive-download-20261111T091602Z-1-001/`](file:///Users/abhishekratnakar/ASL-Generator-recognition-system/drive/drive-download-20261111T091602Z-1-001): `wlasl_final_optimized_signer_split.csv` (6,446 rows), `train.csv` (4,619), `val.csv` (909), `test.csv` (918), `class_to_id.json` (1,800), `id_to_class.json` (1,800).
- [`drive/resnet50_features/`](file:///Users/abhishekratnakar/ASL-Generator-recognition-system/drive/resnet50_features): Full `features.npy`, `labels.npy`, `done.npy`, `manifest.csv`, and `train/extraction_errors.csv`.
- [`drive/asl_23_class/`](file:///Users/abhishekratnakar/ASL-Generator-recognition-system/drive/asl_23_class) & [`drive/feature_cache/`](file:///Users/abhishekratnakar/ASL-Generator-recognition-system/drive/feature_cache).

### 7.3 Dependencies
- [`requirements.txt`](file:///Users/abhishekratnakar/ASL-Generator-recognition-system/requirements.txt) includes `torch>=2.0.0`, `torchvision>=0.15.0`, `torchaudio>=2.0.0`, `opencv-python>=4.8.0`, `Pillow>=10.0.0`, `numpy>=1.24.0`, `pandas>=2.0.0`, `scipy>=1.10.0`, `scikit-learn>=1.3.0`, `matplotlib>=3.7.0`, `seaborn>=0.12.0`, `tqdm>=4.65.0`, `pyyaml>=6.0`, `python-dotenv>=1.0.0`.
- **Missing from `requirements.txt`:** `transformers`, `sentence-transformers` (needed by [`configs/multimodal.yaml`](file:///Users/abhishekratnakar/ASL-Generator-recognition-system/configs/multimodal.yaml) and `09`/`10`), `yt-dlp` (used in `02`), `fastapi`/`uvicorn`/`python-multipart` (for Phase 10 API), `tensorboard`, `pytest`.
- **Local macOS Environment Note:** Python 3.14.6 has `numpy`, `pandas`, `pytest`, while `torch`, `cv2`, `yaml` are not installed locally — GPU training runs in Google Colab (`Tesla T4`, PyTorch `2.11.0+cu130`).

---

## 8. Existing Checkpoints and Feature Caches

### 8.1 Verified Feature Caches (All `float16`, Finite, No NaN/Inf)
| Cache Path | Split | Shape | Size | Manifest Match | Validity Status |
|---|---|---|---|---|---|
| `drive/resnet50_features/train/` | Train | `(4619, 32, 2048)` | 577.4 MB | `manifest.csv` & `labels.npy` match `train.csv` 1:1 (`done=4619/4619`) | Extracted w/o `frame_start`/`frame_end` (2 train rows affected) |
| `drive/resnet50_features/val/` | Val | `(909, 32, 2048)` | 113.6 MB | Matches `val.csv` 1:1 (`done=909/909`) | **INVALIDATED:** 129/909 segmented videos uncropped! |
| `drive/resnet50_features/test/` | Test | `(918, 32, 2048)` | 114.8 MB | Matches `test.csv` 1:1 (`done=918/918`) | Extracted w/o `frame_start`/`frame_end` (0 test rows segmented) |
| `drive/asl_23_class/feature_cache/` (& identical `drive/feature_cache/`) | Train / Val / Test | `(126, 32, 2048)` / `(57, 32, 2048)` / `(79, 32, 2048)` | 16M / 7.1M / 9.9M | Verified exact slice of `resnet50_features` | **INVALIDATED:** 31/57 val videos uncropped! |
| `Drive: experiments/asl_60_class/feature_cache/` | Train / Val / Test | `(248, 32, 2048)` / `(169, 32, 2048)` / `(209, 32, 2048)` | `float32` (on Drive) | Built from `resnet50_features` | **INVALIDATED:** contains uncropped val videos! |

### 8.2 Verified Model Checkpoints
| Checkpoint | Location | Architecture | Optimizer / Saved Epoch | Notes |
|---|---|---|---|---|
| `best_model_23.pt` & `latest_checkpoint_23.pt` | [`drive/asl_23_class/models/`](file:///Users/abhishekratnakar/ASL-Generator-recognition-system/drive/asl_23_class/models) | 1-layer LSTM (`128-D`, `dropout=0.3`, `Linear(23)`, 1,118,103 params) | Best Epoch 2 (Val loss `3.1467`) | Trained on broken val cache. |
| `best_model_60.pt` & `latest_checkpoint_60.pt` | Drive (`experiments/asl_60_class/models/`) | 1-layer LSTM (`128-D`, `dropout=0.2`, `Linear(60)`, 1,122,876 params) | Best Epoch 2 (Val loss `4.1030`) | Trained on broken val cache. |
| `best_model.pt` & `latest_checkpoint.pt` | Drive (`models/cached_feature_baseline_v2/`) | 2-layer LSTM (`256-D`, `dropout=0.3`, `Linear(1800)`) | Best Epoch 3 (Val loss `7.4857`) | Trained on broken val cache + 3 zero-train classes. |

---

## 9. Current Training and Evaluation Results

All figures below are taken directly from recorded notebook outputs and saved JSON/CSV logs in [`drive/`](file:///Users/abhishekratnakar/ASL-Generator-recognition-system/drive) (no fabricated metrics):

| Experiment | Split / Samples / Classes | Best Epoch | Train Metrics | Val Metrics | Test Metrics | Observed Failure Mode |
|---|---|---|---|---|---|---|
| **EXP-00-SMOKE** (`04`) | 1 batch (4 samples), 1,800 cls | N/A | Loss `7.4893` (matches `ln(1800)=7.496`) | N/A | N/A | Smoke test only; no training. |
| **1,800-Class Baseline** (`07_vbt`) | Train `4,619` / Val `909` (1,800 cls, 3 missing in train) | 3 (stopped ep 8) | Loss `7.4725`, Top-1 `0.02%`, Top-5 `0.37%` (ep 3) | Loss `7.4857`, Top-1 `0.00%`, Top-5 `0.11%` (ep 3) | Not evaluated | Collapse to 12 predicted classes (`tender` 315, `home` 266, `admire` 226); severe data scarcity + broken val cache. |
| **8-Class Memorization** (`07_vbt`) | 40 samples (8 cls × 5) | 8 | Loss `0.6589`, Top-1 `95.00%` | N/A | N/A | Proves LSTM optimizes cleanly on small separable training set. |
| **60-Class Baseline** (`07_60`) | Train `248` / Val `169` / Test `209` (60 cls) | 2 (stopped ep 10) | Loss `4.0112`, Top-1 `8.06%` (ep 2); `31.05%` (ep 10) | Loss `4.1030`, Top-1 `4.14%`, Top-5 `11.83%` (ep 2) | Loss `4.1272`, Top-1 `0.96%`, Top-5 `8.13%` | Overfits train; collapses to 19 classes on test (`dance` 53, `who` 44); ResNet50 mean centroid sim `0.8905`; Cosine Nearest Centroid Test `2.87%` (`6/209`). |
| **23-Class Baseline** (`07_23`) | Train `126` / Val `57` / Test `79` (23 cls) | 2 (stopped ep 10) | Loss `3.0101`, Acc `14.29%` (ep 2); `54.76%` (ep 10) | Loss `3.1467`, Acc `7.02%`, Macro-F1 `0.0407` (ep 2) | Loss `3.1596`, Top-1 `3.80%` (`3/79`), Macro-F1 `0.0161` | **31/57 val videos uncropped (avg 218.6s)** + only 5–8 train samples/class. |

---

## 10. Recommended Implementation Sequence & Plan

Following the 11-stage Colab-notebook-first protocol (without deleting existing files or overwriting original feature caches):

1. **Stage 1 & Notebook 01 (`01_repository_dataset_audit.ipynb`) [CURRENT STAGE]:**
   - Independently audit Drive workspace, all split CSVs, class/signer distributions, signer independence, missing train classes, video file availability/readability, and existing feature cache manifests/integrity; produce CSVs, distribution plots, and `dataset_audit.md`.
2. **Stage 2 / Notebook 02 (`02_video_preprocessing_validation.ipynb`):**
   - Inspect actual downloaded FPS, duration, `frame_start`, `frame_end`, 1-based vs 0-based indexing, and `metadata_fps -> actual_fps` conversion on representative segmented videos (`70173`, `12306`, `70348`, `51206`, `70355`, plus full-video and `<32`-frame videos); visually confirm the target sign appears in extracted frames; finalize deterministic preprocessing module + unit tests.
3. **Stage 3 / Notebook 03 (`03_resnet50_feature_extraction.ipynb`):**
   - Regenerate a **separate, new** feature cache (e.g., `processed/resnet50_features_v2/`) using the validated segment-aware preprocessing; verify 1:1 manifest alignment and zero NaNs/Infs. Do **not** overwrite `processed/resnet50_features/`.
4. **Stage 4–5 / Notebooks 04 & 05 (`04_visual_baseline_training.ipynb`, `05_visual_model_experiments.ipynb`):**
   - Re-establish an honest visual baseline on clean features (starting with an appropriate well-populated class subset—e.g. investigating recoverable videos and 23/60/N-class eligibility without missing train classes—plus majority/random baselines).
   - Compare controlled experiments: frozen ResNet50 + LSTM, partially fine-tuned ResNet50 (e.g., `layer4` unfreezing with valid temporal/spatial augmentations), and a justified temporal alternative (e.g., BiLSTM + Attention Pooling / lightweight Transformer Encoder). Keep test set strictly untouched.
5. **Stage 6 / Notebook 06 (`06_language_encoder.ipynb`):**
   - Implement and validate pretrained gloss/text encoder (`all-MiniLM-L6-v2` or justified alternative) projecting gloss inputs to L2-normalized shared embedding space (`256-D`), saving vocabulary embeddings and config.
6. **Stage 7–8 / Notebooks 07 & 08 (`07_multimodal_alignment.ipynb`, `08_joint_multimodal_training.ipynb`):**
   - Build genuine two-branch multimodal alignment (Visual Projection + Language Projection into shared `256-D` space) with InfoNCE contrastive alignment and closed-vocabulary classification/matching evaluation against the visual-only baseline on identical validation splits.
7. **Stage 9 / Notebook 09 (`09_final_evaluation.ipynb`):**
   - Single-pass final evaluation reporting Top-1, Top-5, Macro Precision/Recall/F1, per-class metrics, confusion matrix, retrieval Recall@K, inference latency, model size, and clear caveats about test exposure.
8. **Stage 10–11 / Notebook 10 (`10_asl_inference_demo.ipynb`) + Local Repo Integration:**
   - Refactor all verified modules into [`src/`](file:///Users/abhishekratnakar/ASL-Generator-recognition-system/src) and [`scripts/`](file:///Users/abhishekratnakar/ASL-Generator-recognition-system/scripts), add full pytest coverage in [`tests/`](file:///Users/abhishekratnakar/ASL-Generator-recognition-system/tests), implement FastAPI inference endpoints (`/health`, `/predict`, `/predict/video`) in [`backend/`](file:///Users/abhishekratnakar/ASL-Generator-recognition-system/backend), and update [`README.md`](file:///Users/abhishekratnakar/ASL-Generator-recognition-system/README.md) & [`configs/`](file:///Users/abhishekratnakar/ASL-Generator-recognition-system/configs).


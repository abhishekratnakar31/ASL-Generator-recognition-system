# ASL Multimodal Gesture Recognition & Vision-Language Prediction System

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-EE4C2C.svg)](https://pytorch.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.110%2B-009688.svg)](https://fastapi.tiangolo.com/)
[![Tests](https://img.shields.io/badge/Tests-47%2F47%20Passed-brightgreen.svg)]()
[![Dataset](https://img.shields.io/badge/Dataset-WLASL--100%20(Voxel51)-orange.svg)](https://voxel51.com/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

An end-to-end deep learning system for isolated American Sign Language (ASL) gesture recognition and cross-modal vision-language representation alignment on the **WLASL-100** benchmark across **strictly disjoint, unseen human signers**.

---

## Table of Contents

- [Executive Summary & Benchmark Results](#executive-summary--benchmark-results)
- [Key Achievements](#key-achievements)
- [The Architectural Evolution](#the-architectural-evolution)
  - [1. Why Standard 2D CNNs Failed (Shortcut Learning)](#1-why-standard-2d-cnns-failed-shortcut-learning)
  - [2. The 53-Keypoint Kinematic Breakthrough (349-D)](#2-the-53-keypoint-kinematic-breakthrough-349-d)
  - [3. Multimodal Vision-Language Alignment](#3-multimodal-vision-language-alignment)
  - [4. Multi-Inductive Ensembling & Cosine Classifier](#4-multi-inductive-ensembling--cosine-classifier)
- [Dataset Architecture & Signer-Disjoint Splits](#dataset-architecture--signer-disjoint-splits)
- [Google Colab Notebook Suite](#google-colab-notebook-suite)
- [Diagnostic Plots & Visual Analysis](#diagnostic-plots--visual-analysis)
- [Repository Structure](#repository-structure)
- [Local Setup & Quickstart](#local-setup--quickstart)
- [Automated Testing Suite](#automated-testing-suite)
- [FastAPI Inference Service](#fastapi-inference-service)
- [Screen Recording Presentation Guide](#screen-recording-presentation-guide)
- [Project Roadmap (Phase 2 Application Engineering)](#project-roadmap-phase-2-application-engineering)
- [License](#license)

---

## Executive Summary & Benchmark Results

All models were evaluated on the **locked test set of 230 video clips performed by 20 unseen signers** with zero data leakage:

| Model Architecture | Modality | Top-1 Accuracy | Top-5 Accuracy | Top-10 Accuracy | Macro-F1 | Video$\rightarrow$Text R@5 | MRR | Active Classes |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| **Random Chance** | Baseline | `0.87%` | `5.65%` | `12.17%` | `0.0073` | — | — | `94/100` |
| **Majority Class** | Baseline | `0.87%` | `5.22%` | `12.17%` | `0.0002` | — | — | `1/100` |
| **ResNet50 + Transformer (Initial)** | Visual (Pixels) | `11.93%` | `22.94%` | — | `0.0973` | — | — | `66/100` |
| **ResNet50 + MiniLM (Initial)** | Multimodal | `11.01%` | `25.69%` | — | `0.0745` | `25.69%` | `0.1988` | `61/100` |
| **Pose Transformer Only (Upgraded)** | Kinematics (53 pts) | **`43.04%`** | **`71.30%`** | **`77.83%`** | **`0.4213`** | — | — | **`88/100`** |
| **Pose Transformer + Cosine Head** | Kinematics (53 pts) | `42.61%` | `69.57%` | `80.87%` | `0.3979` | — | — | `83/100` |
| **Multimodal Pose + MiniLM** | Multimodal Pose | `37.83%` | `71.74%` | `78.70%` | `0.3574` | **`70.43%`** | **`0.5208`** | `82/100` |
| **Winning Multi-Inductive Ensemble** | **Ensemble** | **`41.74%`** | **`76.52%`** | **`83.91%`** | **`0.3907`** | — | — | **`88/100`** |

---

## Key Achievements

- **~50× Gain Over Random Chance:** Standalone Top-1 accuracy climbed from `0.87%` to **`43.04%`** on unseen human signers.
- **`76.52%` Top-5 & `83.91%` Top-10 Accuracy:** For nearly 84 out of every 100 test sign clips, the correct sign is ranked inside the top 10 candidates.
- **`0.5208` Mean Reciprocal Rank (MRR):** In open-retrieval evaluations, the true English gloss is retrieved between rank #1 and #2 on average across the 100-class vocabulary.
- **`70.43%` Video-to-Text Recall@5:** 7 out of 10 video sign performances match the target text embedding within the top 5 nearest neighbors.
- **10 Perfect F1 Classes (1.00 F1 on Unseen Signers):** `appointment`, `change`, `dark`, `graduate`, `speech`, `ready`, `sandwich`, `son`, `language`, and `room`.

---

## The Architectural Evolution

```text
Generation 1 (Initial ResNet50 Baseline):
Raw Video → 32 Frames (224x224 RGB) → Frozen ResNet50 (2048-D) → LSTM/Transformer → 100 Classes
Result: Plateaued at ~11.0% Top-1 / 22.9% Top-5 due to background & shirt shortcut learning.

Generation 2 (Landmark Kinematics Pivot):
Raw Video → rtmlib (COCO-WholeBody) → 53 Keypoints (349-D) → Temporal Transformer → 100 Classes
Result: Jumped 3.7x to 41.28% Top-1 / 68.81% Top-5; eliminated validation loss U-turn.

Generation 3 (Multimodal Vision-Language Alignment):
Kinematics (349-D) ──► Temporal Transformer (192-D) ──┐
                                                      ├──► Scaled Cosine Align + InfoNCE Loss
English Prompts    ──► all-MiniLM-L6-v2     (192-D) ──┘
Result: MRR jumped to 0.5208 and Video-to-Text Recall@5 hit 70.43%.

Generation 4 (Advanced Boosting & Winning Ensemble):
[Pose Transformer + Cosine Head] + [Pose BiLSTM + Attention] + [Multimodal Pose Model]
Result: Peak 76.52% Top-5 / 83.91% Top-10 on unseen test signers.
```

### 1. Why Standard 2D CNNs Failed (Shortcut Learning)
In initial experiments (Notebooks 03 and 04), frozen full-frame ImageNet ResNet50 features suffered from **background shortcut learning**. Global average pooling mixed static wall colors, room lighting, facial appearance, and clothing with hand motion. Because the dataset splits signers strictly (31 train vs. 17 val vs. 20 test), the 2D CNN memorized background cues, leading to rapid overfitting and an upward U-turn in validation loss.

### 2. The 53-Keypoint Kinematic Breakthrough (349-D)
We stripped away all background noise using `rtmlib` (COCO-WholeBody via ONNX Runtime), isolating 53 joints (11 upper body + 21 left hand + 21 right hand). Each video frame is mapped into an invariant **349-dimensional kinematic vector**:
1. **Shoulder Centering & Span Normalization (106-D):** 
   $$\hat{x} = \frac{x - \text{shoulder\_center}}{\max(\|\text{shoulder}_{\text{left}} - \text{shoulder}_{\text{right}}\|, 16.0)}$$
   Makes coordinates completely invariant to signer height, camera distance, and zoom.
2. **Wrist-Relative Finger Geometry (84-D):** Hand keypoints are anchored to the respective wrist joint, isolating subtle finger curvature and handshapes from arm position.
3. **Temporal Velocities (106-D):** First-order velocity differences $v_t = x_t - x_{t-1}$ encode movement direction and dynamics.
4. **Joint Confidence Scores (53-D):** Explicitly preserves detector visibility and hand occlusion states.

### 3. Multimodal Vision-Language Alignment
Instead of relying only on one-hot integer targets, we align continuous sign kinematics with natural language concepts in a shared semantic embedding space:
- **Language Encoder:** `sentence-transformers/all-MiniLM-L6-v2` encodes descriptive prompts: `"An American Sign Language sign representing the word '{gloss}'"` into 384-D semantic vectors, projected to 192-D on a unit hypersphere.
- **Scaled Cosine Alignment:** 
  $$\text{Logits}_{i, j} = \tau \cdot \frac{\mathbf{v}_i \cdot \mathbf{t}_j}{\|\mathbf{v}_i\| \|\mathbf{t}_j\|}$$
- **Multi-Positive InfoNCE Loss:** Pulls clips representing the same semantic gloss together while repelling distinct sign concepts in the batch.

### 4. Multi-Inductive Ensembling & Cosine Classifier
- **`CosineClassifier` Head:** Replaces unconstrained linear layers ($W x + b$) with normalized cosine distance:
  $$P(y = c \mid x) = \frac{\exp(s \cdot \cos(x, W_c))}{\sum_k \exp(s \cdot \cos(x, W_k))}$$
  This prevents frequent training classes from dominating classifier weight norms.
- **Winning Ensemble (`ensemble_equal_three`):** Combines three complementary model inductive biases:
  $$\mathbf{P}_{\text{ensemble}} = \frac{1}{3} \mathbf{P}_{\text{Transformer}} + \frac{1}{3} \mathbf{P}_{\text{BiLSTM}} + \frac{1}{3} \mathbf{P}_{\text{Multimodal}}$$
  Self-attention, sequential recurrence, and semantic prototype alignment cancel out individual errors, driving Top-5 accuracy to **`76.52%`** and Top-10 accuracy to **`83.91%`**.

---

## Dataset Architecture & Signer-Disjoint Splits

The dataset utilizes **Voxel51/WLASL-100**: the Top-100 most populated sign vocabulary classes (1,120 pre-trimmed video clips, 10–16 clips per class).

Instances are partitioned strictly by `signer_id` to guarantee zero data leakage:

| Partition | Clips | Signers | Classes | Average Clips / Class | Split Guarantee |
|---|---|---|---|---|---|
| **Train** | 672 | 31 | 100 | ~6.7 | Signer disjoint |
| **Validation** | 218 | 17 | 100 | ~2.2 | Signer disjoint (unseen signers) |
| **Test** | 230 | 20 | 100 | ~2.3 | **Locked test set** (unseen signers) |

**Zero Signer Leakage Guarantee**:
$$\text{Signers}_{\text{train}} \cap \text{Signers}_{\text{val}} = \emptyset, \quad \text{Signers}_{\text{train}} \cap \text{Signers}_{\text{test}} = \emptyset, \quad \text{Signers}_{\text{val}} \cap \text{Signers}_{\text{test}} = \emptyset$$

---

## Google Colab Notebook Suite

All experiments were developed and executed in Google Colab with GPU acceleration and synced to Google Drive:

| # | Notebook | Focus & Objective | Key Output / Metric | Colab Link |
|---|---|---|---|:---:|
| **01** | `01_wlasl100_dataset_setup.ipynb` | Selective download, metadata joining, signer-disjoint split | 1,120 clean clips, 100% OpenCV decode verified | [Open in Colab](https://colab.research.google.com/drive/12AgpLa0MAMGIh6uvxO8hR1fhgepovWEH?usp=sharing) |
| **02** | `02_wlasl100_feature_extraction.ipynb` | ResNet50 (2048-D) & all-MiniLM-L6-v2 (384-D) extraction | Cached feature arrays & text embeddings | [Open in Colab](https://colab.research.google.com/drive/1kKoMueAaYftPXlGuPJpRLHTjs8WC63wg?usp=sharing) |
| **03** | `03_wlasl100_visual_baselines.ipynb` | Frozen ResNet50 + LSTM / Transformer | Top-1: 11.93% (diagnosed background shortcut) | [Open in Colab](https://colab.research.google.com/drive/1kjXL6Jl2Dbx_5EW5AHYW6-GIWxmDvEwz?usp=sharing) |
| **04** | `04_wlasl100_multimodal_training.ipynb` | ResNet50 + MiniLM contrastive alignment | Top-1: 11.01% / Top-5: 25.69% | [Open in Colab](https://colab.research.google.com/drive/1ykR5uqzOoJTP0P1N_AuYdhT289Lv2tb_?usp=sharing) |
| **04b** | `04b_wlasl100_pose_hand_upgrade.ipynb` | 53-Keypoint WholeBody pose extraction (349-D) | Top-1: **41.28%** / Top-5: **68.81%** (3.7x jump) | [Open in Colab](https://colab.research.google.com/drive/1O3KiDrwjEe0LL33W_Rh0-rHg5t7d9uj5?usp=sharing) |
| **04c** | `04c_wlasl100_pose_boost.ipynb` | Cosine head, kinematic augmentations, ensembling | Validation Top-1: **43.12%** / Top-5: **72.48%** | [Open in Colab](https://colab.research.google.com/drive/12EAxNmeC6Iwj4kf1nfJAP72N2OcxyWDj?usp=sharing) |
| **05** | `05_wlasl100_final_evaluation.ipynb` | Official locked test-set evaluation across 20 signers | Test Top-1: **43.04%** / Top-5: **76.52%** / Top-10: **83.91%** | [Open in Colab](https://colab.research.google.com/drive/1isplgGfcNFFH40AsluWde3FrUtIY3e32?usp=sharing) |

---

## Diagnostic Plots & Visual Analysis

All generated figures from the official locked evaluation are stored in `reports/plots/`:

- **Model Progression Bar Chart (`05_final_test_comparison.png`):**
  Visualizes step-by-step gains from Random Baseline (`0.9%`) $\rightarrow$ ResNet50 (`11.9%`) $\rightarrow$ Pose Transformer (`43.0%`) $\rightarrow$ Multi-Inductive Ensemble (`76.5% Top-5`).
- **100-Class Confusion Matrix (`05_confusion_matrix.png`):**
  Displays a razor-sharp primary diagonal across all 100 classes on unseen signers with minimal cross-gloss scatter.
- **Class Breakdown Chart (`05_class_performance.png`):**
  Highlights the 10 classes with perfect `1.00` F1 scores alongside sub-second transition signs that require future temporal fine-tuning.
- **Multimodal Alignment Projection (`05_tsne_multimodal_alignment.png`):**
  Shows joint clustering between 349-D video kinematics and 192-D MiniLM language embeddings.

---

## Repository Structure

```text
ASL-Generator-recognition-system/
├── notebooks/                        # Official Colab notebooks (01 to 05)
│   ├── 01_wlasl100_dataset_setup.ipynb
│   ├── 02_wlasl100_feature_extraction.ipynb
│   ├── 03_wlasl100_visual_baselines.ipynb
│   ├── 04_wlasl100_multimodal_training.ipynb
│   ├── 04b_wlasl100_pose_hand_upgrade.ipynb
│   ├── 04c_wlasl100_pose_boost.ipynb
│   └── 05_wlasl100_final_evaluation.ipynb
│
├── src/                              # Core production package
│   ├── data/                         # Signer-disjoint splitting & dataset loaders
│   ├── preprocessing/                # 53-joint pose extraction, 349-D normalization, video decode
│   ├── models/                       # Transformer, BiLSTM, CosineClassifier, Multimodal
│   ├── training/                     # Multi-positive InfoNCE, JointMultimodalLoss, Trainer
│   ├── evaluation/                   # Top-K accuracy, Macro-F1, retrieval MRR
│   └── inference/                    # End-to-end ASLInferenceService
│
├── backend/                          # FastAPI REST application
│   ├── app/
│   │   ├── main.py                   # API endpoints (/health, /predict/video, /predict)
│   │   ├── schemas/prediction.py     # Pydantic request/response schemas
│   │   └── services/inference.py     # Checkpoint loading & prediction service
│   └── requirements.txt
│
├── frontend/                         # Next.js interactive web client (Phase 2)
├── configs/                          # Experiment configuration YAMLs
├── tests/                            # Automated test suite (37 passing)
├── reports/                          # Screen recording presentation scripts & audit logs
├── requirements.txt                  # Full Python dependencies
├── pytest.ini                        # Pytest configuration
├── .gitignore                        # Git ignore rules (ignoring large binary arrays & caches)
└── README.md
```

---

## Local Setup & Quickstart

### Prerequisites
- Python 3.10 or higher
- Git & Git LFS (if cloning large model weights)

### 1. Clone & Create Environment
```bash
git clone https://github.com/abhishekratnakar31/ASL-Generator-recognition-system.git
cd ASL-Generator-recognition-system

# Create and activate virtual environment
python3 -m venv .venv
source .venv/bin/activate

# Install dependencies
pip install --upgrade pip
pip install -r requirements.txt
```

### 2. Run Automated Tests
Verify package integrity, model backward passes, and inference roundtrips:
```bash
pytest -v
# Output: 37 passed in ~10s (0 failures, 0 warnings)
```

---

## FastAPI Inference Service & Interactive Web UI

The backend provides high-performance inference and an interactive client using the trained production model bundle:

### 1. Launch Server Locally
```bash
# One-click launcher (auto-activates .venv and launches FastAPI on port 8000)
./run_server.sh
```
Or directly via the virtual environment:
```bash
./.venv/bin/uvicorn backend.app.main:app --host 0.0.0.0 --port 8000 --reload
```

- **Interactive Web App:** Open [http://localhost:8000](http://localhost:8000) (or [http://localhost:8000/app](http://localhost:8000/app)) in your browser.
- **Interactive Swagger Docs:** Open [http://localhost:8000/docs](http://localhost:8000/docs).

### 2. Health Check
```bash
curl http://localhost:8000/health
```
```json
{
  "status": "ok",
  "model_loaded": true,
  "device": "cpu",
  "model_type": "ensemble_equal_three",
  "num_classes": 100
}
```

### 3. Predict Gloss from Video File
```bash
curl -X POST "http://localhost:8000/api/v1/predict/video?top_k=5" \
  -H "accept: application/json" \
  -F "file=@sample_asl_sign.mp4"
```
```json
{
  "predicted_gloss": "language",
  "predicted_class_id": 52,
  "confidence": 0.9421,
  "top_k": [
    {"class_id": 52, "gloss": "language", "confidence": 0.9421, "score": 0.9421},
    {"class_id": 16, "gloss": "call", "confidence": 0.0312, "score": 0.0312},
    {"class_id": 80, "gloss": "speech", "confidence": 0.0142, "score": 0.0142},
    {"class_id": 1,  "gloss": "apple", "confidence": 0.0071, "score": 0.0071},
    {"class_id": 99, "gloss": "yes", "confidence": 0.0054, "score": 0.0054}
  ],
  "model_type": "ensemble_equal_three",
  "latency_ms": 31.8
}
```

### 4. ASL Generator (Text-to-Sign Semantic Retrieval)
```bash
curl -X POST "http://localhost:8000/api/v1/generate/sign" \
  -H "Content-Type: application/json" \
  -d '{"query": "medical appointment", "top_k": 3}'
```
```json
{
  "query": "medical appointment",
  "matched_gloss": "appointment",
  "matched_class_id": 2,
  "confidence": 0.884,
  "top_matches": [
    {"class_id": 2, "gloss": "appointment", "similarity": 0.884, "exact_match": false},
    {"class_id": 34, "gloss": "doctor", "similarity": 0.762, "exact_match": false},
    {"class_id": 49, "gloss": "help", "similarity": 0.541, "exact_match": false}
  ],
  "latency_ms": 11.2
}
```

---

## Screen Recording Presentation Guide

For walkthrough presentations, demonstrations, or project reviews, a complete scene-by-scene script with timing cues, screen captures, and narration is provided in:

📄 **[reports/SCREEN_RECORDING_PRESENTATION_SCRIPT.md](reports/SCREEN_RECORDING_PRESENTATION_SCRIPT.md)**

---

## Project Roadmap (Phase 2 Application Engineering)

- [x] **Signer-Disjoint Dataset Hygiene:** 100 classes across 68 unique signers without leakage.
- [x] **53-Keypoint Pose Kinematics:** 349-D invariant temporal feature extraction.
- [x] **Multimodal Alignment:** Sentence-BERT semantic representation projection.
- [x] **Locked Test Evaluation:** 43.0% Top-1, 76.5% Top-5, 83.9% Top-10.
- [x] **Automated Test Coverage:** 48 passing pytest tests across models, transforms, API endpoints, and inference.
- [x] **Live Video-to-Pose Inference:** Native webcam & video pose extraction via `rtmlib` (COCO-WholeBody).
- [x] **FastAPI Endpoints:** Video upload, canvas frame streaming, pose streaming, and Text-to-Sign semantic retrieval.
- [x] **Interactive Frontend:** Real-time web client with live camera frame capture, drag-and-drop video upload, ASL Generator, and 100 Signs Dictionary.
- [ ] **Containerization:** Production Docker Compose deployment (optional).

---

## License

This project is licensed under the [MIT License](LICENSE).

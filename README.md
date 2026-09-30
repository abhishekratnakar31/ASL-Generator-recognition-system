# ASL Multimodal Gesture Recognition & Vision-Language Prediction System

An end-to-end deep learning system designed for isolated American Sign Language (ASL) gesture recognition and vision-language representation alignment using the **WLASL-100** benchmark dataset.

---

## Project Motivation & Goals

Sign language is a primary natural communication medium for millions in the Deaf and Hard-of-Hearing communities. Building robust automated sign language recognition systems bridges communication barriers, but requires addressing complex visual challenges: subtle hand configurations, rapid motion, body posture variations, and significant signer appearance variance.

### Core Objectives
1. **Signer-Independent Recognition**: Develop a video processing and deep learning pipeline that generalizes across unseen signers rather than overfitting to specific individuals.
2. **Temporal Visual Modeling**: Extract spatial features using a deep convolutional backbone (ResNet50) and model dynamic temporal dependencies over time using recurrent networks (LSTM).
3. **Multimodal Alignment**: Align continuous visual sign video embeddings with discrete text/gloss embeddings in a shared 256-dimensional semantic representation space.
4. **End-to-End Product Deployment**: Following full ML experimentation and validation in Google Colab, transition the finalized model into an IDE to build a high-performance FastAPI inference backend and a responsive Next.js frontend with real-time video prediction.

---

## Dataset & Signer-Independent Split

The project utilizes the **WLASL-100** subset, encompassing 100 high-frequency sign vocabulary classes.

- **Classes**: 100
- **Total Expected Videos**: 2,038
- **Total Unique Signers**: 97
- **Primary Metadata**: `metadata/wlasl100_metadata_with_split.csv`

### Partition Strategy

To evaluate genuine generalization to new users, instances are partitioned strictly by `signer_id`:

| Partition | Signers | Classes | Expected Videos | Status |
|---|---|---|---|---|
| **Train** | 68 | 100 | 1,234 | Signer disjoint |
| **Validation** | 15 | 100 | 426 | Signer disjoint |
| **Test** | 14 | 100 | 378 | Signer disjoint |

**Zero Signer Leakage Guarantee**:
$$\text{Signers}_{\text{train}} \cap \text{Signers}_{\text{val}} = \emptyset, \quad \text{Signers}_{\text{train}} \cap \text{Signers}_{\text{test}} = \emptyset, \quad \text{Signers}_{\text{val}} \cap \text{Signers}_{\text{test}} = \emptyset$$

> **Dataset Availability Note**: Currently, ~1,019 videos are available locally/in Google Drive out of the 2,038 expected instances. A recovery request for the remaining videos has been submitted. Initial pipeline sanity testing and training iterations proceed using available instances; definitive benchmark metrics will be established upon full recovery.

---

## System Architecture

### 1. Visual Pipeline (Current Focus)

```text
Video
  ↓
Video Preprocessing (segment slicing, frame extraction)
  ↓
32 Sampled Frames
  ↓
224 × 224 RGB Frames with ImageNet Normalization: [B, 32, 3, 224, 224]
  ↓
ResNet50 Feature Extractor (Pretrained on ImageNet, Frozen)
  ↓
2048-D Spatial Features per Frame: [B, 32, 2048]
  ↓
2-Layer LSTM (Hidden size: 256, Dropout: 0.3)
  ↓
Visual Representation (Temporal sequence modeling): [B, 256]
  ↓
Linear Classification Head
  ↓
100-Class Logits: [B, 100]
```

### 2. Multimodal Vision-Language Alignment (Planned Stage)

```text
Visual Stream                                Language Stream
Video → Preprocessing → ResNet50 + LSTM      ASL Gloss Token
               ↓                                     ↓
     256-D Visual Embedding               256-D Language Embedding
               │                                     │
               └──────────────► ◄────────────────────┘
                           Alignment
                 (Contrastive Cosine Similarity)
```

---

## Repository Structure

```text
ASL-Recognition/
├── README.md
├── LICENSE
├── requirements.txt
├── .gitignore
├── .env.example
│
├── notebooks/                     # Google Colab ML notebooks (to be added; tracked via .gitkeep)
│   └── .gitkeep
│
├── src/                           # Clean, reusable modular implementations
│   ├── data/                      # Dataset, split, and sampling classes
│   ├── preprocessing/             # Video reading, frame transforms, augmentations
│   ├── models/                    # Visual, language, and multimodal architectures
│   ├── training/                  # Trainers, training loops, loss functions
│   └── evaluation/                # Top-1/Top-5 metrics, evaluation scripts, plots
│
├── configs/                       # Experiment configuration YAML files
├── metadata/                      # Dataset metadata schemas and documentation
├── models/                        # Checkpoints and final exported weights
├── results/                       # Metrics logs, loss plots, confusion matrices
├── docs/                          # Comprehensive architecture and project docs
├── backend/                       # Future FastAPI inference service
├── frontend/                      # Future Next.js user interface
├── scripts/                       # Command-line utility scripts
└── .github/                       # CI/CD workflows and issue templates
```

---

## Google Colab Notebooks

ML and model development is actively conducted in **Google Colab** connected to Google Drive storage (`PROJECT_DIR = Path("/content/drive/MyDrive/ASL-Recognition")`).

> [!TIP]
> **How to Add Your Colab Links**:
> To link your notebooks for mentors and collaborators, replace `PASTE_COLAB_URL_HERE` in the table below with your notebook's Google Colab share link (e.g., `https://colab.research.google.com/drive/...`). The actual `.ipynb` files will be saved directly into [`notebooks/`](file:///Users/abhishekratnakar/ASL-Generator-recognition-system/notebooks) once each stage is validated.

| Notebook | Description | Status | Colab Link |
|---|---|---|---|
| `01_environment_setup.ipynb` | Colab setup, PyTorch, GPU verification | **COMPLETED** | [Open in Colab](https://colab.research.google.com/drive/1qZap3MAxTitMWBjwlGohDIFpxJVkPoyo?usp=sharing) |
| `02_dataset_setup.ipynb` | WLASL-100 metadata, class mapping, signer split | **COMPLETED** | [Open in Colab](https://colab.research.google.com/drive/1qX0VA66-4hFp9-WslgDGIO9YdOOl27RU?usp=sharing) |
| `03_video_preprocessing.ipynb` | Video decoding, 32-frame sampling, DataLoader | **COMPLETED** | [Open in Colab](https://colab.research.google.com/drive/1wMA0PbR8gQREaBi20iHGv7yDAMgUDKaf?usp=sharing) |
| `04_visual_baseline.ipynb` | ResNet50 + LSTM architecture & smoke test | **COMPLETED** | [Open in Colab](https://colab.research.google.com/drive/1NgXOM5puF4wOlZNSvdi16dyqnmwnaxuM?usp=sharing) |
| `05_visual_training.ipynb` | Visual model training & checkpointing | **NEXT** | [Open in Colab](PASTE_COLAB_URL_05_HERE) |
| `06_architecture_experiments.ipynb` | Temporal pooling, unfreezing, augmentations | **PLANNED** | [Open in Colab](PASTE_COLAB_URL_06_HERE) |
| `07_language_encoder.ipynb` | Gloss representation & text embeddings | **PLANNED** | [Open in Colab](PASTE_COLAB_URL_07_HERE) |
| `08_multimodal_training.ipynb` | Vision-language contrastive alignment | **PLANNED** | [Open in Colab](PASTE_COLAB_URL_08_HERE) |
| `09_final_evaluation.ipynb` | Final benchmark, error analysis, model export | **PLANNED** | [Open in Colab](PASTE_COLAB_URL_09_HERE) |

> *Note: Replace placeholders with verified Colab URLs as notebooks undergo formal review.*

---

## Current Progress Tracker

- [x] **Environment setup**: Colab environment, PyTorch, CUDA GPU verified
- [x] **WLASL-100 dataset setup**: Raw metadata inspected, 100 classes mapped
- [x] **Metadata processing**: Cleaned and validated dataset records
- [x] **Class mapping**: Bidirectional mapping between 100 glosses and integer class IDs
- [x] **Signer analysis**: Inspected distribution across 97 signers
- [x] **Signer-independent split**: Created mutually exclusive Train/Val/Test partitions by signer ID
- [x] **Video preprocessing**: OpenCV segment decoding, uniform 32-frame sampling
- [x] **Dataset / DataLoader**: PyTorch `Dataset` with ImageNet normalization transforms
- [x] **ResNet50 feature extractor**: Pretrained spatial backbone integrated and frozen
- [x] **LSTM temporal encoder**: 2-layer recurrent network with hidden size 256
- [x] **Visual baseline smoke test**: Forward pass `[4, 32, 3, 224, 224] → [4, 100]`, initial loss ~4.61, backward pass and gradients verified
- [ ] **Full visual model training** (*NEXT STEP: Notebook 05*)
- [ ] **Architecture experiments** (Notebook 06)
- [ ] **Language encoder** (Notebook 07)
- [ ] **Multimodal training** (Notebook 08)
- [ ] **Final evaluation** (Notebook 09)
- [ ] **Model export** (ONNX / TorchScript)
- [ ] **FastAPI backend** (Phase 2)
- [ ] **Next.js frontend** (Phase 2)
- [ ] **Docker containerization** (Phase 2)
- [ ] **Production deployment** (Phase 2)

---

## Experimental Results Policy & Current Status

To maintain academic and engineering rigor:
- **No synthetic metrics**: All reported metrics originate exclusively from verified experimental logs.
- **Smoke Test Results (`EXP-00-SMOKE`)**:
  - Input batch tensor: `torch.Size([4, 32, 3, 224, 224])`
  - Output logits: `torch.Size([4, 100])`
  - CrossEntropyLoss: `~4.61` (matching theoretical random chance $-\ln(0.01) \approx 4.605$)
  - Parameter freezing: ResNet50 parameters verified frozen (`requires_grad = False`); LSTM and classifier weights successfully updated with AdamW.
- **Full Model Training**: Has **not** been performed yet. Scheduled for `05_visual_training.ipynb`.

---

## Development Workflow

We follow a strict verification protocol:
1. **One Notebook at a Time**: Formulate objective, explain logic, write code, run in Google Colab, inspect output, and verify gradients/losses.
2. **Stable Code Refactoring**: Once notebook behavior is verified, refactor stable components into modular files under [`src/`](file:///Users/abhishekratnakar/ASL-Generator-recognition-system/src).
3. **Transition to IDE**: Following completion of Notebooks 01–09 and export of final model checkpoints, backend, frontend, and deployment will be developed locally in an IDE.

---

## Backend & Frontend Architecture (Phase 2)

Application engineering begins strictly after ML model development and final evaluation are complete:

- **Backend (FastAPI)**:
  - Video stream ingestion and validation
  - Preprocessing pipeline execution
  - Model inference using exported weights
  - Top-K prediction response formatting with confidence scores
- **Frontend (Next.js + TypeScript + Tailwind CSS)**:
  - Camera capture and video clip upload UI
  - Real-time gesture prediction display
  - Top confidence ratings and sign vocabulary explorer

---

## Documentation Index

- [Visual Pipeline Architecture](file:///Users/abhishekratnakar/ASL-Generator-recognition-system/docs/architecture/visual_pipeline.md)
- [Multimodal Pipeline Architecture](file:///Users/abhishekratnakar/ASL-Generator-recognition-system/docs/architecture/multimodal_pipeline.md)
- [End-to-End System Architecture](file:///Users/abhishekratnakar/ASL-Generator-recognition-system/docs/architecture/system_architecture.md)
- [Project Roadmap](file:///Users/abhishekratnakar/ASL-Generator-recognition-system/docs/project/roadmap.md)
- [Dataset Documentation](file:///Users/abhishekratnakar/ASL-Generator-recognition-system/docs/project/dataset.md)
- [Experiment Tracking Log](file:///Users/abhishekratnakar/ASL-Generator-recognition-system/docs/project/experiments.md)
- [Metadata Documentation](file:///Users/abhishekratnakar/ASL-Generator-recognition-system/metadata/README.md)

---

## License

This project is open-source under the [MIT License](file:///Users/abhishekratnakar/ASL-Generator-recognition-system/LICENSE).

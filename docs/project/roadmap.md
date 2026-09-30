# Project Roadmap

## Phase 1: Machine Learning & Modeling (Colab Environment)

- [x] **01 Environment Setup**: Google Colab environment, PyTorch, GPU verification.
- [x] **02 Dataset Setup**: WLASL-100 metadata parsing, class mapping, signer-independent split generation.
- [x] **03 Video Preprocessing**: Frame sampling (32 frames), 224x224 resize, ImageNet normalization, Dataset/DataLoader.
- [x] **04 Visual Baseline**: ResNet50 + LSTM architecture definition, smoke test, frozen backbone verification.
- [ ] **05 Visual Training** (*NEXT STEP*): Train ResNet50 + LSTM baseline on available data; scale upon full video recovery.
- [ ] **06 Architecture Experiments**: Temporal pooling, LSTM layer tuning, partial unfreezing, augmentations.
- [ ] **07 Language Encoder**: Text/gloss representation extraction and projection.
- [ ] **08 Multimodal Training**: Contrastive vision-language representation alignment.
- [ ] **09 Final Evaluation**: Top-1/Top-5 test accuracy, confusion matrix, error analysis, model checkpoint export.

## Phase 2: Application Development (IDE Environment)

- [ ] **Backend API**: FastAPI service for video ingestion and batch/stream model inference.
- [ ] **Frontend Application**: Next.js client for webcam capture, video upload, and gloss prediction visualization.
- [ ] **Containerization & Deployment**: Docker Compose setup and production deployment.

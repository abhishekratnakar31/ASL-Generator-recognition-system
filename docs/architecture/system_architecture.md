# End-to-End System Architecture

```text
User / Client Device
       │  (Webcam stream / recorded video)
       ▼
Frontend (Next.js + TypeScript + Tailwind CSS)
       │  (Multipart video upload / WebSocket frames)
       ▼
Backend Inference API (FastAPI + Python + PyTorch)
       │  (Frame sampling & preprocessing)
       ▼
Inference Engine (Trained Visual / Multimodal ASL Model)
       │  (Softmax logits & Top-K predictions)
       ▼
Backend API Response (JSON: Gloss, Confidence, Alternatives)
       │
       ▼
Frontend UI Rendering (Real-time translation & confidence display)
```

## Phase Boundaries

1. **Phase 1: ML & Model Development (Current)**: Executed in Google Colab with Google Drive storage.
2. **Phase 2: Product Engineering (Planned)**: Model exported to ONNX / TorchScript; FastAPI backend, Next.js frontend, and Docker containerization developed in IDE.

# Visual Recognition Pipeline

## Architecture Overview

```text
Video
  ↓
Video Preprocessing (OpenCV, segment slicing)
  ↓
32 Sampled Frames
  ↓
Resized & Normalized: [B, 32, 3, 224, 224]
  ↓
Time-flattened: [B × 32, 3, 224, 224]
  ↓
ResNet50 Backbone (Pretrained on ImageNet, Frozen)
  ↓
2048-D Spatial Features per Frame: [B, 32, 2048]
  ↓
2-Layer LSTM (Hidden size: 256, Dropout: 0.3)
  ↓
Temporal Visual Representation (Last timestep / pooling): [B, 256]
  ↓
Linear Classifier Projection
  ↓
100-Class Logits: [B, 100]
```

## Smoke Test Verification

- **Batch Shape**: `[4, 32, 3, 224, 224]`
- **Output Shape**: `[4, 100]`
- **Initial Loss**: ~4.61 (consistent with $-\ln(1/100) \approx 4.605$)
- **Gradient State**: Verified that ResNet50 backbone parameters remain frozen (`requires_grad = False`) while LSTM and classification head receive non-zero gradients and update via AdamW.

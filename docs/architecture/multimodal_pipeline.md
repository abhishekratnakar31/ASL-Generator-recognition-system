# Multimodal Vision-Language Pipeline

## Conceptual Overview

The planned multimodal architecture aligns dynamic sign video representations with semantic text embeddings:

```text
Video Stream                       ASL Gloss
     ↓                                 ↓
Visual Pipeline (ResNet50 + LSTM)  Language Encoder (Transformer/Embedding)
     ↓                                 ↓
Visual Projection Head             Language Projection Head
     ↓                                 ↓
256-D Visual Embedding             256-D Language Embedding
     │                                 │
     └─────────────► ◄─────────────────┘
                Alignment
          (Contrastive / Cosine Similarity)
```

## Alignment Objective

- **Shared Space Dimension**: 256
- **Normalization**: L2 normalized embeddings
- **Objective**: Contrastive loss (InfoNCE / NT-Xent) pulling paired sign gestures and gloss tokens together while pushing non-matching pairs apart.

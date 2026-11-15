"""End-to-end visual baseline classifier supporting raw videos or cached features."""

import torch
import torch.nn as nn

from src.models.visual.lstm import TemporalLSTM, TemporalTransformerEncoder
from src.models.visual.resnet import ResNet50Backbone


class VisualClassifier(nn.Module):
    """Visual-only ASL recognition model (ResNet50 + Temporal Encoder + Head)."""

    def __init__(
        self,
        num_classes: int,
        *,
        feature_dim: int = 2048,
        hidden_dim: int = 256,
        num_layers: int = 2,
        dropout: float = 0.3,
        temporal_type: str = "lstm",
        bidirectional: bool = False,
        pooling: str = "last",
        use_backbone: bool = False,
        pretrained_backbone: bool = True,
        frozen_backbone: bool = True,
        unfreeze_last_blocks: int = 0,
        frame_chunk_size: int = 32,
        num_heads: int = 4,
    ) -> None:
        super().__init__()
        self.use_backbone = bool(use_backbone)
        self.backbone = (
            ResNet50Backbone(
                pretrained=pretrained_backbone,
                frozen=frozen_backbone,
                unfreeze_last_blocks=unfreeze_last_blocks,
                frame_chunk_size=frame_chunk_size,
            )
            if self.use_backbone
            else None
        )

        if temporal_type == "lstm":
            self.temporal = TemporalLSTM(
                input_dim=feature_dim,
                hidden_dim=hidden_dim,
                num_layers=num_layers,
                dropout=dropout,
                bidirectional=bidirectional,
                pooling=pooling,
            )
        elif temporal_type == "transformer":
            self.temporal = TemporalTransformerEncoder(
                input_dim=feature_dim,
                embed_dim=hidden_dim,
                num_layers=num_layers,
                num_heads=num_heads,
                dim_feedforward=hidden_dim * 2,
                dropout=dropout,
                pooling="mean" if pooling == "last" else pooling,
            )
        else:
            raise ValueError(
                f"Unknown temporal_type='{temporal_type}'; use 'lstm' or 'transformer'."
            )

        self.representation_dim = self.temporal.output_dim
        self.head_dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(self.representation_dim, num_classes)

    def encode_visual(self, inputs: torch.Tensor) -> torch.Tensor:
        """Encode either (B, T, 3, H, W) video frames or (B, T, 2048) features."""
        if inputs.ndim == 5:
            if self.backbone is None:
                raise ValueError(
                    "Received 5D video tensor but model was constructed with use_backbone=False."
                )
            seq_features = self.backbone(inputs)
        elif inputs.ndim == 3:
            seq_features = inputs
        else:
            raise ValueError(
                f"Expected 5D video or 3D feature tensor, got shape {tuple(inputs.shape)}."
            )
        return self.temporal(seq_features)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        repr_vec = self.encode_visual(inputs)
        return self.classifier(self.head_dropout(repr_vec))

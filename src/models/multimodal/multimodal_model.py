"""Multimodal vision-language alignment and fusion architecture for ASL."""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.models.language.language_encoder import TextProjectionHead
from src.models.visual.lstm import TemporalLSTM, TemporalTransformerEncoder
from src.models.visual.resnet import ResNet50Backbone


class VisualProjectionHead(nn.Module):
    """Projects temporal visual features into the L2-normalized shared embedding space."""

    def __init__(
        self,
        input_dim: int = 256,
        embed_dim: int = 256,
        *,
        hidden_dim: int | None = None,
        dropout: float = 0.2,
        normalize: bool = True,
    ) -> None:
        super().__init__()
        self.normalize = bool(normalize)
        if hidden_dim is not None and hidden_dim > 0:
            self.net = nn.Sequential(
                nn.Dropout(dropout),
                nn.Linear(input_dim, hidden_dim),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, embed_dim),
                nn.LayerNorm(embed_dim),
            )
        else:
            self.net = nn.Sequential(
                nn.Dropout(dropout),
                nn.Linear(input_dim, embed_dim),
                nn.LayerNorm(embed_dim),
            )

    def forward(self, visual_features: torch.Tensor) -> torch.Tensor:
        projected = self.net(visual_features)
        return F.normalize(projected, p=2, dim=-1) if self.normalize else projected


class MultimodalASLModel(nn.Module):
    """Dual-branch Vision-Language model supporting alignment, fusion, and retrieval."""

    def __init__(
        self,
        num_classes: int,
        *,
        visual_feature_dim: int = 2048,
        temporal_hidden_dim: int = 256,
        temporal_layers: int = 2,
        temporal_type: str = "lstm",
        bidirectional: bool = False,
        pooling: str = "attention",
        text_dim: int = 384,
        embed_dim: int = 256,
        dropout: float = 0.3,
        fusion_mode: str = "hybrid",
        fusion_alpha: float = 0.5,
        init_temperature: float = 0.1,
        use_backbone: bool = False,
        pretrained_backbone: bool = True,
        frozen_backbone: bool = True,
        unfreeze_last_blocks: int = 0,
    ) -> None:
        super().__init__()
        if fusion_mode not in {"contrastive", "hybrid", "gated_fusion"}:
            raise ValueError(
                f"Unsupported fusion_mode='{fusion_mode}'; "
                "use 'contrastive', 'hybrid', or 'gated_fusion'."
            )

        self.num_classes = int(num_classes)
        self.embed_dim = int(embed_dim)
        self.fusion_mode = fusion_mode
        self.fusion_alpha = float(fusion_alpha)

        self.backbone = (
            ResNet50Backbone(
                pretrained=pretrained_backbone,
                frozen=frozen_backbone,
                unfreeze_last_blocks=unfreeze_last_blocks,
            )
            if use_backbone
            else None
        )

        if temporal_type == "lstm":
            self.temporal_encoder = TemporalLSTM(
                input_dim=visual_feature_dim,
                hidden_dim=temporal_hidden_dim,
                num_layers=temporal_layers,
                dropout=dropout,
                bidirectional=bidirectional,
                pooling=pooling,
            )
        elif temporal_type == "transformer":
            self.temporal_encoder = TemporalTransformerEncoder(
                input_dim=visual_feature_dim,
                embed_dim=temporal_hidden_dim,
                num_layers=temporal_layers,
                dropout=dropout,
                pooling="mean" if pooling == "last" else pooling,
            )
        else:
            raise ValueError(f"Unsupported temporal_type='{temporal_type}'")

        repr_dim = self.temporal_encoder.output_dim
        self.visual_projection = VisualProjectionHead(
            input_dim=repr_dim,
            embed_dim=embed_dim,
            dropout=dropout,
            normalize=True,
        )
        self.text_projection = TextProjectionHead(
            input_dim=text_dim,
            embed_dim=embed_dim,
            dropout=0.1,
            normalize=True,
        )
        self.logit_scale = nn.Parameter(
            torch.tensor(math.log(1.0 / max(init_temperature, 1e-3)), dtype=torch.float32)
        )

        # Closed-vocabulary head for hybrid or cross-modal gated fusion
        self.visual_classifier = nn.Linear(repr_dim, num_classes)
        if fusion_mode == "gated_fusion":
            self.fusion_gate = nn.Sequential(
                nn.Linear(embed_dim * 2, embed_dim),
                nn.Sigmoid(),
            )
            self.fused_classifier = nn.Sequential(
                nn.Dropout(dropout),
                nn.Linear(embed_dim, num_classes),
            )
        else:
            self.fusion_gate = None
            self.fused_classifier = None

    def encode_visual_features(self, visual_inputs: torch.Tensor) -> torch.Tensor:
        """Return pre-projection temporal visual representation (B, repr_dim)."""
        if visual_inputs.ndim == 5:
            if self.backbone is None:
                raise ValueError("5D video input passed when use_backbone=False.")
            seq = self.backbone(visual_inputs)
        elif visual_inputs.ndim == 3:
            seq = visual_inputs
        else:
            raise ValueError(f"Expected 5D or 3D visual input, got {tuple(visual_inputs.shape)}")
        return self.temporal_encoder(seq)

    def encode_video(self, visual_inputs: torch.Tensor) -> torch.Tensor:
        """Return L2-normalized visual embedding (B, embed_dim)."""
        repr_vec = self.encode_visual_features(visual_inputs)
        return self.visual_projection(repr_vec)

    def encode_text(self, raw_text_embeddings: torch.Tensor) -> torch.Tensor:
        """Return L2-normalized language embedding (N, embed_dim)."""
        return self.text_projection(raw_text_embeddings)

    def current_scale(self) -> torch.Tensor:
        return self.logit_scale.exp().clamp(max=100.0)

    def forward(
        self,
        visual_inputs: torch.Tensor,
        vocab_text_embeddings: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        """Compute alignment logits, fused classification logits, and normalized embeddings."""
        repr_vec = self.encode_visual_features(visual_inputs)
        visual_emb = self.visual_projection(repr_vec)
        text_emb = self.text_projection(vocab_text_embeddings)

        scale = self.current_scale()
        cosine_logits = scale * (visual_emb @ text_emb.T)

        if self.fusion_mode == "contrastive":
            cls_logits = cosine_logits
        elif self.fusion_mode == "hybrid":
            vis_logits = self.visual_classifier(repr_vec)
            cls_logits = self.fusion_alpha * cosine_logits + (1.0 - self.fusion_alpha) * vis_logits
        else:
            # Cross-modal gated fusion: attend over language prototypes using visual query
            assert self.fusion_gate is not None and self.fused_classifier is not None
            attn_weights = torch.softmax(cosine_logits, dim=-1)
            lang_context = attn_weights @ text_emb
            gate = self.fusion_gate(torch.cat([visual_emb, lang_context], dim=-1))
            fused = gate * visual_emb + (1.0 - gate) * lang_context
            cls_logits = self.fused_classifier(fused) + self.fusion_alpha * cosine_logits

        return {
            "logits": cls_logits,
            "cosine_logits": cosine_logits,
            "visual_embeddings": visual_emb,
            "text_embeddings": text_emb,
            "logit_scale": scale,
        }

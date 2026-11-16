"""Pose-and-Visual Multimodal Vision-Language architecture for ASL recognition."""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.models.language.language_encoder import TextProjectionHead
from src.models.visual.lstm import TemporalAttentionPool, TemporalLSTM, TemporalTransformerEncoder


class CosineClassifier(nn.Module):
    """Cosine similarity classifier head with learned temperature scale."""

    def __init__(
        self,
        in_features: int,
        num_classes: int,
        init_scale: float = 16.0,
    ) -> None:
        super().__init__()
        self.in_features = int(in_features)
        self.num_classes = int(num_classes)
        self.weight = nn.Parameter(torch.empty(num_classes, in_features))
        nn.init.xavier_uniform_(self.weight)
        self.logit_scale = nn.Parameter(
            torch.tensor(math.log(max(init_scale, 1.0)), dtype=torch.float32)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_norm = F.normalize(x, p=2, dim=-1)
        w_norm = F.normalize(self.weight, p=2, dim=-1)
        scale = self.logit_scale.exp().clamp(min=1.0, max=50.0)
        return scale * F.linear(x_norm, w_norm)


class TemporalBiLSTMEncoder(nn.Module):
    """Bidirectional LSTM with input LayerNorm, sequence LayerNorm, and attention pooling."""

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 96,
        num_layers: int = 2,
        dropout: float = 0.35,
    ) -> None:
        super().__init__()
        self.input_proj = nn.Sequential(
            nn.LayerNorm(input_dim),
            nn.Dropout(dropout),
        )
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.norm = nn.LayerNorm(hidden_dim * 2)
        self.attn_pool = TemporalAttentionPool(hidden_dim * 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.input_proj(x)
        seq, _ = self.lstm(h)
        return self.attn_pool(self.norm(seq))


class PoseVisualMultimodalModel(nn.Module):
    """Flexible dual/triple-stream ASL model supporting pose, visual, and text alignment."""

    def __init__(
        self,
        num_classes: int,
        *,
        pose_dim: int = 349,
        visual_dim: int = 2048,
        text_dim: int = 384,
        hidden_dim: int = 192,
        embed_dim: int = 192,
        num_layers: int = 2,
        num_heads: int = 4,
        dropout: float = 0.35,
        mode: str = "pose_visual_language",
        fusion_alpha: float = 0.5,
        init_temperature: float = 0.1,
        backbone: str = "transformer",
        use_cosine_classifier: bool = False,
    ) -> None:
        super().__init__()
        if mode not in {"pose_only", "pose_language", "pose_visual_language"}:
            raise ValueError(
                f"Unsupported mode='{mode}'; use pose_only, pose_language, or pose_visual_language."
            )
        if backbone not in {"transformer", "bilstm"}:
            raise ValueError(
                f"Unsupported backbone='{backbone}'; use transformer or bilstm."
            )
        self.num_classes = int(num_classes)
        self.mode = mode
        self.fusion_alpha = float(fusion_alpha)
        self.backbone = backbone
        self.use_cosine_classifier = bool(use_cosine_classifier)

        if backbone == "bilstm":
            lstm_hidden = max(16, hidden_dim // 2)
            self.pose_encoder: nn.Module = TemporalBiLSTMEncoder(
                input_dim=pose_dim,
                hidden_dim=lstm_hidden,
                num_layers=num_layers,
                dropout=dropout,
            )
        else:
            self.pose_encoder = TemporalTransformerEncoder(
                input_dim=pose_dim,
                embed_dim=hidden_dim,
                num_layers=num_layers,
                num_heads=num_heads,
                dim_feedforward=hidden_dim * 2,
                dropout=dropout,
                pooling="attention",
            )

        if mode == "pose_visual_language":
            self.visual_encoder: TemporalTransformerEncoder | None = TemporalTransformerEncoder(
                input_dim=visual_dim,
                embed_dim=hidden_dim,
                num_layers=num_layers,
                num_heads=num_heads,
                dim_feedforward=hidden_dim * 2,
                dropout=dropout,
                pooling="attention",
            )
            self.stream_gate: nn.Module | None = nn.Sequential(
                nn.Linear(hidden_dim * 2, hidden_dim),
                nn.Sigmoid(),
            )
        else:
            self.visual_encoder = None
            self.stream_gate = None

        self.motion_projection = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, embed_dim),
            nn.LayerNorm(embed_dim),
        )
        if use_cosine_classifier:
            self.classifier_head: nn.Module = nn.Sequential(
                nn.Dropout(dropout),
                CosineClassifier(hidden_dim, num_classes),
            )
        else:
            self.classifier_head = nn.Sequential(
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, num_classes),
            )

        if mode in {"pose_language", "pose_visual_language"}:
            self.text_projection: TextProjectionHead | None = TextProjectionHead(
                input_dim=text_dim,
                embed_dim=embed_dim,
                dropout=0.15,
                normalize=True,
            )
            self.logit_scale = nn.Parameter(
                torch.tensor(math.log(1.0 / max(init_temperature, 1e-3)), dtype=torch.float32)
            )
            self.lang_fusion_gate: nn.Module | None = nn.Sequential(
                nn.Linear(embed_dim * 2, embed_dim),
                nn.Sigmoid(),
            )
            if use_cosine_classifier:
                self.fused_head: nn.Module | None = nn.Sequential(
                    nn.Dropout(dropout),
                    CosineClassifier(embed_dim, num_classes),
                )
            else:
                self.fused_head = nn.Sequential(
                    nn.Dropout(dropout),
                    nn.Linear(embed_dim, num_classes),
                )
        else:
            self.text_projection = None
            self.register_parameter("logit_scale", None)
            self.lang_fusion_gate = None
            self.fused_head = None

    def encode_kinematics_visual(
        self,
        pose_inputs: torch.Tensor,
        visual_inputs: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Encode pose sequences (and optional visual sequences) into (B, hidden_dim)."""
        pose_repr = self.pose_encoder(pose_inputs)
        if self.mode == "pose_visual_language":
            if visual_inputs is None or self.visual_encoder is None or self.stream_gate is None:
                raise ValueError("visual_inputs required in pose_visual_language mode.")
            vis_repr = self.visual_encoder(visual_inputs)
            gate = self.stream_gate(torch.cat([pose_repr, vis_repr], dim=-1))
            return gate * pose_repr + (1.0 - gate) * vis_repr
        return pose_repr

    def forward(
        self,
        pose_inputs: torch.Tensor,
        *,
        visual_inputs: torch.Tensor | None = None,
        vocab_text_embeddings: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        """Compute classification logits and aligned multimodal embeddings."""
        repr_vec = self.encode_kinematics_visual(pose_inputs, visual_inputs)
        if self.mode == "pose_only":
            logits = self.classifier_head(repr_vec)
            return {"logits": logits, "representation": repr_vec}

        if (
            vocab_text_embeddings is None
            or self.text_projection is None
            or self.logit_scale is None
            or self.lang_fusion_gate is None
            or self.fused_head is None
        ):
            raise ValueError("vocab_text_embeddings required for multimodal pose modes.")

        vis_emb = F.normalize(self.motion_projection(repr_vec), p=2, dim=-1)
        txt_emb = self.text_projection(vocab_text_embeddings)
        scale = self.logit_scale.exp().clamp(max=30.0)
        cosine_logits = scale * (vis_emb @ txt_emb.T)

        attn = torch.softmax(cosine_logits, dim=-1)
        lang_ctx = attn @ txt_emb
        gate = self.lang_fusion_gate(torch.cat([vis_emb, lang_ctx], dim=-1))
        fused = gate * vis_emb + (1.0 - gate) * lang_ctx
        cls_logits = self.fused_head(fused) + self.fusion_alpha * cosine_logits

        return {
            "logits": cls_logits,
            "cosine_logits": cosine_logits,
            "visual_embeddings": vis_emb,
            "text_embeddings": txt_emb,
            "logit_scale": scale,
        }


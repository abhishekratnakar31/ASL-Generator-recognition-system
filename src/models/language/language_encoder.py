"""Language encoder and projection modules for ASL gloss representations."""

from typing import Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F


def build_gloss_texts(
    glosses: Sequence[str],
    *,
    template: str = "{gloss}",
) -> list[str]:
    """Format ASL gloss strings using a consistent documented text template."""
    formatted = []
    for gloss in glosses:
        cleaned = str(gloss).strip()
        if not cleaned:
            raise ValueError("Encountered empty gloss string.")
        formatted.append(template.format(gloss=cleaned))
    return formatted


class TextProjectionHead(nn.Module):
    """Projects frozen or pooled text embeddings into the shared multimodal space."""

    def __init__(
        self,
        input_dim: int = 384,
        embed_dim: int = 256,
        *,
        hidden_dim: int | None = None,
        dropout: float = 0.1,
        normalize: bool = True,
    ) -> None:
        super().__init__()
        self.normalize = bool(normalize)
        if hidden_dim is not None and hidden_dim > 0:
            self.net = nn.Sequential(
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

    def forward(self, text_embeddings: torch.Tensor) -> torch.Tensor:
        if text_embeddings.ndim != 2:
            raise ValueError(
                f"Expected 2D text embeddings (N, D), got {tuple(text_embeddings.shape)}"
            )
        projected = self.net(text_embeddings)
        return F.normalize(projected, p=2, dim=-1) if self.normalize else projected


class PretrainedTextEncoder(nn.Module):
    """HuggingFace Transformer gloss encoder with attention-masked mean pooling."""

    def __init__(
        self,
        model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
        *,
        freeze_backbone: bool = True,
        max_length: int = 32,
        normalize: bool = True,
    ) -> None:
        super().__init__()
        from transformers import AutoModel, AutoTokenizer

        self.model_name = model_name
        self.max_length = int(max_length)
        self.normalize = bool(normalize)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.backbone = AutoModel.from_pretrained(model_name)
        self.output_dim = int(self.backbone.config.hidden_size)

        if freeze_backbone:
            for param in self.backbone.parameters():
                param.requires_grad = False
            self.backbone.eval()

    @staticmethod
    def _mean_pool(
        last_hidden_state: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        mask_expanded = (
            attention_mask.unsqueeze(-1).expand(last_hidden_state.size()).float()
        )
        summed = (last_hidden_state * mask_expanded).sum(dim=1)
        counts = mask_expanded.sum(dim=1).clamp(min=1e-9)
        return summed / counts

    def encode_texts(
        self,
        texts: Sequence[str],
        *,
        batch_size: int = 64,
        device: torch.device | str | None = None,
    ) -> torch.Tensor:
        """Tokenize and encode a list of gloss strings into (N, D) embeddings."""
        if not texts:
            raise ValueError("Cannot encode empty text sequence.")
        target_device = (
            torch.device(device)
            if device is not None
            else next(self.backbone.parameters()).device
        )
        was_training = self.backbone.training
        self.backbone.eval()

        outputs = []
        with torch.no_grad():
            for start in range(0, len(texts), batch_size):
                batch_texts = list(texts[start : start + batch_size])
                encoded = self.tokenizer(
                    batch_texts,
                    padding=True,
                    truncation=True,
                    max_length=self.max_length,
                    return_tensors="pt",
                )
                encoded = {k: v.to(target_device) for k, v in encoded.items()}
                model_out = self.backbone(**encoded)
                pooled = self._mean_pool(
                    model_out.last_hidden_state, encoded["attention_mask"]
                )
                if self.normalize:
                    pooled = F.normalize(pooled, p=2, dim=-1)
                outputs.append(pooled.cpu())

        if was_training:
            self.backbone.train(True)
        return torch.cat(outputs, dim=0)

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        out = self.backbone(input_ids=input_ids, attention_mask=attention_mask)
        pooled = self._mean_pool(out.last_hidden_state, attention_mask)
        return F.normalize(pooled, p=2, dim=-1) if self.normalize else pooled

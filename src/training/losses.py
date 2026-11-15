"""Classification and multimodal contrastive loss functions."""

import torch
import torch.nn as nn
import torch.nn.functional as F


def build_classification_loss(
    *,
    class_weights: torch.Tensor | None = None,
    label_smoothing: float = 0.0,
) -> nn.CrossEntropyLoss:
    """Create CrossEntropyLoss with optional inverse-frequency weights and smoothing."""
    return nn.CrossEntropyLoss(
        weight=class_weights,
        label_smoothing=float(label_smoothing),
    )


def symmetric_infonce_loss(
    visual_embeddings: torch.Tensor,
    text_embeddings: torch.Tensor,
    logit_scale: torch.Tensor | float = 10.0,
    *,
    labels: torch.Tensor | None = None,
) -> torch.Tensor:
    """Compute symmetric InfoNCE contrastive loss.

    When `labels` is provided, positive pairs include any pair sharing the same class ID
    within the mini-batch (multi-positive soft target distribution) so same-class videos
    in the same batch are not falsely pushed apart as negatives.
    """
    if visual_embeddings.shape != text_embeddings.shape:
        raise ValueError(
            f"Shape mismatch: visual {tuple(visual_embeddings.shape)} vs "
            f"text {tuple(text_embeddings.shape)}"
        )

    scale = logit_scale if isinstance(logit_scale, torch.Tensor) else torch.tensor(logit_scale)
    logits = scale * (visual_embeddings @ text_embeddings.T)
    batch_size = logits.size(0)

    if labels is None:
        targets = torch.arange(batch_size, device=logits.device, dtype=torch.long)
        loss_v2t = F.cross_entropy(logits, targets)
        loss_t2v = F.cross_entropy(logits.T, targets)
        return 0.5 * (loss_v2t + loss_t2v)

    pos_mask = (labels.unsqueeze(0) == labels.unsqueeze(1)).to(logits.dtype)
    target_v2t = pos_mask / pos_mask.sum(dim=1, keepdim=True).clamp(min=1.0)
    target_t2v = pos_mask.T / pos_mask.T.sum(dim=1, keepdim=True).clamp(min=1.0)

    loss_v2t = -(target_v2t * F.log_softmax(logits, dim=-1)).sum(dim=-1).mean()
    loss_t2v = -(target_t2v * F.log_softmax(logits.T, dim=-1)).sum(dim=-1).mean()
    return 0.5 * (loss_v2t + loss_t2v)


def vocabulary_contrastive_loss(
    cosine_logits: torch.Tensor,
    labels: torch.Tensor,
    *,
    class_weights: torch.Tensor | None = None,
    label_smoothing: float = 0.0,
) -> torch.Tensor:
    """Contrast each video embedding against the full C-class text vocabulary."""
    return F.cross_entropy(
        cosine_logits,
        labels,
        weight=class_weights,
        label_smoothing=float(label_smoothing),
    )


class JointMultimodalLoss(nn.Module):
    """Weighted combination of vocabulary contrastive alignment and recognition loss."""

    class_weights: torch.Tensor | None

    def __init__(
        self,
        *,
        contrastive_weight: float = 0.5,
        classification_weight: float = 0.5,
        batch_infonce_weight: float = 0.25,
        class_weights: torch.Tensor | None = None,
        label_smoothing: float = 0.05,
    ) -> None:
        super().__init__()
        self.contrastive_weight = float(contrastive_weight)
        self.classification_weight = float(classification_weight)
        self.batch_infonce_weight = float(batch_infonce_weight)
        self.register_buffer("class_weights", class_weights)
        self.label_smoothing = float(label_smoothing)

    def forward(
        self,
        outputs: dict[str, torch.Tensor],
        labels: torch.Tensor,
    ) -> tuple[torch.Tensor, dict[str, float]]:
        cls_loss = F.cross_entropy(
            outputs["logits"],
            labels,
            weight=self.class_weights,
            label_smoothing=self.label_smoothing,
        )
        vocab_con_loss = vocabulary_contrastive_loss(
            outputs["cosine_logits"],
            labels,
            class_weights=self.class_weights,
            label_smoothing=self.label_smoothing,
        )
        batch_text = outputs["text_embeddings"][labels]
        batch_infonce = symmetric_infonce_loss(
            outputs["visual_embeddings"],
            batch_text,
            outputs["logit_scale"],
            labels=labels,
        )

        total = (
            self.classification_weight * cls_loss
            + self.contrastive_weight * vocab_con_loss
            + self.batch_infonce_weight * batch_infonce
        )
        breakdown = {
            "total_loss": float(total.detach()),
            "cls_loss": float(cls_loss.detach()),
            "vocab_contrastive_loss": float(vocab_con_loss.detach()),
            "batch_infonce_loss": float(batch_infonce.detach()),
        }
        return total, breakdown

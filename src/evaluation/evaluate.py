"""Checkpoint evaluation, latency benchmarking, and retrieval diagnostics."""

import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn

from src.evaluation.metrics import (
    compute_classification_metrics,
    compute_confusion_matrix,
    compute_per_class_metrics,
    compute_retrieval_metrics,
)


def count_parameters(model: nn.Module) -> dict[str, int]:
    """Count total and trainable parameters in a PyTorch module."""
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return {"total_params": int(total), "trainable_params": int(trainable)}


def evaluate_checkpoint(
    model: nn.Module,
    loader: Any,
    device: torch.device,
    *,
    vocab_text_embeddings: torch.Tensor | None = None,
    class_names: dict[int, str] | list[str] | None = None,
    checkpoint_path: Path | None = None,
) -> dict[str, Any]:
    """Evaluate classification, retrieval (if multimodal), and inference latency."""
    model.eval()
    all_logits = []
    all_cosine = []
    all_visual_emb = []
    all_labels = []

    if device.type == "cuda":
        torch.cuda.synchronize()
    t0 = time.perf_counter()

    with torch.inference_mode():
        for inputs, labels in loader:
            inputs = inputs.to(device, non_blocking=True)
            if vocab_text_embeddings is None:
                logits = model(inputs)
            else:
                out = model(inputs, vocab_text_embeddings)
                logits = out["logits"]
                all_cosine.append(out["cosine_logits"].float().cpu().numpy())
                all_visual_emb.append(out["visual_embeddings"].float().cpu().numpy())

            all_logits.append(logits.float().cpu().numpy())
            all_labels.append(labels.cpu().numpy())

    if device.type == "cuda":
        torch.cuda.synchronize()
    elapsed_sec = time.perf_counter() - t0

    scores = np.concatenate(all_logits, axis=0)
    y_true = np.concatenate(all_labels, axis=0)
    y_pred = np.argmax(scores, axis=1)

    metrics = compute_classification_metrics(y_true, scores, top_k=5)
    metrics["inference_total_sec"] = round(elapsed_sec, 4)
    metrics["inference_ms_per_sample"] = round(elapsed_sec * 1000.0 / len(y_true), 3)
    metrics.update(count_parameters(model))
    if checkpoint_path is not None and Path(checkpoint_path).is_file():
        metrics["checkpoint_size_mb"] = round(
            Path(checkpoint_path).stat().st_size / (1024**2), 3
        )

    if all_cosine:
        cos_scores = np.concatenate(all_cosine, axis=0)
        retrieval = compute_retrieval_metrics(cos_scores, y_true, ks=(1, 5, 10))
        metrics["retrieval"] = retrieval

    num_classes = scores.shape[1]
    cm = compute_confusion_matrix(y_true, y_pred, num_classes=num_classes)
    per_class = compute_per_class_metrics(
        y_true, y_pred, num_classes=num_classes, class_names=class_names
    )

    return {
        "metrics": metrics,
        "confusion_matrix": cm,
        "per_class": per_class,
        "y_true": y_true,
        "y_pred": y_pred,
        "scores": scores,
    }

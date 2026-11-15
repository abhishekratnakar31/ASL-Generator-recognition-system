"""Reusable PyTorch training loop with checkpointing, AMP, and early stopping."""

import json
import random
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from src.evaluation.metrics import compute_classification_metrics


def set_seed(seed: int = 42) -> None:
    """Set reproducible random seeds across Python, NumPy, and PyTorch."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_device(prefer: str | None = None) -> torch.device:
    """Detect the best available accelerator (CUDA -> Apple MPS -> CPU)."""
    if prefer:
        return torch.device(prefer)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


@dataclass
class TrainerConfig:
    """Serializable configuration for visual or multimodal training."""

    experiment_id: str
    epochs: int = 40
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    grad_clip_norm: float = 1.0
    patience: int = 10
    use_amp: bool = True
    scheduler_type: str = "cosine"
    seed: int = 42


def _create_tb_writer(log_dir: Path | None):
    if log_dir is None:
        return None
    try:
        from torch.utils.tensorboard import SummaryWriter

        return SummaryWriter(log_dir=str(log_dir))
    except Exception:
        return None


def _forward_model(
    model: nn.Module,
    inputs: torch.Tensor,
    vocab_text_embeddings: torch.Tensor | None,
) -> tuple[torch.Tensor, dict[str, torch.Tensor] | None]:
    if vocab_text_embeddings is None:
        return model(inputs), None
    out = model(inputs, vocab_text_embeddings)
    return out["logits"], out


def run_eval_epoch(
    model: nn.Module,
    loader: Any,
    criterion: Any,
    device: torch.device,
    *,
    vocab_text_embeddings: torch.Tensor | None = None,
    use_amp: bool = False,
) -> dict[str, Any]:
    """Run deterministic evaluation over a DataLoader and return metrics + arrays."""
    model.eval()
    total_loss = 0.0
    total_samples = 0
    all_logits = []
    all_labels = []

    amp_enabled = bool(use_amp and device.type == "cuda")
    with torch.inference_mode():
        for inputs, labels in loader:
            inputs = inputs.to(device, non_blocking=True)
            labels = labels.to(device, dtype=torch.long, non_blocking=True)
            with torch.amp.autocast("cuda", enabled=amp_enabled):
                logits, out_dict = _forward_model(model, inputs, vocab_text_embeddings)
                if out_dict is not None:
                    loss, _ = criterion(out_dict, labels)
                else:
                    loss = criterion(logits, labels)

            bs = labels.size(0)
            total_samples += bs
            total_loss += float(loss.detach()) * bs
            all_logits.append(logits.float().cpu().numpy())
            all_labels.append(labels.cpu().numpy())

    if total_samples == 0:
        raise ValueError("Evaluation DataLoader yielded zero samples.")

    scores = np.concatenate(all_logits, axis=0)
    y_true = np.concatenate(all_labels, axis=0)
    metrics = compute_classification_metrics(y_true, scores, top_k=5)
    metrics["loss"] = total_loss / total_samples
    return {"metrics": metrics, "scores": scores, "labels": y_true}


def train_model(
    model: nn.Module,
    train_loader: Any,
    val_loader: Any,
    criterion: Any,
    optimizer: torch.optim.Optimizer,
    output_dir: Path,
    config: TrainerConfig,
    *,
    vocab_text_embeddings: torch.Tensor | None = None,
    scheduler: Any = None,
    class_to_id: dict[str, int] | None = None,
    id_to_class: dict[int, str] | None = None,
    resume: bool = True,
    extra_metadata: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Train model with AMP, gradient clipping, early stopping, and resume support."""
    device = next(model.parameters()).device
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    best_path = output_dir / "best_model.pt"
    latest_path = output_dir / "latest_checkpoint.pt"
    history_path = output_dir / "training_history.csv"
    config_path = output_dir / "config.json"

    amp_enabled = bool(config.use_amp and device.type == "cuda")
    scaler = torch.amp.GradScaler("cuda", enabled=amp_enabled)
    tb_writer = _create_tb_writer(output_dir / "tb_logs")

    start_epoch = 0
    best_val_loss = float("inf")
    best_epoch = 0
    epochs_no_improve = 0
    history: list[dict[str, Any]] = []

    if resume and latest_path.is_file():
        ckpt = torch.load(latest_path, map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model_state_dict"])
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        if scheduler is not None and ckpt.get("scheduler_state_dict") is not None:
            scheduler.load_state_dict(ckpt["scheduler_state_dict"])
        if ckpt.get("scaler_state_dict") is not None and amp_enabled:
            scaler.load_state_dict(ckpt["scaler_state_dict"])
        start_epoch = int(ckpt["epoch"])
        best_val_loss = float(ckpt.get("best_val_loss", float("inf")))
        best_epoch = int(ckpt.get("best_epoch", 0))
        epochs_no_improve = int(ckpt.get("epochs_no_improve", 0))
        history = list(ckpt.get("history", []))

    config_payload = {"trainer": asdict(config), **(extra_metadata or {})}
    config_path.write_text(json.dumps(config_payload, indent=2), encoding="utf-8")

    train_start_wall = time.time()
    for epoch in range(start_epoch, config.epochs):
        ep_t0 = time.time()
        model.train()
        running_loss = 0.0
        total_train = 0
        train_top1_hits = 0

        for inputs, labels in train_loader:
            inputs = inputs.to(device, non_blocking=True)
            labels = labels.to(device, dtype=torch.long, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)

            with torch.amp.autocast("cuda", enabled=amp_enabled):
                logits, out_dict = _forward_model(model, inputs, vocab_text_embeddings)
                if out_dict is not None:
                    loss, _ = criterion(out_dict, labels)
                else:
                    loss = criterion(logits, labels)

            scaler.scale(loss).backward()
            if config.grad_clip_norm > 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(
                    model.parameters(), config.grad_clip_norm
                )
            scaler.step(optimizer)
            scaler.update()

            bs = labels.size(0)
            total_train += bs
            running_loss += float(loss.detach()) * bs
            train_top1_hits += int((logits.argmax(dim=1) == labels).sum().item())

        train_loss = running_loss / max(1, total_train)
        train_top1 = train_top1_hits / max(1, total_train)

        val_out = run_eval_epoch(
            model,
            val_loader,
            criterion,
            device,
            vocab_text_embeddings=vocab_text_embeddings,
            use_amp=amp_enabled,
        )
        val_m = val_out["metrics"]

        if scheduler is not None:
            if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                scheduler.step(val_m["loss"])
            else:
                scheduler.step()

        improved = val_m["loss"] < best_val_loss
        if improved:
            best_val_loss = float(val_m["loss"])
            best_epoch = epoch + 1
            epochs_no_improve = 0
            torch.save(
                {
                    "epoch": best_epoch,
                    "model_state_dict": model.state_dict(),
                    "val_metrics": val_m,
                    "config": config_payload,
                    "class_to_id": class_to_id,
                    "id_to_class": id_to_class,
                },
                best_path,
            )
        else:
            epochs_no_improve += 1

        row = {
            "epoch": epoch + 1,
            "train_loss": train_loss,
            "train_top1": train_top1,
            "val_loss": val_m["loss"],
            "val_top1": val_m["top1_accuracy"],
            "val_top5": val_m.get("top5_accuracy", val_m["top1_accuracy"]),
            "val_macro_f1": val_m["macro_f1"],
            "lr": float(optimizer.param_groups[0]["lr"]),
            "epoch_sec": round(time.time() - ep_t0, 2),
        }
        history.append(row)
        pd.DataFrame(history).to_csv(history_path, index=False)

        if tb_writer is not None:
            tb_writer.add_scalar("loss/train", train_loss, epoch + 1)
            tb_writer.add_scalar("loss/val", val_m["loss"], epoch + 1)
            tb_writer.add_scalar("acc/val_top1", val_m["top1_accuracy"], epoch + 1)
            tb_writer.add_scalar("acc/val_macro_f1", val_m["macro_f1"], epoch + 1)

        torch.save(
            {
                "epoch": epoch + 1,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "scheduler_state_dict": (
                    scheduler.state_dict() if scheduler is not None else None
                ),
                "scaler_state_dict": scaler.state_dict() if amp_enabled else None,
                "best_val_loss": best_val_loss,
                "best_epoch": best_epoch,
                "epochs_no_improve": epochs_no_improve,
                "history": history,
            },
            latest_path,
        )

        if epochs_no_improve >= config.patience:
            break

    if tb_writer is not None:
        tb_writer.close()

    total_duration = round(time.time() - train_start_wall, 2)
    return {
        "best_epoch": best_epoch,
        "best_val_loss": best_val_loss,
        "history": history,
        "best_checkpoint": str(best_path),
        "latest_checkpoint": str(latest_path),
        "training_duration_sec": total_duration,
    }

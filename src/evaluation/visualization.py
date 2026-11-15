"""Plotting utilities for training curves, confusion matrices, and embeddings."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def plot_training_history(
    history_df: pd.DataFrame,
    save_path: Path | None = None,
    *,
    title_prefix: str = "",
) -> None:
    """Plot training/validation loss, Top-1 accuracy, and Macro-F1 curves."""
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    axes[0].plot(history_df["epoch"], history_df["train_loss"], label="Train")
    axes[0].plot(history_df["epoch"], history_df["val_loss"], label="Validation")
    axes[0].set_title(f"{title_prefix}Loss".strip())
    axes[0].set_xlabel("Epoch")
    axes[0].legend()
    axes[0].grid(True)

    axes[1].plot(history_df["epoch"], history_df["train_top1"], label="Train Top-1")
    axes[1].plot(history_df["epoch"], history_df["val_top1"], label="Val Top-1")
    if "val_top5" in history_df.columns:
        axes[1].plot(history_df["epoch"], history_df["val_top5"], label="Val Top-5")
    axes[1].set_title(f"{title_prefix}Accuracy".strip())
    axes[1].set_xlabel("Epoch")
    axes[1].legend()
    axes[1].grid(True)

    if "val_macro_f1" in history_df.columns:
        axes[2].plot(
            history_df["epoch"], history_df["val_macro_f1"], label="Val Macro-F1"
        )
        axes[2].set_title(f"{title_prefix}Validation Macro-F1".strip())
        axes[2].set_xlabel("Epoch")
        axes[2].legend()
        axes[2].grid(True)

    plt.tight_layout()
    if save_path is not None:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_confusion_matrix(
    cm: np.ndarray,
    class_names: list[str] | None = None,
    save_path: Path | None = None,
    *,
    title: str = "Confusion Matrix",
) -> None:
    """Render and optionally save a confusion matrix heatmap."""
    size = (12, 10) if cm.shape[0] <= 30 else (16, 14)
    fig, ax = plt.subplots(figsize=size)
    im = ax.imshow(cm, interpolation="nearest", cmap="Blues")
    fig.colorbar(im, ax=ax)
    ax.set_title(title)
    ax.set_xlabel("Predicted Class")
    ax.set_ylabel("True Class")

    if class_names is not None and len(class_names) <= 30:
        ticks = np.arange(len(class_names))
        ax.set_xticks(ticks)
        ax.set_xticklabels(class_names, rotation=45, ha="right", fontsize=8)
        ax.set_yticks(ticks)
        ax.set_yticklabels(class_names, fontsize=8)

    plt.tight_layout()
    if save_path is not None:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)

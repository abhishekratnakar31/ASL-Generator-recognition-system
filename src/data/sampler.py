"""Class-weighted loss weights and class-balanced sampling utilities."""

import numpy as np


def compute_inverse_frequency_weights(
    labels: np.ndarray,
    num_classes: int,
    *,
    power: float = 1.0,
    normalize_mean: bool = True,
) -> np.ndarray:
    """Compute per-class inverse-frequency weights, assigning 0 to unseen classes."""
    lbl = np.asarray(labels, dtype=np.int64)
    if num_classes <= 0:
        raise ValueError(f"num_classes must be positive, got {num_classes}")
    if power < 0.0:
        raise ValueError(f"power must be non-negative, got {power}")

    counts = np.bincount(lbl, minlength=num_classes).astype(np.float64)
    weights = np.zeros(num_classes, dtype=np.float32)
    present = counts > 0
    if not np.any(present):
        raise ValueError("No valid training labels found.")

    raw = np.power(counts[present], -float(power))
    if normalize_mean:
        raw = raw / raw.mean()
    weights[present] = raw.astype(np.float32)
    return weights


def compute_sample_weights(
    labels: np.ndarray,
    *,
    power: float = 1.0,
) -> np.ndarray:
    """Compute per-sample weights for class-balanced mini-batch sampling."""
    lbl = np.asarray(labels, dtype=np.int64)
    if lbl.ndim != 1 or len(lbl) == 0:
        raise ValueError("labels must be a non-empty 1D array.")
    num_classes = int(lbl.max()) + 1
    class_weights = compute_inverse_frequency_weights(
        lbl, num_classes=num_classes, power=power, normalize_mean=False
    )
    return class_weights[lbl].astype(np.float64)


def build_weighted_sampler(
    labels: np.ndarray,
    *,
    power: float = 1.0,
    num_samples: int | None = None,
    replacement: bool = True,
):
    """Create a PyTorch WeightedRandomSampler for class-balanced training."""
    from torch.utils.data import WeightedRandomSampler

    sample_weights = compute_sample_weights(labels, power=power)
    n_samples = len(sample_weights) if num_samples is None else int(num_samples)
    return WeightedRandomSampler(
        weights=sample_weights.tolist(),
        num_samples=n_samples,
        replacement=replacement,
    )

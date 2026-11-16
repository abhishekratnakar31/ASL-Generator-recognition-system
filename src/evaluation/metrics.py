"""Classification, retrieval, baseline, and confusion matrix metrics."""

from typing import Any

import numpy as np


def compute_confusion_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    num_classes: int,
) -> np.ndarray:
    """Compute a (num_classes, num_classes) confusion matrix using pure NumPy."""
    y_t = np.asarray(y_true, dtype=np.int64)
    y_p = np.asarray(y_pred, dtype=np.int64)
    if y_t.shape != y_p.shape:
        raise ValueError(f"Shape mismatch: y_true {y_t.shape} vs y_pred {y_p.shape}")
    if num_classes <= 0:
        raise ValueError(f"num_classes must be positive, got {num_classes}")

    valid = (y_t >= 0) & (y_t < num_classes) & (y_p >= 0) & (y_p < num_classes)
    if not np.all(valid):
        raise ValueError("Labels/predictions contain out-of-range class indices.")

    flat_idx = y_t * num_classes + y_p
    counts = np.bincount(flat_idx, minlength=num_classes * num_classes)
    return counts.reshape(num_classes, num_classes)


def compute_per_class_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    *,
    num_classes: int | None = None,
    class_names: dict[int, str] | list[str] | None = None,
) -> list[dict[str, Any]]:
    """Compute precision, recall, F1, and support for every class."""
    y_t = np.asarray(y_true, dtype=np.int64)
    y_p = np.asarray(y_pred, dtype=np.int64)
    if num_classes is None:
        num_classes = int(max(y_t.max(initial=0), y_p.max(initial=0)) + 1)

    cm = compute_confusion_matrix(y_t, y_p, num_classes)
    tp = np.diag(cm).astype(np.float64)
    support = cm.sum(axis=1).astype(np.float64)
    pred_count = cm.sum(axis=0).astype(np.float64)

    precision = np.divide(tp, pred_count, out=np.zeros_like(tp), where=pred_count > 0)
    recall = np.divide(tp, support, out=np.zeros_like(tp), where=support > 0)
    denom = precision + recall
    f1 = np.divide(
        2.0 * precision * recall, denom, out=np.zeros_like(tp), where=denom > 0
    )

    rows = []
    for cid in range(num_classes):
        if isinstance(class_names, dict):
            gloss = class_names.get(cid, str(cid))
        elif isinstance(class_names, list) and cid < len(class_names):
            gloss = class_names[cid]
        else:
            gloss = str(cid)

        rows.append(
            {
                "class_id": cid,
                "gloss": gloss,
                "support": int(support[cid]),
                "predicted_count": int(pred_count[cid]),
                "precision": float(precision[cid]),
                "recall": float(recall[cid]),
                "f1": float(f1[cid]),
            }
        )
    return rows


def compute_classification_metrics(
    y_true: np.ndarray,
    scores: np.ndarray,
    *,
    top_k: int = 5,
    macro_over_present_only: bool = True,
) -> dict[str, Any]:
    """Compute Top-1, Top-K, macro precision/recall/F1, and prediction diversity."""
    y_t = np.asarray(y_true, dtype=np.int64)
    logits = np.asarray(scores, dtype=np.float64)
    if logits.ndim != 2:
        raise ValueError(f"Expected 2D scores (N, C), got {logits.shape}")
    if len(y_t) != len(logits):
        raise ValueError(f"Sample count mismatch: {len(y_t)} vs {len(logits)}")
    if len(y_t) == 0:
        raise ValueError("Cannot compute metrics on empty arrays.")

    num_classes = logits.shape[1]
    y_pred = np.argmax(logits, axis=1)
    top1 = float(np.mean(y_pred == y_t))

    k = min(max(1, top_k), num_classes)
    topk_idx = np.argpartition(-logits, kth=k - 1, axis=1)[:, :k]
    topk_acc = float(np.mean(np.any(topk_idx == y_t[:, None], axis=1)))

    per_class = compute_per_class_metrics(y_t, y_pred, num_classes=num_classes)
    if macro_over_present_only:
        eval_rows = [r for r in per_class if r["support"] > 0]
    else:
        eval_rows = per_class

    macro_precision = float(np.mean([r["precision"] for r in eval_rows]))
    macro_recall = float(np.mean([r["recall"] for r in eval_rows]))
    macro_f1 = float(np.mean([r["f1"] for r in eval_rows]))

    return {
        "samples": int(len(y_t)),
        "num_classes": int(num_classes),
        "evaluated_classes": int(len(eval_rows)),
        "top1_accuracy": top1,
        f"top{k}_accuracy": topk_acc,
        "macro_precision": macro_precision,
        "macro_recall": macro_recall,
        "macro_f1": macro_f1,
        "unique_predicted_classes": int(len(np.unique(y_pred))),
    }


def compute_majority_and_random_baselines(
    y_train: np.ndarray,
    y_eval: np.ndarray,
    num_classes: int,
) -> dict[str, float]:
    """Compute theoretical uniform random and empirical majority-class baselines."""
    y_tr = np.asarray(y_train, dtype=np.int64)
    y_ev = np.asarray(y_eval, dtype=np.int64)
    if len(y_tr) == 0 or len(y_ev) == 0:
        raise ValueError("Train and eval label arrays must be non-empty.")

    counts = np.bincount(y_tr, minlength=num_classes)
    majority_class = int(np.argmax(counts))
    majority_top1 = float(np.mean(y_ev == majority_class))

    top5_classes = np.argsort(-counts)[: min(5, num_classes)]
    majority_top5 = float(np.mean(np.isin(y_ev, top5_classes)))

    return {
        "majority_class_id": majority_class,
        "majority_top1": majority_top1,
        "majority_top5": majority_top5,
        "uniform_random_top1": 1.0 / float(num_classes),
        "uniform_random_top5": min(5, num_classes) / float(num_classes),
    }


def compute_retrieval_metrics(
    similarity_matrix: np.ndarray,
    query_labels: np.ndarray,
    gallery_labels: np.ndarray | None = None,
    ks: tuple[int, ...] = (1, 5, 10),
) -> dict[str, float]:
    """Compute Recall@K, MRR, and median rank for embedding retrieval."""
    sim = np.asarray(similarity_matrix, dtype=np.float64)
    q_lbl = np.asarray(query_labels, dtype=np.int64)
    g_lbl = (
        np.arange(sim.shape[1], dtype=np.int64)
        if gallery_labels is None
        else np.asarray(gallery_labels, dtype=np.int64)
    )
    if sim.shape != (len(q_lbl), len(g_lbl)):
        raise ValueError(
            f"Similarity shape {sim.shape} incompatible with "
            f"queries ({len(q_lbl)}) and gallery ({len(g_lbl)})."
        )

    ranked_indices = np.argsort(-sim, axis=1)
    ranks = []
    for row_idx, order in enumerate(ranked_indices):
        matches = np.flatnonzero(g_lbl[order] == q_lbl[row_idx])
        if len(matches) == 0:
            raise ValueError(f"Query label {q_lbl[row_idx]} has no match in gallery.")
        ranks.append(int(matches[0]) + 1)

    ranks_arr = np.asarray(ranks, dtype=np.float64)
    metrics: dict[str, float] = {
        "mrr": float(np.mean(1.0 / ranks_arr)),
        "median_rank": float(np.median(ranks_arr)),
        "mean_rank": float(np.mean(ranks_arr)),
    }
    for k in ks:
        eff_k = min(k, len(g_lbl))
        metrics[f"recall@{k}"] = float(np.mean(ranks_arr <= eff_k))
    return metrics

"""Inference service wrapper loading trained checkpoints or production bundles."""

import os
from pathlib import Path
from typing import Any

from src.inference.predictor import ASLEnsembleService, ASLInferenceService

_SERVICE_INSTANCE: Any | None = None


def get_inference_service() -> Any:
    """Return cached inference service loaded from configured checkpoint paths or bundle."""
    global _SERVICE_INSTANCE
    if _SERVICE_INSTANCE is not None:
        return _SERVICE_INSTANCE

    ckpt_path = os.getenv("ASL_CHECKPOINT_PATH", "").strip()
    vocab_path = os.getenv("ASL_VOCAB_EMBEDDINGS_PATH", "").strip() or None

    if not ckpt_path:
        default_bundle = Path("models/final/production_model")
        if default_bundle.is_dir():
            ckpt_path = str(default_bundle)
        else:
            raise RuntimeError(
                "ASL_CHECKPOINT_PATH is not configured and models/final/production_model not found."
            )

    p = Path(ckpt_path)
    if p.is_dir():
        _SERVICE_INSTANCE = ASLEnsembleService.from_bundle(p)
    else:
        _SERVICE_INSTANCE = ASLInferenceService.from_checkpoint(
            p,
            vocab_embeddings_path=Path(vocab_path) if vocab_path else None,
        )
    return _SERVICE_INSTANCE


def reset_inference_service() -> None:
    """Clear cached inference service (useful for tests or checkpoint reloads)."""
    global _SERVICE_INSTANCE
    _SERVICE_INSTANCE = None

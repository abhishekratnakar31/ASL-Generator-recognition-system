"""Inference service wrapper loading trained checkpoints without placeholders."""

import os
from pathlib import Path

from src.inference.predictor import ASLInferenceService

_SERVICE_INSTANCE: ASLInferenceService | None = None


def get_inference_service() -> ASLInferenceService:
    """Return cached ASLInferenceService loaded from configured checkpoint paths."""
    global _SERVICE_INSTANCE
    if _SERVICE_INSTANCE is not None:
        return _SERVICE_INSTANCE

    ckpt_path = os.getenv("ASL_CHECKPOINT_PATH", "").strip()
    vocab_path = os.getenv("ASL_VOCAB_EMBEDDINGS_PATH", "").strip() or None
    if not ckpt_path:
        raise RuntimeError(
            "ASL_CHECKPOINT_PATH is not configured. Load a trained checkpoint before serving."
        )

    _SERVICE_INSTANCE = ASLInferenceService.from_checkpoint(
        Path(ckpt_path),
        vocab_embeddings_path=Path(vocab_path) if vocab_path else None,
    )
    return _SERVICE_INSTANCE


def reset_inference_service() -> None:
    """Clear cached inference service (useful for tests or checkpoint reloads)."""
    global _SERVICE_INSTANCE
    _SERVICE_INSTANCE = None


"""Unit tests for visual, language, multimodal models, losses, and trainer."""

import ast
from pathlib import Path
from typing import TYPE_CHECKING

import pytest

ROOT_DIR = Path(__file__).resolve().parent.parent

if TYPE_CHECKING:
    import torch

    HAS_TORCH = True
else:
    try:
        import torch

        HAS_TORCH = True
    except ImportError:
        torch = None  # type: ignore[assignment]
        HAS_TORCH = False


def test_all_src_and_script_files_parse_cleanly():
    """Verify every Python module in src/ and scripts/ has valid syntax."""
    py_files = list((ROOT_DIR / "src").rglob("*.py")) + list(
        (ROOT_DIR / "scripts").rglob("*.py")
    )
    assert len(py_files) > 0
    for path in py_files:
        source = path.read_text(encoding="utf-8")
        ast.parse(source, filename=str(path))


@pytest.mark.skipif(not HAS_TORCH, reason="PyTorch not installed in local environment")
def test_visual_classifier_forward_and_backward_from_features():
    from src.models.visual.visual_model import VisualClassifier

    for temporal_type, pooling in [
        ("lstm", "last"),
        ("lstm", "attention"),
        ("transformer", "attention"),
    ]:
        model = VisualClassifier(
            num_classes=23,
            feature_dim=64,
            hidden_dim=32,
            num_layers=2,
            temporal_type=temporal_type,
            pooling=pooling,
            use_backbone=False,
        )
        x = torch.randn(4, 16, 64)
        y = torch.randint(0, 23, (4,))
        logits = model(x)
        assert logits.shape == (4, 23)
        loss = torch.nn.functional.cross_entropy(logits, y)
        loss.backward()


@pytest.mark.skipif(not HAS_TORCH, reason="PyTorch not installed in local environment")
def test_multimodal_model_and_losses():
    from src.models.language.language_encoder import build_gloss_texts
    from src.models.multimodal.multimodal_model import MultimodalASLModel
    from src.training.losses import JointMultimodalLoss, symmetric_infonce_loss

    texts = build_gloss_texts(["book", "drink", "computer"])
    assert texts == ["book", "drink", "computer"]

    for fusion_mode in ["contrastive", "hybrid", "gated_fusion"]:
        model = MultimodalASLModel(
            num_classes=5,
            visual_feature_dim=64,
            temporal_hidden_dim=32,
            temporal_layers=1,
            text_dim=48,
            embed_dim=24,
            fusion_mode=fusion_mode,
        )
        vis = torch.randn(4, 10, 64)
        vocab_txt = torch.randn(5, 48)
        labels = torch.tensor([0, 1, 0, 3], dtype=torch.long)

        out = model(vis, vocab_txt)
        assert out["logits"].shape == (4, 5)
        assert out["cosine_logits"].shape == (4, 5)
        assert out["visual_embeddings"].shape == (4, 24)
        assert out["text_embeddings"].shape == (5, 24)

        norms_v = torch.norm(out["visual_embeddings"], p=2, dim=-1)
        norms_t = torch.norm(out["text_embeddings"], p=2, dim=-1)
        assert torch.allclose(norms_v, torch.ones_like(norms_v), atol=1e-5)
        assert torch.allclose(norms_t, torch.ones_like(norms_t), atol=1e-5)

        criterion = JointMultimodalLoss()
        loss, breakdown = criterion(out, labels)
        assert torch.isfinite(loss)
        assert "cls_loss" in breakdown
        loss.backward()

    # Check multi-positive InfoNCE directly
    v_emb = torch.nn.functional.normalize(torch.randn(4, 16), dim=-1)
    t_emb = torch.nn.functional.normalize(torch.randn(4, 16), dim=-1)
    lbls = torch.tensor([0, 0, 1, 2])
    con_loss = symmetric_infonce_loss(v_emb, t_emb, labels=lbls)
    assert torch.isfinite(con_loss)


@pytest.mark.skipif(not HAS_TORCH, reason="PyTorch not installed in local environment")
def test_trainer_and_evaluation_roundtrip(tmp_path: Path):
    from torch.utils.data import DataLoader

    from src.data.dataset import CachedFeatureDataset
    from src.evaluation.evaluate import evaluate_checkpoint
    from src.models.visual.visual_model import VisualClassifier
    from src.training.trainer import TrainerConfig, get_device, set_seed, train_model

    set_seed(42)
    device = get_device("cpu")
    import numpy as np

    x_tr = np.random.randn(12, 8, 32).astype(np.float32)
    y_tr = np.array([0, 1, 2, 3] * 3, dtype=np.int64)
    x_va = np.random.randn(8, 8, 32).astype(np.float32)
    y_va = np.array([0, 1, 2, 3] * 2, dtype=np.int64)

    tr_loader = DataLoader(CachedFeatureDataset(x_tr, y_tr), batch_size=4, shuffle=True)
    va_loader = DataLoader(
        CachedFeatureDataset(x_va, y_va), batch_size=4, shuffle=False
    )

    model = VisualClassifier(
        num_classes=4, feature_dim=32, hidden_dim=16, num_layers=1
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    criterion = torch.nn.CrossEntropyLoss()
    cfg = TrainerConfig(experiment_id="smoke_unit", epochs=2, use_amp=False, patience=5)

    res = train_model(
        model, tr_loader, va_loader, criterion, optimizer, tmp_path / "run1", cfg
    )
    assert (tmp_path / "run1" / "best_model.pt").is_file()
    assert (tmp_path / "run1" / "training_history.csv").is_file()
    assert len(res["history"]) == 2

    eval_out = evaluate_checkpoint(
        model, va_loader, device, checkpoint_path=tmp_path / "run1" / "best_model.pt"
    )
    assert eval_out["metrics"]["samples"] == 8
    assert eval_out["confusion_matrix"].shape == (4, 4)

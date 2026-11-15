"""Configuration-driven training entrypoint for visual and multimodal ASL models."""

import argparse
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
import yaml
from torch.utils.data import DataLoader

from src.data.dataset import CachedFeatureDataset
from src.data.sampler import build_weighted_sampler, compute_inverse_frequency_weights
from src.data.split import load_class_mappings, map_features_by_manifest
from src.models.multimodal.multimodal_model import MultimodalASLModel
from src.models.visual.visual_model import VisualClassifier
from src.preprocessing.augmentation import augment_feature_sequence
from src.training.losses import JointMultimodalLoss, build_classification_loss
from src.training.trainer import TrainerConfig, get_device, set_seed, train_model


def load_yaml_config(config_path: str | Path) -> dict[str, Any]:
    """Load experiment configuration from YAML."""
    with Path(config_path).open("r", encoding="utf-8") as handle:
        cfg = yaml.safe_load(handle)
    if not isinstance(cfg, dict):
        raise ValueError(f"Expected mapping in YAML config: {config_path}")
    return cfg


def _build_optimizer(model: torch.nn.Module, train_cfg: dict[str, Any]) -> torch.optim.Optimizer:
    name = str(train_cfg.get("optimizer", "AdamW")).lower()
    lr = float(train_cfg.get("learning_rate", 1e-3))
    wd = float(train_cfg.get("weight_decay", 1e-4))
    params = [p for p in model.parameters() if p.requires_grad]
    if name == "adamw":
        return torch.optim.AdamW(params, lr=lr, weight_decay=wd)
    if name == "adam":
        return torch.optim.Adam(params, lr=lr, weight_decay=wd)
    if name == "sgd":
        return torch.optim.SGD(params, lr=lr, momentum=0.9, weight_decay=wd)
    raise ValueError(f"Unsupported optimizer='{name}'")


def _build_scheduler(optimizer: torch.optim.Optimizer, train_cfg: dict[str, Any]):
    sched = str(train_cfg.get("scheduler", "cosine")).lower()
    epochs = int(train_cfg.get("epochs", 40))
    if sched in {"cosine", "cosineannealinglr"}:
        return torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(1, epochs))
    if sched in {"plateau", "reducelronplateau"}:
        return torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="min", factor=0.5, patience=3, min_lr=1e-6
        )
    return None


def _load_split_features(
    cache_dir: Path,
    metadata_csv: Path,
    split_name: str,
) -> tuple[np.ndarray, np.ndarray]:
    meta_df = pd.read_csv(metadata_csv)
    manifest_df = pd.read_csv(cache_dir / f"manifest_{split_name}.csv")
    features = np.load(cache_dir / f"X_{split_name}.npy", mmap_mode="r")
    labels = np.load(cache_dir / f"y_{split_name}.npy")
    return map_features_by_manifest(meta_df, manifest_df, features, labels)


def run_training_from_config(config_path: str | Path) -> dict[str, Any]:
    """Run visual or multimodal training end-to-end from a YAML configuration file."""
    cfg = load_yaml_config(config_path)
    seed = int(cfg.get("seed", 42))
    set_seed(seed)
    device = get_device(cfg.get("device"))

    data_cfg = cfg["dataset"]
    model_cfg = cfg["model"]
    train_cfg = cfg["training"]
    mode = str(cfg.get("mode", "visual")).lower()

    meta_dir = Path(data_cfg["metadata_dir"])
    cache_dir = Path(data_cfg["feature_cache_dir"])
    class_to_id, id_to_class = load_class_mappings(
        meta_dir / "class_to_id.json",
        meta_dir / "id_to_class.json",
    )
    num_classes = len(class_to_id)

    x_train, y_train = _load_split_features(cache_dir, meta_dir / "train.csv", "train")
    x_val, y_val = _load_split_features(cache_dir, meta_dir / "val.csv", "val")

    aug_cfg = cfg.get("augmentation", {})
    aug_fn = None
    if aug_cfg.get("enabled", False):
        raw_crop = aug_cfg.get("crop_ratio_range", [0.85, 1.0])
        crop_range = (float(raw_crop[0]), float(raw_crop[1]))
        f_drop = float(aug_cfg.get("feature_dropout", 0.1))
        n_std = float(aug_cfg.get("noise_std", 0.02))

        def _augment_seq(seq: np.ndarray) -> np.ndarray:
            return augment_feature_sequence(
                seq,
                crop_ratio_range=crop_range,
                feature_dropout=f_drop,
                noise_std=n_std,
            )

        aug_fn = _augment_seq

    train_ds = CachedFeatureDataset(x_train, y_train, augment_fn=aug_fn)
    val_ds = CachedFeatureDataset(x_val, y_val)

    batch_size = int(train_cfg.get("batch_size", 32))
    use_sampler = bool(train_cfg.get("balanced_sampling", False))
    sampler = (
        build_weighted_sampler(y_train, power=float(train_cfg.get("sampler_power", 0.5)))
        if use_sampler
        else None
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=(sampler is None),
        sampler=sampler,
        num_workers=int(train_cfg.get("num_workers", 0)),
        pin_memory=(device.type == "cuda"),
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=int(train_cfg.get("num_workers", 0)),
        pin_memory=(device.type == "cuda"),
    )

    class_weights = None
    if bool(train_cfg.get("class_weighted_loss", False)):
        weights_np = compute_inverse_frequency_weights(
            y_train, num_classes=num_classes, power=float(train_cfg.get("weight_power", 0.5))
        )
        class_weights = torch.tensor(weights_np, dtype=torch.float32, device=device)

    vocab_embeddings = None
    model: torch.nn.Module
    criterion: Any
    if mode == "multimodal":
        vocab_path = Path(cfg["language"]["vocab_embeddings_path"])
        vocab_obj = torch.load(vocab_path, map_location="cpu", weights_only=False)
        vocab_embeddings = vocab_obj["text_embeddings"].float().to(device)
        model = MultimodalASLModel(
            num_classes=num_classes,
            visual_feature_dim=int(model_cfg.get("feature_dim", 2048)),
            temporal_hidden_dim=int(model_cfg.get("hidden_size", 256)),
            temporal_layers=int(model_cfg.get("num_layers", 2)),
            temporal_type=str(model_cfg.get("temporal", "lstm")),
            bidirectional=bool(model_cfg.get("bidirectional", False)),
            pooling=str(model_cfg.get("pooling", "attention")),
            text_dim=int(vocab_embeddings.size(1)),
            embed_dim=int(model_cfg.get("embed_dim", 256)),
            dropout=float(model_cfg.get("dropout", 0.3)),
            fusion_mode=str(model_cfg.get("fusion_mode", "hybrid")),
            fusion_alpha=float(model_cfg.get("fusion_alpha", 0.5)),
            use_backbone=False,
        ).to(device)
        loss_cfg = cfg.get("loss", {})
        criterion = JointMultimodalLoss(
            contrastive_weight=float(loss_cfg.get("contrastive_weight", 0.5)),
            classification_weight=float(loss_cfg.get("classification_weight", 0.5)),
            batch_infonce_weight=float(loss_cfg.get("batch_infonce_weight", 0.25)),
            class_weights=class_weights,
            label_smoothing=float(loss_cfg.get("label_smoothing", 0.05)),
        ).to(device)
    else:
        model = VisualClassifier(
            num_classes=num_classes,
            feature_dim=int(model_cfg.get("feature_dim", 2048)),
            hidden_dim=int(model_cfg.get("hidden_size", 256)),
            num_layers=int(model_cfg.get("num_layers", 2)),
            dropout=float(model_cfg.get("dropout", 0.3)),
            temporal_type=str(model_cfg.get("temporal", "lstm")),
            bidirectional=bool(model_cfg.get("bidirectional", False)),
            pooling=str(model_cfg.get("pooling", "last")),
            use_backbone=False,
        ).to(device)
        criterion = build_classification_loss(
            class_weights=class_weights,
            label_smoothing=float(cfg.get("loss", {}).get("label_smoothing", 0.0)),
        )

    optimizer = _build_optimizer(model, train_cfg)
    scheduler = _build_scheduler(optimizer, train_cfg)
    trainer_cfg = TrainerConfig(
        experiment_id=str(cfg.get("experiment_name", "asl_experiment")),
        epochs=int(train_cfg.get("epochs", 40)),
        learning_rate=float(train_cfg.get("learning_rate", 1e-3)),
        weight_decay=float(train_cfg.get("weight_decay", 1e-4)),
        grad_clip_norm=float(train_cfg.get("grad_clip_norm", 1.0)),
        patience=int(train_cfg.get("patience", 10)),
        use_amp=bool(train_cfg.get("use_amp", True)),
        seed=seed,
    )

    output_dir = Path(train_cfg.get("checkpoint_dir", "models/checkpoints")) / trainer_cfg.experiment_id
    return train_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        criterion=criterion,
        optimizer=optimizer,
        output_dir=output_dir,
        config=trainer_cfg,
        vocab_text_embeddings=vocab_embeddings,
        scheduler=scheduler,
        class_to_id=class_to_id,
        id_to_class=id_to_class,
        resume=bool(train_cfg.get("resume", True)),
        extra_metadata={"full_config": cfg},
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Train ASL visual or multimodal model.")
    parser.add_argument("--config", required=True, help="Path to YAML training config.")
    args = parser.parse_args()
    result = run_training_from_config(args.config)
    print(f"Training completed. Best epoch: {result['best_epoch']} | Best checkpoint: {result['best_checkpoint']}")


if __name__ == "__main__":
    main()

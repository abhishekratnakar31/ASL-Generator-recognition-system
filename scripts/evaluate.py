"""CLI script to evaluate a trained visual or multimodal ASL checkpoint."""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from torch.utils.data import DataLoader

from src.data.dataset import CachedFeatureDataset
from src.data.split import map_features_by_manifest
from src.evaluation.evaluate import evaluate_checkpoint
from src.inference.predictor import ASLInferenceService
from src.training.trainer import get_device


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate a saved ASL checkpoint.")
    parser.add_argument("--checkpoint", required=True, help="Path to best_model.pt")
    parser.add_argument("--metadata-csv", required=True, help="Evaluation split CSV.")
    parser.add_argument("--cache-dir", required=True, help="Feature cache directory.")
    parser.add_argument("--split", default="val", choices=["train", "val", "test"])
    parser.add_argument(
        "--vocab-embeddings", default=None, help="Optional text embeddings .pt"
    )
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument(
        "--output-json", default=None, help="Optional metrics JSON path."
    )
    args = parser.parse_args()

    device = get_device()
    service = ASLInferenceService.from_checkpoint(
        args.checkpoint,
        vocab_embeddings_path=args.vocab_embeddings,
        device=device,
        pretrained_backbone=False,
    )

    cache_dir = Path(args.cache_dir)
    meta_df = pd.read_csv(args.metadata_csv)
    man_df = pd.read_csv(cache_dir / f"manifest_{args.split}.csv")
    feats = np.load(cache_dir / f"X_{args.split}.npy", mmap_mode="r")
    labels = np.load(cache_dir / f"y_{args.split}.npy")
    aligned_x, aligned_y = map_features_by_manifest(meta_df, man_df, feats, labels)

    loader = DataLoader(
        CachedFeatureDataset(aligned_x, aligned_y),
        batch_size=args.batch_size,
        shuffle=False,
    )
    results = evaluate_checkpoint(
        service.model,
        loader,
        device,
        vocab_text_embeddings=service.vocab_text_embeddings,
        class_names=service.id_to_class,
        checkpoint_path=Path(args.checkpoint),
    )

    print(json.dumps(results["metrics"], indent=2))
    if args.output_json:
        out_p = Path(args.output_json)
        out_p.parent.mkdir(parents=True, exist_ok=True)
        out_p.write_text(json.dumps(results["metrics"], indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()

"""CLI utility to audit local WLASL metadata and video file availability."""

import argparse
from pathlib import Path

import pandas as pd

from src.data.split import load_class_mappings, normalize_video_id, validate_splits


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Verify local WLASL metadata splits and video file availability."
    )
    parser.add_argument(
        "--metadata-dir",
        required=True,
        help="Directory containing train.csv, val.csv, test.csv, and class JSONs.",
    )
    parser.add_argument(
        "--videos-dir",
        default=None,
        help="Optional directory containing downloaded video files.",
    )
    args = parser.parse_args()

    meta_dir = Path(args.metadata_dir)
    c2i, _ = load_class_mappings(
        meta_dir / "class_to_id.json", meta_dir / "id_to_class.json"
    )
    train_df = pd.read_csv(meta_dir / "train.csv")
    val_df = pd.read_csv(meta_dir / "val.csv")
    test_df = pd.read_csv(meta_dir / "test.csv")

    summary = validate_splits(train_df, val_df, test_df, c2i)
    print("Split counts:", summary["split_counts"])
    print("Signer counts:", summary["signer_counts"])
    print("Missing training classes:", summary["missing_train_classes"])

    if args.videos_dir is not None:
        vdir = Path(args.videos_dir)
        exts = {".mp4", ".webm", ".mkv", ".avi", ".mov"}
        on_disk = {
            normalize_video_id(p.stem)
            for p in vdir.rglob("*")
            if p.is_file() and p.suffix.lower() in exts
        }
        for split_name, df in [("train", train_df), ("val", val_df), ("test", test_df)]:
            col = (
                "video_id_normalized"
                if "video_id_normalized" in df.columns
                else "video_id"
            )
            expected = set(df[col].map(normalize_video_id))
            found = len(expected & on_disk)
            print(f"{split_name}: {found}/{len(expected)} videos present in {vdir}")


if __name__ == "__main__":
    main()

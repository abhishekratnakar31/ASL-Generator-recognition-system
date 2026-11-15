"""CLI utility for segment-aware video preprocessing and ResNet50 feature caching."""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

from src.data.split import normalize_video_id
from src.models.visual.resnet import ResNet50Backbone
from src.preprocessing.video import preprocess_video


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Extract segment-aware ResNet50 features into a new directory."
    )
    parser.add_argument("--metadata-csv", required=True, help="Split metadata CSV path.")
    parser.add_argument("--videos-dir", required=True, help="Directory with video files.")
    parser.add_argument("--output-dir", required=True, help="Destination cache directory.")
    parser.add_argument("--split", default="val", help="Split label for output files.")
    parser.add_argument("--num-frames", type=int, default=32)
    parser.add_argument("--image-size", type=int, default=224)
    args = parser.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(args.metadata_csv)
    vdir = Path(args.videos_dir)
    exts = {".mp4", ".webm", ".mkv", ".avi", ".mov"}
    index = {
        normalize_video_id(p.stem): p
        for p in vdir.rglob("*")
        if p.is_file() and p.suffix.lower() in exts
    }

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    backbone = ResNet50Backbone(pretrained=True, frozen=True).to(device).eval()

    x_path = out_dir / f"X_{args.split}.npy"
    x_mmap = np.lib.format.open_memmap(
        str(x_path),
        mode="w+",
        dtype=np.float16,
        shape=(len(df), args.num_frames, 2048),
    )
    done = np.zeros(len(df), dtype=bool)
    labels = df["class_id"].to_numpy(dtype=np.int64)
    id_col = "video_id_normalized" if "video_id_normalized" in df.columns else "video_id"

    with torch.inference_mode():
        for row_idx, (_, row) in enumerate(
            tqdm(df.iterrows(), total=len(df), desc=f"Preprocess {args.split}")
        ):
            vid = normalize_video_id(row[id_col])
            vp = index.get(vid)
            if vp is None:
                continue
            fps_val = row.get("fps")
            meta_fps = (
                float(fps_val)
                if fps_val is not None and bool(pd.notna(fps_val))
                else 25.0
            )
            clip = (
                preprocess_video(
                    vp,
                    frame_start=row.get("frame_start", 1),
                    frame_end=row.get("frame_end", -1),
                    metadata_fps=meta_fps,
                    num_frames=args.num_frames,
                    image_size=args.image_size,
                    return_tensor=True,
                )
                .unsqueeze(0)
                .to(device)
            )
            feats = backbone(clip).squeeze(0).float().cpu().numpy().astype(np.float16)
            x_mmap[row_idx] = feats
            done[row_idx] = True

    x_mmap.flush()
    np.save(out_dir / f"y_{args.split}.npy", labels)
    np.save(out_dir / f"done_{args.split}.npy", done)
    df.to_csv(out_dir / f"manifest_{args.split}.csv", index=False)
    print(f"Saved {int(done.sum())}/{len(df)} sequences to {x_path}")


if __name__ == "__main__":
    main()

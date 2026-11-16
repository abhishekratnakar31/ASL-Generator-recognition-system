"""PyTorch Dataset abstractions for cached features and segment-aware videos."""

from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable

import numpy as np
import pandas as pd

if TYPE_CHECKING:
    import torch
    from torch.utils.data import Dataset

    _HAS_TORCH = True
else:
    try:
        import torch
        from torch.utils.data import Dataset

        _HAS_TORCH = True
    except (
        ImportError
    ):  # pragma: no cover - allows non-torch environments to inspect/import
        _HAS_TORCH = False
        torch = None  # type: ignore[assignment]
        Dataset = object  # type: ignore[assignment,misc]

from src.preprocessing.video import preprocess_video


class CachedFeatureDataset(Dataset):
    """Dataset serving aligned pre-extracted (T, D) visual feature sequences."""

    def __init__(
        self,
        features: np.ndarray,
        labels: np.ndarray,
        *,
        augment_fn: Callable[[np.ndarray], np.ndarray] | None = None,
    ) -> None:
        if not _HAS_TORCH:
            raise ImportError(
                "PyTorch is required to instantiate CachedFeatureDataset."
            )
        if len(features) != len(labels):
            raise ValueError(
                f"Feature count ({len(features)}) != label count ({len(labels)})."
            )
        self.features = np.asarray(features)
        self.labels = np.asarray(labels, dtype=np.int64)
        self.augment_fn = augment_fn

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, index: int) -> tuple[Any, Any]:
        seq = np.array(self.features[index], dtype=np.float32, copy=True)
        if self.augment_fn is not None:
            seq = np.asarray(self.augment_fn(seq), dtype=np.float32)
        return torch.from_numpy(seq), torch.tensor(
            int(self.labels[index]), dtype=torch.long
        )


class WLASLVideoDataset(Dataset):
    """Dataset decoding annotated video segments deterministically into (T, 3, H, W)."""

    def __init__(
        self,
        dataframe: pd.DataFrame,
        *,
        path_col: str = "resolved_video_path",
        num_frames: int = 32,
        image_size: int = 224,
        convert_fps: bool = True,
        one_based: bool = True,
        strict_bounds: bool = False,
        augment_fn: Callable[[Any], Any] | None = None,
    ) -> None:
        if not _HAS_TORCH:
            raise ImportError("PyTorch is required to instantiate WLASLVideoDataset.")
        if path_col not in dataframe.columns:
            raise ValueError(f"Missing required path column '{path_col}'.")
        if "class_id" not in dataframe.columns:
            raise ValueError("Missing required 'class_id' column.")

        self.df = dataframe.reset_index(drop=True).copy()
        self.path_col = path_col
        self.num_frames = int(num_frames)
        self.image_size = int(image_size)
        self.convert_fps = bool(convert_fps)
        self.one_based = bool(one_based)
        self.strict_bounds = bool(strict_bounds)
        self.augment_fn = augment_fn

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, index: int) -> tuple[Any, Any]:
        row = self.df.iloc[index]
        video_path = Path(str(row[self.path_col]))
        tensor = preprocess_video(
            video_path=video_path,
            frame_start=row.get("frame_start", 1),
            frame_end=row.get("frame_end", -1),
            metadata_fps=float(row["fps"]) if pd.notna(row.get("fps")) else 25.0,
            num_frames=self.num_frames,
            image_size=self.image_size,
            convert_fps=self.convert_fps,
            one_based=self.one_based,
            strict_bounds=self.strict_bounds,
            return_tensor=True,
        )
        if self.augment_fn is not None:
            tensor = self.augment_fn(tensor)
        label = int(row["class_id"])
        return tensor, torch.tensor(label, dtype=torch.long)

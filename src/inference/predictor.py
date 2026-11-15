"""End-to-end ASL inference pipeline using trained visual or multimodal checkpoints."""

import time
from pathlib import Path
from typing import Any

import torch

from src.models.multimodal.multimodal_model import MultimodalASLModel
from src.models.visual.resnet import ResNet50Backbone
from src.models.visual.visual_model import VisualClassifier
from src.preprocessing.video import preprocess_video
from src.training.trainer import get_device


class ASLInferenceService:
    """Loads saved checkpoints and runs identical training-time video preprocessing."""

    def __init__(
        self,
        model: torch.nn.Module,
        id_to_class: dict[int, str],
        *,
        device: torch.device,
        model_type: str,
        vocab_text_embeddings: torch.Tensor | None = None,
        backbone: ResNet50Backbone | None = None,
        num_frames: int = 32,
        image_size: int = 224,
    ) -> None:
        self.model = model.to(device).eval()
        self.id_to_class = {int(k): str(v) for k, v in id_to_class.items()}
        self.device = device
        self.model_type = model_type
        self.vocab_text_embeddings = (
            vocab_text_embeddings.to(device) if vocab_text_embeddings is not None else None
        )
        self.backbone = backbone.to(device).eval() if backbone is not None else None
        self.num_frames = int(num_frames)
        self.image_size = int(image_size)

    @classmethod
    def from_checkpoint(
        cls,
        checkpoint_path: str | Path,
        *,
        vocab_embeddings_path: str | Path | None = None,
        device: str | torch.device | None = None,
        pretrained_backbone: bool = True,
    ) -> "ASLInferenceService":
        """Reconstruct a trained visual or multimodal model from a saved checkpoint."""
        ckpt_path = Path(checkpoint_path)
        if not ckpt_path.is_file():
            raise FileNotFoundError(f"Trained checkpoint not found: {ckpt_path}")

        resolved_device = (
            torch.device(device) if device is not None else get_device()
        )
        ckpt = torch.load(ckpt_path, map_location=resolved_device, weights_only=False)

        state_dict = ckpt["model_state_dict"]
        raw_i2c = ckpt.get("id_to_class")
        raw_c2i = ckpt.get("class_to_id")
        if raw_i2c is not None:
            id_to_class = {int(k): str(v) for k, v in raw_i2c.items()}
        elif raw_c2i is not None:
            id_to_class = {int(v): str(k) for k, v in raw_c2i.items()}
        else:
            raise ValueError("Checkpoint missing 'id_to_class' / 'class_to_id' mapping.")

        num_classes = len(id_to_class)
        cfg = ckpt.get("config", {})
        full_cfg = cfg.get("full_config", cfg)
        model_cfg = full_cfg.get("model", full_cfg)
        data_cfg = full_cfg.get("dataset", {})
        has_backbone_weights = any(k.startswith("backbone.") for k in state_dict)

        is_pose = (
            any(k.startswith("pose_encoder.") for k in state_dict)
            or "pose_dim" in model_cfg
        )
        is_multimodal = (
            "text_projection.net.1.weight" in state_dict
            or any(k.startswith("text_projection.") for k in state_dict)
        )

        vocab_tensor = None
        if is_multimodal:
            if vocab_embeddings_path is not None:
                vp = Path(vocab_embeddings_path)
                if not vp.is_file():
                    raise FileNotFoundError(f"Vocabulary embeddings not found: {vp}")
                vocab_obj = torch.load(vp, map_location="cpu", weights_only=False)
                vocab_tensor = vocab_obj["text_embeddings"].float()
            elif "vocab_text_embeddings" in ckpt:
                vocab_tensor = ckpt["vocab_text_embeddings"].float()
            else:
                raise ValueError(
                    "Multimodal checkpoint requires `vocab_embeddings_path` or embedded vocabulary tensor."
                )

        if is_pose:
            from src.models.multimodal.pose_multimodal_model import PoseVisualMultimodalModel

            mode = str(model_cfg.get("mode", "pose_language" if is_multimodal else "pose_only"))
            backbone_type = str(model_cfg.get("backbone", "transformer"))
            use_cos = bool(
                model_cfg.get("use_cosine_classifier", False)
                or any("classifier_head.1.logit_scale" in k for k in state_dict)
            )
            model = PoseVisualMultimodalModel(
                num_classes=num_classes,
                pose_dim=int(model_cfg.get("pose_dim", 349)),
                visual_dim=int(model_cfg.get("visual_dim", 2048)),
                text_dim=int(vocab_tensor.size(1)) if vocab_tensor is not None else 384,
                hidden_dim=int(model_cfg.get("hidden_dim", 192)),
                embed_dim=int(model_cfg.get("embed_dim", 192)),
                dropout=float(model_cfg.get("dropout", 0.35)),
                mode=mode,
                backbone=backbone_type,
                use_cosine_classifier=use_cos,
            )
            model.load_state_dict(state_dict, strict=True)
            model_type = f"pose_{mode}"
            backbone = None
        elif is_multimodal:
            model = MultimodalASLModel(
                num_classes=num_classes,
                visual_feature_dim=int(model_cfg.get("feature_dim", 2048)),
                temporal_hidden_dim=int(model_cfg.get("hidden_size", model_cfg.get("hidden_dim", 256))),
                temporal_layers=int(model_cfg.get("num_layers", 2)),
                temporal_type=str(model_cfg.get("temporal", "lstm")),
                bidirectional=bool(model_cfg.get("bidirectional", False)),
                pooling=str(model_cfg.get("pooling", "attention")),
                text_dim=int(vocab_tensor.size(1)),
                embed_dim=int(model_cfg.get("embed_dim", 256)),
                dropout=float(model_cfg.get("dropout", 0.3)),
                fusion_mode=str(model_cfg.get("fusion_mode", "hybrid")),
                fusion_alpha=float(model_cfg.get("fusion_alpha", 0.5)),
                use_backbone=has_backbone_weights,
                pretrained_backbone=False,
            )
            model.load_state_dict(state_dict, strict=True)
            model_type = "multimodal"
            backbone = (
                None
                if has_backbone_weights
                else ResNet50Backbone(pretrained=pretrained_backbone, frozen=True)
            )
        else:
            model = VisualClassifier(
                num_classes=num_classes,
                feature_dim=int(model_cfg.get("feature_dim", 2048)),
                hidden_dim=int(model_cfg.get("hidden_size", model_cfg.get("hidden_dim", 256))),
                num_layers=int(model_cfg.get("num_layers", 2)),
                dropout=float(model_cfg.get("dropout", 0.3)),
                temporal_type=str(model_cfg.get("temporal", "lstm")),
                bidirectional=bool(model_cfg.get("bidirectional", False)),
                pooling=str(model_cfg.get("pooling", "last")),
                use_backbone=has_backbone_weights,
                pretrained_backbone=False,
            )
            model.load_state_dict(state_dict, strict=True)
            model_type = "visual_only"
            backbone = (
                None
                if has_backbone_weights
                else ResNet50Backbone(pretrained=pretrained_backbone, frozen=True)
            )

        return cls(
            model=model,
            id_to_class=id_to_class,
            device=resolved_device,
            model_type=model_type,
            vocab_text_embeddings=vocab_tensor,
            backbone=backbone,
            num_frames=int(data_cfg.get("num_frames", 32)),
            image_size=int(data_cfg.get("image_size", 224)),
        )

    @torch.inference_mode()
    def predict_tensor(
        self,
        clip_tensor: torch.Tensor,
        *,
        top_k: int = 5,
    ) -> dict[str, Any]:
        """Run inference from (1, T, 3, H, W) video frames or (1, T, 2048) features."""
        t0 = time.perf_counter()
        x = clip_tensor.to(self.device)
        if x.ndim == 4:
            x = x.unsqueeze(0)
        if x.ndim == 5 and self.backbone is not None:
            x = self.backbone(x)

        if self.model_type.startswith("pose_"):
            if "language" in self.model_type:
                if self.vocab_text_embeddings is None:
                    raise RuntimeError("Multimodal pose model missing vocabulary embeddings.")
                out = self.model(x, vocab_text_embeddings=self.vocab_text_embeddings)
            else:
                out = self.model(x)
            logits = out["logits"]
        elif self.model_type == "multimodal":
            if self.vocab_text_embeddings is None:
                raise RuntimeError("Multimodal model missing vocabulary embeddings.")
            out = self.model(x, self.vocab_text_embeddings)
            logits = out["logits"]
        else:
            logits = self.model(x)

        probs = torch.softmax(logits[0], dim=-1)
        k = min(max(1, int(top_k)), probs.numel())
        top_probs, top_ids = probs.topk(k)
        raw_scores = logits[0, top_ids]

        predictions = []
        for prob, cid, score in zip(
            top_probs.cpu().tolist(),
            top_ids.cpu().tolist(),
            raw_scores.cpu().tolist(),
        ):
            predictions.append(
                {
                    "class_id": int(cid),
                    "gloss": self.id_to_class.get(int(cid), str(cid)),
                    "confidence": float(prob),
                    "score": float(score),
                }
            )

        latency_ms = round((time.perf_counter() - t0) * 1000.0, 2)
        return {
            "predicted_gloss": predictions[0]["gloss"],
            "predicted_class_id": predictions[0]["class_id"],
            "confidence": predictions[0]["confidence"],
            "top_k": predictions,
            "model_type": self.model_type,
            "latency_ms": latency_ms,
        }

    def predict_video(
        self,
        video_path: str | Path,
        *,
        frame_start: Any = 1,
        frame_end: Any = -1,
        metadata_fps: float | None = 25.0,
        top_k: int = 5,
    ) -> dict[str, Any]:
        """Preprocess a raw video identically to training and return Top-K glosses."""
        clip_tensor = preprocess_video(
            video_path=video_path,
            frame_start=frame_start,
            frame_end=frame_end,
            metadata_fps=metadata_fps,
            num_frames=self.num_frames,
            image_size=self.image_size,
            convert_fps=True,
            one_based=True,
            strict_bounds=False,
            seek_preroll=None,
            return_tensor=True,
        )
        return self.predict_tensor(clip_tensor, top_k=top_k)


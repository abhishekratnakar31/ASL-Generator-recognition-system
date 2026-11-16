"""End-to-end ASL inference pipeline using trained visual or multimodal checkpoints."""

import time
from pathlib import Path
from typing import Any

import torch

from src.models.multimodal.multimodal_model import MultimodalASLModel
from src.models.visual.resnet import ResNet50Backbone
from src.models.visual.visual_model import VisualClassifier
from src.preprocessing.pose import PoseExtractor
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
        pose_extractor: PoseExtractor | None = None,
        text_encoder: Any = None,
        num_frames: int = 32,
        image_size: int = 224,
    ) -> None:
        self.model = model.to(device).eval()
        self.id_to_class = {int(k): str(v) for k, v in id_to_class.items()}
        self.device = device
        self.model_type = model_type
        self.vocab_text_embeddings = (
            vocab_text_embeddings.to(device)
            if vocab_text_embeddings is not None
            else None
        )
        self.backbone = backbone.to(device).eval() if backbone is not None else None
        self._pose_extractor = pose_extractor
        self.text_encoder = text_encoder
        self.num_frames = int(num_frames)
        self.image_size = int(image_size)

    @property
    def pose_extractor(self) -> PoseExtractor:
        if self._pose_extractor is None:
            self._pose_extractor = PoseExtractor(device="cpu")
        return self._pose_extractor

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

        resolved_device = torch.device(device) if device is not None else get_device()
        ckpt = torch.load(ckpt_path, map_location=resolved_device, weights_only=False)

        state_dict = ckpt["model_state_dict"]
        raw_i2c = ckpt.get("id_to_class")
        raw_c2i = ckpt.get("class_to_id")
        if raw_i2c is not None:
            id_to_class = {int(k): str(v) for k, v in raw_i2c.items()}
        elif raw_c2i is not None:
            id_to_class = {int(v): str(k) for k, v in raw_c2i.items()}
        else:
            raise ValueError(
                "Checkpoint missing 'id_to_class' / 'class_to_id' mapping."
            )

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
        is_multimodal = "text_projection.net.1.weight" in state_dict or any(
            k.startswith("text_projection.") for k in state_dict
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
            from src.models.multimodal.pose_multimodal_model import (
                PoseVisualMultimodalModel,
            )

            mode = str(
                model_cfg.get("mode", "pose_language" if is_multimodal else "pose_only")
            )
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
                temporal_hidden_dim=int(
                    model_cfg.get("hidden_size", model_cfg.get("hidden_dim", 256))
                ),
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
                hidden_dim=int(
                    model_cfg.get("hidden_size", model_cfg.get("hidden_dim", 256))
                ),
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
                    raise RuntimeError(
                        "Multimodal pose model missing vocabulary embeddings."
                    )
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
        if self.model_type.startswith("pose_"):
            pose_seq = self.pose_extractor.extract_from_video(
                video_path=video_path,
                num_frames=self.num_frames,
                frame_start=frame_start,
                frame_end=frame_end,
                metadata_fps=metadata_fps,
            )
            clip_tensor = torch.from_numpy(pose_seq).float().unsqueeze(0)
            return self.predict_tensor(clip_tensor, top_k=top_k)

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

    def predict_frames(
        self,
        frames: list[Any] | Any,
        *,
        top_k: int = 5,
    ) -> dict[str, Any]:
        """Run inference from a sequence of raw RGB/BGR image frames."""
        if self.model_type.startswith("pose_"):
            pose_seq = self.pose_extractor.extract_from_frames(frames)
            clip_tensor = torch.from_numpy(pose_seq).float().unsqueeze(0)
            return self.predict_tensor(clip_tensor, top_k=top_k)

        from src.preprocessing.frames import preprocess_bgr_frames

        tensor_frames = preprocess_bgr_frames(frames, image_size=self.image_size)
        clip_tensor = tensor_frames.unsqueeze(0)
        return self.predict_tensor(clip_tensor, top_k=top_k)

    def predict_pose(
        self,
        pose_data: Any,
        *,
        top_k: int = 5,
    ) -> dict[str, Any]:
        """Run direct inference on (T, 349) pre-extracted pose kinematic features."""
        import numpy as np

        if isinstance(pose_data, list):
            arr = np.asarray(pose_data, dtype=np.float32)
            tensor = torch.from_numpy(arr).float()
        elif isinstance(pose_data, np.ndarray):
            tensor = torch.from_numpy(pose_data).float()
        elif isinstance(pose_data, torch.Tensor):
            tensor = pose_data.float()
        else:
            raise TypeError(f"Unsupported pose data type: {type(pose_data)}")

        if tensor.ndim == 2:
            tensor = tensor.unsqueeze(0)
        return self.predict_tensor(tensor, top_k=top_k)

    def retrieve_sign_from_text(
        self,
        query: str,
        *,
        top_k: int = 5,
    ) -> dict[str, Any]:
        """ASL Generator: Semantically search the 100-gloss vocabulary from an English query."""
        t0 = time.perf_counter()
        query_str = str(query).strip().lower()
        if not query_str:
            raise ValueError("Query string cannot be empty.")

        k = min(max(1, int(top_k)), len(self.id_to_class))
        matches: list[dict[str, Any]] = []

        # 1. Exact or partial gloss match check
        class_to_id = {v.lower(): k for k, v in self.id_to_class.items()}
        if query_str in class_to_id:
            cid = class_to_id[query_str]
            matches.append(
                {
                    "class_id": cid,
                    "gloss": self.id_to_class[cid],
                    "similarity": 1.0,
                    "exact_match": True,
                }
            )

        # 2. Semantic text retrieval if vocabulary embeddings are available
        if self.vocab_text_embeddings is not None and len(matches) < k:
            try:
                encoder = self.text_encoder
                if encoder is None:
                    from sentence_transformers import SentenceTransformer

                    encoder = SentenceTransformer(
                        "all-MiniLM-L6-v2", device=str(self.device)
                    )
                    self.text_encoder = encoder

                q_emb = encoder.encode(
                    [
                        f"An American Sign Language sign representing the word '{query_str}'"
                    ],
                    convert_to_tensor=True,
                    device=str(self.device),
                )
                if hasattr(self.model, "text_projection"):
                    q_proj = self.model.text_projection(q_emb.float())
                else:
                    q_proj = q_emb.float()
                q_proj = torch.nn.functional.normalize(q_proj, dim=-1)

                v_proj = self.vocab_text_embeddings
                if hasattr(self.model, "text_projection"):
                    v_proj = self.model.text_projection(v_proj)
                v_proj = torch.nn.functional.normalize(v_proj, dim=-1)

                sims = torch.matmul(q_proj, v_proj.t())[0]
                top_sims, top_ids = sims.topk(min(k * 2, len(sims)))

                seen_cids = {m["class_id"] for m in matches}
                for sim_val, cid_val in zip(
                    top_sims.cpu().tolist(), top_ids.cpu().tolist()
                ):
                    cid = int(cid_val)
                    if cid not in seen_cids:
                        matches.append(
                            {
                                "class_id": cid,
                                "gloss": self.id_to_class.get(cid, str(cid)),
                                "similarity": float(sim_val),
                                "exact_match": False,
                            }
                        )
                        seen_cids.add(cid)
                    if len(matches) >= k:
                        break
            except Exception:
                pass

        # 3. Substring matching fallback
        if len(matches) < k:
            seen_cids = {m["class_id"] for m in matches}
            for cid, gloss in sorted(self.id_to_class.items()):
                if cid not in seen_cids and query_str in gloss.lower():
                    matches.append(
                        {
                            "class_id": cid,
                            "gloss": gloss,
                            "similarity": 0.8,
                            "exact_match": False,
                        }
                    )
                    seen_cids.add(cid)
                if len(matches) >= k:
                    break

        # 4. Fill remaining slots if needed
        if len(matches) < k:
            seen_cids = {m["class_id"] for m in matches}
            for cid, gloss in sorted(self.id_to_class.items()):
                if cid not in seen_cids:
                    matches.append(
                        {
                            "class_id": cid,
                            "gloss": gloss,
                            "similarity": 0.0,
                            "exact_match": False,
                        }
                    )
                    seen_cids.add(cid)
                if len(matches) >= k:
                    break

        latency_ms = round((time.perf_counter() - t0) * 1000.0, 2)
        top_matches = matches[:k]
        return {
            "query": query_str,
            "matched_gloss": top_matches[0]["gloss"],
            "matched_class_id": top_matches[0]["class_id"],
            "confidence": top_matches[0]["similarity"],
            "top_matches": top_matches,
            "latency_ms": latency_ms,
        }


class ASLEnsembleService:
    """Ensemble service combining predictions across multiple model inductive biases."""

    def __init__(
        self,
        services: dict[str, ASLInferenceService],
        *,
        weights: dict[str, float] | None = None,
        id_to_class: dict[int, str] | None = None,
        pose_extractor: PoseExtractor | None = None,
    ) -> None:
        if not services:
            raise ValueError("ASLEnsembleService requires at least one member service.")

        self.services = services
        total_w = sum(weights.values()) if weights else float(len(services))
        self.weights = (
            {k: float(v) / total_w for k, v in weights.items()}
            if weights
            else {k: 1.0 / len(services) for k in services}
        )
        first_svc = next(iter(services.values()))
        self.id_to_class = id_to_class or first_svc.id_to_class
        self.device = first_svc.device
        self.model_type = "ensemble_equal_three"
        self._pose_extractor = pose_extractor
        self.num_frames = first_svc.num_frames

    @property
    def pose_extractor(self) -> PoseExtractor:
        if self._pose_extractor is None:
            self._pose_extractor = PoseExtractor(device="cpu")
        return self._pose_extractor

    @classmethod
    def from_bundle(
        cls,
        bundle_dir: str | Path,
        *,
        device: str | torch.device | None = None,
    ) -> "ASLEnsembleService":
        """Reconstruct the winning ensemble from a saved production model bundle directory."""
        import json

        b_dir = Path(bundle_dir)
        if not b_dir.is_dir():
            raise FileNotFoundError(
                f"Production model bundle directory not found: {b_dir}"
            )

        cfg_path = b_dir / "ensemble_config.json"
        meta_path = b_dir / "production_metadata.json"

        weights: dict[str, float] = {}
        if cfg_path.is_file():
            cfg_data = json.loads(cfg_path.read_text(encoding="utf-8"))
            weights = {str(k): float(v) for k, v in cfg_data.get("weights", {}).items()}

        id_to_class: dict[int, str] | None = None
        if meta_path.is_file():
            meta_data = json.loads(meta_path.read_text(encoding="utf-8"))
            if "id_to_class" in meta_data:
                id_to_class = {
                    int(k): str(v) for k, v in meta_data["id_to_class"].items()
                }

        services: dict[str, ASLInferenceService] = {}
        for ckpt_file in sorted(b_dir.glob("*.pt")):
            name = ckpt_file.stem
            try:
                services[name] = ASLInferenceService.from_checkpoint(
                    ckpt_file,
                    device=device,
                )
            except Exception:
                continue

        if not services:
            raise RuntimeError(
                f"No valid checkpoints could be loaded from bundle: {b_dir}"
            )

        return cls(services=services, weights=weights, id_to_class=id_to_class)

    @torch.inference_mode()
    def predict_tensor(
        self,
        clip_tensor: torch.Tensor,
        *,
        top_k: int = 5,
    ) -> dict[str, Any]:
        """Combine probability distributions across all ensemble members."""
        t0 = time.perf_counter()
        ens_probs: torch.Tensor | None = None

        for name, srv in self.services.items():
            w = self.weights.get(name, 1.0 / len(self.services))
            x = clip_tensor.to(srv.device)
            if srv.model_type.startswith("pose_"):
                if (
                    "language" in srv.model_type
                    and srv.vocab_text_embeddings is not None
                ):
                    out = srv.model(x, vocab_text_embeddings=srv.vocab_text_embeddings)
                else:
                    out = srv.model(x)
                logits = out["logits"]
            else:
                logits = srv.model(x)

            probs = torch.softmax(logits[0], dim=-1)
            ens_probs = probs * w if ens_probs is None else ens_probs + (probs * w)

        assert ens_probs is not None
        k = min(max(1, int(top_k)), ens_probs.numel())
        top_probs, top_ids = ens_probs.topk(k)
        predictions = []
        for prob, cid in zip(top_probs.cpu().tolist(), top_ids.cpu().tolist()):
            predictions.append(
                {
                    "class_id": int(cid),
                    "gloss": self.id_to_class.get(int(cid), str(cid)),
                    "confidence": float(prob),
                    "score": float(prob),
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

    def predict_pose(
        self,
        pose_data: Any,
        *,
        top_k: int = 5,
    ) -> dict[str, Any]:
        """Direct inference on (T, 349) pose features through the ensemble."""
        import numpy as np

        if isinstance(pose_data, list):
            arr = np.asarray(pose_data, dtype=np.float32)
            tensor = torch.from_numpy(arr).float()
        elif isinstance(pose_data, np.ndarray):
            tensor = torch.from_numpy(pose_data).float()
        elif isinstance(pose_data, torch.Tensor):
            tensor = pose_data.float()
        else:
            raise TypeError(f"Unsupported pose data type: {type(pose_data)}")

        if tensor.ndim == 2:
            tensor = tensor.unsqueeze(0)
        return self.predict_tensor(tensor, top_k=top_k)

    def predict_frames(
        self,
        frames: list[Any] | Any,
        *,
        top_k: int = 5,
    ) -> dict[str, Any]:
        """Extract pose and run ensemble prediction from raw image frames."""
        pose_seq = self.pose_extractor.extract_from_frames(frames)
        clip_tensor = torch.from_numpy(pose_seq).float().unsqueeze(0)
        return self.predict_tensor(clip_tensor, top_k=top_k)

    def predict_video(
        self,
        video_path: str | Path,
        *,
        frame_start: Any = 1,
        frame_end: Any = -1,
        metadata_fps: float | None = 25.0,
        top_k: int = 5,
    ) -> dict[str, Any]:
        """Extract pose and run ensemble prediction from a video file."""
        pose_seq = self.pose_extractor.extract_from_video(
            video_path=video_path,
            num_frames=self.num_frames,
            frame_start=frame_start,
            frame_end=frame_end,
            metadata_fps=metadata_fps,
        )
        clip_tensor = torch.from_numpy(pose_seq).float().unsqueeze(0)
        return self.predict_tensor(clip_tensor, top_k=top_k)

    def retrieve_sign_from_text(
        self,
        query: str,
        *,
        top_k: int = 5,
    ) -> dict[str, Any]:
        """Delegate semantic sign retrieval to multimodal member service."""
        for srv in self.services.values():
            if "language" in srv.model_type and srv.vocab_text_embeddings is not None:
                return srv.retrieve_sign_from_text(query, top_k=top_k)
        return next(iter(self.services.values())).retrieve_sign_from_text(
            query, top_k=top_k
        )

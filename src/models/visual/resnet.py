"""Pretrained ResNet50 spatial feature extractor with controlled unfreezing."""

import torch
import torch.nn as nn
from torchvision.models import ResNet50_Weights, resnet50


class ResNet50Backbone(nn.Module):
    """Frame-level ResNet50 feature extractor producing 2048-D embeddings per frame."""

    def __init__(
        self,
        *,
        pretrained: bool = True,
        frozen: bool = True,
        unfreeze_last_blocks: int = 0,
        frame_chunk_size: int = 32,
        freeze_batchnorm: bool = True,
    ) -> None:
        super().__init__()
        weights = ResNet50_Weights.DEFAULT if pretrained else None
        backbone = resnet50(weights=weights)
        # Keep children up to AdaptiveAvgPool2d -> (N, 2048, 1, 1)
        self.stem = nn.Sequential(
            backbone.conv1,
            backbone.bn1,
            backbone.relu,
            backbone.maxpool,
        )
        self.layer1 = backbone.layer1
        self.layer2 = backbone.layer2
        self.layer3 = backbone.layer3
        self.layer4 = backbone.layer4
        self.avgpool = backbone.avgpool
        self.feature_dim = 2048
        self.frame_chunk_size = int(frame_chunk_size)
        self.freeze_batchnorm = bool(freeze_batchnorm)

        self.set_trainable_blocks(frozen=frozen, unfreeze_last_blocks=unfreeze_last_blocks)

    def set_trainable_blocks(
        self,
        *,
        frozen: bool = True,
        unfreeze_last_blocks: int = 0,
    ) -> None:
        """Freeze all layers or unfreeze the final `unfreeze_last_blocks` ResNet stages."""
        for param in self.parameters():
            param.requires_grad = False

        if frozen and unfreeze_last_blocks <= 0:
            self.eval()
            return

        stages = [self.layer4, self.layer3, self.layer2, self.layer1, self.stem]
        n_unfreeze = len(stages) if not frozen and unfreeze_last_blocks <= 0 else unfreeze_last_blocks
        for stage in stages[: max(0, min(n_unfreeze, len(stages)))]:
            for param in stage.parameters():
                param.requires_grad = True

    def train(self, mode: bool = True):
        super().train(mode)
        if self.freeze_batchnorm:
            for module in self.modules():
                if isinstance(module, nn.BatchNorm2d):
                    module.eval()
        return self

    def _forward_frames(self, x: torch.Tensor) -> torch.Tensor:
        x = self.stem(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x)
        return x.flatten(1)

    def forward(self, videos: torch.Tensor) -> torch.Tensor:
        """Encode (B, T, C, H, W) video tensor into (B, T, 2048) features."""
        if videos.ndim != 5:
            raise ValueError(f"Expected 5D tensor (B, T, C, H, W), got {tuple(videos.shape)}")
        b, t, c, h, w = videos.shape
        frames = videos.reshape(b * t, c, h, w)

        has_grad = any(p.requires_grad for p in self.parameters()) and torch.is_grad_enabled()
        chunks = []
        ctx = torch.enable_grad() if has_grad else torch.no_grad()
        with ctx:
            for start in range(0, len(frames), self.frame_chunk_size):
                chunk = frames[start : start + self.frame_chunk_size]
                chunks.append(self._forward_frames(chunk))

        feats = torch.cat(chunks, dim=0)
        return feats.reshape(b, t, self.feature_dim)

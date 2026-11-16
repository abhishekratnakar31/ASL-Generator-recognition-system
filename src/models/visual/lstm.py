"""Temporal sequence encoders (LSTM and lightweight Transformer) for ASL features."""

import math

import torch
import torch.nn as nn


class TemporalAttentionPool(nn.Module):
    """Additive learned attention pooling over temporal sequence (B, T, D) -> (B, D)."""

    def __init__(self, dim: int) -> None:
        super().__init__()
        self.proj = nn.Linear(dim, 1)

    def forward(self, sequence: torch.Tensor) -> torch.Tensor:
        weights = torch.softmax(self.proj(sequence), dim=1)
        return (sequence * weights).sum(dim=1)


class TemporalLSTM(nn.Module):
    """Multi-layer (optionally bidirectional) LSTM temporal encoder."""

    def __init__(
        self,
        input_dim: int = 2048,
        hidden_dim: int = 256,
        num_layers: int = 2,
        dropout: float = 0.3,
        bidirectional: bool = False,
        pooling: str = "last",
    ) -> None:
        super().__init__()
        if pooling not in {"last", "mean", "attention"}:
            raise ValueError(
                f"Unsupported pooling='{pooling}'; choose last, mean, or attention."
            )

        self.input_proj = nn.Sequential(
            nn.LayerNorm(input_dim),
            nn.Dropout(dropout),
        )
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
            bidirectional=bidirectional,
        )
        self.output_dim = hidden_dim * (2 if bidirectional else 1)
        self.pooling = pooling
        self.attn_pool = (
            TemporalAttentionPool(self.output_dim) if pooling == "attention" else None
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        if features.ndim != 3:
            raise ValueError(
                f"Expected 3D features (B, T, D), got {tuple(features.shape)}"
            )
        x = self.input_proj(features)
        sequence, (hidden, _) = self.lstm(x)
        if self.pooling == "mean":
            return sequence.mean(dim=1)
        if self.pooling == "attention" and self.attn_pool is not None:
            return self.attn_pool(sequence)
        if self.lstm.bidirectional:
            return torch.cat([hidden[-2], hidden[-1]], dim=-1)
        return hidden[-1]


class TemporalTransformerEncoder(nn.Module):
    """Lightweight Transformer encoder with sinusoidal positional encoding and pooling."""

    pos_encoding: torch.Tensor

    def __init__(
        self,
        input_dim: int = 2048,
        embed_dim: int = 256,
        num_layers: int = 2,
        num_heads: int = 4,
        dim_feedforward: int = 512,
        dropout: float = 0.3,
        max_seq_len: int = 128,
        pooling: str = "attention",
    ) -> None:
        super().__init__()
        if pooling not in {"mean", "attention"}:
            raise ValueError(
                f"Unsupported pooling='{pooling}'; choose mean or attention."
            )

        self.input_proj = nn.Sequential(
            nn.LayerNorm(input_dim),
            nn.Linear(input_dim, embed_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        pe = torch.zeros(1, max_seq_len, embed_dim)
        pos = torch.arange(0, max_seq_len, dtype=torch.float32).unsqueeze(1)
        div = torch.exp(
            torch.arange(0, embed_dim, 2, dtype=torch.float32)
            * (-math.log(10000.0) / embed_dim)
        )
        pe[0, :, 0::2] = torch.sin(pos * div)
        pe[0, :, 1::2] = torch.cos(pos * div)
        self.register_buffer("pos_encoding", pe, persistent=False)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_layers,
            enable_nested_tensor=False,
        )
        self.norm = nn.LayerNorm(embed_dim)
        self.output_dim = embed_dim
        self.pooling = pooling
        self.attn_pool = (
            TemporalAttentionPool(embed_dim) if pooling == "attention" else None
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        if features.ndim != 3:
            raise ValueError(
                f"Expected 3D features (B, T, D), got {tuple(features.shape)}"
            )
        t_len = features.size(1)
        x = self.input_proj(features) + self.pos_encoding[:, :t_len, :]
        encoded = self.norm(self.encoder(x))
        if self.pooling == "attention" and self.attn_pool is not None:
            return self.attn_pool(encoded)
        return encoded.mean(dim=1)

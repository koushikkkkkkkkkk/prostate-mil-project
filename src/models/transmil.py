"""Simplified TransMIL -- Person 3.

Reference: Shao et al., "TransMIL: Transformer based Correlated Multiple
Instance Learning for Whole Slide Image Classification", NeurIPS 2021.
Implemented from scratch (no official code copied); "simplified" because we use
a small number of standard transformer blocks instead of the full
Nystromformer / PPEG stack described in the paper.

Pipeline
--------
    x_n          -> Linear(D, d_model)                     token sequence
    [CLS] token  -> prepended (learnable)
    PPEG         -> depthwise 3x3 conv over a sqrt(N) x sqrt(N) pseudo-image
    Transformer  -> L pre-norm blocks with multi-head self-attention
    CLS token    -> Linear(d_model, 6)

Heatmaps use the last block's CLS -> patch attention row, averaged over heads.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from typing import Any, Dict, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

from .base import MILModelBase


@dataclass
class TransMILConfig:
    in_dim: int = 1024
    d_model: int = 384
    n_heads: int = 6
    n_layers: int = 2
    ffn_multiplier: int = 4
    n_classes: int = 6
    dropout: float = 0.1

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, values: Dict[str, Any]) -> "TransMILConfig":
        allowed = set(cls.__dataclass_fields__)  # type: ignore[attr-defined]
        unknown = sorted(set(values) - allowed)
        if unknown:
            raise ValueError(f"Unknown TransMILConfig keys: {unknown}")
        return cls(**values)


class PPEG(nn.Module):
    """Pyramid Position Encoding Generator (depthwise conv on a 2-D token grid)."""

    def __init__(self, dim: int, kernel_size: int = 3) -> None:
        super().__init__()
        self.proj = nn.Conv2d(dim, dim, kernel_size, padding=kernel_size // 2, groups=dim)
        self.norm = nn.LayerNorm(dim)

    def forward(self, tokens: torch.Tensor, grid: Tuple[int, int]) -> torch.Tensor:
        """``tokens``: [B, N, C] (N == grid[0] * grid[1])."""
        batch, n_tokens, dim = tokens.shape
        rows, cols = grid
        if rows * cols != n_tokens:
            raise ValueError(f"grid {grid} does not match {n_tokens} tokens")
        image = tokens.transpose(1, 2).reshape(batch, dim, rows, cols)
        image = self.proj(image)
        return self.norm((tokens + image.reshape(batch, dim, n_tokens).transpose(1, 2)))


class SelfAttentionBlock(nn.Module):
    """Pre-norm transformer block that can return its attention matrix."""

    def __init__(self, dim: int, n_heads: int, ffn_multiplier: int, dropout: float) -> None:
        super().__init__()
        if dim % n_heads != 0:
            raise ValueError(f"d_model ({dim}) must be divisible by n_heads ({n_heads})")
        self.n_heads = n_heads
        self.head_dim = dim // n_heads
        self.norm1 = nn.LayerNorm(dim)
        self.qkv = nn.Linear(dim, 3 * dim)
        self.proj = nn.Linear(dim, dim)
        self.norm2 = nn.LayerNorm(dim)
        self.ffn = nn.Sequential(
            nn.Linear(dim, dim * ffn_multiplier),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim * ffn_multiplier, dim),
        )
        self.dropout = nn.Dropout(dropout)

    def forward(
        self, x: torch.Tensor, key_padding_mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        batch, n_tokens, dim = x.shape
        qkv = self.qkv(self.norm1(x)).reshape(batch, n_tokens, 3, self.n_heads, self.head_dim)
        q, k, v = qkv.permute(2, 0, 3, 1, 4)          # [B, H, N, hd] each
        scores = (q @ k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        if key_padding_mask is not None:
            scores = scores.masked_fill(key_padding_mask[:, None, None, :], float("-inf"))
        weights = F.softmax(scores, dim=-1)
        context = (weights @ v).transpose(1, 2).reshape(batch, n_tokens, dim)
        x = x + self.dropout(self.proj(context))
        x = x + self.dropout(self.ffn(self.norm2(x)))
        return x, weights


class TransMIL(MILModelBase):
    """Simplified TransMIL with PPEG positional encoding."""

    def __init__(
        self, config: Union[TransMILConfig, Dict[str, Any], None] = None, **overrides: Any
    ) -> None:
        super().__init__()
        if config is None:
            config = TransMILConfig(**overrides)
        elif isinstance(config, dict):
            config = TransMILConfig.from_dict({**config, **overrides})
        elif overrides:
            raise ValueError("Pass either a config object or keyword overrides, not both")

        self.config = config
        self.n_classes = config.n_classes
        self.embed_dim = config.d_model

        self.input_proj = nn.Sequential(
            nn.Linear(config.in_dim, config.d_model), nn.Dropout(config.dropout)
        )
        self.cls_token = nn.Parameter(torch.zeros(1, 1, config.d_model))
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        self.ppeg = PPEG(config.d_model)
        self.blocks = nn.ModuleList(
            [
                SelfAttentionBlock(config.d_model, config.n_heads, config.ffn_multiplier, config.dropout)
                for _ in range(config.n_layers)
            ]
        )
        self.norm = nn.LayerNorm(config.d_model)
        self.classifier = nn.Linear(config.d_model, config.n_classes)

    @staticmethod
    def _grid_size(n_patches: int) -> Tuple[int, int]:
        side = max(2, int(math.ceil(math.sqrt(max(n_patches, 1)))))
        return side, side

    def forward(
        self,
        features: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,  # interface parity
        return_patch_embeddings: bool = False,
        **_: Any,
    ) -> Dict[str, Any]:
        x, valid_mask, was_2d = self._prepare(features, mask)
        batch, n_patches, _ = x.shape

        tokens = self.input_proj(x)
        rows, cols = self._grid_size(n_patches)
        grid_size = rows * cols

        padding = grid_size - n_patches
        if padding:
            tokens = torch.cat([tokens, tokens.new_zeros(batch, padding, tokens.size(-1))], dim=1)
        tokens = self.ppeg(tokens, (rows, cols))
        seq = torch.cat([self.cls_token.expand(batch, 1, -1), tokens], dim=1)

        keep = torch.ones(batch, grid_size + 1, dtype=torch.bool, device=x.device)
        keep[:, 1 + n_patches :] = False        # padded grid cells never attend
        if valid_mask is not None:
            keep[:, 1 : 1 + n_patches] = valid_mask

        key_padding_mask = ~keep
        weights = None
        for block in self.blocks:
            seq, weights = block(seq, key_padding_mask)

        seq = self.norm(seq)
        bag_feat = seq[:, 0]
        logits = self.classifier(bag_feat)

        # CLS -> patch attention of the last block, averaged over heads.
        cls_attention = weights[:, :, 0, 1 : 1 + n_patches].mean(dim=1)   # [B, N]
        if valid_mask is not None:
            cls_attention = cls_attention.masked_fill(~valid_mask, 0.0)
        attention = cls_attention / cls_attention.sum(dim=1, keepdim=True).clamp_min(1e-8)

        result: Dict[str, Any] = {
            "logits": logits,
            "attention": attention,
            "raw_attention": cls_attention,
            "bag_feat": bag_feat,
        }
        if return_patch_embeddings:
            result["patch_embeddings"] = seq[:, 1 : 1 + n_patches]
        return self._finalise(result, was_2d)


def build_transmil(
    config: Optional[Union[TransMILConfig, Dict[str, Any]]] = None, **overrides: Any
) -> TransMIL:
    return TransMIL(config=config, **overrides)

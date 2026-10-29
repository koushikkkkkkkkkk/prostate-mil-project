"""Gated-Attention ABMIL baseline -- Person 1.

Reference: Ilse, Tomczak & Welling, "Attention-based Deep Multiple Instance
Learning", ICML 2018.  Implemented from scratch in PyTorch.

    h_n   = Dropout(ReLU(V0 x_n))                       [N, L]
    a_n   = softmax_n( w^T ( tanh(V h_n) * sigmoid(U h_n) ) )
    z     = sum_n a_n h_n                               [L]
    logits = W_c z + b_c                                [6]

Implemented by Person 1, but kept importable and testable here because
CLAM-SB's attention branch and the ``without_instance`` ablation are compared
directly against this baseline.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Dict, Optional, Union

import torch
import torch.nn as nn

from .base import MILModelBase, masked_softmax


@dataclass
class ABMILConfig:
    in_dim: int = 1024
    embed_dim: int = 512
    attention_dim: int = 256
    n_classes: int = 6
    dropout: float = 0.25
    gated: bool = True

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, values: Dict[str, Any]) -> "ABMILConfig":
        allowed = set(cls.__dataclass_fields__)  # type: ignore[attr-defined]
        unknown = sorted(set(values) - allowed)
        if unknown:
            raise ValueError(f"Unknown ABMILConfig keys: {unknown}")
        return cls(**values)


class GatedAttention(nn.Module):
    """tanh/sigmoid gated attention scorer (non-gated variant optional)."""

    def __init__(self, in_dim: int, attention_dim: int, gated: bool = True, dropout: float = 0.0):
        super().__init__()
        self.gated = gated
        self.V = nn.Linear(in_dim, attention_dim)
        if gated:
            self.U = nn.Linear(in_dim, attention_dim)
        self.w = nn.Linear(attention_dim, 1)
        self.dropout = nn.Dropout(dropout)

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        """``h``: [B, N, L] -> raw scores [B, N]."""
        v = torch.tanh(self.V(h))
        if self.gated:
            v = v * torch.sigmoid(self.U(h))
        return self.w(self.dropout(v)).squeeze(-1)


class ABMIL(MILModelBase):
    """Gated attention MIL for 6-class ordinal ISUP grading."""

    def __init__(self, config: Union[ABMILConfig, Dict[str, Any], None] = None, **overrides: Any):
        super().__init__()
        if config is None:
            config = ABMILConfig(**overrides)
        elif isinstance(config, dict):
            config = ABMILConfig.from_dict({**config, **overrides})
        elif overrides:
            raise ValueError("Pass either a config object or keyword overrides, not both")

        self.config = config
        self.n_classes = config.n_classes
        self.embed_dim = config.embed_dim

        self.encoder = nn.Sequential(
            nn.Linear(config.in_dim, config.embed_dim),
            nn.ReLU(),
            nn.Dropout(config.dropout),
        )
        self.attention = GatedAttention(
            config.embed_dim, config.attention_dim, config.gated, config.dropout
        )
        self.classifier = nn.Linear(config.embed_dim, config.n_classes)

    def forward(
        self,
        features: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,  # accepted for interface parity
        return_patch_embeddings: bool = False,
        **_: Any,
    ) -> Dict[str, Any]:
        x, valid_mask, was_2d = self._prepare(features, mask)
        h = self.encoder(x)                       # [B, N, L]
        scores = self.attention(h)                # [B, N]
        attention = masked_softmax(scores, valid_mask, dim=1)
        bag_feat = torch.einsum("bn,bne->be", attention, h)
        logits = self.classifier(bag_feat)

        result: Dict[str, Any] = {
            "logits": logits,
            "attention": attention,
            "raw_attention": scores,
            "bag_feat": bag_feat,
        }
        if return_patch_embeddings:
            result["patch_embeddings"] = h
        return self._finalise(result, was_2d)


def build_abmil(config: Optional[Union[ABMILConfig, Dict[str, Any]]] = None, **overrides: Any) -> ABMIL:
    return ABMIL(config=config, **overrides)

"""Shared MIL interfaces and tensor helpers.

Every model in ``src/models`` implements the same tiny contract so that the
shared trainer, evaluator, heatmap renderer and Streamlit app work with any of
them:

    model(features, mask=None, labels=None) -> dict with keys
        "logits"    [B, n_classes]  raw class scores (no softmax)
        "attention" [B, N]          attention weights summing to 1 per bag
        "bag_feat"  [B, embed_dim]  optional pooled slide embedding
        "instance_loss" scalar      optional (CLAM-SB only)

``features`` may be ``[N, D]`` (single slide) or ``[B, N, D]`` (batch); models
accept both and mirror the input rank in their outputs.

Owner: shared (Person 1 maintains, everyone consumes).
"""

from __future__ import annotations

from typing import Any, Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


def ensure_3d(
    features: torch.Tensor, mask: Optional[torch.Tensor] = None
) -> Tuple[torch.Tensor, Optional[torch.Tensor], bool]:
    """Normalise ``[N, D]`` / ``[B, N, D]`` input and its optional mask.

    Returns ``(features[B, N, D], mask[B, N] or None, was_2d)``.
    """
    was_2d = features.ndim == 2
    if was_2d:
        features = features.unsqueeze(0)
    elif features.ndim != 3:
        raise ValueError(f"features must be [N, D] or [B, N, D], got {tuple(features.shape)}")

    valid_mask = mask
    if valid_mask is not None:
        if not torch.is_tensor(valid_mask):
            valid_mask = torch.as_tensor(valid_mask)
        if valid_mask.dtype != torch.bool:
            valid_mask = valid_mask.bool()
        if valid_mask.ndim == 1:
            valid_mask = valid_mask.unsqueeze(0)
        if valid_mask.shape != features.shape[:2]:
            raise ValueError(
                f"mask shape {tuple(valid_mask.shape)} does not match "
                f"features {tuple(features.shape[:2])}"
            )
        if not bool(valid_mask.any(dim=1).all()):
            raise ValueError("Every bag must contain at least one valid patch")
        valid_mask = valid_mask.to(features.device)
    return features, valid_mask, was_2d


def masked_softmax(
    scores: torch.Tensor, mask: Optional[torch.Tensor], dim: int = -1
) -> torch.Tensor:
    """Softmax that gives padded positions exactly zero weight."""
    if mask is None:
        return F.softmax(scores, dim=dim)
    scores = scores.masked_fill(~mask, float("-inf"))
    weights = F.softmax(scores, dim=dim)
    return weights * mask.to(weights.dtype)


def masked_mean(values: torch.Tensor, mask: Optional[torch.Tensor], dim: int = 1) -> torch.Tensor:
    if mask is None:
        return values.mean(dim=dim)
    weights = mask.to(values.dtype)
    return (values * weights).sum(dim=dim) / weights.sum(dim=dim).clamp_min(1.0)


class MILModelBase(nn.Module):
    """Common plumbing: input normalisation, output squeezing, label parsing."""

    n_classes: int = 6
    embed_dim: int = 512

    def _prepare(
        self, features: torch.Tensor, mask: Optional[torch.Tensor]
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], bool]:
        return ensure_3d(features, mask)

    @staticmethod
    def _parse_labels(
        labels: Any, batch_size: int, device: torch.device
    ) -> Optional[torch.Tensor]:
        if labels is None:
            return None
        if isinstance(labels, int):
            return torch.full((batch_size,), int(labels), dtype=torch.long, device=device)
        tensor = torch.as_tensor(labels, device=device).reshape(-1).long()
        if tensor.numel() != batch_size:
            raise ValueError(
                f"Expected {batch_size} labels, received {tensor.numel()}"
            )
        return tensor

    @staticmethod
    def _finalise(result: Dict[str, Any], was_2d: bool) -> Dict[str, Any]:
        if not was_2d:
            return result
        for key, value in list(result.items()):
            if torch.is_tensor(value) and value.ndim >= 1 and value.size(0) == 1:
                result[key] = value.squeeze(0) if value.ndim > 0 else value
        return result


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

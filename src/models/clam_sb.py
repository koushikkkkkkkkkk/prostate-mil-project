"""CLAM-SB (single-branch, clustering-constrained attention MIL) -- Person 2.

Reference: Lu et al., "Data-efficient and weakly supervised computational
pathology on whole-slide images", Nature Biomedical Engineering 5, 555-570
(2021).  Implemented **from scratch** in PyTorch from the paper's equations;
the official ``mahmoodlab/CLAM`` repository was read for conceptual reference
only and no code was copied.

Architecture
------------
    h_n    = Dropout(ReLU(W_e x_n))                     [N, D] -> [N, 512]
    a_n    = softmax_n( w^T ( tanh(V h_n) * sigmoid(U h_n) ) )
    z      = sum_n a_n h_n                              [512]
    logits = W_c z                                      [6]

Instance-level clustering (the part that makes it CLAM and not ABMIL)
---------------------------------------------------------------------
For a slide with true grade ``k`` the ``B`` highest-attention patches are
pseudo-labelled **positive** and the ``B`` lowest-attention patches are
pseudo-labelled **negative**.  A class-specific binary head
``instance_classifiers[k]`` must separate those 2B instances:

    L_total = L_slide(CE or CE-ordinal) + lambda * L_instance

``L_instance`` defaults to the smooth top-1 SVM of the paper
(``alpha = tau = 1``), which is re-implemented here rather than imported.

Setting ``instance_loss_weight = 0`` (or ``--no-instance-loss`` in the
drivers) gives the exact ABMIL-style pooling ablation used in the report:
``CLAM-SB`` vs ``CLAM-SB without instance-level supervision``.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

from .base import MILModelBase, masked_softmax


@dataclass
class CLAMSBConfig:
    in_dim: int = 1024
    embed_dim: int = 512
    attention_dim: int = 256
    n_classes: int = 6
    dropout: float = 0.25
    gated: bool = True
    k_sample: int = 8
    instance_loss: str = "smooth_svm"  # "smooth_svm" | "ce"
    instance_loss_weight: float = 0.30
    subtyping: bool = False
    attention_temperature: float = 1.0
    smooth_svm_alpha: float = 1.0
    smooth_svm_tau: float = 1.0

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, values: Dict[str, Any]) -> "CLAMSBConfig":
        allowed = set(cls.__dataclass_fields__)  # type: ignore[attr-defined]
        unknown = sorted(set(values) - allowed)
        if unknown:
            raise ValueError(f"Unknown CLAMSBConfig keys: {unknown}")
        return cls(**values)


class GatedAttentionScorer(nn.Module):
    """Single-branch gated attention: ``w^T (tanh(Vh) * sigmoid(Uh))``."""

    def __init__(self, in_dim: int, hidden_dim: int, dropout: float = 0.0) -> None:
        super().__init__()
        self.tanh_gate = nn.Sequential(nn.Linear(in_dim, hidden_dim), nn.Tanh(), nn.Dropout(dropout))
        self.sigmoid_gate = nn.Sequential(nn.Dropout(dropout), nn.Linear(in_dim, hidden_dim), nn.Sigmoid())
        self.scorer = nn.Linear(hidden_dim, 1)

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        return self.scorer(self.tanh_gate(h) * self.sigmoid_gate(h))


class PlainAttentionScorer(nn.Module):
    """Non-gated attention variant (ablation only)."""

    def __init__(self, in_dim: int, hidden_dim: int, dropout: float = 0.0) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.Tanh(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        return self.net(h)


class SmoothTop1SVMLoss(nn.Module):
    """Smooth top-1 SVM surrogate for CLAM instance clustering.

    ``tau * softplus(logsumexp_{j != y}((s_j - s_y + alpha) / tau))``

    Written from scratch (not the external ``smooth-topk`` package). With two
    classes it reduces to ``tau * softplus((s_other - s_y + alpha) / tau)``.
    """

    def __init__(self, alpha: float = 1.0, tau: float = 1.0, reduction: str = "mean") -> None:
        super().__init__()
        if alpha < 0:
            raise ValueError("alpha must be non-negative")
        if tau <= 0:
            raise ValueError("tau must be positive")
        if reduction not in {"mean", "sum", "none"}:
            raise ValueError("reduction must be 'mean', 'sum' or 'none'")
        self.alpha = float(alpha)
        self.tau = float(tau)
        self.reduction = reduction

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        if logits.ndim != 2:
            raise ValueError(f"logits must be [M, C], got {tuple(logits.shape)}")
        targets = targets.reshape(-1).long()
        if targets.shape[0] != logits.shape[0]:
            raise ValueError("targets and logits batch sizes differ")

        target_scores = logits.gather(1, targets[:, None])
        margins = (logits - target_scores + self.alpha) / self.tau
        margins = margins.scatter(1, targets[:, None], float("-inf"))
        if margins.size(1) == 1:  # binary edge case: only the true class
            loss = logits.new_zeros(logits.size(0))
        else:
            loss = self.tau * F.softplus(torch.logsumexp(margins, dim=1))
        if self.reduction == "mean":
            return loss.mean()
        if self.reduction == "sum":
            return loss.sum()
        return loss


class CLAMSB(MILModelBase):
    """CLAM-SB with optional instance-level clustering loss."""

    def __init__(self, config: Union[CLAMSBConfig, Dict[str, Any], None] = None, **overrides: Any):
        super().__init__()
        if config is None:
            config = CLAMSBConfig(**overrides)
        elif isinstance(config, dict):
            config = CLAMSBConfig.from_dict({**config, **overrides})
        elif overrides:
            raise ValueError("Pass either a config object or keyword overrides, not both")
        if config.n_classes < 2:
            raise ValueError("n_classes must be >= 2")
        if config.k_sample < 1:
            raise ValueError("k_sample must be >= 1")

        self.config = config
        self.n_classes = config.n_classes
        self.embed_dim = config.embed_dim

        self.feature_encoder = nn.Sequential(
            nn.Linear(config.in_dim, config.embed_dim),
            nn.ReLU(),
            nn.Dropout(config.dropout),
        )
        scorer_cls = GatedAttentionScorer if config.gated else PlainAttentionScorer
        self.attention_scorer = scorer_cls(config.embed_dim, config.attention_dim, dropout=config.dropout)
        self.slide_classifier = nn.Linear(config.embed_dim, config.n_classes)
        # One binary head per grade: "is this patch evidence for grade k?".
        self.instance_classifiers = nn.ModuleList(
            [nn.Linear(config.embed_dim, 2) for _ in range(config.n_classes)]
        )

        if config.instance_loss == "smooth_svm":
            self.instance_criterion: nn.Module = SmoothTop1SVMLoss(
                alpha=config.smooth_svm_alpha, tau=config.smooth_svm_tau
            )
        elif config.instance_loss == "ce":
            self.instance_criterion = nn.CrossEntropyLoss()
        else:
            raise ValueError("instance_loss must be 'smooth_svm' or 'ce'")

    @property
    def instance_loss_weight(self) -> float:
        return float(self.config.instance_loss_weight)

    # ------------------------------------------------------------------ #
    # Internals
    # ------------------------------------------------------------------ #
    def _compute_attention(
        self, h: torch.Tensor, valid_mask: Optional[torch.Tensor]
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        raw = self.attention_scorer(h).squeeze(-1)          # [B, N]
        scaled = raw / self.config.attention_temperature
        return masked_softmax(scaled, valid_mask, dim=1), raw

    @staticmethod
    def _extreme_indices(
        attention: torch.Tensor, valid_mask: Optional[torch.Tensor], k_sample: int
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Top-``k`` and bottom-``k`` attention patch indices (detached keys)."""
        a = attention.detach()
        _, n_patches = a.shape
        if n_patches < 2:
            raise ValueError("A bag needs at least two patches for instance clustering")
        k = min(k_sample, max(1, n_patches // 2))
        if valid_mask is None:
            high_key, low_key = a, a
        else:
            high_key = a.masked_fill(~valid_mask, float("-inf"))
            low_key = a.masked_fill(~valid_mask, float("inf"))
        top_idx = torch.topk(high_key, k=k, dim=1, largest=True).indices
        bottom_idx = torch.topk(low_key, k=k, dim=1, largest=False).indices
        return top_idx, bottom_idx

    @staticmethod
    def _gather_by_index(h: torch.Tensor, idx: torch.Tensor) -> torch.Tensor:
        expand = idx.unsqueeze(-1).expand(-1, -1, h.size(-1))
        return torch.gather(h, 1, expand)

    def _instance_clustering_loss(
        self,
        attention: torch.Tensor,
        h: torch.Tensor,
        labels: torch.Tensor,
        valid_mask: Optional[torch.Tensor],
    ) -> Dict[str, torch.Tensor]:
        top_idx, bottom_idx = self._extreme_indices(attention, valid_mask, self.config.k_sample)
        top_h = self._gather_by_index(h, top_idx)        # [B, k, E]
        bottom_h = self._gather_by_index(h, bottom_idx)  # [B, k, E]

        batch_size, k = top_idx.shape
        device = h.device
        losses: List[torch.Tensor] = []
        all_logits: List[torch.Tensor] = []
        all_targets: List[torch.Tensor] = []

        for b in range(batch_size):
            y = int(labels[b].item())
            if not 0 <= y < self.config.n_classes:
                raise ValueError(f"Label {y} outside [0, {self.config.n_classes - 1}]")

            instances = torch.cat([top_h[b], bottom_h[b]], dim=0)
            targets = torch.cat(
                [
                    torch.ones(k, dtype=torch.long, device=device),
                    torch.zeros(k, dtype=torch.long, device=device),
                ]
            )
            logits = self.instance_classifiers[y](instances)
            losses.append(self.instance_criterion(logits, targets))
            all_logits.append(logits)
            all_targets.append(targets)

            if self.config.subtyping and self.config.n_classes > 1:
                # High-attention evidence is negative for every *other* grade.
                for other in range(self.config.n_classes):
                    if other == y:
                        continue
                    out_logits = self.instance_classifiers[other](top_h[b])
                    out_targets = torch.zeros(k, dtype=torch.long, device=device)
                    losses.append(
                        self.instance_criterion(out_logits, out_targets) / (self.config.n_classes - 1)
                    )
                    all_logits.append(out_logits)
                    all_targets.append(out_targets)

        return {
            "instance_loss": torch.stack(losses).mean(),
            "instance_logits": torch.cat(all_logits, dim=0),
            "instance_targets": torch.cat(all_targets, dim=0),
            "top_indices": top_idx,
            "bottom_indices": bottom_idx,
        }

    # ------------------------------------------------------------------ #
    # Forward
    # ------------------------------------------------------------------ #
    def forward(
        self,
        features: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        labels: Optional[Union[int, torch.Tensor]] = None,
        use_instance_loss: Optional[bool] = None,
        return_patch_embeddings: bool = False,
    ) -> Dict[str, Any]:
        """Run a slide (or bag of slides) through CLAM-SB.

        Returns ``logits`` [B, 6], ``attention`` [B, N] (sums to 1 along N),
        ``raw_attention``, ``bag_feat`` and, when instance loss is active,
        ``instance_loss`` plus the ``top_*``/``bottom_*`` patch indices.
        """
        x, valid_mask, was_2d = self._prepare(features, mask)
        h = self.feature_encoder(x)
        attention, raw_attention = self._compute_attention(h, valid_mask)

        bag_feat = torch.einsum("bne,bn->be", h, attention)
        logits = self.slide_classifier(bag_feat)

        if use_instance_loss is None:
            use_instance_loss = self.instance_loss_weight > 0
        label_tensor = self._parse_labels(labels, x.size(0), x.device)
        use_instance_loss = bool(use_instance_loss and label_tensor is not None)

        if use_instance_loss:
            instance = self._instance_clustering_loss(attention, h, label_tensor, valid_mask)
        else:
            instance = {
                "instance_loss": logits.new_zeros(()),
                "instance_logits": logits.new_zeros((0, 2)),
                "instance_targets": logits.new_zeros((0,), dtype=torch.long),
            }

        result: Dict[str, Any] = {
            "logits": logits,
            "attention": attention,
            "raw_attention": raw_attention,
            "bag_feat": bag_feat,
            "instance_loss": instance["instance_loss"],
            "instance_logits": instance["instance_logits"],
            "instance_targets": instance["instance_targets"],
        }
        if "top_indices" in instance:
            result["top_indices"] = instance["top_indices"]
            result["bottom_indices"] = instance["bottom_indices"]
        if return_patch_embeddings:
            result["patch_embeddings"] = h
        return self._finalise(result, was_2d)


def build_clam_sb(
    config: Optional[Union[CLAMSBConfig, Dict[str, Any]]] = None, **overrides: Any
) -> CLAMSB:
    """Factory used by the shared trainer / notebooks."""
    return CLAMSB(config=config, **overrides)

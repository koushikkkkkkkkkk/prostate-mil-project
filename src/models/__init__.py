"""Model registry and construction helpers.

``build_model("clam_sb", cfg)`` returns a model implementing the shared MIL
contract (see ``src/models/base.py``). Adding a teammate's model means adding
one entry to ``MODEL_REGISTRY``.
"""

from __future__ import annotations

from typing import Any, Dict, Optional

from .abmil import ABMIL, ABMILConfig, build_abmil
from .base import MILModelBase, count_parameters, ensure_3d
from .clam_sb import CLAMSB, CLAMSBConfig, SmoothTop1SVMLoss, build_clam_sb
from .transmil import TransMIL, TransMILConfig, build_transmil

MODEL_REGISTRY: Dict[str, Any] = {
    "abmil": (ABMIL, ABMILConfig),
    "clam_sb": (CLAMSB, CLAMSBConfig),
    "clam-sb": (CLAMSB, CLAMSBConfig),
    "transmil": (TransMIL, TransMILConfig),
}

MIL_MODELS = ("abmil", "clam_sb", "transmil")


def build_model(name: str, config: Optional[Dict[str, Any]] = None, **overrides: Any):
    """Instantiate any registered model by name with a plain-dict config."""
    key = name.strip().lower()
    if key not in MODEL_REGISTRY:
        raise ValueError(f"Unknown model '{name}'. Available: {sorted(MODEL_REGISTRY)}")
    model_cls, config_cls = MODEL_REGISTRY[key]
    values = dict(config or {})
    values.update(overrides)
    return model_cls(config_cls.from_dict(values) if values else config_cls())


__all__ = [
    "ABMIL",
    "ABMILConfig",
    "CLAMSB",
    "CLAMSBConfig",
    "MILModelBase",
    "MODEL_REGISTRY",
    "SmoothTop1SVMLoss",
    "TransMIL",
    "TransMILConfig",
    "build_abmil",
    "build_clam_sb",
    "build_model",
    "build_transmil",
    "count_parameters",
    "ensure_3d",
]

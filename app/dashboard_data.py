"""Data access for the dashboard: read every run's artefacts from results/metrics.

Kept separate from the UI so the same loading logic can be unit-tested and reused by
notebooks. Nothing here trains or mutates anything.
"""
from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any, Dict, List, Optional

# --------------------------------------------------------------------------- #
# Encoder / run labelling
# --------------------------------------------------------------------------- #
ENCODER_LABELS = {
    "abmil": "ResNet50",
    "clam_sb": "ResNet50",
    "abmil_chief": "CHIEF",
    "clam_sb_chief": "CHIEF",
    "abmil_gigap": "GigaPath",
    "clam_sb_gigap": "GigaPath",
}

ENCODER_DIMS = {"ResNet50": 1024, "CHIEF": 768, "GigaPath": 1536, "UNI": 1024}


def pretty_run(folder: str) -> Dict[str, str]:
    """Turn a results/metrics subfolder into display labels.

    ``abmil_chief`` -> {"model": "ABMIL", "encoder": "CHIEF", "family": "pathology"}
    """
    row = csv_dict = {}
    parts = folder.split("/")
    name = parts[0]
    sub = parts[1] if len(parts) > 1 else ""
    encoder = ENCODER_LABELS.get(name, "ResNet50")
    model = "CLAM-SB" if name.startswith("clam") else "ABMIL" if name.startswith("abmil") else name.upper()
    if sub in ("with_instance", "no_instance"):
        model = f"{model} ({sub.replace('_', ' ')})"
    family = "baseline" if encoder == "ResNet50" else "pathology encoder"
    csv_dict = {"folder": folder, "model": model, "encoder": encoder, "family": family}
    return csv_dict


def _read_json(path: Path) -> Optional[Dict[str, Any]]:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None


# --------------------------------------------------------------------------- #
# Run discovery
# --------------------------------------------------------------------------- #
def discover_runs(metrics_root: Path) -> List[Dict[str, Any]]:
    """Every directory containing an ``aggregate_*.json``, parsed into a flat record."""
    runs: List[Dict[str, Any]] = []
    for agg_path in sorted(metrics_root.glob("**/aggregate_*.json")):
        agg = _read_json(agg_path)
        if not agg:
            continue
        folder = str(agg_path.parent.relative_to(metrics_root)).replace("\\", "/")
        if folder == ".":
            folder = agg_path.parent.name
        labels = pretty_run(folder)
        cfg = _read_json(agg_path.parent / "run_config.json") or {}
        runs.append(
            {
                **labels,
                "path": agg_path.parent,
                "qwk": agg.get("qwk", {}).get("mean"),
                "qwk_sd": agg.get("qwk", {}).get("std"),
                "acc": agg.get("accuracy", {}).get("mean"),
                "bal_acc": agg.get("balanced_accuracy", {}).get("mean"),
                "f1": agg.get("macro_f1", {}).get("mean"),
                "n_test": cfg.get("n_test"),
                "n_train": cfg.get("n_train"),
                "n_val": cfg.get("n_val"),
                "seeds": agg.get("seeds") or [],
                "manifest": cfg.get("manifest"),
                "features_root": cfg.get("features_root"),
                "in_dim": (cfg.get("model_config") or {}).get("in_dim"),
                "per_run": agg.get("per_run") or [],
            }
        )
    return runs


def run_matrix(runs: List[Dict[str, Any]], metric: str = "qwk") -> "pd.DataFrame":
    """model x encoder matrix of a metric, NaN where a combination was not run."""
    import numpy as np
    import pandas as pd

    if not runs:
        return pd.DataFrame()
    models = sorted({r["model"] for r in runs})
    encoders = ["ResNet50", "CHIEF", "GigaPath", "UNI"]
    encoders = [e for e in encoders if any(r["encoder"] == e for r in runs)]
    data = np.full((len(models), len(encoders)), np.nan)
    for r in runs:
        if r.get(metric) is None:
            continue
        i, j = models.index(r["model"]), encoders.index(r["encoder"])
        data[i, j] = r[metric]
    return pd.DataFrame(data, index=models, columns=encoders)


# --------------------------------------------------------------------------- #
# Predictions, per-slide
# --------------------------------------------------------------------------- #
def load_predictions(run_path: Path) -> "pd.DataFrame":
    """Concatenate every ``predictions_*.csv`` in a run directory into one frame."""
    import pandas as pd

    frames = []
    for f in sorted(run_path.glob("predictions_*.csv")):
        try:
            df = pd.read_csv(f)
        except Exception:
            continue
        df["seed"] = f.stem.split("_seed")[-1]
        frames.append(df)
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def confusion_from_predictions(preds: "pd.DataFrame", seeds: Optional[List[str]] = None) -> "np.ndarray":
    import numpy as np

    if preds.empty:
        return np.zeros((6, 6), dtype=int)
    if seeds:
        preds = preds[preds["seed"].astype(str).isin([str(s) for s in seeds])]
    cm = np.zeros((6, 6), dtype=int)
    for t, p in zip(preds["y_true"].astype(int), preds["y_pred"].astype(int)):
        cm[t, p] += 1
    return cm


# --------------------------------------------------------------------------- #
# Comparison and history
# --------------------------------------------------------------------------- #
def load_comparison(metrics_root: Path) -> "pd.DataFrame":
    import pandas as pd

    f = metrics_root / "comparison" / "encoder_comparison_57.csv"
    if f.exists():
        return pd.read_csv(f)
    return pd.DataFrame()


def load_history(run_path: Path) -> "pd.DataFrame":
    """All ``history_*.csv`` in a run, tagged by seed — for the training-curve tab."""
    import pandas as pd

    frames = []
    for f in sorted(run_path.glob("history_*.csv")):
        try:
            df = pd.read_csv(f)
        except Exception:
            continue
        df["seed"] = f.stem.split("_seed")[-1]
        frames.append(df)
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def load_attention_agreement(metrics_root: Path) -> Dict[str, Any]:
    f = metrics_root / "attention" / "agreement_abmil_vs_clam_summary.json"
    return _read_json(f) or {}


def load_fused_summary(metrics_root: Path) -> Dict[str, Any]:
    f = metrics_root / "comparison" / "fused_chief_gigap_summary.json"
    return _read_json(f) or {}


def load_statistical(metrics_root: Path) -> List[Dict[str, Any]]:
    out = []
    for f in sorted((metrics_root / "statistical").glob("*.json")):
        d = _read_json(f)
        if d:
            out.append({"model": f.stem.replace("_ordinal_logistic", ""), **d})
    return out

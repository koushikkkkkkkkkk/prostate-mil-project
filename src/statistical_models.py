"""Non-deep-learning baselines on bag-level statistics -- Person 4.

Two things live here:

1. **Bag statistics** derived from a trained MIL model's attention plus the
   patch features -- the 10-ish numbers the PLAN calls ``x_stat``:
   mean/median/std/max attention, top-10 attention mean, attention entropy,
   high-attention ratio, and embedding mean/variance summaries.

2. **Ordinal baselines** over ``x_stat``: a from-scratch cumulative-logit
   ordinal logistic regression (six binary "grade >= k" models sharing one
   design matrix) and, if installed, XGBoost.

These baselines answer the question "is MIL actually needed?", so they are
reported next to ABMIL / CLAM-SB / TransMIL.

Usage::

    python -m src.statistical_models \
        --attention-csv results/metrics/clam_sb_attention.csv \
        --manifest data/common_split.csv --out-dir results/metrics/statistical

Owner: Person 4.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

try:
    from src.evaluate import compute_metrics, quadratic_weighted_kappa
except ImportError:  # pragma: no cover - direct script execution
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from src.evaluate import compute_metrics, quadratic_weighted_kappa  # type: ignore

FEATURE_NAMES = (
    "mean_attention",
    "median_attention",
    "std_attention",
    "max_attention",
    "top10_attention_mean",
    "attention_entropy",
    "high_attention_ratio",
    "embedding_norm_mean",
    "embedding_norm_std",
    "n_patches",
)


# --------------------------------------------------------------------------- #
# x_stat construction
# --------------------------------------------------------------------------- #
def attention_statistics(attention: Sequence[float], k: int = 10) -> Dict[str, float]:
    values = np.asarray(attention, dtype=np.float64).reshape(-1)
    if values.size == 0:
        return {name: 0.0 for name in FEATURE_NAMES[:7]}
    positive = values[values > 0]
    distribution = positive / positive.sum() if positive.size else positive
    entropy = float(-(distribution * np.log(distribution + 1e-12)).sum()) if distribution.size else 0.0
    top_k = min(k, values.size)
    return {
        "mean_attention": float(values.mean()),
        "median_attention": float(np.median(values)),
        "std_attention": float(values.std()),
        "max_attention": float(values.max()),
        "top10_attention_mean": float(np.sort(values)[-top_k:].mean()),
        "attention_entropy": entropy,
        "high_attention_ratio": float((values > 2.0 / values.size).mean()),
    }


def embedding_statistics(
    features: Any, attention: Any, k: int = 10
) -> Dict[str, float]:
    """Norm statistics of the patch embeddings (cheap, model-agnostic)."""
    array = features.detach().cpu().numpy() if hasattr(features, "detach") else np.asarray(features)
    array = np.asarray(array, dtype=np.float64)
    if array.ndim != 2 or array.size == 0:
        return {"embedding_norm_mean": 0.0, "embedding_norm_std": 0.0, "n_patches": 0.0}
    norms = np.linalg.norm(array, axis=1)
    return {
        "embedding_norm_mean": float(norms.mean()),
        "embedding_norm_std": float(norms.std()),
        "n_patches": float(array.shape[0]),
    }


def bag_statistics(attention: Any, features: Any = None, k: int = 10) -> Dict[str, float]:
    stats = attention_statistics(attention, k=k)
    stats.update(embedding_statistics(features, attention, k=k) if features is not None else {
        "embedding_norm_mean": float("nan"),
        "embedding_norm_std": float("nan"),
        "n_patches": float(np.asarray(attention).size),
    })
    return stats


# --------------------------------------------------------------------------- #
# Ordinal logistic regression (cumulative link, from scratch)
# --------------------------------------------------------------------------- #
class OrdinalLogisticRegression:
    """``P(y >= k) = sigmoid(X w + b_k)`` fitted with six weighted binary models.

    Deliberately simple: it uses scikit-learn's logistic regression as the
    optimiser but implements the ordinal decomposition, thresholds and decoding
    itself, so the baseline is transparent and reproducible.
    """

    def __init__(self, n_classes: int = 6, C: float = 1.0, max_iter: int = 2000) -> None:
        self.n_classes = n_classes
        self.C = C
        self.max_iter = max_iter
        self.models: List[Any] = []
        self.mean_: Optional[np.ndarray] = None
        self.scale_: Optional[np.ndarray] = None

    def _standardise(self, x: np.ndarray) -> np.ndarray:
        if self.mean_ is None:
            self.mean_ = x.mean(axis=0)
            self.scale_ = x.std(axis=0)
            self.scale_[self.scale_ < 1e-8] = 1.0
        return (x - self.mean_) / self.scale_

    def fit(self, x: np.ndarray, y: np.ndarray) -> "OrdinalLogisticRegression":
        from sklearn.linear_model import LogisticRegression

        x = np.nan_to_num(np.asarray(x, dtype=np.float64))
        y = np.asarray(y, dtype=int)
        x = self._standardise(x)

        self.models = []
        for threshold in range(1, self.n_classes):
            binary = (y >= threshold).astype(int)
            if binary.min() == binary.max():     # threshold never crossed
                self.models.append(None)
                continue
            model = LogisticRegression(C=self.C, max_iter=self.max_iter, class_weight="balanced")
            model.fit(x, binary)
            self.models.append(model)
        return self

    def predict_proba(self, x: np.ndarray) -> np.ndarray:
        x = (np.nan_to_num(np.asarray(x, dtype=np.float64)) - self.mean_) / self.scale_
        cumulative = np.ones((x.shape[0], self.n_classes), dtype=np.float64)
        for threshold, model in enumerate(self.models, start=1):
            if model is None:
                cumulative[:, threshold:] = 0.0
                continue
            cumulative[:, threshold] = model.predict_proba(x)[:, 1]
        cumulative[:, 0] = 1.0
        cumulative = np.minimum.accumulate(cumulative, axis=1)   # enforce monotonicity
        # P(y = k) = P(y >= k) - P(y >= k + 1), with P(y >= C) := 0.
        extended = np.concatenate([cumulative, np.zeros((x.shape[0], 1))], axis=1)
        probabilities = extended[:, :-1] - extended[:, 1:]
        probabilities = np.clip(probabilities, 1e-9, None)
        return probabilities / probabilities.sum(axis=1, keepdims=True)

    def predict(self, x: np.ndarray) -> np.ndarray:
        """Ordinal decode: expected grade, rounded to the nearest class."""
        probabilities = self.predict_proba(x)
        expectation = probabilities @ np.arange(self.n_classes)
        return np.clip(np.rint(expectation), 0, self.n_classes - 1).astype(int)


def fit_xgboost_if_available(
    x_train: np.ndarray, y_train: np.ndarray, x_test: np.ndarray, seed: int = 42
) -> Optional[np.ndarray]:
    """Optional XGBoost baseline; returns ``None`` when the package is absent."""
    try:
        from xgboost import XGBClassifier  # type: ignore
    except ModuleNotFoundError:
        return None
    model = XGBClassifier(
        n_estimators=300,
        max_depth=3,
        learning_rate=0.05,
        subsample=0.9,
        colsample_bytree=0.9,
        objective="multi:softmax",
        num_class=6,
        random_state=seed,
        eval_metric="mlogloss",
    )
    model.fit(np.nan_to_num(x_train), np.asarray(y_train))
    return model.predict(np.nan_to_num(x_test)).astype(int)


# --------------------------------------------------------------------------- #
# Driver
# --------------------------------------------------------------------------- #
def load_attention_table(path: str | Path) -> Dict[str, Dict[str, float]]:
    """Read ``slide_id,<stat columns>`` produced by any attention export."""
    table: Dict[str, Dict[str, float]] = {}
    with Path(path).open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            slide_id = row.get("slide_id") or row.get("image_id")
            if not slide_id:
                continue
            table[slide_id] = {
                key: float(value)
                for key, value in row.items()
                if key not in {"slide_id", "image_id", "split", "y_true"} and value not in (None, "")
            }
    if not table:
        raise ValueError(f"No rows read from {path}")
    return table


def run_statistical_baseline(
    attention_csv: str | Path,
    manifest: str | Path,
    out_dir: str | Path,
    seed: int = 42,
) -> Dict[str, Any]:
    try:
        from src.dataset import read_manifest
    except ImportError:  # pragma: no cover
        import sys

        sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
        from src.dataset import read_manifest  # type: ignore

    stats = load_attention_table(attention_csv)
    records = {record.image_id: record for record in read_manifest(manifest)}

    def collect(split: str) -> Tuple[np.ndarray, np.ndarray, List[str]]:
        rows, labels, ids = [], [], []
        for slide_id, values in stats.items():
            record = records.get(slide_id)
            if record is None or record.split != split:
                continue
            rows.append([values.get(name, np.nan) for name in FEATURE_NAMES])
            labels.append(record.label)
            ids.append(slide_id)
        return np.asarray(rows, dtype=np.float64).reshape(-1, len(FEATURE_NAMES)), np.asarray(labels, dtype=int), ids

    x_train, y_train, _ = collect("train")
    x_test, y_test, test_ids = collect("test")
    if x_train.size == 0 or x_test.size == 0:
        raise ValueError(
            "Need overlapping slide ids in the attention CSV for both train and test splits"
        )

    ordinal = OrdinalLogisticRegression().fit(x_train, y_train)
    ordinal_pred = ordinal.predict(x_test)
    results: Dict[str, Any] = {
        "ordinal_logistic": compute_metrics(y_test, ordinal_pred, n_classes=6),
        "n_train": int(x_train.shape[0]),
        "n_test": int(x_test.shape[0]),
        "features": list(FEATURE_NAMES),
    }
    results["ordinal_logistic"]["qwk_recompute_check"] = quadratic_weighted_kappa(y_test, ordinal_pred)

    xgb_pred = fit_xgboost_if_available(x_train, y_train, x_test, seed=seed)
    if xgb_pred is not None:
        results["xgboost"] = compute_metrics(y_test, xgb_pred, n_classes=6)
    else:
        results["xgboost"] = {"skipped": "xgboost not installed"}

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "statistical_baselines.json").write_text(json.dumps(results, indent=2), encoding="utf-8")
    with (out_dir / "statistical_predictions.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["slide_id", "y_true", "ordinal_pred", "xgboost_pred"])
        for i, slide_id in enumerate(test_ids):
            writer.writerow(
                [slide_id, int(y_test[i]), int(ordinal_pred[i]), int(xgb_pred[i]) if xgb_pred is not None else ""]
            )
    return results


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Fit ordinal / XGBoost baselines on attention statistics.")
    parser.add_argument("--attention-csv", required=True)
    parser.add_argument("--manifest", default="data/common_split.csv")
    parser.add_argument("--out-dir", default="results/metrics/statistical")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args(argv)

    results = run_statistical_baseline(args.attention_csv, args.manifest, args.out_dir, seed=args.seed)
    for name, payload in results.items():
        if isinstance(payload, dict) and "qwk" in payload:
            print(f"{name}: QWK {payload['qwk']:.4f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

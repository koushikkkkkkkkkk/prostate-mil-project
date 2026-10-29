"""Evaluation suite: QWK, classification metrics, bootstrap CIs, McNemar, overlap.

Primary metric
--------------
**Quadratic Weighted Kappa (QWK)** on ISUP grades 0-5. QWK is the PANDA
competition metric and the clinically meaningful one here: it penalises a
grade-0 -> grade-5 error far more than a grade-2 -> grade-3 error.

Secondary metrics
-----------------
accuracy, balanced accuracy, macro F1, per-class precision/recall/F1/support,
confusion matrix, and (for Person 2) the attention -> tumor-mask overlap score.

Statistics
----------
* ``bootstrap_metric_ci`` -- percentile bootstrap CI for any metric.
* ``paired_bootstrap_difference`` -- CI for the QWK difference between two
  models evaluated on the same slides.
* ``mcnemar_test`` -- exact McNemar test on paired correct/incorrect outcomes.

Owner: Person 4 (metrics) / everyone (consumers).
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, Optional, Sequence, Tuple

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    cohen_kappa_score,
    confusion_matrix,
    f1_score,
    precision_recall_fscore_support,
)

ISUP_CLASSES: Tuple[int, ...] = (0, 1, 2, 3, 4, 5)


# --------------------------------------------------------------------------- #
# Core metrics
# --------------------------------------------------------------------------- #
def quadratic_weighted_kappa(y_true: Sequence[int], y_pred: Sequence[int]) -> float:
    """QWK with a guard for degenerate inputs (single class present)."""
    y_true = np.asarray(y_true).astype(int)
    y_pred = np.asarray(y_pred).astype(int)
    if y_true.size == 0:
        return float("nan")
    if y_true.size != y_pred.size:
        raise ValueError("y_true and y_pred must have the same length")
    try:
        return float(cohen_kappa_score(y_true, y_pred, weights="quadratic"))
    except ValueError:
        return float("nan")


def compute_metrics(
    y_true: Sequence[int],
    y_pred: Sequence[int],
    n_classes: int = 6,
    labels: Optional[Sequence[int]] = None,
) -> Dict[str, Any]:
    """Full metric bundle for a single model's predictions."""
    y_true = np.asarray(y_true).astype(int)
    y_pred = np.asarray(y_pred).astype(int)
    labels_list = list(labels) if labels is not None else list(range(n_classes))

    precision, recall, f1, support = precision_recall_fscore_support(
        y_true, y_pred, labels=labels_list, zero_division=0
    )
    cm = confusion_matrix(y_true, y_pred, labels=labels_list)

    return {
        "n": int(y_true.size),
        "qwk": quadratic_weighted_kappa(y_true, y_pred),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", labels=labels_list, zero_division=0)),
        "weighted_f1": float(
            f1_score(y_true, y_pred, average="weighted", labels=labels_list, zero_division=0)
        ),
        "per_class": {
            str(label): {
                "precision": float(precision[i]),
                "recall": float(recall[i]),
                "f1": float(f1[i]),
                "support": int(support[i]),
            }
            for i, label in enumerate(labels_list)
        },
        "confusion_matrix": cm.tolist(),
    }


# --------------------------------------------------------------------------- #
# Statistics
# --------------------------------------------------------------------------- #
def bootstrap_metric_ci(
    y_true: Sequence[int],
    y_pred: Sequence[int],
    metric: Callable[[Sequence[int], Sequence[int]], float] = quadratic_weighted_kappa,
    n_boot: int = 2000,
    alpha: float = 0.05,
    seed: int = 42,
) -> Dict[str, float]:
    """Percentile bootstrap CI for a metric over slides."""
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    rng = np.random.default_rng(seed)
    n = y_true.size
    if n == 0:
        return {"point": float("nan"), "lo": float("nan"), "hi": float("nan")}

    point = float(metric(y_true, y_pred))
    samples = np.empty(n_boot, dtype=float)
    for i in range(n_boot):
        index = rng.integers(0, n, n)
        samples[i] = metric(y_true[index], y_pred[index])
    finite = samples[np.isfinite(samples)]
    if finite.size == 0:
        return {"point": point, "lo": float("nan"), "hi": float("nan")}
    return {
        "point": point,
        "lo": float(np.quantile(finite, alpha / 2)),
        "hi": float(np.quantile(finite, 1 - alpha / 2)),
        "n_boot": int(n_boot),
    }


def paired_bootstrap_difference(
    y_true: Sequence[int],
    y_pred_a: Sequence[int],
    y_pred_b: Sequence[int],
    metric: Callable[[Sequence[int], Sequence[int]], float] = quadratic_weighted_kappa,
    n_boot: int = 2000,
    alpha: float = 0.05,
    seed: int = 42,
) -> Dict[str, float]:
    """CI for ``metric(b) - metric(a)`` resampling slides (paired design)."""
    y_true = np.asarray(y_true)
    pred_a = np.asarray(y_pred_a)
    pred_b = np.asarray(y_pred_b)
    if not (y_true.size == pred_a.size == pred_b.size):
        raise ValueError("All three arrays must have the same length")

    rng = np.random.default_rng(seed)
    n = y_true.size
    observed = float(metric(y_true, pred_b) - metric(y_true, pred_a))
    diffs = np.empty(n_boot, dtype=float)
    for i in range(n_boot):
        index = rng.integers(0, n, n)
        diffs[i] = metric(y_true[index], pred_b[index]) - metric(y_true[index], pred_a[index])
    finite = diffs[np.isfinite(diffs)]
    return {
        "difference": observed,
        "lo": float(np.quantile(finite, alpha / 2)) if finite.size else float("nan"),
        "hi": float(np.quantile(finite, 1 - alpha / 2)) if finite.size else float("nan"),
        "n_boot": int(n_boot),
    }


def mcnemar_test(
    y_true: Sequence[int], y_pred_a: Sequence[int], y_pred_b: Sequence[int]
) -> Dict[str, float]:
    """Exact McNemar test on paired correctness of two models.

    Uses the binomial exact test on the discordant cells, which is the correct
    choice for the small test sets in this project (60 slides).
    """
    y_true = np.asarray(y_true).astype(int)
    correct_a = np.asarray(y_pred_a).astype(int) == y_true
    correct_b = np.asarray(y_pred_b).astype(int) == y_true

    b = int(np.sum(correct_a & ~correct_b))  # a right, b wrong
    c = int(np.sum(~correct_a & correct_b))  # a wrong, b right
    n = b + c
    if n == 0:
        p_value = 1.0
    else:
        # Two-sided exact binomial test with p = 0.5.
        k = min(b, c)
        tail = sum(math.comb(n, i) for i in range(k + 1)) / (2 ** n)
        p_value = min(1.0, 2 * tail)
    return {
        "a_correct_b_wrong": b,
        "a_wrong_b_correct": c,
        "n_discordant": n,
        "p_value": float(p_value),
    }


# --------------------------------------------------------------------------- #
# Localisation metric (Person 2 / Person 4 joint)
# --------------------------------------------------------------------------- #
def attention_mask_overlap(
    attention: Sequence[float],
    mask: Sequence[float],
    fractions: Iterable[float] = (0.1, 0.2, 0.3),
) -> Dict[str, float]:
    """How well the top-attention patches match the tumor mask.

    ``attention`` is one weight per patch, ``mask`` is per-patch tumor fraction
    in [0, 1] (mean of the binary tissue mask over the patch, or a 0/1 label).
    Positive patches are those with ``mask >= 0.5``.

    Reported scores at each ``fraction`` of patches flagged as "hot":
      * ``precision`` -- fraction of flagged patches that are truly tumor
      * ``recall``    -- fraction of tumor patches that were flagged
      * ``lift``      -- precision / base_rate (>1 means better than chance)

    This is a weak, patch-level proxy for localisation quality: it is not a
    segmentation score, and PANDA provides no pixel-level ground truth, so any
    mask used here must be treated as a coarse reference.
    """
    attention = np.asarray(attention, dtype=float).reshape(-1)
    mask = np.asarray(mask, dtype=float).reshape(-1)
    if attention.size != mask.size:
        raise ValueError("attention and mask must have the same length")
    if attention.size == 0:
        return {"n_patches": 0}

    tumour = mask >= 0.5
    base_rate = float(tumour.mean())
    order = np.argsort(-attention)
    results: Dict[str, Any] = {"n_patches": int(attention.size), "base_rate": base_rate}

    for fraction in fractions:
        k = max(1, int(round(fraction * attention.size)))
        flagged = np.zeros_like(tumour, dtype=bool)
        flagged[order[:k]] = True
        true_positive = int((flagged & tumour).sum())
        precision = true_positive / k
        recall = true_positive / max(int(tumour.sum()), 1)
        results[f"top{int(fraction * 100)}pct"] = {
            "k": int(k),
            "precision": float(precision),
            "recall": float(recall),
            "lift": float(precision / base_rate) if base_rate > 0 else float("nan"),
        }
    return results


# --------------------------------------------------------------------------- #
# Persistence helpers
# --------------------------------------------------------------------------- #
def save_metrics(metrics: Dict[str, Any], path: str | Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    return path


def aggregate_seed_metrics(metrics: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    """Mean/std over seeds for the headline metrics (3 seeds per protocol)."""
    keys = ("qwk", "accuracy", "balanced_accuracy", "macro_f1", "weighted_f1")
    summary: Dict[str, Any] = {"n_seeds": len(metrics)}
    for key in keys:
        values = np.array([m[key] for m in metrics if key in m], dtype=float)
        values = values[np.isfinite(values)]
        if values.size == 0:
            continue
        summary[key] = {
            "mean": float(values.mean()),
            "std": float(values.std(ddof=1)) if values.size > 1 else 0.0,
            "values": values.tolist(),
        }
    return summary


def markdown_table(rows: Sequence[Dict[str, Any]], columns: Sequence[Tuple[str, str]]) -> str:
    """Tiny markdown table writer used for ``table_*.md`` result files."""
    header = "| " + " | ".join(title for _, title in columns) + " |"
    divider = "| " + " | ".join("---" for _ in columns) + " |"
    lines = [header, divider]
    for row in rows:
        cells = []
        for key, _ in columns:
            value = row.get(key, "")
            cells.append(f"{value:.4f}" if isinstance(value, float) else str(value))
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def plot_confusion_matrix(
    cm: Sequence[Sequence[int]],
    out_path: str | Path,
    title: str = "Confusion matrix",
    class_names: Optional[Sequence[str]] = None,
) -> Path:
    """Render a normalised confusion matrix PNG (matplotlib Agg-safe)."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    matrix = np.asarray(cm, dtype=float)
    row_sums = matrix.sum(axis=1, keepdims=True)
    normalised = np.divide(matrix, np.maximum(row_sums, 1e-9))

    names = list(class_names) if class_names else [str(i) for i in range(matrix.shape[0])]
    fig, ax = plt.subplots(figsize=(6.2, 5.2))
    image = ax.imshow(normalised, cmap="Blues", vmin=0, vmax=1)
    ax.set_xticks(range(len(names)), names)
    ax.set_yticks(range(len(names)), names)
    ax.set_xlabel("Predicted ISUP grade")
    ax.set_ylabel("True ISUP grade")
    ax.set_title(title)
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            ax.text(
                j,
                i,
                f"{int(matrix[i, j])}\n{normalised[i, j]:.2f}",
                ha="center",
                va="center",
                fontsize=8,
                color="white" if normalised[i, j] > 0.55 else "black",
            )
    fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04, label="row-normalised")
    fig.tight_layout()

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return out_path

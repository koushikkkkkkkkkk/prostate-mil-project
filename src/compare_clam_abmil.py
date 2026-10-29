"""Paired ABMIL vs CLAM-SB comparison -- Person 2.

Both models are evaluated on the *same* test slides, so the honest comparison
is a paired one:

* per-seed QWK / accuracy / macro F1 table
* paired bootstrap 95% CI for ``QWK(clam_sb) - QWK(abmil)``
* exact McNemar test on discordant correct/incorrect slides
* a comparison figure (QWK per seed with error bars)

Inputs are prediction CSVs in the shared format
(``slide_id, y_true, y_pred, p0..p5``) as written by ``src/train.py`` and
``src/evaluate_clam_sb.py``.

Usage::

    python -m src.compare_clam_abmil \
        --abmil results/metrics/abmil/predictions_abmil_seed0.csv \
        --clam results/metrics/clam_sb/with_instance/predictions_clam_sb_seed0.csv \
        --clam-label "CLAM-SB (+instance loss)" \
        --out-dir results/metrics/comparison
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

try:
    from src.evaluate import (
        compute_metrics,
        mcnemar_test,
        paired_bootstrap_difference,
        quadratic_weighted_kappa,
    )
except ImportError:  # pragma: no cover - direct script execution
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from src.evaluate import (  # type: ignore
        compute_metrics,
        mcnemar_test,
        paired_bootstrap_difference,
        quadratic_weighted_kappa,
    )


def read_predictions(path: str | Path) -> Dict[str, Tuple[int, int]]:
    """``{slide_id: (y_true, y_pred)}`` from a shared-format predictions CSV."""
    table: Dict[str, Tuple[int, int]] = {}
    with Path(path).open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            slide_id = row.get("slide_id") or row.get("image_id")
            if not slide_id:
                continue
            table[slide_id] = (int(float(row["y_true"])), int(float(row["y_pred"])))
    if not table:
        raise ValueError(f"No predictions found in {path}")
    return table


def align(
    pred_a: Dict[str, Tuple[int, int]], pred_b: Dict[str, Tuple[int, int]]
) -> Tuple[List[str], np.ndarray, np.ndarray, np.ndarray]:
    shared = sorted(set(pred_a) & set(pred_b))
    if not shared:
        raise ValueError(
            "The two prediction files share no slide ids -- were they evaluated on "
            "the same split? (both should be the 60 test slides)"
        )
    y_true = np.asarray([pred_a[s][0] for s in shared], dtype=int)
    y_a = np.asarray([pred_a[s][1] for s in shared], dtype=int)
    y_b = np.asarray([pred_b[s][1] for s in shared], dtype=int)
    mismatched = [s for s in shared if pred_a[s][0] != pred_b[s][0]]
    if mismatched:
        raise ValueError(f"Ground-truth labels disagree for {len(mismatched)} slides")
    return shared, y_true, y_a, y_b


def compare(
    abmil_csv: str | Path,
    clam_csv: str | Path,
    out_dir: str | Path,
    abmil_label: str = "ABMIL",
    clam_label: str = "CLAM-SB",
    n_boot: int = 2000,
    seed: int = 42,
) -> Dict[str, Any]:
    pred_a = read_predictions(abmil_csv)
    pred_b = read_predictions(clam_csv)
    shared, y_true, y_a, y_b = align(pred_a, pred_b)

    metrics_a = compute_metrics(y_true, y_a)
    metrics_b = compute_metrics(y_true, y_b)
    difference = paired_bootstrap_difference(
        y_true, y_a, y_b, metric=quadratic_weighted_kappa, n_boot=n_boot, seed=seed
    )
    mcnemar = mcnemar_test(y_true, y_a, y_b)

    rows: List[Dict[str, Any]] = []
    for name, metric in ((abmil_label, metrics_a), (clam_label, metrics_b)):
        rows.append(
            {
                "model": name,
                "qwk": metric["qwk"],
                "accuracy": metric["accuracy"],
                "balanced_accuracy": metric["balanced_accuracy"],
                "macro_f1": metric["macro_f1"],
                "n_slides": metric["n"],
            }
        )

    results: Dict[str, Any] = {
        "n_slides": len(shared),
        "abmil": {"label": abmil_label, **metrics_a},
        "clam_sb": {"label": clam_label, **metrics_b},
        "paired_qwk_difference": difference,
        "mcnemar": mcnemar,
    }

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "comparison.json").write_text(json.dumps(results, indent=2), encoding="utf-8")
    _write_markdown(results, rows, abmil_label, clam_label, out_dir / "comparison.md")
    _plot_comparison(metrics_a, metrics_b, abmil_label, clam_label, out_dir / "comparison_qwk.png")
    return results


def _write_markdown(
    results: Dict[str, Any],
    rows: Sequence[Dict[str, Any]],
    abmil_label: str,
    clam_label: str,
    path: Path,
) -> None:
    lines = [
        "# ABMIL vs CLAM-SB (paired, same test slides)",
        "",
        f"Test slides compared: **{results['n_slides']}**",
        "",
        "| Model | QWK | Accuracy | Balanced acc. | Macro F1 |",
        "| --- | --- | --- | --- | --- |",
    ]
    for row in rows:
        lines.append(
            f"| {row['model']} | {row['qwk']:.4f} | {row['accuracy']:.4f} | "
            f"{row['balanced_accuracy']:.4f} | {row['macro_f1']:.4f} |"
        )
    difference = results["paired_qwk_difference"]
    mcnemar = results["mcnemar"]
    lines += [
        "",
        "## Paired statistics",
        "",
        f"- QWK difference ({clam_label} - {abmil_label}): **{difference['difference']:+.4f}** "
        f"(95% bootstrap CI [{difference['lo']:+.4f}, {difference['hi']:+.4f}], {difference['n_boot']} resamples)",
        f"- McNemar exact test: {mcnemar['a_correct_b_wrong']} vs {mcnemar['a_wrong_b_correct']} "
        f"discordant slides, p = {mcnemar['p_value']:.4f}",
        "",
        "Interpretation guard: with only 60 test slides a non-significant McNemar test "
        "means *inconclusive*, not *equivalent*. Report the CI, not just the point estimate.",
        "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


def _plot_comparison(
    metrics_a: Dict[str, Any],
    metrics_b: Dict[str, Any],
    abmil_label: str,
    clam_label: str,
    path: Path,
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    keys = ["qwk", "accuracy", "balanced_accuracy", "macro_f1"]
    labels = ["QWK", "Accuracy", "Balanced acc.", "Macro F1"]
    x = np.arange(len(keys))
    width = 0.36

    fig, ax = plt.subplots(figsize=(7.0, 3.8))
    ax.bar(x - width / 2, [metrics_a[k] for k in keys], width, label=abmil_label, color="#4c72b0")
    ax.bar(x + width / 2, [metrics_b[k] for k in keys], width, label=clam_label, color="#c44e52")
    ax.set_xticks(x, labels)
    ax.set_ylim(0, 1.0)
    ax.set_ylabel("score")
    ax.set_title("ABMIL vs CLAM-SB on the shared test split")
    ax.legend(frameon=False)
    for offset, metrics in ((-width / 2, metrics_a), (width / 2, metrics_b)):
        for i, key in enumerate(keys):
            ax.text(i + offset, metrics[key] + 0.02, f"{metrics[key]:.2f}", ha="center", fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Paired ABMIL vs CLAM-SB comparison.")
    parser.add_argument("--abmil", required=True, help="ABMIL predictions CSV")
    parser.add_argument("--clam", required=True, help="CLAM-SB predictions CSV")
    parser.add_argument("--out-dir", default="results/metrics/comparison")
    parser.add_argument("--abmil-label", default="ABMIL")
    parser.add_argument("--clam-label", default="CLAM-SB")
    parser.add_argument("--n-boot", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args(argv)

    results = compare(
        args.abmil,
        args.clam,
        args.out_dir,
        abmil_label=args.abmil_label,
        clam_label=args.clam_label,
        n_boot=args.n_boot,
        seed=args.seed,
    )
    difference = results["paired_qwk_difference"]
    print(
        f"QWK {args.abmil_label}={results['abmil']['qwk']:.4f} "
        f"{args.clam_label}={results['clam_sb']['qwk']:.4f} "
        f"diff={difference['difference']:+.4f} [{difference['lo']:+.4f}, {difference['hi']:+.4f}] "
        f"McNemar p={results['mcnemar']['p_value']:.4f}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

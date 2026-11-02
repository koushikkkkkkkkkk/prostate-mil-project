"""Regenerate every paired results table straight from the committed predictions.

Why this exists
---------------
The project has five trained result trees living on four different test sets
(60 slides for the ResNet50 cohort, 57 for CHIEF, 53 for GigaPath). Comparing the
QWK in `results/metrics/*/aggregate_*.json` across those trees is **not** a model
comparison -- it is three different test sets. This script fixes that by only ever
comparing runs on slides they share.

It writes two tables:

1. **Common-cohort table** -- every run restricted to the largest set of test slides
   that *all* runs share, so every row is measured on identical slides.
2. **Native-cohort table** -- each run on its own test set, with `n` printed next to
   the metric so nobody can compare across rows by accident.

Plus paired statistics (paired bootstrap CI for dQWK, exact McNemar) between the
reference run of each cohort and every other run in that cohort.

Everything is recomputed from `results/metrics/*/predictions_*_seed*.csv`; no number
is hand-copied from a document. If a prediction file is missing, that run simply
does not appear -- the script never invents a row.

Usage::

    python scripts/paired_results.py
    python scripts/paired_results.py --results results --out results/metrics/comparison

Outputs:
    results/metrics/comparison/paired_common_cohort.md / .csv
    results/metrics/comparison/paired_native_cohort.md / .csv
    results/metrics/comparison/paired_stats.json / .md
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402

from src.evaluate import (  # noqa: E402
    mcnemar_test,
    paired_bootstrap_difference,
    quadratic_weighted_kappa,
)

#: human-readable encoder label per result folder (falls back to the folder name)
ENCODER_LABELS = {
    "clam_sb/with_instance": "ResNet50 (1024-d)",
    "clam_sb/without_instance": "ResNet50 (1024-d)",
    "abmil": "ResNet50 (1024-d)",
    "transmil": "ResNet50 (1024-d)",
    "abmil_chief": "CHIEF (768-d)",
    "clam_sb_chief": "CHIEF (768-d)",
    "abmil_gigap": "GigaPath (1536-d)",
}

MODEL_LABELS = {
    "clam_sb/with_instance": "CLAM-SB + instance loss",
    "clam_sb/without_instance": "CLAM-SB, no instance loss",
    "abmil": "ABMIL",
    "transmil": "TransMIL",
    "abmil_chief": "ABMIL",
    "clam_sb_chief": "CLAM-SB",
    "abmil_gigap": "ABMIL",
}


# --------------------------------------------------------------------------- #
# Loading
# --------------------------------------------------------------------------- #
def load_run(run_dir: Path) -> Dict[int, Dict[str, Tuple[int, int]]]:
    """``{seed: {slide_id: (y_true, y_pred)}}`` for one result folder."""
    seeds: Dict[int, Dict[str, Tuple[int, int]]] = {}
    for path in sorted(run_dir.glob("predictions_*_seed*.csv")):
        try:
            seed = int(path.stem.rsplit("seed", 1)[-1])
        except ValueError:
            continue
        table: Dict[str, Tuple[int, int]] = {}
        with path.open(newline="", encoding="utf-8") as handle:
            for row in csv.DictReader(handle):
                slide_id = row.get("slide_id") or row.get("image_id")
                if not slide_id:
                    continue
                table[slide_id] = (int(float(row["y_true"])), int(float(row["y_pred"])))
        if table:
            seeds[seed] = table
    return seeds


def discover_runs(results: Path) -> Dict[str, Dict[int, Dict[str, Tuple[int, int]]]]:
    """Every folder under results/metrics that holds prediction CSVs.

    Searched **recursively**: a run can be nested one level down (the CLAM-SB
    ablation writes ``clam_sb/with_instance/`` and ``clam_sb/without_instance/``),
    and a one-level scan silently drops those arms from every table. Keys are
    paths relative to ``results/metrics`` with forward slashes, so a top-level run
    keeps its old name ("abmil") and a nested arm reads "clam_sb/with_instance".
    """
    runs: Dict[str, Dict[int, Dict[str, Tuple[int, int]]]] = {}
    metrics_root = results / "metrics"
    candidates = sorted({p.parent for p in metrics_root.rglob("predictions_*_seed*.csv")})
    for run_dir in candidates:
        if not run_dir.is_dir():
            continue
        seeds = load_run(run_dir)
        if seeds:
            runs[str(run_dir.relative_to(metrics_root)).replace("\\", "/")] = seeds
    return runs


# --------------------------------------------------------------------------- #
# Metrics on a chosen slide set
# --------------------------------------------------------------------------- #
def scores_on(
    seeds: Dict[int, Dict[str, Tuple[int, int]]], slides: Optional[Sequence[str]] = None
) -> Dict[str, Any]:
    """Per-seed QWK / accuracy / macro-F1 on ``slides`` (default: each seed's own set)."""
    per_seed: List[Dict[str, float]] = []
    for seed in sorted(seeds):
        table = seeds[seed]
        keep = sorted(set(table)) if slides is None else [s for s in slides if s in table]
        if not keep:
            continue
        y_true = [table[s][0] for s in keep]
        y_pred = [table[s][1] for s in keep]
        per_seed.append(
            {
                "seed": seed,
                "n": len(keep),
                "qwk": quadratic_weighted_kappa(y_true, y_pred),
                "accuracy": float(np.mean([t == p for t, p in zip(y_true, y_pred)])),
            }
        )
    if not per_seed:
        return {}
    qwk = np.asarray([row["qwk"] for row in per_seed], dtype=float)
    acc = np.asarray([row["accuracy"] for row in per_seed], dtype=float)
    return {
        "n_seeds": len(per_seed),
        "n_slides": int(per_seed[0]["n"]),
        "qwk_mean": float(qwk.mean()),
        "qwk_sd": float(qwk.std(ddof=1)) if qwk.size > 1 else 0.0,
        "acc_mean": float(acc.mean()),
        "acc_sd": float(acc.std(ddof=1)) if acc.size > 1 else 0.0,
        "per_seed": per_seed,
    }


def paired_stats(
    run_a: Dict[int, Dict[str, Tuple[int, int]]],
    run_b: Dict[int, Dict[str, Tuple[int, int]]],
    n_boot: int = 2000,
) -> Dict[str, Any]:
    """Per-seed paired dQWK (b - a) with bootstrap CI + exact McNemar."""
    rows: List[Dict[str, Any]] = []
    for seed in sorted(set(run_a) & set(run_b)):
        shared = sorted(set(run_a[seed]) & set(run_b[seed]))
        if len(shared) < 3:
            continue
        y_true = [run_a[seed][s][0] for s in shared]
        y_a = [run_a[seed][s][1] for s in shared]
        y_b = [run_b[seed][s][1] for s in shared]
        difference = paired_bootstrap_difference(
            y_true, y_a, y_b, metric=quadratic_weighted_kappa, n_boot=n_boot, seed=seed
        )
        test = mcnemar_test(y_true, y_a, y_b)
        rows.append(
            {
                "seed": seed,
                "n": len(shared),
                "qwk_a": quadratic_weighted_kappa(y_true, y_a),
                "qwk_b": quadratic_weighted_kappa(y_true, y_b),
                "delta": difference["difference"],
                "lo": difference["lo"],
                "hi": difference["hi"],
                "mcnemar_p": test["p_value"],
                "a_right_b_wrong": test["a_correct_b_wrong"],
                "a_wrong_b_right": test["a_wrong_b_correct"],
            }
        )
    if not rows:
        return {}
    deltas = np.asarray([row["delta"] for row in rows], dtype=float)
    return {
        "n_seeds": len(rows),
        "mean_delta": float(deltas.mean()),
        "sd_delta": float(deltas.std(ddof=1)) if deltas.size > 1 else 0.0,
        "per_seed": rows,
    }


# --------------------------------------------------------------------------- #
# Table rendering
# --------------------------------------------------------------------------- #
def _fmt(value: float, digits: int = 4) -> str:
    return "n/a" if value != value else f"{value:.{digits}f}"


def render_table(rows: Sequence[Dict[str, Any]], title: str, caption: str) -> str:
    lines = [f"# {title}", "", caption, ""]
    lines += [
        "| model | encoder | n slides | seeds | QWK (mean ± sd) | accuracy |",
        "| --- | --- | --- | --- | --- | --- |",
    ]
    for row in rows:
        lines.append(
            f"| {row['model']} | {row['encoder']} | {row['n_slides']} | {row['n_seeds']} | "
            f"**{_fmt(row['qwk_mean'])} ± {_fmt(row['qwk_sd'])}** | "
            f"{_fmt(row['acc_mean'])} ± {_fmt(row['acc_sd'])} |"
        )
    lines.append("")
    return "\n".join(lines)


def render_paired(entries: Sequence[Tuple[str, str, Dict[str, Any]]]) -> str:
    lines = [
        "# Paired comparisons (identical slides, identical seeds)",
        "",
        "Paired bootstrap CI on ΔQWK (2000 resamples) and the exact McNemar test on "
        "per-slide correctness. A CI that spans 0 means *inconclusive at this sample "
        "size*, not *equivalent*.",
        "",
        "| comparison | seed | n | QWK A | QWK B | ΔQWK | 95% CI | McNemar p | right/wrong |",
        "| --- | --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    for label_a, label_b, stats in entries:
        for row in stats["per_seed"]:
            lines.append(
                f"| {label_a} → {label_b} | {row['seed']} | {row['n']} | {_fmt(row['qwk_a'])} | "
                f"{_fmt(row['qwk_b'])} | {row['delta']:+.4f} | "
                f"[{row['lo']:+.4f}, {row['hi']:+.4f}] | {_fmt(row['mcnemar_p'])} | "
                f"{row['a_right_b_wrong']}/{row['a_wrong_b_right']} |"
            )
    lines.append("")
    return "\n".join(lines)


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #
def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Regenerate paired tables from committed predictions.")
    parser.add_argument("--results", default="results")
    parser.add_argument("--out", default="results/metrics/comparison")
    parser.add_argument("--reference", default=None, help="run folder used as 'A' in paired stats")
    parser.add_argument("--n-boot", type=int, default=2000)
    args = parser.parse_args(argv)

    results = Path(args.results)
    if not results.is_dir():
        print(f"ERROR: results folder not found: {results}", file=sys.stderr)
        return 1

    runs = discover_runs(results)
    if len(runs) < 2:
        print("ERROR: need at least two runs with predictions_*_seed*.csv", file=sys.stderr)
        return 1

    # ---- native cohort (each run on its own test set) ----------------------
    native_rows = []
    for name, seeds in runs.items():
        scores = scores_on(seeds)
        if not scores:
            continue
        native_rows.append(
            {
                "run": name,
                "model": MODEL_LABELS.get(name, name),
                "encoder": ENCODER_LABELS.get(name, "unknown"),
                **scores,
            }
        )
    native_rows.sort(key=lambda row: row["qwk_mean"], reverse=True)

    # ---- common cohort (all runs on the slides they all share) ------------
    common = set.intersection(*[set(seeds[min(seeds)]) for seeds in runs.values()]) if runs else set()
    common_rows = []
    if common:
        for name, seeds in runs.items():
            scores = scores_on(seeds, sorted(common))
            if not scores:
                continue
            common_rows.append(
                {
                    "run": name,
                    "model": MODEL_LABELS.get(name, name),
                    "encoder": ENCODER_LABELS.get(name, "unknown"),
                    **scores,
                }
            )
        common_rows.sort(key=lambda row: row["qwk_mean"], reverse=True)

    # ---- paired stats ----------------------------------------------------
    # The two scientific questions are (a) does the aggregator matter and
    # (b) does the encoder matter, so pair runs that differ in exactly one of
    # them. An explicit --reference adds every other run against that baseline.
    reference = args.reference or "abmil"
    if reference not in runs:
        reference = next(iter(runs))

    pairs: List[Tuple[str, str]] = []
    names = sorted(runs)
    for i, name_a in enumerate(names):
        for name_b in names[i + 1:]:
            same_encoder = ENCODER_LABELS.get(name_a) == ENCODER_LABELS.get(name_b)
            same_model = MODEL_LABELS.get(name_a) == MODEL_LABELS.get(name_b)
            if (same_encoder and not same_model) or (same_model and not same_encoder):
                pairs.append((name_a, name_b))
    for name in names:
        if name != reference and (reference, name) not in pairs and (name, reference) not in pairs:
            pairs.append((reference, name))

    entries = []
    for name_a, name_b in pairs:
        stats = paired_stats(runs[name_a], runs[name_b], n_boot=args.n_boot)
        if stats:
            entries.append((name_a, name_b, stats))

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    native_md = render_table(
        native_rows,
        "Every run on its own test set",
        "Read the `n slides` column: 60 for the ResNet50 cohort, 57 for CHIEF, 53 for "
        "GigaPath. **These rows are not all measured on the same slides** — use the "
        "common-cohort table below for any cross-encoder claim.",
    )
    common_md = render_table(
        common_rows,
        "Every run on the slides all runs share",
        f"Restricted to the {len(common)} test slides covered by every run, so each row "
        "is measured on identical slides and identical labels. This is the only table "
        "that supports a like-for-like comparison.",
    ) if common_rows else "# Common cohort\n\n_No slide set is shared by all runs._\n"
    paired_md = render_paired(entries) if entries else "# Paired comparisons\n\n_None available._\n"

    (out / "paired_native_cohort.md").write_text(native_md, encoding="utf-8")
    (out / "paired_common_cohort.md").write_text(common_md, encoding="utf-8")
    (out / "paired_stats.md").write_text(paired_md, encoding="utf-8")

    with (out / "paired_native_cohort.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["run", "model", "encoder", "n_slides", "n_seeds", "qwk_mean", "qwk_sd", "acc_mean", "acc_sd"],
        )
        writer.writeheader()
        for row in native_rows:
            writer.writerow({key: row[key] for key in writer.fieldnames})
    with (out / "paired_common_cohort.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["run", "model", "encoder", "n_slides", "n_seeds", "qwk_mean", "qwk_sd", "acc_mean", "acc_sd"],
        )
        writer.writeheader()
        for row in common_rows:
            writer.writerow({key: row[key] for key in writer.fieldnames})

    (out / "paired_stats.json").write_text(
        json.dumps(
            {
                "reference": reference,
                "native": native_rows,
                "common": common_rows,
                "n_common_slides": len(common),
                "paired": [
                    {"a": a, "b": b, **stats} for a, b, stats in entries
                ],
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    print(f"runs found: {len(runs)} ({', '.join(sorted(runs))})")
    print(f"common-cohort slides: {len(common)}")
    print()
    print(native_md)
    print(common_md)
    print(paired_md)
    print(f"wrote 5 files -> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

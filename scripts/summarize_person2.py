"""Assemble every Person 2 number into one report-ready summary.

Reads whatever results exist under ``results/`` and writes
``results/PERSON2_SUMMARY.md`` containing:

* ablation table (CLAM-SB with vs without instance loss, mean +/- sd over seeds)
* ABMIL baseline row for context
* **paired** statistics: CLAM-SB+inst vs CLAM-SB(no inst), and CLAM-SB+inst vs ABMIL,
  per seed (paired bootstrap CI for dQWK + exact McNemar p), with the mean delta
* attention sharpness table (entropy / top-10 share) for both ablation arms
* localisation coverage (how many test slides got heatmaps, and the known gap)
* statistical baselines (ordinal logistic / XGBoost) if present

Missing inputs are reported as "not run yet" instead of crashing, so it is safe to
call at any point in the pipeline.

Usage::

    python scripts/summarize_person2.py
    python scripts/summarize_person2.py --results results --out results/PERSON2_SUMMARY.md
"""

from __future__ import annotations

import argparse
import csv
import glob
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

ROOT = Path(__file__).resolve().parents[1]

#: folder-name fragments that mark a "without instance loss" evaluation run
WITHOUT_TAGS = ("without", "noinst", "no_inst", "inst0", "lambda0", "lam0", "l0")
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


# --------------------------------------------------------------------------- #
# small helpers
# --------------------------------------------------------------------------- #
def load_json(path: Path) -> Optional[Dict[str, Any]]:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        return None
    except json.JSONDecodeError:
        return None


def read_predictions(path: Path) -> Dict[str, Tuple[int, int]]:
    table: Dict[str, Tuple[int, int]] = {}
    with path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            slide_id = row.get("slide_id") or row.get("image_id")
            if not slide_id:
                continue
            table[slide_id] = (int(float(row["y_true"])), int(float(row["y_pred"])))
    return table


def mean_sd(values: Sequence[float]) -> Tuple[float, float]:
    import numpy as np

    array = np.asarray([v for v in values if v == v], dtype=float)
    if array.size == 0:
        return float("nan"), float("nan")
    sd = float(array.std(ddof=1)) if array.size > 1 else 0.0
    return float(array.mean()), sd


def fmt(value: Any, digits: int = 4) -> str:
    if isinstance(value, float):
        return "n/a" if value != value else f"{value:.{digits}f}"
    return str(value)


# --------------------------------------------------------------------------- #
# data collection
# --------------------------------------------------------------------------- #
def collect_runs(results: Path) -> Dict[str, Any]:
    """Load the three aggregate files and every per-seed prediction CSV."""
    claw_dir = results / "metrics" / "clam_sb"
    runs: Dict[str, Any] = {"aggregates": {}, "predictions": {}, "attention": {}, "eval_metrics": {}}

    for name, path in (
        ("clam_with", claw_dir / "with_instance" / "aggregate_clam_sb.json"),
        ("clam_without", claw_dir / "without_instance" / "aggregate_clam_sb.json"),
        ("abmil", results / "metrics" / "abmil" / "aggregate_abmil.json"),
    ):
        payload = load_json(path)
        if payload:
            runs["aggregates"][name] = payload

    for name, pattern in (
        ("clam_with", str(claw_dir / "with_instance" / "predictions_clam_sb_seed*.csv")),
        ("clam_without", str(claw_dir / "without_instance" / "predictions_clam_sb_seed*.csv")),
        ("abmil", str(results / "metrics" / "abmil" / "predictions_abmil_seed*.csv")),
    ):
        runs["predictions"][name] = {}
        for file_path in sorted(glob.glob(pattern)):
            seed = int(Path(file_path).stem.rsplit("seed", 1)[-1])
            try:
                runs["predictions"][name][seed] = read_predictions(Path(file_path))
            except Exception:  # noqa: BLE001
                continue

    # Attention CSVs live in ``eval*`` folders whose names the user chooses
    # (eval_seed0, eval_without_seed0, eval_noinst_seed0, ...), so classify each
    # folder by its name instead of hard-coding one spelling.
    runs["attention"] = {"clam_with": {}, "clam_without": {}}
    for file_path in sorted(glob.glob(str(claw_dir / "eval*" / "attention_*.csv"))):
        payload = read_attention_csv(Path(file_path))
        if not payload:
            continue
        folder = Path(file_path).parent.name.lower()
        key = "clam_without" if any(tag in folder for tag in WITHOUT_TAGS) else "clam_with"
        runs["attention"][key][Path(file_path).stem] = payload

    for file_path in sorted(glob.glob(str(claw_dir / "eval*" / "metrics_*.json"))):
        payload = load_json(Path(file_path))
        if payload:
            runs["eval_metrics"][Path(file_path).stem] = payload
    return runs


def read_attention_csv(path: Path) -> Optional[Dict[str, float]]:
    """Average the attention summary columns over the slides in one eval CSV."""
    import numpy as np

    columns = ("attention_entropy", "normalised_entropy", "top10_attention_share")
    values: Dict[str, List[float]] = {c: [] for c in columns}
    try:
        with path.open(newline="", encoding="utf-8") as handle:
            for row in csv.DictReader(handle):
                for column in columns:
                    raw = row.get(column)
                    if raw in (None, ""):
                        continue
                    try:
                        values[column].append(float(raw))
                    except ValueError:
                        continue
    except FileNotFoundError:
        return None
    if not values["attention_entropy"]:
        return None
    return {c: float(np.mean(v)) if v else float("nan") for c, v in values.items()}


def collect_localisation(results: Path) -> List[Dict[str, Any]]:
    entries: List[Dict[str, Any]] = []
    for path in sorted(glob.glob(str(results / "heatmaps" / "*" / "export_summary.json"))):
        payload = load_json(Path(path))
        if payload:
            entries.append(payload)
    return entries


def collect_statistical(results: Path) -> Optional[Dict[str, Any]]:
    return load_json(results / "metrics" / "statistical" / "statistical_baselines.json")


# --------------------------------------------------------------------------- #
# paired statistics
# --------------------------------------------------------------------------- #
def paired_block(
    pred_a: Dict[int, Dict[str, Tuple[int, int]]],
    pred_b: Dict[int, Dict[str, Tuple[int, int]]],
    label_a: str,
    label_b: str,
    n_boot: int = 2000,
) -> List[Dict[str, Any]]:
    """Per-seed paired dQWK (b - a) + McNemar, aligned on shared slide ids."""
    from src.evaluate import mcnemar_test, paired_bootstrap_difference, quadratic_weighted_kappa

    rows: List[Dict[str, Any]] = []
    for seed in sorted(set(pred_a) & set(pred_b)):
        shared = sorted(set(pred_a[seed]) & set(pred_b[seed]))
        if not shared:
            continue
        y_true = [pred_a[seed][s][0] for s in shared]
        y_a = [pred_a[seed][s][1] for s in shared]
        y_b = [pred_b[seed][s][1] for s in shared]
        try:
            difference = paired_bootstrap_difference(
                y_true, y_a, y_b, metric=quadratic_weighted_kappa, n_boot=n_boot, seed=seed
            )
            mcnemar = mcnemar_test(y_true, y_a, y_b)
        except Exception:  # noqa: BLE001
            continue
        rows.append(
            {
                "seed": seed,
                "n": len(shared),
                "qwk_a": quadratic_weighted_kappa(y_true, y_a),
                "qwk_b": quadratic_weighted_kappa(y_true, y_b),
                "delta": difference["difference"],
                "lo": difference["lo"],
                "hi": difference["hi"],
                "mcnemar_p": mcnemar["p_value"],
                "a_b": f"{mcnemar['a_correct_b_wrong']}/{mcnemar['a_wrong_b_correct']}",
                "a": label_a,
                "b": label_b,
            }
        )
    return rows


# --------------------------------------------------------------------------- #
# markdown assembly
# --------------------------------------------------------------------------- #
def build_markdown(runs: Dict[str, Any], localisation: List[Dict[str, Any]], statistical: Optional[Dict[str, Any]]) -> str:
    lines: List[str] = [
        "# Person 2 — CLAM-SB results summary",
        "",
        "_Generated by `scripts/summarize_person2.py` from `results/`. Re-run after every "
        "new training run so the report never drifts from the artifacts._",
        "",
        "**Headline question:** does forcing the model to separate high- and low-attention "
        "patches improve weakly supervised ISUP grading?",
        "",
    ]

    # ---- 1. ablation table -------------------------------------------------
    lines += ["## 1. Instance-loss ablation (test split, 60 slides)", ""]
    rows = []
    for key, label in (("clam_with", "CLAM-SB + instance loss"), ("clam_without", "CLAM-SB, no instance loss")):
        agg = runs["aggregates"].get(key)
        if not agg:
            rows.append({"setting": label, "qwk": "not run", "sd": "", "acc": "", "bacc": "", "f1": "", "seeds": ""})
            continue
        rows.append(
            {
                "setting": label,
                "qwk": agg.get("qwk", {}).get("mean"),
                "sd": agg.get("qwk", {}).get("std"),
                "acc": agg.get("accuracy", {}).get("mean"),
                "bacc": agg.get("balanced_accuracy", {}).get("mean"),
                "f1": agg.get("macro_f1", {}).get("mean"),
                "seeds": agg.get("n_seeds", ""),
            }
        )
    abmil = runs["aggregates"].get("abmil")
    if abmil:
        rows.append(
            {
                "setting": "ABMIL baseline (Person 1)",
                "qwk": abmil.get("qwk", {}).get("mean"),
                "sd": abmil.get("qwk", {}).get("std"),
                "acc": abmil.get("accuracy", {}).get("mean"),
                "bacc": abmil.get("balanced_accuracy", {}).get("mean"),
                "f1": abmil.get("macro_f1", {}).get("mean"),
                "seeds": abmil.get("n_seeds", ""),
            }
        )

    lines += [
        "| Setting | QWK (mean) | QWK (sd) | Accuracy | Balanced acc. | Macro F1 | Seeds |",
        "| --- | --- | --- | --- | --- | --- | --- |",
    ]
    for row in rows:
        lines.append(
            f"| {row['setting']} | {fmt(row['qwk'])} | {fmt(row['sd'])} | {fmt(row['acc'])} | "
            f"{fmt(row['bacc'])} | {fmt(row['f1'])} | {row['seeds']} |"
        )

    with_agg = runs["aggregates"].get("clam_with", {})
    without_agg = runs["aggregates"].get("clam_without", {})
    if with_agg and without_agg:
        delta = with_agg.get("qwk", {}).get("mean", float("nan")) - without_agg.get("qwk", {}).get("mean", float("nan"))
        lines += ["", f"Mean QWK delta (with − without) = **{delta:+.4f}**"]
    if abmil and with_agg:
        delta_abmil = with_agg.get("qwk", {}).get("mean", float("nan")) - abmil.get("qwk", {}).get("mean", float("nan"))
        lines.append(f"Mean QWK delta (CLAM-SB − ABMIL) = **{delta_abmil:+.4f}**")

    # ---- 2. paired statistics ---------------------------------------------
    lines += ["", "## 2. Paired statistics (same slides, same seed)", ""]
    comparisons = [
        ("clam_without", "clam_with", "no instance loss", "with instance loss"),
        ("abmil", "clam_with", "ABMIL", "CLAM-SB + instance loss"),
    ]
    any_comparison = False
    for left, right, label_left, label_right in comparisons:
        block = paired_block(
            runs["predictions"].get(left, {}), runs["predictions"].get(right, {}), label_left, label_right
        )
        if not block:
            continue
        any_comparison = True
        lines += [
            f"### {label_right} vs {label_left}",
            "",
            "| seed | n | QWK (" + label_left + ") | QWK (" + label_right + ") | ΔQWK | 95% CI | McNemar p | discordant (A/B) |",
            "| --- | --- | --- | --- | --- | --- | --- | --- |",
        ]
        for row in block:
            lines.append(
                f"| {row['seed']} | {row['n']} | {fmt(row['qwk_a'])} | {fmt(row['qwk_b'])} | "
                f"{row['delta']:+.4f} | [{row['lo']:+.4f}, {row['hi']:+.4f}] | {fmt(row['mcnemar_p'])} | {row['a_b']} |"
            )
        mean_delta, sd_delta = mean_sd([row["delta"] for row in block])
        lines += ["", f"Mean ΔQWK over seeds = **{mean_delta:+.4f}** (sd {sd_delta:.4f}); "
                      f"CIs spanning 0 mean the difference is inconclusive at this sample size.", ""]
    if not any_comparison:
        lines += ["_No paired prediction files found yet — run the comparison steps._", ""]

    # ---- 3. attention sharpness -------------------------------------------
    lines += ["## 3. Attention sharpness (does instance loss focus the model?)", ""]
    attention = runs["attention"]
    if attention.get("clam_with") or attention.get("clam_without"):
        lines += [
            "| Arm | attention entropy (nats) | normalised entropy | top-10 attention share |",
            "| --- | --- | --- | --- |",
        ]
        for key, label in (("clam_with", "CLAM-SB + instance loss"), ("clam_without", "CLAM-SB, no instance loss")):
            for name, payload in sorted(attention.get(key, {}).items()):
                lines.append(
                    f"| {label} ({name.split('_seed')[-1]}) | {fmt(payload.get('attention_entropy'))} | "
                    f"{fmt(payload.get('normalised_entropy'))} | {fmt(payload.get('top10_attention_share'))} |"
                )
        lines += ["", "Lower entropy / higher top-10 share = the model spreads attention over fewer patches.", ""]
    else:
        lines += ["_No attention CSVs found yet — run `src.evaluate_clam_sb --split all` for both arms._", ""]

    # ---- 4. localisation ---------------------------------------------------
    lines += ["## 4. Heatmaps and localisation coverage", ""]
    if localisation:
        lines += ["| export | split | slides | skipped (no coords) | mean normalised entropy |", "| --- | --- | --- | --- | --- |"]
        for entry in localisation:
            lines.append(
                f"| {Path(entry.get('out_dir', '?')).name} | {entry.get('split', '?')} | "
                f"{entry.get('n_slides', '?')} | {entry.get('n_skipped_no_coords', 0)} | "
                f"{fmt(entry.get('mean_normalised_entropy'))} |"
            )
        lines += [
            "",
            "**Data limitation to state in the report:** the pre-extracted feature release has "
            "patch coordinates for 399/420 slides; 4 of the missing slides are in the test split. "
            "Heatmaps therefore cover 56/60 test slides, and no quantitative overlap score is "
            "possible because the feature-only pipeline has no slide thumbnails or tumour masks.",
            "",
        ]
    else:
        lines += ["_No heatmap exports found yet._", ""]

    # ---- 5. statistical baselines -----------------------------------------
    lines += ["## 5. Statistical baselines (Person 4) on attention statistics", ""]
    if statistical:
        for name in ("ordinal_logistic", "xgboost"):
            payload = statistical.get(name, {})
            if "qwk" in payload:
                lines.append(
                    f"- **{name}**: QWK {fmt(payload['qwk'])}, accuracy {fmt(payload.get('accuracy'))}, "
                    f"macro F1 {fmt(payload.get('macro_f1'))}"
                )
            elif payload.get("skipped"):
                lines.append(f"- {name}: skipped ({payload['skipped']})")
        lines.append("")
    else:
        lines += ["_Not run yet: `python -m src.statistical_models --attention-csv <eval csv>`._", ""]

    # ---- 6. interpretation skeleton ---------------------------------------
    lines += [
        "## 6. Interpretation (fill in after reading the tables)",
        "",
        "1. Did instance loss change grading? Compare the two ablation rows **and** the paired "
        "ΔQWK CI. If the CI spans zero, write \u201cno detectable effect at n=60\u201d, not \u201cno effect\u201d.",
        "2. Did instance loss sharpen attention? Compare the entropy / top-10 share rows.",
        "3. Is CLAM-SB distinguishable from ABMIL? Report the paired ΔQWK CI and McNemar p.",
        "4. Always print **accuracy next to QWK**: QWK around 0.7 with accuracy around 0.4 means "
        "the model is mostly getting the ordinal direction right while collapsing to the extremes.",
        "5. State the sample-size caveat: 60 test slides, 3 seeds, 300 training slides, one feature set.",
        "",
    ]
    return "\n".join(lines)


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Summarise all Person 2 results into one markdown file.")
    parser.add_argument("--results", default="results")
    parser.add_argument("--out", default="results/PERSON2_SUMMARY.md")
    parser.add_argument("--n-boot", type=int, default=2000)
    args = parser.parse_args(argv)

    results = Path(args.results)
    if not results.is_dir():
        print(f"ERROR: results folder not found: {results}", file=sys.stderr)
        return 1

    runs = collect_runs(results)
    markdown = build_markdown(runs, collect_localisation(results), collect_statistical(results))

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(markdown, encoding="utf-8")

    found = sum(1 for key in ("clam_with", "clam_without", "abmil") if key in runs["aggregates"])
    print(f"wrote {out}  (aggregate files found: {found}/3)")
    if found < 3:
        print("  hint: some runs are missing -- the summary marks them as 'not run'.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

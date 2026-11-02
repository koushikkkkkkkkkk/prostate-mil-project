"""Assemble every committed result into one generated `results/ALL_RESULTS.md`.

Why: `docs/RESULTS_LOG.md` was hand-edited and drifted from the artifacts (TransMIL
was listed as "pending" after it had been trained; the statistical baseline was
quoted as 0.7682 while the artifact says 0.7399). A results table that is typed by
hand is a results table that will eventually lie. This script reads only files
under `results/` and writes the table, including a *provenance* column and a
**weights committed?** column so the gap between "we have numbers" and "you can
reproduce them" is visible in the repo itself.

Regenerate after every run:

    python scripts/summarize_results.py
    python scripts/summarize_results.py --results results --out results/ALL_RESULTS.md
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #
def load_json(path: Path) -> Optional[Any]:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:  # noqa: BLE001
        return None


def mean_sd(payload: Optional[Dict[str, Any]], key: str) -> str:
    if not payload:
        return "—"
    value = payload.get(key)
    if isinstance(value, dict):
        mean, std = value.get("mean"), value.get("std")
        if mean is None:
            return "—"
        return f"{mean:.4f} ± {float(std or 0.0):.4f}"
    if isinstance(value, (int, float)):
        return f"{float(value):.4f}"
    return "—"


def run_config_for(run_dir: Path) -> Dict[str, Any]:
    """run_config.json for a run, looking one level up for nested ablation arms."""
    for candidate in (run_dir / "run_config.json", run_dir.parent / "run_config.json"):
        config = load_json(candidate)
        if config:
            return config
    return {}


def rel_to_root(path: Path) -> str:
    """Display path relative to the repo, tolerating a relative --results arg."""
    try:
        return str(path.relative_to(ROOT))
    except ValueError:
        return str(path)


def checkpoint_state(run_dir: Path) -> str:
    """Are the weights for this run actually in the repo?"""
    checkpoints = sorted(run_dir.glob("checkpoints/*.pt")) + sorted(run_dir.glob("*.pt"))
    if checkpoints:
        return f"✅ {len(checkpoints)}"
    candidates = [run_dir / "checkpoints", run_dir] + sorted(run_dir.glob("*/checkpoints"))
    for candidate in candidates:
        if candidate.is_dir() and list(candidate.glob("*.pt")):
            return "❌ (not this run)"
    return "❌ none"


# --------------------------------------------------------------------------- #
# Collectors
# --------------------------------------------------------------------------- #
def collect_runs(results: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    metrics_root = results / "metrics"
    # recursive: the CLAM-SB ablation lives at clam_sb/with_instance/ etc., and a
    # one-level scan would leave both arms out of the results table entirely
    for aggregate_path in sorted(metrics_root.rglob("aggregate_*.json")):
        run_dir = aggregate_path.parent
        aggregate = load_json(aggregate_path) or {}
        config = run_config_for(run_dir)
        model_config = config.get("model_config", {}) or {}
        rows.append(
            {
                "run": str(run_dir.relative_to(metrics_root)).replace("\\", "/"),
                "model": config.get("model", run_dir.name),
                "in_dim": model_config.get("in_dim", "?"),
                "manifest": Path(str(config.get("manifest", "?"))).name,
                "features": Path(str(config.get("features_root", "?"))).name,
                "n_test": config.get("n_test", "?"),
                "seeds": aggregate.get("n_seeds", "?"),
                "qwk": mean_sd(aggregate, "qwk"),
                "acc": mean_sd(aggregate, "accuracy"),
                "bal_acc": mean_sd(aggregate, "balanced_accuracy"),
                "f1": mean_sd(aggregate, "macro_f1"),
                "weights": checkpoint_state(run_dir),
                "per_seed": (aggregate.get("qwk") or {}).get("values", []),
            }
        )
    return rows


def collect_ablations(results: Path) -> List[str]:
    lines: List[str] = []
    for path in sorted((results / "metrics").glob("**/ablation_*.md")):
        lines.append(f"#### `{rel_to_root(path)}`\n")
        lines.append(path.read_text(encoding="utf-8").strip())
        lines.append("")
    if not lines:
        lines.append(
            "_No `ablation_*.md` found. The mandatory CLAM-SB instance-loss ablation has no "
            "committed artifacts yet — run `python -m src.train_clam_sb --seeds 0 1 2` and "
            "commit `results/metrics/clam_sb/`._\n"
        )
    return lines


def collect_statistical(results: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for path in sorted((results / "metrics" / "statistical").glob("*/statistical_baselines.json")):
        payload = load_json(path) or {}
        for name in ("ordinal_logistic", "xgboost"):
            entry = payload.get(name)
            if not isinstance(entry, dict):
                continue
            if "qwk" in entry:
                rows.append(
                    {
                        "source": path.parent.name,
                        "model": name,
                        "qwk": f"{float(entry['qwk']):.4f}",
                        "accuracy": f"{float(entry.get('accuracy', float('nan'))):.4f}",
                        "f1": f"{float(entry.get('macro_f1', float('nan'))):.4f}",
                    }
                )
            elif entry.get("skipped"):
                rows.append({"source": path.parent.name, "model": name, "qwk": "skipped", "accuracy": entry["skipped"], "f1": "—"})
    return rows


def collect_heatmaps(results: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for path in sorted((results / "heatmaps").glob("*/export_summary.json")):
        payload = load_json(path) or {}
        images = len(list((path.parent / "images").glob("*.png"))) if (path.parent / "images").is_dir() else 0
        rows.append(
            {
                "export": path.parent.name,
                "checkpoint": payload.get("checkpoint", "?"),
                "n_slides": payload.get("n_slides", "?"),
                "images": images,
                "entropy": payload.get("mean_normalised_entropy"),
                "localisation": (payload.get("localisation") or {}).get("available", False),
            }
        )
    return rows


def collect_attention(results: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for path in sorted((results / "metrics" / "attention").glob("*/export_summary.json")):
        payload = load_json(path) or {}
        rows.append(
            {
                "arm": path.parent.name,
                "n_slides": payload.get("n_slides", "?"),
                "normalised_entropy": payload.get("mean_normalised_entropy"),
                "top10_share": payload.get("mean_top10_attention_share"),
            }
        )
    return rows


# --------------------------------------------------------------------------- #
# Render
# --------------------------------------------------------------------------- #
def build_markdown(results: Path) -> str:
    runs = collect_runs(results)
    statistical = collect_statistical(results)
    heatmaps = collect_heatmaps(results)
    attention = collect_attention(results)
    patch_cnn = load_json(results / "metrics" / "patch_cnn_demo" / "patch_cnn_summary.json") or load_json(
        results / "metrics" / "patch_cnn_summary.json"
    )

    lines: List[str] = [
        "# All committed results",
        "",
        "_Generated by `python scripts/summarize_results.py` — **do not edit by hand**. "
        "Every number below is read from a file under `results/`; if a row is missing, the "
        "artifact is missing._",
        "",
        "## 1. Trained runs (test split, 3 seeds)",
        "",
        "Read the `n test` column: 60 slides for the ResNet50 cohort, 57 for CHIEF, 53 for "
        "GigaPath. Rows with different `n test` are **not** measured on the same slides — use "
        "`paired_common_cohort.md` for any cross-encoder comparison.",
        "",
        "| run | model | in_dim | manifest | n test | seeds | QWK (mean ± sd) | accuracy | balanced acc. | macro F1 | weights in repo |",
        "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    for row in sorted(runs, key=lambda r: str(r["run"])):
        lines.append(
            f"| `{row['run']}` | {row['model']} | {row['in_dim']} | {row['manifest']} | {row['n_test']} | "
            f"{row['seeds']} | **{row['qwk']}** | {row['acc']} | {row['bal_acc']} | {row['f1']} | {row['weights']} |"
        )
    lines += ["", "Per-seed QWK:", ""]
    for row in sorted(runs, key=lambda r: str(r["run"])):
        values = ", ".join(f"{float(v):.4f}" for v in row["per_seed"]) or "—"
        lines.append(f"- `{row['run']}`: {values}")
    lines.append("")

    lines += [
        "## 2. Paired comparisons (identical slides and seeds)",
        "",
        "_Regenerated from the prediction CSVs by `scripts/paired_results.py`._",
        "",
    ]
    for name in ("paired_common_cohort.md", "paired_stats.md"):
        path = results / "metrics" / "comparison" / name
        if path.is_file():
            text = path.read_text(encoding="utf-8")
            body = "\n".join(
                line for line in text.splitlines() if not line.startswith("# ")
            ).strip()
            lines += [f"### `{name}`", "", body, ""]
        else:
            lines += [f"_`{name}` not found — run `python scripts/paired_results.py`._", ""]

    lines += ["## 3. Ablation (mandatory: CLAM-SB with vs without instance loss)", ""]
    lines += collect_ablations(results)

    lines += ["## 4. Statistical baselines (on attention statistics)", ""]
    if statistical:
        lines += ["| source arm | model | QWK | accuracy | macro F1 |", "| --- | --- | --- | --- | --- |"]
        for row in statistical:
            lines.append(f"| {row['source']} | {row['model']} | **{row['qwk']}** | {row['accuracy']} | {row['f1']} |")
        lines += [
            "",
            "_These consume a MIL model's own attention statistics, so they are **not** "
            "independent baselines — state that when quoting them._",
            "",
        ]
    else:
        lines += ["_No `statistical/*/statistical_baselines.json` found._", ""]

    lines += ["## 5. Heatmaps and attention", ""]
    if heatmaps:
        lines += [
            "| export | n slides | png files | mean normalised entropy | checkpoint | localisation score |",
            "| --- | --- | --- | --- | --- | --- |",
        ]
        for row in heatmaps:
            entropy = f"{float(row['entropy']):.4f}" if isinstance(row["entropy"], (int, float)) else "—"
            lines.append(
                f"| `{row['export']}` | {row['n_slides']} | {row['images']} | {entropy} | "
                f"`{row['checkpoint']}` | {'yes' if row['localisation'] else 'not available (no masks)'} |"
            )
        lines.append("")
    if attention:
        lines += [
            "| attention arm | n slides | normalised entropy | top-10 attention share |",
            "| --- | --- | --- | --- |",
        ]
        for row in attention:
            ent = f"{float(row['normalised_entropy']):.4f}" if isinstance(row["normalised_entropy"], (int, float)) else "—"
            top = f"{float(row['top10_share']):.4f}" if isinstance(row["top10_share"], (int, float)) else "—"
            lines.append(f"| `{row['arm']}` | {row['n_slides']} | {ent} | {top} |")
        lines.append("")

    lines += [
        "## 6. Not reportable",
        "",
        "Synthetic (`mock_*`) runs and the toy patch-CNN demo live outside the tables above "
        "and must never be quoted:",
        "",
    ]
    if patch_cnn:
        lines.append(
            f"- **PatchCNN demo** (`results/metrics/patch_cnn_demo/`): {patch_cnn.get('val_samples', '?')} "
            f"validation tiles, best tile accuracy {patch_cnn.get('best_val_accuracy', '?')}, "
            f"{len(patch_cnn.get('history', []))} epochs. Different data, different task, "
            "different unit of prediction — not comparable to the run table."
        )
    lines.append("- Any `results/mock_*/` folder (synthetic slide generation).")
    lines.append("")

    return "\n".join(lines)


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Generate results/ALL_RESULTS.md from committed artifacts.")
    parser.add_argument("--results", default="results")
    parser.add_argument("--out", default="results/ALL_RESULTS.md")
    args = parser.parse_args(argv)

    results = Path(args.results)
    if not results.is_absolute():
        results = (Path.cwd() / results).resolve()
    if not results.is_dir():
        print(f"ERROR: results folder not found: {results}", file=sys.stderr)
        return 1

    markdown = build_markdown(results)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(markdown, encoding="utf-8")
    runs = len(collect_runs(results))
    print(f"wrote {out}  ({runs} trained runs, {len(collect_statistical(results))} statistical rows, "
          f"{len(collect_heatmaps(results))} heatmap exports)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

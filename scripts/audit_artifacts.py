"""Audit the results tree: is every committed artifact reproducible?

This is the check that would have caught the project's three biggest documentation
drifts before a reader did:

* a heatmap export whose `export_summary.json` points at a checkpoint that is **not
  in the repository**;
* the mandatory CLAM-SB ablation quoted in prose while no `results/metrics/clam_sb/`
  exists;
* trained runs whose weights were never committed, so nobody can re-evaluate them.

It exits non-zero only for **fatal** findings (dangling checkpoint references), and
prints warnings for the rest. Run it before any demo, submission or viva:

    python scripts/audit_artifacts.py
    python scripts/audit_artifacts.py --strict      # warnings become failures
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

FAIL = "FAIL"
WARN = "WARN"
OK = "OK"


def _load(path: Path) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:  # noqa: BLE001
        return None


# --------------------------------------------------------------------------- #
# Checks
# --------------------------------------------------------------------------- #
def check_checkpoint_references(results: Path) -> List[Tuple[str, str, str]]:
    """Every export summary must point at a checkpoint that exists in the repo."""
    findings: List[Tuple[str, str, str]] = []
    for path in sorted(results.rglob("export_summary.json")):
        payload = _load(path) or {}
        checkpoint = payload.get("checkpoint")
        if not checkpoint:
            continue
        candidate = ROOT / str(checkpoint).replace("\\", "/")
        if candidate.exists():
            findings.append((OK, str(path.relative_to(ROOT)), f"checkpoint present: {checkpoint}"))
        else:
            findings.append((FAIL, str(path.relative_to(ROOT)), f"DANGLING checkpoint reference: {checkpoint}"))
    return findings


def check_run_weights(results: Path) -> List[Tuple[str, str, str]]:
    """Each trained run should ship the weights its metrics came from."""
    findings: List[Tuple[str, str, str]] = []
    for run_dir in sorted((results / "metrics").iterdir()):
        if not run_dir.is_dir():
            continue
        if not list(run_dir.glob("aggregate_*.json")):
            continue
        weights = list(run_dir.glob("checkpoints/*.pt")) + list(run_dir.glob("*.pt"))
        rel = str(run_dir.relative_to(ROOT))
        if not weights:
            findings.append((WARN, rel, "trained run with no committed weights -- metrics cannot be reproduced or re-evaluated"))
        else:
            findings.append((OK, rel, f"{len(weights)} checkpoint file(s)"))
    return findings


def check_ablation(results: Path) -> List[Tuple[str, str, str]]:
    """The mandatory Person 2 ablation must have artifacts."""
    findings: List[Tuple[str, str, str]] = []
    ablation_files = list(results.glob("metrics/**/ablation_*.md")) + list(results.glob("metrics/**/ablation_*.json"))
    arms = [p.name for p in (results / "metrics").glob("*/with_instance")] + [
        p.name for p in (results / "metrics").glob("*/without_instance")
    ]
    if ablation_files:
        findings.append((OK, "ablation", f"{len(ablation_files)} ablation artifact(s)"))
    elif arms or (results / "metrics" / "clam_sb").exists():
        findings.append((OK, "ablation", "arm folders present (with_instance / without_instance)"))
    else:
        findings.append(
            (WARN, "ablation", "NO ablation artifacts -- the with/without instance-loss results quoted in docs are unverified")
        )
    return findings


def check_docs_point_at_generated_tables(results: Path) -> List[Tuple[str, str, str]]:
    findings: List[Tuple[str, str, str]] = []
    log = ROOT / "docs" / "RESULTS_LOG.md"
    if not log.is_file():
        return [(WARN, "docs/RESULTS_LOG.md", "missing")]
    text = log.read_text(encoding="utf-8")
    if "ALL_RESULTS.md" in text:
        findings.append((OK, "docs/RESULTS_LOG.md", "points at the generated results table"))
    else:
        findings.append((WARN, "docs/RESULTS_LOG.md", "does not reference results/ALL_RESULTS.md -- numbers may be hand-typed"))
    if (results / "ALL_RESULTS.md").is_file():
        findings.append((OK, "results/ALL_RESULTS.md", "generated table present"))
    else:
        findings.append((WARN, "results/ALL_RESULTS.md", "missing -- run python scripts/summarize_results.py"))
    return findings


def check_predictions_correspond_to_runs(results: Path) -> List[Tuple[str, str, str]]:
    findings: List[Tuple[str, str, str]] = []
    for run_dir in sorted((results / "metrics").iterdir()):
        if not run_dir.is_dir():
            continue
        aggregates = list(run_dir.glob("aggregate_*.json"))
        predictions = list(run_dir.glob("predictions_*_seed*.csv"))
        if aggregates and not predictions:
            findings.append((WARN, str(run_dir.relative_to(ROOT)), "aggregate metrics with no per-seed predictions"))
        elif aggregates and predictions:
            findings.append((OK, str(run_dir.relative_to(ROOT)), f"{len(predictions)} prediction file(s)"))
    return findings


def check_split_leakage() -> List[Tuple[str, str, str]]:
    import csv

    findings: List[Tuple[str, str, str]] = []
    for path in sorted((ROOT / "data").glob("common_split*.csv")):
        rows = list(csv.DictReader(path.open()))
        by_split: Dict[str, set] = {}
        for row in rows:
            by_split.setdefault(row.get("split", "?"), set()).add(row.get("image_id"))
        train, test = by_split.get("train", set()), by_split.get("test", set())
        overlap = len(train & test)
        duplicates = len(rows) - len({row.get("image_id") for row in rows})
        rel = str(path.relative_to(ROOT))
        if overlap or duplicates:
            findings.append((FAIL, rel, f"LEAKAGE: train/test overlap {overlap}, duplicate ids {duplicates}"))
        else:
            findings.append((OK, rel, f"{len(rows)} rows, no overlap ({', '.join(f'{k}:{len(v)}' for k, v in sorted(by_split.items()))})"))
    return findings


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #
def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Audit the results tree for reproducibility gaps.")
    parser.add_argument("--results", default="results")
    parser.add_argument("--strict", action="store_true", help="treat warnings as failures")
    args = parser.parse_args(argv)

    results = ROOT / args.results if not Path(args.results).is_absolute() else Path(args.results)
    if not results.is_dir():
        print(f"ERROR: results folder not found: {results}", file=sys.stderr)
        return 1

    sections = [
        ("checkpoint references", check_checkpoint_references(results)),
        ("run weights", check_run_weights(results)),
        ("instance-loss ablation", check_ablation(results)),
        ("generated tables", check_docs_point_at_generated_tables(results)),
        ("predictions", check_predictions_correspond_to_runs(results)),
        ("split integrity", check_split_leakage()),
    ]

    failures = 0
    warnings = 0
    for title, findings in sections:
        print(f"\n{title}")
        print("-" * len(title))
        for status, where, message in findings:
            print(f"[{status:4}] {where:<52} {message}")
            failures += status == FAIL
            warnings += status == WARN

    print()
    print(f"{failures} fatal, {warnings} warning(s)")
    if failures:
        print("Fatal findings must be fixed: commit the missing weights, or delete the export that references them.")
        return 1
    if warnings and args.strict:
        print("--strict: treating warnings as failures.")
        return 1
    if warnings:
        print("Warnings do not block a demo, but each one is a claim a reviewer cannot verify.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

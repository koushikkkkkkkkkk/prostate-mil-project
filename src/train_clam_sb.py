"""CLAM-SB ablation driver -- Person 2.

Trains CLAM-SB **with** and **without** the instance-level clustering loss on
identical features / split / seeds, then writes the ablation table used in the
report:

    | setting                     | QWK (mean +- sd) | accuracy | macro F1 |

The "without" run is exactly the same network with
``instance_loss_weight = 0`` (the instance classifiers and attention branch are
still built, so this is a clean loss ablation, not an architecture ablation).

Usage::

    python -m src.train_clam_sb \
        --manifest data/common_split.csv --features-root data/features \
        --config configs/clam_sb.yaml --out-dir results/metrics/clam_sb \
        --seeds 0 1 2
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import numpy as np

try:
    from src.evaluate import markdown_table
    from src.train import load_yaml, run_experiment
except ImportError:  # pragma: no cover - direct script execution
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from src.evaluate import markdown_table  # type: ignore
    from src.train import load_yaml, run_experiment  # type: ignore

MODES: Dict[str, float] = {
    "with_instance": 0.30,
    "without_instance": 0.0,
}


def run_ablation(
    manifest: str | Path,
    features_root: str | Path,
    out_dir: str | Path,
    config_path: Optional[str | Path] = None,
    seeds: Sequence[int] = (0, 1, 2),
    modes: Sequence[str] = ("with_instance", "without_instance"),
    instance_loss_weight: float = 0.30,
    verbose: bool = True,
) -> Dict[str, Any]:
    """Run the CLAM-SB instance-loss ablation and persist the comparison."""
    config = load_yaml(config_path) if config_path else {}
    model_config = dict(config.get("model", {}))
    training = dict(config.get("training", {}))
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    weights = {"with_instance": instance_loss_weight, "without_instance": 0.0}
    results: Dict[str, Any] = {}

    for mode in modes:
        mode_config = {**model_config, "instance_loss_weight": weights[mode]}
        mode_out = out_dir / mode
        if verbose:
            print(f"\n=== CLAM-SB [{mode}] instance_loss_weight={weights[mode]} ===")
        summary = run_experiment(
            model_name="clam_sb",
            model_config=mode_config,
            manifest=manifest,
            features_root=features_root,
            out_dir=mode_out,
            seeds=seeds,
            train_kwargs={**training, "instance_loss_weight": weights[mode]},
            verbose=verbose,
        )
        results[mode] = summary["aggregate"]

    rows: List[Dict[str, Any]] = []
    for mode, aggregate in results.items():
        rows.append(
            {
                "setting": "CLAM-SB + instance loss" if mode == "with_instance" else "CLAM-SB, no instance loss",
                "qwk_mean": aggregate.get("qwk", {}).get("mean", float("nan")),
                "qwk_std": aggregate.get("qwk", {}).get("std", 0.0),
                "accuracy": aggregate.get("accuracy", {}).get("mean", float("nan")),
                "balanced_accuracy": aggregate.get("balanced_accuracy", {}).get("mean", float("nan")),
                "macro_f1": aggregate.get("macro_f1", {}).get("mean", float("nan")),
                "seeds": aggregate.get("n_seeds", len(seeds)),
            }
        )

    table = markdown_table(
        rows,
        [
            ("setting", "Setting"),
            ("qwk_mean", "QWK (mean)"),
            ("qwk_std", "QWK (sd)"),
            ("accuracy", "Accuracy"),
            ("balanced_accuracy", "Balanced acc."),
            ("macro_f1", "Macro F1"),
            ("seeds", "Seeds"),
        ],
    )
    header = (
        "# CLAM-SB instance-level supervision ablation\n\n"
        "Same features, same split, same seeds; only the instance clustering "
        "loss changes.\n\n"
    )
    (out_dir / "ablation_clam_sb.md").write_text(header + table + "\n", encoding="utf-8")
    (out_dir / "ablation_clam_sb.json").write_text(json.dumps(results, indent=2), encoding="utf-8")

    with_instance = results.get("with_instance", {}).get("qwk", {}).get("mean", float("nan"))
    without = results.get("without_instance", {}).get("qwk", {}).get("mean", float("nan"))
    delta = with_instance - without if np.isfinite(with_instance) and np.isfinite(without) else float("nan")

    if verbose:
        print("\n" + table)
        print(f"\nQWK delta (with - without) = {delta:+.4f}")
    return {"results": results, "delta_qwk": float(delta), "table": table}


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="CLAM-SB instance-loss ablation.")
    parser.add_argument("--manifest", default="data/common_split.csv")
    parser.add_argument("--features-root", default="data/features")
    parser.add_argument("--out-dir", default="results/metrics/clam_sb")
    parser.add_argument("--config", default="configs/clam_sb.yaml")
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    parser.add_argument(
        "--modes",
        nargs="+",
        default=["with_instance", "without_instance"],
        choices=sorted(MODES),
    )
    parser.add_argument("--instance-loss-weight", type=float, default=0.30)
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args(argv)

    run_ablation(
        manifest=args.manifest,
        features_root=args.features_root,
        out_dir=args.out_dir,
        config_path=args.config,
        seeds=args.seeds,
        modes=args.modes,
        instance_loss_weight=args.instance_loss_weight,
        verbose=not args.quiet,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

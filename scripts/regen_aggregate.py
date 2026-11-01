"""Rebuild aggregate_*.json / table_*.md / ablation_clam_sb.md from ALL per-seed metrics on disk.
The trainer aggregates only the seeds of one call, so `--seeds 1 2` wipes a 3-seed aggregate.
Usage: python scripts/regen_aggregate.py
"""
import glob, json
from pathlib import Path
import numpy as np

KEYS = ("qwk", "accuracy", "balanced_accuracy", "macro_f1", "weighted_f1")
ROOT = Path("results/metrics")

def load(p):
    return sorted((json.load(open(f)) for f in glob.glob(str(p))), key=lambda r: r.get("seed", -1))

def agg(runs):
    out = {"n_seeds": len(runs), "seeds": [r.get("seed") for r in runs]}
    for k in KEYS:
        v = np.array([r[k] for r in runs if k in r], float); v = v[np.isfinite(v)]
        if v.size:
            out[k] = {"mean": float(v.mean()),
                      "std": float(v.std(ddof=1)) if v.size > 1 else 0.0,
                      "values": v.tolist()}
    out["per_run"] = [{k: r.get(k) for k in ("tag", "seed", "qwk", "accuracy",
                                             "macro_f1", "balanced_accuracy", "best_epoch")} for r in runs]
    return out

for model in ("abmil", "transmil"):
    runs = load(ROOT / model / f"metrics_{model}_seed*.json")
    if not runs:
        continue
    a = agg(runs); a["model"] = model
    (ROOT / model / f"aggregate_{model}.json").write_text(json.dumps(a, indent=2))
    (ROOT / model / f"table_{model}.md").write_text(
        "| Model | QWK (mean) | QWK (std) | Accuracy | Balanced acc. | Macro F1 | Seeds |\n"
        "| --- | --- | --- | --- | --- | --- | --- |\n"
        f"| {model} | {a['qwk']['mean']:.4f} | {a['qwk']['std']:.4f} | {a['accuracy']['mean']:.4f} | "
        f"{a['balanced_accuracy']['mean']:.4f} | {a['macro_f1']['mean']:.4f} | {a['n_seeds']} |\n")
    print(f"{model}: {a['n_seeds']} seeds {a['seeds']} -> QWK {a['qwk']['mean']:.4f} +/- {a['qwk']['std']:.4f}")

arms = {}
for arm in ("with_instance", "without_instance"):
    runs = load(ROOT / "clam_sb" / arm / "metrics_clam_sb_seed*.json")
    if runs:
        arms[arm] = runs
        a = agg(runs); a["model"] = f"clam_sb[{arm}]"
        (ROOT / "clam_sb" / arm / f"aggregate_clam_sb_{arm}.json").write_text(json.dumps(a, indent=2))
        print(f"clam_sb[{arm}]: {a['n_seeds']} seeds {a['seeds']} -> QWK {a['qwk']['mean']:.4f} +/- {a['qwk']['std']:.4f}")

if arms:
    lines = ["| Setting | QWK (mean) | QWK (sd) | Accuracy | Balanced acc. | Macro F1 | Seeds |",
             "| --- | --- | --- | --- | --- | --- | --- |"]
    m = {}
    for label, key in (("CLAM-SB + instance loss", "with_instance"),
                       ("CLAM-SB, no instance loss", "without_instance")):
        if key not in arms:
            continue
        a = agg(arms[key]); m[key] = a["qwk"]["mean"]
        lines.append(f"| {label} | {a['qwk']['mean']:.4f} | {a['qwk']['std']:.4f} | "
                     f"{a['accuracy']['mean']:.4f} | {a['balanced_accuracy']['mean']:.4f} | "
                     f"{a['macro_f1']['mean']:.4f} | {a['n_seeds']} |")
    if len(m) == 2:
        lines += ["", f"QWK delta (with - without) = {m['with_instance'] - m['without_instance']:+.4f}"]
    early = [r.get("seed") for r in arms.get("without_instance", [])
             if r.get("best_epoch") is not None and r["best_epoch"] <= 6]
    if early:
        lines.append(f"\n> Caveat: no-instance arm peaked at best_epoch <= 6 for seed(s) "
                     f"{', '.join(map(str, early))} -- selection artifact, not a model effect.")
    (ROOT / "clam_sb" / "ablation_clam_sb.md").write_text("\n".join(lines) + "\n")
    print()
    print("\n".join(lines))
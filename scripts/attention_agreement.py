"""Inter-model attention agreement -- the mask-free substitute for localisation overlap.

Compares two models' exported attention arrays (results/heatmaps/<model>/arrays/*.npz)
per slide: Spearman rho over patch weights, Jaccard of the top-10% patch sets, and
Cohen's kappa on the top-10%/rest binarisation.

This is INTER-MODEL agreement (do two architectures flag the same tiles?), not
tumour-localisation accuracy -- PANDA has no pixel-level ISUP masks for these slides.

Usage:
  python scripts/attention_agreement.py --dir-a results/heatmaps/abmil/arrays \
      --dir-b results/heatmaps/clam_sb/arrays --out results/metrics/attention/agreement.csv
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr


def _load(path: Path):
    try:
        payload = np.load(path, allow_pickle=True)
    except Exception:
        return None
    return np.asarray(payload["attention"], dtype=float).ravel() if "attention" in payload else None


def _kappa(a: np.ndarray, b: np.ndarray) -> float:
    if a.size == 0:
        return float("nan")
    po = float((a == b).mean())
    pa, pb = float(a.mean()), float(b.mean())
    pe = pa * pb + (1 - pa) * (1 - pb)
    return float("nan") if abs(1 - pe) < 1e-12 else (po - pe) / (1 - pe)


def agreement(dir_a: Path, dir_b: Path, top_fraction: float = 0.10, min_patches: int = 20):
    fa = {p.stem: p for p in sorted(dir_a.glob("*.npz"))}
    fb = {p.stem: p for p in sorted(dir_b.glob("*.npz"))}
    rows = []
    for sid in sorted(set(fa) & set(fb)):
        a, b = _load(fa[sid]), _load(fb[sid])
        if a is None or b is None or a.size != b.size or a.size < min_patches:
            continue
        k = max(1, int(round(top_fraction * a.size)))
        ia, ib = set(np.argsort(-a)[:k].tolist()), set(np.argsort(-b)[:k].tolist())
        ba, bb = np.zeros(a.size, bool), np.zeros(b.size, bool)
        ba[list(ia)] = True
        bb[list(ib)] = True
        rows.append({
            "slide_id": sid,
            "n_patches": int(a.size),
            "spearman_rho": float(spearmanr(a, b).statistic),
            f"top{int(top_fraction*100)}pct_jaccard": len(ia & ib) / max(len(ia | ib), 1),
            "top_kappa": _kappa(ba, bb),
        })
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir-a", required=True)
    ap.add_argument("--dir-b", required=True)
    ap.add_argument("--label-a", default="model_a")
    ap.add_argument("--label-b", default="model_b")
    ap.add_argument("--top-fraction", type=float, default=0.10)
    ap.add_argument("--out", default="results/metrics/attention/agreement.csv")
    args = ap.parse_args()

    da, db = Path(args.dir_a), Path(args.dir_b)
    for d in (da, db):
        if not d.exists():
            print(f"ERROR: {d} not found -- render heatmaps with --render-images first.")
            return 1

    rows = agreement(da, db, args.top_fraction)
    if not rows:
        print("No shared slides with matching patch counts. Both models must be exported with --render-images.")
        return 1

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    rho = np.array([r["spearman_rho"] for r in rows])
    jac = np.array([r[[k for k in r if "jaccard" in k][0]] for r in rows])
    kap = np.array([r["top_kappa"] for r in rows])
    summary = {
        "model_a": args.label_a, "model_b": args.label_b, "n_shared_slides": len(rows),
        "spearman_rho": {"mean": float(rho.mean()), "sd": float(rho.std(ddof=1)) if rho.size > 1 else 0.0},
        "jaccard_top": {"mean": float(jac.mean()), "sd": float(jac.std(ddof=1)) if jac.size > 1 else 0.0},
        "kappa_top": {"mean": float(np.nanmean(kap)), "sd": float(np.nanstd(kap, ddof=1)) if kap.size > 1 else 0.0},
    }
    (out.parent / (out.stem + "_summary.json")).write_text(json.dumps(summary, indent=2))

    print(f"shared slides        : {len(rows)}")
    print(f"mean Spearman rho    : {rho.mean():.3f} (sd {summary['spearman_rho']['sd']:.3f})")
    print(f"mean top-{int(args.top_fraction*100)}% Jaccard   : {jac.mean():.3f}")
    print(f"mean Cohen's kappa   : {np.nanmean(kap):.3f}")
    print(f"written              : {out}")
    print("\nReport as INTER-MODEL agreement, never as tumour-localisation accuracy.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
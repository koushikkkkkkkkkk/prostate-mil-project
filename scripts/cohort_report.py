"""Evaluate a fused prediction file per coverage cohort, and each encoder on the same slides.

Reads the CSV written by scripts/fuse_encoders.py (it carries an ``encoders`` column such as
"abmil_chief+abmil_gigap"). Reports:

  1. per cohort: n, QWK, accuracy, error count, off-by-one count
  2. per encoder (optional, via --pred): QWK restricted to exactly the fused slide set,
     so "does fusion beat the best single encoder?" is answered on identical slides
  3. the fused error rate split by single- vs multi-encoder coverage

Usage:
    python scripts/cohort_report.py \
        --fused results/metrics/comparison/fused_chief_gigap.csv \
        --pred "results/metrics/abmil_chief/predictions_abmil_seed*.csv" \
        --pred "results/metrics/abmil_gigap/predictions_abmil_seed*.csv" \
        --out results/metrics/comparison/fused_chief_gigap_cohorts.json
"""
import argparse
import csv
import glob
import json
import os
from collections import defaultdict

import numpy as np


def _read_pred(path):
    out = {}
    for r in csv.DictReader(open(path, newline="", encoding="utf-8")):
        sid = r.get("slide_id") or r.get("image_id")
        probs = [float(r[f"p{k}"]) for k in range(6) if f"p{k}" in r]
        if sid and len(probs) == 6:
            out[sid] = (int(float(r["y_true"])), np.array(probs))
    return out


def _metrics(y, p):
    from sklearn.metrics import accuracy_score, cohen_kappa_score
    if len(y) < 2:
        return float("nan"), (float(accuracy_score(y, p)) if len(y) else float("nan"))
    return (float(cohen_kappa_score(y, p, weights="quadratic")),
            float(accuracy_score(y, p)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fused", required=True)
    ap.add_argument("--pred", action="append", default=[],
                    help="predictions glob per encoder (repeatable)")
    ap.add_argument("--out", default=None, help="optional JSON path for the summary")
    a = ap.parse_args()

    rows = list(csv.DictReader(open(a.fused, newline="", encoding="utf-8")))
    if not rows:
        raise SystemExit("empty fused file")

    summary = {"fused_file": a.fused, "n_total": len(rows), "cohorts": {}}

    # --- 1. per cohort -----------------------------------------------------
    cohorts = defaultdict(list)
    for r in rows:
        cohorts[r["encoders"]].append(r)

    print(f"{a.fused}   n={len(rows)}\n")
    print(f"{'cohort':30s} {'n':>3s} {'QWK':>7s} {'acc':>6s} {'err':>4s} {'off1':>5s}")
    print("-" * 62)
    for key in sorted(cohorts, key=lambda k: -len(cohorts[k])):
        sub = cohorts[key]
        y = np.array([int(r["y_true"]) for r in sub])
        p = np.array([int(r["y_pred"]) for r in sub])
        qwk, acc = _metrics(y, p)
        print(f"{key:30s} {len(sub):3d} {qwk:7.4f} {acc:6.4f} "
              f"{int((y != p).sum()):4d} {int((np.abs(y - p) == 1).sum()):5d}")
        summary["cohorts"][key] = {"n": len(sub), "qwk": qwk, "accuracy": acc,
                                   "errors": int((y != p).sum()),
                                   "off_by_one": int((np.abs(y - p) == 1).sum())}

    # --- 2. each encoder on the SAME fused slide set -----------------------
    fused_ids = [r["slide_id"] for r in rows]
    label = {r["slide_id"]: int(r["y_true"]) for r in rows}
    if a.pred:
        print(f"\n{'encoder (same slide set)':30s} {'n':>3s} {'QWK':>7s} {'acc':>6s}")
        print("-" * 55)
        yf = np.array([label[s] for s in fused_ids])
        pf = np.array([int(r["y_pred"]) for r in rows])
        qwk, acc = _metrics(yf, pf)
        print(f"{'FUSED':30s} {len(fused_ids):3d} {qwk:7.4f} {acc:6.4f}")
        summary["on_fused_slide_set"] = {"FUSED": {"n": len(fused_ids), "qwk": qwk, "accuracy": acc}}

        for pattern in a.pred:
            files = sorted(glob.glob(pattern))
            if not files:
                print(f"  !! no files match {pattern}")
                continue
            name = os.path.basename(os.path.dirname(files[0])) or "run"
            runs = [_read_pred(f) for f in files]
            ids = [s for s in fused_ids if all(s in run for run in runs)]
            if not ids:
                print(f"  {name:30s} -- covers none of the fused slides")
                continue
            probs = np.stack([np.stack([run[s][1] for s in ids]) for run in runs]).mean(axis=0)
            y = np.array([label[s] for s in ids])
            p = probs.argmax(axis=1)
            qwk, acc = _metrics(y, p)
            print(f"{name:30s} {len(ids):3d} {qwk:7.4f} {acc:6.4f}"
                  + ("" if len(ids) == len(fused_ids) else "   (fewer slides)"))
            summary["on_fused_slide_set"][name] = {"n": len(ids), "qwk": qwk, "accuracy": acc}

    # --- 3. coverage split -------------------------------------------------
    multi = [r for r in rows if "+" in r["encoders"]]
    single = [r for r in rows if "+" not in r["encoders"]]
    if multi and single:
        print()
        for name, group in (("both encoders", multi), ("single encoder", single)):
            y = np.array([int(r["y_true"]) for r in group])
            p = np.array([int(r["y_pred"]) for r in group])
            e = int((y != p).sum())
            print(f"  {name:16s} n={len(group):2d}  acc {float((y == p).mean()):.4f}  "
                  f"errors {e}  ({e / len(group) * 100:.1f}% error rate)")
            summary[name.replace(" ", "_")] = {"n": len(group), "errors": e}

    if a.out:
        with open(a.out, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2)
        print(f"\nwritten {a.out}")


if __name__ == "__main__":
    main()

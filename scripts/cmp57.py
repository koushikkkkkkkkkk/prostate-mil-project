"""Compare runs on the identical slide subset defined by an encoder's manifest.

Different encoders cover different slides, so their QWK values are not directly
comparable. This script restricts every run to the test ids present in a reference
manifest (e.g. data/common_split_chief.csv -> 57 slides) and reports mean +/- sd over
seeds, so the comparison is apples-to-apples without any late fusion.

Usage:
    python scripts/cmp57.py \
        --run "abmil r50=results/metrics/abmil/predictions_abmil_seed*.csv" \
        --run "clam_sb r50=results/metrics/clam_sb/with_instance/predictions_clam_sb_seed*.csv" \
        --run "abmil chief=results/metrics/abmil_chief/predictions_abmil_seed*.csv" \
        --run "clam_sb chief=results/metrics/clam_sb_chief/predictions_clam_sb_seed*.csv" \
        --restrict-to data/common_split_chief.csv \
        --out results/metrics/comparison/encoder_comparison_57.csv

Each --run is "label=glob". The glob must match one CSV per seed.
"""
import argparse
import csv
import glob
import os

import numpy as np


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", action="append", required=True, help='"label=glob", repeatable')
    ap.add_argument("--restrict-to", required=True,
                    help="manifest whose test image_ids define the common slide set")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()

    from sklearn.metrics import accuracy_score, cohen_kappa_score

    allowed = {r["image_id"] for r in
               csv.DictReader(open(a.restrict_to, newline="", encoding="utf-8"))
               if r["split"] == "test"}
    print(f"common test set: {len(allowed)} slides from {a.restrict_to}\n")
    print(f"{'run':26s} {'n':>3s} {'seeds':>5s} {'QWK':>15s} {'acc':>7s}")
    print("-" * 60)

    table = []
    for spec in a.run:
        if "=" not in spec:
            raise SystemExit(f"--run must be label=glob, got: {spec}")
        label, pattern = spec.split("=", 1)
        files = sorted(glob.glob(pattern))
        if not files:
            print(f"{label:26s} -- no files match {pattern}")
            continue
        qwk, acc, n = [], [], 0
        for f in files:
            rows = [r for r in csv.DictReader(open(f, newline="", encoding="utf-8"))
                    if (r.get("slide_id") or r.get("image_id")) in allowed]
            if not rows:
                continue
            y = np.array([int(float(r["y_true"])) for r in rows])
            p = np.array([int(float(r["y_pred"])) for r in rows])
            n = len(rows)
            qwk.append(cohen_kappa_score(y, p, weights="quadratic"))
            acc.append(accuracy_score(y, p))
        if not qwk:
            print(f"{label:26s} -- covers none of the common slides")
            continue
        sd = float(np.std(qwk, ddof=1)) if len(qwk) > 1 else 0.0
        print(f"{label:26s} {n:3d} {len(qwk):5d} "
              f"{np.mean(qwk):7.4f} +/- {sd:.4f} {np.mean(acc):7.4f}")
        table.append({"run": label, "n_slides": n, "n_seeds": len(qwk),
                      "qwk_mean": float(np.mean(qwk)), "qwk_sd": sd,
                      "acc_mean": float(np.mean(acc))})

    if a.out and table:
        os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
        with open(a.out, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=list(table[0]))
            w.writeheader()
            w.writerows(table)
        print(f"\nwritten {a.out}")
    print("\nEvery row above is scored on the SAME slides, so the differences are attributable\n"
          "to the encoder/model, not to coverage.")


if __name__ == "__main__":
    main()

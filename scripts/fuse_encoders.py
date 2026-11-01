"""Late-fuse slide predictions from several encoders, covering the union of their slides.

Why: no single PANDA encoder covers every slide in this 420-slide manifest
(r50 60/60 test, chief 57/60, uni 54/60, gigap 53/60) but chief+uni, chief+gigap and
uni+gigap each cover 60/60. Averaging per-slide class probabilities across encoders
both restores full test coverage and is a stronger ensemble than seed averaging alone,
because different encoders make different errors.

Pre-registered rule (do not tune it): for each slide, average p0..p5 over every
(encoder, seed) prediction available for that slide; predict argmax of the average.

Usage:
    python scripts/fuse_encoders.py ^
        --pred "results/metrics/abmil_chief/predictions_abmil_chief_seed*.csv" ^
        --pred "results/metrics/abmil_gigap/predictions_abmil_gigap_seed*.csv" ^
        --manifest data/common_split.csv ^
        --out results/metrics/comparison/fused_chief_gigap.csv
"""
import argparse, csv, glob, json, os
import numpy as np


def read(path):
    rows = {}
    for r in csv.DictReader(open(path, newline="", encoding="utf-8")):
        sid = r.get("slide_id") or r.get("image_id")
        if not sid:
            continue
        probs = [float(r[f"p{k}"]) for k in range(6) if f"p{k}" in r]
        if len(probs) == 6:
            rows[sid] = (int(float(r["y_true"])), np.array(probs))
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pred", action="append", required=True,
                    help="glob of predictions_*.csv for one encoder (repeat per encoder)")
    ap.add_argument("--manifest", default="data/common_split.csv",
                    help="defines the test slide set used for reporting")
    ap.add_argument("--out", required=True)
    ap.add_argument("--tag", default=None)
    a = ap.parse_args()
    from sklearn.metrics import accuracy_score, balanced_accuracy_score, cohen_kappa_score, f1_score

    test_ids = [r["image_id"] for r in csv.DictReader(open(a.manifest, newline="", encoding="utf-8"))
                if r["split"] == "test"]

    groups = []
    for pattern in a.pred:
        files = sorted(glob.glob(pattern))
        if not files:
            print(f"  !! no files match {pattern}")
            continue
                # Label by out-dir (abmil_chief / abmil_gigap / abmil), because the prediction
        # filename only carries the model name, not the encoder.
        model_tag = files[0].split("predictions_")[-1].rsplit("_seed", 1)[0]
        parent = os.path.basename(os.path.dirname(files[0]))
        name = parent if parent not in ("", "metrics") else model_tag
        runs = [read(f) for f in files]
        groups.append((name, runs))
        print(f"{name}: {len(files)} seed(s), covers {len(set().union(*[set(r) for r in runs]))} slides")

    if not groups:
        print("ERROR: nothing to fuse.")
        return 1

    fused, coverage, per_group_alone = {}, {}, {}
    for sid in test_ids:
        stacks = []
        for name, runs in groups:
            seed_probs = [run[sid][1] for run in runs if sid in run]
            if seed_probs:
                stacks.append((name, np.mean(seed_probs, axis=0)))
        if not stacks:
            continue
        fused[sid] = np.mean([p for _, p in stacks], axis=0)
        coverage[sid] = [name for name, _ in stacks]

    ids = [s for s in test_ids if s in fused]
    if not ids:
        print("ERROR: no slide had a prediction.")
        return 1

    label = {r["image_id"]: int(r["isup_grade"]) for r in
             csv.DictReader(open(a.manifest, newline="", encoding="utf-8"))}
    y = np.array([label[s] for s in ids])
    pred = np.array([fused[s].argmax() for s in ids])

    for name, runs in groups:
        sub = [s for s in ids if s in runs[0]]
        if sub:
            p = np.array([runs[0][s][1].argmax() for s in sub])
            yy = np.array([label[s] for s in sub])
            per_group_alone[name] = {
                "n": len(sub),
                "qwk": float(cohen_kappa_score(yy, p, weights="quadratic")),
                "accuracy": float(accuracy_score(yy, p)),
            }

    summary = {
        "tag": a.tag or os.path.splitext(os.path.basename(a.out))[0],
        "encoders": [name for name, _ in groups],
        "n_test_slides": len(test_ids),
        "n_covered": len(ids),
        "n_covered_by_single_encoder": sum(1 for s in ids if len(coverage[s]) == 1),
        "fused": {
            "qwk": float(cohen_kappa_score(y, pred, weights="quadratic")),
            "accuracy": float(accuracy_score(y, pred)),
            "balanced_accuracy": float(balanced_accuracy_score(y, pred)),
            "macro_f1": float(f1_score(y, pred, average="macro")),
        },
        "per_encoder_seed0": per_group_alone,
    }

    out = a.out
    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
    with open(out, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["slide_id", "y_true", "y_pred", "encoders"] +
                                        [f"p{k}" for k in range(6)])
        w.writeheader()
        for i, s in enumerate(ids):
            row = {"slide_id": s, "y_true": int(y[i]), "y_pred": int(pred[i]),
                   "encoders": "+".join(coverage[s])}
            for k in range(6):
                row[f"p{k}"] = float(fused[s][k])
            w.writerow(row)

    json_path = os.path.splitext(out)[0] + "_summary.json"
    open(json_path, "w", encoding="utf-8").write(json.dumps(summary, indent=2))

    print()
    print(f"encoded test slides : {len(ids)}/{len(test_ids)}"
          f"  ({summary['n_covered_by_single_encoder']} covered by one encoder only)")
    for name, info in per_group_alone.items():
        print(f"  {name:22s} alone: n={info['n']:2d}  QWK {info['qwk']:.4f}")
    print(f"  {'FUSED':22s}      : n={len(ids):2d}  QWK {summary['fused']['qwk']:.4f}"
          f"  acc {summary['fused']['accuracy']:.4f}")
    print(f"\nwritten {out}\nsummary {json_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
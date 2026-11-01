#!/usr/bin/env bash
# Person 4 (localisation, statistical baselines, patch CNN) full pipeline on REAL features.
#
#   bash scripts/run_person4_experiments.sh
#
# Override anything with environment variables:
#   MANIFEST   data/common_split.csv     split from Person 1
#   FEATURES   data/features             per-slide {image_id}.pt WITH coords
#   THUMBS     data/thumbnails           optional slide thumbnails for overlays
#   SEED       0                         seed used for the attention analysis
#   ABMIL_CKPT results/metrics/abmil/checkpoints/abmil_seed0_best.pt
#   CLAM_CKPT  results/metrics/clam_sb/with_instance/checkpoints/clam_sb_seed0_best.pt
#   TRAIN      1                         set 0 to reuse existing checkpoints
#
# Produces (all under results/):
#   metrics/attention/abmil/attention_abmil_seed0.csv       x_stat for the baseline
#   metrics/attention/clam_sb/attention_clam_sb_seed0.csv   x_stat for the baseline
#   metrics/attention/attention_stats_abmil_vs_clam.md      sharpness comparison table
#   metrics/statistical/statistical_baselines.json          ordinal logistic (+XGBoost)
#   heatmaps/abmil/, heatmaps/clam_sb/                      attention maps + top-k patches
#   heatmaps/abmil/localisation_scores.csv                  only when MASKS exists
#
# Nothing here replaces Person 2's run_person2_experiments.sh; this script adds the
# ABMIL side and the statistical baselines that Person 4 owns.
set -euo pipefail

cd "$(dirname "$0")/.."
export MPLBACKEND=Agg

MANIFEST="${MANIFEST:-data/common_split.csv}"
FEATURES="${FEATURES:-data/features}"
THUMBS="${THUMBS:-data/thumbnails}"
MASKS="${MASKS:-data/masks}"
SEED="${SEED:-0}"
TRAIN="${TRAIN:-1}"
ABMIL_CKPT="${ABMIL_CKPT:-results/metrics/abmil/checkpoints/abmil_seed${SEED}_best.pt}"
CLAM_CKPT="${CLAM_CKPT:-results/metrics/clam_sb/with_instance/checkpoints/clam_sb_seed${SEED}_best.pt}"
OUT_ATT="${OUT_ATT:-results/metrics/attention}"
OUT_STAT="${OUT_STAT:-results/metrics/statistical}"

[[ -f "$MANIFEST" ]] || { echo "ERROR: manifest not found: $MANIFEST"; exit 1; }
[[ -d "$FEATURES" ]] || { echo "ERROR: features folder not found: $FEATURES"; exit 1; }

# ---- 0. sanity: do the feature files carry coords? -------------------------
python - <<'PY'
import sys, glob, torch
files = sorted(glob.glob("data/features/*.pt")) or sorted(glob.glob("results/../data/features/*.pt"))
files = [f for f in files if not f.endswith(".tmp")]
if not files:
    print("  (skipping the coords pre-check: no .pt files found)"); sys.exit(0)
with_coords = 0
for path in files[:5]:
    try:
        payload = torch.load(path, map_location="cpu", weights_only=False)
    except Exception:
        continue
    if isinstance(payload, dict) and payload.get("coords") is not None:
        with_coords += 1
print(f"  coords present in {with_coords}/{min(len(files),5)} sampled feature files")
if with_coords == 0:
    print("  WARNING: no coords -> heatmaps/localisation cannot be rendered.")
    print("           Fix with: python scripts/download_features_and_coords.py")
PY

# ---- 1. checkpoints -------------------------------------------------------
if [[ "$TRAIN" == "1" ]]; then
  echo "== 1/5  training ABMIL + CLAM-SB (seed ${SEED}) =="
  python -m src.train --model abmil --manifest "$MANIFEST" --features-root "$FEATURES" \
    --config configs/abmil.yaml --out-dir results/metrics/abmil --seeds "$SEED"
  python -m src.train_clam_sb --manifest "$MANIFEST" --features-root "$FEATURES" \
    --config configs/clam_sb.yaml --out-dir results/metrics/clam_sb --seeds "$SEED"
else
  echo "== 1/5  TRAIN=0, reusing checkpoints =="
fi
[[ -f "$ABMIL_CKPT" ]] || { echo "ERROR missing $ABMIL_CKPT"; exit 1; }
[[ -f "$CLAM_CKPT"  ]] || { echo "ERROR missing $CLAM_CKPT";  exit 1; }

# ---- 2. attention statistics for both models (train+val+test) -------------
echo "== 2/5  exporting attention statistics (split=all, needed by the baseline) =="
python -m src.export_attention --checkpoint "$ABMIL_CKPT" --manifest "$MANIFEST" \
  --features-root "$FEATURES" --out-dir "$OUT_ATT/abmil" --split all
python -m src.export_attention --checkpoint "$CLAM_CKPT" --manifest "$MANIFEST" \
  --features-root "$FEATURES" --out-dir "$OUT_ATT/clam_sb" --split all

# ---- 3. attention sharpness: CLAM-SB vs ABMIL -----------------------------
echo "== 3/5  attention sharpness comparison =="
python - "$OUT_ATT" <<'PY'
import json, sys
from pathlib import Path
import numpy as np
root = Path(sys.argv[1])
rows = []
for name, sub in (("ABMIL", "abmil"), ("CLAM-SB", "clam_sb")):
    summary_path = root / sub / "export_summary.json"
    if not summary_path.exists():
        continue
    summary = json.loads(summary_path.read_text())
    rows.append((name, summary.get("n_slides"), summary.get("mean_normalised_entropy"),
                 summary.get("mean_top10_attention_share")))
out = root / "attention_stats_abmil_vs_clam.md"
lines = ["| Model | Slides | Mean normalised entropy (lower = sharper) | Mean top-10 attention share |",
         "| --- | --- | --- | --- |"]
for name, n, ent, share in rows:
    lines.append(f"| {name} | {n} | {ent:.4f} | {share:.4f} |")
out.write_text("\n".join(lines) + "\n")
print("\n".join(lines))
PY

# ---- 4. statistical baselines over x_stat --------------------------------
echo "== 4/5  ordinal logistic / XGBoost baselines over attention statistics =="
for model in abmil clam_sb; do
  CSV=$(ls "$OUT_ATT/$model"/attention_*.csv 2>/dev/null | head -1 || true)
  if [[ -n "${CSV:-}" ]]; then
    python -m src.statistical_models --attention-csv "$CSV" --manifest "$MANIFEST" \
      --out-dir "$OUT_STAT/$model" || echo "  (baseline for $model skipped)"
  fi
done

# ---- 5. heatmaps + localisation ------------------------------------------
echo "== 5/5  heatmaps for the test split =="
THUMB_ARGS=(); [[ -d "$THUMBS" ]] && THUMB_ARGS=(--thumbnails "$THUMBS")
MASK_ARGS=();  [[ -d "$MASKS"  ]] && MASK_ARGS=(--masks "$MASKS")
python -m src.export_attention --checkpoint "$ABMIL_CKPT" --manifest "$MANIFEST" \
  --features-root "$FEATURES" --out-dir results/heatmaps/abmil --split test \
  --render-images --top-k 12 ${THUMB_ARGS[@]+"${THUMB_ARGS[@]}"}
python -m src.export_clam_attention --checkpoint "$CLAM_CKPT" --manifest "$MANIFEST" \
  --features-root "$FEATURES" --out-dir results/heatmaps/clam_sb --split test \
  --top-k 12 ${THUMB_ARGS[@]+"${THUMB_ARGS[@]}"} ${MASK_ARGS[@]+"${MASK_ARGS[@]}"}

echo
echo "Person 4 pipeline complete."
echo "  attention stats : $OUT_ATT/<model>/attention_<model>_seed${SEED}.csv"
echo "  sharpness table : $OUT_ATT/attention_stats_abmil_vs_clam.md"
echo "  baselines       : $OUT_STAT/<model>/statistical_baselines.json"
echo "  heatmaps        : results/heatmaps/abmil/images , results/heatmaps/clam_sb/images"
echo "  localisation    : results/heatmaps/clam_sb/localisation_scores.csv (only with --masks)"
echo
echo "Patch CNN is NOT in this script: it needs tile PNGs, which the pre-extracted"
echo "feature release does not ship. See PERSON4_RUNBOOK.md section 6."

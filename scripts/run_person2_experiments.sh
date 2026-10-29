#!/usr/bin/env bash
# Person 2 (CLAM-SB) full experiment pipeline on the REAL PANDA feature bags.
#
#   bash scripts/run_person2_experiments.sh
#
# Override any path via environment variables:
#   MANIFEST    data/common_split.csv        stratified 300/60/60 split (Person 1)
#   FEATURES    data/features                per-slide {image_id}.pt feature bags
#   THUMBS      data/thumbnails              optional slide thumbnails for overlays
#   MASKS       data/masks                   optional tumour/tissue masks for overlap score
#   OUT         results/metrics/clam_sb
#   CONFIG      configs/clam_sb.yaml
#   SEEDS       "0 1 2"
#
# Produces: instance-loss ablation, per-seed evaluation, attention heatmaps,
# statistical baseline, and the paired ABMIL-vs-CLAM-SB comparison.
set -euo pipefail

cd "$(dirname "$0")/.."
export MPLBACKEND=Agg

MANIFEST="${MANIFEST:-data/common_split.csv}"
FEATURES="${FEATURES:-data/features}"
THUMBS="${THUMBS:-data/thumbnails}"
MASKS="${MASKS:-data/masks}"
OUT="${OUT:-results/metrics/clam_sb}"
CONFIG="${CONFIG:-configs/clam_sb.yaml}"
HEATMAP_OUT="${HEATMAP_OUT:-results/heatmaps/clam_sb}"
COMPARE_OUT="${COMPARE_OUT:-results/metrics/comparison}"
STAT_OUT="${STAT_OUT:-results/metrics/statistical}"
SEEDS="${SEEDS:-0 1 2}"
FIRST_SEED="${SEEDS%% *}"

if [[ ! -f "$MANIFEST" ]]; then
  echo "ERROR: manifest not found: $MANIFEST"
  echo "  Person 1 owns this file. For the pre-Person-1 path use: bash scripts/run_mock_demo.sh"
  exit 1
fi
if [[ ! -d "$FEATURES" ]]; then
  echo "ERROR: feature folder not found: $FEATURES"
  echo "  Extract features with: python -m src.feature_extraction --patches-root data/patches"
  exit 1
fi

read -r -a SEED_ARRAY <<< "$SEEDS"

echo "== 1/6  CLAM-SB instance-loss ablation (seeds: ${SEED_ARRAY[*]}) =="
python -m src.train_clam_sb \
  --manifest "$MANIFEST" \
  --features-root "$FEATURES" \
  --config "$CONFIG" \
  --out-dir "$OUT" \
  --seeds "${SEED_ARRAY[@]}"

CLAM_CKPT="$OUT/with_instance/checkpoints/clam_sb_seed${FIRST_SEED}_best.pt"

echo "== 2/6  evaluating the seed-${FIRST_SEED} CLAM-SB checkpoint =="
python -m src.evaluate_clam_sb \
  --checkpoint "$CLAM_CKPT" \
  --manifest "$MANIFEST" \
  --features-root "$FEATURES" \
  --out-dir "$OUT/eval_seed${FIRST_SEED}" \
  --split all

echo "== 3/6  exporting attention heatmaps =="
THUMB_ARGS=()
[[ -d "$THUMBS" ]] && THUMB_ARGS=(--thumbnails "$THUMBS")
MASK_ARGS=()
[[ -d "$MASKS" ]] && MASK_ARGS=(--masks "$MASKS")
python -m src.export_clam_attention \
  --checkpoint "$CLAM_CKPT" \
  --manifest "$MANIFEST" \
  --features-root "$FEATURES" \
  --out-dir "$HEATMAP_OUT" \
  --split test \
  "${THUMB_ARGS[@]}" "${MASK_ARGS[@]}"

echo "== 4/6  statistical baseline (ordinal logistic / XGBoost) =="
python -m src.statistical_models \
  --attention-csv "$OUT/eval_seed${FIRST_SEED}/attention_clam_sb_seed${FIRST_SEED}.csv" \
  --manifest "$MANIFEST" \
  --out-dir "$STAT_OUT" || echo "  (statistical baseline skipped)"

echo "== 5/6  paired ABMIL vs CLAM-SB comparison =="
ABMIL_CSV="${ABMIL_CSV:-results/metrics/abmil/predictions_abmil_seed${FIRST_SEED}.csv}"
if [[ -f "$ABMIL_CSV" ]]; then
  python -m src.compare_clam_abmil \
    --abmil "$ABMIL_CSV" \
    --clam "$OUT/with_instance/predictions_clam_sb_seed${FIRST_SEED}.csv" \
    --abmil-label "ABMIL" \
    --clam-label "CLAM-SB + instance loss" \
    --out-dir "$COMPARE_OUT"
else
  echo "  SKIPPED: ABMIL predictions not found at $ABMIL_CSV"
  echo "  Person 1 produces them with: python -m src.train --model abmil --manifest $MANIFEST \\"
  echo "      --features-root $FEATURES --config configs/abmil.yaml --out-dir results/metrics/abmil"
fi

echo "== 6/6  ablation table =="
cat "$OUT/ablation_clam_sb.md"
echo
echo "Done."
echo "  ablation     : $OUT/ablation_clam_sb.md"
echo "  comparison   : $COMPARE_OUT/comparison.md"
echo "  heatmaps     : $HEATMAP_OUT/images/"
echo "  localisation : $HEATMAP_OUT/localisation_scores.csv (present only when MASKS was set)"

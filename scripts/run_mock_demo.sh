#!/usr/bin/env bash
# End-to-end Person 2 pipeline on SYNTHETIC mock data.
#
#   bash scripts/run_mock_demo.sh
#
# Generates mock slides, trains ABMIL (for the paired comparison) and CLAM-SB
# with/without instance loss, evaluates, renders heatmaps with localisation
# overlap scores, fits the statistical baseline, and writes the comparison.
#
# EVERYTHING this produces is synthetic. It proves the plumbing works; it says
# nothing about PANDA performance. Do not report these numbers.
set -euo pipefail

cd "$(dirname "$0")/.."
export MPLBACKEND=Agg

SEED="${SEED:-0}"
MOCK_DIR="${MOCK_DIR:-data/mock}"
CLAM_OUT="${CLAM_OUT:-results/mock_clam_sb}"
ABMIL_OUT="${ABMIL_OUT:-results/mock_abmil}"
HEATMAP_OUT="${HEATMAP_OUT:-results/mock_heatmaps}"
EVAL_OUT="${EVAL_OUT:-results/mock_eval}"
STAT_OUT="${STAT_OUT:-results/mock_statistical}"

if ! python -c "import torch, yaml" >/dev/null 2>&1; then
  echo "ERROR: torch (and/or pyyaml) missing. Install with:"
  echo "  pip install -r requirements.txt"
  echo "  # CPU-only, much smaller download:"
  echo "  pip install torch --index-url https://download.pytorch.org/whl/cpu"
  exit 1
fi

echo "== 1/7  generating synthetic mock slides =="
python scripts/make_mock_features.py --out-dir "$MOCK_DIR" --seed 7

echo "== 2/7  training ABMIL baseline (1 seed, mock config) =="
python -m src.train \
  --model abmil \
  --manifest "$MOCK_DIR/manifest.csv" \
  --features-root "$MOCK_DIR/features" \
  --config configs/abmil_mock.yaml \
  --out-dir "$ABMIL_OUT" \
  --seeds "$SEED"

echo "== 3/7  CLAM-SB ablation: with vs without instance loss =="
python -m src.train_clam_sb \
  --manifest "$MOCK_DIR/manifest.csv" \
  --features-root "$MOCK_DIR/features" \
  --config configs/clam_sb_mock.yaml \
  --out-dir "$CLAM_OUT" \
  --seeds "$SEED"

CLAM_CKPT="$CLAM_OUT/with_instance/checkpoints/clam_sb_seed${SEED}_best.pt"

echo "== 4/7  evaluating CLAM-SB checkpoint (all splits, for attention stats) =="
python -m src.evaluate_clam_sb \
  --checkpoint "$CLAM_CKPT" \
  --manifest "$MOCK_DIR/manifest.csv" \
  --features-root "$MOCK_DIR/features" \
  --out-dir "$EVAL_OUT" \
  --split all

echo "== 5/7  exporting attention heatmaps + localisation overlap =="
python -m src.export_clam_attention \
  --checkpoint "$CLAM_CKPT" \
  --manifest "$MOCK_DIR/manifest.csv" \
  --features-root "$MOCK_DIR/features" \
  --thumbnails "$MOCK_DIR/thumbnails" \
  --masks "$MOCK_DIR/masks" \
  --out-dir "$HEATMAP_OUT" \
  --split test

echo "== 6/7  statistical baseline on attention statistics =="
python -m src.statistical_models \
  --attention-csv "$EVAL_OUT/attention_clam_sb_seed${SEED}.csv" \
  --manifest "$MOCK_DIR/manifest.csv" \
  --out-dir "$STAT_OUT" || echo "  (statistical baseline skipped)"

echo "== 7/7  paired ABMIL vs CLAM-SB comparison =="
python -m src.compare_clam_abmil \
  --abmil "$ABMIL_OUT/predictions_abmil_seed${SEED}.csv" \
  --clam "$CLAM_OUT/with_instance/predictions_clam_sb_seed${SEED}.csv" \
  --abmil-label "ABMIL (mock)" \
  --clam-label "CLAM-SB + instance loss (mock)" \
  --out-dir results/mock_comparison

# Every results folder written by this script gets a loud disclaimer.
for dir in "$CLAM_OUT" "$ABMIL_OUT" "$HEATMAP_OUT" "$EVAL_OUT" "$STAT_OUT" results/mock_comparison; do
  mkdir -p "$dir"
  cat > "$dir/README.md" <<'EOF'
# SYNTHETIC MOCK RESULTS -- DO NOT REPORT

Produced by `bash scripts/run_mock_demo.sh` on synthetic slides created by
`scripts/make_mock_features.py`.

* The "slides" are coloured noise, the "features" are random vectors with an
  injected class offset.
* Any QWK / accuracy / F1 / heatmap / checkpoint here is a **plumbing check**.
* Never copy these numbers into the report, slides, or the demo. Re-run
  `scripts/run_person2_experiments.sh` on the real PANDA features for that.

Delete this folder before submission if it is not gitignored.
EOF
done

echo
echo "MOCK DEMO COMPLETE (synthetic -- not reportable)"
echo "  ablation table : $CLAM_OUT/ablation_clam_sb.md"
echo "  comparison     : results/mock_comparison/comparison.md"
echo "  heatmaps       : $HEATMAP_OUT/images/"
echo "  localisation   : $HEATMAP_OUT/localisation_scores.csv"
echo "  attention stats: $EVAL_OUT/attention_clam_sb_seed${SEED}.csv"

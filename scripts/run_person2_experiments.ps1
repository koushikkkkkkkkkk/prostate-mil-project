# Person 2 (CLAM-SB) full experiment pipeline on the REAL PANDA feature bags.
# Windows PowerShell equivalent of scripts/run_person2_experiments.sh.
#
#   powershell -ExecutionPolicy Bypass -File scripts\run_person2_experiments.ps1
#
# Prerequisites (one time):
#   python scripts/download_real_features.py --manifest data/common_split.csv --out-dir data/features
#   pip install h5py
#   python scripts/attach_patch_coords.py --manifest data/common_split.csv --features-dir data/features
#
# Produces: coverage report, instance-loss ablation (3 seeds), per-seed evaluation,
# attention heatmaps, statistical baseline, and the paired ABMIL-vs-CLAM-SB comparison.

$ErrorActionPreference = "Stop"
Set-Location (Split-Path -Parent $PSScriptRoot)
$env:MPLBACKEND = "Agg"

$Manifest = if ($env:MANIFEST)   { $env:MANIFEST }   else { "data/common_split.csv" }
$Features = if ($env:FEATURES)   { $env:FEATURES }   else { "data/features" }
$Thumbs   = if ($env:THUMBS)     { $env:THUMBS }     else { "data/thumbnails" }
$Masks    = if ($env:MASKS)      { $env:MASKS }      else { "data/masks" }
$Out      = if ($env:OUT)        { $env:OUT }        else { "results/metrics/clam_sb" }
$Config   = if ($env:CONFIG)     { $env:CONFIG }     else { "configs/clam_sb.yaml" }
$HeatOut  = if ($env:HEATMAP_OUT){ $env:HEATMAP_OUT }else { "results/heatmaps/clam_sb" }
$CompOut  = if ($env:COMPARE_OUT){ $env:COMPARE_OUT }else { "results/metrics/comparison" }
$StatOut  = if ($env:STAT_OUT)   { $env:STAT_OUT }   else { "results/metrics/statistical" }
$Seeds    = if ($env:SEEDS)      { $env:SEEDS }      else { "0 1 2" }
$AbmilCsv = if ($env:ABMIL_CSV)  { $env:ABMIL_CSV }  else { "results/metrics/abmil/predictions_abmil_seed0.csv" }
$SeedArray = $Seeds -split "\s+"
$FirstSeed = $SeedArray[0]

if (-not (Test-Path $Manifest)) {
    Write-Host "ERROR: manifest not found: $Manifest" -ForegroundColor Red
    Write-Host "  Person 1 owns this file. For the pre-data path use: scripts\run_mock_demo.ps1"
    exit 1
}
if (-not (Test-Path $Features)) {
    Write-Host "ERROR: feature folder not found: $Features" -ForegroundColor Red
    Write-Host "  python scripts/download_real_features.py --manifest $Manifest --out-dir $Features"
    exit 1
}

Write-Host "== 0/6  feature coverage check (must be 300/60/60) =="
python -c @"
import sys
from src.dataset import read_manifest, validate_feature_coverage, format_coverage_table
records = read_manifest('$Manifest')
report = validate_feature_coverage(records, '$Features', fail_fast=False)
print(format_coverage_table(report))
if report['missing']:
    print()
    print('MISSING feature files:', len(report['missing']))
    sys.exit(1)
"@
if ($LASTEXITCODE -ne 0) {
    Write-Host "Feature files are missing - refusing to train on a reduced split." -ForegroundColor Red
    Write-Host "  python scripts/download_real_features.py --manifest $Manifest --out-dir $Features --workers 16"
    exit 1
}

Write-Host "== 1/6  CLAM-SB instance-loss ablation (seeds: $Seeds) =="
python -m src.train_clam_sb --manifest $Manifest --features-root $Features --config $Config --out-dir $Out --seeds @SeedArray

$ClamCkpt = "$Out/with_instance/checkpoints/clam_sb_seed${FirstSeed}_best.pt"

Write-Host "== 2/6  evaluating the seed-$FirstSeed CLAM-SB checkpoint =="
python -m src.evaluate_clam_sb --checkpoint $ClamCkpt --manifest $Manifest --features-root $Features --out-dir "$Out/eval_seed$FirstSeed" --split all

Write-Host "== 3/6  exporting attention heatmaps =="
$thumbArgs = @()
if (Test-Path $Thumbs) { $thumbArgs = @("--thumbnails", $Thumbs) } else { Write-Host "  (no $Thumbs - heatmaps without slide overlay)" }
$maskArgs = @()
if (Test-Path $Masks) { $maskArgs = @("--masks", $Masks) } else { Write-Host "  (no $Masks - localisation overlap score skipped)" }
python -m src.export_clam_attention --checkpoint $ClamCkpt --manifest $Manifest --features-root $Features --out-dir $HeatOut --split test @thumbArgs @maskArgs

Write-Host "== 4/6  statistical baseline (ordinal logistic / XGBoost) =="
python -m src.statistical_models --attention-csv "$Out/eval_seed$FirstSeed/attention_clam_sb_seed$FirstSeed.csv" --manifest $Manifest --out-dir $StatOut

Write-Host "== 5/6  paired ABMIL vs CLAM-SB comparison =="
if (Test-Path $AbmilCsv) {
    python -m src.compare_clam_abmil --abmil $AbmilCsv --clam "$Out/with_instance/predictions_clam_sb_seed${FirstSeed}.csv" --abmil-label "ABMIL" --clam-label "CLAM-SB + instance loss" --out-dir $CompOut
} else {
    Write-Host "  SKIPPED: ABMIL predictions not found at $AbmilCsv" -ForegroundColor Yellow
    Write-Host "  Person 1 produces them with:"
    Write-Host "    python -m src.train --model abmil --manifest $Manifest --features-root $Features --config configs/abmil.yaml --out-dir results/metrics/abmil --seeds 0 1 2"
}

Write-Host "== 6/6  ablation table =="
Get-Content "$Out/ablation_clam_sb.md"

Write-Host ""
Write-Host "Done." -ForegroundColor Green
Write-Host "  ablation     : $Out/ablation_clam_sb.md"
Write-Host "  comparison   : $CompOut/comparison.md"
Write-Host "  heatmaps     : $HeatOut/images/"
Write-Host "  attention    : $Out/eval_seed$FirstSeed/attention_clam_sb_seed$FirstSeed.csv"

# End-to-end Person 2 pipeline on SYNTHETIC mock data (Windows PowerShell).
#
#   powershell -ExecutionPolicy Bypass -File scripts\run_mock_demo.ps1
#
# This is the Windows equivalent of scripts/run_mock_demo.sh. It generates mock
# slides, trains ABMIL (for the paired comparison) and CLAM-SB with/without
# instance loss, evaluates, renders heatmaps with localisation overlap scores,
# fits the statistical baseline, and writes the comparison.
#
# EVERYTHING this produces is synthetic. It proves the plumbing works; it says
# nothing about PANDA performance. Do not report these numbers.

$ErrorActionPreference = "Stop"
Set-Location (Split-Path -Parent $PSScriptRoot)
$env:MPLBACKEND = "Agg"

$Seed        = if ($env:SEED)        { $env:SEED }        else { "0" }
$MockDir     = if ($env:MOCK_DIR)    { $env:MOCK_DIR }    else { "data/mock" }
$ClamOut     = if ($env:CLAM_OUT)    { $env:CLAM_OUT }    else { "results/mock_clam_sb" }
$AbmilOut    = if ($env:ABMIL_OUT)   { $env:ABMIL_OUT }   else { "results/mock_abmil" }
$HeatmapOut  = if ($env:HEATMAP_OUT) { $env:HEATMAP_OUT } else { "results/mock_heatmaps" }
$EvalOut     = if ($env:EVAL_OUT)    { $env:EVAL_OUT }    else { "results/mock_eval" }
$StatOut     = if ($env:STAT_OUT)    { $env:STAT_OUT }    else { "results/mock_statistical" }

Write-Host "== 0/7  checking the environment =="
python scripts/check_env.py
if ($LASTEXITCODE -ne 0) {
    Write-Host "Environment problems detected above. Fix them and re-run." -ForegroundColor Red
    exit 1
}

Write-Host "== 1/7  generating synthetic mock slides =="
python scripts/make_mock_features.py --out-dir $MockDir --seed 7

Write-Host "== 2/7  training ABMIL baseline (1 seed, mock config) =="
python -m src.train --model abmil --manifest "$MockDir/manifest.csv" --features-root "$MockDir/features" --config configs/abmil_mock.yaml --out-dir $AbmilOut --seeds $Seed

Write-Host "== 3/7  CLAM-SB ablation: with vs without instance loss =="
python -m src.train_clam_sb --manifest "$MockDir/manifest.csv" --features-root "$MockDir/features" --config configs/clam_sb_mock.yaml --out-dir $ClamOut --seeds $Seed

$ClamCkpt = "$ClamOut/with_instance/checkpoints/clam_sb_seed${Seed}_best.pt"

Write-Host "== 4/7  evaluating CLAM-SB checkpoint =="
python -m src.evaluate_clam_sb --checkpoint $ClamCkpt --manifest "$MockDir/manifest.csv" --features-root "$MockDir/features" --out-dir $EvalOut --split all

Write-Host "== 5/7  exporting attention heatmaps + localisation overlap =="
python -m src.export_clam_attention --checkpoint $ClamCkpt --manifest "$MockDir/manifest.csv" --features-root "$MockDir/features" --thumbnails "$MockDir/thumbnails" --masks "$MockDir/masks" --out-dir $HeatmapOut --split test

Write-Host "== 6/7  statistical baseline on attention statistics =="
python -m src.statistical_models --attention-csv "$EvalOut/attention_clam_sb_seed${Seed}.csv" --manifest "$MockDir/manifest.csv" --out-dir $StatOut

Write-Host "== 7/7  paired ABMIL vs CLAM-SB comparison =="
python -m src.compare_clam_abmil --abmil "$AbmilOut/predictions_abmil_seed${Seed}.csv" --clam "$ClamOut/with_instance/predictions_clam_sb_seed${Seed}.csv" --abmil-label "ABMIL (mock)" --clam-label "CLAM-SB + instance loss (mock)" --out-dir "results/mock_comparison"

# Every results folder written by this script gets a loud disclaimer.
$disclaimer = @"
# SYNTHETIC MOCK RESULTS -- DO NOT REPORT

Produced by ``scripts/run_mock_demo.ps1`` on synthetic slides created by
``scripts/make_mock_features.py``.

* The "slides" are coloured noise; the "features" are random vectors with an
  injected class offset.
* Any QWK / accuracy / F1 / heatmap / checkpoint here is a **plumbing check**.
* Never copy these numbers into the report, slides, or the demo.

Delete this folder before submission if it is not gitignored.
"@
foreach ($dir in @($ClamOut, $AbmilOut, $HeatmapOut, $EvalOut, $StatOut, "results/mock_comparison")) {
    New-Item -ItemType Directory -Force -Path $dir | Out-Null
    Set-Content -Path (Join-Path $dir "README.md") -Value $disclaimer -Encoding utf8
}

Write-Host ""
Write-Host "MOCK DEMO COMPLETE (synthetic -- not reportable)" -ForegroundColor Green
Write-Host "  ablation table : $ClamOut/ablation_clam_sb.md"
Write-Host "  comparison     : results/mock_comparison/comparison.md"
Write-Host "  heatmaps       : $HeatmapOut/images/"
Write-Host "  localisation   : $HeatmapOut/localisation_scores.csv"
Write-Host "  attention stats: $EvalOut/attention_clam_sb_seed${Seed}.csv"

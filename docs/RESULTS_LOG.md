# Results log

One row per *reportable* run. If a run used synthetic data it does not belong here —
put it in the "mock runs" section at the bottom, which is explicitly non-reportable.

Rule of thumb: if you cannot say exactly which `common_split.csv`, which feature
directory and which config produced a number, it is not reportable.

## Environment of record

| field | value |
| --- | --- |
| date | |
| git commit | |
| split file | `data/common_split.csv` (300 / 60 / 60) |
| feature dir | `data/features` |
| encoder / D | ResNet50 / 1024 |
| patch size | 256 × 256, no overlap |
| seeds | 0, 1, 2 |
| device | |

## Main results (test split, 60 slides)

| model | config | QWK (mean ± sd) | QWK 95% CI | accuracy | balanced acc. | macro F1 | notes |
| --- | --- | --- | --- | --- | --- | --- | --- |
| ABMIL | `configs/abmil.yaml` | 0.7281 ± 0.0132 | seeds 0/1/2: 0.7137 / 0.7308 / 0.7397 | 0.3889 | 0.3889 | 0.3272 | Person 1, 3 seeds |
| CLAM-SB + instance loss | `configs/clam_sb.yaml`, λ=0.30 | 0.6961 ± 0.0164 | see `results/PERSON2_SUMMARY.md` | 0.3944 | 0.3944 | 0.3294 | Person 2, 3 seeds |
| CLAM-SB, no instance loss | `configs/clam_sb.yaml`, λ=0 | **0.7388 ± 0.0040** | see `results/PERSON2_SUMMARY.md` | 0.3944 | 0.3944 | 0.3367 | Person 2 ablation, 3 seeds |
| TransMIL | `configs/transmil.yaml` | pending | | | | | Person 3 |
| Ordinal logistic (x_stat) | `src/statistical_models.py` | 0.7682 | single fit, seed 42 | 0.4500 | — | 0.4100 | Person 4; reuses CLAM-SB attention stats |
| XGBoost (x_stat) | `src/statistical_models.py` | not run | | | | | Person 4, optional; xgboost not installed |

## Attention / localisation analysis (Person 2)

| arm | attention entropy | normalised entropy | top-10 attention share | top-20% overlap precision | lift |
| --- | --- | --- | --- | --- | --- |
| with instance loss | pending (`evaluate_clam_sb --split all`, seeds 0–2) | | | | |
| without instance loss | pending | | | | |

_Note: mask-based overlap (precision / lift) is not computable on the feature-only release — no thumbnails, no masks. Heatmaps cover 56/60 test slides (4 slides have no patch coordinates). See `report/person2_section.md` §7._

## Paired statistics

| comparison | ΔQWK (95% bootstrap CI) | McNemar p | interpretation |
| --- | --- | --- | --- |
| CLAM-SB − ABMIL | seed 0: +0.0003 [−0.1113, +0.1076] | 1.0000 | indistinguishable at n=60; seeds 1–2 pending |
| CLAM-SB(λ=0.3) − CLAM-SB(λ=0) | mean −0.0427 over 3 seeds (per-seed CIs in `results/PERSON2_SUMMARY.md`) | per-seed | instance loss did not help; also destabilised early stopping |

## Mock runs (NOT reportable — for reference only)

| date | command | purpose |
| --- | --- | --- |
| | `bash scripts/run_mock_demo.sh` | plumbing check on synthetic slides |

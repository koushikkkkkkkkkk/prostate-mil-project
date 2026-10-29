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
| ABMIL | `configs/abmil.yaml` | | | | | | Person 1 |
| CLAM-SB + instance loss | `configs/clam_sb.yaml` | | | | | | Person 2 |
| CLAM-SB, no instance loss | `configs/clam_sb.yaml`, λ=0 | | | | | | Person 2 ablation |
| TransMIL | `configs/transmil.yaml` | | | | | | Person 3 |
| Ordinal logistic (x_stat) | `src/statistical_models.py` | | | | | | Person 4 |
| XGBoost (x_stat) | `src/statistical_models.py` | | | | | | Person 4, optional |

## Attention / localisation analysis (Person 2)

| arm | attention entropy | normalised entropy | top-10 attention share | top-20% overlap precision | lift |
| --- | --- | --- | --- | --- | --- |
| with instance loss | | | | | |
| without instance loss | | | | | |

## Paired statistics

| comparison | ΔQWK (95% bootstrap CI) | McNemar p | interpretation |
| --- | --- | --- | --- |
| CLAM-SB − ABMIL | | | |
| CLAM-SB(λ=0.3) − CLAM-SB(λ=0) | | | |

## Mock runs (NOT reportable — for reference only)

| date | command | purpose |
| --- | --- | --- |
| | `bash scripts/run_mock_demo.sh` | plumbing check on synthetic slides |

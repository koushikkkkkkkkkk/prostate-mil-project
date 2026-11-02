# Reproducibility status

**Checked on 2 Nov 2026** with `python scripts/audit_artifacts.py`, which re-runs on every
push (`.github/workflows/tests.yml`, job `artifact-audit`) and earlier tests can never
drift from the committed numbers silently.

What "reproducible" means here: given the repository plus the public feature download, you
can **re-evaluate** the committed predictions and **re-run** the training, and land on the
same numbers. A run whose weights are missing is *not* reproducible — its metrics are
reported, but nobody can interrogate them.

## The five trained families

| run | encoder | weights in repo | features | status |
| --- | --- | --- | --- | --- |
| `abmil` (3 seeds) | ResNet50 1024-d | ✅ committed | public download | **fully reproducible** |
| `transmil` (3 seeds) | ResNet50 1024-d | ✅ committed | public download | **fully reproducible** |
| `clam_sb/with_instance` (3 seeds) | ResNet50 1024-d | ✅ committed | public download | **fully reproducible** |
| `clam_sb/without_instance` (3 seeds) | ResNet50 1024-d | ✅ committed | public download | **fully reproducible** |
| `abmil_chief` (3 seeds) | CHIEF 768-d | ❌ not committed | public download | **metrics only** |
| `clam_sb_chief` (3 seeds) | CHIEF 768-d | ❌ not committed | public download | **metrics only** |
| `abmil_gigap` (3 seeds) | GigaPath 1536-d | ❌ not committed | public download | **metrics only** |

Reproduce everything on the ResNet50 cohort from a clean clone:

```bash
python scripts/download_features_and_coords.py          # ~600-800 MB, once
python scripts/verify_data.py --require all             # ResNet50 row must be OK
python -m src.train --model abmil --seeds 0 1 2         # + --model transmil
python -m src.train_clam_sb --seeds 0 1 2               # both ablation arms at once
python scripts/summarize_results.py && python scripts/paired_results.py
python scripts/audit_artifacts.py                       # must be 0 fatal
```

## The three foundation-encoder runs: an honest statement

`abmil_chief`, `clam_sb_chief` and `abmil_gigap` carry the project's headline numbers
(QWK 0.9124 / 0.9224 / 0.9172). **Their weights were never committed and no longer exist
on the authors' machines.** Their per-seed predictions, metrics, histories and configs *are*
committed, so every statistic in the report can be recomputed and audited — but the models
cannot be re-evaluated or probed.

Say it plainly in the report; do not let a reviewer discover it:

> The CHIEF and GigaPath runs are reported from committed predictions and metrics, not from
> committed weights. Their training occupies a few minutes of GPU time per seed and can be
> reproduced exactly from the public feature release with the commands below, but as of this
> submission they have not been re-run.

To close the gap (one Colab session, ~30 min):

```bash
python scripts/download_alt_encoder_features.py --encoder chief    # 768-d, ~700 MB
python scripts/download_alt_encoder_features.py --encoder gigap    # 1536-d, ~900 MB
python scripts/verify_data.py --require all                        # all four cohorts OK

python -m src.train --model abmil   --manifest data/common_split_chief.csv \
    --features-root data/features_chief --config configs/abmil_chief.yaml \
    --out-dir results/metrics/abmil_chief  --seeds 0 1 2
python -m src.train_clam_sb --manifest data/common_split_chief.csv \
    --features-root data/features_chief --config configs/clam_sb_chief.yaml \
    --out-dir results/metrics/clam_sb_chief --seeds 0 1 2
python -m src.train --model abmil   --manifest data/common_split_gigap.csv \
    --features-root data/features_gigap --config configs/abmil_gigap.yaml \
    --out-dir results/metrics/abmil_gigap --seeds 0 1 2
```

Then commit `results/metrics/{abmil_chief,clam_sb_chief,abmil_gigap}/checkpoints/*.pt`
(~3 MB each, 9 files) and `audit_artifacts.py` reaches **0 fatal, 0 warnings**.

## What is already enforced automatically

| check | where | fails the build? |
| --- | --- | --- |
| 54 unit + end-to-end tests | `.github/workflows/tests.yml` | yes |
| `pyflakes` over `src scripts app tests` | same | yes |
| paired tables regenerated and diffed against the commit | same | yes |
| no slide appears in two splits, no duplicate ids | same | yes |
| dangling checkpoint references, missing weights, missing ablation | `scripts/audit_artifacts.py` | report-only (green once the 9 CHIEF/GigaPath checkpoints land) |

The audit job is deliberately `continue-on-error` until those weights exist: a red build
would hide real failures behind a known, documented one.

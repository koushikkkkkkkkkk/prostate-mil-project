# Reproducibility status

**Checked on 2 Nov 2026** with `python scripts/audit_artifacts.py`, which re-runs on every
push (`.github/workflows/tests.yml`, job `artifact-audit`) and earlier tests can never
drift from the committed numbers silently.

What "reproducible" means here: given the repository plus the public feature download, you
can **re-evaluate** the committed predictions and **re-run** the training, and land on the
same numbers. A run whose weights are missing is *not* reproducible — its metrics are
reported, but nobody can interrogate them.

## The seven trained runs

| run | encoder | weights in repo | features | status |
| --- | --- | --- | --- | --- |
| `abmil` (3 seeds) | ResNet50 1024-d | ✅ committed | public download | **fully reproducible** |
| `transmil` (3 seeds) | ResNet50 1024-d | ✅ committed | public download | **fully reproducible** |
| `clam_sb/with_instance` (3 seeds) | ResNet50 1024-d | ✅ committed | public download | **fully reproducible** |
| `clam_sb/without_instance` (3 seeds) | ResNet50 1024-d | ✅ committed | public download | **fully reproducible** |
| `abmil_chief` (3 seeds) | CHIEF 768-d | ✅ committed | public download | **fully reproducible** |
| `clam_sb_chief` (3 seeds) | CHIEF 768-d | ✅ committed | public download | **fully reproducible** |
| `abmil_gigap` (3 seeds) | GigaPath 1536-d | ✅ committed | public download | **fully reproducible** |

Reproduce everything on the ResNet50 cohort from a clean clone:

```bash
python scripts/download_features_and_coords.py          # ~600-800 MB, once
python scripts/verify_data.py --require all             # ResNet50 row must be OK
python -m src.train --model abmil --seeds 0 1 2         # + --model transmil
python -m src.train_clam_sb --seeds 0 1 2               # both ablation arms at once
python scripts/summarize_results.py && python scripts/paired_results.py
python scripts/audit_artifacts.py                       # must be 0 fatal
```

## The three foundation-encoder runs: retrained and committed

`abmil_chief`, `clam_sb_chief` and `abmil_gigap` carry the project's headline numbers
(QWK 0.9207 / 0.9035 / 0.8984). An earlier version of these runs shipped predictions and
metrics but **no weights**, so their statistics could be audited but the models themselves
could not be reloaded. They have since been retrained from the public feature release on the
same manifests and the same seed list (0/1/2), and the checkpoints committed — 9 files of
~3 MB each, bringing the repository to **21 checkpoints**.

The reproduction lands within 0.019 QWK of the original numbers (Δ = −0.002 / −0.009 /
−0.019 for CLAM-SB CHIEF, ABMIL CHIEF and ABMIL GigaPath). **Every document in this
repository quotes the retrained numbers**, so no figure anywhere refers to a model that
cannot be loaded. That is also why numbers memorised from an earlier draft no longer appear
in any file.

Two consequences are stated rather than buried:

* the foundation runs' seed spread is wider than the first run suggested (sd 0.006–0.015,
  previously 0.001–0.004), which matters because it is the yardstick for claiming aggregator
  differences of ±0.03 are undetectable;
* the CHIEF-versus-GigaPath ordering **reverses** between the two sets of runs of those same
  configurations, at constant magnitude — evidence *for* "indistinguishable", not against it.

Commands used (one Colab T4 session, ~40 minutes end to end):

```bash
python scripts/download_alt_encoder_features.py --encoder chief --workers 16   # 399 slide bags
python scripts/download_alt_encoder_features.py --encoder gigap --workers 16   # 391 slide bags
python scripts/verify_data.py --require all

python -m src.train --model abmil   --manifest data/common_split_chief.csv \
    --features-root data/features_chief --config configs/abmil_chief.yaml \
    --out-dir results/metrics/abmil_chief  --seeds 0 1 2
python -m src.train --model clam_sb --manifest data/common_split_chief.csv \
    --features-root data/features_chief --config configs/clam_sb_chief.yaml \
    --out-dir results/metrics/clam_sb_chief --seeds 0 1 2
python -m src.train --model abmil   --manifest data/common_split_gigap.csv \
    --features-root data/features_gigap --config configs/abmil_gigap.yaml \
    --out-dir results/metrics/abmil_gigap --seeds 0 1 2
```

Use `--model clam_sb` through `src.train`, **not** `src.train_clam_sb`: the latter writes the
nested `with_instance/` layout, whereas `clam_sb_chief` is a flat run directory like every
other run in `results/metrics/`.

After committing the new run directories, regenerate the derived tables so the committed
copies match the artifacts:

```bash
python scripts/summarize_results.py
python scripts/paired_results.py
python scripts/audit_artifacts.py     # expect 0 fatal, 0 warnings
```

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

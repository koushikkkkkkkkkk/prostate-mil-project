# Scripts

Every script here is runnable from the repository root. Paths are relative to it.

## Data acquisition and verification

| script | what it does |
| --- | --- |
| `verify_data.py` | **run this first.** Checks every manifest against the features on disk: rows, missing files, feature dim, coords. `--require all` exits 1 on any gap. |
| `download_features_and_coords.py` | ResNet50 (1024-d) features + patch coords from HuggingFace `Dearcat/CPathPatchFeature` into `data/features/`. |
| `download_alt_encoder_features.py` | CHIEF (768-d) or GigaPath (1536-d) features into `data/features_chief` / `data/features_gigap`. `--encoder chief|gigap`. |
| `download_real_features.py` | legacy r50-only downloader (features, no coords). Prefer the two above. |
| `attach_patch_coords.py` | pulls `panda/patches/{id}.h5` and rewrites each bag as `{features, coords}`. `--verify-only` prints coverage. |
| `make_coords_manifest.py` | writes `data/common_split_with_coords.csv` (only slides that can be plotted) **and** `results/metrics/coord_coverage.csv` (per-slide reason). Supersedes the removed `make_coord_manifest.py`. |
| `create_split.py`, `verify_split.py` | build / check the stratified 300-60-60 split. |
| `check_env.py` | versions + broken-combination detector (numpy 1.x on 3.13+, torch/torchvision mismatch). |
| `check_mask_encoding.py` | audits the Karolinska masks: 99.88 % background, 0.12 % stroma, **zero Gleason pixels** across 24 slides. Evidence for why patch-level supervision is impossible. |
| `download_thumbnails_and_masks.py` | thumbnails/masks for visualisation only (not needed for any metric). |

## Training and evaluation drivers

| script | what it does |
| --- | --- |
| `run_person2_experiments.{sh,ps1}` | CLAM-SB ablation end to end: train 3 seeds both arms, evaluate, export heatmaps, paired comparison. |
| `run_person4_experiments.sh` | localisation, attention agreement, statistical baselines. |
| `run_mock_demo.{sh,ps1}` | synthetic pipeline smoke test. **Never reportable.** |
| `make_mock_features.py` | generates the synthetic bags used by the mock demo. |

## Results assembly (no compute, safe to re-run any time)

| script | writes |
| --- | --- |
| `summarize_results.py` | `results/ALL_RESULTS.md` — every trained run with provenance, per-seed QWK, paired tables, ablation, statistical baselines, heatmap summaries. |
| `paired_results.py` | `results/metrics/comparison/paired_{native,common}_cohort.{md,csv}` + `paired_stats.{md,json}` — recomputed from the prediction CSVs only. |
| `summarize_person2.py` | `results/PERSON2_SUMMARY.md` — Person 2 detail (ablation, attention sharpness, localisation coverage). |
| `attention_agreement.py` | cross-model attention agreement (Spearman / Jaccard / Cohen kappa). |
| `fuse_encoders.py` | CHIEF + GigaPath prediction fusion. |

## One-offs kept for provenance

`cmp57.py` (restrict results to the 57 CHIEF slides), `cohort_report.py` (per-cohort
coverage), `mkcfg.py` (config generator), `regen_aggregate.py` (rebuild an aggregate JSON
from per-seed metrics), `build_colab_notebook.py` (notebook scaffolding). They are
referenced from `report/person4_results.md`; each is small and single-purpose. Prefer
`summarize_results.py` and `paired_results.py` for anything going into the report.

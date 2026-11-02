# Results log

> **Which of these runs can a reviewer actually reproduce?** All of them. See
> [`docs/REPRODUCIBILITY.md`](REPRODUCIBILITY.md): all seven runs ship their weights
> (21 checkpoints), including the three foundation-encoder runs that carry the headline
> numbers.


**This file is a pointer, not a data source.** Every number lives in
`results/ALL_RESULTS.md`, which is *generated* from the artifacts by
`python scripts/summarize_results.py`. Tables typed by hand drifted from the
artifacts three times in this project (a "pending" TransMIL that had been trained, a
statistical baseline quoted as 0.7682 when the artifact said 0.7399, and a CLAM-SB
ablation with two different values in two documents). Do not add numbers here;
regenerate, then quote.

```bash
python scripts/summarize_results.py     # -> results/ALL_RESULTS.md
python scripts/paired_results.py        # -> results/metrics/comparison/paired_*.md
python scripts/summarize_person2.py     # -> results/PERSON2_SUMMARY.md (Person 2 detail)
```

Rule of thumb: if you cannot say exactly which `common_split*.csv`, which feature
directory and which config produced a number, it is not reportable.

## Environment of record

| field | value |
| --- | --- |
| split files | `data/common_split.csv` (300/60/60, r50) · `common_split_chief.csv` (286/56/57) · `common_split_gigap.csv` (281/57/53) |
| feature dirs | `data/features` (r50, 1024-d) · `data/features_chief` (768-d) · `data/features_gigap` (1536-d) |
| patch size | 256 × 256, no overlap |
| seeds | 0, 1, 2 (every run) |
| selection | validation QWK only; test evaluated once per configuration |

## Headline: aggregator vs encoder (50 common test slides)

Restricted to the slides **every** run covers, so the rows are directly comparable
(source: `results/metrics/comparison/paired_common_cohort.md`):

| model | encoder | n | QWK | accuracy |
| --- | --- | --- | --- | --- |
| CLAM-SB | CHIEF (768-d) | 50 | **0.9278 ± 0.0074** | 0.6067 |
| ABMIL | CHIEF (768-d) | 50 | 0.9115 ± 0.0110 | 0.5667 |
| ABMIL | GigaPath (1536-d) | 50 | 0.8907 ± 0.0132 | 0.5667 |
| TransMIL | ResNet50 (1024-d) | 50 | 0.7113 ± 0.0583 | 0.4733 |
| ABMIL | ResNet50 (1024-d) | 50 | 0.6995 ± 0.0136 | 0.3667 |

Paired statistics (`paired_stats.md`):

* **Encoder effect** — ABMIL, same aggregator, ResNet50 → CHIEF: ΔQWK +0.18 to +0.21,
  McNemar p 0.004–0.019, CI excludes zero → **real**.
* **Encoder effect** — CHIEF vs GigaPath: ΔQWK −0.02 to +0.01, CIs span zero →
  the two foundation encoders are indistinguishable here.
* **Aggregator effect** — fixed encoder: CLAM-SB − ABMIL on CHIEF = +0.007 to +0.012
  (McNemar p 0.69–1.0); TransMIL − ABMIL on r50 = −0.01 to +0.03 (p 0.08–1.0) →
  **inconclusive at n = 50–60**, not "better".

Because the CI spans zero, write "no detectable difference", never "equivalent".

## Every run on its own test set

See `results/ALL_RESULTS.md` §1 (with provenance and a *weights in repo* column).
Summary: ABMIL r50 0.7281 ± 0.0132 (60) · CLAM-SB r50 0.7035 ± 0.0088 with instance loss,
0.7271 ± 0.0223 without (60)
· TransMIL r50 0.7446 ± 0.0364 (60) · CLAM-SB CHIEF 0.9207 ± 0.0063 (57) · ABMIL
CHIEF 0.9035 ± 0.0145 (57) · ABMIL GigaPath 0.8984 ± 0.0108 (53).

## Ablation: CLAM-SB with vs without instance loss — **run and committed 2 Nov 2026**

Generated from `results/metrics/clam_sb/{with,without}_instance/`; do not retype it:

| Setting | QWK (mean) | QWK (sd) | Accuracy | Macro F1 | Seeds |
| --- | --- | --- | --- | --- | --- |
| CLAM-SB + instance loss | 0.7035 | 0.0088 | 0.3833 | 0.3142 | 3 |
| CLAM-SB, no instance loss | 0.7271 | 0.0223 | 0.3944 | 0.3386 | 3 |

ΔQWK (with − without) = **−0.0236**, but the per-seed sign flips
(−0.0114 / +0.0345 / +0.0476) and every 95 % bootstrap CI spans zero, so the claim is
**no detectable effect at n = 60**, not "the loss hurts". The 0.6961 / 0.7388 pair quoted
in older documents did not reproduce and is retired.

```bash
python -m src.train_clam_sb --seeds 0 1 2      # retrains both arms
python scripts/summarize_results.py && python scripts/paired_results.py
```

## Attention / localisation

| arm | slides | normalised entropy |
| --- | --- | --- |
| ABMIL (r50, seed 0) | 420 | 0.7245 |
| CLAM-SB (r50, seed 0) | 420 | 0.8003 |

Cross-model agreement (56 shared test slides, `attention/agreement_abmil_vs_clam_summary.json`):
Spearman ρ 0.869, top-k Jaccard 0.502, top-k Cohen κ 0.608.

Mask-based overlap is **not computable**: no thumbnails/masks in the feature-only
release, and the Karolinska masks contain zero Gleason pixels
(`scripts/check_mask_encoding.py`). Heatmaps cover **56/60** test slides; the 21
coord-less slides are listed in `results/metrics/coord_coverage.csv`.

## Mock runs (NOT reportable — reference only)

| command | purpose |
| --- | --- |
| `bash scripts/run_mock_demo.sh` | plumbing check on synthetic slides |

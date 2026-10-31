# Person 2 handbook — CLAM-SB, instance-level supervision, heatmaps

You own CLAM-SB. This document is the whole job in one page: what to build, how to run
it, what to report, and what you are allowed to claim from which numbers.

> Headline sentence for the report:
> **"CLAM-SB tests whether forcing the model to distinguish high- and low-attention
> patches improves weakly supervised grading."**

---

## 1. Deliverables (from the brief)

| # | Deliverable | Where it lives | Status |
| --- | --- | --- | --- |
| 1 | CLAM-SB implemented from scratch in PyTorch | `src/models/clam_sb.py` | ✅ |
| 2 | Instance-level pseudo-labelling (top-`B` positive, bottom-`B` negative) | `CLAMSB._instance_clustering_loss` | ✅ |
| 3 | Ablation: CLAM-SB with vs without instance loss | `src/train_clam_sb.py` → `ablation_clam_sb.md` | ✅ |
| 4 | Comparison against ABMIL on the shared split | `src/compare_clam_abmil.py` → `comparison.md` | ✅ (needs Person 1 predictions) |
| 5 | Attention heatmaps + attention distributions | `src/export_clam_attention.py`, `src/evaluate_clam_sb.py` | ✅ |
| 6 | Quantitative localisation overlap score | `src/heatmaps.py::evaluate_localisation` → `localisation_scores.csv` | ✅ |
| 7 | Report section: does instance loss help grading *and* sharpness? | `report/` | ⛔ blocked on real features |

---

## 1b. Can you start before Person 1 finishes? — Yes

```bash
bash scripts/run_mock_demo.sh      # ~1 minute on CPU
```

That script fabricates 30 synthetic slides that obey the exact file contract Person 1
must deliver, and runs your entire pipeline: ablation, evaluation, heatmaps, localisation
scores, statistical baseline, paired comparison.

**Use mock results for:** shapes, imports, loss behaviour, plots, notebook output,
write-up structure, demo UI.
**Never use mock results for:** any number in the report, slides, or demo. Every mock
output folder contains a `README.md` that says so — keep it there.

When Person 1's split + features arrive, you change paths and nothing else:

```bash
MANIFEST=data/common_split.csv FEATURES=data/features bash scripts/run_person2_experiments.sh
# Windows:
powershell -ExecutionPolicy Bypass -File scripts\run_person2_experiments.ps1
```

### 1c. One extra step the downloader does not cover: patch coordinates

`scripts/download_real_features.py` fetches ``features [N, 1024]`` only — no
``coords``. Without coordinates there are no spatial heatmaps and no localisation
score, so run:

```bash
pip install h5py
python scripts/attach_patch_coords.py --manifest data/common_split.csv --features-dir data/features
python scripts/attach_patch_coords.py --verify-only     # coverage report
```

It pulls `panda/patches/{image_id}.h5` from the same Hugging Face dataset
(a tiny `coords [N,2]` tensor, row-aligned with the features) and rewrites each
`.pt` as `{"features", "coords"}`.

**Known gap:** 399 / 420 slides have a coords file on the Hub; 21 do not
(404). Of those, **4 are test slides**, so heatmaps + localisation cover
**56 / 60** test slides. State that in the report; do not silently drop them
and do not pretend the heatmap analysis used all 60.

---

## 2. The contract you depend on (Person 1)

```
data/common_split.csv            image_id, data_provider, isup_grade, gleason_score, split
data/features/{image_id}.pt      {"features": [N, D], "coords": [N, 2], "patch_paths": [N]}
data/thumbnails/{image_id}.png   optional, for overlays
data/masks/{image_id}.png        optional, for the overlap score
```

`[N, D]` alone is enough to train; `coords` is required for spatial heatmaps. `D` is
auto-detected from the files, so the config's `in_dim` is only a fallback.

Send Person 1 this message:

> I'm consuming `data/common_split.csv` + one `{image_id}.pt` per slide with
> `features [N,D]` and `coords [N,2]`. Please tell me `D` and whether you can also drop
> slide thumbnails into `data/thumbnails/` — that gives me overlay heatmaps, plus
> `data/masks/` if you have tissue/tumour masks for the localisation score.

---

## 3. Model: what is implemented

```
h_n    = Dropout(ReLU(W_e x_n))                    [N, D] → [N, 512]
a_n    = softmax_n( wᵀ( tanh(V h_n) ⊙ sigmoid(U h_n) ) )
z      = Σ_n a_n h_n                               [512]
logits = W_c z                                     [6]        → L_slide (CE, class-weighted)

top-B / bottom-B attention patches (B = k_sample = 8)
  → class-specific binary head for the *true* grade
  → smooth top-1 SVM loss                             → L_instance

L_total = L_slide + λ · L_instance,   λ = 0.30
```

* `λ = 0` → the ablation arm (identical architecture, identical parameter count —
  asserted in `tests/test_models_and_metrics.py::test_ablation_keeps_architecture_identical`).
* `--no-instance-loss` on the CLI does the same thing.
* `subtyping: true` enables the paper's extra out-of-class instance terms (off by default).
* `instance_loss: ce` switches the smooth SVM for plain cross-entropy (useful sanity check).
* Implemented from the paper; **no code copied** from `mahmoodlab/CLAM`.

---

## 4. Run order

```bash
# 0. sanity
python -m unittest discover -s tests -v

# 1. ablation (3 seeds x 2 arms)
python -m src.train_clam_sb \
  --manifest data/common_split.csv --features-root data/features \
  --config configs/clam_sb.yaml --out-dir results/metrics/clam_sb --seeds 0 1 2

# 2. evaluate the best checkpoint + dump attention statistics
python -m src.evaluate_clam_sb \
  --checkpoint results/metrics/clam_sb/with_instance/checkpoints/clam_sb_seed0_best.pt \
  --manifest data/common_split.csv --features-root data/features \
  --out-dir results/metrics/clam_sb/eval_seed0 --split all

# 3. heatmaps (+ overlap score when masks exist)
python -m src.export_clam_attention \
  --checkpoint results/metrics/clam_sb/with_instance/checkpoints/clam_sb_seed0_best.pt \
  --manifest data/common_split.csv --features-root data/features \
  --thumbnails data/thumbnails --masks data/masks \
  --out-dir results/heatmaps/clam_sb_seed0 --split test

# 4. paired comparison vs ABMIL (Person 1's predictions)
python -m src.compare_clam_abmil \
  --abmil results/metrics/abmil/predictions_abmil_seed0.csv \
  --clam  results/metrics/clam_sb/with_instance/predictions_clam_sb_seed0.csv \
  --out-dir results/metrics/comparison

# or everything at once:
bash scripts/run_person2_experiments.sh
```

---

## 5. What to report (and how to phrase it)

**Table 1 — ablation (mean ± sd over 3 seeds):**

| Setting | QWK | Accuracy | Balanced acc. | Macro F1 |
| --- | --- | --- | --- | --- |
| CLAM-SB + instance loss | | | | |
| CLAM-SB, no instance loss | | | | |

**Table 2 — attention sharpness** (from `attention_*.csv`): attention entropy,
normalised entropy, top-10 attention share, for both arms. This is the "does it get
sharper?" half of the claim.

**Table 3 — vs ABMIL** (paired, same test slides): QWK difference with 95% bootstrap CI
and the exact McNemar p-value. Say "inconclusive" when the CI spans zero — with 60 test
slides that will happen often, and pretending otherwise is the fastest way to lose marks.

**Figures:** ablation bar chart with error bars, attention distribution histograms for
both arms, ≥ 10 heatmap overlays, 2-3 top-patch contact sheets, localisation score bars
(at top-10/20/30% of patches).

**Wording to avoid:** "CLAM-SB outperforms ABMIL" without a CI; "the heatmap shows the
tumour" (it shows model attention); "the model is accurate" (accuracy alone is misleading
under ISUP imbalance).

---

## 6. Files you own

```
src/models/clam_sb.py            model + smooth top-1 SVM loss
src/train_clam_sb.py             ablation driver
src/evaluate_clam_sb.py          checkpoint evaluation + attention export
src/export_clam_attention.py     heatmaps, top-k patches, overlap scores
src/compare_clam_abmil.py        paired statistics vs ABMIL
configs/clam_sb.yaml             real hyperparameters
configs/clam_sb_mock.yaml        tiny config for the mock demo
scripts/run_person2_experiments.sh
notebooks/04_clam_training.ipynb
tests/test_models_and_metrics.py (CLAM-related tests)
```

Shared files you may read but not change alone: `src/dataset.py`, `src/train.py`,
`src/evaluate.py`, `src/heatmaps.py`.

---

## 7. Design decisions worth defending in the viva

* **Why `k_sample = 8`?** The paper's default. It is a hyperparameter; with ~500-1500
  patches per slide it selects ~1-3% of patches as pseudo-labelled instances.
* **Why `λ = 0.30`?** From the paper's range; it is a *default*, not tuned on test. If you
  tune it, tune on validation and say so.
* **Why an SVM-style loss?** CLAM's patch evidence is binary and noisy; a soft-margin
  hinge (smoothed via softplus) tolerates mislabelled extremes better than CE.
* **Why not class-specific attention branches (CLAM-MB)?** Out of scope: with 6 grades,
  MB needs 6 attention branches and much more data than 300 slides; SB is the fair
  comparison against ABMIL's single branch.
* **Why is the ablation architecture-identical?** So any difference is attributable to the
  loss, not to extra parameters.

---

## 8. Known gaps / next steps

* Localisation overlap uses coarse tissue masks — it is a sanity check, not a
  segmentation metric. If Person 4 produces tumour masks, drop them in `data/masks/`.
* Heatmaps are rendered on a coarse patch grid (one colour block per 256×256 tile); that is
  inherent to patch-level MIL, not a bug.
* The Streamlit app reads CLAM-SB checkpoints directly, so your model appears in the demo
  without extra work.

---

## One command to assemble the report numbers

After training / evaluating, run:

```powershell
python scripts\summarize_person2.py --results results --out results\PERSON2_SUMMARY.md
```

It reads `results\metrics\clam_sb\{with_instance,without_instance}\aggregate_clam_sb.json`,
`results\metrics\abmil\aggregate_abmil.json`, every `predictions_*_seed*.csv`,
`results\metrics\clam_sb\eval*\attention_*.csv`, `results\heatmaps\*\export_summary.json`
and `results\metrics\statistical\statistical_baselines.json`, then writes a single
report-ready markdown file:

* instance-loss ablation table (QWK mean ± sd, accuracy, balanced accuracy, macro F1)
* **paired** per-seed ΔQWK with bootstrap CI + exact McNemar, CLAM±inst and CLAM vs ABMIL
* attention sharpness (entropy, normalised entropy, top-10 attention share) per arm
* heatmap coverage and the known 56/60 coordinate gap
* statistical baselines and an interpretation skeleton

Missing inputs are reported as "not run yet" instead of crashing, so it is safe to run at any
point. Covered by `tests/test_summarize_person2.py`.

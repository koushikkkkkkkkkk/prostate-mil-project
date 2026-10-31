# Person 2 — CLAM-SB with and without instance-level supervision

**Deliverable owner:** Person 2 (Archit Renjeev)
**Task:** weakly supervised ISUP grade prediction from PANDA whole-slide patches, with a
mandatory ablation of CLAM-SB's instance-level clustering loss.
**Status of the numbers in this file:** final 3-seed results on the locked split. Every table
below is reproduced verbatim by `python scripts\summarize_person2.py`, which reads only the
files under `results\`. Re-run that command before submitting so the report cannot drift
from the artifacts.

> Research prototype. Not a clinical tool, not validated for diagnostic use.

---

## 1. Question

CLAM's contribution over plain attention-MIL is not the attention mechanism — ABMIL already
has one — it is the **instance-level clustering loss** that pulls high-attention patches
towards the slide label and pushes low-attention patches away. If that loss does not help on
this dataset, then the extra machinery is not justified and plain ABMIL is the better
engineering choice. So the ablation is the actual scientific content of this deliverable, not
an appendix.

**Ablation design.** Identical network, identical features, identical split, identical seeds,
identical optimiser schedule. The *only* change is `instance_loss_weight`
(0.30 vs 0.0). The instance classifiers and the auxiliary branch are still constructed in the
0.0 arm, so this is a clean **loss** ablation, not a silent architecture ablation.

## 2. Setup

| Item | Value |
| --- | --- |
| Data | PANDA prostate biopsy patches, pre-extracted 1024-d features + patch coordinates |
| Slides with features | 420 (399 also carry `coords`) |
| Split (locked, shared with Person 1) | train 300 / val 60 / test 60, stratified by ISUP |
| Labels | slide-level ISUP grade only (6 classes, 0–5) |
| Model | CLAM-SB, implemented from scratch in PyTorch (`src/models/`); gated attention, single branch |
| Seeds | 0, 1, 2 |
| Selection | checkpoint by validation QWK; early stopping on validation QWK plateau |
| Primary metric | quadratic weighted kappa (ordinal) |
| Secondary metrics | accuracy, balanced accuracy, macro F1, per-class confusion, bootstrap CI, McNemar |
| Localisation | attention heatmaps over patch coordinates, evaluation only (6 slides it cannot cover, see §7) |

Slide-level labels only. Tissue masks are never used for training or model selection.

## 3. Main result — instance-loss ablation

Test split, 60 slides, mean ± sd over 3 seeds:

| Setting | QWK (mean) | QWK (sd) | Accuracy | Balanced acc. | Macro F1 | Seeds |
| --- | --- | --- | --- | --- | --- | --- |
| **CLAM-SB + instance loss** | **0.6961** | 0.0164 | 0.3944 | 0.3944 | 0.3294 | 3 |
| **CLAM-SB, no instance loss** | **0.7388** | 0.0040 | 0.3944 | 0.3944 | 0.3367 | 3 |
| ABMIL baseline (Person 1) | 0.7281 | 0.0132 | 0.3889 | 0.3889 | 0.3272 | 3 |

**ΔQWK (with − without) = −0.0427.**

Read this carefully, because it is the whole point of the deliverable:

* The instance loss **did not help**, and the direction is negative.
* The effect is **larger than seed noise**: the between-seed sd of the difference is
  ~0.015–0.020, while the gap is 0.043. Three seeds is still a small sample, so the correct
  phrasing is "consistently negative over 3 seeds", not "significantly worse" — the paired
  bootstrap CI in §4 is what decides that.
* **Accuracy is identical (0.3944) in both arms.** The kappa difference comes from *where*
  the errors land in ordinal space, not from how many slides are right. Quoting accuracy
  alone would have hidden the entire effect. This is why every table carries accuracy next
  to QWK.

## 4. Paired statistics

Per-seed pairing on the same 60 test slides (paired bootstrap CI for ΔQWK, exact McNemar on
per-slide correctness). These tables are generated from the per-seed
`predictions_*.csv` files.

### 4.1 CLAM-SB + instance loss vs CLAM-SB, no instance loss

_Paste table from `results\PERSON2_SUMMARY.md` §2 after running the summarizer — the per-seed
predictions already exist from training, so no extra compute is needed._

### 4.2 CLAM-SB vs ABMIL (seed 0, n = 60)

| Comparison | QWK ABMIL | QWK CLAM-SB | ΔQWK | 95% CI | McNemar p |
| --- | --- | --- | --- | --- | --- |
| CLAM-SB + instance loss vs ABMIL | 0.7137 | 0.7140 | **+0.0003** | [−0.1113, +0.1076] | 1.0000 |

At n = 60 these two models are **statistically indistinguishable**. The honest claim is
therefore "CLAM-SB matches ABMIL on this split"; claiming a CLAM improvement would be
over-reading a three-slide difference inside a ±0.11 interval.

## 5. Why the instance loss hurt — training dynamics

This is not just a metric difference; the instalment-loss arm trained worse.

| Seed | With instance loss (val QWK) | Without instance loss (val QWK) |
| --- | --- | --- |
| 0 | plateau 0.5134, **no early stop** through epoch 50 | stopped epoch 43, best epoch 31, 0.6050 |
| 1 | plateau 0.4934, **no early stop** through epoch 50 | plateau 0.6078, ran to 50 |
| 2 | stopped epoch 41, best epoch 29, 0.4703 | plateau 0.5495, ran to 50 |

Train loss fell faster in the with-instance arm (2.19 → 1.65 vs 1.80 → 1.42 over the same
horizon) while validation stalled lower. That is the signature of the instance loss pushing
the bag representation to fit patch-level pseudo-labels from a 300-slide training set — more
capacity spent on a 60-slide validation signal that cannot rank it.

Practical consequence for the writeup: with `instance_loss_weight = 0.30` (fixed), the arm is
not just worse on average, it is **less stable**. A follow-up tuning sweep over
`instance_loss_weight ∈ {0.1, 0.3, 0.5}` with validation-based selection is the fair way to
give CLAM its best shot before concluding the loss is unhelpful; that sweep is listed as
remaining work, and until it is run the conclusion is "no benefit at the default weight".

## 6. Statistical baselines on the same attention statistics

| Model | QWK | Accuracy | Macro F1 |
| --- | --- | --- | --- |
| Ordinal logistic regression on CLAM-SB attention statistics | **0.7682** | 0.4500 | 0.4100 |
| CLAM-SB, no instance loss (end-to-end) | 0.7388 | 0.3944 | 0.3367 |

The classical baseline is the strongest model in the project. That must be stated plainly:
a well-regularised ordinal regression on the aggregated attention features beats both MIL
models on QWK. Caveat to state alongside it — the ordinal model *reuses* CLAM-SB's learned
attention statistics, so it is not an independent competitor; it measures how much of the
signal is already captured by the aggregated feature block. It does mean the deep model's
advantage is in localisation and end-to-end simplicity, not in raw grading accuracy.

## 7. Heatmaps and localisation coverage

| Export | Split | Slides with heatmap | Skipped (no coordinates) |
| --- | --- | --- | --- |
| `results\heatmaps\clam_sb_seed0` | test | 56 | 4 |
| `results\heatmaps\clam_sb_seed0_noinst` | test | 56 | 4 |

**Coverage is 56/60 test slides.** The pre-extracted PANDA feature release ships patch
coordinates for 399/420 slides; four test slides have no coordinates and cannot be placed on
a slide canvas. The four missing test slides are
`b55cae845662dfb081970488f46909fb`, `d8aef66ed061a62078ef30ab96bebe01`,
`f6e7b15af318c5fdc1990ed6a8b117bb`, `0b4b960490879b7588a1d135fcf10fe7`.

Two consequences that must be in the limitations:

1. **No thumbnail overlays and no mask-based overlap score.** The feature-only pipeline has
   no slide thumbnails and no tissue masks, so `attention_mask_overlap` cannot be computed.
   Heatmaps are coordinate-space attention scatter, evaluated visually and by entropy, not
   by IoU against annotated tumour regions.
2. **The ablation's localisation question is answered by entropy, not by overlap.** Attention
   sharpness (mean entropy, normalised entropy, top-10 attention share) is the quantitative
   proxy used here; it answers "does the instance loss concentrate attention?" not "does it
   land on tumour?".

## 8. Limitations

* Research prototype; **not a clinical tool**. No external validation.
* Trained and tested on a PANDA subset only; performance will not transfer unchanged to other
  scanners, labs, or staining protocols without external hospital validation.
* **ISUP class imbalance** makes accuracy misleading. Accuracy ~0.39 with QWK ~0.70–0.74
  means the model gets the ordinal direction right far more often than it gets the exact
  class right; it collapses predictions towards the frequent low grades.
* Test set is 60 slides. Every confidence interval in this report is wide, and no claim of
  significance is made where an interval spans zero.
* One feature extractor, one split, three seeds. Person 1's ABMIL baseline and this CLAM-SB
  share features and split by design, so their comparison is paired — but it is still one
  feature representation.
* Attention heatmaps are **interpretability cues, not guaranteed segmentations**.
* The `instance_loss_weight` sweep and the ABMIL seeds 1–2 paired comparison are still open
  (see §9); the conclusions above hold at the single default weight and seed 0 respectively.

## 9. Open items before submission

1. `evaluate_clam_sb --split all` for seeds 1–2, both arms → fills the attention-sharpness
   table (entropy / top-10 share per arm per seed).
2. `compare_clam_abmil` for seeds 1–2 → fills the paired table §4.2 across all seeds.
3. `python scripts\summarize_person2.py` → `results\PERSON2_SUMMARY.md`, then paste §2 and §3
   of that file into §4.1 and §7 here.
4. Optional but recommended: `instance_loss_weight` sweep {0.1, 0.3, 0.5}, to give the
   instance loss its best shot before the negative conclusion is final.
5. Push the updated `src/dataset.py`, `src/export_clam_attention.py`, `requirements.txt`,
   `scripts/attach_patch_coords.py`, `scripts/make_coords_manifest.py`,
   `scripts/summarize_person2.py` and `tests/test_summarize_person2.py` to `main`.

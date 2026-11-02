# From Patches to Prognosis

## Weakly Supervised Multiple Instance Learning for ISUP Prostate Cancer Grading and Attention-Based Localisation

| | |
| --- | --- |
| **Team** | Koushik Sarathy (Person 1 — data contract, ABMIL baseline) · Archit Renjeev (Person 2 — CLAM-SB, instance-loss ablation) · Himesh (Person 3 — TransMIL) · Aswath (Person 4 — metrics, localisation, statistical baselines, dashboard) |
| **Repository** | `github.com/koushikkkkkkkkkk/prostate-mil-project` |
| **Primary metric** | quadratic weighted kappa (QWK) on ISUP grades 0–5, locked 60-slide test split |
| **Companion documents** | `report/person1_section.md` · `report/person2_section.md` · `report/person3_section.md` · `report/person4_results.md` |

---

## Abstract

We ask a deliberately narrow question: **when four weakly supervised MIL architectures are given the same slide-level labels, the same patches and the same split, which design choice actually determines grading accuracy — the aggregator, or the patch encoder?** We train gated-attention ABMIL, CLAM-SB with and without its instance-level clustering loss, and a simplified TransMIL on 420 H&E prostate biopsies from PANDA (70 slides per ISUP grade, Karolinska provider), using frozen patch features from three encoders: ImageNet ResNet50 (1024-d), CHIEF (768-d) and Prov-GigaPath (1536-d). All comparisons are paired on identical slides, with bootstrap confidence intervals on ΔQWK and exact McNemar tests on per-slide correctness.

The result is unambiguous in one direction and null in the other. **Replacing ImageNet features with a pathology foundation encoder raises QWK by +0.17 to +0.21 with the aggregator held fixed (+0.23 when the aggregator changes too); all six paired confidence intervals across three seeds exclude zero, with McNemar p as low as 0.0007.** Swapping the aggregator — ABMIL → CLAM-SB, ± instance clustering loss, or ABMIL → TransMIL — changes QWK by at most ±0.04, no paired McNemar test reaches significance, and the sign of the instance-loss effect flips between seeds. A three-seed ablation of CLAM-SB's instance-level loss gives 0.7035 ± 0.0088 (with) versus 0.7271 ± 0.0223 (without) — a difference we report as *no detectable effect at n = 60*, not as a benefit or a harm. We further show that the instance loss makes attention **more diffuse** rather than sharper (normalised entropy 0.80 vs 0.70), contrary to the mechanism the CLAM paper proposes.

Two secondary findings matter for how such systems should be reported. First, 80–95 % of the foundation-encoder models' errors are exactly one ISUP grade off, so QWK (0.89–0.93) and accuracy (0.53–0.62) disagree sharply on the *same predictions* — both must be reported. Second, on this cohort 4 of 60 test slides carry no patch coordinates in the public feature release, so localisation is demonstrable on 56/60 slides and no quantitative localisation score is possible at all. The system is a research prototype on 4 % of PANDA, trained and evaluated on a single provider, and is not a clinical device.

---

## 1. Introduction

Prostate cancer grading follows the Gleason system, collapsed into five ISUP grade groups. The clinical task is ordinal: grade 0 is benign, grade 5 is the most aggressive pattern, and the difference between a grade-2 and a grade-3 call can change management. Inter-observer agreement on the middle grades is known to be imperfect, which is what makes computational assistance attractive — and what makes an *ordinal* metric, rather than plain accuracy, the right figure of merit.

Applying deep learning to whole-slide images (WSIs) runs into two obstacles that shape every design choice in this project:

1. **Gigapixel images do not fit in a network.** A 20× prostate biopsy can contain tens of thousands of 256×256 patches. The standard remedy is to tile the slide, embed each tile with a frozen encoder, and treat the resulting bag of feature vectors as the input.
2. **Only slide-level labels exist.** Annotating which patches contain tumour requires pixel-level Gleason masks, which the public PANDA release does not provide for our provider. This is the defining constraint of *weakly supervised* MIL: the model predicts for the bag and must learn, without patch labels, which instances matter.

Multiple instance learning addresses (2) by learning a weighted aggregation of instances. The literature offers several aggregation philosophies — independent attention pooling (ABMIL), attention plus instance-level clustering (CLAM), and token-level self-attention over the whole bag (TransMIL) — and each is usually presented against a different baseline, on a different dataset, with a different encoder. Consequently it is hard to say which of these design choices matters.

Our contribution is an unusual one for a course project: a **controlled comparison** in which the aggregator and the encoder are varied one at a time, on identical slides, with paired statistics. Everything is trained from scratch in PyTorch, without copying the official CLAM or TransMIL implementations, and every reported number is generated from committed artifacts rather than typed by hand.

### 1.1 Contributions

* A reproducible weakly supervised pipeline: locked stratified split, frozen feature bags, four MIL architectures, three encoders, three seeds, all artifacts committed.
* A **paired statistical protocol** (bootstrap CI on ΔQWK + exact McNemar on discordant slides) applied to every model pair, plus a common-cohort table restricted to the 50 test slides every run covers.
* The **encoder-versus-aggregator finding**: encoder choice dominates (+0.17 to +0.21 with the aggregator held fixed, every paired CI excluding zero); aggregator choice is undetectable (±0.04, no McNemar comparison significant, on two independent encoders and three aggregators).
* A **clean negative result** for CLAM-SB's instance-level clustering loss, including the observation that the loss diffuses attention rather than sharpening it.
* An **auditable artifact policy**: `results/ALL_RESULTS.md` is generated from the committed predictions, `scripts/paired_results.py` regenerates every paired table, and `scripts/audit_artifacts.py` reports any result a reviewer could not reproduce. Continuous integration fails the build if the committed tables drift from the artifacts.

---

## 2. Related work

**Multiple instance learning.** Dietterich et al. introduced MIL for drug-activity prediction; Ilse et al. (2018) made it practical for pathology with a gated-attention pooling operator that assigns each instance a weight and is end-to-end differentiable. ABMIL is our baseline.

**Instance-level supervision.** Lu et al. (2021) proposed CLAM, which adds a second branch that clusters the top-*k* and bottom-*k* attended patches and trains them with a smooth top-1 SVM loss. The motivation is localisation: with only a slide label, attention can drift onto irrelevant tissue. CLAM-SB is the single-branch variant; we implement it from scratch and, critically, **ablate the clustering loss**, because the paper's evidence for it is primarily localisation quality, not slide-level grading.

**Correlated instances.** Shao et al. (2021) argued that independent attention cannot express relationships between patches and proposed TransMIL, which encodes patches as tokens with PPEG positional encoding and aggregates through a `[CLS]` token. We implement a simplified two-block version.

**Patch encoders.** He et al. (2016) ResNet50 is the standard ImageNet baseline used throughout the MIL literature. Two pathology foundation models have since displaced it: CHIEF (Wang et al., 2024) and Prov-GigaPath (Xu et al., 2024). We use their pre-extracted features published by Tang et al. (2025), together with the ResNet50 features, so that the encoder is a controlled variable with no re-extraction cost.

**Dataset.** PANDA (Bulten et al., 2022) is a 10,616-slide prostate biopsy grading challenge with two providers. Its public release includes tissue masks that are, for the Karolinska provider, unusable as Gleason labels — a fact we verify in §3.3 and which explains why no patch-level model is trained on real data here.

---

## 3. Dataset

### 3.1 Provenance

| Layer | What it is | Source |
| --- | --- | --- |
| Images | PANDA H&E prostate needle biopsies, ~10,616 WSIs, Karolinska + Radboud | Kaggle PANDA challenge (Bulten et al., 2022) |
| Patch features | Pre-extracted embeddings, 256×256 tiles at 20×, plus patch `(x, y)` coordinates | `Dearcat/CPathPatchFeature` (Tang et al., 2025) |
| Our subset | 420 slide IDs, one bag per slide | `scripts/download_features_and_coords.py` |

We do not re-extract features from WSIs. Re-tiling 420 gigapixel slides is hours of I/O for a feature vector that a frozen encoder would produce identically; using the published release makes the whole project reproducible from a ~600–800 MB download instead of a 347 GB WSI archive. The ResNet50 bags are `[N, 1024]` float32.

### 3.2 Cohort composition

420 slides, stratified 50/10/10 per grade (split seed 42):

| ISUP | Gleason | Train | Val | Test | Total |
| --- | --- | --- | --- | --- | --- |
| 0 — benign | 0+0 | 50 | 10 | 10 | 70 |
| 1 | 3+3 | 50 | 10 | 10 | 70 |
| 2 | 3+4 | 50 | 10 | 10 | 70 |
| 3 | 4+3 | 50 | 10 | 10 | 70 |
| 4 | 4+4 / 3+5 / 5+3 | 50 | 10 | 10 | 70 |
| 5 | 4+5 / 5+4 / 5+5 | 50 | 10 | 10 | 70 |
| **Total** | | **300** | **60** | **60** | **420** |

* **70 benign slides, 350 cancerous** (5:1). The cohort is class-balanced by construction — which is why *imbalance* is not the explanation for low accuracy in this project (§6.4).
* **All 420 slides are Karolinska.** No Radboud slides are present, so no provider-shift estimate is possible. This is the single biggest external-validity limitation.
* The subset is 4 % of PANDA and does not follow PANDA's real class distribution, in which low grades dominate.

Encoder-specific cohorts are the same split minus slides whose features were unavailable: CHIEF 399 (68 benign / 331 cancer), GigaPath 391 (66 / 325), coordinate-bearing subset 399. Test-set sizes are therefore 60, 57, 53 and 56 — never compared across cohorts without the common-cohort table.

### 3.3 Why there is no patch-level supervision

The public PANDA release ships `train_label_masks` for the Karolinska provider. Auditing them with `scripts/check_mask_encoding.py` across 24 slides shows the masks are **99.88 % background and 0.12 % stroma, with no Gleason 3/4/5 pixels anywhere** — ISUP-4 slides, which by definition contain annotated tumour, have entirely empty masks. Pixel-level labels usable for patch supervision therefore do not exist in our provider, and the only supervised learning available is slide-level. This is a data finding, not an implementation gap, and it is why the project contains no trained patch classifier. (Reproduction command and the accompanying caveat that the audit output is not committed as an artifact are noted in §12.)

---

## 4. Methods

### 4.1 Pipeline

```
WSI ──tiling(Otsu tissue mask, 256×256 @20×)──▶ patches
     ──frozen encoder──▶ bag X ∈ ℝ^{N×D}    (D = 1024 r50, 768 CHIEF, 1536 GigaPath)
     ──MIL aggregator──▶ slide embedding z
     ──linear head──▶ 6-way ISUP logits
     ──softmax attention──▶ per-patch weights (interpretability)
```

Tissue detection, tiling and the frozen-encoder path are implemented in `src/preprocessing.py` and `src/feature_extraction.py`; for the reported runs the bags come from the published release, and `data/common_split.csv` is the contract every component reads (`docs/DATA_CONTRACT.md`).

### 4.2 Models

| Model | Aggregation | Instance-level supervision | Parameters (r50) |
| --- | --- | --- | --- |
| **ABMIL** | gated attention, `tanh ⊙ sigmoid`, masked softmax | none | ~0.8 M |
| **CLAM-SB** | gated attention + top-*k* / bottom-*k* clustering branch | smooth top-1 SVM, weight λ = 0.30, *k* = 8 | 796,947 |
| **CLAM-SB λ=0** | identical architecture, loss weight set to zero | none (ablation arm) | 796,947 |
| **TransMIL** | 2-block transformer, PPEG, `[CLS]` token | none | 3,950,598 |

CLAM-SB's ablation is a genuine single-variable comparison: both arms are built from the same config with only `instance_loss_weight` differing, and the notebook asserts the parameter counts are identical before training.

### 4.3 Training protocol (identical for all runs)

AdamW, lr 2e-4 (MIL heads) / 1e-4 (TransMIL), weight decay 1e-5, class-balanced cross-entropy, gradient clipping, batch 16 (CLAM-SB) or 4 (TransMIL, larger bags), max 50 epochs, early stopping on **validation QWK** with patience 12, training-time patch cap 4096 (CLAM-SB) / 2048 (TransMIL). Three seeds (0, 1, 2) per configuration. Hyperparameters were taken from the source papers, not tuned on test.

### 4.4 Evaluation protocol

* **Primary metric:** QWK over ISUP 0–5, which weights a 0↔5 error 25× a 2↔3 error.
* **Secondary:** accuracy, balanced accuracy, macro F1, per-class confusion, attention entropy.
* **Aggregation:** mean ± sd over 3 seeds. The test split is evaluated once per run; no test-based selection anywhere.
* **Paired tests:** for two models A and B on the same slides, we report ΔQWK with a paired bootstrap CI (2000 resamples) and the exact McNemar test on discordant slides. A CI spanning zero is reported as **inconclusive at this sample size**, never as "equivalent".
* **Common-cohort restriction:** cross-encoder claims use only the 50 test slides that *every* run covers.

---

## 5. Results

### 5.1 Every run on its own test set

| Run | Encoder | n test | QWK (mean ± sd) | Accuracy (mean ± sd) | Bal. acc. (seed 0) | Macro F1 (seed 0) |
| --- | --- | --- | --- | --- | --- | --- |
| CLAM-SB | CHIEF | 57 | **0.9207 ± 0.0063** | 0.6023 ± 0.0101 | 0.6204 | 0.6013 |
| ABMIL | CHIEF | 57 | **0.9035 ± 0.0145** | 0.5614 ± 0.0351 | 0.6037 | 0.5785 |
| ABMIL | GigaPath | 53 | **0.8984 ± 0.0108** | 0.5849 ± 0.0327 | 0.6190 | 0.6036 |
| TransMIL | ResNet50 | 60 | 0.7446 ± 0.0364 | 0.4833 | 0.4833 | 0.4475 |
| ABMIL | ResNet50 | 60 | 0.7281 ± 0.0132 | 0.3889 | 0.3889 | 0.3272 |
| CLAM-SB λ=0 | ResNet50 | 60 | 0.7271 ± 0.0223 | 0.3944 | 0.3944 | 0.3386 |
| CLAM-SB λ=0.3 | ResNet50 | 60 | 0.7035 ± 0.0088 | 0.3833 | 0.3833 | 0.3142 |

Generated from `results/ALL_RESULTS.md`. QWK and accuracy are three-seed means; balanced accuracy and macro-F1 are reported for seed 0 only. All 21 checkpoints are committed (§10). **Rows are not mutually comparable** — three different test sets are mixed in this table; that is what §5.2 is for.

### 5.2 The like-for-like table

Restricted to the 50 test slides covered by all seven runs:

| Model | Encoder | QWK | Accuracy |
| --- | --- | --- | --- |
| CLAM-SB | CHIEF (768-d) | **0.9278 ± 0.0074** | 0.6067 ± 0.0115 |
| ABMIL | CHIEF (768-d) | 0.9115 ± 0.0110 | 0.5667 ± 0.0306 |
| ABMIL | GigaPath (1536-d) | 0.8907 ± 0.0132 | 0.5667 ± 0.0306 |
| TransMIL | ResNet50 (1024-d) | 0.7113 ± 0.0583 | 0.4733 |
| CLAM-SB λ=0 | ResNet50 | 0.7001 ± 0.0313 | 0.3733 |
| ABMIL | ResNet50 | 0.6995 ± 0.0136 | 0.3667 |
| CLAM-SB λ=0.3 | ResNet50 | 0.6680 ± 0.0110 | 0.3600 |

Two clean bands: 0.89–0.93 for the foundation encoders, 0.67–0.71 for ResNet50. Within each band the differences between aggregators are smaller than the seed-to-seed spread of the band.

### 5.3 The central comparison: encoder versus aggregator

Per-seed paired statistics (`results/metrics/comparison/paired_stats.md`):

| Factor varied | Comparison | ΔQWK per seed | 95 % CI | McNemar p | Verdict |
| --- | --- | --- | --- | --- | --- |
| **Encoder** | ABMIL r50 → ABMIL CHIEF | +0.2149 / +0.1681 / +0.1762 | excludes 0 | 0.0013 / 0.0225 / 0.0636 | **real** |
| **Encoder** | ABMIL r50 → ABMIL GigaPath | +0.1872 / +0.1934 / +0.1744 | excludes 0 | 0.0013 / 0.0414 / 0.0963 | **real** |
| Encoder (within foundation) | ABMIL CHIEF → ABMIL GigaPath | −0.0456 / +0.0027 / −0.0193 | spans 0 | 1.0 / 1.0 / 1.0 | indistinguishable |
| **Aggregator** | ABMIL → CLAM-SB (CHIEF) | +0.0107 / +0.0318 / +0.0092 | −0.008 to 0.000 | 1.0 / 0.22 / 0.63 | no detectable effect |
| **Aggregator** | ABMIL → TransMIL (r50) | −0.0106 / +0.0294 / +0.0309 | spans 0 | 0.077 / 0.077 / 1.0 | no detectable effect |
| **Aggregator** | ABMIL → CLAM-SB λ=0.3 (r50) | 0.0000 / −0.0317 / −0.0419 | spans 0 | 0.63 / 0.63 / 1.0 | no detectable effect |
| **Aggregator** | ABMIL → CLAM-SB λ=0 (r50) | −0.0114 / +0.0029 / +0.0057 | spans 0 | 0.63 / 1.0 / 1.0 | no detectable effect |
| **Encoder + aggregator** | ABMIL r50 → CLAM-SB CHIEF | +0.2255 / +0.2000 / +0.1854 | excludes 0 | 0.0007 / 0.0044 / 0.0192 | **real** |

**Encoder effects run +0.17 to +0.23; aggregator effects never exceed 0.04.** All six encoder intervals exclude zero; no aggregator interval excludes zero on more than one of three seeds, and no aggregator McNemar test reaches significance. The same aggregator conclusion holds on two different encoders (CHIEF and ResNet50), which is what distinguishes it from a single lucky pairing.

Two of the six encoder McNemar tests cross 0.05 (p = 0.064 for CHIEF seed 2, 0.096 for GigaPath seed 2) even though both intervals exclude zero. At n ≈ 55 exact McNemar has little power, so we treat the bootstrap interval as the primary test and quote the p-values in full rather than reporting only the smallest.

An important corollary: **CHIEF and GigaPath are indistinguishable from each other** (−0.046 to +0.003 on the shared 50 slides, every CI spanning zero, all three McNemar p = 1.0). The gain comes from *pathology pre-training at all*, not from which foundation model. Retraining corroborates this: between the original and the reproduced runs of these same two configurations the ordering **reverses** (GigaPath ahead by 0.005, then CHIEF ahead by 0.005) while the magnitude stays fixed at 0.005. A ranking that flips on retraining at unchanged effect size is what an undetectable difference looks like.

### 5.4 Why accuracy and QWK disagree so sharply

All five ResNet50/CHIEF/GigaPath runs are evaluated on class-balanced cohorts, so imbalance cannot explain the accuracy–QWK gap. Measuring where the errors land (seed 0, from the committed confusion matrices):

| Model | Accuracy | QWK | Errors | Off by exactly one grade |
| --- | --- | --- | --- | --- |
| CLAM-SB, CHIEF | 0.614 | 0.9274 | 22 / 57 | 21 → **95 %** |
| ABMIL, CHIEF | 0.597 | 0.9168 | 23 / 57 | 21 → **91 %** |
| ABMIL, GigaPath | 0.623 | 0.8871 | 20 / 53 | 16 → **80 %** |
| CLAM-SB λ=0.3, r50 | 0.400 | 0.7137 | 36 / 60 | 23 → 64 % |
| ABMIL, r50 | 0.367 | 0.7137 | 38 / 60 | 20 → 53 % |

The foundation-encoder models make mostly near errors: 80–95 % of their mistakes are one grade group off, against 53–64 % for ResNet50, which QWK credits as near-correct and accuracy counts as a total miss. GigaPath's 80 % is its weakest seed (that seed's QWK is 0.887 against 0.908 and 0.900 for the other two), so it understates the typical case rather than overstating it. Accuracy therefore **understates** ordinal performance on this task, while still being the number a clinician would first ask for. Both belong in any table; neither alone describes the system.

---

## 6. The CLAM-SB instance-loss ablation

The project's mandatory ablation: same features, same split, same seeds, same architecture; only the smooth top-1 SVM clustering loss changes (λ = 0.30 versus 0).

| Setting | QWK (mean) | QWK (sd) | Accuracy | Macro F1 | Seeds |
| --- | --- | --- | --- | --- | --- |
| CLAM-SB + instance loss | 0.7035 | 0.0088 | 0.3833 | 0.3142 | 3 |
| CLAM-SB, no instance loss | 0.7271 | 0.0223 | 0.3944 | 0.3386 | 3 |
| ABMIL reference | 0.7281 | 0.0132 | 0.3889 | 0.3272 | 3 |

ΔQWK (with − without) = **−0.0236**.

### 6.1 The sign is not stable, and nothing is significant

| Seed | QWK with | QWK without | ΔQWK (without − with) | 95 % CI | McNemar p | discordant |
| --- | --- | --- | --- | --- | --- | --- |
| 0 | 0.7137 | 0.7023 | −0.0114 | [−0.109, +0.088] | 1.0000 | 4 / 4 |
| 1 | 0.6991 | 0.7336 | +0.0345 | [−0.033, +0.109] | 1.0000 | 2 / 3 |
| 2 | 0.6978 | 0.7454 | +0.0476 | [−0.015, +0.125] | 1.0000 | 1 / 2 |

Seed 0 favours the loss; seeds 1 and 2 favour removing it; every CI spans zero and every McNemar test is exactly 1.0. The defensible statement is **"no detectable effect of the instance-level loss on slide-level QWK"**, with the explicit qualifier that a 60-slide test set is powered only for differences above ≈0.05.

### 6.2 Validation and test disagree

From the committed training histories:

| Seed | With instance loss | Without instance loss |
| --- | --- | --- |
| 0 | best val QWK 0.5132 @ 35, stopped 47, test 0.7137 | best 0.6068 @ 39, ran 50, test 0.7023 |
| 1 | best 0.4955 @ 44, ran 50, test 0.6991 | best 0.5864 @ 36, stopped 48, test 0.7336 |
| 2 | best 0.4871 @ 23, stopped 35, test 0.6978 | best 0.5495 @ 36, stopped 48, test 0.7454 |

Validation QWK is consistently *worse* with the loss (0.487–0.513 vs 0.550–0.607), yet test QWK is not, and seed 0 reverses the ordering. Model selection on validation would therefore have chosen the no-instance arm every time — for a signal that does not transfer to test. This is a useful caution about 60-slide validation sets, not a property of the loss.

### 6.3 The loss diffuses attention rather than sharpening it

Attention statistics from the committed exports (seed 0, all 420 slides):

| Arm | attention entropy (nats) | normalised entropy | top-10 attention share |
| --- | --- | --- | --- |
| CLAM-SB + instance loss | 4.9212 | 0.7996 | 0.3438 |
| CLAM-SB, no instance loss | 4.3021 | 0.6986 | 0.4368 |

The instance loss produces **higher** entropy and a **lower** top-10 share: attention is spread more widely, not concentrated on the top-*k* patches. This is the opposite direction from the localisation mechanism attributed to CLAM's clustering term, and we report it as a negative result. Two caveats: the with-instance arm under-fits on validation (§6.2), so a model that has not separated the classes has little reason to concentrate attention; and the attention exports come from seed-0 checkpoints only, so no per-seed spread is available.

---

## 7. Attention, heatmaps and localisation coverage

| Export | Split | Slides with heatmap | Skipped (no coordinates) | Mean normalised entropy |
| --- | --- | --- | --- | --- |
| `results/heatmaps/abmil` | test | 56 | 4 | 0.6924 |
| `results/heatmaps/clam_sb` | test | 56 | 4 | 0.7744 |

* **Coverage is 56/60 test slides.** The published feature release contains patch coordinates for 399 of 420 slides; four test slides can be graded but not placed on a canvas.
* **No quantitative localisation score is possible.** `results/heatmaps/*/export_summary.json` records `"localisation": {"available": false}`: the feature-only pipeline has neither slide thumbnails nor tissue masks, so IoU against annotated tumour regions cannot be computed. Attention maps are coordinate-space visualisations evaluated by entropy, and must be described as an **interpretability cue, not a segmentation**.
* Attention is diffuse: normalised entropy 0.69 (ABMIL) and 0.77 (CLAM-SB) with a top-10 share of 0.40 and 0.34 respectively. Any claim that these maps "find the tumour" would require patch-level labels that do not exist for this provider (§3.3).

---

## 8. Statistical baselines on attention statistics

Each MIL model's own attention statistics are summarised per slide (mean/median/std/max attention, top-10 mean, entropy, high-attention ratio, embedding norm moments, patch count) and passed to a proportional-odds ordinal logistic regression:

| Attention source | Model | QWK | Accuracy | Macro F1 |
| --- | --- | --- | --- | --- |
| CLAM-SB | ordinal logistic | 0.7399 | 0.3667 | 0.3255 |
| ABMIL | ordinal logistic | 0.7255 | 0.2833 | 0.2224 |

The CG-XGBoost arm is skipped (`xgboost` not installed in the reporting environment) and is **not** reported as a result.

Interpretation: a well-regularised ordinal head on a model's *own* attention summary is competitive with the end-to-end network (0.7399 vs 0.7271 for CLAM-SB λ=0). Two caveats must travel with that sentence: it consumes features produced by the very model it is compared against, so it is not an independent baseline; and a +0.01 margin is well inside 60-slide noise. What it does show is that most of the slide-level signal is already captured by the aggregated feature block — consistent with the paper's central finding that the encoder, not the aggregator, carries the information.

---

## 9. Interactive dashboard

`app/streamlit_app.py` is a checkpoint-driven Streamlit application (four tabs), tested headlessly for import and render (0 exceptions; all 12 committed checkpoints discovered, 0 unreadable; ABMIL, CLAM-SB and TransMIL each load and forward-pass correctly with attention summing to 1.0).

| Tab | Function | Requires |
| --- | --- | --- |
| Overview | problem, method, dataset card, verified results, team & findings, provenance | nothing |
| Slide analysis | pick a slide from the locked split, load a **real** checkpoint, show predicted grade, probabilities and attention | `data/features/{slide_id}.pt` |
| Upload & tile | tile an arbitrary image, extract features with the same frozen encoder, **measure** out-of-distribution distance before grading and abstain if too far | torchvision weights (first use) |
| Results & reports | render the committed artifacts verbatim (tables, metrics JSON, heatmaps) | nothing |

Design rule: **the app never invents a number.** If a checkpoint or a feature bag is missing, it raises a visible error rather than displaying a plausible grade. The Upload tab will not grade an image whose features fall outside the training distribution; it reports the out-of-distribution percentile and abstains.

---

## 10. Reproducibility

| Tier | Runs | Weights | Status |
| --- | --- | --- | --- |
| A | ABMIL, TransMIL, CLAM-SB λ=0.3, CLAM-SB λ=0 (12 checkpoints) | committed | fully reproducible |
| A | ABMIL CHIEF, CLAM-SB CHIEF, ABMIL GigaPath (9 checkpoints) | committed | fully reproducible |

All 21 checkpoints ship with the repository, so every row of §5.1 can be reloaded and re-evaluated, not merely re-audited. The two feature sets they depend on are public downloads (`scripts/download_alt_encoder_features.py`), so the full pipeline is reproducible end to end on a single free GPU in well under an hour.

A note on provenance, because it affects how §5.1 should be read. An earlier version of the three foundation-encoder runs had committed predictions and metrics but **no weight files**, so those numbers referred to models nobody could reload. They were retrained from the public feature release on the same manifests and seed list, and the checkpoints committed. The reproduction lands within 0.019 QWK of the original (Δ = −0.009 / −0.002 / −0.019 for ABMIL CHIEF, CLAM-SB CHIEF and ABMIL GigaPath respectively), and **this report quotes the retrained numbers throughout** — no figure in it refers to a model that cannot be loaded. Two consequences are worth stating rather than burying: the seed spread of the foundation runs is wider than the original suggested (sd 0.006–0.015 rather than 0.001–0.004), and the CHIEF-versus-GigaPath ordering reverses between the two sets of runs of those same configurations (§5.3).

Enforcement, on every push (`.github/workflows/tests.yml`): 54 unit and end-to-end tests; `pyflakes` over `src scripts app tests`; a job that regenerates the paired tables and fails if the committed copies differ; and a train/test-leakage check across all four manifests. `scripts/audit_artifacts.py` additionally reports dangling checkpoint references, missing weights, and absent ablation artifacts.

---

## 11. Limitations

1. **Single provider, one cohort.** 420 Karolinska slides, 4 % of PANDA, stratified rather than sampled from PANDA's true distribution. No external or cross-provider validation. Every number in this report is an internal-split estimate.
2. **Small test set.** 60 slides (50 on the common cohort). Paired CIs are ±0.10 on ΔQWK, so the study can only resolve effects above ≈0.05. Several comparisons here are genuinely inconclusive rather than negative, and are reported as such.
3. **Three seeds.** Enough to report a mean and sd, not enough to characterise the variance of a 4 M-parameter transformer. TransMIL's sd (0.0364) is the largest in the project.
4. **Frozen, pre-extracted features.** No encoder fine-tuning, and encoder comparisons inherit any preprocessing differences in the public feature release.
5. **The foundation-encoder runs were reproduced, not recovered.** Their earlier numbers referred to models whose weights were never committed; the committed runs are a retrain that lands within 0.019 QWK. Seed spread of the retrained foundation runs (sd 0.006–0.015) is larger than the original three suggested.
6. **No localisation ground truth.** Masks are empty for our provider (§3.3); localisation is qualitative and coverage is 56/60.
7. **One retrain, not a repeated study.** The reproduction confirms the rank order of encoder and aggregator effects but also flips the CHIEF-versus-GigaPath ordering, which is direct evidence for the "indistinguishable" claim rather than against it — but a third training run would be needed to put an sd on that flip.
8. **Single instance-loss weight.** λ = 0.30 was fixed from the CLAM paper, not swept; the ablation therefore speaks to *that* weight, not to the loss family.

---

## 12. Conclusion and future work

On a controlled comparison of MIL aggregators and patch encoders for ISUP grading, the encoder dominates. Moving from ImageNet ResNet50 to a pathology foundation encoder raises QWK by +0.17 to +0.23 with every paired confidence interval excluding zero across three seeds; moving between gated attention pooling, CLAM's instance-clustered pooling and TransMIL's token-level self-attention changes QWK by at most ±0.04, never with a significant McNemar test. CLAM-SB's instance-level clustering loss, the object of our mandatory ablation, produces no detectable grading benefit at λ = 0.30 and makes attention measurably *more* diffuse. These are negative results about architecture and honest positives about representation, and the paired protocol is what allows the two to be distinguished.

Highest-value next steps, in order:

1. **Retrain and commit the three foundation-encoder runs** so the headline numbers become re-evaluable (≈30 min).
2. **Scale to a few thousand slides** and re-run the paired protocol; this is the only way the aggregator question can be settled, since 60 slides cannot resolve a 0.02 effect.
3. **Sweep λ ∈ {0.1, 0.3, 0.5}** with validation-based selection and ≥5 seeds before concluding anything further about instance supervision.
4. **Cross-provider evaluation** using the Radboud slides, which is also the only route to patch-level Gleason supervision and therefore to a quantitative localisation metric.

---

## References

1. Bulten, W., Kartasalo, K., Chen, P.-H. C., et al. *Artificial intelligence for diagnosis and Gleason grading of prostate cancer: the PANDA challenge.* Nature Medicine 28, 154–163 (2022).
2. Dietterich, T. G., Lathrop, R. H., Lozano-Pérez, T. *Solving the multiple instance problem with axis-parallel rectangles.* Artificial Intelligence 89 (1997).
3. Ilse, M., Tomczak, J. M., Welling, M. *Attention-based Deep Multiple Instance Learning.* ICML (2018).
4. Lu, M. Y., Williamson, D. F. K., Chen, T. Y., et al. *Data-efficient and weakly supervised computational pathology on whole-slide images.* Nature Biomedical Engineering 5, 555–570 (2021).
5. Shao, Z., Bian, H., Chen, Y., et al. *TransMIL: Transformer based Correlated Multiple Instance Learning for Whole Slide Image Classification.* NeurIPS (2021).
6. He, K., Zhang, X., Ren, S., Sun, J. *Deep Residual Learning for Image Recognition.* CVPR (2016).
7. Wang, X., Yang, S., Zhang, J., et al. *CHIEF: A pathology foundation model for cancer diagnosis and prognosis prediction.* Nature (2024).
8. Xu, H., Usuyama, N., Bagga, J., et al. *Prov-GigaPath: A whole-slide foundation model for digital pathology from real-world data.* Nature (2024).
9. Tang, W., Fang, H., Wu, G., Li, X., Cheng, M.-M. *Revisiting Data Challenges of Computational Pathology: A Pack-based Multiple Instance Learning Framework.* arXiv:2509.20923 (2025). — source of the pre-extracted PANDA features.
10. Tang, W., Qin, R., Fang, H., Zhou, F., Chen, H., Li, X., Cheng, M.-M. *Revisiting End-to-End Learning with Slide-level Supervision in Computational Pathology.* arXiv:2506.02408 (2025).
11. Cohen, J. *Weighted kappa: nominal scale agreement with provision for scaled disagreement or partial credit.* Psychological Bulletin 70 (1968).
12. McNemar, Q. *Note on the sampling error of the difference between correlated proportions or percentages.* Psychometrika 12 (1947).
13. Efron, B., Tibshirani, R. J. *An Introduction to the Bootstrap.* Chapman & Hall (1993).

---

## Appendix A — Reproducing the results

```bash
# 1. data (~600-800 MB, once; writes data/features/)
python scripts/download_features_and_coords.py --manifest data/common_split.csv --out-dir data/features
python scripts/verify_data.py --require all          # ResNet50 cohort must report OK

# 2. training (3 seeds each, T4)
python -m src.train --model abmil    --seeds 0 1 2   # ABMIL
python -m src.train --model transmil --seeds 0 1 2   # TransMIL
python -m src.train_clam_sb          --seeds 0 1 2   # both ablation arms

# 3. tables (no compute; regenerates every number quoted above)
python scripts/summarize_results.py                  # results/ALL_RESULTS.md
python scripts/paired_results.py                     # paired_*.md / paired_stats.md
python scripts/audit_artifacts.py                    # must report 0 fatal

# 4. verification
python -m unittest discover -s tests                 # 54 tests
python -m pyflakes src scripts app tests             # clean
streamlit run app/streamlit_app.py                   # dashboard
```

## Appendix B — Artifact map

| Path | Contents |
| --- | --- |
| `results/ALL_RESULTS.md` | every trained run, generated from artifacts (single source of truth) |
| `results/metrics/<run>/` | weights, per-seed metrics/predictions/histories, config, confusion matrices |
| `results/metrics/clam_sb/{with,without}_instance/` | the ablation arms |
| `results/metrics/comparison/paired_*.md` | common/native cohort tables and per-seed paired statistics |
| `results/metrics/statistical/` | ordinal-logistic baselines on attention statistics |
| `results/heatmaps/`, `results/metrics/attention/` | attention exports, heatmaps, entropy summaries |
| `docs/REPRODUCIBILITY.md` | what ships with weights, what does not, and how to close the gap |
| `docs/DATA_CONTRACT.md` | the three files every component depends on |
| `docs/MODEL_COMPENDIUM.md` | per-model architecture notes and paper references |

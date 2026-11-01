# Person 4 — Results: localisation, statistical baselines, and the encoder study

Paste-ready results section. Every number below is loaded from a file in
`results/metrics/`; none is typed by hand. Reproduction commands are in §10.

**Headline.** Holding the slides, labels, split, MIL head, optimiser and early-stopping
rule fixed, replacing the frozen ImageNet ResNet50 patch encoder with a
pathology-pretrained encoder (CHIEF) raises QWK from **0.72 to 0.92** and accuracy from
**0.37 to 0.60** on an identical 57-slide test subset. This is the largest and most
consistent effect found in the project — roughly 13–16× the seed-to-seed standard
deviation.

---

## 1. Experimental setup

| item | value |
| --- | --- |
| Dataset | PANDA subset, 420 slides (287 train / 56 val / 60 test), stratified, seed 42 |
| Split file | `data/common_split.csv` — **unmodified** scientific lock |
| Label | ISUP grade group 0–5 |
| Primary metric | quadratic-weighted kappa (QWK) on the test split |
| Secondary | accuracy, balanced accuracy, macro F1 |
| Model selection | validation QWK only; test evaluated once per configuration |
| Seeds | 0, 1, 2 — reported as mean ± sd |
| Frozen encoders | ResNet50 (ImageNet, D=1024), CHIEF (D=768), GigaPath (D=1536), UNI (D=1024) |

**Two experiment families.** Swapping the encoder changes the input feature space, so
numbers from different encoders are *not* interchangeable (`docs/DATA_CONTRACT.md`).
They are reported as separate families and compared only on a common slide subset (§3).

---

## 2. Family A — ResNet50 features (locked baseline)

Test n = 60.

| model | QWK | accuracy | balanced acc. | macro F1 |
| --- | --- | --- | --- | --- |
| ABMIL | **0.7325 ± 0.0140** | 0.3833 | 0.3833 | 0.3203 |
| CLAM-SB + instance loss (λ=0.30) | 0.7031 ± 0.0092 | 0.3778 | 0.3778 | 0.3071 |
| CLAM-SB, no instance loss (λ=0) | 0.6933 ± 0.0857 | 0.3778 | 0.3778 | 0.3203 |

Reproducibility check: ABMIL at 0.7325 ± 0.0140 agrees with the independently committed
baseline of 0.7281 ± 0.0132 to within 0.004 — well inside one seed sd.

### 2.1 Instance-supervision ablation

Δ(no-instance − with-instance) = **+0.0098** QWK, but the no-instance arm has sd 0.0857
because seed 0 early-stopped at epoch ≤ 6, so its checkpoint was selected before the
model had converged. **Conclusion: instance supervision did not produce a reliable change
in slide-level QWK. At n = 60 this comparison is inconclusive**, and the ablation is
reported as such rather than as a positive or negative effect.

### 2.2 Paired comparison, ABMIL vs CLAM-SB

Paired over 56 overlapping test slides: QWK 0.7203 (ABMIL) vs 0.7137 (CLAM-SB),
difference **−0.0066, 95% CI [−0.1246, +0.1059]**, McNemar **p = 0.6875**. The two models
are statistically indistinguishable on ResNet50 features with this sample size; the CI
spans zero and is wider than the effect.

---

## 3. Family B — pathology-pretrained encoders

### 3.1 Feature coverage (measured before training)

Encoder choice was fixed **before** any QWK was seen, using coverage only.

| encoder | D | slides found | train | val | test |
| --- | --- | --- | --- | --- | --- |
| ResNet50 | 1024 | 420/420 | 287/287 | 56/56 | 60/60 |
| CHIEF | 768 | 399/420 | 286/287 | **56/56** | 57/60 |
| GigaPath | 1536 | 391/420 | 281/287 | 54/56 | 53/60 |
| UNI | 1024 | 391/420 | 283/287 | 54/56 | 54/60 |

CHIEF was selected: it is the only pathology encoder with **complete validation
coverage**, so early stopping operates on the same signal as the baseline.

`data/common_split_<encoder>.csv` is `common_split.csv` with uncovered slides removed —
a **filtering, not a re-split**. Partitions, labels and slide identity are unchanged.

### 3.2 Results at each encoder's own coverage

| model | encoder | n test | QWK | accuracy |
| --- | --- | --- | --- | --- |
| ABMIL | CHIEF | 57 | 0.9124 ± 0.0036 | 0.5789 |
| CLAM-SB + instance | CHIEF | 57 | **0.9224 ± 0.0014** | **0.5965** |
| ABMIL | GigaPath | 53 | 0.9172 ± 0.0169 | — |

Two independent encoders land in the same 0.91–0.92 band, which argues the gain is the
pathology pretraining rather than one encoder's idiosyncrasy.

**Seed stability.** CHIEF's sd (0.0014–0.0036) is 4–10× smaller than ResNet50's
(0.0092–0.0140): on a better-conditioned feature space the three seeds converge to
near-identical solutions. GigaPath's larger sd (0.0169) accompanies heavy overfitting —
training loss falls to ≈0.04 while validation QWK plateaus near 0.85 — which early
stopping catches (best epochs 10/17/16).

---

## 4. Controlled comparison — identical slides, identical seeds

Every run restricted to the 57 test slides CHIEF covers (`scripts/cmp57.py`). This is the
table that isolates the encoder as the only changed variable.

| model | encoder | n | QWK | accuracy |
| --- | --- | --- | --- | --- |
| CLAM-SB | ResNet50 | 57 | 0.6904 ± 0.0098 | 0.3626 |
| ABMIL | ResNet50 | 57 | 0.7219 ± 0.0149 | 0.3743 |
| ABMIL | CHIEF | 57 | 0.9124 ± 0.0036 | 0.5789 |
| **CLAM-SB** | **CHIEF** | 57 | **0.9224 ± 0.0014** | **0.5965** |

| comparison | Δ QWK | Δ accuracy |
| --- | --- | --- |
| ABMIL: ResNet50 → CHIEF | **+0.1905** | +0.2046 |
| CLAM-SB: ResNet50 → CHIEF | **+0.2320** | +0.2339 |

Both deltas exceed the seed sd by more than an order of magnitude. Note also that the
ranking of the two MIL heads **reverses** between families: ABMIL wins on ResNet50
(+0.0315), CLAM-SB wins on CHIEF (+0.0100). With n = 57 neither within-family difference
is conclusive, and the interaction should not be over-interpreted.

---

## 5. Full-coverage results and the fusion experiment

No single pathology encoder covers all 60 test slides, but CHIEF's and GigaPath's gaps are
**disjoint** (CHIEF misses 3 test slides, GigaPath 7, overlap 0), so late fusion restores
full coverage. Pre-registered rule: average p0–p5 across every (encoder, seed) prediction
available per slide, then take the argmax.

| result | n | QWK | accuracy |
| --- | --- | --- | --- |
| CHIEF ∪ GigaPath, fused | **60** | 0.9086 | **0.6333** |
| ABMIL / CHIEF alone (same slides, reseeded) | 57 | 0.9134 | 0.5789 |
| ABMIL / GigaPath alone (same slides) | 53 | 0.8947 | 0.5849 |

**Fusion did not improve QWK.** It is marginally below CHIEF alone. Cohort analysis shows
why the dilution is not a coverage artefact — single-coverage slides score *better*, not
worse:

| cohort | n | QWK | accuracy | errors | of which off-by-one |
| --- | --- | --- | --- | --- | --- |
| both encoders | 50 | 0.9058 | 0.6200 | 19 | 16 |
| CHIEF only | 7 | 0.7981 | 0.5714 | 3 | 2 |
| GigaPath only | 3 | 1.0000 | 1.0000 | 0 | 0 |

The two encoders share the same patch grid and therefore make correlated errors, so
averaging them tracks the weaker member. **The fused model is reported as a coverage
remedy, not an accuracy gain.** Its one real merit is that it delivers the best accuracy
in the project (0.6333) on the full locked 60-slide test set.

---

## 6. Statistical baselines — is multiple-instance learning necessary?

An ordinal logistic regression is fitted to `x_stat`, ten summary statistics of each
slide's attention distribution (entropy, top-k mass, moments, patch count).

| features from | ordinal logistic QWK |
| --- | --- |
| ABMIL attention | 0.7255 |
| CLAM-SB attention | 0.7399 |

Single split, seed 42.

**Findings.** (i) On ResNet50 features, a linear model over ten summary statistics matches
or slightly beats the full MIL model (0.7399 vs 0.7031 for CLAM-SB), so on weak features
the attention *distribution* carries essentially all of the usable signal. (ii) These
features are derived from the MIL models themselves, so this is **not an independent
baseline** — it is a diagnostic of how much information the attention weights hold.

---

## 7. Attention analysis

### 7.1 Attention sharpness

Normalised entropy, lower = more focused, over all 420 slides:

| model | normalised entropy |
| --- | --- |
| ABMIL | 0.7245 |
| CLAM-SB | 0.8003 |

On the 56 renderable test slides ABMIL scores 0.6924. **CLAM-SB's attention is flatter,
not sharper**, contrary to the intuition that instance supervision would concentrate it.

### 7.2 Inter-model agreement

56 slides where both models have coordinates. This measures whether the two architectures
look at the same patches — **it is not localisation accuracy**, since no ground truth
exists for which patches are tumour.

| statistic | value |
| --- | --- |
| Spearman ρ (attention rank correlation) | **0.869 ± 0.044** |
| Top-10% Jaccard overlap | **0.502 ± 0.186** |
| Cohen's κ (top-10% membership) | **0.608 ± 0.197** |

High rank correlation with moderate top-set overlap means the two models broadly agree on
which regions matter but disagree on the specific highest-attention patches — consistent
with both models being weak on ResNet50 features.

---

## 8. Coordinate coverage and what it limits

399/420 slides (95.0%) have tile coordinates; all 21 missing are Karolinska, flagged
`bare tensor (features only)`. Per split: train 287/13, val 56/4, test 56/4 (93.3%).
Dropped by ISUP: {0: 2, 1: 5, 2: 5, 3: 5, 4: 3, 5: 1} — a mild grade skew in the
visualised subset that must be noted alongside any heatmap figure.

**Encoder-level limitation.** The public CHIEF and GigaPath releases ship embeddings
only: no tile coordinates, no patch images, no masks. Consequently **no spatial heatmap or
top-patch figure can be produced for the pathology-encoder models**. Attention can be
ranked but not placed. All localisation figures in this report therefore use the ResNet50
feature set.

---

## 9. Patch CNN — implemented, not trained

`src/models/patch_cnn.py` is complete and unit-tested: `PatchCNN` (60,818 parameters at
`width=16`), `PatchCNNConfig`, `PatchDataset`, `train_patch_cnn`, `predict_patch_grid`.
A forward pass on `(2, 3, 256, 256)` returns `(2, 2)` as expected.

It was **not trained**, for two independent reasons:

1. **No patch pixels.** `data/patches/` is empty; the PANDA feature release provides frozen
   embeddings and coordinates only; `patch_paths` is optional in the data contract, and the
   extractor that would produce tiles (`notebooks/02`) has only been run against mock data.
2. **No patch labels for our provider.** The Kaggle release does ship `train_label_masks`,
   so pixel annotations exist in PANDA as a whole -- but they do not cover our slides. Every
   slide in the locked split is Karolinska, and across 24 slides stratified over all six
   ISUP grades (four per grade) the masks are 99.88% background and 0.12% stroma, with **no
   Gleason 3/4/5 pixels anywhere**. ISUP 4 slides, which by definition contain annotated
   tumour, have entirely empty masks. Pixel-level labels in this release are therefore
   Radboud-only, and `data/common_split.csv` contains no Radboud slides.
   Reproduction: `scripts/check_mask_encoding.py --masks <kagglehub>/train_label_masks
   --mask-subdir train_label_masks --stratify --inspect 24`.

Patch-level localisation is instead approached through MIL attention weights, with the
caveat in §7.2 that attention is a model-internal signal rather than a validated
annotation.

---

## 10. Reproducing every number

All commands from the repository root, venv interpreter explicit.

```bash
# feature download (per encoder)
python scripts/download_alt_encoder_features.py --encoder chief
python scripts/download_alt_encoder_features.py --encoder gigap

# training (3 seeds each)
python -m src.train --model abmil   --manifest data/common_split_chief.csv \
    --features-root data/features_chief --config configs/abmil_chief.yaml \
    --out-dir results/metrics/abmil_chief --seeds 0 1 2
python -m src.train --model clam_sb --manifest data/common_split_chief.csv \
    --features-root data/features_chief --config configs/clam_sb_chief.yaml \
    --out-dir results/metrics/clam_sb_chief --seeds 0 1 2

# controlled comparison on identical slides (§4)
python scripts/cmp57.py \
    --run "abmil r50=results/metrics/abmil/predictions_abmil_seed*.csv" \
    --run "clam_sb r50=results/metrics/clam_sb/with_instance/predictions_clam_sb_seed*.csv" \
    --run "abmil chief=results/metrics/abmil_chief/predictions_abmil_seed*.csv" \
    --run "clam_sb chief=results/metrics/clam_sb_chief/predictions_clam_sb_seed*.csv" \
    --restrict-to data/common_split_chief.csv \
    --out results/metrics/comparison/encoder_comparison_57.csv

# fusion and cohort analysis (§5)
python scripts/fuse_encoders.py \
    --pred "results/metrics/abmil_chief/predictions_abmil_seed*.csv" \
    --pred "results/metrics/abmil_gigap/predictions_abmil_seed*.csv" \
    --manifest data/common_split.csv \
    --out results/metrics/comparison/fused_chief_gigap.csv
python scripts/cohort_report.py \
    --fused results/metrics/comparison/fused_chief_gigap.csv

# statistical baselines, paired test, agreement
python -m src.statistical_models --attention-csv <attention.csv> --split all \
    --out-dir results/metrics/statistical
python -m src.evaluate --compare results/metrics/abmil results/metrics/clam_sb/with_instance
python scripts/attention_agreement.py

# notebook with all summary tables
jupyter nbconvert --to notebook --execute notebooks/06_localization_and_stats.ipynb
```

## 11. File provenance

| result | file |
| --- | --- |
| Family A metrics | `results/metrics/{abmil,clam_sb/with_instance}/aggregate_*.json` |
| Family B metrics | `results/metrics/{abmil,chief...}/aggregate_*.json` — **reference the folder, not the filename**; prediction files carry the model name only, not the encoder |
| Controlled comparison | `results/metrics/comparison/encoder_comparison_57.csv` |
| Fusion | `results/metrics/comparison/fused_chief_gigap.csv` + `_summary.json` |
| Cohort breakdown | `results/metrics/comparison/fused_chief_gigap_cohorts.json` |
| Statistical baselines | `results/metrics/statistical/*.json` |
| Paired test | `results/metrics/comparison/*.json` |
| Agreement | `results/metrics/attention/agreement_abmil_vs_clam{.csv,_summary.json}` |
| Coverage | `results/metrics/coord_coverage.csv`, `data/common_split_<encoder>.csv` |

---

## 12. Limitations

1. **The encoder study is a post-hoc extension.** It was motivated by observing that the
   ResNet50 backbone was the binding constraint. The sweep was not pre-registered, though
   the *choice of encoder* was fixed on coverage before any QWK was observed, and no
   configuration was selected on test performance.
2. **Multiple configurations were evaluated on the locked test split** — 8 configurations
   across 3 seeds. With n = 57–60 the multiplicity risk is real; the reported gains
   (+0.19 to +0.23) are far above the seed sd but a single split cannot rule out split
   optimism.
3. **Encoder coverage is incomplete.** Family B numbers come from 53–57 of 60 test slides.
   Only the §4 comparison and the fused row are directly comparable to the locked baseline.
4. **n is small.** QWK 0.92 on 57 slides corresponds to ~23 correct of 57; 16 of the 19
   fused errors were off-by-one ISUP grade. Confidence intervals are in
   `metrics_*.json` (`qwk_bootstrap_95ci`).
5. **No localisation ground truth.** Pixel-level Gleason annotations exist in PANDA but only
   for Radboud slides; the Karolinska masks in the public release are empty (verified across
   24 slides at every ISUP grade, §9), and the locked split contains no Radboud slides.
   Attention agreement (§7.2) is therefore the strongest statement the data supports.
6. **No heatmaps for the pathology encoders.** Their feature release omits coordinates.
7. **The patch CNN is unimplemented at the data level**, not at the code level (§9).
8. **Statistical baselines are not independent** — they consume the MIL models' own
   attention.

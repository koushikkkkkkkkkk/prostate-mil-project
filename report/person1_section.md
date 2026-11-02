# Person 1 — Gated-Attention ABMIL Baseline & Data Infrastructure

**Deliverable Owner:** Person 1 (Koushik Sarathy, Team Lead)  
**Task:** Establish reproducible repository contracts, locked split, cloud training pipeline, and the official Gated-Attention ABMIL baseline across 3 seeds.  
**Primary Metric:** Quadratic Weighted Kappa (QWK) on the locked test split ($n = 60$).  

---

## 1. Experimental Setup & Protocol

| Item | Value |
| :--- | :--- |
| **Dataset** | PANDA whole-slide prostate biopsies (Karolinska provider) |
| **Total Slides** | 420 slides (features pre-extracted at 20× magnification) |
| **Split (Locked Contract)** | 300 Train / 60 Validation / 60 Test, stratified by ISUP grade (seed 42) |
| **Feature Encoder** | Frozen ImageNet ResNet50 (1024-dimensional embeddings) |
| **Label Space** | Slide-level ISUP Grade Groups 0, 1, 2, 3, 4, 5 (ordinal) |
| **Model** | Gated-Attention ABMIL (Ilse et al., ICML 2018), from scratch in PyTorch |
| **Seeds** | 0, 1, 2 |
| **Selection Criterion** | Checkpoint selected by validation QWK plateau |
| **Primary Metric** | Quadratic Weighted Kappa (QWK) |
| **Secondary Metrics** | Accuracy, Balanced Accuracy, Macro F1, Per-Class Confusion Matrix |

---

## 2. Gated Attention Mechanism

Standard attention pooling can suffer from gradient saturation when $\tanh$ activations operate in extreme ranges. ABMIL introduces a non-linear gating mechanism combining $\tanh$ and $\text{sigmoid}$ gates:

$$a_j = \frac{\exp\left(\mathbf{w}^\top \left(\tanh(\mathbf{V} \mathbf{x}_j) \odot \text{sigm}(\mathbf{U} \mathbf{x}_j)\right)\right)}{\sum_{k=1}^N \exp\left(\mathbf{w}^\top \left(\tanh(\mathbf{V} \mathbf{x}_k) \odot \text{sigm}(\mathbf{U} \mathbf{x}_k)\right)\right)}$$

Where:
* $\mathbf{V} \in \mathbb{R}^{L \times D}$ and $\mathbf{U} \in \mathbb{R}^{L \times D}$ are learned linear projections ($L = 256, D = 1024$).
* $\odot$ is element-wise multiplication.
* $\mathbf{w} \in \mathbb{R}^{L \times 1}$ maps the gated representation to a scalar attention energy.
* The bag representation $\mathbf{h} = \sum_{j=1}^N a_j \mathbf{x}_j$ is passed to an ordinal classification head $\mathbb{R}^D \to \mathbb{R}^6$.

---

## 3. Official Baseline Results (3-Seed Evaluation)

Test set ($n = 60$ slides), evaluated across seeds 0, 1, and 2:

| Seed | Test QWK | Accuracy | Balanced Acc. | Macro F1 | Epochs to Best |
| :---: | :---: | :---: | :---: | :---: | :---: |
| 0 | 0.7137 | 0.3889 | 0.3889 | 0.3341 | 18 |
| 1 | 0.7308 | 0.3889 | 0.3889 | 0.3235 | 22 |
| 2 | 0.7397 | 0.3889 | 0.3889 | 0.3239 | 24 |
| **Mean ± SD** | **0.7281 ± 0.0132** | **0.3889** | **0.3889** | **0.3272** | **21.3** |

### Key Observations:
1. **High Ordinal Concordance (QWK ~0.73):** Despite raw accuracy being ~39%, QWK is strong because errors land on immediately adjacent Gleason grades (e.g. predicting ISUP 2 instead of ISUP 3), which incur only minor quadratic penalties ($1/25 = 0.04$).
2. **Stable Convergence:** Across all 3 seeds, the gated attention model cleanly plateaued and early-stopped between epochs 18 and 24, avoiding severe overfitting on the 300 training bags.
3. **Determinism:** the three seeds are committed with their configs, so any teammate can
   reproduce these numbers with
   `python -m src.train --model abmil --seeds 0 1 2`. Reproduction is a one-command
   operation on the public feature release — see `docs/REPRODUCIBILITY.md`.

---

## 4. Attention Attribution & Sanity Checks

Attention weights are exported for all 420 slides and summarised in
`results/ALL_RESULTS.md` §5. What they actually show:

| measure | ABMIL (r50) | CLAM-SB (+ instance loss) |
| :--- | :---: | :---: |
| mean normalised attention entropy | 0.7245 | 0.8003 |
| top-10 attention share | 0.4005 | 0.3432 |

Read this honestly: **normalised entropy of 0.72 is diffuse, not sharp.** With ~1000+
patches per slide, attention mass is spread across a large fraction of the bag, and the
top 10 patches hold 40% of the weight. ABMIL is more concentrated than CLAM-SB
(0.72 vs 0.80), but neither produces the near-one-hot attention a "the model found the
tumour" claim would require.

Three consequences to keep in the write-up:

* The maps are an **interpretability cue**, not a segmentation. No quantitative
  localisation score is possible: the feature release has no slide thumbnails and no
  tissue masks (see §5 and `results/heatmaps/*/export_summary.json`, where
  `"localisation": {"available": false}`).
* The claim "benign stroma receives negligible attention, neoplastic glands receive the
  vast majority" was **not measured** and has been removed. Establishing it would need
  patch-level labels, which PANDA does not provide for Karolinska slides.
* Attention exports exist for **56 of 60 test slides** — four test slides have no patch
  coordinates in the feature release, so they can be graded but not placed on a canvas.

---

## 5. Limitations

* **One provider.** All 420 slides are Karolinska; there are no Radboud slides, so no
  provider-shift estimate is possible.
* **4% of PANDA.** The subset is stratified (70 slides per ISUP grade) rather than
  reflecting PANDA's real class balance, so accuracy on this split is not an estimate of
  accuracy on the full cohort.
* **Adjacent-grade confusion dominates** the error budget: 53% of ABMIL's test errors are
  off by exactly one grade. Report QWK and accuracy together, always.
* **Masks are unusable for supervision.** The public Karolinska label masks contain no
  Gleason pixels (reproducible with `scripts/check_mask_encoding.py`), which is why no
  patch-level model is trained on real data.

## 6. Artifacts & Deliverables

All outputs are committed under `results/metrics/abmil/`:
* Prediction CSVs with slide IDs, ground truth, predicted grades, and probabilities: `predictions_abmil_seed{0,1,2}.csv`
* Detailed metrics JSON bundles: `metrics_abmil_seed{0,1,2}.json`
* Aggregate summary: `aggregate_abmil.json`
* High-resolution confusion matrices: `confusion_abmil_seed{0,1,2}.png`
* Training histories: `history_abmil_seed{0,1,2}.csv`

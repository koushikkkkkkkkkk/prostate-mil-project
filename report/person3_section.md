# Person 3 — TransMIL (Transformer-based Correlated MIL)

**Deliverable owner:** Person 3 (Himesh)
**Model:** simplified TransMIL with PPEG positional encoding, written from scratch in
PyTorch (`src/models/transmil.py`)
**Question it answers:** does letting patches *communicate with each other* beat pooling
them independently, as ABMIL and CLAM-SB do?
**Primary metric:** Quadratic Weighted Kappa (QWK) on the locked 60-slide test split.

---

## 1. Why this model is in the project

ABMIL and CLAM-SB both treat a slide as a **bag of independent patches**: each patch gets
an attention weight, the weights are normalised, and the slide prediction is a weighted
average. Nothing in that formulation lets patch *j* see patch *k*.

TransMIL removes that restriction. Every patch becomes a token, tokens are processed by
self-attention, and a learnable `[CLS]` token accumulates the slide representation. It is
therefore the only model in this project that can express **correlated** morphology —
e.g. "cribriform architecture next to high-grade nuclei" — rather than scoring patches in
isolation. That is the specific hypothesis it tests.

It is also the model with the best ResNet50 result in the repository
(**0.7446 ± 0.0364** vs ABMIL's 0.7281 ± 0.0132), and — importantly — that difference is
**not statistically significant** (see §5). The honest headline is that context modelling
does not rescue ImageNet features.

---

## 2. Architecture as implemented

```
patches  x  [N, 1024]
   └─ Linear(1024 → 384) + dropout            token sequence   [N, 384]
      └─ PPEG: depthwise 3×3 conv over a ceil(√N) × ceil(√N) pseudo-image
         └─ prepend learnable [CLS] token      [1 + N, 384]
            └─ 2 × pre-norm transformer block (6 heads, FFN ×4, GELU, dropout 0.1)
               └─ LayerNorm
                  └─ Linear(384 → 6) on the [CLS] token → ISUP logits
```

| Component | Implementation detail | Source |
| --- | --- | --- |
| Token projection | `Linear(in_dim → d_model)` + dropout | `src/models/transmil.py` |
| PPEG | depthwise `Conv2d(384, 384, k=3, padding=1, groups=384)` over the reshaped grid, added residually | `PPEG` class |
| Self-attention | standard multi-head attention, pre-norm residual blocks | `SelfAttentionBlock` |
| Masking | padded grid cells and invalid patches are excluded via a key-padding mask, so a short bag is not polluted by zeros | `TransMIL.forward` |
| Attention export | last block's **CLS → patch** attention row, averaged over heads | `forward(..., return_patch_embeddings=True)` |
| Head | `Linear(d_model, 6)` | `TransMIL.classifier` |

Deliberate simplifications, to be stated in the report rather than hidden:

* The paper's **Nystromformer** approximation is replaced by full self-attention over a
  capped number of patches (`max_patches: 2048` during training). At 2048 tokens this is
  affordable; the paper's approximation exists for bags an order of magnitude larger.
* **No pyramid patch embedding**, no 4-stage hierarchy. Two blocks, one PPEG, one CLS.
* Consequence: it is a *simplified* TransMIL, which is why the module docstring and this
  section both say so. It is not a claim of reproducing the paper's reported numbers.

Parameter count: **3,950,598** (vs ABMIL's ~0.8 M and CLAM-SB's 796,947) — roughly 5× the
parameters of the pooling models, for a QWK difference inside the noise band.

---

## 3. Training protocol

Identical to every other run in the repository, so the comparison is clean:

| Item | Value |
| --- | --- |
| Manifest | `data/common_split.csv` (300 train / 60 val / 60 test, stratified, seed 42) |
| Features | frozen ImageNet ResNet50, 1024-d, one bag per slide |
| Loss | cross-entropy, class weights balanced |
| Optimiser | AdamW, lr 1e-4, weight decay 1e-5, grad clip 1.0 |
| Batch | 4 slides (bags are padded, so batches must stay small) |
| Schedule | max 50 epochs, early stopping on validation QWK, patience 12 |
| Patch cap | 2048 random patches per slide during training |
| Seeds | 0, 1, 2 |
| Hardware / time | one T4, **203–316 s per seed** |
| Config of record | `results/metrics/transmil/run_config.json` |

---

## 4. Results

Locked test split, **n = 60 slides**, 3 seeds:

| Seed | QWK | Accuracy | Balanced acc. | Macro F1 | Best epoch | Best val QWK | Train time |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 0 | 0.7030 | 0.5000 | 0.5000 | 0.4653 | 42 | 0.8208 | 316 s |
| 1 | 0.7602 | 0.5167 | 0.5167 | 0.4288 | 36 | 0.7557 | 279 s |
| 2 | 0.7706 | 0.4333 | 0.4333 | 0.4485 | 23 | 0.8250 | 203 s |
| **Mean ± SD** | **0.7446 ± 0.0364** | **0.4833 ± 0.0441** | 0.4833 ± 0.0441 | 0.4475 ± 0.0371 | — | — | — |

Source: `results/metrics/transmil/metrics_transmil_seed{0,1,2}.json`;
weights committed (`results/metrics/transmil/checkpoints/transmil_seed{0,1,2}_best.pt`,
15.1 MB each).

### 4.1 On the common 50-slide cohort (the only like-for-like table)

Every run in the project restricted to the 50 test slides *all* runs cover:

| model | encoder | QWK | accuracy |
| --- | --- | --- | --- |
| CLAM-SB | CHIEF | **0.9298 ± 0.0016** | 0.6000 |
| ABMIL | CHIEF | 0.9182 ± 0.0043 | 0.5800 |
| ABMIL | GigaPath | 0.9099 ± 0.0183 | 0.6133 |
| **TransMIL** | **ResNet50** | **0.7113 ± 0.0583** | 0.4733 |
| ABMIL | ResNet50 | 0.6995 ± 0.0136 | 0.3667 |

TransMIL is the best **ResNet50** model on this cohort, and still ~0.20 QWK below every
foundation-encoder run.

---

## 5. Is TransMIL actually better than ABMIL? No — not detectably

Paired per-seed statistics on the same 60 slides (paired bootstrap CI, 2000 resamples;
exact McNemar on per-slide correctness). Source: `results/metrics/comparison/paired_stats.md`.

| Seed | ABMIL QWK | TransMIL QWK | ΔQWK | 95 % CI | McNemar p |
| --- | --- | --- | --- | --- | --- |
| 0 | 0.7137 | 0.7030 | −0.0106 | [−0.1488, +0.1048] | 0.0768 |
| 1 | 0.7308 | 0.7602 | +0.0294 | [−0.0632, +0.1218] | 0.0768 |
| 2 | 0.7397 | 0.7706 | +0.0309 | [−0.1076, +0.1752] | 1.0000 |

* The mean is **+0.0166**, but the sign flips at seed 0 and **every CI spans zero**.
* TransMIL is also **less stable across seeds** (sd 0.0364 vs ABMIL's 0.0132), which is
  expected for a 4 M-parameter transformer on 300 training bags.
* Correct wording: *"TransMIL is numerically the strongest ResNet50 model, but at
  n = 60 the difference from ABMIL is not detectable; the study is powered for
  differences above ≈0.05, and TransMIL is the higher-variance of the two."*

A tempting quote to avoid: TransMIL's mean is higher than CLAM-SB-with-instance-loss on
the common cohort (0.7113 vs 0.6680). That is **one cell of a grid whose arms all have
CIs spanning zero**, not a ranking — see the multiple-comparison note in
`report/person2_section.md` §5.4.

---

## 6. What TransMIL contributes to the paper's conclusion

It is the third independent test of the same question, and it agrees with the other two:

| Aggregator swap (encoder held fixed) | ΔQWK | Verdict |
| --- | --- | --- |
| ABMIL → CLAM-SB (CHIEF) | +0.007 … +0.0117 | no detectable effect |
| ABMIL → CLAM-SB ± instance loss (ResNet50) | 0.0000 / −0.0114 | no detectable effect |
| **ABMIL → TransMIL (ResNet50)** | **−0.0106 … +0.0309** | **no detectable effect** |
| *Encoder swap (aggregator held fixed)* | **+0.18 … +0.24** | **real, p ≤ 0.02** |

Three aggregators — pooling with gated attention, pooling with instance clustering, and
full token-level self-attention with positional encoding — move the number by at most
±0.03 on 60 slides. Replacing ImageNet features with a pathology foundation encoder moves
it by +0.19 with p = 0.004. That contrast is the paper's central claim, and TransMIL is
what makes it a claim about *MIL architectures* rather than about one architecture.

---

## 7. Limitations specific to this model

* **Simplified, not the paper's TransMIL.** No Nystromformer approximation, no pyramid
  embedding. Report it as "a two-block transformer MIL with PPEG", not as a reproduction.
* **No heatmaps exported for this model.** Attention export exists
  (`forward(..., return_patch_embeddings=True)`), but no `results/heatmaps/transmil/`
  export was produced, so TransMIL contributes no localisation evidence. If a heatmap
  figure is wanted for Person 3's slide, that is one `src/export_attention.py` run.
* **Highest seed variance** of any model in the project (sd 0.0364). Quote the CI when
  mentioning the mean, or the reader will over-read a 3-seed average.
* **Same data limitations as everyone else:** 420 Karolinska slides, 4 % of PANDA, no
  masks, no external validation.

---

## 8. Artifacts

| Artifact | Path |
| --- | --- |
| Weights, 3 seeds | `results/metrics/transmil/checkpoints/transmil_seed{0,1,2}_best.pt` |
| Per-seed metrics | `results/metrics/transmil/metrics_transmil_seed{0,1,2}.json` |
| Predictions | `results/metrics/transmil/predictions_transmil_seed{0,1,2}.csv` |
| Training histories | `results/metrics/transmil/history_transmil_seed{0,1,2}.csv` |
| Aggregate + config | `results/metrics/transmil/aggregate_transmil.json`, `run_config.json` |
| Confusion matrices | `results/metrics/transmil/confusion_transmil_seed{0,1,2}.png` |
| Reproduce | `python -m src.train --model transmil --manifest data/common_split.csv --features-root data/features --config configs/transmil.yaml --out-dir results/metrics/transmil --seeds 0 1 2` |

Reference: Shao et al., *TransMIL: Transformer based Correlated Multiple Instance Learning
for Whole Slide Image Classification*, NeurIPS 2021. Implemented from scratch; no official
code copied.

# From Patches to Prognosis

**Weakly Supervised Multiple Instance Learning for ISUP Prostate Cancer Grading and Tumor Localization**

[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-EE4C2C.svg?logo=pytorch&logoColor=white)](https://pytorch.org)
[![Python](https://img.shields.io/badge/Python-3.10+-blue.svg?logo=python&logoColor=white)](https://python.org)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

An end-to-end MIL framework, written **from scratch in PyTorch**, that grades pan-cancer
prostate biopsy whole-slide images into ISUP Grade Groups 0-5 using **slide-level labels
only**, and produces attention heatmaps that localise suspicious regions.

> **Status: research prototype.** Not a medical device. It is trained on a 420-slide
> subset of PANDA, has never been externally validated, and its attention maps are
> interpretability cues, not tumour segmentations.

---

## 1. What works right now

| Component | Owner | File | State |
| --- | --- | --- | --- |
| Data contract + split | Person 1 | `src/dataset.py`, `scripts/create_split.py` | ✅ implemented (needs PANDA `train.csv`) |
| Tissue detection + 256×256 tiling | Person 1 | `src/preprocessing.py` | ✅ implemented |
| Frozen ResNet50 feature extraction | Person 1 | `src/feature_extraction.py` | ✅ implemented (needs download of weights) |
| Mock data bridge | Person 1 | `src/utils/mock_data.py` | ✅ implemented |
| **ABMIL / gated attention** | Person 1 | `src/models/abmil.py` | ✅ implemented, tested |
| **CLAM-SB + instance clustering** | **Person 2** | `src/models/clam_sb.py` | ✅ implemented, tested |
| **CLAM-SB ablation (with/without instance loss)** | **Person 2** | `src/train_clam_sb.py` | ✅ implemented, runs end-to-end |
| **CLAM-SB attention heatmaps + localisation score** | **Person 2** | `src/export_clam_attention.py`, `src/heatmaps.py` | ✅ implemented |
| **Paired ABMIL-vs-CLAM-SB statistics** | **Person 2** | `src/compare_clam_abmil.py` | ✅ implemented |
| Simplified TransMIL (PPEG + CLS) | Person 3 | `src/models/transmil.py` | ✅ implemented, tested |
| Patch CNN + ordinal/XGBoost baselines | Person 4 | `src/models/patch_cnn.py`, `src/statistical_models.py` | ✅ implemented (patch CNN needs patch labels) |
| Metrics: QWK, bootstrap CI, McNemar, overlap | Person 4 | `src/evaluate.py` | ✅ implemented |
| Streamlit demo | Person 4 | `app/streamlit_app.py` | ✅ implemented |
| Shared trainer (all models, 3 seeds) | Person 1 | `src/train.py` | ✅ implemented |
| Notebooks 01-06 | everyone | `notebooks/` | ✅ 04 is complete; 01-03/05-06 are runnable starters |
| **Real PANDA results** | Person 1 | `results/metrics/` | ⛔ blocked on the PANDA download + feature extraction |

Everything above except the last row can be exercised **today** on synthetic data.
No number produced on synthetic data may be reported — see `docs/RESULTS_LOG.md`.

---

## 2. Architecture at a glance

```
WSI (.tiff/.svs)
   │  preprocessing.py      Otsu tissue mask → non-overlapping 256×256 tiles
   ▼
data/patches/{image_id}/patches/*.png + coords.csv
   │  feature_extraction.py  frozen ImageNet ResNet50 (2048-d, optional 1024-d projection)
   ▼
data/features/{image_id}.pt   {"features": [N, D], "coords": [N, 2], "patch_paths": [N]}
   │
   ├── ABMIL           gated attention pooling            (Person 1)
   ├── CLAM-SB         + instance clustering loss         (Person 2)
   ├── TransMIL        PPEG + transformer over patches    (Person 3)
   └── statistical     attention stats → ordinal logreg   (Person 4)
   ▼
ISUP grade 0-5 + attention heatmap + evaluation metrics
```

The shared tensor contract (`[N, D]` per slide, `[N, 2]` patch coordinates) is documented
in [`docs/DATA_CONTRACT.md`](docs/DATA_CONTRACT.md). Every model implements the same
`forward(features, mask, labels) -> {"logits", "attention", ...}` interface, so the
trainer, evaluator, heatmap renderer and app work with all of them.

---

## 3. Quickstart

```bash
git clone <repo-url> && cd prostate-mil-project
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
# CPU-only machine? install torch separately, it is much smaller:
# pip install torch --index-url https://download.pytorch.org/whl/cpu
```

### 3a. Before Person 1's real features exist (synthetic, ~1 minute)

```bash
bash scripts/run_mock_demo.sh
```

Generates 30 synthetic slides that obey the real file contract, then runs the whole
Person 2 pipeline: ABMIL baseline → CLAM-SB with/without instance loss → evaluation →
attention heatmaps with localisation overlap → statistical baseline → paired comparison.

Outputs land in `results/mock_*` and every folder carries a `README.md` saying
**SYNTHETIC — DO NOT REPORT**.

### 3b. On real PANDA features

```bash
# 1. split  (needs the PANDA train.csv metadata in data/raw/)
python scripts/create_split.py --panda-csv data/raw/train.csv --out data/common_split.csv

# 2. tile + featurise (per slide)
python -m src.preprocessing      --slide data/raw/<slide>.tiff --out data/patches/<image_id>
python -m src.feature_extraction --patches-root data/patches --features-root data/features

# 3. everybody trains on the same split
python -m src.train --model abmil    --config configs/abmil.yaml    --out-dir results/metrics/abmil
python -m src.train --model transmil --config configs/transmil.yaml --out-dir results/metrics/transmil
bash scripts/run_person2_experiments.sh     # CLAM-SB ablation + heatmaps + comparison

# 4. demo
streamlit run app/streamlit_app.py
```

### 3c. Tests

```bash
python -m unittest discover -s tests -v       # 21 tests, ~30 s, CPU only
```

---

## 4. CLAM-SB (Person 2) in one paragraph

CLAM-SB adds *instance-level supervision* to a gated-attention MIL pooler: for a slide
whose true grade is `k`, the `B=8` highest-attention patches are pseudo-labelled
**positive** and the `B` lowest-attention patches **negative**, and a class-specific
binary head is trained with a smooth top-1 SVM loss to separate them. The total loss is
`L_slide + λ · L_instance` with `λ = 0.3`. Setting `λ = 0` gives the ablation arm — same
network, same seed, same features, no instance supervision.

> **CLAM-SB tests whether forcing the model to distinguish high- and low-attention patches
> improves weakly supervised grading.**

Reports two things for each arm: QWK (does grading improve?) and attention entropy /
top-10 attention share (does attention get sharper?).

See [`PERSON2_README.md`](PERSON2_README.md) for the full Person 2 handbook.

---

## 5. Evaluation protocol

* **Primary metric:** Quadratic Weighted Kappa (QWK) on ISUP 0-5 — the PANDA metric, and
  the one that respects the ordinal structure (grade 0→5 errors are punished far more
  than 2→3).
* **Secondary:** accuracy, balanced accuracy, macro F1, per-class precision/recall,
  confusion matrix, attention entropy, inference time.
* **Statistics:** mean ± sd over 3 seeds, bootstrap 95% CIs (2000 resamples), paired
  bootstrap CI for model differences, exact McNemar test on discordant slides.
* **Model selection:** validation QWK only. The test split (60 slides) is touched once.
* **Guard:** with 60 test slides a non-significant McNemar test means *inconclusive*,
  not *equivalent*. Report CIs, never bare point estimates.

---

## 6. Repository layout

```
prostate-mil-project/
├── README.md                  ← you are here
├── PLAN.md                    team plan & milestones
├── PERSON2_README.md          Person 2 (CLAM-SB) handbook
├── AGENT_MASTER_INSTRUCTIONS.md  original build blueprint (kept for reference)
├── configs/                   YAML hyperparameters (real + mock variants)
├── data/                      raw / patches / features (all gitignored) + common_split.csv
├── docs/                      DATA_CONTRACT.md, RESULTS_LOG.md
├── notebooks/                 01 EDA … 06 localisation & stats
├── scripts/                   create_split, make_mock_features, run_* pipelines
├── src/
│   ├── dataset.py             split manifest + feature-bag loader (shared contract)
│   ├── preprocessing.py       Otsu tissue detection + tiling
│   ├── feature_extraction.py  frozen ResNet50 features
│   ├── train.py               shared multi-seed trainer (any model)
│   ├── evaluate.py            metrics, bootstrap, McNemar, overlap score
│   ├── heatmaps.py            attention → heatmap/overlay + localisation scoring
│   ├── statistical_models.py  x_stat features + ordinal logreg / XGBoost
│   ├── train_clam_sb.py       Person 2: instance-loss ablation
│   ├── evaluate_clam_sb.py    Person 2: checkpoint evaluation + attention export
│   ├── export_clam_attention.py Person 2: heatmaps + top-k patches + overlap
│   ├── compare_clam_abmil.py  Person 2: paired statistics vs ABMIL
│   ├── models/                abmil, clam_sb, transmil, patch_cnn, base
│   └── utils/mock_data.py     synthetic bag generator
├── tests/                     unit + end-to-end tests
├── app/streamlit_app.py       interactive demo
├── results/                   metrics / heatmaps / figures (generated)
└── report/                    final write-up
```

---

## 7. Team & ownership

| Person | Owns | Do not edit without asking |
| --- | --- | --- |
| 1 (lead) | split, preprocessing, features, ABMIL, shared trainer | `data/common_split.csv`, `src/dataset.py`, `src/train.py` |
| 2 | CLAM-SB, ablation, heatmaps, ABMIL comparison | `src/models/clam_sb.py`, `src/{train,evaluate}_clam_sb.py`, `src/export_clam_attention.py`, `src/compare_clam_abmil.py` |
| 3 | TransMIL | `src/models/transmil.py` |
| 4 | patch CNN, statistical baselines, metrics, app | `src/evaluate.py`, `src/statistical_models.py`, `app/` |

Shared rules: no test-set tuning, no copying official CLAM/TransMIL code, no pixel-level
labels during training, fixed seeds (`0, 1, 2`), everything reproducible from the repo.

---

## 8. Limitations to carry into the write-up

* Research prototype; **not** a clinical diagnostic tool and not validated for clinical use.
* 420-slide stratified subset of PANDA — the full cohort (10,616 slides) and other
  scanners/labs are not represented, so generalisation is untested.
* ISUP grades are imbalanced (grades 0 and 5 are scarce), so **accuracy alone is
  misleading**; QWK, balanced accuracy and per-class recall are the honest numbers.
* Attention heatmaps are interpretability cues, not guaranteed tumour segmentations;
  overlap scores use coarse masks because PANDA has no pixel-level ISUP annotation.
* External hospital validation is required before any real-world deployment.

---

## 9. References

* Lu et al., *Data-efficient and weakly supervised computational pathology on whole-slide
  images*, Nature Biomedical Engineering 5, 555-570 (2021) — CLAM.
* Ilse, Tomczak & Welling, *Attention-based Deep Multiple Instance Learning*, ICML 2018 — ABMIL.
* Shao et al., *TransMIL: Transformer based Correlated Multiple Instance Learning for WSI
  Classification*, NeurIPS 2021.
* Bulten et al., *Artificial intelligence for diagnosis and Gleason grading of prostate
  cancer: the PANDA challenge*, Nature Medicine 28, 154-163 (2022).

Implementations here are written from the papers and equations; no code from the
official CLAM or TransMIL repositories was copied.

## License

MIT — see [LICENSE](LICENSE).

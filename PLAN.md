# 🔬 Master Project Plan: From Patches to Prognosis

> **Implementation status (read me first):** the scaffolding in this plan now exists as
> working code. See [`README.md`](README.md) §1 for what is implemented, tested and
> blocked, [`docs/DATA_CONTRACT.md`](docs/DATA_CONTRACT.md) for the file contract every
> model depends on, and [`docs/RESULTS_LOG.md`](docs/RESULTS_LOG.md) for where reportable
> numbers go. Person 2's handbook is [`PERSON2_README.md`](PERSON2_README.md).

**Weakly Supervised Multiple Instance Learning for ISUP Prostate Cancer Grading & Tumor Localization**

---

## 1. Executive Summary & Objective

This project implements a weakly supervised Multiple Instance Learning (MIL) pipeline to predict prostate cancer severity (ISUP grade groups 0–5) from gigapixel Whole-Slide Images (WSIs) using **slide-level labels only**. In addition to grading, the pipeline generates spatial attention heatmaps to localize diagnostically relevant tumor regions.

- **Dataset**: PANDA (Prostate cANcer graDe Assessment) challenge dataset.
- **Initial Practical Scope**: 420 slides total (300 Train / 60 Val / 60 Test) with stratified ISUP grade group distribution.
- **Primary Metric**: Quadratic Weighted Kappa (QWK) — penalizing large ordinal discrepancies more heavily than adjacent grade shifts.
- **Core Models to Implement From Scratch**:
  1. **ABMIL / Gated Attention MIL** (Person 1 — Team Lead)
  2. **CLAM-SB** (Person 2)
  3. **Simplified TransMIL** (Person 3)
  4. **Patch-level CNN Classifier & Statistical Baselines** (Person 4)

---

## 2. Repository Structure

```
prostate-mil-project/
├── PLAN.md                        # Master Execution & Integration Plan (This document)
├── README.md                      # Team onboarding, overview, and quickstart guide
├── requirements.txt               # Pinned project dependencies
├── .gitignore                     # Ignores .svs, .npy, .pt, and large caches
├── data/
│   ├── raw/                       # Downloaded PANDA .svs whole slide images
│   ├── patches/                   # Extracted 256x256 image patches
│   ├── features/                  # Extracted feature vectors (.pt / .npy)
│   └── common_split.csv           # Stratified train/val/test slide IDs
├── src/
│   ├── __init__.py
│   ├── dataset.py                 # PyTorch Bag / Slide Dataset (generator-based)
│   ├── preprocessing.py           # Tissue detection & 256x256 patch extraction
│   ├── feature_extraction.py      # ResNet50 / Foundation model feature extractor
│   ├── train.py                   # Shared MIL training loop
│   ├── evaluate.py                # Evaluation suite (QWK, Macro F1, Confusion Matrix)
│   ├── heatmaps.py                # Attention/probability heatmap rendering
│   ├── statistical_models.py      # Ordinal logistic regression & XGBoost
│   ├── models/
│   │   ├── __init__.py
│   │   ├── abmil.py               # ABMIL baseline (Person 1)
│   │   ├── clam_sb.py             # CLAM-SB model (Person 2)
│   │   ├── transmil.py            # TransMIL model (Person 3)
│   │   └── patch_cnn.py           # Patch-level CNN (Person 4)
│   └── utils/
│       ├── __init__.py
│       └── mock_data.py           # Synthetic bag tensor generator for team unblocking
├── notebooks/
│   ├── 01_eda.ipynb               # Dataset distribution and slide exploration
│   ├── 02_patch_extraction.ipynb  # Tissue masking and tiling inspection
│   ├── 03_abmil_training.ipynb    # Person 1 ABMIL training & evaluation
│   ├── 04_clam_training.ipynb     # Person 2 CLAM-SB training & ablation
│   ├── 05_transmil_training.ipynb # Person 3 TransMIL training & attention analysis
│   └── 06_localization_and_stats.ipynb # Person 4 localization & statistical baseline
├── results/
│   ├── metrics/                   # Saved CSVs, JSON classification reports
│   ├── heatmaps/                  # Generated attention overlay images
│   └── figures/                   # Loss curves, confusion matrices, ROC plots
├── app/
│   └── streamlit_app.py           # Interactive slide viewer and grading demo
└── report/
    └── final_report.pdf           # Final IEEE/Springer style research report
```

---

## 3. Team Role Allocations & Deliverables

### Person 1 — Project Lead & ABMIL Baseline (Your Role)
- **Primary Goal**: Set up repository infrastructure, freeze the scientific data split, build the data & patch pipeline, and implement the ABMIL baseline from scratch to unblock all teammates.
- **Responsibilities**:
  1. Initialize and maintain the shared GitHub repository, environment, and `.gitignore`.
  2. Create `common_split.csv` (300 Train / 60 Val / 60 Test, stratified by ISUP grade).
  3. Implement `src/utils/mock_data.py` so Persons 2 & 3 can immediately build and test their model architectures before slide preprocessing finishes.
  4. Implement `src/preprocessing.py` (Otsu tissue masking + non-overlapping $256 \times 256$ patch tiling).
  5. Implement `src/feature_extraction.py` (pretrained CNN feature extraction $\rightarrow$ saved `.pt` vectors).
  6. Implement **ABMIL / Gated Attention MIL** from scratch in `src/models/abmil.py`:
     $$a_n = \frac{\exp(W^T (\tanh(V h_n) \odot \mathrm{sigmoid}(U h_n)))}{\sum_{j=1}^N \exp(W^T (\tanh(V h_j) \odot \mathrm{sigmoid}(U h_j)))}$$
  7. Train baseline model, record baseline QWK and Macro F1, and render attention heatmaps for $\ge 10$ test slides.

### Person 2 — CLAM-SB Model Owner
- **Primary Goal**: Instance-level clustering and pseudo-label regularization.
- **Responsibilities**:
  1. Implement **CLAM-SB** (Single-Branch Clustering-constrained Attention MIL) from scratch in `src/models/clam_sb.py`.
  2. Implement instance-level pseudo-labeling: highest-attention patches as positive instances, lowest-attention as negative instances.
  3. Evaluate the impact of instance-level loss on bag classification accuracy and attention sharpness.
  4. Compare performance directly against Person 1's ABMIL baseline using `common_split.csv`.

### Person 3 — TransMIL Model Owner
- **Primary Goal**: Spatial context and instance correlation modeling.
- **Responsibilities**:
  1. Implement a **Simplified TransMIL** from scratch in `src/models/transmil.py`.
  2. Implement positional encoding / spatial coordinates for extracted patches.
  3. Model patch-to-patch correlations using Transformer self-attention layers (or Nyström self-attention).
  4. Aggregate sequence tokens into slide-level ISUP grade group predictions.
  5. Compare grading performance and attention behavior against ABMIL and CLAM-SB.

### Person 4 — Localization, Interpretability & Statistical Baselines
- **Primary Goal**: Patch-level validation, heatmap stitching, and non-deep-learning baselines.
- **Responsibilities**:
  1. Implement a small **Patch CNN Classifier** in `src/models/patch_cnn.py` to classify patches (tumor vs. benign).
  2. Stitch patch probabilities into slide-level spatial heatmaps and compare against MIL attention maps.
  3. Extract bag-level summary statistics (mean attention, top-$k$ attention, attention entropy, embedding variance).
  4. Train an **Ordinal Logistic Regression** or **XGBoost** baseline in `src/statistical_models.py`.
  5. Implement evaluation metrics (QWK, bootstrap 95% confidence intervals, McNemar test) and build `app/streamlit_app.py`.

---

## 4. Execution Phases & Milestones

### Phase 1: Unblocking the Team (Day 1 — Immediate)
1. **Repository Setup**: Complete directory layout, `.gitignore`, and `requirements.txt`.
2. **The Scientific Lock**: Generate `common_split.csv` using stratified sampling across ISUP grades 0, 1, 2, 3, 4, 5.
3. **The Mock Data Bridge**: Provide `src/utils/mock_data.py` returning synthetic bag tensors of shape `(batch_size, num_patches, feature_dim)` to enable model architecture development in parallel.

### Phase 2: Slide Ingestion & Preprocessing (Day 2)
1. **Tissue Detection**: Convert WSI thumbnail to HSV/grayscale $\rightarrow$ apply Otsu thresholding $\rightarrow$ generate binary tissue mask.
2. **Patch Extraction**: Extract $256 \times 256$ non-overlapping patches at $20\times$ equivalent magnification where tissue percentage $> 50\%$.
3. **Feature Extraction**: Pass patches through a frozen ImageNet ResNet50 (or medical pathology encoder) to extract 1024-D or 2048-D feature vectors per patch. Save per slide as `data/features/{slide_id}.pt`.

### Phase 3: Model Development & Baseline Training (Days 3–4)
1. Train ABMIL on Google Colab GPU using precomputed feature tensors.
2. Persons 2 & 3 train CLAM-SB and TransMIL on the identical precomputed features and data splits.
3. Person 4 trains the patch CNN and statistical baselines.

### Phase 4: Evaluation, Interpretability & Final Deliverables (Days 5–6)
1. Compute Quadratic Weighted Kappa (QWK), macro F1, and confusion matrices across all 4 models.
2. Generate attention and probability heatmaps overlaid on slide thumbnails.
3. Launch Streamlit interactive demo (`app/streamlit_app.py`).
4. Assemble final presentation slides and IEEE-style research report.

---

## 5. Key Scientific & Technical Rules

1. **No Data Leakage**: Slides from the test set must never be used in patch feature extraction fitting, hyperparameter selection, or training.
2. **Memory Efficiency**: Always use PyTorch `Dataset` with on-demand tensor loading; never load the full dataset of slide embeddings into memory at once.
3. **Reproducibility**: Fix random seeds (`seed=42`) across PyTorch, NumPy, and Scikit-learn for all training and split generation.

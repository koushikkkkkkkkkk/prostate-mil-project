# 🤖 Autonomous Agent Blueprint: From-Scratch Multiple Instance Learning for Prostate Cancer Grading (PANDA)
**Document Version:** 1.0.0  
**Target Execution Environment:** Linux (Omarchy / Ubuntu) + Python 3.10+ / PyTorch 2.0+ + Google Colab Pro/Plus GPU  
**Project Name:** `prostate-mil-project`  
**Purpose:** Self-contained instruction manual for an autonomous AI agent to scaffold, configure, implement, and benchmark the entire end-to-end weakly supervised Multiple Instance Learning (MIL) prostate cancer grading pipeline from scratch, assuming zero pre-existing project files.

---

## Table of Contents
1. [Agent Persona, Role & Objectives](#1-agent-persona-role--objectives)
2. [Project Overview & Scientific Problem](#2-project-overview--scientific-problem)
3. [Phase 0: Workspace & Directory Initialization](#3-phase-0-workspace--directory-initialization)
4. [Phase 1: Environment & Dependency Setup](#4-phase-1-environment--dependency-setup)
5. [Phase 2: Git Infrastructure & Strict Ignore Rules](#5-phase-2-git-infrastructure--strict-ignore-rules)
6. [Phase 3: The Scientific Lock (Stratified Data Split Protocol)](#6-phase-3-the-scientific-lock-stratified-data-split-protocol)
7. [Phase 4: Synthetic Mock Data Bridge (Immediate Teammate Unblocking)](#7-phase-4-synthetic-mock-data-bridge-immediate-teammate-unblocking)
8. [Phase 5: Data Pipeline Specifications (WSI $\rightarrow$ Patches $\rightarrow$ Embeddings)](#8-phase-5-data-pipeline-specifications-wsi--patches--embeddings)
9. [Phase 6: Model Architectures From Scratch (Detailed Mathematical Contracts)](#9-phase-6-model-architectures-from-scratch-detailed-mathematical-contracts)
   - [Model 1: Gated Attention ABMIL (Person 1 — Team Lead)](#model-1-gated-attention-abmil-person-1--team-lead)
   - [Model 2: CLAM-SB (Person 2)](#model-2-clam-sb-person-2)
   - [Model 3: Simplified TransMIL (Person 3)](#model-3-simplified-transmil-person-3)
   - [Model 4: Patch CNN & Statistical Baselines (Person 4)](#model-4-patch-cnn--statistical-baselines-person-4)
10. [Phase 7: Shared Training & Evaluation Loops](#10-phase-7-shared-training--evaluation-loops)
11. [Phase 8: Interpretability & Heatmap Generation](#11-phase-8-interpretability--heatmap-generation)
12. [Phase 9: Interactive Demo & Reporting](#12-phase-9-interactive-demo--reporting)
13. [Phase 10: Notebook Walkthrough Templates](#13-phase-10-notebook-walkthrough-templates)
14. [Agent Verification Checklist & Sanity Commands](#14-agent-verification-checklist--sanity-commands)

---

## 1. Agent Persona, Role & Objectives

You are acting as **Person 1 (Team Lead & Baseline MIL Architect)**. Your mission is two-fold:
1. **Unblock the Team**: Provide the complete folder structure, reproducible environment configurations, unified stratified data split (`common_split.csv`), and mock data generators so your teammates (Persons 2, 3, and 4) can develop their model architectures in parallel without waiting for whole-slide preprocessing.
2. **Deliver the Baseline**: Build the data ingestion pipeline, implement the **ABMIL (Gated Attention MIL)** baseline from scratch in PyTorch, train the first working model on the PANDA dataset, report Quadratic Weighted Kappa (QWK), and generate spatial attention heatmaps.

---

## 2. Project Overview & Scientific Problem

- **Dataset**: Prostate cANcer graDe Assessment (PANDA) challenge.
- **Input**: Gigapixel Whole-Slide Images (WSIs) of prostate biopsy core needle specimens in `.svs` or `.tiff` format.
- **Labels**: Slide-level ISUP Grade Group ($0, 1, 2, 3, 4, 5$). Ordinal classification task.
- **Weakly Supervised Constraint**: No pixel-level annotations are used during training. Only slide-level grade is provided.
- **Evaluation Metric**: **Quadratic Weighted Kappa (QWK)** is the primary metric:
  $$\kappa = 1 - \frac{\sum_{i,j} w_{i,j} O_{i,j}}{\sum_{i,j} w_{i,j} E_{i,j}}, \quad w_{i,j} = \frac{(i - j)^2}{(N - 1)^2}$$
  Penalizes severe grade misclassifications (e.g., Grade 0 misclassified as Grade 5) exponentially more than adjacent grade errors (e.g., Grade 2 misclassified as Grade 3).
- **Practical Dataset Scope**: 420 slides total:
  - **Train**: 300 slides (stratified by ISUP 0–5)
  - **Validation**: 60 slides (stratified by ISUP 0–5)
  - **Test**: 60 slides (stratified by ISUP 0–5)

---

## 3. Phase 0: Workspace & Directory Initialization

If the target folder `prostate-mil-project` does not exist, the agent must create the exact hierarchy below using shell commands:

```bash
mkdir -p prostate-mil-project
cd prostate-mil-project

# Data landing folders (must be tracked by git using .gitkeep)
mkdir -p data/raw data/patches data/features

# Core Python package modules
mkdir -p src/models src/utils

# Jupyter analysis and training notebooks
mkdir -p notebooks

# Metric logs, figures, and heatmap outputs
mkdir -p results/metrics results/heatmaps results/figures

# Interactive web demo
mkdir -p app

# Final research paper and documentation
mkdir -p report

# Preserve empty directories in Git
touch data/raw/.gitkeep \
      data/patches/.gitkeep \
      data/features/.gitkeep \
      results/metrics/.gitkeep \
      results/heatmaps/.gitkeep \
      results/figures/.gitkeep \
      report/.gitkeep
```

### Complete Target Directory Layout
```
prostate-mil-project/
├── .gitignore
├── requirements.txt
├── README.md
├── PLAN.md
├── AGENT_MASTER_INSTRUCTIONS.md
├── data/
│   ├── raw/.gitkeep
│   ├── patches/.gitkeep
│   ├── features/.gitkeep
│   └── common_split.csv
├── src/
│   ├── __init__.py
│   ├── dataset.py
│   ├── preprocessing.py
│   ├── feature_extraction.py
│   ├── train.py
│   ├── evaluate.py
│   ├── heatmaps.py
│   ├── statistical_models.py
│   ├── models/
│   │   ├── __init__.py
│   │   ├── abmil.py
│   │   ├── clam_sb.py
│   │   ├── transmil.py
│   │   └── patch_cnn.py
│   └── utils/
│       ├── __init__.py
│       └── mock_data.py
├── notebooks/
│   ├── 01_eda.ipynb
│   ├── 02_patch_extraction.ipynb
│   ├── 03_abmil_training.ipynb
│   ├── 04_clam_training.ipynb
│   ├── 05_transmil_training.ipynb
│   └── 06_localization_and_stats.ipynb
├── results/
│   ├── metrics/.gitkeep
│   ├── heatmaps/.gitkeep
│   └── figures/.gitkeep
├── app/
│   └── streamlit_app.py
└── report/
    └── .gitkeep
```

---

## 4. Phase 1: Environment & Dependency Setup

The agent must create `requirements.txt` with verified production dependencies:

```text
# Deep Learning Core
torch>=2.0.0
torchvision>=0.15.0
torchaudio

# Histopathology & Computer Vision
openslide-python
opencv-python-headless
Pillow

# Data Analysis, Metrics & ML
numpy<2.0.0
pandas>=2.0.0
scipy>=1.10.0
scikit-learn>=1.2.0

# Visualization & Reporting
matplotlib>=3.7.0
seaborn>=0.12.0

# Interactive Web Application
streamlit>=1.28.0

# Notebook Environment & Progress Tracking
tqdm>=4.65.0
jupyter
ipykernel
```

### OS-Level Dependencies (Linux / Ubuntu / Arch)
When running whole-slide image loaders on Linux, `openslide` native C libraries are required:
- Debian/Ubuntu: `sudo apt-get install -y libopenslide-dev libvips-dev`
- Arch Linux: `sudo pacman -S openslide libvips`

---

## 5. Phase 2: Git Infrastructure & Strict Ignore Rules

Histopathology files (`.svs`) are 500MB to 2GB each; feature embeddings (`.npy`, `.pt`) reach tens of gigabytes. Under no circumstances should raw binary data or model weights be committed to Git.

Create `.gitignore`:

```gitignore
# Gigapixel WSI and Archive Files
*.svs
*.tif
*.tiff
*.ndpi
*.h5
*.hdf5
*.zip
*.tar.gz

# Data directories (keep structure with .gitkeep)
data/raw/*
!data/raw/.gitkeep
data/patches/*
!data/patches/.gitkeep
data/features/*
!data/features/.gitkeep

# PyTorch Weights & Cached Numerical Tensors
*.npy
*.npz
*.pt
*.pth
*.bin
*.ckpt
*.onnx

# Python bytecode & environments
__pycache__/
*.py[cod]
*$py.class
.venv/
env/
venv/
ENV/

# Jupyter Checkpoints & OS Junk
.ipynb_checkpoints
*/.ipynb_checkpoints/*
.DS_Store
.idea/
.vscode/
*.swp
```

Initialize Git repository:
```bash
git init -b main
git add .
git commit -m "chore: scaffold complete repository structure and instructions"
```

---

## 6. Phase 3: The Scientific Lock (Stratified Data Split Protocol)

To prevent data leakage and guarantee that Persons 1, 2, 3, and 4 benchmark their models on identical samples, create a stratified split script.

### Specifications for `data/common_split.csv`
1. **Total Slides**: 420 (300 Train / 60 Validation / 60 Test).
2. **Stratification Key**: `isup_grade` (Values: 0, 1, 2, 3, 4, 5).
3. **Reproducibility**: Random seed fixed to `42`.
4. **Columns**:
   - `image_id`: Unique slide identifier string (e.g. `0005f7aa...`).
   - `data_provider`: `karolinska` or `radboud`.
   - `isup_grade`: Integer from 0 to 5.
   - `gleason_score`: String (e.g. `negative`, `0+0`, `3+3`, `3+4`, `4+3`, `4+4`, `3+5`, `4+5`, `5+4`, `5+5`).
   - `split`: String enum: `train`, `val`, `test`.

### Split Generation Logic (`scripts/create_split.py` or within `src/dataset.py`)
```python
import pandas as pd
from sklearn.model_selection import train_test_split

def generate_stratified_split(panda_csv_path, output_path="data/common_split.csv", seed=42):
    df = pd.read_csv(panda_csv_path)
    
    # Filter valid slides and balance across ISUP grades
    # Target: 50 per class for train (300 total), 10 per class for val (60 total), 10 per class for test (60 total)
    train_dfs, val_dfs, test_dfs = [], [], []
    for grade in range(6):
        grade_df = df[df["isup_grade"] == grade].sample(n=70, random_state=seed)
        train_dfs.append(grade_df.iloc[:50])
        val_dfs.append(grade_df.iloc[50:60])
        test_dfs.append(grade_df.iloc[60:70])
        
    train_df = pd.concat(train_dfs).assign(split="train")
    val_df = pd.concat(val_dfs).assign(split="val")
    test_df = pd.concat(test_dfs).assign(split="test")
    
    split_df = pd.concat([train_df, val_df, test_df]).sample(frac=1.0, random_state=seed).reset_index(drop=True)
    split_df[["image_id", "data_provider", "isup_grade", "gleason_score", "split"]].to_csv(output_path, index=False)
```

---

## 7. Phase 4: Synthetic Mock Data Bridge (Immediate Teammate Unblocking)

Create `src/utils/mock_data.py`. This allows Persons 2 and 3 to build their PyTorch models (`clam_sb.py`, `transmil.py`) and Person 4 to write the evaluation scripts immediately without waiting for gigapixel slide downloads.

```python
"""
Mock Data Generator for PANDA Multiple Instance Learning.
Mimics bag-level feature embeddings and ordinal ISUP targets.
"""
import torch
from typing import Tuple, List

def generate_mock_bag(
    num_patches: int = 400,
    feature_dim: int = 1024,
    device: str = "cpu"
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Generates a single slide bag tensor with random instance embeddings.
    
    Returns:
        features: Tensor of shape (num_patches, feature_dim)
        label: Tensor of shape (1,) with an ISUP grade integer [0..5]
    """
    features = torch.randn(num_patches, feature_dim, device=device)
    label = torch.randint(0, 6, (1,), device=device)
    return features, label

def generate_mock_batch(
    batch_size: int = 4,
    min_patches: int = 200,
    max_patches: int = 600,
    feature_dim: int = 1024
) -> Tuple[List[torch.Tensor], torch.Tensor]:
    """
    Generates a batch of bags with variable patch counts (standard in WSI MIL).
    
    Returns:
        bags: List of Tensors, each of shape (N_i, feature_dim)
        labels: Tensor of shape (batch_size,) containing ISUP grades
    """
    bags = []
    labels = torch.randint(0, 6, (batch_size,))
    for _ in range(batch_size):
        n_patches = torch.randint(min_patches, max_patches, (1,)).item()
        bags.append(torch.randn(n_patches, feature_dim))
    return bags, labels
```

---

## 8. Phase 5: Data Pipeline Specifications (WSI $\rightarrow$ Patches $\rightarrow$ Embeddings)

### 1. Tissue Detection & Patch Extraction (`src/preprocessing.py`)
- **Step 1: Thumbnail Generation**: Read slide thumbnail at level 2 or low resolution ($32\times$ downsample).
- **Step 2: Otsu Masking**: Convert thumbnail to HSV or grayscale $\rightarrow$ Gaussian blur ($5\times 5$) $\rightarrow$ Otsu thresholding to segment tissue from glass white background.
- **Step 3: Coordinate Grid Generation**: Step across the slide at highest magnification ($20\times$ or level 0) in non-overlapping $256 \times 256$ pixel steps.
- **Step 4: Filtering**: For each bounding box $(x, y, 256, 256)$, check tissue mask overlap. Discard if tissue ratio $< 50\%$.
- **Output**: Save coordinates as a CSV `data/patches/{slide_id}_coords.csv` or extract patch images to `data/patches/{slide_id}/`.

### 2. Feature Extraction (`src/feature_extraction.py`)
- Use a pre-trained feature extractor:
  - Standard Baseline: `torchvision.models.resnet50(weights='IMAGENET1K_V2')` with classification head removed (`avgpool` output: 2048-D).
  - Pathology Foundation Alternative: CTransPath / UNI / Phikon (1024-D).
- Extract embeddings in batches ($B=128$) on GPU:
  ```python
  # Output: Tensor of shape (N_patches, feature_dim)
  torch.save(slide_features, f"data/features/{slide_id}.pt")
  ```

### 3. Memory-Safe Dataset Loader (`src/dataset.py`)
- **Rule**: Never load all slide embeddings into RAM at once.
- Subclass `torch.utils.data.Dataset`:
  - `__getitem__(idx)`: Reads `data/features/{slide_id}.pt` from disk dynamically on demand, returning `features` $(N, D)$ and `label` $(1)$.

---

## 9. Phase 6: Model Architectures From Scratch (Detailed Mathematical Contracts)

### Model 1: Gated Attention ABMIL (Person 1 — Team Lead)
**File:** `src/models/abmil.py`  
**Reference Paper:** Ilse et al., *"Attention-based Deep Multiple Instance Learning"*, ICML 2018.

#### Mathematical Formulation
Given a bag of $N$ instance embeddings $H = \{h_1, h_2, \dots, h_N\} \in \mathbb{R}^{N \times D}$:
1. **Low-dimensional projection**:
   $$h'_n = \mathrm{ReLU}(V_0 h_n), \quad V_0 \in \mathbb{R}^{L \times D}$$
2. **Gated Attention Mechanism**:
   $$a_n = \frac{\exp\left(W^T (\tanh(V h'_n) \odot \mathrm{sigmoid}(U h'_n))\right)}{\sum_{j=1}^N \exp\left(W^T (\tanh(V h'_j) \odot \mathrm{sigmoid}(U h'_j))\right)}$$
   Where $V \in \mathbb{R}^{M \times L}$, $U \in \mathbb{R}^{M \times L}$, $W \in \mathbb{R}^{M \times 1}$, and $\odot$ is element-wise multiplication.
3. **Bag Representation (Attention-weighted pooling)**:
   $$z = \sum_{n=1}^N a_n h'_n \in \mathbb{R}^L$$
4. **Slide-level ISUP Prediction**:
   $$\hat{y} = \mathrm{Classifier}(z) \in \mathbb{R}^{6}$$

#### Class Skeleton to Implement
```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class GatedAttention(nn.Module):
    def __init__(self, in_features: int = 1024, hidden_dim: int = 512, attn_dim: int = 256):
        super().__init__()
        self.feature_extractor = nn.Sequential(
            nn.Linear(in_features, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.25)
        )
        self.attention_V = nn.Linear(hidden_dim, attn_dim)
        self.attention_U = nn.Linear(hidden_dim, attn_dim)
        self.attention_weights = nn.Linear(attn_dim, 1)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        # x shape: (N, in_features)
        h = self.feature_extractor(x)  # (N, hidden_dim)
        v = torch.tanh(self.attention_V(h))  # (N, attn_dim)
        u = torch.sigmoid(self.attention_U(h))  # (N, attn_dim)
        raw_weights = self.attention_weights(v * u)  # (N, 1)
        A = F.softmax(raw_weights, dim=0)  # (N, 1) normalized attention
        M = torch.sum(A * h, dim=0, keepdim=True)  # (1, hidden_dim) bag embedding
        return M, A

class ABMIL(nn.Module):
    def __init__(self, in_features: int = 1024, num_classes: int = 6):
        super().__init__()
        self.attention_net = GatedAttention(in_features=in_features)
        self.classifier = nn.Linear(512, num_classes)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        # returns logits: (1, 6), attention_weights: (N, 1)
        bag_embedding, attention_weights = self.attention_net(x)
        logits = self.classifier(bag_embedding)
        return logits, attention_weights
```

---

### Model 2: CLAM-SB (Person 2)
**File:** `src/models/clam_sb.py`  
**Reference Paper:** Lu et al., *"Data-efficient and weakly supervised computational pathology on whole-slide images"*, Nature Biomedical Engineering 2021.

#### Mathematical Formulation & Architecture
1. **Single-Branch Attention**: Computes attention $A \in \mathbb{R}^{N \times 1}$ and pools into slide embedding $z$.
2. **Instance-Level Pseudo-Labeling**:
   - For a slide of class $k$, patches with the top-$K$ highest attention values are pseudo-labeled as **positive instances**.
   - Patches with the bottom-$K$ lowest attention values are pseudo-labeled as **negative instances**.
3. **Clustering Auxiliary Loss**:
   $$\mathcal{L}_{\mathrm{total}} = \mathcal{L}_{\mathrm{slide}} + c_1 \mathcal{L}_{\mathrm{instance\_pos}} + c_2 \mathcal{L}_{\mathrm{instance\_neg}}$$
   Forces positive high-attention patches and negative low-attention patches to separate into distinct clusters in embedding space.

---

### Model 3: Simplified TransMIL (Person 3)
**File:** `src/models/transmil.py`  
**Reference Paper:** Shao et al., *"TransMIL: Transformer based Correlated Multiple Instance Learning for Whole Slide Image Classification"*, NeurIPS 2021.

#### Mathematical Formulation & Architecture
1. **Dimension Reduction**: Project input patch features from $D$ to $d_{\mathrm{model}}$ (e.g. 512).
2. **Spatial Positional Encoding (PPEG)**:
   - Reshape the 1D sequence of patch tokens into a 2D pseudo-image grid based on patch spatial $(x, y)$ coordinates.
   - Apply depthwise convolutional layers with residual connections (Pyramid Position Encoding Generator) to encode spatial relationships without fixed sequence length limits.
3. **Transformer Self-Attention Layers**:
   - Multi-Head Self-Attention (MHSA) computes pairwise patch correlation:
     $$\mathrm{Attention}(Q, K, V) = \mathrm{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)V$$
4. **Classification Token**: Prepend a learnable `[CLS]` token to aggregate sequence context for ISUP grade prediction.

---

### Model 4: Patch CNN & Statistical Baselines (Person 4)
**Files:** `src/models/patch_cnn.py` & `src/statistical_models.py`

#### 1. Patch CNN
- Small 4-layer convolutional network (Conv2D $\rightarrow$ BatchNorm $\rightarrow$ ReLU $\rightarrow$ MaxPool) operating directly on $256 \times 256$ RGB patches.
- Binary classification: Benign tissue (ISUP 0 patterns) vs. Malignant tumor (Gleason 3/4/5 patterns).

#### 2. Statistical Baseline Feature Engineering
Person 4 extracts bag-level feature vectors $x_{\mathrm{stat}} \in \mathbb{R}^{10}$ using attention weights and patch embeddings:
- `mean_attention`: $\frac{1}{N} \sum_{i=1}^N a_i$
- `top_10_attention_mean`: Mean of the 10 highest attention values.
- `attention_entropy`: $-\sum_{i=1}^N a_i \log(a_i + \epsilon)$
- `high_attention_ratio`: Proportion of patches where $a_i > \frac{2}{N}$
- `embedding_mean`: Mean vector across top-10 patch embeddings.
- `embedding_variance`: Variance across patch embeddings.

#### 3. Ordinal Logistic Regression & XGBoost
Fit `OrdinalRidge` or `XGBClassifier` on $x_{\mathrm{stat}}$ to provide a non-deep-learning statistical baseline for ISUP grading.

---

## 10. Phase 7: Shared Training & Evaluation Loops

### Shared Training Script (`src/train.py`)
- **Loss Function**: `nn.CrossEntropyLoss()` or `SmoothL1Loss` / Ordinal Loss.
- **Optimizer**: `torch.optim.AdamW(model.parameters(), lr=2e-4, weight_decay=1e-5)`.
- **LR Scheduler**: `CosineAnnealingLR(optimizer, T_max=epochs)`.
- **Early Stopping**: Monitored on validation Quadratic Weighted Kappa (patience: 10 epochs).

### Primary Metric: Quadratic Weighted Kappa (`src/evaluate.py`)
```python
from sklearn.metrics import cohen_kappa_score, f1_score, confusion_matrix, accuracy_score
import numpy as np

def compute_all_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    return {
        "qwk": float(cohen_kappa_score(y_true, y_pred, weights="quadratic")),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro")),
        "confusion_matrix": confusion_matrix(y_true, y_pred).tolist()
    }
```

---

## 11. Phase 8: Interpretability & Heatmap Generation

**File:** `src/heatmaps.py`
1. For any test slide, forward pass the patch features through ABMIL/CLAM to obtain attention weights $a_n \in [0, 1]$.
2. Normalize attention scores to range $[0, 1]$ using percentile clipping (e.g. 5th and 95th percentiles to eliminate extreme outliers).
3. Map normalized weights onto patch coordinates $(x, y)$.
4. Create an attention overlay using OpenCV's `applyColorMap(..., cv2.COLORMAP_JET)`.
5. Alpha-blend the color map onto the original whole-slide thumbnail ($\alpha = 0.5, \beta = 0.5$).
6. Save heatmaps to `results/heatmaps/{model_name}_{slide_id}.png`.

---

## 12. Phase 9: Interactive Demo & Reporting

### Streamlit Application (`app/streamlit_app.py`)
- **Features**:
  - Slide selector dropdown (choose from the 60 test slides).
  - Displays slide metadata (Data provider, ground-truth ISUP grade, Gleason score).
  - Displays Model Predicted ISUP Grade Group with class confidence bar chart.
  - Side-by-side visualization: Whole Slide Thumbnail vs. Attention Heatmap Overlay.
  - Top-5 Suspicious Patches: Displays the 5 image patches assigned highest attention by the model.

---

## 13. Phase 10: Notebook Walkthrough Templates

The agent must create clean, runnable Jupyter notebooks in `notebooks/`:
- `01_eda.ipynb`: ISUP grade group distribution, provider breakdown (Radboud vs. Karolinska), slide dimensions, missing value checks.
- `02_patch_extraction.ipynb`: Interactive visualization of Otsu tissue thresholding and patch coordinate grid generation.
- `03_abmil_training.ipynb`: Colab-ready notebook for ABMIL training, tracking training loss and validation QWK per epoch.
- `04_clam_training.ipynb`: CLAM-SB model training and instance loss ablation.
- `05_transmil_training.ipynb`: TransMIL model training and correlation matrix visualization.
- `06_localization_and_stats.ipynb`: Patch CNN evaluation, XGBoost training on attention statistics, and ROC/PR curve plotting.

---

## 14. Agent Verification Checklist & Sanity Commands

Before declaring setup or execution complete, an autonomous agent must execute these checks:

```bash
# 1. Directory Tree Integrity
test -d data/raw && test -d data/patches && test -d data/features
test -d src/models && test -d src/utils && test -d notebooks && test -d results

# 2. Syntax Compilation (Ensure zero syntax errors in all python modules)
python -m py_compile src/**/*.py src/*.py

# 3. Notebook JSON Integrity (Ensure notebooks are valid parseable JSON)
python -c '
import json, glob
for nb in glob.glob("notebooks/*.ipynb"):
    with open(nb) as f:
        json.load(f)
print("All notebooks valid JSON.")
'

# 4. Mock Data Generation Test
python -c '
from src.utils.mock_data import generate_mock_batch
bags, labels = generate_mock_batch(batch_size=2, feature_dim=1024)
assert len(bags) == 2
assert labels.shape == (2,)
print("Mock data sanity check passed.")
'

# 5. Git Status Check (Ensure large files ignored and .gitkeep tracked)
git status
```

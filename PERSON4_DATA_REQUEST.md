# Data request to Person 1 (owner of `data/`)

**From:** Person 4 (app, metrics, statistical baselines, patch CNN)
**Status:** five of six deliverables are complete without this. The items below are the only
things still blocked, in priority order. Nothing here is needed to report the current results.

---

## Priority 1 — `data/thumbnails/{image_id}.png` (420 files, ~200 MB)

Slide-level thumbnails, one per slide, named by `image_id`.

* **Why:** the app's "Slide" panel currently says *"No thumbnail available"*. The attention
  heatmap is alpha-blended onto the thumbnail, so without these the overlay renders on a
  blank white canvas.
* **Format:** PNG or JPG, any resolution ≥ 768 px wide, filename = `<image_id>.png`.
* **Source:** the Kaggle PANDA thumbnails, or downscaled from the WSIs.
* **What it unlocks:** real slide image + attention overlay in the demo, and the
  "top suspicious patches" gallery becomes visually meaningful.

## Priority 2 — `data/masks/{image_id}.png` (420 files, ~100 MB)

Binary tissue / tumour masks matching the thumbnail geometry.

* **Why:** `src.heatmaps.evaluate_localisation` needs a mask to produce an overlap score.
  Right now I cannot report a single localisation number, only inter-model agreement.
* **Format:** same pixel grid as the thumbnail, 0 = background, >0 = tissue/tumour.
* **Caveat I will state in the report:** PANDA has no pixel-level ISUP annotation, so this
  measures tissue coverage, not tumour accuracy. I will label it a sanity check.
* **What it unlocks:** a localisation table (mean overlap ± sd) for the ResNet50 models.

## Priority 3 — `patch_paths` inside the feature files (optional)

Currently `data/features/{image_id}.pt` holds `features` + `coords` but not `patch_paths`.

* **Why:** the data contract marks `patch_paths` as "preferred content", and the app's
  top-k patch gallery checks `row["patch_path"].exists()`.
* **Format:** `{"features": [N,D], "coords": [N,2], "patch_paths": [N] list[str]}` where each
  path points at a 256×256 PNG of that patch.
* **What it unlocks:** the app can show the actual highest-attention patch images instead of
  a bare ranking table.

## Priority 4 — patch pixels + patch labels (the patch CNN)

This is the only genuinely large ask, and it is **optional** — the report already documents
the CNN as implemented-but-blocked with this reason.

Needs **all three**:

1. **Patch images.** Either the raw WSIs (Kaggle `prostate-cancer-grade-assessment`, ~347 GB)
   or pre-cut 256×256 tiles. Nothing on HuggingFace offers PANDA patch pixels; I checked.
2. **Pixel-level labels.** The Kaggle release ships **Gleason masks**, which can be
   thresholded per tile into benign / malignant. PANDA itself annotates ISUP at *slide*
   level only, so this is the only non-circular source.
3. **A CSV per split.** `data/patch_labels/train.csv` and `val.csv`, columns
   `patch_path,label` — exactly what `src.models.patch_cnn.train_patch_cnn` consumes.
   No script in the repo produces these today.

* **What it unlocks:** `train_patch_cnn` can run, and patch-level localisation stops being
  an attention proxy.

---

## What I do once each item lands

| item arrives | I run |
|---|---|
| thumbnails | nothing — the app picks them up from `data/thumbnails/` automatically |
| masks | `python -c "from src.heatmaps import evaluate_localisation; ..."` over the exported attention arrays, then add a localisation table to §3 of the report |
| `patch_paths` | nothing — the app's top-k gallery activates |
| patch images + labels CSVs | `train_patch_cnn(...)`, then patch-level AUC/F1 into §9 |

## Two smaller asks

* **Feature folders.** `data/features_chief/` (399 files) and `data/features_gigap/` (391
  files) exist only on my machine and are gitignored. Either share them, or run
  `scripts/download_alt_encoder_features.py --encoder chief` — it rebuilds them from
  HuggingFace in ~10 minutes. The repo needs the script committed either way, or my results
  are not reproducible.
* **`python -m src.train` in_dim bug** (your file). `run_experiment` lets `data.in_dim` from
  the YAML override the dimension probed from disk, and `--in-dim` is discarded afterwards.
  Any encoder that is not 1024-d crashes on the first forward pass. I worked around it with
  per-encoder configs (`configs/abmil_chief.yaml`, `configs/clam_sb_chief.yaml`,
  `configs/abmil_gigap.yaml`); the two-line fix is in my notes if you want it.

---

## Nothing else is blocking

Current state, all reproducible from `results/metrics/`:

| deliverable | state |
|---|---|
| `src/evaluate.py` — metrics, bootstrap CI, McNemar | done |
| `src/statistical_models.py` — ordinal logistic baseline | done |
| `app/streamlit_app.py` — 4-page dashboard | done |
| `notebooks/06` — 16 cells, executes clean | done |
| `report/person4_results.md` — 12 sections | done |
| `src/models/patch_cnn.py` | implemented, blocked on Priority 4 only |

---

## Resolution from Person 1 (Lead / Data Owner) — Completed

All actionable requests have been resolved:

1. **`src/train.py` `in_dim` bug fixed:**
   - Probed feature dimension on disk is now ground truth for model instantiation.
   - Command-line `--in-dim` explicit override takes precedence over YAML defaults.
   - Zero crashes when training alternative foundation models (CHIEF 768-d, GigaPath 1536-d).

2. **Alternative encoder features script committed:**
   - `scripts/download_alt_encoder_features.py`:
     ```bash
     python scripts/download_alt_encoder_features.py --encoder chief
     python scripts/download_alt_encoder_features.py --encoder gigap
     ```
   - Automatically downloads the foundation model embeddings for all 420 slides into `data/features_{encoder}/`.

3. **Thumbnails & Masks pipeline provided (Priority 1 & 2):**
   - `scripts/download_thumbnails_and_masks.py`:
     ```bash
     python scripts/download_thumbnails_and_masks.py \
         --manifest data/common_split.csv \
         --thumbnails-dir data/thumbnails \
         --masks-dir data/masks
     ```
   - Downloads the 420 slide thumbnails (~35 MB) from the open PANDA 512×512 Kaggle release via `kagglehub` and generates matching binary tissue masks with Otsu thresholding.
   - Unlocks the Streamlit app's real biopsy view and `evaluate_localisation` overlap scoring.

4. **Priority 4 (Patch CNN):**
   - Retained as documented: implemented in `src/models/patch_cnn.py`, blocked by the 347 GB raw WSI storage constraint.


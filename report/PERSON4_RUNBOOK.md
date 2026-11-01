# Person 4 runbook — localisation, statistical baselines, patch CNN, evaluation, app

**Scope of this runbook:** everything in `notebooks/06_localization_and_stats.ipynb`, plus
the Person 4 code that notebook depends on (`src/heatmaps.py` localisation scoring,
`src/statistical_models.py`, `src/models/patch_cnn.py`, `src/evaluate.py`, `app/streamlit_app.py`).

**Your hardware:** RTX 5060 laptop (Blackwell, `sm_120`), so **run everything locally**.
Do **not** move Person 4's work to Colab — it is mostly CPU work on small attention tables
and image rendering, and Colab adds a 600 MB re-download plus Drive round-trips for no gain.
Colab is for Person 1/2/3 training bursts if your laptop is busy (see §9).

**Verified facts this runbook relies on** (checked against the live repo and the Hugging Face
feature release on 2026-10-31):

| thing | value |
| --- | --- |
| split file | `data/common_split.csv` — 420 rows, 300 train / 60 val / 60 test, seed 42 |
| features | `Dearcat/CPathPatchFeature` → `panda/r50/pt_files/{image_id}.pt`, **1024-d**, bare `[N, 1024]` tensor |
| patch coords | `panda/patches/{image_id}.h5` → dataset `coords`, int64 `[N, 2]`, **row-aligned** with the `.pt` (verified on 3 slides: 452/452, 203/203, 380/380) |
| coord coverage | ~10/11 sampled manifest slides have `.h5` upstream (a few 404 — a small `b55cae84…` batch is absent) |
| tilings differ per provider | Karolinska grid starts at 3520/3856 with 256 px steps; Radboud slides have a different origin/step. Never assume a global origin. |
| no patch images upstream | the release ships **features + coords only**. `panda/patches/*.h5` contains coords, not pixels. Patch PNGs exist only for the synthetic mock data. |
| PANDA raw (if you ever want pixels) | Kaggle `prostate-cancer-grade-assessment`, ~347 GB images + ~37 GB masks |
| tests | `python -m unittest discover -s tests` → 21 tests pass (~25 s CPU) |
| GPU note | Blackwell needs **torch ≥ 2.7 with cu128** wheels; the default PyPI torch is CPU-only |

---

## 1. What Person 4 actually delivers (and what each number means)

| # | Deliverable | Code | Output file | Reportable? |
| --- | --- | --- | --- | --- |
| 1 | Quantitative localisation score: attention vs. tumour mask, top-k% precision / recall / lift | `src/heatmaps.py::evaluate_localisation` → `src/evaluate.py::attention_mask_overlap` | `results/heatmaps/<model>/localisation_scores.csv` | Only with real masks; PANDA masks are coarse → sanity check, not a segmentation metric |
| 2 | Attention heatmaps + top-k patch tables + contact sheets | `src/export_attention.py` (ABMIL, added by me), `src/export_clam_attention.py` (CLAM-SB, Person 2) | `results/heatmaps/<model>/images/`, `top_patches/`, `arrays/*.npz` | Figures yes, numbers only as qualitative examples |
| 3 | `x_stat` bag statistics → ordinal logistic regression (+ optional XGBoost) | `src/statistical_models.py` | `results/metrics/statistical/<model>/statistical_baselines.json`, `statistical_predictions.csv` | **Yes** — "is MIL needed at all?" baseline |
| 4 | Attention sharpness comparison ABMIL vs CLAM-SB (entropy, top-10 share) | new `scripts/run_person4_experiments.sh` | `results/metrics/attention/attention_stats_abmil_vs_clam.md` | Yes (supports Person 2's claim) |
| 5 | Patch CNN benign-vs-malignant probability map | `src/models/patch_cnn.py` | `results/metrics/patch_cnn.pt` + prob maps | **No AUC/F1** — PANDA has no patch labels (§6) |
| 6 | Metrics library: QWK, bootstrap CI, paired bootstrap, McNemar | `src/evaluate.py` | used by everyone | Yes — you own the metric code, so keep it stable |
| 7 | Streamlit demo | `app/streamlit_app.py` | running app | Demo only |

Rule from the repo (`AGENT_MASTER_INSTRUCTIONS.md`, `docs/RESULTS_LOG.md`): never report a
mock number. Every localisation/patch-CNN claim must say which split, which checkpoint and
which mask source produced it.

---

## 2. Where to run things (the short answer)

| task | where | why |
| --- | --- | --- |
| everything in §3–§8 (attention export, heatmaps, statistical baselines, app, tests) | **local VS Code terminal**, inside the project's venv | CPU-bound, needs your GPU only for training, small files |
| notebook `06_localization_and_stats.ipynb` | **local VS Code Jupyter** (select the project venv kernel) | it shells out to `python -m src…` with `cwd=ROOT`, so paths match the repo |
| training ABMIL / CLAM-SB (needed to *produce* the checkpoints you analyse) | **local, on the RTX 5060** (~2–6 min each for 3 seeds once features exist) | 420 small bags; GPU is idle-fast here |
| big downloads (600 MB features, 347 GB PANDA if used) | **local** (resumable, cached) — Colab kills long downloads | avoids re-downloading after a disconnect |
| Colab | only if you need a second GPU in parallel with teammates | then mount Drive and reuse the same repo + feature folder |

**VS Code setup rules that matter here**

1. `File → Open Folder → prostate-mil-project` (open the *repo root*, not `notebooks/`), otherwise
   `sys.path` and `data/...` relative paths break.
2. `Terminal → New Terminal` gives you a PowerShell prompt in the repo root on Windows.
3. `Ctrl+Shift+P → Python: Select Interpreter → ./.venv/Scripts/python.exe`.
4. Every command in this runbook is run **from the repo root**.
5. On Windows, if `python -m src.train_clam_sb` says *"No module named src"*, you are in the wrong
   folder — `cd` to the repo root. (`python -m src.<module>` always resolves from the cwd.)

---

## 3. Step 0 — one-time environment (local, Windows/PowerShell)

```powershell
git clone https://github.com/koushikkkkkkkkkk/prostate-mil-project.git
cd prostate-mil-project
python -m venv .venv
.\.venv\Scripts\Activate.ps1        # (if blocked: Set-ExecutionPolicy -Scope Process RemoteSigned)

# 1) Blackwell-correct torch FIRST (cu128 wheels; torch >= 2.7 is mandatory for sm_120)
pip install --upgrade pip
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128

# 2) verify the GPU is actually visible to torch
python -c "import torch; print(torch.__version__, torch.cuda.is_available(), torch.cuda.get_device_name(0)); print(torch.cuda.get_arch_list())"
#   expect: 2.x.y+cu128 True NVIDIA GeForce RTX 5060 ... and 'sm_120' in the arch list

# 3) the rest of the project
pip install -r requirements.txt
pip install h5py            # needed for the patch-coordinate download (Person 4 only)
pip install xgboost         # OPTIONAL - enables the second statistical baseline
```

macOS / Linux equivalents: `source .venv/bin/activate`, same pip commands.

Common install failures:

| symptom | fix |
| --- | --- |
| `torch.cuda.is_available()` is `False`, version ends `+cpu` | you installed torch before the cu128 line. `pip uninstall -y torch torchvision` then reinstall from the cu128 index |
| `sm_120 is not compatible with the current PyTorch installation` | torch < 2.7 → upgrade torch from the cu128 index |
| `ModuleNotFoundError: h5py` | `pip install h5py` |
| `openslide` fails to install on Windows | ignore it — nothing in Person 4's path uses OpenSlide (features are pre-extracted) |

Sanity check the repo itself before you touch data:

```powershell
python -m unittest discover -s tests -v      # 21 tests, all OK
```

Optional 30-second end-to-end smoke test on synthetic data (no download at all):

```powershell
bash scripts/run_mock_demo.sh               # Git Bash / WSL / Linux / macOS
# or, on plain PowerShell:
python scripts/make_mock_features.py --out-dir data/mock --n-train 18 --n-val 6 --n-test 6
python -m src.train --model abmil --manifest data/mock/manifest.csv --features-root data/mock/features --config configs/abmil_mock.yaml --out-dir results/mock_abmil --seeds 0
```

Everything that produces lands in `results/mock_*` and carries a "DO NOT REPORT" README.

---

## 4. Step 1 — get the data Person 4 needs

### 4.1 What Person 4 needs, precisely

```
data/common_split.csv                          already in the repo (420 slides)
data/features/{image_id}.pt     {"features":[N,1024], "coords":[N,2]}   <- you need BOTH keys
results/metrics/abmil/checkpoints/abmil_seed0_best.pt        Person 1 (already committed!)
results/metrics/clam_sb/with_instance/checkpoints/clam_sb_seed0_best.pt   Person 2
data/masks/{image_id}.png        OPTIONAL, coarse tumour/tissue mask  -> localisation score
data/thumbnails/{image_id}.png   OPTIONAL, for readable overlays
data/patches/{image_id}/*.png    OPTIONAL, only for the patch CNN (§6)
```

The repo's existing `scripts/download_real_features.py` downloads the features **only**, as a
bare tensor. That trains fine but **cannot render heatmaps or localisation scores**, because
`src/export_clam_attention.py` raises *"has no coords"*. I added a script that downloads both
and merges them into the contract format.

### 4.2 Download features + coords (≈5–15 min, ~700 MB)

```powershell
python scripts/download_features_and_coords.py --limit 5      # smoke test first
python scripts/download_features_and_coords.py               # all 420 slides in the manifest
```

Expected output at the end:

```
features downloaded : 420/420
coords downloaded   : 41x/420
coords merged       : 41x
slides WITHOUT coords (no heatmap/localisation, still trainable): ~5
Percentages for the report: coords coverage = 98.x% of the requested slides.
```

Write that coverage number down — it belongs in the report's limitations ("localisation
analysis covers X of the 420 slides because the public release omits coordinates for N slides").

Verify one file:

```powershell
python -c "import torch; d=torch.load('data/features/85b43e64b68fce39237fcfaf32d5a7d7.pt', weights_only=False); print(list(d), d['features'].shape, d['coords'].shape, d['coords'][:3])"
# expect: ['features', 'coords'] torch.Size([N, 1024]) torch.Size([N, 2]) tensor([[3520., 3856.], ...])
```

### 4.3 Masks and thumbnails (needed for deliverable #1)

Neither the HF release nor the repo ships them, and the three options are genuinely different
choices — pick one and say which in the report:

* **A. PANDA Kaggle masks (best).** Download `train_label_masks.zip` (~37 GB, not the 347 GB
  images) with `kaggle competitions download -c prostate-cancer-grade-assessment -f train_label_masks.zip`,
  unzip into `data/masks_raw/`, then for each slide keep the downsampled mask:
  Radboud masks have per-pixel tissue classes (0 background, 1 stroma, 2 benign epithelium,
  3–5 Gleason 3/4/5) → `mask > 2` is "cancer epithelium", which is the mask the overlap score
  wants. Karolinska masks are coarser (tumour vs non-tumour). Resize to `data/masks/{image_id}.png`,
  grayscale, white = tumour. **This turns deliverable #1 into a real number.**
* **B. Tissue-only masks (cheap fallback).** Otsu on a thumbnail
  (`src/preprocessing.py::detect_tissue_mask`) gives you a tissue mask. Then the overlap score
  measures "does attention sit on tissue" — a much weaker claim; label the table accordingly.
* **C. No masks.** Skip `--masks`. Deliverable #1 becomes qualitative (heatmap figures +
  top-k patch contact sheets) and say exactly that in the report.

Thumbnails, if you take option A/B: the mask downsampled to ~1–2k px works as a background for
the overlay, or grab one low-res level from the Kaggle `.tiff`.

> Cost check before committing to option A: 37 GB download + unzip. On a laptop that is ~1–3 h.
> If the deadline is close, option B or C is defensible in the write-up.

---

## 5. Step 2 — produce the checkpoints you analyse

You need one ABMIL checkpoint (Person 1's, already committed as
`results/metrics/abmil/…` **prediction/metric files only — no checkpoints**), so train one:

```powershell
python -m src.train --model abmil --manifest data/common_split.csv --features-root data/features --config configs/abmil.yaml --out-dir results/metrics/abmil --seeds 0
# ~2-4 min on the RTX 5060 for 1 seed; 3 seeds for the official table

python -m src.train_clam_sb --manifest data/common_split.csv --features-root data/features --config configs/clam_sb.yaml --out-dir results/metrics/clam_sb --seeds 0
# trains BOTH arms (with / without instance loss) and writes ablation_clam_sb.md
```

Checkpoints land at:

```
results/metrics/abmil/checkpoints/abmil_seed0_best.pt
results/metrics/clam_sb/with_instance/checkpoints/clam_sb_seed0_best.pt
results/metrics/clam_sb/without_instance/checkpoints/clam_sb_seed0_best.pt
```

If you just want to *test your Person 4 code* without waiting for real training, reuse the
mock checkpoints from §3 (`results/mock_clam_sb/with_instance/checkpoints/…`) — but only for
plumbing, never for numbers.

---

## 6. Step 3 — run the Person 4 pipeline (one command)

I added `scripts/run_person4_experiments.sh`, which chains the whole Person 4 path on real
features and reuses Person 2's exporters where they already exist:

```powershell
# Git Bash / WSL / Linux / macOS
bash scripts/run_person4_experiments.sh

# variants
SEEDS="0 1 2" SEED=1 bash scripts/run_person4_experiments.sh     # analyse seed 1
TRAIN=0 bash scripts/run_person4_experiments.sh                 # reuse checkpoints, skip training
MASKS=data/masks THUMBS=data/thumbnails bash scripts/run_person4_experiments.sh
```

Equivalent Windows-native commands (run them one by one in PowerShell if bash is unavailable):

```powershell
# A. attention statistics for BOTH models, over train+val+test
python -m src.export_attention --checkpoint results/metrics/abmil/checkpoints/abmil_seed0_best.pt --manifest data/common_split.csv --features-root data/features --out-dir results/metrics/attention/abmil --split all
python -m src.export_attention --checkpoint results/metrics/clam_sb/with_instance/checkpoints/clam_sb_seed0_best.pt --manifest data/common_split.csv --features-root data/features --out-dir results/metrics/attention/clam_sb --split all

# B. statistical baselines (ordinal logistic, + XGBoost if installed)
python -m src.statistical_models --attention-csv results/metrics/attention/abmil/attention_abmil_seed0.csv   --manifest data/common_split.csv --out-dir results/metrics/statistical/abmil
python -m src.statistical_models --attention-csv results/metrics/attention/clam_sb/attention_clam_sb_seed0.csv --manifest data/common_split.csv --out-dir results/metrics/statistical/clam_sb

# C. heatmaps + top-k patches for the test split (ABMIL side)
python -m src.export_attention --checkpoint results/metrics/abmil/checkpoints/abmil_seed0_best.pt --manifest data/common_split.csv --features-root data/features --out-dir results/heatmaps/abmil --split test --render-images --top-k 12 --thumbnails data/thumbnails

# D. heatmaps + localisation score (CLAM-SB side, Person 2's exporter)
python -m src.export_clam_attention --checkpoint results/metrics/clam_sb/with_instance/checkpoints/clam_sb_seed0_best.pt --manifest data/common_split.csv --features-root data/features --out-dir results/heatmaps/clam_sb --split test --top-k 12 --thumbnails data/thumbnails --masks data/masks
```

Why `--split all` for step A: `src/statistical_models.py` fits on the **train** rows of the
attention CSV and scores the **test** rows. A test-only CSV fails with
*"Need overlapping slide ids … for both train and test splits"*. `--split all` is the simplest
correct answer (the model never trains on the CSV — it just reads attention weights).

Outputs you should see:

```
results/metrics/attention/abmil/attention_abmil_seed0.csv         10 x_stat columns + split + labels
results/metrics/attention/abmil/attention_distribution_abmil_seed0.png
results/metrics/attention/attention_stats_abmil_vs_clam.md         sharpness table
results/metrics/statistical/abmil/statistical_baselines.json       QWK per baseline
results/metrics/statistical/abmil/statistical_predictions.csv
results/heatmaps/abmil/images/<slide>_attention.png (+ _overlay.png with thumbnails)
results/heatmaps/abmil/top_patches/<slide>_top12.csv
results/heatmaps/clam_sb/localisation_scores.csv                   only if --masks was given
```

Then check:

```powershell
Get-Content results/metrics/statistical/abmil/statistical_baselines.json
Get-Content results/metrics/attention/attention_stats_abmil_vs_clam.md
```

---

## 7. Step 4 — the notebook `06_localization_and_stats.ipynb`

Your Drive copy and the repo copy are **identical** (I diffed them cell by cell), so work from
the repo copy — do not fork a second version, that is how numbers get out of sync.

1. VS Code → open `notebooks/06_localization_and_stats.ipynb` → kernel = `./.venv`.
2. Run cell 2 (setup). If `Path.cwd().name == "notebooks"` it walks one level up — fine either way.
3. The notebook as committed is **mock-oriented**: section 1 looks for
   `results/mock_heatmaps/arrays` + `data/mock/masks`, section 2 looks for
   `results/mock_eval/attention_*.csv`. So either run `bash scripts/run_mock_demo.sh` first
   (recommended — it makes every cell light up in ~30 s), **or** point the paths at the real
   outputs. Replace the two path blocks with:

```python
# 1. attention-vs-mask overlap (localisation score), real data
from src.heatmaps import evaluate_localisation
import numpy as np, json

slide = "85b43e64b68fce39237fcfaf32d5a7d7"          # any slide from data/common_split.csv
payload = np.load(f"results/heatmaps/clam_sb/arrays/{slide}.npz", allow_pickle=True)
score = evaluate_localisation(payload["attention"], f"data/masks/{slide}.png",
                              payload["coords"], patch_size=256)
print(json.dumps(score, indent=2, default=float))
```

```python
# 2. statistical baselines, real data
import subprocess, sys
subprocess.run([sys.executable, "-m", "src.statistical_models",
                "--attention-csv", "results/metrics/attention/abmil/attention_abmil_seed0.csv",
                "--manifest", "data/common_split.csv",
                "--out-dir", "results/metrics/statistical/abmil"], check=True)
```

4. Section 3 (patch CNN) will only *instantiate* the model (it needs no patch images) — that is
   the intended behaviour; see §8 for what real training needs.

**Figures to lift into the report from this notebook**

* attention-vs-mask overlap JSON (precision/recall/lift at top-10/20/30%) — only with real masks;
* attention distribution histograms (`attention_distribution_*.png`) for ABMIL vs CLAM-SB;
* heatmap + `_overlay.png` pairs for 3–5 test slides with `pred` vs `true` in the title;
* top-12 patch contact sheets (only if patch paths exist — see §8);
* the ordinal-baseline vs MIL QWK comparison table.

---

## 8. Step 5 — patch CNN (the honest version)

What the code does: `src/models/patch_cnn.py::PatchCNN` is a 4-conv-layer classifier
(`num_classes=2` benign/malignant, or 6 for ISUP), with `PatchDataset` reading a CSV of
`patch_path,label` rows, a `train_patch_cnn(...)` loop and `predict_patch_grid(...)` that turns
per-patch malignant probabilities into a slide-level grid.

What you **cannot** do with the current data: PANDA has no *patch-level* labels, and the public
feature release ships no tile images at all (`panda/patches/{id}.h5` = coords only). So a
patch CNN trained on real PANDA tiles requires downloading the 347 GB Kaggle `.tiff` set and
tiling it with `src/preprocessing.py` — that is Person 1's patch pipeline, and it is a
multi-hour, multi-hundred-GB job on a laptop.

Three defensible paths, in order of effort:

1. **Weakly supervised proxy (recommended, no new download).** Use CLAM-SB's instance clustering
   pseudo-labels as patch labels: for each slide take the top-B / bottom-B attention patches as
   malignant / benign, write `data/patch_labels/train.csv` (`patch_path,label`), train the CNN
   on those. You still need patch PNGs for this (below), but the *labels* are free.
2. **Mock patching (plumbing only).** `scripts/make_mock_features.py` writes real PNG tiles +
   `data/mock/patches/<id>/*.png`, so you can build and validate the whole
   `train_patch_cnn → predict_patch_grid → heatmap` chain today on synthetic tiles. Results are
   non-reportable, but the *code path* is proven.
3. **Full PANDA path (expensive).** Kaggle images → `python -m src.preprocessing --slide … --out data/patches/<image_id>`
   per slide (Person 1's script, ~1–3 min/slide CPU) → labels from CLAM-SB attention → train.
   Realistically: do this only for a subset (say 20 slides) and state it.

The "patch CNN probability map vs MIL attention map" comparison figure is the deliverable —
report it as a *proxy* comparison, not as patch-level AUC, unless you built path 3 with real
labels.

Minimal training call once you have a label CSV and tiles:

```powershell
python -c "from src.models.patch_cnn import train_patch_cnn; print(train_patch_cnn('data/patch_labels/train.csv','data/patch_labels/val.csv', out_path='results/metrics/patch_cnn.pt', epochs=5))"
```

---

## 9. Step 6 — demo app, tests, git

```powershell
streamlit run app/streamlit_app.py         # opens http://localhost:8501
```

In the sidebar point it at `data/features` (and `data/thumbnails`, `results/heatmaps/...` if it
asks). It shows the slide selector, prediction + class-confidence bars, thumbnail vs attention
overlay and the top-5 suspicious patches. Top-5 patches need `patch_paths` inside the `.pt`
files — the public release has none, so expect that panel to stay empty unless you tile yourself.

Tests (run before every commit — you own `src/evaluate.py`):

```powershell
python -m unittest discover -s tests -v          # 21 tests, ~25 s, CPU only
```

Git — you are Person 4, so stay inside your files and branch:

```powershell
git checkout -b person4/localization-stats
git add src/export_attention.py scripts/download_features_and_coords.py scripts/run_person4_experiments.sh PERSON4_RUNBOOK.md
git commit -m "feat(person4): model-agnostic attention export, feature+coord download, Person 4 pipeline script, runbook"
git push -u origin person4/localization-stats
```

Do **not** commit `data/features/` (600 MB), `results/heatmaps/` images or checkpoints —
`.gitignore` already excludes `*.pt`; add `results/metrics/**/checkpoints/` if git ever offers it.
Commit only the small CSVs/JSONs/MD tables that the report cites.

### When to actually use Colab

Only if you want a second machine. Then:

```python
from google.colab import drive; drive.mount('/content/drive')
!git clone https://github.com/koushikkkkkkkkkk/prostate-mil-project.git /content/prostate-mil-project
%cd /content/prostate-mil-project
!ln -s /content/drive/MyDrive/panda/features data/features     # reuse one feature folder
```

…and run the same `python -m src.…` commands. Never train on GPU in Colab and evaluate on a
different feature folder — mismatched feature dirs are the fastest way to invalidate a
comparison (`docs/DATA_CONTRACT.md` is explicit about this).

---

## 10. Troubleshooting table (things that will actually happen)

| error | cause | fix |
| --- | --- | --- |
| `ValueError: Slide … has no coords` | used `scripts/download_real_features.py` (features only) | `python scripts/download_features_and_coords.py` |
| `Need overlapping slide ids in the attention CSV for both train and test splits` | attention CSV built with `--split test` | rebuild with `--split all` |
| `No module named src` | wrong cwd | `cd` to the repo root; always `python -m src.<x>` |
| `torch.cuda.OutOfMemoryError` | batch too large for 8 GB | `--batch-size 8`, and `--max-patches 2048` |
| `sm_120 is not compatible…` | torch < 2.7 / cu126 wheel | reinstall torch from the `cu128` index |
| heatmap is a single flat colour | attention nearly uniform on a lightly-trained model | normal; check `attention_distribution_*.png`, and use a trained checkpoint |
| `evaluate_localisation` raises on a mask | mask size ≠ slide dimension → coords scaled against the wrong grid | pass `slide_dimensions=(W, H)`, or resize the mask to the slide's tiling grid |
| `localisation_scores.csv` missing | `--masks` not passed / no matching files | check `data/masks/{image_id}.png` naming |
| QWK `nan` in eval | one test subset with a single class (only happens on tiny subsets) | expected on mock/small subsets; use the full 60-slide test split |
| streamlit "port in use" | old process alive | `streamlit run app/streamlit_app.py --server.port 8502` |
| `openslide` build error on Windows | no native lib | ignore, not used |

---

## 11. Definition of done for Person 4

- [ ] `data/features/*.pt` contain `features` **and** `coords`; coverage % recorded.
- [ ] ABMIL + CLAM-SB checkpoints exist for seeds 0,1,2 on the shared split.
- [ ] `results/metrics/attention/{abmil,clam_sb}/attention_*.csv` produced with `--split all`.
- [ ] `results/metrics/attention/attention_stats_abmil_vs_clam.md` — sharpness table (entropy, top-10 share).
- [ ] `results/metrics/statistical/{abmil,clam_sb}/statistical_baselines.json` — ordinal baseline QWK (+XGBoost if installed).
- [ ] Heatmaps + overlays + top-k tables for the 60 test slides, both models.
- [ ] `localisation_scores.csv` **or** an explicit "masks unavailable, qualitative only" statement.
- [ ] Patch CNN: either a proxy experiment with clearly labelled weak labels, or a stated scope limitation.
- [ ] `notebooks/06_localization_and_stats.ipynb` re-run end to end, outputs committed (small only).
- [ ] `src/evaluate.py` untouched by others since your last test run; 21 tests green.
- [ ] Every table in the report cites split file + feature dir + checkpoint + seed, and no mock number appears.

---

## 12. Suggested two-day plan

| when | what | time |
| --- | --- | --- |
| Day 1 AM | §3 env + cu128 torch + 21 tests + `run_mock_demo.sh` smoke test | 45 min |
| Day 1 midday | §4 `download_features_and_coords.py` (all 420) while you read `src/evaluate.py` | 15 min active |
| Day 1 PM | §6 pipeline on real features with `TRAIN=1` (1 seed first) | 1 h |
| Day 1 evening | §7 notebook re-run on real numbers; capture 6 figures | 1 h |
| Day 2 AM | masks decision (§4.3) → either Kaggle masks download or option B; re-run with `--masks` | 0.5–3 h |
| Day 2 PM | §8 patch CNN path 1 or 2; §9 app + tests + commit + report tables | 2 h |

---

## 13. Files I added/changed in this workspace (all verified to run)

`repo/` in this workspace is the cloned project plus these additions:

| file | what |
| --- | --- |
| `scripts/download_features_and_coords.py` | downloads `panda/r50/pt_files/{id}.pt` **+** `panda/patches/{id}.h5`, verifies `N` matches, merges into `{"features", "coords"}`. Tested on 24 slides: 24/24 merged, `N` matched on all of them. |
| `src/export_attention.py` | model-agnostic attention export (`--split all`, `--render-images`, `--top-k`), writes the 10-column `x_stat` CSV that `src/statistical_models.py` expects, plus distribution plot / heatmaps / `.npz`. Tested on an ABMIL checkpoint → CSV → ordinal baseline ran. |
| `scripts/run_person4_experiments.sh` | the end-to-end Person 4 pipeline (§6). Tested on a 24-slide real-feature subset with `TRAIN=1` and `TRAIN=0`. |
| `PERSON4_RUNBOOK.md` | this document. |

Nothing in Person 1/2/3's files was modified, and `python -m unittest discover -s tests`
still passes 21/21 after the additions.

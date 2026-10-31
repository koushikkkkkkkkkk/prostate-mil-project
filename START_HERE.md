# START HERE

You have the code. This page is the shortest path from "zip on my laptop" to
"tests green + a full pipeline run". Follow it in order — steps 1-3 take about
5 minutes, step 4 takes about a minute of compute.

---

## 0. What is in this folder

A complete, working MIL codebase for the PANDA ISUP grading project:

* `src/` — models (ABMIL, CLAM-SB, TransMIL, patch CNN), data loaders, trainer,
  metrics, heatmaps, CLAM-SB ablation/eval/export/compare scripts
* `notebooks/01…06` — EDA, patching, ABMIL, CLAM-SB, TransMIL, localisation
* `configs/` — YAML hyperparameters (real + tiny "mock" variants)
* `scripts/` — split creation, mock-data generator, one-command pipelines
* `tests/` — 21 unit + end-to-end tests
* `app/streamlit_app.py` — demo UI
* `docs/` — data contract, results log
* `PERSON2_README.md` — Person 2 (CLAM-SB) handbook

**Not included on purpose:** `data/mock/` (synthetic slides, 15 MB) and
`results/mock_*` (their outputs). They are generated in step 4 — they are
gitignored and are never reportable anyway.

---

## 1. Unzip and enter the project

```bash
unzip prostate-mil-project.zip
cd prostate-mil-project
```

## 2. Create an environment and install dependencies

**Python version matters.** Use **3.10 – 3.13**. Verified combination: Python 3.13,
torch 2.15 (CPU), numpy 2.3, pandas 2.2, scikit-learn 1.6, matplotlib 3.10.

If you are on Python 3.14, keep numpy **>= 2.0**. numpy 1.x does not work on 3.14 and
fails at import with `OverflowError: cannot convert longdouble infinity to integer`.

```bash
python -m venv .venv
source .venv/bin/activate            # Windows PowerShell: .venv\Scripts\Activate.ps1
                                     # Windows cmd:        .venv\Scripts\activate.bat

python -m pip install --upgrade pip
pip install -r requirements.txt
```

On a CPU-only laptop the PyTorch install is the slow part. Smaller route:

```bash
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements.txt
```

Optional extras (Streamlit demo, notebooks, real WSI feature extraction):

```bash
pip install -r requirements-full.txt
```

Always confirm the environment before going further:

```bash
python scripts/check_env.py
```

It prints every version, flags broken combinations (numpy 1.x on Python 3.13+, missing
venv, mismatched torch/torchvision) and prints the exact command that fixes each one.

## 3. Run the tests (30 seconds, no data needed)

```bash
python -m unittest discover -s tests -v
```

Expect `Ran 21 tests ... OK`. If anything fails, stop and fix the environment
before going further — everything downstream depends on these passing.

## 4. Run the full pipeline on synthetic data (~1 minute)

```bash
bash scripts/run_mock_demo.sh          # macOS / Linux / Git Bash / WSL
powershell -ExecutionPolicy Bypass -File scripts\run_mock_demo.ps1   # Windows PowerShell
```

This generates synthetic slides, trains ABMIL and CLAM-SB (with and without the
instance loss), evaluates, exports heatmaps with localisation overlap scores,
and writes the paired comparison.

You should end with:

```
MOCK DEMO COMPLETE (synthetic -- not reportable)
  ablation table : results/mock_clam_sb/ablation_clam_sb.md
  comparison     : results/mock_comparison/comparison.md
  heatmaps       : results/mock_heatmaps/images/
```

Open `results/mock_clam_sb/ablation_clam_sb.md` and one image in
`results/mock_heatmaps/images/` to confirm the plumbing works end to end.

> ⚠️ Every number from step 4 is synthetic. It proves the code runs; it says
> nothing about PANDA performance. Never put it in the report.

## 5. Optional: look at the demo UI

```bash
streamlit run app/streamlit_app.py
```

It will auto-detect the mock checkpoint from step 4 and show a slide, the
predicted grade, a confidence chart, the attention overlay and the top patches.

## 6. Push this to your GitHub repo

The repo you sent me (`koushikkkkkkkkkk/prostate-mil-project`) still contains the
empty scaffold. Replace it with this code:

```bash
git init -b main          # already initialised if a .git folder came with the zip
git add -A
git commit -m "feat: full MIL pipeline (ABMIL, CLAM-SB + ablation, TransMIL, metrics, demo)"
git remote add origin https://github.com/koushikkkkkkkkkk/prostate-mil-project.git
git push -u origin main --force
```

`--force` is only needed because the remote currently holds the scaffold commit;
if you would rather keep history, use `git pull --rebase origin main` first.

Note: `data/mock/` and `results/mock_*` are gitignored, so the push stays small.

## 6b. Real PANDA data path (once Person 1 has published the split)

```bash
git pull origin main                                   # gets data/common_split.csv + ABMIL predictions
python scripts/download_real_features.py --manifest data/common_split.csv --out-dir data/features --workers 16
pip install h5py
python scripts/attach_patch_coords.py                 # REAL patch coords -> real heatmaps
python scripts/attach_patch_coords.py --verify-only    # expect 399/420 with coords
```

Then run the experiments (bash or Windows PowerShell):

```bash
bash scripts/run_person2_experiments.sh
powershell -ExecutionPolicy Bypass -File scripts\run_person2_experiments.ps1
```

The runner refuses to start unless all 420 feature files (300/60/60) are present —
a silently reduced test split would invalidate every metric.

## 7. What comes next (the blocking dependency)

Everything real needs **Person 1's outputs**:

| needed | from | used for |
| --- | --- | --- |
| `data/common_split.csv` (300/60/60) | Person 1 | the frozen split for every model |
| `data/features/{image_id}.pt` — `features [N,D]`, `coords [N,2]` | Person 1 | training + heatmaps |
| `data/thumbnails/{image_id}.png` (optional) | Person 1 | overlay heatmaps, demo |
| ABMIL predictions CSV | Person 1 | the paired CLAM-SB comparison |

Send them this:

> I've got the full pipeline running on synthetic data. What I need from you is
> `data/common_split.csv`, one `data/features/{image_id}.pt` per slide containing
> `features` as `[N, D]` and `coords` as `[N, 2]`, the value of `D`, and (if easy)
> slide thumbnails in `data/thumbnails/`. Once those exist I swap three arguments
> and rerun the real experiments.

When they arrive:

```bash
# 1. split (if Person 1 has not committed common_split.csv yet)
python scripts/create_split.py --panda-csv data/raw/train.csv --out data/common_split.csv

# 2. CLAM-SB ablation + evaluation + heatmaps + comparison
bash scripts/run_person2_experiments.sh

# 3. ABMIL baseline, if Person 1 has not produced predictions yet
python -m src.train --model abmil --config configs/abmil.yaml --out-dir results/metrics/abmil
```

Then fill in `docs/RESULTS_LOG.md` with the real numbers.

---

## Troubleshooting

| symptom | fix |
| --- | --- |
| `OverflowError: cannot convert longdouble infinity to integer` at numpy import | numpy 1.x on Python 3.13+. `pip install --upgrade "numpy>=2.0"`, or rebuild the venv on Python 3.12/3.13 |
| `ImportError: Failed to import test module` | usually a package import blowing up — run `python scripts/check_env.py` to see which one |
| `ModuleNotFoundError: No module named 'torch'` | step 2 — packages live in the venv, activate it in every new shell |
| `torch` install is huge / times out | use the CPU index URL shown in step 2 |
| `FileNotFoundError: data/common_split.csv` | you ran a real-data command before Person 1 delivered; use the mock demo |
| `bash: scripts/run_mock_demo.sh: No such file` | wrong directory; on Windows use `scripts\run_mock_demo.ps1` or copy the commands out of the `.sh` |
| `run_mock_demo.ps1 cannot be loaded because running scripts is disabled` | `powershell -ExecutionPolicy Bypass -File scripts\run_mock_demo.ps1` |
| `ModuleNotFoundError: yaml` | `pip install pyyaml` |
| matplotlib errors on a headless machine | prefix commands with `MPLBACKEND=Agg` |
| `streamlit: command not found` | `pip install -r requirements-full.txt` (optional) |
| heatmap export warns about missing coords | the `.pt` file has no `coords`; re-export features with `src/feature_extraction.py` |

## The one rule

Synthetic numbers are for debugging the pipeline. Reportable numbers come only
from `data/common_split.csv` + real `data/features` + 3 seeds, written into
`docs/RESULTS_LOG.md`.

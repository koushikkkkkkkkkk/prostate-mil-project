# Closing the audit: what one person actually has to do

**Status: 7 of the 9 remediation items are finished and are sitting in a git bundle of eight commits.**
Two items cannot be finished from inside the repository, because they need files that
only exist on your Windows machine:

1. **P0-1** — the CLAM-SB / TransMIL weight files were never committed.
2. **P0-3** — the mandatory instance-loss ablation has no committed artifacts, so the
   two numbers in the report (0.6961 vs 0.7388) are currently unverifiable.

Everything else is done. Total remaining work: **about two hours, one person, one machine.**

| # | item | state | where |
| --- | --- | --- | --- |
| P0-1 | track the missing seed checkpoints | **you** — 30 min | Part B |
| P0-2 | fix the sample app | **done** | commit `d004422`, `app/streamlit_app.py` |
| P0-3 | run + commit the CLAM-SB ablation | **you** — 90 min | Part C |
| P1-4 | generated results table, no hand-typed numbers | **done** | `results/ALL_RESULTS.md`, `docs/RESULTS_LOG.md` |
| P1-5 | lead with the paired table | **done** | `results/metrics/comparison/paired_*` |
| P1-6 | quarantine the toy patch-CNN | **done** | `results/metrics/patch_cnn_demo/NOT_REPORTABLE.md` |
| P2-7 | Colab setup cells + provenance guard | **done** | notebooks 03/04/05/06 |
| P2-8 | document the data + `verify_data.py` | **done** | README §3-BIS, `scripts/verify_data.py` |
| P2-9 | delete duplicate script, "54 tests", CI | **done** | `.github/workflows/tests.yml`, `scripts/README.md` |

Do the parts in order. A → B → C → D → E. Do not skip Part D; it is ten minutes and it is
what proves the job is finished.

---

## Part A — get the work onto your machine (5 minutes, PowerShell)

The work is in `prostate-mil-updates.bundle`: eight commits on top of `main` (`8b08606`),
branch `archit-p0-p2`. A bundle is a single file that behaves exactly like a remote.

```powershell
cd D:\College\IBD\prostate-mil-project\prostate-mil-project

# 1. make sure you have nothing half-finished that would block a merge
git status

# 2. main must be at 8b08606 (the bundle's prerequisite)
git switch main
git pull --ff-only
git log --oneline -1          # expect: 8b08606 ...

# 3. check the bundle is intact, then fetch the branch out of it
git bundle verify C:\Users\<you>\Downloads\prostate-mil-updates.bundle
git fetch C:\Users\<you>\Downloads\prostate-mil-updates.bundle archit-p0-p2:archit-p0-p2

# 4. look at what you got before merging anything
git log --oneline main..archit-p0-p2      # expect exactly 8 lines
git diff --stat main..archit-p0-p2 | Select-Object -Last 1
```

You should see these eight commits:

```
<newest> docs: add the one-person action plan for the two items that need local files
f89461b feat(audit): scripts/audit_artifacts.py + a report-only CI job
e8571e2 ci: add a test workflow and make the tree lint-clean
e6a9513 chore(scripts): retire the duplicate coords script, document every script
1389bb7 feat(colab): self-contained setup for notebooks 04 and 06 + provenance guard in all four trainers
d004422 docs: make every document agree with the committed artifacts
789ea99 feat(results): generate results/ALL_RESULTS.md from artifacts, quarantine the toy patch CNN
056d862 feat(results): regenerate every paired table from the committed predictions
```

Now bring it into your branch. If `Archit` has no unmerged local work, the cleanest route
is to fast-forward `main` over it and push:

```powershell
git switch main
git merge --ff-only archit-p0-p2
git push origin main
```

If you would rather keep `main` untouched and open a pull request:

```powershell
git push origin archit-p0-p2
# then on GitHub: "Compare & pull request" -> base main, compare archit-p0-p2
```

**Sanity check after the merge, before you do anything else:**

```powershell
python -m unittest discover -s tests          # expect: Ran 54 tests ... OK
python scripts\audit_artifacts.py             # expect: 2 fatal  <- the two P0-1/P0-3 gaps
```

If the test count is not 54, or the audit prints no fatal findings, the merge is wrong.

---

## Part B — P0-1: commit the nine checkpoints (30 minutes)

### B1. Find out what you already have (2 min)

```powershell
# what git currently tracks
(git ls-files | Select-String "_best\.pt").Count      # expect 3 today

# what is actually on this disk, with sizes
Get-ChildItem -Recurse -Filter *_best.pt |
  Select-Object @{n='MB';e={[math]::Round($_.Length/1MB,1)}}, LastWriteTime, FullName |
  Sort-Object FullName | Format-Table -AutoSize
```

A trained checkpoint in this project is about **3 MB** (the committed ABMIL seeds are
3.1 MB each), so even a dozen files cost you ~40 MB of repository — nothing. There is no
LFS traffic, no size problem, no reason they should not be in the repo.

### B2. The files that are missing, and exactly where they must live

The paths are not a matter of taste: two committed files already *point* at them, and the
audit verdict depends on those paths existing.

| file | required path | why |
| --- | --- | --- |
| CLAM-SB seed 0/1/2, **with** instance loss | `results/metrics/clam_sb/with_instance/checkpoints/clam_sb_seed{0,1,2}_best.pt` | `results/heatmaps/clam_sb/export_summary.json` and `results/metrics/attention/clam_sb/export_summary.json` already reference seed 0 — **this is the dangling reference the audit fails on** |
| CLAM-SB seed 0/1/2, **without** instance loss | `results/metrics/clam_sb/without_instance/checkpoints/clam_sb_seed{0,1,2}_best.pt` | the ablation's second arm; without it the 0.7388 row is a number with no weight behind it |
| TransMIL seed 0/1/2 | `results/metrics/transmil/checkpoints/transmil_seed{0,1,2}_best.pt` | 0.7446 is in the generated table with no weights |
| CHIEF / GigaPath runs | `results/metrics/{abmil_chief,clam_sb_chief,abmil_gigap}/checkpoints/…_seed{n}_best.pt` | 0.91–0.93 QWK, the headline result, currently unreproducible |

Today the repository tracks **three** `_best.pt` files — ABMIL seeds 0/1/2. The six
CLAM-SB arm checkpoints bring the count to exactly **9**, which is the acceptance bar
(`git ls-files | grep -c "_best.pt"` ≥ 9). Commit the TransMIL and CHIEF/GigaPath ones too
if you have them: they push it well past 9 and they are the numbers a reviewer will probe.

Search for them anywhere on the machine, not just in `results\` — they may be on the
Colab Drive copy or an old run folder:

```powershell
Get-ChildItem D:\ -Recurse -Filter "clam_sb_seed*_best.pt" -ErrorAction SilentlyContinue |
  Select-Object MB = @{n='MB';e={[math]::Round($_.Length/1MB,1)}}, FullName, LastWriteTime
```

### B3. Put them in place and commit

`results/**/*.pt` is already whitelisted in `.gitignore` (lines 32–34), so these files are
committable as-is. Adjust the source paths to whatever B2 found.

```powershell
New-Item -ItemType Directory -Force -Path `
  results\metrics\clam_sb\with_instance\checkpoints, `
  results\metrics\clam_sb\without_instance\checkpoints, `
  results\metrics\transmil\checkpoints | Out-Null

Copy-Item "<wherever>\clam_sb_seed0_best.pt" results\metrics\clam_sb\with_instance\checkpoints\
Copy-Item "<wherever>\clam_sb_seed1_best.pt" results\metrics\clam_sb\with_instance\checkpoints\
Copy-Item "<wherever>\clam_sb_seed2_best.pt" results\metrics\clam_sb\with_instance\checkpoints\
Copy-Item "<wherever>\clam_sb_noinst_seed0_best.pt" results\metrics\clam_sb\without_instance\checkpoints\clam_sb_seed0_best.pt
# ... and so on for the rest

git add results/metrics/clam_sb results/metrics/transmil results/metrics/abmil_chief results/metrics/clam_sb_chief results/metrics/abmil_gigap
git status --short                     # every added line should say "A  ..._best.pt"
git commit -m "results(P0-1): commit the trained seed checkpoints

The CLAM-SB, TransMIL, CHIEF and GigaPath metrics were committed without the
weights they came from, so none of them could be re-evaluated and the two
committed heatmap exports pointed at checkpoints that did not exist. Adds the
seed-0/1/2 best checkpoints for each run (~3 MB each)."
```

**Acceptance test** — run all three, all three must pass:

```powershell
(git ls-files | Select-String "_best\.pt").Count     # >= 9
python scripts\audit_artifacts.py                    # 0 fatal
git lfs ls-files                                     # empty (you are not using LFS)
```

### B4. If the files are genuinely gone

Then they have to be retrained, and that is a GPU job, not a documentation job. Run
`scripts\run_person2_experiments.ps1` (Part C) — it retrains all six CLAM-SB arms and
writes the checkpoints into exactly the paths above. Do Part C first, then come back to
B3 with the files it produced.

Skip nothing here. An unverifiable 0.93 is worth less than a verified 0.70, and a reviewer
who asks "show me the weights" is not being awkward — it is the first question in the
checklist for any applied-ML submission.

---

## Part C — P0-3: the instance-loss ablation (90 minutes)

This is the deliverable the brief actually asked Person 2 for, and it is the one item with
no artifacts at all. Two documents currently disagree about its result (0.6961 vs 0.7388,
and 0.7031 in `report/person4_results.md`), which is exactly why it has to be re-run
rather than reconciled on paper.

### C1. Check the inputs before spending an hour of GPU time (2 min)

```powershell
.\.venv\Scripts\Activate.ps1
python scripts\verify_data.py --require all
```

Expect every cohort **OK**, dims 1024 / 768 / 1536, and `missing 0`. If it exits 1, fix
that first (`README.md` §3-BIS has the three download commands). Every training script in
this repo refuses to run on a reduced split, so a partial data folder will just stop you
later with a worse error message.

### C2. Run the whole Person 2 pipeline (one command)

```powershell
$env:SEEDS = "0 1 2"
powershell -ExecutionPolicy Bypass -File scripts\run_person2_experiments.ps1
```

It does six things in order: coverage check → train both arms × 3 seeds → evaluate seed 0
→ export heatmaps → statistical baseline → paired ABMIL-vs-CLAM-SB comparison → print the
ablation table. Budget **20–40 minutes** on the RTX 5060: the only comparable figure in the
repo is 2–6 minutes per 3-seed ABMIL run (`report/PERSON4_RUNBOOK.md:53`), and CLAM-SB
trains a second (instance) head at every step.

Two environment variables are worth setting if your data is not in the default location:

```powershell
$env:MANIFEST = "data\common_split.csv"
$env:FEATURES = "data\features"
$env:OUT      = "results\metrics\clam_sb"
```

### C3. Read the artifacts it produced

```powershell
Get-ChildItem results\metrics\clam_sb -Recurse -File | Select-Object FullName, Length
```

You are looking for, and should find:

- `results\metrics\clam_sb\with_instance\aggregate_clam_sb.json` and `predictions_clam_sb_seed{0,1,2}.csv`
- `results\metrics\clam_sb\without_instance\aggregate_clam_sb.json` and predictions
- `results\metrics\clam_sb\ablation_clam_sb.md` ← the with-vs-without table
- `results\metrics\clam_sb\with_instance\checkpoints\clam_sb_seed{0,1,2}_best.pt` ← feeds Part B3
- `results\heatmaps\clam_sb\…` ← makes the dangling reference in B2 real

Then regenerate the derived documents **from these artifacts**, never by typing numbers:

```powershell
python scripts\summarize_results.py        # -> results/ALL_RESULTS.md
python scripts\paired_results.py           # -> results/metrics/comparison/paired_*
python scripts\summarize_person2.py        # -> results/PERSON2_SUMMARY.md
python scripts\audit_artifacts.py          # 0 fatal, and the ablation warning is gone
```

### C4. Update the three places that currently say "unverified"

```powershell
Select-String -Path docs\*.md,report\*.md,presentation_guide_and_team_handbook.md -Pattern "unverified"
```

Three edits, each a two-minute copy from `results/ALL_RESULTS.md`:

1. **`report/person2_section.md` §4** — replace the flagged table with the real
   with/without rows, the real ΔQWK, and the real per-seed spread.
2. **`docs/MODEL_COMPENDIUM.md`** and **`docs/PRESENTATION_GUIDE.md`** — drop the
   ⚠️ markers, paste the same numbers.
3. **`docs/RESULTS_LOG.md`** — the ablation block goes from "artifacts missing" to the
   generated table.

Then commit:

```powershell
git add results/metrics/clam_sb results/metrics/statistical results/heatmaps/clam_sb results/ALL_RESULTS.md results/metrics/comparison results/PERSON2_SUMMARY.md docs report presentation_guide_and_team_handbook.md
git commit -m "results(P0-3): rerun and commit the CLAM-SB instance-loss ablation

with_instance/ and without_instance/ now exist for seeds 0/1/2, so the ablation
table is generated rather than quoted. Docs regenerated from the artifacts."
```

### C5. Whichever way it comes out, you win — if you write it honestly

**If ΔQWK (with − without) ≤ 0:** that is a *negative result about the loss*, not a failed
deliverable. Write exactly this: *"Adding the instance-level clustering loss did not improve
slide-level QWK on this cohort at λ = 0.30. The 95 % CI spans zero, so we cannot claim a
benefit; with 60 test slides the study is only powered to detect differences above ≈0.05.
The loss's real purpose is localisation, which we demonstrate separately in §8."*

**If ΔQWK > 0:** do not over-claim either. Say the point estimate improved by X with a CI
of [a, b], and check whether a single seed drives it before calling it a win.

Do **not** run the ablation twice and keep the better number. That is the one thing that
would make every other number in the repository untrustworthy.

---

## Part D — prove all nine items are done (10 minutes)

Run this block top to bottom. Every line has a stated expected output; anything else means
the item is not actually closed.

```powershell
# P0-1  nine tracked weights, no dangling references
(git ls-files | Select-String "_best\.pt").Count                 # >= 9 (3 + the 6 ablation arms)
python scripts\audit_artifacts.py                                 # 0 fatal

# P0-2  the app loads a real checkpoint and predicts a real grade
python -m unittest tests.test_app_logic                           # Ran 22 tests ... OK
streamlit run app\streamlit_app.py                                # pick a slide from common_split.csv

# P0-3  the ablation has artifacts
Test-Path results\metrics\clam_sb\with_instance\aggregate_clam_sb.json      # True
Test-Path results\metrics\clam_sb\without_instance\aggregate_clam_sb.json   # True
Select-String -Path docs\*.md,report\*.md -Pattern "unverified"             # no output

# P1-4 / P1-5  the tables are generated, and they are current
python scripts\summarize_results.py; python scripts\paired_results.py
git status --short                                                # no modified generated files

# P1-6  the toy CNN is quarantined
Test-Path results\metrics\patch_cnn_demo\NOT_REPORTABLE.md         # True
Select-String -Path README.md -Pattern "patch_cnn"                 # at most a "demo, not reportable" line

# P2-7  every trainer guards against mock data
Select-String -Path notebooks\*.ipynb -Pattern "REFUSING TO TRAIN" | Measure-Object | % Count   # 4

# P2-8  the data is documented and checkable
Select-String -Path README.md -Pattern "3-BIS"                     # 1
python scripts\verify_data.py --require all                        # exit 0

# P2-9  tests, lint and CI
python -m unittest discover -s tests                               # Ran 54 tests ... OK
python -m pyflakes src scripts app tests                           # no output (pip install pyflakes)
Test-Path .github\workflows\tests.yml                              # True
```

If all of that passes, the audit is closed and you can print the repository URL in the
report without a caveat.

---

## Part E — the one paragraph to say out loud

> "Our strongest result is a QWK of 0.93 with CLAM-SB on CHIEF features, but the thing we
> actually established is that the **encoder** decides the outcome: moving from ImageNet
> ResNet50 to a pathology foundation encoder is worth +0.19 QWK with a McNemar p of 0.004,
> while swapping the aggregator — ABMIL, CLAM-SB, TransMIL — moves the number by less than
> 0.02 with a confidence interval that spans zero. The attention maps are an
> interpretability cue on 56 of 60 slides, not a segmentation; the masks contain no
> Gleason pixels at all, which is why patch-level supervision was impossible. This is a
> research prototype on a PANDA subset, and the ISUP imbalance means accuracy understates
> performance, so QWK is the metric of record."

---

## Appendix — retraining everything from scratch

Only needed if the weights are gone (B4). On the RTX 5060, after
`python scripts\verify_data.py --require all` passes:

```powershell
# Person 1 baselines
python -m src.train --model abmil --manifest data/common_split.csv --features-root data/features --config configs/abmil.yaml --out-dir results/metrics/abmil --seeds 0 1 2
python -m src.train --model transmil --manifest data/common_split.csv --features-root data/features --config configs/transmil.yaml --out-dir results/metrics/transmil --seeds 0 1 2

# Person 2 ablation (this also writes the six CLAM-SB checkpoints)
powershell -ExecutionPolicy Bypass -File scripts\run_person2_experiments.ps1

# Foundation-encoder runs (need data/features_chief and data/features_gigap)
python -m src.train --model abmil --manifest data/common_split_chief.csv --features-root data/features_chief --config configs/abmil_chief.yaml --out-dir results/metrics/abmil_chief --seeds 0 1 2
python -m src.train --model clam_sb --manifest data/common_split_chief.csv --features-root data/features_chief --config configs/clam_sb_chief.yaml --out-dir results/metrics/clam_sb_chief --seeds 0 1 2
python -m src.train --model abmil --manifest data/common_split_gigap.csv --features-root data/features_gigap --config configs/abmil_gigap.yaml --out-dir results/metrics/abmil_gigap --seeds 0 1 2

# then regenerate every table
python scripts\summarize_results.py
python scripts\paired_results.py
python scripts\audit_artifacts.py
```

If a config file listed above does not exist for a run, that run was made with a
hand-written config and the config has to be reconstructed from its
`results/metrics/<run>/run_config.json` — which is itself evidence for why every config is
committed next to its run.

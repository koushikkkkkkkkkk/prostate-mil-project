# NOT REPORTABLE — toy patch-CNN demo

`patch_cnn.pt` and `patch_cnn_summary.json` in this folder are a **plumbing
demonstration**, not a project result.

| | this demo | the reportable runs |
| --- | --- | --- |
| data | 988 train / 333 val **tiles** from an undocumented tile set | 420 PANDA slides, locked split |
| epochs | 3 | up to 50 with early stopping on validation QWK |
| task | binary benign vs malignant per tile | 6-class ISUP grade per slide |
| metric | tile accuracy 83.78 % | slide-level QWK with paired CIs |

Why the real thing cannot be trained on this dataset: the PANDA feature release
provides **slide-level labels only**, and the Karolinska tissue masks shipped with it
contain 99.88 % background / 0.12 % stroma / **zero Gleason pixels** across the 24
slides audited by `scripts/check_mask_encoding.py`. There is no patch-level ground
truth to fit, and no patch images in the feature-only release.

Keep this folder as evidence that the code path works end to end. Do not put the
83.78 % next to a QWK, and do not cite it in the report.

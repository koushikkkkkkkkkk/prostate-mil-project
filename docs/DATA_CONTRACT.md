# Data contract

Everything in this repository depends on three files. Agree on them once and no
component has to change again.

## 1. `data/common_split.csv` — the scientific lock (Person 1)

| column | type | example | notes |
| --- | --- | --- | --- |
| `image_id` | str | `0005f7aaab28080e` | PANDA slide id; also the feature-file stem |
| `data_provider` | str | `karolinska` / `radboud` | used for the shift discussion |
| `isup_grade` | int 0-5 | `2` | the only label used in training |
| `gleason_score` | str | `3+4` | metadata only, never a training target |
| `split` | str | `train` / `val` / `test` | 300 / 60 / 60 slides, stratified, seed 42 |

Created by:

```bash
python scripts/create_split.py --panda-csv data/raw/train.csv --out data/common_split.csv
```

Rules:

* `test` slides are never used for training, early stopping or hyperparameter choice.
* The file is committed to git (it is small text) so every teammate and the marker can
  reproduce the exact same evaluation set.
* If a grade cannot reach the per-class target, the shortfall is printed and must be
  reported — never "fixed" by borrowing slides from another grade.

## 2. `data/features/{image_id}.pt` — one bag per slide (Person 1)

Minimum accepted content:

```python
torch.save({"features": torch.tensor(...)}, "data/features/<image_id>.pt")   # [N, D] float32
```

Preferred content (enables heatmaps and top-patch figures):

```python
torch.save(
    {
        "features": features,        # [N, D] float32, N patches, D = 1024 (ResNet50) or 2048
        "coords": coords,            # [N, 2] float32, patch origin (x, y) in slide pixels
        "patch_paths": patch_paths,  # [N] list[str], PNG tiles for the "top suspicious patches" view
    },
    f"data/features/{image_id}.pt",
)
```

Loader behaviour (`src/dataset.py`):

* accepts a bare tensor, a `dict` (`features`/`feats`/`embeddings`, `coords`/`coordinates`,
  `patch_paths`/`paths`), `.npy` or `.npz`;
* `N` may differ per slide (variable bag length is the point);
* patch order is irrelevant — all models are permutation invariant over patches;
* `D` is auto-detected from the first few files; a config `in_dim` that disagrees is
  overridden with a warning.

## 3. Optional assets

| path | used by | why |
| --- | --- | --- |
| `data/thumbnails/{image_id}.png` | heatmap overlays, Streamlit app | alpha-blend the attention map onto something readable |
| `data/masks/{image_id}.png` | localisation overlap score | bright = tumour/tissue; **coarse reference only** |
| `data/patches/{image_id}/coords.csv` | `src.feature_extraction` | `patch_index, x, y, patch_path` |

## Model interface (all four models)

```python
output = model(features, mask=None, labels=None)
output["logits"]     # [B, 6] raw scores
output["attention"]  # [B, N] softmax over real patches, 0 on padding
output["bag_feat"]   # [B, embed_dim] pooled slide embedding
output["instance_loss"]  # scalar, CLAM-SB only
```

`features` can be `[N, D]` (single slide) or `[B, N, D]`; `mask` is a bool `[B, N]` where
`True` marks a real patch. Models accept both and mirror the input rank in their outputs.

## Numbers that must never be compared across contracts

Feature dimension, encoder, patch size and split all change what a QWK means. If Person 1
re-extracts features (new encoder, new D, new tile size), every model must be retrained
before any table is compared again. Record the settings in `docs/RESULTS_LOG.md`.

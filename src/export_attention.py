"""Export per-slide attention statistics for any trained MIL checkpoint.

Why this script exists
----------------------
Notebook 06 needs an ``attention_*.csv`` table to fit the statistical baselines
(ordinal logistic / XGBoost on top of attention statistics). The repo ships
``src.evaluate_clam_sb`` which produces that table **only for CLAM-SB**, using a
private (underscore-prefixed) helper internally. Person 4's notebook is written
to be model-agnostic, so this script provides the same table for *any* model
registered in ``src.models.build_model`` -- in practice the ABMIL baseline
(``results/metrics/abmil/checkpoints/abmil_seed0_best.pt``) and CLAM-SB.

It also optionally renders attention heatmaps for any model (the same renderer
CLAM-SB uses), which is the "does ABMIL attend to the same regions as CLAM-SB?"
figure for the write-up.

Outputs (in ``--out-dir``)::

    attention_<tag>.csv                 per-slide attention statistics (x_stat inputs)
    attention_distribution_<tag>.png    histogram of all attention weights
    images/<slide_id>_attention.png     heatmap (+ _overlay.png when --thumbnails given)
    arrays/<slide_id>.npz              attention, coords, probs, y_true, y_pred
    export_summary.json

Usage::

    python -m src.export_attention --checkpoint results/metrics/abmil/checkpoints/abmil_seed0_best.pt \
        --manifest data/common_split.csv --features-root data/features \
        --out-dir results/metrics/attention/abmil --split train        # <- train+test needed!
    python -m src.export_attention --checkpoint results/metrics/abmil/checkpoints/abmil_seed0_best.pt \
        --manifest data/common_split.csv --features-root data/features \
        --out-dir results/heatmaps/abmil --split test --thumbnails data/thumbnails

IMPORTANT: ``src/statistical_models`` fits on the **train** split attention
statistics and evaluates on **test**, so run this twice (``--split train`` and
``--split test``) and concatenate the two CSVs, or run with ``--split all``,
which is supported here and is the easiest path.

Owner: Person 4 (with thanks to Person 2's CLAM-SB exporter for the layout).
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import numpy as np
import torch

try:
    from src.dataset import FeatureBagDataset, read_manifest, split_records
    from src.export_clam_attention import (  # reuse the CLAM renderer, it is model-agnostic
        _resolve_thumbnail,
        _write_csv,
    )
    from src.heatmaps import (
        attention_entropy,
        make_patch_contact_sheet,
        normalised_attention_entropy,
        plot_attention_distribution,
        render_attention_heatmap,
        top_attention_patches,
        write_patch_table,
    )
    from src.models import build_model
except ImportError:  # pragma: no cover - direct script execution
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from src.dataset import FeatureBagDataset, read_manifest, split_records  # type: ignore
    from src.export_clam_attention import _resolve_thumbnail, _write_csv  # type: ignore
    from src.heatmaps import (  # type: ignore
        attention_entropy,
        make_patch_contact_sheet,
        normalised_attention_entropy,
        plot_attention_distribution,
        render_attention_heatmap,
        top_attention_patches,
        write_patch_table,
    )
    from src.models import build_model  # type: ignore


def torch_load_any(path: str | Path) -> Dict[str, Any]:
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:  # pragma: no cover - torch < 2.0
        return torch.load(path, map_location="cpu")


def load_checkpoint(path: str | Path, device: str = "cpu"):
    payload = torch_load_any(path)
    if "state_dict" not in payload:
        raise ValueError(f"{path} does not look like a training checkpoint")
    model = build_model(payload["model_name"], payload.get("model_config", {}))
    model.load_state_dict(payload["state_dict"])
    model.to(device).eval()
    return model, payload


@torch.no_grad()
def export_attention(
    checkpoint: str | Path,
    manifest: str | Path,
    features_root: str | Path,
    out_dir: str | Path,
    split: str = "all",
    thumbnails: Optional[str | Path] = None,
    max_patches: Optional[int] = None,
    top_k: int = 20,
    render_images: bool = False,
    device: str = "cpu",
) -> Dict[str, Any]:
    model, payload = load_checkpoint(checkpoint, device=device)
    tag = Path(checkpoint).stem.replace("_best", "")
    all_records = read_manifest(manifest)
    records = all_records if split == "all" else split_records(all_records, split)
    if not records:
        raise ValueError(f"split '{split}' is empty in {manifest}")

    dataset = FeatureBagDataset(records, features_root, load_coords=render_images, train=False)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    rows: List[Dict[str, Any]] = []
    all_attention: List[np.ndarray] = []

    for index in range(len(dataset)):
        item = dataset[index]
        record = dataset.records[index]
        features = item["features"].to(device)
        if max_patches and features.size(0) > max_patches:
            features = features[:max_patches]
        output = model(features.unsqueeze(0), use_instance_loss=False)
        attention = output["attention"].squeeze(0).cpu().numpy()
        probs = torch.softmax(output["logits"], dim=-1).squeeze(0).cpu().numpy()
        prediction = int(np.argmax(probs))
        all_attention.append(attention)

        # embedding summary statistics -- the last two x_stat features used by
        # src.statistical_models (computed here so the baseline sees all 10 columns)
        norms = features.detach().float().norm(dim=1).cpu().numpy()

        row: Dict[str, Any] = {
            "slide_id": record.image_id,
            "split": record.split,
            "y_true": record.label,
            "y_pred": prediction,
            "prob_pred": float(probs[prediction]),
            "prob_true": float(probs[record.label]) if 0 <= record.label < probs.size else float("nan"),
            "n_patches": int(attention.size),
            "mean_attention": float(attention.mean()),
            "median_attention": float(np.median(attention)),
            "std_attention": float(attention.std()),
            "max_attention": float(attention.max()),
            "top10_attention_mean": float(np.sort(attention)[-min(10, attention.size):].mean()),
            "attention_entropy": attention_entropy(attention),
            "normalised_entropy": normalised_attention_entropy(attention),
            "top10_attention_share": float(np.sort(attention)[-min(10, attention.size):].sum()),
            "high_attention_ratio": float((attention > 2.0 / attention.size).mean()),
            "embedding_norm_mean": float(norms.mean()),
            "embedding_norm_std": float(norms.std()),
        }

        if render_images:
            if "coords" not in item:
                # public release has no coords for ~5% of slides: skip rendering,
                # keep the attention statistics, never abort the whole export
                print(f"  skipping {record.image_id}: no coords")
                rows.append(row)
                continue
            coords = item["coords"].numpy()
            images_dir = out_dir / "images"
            arrays_dir = out_dir / "arrays"
            patches_dir = out_dir / "top_patches"
            for directory in (images_dir, arrays_dir, patches_dir):
                directory.mkdir(parents=True, exist_ok=True)

            rendered = render_attention_heatmap(
                attention,
                coords,
                images_dir / f"{record.image_id}_attention.png",
                thumbnail=_resolve_thumbnail(Path(thumbnails) if thumbnails else None, record.image_id),
                title=f"{record.image_id} {tag} pred={prediction} true={record.label}",
            )
            row["heatmap"] = rendered["heatmap"]
            row["overlay"] = rendered.get("overlay", "")

            patch_rows = top_attention_patches(attention, coords, item.get("patch_paths"), k=top_k)
            write_patch_table(patch_rows, patches_dir / f"{record.image_id}_top{top_k}.csv")
            row["contact_sheet"] = make_patch_contact_sheet(
                patch_rows, images_dir / f"{record.image_id}_top{top_k}.png"
            ) or ""

            np.savez_compressed(
                arrays_dir / f"{record.image_id}.npz",
                attention=attention,
                coords=coords,
                probs=probs,
                y_true=np.asarray([record.label]),
                y_pred=np.asarray([prediction]),
            )
        rows.append(row)

    _write_csv(out_dir / f"attention_{tag}.csv", rows)
    plot_attention_distribution(
        np.concatenate(all_attention),
        out_dir / f"attention_distribution_{tag}.png",
        title=f"{tag} attention ({split})",
    )

    tag_from_payload = f"{payload.get('model_name', 'model')}_seed{payload.get('seed')}"
    summary = {
        "checkpoint": str(checkpoint),
        "attention_csv": str(out_dir / f"attention_{tag}.csv"),
        "model": payload.get("model_name"),
        "seed": payload.get("seed"),
        "split": split,
        "n_slides": len(rows),
        "mean_normalised_entropy": float(np.mean([r["normalised_entropy"] for r in rows])),
        "mean_top10_attention_share": float(np.mean([r["top10_attention_share"] for r in rows])),
        "out_dir": str(out_dir),
        "rendered_heatmaps": bool(render_images),
    }
    (out_dir / "export_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Export attention statistics/heatmaps for any MIL checkpoint.")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--manifest", default="data/common_split.csv")
    parser.add_argument("--features-root", default="data/features")
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--split", default="all", choices=["train", "val", "test", "all"])
    parser.add_argument("--thumbnails", default=None)
    parser.add_argument("--max-patches", type=int, default=None)
    parser.add_argument("--top-k", type=int, default=20)
    parser.add_argument("--render-images", action="store_true", help="also render heatmaps (needs coords)")
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args(argv)

    summary = export_attention(
        checkpoint=args.checkpoint,
        manifest=args.manifest,
        features_root=args.features_root,
        out_dir=args.out_dir,
        split=args.split,
        thumbnails=args.thumbnails,
        max_patches=args.max_patches,
        top_k=args.top_k,
        render_images=args.render_images,
        device=args.device,
    )
    print(f"exported {summary['n_slides']} slides -> {summary['out_dir']}")
    print(f"  attention csv : {Path(args.out_dir) / ('attention_' + summary['model'] + '_seed' + str(summary['seed']) + '.csv')}")
    print(f"  mean normalised entropy = {summary['mean_normalised_entropy']:.4f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

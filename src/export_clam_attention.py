"""Export CLAM-SB attention heatmaps for the test slides -- Person 2.

For every slide of a split this writes:

    images/<slide_id>_attention.png          raw attention heatmap
    images/<slide_id>_overlay.png            heatmap alpha-blended on thumbnail
    top_patches/<slide_id>_top20.csv         highest-attention patches
    arrays/<slide_id>.npz                    attention + coords + patch paths
    heatmap_manifest.csv                     index of everything produced
    localisation_scores.csv                  (optional) attention-vs-mask overlap

If a thumbnail for a slide exists (``--thumbnails``) the overlay is rendered on
top of it; the heatmap itself only needs ``coords``. ``--masks`` enables the
overlap score, which is the quantitative stand-in for "the heatmap looks
plausible" -- see the caveat in ``src/heatmaps.py``.

Usage::

    python -m src.export_clam_attention \
        --checkpoint results/metrics/clam_sb/with_instance/checkpoints/clam_sb_seed0_best.pt \
        --manifest data/common_split.csv --features-root data/features \
        --thumbnails data/thumbnails --out-dir results/heatmaps/clam_sb_seed0
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
    from src.evaluate_clam_sb import load_checkpoint
    from src.heatmaps import (
        attention_entropy,
        evaluate_localisation,
        make_patch_contact_sheet,
        normalised_attention_entropy,
        render_attention_heatmap,
        top_attention_patches,
        write_patch_table,
    )
except ImportError:  # pragma: no cover - direct script execution
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from src.dataset import FeatureBagDataset, read_manifest, split_records  # type: ignore
    from src.evaluate_clam_sb import load_checkpoint  # type: ignore
    from src.heatmaps import (  # type: ignore
        attention_entropy,
        evaluate_localisation,
        make_patch_contact_sheet,
        normalised_attention_entropy,
        render_attention_heatmap,
        top_attention_patches,
        write_patch_table,
    )


def _resolve_thumbnail(thumbnails: Optional[Path], slide_id: str) -> Optional[Path]:
    if thumbnails is None:
        return None
    for suffix in (".png", ".jpg", ".jpeg", ".tif", ".tiff"):
        candidate = thumbnails / f"{slide_id}{suffix}"
        if candidate.exists():
            return candidate
    return None


def export_attention_heatmaps(
    checkpoint: str | Path,
    manifest: str | Path,
    features_root: str | Path,
    out_dir: str | Path,
    split: str = "test",
    thumbnails: Optional[str | Path] = None,
    masks: Optional[str | Path] = None,
    patch_size: int = 256,
    top_k: int = 20,
    max_slides: Optional[int] = None,
    colormap: str = "jet",
    alpha: float = 0.5,
    device: str = "cpu",
) -> Dict[str, Any]:
    """Render and persist heatmaps/attention arrays for one checkpoint."""
    torch_device = device
    model, payload = load_checkpoint(checkpoint, device=torch_device)
    records = split_records(read_manifest(manifest), split)
    if max_slides:
        records = records[:max_slides]
    if not records:
        raise ValueError(f"Split '{split}' is empty in {manifest}")

    out_dir = Path(out_dir)
    images_dir = out_dir / "images"
    patches_dir = out_dir / "top_patches"
    arrays_dir = out_dir / "arrays"
    for directory in (images_dir, patches_dir, arrays_dir):
        directory.mkdir(parents=True, exist_ok=True)

    thumbnail_root = Path(thumbnails) if thumbnails else None
    mask_root = Path(masks) if masks else None
    dataset = FeatureBagDataset(records, features_root, load_coords=True, train=False)

    manifest_rows: List[Dict[str, Any]] = []
    localisation_rows: List[Dict[str, Any]] = []
    skipped_no_coords: List[str] = []

    for index in range(len(dataset)):
        item = dataset[index]
        record = dataset.records[index]
        if "coords" not in item:
            # A slide without patch coordinates cannot be placed on the slide, so it
            # is skipped with a warning instead of aborting the whole export.
            skipped_no_coords.append(record.image_id)
            continue

        features = item["features"].to(torch_device).unsqueeze(0)
        with torch.no_grad():
            output = model(features, use_instance_loss=False)
        attention = output["attention"].squeeze(0).cpu().numpy()
        probs = torch.softmax(output["logits"], dim=-1).squeeze(0).cpu().numpy()
        prediction = int(np.argmax(probs))
        coords = item["coords"].numpy()
        patch_paths = item.get("patch_paths")

        rendered = render_attention_heatmap(
            attention,
            coords,
            images_dir / f"{record.image_id}_attention.png",
            thumbnail=_resolve_thumbnail(thumbnail_root, record.image_id),
            patch_size=patch_size,
            colormap=colormap,
            alpha=alpha,
            title=f"{record.image_id} pred={prediction} true={record.label}",
        )

        rows = top_attention_patches(attention, coords, patch_paths, k=top_k)
        write_patch_table(rows, patches_dir / f"{record.image_id}_top{top_k}.csv")
        contact = make_patch_contact_sheet(rows, images_dir / f"{record.image_id}_top{top_k}.png")

        npz_payload: Dict[str, Any] = {
            "attention": attention,
            "coords": coords,
            "probs": probs,
            "y_true": np.asarray([record.label]),
            "y_pred": np.asarray([prediction]),
        }
        if patch_paths is not None:
            npz_payload["patch_paths"] = np.asarray(patch_paths, dtype=object)
        np.savez_compressed(arrays_dir / f"{record.image_id}.npz", **npz_payload)

        row = {
            "slide_id": record.image_id,
            "split": record.split,
            "y_true": record.label,
            "y_pred": prediction,
            "prob_pred": float(probs[prediction]),
            "n_patches": int(attention.size),
            "attention_entropy": attention_entropy(attention),
            "normalised_entropy": normalised_attention_entropy(attention),
            "top10_attention_share": float(np.sort(attention)[-10:].sum()),
            "heatmap": rendered["heatmap"],
            "overlay": rendered.get("overlay", ""),
            "top_patch_table": str(patches_dir / f"{record.image_id}_top{top_k}.csv"),
            "contact_sheet": contact or "",
            "arrays": str(arrays_dir / f"{record.image_id}.npz"),
        }
        manifest_rows.append(row)

        if mask_root is not None:
            mask_path = _resolve_thumbnail(mask_root, record.image_id)
            if mask_path is not None:
                overlap = evaluate_localisation(
                    attention, mask_path, coords, patch_size=patch_size
                )
                flat = {
                    "slide_id": record.image_id,
                    "y_true": record.label,
                    "y_pred": prediction,
                    "n_patches": overlap.get("n_patches"),
                    "base_rate": overlap.get("base_rate"),
                }
                for fraction in ("top10pct", "top20pct", "top30pct"):
                    entry = overlap.get(fraction, {})
                    if isinstance(entry, dict):
                        flat[f"{fraction}_precision"] = entry.get("precision")
                        flat[f"{fraction}_recall"] = entry.get("recall")
                        flat[f"{fraction}_lift"] = entry.get("lift")
                localisation_rows.append(flat)

    _write_csv(out_dir / "heatmap_manifest.csv", manifest_rows)
    if localisation_rows:
        _write_csv(out_dir / "localisation_scores.csv", localisation_rows)

    if skipped_no_coords:
        print(
            f"[warn] {len(skipped_no_coords)} slide(s) skipped: no patch coords. "
            "Run scripts/attach_patch_coords.py to fetch them."
        )

    summary = {
        "checkpoint": str(checkpoint),
        "split": split,
        "n_slides": len(manifest_rows),
        "n_skipped_no_coords": len(skipped_no_coords),
        "skipped_no_coords": skipped_no_coords,
        "out_dir": str(out_dir),
        "colormap": colormap,
        "patch_size": patch_size,
        "mean_normalised_entropy": float(
            np.mean([row["normalised_entropy"] for row in manifest_rows])
        )
        if manifest_rows
        else float("nan"),
        "localisation": _summarise_localisation(localisation_rows),
        "instance_loss_active": bool(getattr(model, "instance_loss_weight", 0.0) > 0),
        "seed": payload.get("seed"),
    }
    (out_dir / "export_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def _summarise_localisation(rows: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    if not rows:
        return {"available": False}
    summary: Dict[str, Any] = {"available": True, "n_slides": len(rows)}
    for key in ("top10pct_precision", "top20pct_precision", "top30pct_precision", "top20pct_lift"):
        values = np.asarray([row.get(key, np.nan) for row in rows], dtype=float)
        values = values[np.isfinite(values)]
        if values.size:
            summary[key] = {"mean": float(values.mean()), "std": float(values.std(ddof=1)) if values.size > 1 else 0.0}
    return summary


def _write_csv(path: Path, rows: Sequence[Dict[str, Any]]) -> None:
    if not rows:
        return
    fieldnames: List[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Export CLAM-SB attention heatmaps.")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--manifest", default="data/common_split.csv")
    parser.add_argument("--features-root", default="data/features")
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--split", default="test", choices=["train", "val", "test"])
    parser.add_argument("--thumbnails", default=None)
    parser.add_argument("--masks", default=None, help="tumour/tissue masks for overlap scoring")
    parser.add_argument("--patch-size", type=int, default=256)
    parser.add_argument("--top-k", type=int, default=20)
    parser.add_argument("--max-slides", type=int, default=None)
    parser.add_argument("--colormap", default="jet")
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args(argv)

    summary = export_attention_heatmaps(
        checkpoint=args.checkpoint,
        manifest=args.manifest,
        features_root=args.features_root,
        out_dir=args.out_dir,
        split=args.split,
        thumbnails=args.thumbnails,
        masks=args.masks,
        patch_size=args.patch_size,
        top_k=args.top_k,
        max_slides=args.max_slides,
        colormap=args.colormap,
        device=args.device,
    )
    print(f"exported {summary['n_slides']} slides -> {summary['out_dir']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

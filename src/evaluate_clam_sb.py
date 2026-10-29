"""Evaluate a saved CLAM-SB checkpoint and export its attention profile -- Person 2.

Produces, for the test split:

* ``predictions_<tag>.csv`` -- slide_id, y_true, y_pred, p0..p5
* ``metrics_<tag>.json``    -- QWK (+ bootstrap CI), accuracy, macro F1, per class
* ``attention_<tag>.csv``   -- per-slide attention statistics (fed to the
  statistical baseline and used for the "does instance loss sharpen attention?"
  analysis: mean / max attention, entropy, normalised entropy, top-10 share)
* ``attention_distribution_<tag>.png`` and optional per-slide ``.npz`` arrays

Usage::

    python -m src.evaluate_clam_sb \
        --checkpoint results/metrics/clam_sb/with_instance/checkpoints/clam_sb_seed0_best.pt \
        --manifest data/common_split.csv --features-root data/features \
        --out-dir results/metrics/clam_sb/with_instance/eval
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
    from src.evaluate import bootstrap_metric_ci, compute_metrics, plot_confusion_matrix
    from src.heatmaps import attention_entropy, normalised_attention_entropy, plot_attention_distribution
    from src.models import build_model
    from src.statistical_models import bag_statistics
except ImportError:  # pragma: no cover - direct script execution
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from src.dataset import FeatureBagDataset, read_manifest, split_records  # type: ignore
    from src.evaluate import bootstrap_metric_ci, compute_metrics, plot_confusion_matrix  # type: ignore
    from src.heatmaps import (  # type: ignore
        attention_entropy,
        normalised_attention_entropy,
        plot_attention_distribution,
    )
    from src.models import build_model  # type: ignore
    from src.statistical_models import bag_statistics  # type: ignore


def torch_load_any(path: str | Path) -> Dict[str, Any]:
    """Load a checkpoint across torch versions (``weights_only`` default flip)."""
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
def evaluate_checkpoint(
    checkpoint: str | Path,
    manifest: str | Path,
    features_root: str | Path,
    out_dir: str | Path,
    split: str = "test",
    max_patches: Optional[int] = None,
    save_arrays: bool = False,
    device: str = "cpu",
) -> Dict[str, Any]:
    model, payload = load_checkpoint(checkpoint, device=device)
    all_records = read_manifest(manifest)
    records = all_records if split == "all" else split_records(all_records, split)
    if not records:
        raise ValueError(f"Split '{split}' is empty in {manifest}")

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    tag = Path(checkpoint).stem.replace("_best", "")

    dataset = FeatureBagDataset(records, features_root, max_patches=max_patches, train=False)
    arrays_dir = out_dir / "arrays"
    if save_arrays:
        arrays_dir.mkdir(parents=True, exist_ok=True)

    y_true: List[int] = []
    y_pred: List[int] = []
    probabilities: List[np.ndarray] = []
    slide_rows: List[Dict[str, Any]] = []
    all_attention: List[np.ndarray] = []

    for index in range(len(dataset)):
        item = dataset[index]
        record = dataset.records[index]
        features = item["features"].to(device).unsqueeze(0)
        output = model(features, use_instance_loss=False)
        attention = output["attention"].squeeze(0).cpu().numpy()
        probs = torch.softmax(output["logits"], dim=-1).squeeze(0).cpu().numpy()

        prediction = int(np.argmax(probs))
        y_true.append(record.label)
        y_pred.append(prediction)
        probabilities.append(probs)
        all_attention.append(attention)

        stats = bag_statistics(attention, item["features"].numpy())
        stats.update(
            {
                "slide_id": record.image_id,
                "split": record.split,
                "y_true": record.label,
                "y_pred": prediction,
                "prob_true": float(probs[record.label]),
                "normalised_entropy": normalised_attention_entropy(attention),
                "attention_entropy": attention_entropy(attention),
                "top10_attention_share": float(np.sort(attention)[-10:].sum()),
            }
        )
        slide_rows.append(stats)

        if save_arrays:
            payload_npz: Dict[str, Any] = {"attention": attention, "probs": probs}
            if "coords" in item:
                payload_npz["coords"] = item["coords"].numpy()
            if "patch_paths" in item:
                payload_npz["patch_paths"] = np.asarray(item["patch_paths"], dtype=object)
            np.savez_compressed(arrays_dir / f"{record.image_id}.npz", **payload_npz)

    metrics = compute_metrics(y_true, y_pred, n_classes=model.n_classes)
    metrics.update(
        {
            "checkpoint": str(checkpoint),
            "tag": tag,
            "split": split,
            "seed": payload.get("seed"),
            "instance_loss_active": bool(getattr(model, "instance_loss_weight", 0.0) > 0),
            "attention_entropy_mean": float(np.mean([r["attention_entropy"] for r in slide_rows])),
            "normalised_entropy_mean": float(np.mean([r["normalised_entropy"] for r in slide_rows])),
            "top10_attention_share_mean": float(np.mean([r["top10_attention_share"] for r in slide_rows])),
        }
    )
    metrics["qwk_bootstrap_95ci"] = bootstrap_metric_ci(y_true, y_pred)

    (out_dir / f"metrics_{tag}.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    with (out_dir / f"predictions_{tag}.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["slide_id", "y_true", "y_pred"] + [f"p{c}" for c in range(len(probabilities[0]))])
        for i, row in enumerate(slide_rows):
            writer.writerow([row["slide_id"], y_true[i], y_pred[i]] + [float(p) for p in probabilities[i]])

    if slide_rows:
        fieldnames: List[str] = []
        for row in slide_rows:
            for key in row:
                if key not in fieldnames:
                    fieldnames.append(key)
        with (out_dir / f"attention_{tag}.csv").open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(slide_rows)

    plot_confusion_matrix(
        metrics["confusion_matrix"],
        out_dir / f"confusion_{tag}.png",
        title=f"CLAM-SB {tag} ({split}) QWK={metrics['qwk']:.3f}",
    )
    plot_attention_distribution(
        np.concatenate(all_attention) if all_attention else np.zeros(1),
        out_dir / f"attention_distribution_{tag}.png",
        title=f"Attention weights ({tag}, {split})",
    )
    return metrics


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Evaluate a CLAM-SB checkpoint.")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--manifest", default="data/common_split.csv")
    parser.add_argument("--features-root", default="data/features")
    parser.add_argument("--out-dir", required=True)
    parser.add_argument(
        "--split", default="test", choices=["train", "val", "test", "all"],
        help="'all' is used to emit an attention table across splits for the statistical baseline",
    )
    parser.add_argument("--max-patches", type=int, default=None)
    parser.add_argument("--save-arrays", action="store_true")
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args(argv)

    metrics = evaluate_checkpoint(
        checkpoint=args.checkpoint,
        manifest=args.manifest,
        features_root=args.features_root,
        out_dir=args.out_dir,
        split=args.split,
        max_patches=args.max_patches,
        save_arrays=args.save_arrays,
        device=args.device,
    )
    print(f"QWK {metrics['qwk']:.4f} | accuracy {metrics['accuracy']:.4f} | macro F1 {metrics['macro_f1']:.4f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

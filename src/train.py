"""Shared MIL training / evaluation loop.

Works for every MIL model in the registry (ABMIL, CLAM-SB, TransMIL) so that
all reported numbers come from the same code path: same split, same early
stopping, same metric code, same prediction-file format.

Usage (single model, 3 seeds)::

    python -m src.train --model clam_sb \
        --manifest data/common_split.csv --features-root data/features \
        --config configs/clam_sb.yaml --out-dir results/metrics/clam_sb \
        --seeds 0 1 2

Outputs per run (``tag = f"{model}_{seed}"``)::

    checkpoints/{tag}_best.pt
    history_{tag}.csv
    metrics_{tag}.json
    predictions_{tag}.csv        (slide_id, y_true, y_pred, p0..p5)
    confusion_{tag}.png
and across seeds::

    aggregate_{model}.json, table_{model}.md, run_config.json

Loss: cross-entropy on the 6 ISUP classes (+ optional class weights for the
ISUP imbalance). CLAM-SB adds ``instance_loss_weight * instance_loss``.
Model selection: validation QWK (never test), early stopping with patience.

Owner: Person 1 (shared loop). Model-specific drivers live next to it.
"""

from __future__ import annotations

import argparse
import copy
import csv
import json
import random
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

try:  # works both as ``python -m src.train`` and ``python src/train.py``
    from src.dataset import (
        FeatureBagDataset,
        SlideRecord,
        collate_bags,
        infer_feature_dim,
        read_manifest,
        split_records,
    )
    from src.evaluate import (
        aggregate_seed_metrics,
        bootstrap_metric_ci,
        compute_metrics,
        markdown_table,
        plot_confusion_matrix,
        quadratic_weighted_kappa,
    )
    from src.models import build_model, count_parameters
except ImportError:  # pragma: no cover - direct script execution
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from src.dataset import (  # type: ignore
        FeatureBagDataset,
        SlideRecord,
        collate_bags,
        infer_feature_dim,
        read_manifest,
        split_records,
    )
    from src.evaluate import (  # type: ignore
        aggregate_seed_metrics,
        bootstrap_metric_ci,
        compute_metrics,
        markdown_table,
        plot_confusion_matrix,
        quadratic_weighted_kappa,
    )
    from src.models import build_model, count_parameters  # type: ignore

ISUP_CLASSES: Tuple[int, ...] = (0, 1, 2, 3, 4, 5)


# --------------------------------------------------------------------------- #
# Utilities
# --------------------------------------------------------------------------- #
def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def resolve_device(requested: str = "auto") -> torch.device:
    if requested and requested != "auto":
        return torch.device(requested)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def _loader(
    records: Sequence[SlideRecord],
    features_root: str | Path,
    batch_size: int,
    shuffle: bool,
    max_patches: Optional[int] = None,
    train: bool = False,
    num_workers: int = 0,
    seed: int = 42,
) -> DataLoader:
    dataset = FeatureBagDataset(
        records, features_root, max_patches=max_patches, train=train, seed=seed
    )
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        collate_fn=collate_bags,
        num_workers=num_workers,
        drop_last=False,
    )


def compute_class_weights(
    records: Sequence[SlideRecord], n_classes: int = 6
) -> torch.Tensor:
    """Inverse-frequency weights; ``none``/``balanced`` handled by the caller."""
    counts = np.zeros(n_classes, dtype=np.float64)
    for record in records:
        counts[record.label] += 1
    counts = np.maximum(counts, 1.0)
    weights = counts.sum() / (counts * n_classes)
    return torch.tensor(weights, dtype=torch.float32)


# --------------------------------------------------------------------------- #
# Train / evaluate one run
# --------------------------------------------------------------------------- #
def _forward(model: nn.Module, batch: Dict[str, Any], with_instance_loss: bool) -> Dict[str, Any]:
    kwargs: Dict[str, Any] = {"mask": batch["mask"]}
    if with_instance_loss:
        kwargs["labels"] = batch["label"]
    else:
        kwargs["use_instance_loss"] = False
    return model(batch["features"], **kwargs)


def train_one_run(
    model_name: str,
    model_config: Dict[str, Any],
    train_records: Sequence[SlideRecord],
    val_records: Sequence[SlideRecord],
    test_records: Sequence[SlideRecord],
    features_root: str | Path,
    out_dir: str | Path,
    tag: str,
    seed: int = 42,
    max_epochs: int = 50,
    patience: int = 12,
    batch_size: int = 16,
    lr: float = 2e-4,
    weight_decay: float = 1e-5,
    max_patches: Optional[int] = 4096,
    class_weights: str = "balanced",
    grad_clip: float = 5.0,
    instance_loss_weight: Optional[float] = None,
    device: str = "auto",
    num_workers: int = 0,
    verbose: bool = True,
) -> Dict[str, Any]:
    """Train one model for one seed and evaluate it on the test split."""
    set_seed(seed)
    device_obj = resolve_device(device)
    out_dir = Path(out_dir)
    (out_dir / "checkpoints").mkdir(parents=True, exist_ok=True)

    model = build_model(model_name, model_config).to(device_obj)
    with_instance_loss = (
        getattr(model, "instance_loss_weight", 0.0) > 0
        if instance_loss_weight is None
        else instance_loss_weight > 0
    )
    if instance_loss_weight is not None and hasattr(model, "config"):
        try:
            model.config.instance_loss_weight = float(instance_loss_weight)
        except AttributeError:  # pragma: no cover - non-CLAM models
            pass
    weight = float(getattr(model, "instance_loss_weight", 0.0)) if with_instance_loss else 0.0

    weights_tensor = None
    if class_weights == "balanced":
        weights_tensor = compute_class_weights(train_records, model.n_classes).to(device_obj)
    slide_criterion = nn.CrossEntropyLoss(weight=weights_tensor)

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(max_epochs, 1))

    train_loader = _loader(
        train_records,
        features_root,
        batch_size,
        shuffle=True,
        max_patches=max_patches,
        train=True,
        num_workers=num_workers,
        seed=seed,
    )
    val_loader = _loader(
        val_records, features_root, batch_size=1, shuffle=False, max_patches=max_patches
    )
    test_loader = _loader(
        test_records, features_root, batch_size=1, shuffle=False, max_patches=max_patches
    )

    history: List[Dict[str, Any]] = []
    best_qwk = -np.inf
    best_state = copy.deepcopy(model.state_dict())
    best_epoch = 0
    patience_left = patience
    started = time.time()

    for epoch in range(1, max_epochs + 1):
        model.train()
        if hasattr(train_loader.dataset, "set_epoch"):
            train_loader.dataset.set_epoch(epoch)

        running_loss, running_slide, running_instance, seen = 0.0, 0.0, 0.0, 0
        for batch in train_loader:
            features = batch["features"].to(device_obj)
            mask = batch["mask"].to(device_obj)
            labels = batch["label"].to(device_obj)
            batch = {"features": features, "mask": mask, "label": labels}

            optimizer.zero_grad(set_to_none=True)
            output = _forward(model, batch, with_instance_loss)
            slide_loss = slide_criterion(output["logits"], labels)
            instance_loss = output.get("instance_loss")
            loss = slide_loss
            if weight > 0 and torch.is_tensor(instance_loss):
                loss = loss + weight * instance_loss

            loss.backward()
            if grad_clip:
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            optimizer.step()

            running_loss += float(loss.detach()) * labels.size(0)
            running_slide += float(slide_loss.detach()) * labels.size(0)
            if torch.is_tensor(instance_loss):
                running_instance += float(instance_loss.detach()) * labels.size(0)
            seen += labels.size(0)
        scheduler.step()

        val_scores = predict(model, val_loader, device_obj, with_instance_loss)
        val_qwk = quadratic_weighted_kappa(val_scores["y_true"], val_scores["y_pred"])
        record = {
            "epoch": epoch,
            "train_loss": running_loss / max(seen, 1),
            "slide_loss": running_slide / max(seen, 1),
            "instance_loss": running_instance / max(seen, 1),
            "val_qwk": val_qwk,
            "val_accuracy": float(np.mean(np.asarray(val_scores["y_true"]) == np.asarray(val_scores["y_pred"]))),
            "lr": optimizer.param_groups[0]["lr"],
        }
        history.append(record)
        if verbose:
            print(
                f"[{tag}] epoch {epoch:03d} loss={record['train_loss']:.4f} "
                f"val_qwk={val_qwk:.4f} val_acc={record['val_accuracy']:.4f}"
            )

        if np.isfinite(val_qwk) and val_qwk > best_qwk:
            best_qwk = val_qwk
            best_epoch = epoch
            best_state = copy.deepcopy(model.state_dict())
            patience_left = patience
        else:
            patience_left -= 1
            if patience_left <= 0:
                if verbose:
                    print(f"[{tag}] early stopping at epoch {epoch} (best epoch {best_epoch})")
                break

    model.load_state_dict(best_state)
    checkpoint_path = out_dir / "checkpoints" / f"{tag}_best.pt"
    torch.save(
        {
            "model_name": model_name,
            "model_config": getattr(model, "config", {}).to_dict() if hasattr(getattr(model, "config", None), "to_dict") else model_config,
            "state_dict": best_state,
            "seed": seed,
            "best_epoch": best_epoch,
            "best_val_qwk": best_qwk,
        },
        checkpoint_path,
    )

    test_scores = predict(model, test_loader, device_obj, with_instance_loss)
    metrics = compute_metrics(test_scores["y_true"], test_scores["y_pred"], n_classes=model.n_classes)
    metrics.update(
        {
            "model": model_name,
            "tag": tag,
            "seed": seed,
            "best_epoch": best_epoch,
            "best_val_qwk": float(best_qwk),
            "instance_loss_active": bool(weight > 0),
            "instance_loss_weight": weight,
            "n_parameters": count_parameters(model),
            "train_seconds": time.time() - started,
        }
    )
    metrics["qwk_bootstrap_95ci"] = bootstrap_metric_ci(
        test_scores["y_true"], test_scores["y_pred"], seed=seed
    )

    with (out_dir / f"history_{tag}.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(history[0]))
        writer.writeheader()
        writer.writerows(history)
    (out_dir / f"metrics_{tag}.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")

    probabilities = test_scores["probs"]
    fieldnames = ["slide_id", "y_true", "y_pred"] + [f"p{c}" for c in range(probabilities.shape[1])]
    with (out_dir / f"predictions_{tag}.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for i, slide_id in enumerate(test_scores["slide_id"]):
            row = {
                "slide_id": slide_id,
                "y_true": int(test_scores["y_true"][i]),
                "y_pred": int(test_scores["y_pred"][i]),
            }
            for c in range(probabilities.shape[1]):
                row[f"p{c}"] = float(probabilities[i, c])
            writer.writerow(row)

    plot_confusion_matrix(
        metrics["confusion_matrix"],
        out_dir / f"confusion_{tag}.png",
        title=f"{model_name} seed {seed} (test QWK={metrics['qwk']:.3f})",
    )
    metrics["checkpoint"] = str(checkpoint_path)
    return metrics


@torch.no_grad()
def predict(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    with_instance_loss: bool = False,
) -> Dict[str, Any]:
    """Run inference slide-by-slide (batch size 1 keeps bags unpadded)."""
    model.eval()
    y_true: List[int] = []
    y_pred: List[int] = []
    probs: List[np.ndarray] = []
    slide_ids: List[str] = []
    attention_rows: List[np.ndarray] = []

    for batch in loader:
        features = batch["features"].to(device)
        mask = batch["mask"].to(device)
        labels = batch["label"].to(device)
        output = _forward(model, {"features": features, "mask": mask, "label": labels}, with_instance_loss)
        logits = output["logits"]
        probabilities = torch.softmax(logits, dim=-1)
        y_true.extend(labels.cpu().tolist())
        y_pred.extend(probabilities.argmax(dim=1).cpu().tolist())
        probs.append(probabilities.cpu().numpy())
        slide_ids.extend(batch["image_id"])
        attention = output["attention"].detach().cpu().numpy()
        attention_rows.extend(list(attention))

    return {
        "y_true": y_true,
        "y_pred": y_pred,
        "probs": np.concatenate(probs, axis=0) if probs else np.zeros((0, model.n_classes)),
        "slide_id": slide_ids,
        "attention": attention_rows,
    }


# --------------------------------------------------------------------------- #
# Multi-seed driver
# --------------------------------------------------------------------------- #
def run_experiment(
    model_name: str,
    model_config: Dict[str, Any],
    manifest: str | Path,
    features_root: str | Path,
    out_dir: str | Path,
    seeds: Sequence[int] = (0, 1, 2),
    train_kwargs: Optional[Dict[str, Any]] = None,
    in_dim: Optional[int] = None,
    verbose: bool = True,
) -> Dict[str, Any]:
    """Train/evaluate ``model_name`` for several seeds and aggregate results."""
    train_kwargs = dict(train_kwargs or {})
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    records = read_manifest(manifest)
    train_records = split_records(records, "train")
    val_records = split_records(records, "val")
    test_records = split_records(records, "test")
    if not test_records:
        raise ValueError("Test split is empty -- check the manifest's 'split' column")

    # The features on disk are ground truth for D; the config file only provides
    # a default. An explicit --in-dim / in_dim argument still wins.
    detected_in_dim = infer_feature_dim(train_records, features_root)
    resolved_in_dim = int(train_kwargs.pop("in_dim", None) or in_dim or detected_in_dim)
    if model_config.get("in_dim") not in (None, resolved_in_dim):
        print(
            f"[warn] config in_dim={model_config['in_dim']} but features are "
            f"{resolved_in_dim}-d; using the feature dimension."
        )
    model_config = {**model_config, "in_dim": resolved_in_dim}

    run_metrics: List[Dict[str, Any]] = []
    for seed in seeds:
        tag = f"{model_name}_seed{seed}"
        run_metrics.append(
            train_one_run(
                model_name=model_name,
                model_config=model_config,
                train_records=train_records,
                val_records=val_records,
                test_records=test_records,
                features_root=features_root,
                out_dir=out_dir,
                tag=tag,
                seed=seed,
                verbose=verbose,
                **train_kwargs,
            )
        )

    aggregate = aggregate_seed_metrics(run_metrics)
    aggregate["model"] = model_name
    aggregate["seeds"] = list(seeds)
    aggregate["per_run"] = [
        {k: m[k] for k in ("tag", "seed", "qwk", "accuracy", "macro_f1", "balanced_accuracy", "best_epoch")}
        for m in run_metrics
    ]
    (out_dir / f"aggregate_{model_name}.json").write_text(json.dumps(aggregate, indent=2), encoding="utf-8")

    table = markdown_table(
        [
            {
                "model": model_name,
                "qwk": aggregate.get("qwk", {}).get("mean", float("nan")),
                "qwk_std": aggregate.get("qwk", {}).get("std", 0.0),
                "accuracy": aggregate.get("accuracy", {}).get("mean", float("nan")),
                "balanced_accuracy": aggregate.get("balanced_accuracy", {}).get("mean", float("nan")),
                "macro_f1": aggregate.get("macro_f1", {}).get("mean", float("nan")),
                "seeds": len(seeds),
            }
        ],
        [
            ("model", "Model"),
            ("qwk", "QWK (mean)"),
            ("qwk_std", "QWK (std)"),
            ("accuracy", "Accuracy"),
            ("balanced_accuracy", "Balanced acc."),
            ("macro_f1", "Macro F1"),
            ("seeds", "Seeds"),
        ],
    )
    (out_dir / f"table_{model_name}.md").write_text(table + "\n", encoding="utf-8")

    (out_dir / "run_config.json").write_text(
        json.dumps(
            {
                "model": model_name,
                "model_config": model_config,
                "train_kwargs": train_kwargs,
                "manifest": str(manifest),
                "features_root": str(features_root),
                "seeds": list(seeds),
                "n_train": len(train_records),
                "n_val": len(val_records),
                "n_test": len(test_records),
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    return {"aggregate": aggregate, "per_run": run_metrics}


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
def load_yaml(path: str | Path) -> Dict[str, Any]:
    try:
        import yaml
    except ModuleNotFoundError as exc:  # pragma: no cover
        raise SystemExit("PyYAML is required for --config: pip install pyyaml") from exc
    return yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train a MIL model on PANDA feature bags.")
    parser.add_argument("--model", default="clam_sb", choices=["abmil", "clam_sb", "transmil"])
    parser.add_argument("--manifest", default="data/common_split.csv")
    parser.add_argument("--features-root", default="data/features")
    parser.add_argument("--out-dir", default=None)
    parser.add_argument("--config", default=None, help="YAML file with model/training keys")
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--patience", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--max-patches", type=int, default=None)
    parser.add_argument("--in-dim", type=int, default=None)
    parser.add_argument("--class-weights", default=None, choices=["balanced", "none"])
    parser.add_argument("--instance-loss-weight", type=float, default=None)
    parser.add_argument("--no-instance-loss", action="store_true")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--quiet", action="store_true")
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_argparser().parse_args(argv)
    config = load_yaml(args.config) if args.config else {}

    model_config = dict(config.get("model", {}))
    data_config = dict(config.get("data", {}))
    training = dict(config.get("training", {}))

    if args.epochs is not None:
        training["max_epochs"] = args.epochs
    if args.patience is not None:
        training["patience"] = args.patience
    if args.batch_size is not None:
        training["batch_size"] = args.batch_size
    if args.lr is not None:
        training["lr"] = args.lr
    if args.max_patches is not None:
        training["max_patches"] = args.max_patches
    if args.class_weights is not None:
        training["class_weights"] = args.class_weights
    if args.in_dim is not None:
        model_config["in_dim"] = args.in_dim
    training["device"] = args.device

    if args.no_instance_loss:
        model_config["instance_loss_weight"] = 0.0
        training["instance_loss_weight"] = 0.0
    elif args.instance_loss_weight is not None:
        model_config["instance_loss_weight"] = args.instance_loss_weight
        training["instance_loss_weight"] = args.instance_loss_weight

    out_dir = args.out_dir or f"results/metrics/{args.model}"
    summary = run_experiment(
        model_name=args.model,
        model_config=model_config,
        manifest=args.manifest,
        features_root=args.features_root,
        out_dir=out_dir,
        seeds=args.seeds,
        train_kwargs=training,
        in_dim=data_config.get("in_dim"),
        verbose=not args.quiet,
    )
    headline = summary["aggregate"].get("qwk", {})
    print(
        f"\n{args.model}: QWK {headline.get('mean', float('nan')):.4f} "
        f"+/- {headline.get('std', 0.0):.4f} over {len(args.seeds)} seed(s) -> {out_dir}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""Shared data layer: the PANDA split manifest and the per-slide feature bag.

This module defines the **one contract everyone must obey**:

    data/common_split.csv
        image_id, data_provider, isup_grade, gleason_score, split

    data/features/{image_id}.pt
        a) torch.Tensor  [N, D]                      -- minimum requirement
        b) dict with "features" [N, D] and "coords"  [N, 2] (x, y patch origin)
        c) dict with "features" [N, D] and "patch_paths" [N] strings

``N`` is the number of tissue patches kept for that slide, ``D`` the frozen
encoder dimension (1024 for ImageNet ResNet50, 2048 for the 2048-d variant).
Nothing in the models depends on the value of ``N`` or on patch ordering.

Owner: Person 1 (contract) / everybody (consumers).
"""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset

ISUP_CLASSES: Tuple[int, ...] = (0, 1, 2, 3, 4, 5)
SPLITS: Tuple[str, ...] = ("train", "val", "test")

# Column aliases tolerated when reading a manifest.
_ID_KEYS = ("image_id", "slide_id", "slide", "id")
_LABEL_KEYS = ("isup_grade", "label", "grade", "y")
_SPLIT_KEYS = ("split", "fold", "partition")
_PROVIDER_KEYS = ("data_provider", "provider", "source")
_GLEASON_KEYS = ("gleason_score", "gleason", "isup_grade_group")

_FEATURE_KEYS = ("features", "feats", "feature", "embeddings", "emb", "x")
_COORD_KEYS = ("coords", "coordinates", "coord", "locs", "locations", "xy")
_PATH_KEYS = ("patch_paths", "paths", "patch_files", "filenames")


# --------------------------------------------------------------------------- #
# Manifest handling
# --------------------------------------------------------------------------- #
def _first_present(row: Dict[str, Any], keys: Sequence[str]) -> Optional[Any]:
    for key in keys:
        if key in row and row[key] not in (None, "", "nan"):
            return row[key]
    return None


class SlideRecord(dict):
    """A single manifest row with normalised key names."""

    @property
    def image_id(self) -> str:
        return str(self["image_id"])

    @property
    def label(self) -> int:
        return int(self["isup_grade"])

    @property
    def split(self) -> str:
        return str(self["split"])

    @property
    def data_provider(self) -> str:
        return str(self.get("data_provider", "unknown"))

    @property
    def gleason_score(self) -> str:
        return str(self.get("gleason_score", "unknown"))


def read_manifest(path: str | Path) -> List[SlideRecord]:
    """Read ``common_split.csv`` and normalise its column names.

    Raises a descriptive error if the slide-id, label or split column is
    missing, because silently training on a mis-parsed manifest is exactly the
    kind of bug that invalidates a whole experiment.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(
            f"Manifest not found: {path}. Run scripts/create_split.py first "
            "(or point --manifest at data/mock/manifest.csv for the mock demo)."
        )

    records: List[SlideRecord] = []
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            raise ValueError(f"Manifest {path} has no header row")
        unknown_id = _first_present({k: k for k in reader.fieldnames}, _ID_KEYS)
        if unknown_id is None:
            raise ValueError(
                f"Manifest {path} needs one of these id columns: {_ID_KEYS}; "
                f"found {reader.fieldnames}"
            )
        for row in reader:
            image_id = _first_present(row, _ID_KEYS)
            label = _first_present(row, _LABEL_KEYS)
            split = _first_present(row, _SPLIT_KEYS)
            if image_id is None or label is None or split is None:
                continue
            split = str(split).strip().lower()
            if split == "valid":
                split = "val"
            if split not in SPLITS:
                continue
            try:
                label_int = int(float(label))
            except (TypeError, ValueError):
                continue
            if label_int not in ISUP_CLASSES:
                continue
            records.append(
                SlideRecord(
                    image_id=str(image_id).strip(),
                    isup_grade=label_int,
                    split=split,
                    data_provider=str(_first_present(row, _PROVIDER_KEYS) or "unknown"),
                    gleason_score=str(_first_present(row, _GLEASON_KEYS) or "unknown"),
                )
            )

    if not records:
        raise ValueError(f"Manifest {path} produced zero usable rows")
    return records


def split_records(
    records: Iterable[SlideRecord], split: str
) -> List[SlideRecord]:
    split = "val" if split == "valid" else split
    return [record for record in records if record.split == split]


def class_distribution(records: Iterable[SlideRecord]) -> Dict[str, Dict[int, int]]:
    """Counts per ISUP grade, per split -- used by EDA and by the report."""
    distribution: Dict[str, Dict[int, int]] = {s: {c: 0 for c in ISUP_CLASSES} for s in SPLITS}
    for record in records:
        distribution.setdefault(record.split, {c: 0 for c in ISUP_CLASSES})
        distribution[record.split][record.label] = (
            distribution[record.split].get(record.label, 0) + 1
        )
    return distribution


# --------------------------------------------------------------------------- #
# Feature bag loading
# --------------------------------------------------------------------------- #
def _extract_arrays(payload: Any) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[List[str]]]:
    """Pull (features, coords, patch_paths) out of whatever Person 1 saved."""
    if isinstance(payload, torch.Tensor):
        features, coords, patch_paths = payload, None, None
    elif isinstance(payload, np.ndarray):
        features, coords, patch_paths = torch.from_numpy(payload), None, None
    elif isinstance(payload, dict):
        feature_key = next((k for k in _FEATURE_KEYS if k in payload), None)
        if feature_key is None:
            raise KeyError(
                f"Feature dict has none of {_FEATURE_KEYS}; keys are {sorted(payload)}"
            )
        features = payload[feature_key]
        coord_key = next((k for k in _COORD_KEYS if k in payload), None)
        coords = payload[coord_key] if coord_key is not None else None
        path_key = next((k for k in _PATH_KEYS if k in payload), None)
        patch_paths = list(payload[path_key]) if path_key is not None else None
        if not isinstance(features, torch.Tensor):
            features = torch.as_tensor(np.asarray(features))
        if coords is not None and not isinstance(coords, torch.Tensor):
            coords = torch.as_tensor(np.asarray(coords))
    else:
        raise TypeError(f"Unsupported feature payload type: {type(payload)!r}")

    features = features.float()
    if features.ndim != 2:
        raise ValueError(f"features must be [N, D], got {tuple(features.shape)}")
    if coords is not None:
        coords = coords.float().reshape(features.size(0), -1)[:, :2]
    return features, coords, patch_paths


def load_feature_bag(
    path: str | Path,
) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[List[str]]]:
    """Load one slide bag from ``.pt`` (also accepts ``.npy`` / ``.npz``)."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Feature file not found: {path}")

    suffix = path.suffix.lower()
    if suffix in {".pt", ".pth", ".bin"}:
        try:
            payload = torch.load(path, map_location="cpu", weights_only=False)
        except TypeError:  # torch < 2.0
            payload = torch.load(path, map_location="cpu")
    elif suffix == ".npy":
        payload = np.load(path)
    elif suffix == ".npz":
        archive = np.load(path)
        payload = {key: archive[key] for key in archive.files}
    else:
        raise ValueError(f"Unsupported feature file type: {path.suffix}")

    return _extract_arrays(payload)


def feature_path(features_root: str | Path, image_id: str) -> Path:
    """Resolve ``data/features/{image_id}.pt`` (or .npy/.npz) for a slide."""
    root = Path(features_root)
    for suffix in (".pt", ".pth", ".npy", ".npz"):
        candidate = root / f"{image_id}{suffix}"
        if candidate.exists():
            return candidate
    # Default guess so the error message names the file we expected.
    return root / f"{image_id}.pt"


# --------------------------------------------------------------------------- #
# Coverage validation (never silently change the evaluation set)
# --------------------------------------------------------------------------- #
def validate_feature_coverage(
    records: Sequence[SlideRecord],
    features_root: str | Path,
    fail_fast: bool = True,
) -> Dict[str, Any]:
    """Check that every slide in the manifest has a feature file.

    Silent filtering of missing slides is dangerous: if two test slides are
    missing, the test split quietly becomes 58 slides and every metric stops
    being comparable to a teammate's run. This function reports the damage per
    split and, by default, refuses to continue.
    """
    features_root = Path(features_root)
    report: Dict[str, Any] = {"per_split": {}, "missing": []}

    for split in SPLITS + ("unknown",):
        subset = [r for r in records if r.split == split]
        if not subset:
            continue
        present = [r for r in subset if feature_path(features_root, r.image_id).exists()]
        missing = [r.image_id for r in subset if not feature_path(features_root, r.image_id).exists()]
        report["per_split"][split] = {
            "expected": len(subset),
            "present": len(present),
            "missing": len(missing),
        }
        report["missing"].extend(missing)

    if report["missing"] and fail_fast:
        preview = ", ".join(report["missing"][:5])
        more = "" if len(report["missing"]) <= 5 else f" (+{len(report['missing']) - 5} more)"
        raise FileNotFoundError(
            f"{len(report['missing'])} slide(s) in the manifest have no feature file under "
            f"'{features_root}': {preview}{more}\n"
            "  Download them with:  python scripts/download_real_features.py "
            "--manifest data/common_split.csv --out-dir "
            f"{features_root}\n"
            "  Or check coverage first:  python scripts/attach_patch_coords.py --verify-only"
        )
    return report


def format_coverage_table(report: Dict[str, Any]) -> str:
    """One-line-per-split summary used by the CLI drivers."""
    lines = ["split     expected  present  missing"]
    for split, counts in report["per_split"].items():
        lines.append(
            f"{split:<9} {counts['expected']:>8} {counts['present']:>8} {counts['missing']:>8}"
        )
    return "\n".join(lines)


# --------------------------------------------------------------------------- #
# Torch datasets
# --------------------------------------------------------------------------- #
class FeatureBagDataset(Dataset):
    """One item per slide: ``{"features", "label", "image_id", ...}``.

    ``max_patches`` randomly subsamples large bags during training (a cheap
    memory guard for 10k-patch slides) and takes an evenly spaced subset at
    evaluation time so predictions stay deterministic.
    """

    def __init__(
        self,
        records: Sequence[SlideRecord],
        features_root: str | Path,
        max_patches: Optional[int] = None,
        train: bool = False,
        load_coords: bool = False,
        seed: int = 42,
    ) -> None:
        self.records = list(records)
        self.features_root = Path(features_root)
        missing = [r.image_id for r in self.records if not feature_path(self.features_root, r.image_id).exists()]
        if missing:
            raise FileNotFoundError(
                f"{len(missing)} of {len(self.records)} slides have no feature file under "
                f"'{self.features_root}' (first: {missing[0]}). Refusing to train on a "
                "silently reduced dataset -- download the features first."
            )
        self.max_patches = max_patches
        self.train = train
        self.load_coords = load_coords
        self.seed = seed
        self._epoch = 0

    def __len__(self) -> int:
        return len(self.records)

    def set_epoch(self, epoch: int) -> None:
        self._epoch = int(epoch)

    def _subsample(self, tensor: torch.Tensor) -> torch.Tensor:
        n = tensor.size(0)
        if not self.max_patches or n <= self.max_patches:
            return tensor
        if self.train:
            generator = torch.Generator().manual_seed(self.seed + 1000 * self._epoch + n)
            index = torch.randperm(n, generator=generator)[: self.max_patches]
        else:
            index = torch.linspace(0, n - 1, self.max_patches).long()
        return tensor[index]

    def __getitem__(self, index: int) -> Dict[str, Any]:
        record = self.records[index]
        path = feature_path(self.features_root, record.image_id)
        features, coords, patch_paths = load_feature_bag(path)

        if self.max_patches and features.size(0) > self.max_patches:
            keep = self._subsample(torch.arange(features.size(0)))
            features = features[keep]
            coords = coords[keep] if coords is not None else None
            patch_paths = [patch_paths[i] for i in keep.tolist()] if patch_paths else None

        item: Dict[str, Any] = {
            "features": features,
            "label": torch.tensor(record.label, dtype=torch.long),
            "image_id": record.image_id,
            "data_provider": record.data_provider,
            "gleason_score": record.gleason_score,
        }
        if self.load_coords and coords is not None:
            item["coords"] = coords
        if patch_paths is not None:
            item["patch_paths"] = patch_paths
        return item


def collate_bags(batch: Sequence[Dict[str, Any]], pad_value: float = 0.0) -> Dict[str, Any]:
    """Pad variable-length bags into ``[B, N_max, D]`` plus a validity mask."""
    max_n = max(item["features"].size(0) for item in batch)
    dim = batch[0]["features"].size(1)
    batch_size = len(batch)

    features = torch.full((batch_size, max_n, dim), pad_value, dtype=torch.float32)
    mask = torch.zeros((batch_size, max_n), dtype=torch.bool)
    for i, item in enumerate(batch):
        n = item["features"].size(0)
        features[i, :n] = item["features"]
        mask[i, :n] = True

    collated: Dict[str, Any] = {
        "features": features,
        "mask": mask,
        "label": torch.stack([item["label"] for item in batch]),
        "image_id": [item["image_id"] for item in batch],
        "data_provider": [item.get("data_provider", "unknown") for item in batch],
        "gleason_score": [item.get("gleason_score", "unknown") for item in batch],
    }
    if all("coords" in item for item in batch):
        max_c = max(item["coords"].size(1) for item in batch)
        coords = torch.full((batch_size, max_n, max_c), float("nan"), dtype=torch.float32)
        for i, item in enumerate(batch):
            n = item["coords"].size(0)
            coords[i, :n, : item["coords"].size(1)] = item["coords"]
        collated["coords"] = coords
    return collated


def infer_feature_dim(
    records: Sequence[SlideRecord], features_root: str | Path, max_probe: int = 5
) -> int:
    """Probe the first few slides to report ``D`` (used by configs and CLIs)."""
    errors: List[str] = []
    for record in list(records)[:max_probe]:
        path = feature_path(features_root, record.image_id)
        try:
            features, _, _ = load_feature_bag(path)
            return int(features.size(1))
        except Exception as exc:  # pragma: no cover - diagnostic path
            errors.append(f"{record.image_id}: {exc}")
    raise RuntimeError(
        "Could not infer feature dimension from any slide. Tried:\n  "
        + "\n  ".join(errors)
    )


def describe_dataset(
    manifest: str | Path, features_root: str | Path
) -> Dict[str, Any]:
    """Small JSON-friendly summary used by notebooks and the report."""
    records = read_manifest(manifest)
    distribution = class_distribution(records)
    summary: Dict[str, Any] = {
        "manifest": str(manifest),
        "features_root": str(features_root),
        "n_slides": len(records),
        "per_split": {split: len(split_records(records, split)) for split in SPLITS},
        "class_distribution": {k: {str(c): v for c, v in d.items()} for k, d in distribution.items()},
    }
    try:
        summary["feature_dim"] = infer_feature_dim(records, features_root)
    except RuntimeError as exc:
        summary["feature_dim"] = None
        summary["feature_dim_error"] = str(exc)
    return summary


def write_json(path: str | Path, payload: Dict[str, Any]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

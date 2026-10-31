"""Build a feature-manifest restricted to slides that actually carry patch coordinates.

The PANDA feature release has two parallel trees:

* ``panda/r50/pt_files/{image_id}.pt``   -- CLAM-style ``{"features", "coords"}``
  for most slides, but the ``pt`` files we were given contain **features only**;
* ``panda/patches/{image_id}.h5``        -- ``{"features": [N, 1024], "coords": [N, 2]}``
  for 399 of the 420 slides.

Attention heatmaps need ``coords``; the 21 slides without them are hard 404s and can
never be plotted. Rather than making the exporter fail-fast (or silently drop slides
without telling anyone), this script writes a *coords-only* manifest that the explicit
export step consumes. Training/evaluation keep using the full manifest so no slide is
lost from the metrics.

Usage::

    python scripts/make_coords_manifest.py \
        --manifest data/common_split.csv \
        --features-root data/features \
        --out data/common_split_with_coords.csv

    python -m src.export_clam_attention \
        --checkpoint results/metrics/clam_sb/with_instance/checkpoints/clam_sb_seed0_best.pt \
        --manifest data/common_split_with_coords.csv \
        --out-dir results/heatmaps/clam_sb_seed0
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import h5py

H5_SUFFIXES = (".h5", ".hdf5")
PT_SUFFIXES = (".pt", ".pth")
REQUIRED_COLUMNS = ("image_id", "split")


def find_feature_files(features_root: Path, image_id: str) -> List[Path]:
    """Every feature file for one slide, ``.h5`` first (the release that carries coords)."""
    candidates: List[Path] = []
    for suffix in H5_SUFFIXES + PT_SUFFIXES:
        candidates += [
            features_root / f"{image_id}{suffix}",
            features_root / "patches" / f"{image_id}{suffix}",
            features_root / "panda" / "patches" / f"{image_id}{suffix}",
            features_root / "pt_files" / f"{image_id}{suffix}",
            features_root / "panda" / "pt_files" / f"{image_id}{suffix}",
        ]
    return [c for c in candidates if c.is_file()]


def has_coords(path: Path) -> Tuple[bool, int]:
    """Return ``(has_coords, n_patches)`` for one feature file (``.h5`` or ``.pt``)."""
    if path.suffix in PT_SUFFIXES:
        try:
            import torch  # imported lazily: only needed for the CLAM-style .pt release
        except ImportError:
            return False, 0
        try:
            payload = torch.load(path, map_location="cpu", weights_only=False)
        except Exception:  # noqa: BLE001 - corrupt/foreign file, treat as unusable
            return False, 0
        if not isinstance(payload, dict) or "coords" not in payload or "features" not in payload:
            return False, 0
        n_patches = int(len(payload["features"]))
        return (int(len(payload["coords"])) == n_patches), n_patches

    try:
        with h5py.File(path, "r") as handle:
            if "coords" not in handle or "features" not in handle:
                return False, 0
            n_patches = int(handle["features"].shape[0])
            if int(handle["coords"].shape[0]) != n_patches:
                return False, n_patches
            return True, n_patches
    except (OSError, KeyError):
        return False, 0


def build_manifest(
    manifest: str | Path,
    features_root: str | Path,
    out: str | Path,
    verbose: bool = True,
) -> Dict[str, object]:
    """Write a copy of ``manifest`` keeping only slides whose feature file has coords."""
    manifest = Path(manifest)
    features_root = Path(features_root)
    out = Path(out)

    if not manifest.is_file():
        raise FileNotFoundError(f"manifest not found: {manifest}")
    if not features_root.is_dir():
        raise FileNotFoundError(f"features root not found: {features_root}")

    with manifest.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        rows = list(reader)
        fieldnames = list(reader.fieldnames or [])
    if not rows:
        raise ValueError(f"manifest is empty: {manifest}")
    missing_columns = [c for c in REQUIRED_COLUMNS if c not in fieldnames]
    if missing_columns:
        raise ValueError(f"manifest {manifest} is missing columns: {missing_columns}")

    kept: List[Dict[str, str]] = []
    no_file: List[str] = []
    no_coords: List[str] = []
    per_split_kept: Dict[str, int] = {}
    per_split_total: Dict[str, int] = {}
    patch_counts: List[int] = []

    for row in rows:
        image_id = (row.get("image_id") or "").strip()
        split = (row.get("split") or "").strip()
        per_split_total[split] = per_split_total.get(split, 0) + 1

        candidates = find_feature_files(features_root, image_id)
        if not candidates:
            no_file.append(image_id)
            continue
        usable, n_patches = False, 0
        for candidate in candidates:
            usable, n_patches = has_coords(candidate)
            if usable:
                break
        if not usable:
            no_coords.append(image_id)
            continue
        kept.append(row)
        patch_counts.append(n_patches)
        per_split_kept[split] = per_split_kept.get(split, 0) + 1

    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(kept)

    summary = {
        "manifest": str(manifest),
        "features_root": str(features_root),
        "out": str(out),
        "n_total": len(rows),
        "n_kept": len(kept),
        "n_no_coords": len(no_coords),
        "n_missing_file": len(no_file),
        "per_split_kept": per_split_kept,
        "per_split_total": per_split_total,
        "missing_coords_ids": no_coords,
        "missing_file_ids": no_file,
    }

    if verbose:
        drop = len(rows) - len(kept)
        print(f"{manifest} -> {out}")
        print(f"  kept {len(kept)}/{len(rows)} slides ({drop} dropped: {len(no_coords)} no coords, {len(no_file)} missing file)")
        for split in sorted(per_split_total):
            print(f"    {split:<6} {per_split_kept.get(split, 0):>4} / {per_split_total[split]:<4} kept")
        if patch_counts:
            print(f"  patches per slide: min {min(patch_counts)}, max {max(patch_counts)}")
        if no_file:
            print(f"  WARNING: {len(no_file)} slides have no feature file at all -- check --features-root")
        if no_coords:
            print("  no-coords slides (cannot be heatmapped, still used for training/eval):")
            for image_id in no_coords:
                print(f"    {image_id}")
    return summary


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Manifest restricted to slides with patch coordinates.")
    parser.add_argument("--manifest", default="data/common_split.csv")
    parser.add_argument("--features-root", default="data/features")
    parser.add_argument("--out", default="data/common_split_with_coords.csv")
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args(argv)

    try:
        summary = build_manifest(
            manifest=args.manifest,
            features_root=args.features_root,
            out=args.out,
            verbose=not args.quiet,
        )
    except (FileNotFoundError, ValueError) as error:
        print(f"ERROR: {error}", file=sys.stderr)
        return 1

    if summary["n_kept"] == 0:
        print("ERROR: no slide has patch coordinates -- nothing written that the exporter could use.", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

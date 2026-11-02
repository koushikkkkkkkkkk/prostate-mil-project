"""Verify that the data this project needs is actually on disk.

A fresh clone contains **no** features and **no** patches -- GitHub rejects files
over 100 MB and the release is ~600 MB per encoder, so the data lives outside git
(HuggingFace + Google Drive). That makes "did the download work?" a question
nobody could answer without guessing, which is how people end up training on a
silently-truncated cohort.

This script answers it in one command: for every manifest it finds, it reports how
many slides have a feature file, what the feature dimension is, whether that
dimension matches the config that will consume it, and whether coordinates exist
for the heatmap step. Pass --manifest to check one cohort (exits non-zero if that
cohort is incomplete) or --require all to fail on any gap.

Usage::

    python scripts/verify_data.py                       # check everything present
    python scripts/verify_data.py --features data/features --manifest data/common_split.csv
    python scripts/verify_data.py --require all         # non-zero exit on any gap
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

#: manifest -> (encoder, expected feature dim, expected rows, expected features root)
COHORTS: Tuple[Tuple[str, str, int, int, str], ...] = (
    ("data/common_split.csv", "ResNet50", 1024, 420, "data/features"),
    ("data/common_split_chief.csv", "CHIEF", 768, 399, "data/features_chief"),
    ("data/common_split_gigap.csv", "GigaPath", 1536, 391, "data/features_gigap"),
    ("data/common_split_with_coords.csv", "ResNet50 (coords only)", 1024, 399, "data/features"),
)

SUFFIXES = (".pt", ".pth", ".npy", ".npz", ".h5")


# --------------------------------------------------------------------------- #
# Inspection
# --------------------------------------------------------------------------- #
def find_feature(features_root: Path, image_id: str) -> Optional[Path]:
    for suffix in SUFFIXES:
        candidate = features_root / f"{image_id}{suffix}"
        if candidate.is_file():
            return candidate
    return None


def inspect_feature(path: Path) -> Tuple[Optional[int], bool]:
    """``(dim, has_coords)`` for one feature file; ``(None, False)`` if unreadable."""
    import h5py

    if path.suffix in (".h5", ".hdf5"):
        try:
            with h5py.File(path, "r") as handle:
                dim = int(handle["features"].shape[1]) if "features" in handle else None
                return dim, "coords" in handle
        except Exception:  # noqa: BLE001
            return None, False
    try:
        import torch

        payload = torch.load(path, map_location="cpu", weights_only=False)
        tensor = payload.get("features") if isinstance(payload, dict) else payload
        dim = int(tensor.shape[1]) if getattr(tensor, "ndim", 0) == 2 else None
        has_coords = isinstance(payload, dict) and payload.get("coords") is not None
        return dim, has_coords
    except Exception:  # noqa: BLE001
        return None, False


def check_cohort(manifest: Path, encoder: str, expected_dim: int, expected_rows: int, features_root: Path) -> Dict[str, Any]:
    import csv

    from collections import Counter

    if not manifest.is_file():
        return {"manifest": str(manifest), "status": "missing manifest"}
    with manifest.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))

    per_split = Counter(row.get("split", "?") for row in rows)
    present, missing, dims, with_coords = 0, [], Counter(), 0
    for row in rows:
        image_id = (row.get("image_id") or "").strip()
        path = find_feature(features_root, image_id)
        if path is None:
            missing.append(image_id)
            continue
        present += 1
        dim, has_coords = inspect_feature(path)
        if dim is not None:
            dims[dim] += 1
        with_coords += int(bool(has_coords))

    dominant = dims.most_common(1)[0][0] if dims else None
    problems: List[str] = []
    if len(rows) < expected_rows:
        problems.append(f"only {len(rows)} rows (expected {expected_rows})")
    if present < len(rows):
        problems.append(f"{len(missing)} of {len(rows)} slides have no feature file")
    if dominant is not None and dominant != expected_dim:
        problems.append(f"feature dim {dominant} != {expected_dim} expected for {encoder}")
    if len(dims) > 1:
        problems.append(f"mixed feature dims on disk: {dict(dims)}")

    return {
        "manifest": str(manifest),
        "encoder": encoder,
        "expected_dim": expected_dim,
        "rows": len(rows),
        "per_split": dict(per_split),
        "features_root": str(features_root),
        "present": present,
        "missing": len(missing),
        "first_missing": missing[:3],
        "feature_dim": dominant,
        "with_coords": with_coords,
        "status": "ok" if not problems else "INCOMPLETE",
        "problems": problems,
    }


# --------------------------------------------------------------------------- #
# Reporting
# --------------------------------------------------------------------------- #
def render(results: Sequence[Dict[str, Any]]) -> str:
    lines = [
        "cohort                        encoder                    rows  present  missing  dim  coords  status",
        "-" * 101,
    ]
    for entry in results:
        if "rows" not in entry:
            lines.append(f"{entry['manifest']:<29} {'-':<26} {'-':>4}  {'-':>7}  {'-':>7}  {'-':>3}  {'-':>6}  MISSING MANIFEST")
            continue
        lines.append(
            f"{Path(entry['manifest']).name:<29} {entry['encoder']:<26} {entry['rows']:>4}  "
            f"{entry['present']:>7}  {entry['missing']:>7}  {str(entry['feature_dim'] or '?'):>3}  "
            f"{entry['with_coords']:>6}  {entry['status']}"
        )
    return "\n".join(lines)


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Verify feature/manifest coverage before training.")
    parser.add_argument("--manifest", default=None, help="check a single manifest instead of all known cohorts")
    parser.add_argument("--features", default=None, help="feature root for --manifest")
    parser.add_argument("--dim", type=int, default=None, help="expected feature dim for --manifest")
    parser.add_argument("--require", choices=["none", "all"], default="none",
                        help="'all' exits non-zero unless every cohort is complete")
    parser.add_argument("--json", default=None, help="also write the report to this path")
    args = parser.parse_args(argv)

    if args.manifest:
        cohorts = ((args.manifest, "custom", args.dim or 1024, 0, args.features or "data/features"),)
    else:
        cohorts = COHORTS

    results: List[Dict[str, Any]] = []
    for manifest, encoder, dim, rows, features_root in cohorts:
        results.append(
            check_cohort(ROOT / manifest, encoder, dim, rows, ROOT / features_root)
        )

    print(render(results))
    print()

    incomplete = [entry for entry in results if entry.get("status") != "ok"]
    if incomplete:
        print("Problems:")
        for entry in incomplete:
            if "problems" not in entry:
                print(f"  {entry['manifest']}: missing")
                continue
            for problem in entry["problems"]:
                print(f"  {Path(entry['manifest']).name}: {problem}")
            if entry.get("first_missing"):
                print(f"      e.g. {', '.join(entry['first_missing'])}")
        print()
        print("Fix, per encoder (see README 'Get the data'):")
        print("  ResNet50 : python scripts/download_features_and_coords.py")
        print("  CHIEF    : python scripts/download_alt_encoder_features.py --encoder chief")
        print("  GigaPath : python scripts/download_alt_encoder_features.py --encoder gigap")
    else:
        print("All cohorts complete: every manifest row has a feature file with the expected dimension.")

    if args.json:
        import json

        out = ROOT / args.json if not Path(args.json).is_absolute() else Path(args.json)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(results, indent=2), encoding="utf-8")
        print(f"wrote {out}")

    if args.require == "all" and incomplete:
        return 1
    if incomplete and args.manifest:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

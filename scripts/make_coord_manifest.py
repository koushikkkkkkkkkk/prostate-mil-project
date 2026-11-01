"""Build a coord-filtered copy of the shared manifest (for heatmaps/localisation only).

21 of the 420 PANDA slides have no coordinates in the public feature release, so
their data/features/{id}.pt contains only "features". Training does not care, but
the heatmap exporters raise on the first coord-less slide and kill the whole run.

This writes a filtered manifest for the *visualisation* steps only:

    data/common_split_with_coords.csv     <- pass this to --render-images / --masks
    results/metrics/coord_coverage.csv    <- per-slide reason

data/common_split.csv is never modified: all 420 slides still train and evaluate
exactly as agreed. Owner: Person 4.
"""

from __future__ import annotations

import argparse
import csv
from collections import Counter
from pathlib import Path
from typing import Dict, List, Tuple

import torch


def inspect_feature_file(path: Path) -> Tuple[bool, bool, str]:
    """Return (has_coords, has_patch_paths, reason)."""
    if not path.exists():
        return False, False, "missing feature file"
    try:
        payload = torch.load(path, map_location="cpu", weights_only=False)
    except Exception as exc:
        return False, False, f"unreadable ({type(exc).__name__})"
    if not isinstance(payload, dict):
        return False, False, "bare tensor (features only)"
    has_coords = payload.get("coords") is not None
    has_paths = payload.get("patch_paths") is not None
    return has_coords, has_paths, "ok" if has_coords else "no coords key"


def main(argv: List[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Filter the manifest to renderable slides.")
    parser.add_argument("--manifest", default="data/common_split.csv")
    parser.add_argument("--features-root", default="data/features")
    parser.add_argument("--out", default="data/common_split_with_coords.csv")
    parser.add_argument("--require-patch-paths", action="store_true")
    parser.add_argument("--report", default="results/metrics/coord_coverage.csv")
    args = parser.parse_args(argv)

    manifest = Path(args.manifest)
    if not manifest.exists():
        print(f"ERROR: manifest not found: {manifest}")
        return 1
    features_root = Path(args.features_root)

    with manifest.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        fieldnames = reader.fieldnames or []
        rows = list(reader)

    kept, dropped = [], []
    for row in rows:
        image_id = (row.get("image_id") or row.get("slide_id") or "").strip()
        has_coords, has_paths, reason = inspect_feature_file(features_root / f"{image_id}.pt")
        ok = has_coords and (has_paths or not args.require_patch_paths)
        if ok:
            kept.append(row)
        else:
            dropped.append((row, reason))

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(kept)

    kept_by_split: Dict[str, int] = Counter(row.get("split", "?") for row in kept)
    dropped_by_split: Dict[str, int] = Counter(row.get("split", "?") for row, _ in dropped)

    print(f"manifest          : {manifest} ({len(rows)} slides)")
    print(f"features root     : {features_root}")
    print(f"with coords       : {len(kept)}/{len(rows)}  ({100.0 * len(kept) / max(len(rows), 1):.1f}%)")
    print(f"written           : {out_path}")
    print()
    print("| split | kept | dropped | kept % |")
    print("| --- | --- | --- | --- |")
    for split in ("train", "val", "test"):
        total = len([r for r in rows if r.get("split") == split])
        keep = kept_by_split.get(split, 0)
        drop = dropped_by_split.get(split, 0)
        pct = 100.0 * keep / total if total else 0.0
        print(f"| {split} | {keep} | {drop} | {pct:.1f}% |")

    if dropped:
        print()
        print("dropped by provider:", dict(Counter(r.get("data_provider", "?") for r, _ in dropped)))
        print("dropped by ISUP    :", dict(sorted(Counter(r.get("isup_grade", "?") for r, _ in dropped).items())))
        print("reasons            :", dict(Counter(reason for _, reason in dropped)))
        report_path = Path(args.report)
        report_path.parent.mkdir(parents=True, exist_ok=True)
        with report_path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(
                handle, fieldnames=["image_id", "split", "data_provider", "isup_grade", "reason"]
            )
            writer.writeheader()
            for row, reason in dropped:
                writer.writerow({
                    "image_id": row.get("image_id"),
                    "split": row.get("split"),
                    "data_provider": row.get("data_provider"),
                    "isup_grade": row.get("isup_grade"),
                    "reason": reason,
                })
        print(f"per-slide details  : {report_path}")

    print()
    print("Use the filtered manifest ONLY for heatmaps/localisation:")
    print(f"  --manifest {out_path}")
    print(f"Train/evaluate on the real lock: {manifest}")
    return 0 if kept else 1


if __name__ == "__main__":
    raise SystemExit(main())
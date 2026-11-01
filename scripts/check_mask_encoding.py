"""Print the pixel-value histogram of PANDA label masks, so the encoding can be
confirmed before any patch labels are trusted. Builds nothing, writes nothing.

    python scripts/check_mask_encoding.py --masks <dir> --mask-subdir train_label_masks
"""
from __future__ import annotations

import argparse
import csv
from collections import Counter
from pathlib import Path

import numpy as np
from PIL import Image


def find_mask(root: Path, image_id: str, subdir: str | None):
    dirs = ([root / subdir] if subdir else []) + [root, root / "train_label_masks"]
    for d in dirs:
        for name in (f"{image_id}_mask.png", f"{image_id}.png",
                     f"{image_id}_mask.tiff", f"{image_id}.tiff"):
            if (d / name).is_file():
                return d / name
    return None


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--masks", required=True)
    ap.add_argument("--mask-subdir", default=None)
    ap.add_argument("--split", default="data/common_split.csv")
    ap.add_argument("--splits", default="train,val")
    ap.add_argument("--inspect", type=int, default=20)
    ap.add_argument("--isup-min", type=int, default=None,
                    help="only inspect slides at or above this ISUP grade")
    ap.add_argument("--stratify", action="store_true",
                    help="spread the inspection evenly across ISUP grades (recommended)")
    args = ap.parse_args()

    wanted = {s.strip() for s in args.splits.split(",") if s.strip()}
    with Path(args.split).open(newline="", encoding="utf-8") as handle:
        slides = [r for r in csv.DictReader(handle) if r["split"] in wanted]

    if args.isup_min is not None:
        slides = [r for r in slides if int(r["isup_grade"]) >= args.isup_min]

    if args.stratify and slides:
        by_grade: dict = {}
        for row in slides:
            by_grade.setdefault(int(row["isup_grade"]), []).append(row)
        per_grade = max(1, args.inspect // max(len(by_grade), 1))
        chosen = []
        for grade in sorted(by_grade):
            chosen.extend(by_grade[grade][:per_grade])
        slides = chosen

    root = Path(args.masks)
    pooled: Counter = Counter()
    maxima, shown, missing = [], 0, 0
    grade_stats: dict = {}
    for row in slides:
        if shown >= args.inspect:
            break
        path = find_mask(root, row["image_id"], args.mask_subdir)
        if path is None:
            missing += 1
            continue
        with Image.open(path) as im:
            array = np.asarray(im.convert("L"), dtype=np.uint8)
        values, counts = np.unique(array, return_counts=True)
        fractions = {int(v): c / array.size for v, c in zip(values, counts)}
        pooled.update({int(v): int(c) for v, c in zip(values, counts)})
        maxima.append(int(values.max()))
        shown += 1
        grade = int(row["isup_grade"])
        seen_rows, seen_values = grade_stats.get(grade, (0, set()))
        grade_stats[grade] = (seen_rows + 1, seen_values | set(int(v) for v in values))
        top = "  ".join(f"{v}:{fractions[v]:.2%}" for v in sorted(fractions, key=lambda k: -fractions[k])[:6])
        print(f"{row['image_id']}  {row['data_provider']:<11} isup={row['isup_grade']}  {path.name}")
        print(f"    {top}")

    if not shown:
        print("No masks found -- check --masks / --mask-subdir", flush=True)
        return 1

    total = sum(pooled.values())
    print("\npooled pixel fractions across inspected masks:")
    for value in sorted(pooled):
        print(f"    value {value}: {pooled[value] / total:7.2%}")
    print("\nper-grade summary:")
    for grade in sorted(grade_stats):
        rows_seen, values_seen = grade_stats[grade]
        label = ", ".join(f"{v}" for v in sorted(values_seen)) or "none"
        print(f"    isup={grade}: {rows_seen} slides, distinct values seen: {label}")

    print(f"\nslides inspected   {shown}   (missing masks: {missing})")
    print(f"max value seen     {max(maxima)}   (per-slide {min(maxima)}..{max(maxima)})")

    seen = set(pooled)
    expected = {0, 1, 2, 3, 4, 5}
    if not seen & {3, 4, 5}:
        print("\nWARNING: no Gleason pixels (3/4/5) found. Either these slides have")
        print("         no annotated tumour, or Karolinska masks use a different")
        print("         encoding than Radboud's 0-5. Do NOT build patch labels yet.")
    if seen - expected:
        print(f"\nWARNING: unexpected values {sorted(seen - expected)} -- encoding differs from docs.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

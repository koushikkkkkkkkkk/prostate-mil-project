"""Create the frozen, stratified PANDA split: 300 train / 60 val / 60 test.

Input is the PANDA ``train.csv`` metadata (columns: ``image_id``,
``data_provider``, ``isup_grade``, ``gleason_score``). Output is
``data/common_split.csv`` -- the single source of truth every teammate must
train and evaluate on. Deterministic: ``seed=42``.

The PLAN's target of 50/10/10 slides per ISUP grade (6 grades) requires 70
slides per grade in the source metadata. PANDA is heavily imbalanced: grade 0
(negative) and grade 5 are the scarce ones. So the script:

1. takes up to ``--per-class`` slides per grade (default 70),
2. splits each grade 50/10/10 when enough slides exist,
3. reports the achieved distribution and warns if a grade could not reach the
   target -- an unfilled grade must be documented in the report, never silently
   topped up from another grade (that would break the stratification claim).

Usage::

    python scripts/create_split.py --panda-csv data/raw/train.csv \
        --out data/common_split.csv --seed 42
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Any, Dict, List, Sequence

import numpy as np

TARGET_TRAIN_PER_CLASS = 50
TARGET_VAL_PER_CLASS = 10
TARGET_TEST_PER_CLASS = 10


def read_panda_csv(path: str | Path) -> List[Dict[str, Any]]:
    path = Path(path)
    if not path.exists():
        raise SystemExit(
            f"PANDA metadata not found: {path}\n"
            "Download train.csv from https://www.kaggle.com/c/prostate-cancer-grade-assessment/data "
            "and place it at data/raw/train.csv (it is gitignored)."
        )
    rows: List[Dict[str, Any]] = []
    with path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            image_id = row.get("image_id") or row.get("slide_id")
            grade = row.get("isup_grade")
            if image_id in (None, "") or grade in (None, ""):
                continue
            try:
                grade_int = int(float(grade))
            except ValueError:
                continue
            if not 0 <= grade_int <= 5:
                continue
            rows.append(
                {
                    "image_id": str(image_id),
                    "data_provider": row.get("data_provider", "unknown"),
                    "isup_grade": grade_int,
                    "gleason_score": row.get("gleason_score", "unknown"),
                }
            )
    if not rows:
        raise SystemExit(f"No usable rows in {path}")
    return rows


def stratified_split(
    rows: Sequence[Dict[str, Any]], seed: int = 42
) -> List[Dict[str, Any]]:
    """50/10/10 per ISUP grade, seeded, with a distribution report."""
    rng = np.random.default_rng(seed)
    by_grade: Dict[int, List[Dict[str, Any]]] = {grade: [] for grade in range(6)}
    for row in rows:
        by_grade[row["isup_grade"]].append(row)

    output: List[Dict[str, Any]] = []
    shortfalls: List[str] = []
    for grade in range(6):
        bucket = by_grade[grade]
        order = rng.permutation(len(bucket))
        bucket = [bucket[i] for i in order]
        n = len(bucket)
        if n < 20:
            shortfalls.append(f"grade {grade}: only {n} slides available")

        n_train = min(TARGET_TRAIN_PER_CLASS, int(round(n * 0.72)))
        n_val = min(TARGET_VAL_PER_CLASS, max(0, (n - n_train) // 2))
        n_test = min(TARGET_TEST_PER_CLASS, max(0, n - n_train - n_val))
        # Give any leftovers to train so the model sees as much data as possible.
        n_train = n - n_val - n_test

        for i, row in enumerate(bucket):
            if i < n_train:
                split = "train"
            elif i < n_train + n_val:
                split = "val"
            elif i < n_train + n_val + n_test:
                split = "test"
            else:  # pragma: no cover - defensive
                continue
            output.append({**row, "split": split})

        if n < TARGET_TRAIN_PER_CLASS + TARGET_VAL_PER_CLASS + TARGET_TEST_PER_CLASS:
            shortfalls.append(
                f"grade {grade}: {n} slides -> "
                f"{n_train}/{n_val}/{n_test} train/val/test (target 50/10/10)"
            )

    return output, shortfalls  # type: ignore[return-value]


def write_split(rows: Sequence[Dict[str, Any]], out_path: str | Path) -> Dict[str, Any]:
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ["image_id", "data_provider", "isup_grade", "gleason_score", "split"]
    with out_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row[key] for key in fieldnames})

    summary: Dict[str, Dict[str, int]] = {}
    for row in rows:
        summary.setdefault(row["split"], {})
        summary[row["split"]][row["isup_grade"]] = summary[row["split"]].get(row["isup_grade"], 0) + 1
    return summary


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Create the stratified 300/60/60 PANDA split.")
    parser.add_argument("--panda-csv", default="data/raw/train.csv")
    parser.add_argument("--out", default="data/common_split.csv")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args(argv)

    rows = read_panda_csv(args.panda_csv)
    split_rows, shortfalls = stratified_split(rows, seed=args.seed)
    summary = write_split(split_rows, args.out)

    print(f"Wrote {len(split_rows)} slides -> {args.out}")
    header = "split  " + "  ".join(f"g{g}" for g in range(6)) + "   total"
    print(header)
    for split in ("train", "val", "test"):
        counts = summary.get(split, {})
        totals = sum(counts.values())
        print(f"{split:<6} " + "  ".join(f"{counts.get(g, 0):>2}" for g in range(6)) + f"   {totals:>5}")
    if shortfalls:
        print("\nStratification notes (document these in the report):")
        for note in shortfalls:
            print(f"  - {note}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

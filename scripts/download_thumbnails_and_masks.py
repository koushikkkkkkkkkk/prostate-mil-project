"""Download real PANDA slide thumbnails and generate binary tissue masks.

Uses Kaggle's open 512x512 PANDA dataset (xhlulu/panda-resized-train-data-512x512)
via kagglehub, filters only the 420 slides in common_split.csv (~35 MB total),
and generates matching binary tissue masks using Otsu thresholding.

Usage::

    # In Google Colab / Cloud (recommended):
    python scripts/download_thumbnails_and_masks.py \\
        --manifest data/common_split.csv \\
        --thumbnails-dir /content/drive/MyDrive/prostate_mil_data/thumbnails \\
        --masks-dir /content/drive/MyDrive/prostate_mil_data/masks
"""

from __future__ import annotations

import argparse
import csv
import shutil
import sys
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.preprocessing import detect_tissue_mask


def prepare_thumbnails_and_masks(
    manifest_path: Path,
    thumbnails_dir: Path,
    masks_dir: Path,
    force: bool = False,
) -> tuple[int, int]:
    thumbnails_dir.mkdir(parents=True, exist_ok=True)
    masks_dir.mkdir(parents=True, exist_ok=True)

    slide_ids: list[str] = []
    with open(manifest_path, mode="r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            sid = row.get("image_id") or row.get("slide_id")
            if sid and sid not in slide_ids:
                slide_ids.append(str(sid).strip())

    existing_thumbs = [sid for sid in slide_ids if (thumbnails_dir / f"{sid}.png").exists()]
    if len(existing_thumbs) == len(slide_ids) and not force:
        print(f"✓ All {len(slide_ids)} thumbnails already present in {thumbnails_dir}")
    else:
        print(f"Fetching 512x512 PANDA slide images via kagglehub...")
        try:
            import kagglehub
            dataset_dir = Path(kagglehub.dataset_download("xhlulu/panda-resized-train-data-512x512"))
        except Exception as exc:
            print(f"Error downloading from kagglehub: {exc}", file=sys.stderr)
            print("Please ensure kagglehub is installed: pip install kagglehub", file=sys.stderr)
            return 0, 0

        print(f"Dataset extracted to: {dataset_dir}. Copying our {len(slide_ids)} slides...")
        # Search for images in train_images or root of the downloaded dataset
        image_search_dirs = [dataset_dir / "train_images", dataset_dir]
        found_count = 0

        for sid in slide_ids:
            dest = thumbnails_dir / f"{sid}.png"
            if dest.exists() and not force:
                found_count += 1
                continue

            for sdir in image_search_dirs:
                candidate = sdir / f"{sid}.png"
                if candidate.exists():
                    shutil.copy2(candidate, dest)
                    found_count += 1
                    break
                candidate_jpg = sdir / f"{sid}.jpg"
                if candidate_jpg.exists():
                    with Image.open(candidate_jpg) as img:
                        img.convert("RGB").save(dest)
                    found_count += 1
                    break

        print(f"✓ Copied {found_count} / {len(slide_ids)} slide thumbnails to {thumbnails_dir}")

    # Generate matching tissue masks using Otsu thresholding
    print("Generating binary tissue masks for all thumbnails...")
    mask_count = 0
    for sid in slide_ids:
        thumb_path = thumbnails_dir / f"{sid}.png"
        mask_path = masks_dir / f"{sid}.png"
        if mask_path.exists() and not force:
            mask_count += 1
            continue

        if not thumb_path.exists():
            continue

        with Image.open(thumb_path) as thumb:
            mask = detect_tissue_mask(thumb, min_saturation=15, invert=True)
            # Save binary mask (0 = background, 255 = tissue)
            mask_img = Image.fromarray((mask * 255).astype(np.uint8))
            mask_img.save(mask_path)
            mask_count += 1

    print(f"✓ Successfully verified {mask_count} / {len(slide_ids)} tissue masks in {masks_dir}")
    return len(existing_thumbs), mask_count


def main() -> int:
    parser = argparse.ArgumentParser(description="Download PANDA thumbnails and generate masks.")
    parser.add_argument("--manifest", default="data/common_split.csv", help="Manifest path")
    parser.add_argument(
        "--thumbnails-dir",
        default="data/thumbnails",
        help="Target folder for thumbnails (default: data/thumbnails)",
    )
    parser.add_argument(
        "--masks-dir",
        default="data/masks",
        help="Target folder for masks (default: data/masks)",
    )
    parser.add_argument("--force", action="store_true", help="Force re-generation")
    args = parser.parse_args()

    n_thumbs, n_masks = prepare_thumbnails_and_masks(
        manifest_path=Path(args.manifest),
        thumbnails_dir=Path(args.thumbnails_dir),
        masks_dir=Path(args.masks_dir),
        force=args.force,
    )
    return 0 if (n_thumbs > 0 and n_masks > 0) else 1


if __name__ == "__main__":
    raise SystemExit(main())

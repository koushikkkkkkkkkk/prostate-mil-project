"""Download pre-extracted real ResNet50 features for the 420 PANDA slides.

Downloads features directly from the Hugging Face computational pathology repository
(Dearcat/CPathPatchFeature) into data/features/{image_id}.pt.

Total download: ~600 MB (instead of downloading 180 GB raw WSI slides).
Uses multi-threaded parallel downloads with progress reporting.

Usage::

    python scripts/download_real_features.py
"""

from __future__ import annotations

import argparse
import csv
import os
import sys
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

try:
    from tqdm import tqdm
except ImportError:
    class tqdm:  # type: ignore[no-redef]
        def __init__(self, total=0, desc="", unit=""):
            self.total = total
            self.count = 0
            self.desc = desc
        def update(self, n=1):
            self.count += n
            if self.count % 20 == 0 or self.count == self.total:
                print(f"{self.desc}: {self.count}/{self.total}")
        def __enter__(self):
            return self
        def __exit__(self, *args):
            pass
        @staticmethod
        def write(msg):
            print(msg)

HF_BASE_URL = (
    "https://huggingface.co/datasets/Dearcat/CPathPatchFeature/resolve/main/panda/r50/pt_files"
)


def download_slide_feature(image_id: str, out_dir: Path, timeout: int = 30) -> tuple[str, bool, str]:
    target_path = out_dir / f"{image_id}.pt"
    if target_path.exists() and target_path.stat().st_size > 1024:
        return image_id, True, "cached"

    url = f"{HF_BASE_URL}/{image_id}.pt"
    temp_path = out_dir / f"{image_id}.pt.tmp"

    try:
        req = urllib.request.Request(
            url,
            headers={"User-Agent": "Mozilla/5.0 (compatible; PANDADownloader/1.0)"},
        )
        with urllib.request.urlopen(req, timeout=timeout) as response, open(temp_path, "wb") as out_file:
            out_file.write(response.read())

        temp_path.rename(target_path)
        return image_id, True, "downloaded"
    except Exception as exc:
        if temp_path.exists():
            temp_path.unlink()
        return image_id, False, str(exc)


def main() -> int:
    parser = argparse.ArgumentParser(description="Download real PANDA ResNet50 features.")
    parser.add_argument(
        "--manifest",
        default="data/common_split.csv",
        help="Path to common_split.csv manifest (default: data/common_split.csv)",
    )
    parser.add_argument(
        "--out-dir",
        default="data/features",
        help="Output directory for .pt files (default: data/features)",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=8,
        help="Number of concurrent download threads (default: 8)",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Optional limit on number of slides to download (for testing)",
    )
    args = parser.parse_args()

    manifest_path = Path(args.manifest)
    if not manifest_path.exists():
        print(f"Error: Manifest not found at {manifest_path}", file=sys.stderr)
        return 1

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    slide_ids: list[str] = []
    with open(manifest_path, mode="r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            sid = row.get("image_id") or row.get("slide_id")
            if sid and sid not in slide_ids:
                slide_ids.append(str(sid).strip())
    if args.limit:
        slide_ids = slide_ids[: args.limit]

    print(f"Downloading real ResNet50 features for {len(slide_ids)} slides to {out_dir}...")
    print("Using 8 concurrent workers. Total size is ~600 MB.")

    success_count = 0
    fail_count = 0

    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = {
            executor.submit(download_slide_feature, sid, out_dir): sid for sid in slide_ids
        }
        with tqdm(total=len(slide_ids), desc="Downloading features", unit="slide") as pbar:
            for future in as_completed(futures):
                sid, success, msg = future.result()
                if success:
                    success_count += 1
                else:
                    fail_count += 1
                    tqdm.write(f"Failed {sid}: {msg}")
                pbar.update(1)

    print(f"\nDownload complete: {success_count}/{len(slide_ids)} succeeded, {fail_count} failed.")
    return 0 if fail_count == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())

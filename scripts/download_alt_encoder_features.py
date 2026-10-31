"""Download alternative pathology foundation model features (CHIEF, GigaPath, UNI).

Hugging Face source: Dearcat/CPathPatchFeature/panda/{encoder}/pt_files/{image_id}.pt

Usage::

    python scripts/download_alt_encoder_features.py --encoder chief
    python scripts/download_alt_encoder_features.py --encoder gigap
    python scripts/download_alt_encoder_features.py --encoder uni
"""

from __future__ import annotations

import argparse
import csv
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
    "https://huggingface.co/datasets/Dearcat/CPathPatchFeature/resolve/main/panda/{encoder}/pt_files"
)


def download_slide(
    image_id: str,
    out_dir: Path,
    encoder: str,
    timeout: int = 30,
) -> tuple[str, bool, str]:
    target_path = out_dir / f"{image_id}.pt"
    if target_path.exists() and target_path.stat().st_size > 1024:
        return image_id, True, "cached"

    url = f"{HF_BASE_URL.format(encoder=encoder)}/{image_id}.pt"
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
    parser = argparse.ArgumentParser(
        description="Download alternative encoder features (CHIEF, GigaPath, UNI)."
    )
    parser.add_argument(
        "--encoder",
        choices=["chief", "gigap", "uni", "r50"],
        default="chief",
        help="Feature encoder to download (default: chief)",
    )
    parser.add_argument(
        "--manifest",
        default="data/common_split.csv",
        help="Path to manifest (default: data/common_split.csv)",
    )
    parser.add_argument(
        "--out-dir",
        default=None,
        help="Output directory (default: data/features_{encoder})",
    )
    parser.add_argument(
        "--workers",
        type=int,
        nargs="?",
        const=8,
        default=8,
        help="Number of concurrent download threads (default: 8)",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Optional limit on number of slides to download",
    )
    args = parser.parse_args()

    manifest_path = Path(args.manifest)
    if not manifest_path.exists():
        print(f"Error: Manifest not found at {manifest_path}", file=sys.stderr)
        return 1

    out_dir = Path(args.out_dir or f"data/features_{args.encoder}")
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

    print(f"Downloading {args.encoder.upper()} features for {len(slide_ids)} slides to {out_dir}...")
    success_count = 0
    fail_count = 0

    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = {
            executor.submit(download_slide, sid, out_dir, args.encoder): sid for sid in slide_ids
        }
        with tqdm(total=len(slide_ids), desc=f"Downloading {args.encoder}", unit="slide") as pbar:
            for future in as_completed(futures):
                sid, success, msg = future.result()
                if success:
                    success_count += 1
                else:
                    fail_count += 1
                pbar.update(1)

    print(f"\nDownload complete: {success_count}/{len(slide_ids)} succeeded, {fail_count} failed.")
    return 0 if fail_count == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())

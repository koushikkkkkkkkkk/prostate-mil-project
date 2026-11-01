"""Download real PANDA ResNet50 features AND their patch coordinates.

Why this script exists
----------------------
``scripts/download_real_features.py`` downloads only the feature tensors
(``data/features/{image_id}.pt`` with a bare ``[N, D]`` tensor). That is enough
to *train* every MIL model, but Person 4's work (notebook 06) needs patch
``(x, y)`` origins for two of its three deliverables:

* attention heatmaps / localisation overlap  -> needs ``coords``
* patch-CNN probability maps                 -> needs ``coords``

The upstream Hugging Face repo (``Dearcat/CPathPatchFeature``) ships the patch
coordinates as a separate per-slide HDF5 file::

    panda/patches/{image_id}.h5      ->  dataset "coords", int64 [N, 2]

This script downloads both, verifies that the row counts agree, and rewrites
``data/features/{image_id}.pt`` in the *preferred* contract format::

    {"features": [N, D] float32, "coords": [N, 2] float32}

Slides whose ``.h5`` is missing (a handful are absent upstream) are still
written, but without ``coords``; the script prints which ones so the report can
state how many slides were usable for localisation analysis.

Usage::

    python scripts/download_features_and_coords.py                 # all manifest slides
    python scripts/download_features_and_coords.py --limit 20      # quick test
    python scripts/download_features_and_coords.py --skip-coords   # features only

Total download: ~600-800 MB. Re-running is cheap: existing files are reused.
"""

from __future__ import annotations

import argparse
import csv
import sys
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import List, Optional, Tuple

HF_BASE = "https://huggingface.co/datasets/Dearcat/CPathPatchFeature/resolve/main/panda"
FEATURE_URL = HF_BASE + "/r50/pt_files/{image_id}.pt"
COORD_URL = HF_BASE + "/patches/{image_id}.h5"

try:
    from tqdm import tqdm
except ImportError:  # tiny fallback so the script works before pip install -r
    class tqdm:  # type: ignore[no-redef]
        def __init__(self, total=0, desc="", unit="", **kwargs):
            self.total, self.n, self.desc = total, 0, desc

        def update(self, k=1):
            self.n += k
            if self.n % 10 == 0 or self.n == self.total:
                print(f"{self.desc}: {self.n}/{self.total}")

        def write(self, msg):
            print(msg)

        def __enter__(self):
            return self

        def __exit__(self, *exc):
            return False


def _download(url: str, target: Path, timeout: int = 120) -> bool:
    """Download ``url`` to ``target`` atomically. Returns False on failure."""
    if target.exists() and target.stat().st_size > 1024:
        return True
    tmp = target.with_suffix(target.suffix + ".tmp")
    try:
        request = urllib.request.Request(url, headers={"User-Agent": "panda-mil/1.0"})
        with urllib.request.urlopen(request, timeout=timeout) as response, tmp.open("wb") as handle:
            while True:
                chunk = response.read(1 << 20)
                if not chunk:
                    break
                handle.write(chunk)
        tmp.replace(target)
        return True
    except Exception:
        if tmp.exists():
            tmp.unlink()
        return False


def _read_manifest_image_ids(manifest: Path) -> List[str]:
    ids: List[str] = []
    with manifest.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            image_id = (row.get("image_id") or row.get("slide_id") or "").strip()
            if image_id and image_id not in ids:
                ids.append(image_id)
    return ids


def _load_any_payload(path: Path):
    """Load a .pt that is either a bare tensor or a dict."""
    import torch

    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:  # torch < 2.0
        return torch.load(path, map_location="cpu")


def _merge_coords(feature_path: Path, coord_path: Path) -> Tuple[bool, str]:
    """Union of the feature payload and the h5 coords -> preferred .pt contract."""
    import h5py
    import torch

    payload = _load_any_payload(feature_path)
    if isinstance(payload, dict):
        features = payload.get("features", payload.get("feats", payload.get("embeddings")))
    else:
        features = payload
    if features is None:
        return False, "no feature tensor in payload"
    features = torch.as_tensor(features, dtype=torch.float32)

    with h5py.File(coord_path, "r") as handle:
        coords = torch.as_tensor(handle["coords"][:], dtype=torch.float32)

    if coords.shape[0] != features.shape[0]:
        return False, f"N mismatch: features {tuple(features.shape)} vs coords {tuple(coords.shape)}"

    out = {"features": features, "coords": coords}
    if isinstance(payload, dict) and payload.get("patch_paths") is not None:
        out["patch_paths"] = list(payload["patch_paths"])
    torch.save(out, feature_path)
    return True, f"N={features.shape[0]} D={features.shape[1]}"


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--manifest", default="data/common_split.csv")
    parser.add_argument("--out-dir", default="data/features")
    parser.add_argument("--coords-dir", default="data/coords_h5", help="where the raw .h5 files are cached")
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--limit", type=int, default=None, help="only the first N slides (smoke test)")
    parser.add_argument("--skip-coords", action="store_true", help="features only (training works, heatmaps do not)")
    args = parser.parse_args(argv)

    manifest = Path(args.manifest)
    if not manifest.exists():
        print(f"ERROR: manifest not found: {manifest}", file=sys.stderr)
        return 1
    try:
        import h5py  # noqa: F401
    except ModuleNotFoundError:
        print("ERROR: h5py is required for coordinates.  pip install h5py", file=sys.stderr)
        return 1

    out_dir = Path(args.out_dir)
    coord_dir = Path(args.coords_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    coord_dir.mkdir(parents=True, exist_ok=True)

    image_ids = _read_manifest_image_ids(manifest)
    if args.limit:
        image_ids = image_ids[: args.limit]
    print(f"{len(image_ids)} slides -> {out_dir}  (+ coords cached in {coord_dir})")

    # 1. features
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {
            pool.submit(_download, FEATURE_URL.format(image_id=i), out_dir / f"{i}.pt"): i
            for i in image_ids
        }
        missing_features: List[str] = []
        with tqdm(total=len(futures), desc="features", unit="slide") as bar:
            for future in as_completed(futures):
                if not future.result():
                    missing_features.append(futures[future])
                    tqdm.write(f"  missing feature: {futures[future]}")
                bar.update(1)

    if args.skip_coords:
        print("Skipping coordinates (--skip-coords).")
        return 0 if not missing_features else 1

    # 2. coordinates
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {
            pool.submit(_download, COORD_URL.format(image_id=i), coord_dir / f"{i}.h5"): i
            for i in image_ids
            if i not in missing_features
        }
        missing_coords: List[str] = []
        with tqdm(total=len(futures), desc="coords", unit="slide") as bar:
            for future in as_completed(futures):
                if not future.result():
                    missing_coords.append(futures[future])
                    tqdm.write(f"  missing coords: {futures[future]}")
                bar.update(1)

    # 3. merge into the preferred contract
    merged, failed = [], []
    for image_id in image_ids:
        feature_path = out_dir / f"{image_id}.pt"
        coord_path = coord_dir / f"{image_id}.h5"
        if not feature_path.exists() or not coord_path.exists():
            continue
        ok, message = _merge_coords(feature_path, coord_path)
        (merged if ok else failed).append(image_id if ok else f"{image_id} ({message})")
        if not ok:
            print(f"  merge failed {image_id}: {message}")

    print()
    print(f"features downloaded : {len(image_ids) - len(missing_features)}/{len(image_ids)}")
    print(f"coords downloaded   : {len(image_ids) - len(missing_features) - len(missing_coords)}/{len(image_ids)}")
    print(f"coords merged       : {len(merged)}")
    if missing_coords:
        print(f"slides WITHOUT coords (no heatmap/localisation, still trainable): {len(missing_coords)}")
        print("  " + ", ".join(sorted(missing_coords)[:20]) + (" ..." if len(missing_coords) > 20 else ""))
    if failed:
        print(f"merge failures: {len(failed)} -> {failed[:5]}")
    print()
    print("Percentages for the report: coords coverage = "
          f"{100.0 * len(merged) / max(len(image_ids), 1):.1f}% of the requested slides.")
    return 0 if merged else 1


if __name__ == "__main__":
    raise SystemExit(main())

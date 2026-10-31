"""Attach real patch coordinates to the downloaded PANDA feature bags.

Why this exists
---------------
``scripts/download_real_features.py`` pulls the pre-extracted ResNet50 features,
which are stored as a bare tensor ``[N, 1024]`` — **no coordinates**. Without
coordinates the attention weights cannot be placed on the slide, so heatmaps,
overlays and the localisation-overlap score (Person 2's deliverables 5-6) are
impossible.

The same Hugging Face dataset stores the patch origins separately:

    https://huggingface.co/datasets/Dearcat/CPathPatchFeature/resolve/main/panda/patches/{image_id}.h5
        HDF5 file with a single dataset "coords" of shape [N, 2] (x, y) in slide
        pixels, 256 px spacing -> matches the feature tensor row for row.

This script downloads that small file per slide and rewrites
``data/features/{image_id}.pt`` as::

    {"features": [N, 1024] float32, "coords": [N, 2] float32}

so every downstream tool (heatmaps, Streamlit app, top-patch figures) works
unchanged. It never modifies the features themselves and is safe to re-run.

Usage::

    python scripts/attach_patch_coords.py                       # all slides in the split
    python scripts/attach_patch_coords.py --limit 5             # smoke test
    python scripts/attach_patch_coords.py --verify-only         # report coverage
"""

from __future__ import annotations

import argparse
import csv
import sys
import tempfile
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import torch

HF_COORD_BASE = (
    "https://huggingface.co/datasets/Dearcat/CPathPatchFeature/resolve/main/panda/patches"
)


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #
def _h5py_available() -> bool:
    """h5py is only needed for this script; check it without a bare import."""
    import importlib.util

    return importlib.util.find_spec("h5py") is not None


def read_slide_ids(manifest: Path) -> List[str]:
    ids: List[str] = []
    with manifest.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            sid = (row.get("image_id") or row.get("slide_id") or "").strip()
            if sid and sid not in ids:
                ids.append(sid)
    return ids


def load_features_payload(path: Path) -> Tuple[torch.Tensor, Dict[str, Any]]:
    """Return (features, rest-of-payload) so extra keys survive the rewrite."""
    try:
        payload = torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:  # torch < 2.0
        payload = torch.load(path, map_location="cpu")

    if isinstance(payload, torch.Tensor):
        return payload.float(), {}
    if isinstance(payload, dict):
        key = next((k for k in ("features", "feats", "feature", "embeddings", "x") if k in payload), None)
        if key is None:
            raise KeyError(f"{path} dict has no feature key; keys={sorted(payload)}")
        features = payload[key]
        if not torch.is_tensor(features):
            features = torch.as_tensor(features)
        rest = {k: v for k, v in payload.items() if k != key}
        return features.float(), rest
    raise TypeError(f"unsupported payload type in {path}: {type(payload)!r}")


def download_coords(image_id: str, timeout: int = 60) -> torch.Tensor:
    """Fetch the coordinate HDF5 for one slide and return an ``[N, 2]`` tensor."""
    import h5py  # imported lazily so the rest of the repo does not depend on it

    url = f"{HF_COORD_BASE}/{image_id}.h5"
    request = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0 (compatible; PANDACoords/1.0)"})
    with urllib.request.urlopen(request, timeout=timeout) as response:
        data = response.read()

    with tempfile.NamedTemporaryFile(suffix=".h5", delete=False) as handle:
        handle.write(data)
        temp_path = Path(handle.name)
    try:
        with h5py.File(temp_path, "r") as h5:
            if "coords" not in h5:
                raise KeyError(f"{image_id}.h5 has no 'coords' dataset (keys={list(h5.keys())})")
            coords = h5["coords"][:]
    finally:
        temp_path.unlink(missing_ok=True)

    tensor = torch.as_tensor(coords).float()
    if tensor.ndim != 2 or tensor.size(1) < 2:
        raise ValueError(f"unexpected coords shape {tuple(tensor.shape)} for {image_id}")
    return tensor[:, :2]


def save_payload(path: Path, payload: Dict[str, Any]) -> None:
    """Atomic write so an interrupted run never leaves a corrupt .pt behind."""
    temp_path = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, temp_path)
    temp_path.replace(path)


# --------------------------------------------------------------------------- #
# per-slide work
# --------------------------------------------------------------------------- #
def attach_one(image_id: str, features_dir: Path, overwrite: bool = False) -> Tuple[str, str, str]:
    """Returns ``(image_id, status, detail)`` with status in ok/skip/fail."""
    feature_path = features_dir / f"{image_id}.pt"
    if not feature_path.exists():
        return image_id, "fail", "feature file missing (run download_real_features.py first)"

    try:
        features, rest = load_features_payload(feature_path)
    except Exception as exc:  # noqa: BLE001
        return image_id, "fail", f"cannot read features: {type(exc).__name__}: {exc}"

    existing = rest.get("coords")
    if existing is not None and not overwrite:
        return image_id, "skip", f"coords already present ({tuple(torch.as_tensor(existing).shape)})"

    try:
        coords = download_coords(image_id)
    except Exception as exc:  # noqa: BLE001
        return image_id, "fail", f"coords download failed: {type(exc).__name__}: {exc}"

    if coords.size(0) != features.size(0):
        return image_id, "fail", (
            f"N mismatch: {features.size(0)} features vs {coords.size(0)} coords "
            "(feature/coord files may be from different extractions)"
        )

    payload = {"features": features, "coords": coords, **rest}
    try:
        save_payload(feature_path, payload)
    except Exception as exc:  # noqa: BLE001
        return image_id, "fail", f"write failed: {type(exc).__name__}: {exc}"
    return image_id, "ok", f"{coords.size(0)} coords"


def verify_only(features_dir: Path, slide_ids: List[str]) -> int:
    """Report which feature bags can drive heatmaps."""
    with_coords, without_coords, missing = [], [], []
    for image_id in slide_ids:
        path = features_dir / f"{image_id}.pt"
        if not path.exists():
            missing.append(image_id)
            continue
        try:
            features, rest = load_features_payload(path)
        except Exception:  # noqa: BLE001
            missing.append(image_id)
            continue
        coords = rest.get("coords")
        if coords is None:
            without_coords.append(image_id)
        elif torch.as_tensor(coords).size(0) != features.size(0):
            without_coords.append(image_id)
        else:
            with_coords.append(image_id)

    print(f"features present with coords   : {len(with_coords)}")
    print(f"features present, NO coords    : {len(without_coords)}")
    print(f"feature files missing entirely : {len(missing)}")
    if without_coords:
        print("\nHeatmaps/localisation are blocked for the slides without coords.")
        print("Fix: python scripts/attach_patch_coords.py")
    if missing:
        print("\nMissing feature files: python scripts/download_real_features.py")
    return 0 if not without_coords and not missing else 1


# --------------------------------------------------------------------------- #
# main
# --------------------------------------------------------------------------- #
def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Attach PANDA patch coordinates to feature bags.")
    parser.add_argument("--manifest", default="data/common_split.csv")
    parser.add_argument("--features-dir", default="data/features")
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--limit", type=int, default=None, help="only the first N slides (smoke test)")
    parser.add_argument("--overwrite", action="store_true", help="re-download coords even if present")
    parser.add_argument("--verify-only", action="store_true", help="report coverage without downloading")
    args = parser.parse_args(argv)

    if not _h5py_available():
        print("ERROR: h5py is required: pip install h5py", file=sys.stderr)
        return 1

    manifest = Path(args.manifest)
    if not manifest.exists():
        print(f"ERROR: manifest not found: {manifest}", file=sys.stderr)
        return 1
    features_dir = Path(args.features_dir)
    if not features_dir.is_dir():
        print(f"ERROR: features dir not found: {features_dir}", file=sys.stderr)
        return 1

    slide_ids = read_slide_ids(manifest)
    if args.limit:
        slide_ids = slide_ids[: args.limit]
    print(f"{len(slide_ids)} slides in {manifest} -> {features_dir}")

    if args.verify_only:
        return verify_only(features_dir, slide_ids)

    counts: Dict[str, int] = {"ok": 0, "skip": 0, "fail": 0}
    failures: List[Tuple[str, str]] = []

    with ThreadPoolExecutor(max_workers=max(1, args.workers)) as pool:
        futures = {
            pool.submit(attach_one, sid, features_dir, args.overwrite): sid for sid in slide_ids
        }
        done = 0
        for future in as_completed(futures):
            image_id, status, detail = future.result()
            counts[status] += 1
            if status == "fail":
                failures.append((image_id, detail))
            done += 1
            if done % 25 == 0 or done == len(slide_ids):
                print(f"  {done}/{len(slide_ids)}  ok={counts['ok']} skip={counts['skip']} fail={counts['fail']}")

    print(f"\nDone: {counts['ok']} updated, {counts['skip']} already had coords, {counts['fail']} failed.")
    if failures:
        print("\nFailures:")
        for image_id, detail in failures[:20]:
            print(f"  {image_id}: {detail}")
        if len(failures) > 20:
            print(f"  ... and {len(failures) - 20} more")
        return 1

    print("\nVerify coverage with: python scripts/attach_patch_coords.py --verify-only")
    print("Then heatmaps are unblocked: python -m src.export_clam_attention --out-dir results/heatmaps/clam_sb --split test")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

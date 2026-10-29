"""Whole-slide -> 256x256 patch tiling with Otsu tissue detection -- Person 1.

Kept dependency-light: ``openslide`` is used when it is installed (real .svs
files), otherwise the module falls back to Pillow, which is enough for PANDA
``.tiff`` thumbnails and for the synthetic mock slides used in development.

Pipeline
--------
1. Read a low-resolution thumbnail.
2. Otsu-threshold the thumbnail to find tissue (implemented here with NumPy so
   the module works without OpenCV/OpenSlide).
3. Build a non-overlapping 256x256 grid over the whole slide and keep tiles
   whose tissue fraction exceeds ``min_tissue_fraction``.
4. Save the tiles plus a ``coords.csv`` (``patch_path, x, y``).

Every downstream component needs ``(x, y)`` patch origins, so ``coords.csv``
is part of the data contract just like the feature files.

Owner: Person 1.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
from PIL import Image

# --------------------------------------------------------------------------- #
# Tissue detection
# --------------------------------------------------------------------------- #
def otsu_threshold(gray: np.ndarray, bins: int = 256) -> float:
    """Otsu's method on a uint8 grayscale image (from scratch, NumPy only)."""
    if gray.dtype != np.uint8:
        gray = np.clip(gray, 0, 255).astype(np.uint8)
    histogram, edges = np.histogram(gray.reshape(-1), bins=bins, range=(0, 256))
    total = histogram.sum()
    if total == 0:
        return 0.0

    probabilities = histogram / total
    centres = (edges[:-1] + edges[1:]) / 2.0
    omega = np.cumsum(probabilities)
    mu = np.cumsum(probabilities * centres)
    mu_total = mu[-1]

    denominator = omega * (1.0 - omega)
    denominator[denominator <= 0] = np.nan
    between = (mu_total * omega - mu) ** 2 / denominator
    index = int(np.nanargmax(between))
    return float(centres[index])


def detect_tissue_mask(
    thumbnail: Image.Image | np.ndarray,
    min_saturation: int = 20,
    invert: bool = True,
) -> np.ndarray:
    """Binary tissue mask from a thumbnail.

    Tissue is dark and coloured (haematoxylin), background glass is bright and
    unsaturated. We combine an Otsu threshold on the grayscale channel with a
    saturation gate, which is more robust than grayscale alone on pale slides.
    """
    if isinstance(thumbnail, Image.Image):
        rgb = np.asarray(thumbnail.convert("RGB"), dtype=np.uint8)
    else:
        rgb = np.asarray(thumbnail, dtype=np.uint8)
        if rgb.ndim == 2:
            rgb = np.stack([rgb] * 3, axis=-1)

    gray = (0.299 * rgb[..., 0] + 0.587 * rgb[..., 1] + 0.114 * rgb[..., 2]).astype(np.uint8)
    threshold = otsu_threshold(gray)
    dark = gray < threshold if invert else gray > threshold

    maximum = rgb.max(axis=2).astype(np.int16)
    minimum = rgb.min(axis=2).astype(np.int16)
    saturation = maximum - minimum
    coloured = saturation > min_saturation

    return (dark | coloured).astype(np.uint8)


# --------------------------------------------------------------------------- #
# Tiling
# --------------------------------------------------------------------------- #
def grid_coordinates(
    width: int, height: int, patch_size: int = 256
) -> List[Tuple[int, int]]:
    """All ``(x, y)`` origins of a non-overlapping grid covering the slide."""
    return [
        (x, y)
        for y in range(0, max(height - patch_size + 1, 1), patch_size)
        for x in range(0, max(width - patch_size + 1, 1), patch_size)
    ]


def tissue_fraction_in_tile(
    tissue_mask: np.ndarray,
    x: int,
    y: int,
    patch_size: int,
    level_scale: float = 1.0,
) -> float:
    patch_size_level = max(1, int(round(patch_size * level_scale)))
    x_level = int(round(x * level_scale))
    y_level = int(round(y * level_scale))
    tile = tissue_mask[
        y_level : y_level + patch_size_level, x_level : x_level + patch_size_level
    ]
    if tile.size == 0:
        return 0.0
    return float(tile.mean())


def select_tiles(
    tissue_mask: np.ndarray,
    slide_dimensions: Tuple[int, int],
    patch_size: int = 256,
    min_tissue_fraction: float = 0.5,
    max_patches: Optional[int] = None,
) -> List[Tuple[int, int]]:
    """Grid origins whose tissue fraction passes the threshold."""
    width, height = slide_dimensions
    scale = tissue_mask.shape[1] / max(width, 1)
    kept = [
        (x, y)
        for x, y in grid_coordinates(width, height, patch_size)
        if tissue_fraction_in_tile(tissue_mask, x, y, patch_size, scale) >= min_tissue_fraction
    ]
    if max_patches and len(kept) > max_patches:
        index = np.linspace(0, len(kept) - 1, max_patches).astype(int)
        kept = [kept[i] for i in index]
    return kept


# --------------------------------------------------------------------------- #
# Slide reading / writing
# --------------------------------------------------------------------------- #
def open_slide(path: str | Path):
    """Return ``(reader, dimensions)`` using OpenSlide when available."""
    path = Path(path)
    try:
        import openslide  # type: ignore

        slide = openslide.OpenSlide(str(path))
        return slide, tuple(slide.dimensions)
    except Exception:
        with Image.open(path) as image:
            return image.convert("RGB"), image.size


def read_thumbnail(slide: Any, width: int = 1024) -> Image.Image:
    """Downsampled RGB view of the slide."""
    if hasattr(slide, "get_thumbnail"):
        return slide.get_thumbnail((width, width))
    dimensions = getattr(slide, "size", None) or slide.dimensions
    scale = width / max(dimensions[0], 1)
    return slide.resize((max(1, int(dimensions[0] * scale)), max(1, int(dimensions[1] * scale))))


def read_region(slide: Any, x: int, y: int, patch_size: int = 256) -> Image.Image:
    if hasattr(slide, "read_region"):
        region = slide.read_region((x, y), 0, (patch_size, patch_size))
        return region.convert("RGB")
    return slide.crop((x, y, x + patch_size, y + patch_size)).convert("RGB")


def extract_patches(
    slide_path: str | Path,
    out_dir: str | Path,
    patch_size: int = 256,
    min_tissue_fraction: float = 0.5,
    thumbnail_width: int = 1024,
    max_patches: Optional[int] = None,
    save_thumbnail: bool = True,
) -> Dict[str, Any]:
    """Tile one slide into ``out_dir/patches/*.png`` + ``out_dir/coords.csv``."""
    out_dir = Path(out_dir)
    patches_dir = out_dir / "patches"
    patches_dir.mkdir(parents=True, exist_ok=True)

    slide, dimensions = open_slide(slide_path)
    thumbnail = read_thumbnail(slide, thumbnail_width)
    tissue_mask = detect_tissue_mask(thumbnail)

    kept = select_tiles(
        tissue_mask,
        dimensions,
        patch_size=patch_size,
        min_tissue_fraction=min_tissue_fraction,
        max_patches=max_patches,
    )

    rows: List[Dict[str, Any]] = []
    for index, (x, y) in enumerate(kept):
        patch = read_region(slide, x, y, patch_size)
        patch_path = patches_dir / f"{Path(slide_path).stem}_x{x}_y{y}.png"
        patch.save(patch_path)
        rows.append({"patch_path": str(patch_path), "x": x, "y": y, "patch_index": index})

    with (out_dir / "coords.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["patch_index", "x", "y", "patch_path"])
        writer.writeheader()
        writer.writerows(rows)

    if save_thumbnail:
        thumbnail.save(out_dir / "thumbnail.png")
        Image.fromarray((tissue_mask * 255).astype(np.uint8)).save(out_dir / "tissue_mask.png")

    return {
        "slide": str(slide_path),
        "dimensions": list(dimensions),
        "thumbnail": str(out_dir / "thumbnail.png"),
        "coords_csv": str(out_dir / "coords.csv"),
        "n_patches": len(rows),
        "patch_size": patch_size,
        "min_tissue_fraction": min_tissue_fraction,
    }


def read_coords_csv(path: str | Path) -> List[Dict[str, Any]]:
    with Path(path).open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Tile a WSI into 256x256 tissue patches.")
    parser.add_argument("--slide", required=True)
    parser.add_argument("--out", required=True, help="output directory for one slide")
    parser.add_argument("--patch-size", type=int, default=256)
    parser.add_argument("--min-tissue", type=float, default=0.5)
    parser.add_argument("--max-patches", type=int, default=None)
    args = parser.parse_args(argv)

    summary = extract_patches(
        args.slide,
        args.out,
        patch_size=args.patch_size,
        min_tissue_fraction=args.min_tissue,
        max_patches=args.max_patches,
    )
    print(f"{summary['n_patches']} patches -> {summary['coords_csv']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

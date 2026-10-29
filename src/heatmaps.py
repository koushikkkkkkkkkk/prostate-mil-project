"""Attention -> spatial heatmap rendering and localisation scoring.

Two consumers:
* Person 1 / Person 3 -- render attention maps for their models' test slides.
* Person 2 -- CLAM-SB attention maps, plus the quantitative overlap score that
  replaces "the heatmap looks right" with a number.

Pipeline
--------
1. Take per-patch attention weights ``a_n`` and patch origins ``(x, y)``.
2. Scatter them into a coarse grid at the level of the slide thumbnail.
3. Percentile-clip (default 5th-95th) so one saturated patch does not flatten
   the whole map, then map through a colormap and alpha-blend on the thumbnail.
4. Optionally crop the top-k highest-attention patches for the report figures.

Caveat to reproduce in the write-up: attention is a *model cue*, not a
segmentation. PANDA has no pixel-level ISUP annotation, so overlap scores use
coarse tissue/tumour masks and are only a sanity check, never ground truth.

Owner: Person 2 (CLAM-SB heatmaps) / Person 4 (app + stitch tooling).
"""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import torch
from PIL import Image

DEFAULT_COLORMAP = "jet"


# --------------------------------------------------------------------------- #
# Coordinate helpers
# --------------------------------------------------------------------------- #
def _to_numpy(values: Any) -> np.ndarray:
    if torch.is_tensor(values):
        return values.detach().cpu().numpy()
    return np.asarray(values)


def estimate_slide_dimensions(
    coords: np.ndarray, patch_size: int
) -> Tuple[int, int]:
    """Best-effort slide size from patch origins (coords start at 0)."""
    if coords.size == 0:
        return patch_size, patch_size
    width = int(np.nanmax(coords[:, 0]) + patch_size)
    height = int(np.nanmax(coords[:, 1]) + patch_size)
    return max(width, patch_size), max(height, patch_size)


def scatter_attention(
    attention: Any,
    coords: Any,
    patch_size: int = 256,
    out_width: int = 512,
    out_height: Optional[int] = None,
    slide_dimensions: Optional[Tuple[int, int]] = None,
) -> np.ndarray:
    """Scatter per-patch attention into a ``[H, W]`` float map."""
    attention = _to_numpy(attention).astype(np.float64).reshape(-1)
    coords = _to_numpy(coords).astype(np.float64).reshape(-1, 2)
    if attention.size != coords.shape[0]:
        raise ValueError(
            f"attention ({attention.size}) and coords ({coords.shape[0]}) length mismatch"
        )

    slide_w, slide_h = slide_dimensions or estimate_slide_dimensions(coords, patch_size)
    out_height = out_height or max(1, int(round(out_width * slide_h / max(slide_w, 1))))
    scale_x = out_width / max(slide_w, 1)
    scale_y = out_height / max(slide_h, 1)
    step_x = max(1, int(round(patch_size * scale_x)))
    step_y = max(1, int(round(patch_size * scale_y)))

    accumulator = np.zeros((out_height, out_width), dtype=np.float64)
    counts = np.zeros((out_height, out_width), dtype=np.float64)
    for value, (x, y) in zip(attention, coords):
        col = int(round(x * scale_x))
        row = int(round(y * scale_y))
        if not (0 <= row < out_height and 0 <= col < out_width):
            continue
        row_end = min(row + step_y, out_height)
        col_end = min(col + step_x, out_width)
        if row_end <= row or col_end <= col:
            continue
        accumulator[row:row_end, col:col_end] += value
        counts[row:row_end, col:col_end] += 1.0

    filled = counts > 0
    accumulator[filled] /= counts[filled]
    return accumulator


def normalise_attention(
    attention_map: np.ndarray, percentile_clip: Tuple[float, float] = (5.0, 95.0)
) -> np.ndarray:
    """Percentile-clip then min-max to [0, 1]; empty regions stay at 0."""
    values = attention_map.astype(np.float64).copy()
    nonzero = values[values > 0]
    if nonzero.size == 0:
        return np.zeros_like(values, dtype=np.float32)
    low, high = np.percentile(nonzero, percentile_clip)
    if high - low < 1e-12:
        high = low + 1e-12
    values = np.clip(values, low, high)
    values = (values - low) / (high - low)
    values[attention_map <= 0] = 0.0
    return values.astype(np.float32)


def colourise(normalised: np.ndarray, colormap: str = DEFAULT_COLORMAP) -> np.ndarray:
    """Apply a matplotlib colormap to a [0, 1] map -> RGB uint8."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.cm as cm

    if hasattr(matplotlib, "colormaps"):          # matplotlib >= 3.6
        mapper = matplotlib.colormaps[colormap]
    else:                                          # pragma: no cover - old versions
        mapper = cm.get_cmap(colormap)
    rgba = mapper(np.clip(normalised, 0.0, 1.0))
    return (rgba[..., :3] * 255).astype(np.uint8)


# --------------------------------------------------------------------------- #
# Rendering
# --------------------------------------------------------------------------- #
def render_attention_heatmap(
    attention: Any,
    coords: Any,
    out_path: str | Path,
    thumbnail: Optional[str | Path | Image.Image] = None,
    patch_size: int = 256,
    out_width: int = 768,
    alpha: float = 0.5,
    percentile_clip: Tuple[float, float] = (5.0, 95.0),
    colormap: str = DEFAULT_COLORMAP,
    title: Optional[str] = None,
    slide_dimensions: Optional[Tuple[int, int]] = None,
) -> Dict[str, Any]:
    """Render a heatmap, plus an overlay when a thumbnail is available."""
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    raw = scatter_attention(attention, coords, patch_size, out_width, slide_dimensions=slide_dimensions)
    normalised = normalise_attention(raw, percentile_clip)
    colour = colourise(normalised, colormap)

    heat_image = Image.fromarray(colour)
    heat_image.save(out_path)

    saved: Dict[str, Any] = {
        "heatmap": str(out_path),
        "attention_map_shape": list(normalised.shape),
        "colormap": colormap,
        "percentile_clip": list(percentile_clip),
    }

    if thumbnail is not None:
        if isinstance(thumbnail, Image.Image):
            thumb = thumbnail
        else:
            with Image.open(thumbnail) as handle:
                thumb = handle.convert("RGB")
        thumb = thumb.resize((normalised.shape[1], normalised.shape[0]))
        base = np.asarray(thumb, dtype=np.float32)
        overlay = (1 - alpha) * base + alpha * colour.astype(np.float32)
        overlay_path = out_path.with_name(out_path.stem + "_overlay.png")
        Image.fromarray(np.clip(overlay, 0, 255).astype(np.uint8)).save(overlay_path)
        saved["overlay"] = str(overlay_path)

    if title:
        saved["title"] = title
    return saved


def plot_attention_distribution(
    attention: Any,
    out_path: str | Path,
    title: str = "Attention distribution",
    instance_mask: Optional[Any] = None,
) -> str:
    """Histogram of attention weights (optionally highlighting clustered patches)."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    values = _to_numpy(attention).astype(float).reshape(-1)
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(6.4, 3.6))
    ax.hist(values, bins=min(60, max(10, values.size // 2)), color="#4c72b0", edgecolor="white")
    if instance_mask is not None:
        flagged = _to_numpy(instance_mask).astype(bool).reshape(-1)
        if flagged.size == values.size and flagged.any():
            ax.hist(values[flagged], bins=20, color="#c44e52", alpha=0.75, label="clustered (top/bottom-k)")
            ax.legend()
    ax.set_xlabel("attention weight")
    ax.set_ylabel("patch count")
    ax.set_title(title)
    ax.text(
        0.99,
        0.95,
        f"n={values.size}\nmean={values.mean():.4g}\nmax={values.max():.4g}\n"
        f"entropy={attention_entropy(values):.3f}",
        transform=ax.transAxes,
        ha="right",
        va="top",
        fontsize=8,
        bbox=dict(boxstyle="round", facecolor="white", alpha=0.8),
    )
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return str(out_path)


def attention_entropy(attention: Any, eps: float = 1e-12) -> float:
    """Shannon entropy of the attention distribution, in nats.

    Lower entropy = sharper focus. This is the "attention sharpness" number the
    CLAM ablation is judged on, alongside QWK.
    """
    values = _to_numpy(attention).astype(np.float64).reshape(-1)
    values = values[values > 0]
    if values.size == 0:
        return float("nan")
    values = values / values.sum()
    return float(-(values * np.log(values + eps)).sum())


def normalised_attention_entropy(attention: Any, eps: float = 1e-12) -> float:
    """Entropy divided by ``log(N)`` so slides with different N are comparable."""
    values = _to_numpy(attention).astype(np.float64).reshape(-1)
    n = int((values > 0).sum())
    if n <= 1:
        return 0.0
    return float(attention_entropy(values, eps) / np.log(n))


def top_attention_patches(
    attention: Any, coords: Any, patch_paths: Optional[Sequence[str]] = None, k: int = 20
) -> List[Dict[str, Any]]:
    """Rank patches by attention; used for top-k crop figures and the app."""
    values = _to_numpy(attention).astype(float).reshape(-1)
    points = _to_numpy(coords).astype(float).reshape(-1, 2)
    order = np.argsort(-values)[:k]
    rows: List[Dict[str, Any]] = []
    for rank, index in enumerate(order, start=1):
        row: Dict[str, Any] = {
            "rank": rank,
            "patch_index": int(index),
            "attention": float(values[index]),
            "x": float(points[index, 0]) if index < len(points) else float("nan"),
            "y": float(points[index, 1]) if index < len(points) else float("nan"),
        }
        if patch_paths is not None and index < len(patch_paths):
            row["patch_path"] = str(patch_paths[index])
        rows.append(row)
    return rows


def write_patch_table(rows: Sequence[Dict[str, Any]], out_path: str | Path) -> str:
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames: List[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with out_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames or ["rank"])
        writer.writeheader()
        writer.writerows(rows)
    return str(out_path)


def make_patch_contact_sheet(
    rows: Sequence[Dict[str, Any]], out_path: str | Path, tile: int = 128, columns: int = 4
) -> Optional[str]:
    """Grid of the highest-attention patch crops with attention labels."""
    from PIL import ImageDraw

    usable = [row for row in rows if row.get("patch_path") and Path(row["patch_path"]).exists()]
    if not usable:
        return None

    rows_count = (len(usable) + columns - 1) // columns
    sheet = Image.new("RGB", (columns * tile, rows_count * tile), "white")
    draw = ImageDraw.Draw(sheet)
    for i, row in enumerate(usable):
        with Image.open(row["patch_path"]) as patch:
            patch = patch.convert("RGB").resize((tile, tile))
            col, line = i % columns, i // columns
            sheet.paste(patch, (col * tile, line * tile))
            draw.text((col * tile + 4, line * tile + 4), f"#{row['rank']} a={row['attention']:.3f}", fill="yellow")

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(out_path)
    return str(out_path)


# --------------------------------------------------------------------------- #
# Localisation scoring against a tumour mask
# --------------------------------------------------------------------------- #
def patch_tumor_fractions(
    mask_path: str | Path,
    coords: Any,
    patch_size: int = 256,
    threshold: float = 127,
    slide_dimensions: Optional[Tuple[int, int]] = None,
) -> np.ndarray:
    """Mean mask value inside every patch region, in [0, 1].

    ``mask_path`` is any image where bright pixels mark tumour (or tissue)
    regions -- PANDA ships tissue masks; a radboud/karolinska tumour mask can be
    substituted when available.
    """
    points = _to_numpy(coords).astype(np.float64).reshape(-1, 2)
    with Image.open(mask_path) as handle:
        mask = np.asarray(handle.convert("L"), dtype=np.float32) / 255.0
    mask_h, mask_w = mask.shape[:2]
    slide_w, slide_h = slide_dimensions or estimate_slide_dimensions(points, patch_size)
    scale_x = mask_w / max(slide_w, 1)
    scale_y = mask_h / max(slide_h, 1)
    win_w = max(1, int(round(patch_size * scale_x)))
    win_h = max(1, int(round(patch_size * scale_y)))

    fractions = np.zeros(points.shape[0], dtype=np.float32)
    for i, (x, y) in enumerate(points):
        col = int(round(x * scale_x))
        row = int(round(y * scale_y))
        row_end = min(row + win_h, mask_h)
        col_end = min(col + win_w, mask_w)
        if row_end <= row or col_end <= col or row >= mask_h or col >= mask_w:
            continue
        region = mask[max(row, 0) : row_end, max(col, 0) : col_end]
        binarised = region > (threshold / 255.0)
        fractions[i] = float(binarised.mean()) if binarised.size else 0.0
    return fractions


def evaluate_localisation(
    attention: Any,
    mask_path: str | Path,
    coords: Any,
    patch_size: int = 256,
    fractions: Iterable[float] = (0.1, 0.2, 0.3),
    slide_dimensions: Optional[Tuple[int, int]] = None,
) -> Dict[str, Any]:
    """Attention-vs-mask overlap bundle (precision/recall/lift at top-k%)."""
    try:
        from src.evaluate import attention_mask_overlap
    except ImportError:  # pragma: no cover - direct script execution
        import sys

        sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
        from src.evaluate import attention_mask_overlap  # type: ignore

    patch_fractions = patch_tumor_fractions(
        mask_path, coords, patch_size=patch_size, slide_dimensions=slide_dimensions
    )
    return attention_mask_overlap(attention, patch_fractions, fractions=fractions)

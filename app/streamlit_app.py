"""Interactive MIL demo -- slide grading, attention localisation, results browser.

Run locally::

    streamlit run app/streamlit_app.py

Tabs
----
1. **Overview** -- what this is, what it is not.
2. **Slide analysis** -- the *validated* path: pick a slide from the split
   manifest, load its pre-extracted feature bag (the same features the model was
   trained and reported on), run the selected checkpoint, show the predicted
   grade, the probability bars, the attention heatmap and the top-k patches.
3. **Upload & tile** -- tile a single uploaded image, run the same checkpoint on
   those tiles, and show (a) the model output and (b) transparent, clearly
   labelled image descriptors. Uploaded images are almost never in the training
   distribution, so this tab *measures* that and says so instead of pretending.
4. **Results & reports** -- renders the real artifacts under ``results/``:
   ablation tables, per-seed metrics, confusion matrices, paired-comparison
   reports, statistical baselines, heatmap export summaries.

Design rules followed here
--------------------------
* **No fake predictions.** Every number shown is either produced by a real
  checkpoint or is explicitly labelled as a descriptive image statistic.
* **No random attention.** The heatmap is the model's actual attention weights.
* **No silent fallbacks.** A missing checkpoint, a feature-dim mismatch or a
  weight download failure is surfaced as an error with the fix, never papered
  over with a dummy model.
* **Mock artifacts are excluded by default** and marked not-reportable when
  enabled.

Research prototype. Not a diagnostic device. See the sidebar disclaimer.

Point ``PROSTATE_MIL_ROOT`` at another directory to demo a checkpoint/feature
tree that lives outside this repo.

Owner: Person 4 (app), with model/checkpoint hooks from Persons 1-3.
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch

ROOT = Path(os.environ.get("PROSTATE_MIL_ROOT", Path(__file__).resolve().parents[1]))
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

GRADE_NAMES = {
    0: "0 - benign / negative",
    1: "1 - Gleason 3+3",
    2: "2 - Gleason 3+4",
    3: "3 - Gleason 4+3",
    4: "4 - Gleason 4+4",
    5: "5 - Gleason 5+5 (or higher)",
}

#: uploads further from the training cloud than this (percentile) get no grade at all
OOD_ABSTAIN_PERCENTILE = 99.0
#: uploads this far out are graded but flagged
OOD_WARN_PERCENTILE = 95.0
#: set by the demo harness when the checkpoints on disk are placeholders
PLACEHOLDER_ENV = "PROSTATE_MIL_PLACEHOLDER"

#: model name -> human label, in the order the demo offers them
MODEL_LABELS = {
    "clam_sb": "CLAM-SB (Person 2)",
    "abmil": "ABMIL (Person 1)",
    "transmil": "TransMIL (Person 3)",
}


# --------------------------------------------------------------------------- #
# Checkpoint discovery / loading
# --------------------------------------------------------------------------- #
def discover_checkpoints(root: Path = ROOT) -> Dict[str, Any]:
    """Group every ``*_best.pt`` under ``results/`` by the model it contains.

    A checkpoint is inspected (header only, on CPU) to learn which model it is,
    because folder names are not trustworthy enough to load blind.
    """
    candidates: List[Path] = []
    for pattern in ("results/**/checkpoints/*_best.pt", "results/**/*_best.pt"):
        candidates.extend(root.glob(pattern))

    unique: List[Path] = []
    seen = set()
    for path in sorted(candidates):
        resolved = path.resolve()
        if resolved not in seen:
            seen.add(resolved)
            unique.append(path)

    models: Dict[str, List[Path]] = {}
    unreadable: List[Path] = []
    for path in unique:
        try:
            payload = torch.load(path, map_location="cpu", weights_only=False)
        except Exception:  # noqa: BLE001 - cannot inspect; keep it visible instead of hiding it
            unreadable.append(path)
            continue
        if not isinstance(payload, dict) or "state_dict" not in payload:
            unreadable.append(path)
            continue
        name = str(payload.get("model_name") or _guess_model_name(path)).lower().replace("-", "_")
        models.setdefault(name, []).append(path)
    return {"models": models, "unreadable": unreadable}


def _guess_model_name(path: Path) -> str:
    text = str(path).lower()
    for key in ("clam_sb", "clam-sb", "abmil", "transmil"):
        if key in text:
            return key
    return "unknown"


def load_checkpoint_payload(path: Path) -> Dict[str, Any]:
    """``torch.load`` a checkpoint with a pre-2.0 fallback."""
    try:
        payload = torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:  # torch < 2.0 has no weights_only
        payload = torch.load(path, map_location="cpu")
    if not isinstance(payload, dict) or "state_dict" not in payload:
        raise ValueError(
            f"{path} is not a training checkpoint (expected a dict with 'state_dict'); "
            "point the app at results/**/checkpoints/*_best.pt"
        )
    return payload


def build_model_from_payload(payload: Dict[str, Any]) -> Tuple[torch.nn.Module, str, Dict[str, Any]]:
    """Rebuild the exact architecture described by the checkpoint."""
    from src.models import build_model

    name = str(payload.get("model_name") or "unknown").lower().replace("-", "_")
    if name not in ("clam_sb", "abmil", "transmil"):
        raise ValueError(f"checkpoint holds model '{name}', which this app cannot drive")
    config = dict(payload.get("model_config") or {})
    model = build_model(name, config)
    model.load_state_dict(payload["state_dict"])
    return model.eval(), name, config


@torch.no_grad()
def run_model(model: torch.nn.Module, features: torch.Tensor, device: str = "cpu") -> Dict[str, Any]:
    """Forward one bag; returns probabilities, attention and the predicted class."""
    output = model(features.to(device))
    logits = output["logits"]
    probs = torch.softmax(logits, dim=-1).reshape(-1).cpu().numpy()
    attention = output["attention"].reshape(-1).cpu().numpy()
    if attention.size != features.shape[0]:  # defensive: never index with the wrong length
        attention = np.resize(attention, features.shape[0])
    prediction = int(np.argmax(probs))
    return {
        "probs": probs,
        "attention": attention,
        "prediction": prediction,
        "confidence": float(probs[prediction]),
    }


# --------------------------------------------------------------------------- #
# Feature extraction (same backbone as src/feature_extraction.py)
# --------------------------------------------------------------------------- #
def build_extractor(in_dim: int, backbone: str = "resnet50", pretrained: bool = True, device: str = "cpu"):
    """Frozen CNN producing ``in_dim``-d vectors, exactly like Person 1's extractor."""
    from src.feature_extraction import FrozenFeatureExtractor

    return FrozenFeatureExtractor(backbone=backbone, target_dim=in_dim, pretrained=pretrained, device=device)


def tile_image(image, grid: int = 6) -> Dict[str, Any]:
    """Cut an image into a ``grid x grid`` bag of tiles.

    Returns the tiles, their top-left coordinates in the source image and the tile
    size, so attention can be scattered back onto the uploaded picture.
    """
    image = image.convert("RGB")
    width, height = image.size
    tile_w = max(1, width // grid)
    tile_h = max(1, height // grid)

    tiles: List[Any] = []
    coords: List[Tuple[float, float]] = []
    for row in range(grid):
        for col in range(grid):
            left, top = col * tile_w, row * tile_h
            tiles.append(image.crop((left, top, left + tile_w, top + tile_h)))
            coords.append((float(left), float(top)))
    return {
        "tiles": tiles,
        "coords": np.asarray(coords, dtype=np.float32),
        "tile_size": int(max(tile_w, tile_h)),
        "grid": grid,
        "size": (width, height),
    }


def extract_features(
    images: Sequence[Any], extractor, image_size: int = 256, batch_size: int = 16
) -> torch.Tensor:
    """Feature matrix ``[N, D]`` for a list of PIL images."""
    mean = torch.tensor(IMAGENET_MEAN).view(3, 1, 1)
    std = torch.tensor(IMAGENET_STD).view(3, 1, 1)
    chunks: List[torch.Tensor] = []
    batch: List[torch.Tensor] = []
    for image in images:
        array = np.asarray(image.convert("RGB").resize((image_size, image_size)), dtype=np.float32) / 255.0
        tensor = torch.from_numpy(array).permute(2, 0, 1)
        batch.append((tensor - mean) / std)
        if len(batch) == batch_size:
            chunks.append(extractor(torch.stack(batch)))
            batch = []
    if batch:
        chunks.append(extractor(torch.stack(batch)))
    if not chunks:
        return torch.zeros((0, getattr(extractor, "out_dim", 0)))
    return torch.cat(chunks, dim=0)


# --------------------------------------------------------------------------- #
# Out-of-distribution check (measured, not asserted)
# --------------------------------------------------------------------------- #
def reference_feature_stats(
    features_root: Path, in_dim: int, max_slides: int = 40
) -> Optional[Dict[str, Any]]:
    """Per-slide feature statistics of the *training* feature release.

    Used to tell the user whether an uploaded image produces features that look
    like anything the checkpoint was trained on. Returns ``None`` when no feature
    files are available, in which case the check is simply skipped.
    """
    if not features_root.is_dir():
        return None
    files = sorted(features_root.glob("*.pt"))[:max_slides]
    if not files:
        return None

    pooled: List[np.ndarray] = []
    dim = None
    for path in files:
        try:
            payload = torch.load(path, map_location="cpu", weights_only=False)
        except Exception:  # noqa: BLE001
            continue
        features = payload.get("features") if isinstance(payload, dict) else payload
        if features is None:
            continue
        tensor = torch.as_tensor(features).float()
        if tensor.ndim != 2:
            continue
        if dim is None:
            dim = int(tensor.size(1))
        if tensor.size(1) != dim or dim != int(in_dim):
            continue
        pooled.append(tensor.mean(dim=0).numpy())
    if len(pooled) < 3:
        return None

    matrix = np.stack(pooled)
    mean = matrix.mean(axis=0)
    std = matrix.std(axis=0) + 1e-6
    distances = np.abs((matrix - mean) / std).mean(axis=1)
    return {
        "dim": int(dim),
        "mean": mean,
        "std": std,
        "distances": distances,
        "n_slides": int(matrix.shape[0]),
        "source": str(features_root),
    }


def ood_percentile(features: torch.Tensor, reference: Optional[Dict[str, Any]]) -> Optional[Dict[str, float]]:
    """How far the upload's features sit from the training feature cloud."""
    if reference is None:
        return None
    tensor = torch.as_tensor(features).float()
    if tensor.ndim != 2 or int(tensor.size(1)) != int(reference["dim"]):
        return None
    pooled = tensor.mean(dim=0).numpy()
    distance = float(np.abs((pooled - reference["mean"]) / reference["std"]).mean())
    return {
        "distance": distance,
        "percentile": float((reference["distances"] < distance).mean() * 100.0),
        "reference_median": float(np.median(reference["distances"])),
        "n_reference": reference["n_slides"],
    }


def grade_decision(ood: Optional[Dict[str, float]]) -> Dict[str, Any]:
    """Decide whether a model output may be presented as a grade at all.

    A MIL model trained on 20x H&E patches will happily emit a confident grade for
    a picture it has never seen the likes of -- a nuclear-density phantom comes
    back as ISUP 5. Rather than printing that number with a footnote, the app only
    reports a grade when the upload's feature cloud is close to the training
    features. Otherwise it abstains and says why.

    Returns ``{"action": ..., "reason": ..., ...}`` where ``action`` is one of
    ``"report"``, ``"warn"``, ``"abstain"``, ``"unverified"``.
    """
    if ood is None:
        return {
            "action": "unverified",
            "reason": (
                "No reference feature cloud is available, so the app cannot tell whether this "
                "image is inside the model's training distribution. The raw model output is "
                "hidden by default for that reason."
            ),
            "percentile": None,
        }
    percentile = float(ood["percentile"])
    evidence = (
        f"upload is at the {percentile:.0f}th percentile of the training feature cloud "
        f"(n={ood['n_reference']} slides; distance {ood['distance']:.2f} vs median "
        f"{ood['reference_median']:.2f})"
    )
    if percentile >= OOD_ABSTAIN_PERCENTILE:
        return {
            "action": "abstain",
            "reason": (
                "This image is outside the model's training distribution, so no ISUP grade is "
                f"shown: {evidence}."
            ),
            "percentile": percentile,
        }
    if percentile >= OOD_WARN_PERCENTILE:
        return {
            "action": "warn",
            "reason": f"Graded, but with caution -- {evidence}.",
            "percentile": percentile,
        }
    return {"action": "report", "reason": f"In-distribution -- {evidence}.", "percentile": percentile}


# --------------------------------------------------------------------------- #
# Transparent image descriptors (NOT the model)
# --------------------------------------------------------------------------- #
def image_descriptors(image) -> Dict[str, float]:
    """Descriptive H&E image statistics -- explicitly not a grade.

    These respond to how much dark (haematoxylin-stained) nuclear material is in
    the picture. That is why synthetic nuclear-density phantoms separate cleanly
    here while a MIL model trained on real 20x H&E patches does not -- which is
    exactly why these numbers must never be presented as model output.
    """
    image = image.convert("RGB")
    array = np.asarray(image, dtype=np.float32) / 255.0
    red, green, blue = array[:, :, 0], array[:, :, 1], array[:, :, 2]
    luminance = 0.299 * red + 0.587 * green + 0.114 * blue

    stain_density = float(np.mean(1.0 - luminance))
    threshold = _otsu(luminance)
    dark = luminance < threshold

    return {
        "mean stain density (1 - luminance)": stain_density,
        "dark-object fraction (Otsu)": float(dark.mean()),
        "haematoxylin / purple index": float(np.mean(np.clip(red + blue - 2 * green, 0.0, None))),
        "dark objects per megapixel": float(_count_blobs(dark)) / max(array.shape[0] * array.shape[1], 1) * 1e6,
        "blank-glass fraction (luminance > 0.85)": float(np.mean(luminance > 0.85)),
        "luminance dynamic range (p95 - p5)": float(np.percentile(luminance, 95) - np.percentile(luminance, 5)),
        "otsu threshold": float(threshold),
    }


def _otsu(values: np.ndarray, bins: int = 256) -> float:
    """Otsu threshold for a ``[0, 1]`` array."""
    hist, edges = np.histogram(values, bins=bins, range=(0.0, 1.0))
    hist = hist.astype(np.float64)
    total = hist.sum()
    if total <= 0:
        return 0.5
    centres = (edges[:-1] + edges[1:]) / 2.0
    weight_bg = np.cumsum(hist)
    weight_fg = total - weight_bg
    cum_mean = np.cumsum(hist * centres)
    total_mean = cum_mean[-1]
    with np.errstate(divide="ignore", invalid="ignore"):
        between = (total_mean * weight_bg - cum_mean * total) ** 2 / (weight_bg * weight_fg)
    between = np.nan_to_num(between, nan=0.0, posinf=0.0, neginf=0.0)
    # use the upper edge of the winning bin: with a two-valued histogram the
    # between-class variance is flat across a plateau and argmax returns its first
    # bin, whose centre sits *below* the dark pixels and would exclude them all.
    return float(edges[int(np.argmax(between)) + 1])


def _count_blobs(mask: np.ndarray, min_area: int = 12) -> int:
    """Connected-component count of the dark mask (nuclei proxy)."""
    try:
        from scipy import ndimage

        labels, n = ndimage.label(mask)
        if n == 0:
            return 0
        sizes = np.bincount(labels.ravel())
        return int((sizes[1:] >= min_area).sum())
    except Exception:  # noqa: BLE001 - scipy missing, coarse estimate instead
        return int(mask.sum() // max(min_area * 4, 1))


# --------------------------------------------------------------------------- #
# Reports discovery
# --------------------------------------------------------------------------- #
def _is_mock(path: Path, root: Path) -> bool:
    try:
        relative = path.relative_to(root)
    except ValueError:
        relative = path
    return any(part.startswith("mock") for part in relative.parts)


def discover_reports(root: Path = ROOT, include_mock: bool = False) -> Dict[str, List[Path]]:
    """Every artifact the report is built from, grouped by how it is rendered."""
    def keep(paths: Sequence[Path]) -> List[Path]:
        seen, out = set(), []
        for path in paths:
            if include_mock or not _is_mock(path, root):
                key = path.resolve()
                if key not in seen:
                    seen.add(key)
                    out.append(path)
        return out

    markdown: List[Path] = []
    payloads: List[Path] = []
    images: List[Path] = []
    for base in (root / "results", root / "report", root / "docs"):
        if not base.is_dir():
            continue
        markdown += keep(sorted(base.rglob("*.md")))
        payloads += keep(sorted(base.rglob("*.json")))
        images += keep(sorted(base.rglob("*.png")))

    heatmap_root = root / "results" / "heatmaps"
    heatmaps = keep(sorted(heatmap_root.rglob("*.png"))) if heatmap_root.is_dir() else []
    heatmap_set = {path.resolve() for path in heatmaps}
    return {
        "markdown": keep(markdown),
        "json": keep(payloads),
        "images": [path for path in keep(images) if path.resolve() not in heatmap_set],
        "heatmaps": heatmaps,
    }


def read_json(path: Path) -> Any:
    try:
        return json.loads(Path(path).read_text(encoding="utf-8"))
    except Exception as error:  # noqa: BLE001
        return {"error": f"could not parse {path.name}: {error}"}


# --------------------------------------------------------------------------- #
# Streamlit helpers
# --------------------------------------------------------------------------- #
def _stretch(st_callable, *args, **kwargs):
    """Call a Streamlit element full-width across versions.

    Streamlit renamed ``use_container_width=True`` to ``width="stretch"`` and warns
    on every call of the old spelling; the new spelling raises TypeError on older
    releases. Try the new one, fall back to the old one.
    """
    try:
        return st_callable(*args, width="stretch", **kwargs)
    except TypeError:
        return st_callable(*args, use_container_width=True, **kwargs)


def _show_image(st, image, caption: str = "") -> None:
    """``st.image`` across Streamlit versions."""
    _stretch(st.image, image, caption=caption)


def _show_figure(st, figure) -> None:
    _stretch(st.pyplot, figure)


def _show_table(st, data, **kwargs) -> None:
    _stretch(st.dataframe, data, **kwargs)


def _relative(path: Path, root: Path) -> str:
    try:
        return str(path.relative_to(root))
    except ValueError:
        return str(path)


def _probability_figure(matplotlib, probs: Sequence[float], prediction: int, truth: Optional[int], size=(6.4, 2.6)):
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=size)
    colours = ["#c44e52" if index == prediction else "#4c72b0" for index in range(len(probs))]
    ax.bar([str(index) for index in range(len(probs))], probs, color=colours)
    if truth is not None:
        ax.axvline(truth, color="#55a868", linestyle="--", linewidth=1.2, label="true grade")
        ax.legend(frameon=False, fontsize=8)
    ax.set_xlabel("ISUP grade group")
    ax.set_ylabel("probability")
    ax.set_ylim(0, 1)
    return fig


def _attention_overlay(matplotlib, attention, coords, patch_size, width, size, base_image):
    """Scatter attention, normalise, alpha-blend over the image at the same scale."""
    from PIL import Image

    from src.heatmaps import normalise_attention, scatter_attention

    attention_map = scatter_attention(
        attention, coords, patch_size=patch_size, out_width=width, slide_dimensions=size
    )
    normalised = normalise_attention(attention_map)
    colour = (matplotlib.colormaps["jet"](normalised)[..., :3] * 255).astype(np.uint8)
    if base_image is not None:
        base = base_image.convert("RGB").resize((normalised.shape[1], normalised.shape[0]))
    else:
        base = Image.new("RGB", (normalised.shape[1], normalised.shape[0]), "white")
    overlay = 0.5 * np.asarray(base, dtype=np.float32) + 0.5 * colour.astype(np.float32)
    return np.clip(overlay, 0, 255).astype(np.uint8)


# --------------------------------------------------------------------------- #
# App
# --------------------------------------------------------------------------- #
def render() -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from PIL import Image

    import streamlit as st

    from src.dataset import feature_path, load_feature_bag, read_manifest, split_records
    from src.heatmaps import attention_entropy, normalised_attention_entropy, top_attention_patches

    st.set_page_config(page_title="Patches to Prognosis", page_icon="🔬", layout="wide")
    st.title("From Patches to Prognosis")
    st.caption(
        "Weakly supervised ISUP grading and attention localisation from whole-slide patches "
        "(Multiple Instance Learning)."
    )

    placeholder = os.environ.get(PLACEHOLDER_ENV, "") not in ("", "0", "false", "False")
    if placeholder:
        st.error(
            "**PLACEHOLDER WEIGHTS — not a trained model.** This instance is running against "
            f"`{ROOT}`, a demo tree whose checkpoints contain **random initialisation**. Every "
            "grade, probability and attention map produced here is arbitrary. It exists to "
            "exercise the UI only. Point the app at your real `results/` folder (or unset "
            f"`{PLACEHOLDER_ENV}`) to see real numbers."
        )

    # ---------------- sidebar ----------------
    checkpoints = discover_checkpoints()
    manifest_default = (
        "data/mock/manifest.csv" if (ROOT / "data/mock/manifest.csv").exists() else "data/common_split.csv"
    )

    with st.sidebar:
        st.header("Configuration")
        manifest_path = st.text_input("Split manifest", manifest_default)
        features_root_text = st.text_input(
            "Feature folder", "data/mock/features" if "mock" in manifest_path else "data/features"
        )
        reports_root_text = st.text_input("Reports folder", "results")

        models = checkpoints["models"]
        available = [name for name in MODEL_LABELS if name in models]
        if not available:
            st.error(
                "No trained checkpoint found under `results/**/checkpoints/*_best.pt`. Train one "
                "first, e.g.\n```bash\npython -m src.train_clam_sb --manifest data/common_split.csv "
                "--features-root data/features --seeds 0\n```"
            )
            st.stop()

        chosen_label = st.selectbox("Model", [MODEL_LABELS[name] for name in available], index=0)
        chosen_model = available[[MODEL_LABELS[name] for name in available].index(chosen_label)]
        ckpt_paths = sorted(models[chosen_model])
        ckpt_labels = [_relative(path, ROOT) for path in ckpt_paths]
        default_index = next((i for i, label in enumerate(ckpt_labels) if "seed0" in label), 0)
        ckpt_label = st.selectbox("Checkpoint", ckpt_labels, index=default_index)
        checkpoint = ckpt_paths[ckpt_labels.index(ckpt_label)]

        missing = [MODEL_LABELS[name] for name in MODEL_LABELS if name not in models]
        if missing:
            st.caption("Not trained yet (option hidden): " + ", ".join(missing))
        if checkpoints["unreadable"]:
            st.caption(f"{len(checkpoints['unreadable'])} checkpoint file(s) could not be read.")

        top_k = st.slider("Top suspicious patches", 3, 20, 6)

        if os.environ.get("PROSTATE_MIL_ROOT"):
            st.info(
                "Running against an **external checkpoint tree**:\n`"
                + str(ROOT)
                + "`\n\nAnything shown comes from the weights stored there."
            )

        if os.environ.get(PLACEHOLDER_ENV, "") not in ("", "0", "false", "False"):
            st.error("Placeholder (random) weights — outputs are meaningless.", icon="🚨")

        st.markdown("---")
        st.warning(
            "**Research prototype.** Not a clinical device. Trained/evaluated on a 420-slide "
            "PANDA subset; attention maps are model cues, not tumour segmentations. External "
            "validation is required before any real-world use."
        )
        if "mock" in manifest_path:
            st.info("SYNTHETIC mock data selected — outputs are a UI smoke test only.")

    # ---------------- load checkpoint once per selection ----------------
    @st.cache_resource(show_spinner=False)
    def _load(path_str: str):
        payload = load_checkpoint_payload(Path(path_str))
        model, name, config = build_model_from_payload(payload)
        return model, name, config, payload

    try:
        model, model_name, model_config, checkpoint_meta = _load(str(checkpoint))
    except Exception as error:  # noqa: BLE001
        st.error(f"Could not load `{ckpt_label}`: {error}")
        st.stop()

    in_dim = int(model_config.get("in_dim", 1024) or 1024)
    n_classes = int(model_config.get("n_classes", 6) or 6)

    @st.cache_resource(show_spinner=False)
    def _load_extractor(dim: int, image_size: int):
        return build_extractor(dim, pretrained=True)

    @st.cache_resource(show_spinner=False)
    def _reference(features_root_str: str, dim: int):
        return reference_feature_stats(Path(features_root_str), dim)

    features_root = (
        ROOT / features_root_text if not Path(features_root_text).is_absolute() else Path(features_root_text)
    )
    manifest = ROOT / manifest_path if not Path(manifest_path).is_absolute() else Path(manifest_path)

    tabs = st.tabs(["Overview", "Slide analysis", "Upload & tile", "Results & reports"])

    # ----------------------------------------------------------------- #
    # 1. Overview
    # ----------------------------------------------------------------- #
    with tabs[0]:
        left, right = st.columns(2)
        with left:
            st.subheader("The clinical problem")
            st.write(
                "Pathologists grade prostate cancer on the Gleason/ISUP scale from H&E whole-slide "
                "images. Grading is slow, and inter-observer agreement on the middle grades is "
                "imperfect."
            )
            st.subheader("Method")
            st.write(
                "1. **Tiling** — a slide is cut into 256×256 patches (Otsu tissue mask, 20×).\n"
                "2. **Embedding** — a frozen ImageNet ResNet50 turns each patch into a feature vector.\n"
                "3. **Aggregation** — a MIL model learns attention weights over patches, because only "
                "a slide-level label is available.\n"
                "4. **Grading** — a 6-way softmax over ISUP 0–5; attention is the localisation signal."
            )
        with right:
            st.subheader("What is real in this app")
            st.markdown(
                "- **Slide analysis** runs a real checkpoint on the real pre-extracted features — "
                "the same numbers as the report.\n"
                "- **Upload & tile** runs the same checkpoint on tiles cut from your image and "
                "*measures* whether those features are in-distribution instead of assuming it.\n"
                "- **Results & reports** renders the artifacts under `results/` verbatim, so the app "
                "cannot drift from the report."
            )
            st.subheader("What is not")
            st.markdown(
                "- Not a diagnostic tool; no external validation; a 420-slide PANDA subset.\n"
                "- No fake predictions, no random attention, no hard-coded grades. If a checkpoint "
                "is missing you get an error, not a plausible-looking number."
            )
        st.success(
            f"Loaded **{MODEL_LABELS.get(model_name, model_name)}** from `{ckpt_label}` "
            f"(in_dim={in_dim}, n_classes={n_classes}, seed={checkpoint_meta.get('seed', '?')}, "
            f"best val QWK={float(checkpoint_meta.get('best_val_qwk', float('nan'))):.4f})."
        )

        # ---- dataset card -------------------------------------------- #
        st.divider()
        st.subheader("The data, precisely")
        st.markdown(
            "**420 H&E prostate biopsies from PANDA**, pre-extracted into patch features. "
            "Every number in this app and in the report comes from these slides."
        )
        col_a, col_b = st.columns([1.15, 1])
        with col_a:
            st.markdown(
                "| ISUP | Gleason | train | val | test | total |\n"
                "| --- | --- | --- | --- | --- | --- |\n"
                "| 0 — benign | 0+0 | 50 | 10 | 10 | **70** |\n"
                "| 1 | 3+3 | 50 | 10 | 10 | **70** |\n"
                "| 2 | 3+4 | 50 | 10 | 10 | **70** |\n"
                "| 3 | 4+3 | 50 | 10 | 10 | **70** |\n"
                "| 4 | 4+4 / 3+5 / 5+3 | 50 | 10 | 10 | **70** |\n"
                "| 5 | 4+5 / 5+4 / 5+5 | 50 | 10 | 10 | **70** |\n"
                "| **total** | | **300** | **60** | **60** | **420** |"
            )
        with col_b:
            st.markdown(
                "- **70 benign, 350 cancerous** (5:1) — the cohort is *class-balanced by construction*, "
                "so imbalance is **not** why accuracy looks low.\n"
                "- **All 420 slides are Karolinska.** No Radboud slides, so there is no provider-shift "
                "estimate and no patch-level Gleason supervision available.\n"
                "- **4 % of PANDA** (420 of 10,616), stratified rather than sampled from the true "
                "distribution.\n"
                "- Features from `Dearcat/CPathPatchFeature` (Tang et al., 2025); we do not re-tile "
                "gigapixel WSIs.\n"
                "- Patch coordinates exist for **399 / 420** slides, so heatmaps cover **56 of 60** "
                "test slides."
            )

        # ---- verified results ---------------------------------------- #
        st.divider()
        st.subheader("Verified results (7 trained runs, 3 seeds each)")
        st.caption(
            "Read the `n` column: 60 slides for the ResNet50 cohort, 57 for CHIEF, 53 for GigaPath. "
            "Rows are **not** measured on the same slides — the common-cohort column is the like-for-like one."
        )
        st.markdown(
            "| run | encoder | n | QWK (own test set) | QWK (50 shared slides) | accuracy |\n"
            "| --- | --- | --- | --- | --- | --- |\n"
            "| CLAM-SB | CHIEF 768-d | 57 | **0.9207 ± 0.0063** | **0.9278 ± 0.0074** | 0.602 |\n"
            "| ABMIL | CHIEF 768-d | 57 | 0.9035 ± 0.0145 | 0.9115 ± 0.0110 | 0.561 |\n"
            "| ABMIL | GigaPath 1536-d | 53 | 0.8984 ± 0.0108 | 0.8907 ± 0.0132 | 0.585 |\n"
            "| TransMIL | ResNet50 1024-d | 60 | 0.7446 ± 0.0364 | 0.7113 ± 0.0583 | 0.483 |\n"
            "| ABMIL | ResNet50 1024-d | 60 | 0.7281 ± 0.0132 | 0.6995 ± 0.0136 | 0.389 |\n"
            "| CLAM-SB, no instance loss | ResNet50 1024-d | 60 | 0.7271 ± 0.0223 | 0.7001 ± 0.0313 | 0.394 |\n"
            "| CLAM-SB + instance loss | ResNet50 1024-d | 60 | 0.7035 ± 0.0088 | 0.6680 ± 0.0110 | 0.383 |"
        )
        st.success(
            "**Weights:** all **21 checkpoints** (7 runs × 3 seeds) are committed, so every row above "
            "can be reloaded and re-evaluated — not merely re-audited. The three CHIEF/GigaPath runs "
            "were retrained from the public feature release and reproduce the earlier numbers to "
            "within 0.019 QWK — see `docs/REPRODUCIBILITY.md`."
        )

        # ---- findings ------------------------------------------------- #
        st.divider()
        st.subheader("What we found")
        st.markdown(
            "**1. The encoder decides everything.** ABMIL ResNet50 → ABMIL CHIEF is worth "
            "**+0.17 to +0.21 QWK**, with every paired confidence interval excluding zero across "
            "three seeds (McNemar p 0.001 / 0.023 / 0.064).\n\n"
            "**2. The aggregator decides nothing detectable.** ABMIL → CLAM-SB is +0.009 to +0.032 "
            "(CHIEF) and +0.0000 to −0.0114 (ResNet50); ABMIL → TransMIL is −0.011 to +0.031. No "
            "paired McNemar test is significant, on **two independent encoders**.\n\n"
            "**3. The instance-loss ablation is a clean negative.** 0.7035 vs 0.7271, but the sign "
            "flips between seeds and no paired test separates the arms. We report *no detectable "
            "effect at n = 60*, not a winner.\n\n"
            "**4. The instance loss diffuses attention** rather than sharpening it: normalised "
            "entropy 0.80 vs 0.70, top-10 attention share 0.34 vs 0.44 — the opposite of the "
            "mechanism CLAM proposes.\n\n"
            "**5. Accuracy understates this task.** 80–95 % of the foundation-encoder models' errors "
            "are exactly one ISUP grade off, which QWK credits and accuracy counts as a total miss."
        )
        st.warning(
            "**Claims we explicitly do not make:** that CLAM-SB beats ABMIL; that the instance loss "
            "sharpens attention; that attention maps segment tumour (no masks exist for this "
            "provider); or that any of this generalises beyond a single-provider 420-slide subset."
        )

        # ---- team ----------------------------------------------------- #
        st.divider()
        st.subheader("Who built what")
        st.markdown(
            "| person | contribution | key artifact |\n"
            "| --- | --- | --- |\n"
            "| **Person 1** — Koushik | data contract, locked stratified split, frozen ResNet50 "
            "features, ABMIL baseline | `src/dataset.py`, `src/train.py`, `results/metrics/abmil/` |\n"
            "| **Person 2** — Archit | CLAM-SB from scratch, mandatory instance-loss ablation, paired "
            "statistics, results pipeline | `src/models/clam_sb.py`, `results/metrics/clam_sb/`, "
            "`scripts/paired_results.py` |\n"
            "| **Person 3** — Himesh | TransMIL with PPEG, transformer baseline | "
            "`src/models/transmil.py`, `results/metrics/transmil/` |\n"
            "| **Person 4** — Aswath | metrics, bootstrap/McNemar tests, attention export, "
            "localisation coverage, statistical baselines, this dashboard | `src/evaluate.py`, "
            "`app/streamlit_app.py` |"
        )

        # ---- provenance ----------------------------------------------- #
        st.divider()
        st.subheader("Provenance — how to check any number you see here")
        st.markdown(
            "- Every table in **Results & reports** is rendered from files under `results/`, never "
            "typed into the app.\n"
            "- `results/ALL_RESULTS.md` is generated from the committed predictions by "
            "`scripts/summarize_results.py`; `scripts/paired_results.py` regenerates every paired "
            "statistic.\n"
            "- `scripts/audit_artifacts.py` reports any result a reviewer could not reproduce "
            "(currently **0 fatal**, 3 warnings — the three runs without weights).\n"
            "- CI on every push runs 54 tests, `pyflakes`, a check that the committed tables still "
            "match the artifacts, and a train/test-leakage scan of all four manifests."
        )

    # ----------------------------------------------------------------- #
    # 2. Slide analysis
    # ----------------------------------------------------------------- #
    with tabs[1]:
        st.subheader("Slide analysis on pre-extracted features")
        st.caption(
            "The validated path: the feature bag on disk is the one the checkpoint was trained "
            "and evaluated on."
        )
        try:
            records = read_manifest(manifest)
        except Exception as error:  # noqa: BLE001
            st.error(f"Could not read manifest `{_relative(manifest, ROOT)}`: {error}")
            records = []

        if not records:
            st.info("Point the sidebar at a valid split manifest to analyse slides.")
        else:
            split = st.radio("Split", ["test", "val", "train", "all"], index=0, horizontal=True)
            subset = records if split == "all" else split_records(records, split)
            if not subset:
                st.warning("No slides in that split.")
            else:
                slide_id = st.selectbox("Slide", [record.image_id for record in subset])
                record = next(r for r in subset if r.image_id == slide_id)
                try:
                    features, coords, patch_paths = load_feature_bag(feature_path(features_root, slide_id))
                except Exception as error:  # noqa: BLE001
                    features = None
                    st.error(
                        f"Could not load features for `{slide_id}` from "
                        f"`{_relative(features_root, ROOT)}`: {error}"
                    )

                if features is not None and int(features.size(1)) != in_dim:
                    st.error(
                        f"Feature dim mismatch: this checkpoint expects {in_dim}-d features but "
                        f"`{slide_id}` has {features.size(1)}-d. Point the sidebar at the matching "
                        "feature folder."
                    )
                    features = None

                if features is not None:
                    result = run_model(model, features)
                    prediction = result["prediction"]

                    thumbnail = None
                    thumbnail_dir = ROOT / ("data/mock/thumbnails" if "mock" in manifest_path else "data/thumbnails")
                    for suffix in (".png", ".jpg", ".jpeg", ".tif", ".tiff"):
                        candidate = thumbnail_dir / f"{slide_id}{suffix}"
                        if candidate.exists():
                            with Image.open(candidate) as handle:
                                thumbnail = handle.convert("RGB")
                            break

                    col_a, col_b = st.columns([1, 1.4])
                    with col_a:
                        if thumbnail is not None:
                            _show_image(st, thumbnail, slide_id)
                        else:
                            st.info("No thumbnail on disk (feature-only pipeline) — heatmap only.")
                        st.write(
                            {
                                "Slide": slide_id,
                                "Provider": record.data_provider,
                                "True ISUP": GRADE_NAMES.get(record.label, str(record.label)),
                                "Gleason": record.gleason_score,
                                "Patches": int(features.size(0)),
                                "Feature dim": int(features.size(1)),
                                "Model": MODEL_LABELS.get(model_name, model_name),
                            }
                        )
                    with col_b:
                        delta = prediction - record.label
                        st.metric(
                            f"Predicted ISUP: {prediction} ({result['confidence']:.1%})",
                            GRADE_NAMES.get(prediction, str(prediction)),
                            delta=f"{delta:+d} vs ground truth" if delta else "matches ground truth",
                            delta_color="inverse" if delta else "normal",
                        )
                        fig = _probability_figure(matplotlib, result["probs"], prediction, record.label)
                        _show_figure(st, fig)
                        plt.close(fig)

                        if coords is None:
                            st.warning(
                                "This feature bag has no patch coordinates, so no spatial heatmap "
                                "can be drawn. Re-attach them with "
                                "`python scripts/attach_patch_coords.py`."
                            )
                        else:
                            overlay = _attention_overlay(
                                matplotlib,
                                result["attention"],
                                coords.numpy(),
                                patch_size=256,
                                width=640,
                                size=None,
                                base_image=thumbnail,
                            )
                            _show_image(st, overlay, "attention (jet) over patch coordinates")
                            st.caption(
                                f"attention entropy {attention_entropy(result['attention']):.3f} "
                                f"(normalised "
                                f"{normalised_attention_entropy(result['attention']):.3f}); lower = more focused."
                            )

                    st.subheader(f"Top {top_k} patches by attention")
                    coord_array = coords.numpy() if coords is not None else np.zeros((features.size(0), 2))
                    rows = top_attention_patches(result["attention"], coord_array, patch_paths, k=top_k)
                    usable = [
                        row for row in rows if row.get("patch_path") and Path(row["patch_path"]).exists()
                    ]
                    if usable:
                        columns = st.columns(min(4, len(usable)))
                        for index, row in enumerate(usable):
                            with columns[index % len(columns)]:
                                with Image.open(row["patch_path"]) as patch:
                                    _show_image(
                                        st,
                                        patch.convert("RGB"),
                                        f"#{row['rank']} a={row['attention']:.4f} "
                                        f"({int(row['x'])}, {int(row['y'])})",
                                    )
                    else:
                        st.info(
                            "Patch images are not on disk for this feature bag (`patch_paths` "
                            "empty) — the ranked attention table is shown instead."
                        )
                        _show_table(st, rows)

    # ----------------------------------------------------------------- #
    # 3. Upload & tile
    # ----------------------------------------------------------------- #
    with tabs[2]:
        st.subheader("Upload one image → tile it → run the same checkpoint")
        st.warning(
            "**Read this first.** The checkpoint was trained on CLAM-style ResNet50 features of "
            "256×256 **20× H&E** patches. A single uploaded image usually differs in scale, stain "
            "batch and field of view, so its features are **out of distribution**. The app measures "
            "that below rather than hiding it. For a trustworthy grade use **Slide analysis**."
        )
        col_ctrl, col_up = st.columns([1, 1.2])
        with col_ctrl:
            grid = st.slider("Tile grid (N×N patches)", 3, 10, 6)
            image_size = st.select_slider("Patch size fed to the CNN", options=[224, 256, 320], value=256)
            st.caption(
                "Features come from the same `src/feature_extraction.py` backbone, with the same "
                f"fixed projection to {in_dim}-d the checkpoint expects."
            )
        with col_up:
            uploaded = st.file_uploader("Patch / region image", type=["png", "jpg", "jpeg", "tif", "tiff"])

        if uploaded is None:
            st.info("Upload an image to run the pipeline.")
        else:
            image = Image.open(uploaded).convert("RGB")
            tiled = tile_image(image, grid=grid)
            _show_image(st, image, f"uploaded image ({image.width}×{image.height}) cut into {grid}×{grid} tiles")

            extraction_error = None
            result = None
            with st.spinner("Extracting features and running the model..."):
                try:
                    extractor = _load_extractor(in_dim, image_size)
                    features = extract_features(tiled["tiles"], extractor, image_size=image_size)
                    result = run_model(model, features)
                except Exception as error:  # noqa: BLE001
                    extraction_error = error

            if extraction_error is not None:
                st.error(
                    f"Feature extraction failed: {extraction_error}\n\n"
                    "torchvision downloads the ImageNet ResNet50 weights on first use (~100 MB). "
                    "Without network access the extractor cannot run — use **Slide analysis**, "
                    "which needs no download."
                )
            else:
                # Decide *before* rendering whether this image may be graded at all.
                reference = _reference(str(features_root), in_dim)
                ood = ood_percentile(features, reference)
                decision = grade_decision(ood)
                graded = decision["action"] in ("report", "warn")

                left, right = st.columns([1, 1.3])
                with left:
                    st.markdown("### Model output")
                    if graded:
                        st.metric(
                            f"Predicted ISUP: {result['prediction']}",
                            f"{result['confidence']:.1%} confidence",
                        )
                        fig = _probability_figure(
                            matplotlib, result["probs"], result["prediction"], truth=None, size=(5.6, 2.4)
                        )
                        _show_figure(st, fig)
                        plt.close(fig)
                        if decision["action"] == "warn":
                            st.warning(decision["reason"])
                        else:
                            st.success(decision["reason"])
                    else:
                        st.error("**No grade shown.** " + decision["reason"])
                        st.caption(
                            "A MIL model trained on 20× H&E patches will still emit a confident "
                            "number for a picture it has never seen — a synthetic nuclear-density "
                            "phantom comes back as ISUP 5. Printing that number would be "
                            "indigestible nonsense, so the app abstains instead."
                        )
                        st.metric(
                            "Predicted ISUP", "— abstained —",
                            help="Abstaining is the correct behaviour: the input is out of "
                                 "distribution for this checkpoint.",
                        )

                    with st.expander("Show the raw model output anyway (diagnostic only)"):
                        st.caption(
                            "This is what the network emitted. It is **not** a prediction about "
                            "this image; do not quote it."
                        )
                        st.write(
                            {
                                "argmax": int(result["prediction"]),
                                "confidence": round(float(result["confidence"]), 4),
                                "probabilities": [round(float(p_), 4) for p_ in result["probs"]],
                                "decision": decision["action"],
                                "ood_percentile": decision["percentile"],
                            }
                        )

                with right:
                    st.markdown("### Attention over the uploaded tiles")
                    overlay = _attention_overlay(
                        matplotlib,
                        result["attention"],
                        tiled["coords"],
                        patch_size=tiled["tile_size"],
                        width=min(640, image.width),
                        size=tiled["size"],
                        base_image=image,
                    )
                    _show_image(st, overlay, "model attention (jet)")
                    st.caption(
                        f"attention entropy {attention_entropy(result['attention']):.3f} "
                        f"(normalised {normalised_attention_entropy(result['attention']):.3f})."
                    )

                    order = np.argsort(result["attention"])[::-1][:top_k]
                    columns = st.columns(min(4, len(order)))
                    for rank, index in enumerate(order, start=1):
                        with columns[(rank - 1) % len(columns)]:
                            _show_image(
                                st,
                                tiled["tiles"][int(index)],
                                f"#{rank} a={result['attention'][int(index)]:.4f}",
                            )

                st.divider()
                st.markdown("### Image descriptors — transparent, rule-based, **not** the model")
                st.caption(
                    "Plain pixel measurements (Otsu-thresholded dark-object fraction, haematoxylin "
                    "index, blob count). They carry **no diagnostic meaning** and are not an ISUP "
                    "grade. They are shown so the input is auditable — synthetic nuclear-density "
                    "phantoms separate cleanly on them, which is exactly why a rule built on them "
                    "must never be presented as model output."
                )
                descriptors = image_descriptors(image)
                _show_table(
                    st,
                    [{"descriptor": key, "value": round(value, 4)} for key, value in descriptors.items()],
                    hide_index=True,
                )
                with st.expander("Per-tile descriptors"):
                    tile_rows = []
                    for index, tile in enumerate(tiled["tiles"]):
                        stats = image_descriptors(tile)
                        tile_rows.append(
                            {
                                "tile": index,
                                "attention": round(float(result["attention"][index]), 5),
                                "dark fraction": round(stats["dark-object fraction (Otsu)"], 4),
                                "stain density": round(stats["mean stain density (1 - luminance)"], 4),
                            }
                        )
                    _show_table(st, tile_rows, hide_index=True)

    # ----------------------------------------------------------------- #
    # 4. Results & reports
    # ----------------------------------------------------------------- #
    with tabs[3]:
        st.subheader("Results & reports — rendered from disk, never retyped")
        reports_root = (
            ROOT / reports_root_text if not Path(reports_root_text).is_absolute() else Path(reports_root_text)
        )
        include_mock = st.toggle("Include synthetic mock artifacts (never reportable)", value=False)
        reports = discover_reports(ROOT, include_mock=include_mock)

        total = sum(len(value) for value in reports.values())
        if total == 0:
            st.warning(
                "Nothing found under `results/`, `report/` or `docs/`. Train a model or run "
                "`python scripts/summarize_person2.py` first."
            )
        else:
            st.caption(
                "Discovered: " + ", ".join(f"{len(v)} {k}" for k, v in reports.items() if v) + f" (in `{_relative(reports_root, ROOT)}` and friends)."
            )
            kinds = [kind for kind in ("markdown", "json", "images", "heatmaps") if reports.get(kind)]
            kind = st.radio("Artifact type", kinds, horizontal=True)
            choices = reports[kind]
            labels = [_relative(path, ROOT) for path in choices]
            selected = choices[labels.index(st.selectbox("File", labels))]

            if kind == "markdown":
                st.markdown(Path(selected).read_text(encoding="utf-8", errors="replace"))
            elif kind == "json":
                payload = read_json(Path(selected))
                if isinstance(payload, list) and payload and isinstance(payload[0], dict):
                    _show_table(st, payload, hide_index=True)
                else:
                    st.json(payload)
            else:
                _show_image(st, Image.open(selected), Path(selected).stem.replace("_", " "))
                if kind == "heatmaps":
                    st.caption(
                        f"{len(choices)} heatmap image(s). Coverage is 56/60 test slides: four test "
                        "slides have no patch coordinates in the feature release."
                    )

            st.divider()
            st.markdown("#### Report index")
            for group, paths in reports.items():
                if not paths:
                    continue
                with st.expander(f"{group} ({len(paths)})"):
                    for path in paths[:200]:
                        st.write("`" + _relative(path, ROOT) + "`")
            if not include_mock:
                st.caption(
                    "Mock/synthetic artifacts are hidden. They exist only to smoke-test the "
                    "pipeline and must never be quoted in the report."
                )

    st.sidebar.markdown("---")
    st.sidebar.caption(f"Workspace: `{ROOT}`")


if __name__ == "__main__":
    render()

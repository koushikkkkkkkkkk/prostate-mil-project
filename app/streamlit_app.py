"""Interactive MIL demo -- slide grading + attention localisation.

Run locally::

    streamlit run app/streamlit_app.py

What it shows for one slide
---------------------------
* slide metadata (provider, true ISUP grade, Gleason score)
* predicted ISUP grade group + class-confidence bars
* attention heatmap overlay on the slide thumbnail
* the top-k highest-attention patches ("what the model looked at")

It works on the real PANDA features once they exist, and on the synthetic mock
features before that (``data/mock``) so the UI can be built and demoed early.

IMPORTANT: this is a research prototype. It is not a diagnostic tool and must
not be presented as one -- see the disclaimer rendered in the sidebar.

Owner: Person 4 (app), with model/checkpoint hooks from Persons 1-3.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.dataset import (  # noqa: E402
    feature_path,
    load_feature_bag,
    read_manifest,
    split_records,
)
from src.heatmaps import (  # noqa: E402
    attention_entropy,
    normalised_attention_entropy,
    normalise_attention,
    scatter_attention,
    top_attention_patches,
)
from src.models import build_model  # noqa: E402

GRADE_NAMES = {
    0: "0 - benign / negative",
    1: "1 - Gleason 3+3",
    2: "2 - Gleason 3+4",
    3: "3 - Gleason 4+3",
    4: "4 - Gleason 4+4",
    5: "5 - Gleason 5+5 (or higher)",
}


# --------------------------------------------------------------------------- #
# Cached helpers
# --------------------------------------------------------------------------- #
def list_checkpoints(root: Path) -> List[Path]:
    """Every ``*_best.pt`` under the results folder, newest first."""
    candidates = [*root.glob("results/**/checkpoints/*_best.pt"), *root.glob("results/**/*_best.pt")]
    seen, unique = set(), []
    for path in candidates:
        if path not in seen:
            seen.add(path)
            unique.append(path)
    return sorted(unique, key=lambda p: p.stat().st_mtime, reverse=True)


def load_model(checkpoint: Path, device: str = "cpu") -> Tuple[torch.nn.Module, Dict[str, Any]]:
    try:
        payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    except TypeError:  # torch < 2.0
        payload = torch.load(checkpoint, map_location="cpu")
    model = build_model(payload["model_name"], payload.get("model_config", {}))
    model.load_state_dict(payload["state_dict"])
    return model.to(device).eval(), payload


@torch.no_grad()
def predict_slide(
    model: torch.nn.Module, features: torch.Tensor, device: str = "cpu"
) -> Tuple[np.ndarray, np.ndarray]:
    output = model(features.to(device), use_instance_loss=False)
    probs = torch.softmax(output["logits"], dim=-1).squeeze(0).cpu().numpy()
    attention = output["attention"].squeeze(0).cpu().numpy()
    return probs, attention


def load_thumbnail(path: Optional[Path], width: int = 768):
    from PIL import Image

    if path is None or not Path(path).exists():
        return None
    with Image.open(path) as image:
        image = image.convert("RGB")
        scale = width / max(image.width, 1)
        return image.resize((width, max(1, int(image.height * scale))))


def find_thumbnail(thumbnails_dir: Optional[str], slide_id: str) -> Optional[Path]:
    if not thumbnails_dir:
        return None
    base = Path(thumbnails_dir)
    for suffix in (".png", ".jpg", ".jpeg", ".tif", ".tiff"):
        candidate = base / f"{slide_id}{suffix}"
        if candidate.exists():
            return candidate
    return None


# --------------------------------------------------------------------------- #
# App
# --------------------------------------------------------------------------- #
def render() -> None:
    import matplotlib
    import streamlit as st
    from PIL import Image

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    st.set_page_config(page_title="Patches to Prognosis", page_icon="🔬", layout="wide")
    st.title("From Patches to Prognosis")
    st.caption(
        "Weakly supervised ISUP grading and attention localisation from whole-slide patches "
        "(Multiple Instance Learning)."
    )

    st.sidebar.header("Configuration")
    default_manifest = "data/mock/manifest.csv" if (ROOT / "data/mock/manifest.csv").exists() else "data/common_split.csv"
    manifest_path = st.sidebar.text_input("Split manifest", default_manifest)
    features_root = st.sidebar.text_input(
        "Feature folder",
        "data/mock/features" if "mock" in manifest_path else "data/features",
    )
    thumbnails_dir = st.sidebar.text_input(
        "Thumbnails folder (optional)",
        "data/mock/thumbnails" if "mock" in manifest_path else "data/thumbnails",
    )

    checkpoints = list_checkpoints(ROOT)
    if not checkpoints:
        st.error(
            "No trained checkpoint found under `results/`. Train one first:\n\n"
            "```bash\nbash scripts/run_mock_demo.sh\n```"
        )
        st.stop()
    labels = [str(path.relative_to(ROOT)) for path in checkpoints]
    default_index = next((i for i, l in enumerate(labels) if "clam_sb" in l), 0)
    checkpoint_label = st.sidebar.selectbox("Model checkpoint", labels, index=default_index)
    checkpoint = ROOT / checkpoint_label

    split = st.sidebar.selectbox("Split", ["test", "val", "train", "all"], index=0)
    top_k = st.sidebar.slider("Top suspicious patches", 3, 20, 6)

    st.sidebar.markdown("---")
    st.sidebar.warning(
        "**Research prototype.** Not a clinical device. Trained/evaluated on a 420-slide "
        "PANDA subset; attention maps are model cues, not tumour segmentations. "
        "External validation is required before any real-world use."
    )
    if "mock" in manifest_path:
        st.sidebar.info("SYNTHETIC mock data detected -- outputs are a UI smoke test only.")

    try:
        records = read_manifest(ROOT / manifest_path if not Path(manifest_path).is_absolute() else manifest_path)
    except Exception as exc:  # noqa: BLE001
        st.error(f"Could not read manifest: {exc}")
        st.stop()

    records = records if split == "all" else split_records(records, split)
    if not records:
        st.warning("No slides in that split.")
        st.stop()

    slide_ids = [record.image_id for record in records]
    selected_id = st.selectbox("Slide", slide_ids)
    record = next(r for r in records if r.image_id == selected_id)

    try:
        model, payload = load_model(checkpoint)
    except Exception as exc:  # noqa: BLE001
        st.error(f"Could not load checkpoint {checkpoint_label}: {exc}")
        st.stop()

    root_features = ROOT / features_root if not Path(features_root).is_absolute() else Path(features_root)
    try:
        features, coords, patch_paths = load_feature_bag(feature_path(root_features, selected_id))
    except Exception as exc:  # noqa: BLE001
        st.error(f"Could not load features for {selected_id}: {exc}")
        st.stop()

    probs, attention = predict_slide(model, features)
    prediction = int(np.argmax(probs))

    left, right = st.columns([1, 1.35])
    with left:
        st.subheader("Slide")
        thumbnail = load_thumbnail(find_thumbnail(thumbnails_dir, selected_id))
        if thumbnail is not None:
            st.image(thumbnail, caption=f"{selected_id}", use_container_width=True)
        else:
            st.info("No thumbnail available for this slide.")

        st.subheader("Metadata")
        st.write(
            {
                "Slide ID": selected_id,
                "Data provider": record.data_provider,
                "True ISUP grade": GRADE_NAMES.get(record.label, str(record.label)),
                "Gleason score": record.gleason_score,
                "Patches": int(features.size(0)),
                "Feature dim": int(features.size(1)),
                "Model": payload.get("model_name", "unknown"),
                "Seed": payload.get("seed", "?"),
            }
        )

    with right:
        st.subheader("Prediction")
        delta = prediction - record.label
        st.metric(
            f"Predicted ISUP grade: {prediction}",
            f"{probs[prediction]:.1%} confidence",
            delta=f"{delta:+d} vs ground truth" if delta != 0 else "matches ground truth",
            delta_color="inverse" if delta != 0 else "normal",
        )

        fig, ax = plt.subplots(figsize=(6.0, 2.6))
        colours = ["#c44e52" if i == prediction else "#4c72b0" for i in range(len(probs))]
        ax.bar([str(i) for i in range(len(probs))], probs, color=colours)
        ax.axvline(record.label, color="#55a868", linestyle="--", linewidth=1.2, label="true grade")
        ax.set_xlabel("ISUP grade group")
        ax.set_ylabel("probability")
        ax.set_ylim(0, 1)
        ax.legend(frameon=False, fontsize=8)
        st.pyplot(fig, use_container_width=True)
        plt.close(fig)

        st.subheader("Attention heatmap")
        if coords is None:
            st.warning(
                "This feature file has no patch coordinates, so a spatial heatmap cannot be "
                "drawn. Re-export features with `src/feature_extraction.py`."
            )
        else:
            width = 768
            attention_map = scatter_attention(
                attention, coords.numpy(), patch_size=256, out_width=width
            )
            normalised = normalise_attention(attention_map)
            colour = (matplotlib.colormaps["jet"](normalised)[..., :3] * 255).astype(np.uint8)
            base = thumbnail.resize((normalised.shape[1], normalised.shape[0])) if thumbnail else Image.new(
                "RGB", (normalised.shape[1], normalised.shape[0]), "white"
            )
            overlay = 0.5 * np.asarray(base, dtype=np.float32) + 0.5 * colour.astype(np.float32)
            st.image(
                np.clip(overlay, 0, 255).astype(np.uint8),
                caption="attention (jet) alpha-blended on the slide thumbnail",
                use_container_width=True,
            )
            st.caption(
                f"attention entropy {attention_entropy(attention):.3f} "
                f"(normalised {normalised_attention_entropy(attention):.3f}); "
                "lower = more focused."
            )

    st.subheader("Top suspicious patches")
    coords_array = coords.numpy() if coords is not None else np.zeros((features.size(0), 2))
    rows = top_attention_patches(attention, coords_array, patch_paths, k=top_k)
    usable = [row for row in rows if row.get("patch_path") and Path(row["patch_path"]).exists()]
    if usable:
        columns = st.columns(min(4, len(usable)))
        for i, row in enumerate(usable):
            with columns[i % len(columns)]:
                with Image.open(row["patch_path"]) as patch:
                    st.image(
                        patch.convert("RGB"),
                        caption=f"#{row['rank']} a={row['attention']:.4f}\n({int(row['x'])}, {int(row['y'])})",
                        use_container_width=True,
                    )
    else:
        st.info(
            "Patch images are not available for this slide (the feature file has no "
            "`patch_paths`). The ranked attention table is shown instead."
        )
        st.table(rows)

    with st.expander("Attention table (top-k)"):
        st.dataframe(rows, use_container_width=True)

    with st.expander("How to read this"):
        st.markdown(
            "- **QWK** is the headline metric in the report; accuracy alone is misleading "
            "because ISUP grades are imbalanced.\n"
            "- **Attention** shows where the model looked, not where tumour definitely is. "
            "PANDA has no pixel-level annotations, so overlap scores use coarse masks.\n"
            "- A confident wrong prediction here is a *feature* of a weak-label MIL model "
            "on a small subset, and is exactly what the report's error analysis discusses."
        )


if __name__ == "__main__":
    render()

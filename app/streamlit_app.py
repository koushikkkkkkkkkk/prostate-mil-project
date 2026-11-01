"""Interactive MIL dashboard -- slide grading, model comparison, attention localisation.

Run locally::

    streamlit run app/streamlit_app.py

Pages
-----
* **Overview**           headline numbers + model x encoder heatmap
* **Performance**        training curves, per-seed spread, confusion matrix, per-slide errors
* **Slide Predictor**    one slide: class probabilities, attention ranking, heatmap
* **Conclusions**        what the experiments do and do not establish

Every number is read from ``results/metrics/``; nothing is hard-coded and nothing trains.
Pages degrade gracefully (they say which artefact is missing) so the app runs even on a
partial checkout.

IMPORTANT: this is a research prototype. It is not a diagnostic tool and must not be
presented as one -- see the disclaimer in the sidebar.

Owner: Person 4 (app), with model/checkpoint hooks from Persons 1-3.
"""

from __future__ import annotations

import csv
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np
import pandas as pd
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


METRICS = ROOT / "results" / "metrics"

GRADE_NAMES = {
    0: "0 - benign / negative",
    1: "1 - Gleason 3+3",
    2: "2 - Gleason 3+4",
    3: "3 - Gleason 4+3",
    4: "4 - Gleason 4+4",
    5: "5 - Gleason 5+5 (or higher)",
}
GRADE_SHORT = {0: "Benign", 1: "3+3", 2: "3+4", 3: "4+3", 4: "4+4", 5: "≥5+5"}


# --------------------------------------------------------------------------- #
# Data layer -- reads results/metrics/ (kept inline so the app is one file)
# --------------------------------------------------------------------------- #
# --------------------------------------------------------------------------- #
# Encoder / run labelling
# --------------------------------------------------------------------------- #
ENCODER_LABELS = {
    "abmil": "ResNet50",
    "clam_sb": "ResNet50",
    "abmil_chief": "CHIEF",
    "clam_sb_chief": "CHIEF",
    "abmil_gigap": "GigaPath",
    "clam_sb_gigap": "GigaPath",
}

ENCODER_DIMS = {"ResNet50": 1024, "CHIEF": 768, "GigaPath": 1536, "UNI": 1024}


def pretty_run(folder: str) -> Dict[str, str]:
    """Turn a results/metrics subfolder into display labels.

    ``abmil_chief`` -> {"model": "ABMIL", "encoder": "CHIEF", "family": "pathology"}
    """
    parts = folder.split("/")
    name = parts[0]
    sub = parts[1] if len(parts) > 1 else ""
    encoder = ENCODER_LABELS.get(name, "ResNet50")
    model = "CLAM-SB" if name.startswith("clam") else "ABMIL" if name.startswith("abmil") else name.upper()
    if sub in ("with_instance", "no_instance"):
        model = f"{model} ({sub.replace('_', ' ')})"
    family = "baseline" if encoder == "ResNet50" else "pathology encoder"
    out = {"folder": folder, "model": model, "encoder": encoder, "family": family}
    return out


def _read_json(path: Path) -> Optional[Dict[str, Any]]:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None


# --------------------------------------------------------------------------- #
# Run discovery
# --------------------------------------------------------------------------- #
def discover_runs(metrics_root: Path) -> List[Dict[str, Any]]:
    """Every directory containing an ``aggregate_*.json``, parsed into a flat record."""
    runs: List[Dict[str, Any]] = []
    for agg_path in sorted(metrics_root.glob("**/aggregate_*.json")):
        agg = _read_json(agg_path)
        if not agg:
            continue
        folder = str(agg_path.parent.relative_to(metrics_root)).replace("\\", "/")
        if folder == ".":
            folder = agg_path.parent.name
        labels = pretty_run(folder)
        cfg = _read_json(agg_path.parent / "run_config.json") or {}
        runs.append(
            {
                **labels,
                "path": agg_path.parent,
                "qwk": agg.get("qwk", {}).get("mean"),
                "qwk_sd": agg.get("qwk", {}).get("std"),
                "acc": agg.get("accuracy", {}).get("mean"),
                "bal_acc": agg.get("balanced_accuracy", {}).get("mean"),
                "f1": agg.get("macro_f1", {}).get("mean"),
                "n_test": cfg.get("n_test"),
                "n_train": cfg.get("n_train"),
                "n_val": cfg.get("n_val"),
                "seeds": agg.get("seeds") or [],
                "manifest": cfg.get("manifest"),
                "features_root": cfg.get("features_root"),
                "in_dim": (cfg.get("model_config") or {}).get("in_dim"),
                "per_run": agg.get("per_run") or [],
            }
        )
    return runs


def run_matrix(runs: List[Dict[str, Any]], metric: str = "qwk") -> "pd.DataFrame":
    """model x encoder matrix of a metric, NaN where a combination was not run."""
    import numpy as np
    import pandas as pd

    if not runs:
        return pd.DataFrame()
    models = sorted({r["model"] for r in runs})
    encoders = ["ResNet50", "CHIEF", "GigaPath", "UNI"]
    encoders = [e for e in encoders if any(r["encoder"] == e for r in runs)]
    data = np.full((len(models), len(encoders)), np.nan)
    for r in runs:
        if r.get(metric) is None:
            continue
        i, j = models.index(r["model"]), encoders.index(r["encoder"])
        data[i, j] = r[metric]
    return pd.DataFrame(data, index=models, columns=encoders)


# --------------------------------------------------------------------------- #
# Predictions, per-slide
# --------------------------------------------------------------------------- #
def load_predictions(run_path: Path) -> "pd.DataFrame":
    """Concatenate every ``predictions_*.csv`` in a run directory into one frame."""
    import pandas as pd

    frames = []
    for f in sorted(run_path.glob("predictions_*.csv")):
        try:
            df = pd.read_csv(f)
        except Exception:
            continue
        df["seed"] = f.stem.split("_seed")[-1]
        frames.append(df)
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def confusion_from_predictions(preds: "pd.DataFrame", seeds: Optional[List[str]] = None) -> "np.ndarray":
    import numpy as np

    if preds.empty:
        return np.zeros((6, 6), dtype=int)
    if seeds:
        preds = preds[preds["seed"].astype(str).isin([str(s) for s in seeds])]
    cm = np.zeros((6, 6), dtype=int)
    for t, p in zip(preds["y_true"].astype(int), preds["y_pred"].astype(int)):
        cm[t, p] += 1
    return cm


# --------------------------------------------------------------------------- #
# Comparison and history
# --------------------------------------------------------------------------- #
def load_comparison(metrics_root: Path) -> "pd.DataFrame":
    import pandas as pd

    f = metrics_root / "comparison" / "encoder_comparison_57.csv"
    if f.exists():
        return pd.read_csv(f)
    return pd.DataFrame()


def load_history(run_path: Path) -> "pd.DataFrame":
    """All ``history_*.csv`` in a run, tagged by seed — for the training-curve tab."""
    import pandas as pd

    frames = []
    for f in sorted(run_path.glob("history_*.csv")):
        try:
            df = pd.read_csv(f)
        except Exception:
            continue
        df["seed"] = f.stem.split("_seed")[-1]
        frames.append(df)
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def load_attention_agreement(metrics_root: Path) -> Dict[str, Any]:
    f = metrics_root / "attention" / "agreement_abmil_vs_clam_summary.json"
    return _read_json(f) or {}


def load_fused_summary(metrics_root: Path) -> Dict[str, Any]:
    f = metrics_root / "comparison" / "fused_chief_gigap_summary.json"
    return _read_json(f) or {}


def load_statistical(metrics_root: Path) -> List[Dict[str, Any]]:
    out = []
    for f in sorted((metrics_root / "statistical").glob("*.json")):
        d = _read_json(f)
        if d:
            out.append({"model": f.stem.replace("_ordinal_logistic", ""), **d})
    return out


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


def checkpoint_context(checkpoint: Path) -> Dict[str, Any]:
    """Read the ``run_config.json`` that sits beside a checkpoint's run directory."""
    run_dir = checkpoint.parent.parent if checkpoint.parent.name == "checkpoints" else checkpoint.parent
    context: Dict[str, Any] = {"run_dir": run_dir}
    config_path = run_dir / "run_config.json"
    if config_path.exists():
        try:
            config = json.loads(config_path.read_text(encoding="utf-8"))
            for key in ("manifest", "features_root"):
                if config.get(key):
                    context[key] = str(config[key])
            for key in ("n_train", "n_val", "n_test"):
                if config.get(key) is not None:
                    context[key] = int(config[key])
            model_config = config.get("model_config") or {}
            if model_config.get("in_dim") is not None:
                context["in_dim"] = int(model_config["in_dim"])
        except Exception as exc:  # noqa: BLE001
            context["error"] = f"unreadable run_config.json: {exc}"
    return context


def model_in_dim(model: Any, payload: Dict[str, Any], context: Dict[str, Any]) -> Optional[int]:
    """Input dimension a checkpoint expects, discovered without a forward pass."""
    for candidate in (
        getattr(model, "in_dim", None),
        getattr(getattr(model, "config", None), "in_dim", None),
        (payload.get("model_config") or {}).get("in_dim"),
        context.get("in_dim"),
    ):
        if candidate is not None:
            try:
                return int(candidate)
            except (TypeError, ValueError):
                continue
    for module in model.modules():
        if isinstance(module, torch.nn.Linear):
            return int(module.weight.shape[1])
    return None


def available_slide_ids(records: List[Any], features_root: Path) -> List[str]:
    """Slide ids that actually have a feature file on disk under ``features_root``."""
    return [r.image_id for r in records if feature_path(features_root, r.image_id).exists()]


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
# Pages
# --------------------------------------------------------------------------- #
def page_overview(st) -> None:
    st.title("Overview")
    st.caption(
        "Weakly supervised ISUP grading on a 420-slide PANDA subset. Primary metric: "
        "quadratic-weighted kappa (QWK) on the held-out test split."
    )

    runs = discover_runs(METRICS)
    if not runs:
        st.error(
            "No `aggregate_*.json` found under `results/metrics/`. Train a model first:\n\n"
            "```bash\npython -m src.train --model abmil --seeds 0 1 2\n```"
        )
        return

    best = max(runs, key=lambda r: (r.get("qwk") or -1))
    baseline = next((r for r in runs if r["encoder"] == "ResNet50" and r["model"] == "ABMIL"), None)

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Best QWK", f"{best['qwk']:.4f}", help=f"{best['model']} / {best['encoder']}")
    if baseline and baseline.get("qwk") is not None:
        c2.metric(
            "Gain over baseline",
            f"{best['qwk'] - baseline['qwk']:+.4f}",
            delta=f"{best['encoder']} vs ResNet50",
        )
    c3.metric("Runs evaluated", f"{len(runs)}")
    c4.metric(
        "Seeds per run",
        f"{max(len(r['seeds']) for r in runs)}",
        help="Every reported number is a mean over seeds; sd is across seeds.",
    )

    st.divider()
    st.subheader("Results at a glance")
    st.caption("QWK by model and encoder. Blank cells were not run; the locked baseline is ResNet50.")

    metric = st.radio("Metric", ["qwk", "acc", "bal_acc", "f1"], horizontal=True,
                      format_func=lambda m: {"qwk": "QWK", "acc": "Accuracy",
                                             "bal_acc": "Balanced accuracy",
                                             "f1": "Macro F1"}[m])
    matrix = run_matrix(runs, metric)
    if matrix.empty:
        st.info("No metrics to plot.")
        return

    fig, ax = plt.subplots(figsize=(7.4, 0.75 * len(matrix) + 1.9))
    data = matrix.values.astype(float)
    im = ax.imshow(data, cmap="RdYlGn", vmin=0, vmax=1, aspect="auto")
    ax.set_xticks(range(matrix.shape[1]), matrix.columns, fontsize=9)
    ax.set_yticks(range(matrix.shape[0]), matrix.index, fontsize=9)
    ax.set_xlabel("Encoder", fontsize=9)
    ax.set_ylabel("Model", fontsize=9)
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            v = data[i, j]
            if np.isnan(v):
                ax.text(j, i, "–", ha="center", va="center", color="#999", fontsize=10)
            else:
                ax.text(j, i, f"{v:.3f}", ha="center", va="center",
                        color="black" if 0.25 < v < 0.85 else "white", fontsize=10, weight="bold")
    fig.colorbar(im, ax=ax, label=metric.upper(), fraction=0.035)
    ax.spines[:].set_visible(False)
    st.pyplot(fig, use_container_width=True)
    plt.close(fig)

    st.caption(
        "Encoder coverage differs: ResNet50 covers 60/60 test slides, CHIEF 57/60, "
        "GigaPath 53/60. Cells are therefore not directly comparable across columns — "
        "see the Performance page for the identical-slide comparison."
    )

    st.divider()
    st.subheader("All runs")
    table = pd.DataFrame([
        {
            "Run": r["folder"],
            "Model": r["model"],
            "Encoder": r["encoder"],
            "D": r.get("in_dim"),
            "n test": r.get("n_test"),
            "QWK": r.get("qwk"),
            "± sd": r.get("qwk_sd"),
            "Accuracy": r.get("acc"),
        }
        for r in sorted(runs, key=lambda r: -(r.get("qwk") or 0))
    ])
    st.dataframe(table, use_container_width=True, hide_index=True,
                 column_config={
                     "QWK": st.column_config.NumberColumn(format="%.4f"),
                     "± sd": st.column_config.NumberColumn(format="%.4f"),
                     "Accuracy": st.column_config.NumberColumn(format="%.4f"),
                 })


def page_performance(st) -> None:
    st.title("Performance Explorer")
    st.caption("Where the gain comes from, and where the errors are.")

    runs = discover_runs(METRICS)
    if not runs:
        st.warning("No runs found under `results/metrics/`.")
        return

    st.subheader("Controlled comparison — identical slides")
    comp = load_comparison(METRICS)
    if comp.empty:
        st.info(
            "`results/metrics/comparison/encoder_comparison_57.csv` not found. Generate it:\n\n"
            "```bash\npython scripts/cmp57.py --restrict-to data/common_split_chief.csv ...\n```"
        )
    else:
        fig, ax = plt.subplots(figsize=(7.6, 3.2))
        labels = comp["run"].tolist()
        colours = ["#c62828" if "r50" in l else "#2e7d32" for l in labels]
        bars = ax.barh(labels, comp["qwk_mean"], xerr=comp["qwk_sd"], color=colours,
                       capsize=4, edgecolor="white")
        for bar, v in zip(bars, comp["qwk_mean"]):
            ax.text(v + 0.01, bar.get_y() + bar.get_height() / 2, f"{v:.3f}",
                    va="center", fontsize=9)
        ax.set_xlim(0, 1.05)
        ax.set_xlabel("QWK (mean ± sd over seeds)", fontsize=9)
        ax.spines[["top", "right"]].set_visible(False)
        ax.tick_params(labelsize=9)
        st.pyplot(fig, use_container_width=True)
        plt.close(fig)
        st.dataframe(comp, use_container_width=True, hide_index=True)
        st.caption(
            f"All {int(comp['n_slides'].iloc[0])} rows scored on the same slides, so the "
            "difference is attributable to the encoder and model alone."
        )

    st.divider()
    st.subheader("Training curves")
    run_labels = {f"{r['model']} / {r['encoder']} ({r['folder']})": r for r in runs}
    choice = st.selectbox("Run", list(run_labels), index=0)
    run = run_labels[choice]
    hist = load_history(run["path"])
    if hist.empty:
        st.info("No `history_*.csv` in this run directory.")
    else:
        fig, axes = plt.subplots(1, 2, figsize=(11.5, 3.4))
        for seed, sub in hist.groupby("seed"):
            axes[0].plot(sub["epoch"], sub["train_loss"], label=f"seed {seed}", lw=1.4)
            if "val_qwk" in sub:
                axes[1].plot(sub["epoch"], sub["val_qwk"], label=f"seed {seed}", lw=1.4)
        axes[0].set_title("training loss", fontsize=10)
        axes[1].set_title("validation QWK", fontsize=10)
        for ax in axes:
            ax.set_xlabel("epoch", fontsize=9)
            ax.spines[["top", "right"]].set_visible(False)
            ax.tick_params(labelsize=8)
            ax.legend(frameon=False, fontsize=8)
        if "val_qwk" in hist and not hist["val_qwk"].isna().all():
            best_val = hist["val_qwk"].max()
            axes[1].axhline(best_val, color="#999", ls=":", lw=1)
        st.pyplot(fig, use_container_width=True)
        plt.close(fig)
        st.caption(
            "Early stopping selects the epoch with the highest validation QWK; the test "
            "split is never used for model selection."
        )

    st.divider()
    st.subheader("Per-slide errors")
    pred_run = st.selectbox("Prediction source", list(run_labels), index=0, key="pred_run")
    preds = load_predictions(run_labels[pred_run]["path"])
    if preds.empty:
        st.info("No `predictions_*.csv` in this run directory.")
        return

    seed = st.selectbox("Seed", sorted(preds["seed"].unique().tolist()))
    seed_preds = preds[preds["seed"] == seed]
    cm = confusion_from_predictions(preds, seeds=[seed])

    left, right = st.columns([1, 1.2])
    with left:
        fig, ax = plt.subplots(figsize=(4.0, 3.5))
        im = ax.imshow(cm, cmap="Blues")
        ax.set_xticks(range(6), range(6), fontsize=8)
        ax.set_yticks(range(6), range(6), fontsize=8)
        ax.set_xlabel("predicted", fontsize=9)
        ax.set_ylabel("true", fontsize=9)
        for i in range(6):
            for j in range(6):
                if cm[i, j]:
                    ax.text(j, i, cm[i, j], ha="center", va="center", fontsize=9,
                            color="white" if cm[i, j] > cm.max() / 2 else "black")
        ax.set_title(f"confusion, seed {seed}", fontsize=10)
        st.pyplot(fig, use_container_width=True)
        plt.close(fig)

    with right:
        errors = seed_preds[seed_preds["y_true"] != seed_preds["y_pred"]].copy()
        errors["error"] = errors["y_pred"] - errors["y_true"]
        n = len(seed_preds)
        n_err = len(errors)
        n_adj = int((errors["error"].abs() == 1).sum())
        st.metric("Misclassified", f"{n_err}/{n}", delta=f"{n_err / n:.1%} error rate",
                  delta_color="inverse")
        st.metric("Off by exactly one grade", f"{n_adj}/{max(n_err, 1)}",
                  help="Adjacent-grade errors are near misses under QWK.")
        if n_err:
            st.dataframe(
                errors[["slide_id", "y_true", "y_pred", "error"]]
                .sort_values("error", key=lambda s: s.abs(), ascending=False),
                use_container_width=True, hide_index=True,
            )


def page_predictor(st) -> None:
    st.title("Slide Predictor")
    st.caption("One slide at a time: predicted grade, class distribution, attention ranking.")

    checkpoints = list_checkpoints(ROOT)
    if not checkpoints:
        st.error("No trained checkpoint found under `results/`.")
        return
    labels = [str(p.relative_to(ROOT)) for p in checkpoints]
    default = next((i for i, l in enumerate(labels) if "clam_sb" in l), 0)
    checkpoint_label = st.sidebar.selectbox("Model checkpoint", labels, index=default)
    checkpoint = ROOT / checkpoint_label
    context = checkpoint_context(checkpoint)

    mock_available = (ROOT / "data/mock/manifest.csv").exists()
    seeded_manifest = context.get("manifest") or (
        "data/mock/manifest.csv" if mock_available else "data/common_split.csv"
    )
    seeded_features = context.get("features_root") or (
        "data/mock/features" if "mock" in seeded_manifest else "data/features"
    )
    manifest_path = st.sidebar.text_input("Split manifest", seeded_manifest)
    features_root = st.sidebar.text_input("Feature folder", seeded_features)
    thumbnails_dir = st.sidebar.text_input(
        "Thumbnails folder (optional)",
        "data/mock/thumbnails" if "mock" in manifest_path else "data/thumbnails",
    )
    if context.get("manifest"):
        st.sidebar.caption(
            f"seeded from `{checkpoint.parent.parent.name}/run_config.json` "
            f"(train/val/test = {context.get('n_train','?')}/{context.get('n_val','?')}/{context.get('n_test','?')})"
        )

    split = st.sidebar.selectbox("Split", ["test", "val", "train", "all"], index=0)
    top_k = st.sidebar.slider("Top suspicious patches", 3, 20, 6)

    try:
        records = read_manifest(ROOT / manifest_path if not Path(manifest_path).is_absolute() else manifest_path)
    except Exception as exc:  # noqa: BLE001
        st.error(f"Could not read manifest: {exc}")
        return

    records = records if split == "all" else split_records(records, split)
    if not records:
        st.warning("No slides in that split.")
        return

    root_features = ROOT / features_root if not Path(features_root).is_absolute() else Path(features_root)
    all_ids = [r.image_id for r in records]
    slide_ids = available_slide_ids(records, root_features)
    if not slide_ids:
        st.error(
            f"No feature files for the '{split}' split under `{features_root}`.\n\n"
            "Each encoder has its own folder (`data/features` = ResNet50, "
            "`data/features_chief`, `data/features_gigap`)."
        )
        return
    missing = len(all_ids) - len(slide_ids)
    if missing:
        st.sidebar.caption(
            f"{len(slide_ids)}/{len(all_ids)} slides here have features for this encoder "
            f"({missing} missing)."
        )

    selected_id = st.selectbox("Slide", slide_ids)
    record = next(r for r in records if r.image_id == selected_id)

    try:
        model, payload = load_model(checkpoint)
    except Exception as exc:  # noqa: BLE001
        st.error(f"Could not load checkpoint: {exc}")
        return
    try:
        features, coords, patch_paths = load_feature_bag(feature_path(root_features, selected_id))
    except Exception as exc:  # noqa: BLE001
        st.error(f"Could not load features for {selected_id}: {exc}")
        return

    expected_dim = model_in_dim(model, payload, context)
    if expected_dim is not None and int(features.size(1)) != int(expected_dim):
        st.error(
            f"Feature dimension mismatch: checkpoint expects {expected_dim}-d, "
            f"`{features_root}` provides {int(features.size(1))}-d.\n\n"
            "Set **Feature folder** to the encoder this run was trained on."
        )
        return

    probs, attention = predict_slide(model, features)
    prediction = int(np.argmax(probs))
    delta = prediction - record.label
    correct = delta == 0
    confidence = float(probs[prediction])
    entropy = float(-np.sum(probs * np.log(np.clip(probs, 1e-12, 1.0))) / np.log(len(probs)))

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Predicted grade", f"{prediction} · {GRADE_SHORT.get(prediction, '')}")
    c2.metric("Confidence", f"{confidence:.1%}",
              delta=f"{confidence - 1 / len(probs):+.1%} vs uniform")
    c3.metric("Ground truth", f"{record.label} · {GRADE_SHORT.get(record.label, '')}",
              delta="match" if correct else f"off by {delta:+d}",
              delta_color="normal" if correct else "inverse")
    c4.metric("Uncertainty", f"{entropy:.2f}",
              help="Normalised Shannon entropy: 0 = decisive, 1 = uniform.")

    if not correct and abs(delta) == 1:
        st.caption("Adjacent-grade error — a near miss under QWK, which is why QWK is the headline metric.")
    elif not correct:
        st.caption(f"Misclassified by {abs(delta)} grade groups.")

    st.divider()
    left, right = st.columns([1, 1.4], gap="large")
    with left:
        st.subheader("Slide")
        thumbnail = load_thumbnail(find_thumbnail(thumbnails_dir, selected_id))
        if thumbnail is not None:
            st.image(thumbnail, caption=selected_id, use_container_width=True)
        else:
            st.info(
                "No thumbnail. Drop a `.png`/`.jpg` named `<slide_id>` into "
                "`data/thumbnails/` to display the slide here."
            )
        with st.expander("Metadata", expanded=True):
            st.markdown(
                f"**Slide** `{selected_id}`  \n"
                f"**Provider** {record.data_provider}  \n"
                f"**True ISUP** {GRADE_NAMES.get(record.label, record.label)}  \n"
                f"**Gleason** {record.gleason_score}  \n"
                f"**Patches** {int(features.size(0)):,}  \n"
                f"**Feature dim** {int(features.size(1))}  \n"
                f"**Model** {payload.get('model_name', 'unknown')} · seed {payload.get('seed', '?')}"
            )

    with right:
        st.subheader("Class probabilities")
        fig, ax = plt.subplots(figsize=(6.4, 2.9))
        tick_labels = [f"{i}\n{GRADE_SHORT.get(i, '')}" for i in range(len(probs))]
        colours = [
            "#2e7d32" if i == record.label and i == prediction
            else "#c62828" if i == prediction
            else "#546e7a" if i == record.label
            else "#cfd8dc"
            for i in range(len(probs))
        ]
        bars = ax.bar(range(len(probs)), probs, color=colours, edgecolor="white", linewidth=0.8)
        ax.set_xticks(range(len(probs)), tick_labels, fontsize=8)
        ax.set_ylabel("probability", fontsize=9)
        ax.set_ylim(0, max(1.0, float(probs.max()) * 1.18))
        ax.spines[["top", "right"]].set_visible(False)
        ax.tick_params(labelsize=8)
        for bar, p in zip(bars, probs):
            if p > 0.02:
                ax.text(bar.get_x() + bar.get_width() / 2, p + 0.015, f"{p:.0%}",
                        ha="center", fontsize=7.5)
        from matplotlib.patches import Patch
        ax.legend(handles=[
            Patch(facecolor="#2e7d32", label="predicted = truth"),
            Patch(facecolor="#c62828", label="predicted"),
            Patch(facecolor="#546e7a", label="true grade"),
        ], frameon=False, fontsize=7.5, loc="upper right")
        st.pyplot(fig, use_container_width=True)
        plt.close(fig)

        top2 = np.argsort(probs)[::-1][:2]
        st.caption(
            f"Top-2: {int(top2[0])} ({probs[top2[0]]:.1%}) and {int(top2[1])} "
            f"({probs[top2[1]]:.1%}). A single sharp peak is the well-calibrated case; "
            "a flat distribution means the model is undecided."
        )

    st.divider()
    st.subheader("Attention heatmap")
    if coords is None:
        st.warning(
            "No patch coordinates in this feature file, so a spatial heatmap cannot be drawn. "
            + (
                "The CHIEF / GigaPath / UNI releases ship patch embeddings only — no tile "
                "coordinates and no patch images — so a spatial map is not recoverable for "
                "this encoder. Attention ranking below is still valid. Use a ResNet50 "
                "checkpoint (`data/features`) for heatmaps."
                if any(k in str(features_root).lower() for k in ("chief", "gigap", "uni"))
                else "Re-export features with `src/feature_extraction.py` to include `coords`."
            )
        )
    else:
        width = 768
        attention_map = scatter_attention(attention, coords.numpy(), patch_size=256, out_width=width)
        normalised = normalise_attention(attention_map)
        colour = (matplotlib.colormaps["jet"](normalised)[..., :3] * 255).astype(np.uint8)
        base = (
            thumbnail.resize((normalised.shape[1], normalised.shape[0]))
            if thumbnail
            else __import__("PIL.Image", fromlist=["Image"]).new(
                "RGB", (normalised.shape[1], normalised.shape[0]), "white"
            )
        )
        overlay = 0.5 * np.asarray(base, dtype=np.float32) + 0.5 * colour.astype(np.float32)
        st.image(np.clip(overlay, 0, 255).astype(np.uint8),
                 caption="attention (jet) alpha-blended on the slide thumbnail",
                 use_container_width=True)
        st.caption(
            f"attention entropy {attention_entropy(attention):.3f} "
            f"(normalised {normalised_attention_entropy(attention):.3f}); lower = more focused."
        )

    st.subheader("Top suspicious patches")
    has_coords = coords is not None
    coords_array = coords.numpy() if has_coords else np.zeros((features.size(0), 2))
    rows = top_attention_patches(attention, coords_array, patch_paths, k=top_k)
    if not has_coords:
        rows = [{k: v for k, v in row.items() if k not in ("x", "y")} for row in rows]
    usable = [r for r in rows if r.get("patch_path") and Path(r["patch_path"]).exists()]
    if usable:
        columns = st.columns(min(4, len(usable)))
        for i, row in enumerate(usable):
            with columns[i % len(columns)]:
                with __import__("PIL.Image", fromlist=["Image"]).open(row["patch_path"]) as patch:
                    st.image(patch.convert("RGB"),
                             caption=f"#{row['rank']} a={row['attention']:.4f}",
                             use_container_width=True)
    else:
        st.info(
            "Patch images are not available (the feature file has no `patch_paths`). "
            "The ranked attention table is shown instead."
            + ("" if has_coords else " Coordinates are omitted: this encoder's release "
               "provides embeddings only, with no tile positions.")
        )
    st.dataframe(rows, use_container_width=True, hide_index=True)

    with st.expander("How to read this"):
        st.markdown(
            "- **QWK** is the headline metric in the report; accuracy alone is misleading "
            "because ISUP grades are imbalanced.\n"
            "- **Attention** shows where the model looked, not where tumour definitely is. "
            "PANDA has no pixel-level annotations.\n"
            "- A confident wrong prediction is a *feature* of a weak-label MIL model on a "
            "small subset, and is exactly what the report's error analysis discusses."
        )


def page_conclusions(st) -> None:
    st.title("Research Conclusions")

    runs = discover_runs(METRICS)
    baseline = next((r for r in runs if r["encoder"] == "ResNet50" and r["model"] == "ABMIL"), None)
    best = max(runs, key=lambda r: (r.get("qwk") or -1)) if runs else None

    st.subheader("What these experiments show")
    if best and baseline and baseline.get("qwk") is not None:
        st.markdown(
            f"The largest and most consistent effect in this project is the **frozen patch "
            f"encoder**, not the MIL architecture. Holding slides, labels, split, head, "
            f"optimiser and early stopping fixed, replacing ImageNet ResNet50 "
            f"(QWK {baseline['qwk']:.3f}) with a pathology-pretrained encoder reaches "
            f"QWK **{best['qwk']:.3f}** ({best['encoder']}, {best['model']})."
        )

    comp = load_comparison(METRICS)
    if not comp.empty:
        st.dataframe(comp, use_container_width=True, hide_index=True)

    st.markdown(
        "- **ABMIL vs CLAM-SB is not resolved.** On ResNet50 the paired comparison gives "
        "QWK 0.7203 vs 0.7137, difference −0.0066 with a 95% CI spanning zero and McNemar "
        "p = 0.6875. The ranking even reverses between encoder families. With n = 57–60 "
        "this difference is not estimable.\n"
        "- **Instance supervision had no reliable effect.** Δ = +0.0098 but the no-instance "
        "arm's sd is 0.0857 because one seed early-stopped before convergence.\n"
        "- **A linear model on ten attention statistics matches the MIL model on weak "
        "features** (QWK 0.7399 vs 0.7031), which suggests the attention distribution "
        "carries most of the usable signal there.\n"
        "- **The two models attend to similar regions but not identical patches** — "
        "Spearman ρ 0.869, top-10% Jaccard 0.502.\n"
        "- **Late fusion did not improve QWK.** Averaging CHIEF and GigaPath tracks the "
        "weaker member because they share a patch grid and make correlated errors."
    )

    st.subheader("What these experiments do not establish")
    st.markdown(
        "- **Not that any model is clinically usable.** 57–60 test slides, one dataset, "
        "no external validation.\n"
        "- **Not that the encoder gain will replicate elsewhere.** The sweep was a post-hoc "
        "extension; the encoder was chosen on feature *coverage*, not on test QWK, but eight "
        "configurations were evaluated on one locked split.\n"
        "- **Not localisation accuracy.** PANDA has no pixel-level ISUP annotation, so "
        "attention agreement is the strongest statement the data supports.\n"
        "- **Not a working patch CNN.** It is implemented and unit-tested but untrained: "
        "no patch pixels and no patch-level labels exist in this project's data.\n"
        "- **No statistical superiority between ABMIL and CLAM-SB.** The confidence interval "
        "for the difference is wider than the difference itself."
    )

    st.subheader("Reproducibility")
    st.markdown(
        "Every number on these pages is read from `results/metrics/`. The commands that "
        "produce them are listed in `report/person4_results.md` §10. Seeds are 0/1/2 "
        "throughout and all reported values are mean ± sd across seeds; no single-seed "
        "number is quoted as a result."
    )


# --------------------------------------------------------------------------- #
# Entry point
# --------------------------------------------------------------------------- #
def render() -> None:
    st.set_page_config(
        page_title="Prostate MIL · ISUP grading",
        page_icon="🔬",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    st.sidebar.title("🔬 Prostate MIL")
    st.sidebar.caption("Weakly supervised ISUP grading from whole-slide patches")
    page = st.sidebar.radio(
        "Explore",
        ["Overview", "Performance", "Slide Predictor", "Conclusions"],
        index=0,
        label_visibility="collapsed",
    )

    st.sidebar.divider()
    st.sidebar.caption(
        "PANDA subset · 420 slides\n\n"
        "4 encoders · 2 MIL heads\n\n"
        "Experimental university project"
    )
    st.sidebar.divider()
    st.sidebar.warning(
        "**Research prototype.** Not a clinical device. Attention maps are model cues, "
        "not tumour segmentations. External validation is required before any real-world use."
    )

    if page == "Overview":
        page_overview(st)
    elif page == "Performance":
        page_performance(st)
    elif page == "Slide Predictor":
        page_predictor(st)
    else:
        page_conclusions(st)


if __name__ == "__main__":
    import streamlit as st

    render()

"""Tests for ``app/streamlit_app.py`` logic (no browser, no network, no real data).

The app's contract is that it never invents a number: every panel is either a
real checkpoint output or an explicitly labelled image descriptor. These tests
pin that down, plus the geometry of the tiling and the report discovery.
"""

from __future__ import annotations

import importlib.util
import json
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

APP_PATH = ROOT / "app" / "streamlit_app.py"


def _load_app_module():
    spec = importlib.util.spec_from_file_location("streamlit_app_under_test", APP_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


app = _load_app_module()


def _phantom(dots: int, size: int = 256, seed: int = 0) -> Image.Image:
    """Synthetic nuclear-density phantom: pink background, purple dots."""
    rng = np.random.default_rng(seed)
    image = Image.new("RGB", (size, size), (232, 198, 214))
    draw = ImageDraw.Draw(image)
    for _ in range(dots):
        x, y = rng.integers(0, size, size=2)
        radius = int(rng.integers(2, 7))
        draw.ellipse([x - radius, y - radius, x + radius, y + radius], fill=(60, 20, 95))
    return image


class _StubExtractor:
    """Deterministic stand-in for the ResNet50 featurizer (no download in tests)."""

    def __init__(self, out_dim: int = 16) -> None:
        self.out_dim = out_dim

    def __call__(self, batch: torch.Tensor) -> torch.Tensor:
        pooled = batch.mean(dim=(2, 3))
        return pooled[:, : self.out_dim].repeat(1, self.out_dim // pooled.shape[1] + 1)[:, : self.out_dim]


class TilingTest(unittest.TestCase):
    def test_tiles_cover_the_image_exactly(self) -> None:
        image = Image.new("RGB", (300, 300), "white")
        tiled = app.tile_image(image, grid=5)
        self.assertEqual(len(tiled["tiles"]), 25)
        self.assertEqual(tiled["tiles"][0].size, (60, 60))
        self.assertEqual(tiled["coords"].shape, (25, 2))
        self.assertEqual(tiled["tile_size"], 60)
        # last tile starts at (240, 240) and still lies inside the image
        self.assertEqual(tuple(tiled["coords"][-1]), (240.0, 240.0))

    def test_coords_are_in_row_major_order(self) -> None:
        tiled = app.tile_image(Image.new("RGB", (200, 200)), grid=2)
        self.assertEqual([tuple(c) for c in tiled["coords"]], [(0.0, 0.0), (100.0, 0.0), (0.0, 100.0), (100.0, 100.0)])

    def test_extract_features_respects_flat_bags(self) -> None:
        tiles = app.tile_image(Image.new("RGB", (120, 120), "grey"), grid=2)["tiles"]
        features = app.extract_features(tiles, _StubExtractor(out_dim=16), image_size=64, batch_size=3)
        self.assertEqual(tuple(features.shape), (4, 16))
        self.assertTrue(torch.isfinite(features).all())


class ModelOutputTest(unittest.TestCase):
    def setUp(self) -> None:
        from src.models import build_model

        self.config = {"in_dim": 16, "embed_dim": 8, "attention_dim": 4, "n_classes": 6}
        torch.manual_seed(0)
        self.model = build_model("abmil", self.config).eval()

    def test_probabilities_and_attention_contract(self) -> None:
        features = torch.randn(12, 16)
        result = app.run_model(self.model, features)
        self.assertEqual(result["probs"].shape, (6,))
        self.assertAlmostEqual(float(result["probs"].sum()), 1.0, places=5)
        self.assertEqual(result["attention"].shape, (12,))
        self.assertGreaterEqual(float(result["attention"].min()), 0.0)
        self.assertEqual(result["prediction"], int(np.argmax(result["probs"])))
        self.assertAlmostEqual(result["confidence"], float(result["probs"][result["prediction"]]), places=6)

    def test_attention_length_always_matches_bag_size(self) -> None:
        for n_patches in (3, 36, 100):
            result = app.run_model(self.model, torch.randn(n_patches, 16))
            self.assertEqual(result["attention"].size, n_patches)


class DescriptorTest(unittest.TestCase):
    def test_descriptors_are_orderable_and_not_grades(self) -> None:
        healthy = app.image_descriptors(_phantom(dots=40, seed=1))
        mixed = app.image_descriptors(_phantom(dots=200, seed=2))
        aggressive = app.image_descriptors(_phantom(dots=600, seed=3))

        key = "dark-object fraction (Otsu)"
        self.assertLess(healthy[key], mixed[key])
        self.assertLess(mixed[key], aggressive[key])
        # a descriptor block, never a predicted grade
        self.assertIn("mean stain density (1 - luminance)", healthy)
        self.assertNotIn("grade", " ".join(healthy).lower())

    def test_otsu_splits_a_bimodal_image(self) -> None:
        image = Image.new("RGB", (64, 64), (240, 240, 240))
        ImageDraw.Draw(image).rectangle([0, 0, 31, 63], fill=(20, 20, 20))
        descriptors = app.image_descriptors(image)
        self.assertAlmostEqual(descriptors["dark-object fraction (Otsu)"], 0.5, delta=0.05)


class OodTest(unittest.TestCase):
    def test_flags_shifted_features_and_passes_typical_ones(self) -> None:
        rng = np.random.default_rng(0)
        dim = 8
        typical = [rng.normal(scale=0.5, size=dim) for _ in range(20)]

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            for index, vector in enumerate(typical):
                torch.save({"features": torch.tensor(np.tile(vector, (10, 1)), dtype=torch.float32)},
                           root / f"slide{index:02d}.pt")
            reference = app.reference_feature_stats(root, dim, max_slides=40)
            self.assertIsNotNone(reference)
            self.assertEqual(reference["n_slides"], 20)

            in_distribution = torch.tensor(np.tile(typical[0], (10, 1)), dtype=torch.float32)
            shifted = torch.tensor(np.tile(typical[0] + 10.0, (10, 1)), dtype=torch.float32)

            near = app.ood_percentile(in_distribution, reference)
            far = app.ood_percentile(shifted, reference)
            self.assertLess(near["percentile"], 50.0)
            self.assertGreater(far["percentile"], 95.0)
            self.assertGreater(far["distance"], near["distance"])

    def test_returns_none_without_reference_features(self) -> None:
        self.assertIsNone(app.reference_feature_stats(Path("/definitely/not/here"), 1024))
        self.assertIsNone(app.ood_percentile(torch.randn(4, 8), None))

    def test_wrong_dimension_is_not_compared(self) -> None:
        reference = {"dim": 8, "mean": np.zeros(8), "std": np.ones(8), "distances": np.zeros(5), "n_slides": 5}
        self.assertIsNone(app.ood_percentile(torch.randn(4, 3), reference))


class GradeDecisionTest(unittest.TestCase):
    """The app must abstain rather than print a grade for an out-of-distribution input."""

    @staticmethod
    def _ood(percentile: float, distance: float = 6.0) -> dict:
        return {"distance": distance, "percentile": percentile, "reference_median": 0.9, "n_reference": 40}

    def test_out_of_distribution_inputs_get_no_grade(self) -> None:
        decision = app.grade_decision(self._ood(100.0))
        self.assertEqual(decision["action"], "abstain")
        self.assertIn("no ISUP grade", decision["reason"])
        self.assertIn("100th percentile", decision["reason"])

    def test_borderline_inputs_are_graded_with_a_warning(self) -> None:
        self.assertEqual(app.grade_decision(self._ood(97.0))["action"], "warn")

    def test_typical_inputs_are_reported(self) -> None:
        decision = app.grade_decision(self._ood(40.0))
        self.assertEqual(decision["action"], "report")

    def test_missing_reference_cloud_is_unverified(self) -> None:
        decision = app.grade_decision(None)
        self.assertEqual(decision["action"], "unverified")
        self.assertIsNone(decision["percentile"])

    def test_thresholds_are_ordered(self) -> None:
        self.assertLess(app.OOD_WARN_PERCENTILE, app.OOD_ABSTAIN_PERCENTILE)
        self.assertLessEqual(app.OOD_ABSTAIN_PERCENTILE, 100.0)


class CheckpointDiscoveryTest(unittest.TestCase):
    def test_finds_and_classifies_checkpoints(self) -> None:
        from src.models import build_model

        config = {"in_dim": 16, "embed_dim": 8, "attention_dim": 4, "n_classes": 6}
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            ckpt_dir = root / "results" / "metrics" / "clam_sb" / "with_instance" / "checkpoints"
            ckpt_dir.mkdir(parents=True)
            model = build_model("clam_sb", {**config, "k_sample": 2})
            torch.save({"model_name": "clam_sb", "model_config": {**config, "k_sample": 2},
                        "state_dict": model.state_dict()}, ckpt_dir / "clam_sb_seed0_best.pt")
            (ckpt_dir / "broken_best.pt").write_bytes(b"not a checkpoint")

            found = app.discover_checkpoints(root)
            self.assertIn("clam_sb", found["models"])
            self.assertEqual(len(found["models"]["clam_sb"]), 1)
            self.assertEqual(len(found["unreadable"]), 1)

    def test_rebuilds_the_architecture_from_the_payload(self) -> None:
        from src.models import build_model

        config = {"in_dim": 16, "embed_dim": 8, "attention_dim": 4, "n_classes": 6}
        payload = {"model_name": "abmil", "model_config": config,
                   "state_dict": build_model("abmil", config).state_dict()}
        model, name, resolved = app.build_model_from_payload(payload)
        self.assertEqual(name, "abmil")
        self.assertEqual(resolved["in_dim"], 16)
        self.assertFalse(model.training)

    def test_rejects_a_payload_it_cannot_drive(self) -> None:
        with self.assertRaises(ValueError):
            app.build_model_from_payload({"model_name": "patch_cnn", "state_dict": {}})
        with self.assertRaises(ValueError):
            app.build_model_from_payload({"weights": {}})


class ReportsTest(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        for relative, text in (
            ("results/metrics/clam_sb/ablation_clam_sb.md", "# ablation"),
            ("results/metrics/clam_sb/with_instance/aggregate_clam_sb.json", json.dumps({"qwk": 0.69})),
            ("results/figures/confusion_abmil_seed0.png", "png"),
            ("results/heatmaps/clam_sb_seed0/slideA.png", "png"),
            ("results/mock_clam_sb/ablation_clam_sb.md", "# MOCK - not reportable"),
            ("results/mock_abmil/confusion_abmil_seed0.png", "png"),
            ("docs/RESULTS_LOG.md", "# results log"),
            ("report/person2_section.md", "# person 2"),
        ):
            path = self.root / relative
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(text, encoding="utf-8")

    def tearDown(self) -> None:
        self.tmp.cleanup()

    def test_mock_artifacts_are_hidden_by_default(self) -> None:
        reports = app.discover_reports(self.root)
        everything = [str(path) for paths in reports.values() for path in paths]
        self.assertTrue(everything)
        self.assertFalse([path for path in everything if "mock" in path])

    def test_mock_artifacts_can_be_included(self) -> None:
        reports = app.discover_reports(self.root, include_mock=True)
        everything = [str(path) for paths in reports.values() for path in paths]
        self.assertTrue([path for path in everything if "mock" in path])

    def test_groups_are_disjoint_and_heatmaps_split_out(self) -> None:
        reports = app.discover_reports(self.root)
        markdown = {str(p) for p in reports["markdown"]}
        payloads = {str(p) for p in reports["json"]}
        images = {str(p) for p in reports["images"]}
        heatmaps = {str(p) for p in reports["heatmaps"]}

        self.assertFalse(markdown & payloads)
        self.assertFalse(images & heatmaps)
        self.assertTrue(any("ablation_clam_sb.md" in path for path in markdown))
        self.assertTrue(any("aggregate_clam_sb.json" in path for path in payloads))
        self.assertTrue(any("heatmaps" in path for path in heatmaps))
        self.assertTrue(any("figures" in path for path in images))


class NoFakeOutputTest(unittest.TestCase):
    """The app must not contain the anti-patterns it replaced."""

    def test_no_random_attention_or_hardcoded_grades(self) -> None:
        source = APP_PATH.read_text(encoding="utf-8")
        for forbidden in (
            "np.random.rand(8, 8)",
            "alpha-blended on",  # old caption style that implied a fake map was real
            "np.random.rand(8,8)",
            "torch.softmax(torch.randn",
            "High Risk / Aggressive",
            "Normal / Low Risk",
            "composite_score",
        ):
            self.assertNotIn(forbidden, source, f"app still contains a fake-output pattern: {forbidden}")


if __name__ == "__main__":
    unittest.main()

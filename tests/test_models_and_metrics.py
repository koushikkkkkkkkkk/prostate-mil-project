"""Unit tests for the MIL models, metrics and data contract.

Run with::

    MPLBACKEND=Agg python -m unittest discover -s tests -v
"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.dataset import collate_bags  # noqa: E402
from src.evaluate import (  # noqa: E402
    attention_mask_overlap,
    bootstrap_metric_ci,
    compute_metrics,
    mcnemar_test,
    paired_bootstrap_difference,
    quadratic_weighted_kappa,
)
from src.heatmaps import (  # noqa: E402
    attention_entropy,
    normalised_attention_entropy,
    scatter_attention,
    top_attention_patches,
)
from src.models import build_model  # noqa: E402
from src.models.clam_sb import CLAMSB, CLAMSBConfig, SmoothTop1SVMLoss  # noqa: E402
from src.statistical_models import OrdinalLogisticRegression, bag_statistics  # noqa: E402
from src.utils.mock_data import (  # noqa: E402
    collate_mock,
    generate_mock_bag,
    generate_mock_batch,
)

N_CLASSES = 6


class TestMockData(unittest.TestCase):
    def test_bag_shapes(self):
        features, label = generate_mock_bag(num_patches=64, feature_dim=128)
        self.assertEqual(tuple(features.shape), (64, 128))
        self.assertIn(label, range(N_CLASSES))

    def test_batch_contract_from_plan(self):
        bags, labels = generate_mock_batch(batch_size=2, feature_dim=1024)
        self.assertEqual(len(bags), 2)
        self.assertEqual(tuple(labels.shape), (2,))

    def test_variable_length_collate(self):
        bags, _ = generate_mock_batch(batch_size=3, num_patches=40, feature_dim=32, seed=1)
        features, mask = collate_mock(bags)
        self.assertEqual(features.shape[0], 3)
        self.assertEqual(mask.shape[:2], features.shape[:2])
        self.assertTrue(bool(mask[:, 0].all()))


class TestModels(unittest.TestCase):
    def _bags(self, n=40, dim=32, batch=2, seed=0):
        generator = torch.Generator().manual_seed(seed)
        features = torch.randn(batch, n, dim, generator=generator)
        mask = torch.ones(batch, n, dtype=torch.bool)
        mask[1, -5:] = False
        labels = torch.tensor([2, 4])
        return features, mask, labels

    def test_abmil_shapes(self):
        model = build_model("abmil", {"in_dim": 32, "embed_dim": 16, "attention_dim": 8})
        features, mask, labels = self._bags()
        output = model(features, mask=mask)
        self.assertEqual(tuple(output["logits"].shape), (2, N_CLASSES))
        self.assertEqual(tuple(output["attention"].shape), (2, 40))
        # padded patches must receive zero attention
        self.assertAlmostEqual(float(output["attention"][1, -5:].sum()), 0.0, places=6)
        self.assertAlmostEqual(float(output["attention"].sum(dim=1)[0]), 1.0, places=5)

    def test_clam_sb_shapes_and_instance_loss(self):
        model = build_model("clam_sb", {"in_dim": 32, "embed_dim": 16, "attention_dim": 8, "k_sample": 4})
        features, mask, labels = self._bags()
        output = model(features, mask=mask, labels=labels)
        self.assertEqual(tuple(output["logits"].shape), (2, N_CLASSES))
        self.assertGreater(float(output["instance_loss"]), 0.0)
        self.assertEqual(output["top_indices"].shape, (2, 4))

    def test_clam_sb_without_instance_loss(self):
        model = build_model(
            "clam_sb",
            {"in_dim": 32, "embed_dim": 16, "attention_dim": 8, "instance_loss_weight": 0.0},
        )
        features, mask, labels = self._bags()
        output = model(features, mask=mask, labels=labels)
        self.assertEqual(float(output["instance_loss"]), 0.0)

    def test_single_bag_input_is_squeezed(self):
        model = build_model("clam_sb", {"in_dim": 32, "embed_dim": 16, "attention_dim": 8})
        output = model(torch.randn(30, 32))
        self.assertEqual(tuple(output["logits"].shape), (N_CLASSES,))
        self.assertEqual(tuple(output["attention"].shape), (30,))

    def test_ablation_keeps_architecture_identical(self):
        with_loss = CLAMSB(CLAMSBConfig(in_dim=32, embed_dim=16, attention_dim=8, instance_loss_weight=0.3))
        without = CLAMSB(CLAMSBConfig(in_dim=32, embed_dim=16, attention_dim=8, instance_loss_weight=0.0))
        self.assertEqual(
            sum(p.numel() for p in with_loss.parameters()),
            sum(p.numel() for p in without.parameters()),
        )

    def test_transmil_shapes_and_attention(self):
        model = build_model(
            "transmil", {"in_dim": 32, "d_model": 16, "n_heads": 4, "n_layers": 1, "ffn_multiplier": 2}
        )
        features, mask, _ = self._bags(n=37)
        output = model(features, mask=mask)
        self.assertEqual(tuple(output["logits"].shape), (2, N_CLASSES))
        self.assertEqual(tuple(output["attention"].shape), (2, 37))
        self.assertAlmostEqual(float(output["attention"].sum(dim=1)[1]), 1.0, places=4)

    def test_gradients_flow(self):
        model = build_model("clam_sb", {"in_dim": 32, "embed_dim": 16, "attention_dim": 8, "k_sample": 3})
        features, mask, labels = self._bags()
        output = model(features, mask=mask, labels=labels)
        loss = torch.nn.functional.cross_entropy(output["logits"], labels) + output["instance_loss"]
        loss.backward()
        grads = [p.grad for p in model.parameters() if p.grad is not None]
        self.assertTrue(grads)
        self.assertTrue(any(float(g.abs().sum()) > 0 for g in grads))


class TestLossesAndMetrics(unittest.TestCase):
    def test_smooth_top1_svm_prefers_correct_class(self):
        criterion = SmoothTop1SVMLoss(alpha=1.0, tau=1.0)
        targets = torch.tensor([1, 1, 0])
        good = torch.tensor([[0.0, 5.0], [0.0, 4.0], [4.0, 0.0]])
        bad = torch.tensor([[5.0, 0.0], [4.0, 0.0], [0.0, 4.0]])
        self.assertLess(float(criterion(good, targets)), float(criterion(bad, targets)))

    def test_qwk_bounds(self):
        y = [0, 1, 2, 3, 4, 5]
        self.assertAlmostEqual(quadratic_weighted_kappa(y, y), 1.0, places=6)
        self.assertLess(quadratic_weighted_kappa(y, [5, 4, 3, 2, 1, 0]), 0.0)

    def test_compute_metrics_bundle(self):
        y_true = [0, 1, 2, 3, 4, 5]
        y_pred = [0, 1, 2, 3, 4, 4]
        metrics = compute_metrics(y_true, y_pred)
        for key in ("qwk", "accuracy", "balanced_accuracy", "macro_f1", "confusion_matrix", "per_class"):
            self.assertIn(key, metrics)
        self.assertEqual(len(metrics["confusion_matrix"]), N_CLASSES)

    def test_bootstrap_and_paired_stats(self):
        y_true = [0, 1, 2, 3, 4, 5, 5, 4, 3, 2]
        y_a = [0, 1, 2, 3, 4, 5, 4, 4, 3, 2]
        y_b = [0, 1, 2, 3, 4, 5, 5, 5, 3, 2]
        ci = bootstrap_metric_ci(y_true, y_a, n_boot=50, seed=0)
        self.assertLessEqual(ci["lo"], ci["hi"])
        difference = paired_bootstrap_difference(y_true, y_a, y_b, n_boot=50, seed=0)
        self.assertLessEqual(difference["lo"], difference["hi"])
        mcnemar = mcnemar_test(y_true, y_a, y_b)
        self.assertTrue(0.0 <= mcnemar["p_value"] <= 1.0)

    def test_attention_mask_overlap(self):
        attention = [0.9, 0.8, 0.05, 0.05]
        mask = [1.0, 1.0, 0.0, 0.0]
        result = attention_mask_overlap(attention, mask, fractions=(0.5,))
        self.assertAlmostEqual(result["top50pct"]["precision"], 1.0)
        self.assertAlmostEqual(result["base_rate"], 0.5)


class TestHeatmaps(unittest.TestCase):
    def test_scatter_and_entropy(self):
        coords = [[0, 0], [256, 0], [0, 256], [256, 256]]
        attention = [0.7, 0.1, 0.1, 0.1]
        attention_map = scatter_attention(attention, coords, patch_size=256, out_width=64)
        self.assertEqual(attention_map.ndim, 2)
        self.assertGreater(attention_map[0, 0], attention_map[0, -1])

        peaked = attention_entropy([0.97, 0.01, 0.01, 0.01])
        flat = attention_entropy([0.25] * 4)
        self.assertLess(peaked, flat)
        self.assertLessEqual(normalised_attention_entropy([0.97, 0.01, 0.01, 0.01]), 1.0)

    def test_top_patches_sorted(self):
        rows = top_attention_patches([0.1, 0.5, 0.9], [[0, 0], [1, 1], [2, 2]], k=2)
        self.assertEqual(rows[0]["patch_index"], 2)
        self.assertEqual(rows[1]["patch_index"], 1)


class TestStatisticsAndData(unittest.TestCase):
    def test_ordinal_logistic_recovers_order(self):
        rng = np.random.default_rng(0)
        grades = np.repeat(np.arange(6), 30)
        order = rng.permutation(grades.size)          # every class present in train
        grades = grades[order]
        x = grades[:, None] + rng.normal(scale=0.45, size=(grades.size, 1))
        model = OrdinalLogisticRegression().fit(x[:150], grades[:150])
        prediction = model.predict(x[150:])
        accuracy = float((prediction == grades[150:]).mean())
        self.assertGreater(accuracy, 0.4)
        # predictions must be monotone non-decreasing in x (ordinal decoding)
        ordered = prediction[np.argsort(x[150:, 0])]
        self.assertTrue(np.all(np.diff(ordered) >= 0))

    def test_bag_statistics_keys(self):
        stats = bag_statistics(np.array([0.5, 0.3, 0.2]), np.random.randn(3, 8))
        self.assertIn("attention_entropy", stats)
        self.assertIn("embedding_norm_mean", stats)

    def test_collate_pads_and_masks(self):
        batch = [
            {"features": torch.randn(5, 4), "label": torch.tensor(1), "image_id": "a"},
            {"features": torch.randn(3, 4), "label": torch.tensor(2), "image_id": "b"},
        ]
        collated = collate_bags(batch)
        self.assertEqual(tuple(collated["features"].shape), (2, 5, 4))
        self.assertEqual(int(collated["mask"].sum()), 8)
        self.assertEqual(collated["image_id"], ["a", "b"])


if __name__ == "__main__":
    unittest.main(verbosity=2)

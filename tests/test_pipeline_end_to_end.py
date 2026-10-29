"""Integration test: synthetic data -> train -> evaluate -> heatmaps -> compare.

It exercises the exact command-line entry points used in
``scripts/run_mock_demo.sh`` with tiny sizes, so a broken import, argument or
file contract fails here instead of during the real run.

The metrics produced here are meaningless by construction (synthetic data);
only the *plumbing* is under test.
"""

from __future__ import annotations

import csv
import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ENV_MPL = {"MPLBACKEND": "Agg"}


def run(module: str, *args: str, cwd: Path = ROOT) -> subprocess.CompletedProcess:
    command = [sys.executable, "-m", module, *args]
    result = subprocess.run(
        command, cwd=cwd, capture_output=True, text=True, env={**os.environ, **ENV_MPL}
    )
    if result.returncode != 0:
        raise AssertionError(
            f"{' '.join(command)} failed\nSTDOUT:\n{result.stdout}\nSTDERR:\n{result.stderr}"
        )
    return result


class TestEndToEndMockPipeline(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.tmp = tempfile.TemporaryDirectory()
        cls.root = Path(cls.tmp.name)
        cls.mock = cls.root / "mock"
        subprocess.run(
            [
                sys.executable,
                "scripts/make_mock_features.py",
                "--out-dir", str(cls.mock),
                "--n-train", "3",
                "--n-val", "2",
                "--n-test", "2",
                "--feature-dim", "16",
                "--grid-cols", "4",
                "--grid-rows", "4",
                "--patch-size", "64",
                "--patch-px", "16",
                "--thumbnail-width", "96",
                "--seed", "3",
            ],
            cwd=ROOT,
            check=True,
            capture_output=True,
            text=True,
        )

    @classmethod
    def tearDownClass(cls) -> None:
        cls.tmp.cleanup()

    def test_full_chain(self):
        mock = self.mock
        manifest = str(mock / "manifest.csv")
        features = str(mock / "features")

        # 1. ABMIL baseline (needed for the paired comparison later)
        abmil_out = self.root / "abmil"
        run(
            "src.train",
            "--model", "abmil",
            "--manifest", manifest,
            "--features-root", features,
            "--out-dir", str(abmil_out),
            "--seeds", "0",
            "--epochs", "1",
            "--patience", "1",
            "--batch-size", "2",
            "--lr", "0.001",
            "--max-patches", "32",
            "--in-dim", "16",
            "--quiet",
        )
        self.assertTrue((abmil_out / "aggregate_abmil.json").exists())
        self.assertTrue((abmil_out / "predictions_abmil_seed0.csv").exists())

        # 2. CLAM-SB ablation (with vs without instance loss)
        clam_out = self.root / "clam"
        run(
            "src.train_clam_sb",
            "--manifest", manifest,
            "--features-root", features,
            "--out-dir", str(clam_out),
            "--seeds", "0",
            "--instance-loss-weight", "0.3",
            "--quiet",
            "--config", "configs/clam_sb_mock.yaml",
        )
        self.assertTrue((clam_out / "ablation_clam_sb.md").exists())
        with (clam_out / "ablation_clam_sb.json").open() as handle:
            ablation = json.load(handle)
        self.assertIn("with_instance", ablation)
        self.assertIn("without_instance", ablation)

        # 3. Evaluate the trained checkpoint (all splits -> attention table)
        checkpoint = clam_out / "with_instance" / "checkpoints" / "clam_sb_seed0_best.pt"
        self.assertTrue(checkpoint.exists())
        eval_out = self.root / "eval"
        run(
            "src.evaluate_clam_sb",
            "--checkpoint", str(checkpoint),
            "--manifest", manifest,
            "--features-root", features,
            "--out-dir", str(eval_out),
            "--split", "all",
        )
        attention_csv = eval_out / "attention_clam_sb_seed0.csv"
        self.assertTrue(attention_csv.exists())
        with attention_csv.open() as handle:
            rows = list(csv.DictReader(handle))
        self.assertEqual(len(rows), 7)  # 3 train + 2 val + 2 test

        # 4. Heatmaps + localisation overlap (thumbnails and masks exist in mock data)
        heatmap_out = self.root / "heatmaps"
        run(
            "src.export_clam_attention",
            "--checkpoint", str(checkpoint),
            "--manifest", manifest,
            "--features-root", features,
            "--thumbnails", str(mock / "thumbnails"),
            "--masks", str(mock / "masks"),
            "--out-dir", str(heatmap_out),
            "--split", "test",
        )
        images = sorted((heatmap_out / "images").glob("*_attention.png"))
        self.assertEqual(len(images), 2)
        self.assertTrue((heatmap_out / "localisation_scores.csv").exists())

        # 5. Statistical baseline
        run(
            "src.statistical_models",
            "--attention-csv", str(attention_csv),
            "--manifest", manifest,
            "--out-dir", str(self.root / "stat"),
        )

        # 6. Paired comparison
        compare_out = self.root / "compare"
        run(
            "src.compare_clam_abmil",
            "--abmil", str(abmil_out / "predictions_abmil_seed0.csv"),
            "--clam", str(clam_out / "with_instance" / "predictions_clam_sb_seed0.csv"),
            "--out-dir", str(compare_out),
            "--n-boot", "20",
        )
        comparison = json.loads((compare_out / "comparison.json").read_text())
        self.assertEqual(comparison["n_slides"], 2)
        self.assertIn("paired_qwk_difference", comparison)


if __name__ == "__main__":
    unittest.main(verbosity=2)

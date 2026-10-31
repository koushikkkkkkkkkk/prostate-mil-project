"""Regression test for ``scripts/summarize_person2.py``.

Builds a miniature `results/` tree that mimics exactly what
``src.train_clam_sb`` / ``src.train`` / ``src.evaluate_clam_sb`` /
``src.export_clam_attention`` write, then asserts the summary report picks the
numbers up instead of silently degrading to "not run".

Run with::

    MPLBACKEND=Agg python -m unittest tests.test_summarize_person2 -v
"""

from __future__ import annotations

import csv
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "summarize_person2.py"
N_SLIDES = 60


def _write_predictions(path: Path, y_true, y_pred) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["slide_id", "y_true", "y_pred"] + [f"p{c}" for c in range(6)])
        for index, (truth, prediction) in enumerate(zip(y_true, y_pred)):
            writer.writerow([f"slide{index:03d}", truth, prediction] + [0.1667] * 6)


def _aggregate(qwk_values, accuracy=0.40) -> dict:
    import numpy as np

    array = np.asarray(qwk_values, dtype=float)
    return {
        "n_seeds": int(array.size),
        "qwk": {"mean": float(array.mean()), "std": float(array.std(ddof=1)), "values": qwk_values},
        "accuracy": {"mean": accuracy, "std": 0.0, "values": [accuracy] * array.size},
        "balanced_accuracy": {"mean": accuracy, "std": 0.0, "values": [accuracy] * array.size},
        "macro_f1": {"mean": 0.33, "std": 0.0, "values": [0.33] * array.size},
    }


class SummarizePerson2Test(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.results = Path(self.tmp.name) / "results"
        self.y_true = [index // 10 for index in range(N_SLIDES)]
        # the with-instance arm is deliberately worse: one ordinal error every 4th
        # slide, versus one every 7th slide for the no-instance arm.
        self.y_with = [
            (truth + 1) if index % 4 == 0 else truth for index, truth in enumerate(self.y_true)
        ]
        self.y_without = [
            (truth + 1) if index % 7 == 0 else truth for index, truth in enumerate(self.y_true)
        ]

        clamsb = self.results / "metrics" / "clam_sb"
        _write_predictions(clamsb / "with_instance" / "predictions_clam_sb_seed0.csv", self.y_true, self.y_with)
        _write_predictions(clamsb / "without_instance" / "predictions_clam_sb_seed0.csv", self.y_true, self.y_without)
        _write_predictions(self.results / "metrics" / "abmil" / "predictions_abmil_seed0.csv", self.y_true, self.y_with)

        (clamsb / "with_instance").mkdir(parents=True, exist_ok=True)
        (clamsb / "with_instance" / "aggregate_clam_sb.json").write_text(
            json.dumps(_aggregate([0.60, 0.62, 0.64])), encoding="utf-8"
        )
        (clamsb / "without_instance").mkdir(parents=True, exist_ok=True)
        (clamsb / "without_instance" / "aggregate_clam_sb.json").write_text(
            json.dumps(_aggregate([0.70, 0.72, 0.71])), encoding="utf-8"
        )
        (self.results / "metrics" / "abmil" / "aggregate_abmil.json").write_text(
            json.dumps(_aggregate([0.71, 0.72, 0.73])), encoding="utf-8"
        )

        for arm_tag, entropy in (("eval_seed0", 4.10), ("eval_without_seed0", 4.50)):
            attention = clamsb / arm_tag / "attention_clam_sb_seed0.csv"
            attention.parent.mkdir(parents=True, exist_ok=True)
            with attention.open("w", newline="", encoding="utf-8") as handle:
                writer = csv.writer(handle)
                writer.writerow(["slide_id", "attention_entropy", "normalised_entropy", "top10_attention_share"])
                for index in range(N_SLIDES):
                    writer.writerow([f"slide{index:03d}", entropy, 0.95, 0.08])

        for name, slides in (("clam_sb_seed0", 56), ("clam_sb_seed0_noinst", 56)):
            summary = self.results / "heatmaps" / name / "export_summary.json"
            summary.parent.mkdir(parents=True, exist_ok=True)
            summary.write_text(
                json.dumps(
                    {
                        "out_dir": f"results/heatmaps/{name}",
                        "split": "test",
                        "n_slides": slides,
                        "n_skipped_no_coords": 4,
                        "mean_normalised_entropy": 0.95,
                    }
                ),
                encoding="utf-8",
            )

        statistics_dir = self.results / "metrics" / "statistical"
        statistics_dir.mkdir(parents=True, exist_ok=True)
        (statistics_dir / "statistical_baselines.json").write_text(
            json.dumps({"ordinal_logistic": {"qwk": 0.7682, "accuracy": 0.45, "macro_f1": 0.41}}),
            encoding="utf-8",
        )
        self.out = Path(self.tmp.name) / "PERSON2_SUMMARY.md"

    def tearDown(self) -> None:
        self.tmp.cleanup()

    def _run(self) -> str:
        completed = subprocess.run(
            [sys.executable, str(SCRIPT), "--results", str(self.results), "--out", str(self.out), "--n-boot", "200"],
            cwd=ROOT,
            capture_output=True,
            text=True,
        )
        self.assertEqual(completed.returncode, 0, completed.stderr)
        return self.out.read_text(encoding="utf-8")

    def test_reports_every_section(self) -> None:
        text = self._run()
        self.assertIn("| CLAM-SB + instance loss | 0.6200 |", text)
        self.assertIn("| CLAM-SB, no instance loss | 0.7100 |", text)
        self.assertIn("| ABMIL baseline (Person 1) | 0.7200 |", text)
        self.assertIn("Mean QWK delta (with − without) = **-0.0900**", text)
        self.assertIn("Mean QWK delta (CLAM-SB − ABMIL) = **-0.1000**", text)
        self.assertNotIn("not run", text)
        self.assertNotIn("_No paired prediction files found yet", text)
        self.assertNotIn("_No attention CSVs found yet", text)
        self.assertNotIn("_No heatmap exports found yet", text)
        self.assertNotIn("_Not run yet", text)

    def test_paired_section_uses_shared_slides(self) -> None:
        text = self._run()
        self.assertIn("| 0 | 60 |", text)
        self.assertIn("McNemar p", text)

    def test_with_arm_is_worse_in_paired_table(self) -> None:
        text = self._run()
        paired = text.split("### with instance loss vs no instance loss")[1].split("###")[0]
        for line in paired.splitlines():
            if line.startswith("| 0 |"):
                delta = float(line.split("|")[5].strip())
                self.assertLess(delta, 0.0)

    def test_alternate_eval_folder_name_is_detected(self) -> None:
        """`eval_noinst_seed0` must be classified as the no-instance arm, not silently dropped."""
        clamsb = self.results / "metrics" / "clam_sb"
        source = clamsb / "eval_without_seed0"
        target = clamsb / "eval_noinst_seed0"
        target.mkdir(parents=True, exist_ok=True)
        (target / "attention_clam_sb_seed0.csv").write_text(
            (source / "attention_clam_sb_seed0.csv").read_text(encoding="utf-8"), encoding="utf-8"
        )
        text = self._run()
        self.assertIn("CLAM-SB, no instance loss (0)", text)
        self.assertIn("CLAM-SB + instance loss (0)", text)

    def test_missing_results_dir_fails_loudly(self) -> None:
        completed = subprocess.run(
            [sys.executable, str(SCRIPT), "--results", str(Path(self.tmp.name) / "absent")],
            cwd=ROOT,
            capture_output=True,
            text=True,
        )
        self.assertEqual(completed.returncode, 1)
        self.assertIn("results folder not found", completed.stderr)

    def test_empty_results_dir_degrades_gracefully(self) -> None:
        empty = Path(self.tmp.name) / "empty" / "results"
        empty.mkdir(parents=True)
        completed = subprocess.run(
            [sys.executable, str(SCRIPT), "--results", str(empty), "--out", str(Path(self.tmp.name) / "empty.md")],
            cwd=ROOT,
            capture_output=True,
            text=True,
        )
        self.assertEqual(completed.returncode, 0, completed.stderr)
        self.assertIn("0/3", completed.stdout)
        self.assertIn("not run", (Path(self.tmp.name) / "empty.md").read_text(encoding="utf-8"))


if __name__ == "__main__":
    unittest.main()

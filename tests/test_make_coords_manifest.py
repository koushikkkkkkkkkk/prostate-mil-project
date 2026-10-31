"""Tests for ``scripts/make_coords_manifest.py``.

The exporter must never be handed a slide it cannot plot, and must never silently
drop a slide without saying so -- those are the two properties checked here.
"""

from __future__ import annotations

import csv
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

try:
    import h5py
except ImportError:
    h5py = None
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "make_coords_manifest.py"

FIELDS = ["image_id", "data_provider", "isup_grade", "gleason_score", "split"]


class MakeCoordsManifestTest(unittest.TestCase):
    def setUp(self) -> None:
        if h5py is None:
            self.skipTest("h5py is not installed")
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.features = self.root / "features"
        (self.features / "patches").mkdir(parents=True)

        self.rows = []
        for index in range(6):
            image_id = f"slide{index:03d}"
            self.rows.append(
                {
                    "image_id": image_id,
                    "data_provider": "panda",
                    "isup_grade": index % 6,
                    "gleason_score": "3+3",
                    "split": ("train", "val", "test")[index % 3],
                }
            )
            if index == 5:  # no file at all
                continue
            n_patches = 20 + index
            with h5py.File(self.features / "patches" / f"{image_id}.h5", "w") as handle:
                handle.create_dataset("features", data=np.zeros((n_patches, 1024), dtype="float32"))
                if index not in (1, 4):  # slides 1 and 4 have features but no coords
                    handle.create_dataset("coords", data=np.zeros((n_patches, 2), dtype="float32"))

        self.manifest = self.root / "split.csv"
        with self.manifest.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=FIELDS)
            writer.writeheader()
            writer.writerows(self.rows)

        self.out = self.root / "with_coords.csv"

    def tearDown(self) -> None:
        self.tmp.cleanup()

    def _run(self, *extra: str) -> subprocess.CompletedProcess:
        return subprocess.run(
            [
                sys.executable,
                str(SCRIPT),
                "--manifest",
                str(self.manifest),
                "--features-root",
                str(self.features),
                "--out",
                str(self.out),
                *extra,
            ],
            cwd=ROOT,
            capture_output=True,
            text=True,
        )

    def test_keeps_only_slides_with_coords(self) -> None:
        completed = self._run()
        self.assertEqual(completed.returncode, 0, completed.stderr)
        self.assertIn("kept 3/6 slides", completed.stdout)

        with self.out.open(newline="", encoding="utf-8") as handle:
            kept = [row["image_id"] for row in csv.DictReader(handle)]
        self.assertEqual(kept, ["slide000", "slide002", "slide003"])

    def test_preserves_columns_and_order(self) -> None:
        self._run()
        with self.out.open(newline="", encoding="utf-8") as handle:
            reader = csv.DictReader(handle)
            self.assertEqual(reader.fieldnames, FIELDS)
            rows = list(reader)
        self.assertEqual(rows[0]["image_id"], "slide000")
        self.assertEqual(rows[0]["split"], "train")

    def test_reports_the_dropped_slides(self) -> None:
        completed = self._run()
        self.assertIn("2 no coords", completed.stdout)
        self.assertIn("1 missing file", completed.stdout)
        self.assertIn("slide001", completed.stdout)
        self.assertIn("slide004", completed.stdout)

    def test_missing_manifest_fails_loudly(self) -> None:
        self.manifest.unlink()
        completed = self._run()
        self.assertEqual(completed.returncode, 1)
        self.assertIn("manifest not found", completed.stderr)
        self.assertFalse(self.out.exists())

    def test_no_coords_anywhere_fails_loudly(self) -> None:
        for path in (self.features / "patches").glob("*.h5"):
            with h5py.File(path, "w") as handle:
                handle.create_dataset("features", data=np.zeros((5, 1024), dtype="float32"))
        completed = self._run()
        self.assertEqual(completed.returncode, 1)
        self.assertIn("nothing written", completed.stderr)


if __name__ == "__main__":
    unittest.main()

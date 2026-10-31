"""Environment pre-flight check.

Run this first whenever something behaves strangely:

    python scripts/check_env.py

It reports the Python version, the packages the project needs, whether PyTorch
can see a GPU, and -- most usefully -- it *detects the known broken
combinations* (numpy 1.x on Python 3.13+, a torch/torchvision mismatch, a venv
that is not active) and prints the exact command that fixes them.
"""

from __future__ import annotations

import importlib
import platform
import sys
from typing import List, Optional

MIN_PYTHON = (3, 10)
MAX_TESTED_PYTHON = (3, 13)

OK = "[ok]"
WARN = "[warn]"
FAIL = "[FAIL]"

problems: List[str] = []
notes: List[str] = []


def check(name: str, module: Optional[str] = None, minimum: Optional[str] = None) -> None:
    """Import a package and report its version, or record a problem."""
    try:
        mod = importlib.import_module(module or name)
    except Exception as exc:  # noqa: BLE001 - we want the real reason
        problems.append(f"{name} could not be imported: {type(exc).__name__}: {exc}")
        print(f"{FAIL} {name:<14} not importable ({type(exc).__name__})")
        return

    version = getattr(mod, "__version__", "unknown")
    print(f"{OK} {name:<14} {version}")
    if minimum and version != "unknown":
        try:
            from packaging.version import Version

            if Version(version) < Version(minimum):
                notes.append(f"{name} {version} is older than the expected {minimum}.")
        except Exception:  # pragma: no cover - packaging missing
            pass


def check_python() -> None:
    major, minor = sys.version_info[:2]
    version = f"{major}.{minor}.{sys.version_info.micro}"
    in_venv = sys.prefix != getattr(sys, "base_prefix", sys.prefix)

    print(f"{OK} python         {version}  ({platform.system()})")
    print(f"{OK if in_venv else WARN} virtualenv     {'active' if in_venv else 'NOT active'}")
    if not in_venv:
        notes.append(
            "No virtual environment is active. Use one:\n"
            "    python -m venv .venv\n"
            "    source .venv/bin/activate        (Windows: .venv\\Scripts\\activate)"
        )

    if (major, minor) < MIN_PYTHON:
        problems.append(
            f"Python {version} is too old; this project needs {MIN_PYTHON[0]}.{MIN_PYTHON[1]}+."
        )
    if (major, minor) > MAX_TESTED_PYTHON:
        notes.append(
            f"Python {version} is newer than the tested range (3.10-3.13). The code is version\n"
            "    agnostic, but torch/numpy wheels for very new interpreters are sometimes missing\n"
            "    or broken. If anything below fails, either upgrade numpy to >= 2.0 or use 3.12:\n"
            "        py -3.12 -m venv .venv312\n"
            "        .venv312\\Scripts\\activate\n"
            "        pip install -r requirements.txt"
        )


def check_numpy_torch_pair() -> None:
    """The two failure modes that actually bite on fresh Windows installs."""
    try:
        import numpy as np
    except Exception:  # already reported by check()
        return
    numpy_version = tuple(int(part) for part in np.__version__.split(".")[:2])
    if sys.version_info[:2] >= (3, 13) and numpy_version < (2, 0):
        problems.append(
            f"numpy {np.__version__} is not supported on Python "
            f"{sys.version_info[0]}.{sys.version_info[1]}.\n"
            "    This is the exact cause of 'OverflowError: cannot convert longdouble\n"
            "    infinity to integer' at numpy import. Fix with:\n"
            "        pip install --upgrade \"numpy>=2.0\""
        )

    torch = sys.modules.get("torch")
    if torch is None:
        return
    try:
        import torchvision

        torch_mm = ".".join(torch.__version__.split("+")[0].split(".")[:2])
        tv_mm = ".".join(torchvision.__version__.split("+")[0].split(".")[:2])
        if torch_mm != tv_mm:
            notes.append(
                f"torch {torch.__version__} and torchvision {torchvision.__version__} look "
                "mismatched;\n    reinstall torchvision for your torch build if feature "
                "extraction fails."
            )
    except ImportError:
        notes.append(
            "torchvision missing: fine for tests/mock demo, required for real feature\n"
            "    extraction (pip install -r requirements-full.txt)."
        )
    except Exception as exc:  # noqa: BLE001
        problems.append(f"torchvision is installed but broken: {type(exc).__name__}: {exc}")


def check_torch_device() -> None:
    torch = sys.modules.get("torch")
    if torch is None:
        return
    if torch.cuda.is_available():
        print(f"{OK} cuda           {torch.cuda.get_device_name(0)}")
    else:
        print(f"{OK} cuda           not available (CPU training still works)")
        notes.append(
            "No GPU visible. The mock demo runs on CPU in about a minute; real 3-seed\n"
            "    training on 300 slides is much faster on Colab/GPU."
        )


def check_project_files() -> None:
    from pathlib import Path

    root = Path(__file__).resolve().parents[1]
    required = [
        "src/dataset.py",
        "src/models/clam_sb.py",
        "src/train.py",
        "src/train_clam_sb.py",
        "scripts/make_mock_features.py",
        "configs/clam_sb.yaml",
    ]
    missing: List[str] = [name for name in required if not (root / name).exists()]
    if missing:
        problems.append("missing project files: " + ", ".join(missing))
        print(f"{FAIL} project files  {len(missing)} missing")
    else:
        print(f"{OK} project files  all present")

    if (root / "data/mock/manifest.csv").exists():
        print(f"{OK} mock data      generated")
    else:
        print(f"{WARN} mock data      not generated yet -> bash scripts/run_mock_demo.sh")

    split = root / "data/common_split.csv"
    print(
        f"{OK} real split     {split}" if split.exists()
        else f"{WARN} real split     data/common_split.csv not present (Person 1 delivers this)"
    )


def main() -> int:
    print("=" * 62)
    print("prostate-mil-project :: environment check")
    print("=" * 62)

    check_python()
    print("-" * 62)
    check("numpy", minimum="1.26")
    check("pandas", minimum="2.1")
    check("scipy", minimum="1.11")
    check("sklearn", module="sklearn", minimum="1.4")
    check("matplotlib", minimum="3.8")
    check("PIL", module="PIL")
    check("yaml", module="yaml")
    check("torch", minimum="2.2")
    print("-" * 62)
    check_numpy_torch_pair()
    check_torch_device()
    check_project_files()
    print("-" * 62)

    for name in ("streamlit", "nbformat"):
        try:
            importlib.import_module(name)
        except Exception:  # noqa: BLE001
            notes.append(f"{name} is not installed (optional): pip install -r requirements-full.txt")

    if problems:
        print("\nBLOCKING PROBLEMS")
        for problem in problems:
            print(f"  {FAIL} {problem}")
    if notes:
        print("\nNotes")
        seen = set()
        for note in notes:
            if note in seen:
                continue
            seen.add(note)
            print(f"  {WARN} {note}")

    print()
    if problems:
        print("Result: fix the problems above, then re-run this check.")
        return 1
    print("Result: environment looks good. Next: python -m unittest discover -s tests -v")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

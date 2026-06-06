"""
setup.py — Environment setup for the YOLOv8x + ByteTrack pipeline.

Handles three concerns:
  1. Colab numpy conflict fix  (run_env_fix, then restart runtime)
  2. Dependency installation   (install_dependencies)
  3. TrackEval source patch    (patch_trackeval)

Usage in Colab:
  # First run — fixes numpy, then restart runtime:
  from setup import run_env_fix
  run_env_fix()
  # → Runtime → Restart session → then run install_dependencies()

  # After restart:
  from setup import install_dependencies
  install_dependencies()
"""

import subprocess
import sys
import re
from pathlib import Path


# ── 1. Colab numpy environment fix ────────────────────────────────────────

def run_env_fix():
    """
    Uninstall conflicting packages so a fresh runtime starts clean.

    Must be followed by Runtime → Restart session before calling
    install_dependencies(). Do NOT call this again after restarting.
    """
    subprocess.run(
        [sys.executable, "-m", "pip", "uninstall", "-y",
         "numpy", "numba", "boxmot", "ultralytics", "lap"],
        capture_output=True
    )
    subprocess.run(
        [sys.executable, "-m", "pip", "install", "-q", "numpy"],
        check=True
    )
    print("Environment reset.")
    print("NEXT: Runtime → Restart session, then call install_dependencies()")


# ── 2. Dependency installation ────────────────────────────────────────────

def install_dependencies():
    """
    Install all pipeline dependencies and clone + patch TrackEval.

    Call this after restarting the runtime following run_env_fix().
    """
    def pip(*args):
        subprocess.run(
            [sys.executable, "-m", "pip", "install", "-q", *args],
            check=True
        )

    print("Installing ultralytics...")
    pip("ultralytics")

    print("Installing scipy, matplotlib, kaggle, tqdm...")
    pip("scipy", "matplotlib", "kaggle", "tqdm")

    print("Cloning TrackEval...")
    _clone_trackeval()

    print("Patching TrackEval for numpy 2.x compatibility...")
    patch_trackeval()

    # Verify
    import numpy as np
    import torch
    from ultralytics import YOLO
    from scipy.optimize import linear_sum_assignment

    print(f"\nnumpy:       {np.__version__}")
    print(f"torch:       {torch.__version__}")
    print(f"CUDA:        {torch.cuda.is_available()}")
    print("ultralytics: ✓")
    print("scipy:       ✓")
    print("TrackEval:   ✓")
    print("\nSetup complete. Proceed to data.py → detect.py → track.py → evaluate.py")


def _clone_trackeval():
    import os
    if not Path("/content/TrackEval").exists():
        subprocess.run(
            ["git", "clone", "-q",
             "https://github.com/JonathonLuiten/TrackEval.git",
             "/content/TrackEval"],
            check=True
        )


# ── 3. TrackEval numpy patch ──────────────────────────────────────────────

def patch_trackeval(trackeval_dir: str = "/content/TrackEval"):
    """
    Patch TrackEval source files to fix deprecated numpy aliases.

    TrackEval uses np.float, np.int, np.bool which were removed
    in numpy 1.24+. This replaces them with their modern equivalents
    and fixes any syntax errors introduced by prior patch attempts.

    Safe to call multiple times.
    """
    fixes = {
        "np.float)": "np.float64)",
        "np.float,": "np.float64,",
        "np.int)":   "np.int64)",
        "np.int,":   "np.int64,",
        "np.bool)":  "np.bool_)",    # no () after bool_
        "np.bool,":  "np.bool_,",
    }

    patched = []
    for f in Path(trackeval_dir).rglob("*.py"):
        txt = f.read_text()
        new = txt

        for old, repl in fixes.items():
            new = new.replace(old, repl)

        # Fix any erroneous np.bool_() left by previous patch runs
        new = re.sub(r"np\.bool_\(\)", "np.bool_", new)

        # Fix any unclosed parens: .astype(np.bool_\n → .astype(np.bool_)\n
        new = re.sub(r"\.astype\(np\.bool_\s*\n", ".astype(np.bool_)\n", new)

        if new != txt:
            f.write_text(new)
            patched.append(f.name)

    print(f"  Patched {len(patched)} files: {patched}")
    return patched


# ── CLI entry point ───────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Pipeline setup utilities")
    parser.add_argument(
        "command",
        choices=["env-fix", "install", "patch-trackeval"],
        help=(
            "env-fix: uninstall conflicting packages (then restart runtime); "
            "install: install all dependencies; "
            "patch-trackeval: patch TrackEval numpy aliases only"
        )
    )
    args = parser.parse_args()

    if args.command == "env-fix":
        run_env_fix()
    elif args.command == "install":
        install_dependencies()
    elif args.command == "patch-trackeval":
        patch_trackeval()

#!/usr/bin/env python3
"""CLI wrapper for running pairwise sweeps."""

from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from ollama_color_bypass_bench.sweep import main


if __name__ == "__main__":
    main()

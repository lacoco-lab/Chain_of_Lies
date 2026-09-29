#!/usr/bin/env python3
"""Run the preliminary filler gate without installing the local package."""

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))

from filler_gate.cli import main

if __name__ == "__main__":
    main()

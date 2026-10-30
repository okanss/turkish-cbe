"""Frozen test evaluation; execution additionally requires real pilot approval."""

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from inference.cli import stage_main

if __name__ == "__main__":
    raise SystemExit(stage_main("full", project_root=ROOT))

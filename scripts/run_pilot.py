"""Small matched pilot; the default action is a dry-run, never generation."""

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from inference.cli import stage_main

if __name__ == "__main__":
    raise SystemExit(stage_main("pilot", project_root=ROOT))

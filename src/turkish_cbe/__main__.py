"""Offline planning commands. There is intentionally no experiment command."""

import argparse
import json
from pathlib import Path
import sys

from .config import ConfigError, load_config
from .provenance import build_manifest, write_manifest


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Turkish-CBE: offline planning only")
    commands = parser.add_subparsers(dest="command", required=True)
    validate = commands.add_parser("validate", help="validate a plan and count requests")
    validate.add_argument("config", type=Path)
    manifest = commands.add_parser("manifest", help="write planning provenance, not results")
    manifest.add_argument("config", type=Path)
    manifest.add_argument("--output", required=True, type=Path)
    manifest.add_argument("--project-root", type=Path, default=Path.cwd(),
                          help="source checkout root (default: current directory)")
    args = parser.parse_args(argv)
    try:
        if args.command == "validate":
            config = load_config(args.config)
            result = {"status": config.status, "stage": config.stage,
                      "execution_ready": False, "budget": config.budget(),
                      "pending_gates": config.pending_gates}
        else:
            if not (args.project_root / "pyproject.toml").is_file():
                raise ValueError("--project-root must contain pyproject.toml")
            record = build_manifest(args.config, args.project_root)
            write_manifest(record, args.output)
            result = {"status": "planning_only", "manifest": str(args.output),
                      "source_tree_sha256": record["source_tree_sha256"]}
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return 0
    except (ConfigError, OSError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())

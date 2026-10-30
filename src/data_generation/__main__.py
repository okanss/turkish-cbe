"""Offline data generation and verification, separate from model evaluation."""

import argparse
import json
from pathlib import Path
import sys

from .dataset import verify_dataset, write_dataset
from .entities import ENTITY_TYPES
from .generator import DEFAULT_DISTANCES, DEFAULT_SIZES, GenerationPlan, QUERY_TYPES


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Turkish-CBE deterministic latent datasets")
    commands = parser.add_subparsers(dest="command", required=True)
    generate = commands.add_parser("generate", help="create a new validated JSONL dataset")
    generate.add_argument("--output", type=Path, default=Path("data/latent"))
    generate.add_argument("--manifest", type=Path, default=Path("dataset_manifest.json"))
    generate.add_argument("--seed", type=int, default=11)
    generate.add_argument("--sizes", type=int, nargs="+", default=DEFAULT_SIZES)
    generate.add_argument("--examples-per-size", type=int, default=200)
    generate.add_argument("--entity-types", nargs="+", choices=ENTITY_TYPES, default=ENTITY_TYPES)
    generate.add_argument("--query-types", nargs="+", choices=QUERY_TYPES, default=QUERY_TYPES)
    generate.add_argument("--distances", type=int, nargs="+", default=DEFAULT_DISTANCES)
    verify = commands.add_parser("verify", help="validate hashes, records, cells and shared orders")
    verify.add_argument("manifest", type=Path)
    verify.add_argument("--replay", action="store_true", help="regenerate bytes in memory and compare hashes")
    args = parser.parse_args(argv)
    try:
        if args.command == "generate":
            plan = GenerationPlan(seed=args.seed, sizes=tuple(args.sizes),
                                  examples_per_size=args.examples_per_size,
                                  entity_types=tuple(args.entity_types),
                                  query_types=tuple(args.query_types), distances=tuple(args.distances))
            manifest = write_dataset(args.output, args.manifest, plan)
            result = {"status": "validated_latent_data", "manifest": str(args.manifest),
                      "contains_model_results": False, "files": len(manifest["files"]),
                      "summary": manifest["summary"]}
        else:
            result = {"status": "verified", "replayed": args.replay,
                      "summary": verify_dataset(args.manifest, replay=args.replay)}
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return 0
    except (OSError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())

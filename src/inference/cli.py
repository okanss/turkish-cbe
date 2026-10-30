"""Dry-run-first command line. Live runs consume an already reviewed plan."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

from .config import load_models, select_models, ConfigError
from .dataset import default_selection
from .planning import make_plan, write_plan, validate_plan, DEFAULT_POLICY
from .runner import run_plan, approve_pilot
from .storage import RunStore, StorageError


def _path(root, value):
    path = Path(value)
    return path if path.is_absolute() else root / path


def stage_main(stage, argv=None, project_root=None):
    p = argparse.ArgumentParser(description="Plan first; --execute requires a saved plan.")
    p.add_argument("--project-root", type=Path, default=project_root or Path.cwd())
    p.add_argument("--models", default="configs/models.yaml")
    p.add_argument("--model", nargs="+")
    p.add_argument("--dataset-manifest", default="data/complete_grid/manifest.json")
    p.add_argument("--sizes", nargs="+", type=int)
    p.add_argument("--groups-per-size", type=int)
    for key in ("query-types", "distances", "name-controls", "relations", "languages", "conditions"):
        p.add_argument("--"+key, nargs="+", type=int if key == "distances" else str)
    for key in ("requests-per-minute", "tokens-per-minute", "max-attempts", "max-calls", "replicates"):
        p.add_argument("--"+key, type=int)
    p.add_argument("--max-estimated-cost", type=float)
    p.add_argument("--cache", action="store_true")
    mode = p.add_mutually_exclusive_group()
    mode.add_argument("--execute", action="store_true")
    mode.add_argument("--dry-run", action="store_true")
    p.add_argument("--plan", help="Previously generated immutable plan JSON")
    p.add_argument("--output", default=f"logs/{stage}_dry_run.json")
    p.add_argument("--run-id")
    p.add_argument("--pilot-approval")
    p.add_argument("--linguistic-review", help="Reviewed sample receipt required for a large pilot")
    args = p.parse_args(argv)
    root = args.project_root.resolve()
    try:
        if args.plan:
            selection_keys = ("sizes", "groups_per_size", "query_types", "distances", "name_controls",
                              "relations", "languages", "conditions")
            policy_keys = tuple(k for k in DEFAULT_POLICY if k != "cache")
            if args.model or args.cache or any(getattr(args, k, None) is not None for k in selection_keys+policy_keys):
                raise ValueError("a saved plan cannot be combined with new selection/policy/model flags")
            plan = json.loads(_path(root, args.plan).read_bytes())
            validate_plan(plan, root)
            if plan["stage"] != stage:
                raise ValueError("plan belongs to another stage")
        else:
            if args.execute:
                raise ValueError("--execute requires --plan; generate and review a dry-run first")
            if not args.model:
                raise ValueError("select explicit model IDs with --model")
            models = select_models(load_models(_path(root, args.models)), args.model)
            selection = default_selection(stage)
            for key in selection:
                if hasattr(args, key) and getattr(args, key) is not None:
                    selection[key] = getattr(args, key)
            policy = dict(DEFAULT_POLICY)
            for key in policy:
                value = getattr(args, key, None)
                if value is not None:
                    policy[key] = value
            plan = make_plan(root, args.dataset_manifest, models, stage, selection, policy)
            write_plan(plan, _path(root, args.output))
        result = run_plan(root, plan, args.run_id, execute=args.execute,
                          approval=_path(root, args.pilot_approval) if args.pilot_approval else None,
                          linguistic_review=_path(root,args.linguistic_review) if args.linguistic_review else None)
        result["model_budgets"] = plan["model_budgets"]
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return 0 if result.get("dry_run") or result.get("complete") else 2
    except (ValueError, OSError, ConfigError, StorageError) as error:
        p.exit(2, f"error: {error}\n")


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    p = argparse.ArgumentParser(description="Turkish-CBE inference and approval controls")
    p.add_argument("command", choices=("pilot", "full", "approve-pilot", "resolve"))
    args, remaining = p.parse_known_args(argv)
    if args.command in {"pilot", "full"}:
        return stage_main(args.command, remaining)
    q = argparse.ArgumentParser(prog="turkish-cbe-evaluate "+args.command)
    q.add_argument("--project-root", type=Path, default=Path.cwd())
    q.add_argument("--run-id", required=True)
    if args.command == "approve-pilot":
        q.add_argument("--approved-by", required=True)
        q.add_argument("--linguistic-review", required=True)
        q.add_argument("--output", required=True)
    else:
        q.add_argument("--plan", required=True)
        q.add_argument("--task-id", required=True)
        q.add_argument("--decision", choices=("retry", "abandon"), required=True)
        q.add_argument("--reason", required=True)
        q.add_argument("--acknowledge-duplicate-risk", action="store_true")
    a = q.parse_args(remaining)
    root = a.project_root.resolve()
    try:
        if args.command == "approve-pilot":
            result = approve_pilot(root, a.run_id, _path(root, a.output), approved_by=a.approved_by,
                                   linguistic_review=_path(root, a.linguistic_review))
        else:
            plan = json.loads(_path(root, a.plan).read_bytes())
            validate_plan(plan, root)
            with RunStore(root, a.run_id, plan) as store:
                result = {"resolution_id": store.resolve_ambiguous(
                    a.task_id, a.decision, a.reason, ack_duplicate_risk=a.acknowledge_duplicate_risk)}
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return 0
    except (ValueError, OSError, StorageError) as error:
        q.exit(2, f"error: {error}\n")

"""Inspect/plan first; optimization and held-out inference require --execute."""

import argparse
import json
from pathlib import Path
import sys

from .config import load_config
from .data import export_data
from .engine import run_training
from .evaluation import evaluate
from .hardware import inspect_hardware
from .planning import make_plan, validate_plan, write_plan
from .reporting import report


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("command", choices=("inspect", "plan", "export", "run", "evaluate", "report"))
    p.add_argument("--project-root", type=Path, default=Path.cwd())
    p.add_argument("--config", default=None)
    p.add_argument("--plan", default=None)
    p.add_argument("--seed", type=int)
    p.add_argument("--output")
    p.add_argument("--arm")
    p.add_argument("--adapter-run")
    p.add_argument("--linguistic-review")
    p.add_argument("--evaluation", nargs="+", default=[])
    p.add_argument("--training-run", nargs="+", default=[])
    p.add_argument("--max-calls", type=int, default=2000)
    p.add_argument("--bootstrap-repeats", type=int, default=2000)
    p.add_argument("--probe-cuda", action="store_true")
    p.add_argument("--execute", action="store_true")
    a = p.parse_args(argv)
    root = a.project_root.resolve()
    def path(value):
        value = Path(value)
        return value if value.is_absolute() else root / value
    try:
        if a.command == "inspect":
            if a.execute: raise ValueError("inspection never executes training")
            result = inspect_hardware(probe_torch=a.probe_cuda)
            if a.output:
                from inference.runner import _immutable
                _immutable(path(a.output), result)
        elif a.command == "plan":
            if a.plan or a.execute: raise ValueError("planning accepts config, never execution or a saved plan")
            config = load_config(path(a.config or "configs/training/pilot.toml"))
            if a.seed is not None: config["seed"] = a.seed
            result = make_plan(root, config, hardware=inspect_hardware(probe_torch=a.probe_cuda))
            if not a.output: raise ValueError("choose a new --output plan JSON")
            write_plan(result, path(a.output))
        else:
            if not a.plan or a.config or a.seed is not None or a.probe_cuda:
                raise ValueError("consume an explicit --plan without new config/seed/probe overrides")
            plan = json.loads(path(a.plan).read_bytes()); validate_plan(plan, root)
            if not a.output: raise ValueError("choose a new --output")
            if a.command == "export":
                if a.execute: raise ValueError("data export does not execute training")
                result = export_data(root, plan["data"], path(a.output))
            elif a.command == "run":
                result = run_training(root, plan, a.arm, path(a.output), execute=a.execute)
            elif a.command == "evaluate":
                result = evaluate(root, plan, a.arm, path(a.output),
                    adapter_run=path(a.adapter_run) if a.adapter_run else None,
                    execute=a.execute, max_calls=a.max_calls,
                    linguistic_review=path(a.linguistic_review) if a.linguistic_review else None)
            else:
                if a.execute: raise ValueError("reporting never executes a model")
                result = report(root, plan, path(a.output),
                    evaluations=[path(value) for value in a.evaluation], training_runs=[path(value) for value in a.training_run],
                    repeats=a.bootstrap_repeats, seed=plan["config"]["seed"])
        print(json.dumps(result, sort_keys=True, ensure_ascii=False, allow_nan=False))
        return 0
    except (ValueError, OSError, KeyError, ImportError) as error:
        print("error: "+str(error), file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())

"""Dry-run-first reasoning plans, controlled execution, and immutable analysis."""

from __future__ import annotations

import argparse
import importlib.metadata
import json
from pathlib import Path
import platform
import shutil
import sys
import tempfile

from analysis.io import canonical, digest
from analysis.reporting import export_tables
from inference.config import load_models, select_models
from inference.dataset import default_selection
from inference.planning import DEFAULT_POLICY
from inference.storage import StorageError

from . import REASONING_VERSION
from .analysis import build_tables, export_figures
from .io import discover_runs, load_run
from .planning import load_settings, make_plan, write_plan, validate_plan
from .runner import run_plan, approve_pilot, ReasoningStore


def _path(root, value):
    path = Path(value)
    return path if path.is_absolute() else root / path


def analyse(root, output, *, run_ids=None, seed=20261025, repeats=2000, figures=False):
    """Write a new immutable analysis directory; never mutate inference files."""
    root, output = Path(root).resolve(), Path(output).absolute()
    if type(seed) is not int or seed < 0 or type(repeats) is not int or repeats < 1:
        raise ValueError("invalid analysis seed or bootstrap repeat count")
    if output.exists():
        raise FileExistsError("analysis output already exists; choose a new directory")
    if any(p.is_symlink() for p in (output, *output.parents)):
        raise ValueError("analysis output must not traverse symlinks")
    # Never place a new analysis inside raw data, normalized data, source, or
    # frozen datasets. Destination defaults to a separate results/analysis tree.
    for directory in ("results/raw", "results/normalized", "data", "src", "configs", "tests", "logs",
                      "results/reasoning/raw", "results/reasoning/normalized", "results/reasoning/checkpoints",
                      "results/reasoning/plans", "results/reasoning/cache"):
        if output.is_relative_to(root / directory):
            raise ValueError("analysis destination overlaps protected input")
    ids = discover_runs(root) if run_ids is None else run_ids
    if not isinstance(ids, list) or len(set(ids)) != len(ids):
        raise ValueError("run IDs must be a list without duplicates")
    loaded = [load_run(root, identifier) for identifier in sorted(ids)]
    if any(run["synthetic"] for run in loaded):
        raise ValueError("synthetic fixtures cannot be published by the research CLI")
    observations, keys = [], set()
    for run in loaded:
        if run["plan"]["cohort"]["cohort_sha256"] != loaded[0]["plan"]["cohort"]["cohort_sha256"]:
            raise ValueError("runs have different planned cohorts; analyse them separately")
        for row in run["observations"]:
            # Different replicas/runs of one task are not new latent evidence.
            if row["task_id"] in keys:
                raise ValueError("overlapping tasks across runs; analyse independent replications separately")
            keys.add(row["task_id"])
            observations.append(row)
    from .planning import source_fingerprints
    source_files = source_fingerprints(root)
    config = {"seed":seed, "bootstrap_repeats":repeats, "confidence":0.95,
              "replicate":0, "cluster":"latent_instance_id", "strata":["n"],
              "run_ids":sorted(ids), "figures":figures}
    versions = {}
    for name in ("matplotlib", "numpy"):
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = None
    has_responses = any(r["response_observed"] for r in observations)
    result = {"reasoning_version":REASONING_VERSION, "artifact_type":"reasoning_cbe_analysis",
        "status":"completed" if has_responses else "no_model_results",
        "contains_model_results":has_responses, "synthetic":False,
        "config":config, "config_sha256":digest(canonical(config).encode()),
        "environment":{"python":platform.python_version(),"platform":platform.platform(),"packages":versions},
        "source_files_sha256":source_files,
        "input_runs":[{"run_id":r["run_id"],"plan_sha256":r["plan"]["plan_sha256"],
            "dataset_manifest_sha256":r["plan"]["cohort"]["dataset_manifest_sha256"],
            "input_files_sha256":r["input_files_sha256"],"ignored_replicates":r["ignored_replicates"]} for r in loaded],
        "n_planned_responses":len(observations),
        "n_observed_model_responses":sum(r["response_observed"] for r in observations),
        "n_missing_responses":sum(r["response_status"]=="missing" for r in observations),
        "n_invalid_responses":sum(r["prediction"] is None for r in observations),
        "human_review_status":sorted({r["human_review_status"] for r in observations}),
        "eligible_for_confirmatory_evaluation": bool(observations) and all(
            r["eligible_for_confirmatory_evaluation"] for r in observations),
        "limitations":["Exploratory implementation; not an external preregistration.",
            "Conditional metrics share joint-valid support; marginal accuracy retains failures.",
            "Intervals are pointwise, not simultaneous; partial undefined bootstrap intervals remain null.",
            "Model families, native effort and explicit prompting interventions remain distinct.",
            "Reasoning text is not scored or treated as faithful evidence of internal computation.",
            "Improvements on this frozen family do not establish competence on all family members."]}
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=".analysis-", dir=output.parent))
    try:
        artifacts = []
        if has_responses:
            tables = build_tables(observations, seed=seed, repeats=repeats)
            artifacts += export_tables(tables, temporary / "tables")
            if figures:
                artifacts += export_figures(tables, temporary / "figures")
            result["table_rows"] = {key:len(rows) for key,rows in tables.items()}
        else:
            result["table_rows"] = {}
            result["message"] = "No recorded model runs; no performance tables, p-values or figures were created."
        result["files"] = [{"path":str(path.relative_to(temporary)),"sha256":digest(path.read_bytes()),
                            "bytes":path.stat().st_size} for path in sorted(artifacts)]
        (temporary / "analysis_manifest.json").write_text(json.dumps(result,sort_keys=True,
            ensure_ascii=False,indent=2,allow_nan=False)+"\n",encoding="utf-8")
        # Rename publishes only a completed report. A nonempty destination
        # created concurrently is never replaced by Path.rename.
        if output.exists():
            raise FileExistsError("analysis destination appeared concurrently")
        temporary.rename(output)
    except BaseException:
        shutil.rmtree(temporary, ignore_errors=True)
        raise
    return result


def stage_main(stage, argv):
    p = argparse.ArgumentParser(description="Plan the same cohort for all reasoning arms.")
    p.add_argument("--project-root", type=Path, default=Path.cwd())
    p.add_argument("--models", default=None)
    p.add_argument("--model", nargs="+")
    p.add_argument("--settings", default=None)
    p.add_argument("--dataset-manifest", default=None)
    p.add_argument("--sizes", nargs="+", type=int)
    p.add_argument("--groups-per-size", type=int)
    for key in ("query-types", "distances", "name-controls", "relations", "languages", "conditions"):
        p.add_argument("--"+key, nargs="+", type=int if key == "distances" else str)
    for key in ("requests-per-minute", "tokens-per-minute", "max-attempts", "max-calls", "replicates"):
        p.add_argument("--"+key, type=int)
    p.add_argument("--max-estimated-cost", type=float)
    p.add_argument("--cache", action="store_true", default=None)
    mode = p.add_mutually_exclusive_group()
    mode.add_argument("--execute", action="store_true")
    mode.add_argument("--dry-run", action="store_true")
    p.add_argument("--plan")
    p.add_argument("--output", default=f"results/reasoning/plans/{stage}.json")
    p.add_argument("--run-id")
    p.add_argument("--pilot-approval")
    p.add_argument("--linguistic-review", help="Reviewed sample receipt required for a large pilot")
    a = p.parse_args(argv)
    root = a.project_root.resolve()
    try:
        selection_keys = ("sizes", "groups_per_size", "query_types", "distances", "name_controls",
                          "relations", "languages", "conditions")
        if a.plan:
            incompatible = selection_keys + tuple(DEFAULT_POLICY) + ("model", "models", "settings", "dataset_manifest")
            if any(getattr(a, key, None) is not None for key in incompatible):
                raise ValueError("saved plans cannot be combined with new design/config/policy flags")
            plan = json.loads(_path(root, a.plan).read_bytes())
            validate_plan(plan, root)
            if plan["stage"] != stage:
                raise ValueError("plan belongs to another stage")
        else:
            if a.execute:
                raise ValueError("--execute requires a saved --plan; review the dry-run first")
            if not a.model:
                raise ValueError("select explicit registry IDs with --model")
            models = select_models(load_models(_path(root, a.models or "configs/models.yaml")), a.model)
            settings = load_settings(_path(root, a.settings or "configs/reasoning.toml"))
            selection = default_selection(stage)
            selection["query_types"] = ["endpoint", "internal", "distance_controlled"]
            for key in selection_keys:
                if getattr(a, key) is not None:
                    selection[key] = getattr(a, key)
            policy = dict(DEFAULT_POLICY)
            for key in policy:
                value = getattr(a, key, None)
                if value is not None:
                    policy[key] = value
            plan = make_plan(root, a.dataset_manifest or "data/complete_grid/manifest.json", models,
                             stage, selection, policy, settings)
            write_plan(plan, _path(root, a.output))
        result = run_plan(root, plan, a.run_id, execute=a.execute,
                          approval=_path(root, a.pilot_approval) if a.pilot_approval else None,
                          linguistic_review=_path(root,a.linguistic_review) if a.linguistic_review else None)
        result.update(arm_budgets=plan["arm_budgets"], unavailable_arms=plan["unavailable_arms"])
        print(json.dumps(result, ensure_ascii=False, sort_keys=True, allow_nan=False))
        return 0 if result.get("dry_run") or result.get("complete") else 2
    except (ValueError, OSError, KeyError, StorageError) as error:
        p.exit(2, f"error: {error}\n")


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("pilot", "full", "analyse", "approve-pilot", "resolve"))
    command, remaining = parser.parse_known_args(argv)
    if command.command in {"pilot", "full"}:
        return stage_main(command.command, remaining)
    p = argparse.ArgumentParser(prog="turkish-cbe-reasoning " + command.command)
    p.add_argument("--project-root", type=Path, default=Path.cwd())
    if command.command == "analyse":
        p.add_argument("--run", nargs="+", dest="run_ids")
        p.add_argument("--output", type=Path, required=True)
        p.add_argument("--seed", type=int, default=20261025)
        p.add_argument("--bootstrap-repeats", type=int, default=2000)
        p.add_argument("--figures", action="store_true")
    else:
        p.add_argument("--run-id", required=True)
        if command.command == "approve-pilot":
            p.add_argument("--approved-by", required=True)
            p.add_argument("--linguistic-review", required=True)
            p.add_argument("--output", required=True)
        else:
            p.add_argument("--plan", required=True)
            p.add_argument("--task-id", required=True)
            p.add_argument("--decision", choices=("retry", "abandon"), required=True)
            p.add_argument("--reason", required=True)
            p.add_argument("--acknowledge-duplicate-risk", action="store_true")
    a = p.parse_args(remaining)
    root = a.project_root.resolve()
    try:
        if command.command == "analyse":
            result = analyse(root, _path(root, a.output), run_ids=a.run_ids, seed=a.seed,
                             repeats=a.bootstrap_repeats, figures=a.figures)
        elif command.command == "approve-pilot":
            result = approve_pilot(root, a.run_id, _path(root, a.output), approved_by=a.approved_by,
                                   linguistic_review=_path(root, a.linguistic_review))
        else:
            plan = json.loads(_path(root, a.plan).read_bytes())
            validate_plan(plan, root)
            with ReasoningStore(root, a.run_id, plan) as store:
                result = {"resolution_id": store.resolve_ambiguous(
                    a.task_id, a.decision, a.reason, ack_duplicate_risk=a.acknowledge_duplicate_risk)}
        print(json.dumps(result, ensure_ascii=False, sort_keys=True, allow_nan=False))
        return 0
    except (ValueError, OSError, KeyError, StorageError) as error:
        print("error: " + str(error), file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())


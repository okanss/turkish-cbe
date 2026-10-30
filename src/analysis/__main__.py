"""Analyse recorded runs, or explicitly report the absence of model results."""

from __future__ import annotations

import argparse
import importlib.metadata
import json
from pathlib import Path
import platform
import shutil
import sys
import tempfile

from . import ANALYSIS_VERSION
from .io import canonical, digest, discover_runs, load_run
from .pipeline import build_tables
from .reporting import export_figures, export_tables


def analyse(root, output, *, run_ids=None, seed=20261024, repeats=2000, figures=False):
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
    for directory in ("results/raw", "results/normalized", "data", "src", "configs", "tests", "logs"):
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
    source_files = {str(path.relative_to(root)): digest(path.read_bytes())
                    for path in sorted((root / "src/analysis").glob("*.py"))}
    active = Path(__file__).resolve().parent
    for relative, sha in source_files.items():
        if digest((active / Path(relative).name).read_bytes()) != sha:
            raise ValueError("installed analysis code differs from project sources")
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
    result = {"analysis_version":ANALYSIS_VERSION, "artifact_type":"cbe_analysis",
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
            "Model families remain separated by model and revision; token associations are descriptive."]}
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


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root",type=Path,default=Path.cwd())
    parser.add_argument("--run",nargs="+",dest="run_ids")
    parser.add_argument("--output",type=Path,required=True)
    parser.add_argument("--seed",type=int,default=20261024)
    parser.add_argument("--bootstrap-repeats",type=int,default=2000)
    parser.add_argument("--figures",action="store_true",help="export PDF/SVG/PNG with pinned Matplotlib")
    args=parser.parse_args(argv)
    try:
        result=analyse(args.root,args.output,run_ids=args.run_ids,seed=args.seed,
                       repeats=args.bootstrap_repeats,figures=args.figures)
    except (ValueError, OSError, KeyError) as error:
        print("error: "+str(error),file=sys.stderr)
        return 2
    print(json.dumps({key:result[key] for key in ("status","contains_model_results",
          "n_planned_responses","n_missing_responses","n_invalid_responses","table_rows")},
          ensure_ascii=False,sort_keys=True,allow_nan=False))
    return 0


if __name__=="__main__":
    raise SystemExit(main())

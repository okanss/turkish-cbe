"""Reporting is separate from optimization; unrun arms never become results."""

import hashlib
import json
from pathlib import Path
import shutil
import tempfile

from analysis.pipeline import build_tables
from analysis.reporting import export_tables
from inference.dataset import fingerprint
from .evaluation import load_evaluation, adapter_identity
from .planning import validate_plan
from .comparison import training_contrasts


def report(root, plan, output, *, evaluations=None, training_runs=None, repeats=2000, seed=42):
    validate_plan(plan, root)
    if type(repeats) is not int or repeats < 1 or type(seed) is not int or seed < 0:
        raise ValueError("invalid bootstrap settings")
    output, root = Path(output).absolute(), Path(root).resolve()
    if (output.exists() or not output.is_relative_to(root / "results/training/reports")
            or any(p.is_symlink() for p in (output, *output.parents))):
        raise ValueError("choose a new training reports directory")
    loaded = [load_evaluation(root, path) for path in (evaluations or [])]
    if any(item["plan"] != plan for item in loaded): raise ValueError("comparison requires the same model/data/config/seed plan")
    arms = [item["manifest"]["identity"]["arm"] for item in loaded]
    if len(set(arms)) != len(arms): raise ValueError("duplicate arm/run; repeated answers are not independent examples")
    trained, identities = {}, {}
    for directory in (training_runs or []):
        directory = Path(directory)
        summary = json.loads((directory / "training_summary.json").read_bytes())
        arm = summary["arm"]
        identities[arm] = adapter_identity(plan, arm, directory)
        if arm in trained: raise ValueError("duplicate trained arm; analyse seeds/runs separately")
        trained[arm] = summary
    for item in loaded:
        identity = item["manifest"]["identity"]
        arm = identity["arm"]
        if arm != "base" and (arm not in identities or identities[arm] != identity):
            raise ValueError("adapter evaluation requires its exact verified completed training evidence")
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=".training-report-", dir=output.parent))
    try:
        status = {"artifact_type": "training_comparison_report", "training_version": plan["training_version"],
            "plan_sha256": plan["plan_sha256"], "status": "completed_evaluation_analysis" if loaded else (
                "training_runs_recorded_no_evaluation" if trained else "no_training_or_model_results"),
            "contains_model_results": bool(loaded), "contains_training_run": bool(trained),
            "bootstrap_repeats": repeats, "seed": seed,
            "actual_optimizer_steps": sum(item["stats"]["optimizer_steps"] for item in trained.values()),
            "evaluated_arms": arms, "missing_evaluation_arms": [arm for arm in plan["config"]["arms"] if arm not in arms],
            "arm_states": [{"arm": budget["arm"], "evaluation": "observed" if budget["arm"] in arms else "not_executed",
                "training": "no_training_base" if budget["arm"] == "base" else (
                    "completed_with_verified_adapter" if budget["arm"] in trained else "no_training_evidence_in_report"),
                "budget": budget, "observed_training": trained.get(budget["arm"])} for budget in plan["arm_budgets"]],
            "input_evaluations": [{"identity": item["manifest"]["identity"], "manifest_sha256": item["manifest_sha256"]} for item in loaded],
            "eligible_for_confirmatory_evaluation": False,
            "lexical_overlap_audit": plan["data"]["overlap_audit"]["lexical"],
            "limitations": ["Planned steps/tokens are not actual training usage.",
                "Equal optimizer steps do not equalize GPU time, tokens or number of unique examples.",
                "Held-out predicate vocabulary is clean; English contexts share higher/lower.",
                "No selection of a training seed or checkpoint on the frozen test set is allowed.",
                "Improvement on a finite benchmark does not prove family-wide competence."]}
        files, all_tables = [], {}
        # Arm identity is part of MODEL, so pooling the input retains separate
        # profiles while correcting shared scientific families across all arms.
        # Calling the legacy analyser separately per arm would reset Holm.
        arm_by_model = {row["model_id"]: item["manifest"]["identity"]["arm"]
                        for item in loaded for row in item["observations"]}
        for transfer in plan["data"]["evaluation"]:
            observations = [row for item in loaded for row in item["observations"] if row["transfer"] == transfer]
            if not observations: continue
            tables = build_tables(observations, seed=seed, repeats=repeats)
            for name, rows in tables.items():
                for row in rows: row.update(training_arm=arm_by_model[row["model_id"]], transfer=transfer)
                all_tables.setdefault(name, []).extend(rows)
            contrasts=training_contrasts(observations,seed=seed,repeats=repeats,
                declared_arms=[a for a in plan["config"]["arms"] if a!="base"])
            for row in contrasts:row["transfer"]=transfer
            all_tables.setdefault("training_arm_contrasts",[]).extend(contrasts)
        if loaded: files = export_tables(all_tables, temporary / "tables")
        status["table_rows"] = {key: len(rows) for key, rows in all_tables.items()}
        status["files"] = [{"path": str(p.relative_to(temporary)), "sha256": hashlib.sha256(p.read_bytes()).hexdigest(),
                            "bytes": p.stat().st_size} for p in files]
        status["config_sha256"] = fingerprint({"plan": plan["plan_sha256"], "repeats": repeats, "seed": seed})
        (temporary / "training_report.json").write_text(json.dumps(status, sort_keys=True, ensure_ascii=False, indent=2, allow_nan=False)+"\n")
        if output.exists(): raise FileExistsError("report destination appeared concurrently")
        temporary.rename(output)
        return status
    except BaseException:
        shutil.rmtree(temporary, ignore_errors=True)
        raise

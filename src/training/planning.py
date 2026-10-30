"""Metadata-only training budgets, immutable identities and resource blockers."""

from copy import deepcopy
import hashlib
import importlib.metadata
import json
import math
from pathlib import Path

from inference.config import load_models, select_models, validate_model
from inference.dataset import fingerprint
from inference.planning import runtime_environment
from . import TRAINING_VERSION
from .config import validate_config
from .data import plan_data
from .hardware import inspect_hardware, check_resources


def source_fingerprints(root):
    root = Path(root)
    active = Path(__file__).resolve().parents[1]
    result = {}
    for path in sorted((root / "src").rglob("*.py")):
        relative = path.relative_to(root / "src")
        if not (active / relative).is_file() or (active / relative).read_bytes() != path.read_bytes():
            raise ValueError("installed training source differs from project: " + str(relative))
        result["src/"+relative.as_posix()] = hashlib.sha256(path.read_bytes()).hexdigest()
    return result


def arm_budgets(config, data):
    counts = data["counts"]
    result = []
    for arm in config["arms"]:
        paired = arm in {"paired_sft", "sft_consistency"}
        grpo = arm.startswith("grpo_")
        rows = counts["paired_rows"] if paired else counts["canonical_rows"]
        if arm == "base":
            steps = exposures = 0
        elif grpo:
            steps = config["grpo"]["max_steps"]
            exposures = steps * 4 * config["data"]["grpo_rollouts_per_view"]
        else:
            steps = config["sft"]["max_steps"] or math.ceil(rows / config["sft"]["effective_batch_size"]) * config["sft"]["epochs"]
            exposures = (steps * config["sft"]["effective_batch_size"] if config["sft"]["max_steps"]
                         else rows * config["sft"]["epochs"])
        row = {"arm": arm, "state": "planned_not_executed", "planned_optimizer_steps": steps,
            "unique_training_latents_available": counts["training_latent_instances"],
            "unique_query_relation_units_available": counts["grpo_selected_units"] if grpo else counts["training_units"],
            "unique_surface_rows_available": counts["grpo_views"] if grpo else rows,
            "planned_sequence_exposures": exposures,
            "exposure_scope": "rollout samples; four views and G=4" if grpo else "supervised rows; repeated epochs/cycling are exposures, not new examples",
            "maximum_training_token_exposures": exposures * config["budget"]["max_length"],
            "maximum_accounted_forward_tokens": exposures * config["budget"]["max_length"] * (
                3 if grpo or arm == "sft_consistency" else 1),
            "exact_planned_training_tokens": None, "actual_training_tokens": None,
            "actual_optimizer_steps": 0, "actual_gpu_seconds": None, "actual_cost": None,
            "trainable_parameters": 0 if arm == "base" else None,
            "parameter_update_budget": 0 if arm == "base" else None,
            "agreement_weight": config["consistency"]["agreement_bonus_weight"] if arm == "grpo_consistency" else 0.0,
            "js_weight": config["consistency"]["js_weight"] if arm == "sft_consistency" else 0.0,
            "advantage_mode": config["grpo"]["advantage_mode"] if grpo else None}
        if arm == "base":
            row.update(unique_training_latents_available=0, unique_query_relation_units_available=0,
                       unique_surface_rows_available=0, exact_planned_training_tokens=0)
        result.append(row)
    return result


def make_plan(root, config, *, hardware=None):
    root = Path(root).resolve()
    config = deepcopy(config)
    validate_config(config)
    model = select_models(load_models(root / config["model_registry"]), [config["model_id"]])[0]
    data = plan_data(root, config["dataset_manifest"], config["data"], seed=config["seed"])
    inventory = deepcopy(inspect_hardware() if hardware is None else hardware)
    resources = check_resources(config, model, inventory)
    budgets = arm_budgets(config, data)
    blockers = list(resources["blockers"])
    for budget in budgets:
        if budget["planned_optimizer_steps"] > config["budget"]["max_optimizer_steps"]:
            blockers.append("optimizer_step_budget_exceeded:"+budget["arm"])
    environment = runtime_environment()
    distributions = {}
    for distribution in importlib.metadata.distributions():
        name = distribution.metadata.get("Name")
        if name:
            distributions.setdefault(name, set()).add(distribution.version)
    environment["all_installed_distribution_versions"] = {name: sorted(versions) for name, versions in sorted(distributions.items())}
    plan = {"schema_version": 1, "artifact_type": "training_dry_run_plan",
        "training_version": TRAINING_VERSION, "config": config, "config_sha256": fingerprint(config),
        "model": model, "model_spec_sha256": fingerprint(model), "data": data,
        "hardware": inventory, "resources": resources, "arm_budgets": budgets,
        "blockers": sorted(set(blockers)), "environment": environment,
        "source_files_sha256": source_fingerprints(root), "contains_training_run": False,
        "contains_model_results": False, "actual_optimizer_steps": 0,
        "estimated_gpu_hours": None, "estimated_cost": None,
        "fairness": {"matched_optimizer_steps": config["profile"] == "matched_steps",
            "matched_token_budget": False, "matched_gpu_time": False,
            "note": "Same LoRA target set; equal steps do not imply equal token, compute or unique-example budgets."},
        "warnings": ["Base BF16 weights are a memory lower bound, not a training-fit estimate.",
            "Exact chat/assistant tokens require encoding every selected sequence before optimizer initialization.",
            "Full English contexts share higher/lower; predicate lexicon holdout does not establish full-context lexical isolation.",
            "Prompt-normalized binary advantages can make the agreement term inert; diagnostics are required.",
            "No seed/checkpoint selection on test results is permitted; report every predeclared seed separately."]}
    plan["plan_sha256"] = fingerprint(plan)
    return plan


def validate_plan(plan, root=None):
    if not isinstance(plan, dict):
        raise ValueError("training plan must be an object")
    payload = {k: v for k, v in plan.items() if k != "plan_sha256"}
    if (plan.get("plan_sha256") != fingerprint(payload) or plan.get("training_version") != TRAINING_VERSION
            or plan.get("artifact_type") != "training_dry_run_plan"):
        raise ValueError("training plan hash/version mismatch")
    validate_config(plan["config"]); validate_model(plan["model"])
    if plan["config_sha256"] != fingerprint(plan["config"]) or plan["model_spec_sha256"] != fingerprint(plan["model"]):
        raise ValueError("training config/model hash mismatch")
    data = plan["data"]
    if data["data_sha256"] != fingerprint({k:v for k,v in data.items() if k != "data_sha256"}):
        raise ValueError("training data hash mismatch")
    if plan["arm_budgets"] != arm_budgets(plan["config"], data):
        raise ValueError("derived training budget changed")
    resources = check_resources(plan["config"], plan["model"], plan["hardware"])
    blockers = list(resources["blockers"])
    blockers.extend("optimizer_step_budget_exceeded:"+row["arm"] for row in plan["arm_budgets"]
                    if row["planned_optimizer_steps"] > plan["config"]["budget"]["max_optimizer_steps"])
    if resources != plan["resources"] or sorted(set(blockers)) != plan["blockers"]:
        raise ValueError("derived training resource/blocker fields changed")
    if plan["estimated_gpu_hours"] is not None or plan["estimated_cost"] is not None:
        raise ValueError("unmeasured training time/cost must remain unknown")
    if plan["contains_training_run"] is not False or plan["contains_model_results"] is not False or plan["actual_optimizer_steps"] != 0:
        raise ValueError("a plan cannot contain a completed training run")
    if root is not None and plan["source_files_sha256"] != source_fingerprints(root):
        raise ValueError("training source changed; create a new plan")


def write_plan(plan, path):
    validate_plan(plan)
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as handle:
        json.dump(plan, handle, sort_keys=True, ensure_ascii=False, indent=2, allow_nan=False)
        handle.write("\n")

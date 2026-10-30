"""Offline call/token budgets and deterministic execution provenance."""

from __future__ import annotations

from copy import deepcopy
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
import math
from pathlib import Path
import platform

from . import INFERENCE_VERSION
from .config import validate_model
from .dataset import GridDataset, default_selection, fingerprint
from .parser import PARSER_VERSION
from .prompts import PROMPT_VERSION


DEFAULT_POLICY = {"requests_per_minute": 30, "tokens_per_minute": 60000,
                  "max_attempts": 3, "backoff_initial_seconds": 1.0,
                  "backoff_max_seconds": 30.0, "max_calls": 2000,
                  "max_estimated_cost": None, "cache": False, "replicates": 1}


def validate_policy(policy):
    if not isinstance(policy, dict) or set(policy) != set(DEFAULT_POLICY):
        raise ValueError("unknown or missing execution policy fields")
    for key in ("requests_per_minute", "tokens_per_minute", "max_attempts", "max_calls", "replicates"):
        if type(policy[key]) is not int or policy[key] < 1:
            raise ValueError("positive integer required: " + key)
    if policy["max_attempts"] > 10:
        raise ValueError("at most ten transport attempts per logical call")
    for key in ("backoff_initial_seconds", "backoff_max_seconds"):
        if type(policy[key]) not in (float, int) or not math.isfinite(policy[key]) or not 0 < policy[key] <= 60:
            raise ValueError("backoff must be finite and within (0,60]")
    if policy["backoff_initial_seconds"] > policy["backoff_max_seconds"]:
        raise ValueError("backoff initial exceeds maximum")
    cost = policy["max_estimated_cost"]
    if cost is not None and (type(cost) not in (float, int) or not math.isfinite(cost) or cost < 0):
        raise ValueError("invalid cost cap")
    if type(policy["cache"]) is not bool:
        raise ValueError("cache must be boolean")
    if policy["cache"] and policy["replicates"] != 1:
        raise ValueError("cached responses cannot be independent replicates")


def source_fingerprints(root):
    root = Path(root)
    files = sorted((root / "src/inference").glob("*.py"))
    active = Path(__file__).resolve().parent
    for path in files:
        imported = active / path.name
        if not imported.is_file() or hashlib.sha256(imported.read_bytes()).digest() != hashlib.sha256(path.read_bytes()).digest():
            raise ValueError("installed inference code differs from project sources")
    files += sorted((root / "scripts").glob("run_*.py"))
    return {str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest() for p in files}


def runtime_environment():
    versions = {}
    for name in ("PyYAML", "tiktoken", "regex", "transformers", "torch", "vllm", "huggingface-hub"):
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = None
    return {"python": platform.python_version(), "platform": platform.platform(), "packages": versions}


def price_usage(spec, usage):
    """No missing price, missing usage or hardware cost is silently treated as zero."""
    p = spec["pricing"]
    inp, out = usage.get("input_tokens"), usage.get("output_tokens")
    if p["input_per_million"] is None or p["output_per_million"] is None or inp is None or out is None:
        return {"amount": None, "currency": p["currency"], "status": "unknown",
                "pricing_as_of": p["as_of"], "pricing_source": p["source"]}
    cached = usage.get("cached_input_tokens")
    cache_unknown = cached is None
    cached = cached or 0
    if cached > inp:
        raise ValueError("cached input cannot exceed total input")
    if cached and p["cached_input_per_million"] is None:
        return {"amount": None, "currency": p["currency"], "status": "unknown_cached_rate",
                "pricing_as_of": p["as_of"], "pricing_source": p["source"]}
    amount = ((inp-cached)*p["input_per_million"] + cached*(p["cached_input_per_million"] or 0)
              + out*p["output_per_million"]) / 1_000_000
    return {"amount": amount, "currency": p["currency"],
            "status": "conservative_no_cache_breakdown" if cache_unknown else "computed_from_reported_usage",
            "pricing_as_of": p["as_of"], "pricing_source": p["source"]}


def make_plan(root, manifest, models, stage="pilot", selection=None, policy=None):
    root = Path(root).resolve()
    if stage not in {"pilot", "full"} or not models:
        raise ValueError("stage and at least one explicit model are required")
    if len({m["id"] for m in models}) != len(models):
        raise ValueError("duplicate selected model")
    models = deepcopy(models)
    for model in models:
        validate_model(model)
    p = deepcopy(DEFAULT_POLICY if policy is None else policy)
    validate_policy(p)
    selected = deepcopy(default_selection(stage) if selection is None else selection)
    if selected["split"] != ("pilot" if stage == "pilot" else "test"):
        raise ValueError("pilot/full must use their respective disjoint source partitions")
    with GridDataset(root, manifest) as data:
        cohort = data.select(selected)
        lengths = data.source_lengths(cohort)
    calls = cohort["prompt_count"] * p["replicates"]
    budgets = []
    for model in models:
        encoding = model["tokenizer"]["encoding"]
        proxy = encoding if encoding in {"cl100k_base", "o200k_base"} else "o200k_base"
        estimate = (lengths[proxy] + 64*cohort["prompt_count"]) * p["replicates"]
        maximum = lengths["max_"+proxy] + 64 + model["sampling"]["max_output_tokens"]
        output = calls * model["sampling"]["max_output_tokens"]
        price = price_usage(model, {"input_tokens": estimate, "output_tokens": output})
        price["status"] = "estimated_using_configured_prices" if price["amount"] is not None else price["status"]
        budgets.append({"model_id": model["id"], "generation_calls": calls,
                        "maximum_generation_attempts": calls*p["max_attempts"],
                        "auxiliary_token_count_requests": calls if model["provider"] in {"openai", "vllm"} else 0,
                        "maximum_auxiliary_token_count_requests": calls*p["max_attempts"]
                            if model["provider"] in {"openai", "vllm"} else 0,
                        "estimated_input_tokens": estimate, "maximum_generated_tokens": output,
                        "token_estimate_method": f"frozen_old_prompt_{proxy}_proxy_plus_64_per_new_chat",
                        "actual_new_message_token_count": None, "chat_count_exact": False,
                        "potential_context_overflow": maximum > model["limits"]["context_tokens"],
                        "cost": price})
    total = calls * len(models)
    blockers = []
    if any(not m["enabled"] for m in models):
        blockers.append("selected_models_disabled")
    if total > p["max_calls"]:
        blockers.append("call_budget_exceeded")
    if any(b["potential_context_overflow"] for b in budgets):
        blockers.append("possible_context_overflow_requires_exact_preflight")
    if p["max_estimated_cost"] is not None:
        currencies = {b["cost"]["currency"] for b in budgets}
        if any(b["cost"]["amount"] is None for b in budgets) or len(currencies) != 1:
            blockers.append("cost_cap_cannot_be_verified")
        elif sum(b["cost"]["amount"] for b in budgets) > p["max_estimated_cost"]:
            blockers.append("estimated_cost_cap_exceeded")
    if stage == "full":
        blockers.append("completed_real_pilot_and_explicit_approval_required")
    manifest = Path(manifest)
    if not manifest.is_absolute():
        manifest = root / manifest
    plan = {"schema_version": 1, "artifact_type": "inference_dry_run_plan",
            "inference_version": INFERENCE_VERSION, "prompt_version": PROMPT_VERSION,
            "parser_version": PARSER_VERSION, "stage": stage,
            "dataset_manifest": str(manifest.resolve().relative_to(root)),
            "cohort": cohort, "models": models, "policy": p, "model_budgets": budgets,
            "generation_calls": total, "contains_model_results": False,
            "actual_generation_calls": 0, "execution_blockers": blockers,
            "estimated_cost_total": None if any(b["cost"]["amount"] is None for b in budgets)
                or len({b["cost"]["currency"] for b in budgets}) != 1
                else sum(b["cost"]["amount"] for b in budgets),
            "source_files_sha256": source_fingerprints(root), "environment": runtime_environment(),
            "assumptions": ["Budget counts every selected presentation for every model; groups are shared.",
                "64 extra tokens per chat is a planning heuristic, not a measured framing bound.",
                "Old source BPE counts are proxies; HF/vLLM actual tokenizer counts remain unknown offline.",
                "Output budget includes reasoning tokens where the provider includes them.",
                "Unknown prices are null, not free; hardware and auxiliary HTTP costs are not estimated."]}
    plan["plan_sha256"] = fingerprint(plan)
    return plan


def validate_plan(plan, root=None):
    if not isinstance(plan, dict):
        raise ValueError("plan must be a mapping")
    value = dict(plan)
    claimed = value.pop("plan_sha256", None)
    if claimed != fingerprint(value) or plan["inference_version"] != INFERENCE_VERSION:
        raise ValueError("plan fingerprint or version changed")
    validate_policy(plan["policy"])
    for model in plan["models"]:
        validate_model(model)
    if root is not None and plan["source_files_sha256"] != source_fingerprints(root):
        raise ValueError("inference source files changed; create a new plan")


def write_plan(plan, path):
    validate_plan(plan)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as output:
        json.dump(plan, output, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False)
        output.write("\n")

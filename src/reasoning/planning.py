"""Offline, capability-bound reasoning experiment plans; no provider calls."""

from __future__ import annotations

from copy import deepcopy
from datetime import date
import hashlib
import json
from pathlib import Path
import tomllib
from urllib.parse import urlsplit

import analysis as base_analysis
from inference import INFERENCE_VERSION
from inference.config import validate_model
from inference.dataset import GridDataset, default_selection, fingerprint, validate_selection
from inference.planning import (DEFAULT_POLICY, price_usage, runtime_environment,
                                source_fingerprints as inference_fingerprints, validate_policy)
from . import REASONING_VERSION
from .parser import PARSER_VERSIONS
from .prompts import PROMPT_VERSIONS, instruction_overhead_utf8_bytes


STRATEGIES = ("direct", "zero_shot_reasoning", "structured_reasoning")
EFFORTS = ("none", "low", "medium", "high")
DEFAULT_SETTINGS = {"schema_version": 1, "output_budget": 2048,
                    "strategies": list(STRATEGIES), "native_efforts": list(EFFORTS), "capabilities": []}


def _keys(value, keys, name):
    if not isinstance(value, dict) or set(value) != set(keys):
        raise ValueError("unknown or missing " + name + " fields")


def _text(value):
    return isinstance(value, str) and bool(value) and value == value.strip() and not any(ord(c) < 32 for c in value)


def _choices(values, choices, name, *, empty=False):
    if (not isinstance(values, list) or (not values and not empty)
            or any(type(item) is not str or item not in choices for item in values)
            or len(set(values)) != len(values)):
        raise ValueError("invalid " + name)


def validate_settings(settings):
    """Strict JSON-shaped TOML settings; evidence is recorded, not fetched."""
    _keys(settings, DEFAULT_SETTINGS, "reasoning settings")
    if type(settings["schema_version"]) is not int or settings["schema_version"] != 1:
        raise ValueError("unsupported reasoning settings schema")
    if type(settings["output_budget"]) is not int or settings["output_budget"] < 16:
        raise ValueError("common output_budget must be an exact integer >=16")
    _choices(settings["strategies"], STRATEGIES, "strategies")
    if "direct" not in settings["strategies"]:
        raise ValueError("the matched direct baseline is required")
    _choices(settings["native_efforts"], EFFORTS, "native_efforts", empty=True)
    if not isinstance(settings["capabilities"], list):
        raise ValueError("capabilities must be a list")
    seen = set()
    for capability in settings["capabilities"]:
        _keys(capability, {"provider", "model_id", "revision", "evidence_url", "verified_on",
                           "supported_efforts", "sampling_restrictions"}, "capability")
        if type(capability["provider"]) is not str or capability["provider"] not in {"openai", "transformers", "vllm"}:
            raise ValueError("unknown capability provider")
        if any(not _text(capability[key]) for key in ("model_id", "revision", "evidence_url", "verified_on")):
            raise ValueError("capability needs exact identity, evidence URL and verification date")
        try:
            url = urlsplit(capability["evidence_url"])
            valid_url = (url.scheme == "https" and bool(url.hostname) and url.username is None
                         and url.password is None and not url.fragment and not url.query)
            url.port
            valid_date = date.fromisoformat(capability["verified_on"]).isoformat() == capability["verified_on"]
        except ValueError as error:
            raise ValueError("malformed capability evidence/date") from error
        if not valid_url or not valid_date:
            raise ValueError("capability evidence requires HTTPS and a canonical YYYY-MM-DD date")
        _choices(capability["supported_efforts"], EFFORTS, "supported_efforts")
        _keys(capability["sampling_restrictions"], capability["supported_efforts"], "effort sampling restriction")
        for rule in capability["sampling_restrictions"].values():
            _keys(rule, {"temperature", "top_p"}, "sampling rule")
            if any(type(value) is not str or value not in {"allowed", "must_be_null"} for value in rule.values()):
                raise ValueError("sampling rules must be allowed or must_be_null")
        identity = tuple(capability[key] for key in ("provider", "model_id", "revision"))
        if identity in seen:
            raise ValueError("duplicate capability identity")
        seen.add(identity)
    # Reject date/time objects, nonfinite numbers and other non-JSON values in
    # programmatic settings just as the bounded TOML loader does.
    try:
        json.dumps(settings, allow_nan=False)
    except (ValueError, TypeError) as error:
        raise ValueError("reasoning settings must have finite JSON values") from error


def load_settings(path):
    data = Path(path).read_bytes()
    if len(data) > 1_048_576:
        raise ValueError("reasoning settings exceed 1 MiB")
    try:
        settings = tomllib.loads(data.decode("utf-8"))
    except (tomllib.TOMLDecodeError, UnicodeDecodeError) as error:
        raise ValueError("invalid reasoning TOML") from error
    validate_settings(settings)
    return settings


def source_fingerprints(root):
    """Bind old inference/analysis and every new reasoning source module."""
    root = Path(root)
    result = inference_fingerprints(root)
    for relative, active in (("src/analysis", Path(base_analysis.__file__).resolve().parent),
                             ("src/reasoning", Path(__file__).resolve().parent)):
        paths = sorted((root / relative).glob("*.py"))
        if not paths:
            raise ValueError("missing source modules: " + relative)
        for path in paths:
            imported = active / path.name
            if not imported.is_file() or imported.read_bytes() != path.read_bytes():
                raise ValueError("installed reasoning/analysis source differs from project: " + str(path))
            result[str(path.relative_to(root))] = hashlib.sha256(path.read_bytes()).hexdigest()
    return result


def _arms(models, settings):
    arms, unavailable, sampling_blockers = [], [], []
    capabilities = {tuple(c[key] for key in ("provider", "model_id", "revision")): c
                    for c in settings["capabilities"]}
    for base in models:
        if settings["output_budget"] > base["limits"]["max_output_tokens"]:
            raise ValueError("common output budget exceeds declared output limit for " + base["id"])
        effective = deepcopy(base)
        effective["sampling"]["max_output_tokens"] = settings["output_budget"]
        validate_model(effective)
        for strategy in settings["strategies"]:
            arms.append({"arm_id": base["id"] + "--" + strategy, "model_id": base["id"],
                "strategy": strategy, "effort": None, "effective_model": deepcopy(effective),
                "prompt_version": PROMPT_VERSIONS[strategy], "parser_version": PARSER_VERSIONS[strategy],
                "capability": None})
        capability = capabilities.get((base["provider"], base["model_id"], base["revision"]))
        for effort in settings["native_efforts"]:
            arm_id = base["id"] + "--provider_effort_" + effort
            reason = ("provider_adapter_does_not_support_native_effort" if base["provider"] != "openai" else
                      "no_verified_native_effort_capability_for_exact_model_revision" if capability is None else
                      "effort_not_declared_supported_for_exact_model_revision" if effort not in capability["supported_efforts"] else None)
            if reason:
                unavailable.append({"arm_id": arm_id, "model_id": base["id"], "strategy": "provider_effort",
                    "effort": effort, "reason": reason, "capability_verified": capability is not None,
                    "generation_calls": 0})
                continue
            native = deepcopy(effective)
            native["reasoning"] = {"mode": "none" if effort == "none" else "thinking",
                                   "effort": effort, "enable_thinking": None}
            validate_model(native)
            incompatibilities = [name for name, rule in capability["sampling_restrictions"][effort].items()
                                 if rule == "must_be_null" and native["sampling"][name] is not None]
            sampling_blockers.extend("native_effort_sampling_incompatible:" + arm_id + ":" + name
                                     for name in incompatibilities)
            arms.append({"arm_id": arm_id, "model_id": base["id"], "strategy": "provider_effort",
                "effort": effort, "effective_model": native, "prompt_version": PROMPT_VERSIONS["provider_effort"],
                "parser_version": PARSER_VERSIONS["provider_effort"], "capability": deepcopy(capability)})
    return arms, unavailable, sampling_blockers


def _budgets(arms, cohort, lengths, policy):
    selection = cohort["selection"]
    per_language = {language: cohort["query_count"] * len(selection["name_controls"]) * len(selection["relations"])
        * sum(language == "tr" or condition in {"canonical", "equivalent"} for condition in selection["conditions"])
        for language in selection["languages"]}
    if sum(per_language.values()) != cohort["prompt_count"] or lengths["prompts"] != cohort["prompt_count"]:
        raise ValueError("planned prompt or source-length denominator mismatch")
    calls = cohort["prompt_count"] * policy["replicates"]
    budgets = []
    for arm in arms:
        model, strategy = arm["effective_model"], arm["strategy"]
        encoding = model["tokenizer"]["encoding"]
        proxy = encoding if encoding in {"cl100k_base", "o200k_base"} else "o200k_base"
        overhead = {language: instruction_overhead_utf8_bytes(language, strategy) for language in per_language}
        if any(type(value) is not int or value < 0 for value in overhead.values()):
            raise ValueError("instruction byte increments must be nonnegative exact integers")
        added = sum(overhead[language] * count for language, count in per_language.items())
        estimated = (lengths[proxy] + 64 * cohort["prompt_count"] + added) * policy["replicates"]
        output = calls * model["sampling"]["max_output_tokens"]
        maximum = lengths["max_" + proxy] + 64 + max(overhead.values()) + model["sampling"]["max_output_tokens"]
        cost = price_usage(model, {"input_tokens": estimated, "output_tokens": output})
        if cost["amount"] is not None:
            cost["status"] = "estimated_using_configured_prices"
        http = model["provider"] in {"openai", "vllm"}
        preflight = {"openai": 1, "vllm": 3, "transformers": 0}[model["provider"]]
        budgets.append({"arm_id": arm["arm_id"], "model_id": arm["model_id"], "strategy": strategy,
            "effort": arm["effort"], "generation_calls": calls,
            "maximum_generation_attempts": calls * policy["max_attempts"],
            "auxiliary_token_count_requests": calls if http else 0,
            "maximum_auxiliary_token_count_requests": calls * policy["max_attempts"] if http else 0,
            "count_input_tokens_method_invocations_per_generation": 2 if http else 1,
            "auxiliary_http_count_assumption": "runner count is reused by generate through the exact-last-payload provider cache" if http else "local tokenizer; no token-count HTTP request",
            "nominal_preflight_http_requests_per_arm_invocation": preflight,
            "maximum_preflight_http_requests_per_arm_invocation": preflight * policy["max_attempts"],
            "preflight_budget_scope": "one invocation only; unused/completed arms make no request; repeated resumes are not bounded globally",
            "estimated_input_tokens": estimated, "maximum_generated_tokens": output,
            "instruction_overhead_utf8_bytes": overhead, "source_prompt_counts_by_language": per_language,
            "instruction_bytes_used_as_proxy_tokens": added * policy["replicates"],
            "token_estimate_method": "frozen_old_prompt_" + proxy + "_plus_64_chat_heuristic_plus_exact_added_instruction_bytes_as_token_proxy",
            "actual_new_message_token_count": None, "chat_count_exact": False,
            "potential_context_overflow": maximum > model["limits"]["context_tokens"], "cost": cost})
    return budgets


def _blockers(models, arms, budgets, policy, stage, sampling_blockers):
    values = list(sampling_blockers)
    total = sum(row["generation_calls"] for row in budgets)
    if any(not model["enabled"] for model in models):
        values.append("selected_models_disabled")
    if total > policy["max_calls"]:
        values.append("call_budget_exceeded")
    if any(row["potential_context_overflow"] for row in budgets):
        values.append("possible_context_overflow_requires_exact_preflight")
    known = all(row["cost"]["amount"] is not None for row in budgets)
    one_currency = len({row["cost"]["currency"] for row in budgets}) == 1
    estimated_cost = sum(row["cost"]["amount"] for row in budgets) if known and one_currency else None
    if policy["max_estimated_cost"] is not None:
        if estimated_cost is None:
            values.append("cost_cap_cannot_be_verified")
        elif estimated_cost > policy["max_estimated_cost"]:
            values.append("estimated_cost_cap_exceeded")
    if stage == "full":
        values.append("completed_real_pilot_and_explicit_approval_required")
    return sorted(set(values)), estimated_cost


def make_plan(root, manifest, models, stage="pilot", selection=None, policy=None, settings=None):
    root = Path(root).resolve()
    if stage not in {"pilot", "full"} or not isinstance(models, list) or not models:
        raise ValueError("pilot/full and at least one explicit model are required")
    models = deepcopy(models)
    for model in models:
        validate_model(model)
    if len({model["id"] for model in models}) != len(models):
        raise ValueError("duplicate selected model")
    settings = deepcopy(load_settings(root / "configs/reasoning.toml") if settings is None else settings)
    validate_settings(settings)
    arms, unavailable, sampling_blockers = _arms(models, settings)
    policy = deepcopy(DEFAULT_POLICY if policy is None else policy)
    validate_policy(policy)
    selected = deepcopy(default_selection(stage) if selection is None else selection)
    if selection is None:
        selected["query_types"] = ["endpoint", "internal", "distance_controlled"]
    validate_selection(selected)
    if selected["split"] != ("pilot" if stage == "pilot" else "test"):
        raise ValueError("reasoning pilot/full must retain disjoint pilot/test source partitions")
    with GridDataset(root, manifest) as data:
        cohort = data.select(selected)
        lengths = data.source_lengths(cohort)
    budgets = _budgets(arms, cohort, lengths, policy)
    blockers, cost = _blockers(models, arms, budgets, policy, stage, sampling_blockers)
    path = Path(manifest)
    path = path if path.is_absolute() else root / path
    plan = {"schema_version": 1, "artifact_type": "reasoning_dry_run_plan", "reasoning_version": REASONING_VERSION,
        "inference_version": INFERENCE_VERSION, "prompt_versions": deepcopy(PROMPT_VERSIONS),
        "parser_versions": deepcopy(PARSER_VERSIONS), "stage": stage,
        "dataset_manifest": str(path.resolve().relative_to(root)), "cohort": cohort, "source_lengths": lengths,
        "models": models, "settings": settings, "settings_sha256": fingerprint(settings), "policy": policy,
        "arms": arms, "unavailable_arms": unavailable, "arm_budgets": budgets,
        "generation_calls": sum(row["generation_calls"] for row in budgets),
        "actual_generation_calls": 0, "contains_model_results": False,
        "execution_blockers": blockers, "estimated_cost_total": cost,
        "source_files_sha256": source_fingerprints(root), "environment": runtime_environment(),
        "assumptions": ["All active arms share the entire frozen query/presentation cohort; no arm-specific dropping.",
            "All arms use the common output cap, including rerun direct; base model records remain unchanged.",
            "Old frozen BPE counts plus 64/chat and added instruction UTF-8 bytes are planning proxies, not measured HF/chat token counts or proven context bounds.",
            "The UTF-8 instruction increment is exact; equating each added byte to a token is only an offline budgeting heuristic.",
            "Provider reasoning tokens can consume the common output cap; adequate budget is not assumed.",
            "Capability evidence is an explicit dated declaration, not a fresh provider availability check.",
            "HTTP count methods are invoked twice per generation, but exact-last-payload caching reuses the first count; preflight counts are scoped per invocation.",
            "Unknown prices/hardware costs are null; no unavailable native effort arm schedules a call."]}
    plan["plan_sha256"] = fingerprint(plan)
    return plan


def validate_plan(plan, root=None):
    if not isinstance(plan, dict):
        raise ValueError("reasoning plan must be a mapping")
    value = dict(plan)
    claimed = value.pop("plan_sha256", None)
    if (claimed != fingerprint(value) or plan.get("schema_version") != 1 or type(plan.get("schema_version")) is not int
            or plan.get("artifact_type") != "reasoning_dry_run_plan" or plan.get("reasoning_version") != REASONING_VERSION
            or plan.get("inference_version") != INFERENCE_VERSION):
        raise ValueError("reasoning plan fingerprint, schema or version changed")
    if plan["prompt_versions"] != PROMPT_VERSIONS or plan["parser_versions"] != PARSER_VERSIONS:
        raise ValueError("reasoning prompt/parser version changed")
    validate_settings(plan["settings"])
    if plan["settings_sha256"] != fingerprint(plan["settings"]):
        raise ValueError("reasoning settings hash mismatch")
    validate_policy(plan["policy"])
    for model in plan["models"]:
        validate_model(model)
    if not plan["models"] or len({model["id"] for model in plan["models"]}) != len(plan["models"]):
        raise ValueError("empty or duplicate selected model")
    if plan["stage"] not in {"pilot", "full"}:
        raise ValueError("invalid reasoning stage")
    validate_selection(plan["cohort"]["selection"])
    if plan["cohort"]["selection"]["split"] != ("pilot" if plan["stage"] == "pilot" else "test"):
        raise ValueError("reasoning split mismatch")
    cohort = dict(plan["cohort"])
    if cohort.pop("cohort_sha256", None) != fingerprint(cohort):
        raise ValueError("cohort hash mismatch")
    arms, unavailable, sampling = _arms(plan["models"], plan["settings"])
    budgets = _budgets(arms, plan["cohort"], plan["source_lengths"], plan["policy"])
    blockers, cost = _blockers(plan["models"], arms, budgets, plan["policy"], plan["stage"], sampling)
    for key, expected in (("arms", arms), ("unavailable_arms", unavailable), ("arm_budgets", budgets),
                          ("execution_blockers", blockers), ("estimated_cost_total", cost),
                          ("generation_calls", sum(row["generation_calls"] for row in budgets)),
                          ("actual_generation_calls", 0), ("contains_model_results", False)):
        if json.dumps(plan[key], sort_keys=True) != json.dumps(expected, sort_keys=True):
            raise ValueError("reasoning derived plan field changed: " + key)
    if root is not None and plan["source_files_sha256"] != source_fingerprints(Path(root)):
        raise ValueError("reasoning/inference/analysis source files changed; create a new plan")


def write_plan(plan, path):
    validate_plan(plan)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as handle:
        json.dump(plan, handle, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False)
        handle.write("\n")

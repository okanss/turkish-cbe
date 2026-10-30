"""Strict stdlib TOML configuration for explicitly disabled training plans."""

from __future__ import annotations

import math
from pathlib import Path, PurePosixPath
import re
import tomllib


ARMS = ("base", "sft_lora", "paired_sft", "sft_consistency", "grpo_correctness", "grpo_consistency")
LORA_TARGETS = ("q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj")
DEPENDENCY_PINS = {"torch": "2.9.0", "transformers": "4.57.1", "peft": "0.17.1", "accelerate": "1.11.0"}
_TOP = {"schema_version", "profile", "seed", "enable_execution", "model_registry", "model_id", "dataset_manifest",
        "arms", "data", "lora", "sft", "grpo", "consistency", "runtime", "budget", "dependencies"}


def _keys(value, names, label, optional=()):
    if not isinstance(value, dict) or not set(names) <= value.keys() or set(value) - set(names) - set(optional):
        raise ValueError("unknown or missing " + label + " fields")


def _text(value):
    return (type(value) is str and bool(value) and value == value.strip()
            and not any(ord(char) < 32 for char in value) and "${" not in value)


def _number(value, label, *, minimum=0, maximum=None, integer=False):
    valid = type(value) is int if integer else type(value) in (int, float)
    try:
        valid = valid and math.isfinite(value) and value >= minimum and (maximum is None or value <= maximum)
    except OverflowError:
        valid = False
    if not valid:
        raise ValueError("invalid finite numeric setting: " + label)


def _list(value, choices, label, *, require_all=False):
    if (not isinstance(value, list) or not value or any(type(item) is not type(choices[0]) or item not in choices for item in value)
            or len(set(value)) != len(value) or (require_all and set(value) != set(choices))):
        raise ValueError("invalid unique choices: " + label)


def _relative(value, label):
    if not _text(value) or "\\" in value:
        raise ValueError("invalid project-relative path: " + label)
    path = PurePosixPath(value)
    if path.is_absolute() or ".." in path.parts or path.as_posix() != value or value in {".", ""}:
        raise ValueError("path must stay within the project: " + label)


def validate_config(config):
    """Validate without mutation; no imports of a model/backend or any downloads."""
    _keys(config, _TOP, "training configuration")
    if type(config["schema_version"]) is not int or config["schema_version"] != 1:
        raise ValueError("unsupported training schema")
    if config["profile"] not in ("pilot", "reference_inspired", "matched_steps"):
        raise ValueError("unregistered training profile")
    _number(config["seed"], "seed", minimum=0, maximum=2**63 - 1, integer=True)
    if type(config["enable_execution"]) is not bool:
        raise ValueError("enable_execution must be an exact Boolean")
    for field in ("model_registry", "dataset_manifest"):
        _relative(config[field], field)
    if not _text(config["model_id"]) or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*", config["model_id"]):
        raise ValueError("model_id must be an explicit registry identifier")
    _list(config["arms"], ARMS, "arms", require_all=True)
    data = config["data"]
    _keys(data, {"training_sizes", "groups_per_size", "training_relations", "heldout_relations", "query_types", "conditions",
                "evaluation_sizes", "evaluation_groups_per_size", "evaluation_split", "grpo_min_n", "grpo_max_instances",
                "grpo_rollouts_per_view"}, "data")
    sizes = (4, 8, 16, 32, 64, 128, 256, 512, 1024)
    _list(data["training_sizes"], sizes[:6], "training_sizes")
    _list(data["evaluation_sizes"], sizes, "evaluation_sizes", require_all=True)
    _list(data["training_relations"], ("comparison_height", "comparison_speed", "comparison_weight"), "training_relations", require_all=True)
    _list(data["heldout_relations"], ("spatial_front", "spatial_right", "spatial_above"), "heldout_relations", require_all=True)
    _list(data["query_types"], ("endpoint", "internal", "distance_controlled"), "query_types", require_all=True)
    _list(data["conditions"], ("canonical", "equivalent", "P1", "P2"), "conditions", require_all=True)
    if data["evaluation_split"] != "test":
        raise ValueError("frozen final evaluation must retain the test split")
    for field in ("groups_per_size", "evaluation_groups_per_size", "grpo_max_instances"):
        _number(data[field], "data." + field, minimum=1, integer=True)
    if data["groups_per_size"] not in (4, 50) or data["evaluation_groups_per_size"] not in (4, 200):
        raise ValueError("group counts must retain the pilot or full protocol design")
    if data["grpo_max_instances"] % 4 or data["grpo_max_instances"] > 600:
        raise ValueError("grpo_max_instances must be a multiple of four, at most 600")
    _number(data["grpo_min_n"], "grpo_min_n", minimum=4, maximum=128, integer=True)
    if data["grpo_min_n"] not in data["training_sizes"]:
        raise ValueError("GRPO minimum n must be a selected training size")
    if type(data["grpo_rollouts_per_view"]) is not int or data["grpo_rollouts_per_view"] != 4:
        raise ValueError("this protocol requires four rollouts per presentation")
    lora = config["lora"]
    _keys(lora, {"rank", "alpha", "dropout", "target_modules"}, "LoRA")
    _number(lora["rank"], "LoRA rank", minimum=1, integer=True)
    _number(lora["alpha"], "LoRA alpha", minimum=1, integer=True)
    _number(lora["dropout"], "LoRA dropout", minimum=0, maximum=0.999999)
    _list(lora["target_modules"], LORA_TARGETS, "LoRA targets", require_all=True)
    sft = config["sft"]
    _keys(sft, {"learning_rate", "epochs", "effective_batch_size", "max_steps"}, "SFT")
    _number(sft["learning_rate"], "SFT learning rate", minimum=1e-12)
    _number(sft["epochs"], "SFT epochs", minimum=1, integer=True)
    _number(sft["effective_batch_size"], "SFT effective batch size", minimum=1, integer=True)
    _number(sft["max_steps"], "SFT step override", minimum=0, integer=True)
    grpo = config["grpo"]
    _keys(grpo, {"learning_rate", "beta", "max_steps", "advantage_mode", "temperature", "top_p"}, "GRPO")
    _number(grpo["learning_rate"], "GRPO learning rate", minimum=1e-12)
    _number(grpo["beta"], "GRPO beta", minimum=0, maximum=0)
    _number(grpo["max_steps"], "GRPO steps", minimum=1, integer=True)
    _number(grpo["temperature"], "GRPO temperature", minimum=1e-12)
    _number(grpo["top_p"], "GRPO top_p", minimum=1, maximum=1)
    if grpo["advantage_mode"] not in ("prompt_normalized", "instance_centered"):
        raise ValueError("unknown GRPO advantage mode")
    consistency = config["consistency"]
    _keys(consistency, {"js_weight", "guard", "agreement_bonus_weight"}, "consistency")
    _number(consistency["js_weight"], "JS weight", minimum=0)
    _number(consistency["agreement_bonus_weight"], "agreement bonus weight", minimum=0, maximum=0.5)
    if consistency["guard"] != "detached_both_gold_probability_above_half":
        raise ValueError("unregistered JS guard")
    runtime = config["runtime"]
    _keys(runtime, {"device", "dtype", "local_files_only", "weights_path", "weights_manifest", "minimum_free_gpu_memory_gib"}, "runtime")
    if runtime["device"] != "cuda" or runtime["dtype"] != "bfloat16" or runtime["local_files_only"] is not True:
        raise ValueError("execution requires the validated local-only CUDA bfloat16 backend")
    for field in ("weights_path", "weights_manifest"):
        value = runtime[field]
        if type(value) is not str or (value and (not _text(value) or not Path(value).is_absolute())):
            raise ValueError(field + " must be empty or an explicit absolute local path")
    _number(runtime["minimum_free_gpu_memory_gib"], "minimum free GPU GiB", minimum=1e-12)
    budget = config["budget"]
    _keys(budget, {"max_optimizer_steps", "max_wall_seconds", "output_tokens", "max_length", "optimizer", "scheduler", "warmup_ratio"},
          "budget", optional={"cost_per_gpu_hour", "currency"})
    for field in ("max_optimizer_steps", "max_wall_seconds", "output_tokens", "max_length"):
        _number(budget[field], "budget." + field, minimum=1, integer=True)
    if budget["output_tokens"] >= budget["max_length"]:
        raise ValueError("output cap must leave room for training input tokens")
    if (sft["max_steps"] > budget["max_optimizer_steps"] or grpo["max_steps"] > budget["max_optimizer_steps"]):
        raise ValueError("arm step override exceeds the declared optimizer-step budget")
    if budget["optimizer"] != "adamw" or budget["scheduler"] != "cosine":
        raise ValueError("this backend implements AdamW and cosine scheduling only")
    _number(budget["warmup_ratio"], "warmup ratio", minimum=0, maximum=1)
    cost, currency = budget.get("cost_per_gpu_hour"), budget.get("currency")
    if cost is not None:
        _number(cost, "GPU hourly cost", minimum=0)
        if type(currency) is not str or not re.fullmatch(r"[A-Z]{3}", currency):
            raise ValueError("a declared GPU price requires an ISO currency code")
    elif currency is not None:
        raise ValueError("currency without a measured/declared hourly price is ambiguous")
    _keys(config["dependencies"], DEPENDENCY_PINS, "dependencies")
    for name, version in config["dependencies"].items():
        if not _text(version) or not re.fullmatch(r"\d+\.\d+\.\d+(?:\+[A-Za-z0-9.]+)?", version):
            raise ValueError("dependencies need exact version pins: " + name)
    if config["profile"] == "matched_steps" and not (sft["max_steps"] == grpo["max_steps"] == budget["max_optimizer_steps"]):
        raise ValueError("matched_steps must explicitly match every trained arm's optimizer-step cap")


def load_config(path):
    data = Path(path).read_bytes()
    if len(data) > 1_048_576:
        raise ValueError("training configuration exceeds 1 MiB")
    try:
        config = tomllib.loads(data.decode("utf-8"))
    except (tomllib.TOMLDecodeError, UnicodeDecodeError) as error:
        raise ValueError("invalid training TOML") from error
    validate_config(config)
    config["budget"].setdefault("cost_per_gpu_hour", None)
    config["budget"].setdefault("currency", None)
    return config

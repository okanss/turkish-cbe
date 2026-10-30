"""Strict, explicit model registry. Loading never downloads a model or reads a key.

PyYAML is optional until ``load_models`` is called. YAML aliases and implicit
timestamps are deliberately excluded, so the registry has a plain JSON shape.
"""

from __future__ import annotations

from copy import deepcopy
from datetime import date
import math
from pathlib import Path
import re
from typing import Any
from urllib.parse import urlsplit


class ConfigError(ValueError):
    """A registry cannot be interpreted without guessing or losing a setting."""


_MODEL_KEYS = {
    "id", "provider", "family", "parameters_billion", "model_id", "revision",
    "tokenizer", "serving", "sampling", "limits", "reasoning", "seed",
    "pricing", "enabled",
}
_NESTED_KEYS = {
    "tokenizer": {"id", "revision", "encoding", "chat_template_sha256"},
    "serving": {"base_url", "api_key_env", "timeout_seconds", "device", "dtype",
                "trust_remote_code", "local_files_only", "server_revision", "chat_template_date"},
    "sampling": {"temperature", "top_p", "max_output_tokens"},
    "limits": {"context_tokens", "max_output_tokens"},
    "reasoning": {"mode", "effort", "enable_thinking"},
    "pricing": {"currency", "input_per_million", "output_per_million",
                "cached_input_per_million", "as_of", "source"},
}
_COMMIT = re.compile(r"[0-9a-f]{40}\Z")
_HASH = re.compile(r"[0-9a-f]{64}\Z")
_IDENTIFIER = re.compile(r"[a-zA-Z0-9][a-zA-Z0-9_.-]*\Z")


def _mapping(value: Any, keys: set[str], name: str) -> dict:
    if not isinstance(value, dict) or any(not isinstance(k, str) for k in value):
        raise ConfigError(f"{name} must be a string-keyed mapping")
    missing, unknown = keys - value.keys(), value.keys() - keys
    if missing or unknown:
        raise ConfigError(f"{name}: missing={sorted(missing)}, unknown={sorted(unknown)}")
    return value


def _text(value: Any, name: str, *, nullable: bool = False) -> None:
    if value is None and nullable:
        return
    if not isinstance(value, str) or not value or value != value.strip():
        raise ConfigError(f"{name} must be a nonempty, trimmed string")
    if any(ord(c) < 32 for c in value) or "${" in value:
        raise ConfigError(f"{name} contains controls or unsupported interpolation")


def _number(value: Any, name: str, low: float, high: float | None = None,
            *, nullable: bool = False, integer: bool = False) -> None:
    if value is None and nullable:
        return
    valid_type = type(value) is int if integer else type(value) in (int, float)
    try:
        finite = valid_type and math.isfinite(value)
    except OverflowError:
        finite = False
    if not finite or value < low:
        raise ConfigError(f"{name} must be finite and >= {low}")
    if high is not None and value > high:
        raise ConfigError(f"{name} must be <= {high}")


def _url(value: Any, name: str, *, nullable: bool = True) -> None:
    _text(value, name, nullable=nullable)
    if value is None:
        return
    try:
        parsed = urlsplit(value)
    except ValueError as exc:
        raise ConfigError(f"{name}: malformed URL") from exc
    if (parsed.scheme not in {"http", "https"} or not parsed.hostname
            or parsed.username is not None or parsed.password is not None
            or parsed.query or parsed.fragment):
        raise ConfigError(f"{name} must be an HTTP(S) URL without embedded credentials/query")
    try:
        parsed.port
    except ValueError as exc:
        raise ConfigError(f"{name}: invalid port") from exc


def _iso_date(value: Any, name: str) -> None:
    _text(value, name, nullable=True)
    if value is not None:
        try:
            if date.fromisoformat(value).isoformat() != value:
                raise ValueError("date is not canonical")
        except ValueError as exc:
            raise ConfigError(f"{name} must be YYYY-MM-DD") from exc


def validate_model(spec: dict) -> None:
    """Validate all fields, including disabled examples; never alter ``spec``.

    Disabled examples may leave an immutable revision/template pin unselected.
    Setting enabled=true requires every executable identity to be pinned. Actual
    model availability and template capabilities are checked by each adapter.
    """
    _mapping(spec, _MODEL_KEYS, "model")
    for name in ("id", "provider", "family", "model_id"):
        _text(spec[name], name)
    if not _IDENTIFIER.fullmatch(spec["id"]):
        raise ConfigError("id must be a portable registry identifier")
    if spec["provider"] not in {"openai", "transformers", "vllm"}:
        raise ConfigError("provider must be openai, transformers, or vllm")
    if type(spec["enabled"]) is not bool:
        raise ConfigError("enabled must be a boolean")
    _text(spec["revision"], "revision", nullable=not spec["enabled"])
    _number(spec["parameters_billion"], "parameters_billion", 1e-12, nullable=True)
    if spec["provider"] == "openai" and spec["parameters_billion"] is not None:
        raise ConfigError("OpenAI parameter count is undisclosed; use null")
    for key, keys in _NESTED_KEYS.items():
        _mapping(spec[key], keys, key)

    tokenizer = spec["tokenizer"]
    _text(tokenizer["id"], "tokenizer.id")
    _text(tokenizer["revision"], "tokenizer.revision", nullable=not spec["enabled"])
    _text(tokenizer["encoding"], "tokenizer.encoding", nullable=True)
    template_hash = tokenizer["chat_template_sha256"]
    if template_hash is not None and (not isinstance(template_hash, str)
                                      or not _HASH.fullmatch(template_hash)):
        raise ConfigError("tokenizer.chat_template_sha256 must be a 64-character SHA256")
    if spec["provider"] in {"transformers", "vllm"}:
        for name, revision in (("revision", spec["revision"]),
                               ("tokenizer.revision", tokenizer["revision"])):
            if revision is not None and not _COMMIT.fullmatch(revision):
                raise ConfigError(f"{name} must be an immutable 40-character commit")
        if spec["enabled"] and template_hash is None:
            raise ConfigError("enabled open-weight models require a chat template hash")
        if tokenizer["encoding"] is not None:
            raise ConfigError("open-weight tokenizers use pinned tokenizer files, not a tiktoken encoding")
    elif spec["revision"] is not None and spec["revision"] != spec["model_id"]:
        raise ConfigError("OpenAI revision must equal the exact requested model snapshot")
    if spec["provider"] == "openai":
        if spec["enabled"] and not re.search(r"-\d{4}-\d{2}-\d{2}\Z", spec["model_id"]):
            raise ConfigError("enabled OpenAI models require a dated snapshot, not a moving alias")
        if tokenizer["encoding"] is None:
            raise ConfigError("OpenAI tokenizer encoding must be explicitly selected")

    serving = spec["serving"]
    _url(serving["base_url"], "serving.base_url")
    _text(serving["api_key_env"], "serving.api_key_env", nullable=True)
    if serving["api_key_env"] is not None and not re.fullmatch(
            r"[A-Za-z_][A-Za-z0-9_]*", serving["api_key_env"]):
        raise ConfigError("api_key_env is an environment variable name, never a literal key")
    _number(serving["timeout_seconds"], "serving.timeout_seconds", 0.001)
    _text(serving["device"], "serving.device", nullable=True)
    if serving["dtype"] is not None and (not isinstance(serving["dtype"], str)
            or serving["dtype"] not in {"auto", "float16", "bfloat16", "float32"}):
        raise ConfigError("serving.dtype is not supported")
    for name in ("trust_remote_code", "local_files_only"):
        if type(serving[name]) is not bool:
            raise ConfigError(f"serving.{name} must be a boolean")
    _text(serving["server_revision"], "serving.server_revision", nullable=True)
    _iso_date(serving["chat_template_date"], "serving.chat_template_date")
    if spec["provider"] == "openai" and serving["chat_template_date"] is not None:
        raise ConfigError("OpenAI has no configurable local chat template date; use null")
    if spec["provider"] == "transformers":
        if serving["base_url"] is not None or serving["api_key_env"] is not None:
            raise ConfigError("transformers is local; HTTP endpoint/key fields must be null")
        if serving["device"] is None or serving["dtype"] is None:
            raise ConfigError("transformers requires explicit device and dtype")
        if serving["server_revision"] is not None:
            raise ConfigError("transformers has no server_revision; runtime package versions are recorded")
    else:
        if serving["base_url"] is None:
            raise ConfigError("HTTP providers require a base_url")
        if serving["device"] is not None or serving["dtype"] is not None:
            raise ConfigError("HTTP serving device/dtype are not request parameters; set null")
        if serving["trust_remote_code"] or not serving["local_files_only"]:
            raise ConfigError("HTTP adapters cannot apply trust_remote_code/local_files_only overrides")
        if spec["provider"] == "openai" and serving["server_revision"] is not None:
            raise ConfigError("OpenAI server implementation revision is not exposed; use null")
        if spec["provider"] == "vllm" and spec["enabled"]:
            if serving["server_revision"] is None or serving["server_revision"].lower() in {
                    "main", "latest", "unselected", "unknown"}:
                raise ConfigError("enabled vllm requires an explicit serving implementation version")

    limits, sampling = spec["limits"], spec["sampling"]
    for name, value in limits.items():
        _number(value, f"limits.{name}", 1, integer=True)
    _number(sampling["max_output_tokens"], "sampling.max_output_tokens",
            16 if spec["provider"] == "openai" else 1, integer=True)
    if sampling["max_output_tokens"] > limits["max_output_tokens"]:
        raise ConfigError("sampling output budget exceeds declared model output limit")
    if limits["max_output_tokens"] > limits["context_tokens"]:
        raise ConfigError("output limit exceeds context limit")
    _number(sampling["temperature"], "sampling.temperature", 0, 2, nullable=True)
    _number(sampling["top_p"], "sampling.top_p", 1e-12, 1, nullable=True)
    _number(spec["seed"], "seed", 0, 2**63 - 1, nullable=True, integer=True)
    if spec["provider"] == "openai" and spec["seed"] is not None:
        raise ConfigError("OpenAI Responses does not support seed; use null")

    reasoning = spec["reasoning"]
    if not isinstance(reasoning["mode"], str) or reasoning["mode"] not in {"none", "provider_default", "thinking"}:
        raise ConfigError("reasoning.mode must be none, provider_default, or thinking")
    if reasoning["effort"] is not None and (not isinstance(reasoning["effort"], str)
            or reasoning["effort"] not in {"none", "minimal", "low", "medium", "high", "xhigh", "max"}):
        raise ConfigError("unsupported reasoning.effort")
    thinking = reasoning["enable_thinking"]
    if thinking is not None and type(thinking) is not bool:
        raise ConfigError("reasoning.enable_thinking must be null or a boolean")
    if reasoning["mode"] == "provider_default":
        if reasoning["effort"] is not None or thinking is not None:
            raise ConfigError("provider_default must not override effort or enable_thinking")
    if spec["provider"] == "openai":
        if thinking is not None:
            raise ConfigError("OpenAI uses reasoning.effort, not enable_thinking")
        if reasoning["mode"] == "none" and reasoning["effort"] not in {None, "none"}:
            raise ConfigError("reasoning none cannot request a positive effort")
        if reasoning["mode"] == "thinking" and reasoning["effort"] in {None, "none"}:
            raise ConfigError("OpenAI thinking requires an explicit positive effort")
    else:
        if reasoning["effort"] is not None:
            raise ConfigError("open-weight adapters do not support an effort setting")
        if reasoning["mode"] != "provider_default" and thinking != (reasoning["mode"] == "thinking"):
            raise ConfigError("open-weight reasoning mode must agree with enable_thinking")

    pricing = spec["pricing"]
    _text(pricing["currency"], "pricing.currency", nullable=True)
    if pricing["currency"] is not None and not re.fullmatch(r"[A-Z]{3}", pricing["currency"]):
        raise ConfigError("pricing.currency must be a three-letter currency code")
    for name in ("input_per_million", "output_per_million", "cached_input_per_million"):
        _number(pricing[name], f"pricing.{name}", 0, nullable=True)
    _iso_date(pricing["as_of"], "pricing.as_of")
    _url(pricing["source"], "pricing.source")
    rates = [pricing[k] for k in ("input_per_million", "output_per_million", "cached_input_per_million")]
    if any(value is not None for value in rates) and any(
            pricing[name] is None for name in ("currency", "as_of", "source")):
        raise ConfigError("non-null prices require currency, date, and source")
    if (pricing["cached_input_per_million"] is not None and pricing["input_per_million"] is not None
            and pricing["cached_input_per_million"] > pricing["input_per_million"]):
        raise ConfigError("cached input price exceeds ordinary input price")


def _validate_registry(registry: Any) -> None:
    _mapping(registry, {"schema_version", "models"}, "registry")
    if type(registry["schema_version"]) is not int or registry["schema_version"] != 1:
        raise ConfigError("unsupported models schema_version")
    if not isinstance(registry["models"], list) or not registry["models"]:
        raise ConfigError("models must be a nonempty list")
    seen: set[str] = set()
    for spec in registry["models"]:
        validate_model(spec)
        if spec["id"] in seen:
            raise ConfigError(f"duplicate model id: {spec['id']}")
        seen.add(spec["id"])


def load_models(path: str | Path) -> dict:
    """Load a bounded YAML document without aliases, duplicate keys or object tags."""
    try:
        import yaml
    except ImportError as exc:
        raise RuntimeError("models.yaml requires the optional PyYAML dependency") from exc
    path = Path(path)
    raw = path.read_bytes()
    if len(raw) > 1_048_576:
        raise ConfigError("model registry exceeds 1 MiB")
    try:
        source = raw.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise ConfigError("model registry must be UTF-8") from exc

    class StrictLoader(yaml.SafeLoader):
        def compose_node(self, parent, index):
            if self.check_event(yaml.AliasEvent):
                raise ConfigError("YAML aliases are not supported")
            return super().compose_node(parent, index)

        def construct_mapping(self, node, deep=False):
            result = {}
            for key_node, value_node in node.value:
                key = self.construct_object(key_node, deep=deep)
                if not isinstance(key, str):
                    raise ConfigError("YAML mapping keys must be strings")
                if key in result:
                    raise ConfigError(f"duplicate YAML key: {key}")
                result[key] = self.construct_object(value_node, deep=deep)
            return result

    # Do not mutate SafeLoader's global resolver tables.
    StrictLoader.yaml_implicit_resolvers = {
        key: [(tag, pattern) for tag, pattern in rules
              if tag not in {"tag:yaml.org,2002:timestamp", "tag:yaml.org,2002:bool"}]
        for key, rules in yaml.SafeLoader.yaml_implicit_resolvers.items()
    }
    StrictLoader.add_implicit_resolver(
        "tag:yaml.org,2002:bool", re.compile(r"^(?:true|false)$"), ["t", "f"])
    try:
        registry = yaml.load(source, Loader=StrictLoader)
    except (yaml.YAMLError, RecursionError) as exc:
        raise ConfigError("invalid or excessively nested model YAML") from exc
    _validate_registry(registry)
    return registry


def select_models(registry: dict, ids: list[str]) -> list[dict]:
    """Select exactly the requested IDs, in order, without enabling or substituting."""
    _validate_registry(registry)
    if not isinstance(ids, list) or not ids or any(not isinstance(x, str) for x in ids):
        raise ConfigError("select at least one explicit model id")
    if len(ids) != len(set(ids)):
        raise ConfigError("duplicate model selection")
    indexed = {spec["id"]: spec for spec in registry["models"]}
    unknown = [identifier for identifier in ids if identifier not in indexed]
    if unknown:
        raise ConfigError(f"unknown model IDs: {unknown}")
    return [deepcopy(indexed[identifier]) for identifier in ids]

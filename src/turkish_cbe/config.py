"""Strict, dependency-free validation of complete TOML experiment plans."""

from dataclasses import asdict, dataclass
from fractions import Fraction
from pathlib import Path
import tomllib
from typing import Any


class ConfigError(ValueError):
    """A plan is malformed or outside the implemented planning scope."""


def _keys(value: Any, expected: str, name: str) -> dict:
    if not isinstance(value, dict) or set(value) != set(expected.split()):
        raise ConfigError(f"{name}: exact keys required: {expected}")
    return value


def _integer(value: Any, name: str, minimum: int = 0) -> int:
    if type(value) is not int or value < minimum:
        raise ConfigError(f"{name}: integer >= {minimum} required")
    return value


def _strings(value: Any, name: str) -> tuple[str, ...]:
    if (not isinstance(value, list) or not value
            or any(not isinstance(x, str) or not x.strip() for x in value)
            or len(set(value)) != len(value)):
        raise ConfigError(f"{name}: nonempty unique string array required")
    return tuple(value)


def _numbers(value: Any, name: str, minimum: int) -> tuple[int, ...]:
    if not isinstance(value, list) or not value:
        raise ConfigError(f"{name}: nonempty integer array required")
    values = tuple(_integer(x, name, minimum) for x in value)
    if len(set(values)) != len(values):
        raise ConfigError(f"{name}: duplicates are not allowed")
    return values


@dataclass(frozen=True)
class Presentation:
    language: str
    variant: str


@dataclass(frozen=True)
class ModelSpec:
    slot: str
    provider: str
    model_id: str
    revision: str
    tokenizer_revision: str


@dataclass(frozen=True)
class RepeatPlan:
    fraction: float
    total_observations: int
    seed: int


@dataclass(frozen=True)
class DecodingPlan:
    temperature: float
    top_p: float
    max_output_tokens: int
    reasoning_effort: str
    seed_policy: str


@dataclass(frozen=True)
class AnalysisPlan:
    bootstrap_resamples: int
    seed: int
    alpha: float
    multiplicity: str


@dataclass(frozen=True)
class TrainingPlan:
    enabled: bool


@dataclass(frozen=True)
class ExperimentConfig:
    schema_version: int
    protocol_id: str
    stage: str
    status: str
    task: str
    relation: str
    seeds: tuple[int, ...]
    sizes: tuple[int, ...]
    instances_per_size_per_seed: int
    languages: tuple[str, ...]
    presentations: tuple[Presentation, ...]
    models: tuple[ModelSpec, ...]
    prompting: tuple[str, ...]
    repeats: RepeatPlan
    decoding: DecodingPlan
    analysis: AnalysisPlan
    training: TrainingPlan
    pending_gates: tuple[str, ...]

    def to_dict(self) -> dict:
        return asdict(self)

    def budget(self) -> dict[str, int]:
        """Counts requests, not experimental outcomes or monetary costs."""
        blocks = len(self.sizes) * len(self.seeds)
        instances = blocks * self.instances_per_size_per_seed
        per_cell = Fraction(str(self.repeats.fraction)) * (
            self.instances_per_size_per_seed // 4
        )
        selected = int(per_cell) * 4 * blocks
        arms = len(self.presentations) * len(self.models) * len(self.prompting)
        first = instances * arms
        extra = selected * arms * (self.repeats.total_observations - 1)
        return {"latent_instances": instances, "first_observation_requests": first,
                "repeat_selected_instances": selected,
                "additional_repeat_requests": extra,
                "total_planned_requests": first + extra}


def parse_config(raw: dict) -> ExperimentConfig:
    _keys(raw, "schema_version protocol_id stage status task relation seeds sizes "
          "instances_per_size_per_seed languages presentations models prompting "
          "repeats decoding analysis training pending_gates", "root")
    if type(raw["schema_version"]) is not int or raw["schema_version"] != 1:
        raise ConfigError("schema_version: only integer 1 is supported")
    for name, expected in (("protocol_id", "tr_cbe_v0_1"),
                           ("status", "planning_only"),
                           ("task", "ordering_endpoint"),
                           ("relation", "front_behind")):
        if raw[name] != expected:
            raise ConfigError(f"{name}: only {expected!r} is supported")
    if raw["stage"] not in ("pilot", "baseline", "advanced"):
        raise ConfigError("stage: expected pilot, baseline or advanced")
    seeds = _numbers(raw["seeds"], "seeds", 0)
    sizes = _numbers(raw["sizes"], "sizes", 4)
    if tuple(sorted(sizes)) != sizes:
        raise ConfigError("sizes: must be strictly increasing")
    count = _integer(raw["instances_per_size_per_seed"],
                     "instances_per_size_per_seed", 4)
    if count % 4:
        raise ConfigError("instances_per_size_per_seed: must balance four query cells")
    languages = _strings(raw["languages"], "languages")
    if set(languages) != {"tr", "en"}:
        raise ConfigError("languages: this paired design requires tr and en")
    if not isinstance(raw["presentations"], list) or not raw["presentations"]:
        raise ConfigError("presentations: nonempty table array required")
    presentations = []
    allowed = {"canonical", "converse", "premise_shuffle", "syntax", "morph_joint"}
    for entry in raw["presentations"]:
        _keys(entry, "language variant", "presentations")
        if any(not isinstance(x, str) for x in entry.values()):
            raise ConfigError("presentations: string fields required")
        if (entry["language"] not in languages or entry["variant"] not in allowed
                or (entry["language"] == "en"
                    and entry["variant"] not in {"canonical", "converse"})):
            raise ConfigError("presentations: unsupported language/variant combination")
        presentations.append(Presentation(**entry))
    if len(set(presentations)) != len(presentations):
        raise ConfigError("presentations: duplicate language/variant")
    for language in languages:
        for variant in ("canonical", "converse"):
            if Presentation(language, variant) not in presentations:
                raise ConfigError("presentations: canonical/converse required in each language")
    if not isinstance(raw["models"], list) or not raw["models"]:
        raise ConfigError("models: nonempty table array required")
    models = []
    for entry in raw["models"]:
        _keys(entry, "slot provider model_id revision tokenizer_revision", "models")
        if any(not isinstance(x, str) or not x.strip() for x in entry.values()):
            raise ConfigError("models: nonempty strings required for all fields")
        models.append(ModelSpec(**entry))
    if len({m.slot for m in models}) != len(models):
        raise ConfigError("models: slots must be unique")
    conditions = _strings(raw["prompting"], "prompting")
    if set(conditions) - {"direct", "reasoning"} or "direct" not in conditions:
        raise ConfigError("prompting: direct required; only direct/reasoning supported")
    if "reasoning" in conditions and raw["stage"] != "advanced":
        raise ConfigError("prompting: reasoning belongs to advanced stage")
    repeat = _keys(raw["repeats"], "fraction total_observations seed", "repeats")
    fraction = repeat["fraction"]
    if type(fraction) not in (int, float) or not 0 < fraction <= 1:
        raise ConfigError("repeats.fraction: required in (0, 1]")
    if (Fraction(str(fraction)) * (count // 4)).denominator != 1:
        raise ConfigError("repeats.fraction: selection must be integral in every query cell")
    _integer(repeat["total_observations"], "repeats.total_observations", 2)
    _integer(repeat["seed"], "repeats.seed")
    decode = _keys(raw["decoding"], "temperature top_p max_output_tokens "
                   "reasoning_effort seed_policy", "decoding")
    for key, lower, upper in (("temperature", 0, 2), ("top_p", 0, 1)):
        val = decode[key]
        if type(val) not in (int, float) or not lower <= val <= upper:
            raise ConfigError(f"decoding.{key}: invalid numeric range")
    if decode["top_p"] == 0:
        raise ConfigError("decoding.top_p: must be positive")
    _integer(decode["max_output_tokens"], "decoding.max_output_tokens", 1)
    if decode["reasoning_effort"] != "none":
        raise ConfigError("decoding.reasoning_effort: provider effort is not implemented")
    if decode["seed_policy"] != "record_if_supported":
        raise ConfigError("decoding.seed_policy: expected record_if_supported")
    analysis = _keys(raw["analysis"], "bootstrap_resamples seed alpha multiplicity", "analysis")
    _integer(analysis["bootstrap_resamples"], "analysis.bootstrap_resamples", 1)
    _integer(analysis["seed"], "analysis.seed")
    if (type(analysis["alpha"]) not in (int, float)
            or not 0 < analysis["alpha"] < 1 or analysis["multiplicity"] != "holm"):
        raise ConfigError("analysis: alpha in (0,1) and multiplicity=holm required")
    training = _keys(raw["training"], "enabled", "training")
    if training["enabled"] is not False:
        raise ConfigError("training.enabled: must remain false in this scaffold")
    gates = _strings(raw["pending_gates"], "pending_gates")
    return ExperimentConfig(
        schema_version=1, protocol_id=raw["protocol_id"], stage=raw["stage"],
        status=raw["status"], task=raw["task"], relation=raw["relation"],
        seeds=seeds, sizes=sizes, instances_per_size_per_seed=count,
        languages=languages, presentations=tuple(presentations), models=tuple(models),
        prompting=conditions, repeats=RepeatPlan(**repeat),
        decoding=DecodingPlan(**decode), analysis=AnalysisPlan(**analysis),
        training=TrainingPlan(enabled=False),
        pending_gates=gates,
    )


def loads_config(data: bytes) -> ExperimentConfig:
    try:
        return parse_config(tomllib.loads(data.decode("utf-8")))
    except (tomllib.TOMLDecodeError, UnicodeDecodeError) as exc:
        raise ConfigError(f"invalid TOML: {exc}") from exc


def load_config(path: str | Path) -> ExperimentConfig:
    return loads_config(Path(path).read_bytes())

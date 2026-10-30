"""Small, explicit statistical procedures for the registered CBE contrasts.

Only the Python standard library is required.  Undefined estimates and intervals
are represented by ``None`` together with a status, never by zero.  Cluster
bootstrap confidence intervals concern independent latent orders, rather than
the number of surface presentations generated from an order.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
import json
import math
import random
from statistics import NormalDist
from typing import Any


def _probability(value: Any, name: str, *, open_interval: bool = False) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be a finite number")
    try:
        result = float(value)
    except OverflowError as exc:
        raise ValueError(f"{name} must be a finite number") from exc
    if not math.isfinite(result):
        raise ValueError(f"{name} must be finite")
    permitted = 0 < result < 1 if open_interval else 0 <= result <= 1
    if not permitted:
        raise ValueError(f"{name} is outside its permitted probability interval")
    return result


def _integer(value: Any, name: str, *, minimum: int = 0) -> int:
    if type(value) is not int or value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}")
    return value


def wilson_interval(successes: int, total: int, confidence: float = 0.95) -> dict:
    """Return a two-sided Wilson score interval for a binomial proportion.

    The binomial independence assumption is not asserted by this function.
    Callers must label an interval descriptive if multiple rows share a latent
    order.  An empty sample has no interval or point estimate.
    """
    _integer(successes, "successes")
    _integer(total, "total")
    confidence = _probability(confidence, "confidence", open_interval=True)
    if successes > total:
        raise ValueError("successes cannot exceed total")
    result = {"estimate": None, "low": None, "high": None,
              "successes": successes, "total": total, "confidence": confidence,
              "method": "wilson", "status": "undefined_empty_sample"}
    if total == 0:
        result["reason"] = "No independent binomial observations are available."
        return result
    z = -NormalDist().inv_cdf((1 - confidence) / 2)
    proportion = successes / total
    squared = z * z
    inverse_total = 1 / total
    divisor = 1 + squared * inverse_total
    midpoint = (proportion + squared * inverse_total / 2) / divisor
    half_width = z * math.sqrt(
        (proportion * (1 - proportion) + squared * inverse_total / 4) * inverse_total
    ) / divisor
    result.update(estimate=proportion, low=max(0.0, midpoint - half_width),
                  high=min(1.0, midpoint + half_width), status="ok")
    return result


def _booleans(values: Any, name: str) -> list[bool]:
    if not isinstance(values, Sequence) or isinstance(values, (str, bytes)):
        raise ValueError(f"{name} must be a sequence of exact booleans")
    result = list(values)
    if any(type(value) is not bool for value in result):
        raise ValueError(f"{name} must contain exact booleans only")
    return result


def _stable_json(value: Any) -> str:
    try:
        return json.dumps(value, sort_keys=True, ensure_ascii=False,
                          allow_nan=False, separators=(",", ":"))
    except (TypeError, ValueError) as exc:
        raise ValueError("Rows and identifiers must contain finite JSON values") from exc


def _identifier(value: Any) -> str:
    # Type tags prevent the Python equality True == 1 from merging clusters.
    if type(value) not in (str, int, float) or (isinstance(value, str) and not value):
        raise ValueError("Cluster and stratum identifiers must be nonempty strings or numbers")
    if isinstance(value, float) and not math.isfinite(value):
        raise ValueError("Cluster and stratum identifiers must be finite")
    return f"{type(value).__name__}:{_stable_json(value)}"


def mcnemar(left: Sequence[bool], right: Sequence[bool], *,
            clusters: Sequence | None = None, method: str = "auto") -> dict:
    """Test paired correctness with exact or continuity-corrected McNemar.

    ``b`` counts correct-left/wrong-right pairs, and ``c`` the converse.
    ``auto`` uses the two-sided Binomial(discordant, .5) test through 25
    discordant pairs, and a continuity-corrected chi-square approximation above
    that.  Repeated cluster identifiers invalidate the ordinary McNemar p-value;
    they must be analysed with a cluster contrast instead.
    """
    if method not in ("auto", "exact", "asymptotic"):
        raise ValueError("method must be auto, exact, or asymptotic")
    left = _booleans(left, "left")
    right = _booleans(right, "right")
    if len(left) != len(right):
        raise ValueError("Paired sequences must have equal lengths")
    b = sum(a and not z for a, z in zip(left, right))
    c = sum(not a and z for a, z in zip(left, right))
    discordant = b + c
    selected = ("exact" if discordant <= 25 else "asymptotic") if method == "auto" else method
    result = {"n": len(left), "b": b, "c": c, "discordant": discordant,
              "both_correct": sum(a and z for a, z in zip(left, right)),
              "both_wrong": sum(not a and not z for a, z in zip(left, right)),
              "p_value": None, "statistic": None, "method": selected,
              "status": "ok"}
    if clusters is not None:
        if not isinstance(clusters, Sequence) or isinstance(clusters, (str, bytes)):
            raise ValueError("clusters must be a sequence of identifiers")
        if len(clusters) != len(left):
            raise ValueError("clusters must have the same length as the paired data")
        identifiers = [_identifier(value) for value in clusters]
        if len(set(identifiers)) != len(identifiers):
            result.update(status="undefined_dependent_clusters",
                          reason="Repeated latent clusters violate ordinary McNemar independence.")
            return result
    if not left:
        result.update(status="undefined_empty_sample", reason="No matched observations are available.")
        return result
    if discordant == 0:
        result.update(p_value=1.0, statistic=0.0)
        return result
    if selected == "exact":
        # Sum exact integer binomial coefficients before the single division.
        # This avoids accumulating rounding errors in the small-sample test.
        limit = min(b, c)
        coefficient = 1
        lower_tail = 1
        for k in range(1, limit + 1):
            coefficient = coefficient * (discordant - k + 1) // k
            lower_tail += coefficient
        result["p_value"] = min(1.0, (2 * lower_tail) / (1 << discordant))
    else:
        statistic = max(0, abs(b - c) - 1) ** 2 / discordant
        result.update(statistic=statistic,
                      p_value=math.erfc(math.sqrt(statistic / 2)))
    return result


def holm_adjust(p_values: Mapping[str, float | None]) -> dict[str, float | None]:
    """Apply Holm's step-down family-wise correction, retaining missing tests.

    A ``None`` test remains ``None`` and still occupies a slot in the declared
    family.  This conservative choice prevents missing model/contrast results
    from silently making the remaining tests easier to declare significant.
    """
    if not isinstance(p_values, Mapping):
        raise ValueError("p_values must be a mapping")
    if any(not isinstance(key, str) or not key for key in p_values):
        raise ValueError("Each hypothesis must have a nonempty string identifier")
    valid = [(key, _probability(value, f"p_values[{key}]"))
             for key, value in p_values.items() if value is not None]
    valid.sort(key=lambda item: (item[1], item[0]))
    adjusted: dict[str, float | None] = {key: None for key in p_values}
    family_size = len(p_values)
    previous = 0.0
    for rank, (key, value) in enumerate(valid):
        previous = max(previous, min(1.0, (family_size - rank) * value))
        adjusted[key] = previous
    return adjusted


def _metric_values(value: Any) -> dict[str, float | None]:
    if not isinstance(value, Mapping):
        raise ValueError("statistic must return a mapping of named scalar metrics")
    output: dict[str, float | None] = {}
    for key, item in value.items():
        if not isinstance(key, str) or not key:
            raise ValueError("Metric names must be nonempty strings")
        if item is None:
            output[key] = None
        elif isinstance(item, bool) or not isinstance(item, (int, float)):
            raise ValueError("Metric values must be finite numeric scalars or None")
        else:
            try:
                numeric = float(item)
            except OverflowError as exc:
                raise ValueError("Metric values must be finite numeric scalars or None") from exc
            if not math.isfinite(numeric):
                raise ValueError("Metric values must be finite numeric scalars or None")
            output[key] = numeric
    return output


def _percentile(sorted_values: list[float], fraction: float) -> float:
    index = (len(sorted_values) - 1) * fraction
    lower = math.floor(index)
    upper = math.ceil(index)
    return sorted_values[lower] + (index - lower) * (sorted_values[upper] - sorted_values[lower])


def bootstrap(rows: Sequence[Mapping], statistic: Callable, *, seed: int = 20261024,
              repeats: int = 2000, cluster_key: str = "cluster_id",
              strata_keys: Sequence[str] = ("n",), confidence: float = 0.95) -> dict:
    """Percentile bootstrap of all observations belonging to each sampled order.

    Each stratum independently resamples its original number of clusters with
    replacement.  All rows of a drawn cluster move together, including its
    languages and presentations; a cluster drawn twice contributes its rows
    twice.  Cluster identities may not span strata.  Canonical ordering makes
    a fixed seed independent of the input row ordering.

    No interval is emitted if any resampling stratum has fewer than two distinct
    clusters, the original estimate is undefined, or *any* bootstrap replicate is undefined.
    ``defined_replicates`` always exposes that last denominator explicitly.
    """
    _integer(seed, "seed")
    _integer(repeats, "repeats", minimum=1)
    confidence = _probability(confidence, "confidence", open_interval=True)
    if not isinstance(cluster_key, str) or not cluster_key:
        raise ValueError("cluster_key must be a nonempty string")
    if (not isinstance(strata_keys, Sequence) or isinstance(strata_keys, (str, bytes))
            or any(not isinstance(key, str) or not key for key in strata_keys)
            or len(set(strata_keys)) != len(strata_keys)):
        raise ValueError("strata_keys must contain unique nonempty field names")
    if cluster_key in strata_keys:
        raise ValueError("cluster_key cannot also be a stratum key")
    if not isinstance(rows, Sequence) or isinstance(rows, (str, bytes)):
        raise ValueError("rows must be a sequence of mappings")
    if not callable(statistic):
        raise ValueError("statistic must be callable")

    grouped: dict[str, dict[str, list[dict]]] = {}
    cluster_strata: dict[str, str] = {}
    for row in rows:
        if not isinstance(row, Mapping) or any(not isinstance(key, str) for key in row):
            raise ValueError("Each row must be a mapping with string field names")
        if cluster_key not in row or any(key not in row for key in strata_keys):
            raise ValueError("Each row must include its cluster and all stratum keys")
        cluster = _identifier(row[cluster_key])
        stratum = _stable_json([_identifier(row[key]) for key in strata_keys])
        if cluster in cluster_strata and cluster_strata[cluster] != stratum:
            raise ValueError("A latent cluster cannot span bootstrap strata")
        cluster_strata[cluster] = stratum
        copied = dict(row)
        _stable_json(copied)
        grouped.setdefault(stratum, {}).setdefault(cluster, []).append(copied)
    ordered = []
    for stratum in sorted(grouped):
        clusters = grouped[stratum]
        ordered.append([sorted(clusters[key], key=_stable_json) for key in sorted(clusters)])
    canonical_rows = [row for stratum in ordered for cluster in stratum for row in cluster]
    estimate = _metric_values(statistic(canonical_rows))
    metadata = {"repeats": repeats, "confidence": confidence, "seed": seed,
                "method": "stratified_cluster_percentile", "independent_clusters": len(cluster_strata),
                "strata": len(ordered)}
    output = {key: {"estimate": value, "low": None, "high": None,
                    "defined_replicates": 0, **metadata, "status": "insufficient_clusters"}
              for key, value in estimate.items()}
    if not ordered or any(len(stratum) < 2 for stratum in ordered):
        for interval in output.values():
            interval["reason"] = ("At least two independent latent clusters are required in every "
                                  "resampling stratum; within-stratum variation is unestimable.")
        return output
    values: dict[str, list[float]] = {key: [] for key in estimate}
    rng = random.Random(seed)
    for _ in range(repeats):
        sampled = [row for stratum in ordered for _ in range(len(stratum))
                   for row in rng.choice(stratum)]
        current = _metric_values(statistic(sampled))
        if current.keys() != estimate.keys():
            raise ValueError("statistic must return the same metric names in every replicate")
        for key, value in current.items():
            if value is not None:
                values[key].append(value)
    tail = (1 - confidence) / 2
    for key, interval in output.items():
        defined = len(values[key])
        interval["defined_replicates"] = defined
        if interval["estimate"] is None:
            interval.update(status="undefined_estimate", reason="The original-sample metric is undefined.")
        elif defined < repeats:
            interval.update(status="partial_undefined", reason="One or more bootstrap replicates are undefined.")
        else:
            values[key].sort()
            interval.update(low=_percentile(values[key], tail),
                            high=_percentile(values[key], 1 - tail), status="ok")
    return output

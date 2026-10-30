"""Descriptive lexical associations with explicit dependence and design checks.

Repeated latent presentations do not create independent vocabulary evidence.
Consequently neither procedure emits ordinary independent-observation p-values,
standard errors, or confidence intervals.  NumPy is imported only for the
optional controlled least-squares diagnostic.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Mapping, Sequence
import math
from typing import Any


def _number(value: Any) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    try:
        result = float(value)
    except OverflowError:
        return None
    return result if math.isfinite(result) else None


def _ranks(values: list[float]) -> list[float]:
    order = sorted(range(len(values)), key=values.__getitem__)
    ranks = [0.0] * len(values)
    start = 0
    while start < len(order):
        end = start + 1
        while end < len(order) and values[order[end]] == values[order[start]]:
            end += 1
        average = (start + end - 1) / 2 + 1
        for position in range(start, end):
            ranks[order[position]] = average
        start = end
    return ranks


def spearman(x, y) -> dict:
    """Return Spearman's average-tie rank correlation without an iid p-value.

    Invalid, missing, misaligned, and constant inputs produce a null coefficient
    and an explicit status.  Missing pairs are counted, never silently dropped.
    ``n`` is the supplied aligned pair count; ``n_complete`` counts finite pairs.
    """
    result = {"coefficient": None, "n": 0, "n_complete": 0,
              "status": "invalid_input", "p_value": None,
              "inference": "descriptive_no_independent_observation_p_value",
              "tie_method": "average_rank"}
    if (not isinstance(x, Sequence) or not isinstance(y, Sequence)
            or isinstance(x, (str, bytes)) or isinstance(y, (str, bytes))):
        result["reason"] = "Inputs must be finite numeric sequences."
        return result
    if len(x) != len(y):
        result.update(status="length_mismatch", reason="Paired inputs have different lengths.")
        return result
    result["n"] = len(x)
    converted = [(_number(a), _number(b)) for a, b in zip(x, y)]
    result["n_complete"] = sum(a is not None and b is not None for a, b in converted)
    if result["n_complete"] != len(x):
        result.update(status="invalid_values", reason="Missing/nonfinite/non-numeric pairs were not dropped.")
        return result
    if len(x) < 2:
        result.update(status="insufficient_pairs", reason="At least two finite pairs are required.")
        return result
    xs = _ranks([a for a, _ in converted])
    ys = _ranks([b for _, b in converted])
    average = (len(xs) + 1) / 2
    xx = math.fsum((value - average) ** 2 for value in xs)
    yy = math.fsum((value - average) ** 2 for value in ys)
    if xx == 0 or yy == 0:
        result.update(status="constant_input", reason="A constant variable has no rank correlation.")
        return result
    covariance = math.fsum((a - average) * (b - average) for a, b in zip(xs, ys))
    result.update(coefficient=max(-1.0, min(1.0, covariance / math.sqrt(xx * yy))), status="ok")
    return result


def _text(value):
    return isinstance(value, str) and bool(value.strip())


def fit_controlled(rows, *, outcome, predictor="log_frequency", interaction=False) -> dict:
    """Fit a prespecified exploratory OLS diagnostic, with no naive inference.

    Main design: intercept, predictor, log2(n), log(prompt_tokens), reference-
    coded model/family factors, and the after-query indicator.  The interaction
    design instead uses model, vocabulary, and model-by-vocabulary fixed effects
    plus the two numeric controls.  Frequency/family/direction are omitted from
    that design because vocabulary fixed effects absorb vocabulary attributes.

    All supplied rows remain accounted for. Any missing required field/value,
    rank deficiency, or insufficient residual observations yields no fit.
    Sparse vocabulary counts are exposed even when descriptive coefficients
    can be estimated; repeated observations never justify iid uncertainty.
    """
    if not _text(outcome) or not _text(predictor) or type(interaction) is not bool:
        raise ValueError("outcome/predictor must be nonblank names and interaction an exact bool")
    reserved = {"Intercept", "log2_n", "log_prompt_tokens", "model_id", "n", "relation_family",
                "vocabulary_id", "query_relation", "prompt_tokens", "cluster_id"}
    if predictor in reserved or predictor.startswith(("model[", "family[", "vocabulary[")):
        raise ValueError("predictor name collides with a design control or identifier")
    if outcome in {"model_id", "n", "relation_family", "vocabulary_id", "query_relation",
                   "prompt_tokens", "cluster_id"} or outcome == predictor:
        raise ValueError("outcome must be distinct from predictors and design identifiers")
    if (not isinstance(rows, Sequence) or isinstance(rows, (str, bytes))
            or any(not isinstance(row, Mapping) for row in rows)):
        raise ValueError("rows must be a sequence of mappings")
    result = {"status": "no_rows", "coefficients": None, "n_rows": len(rows), "n_complete": 0,
              "n_parameters": 0, "rank": None, "n_vocabulary": 0, "n_vocabulary_cells": 0, "n_latent": 0,
              "n_models": 0, "outcome": outcome, "requested_predictor": predictor,
              "interaction": interaction, "model_design": None, "warnings": [],
              "p_values": None, "standard_errors": None, "confidence_intervals": None,
              "inference": "descriptive_only_no_iid_inference", "residual_sum_squares": None,
              "r_squared": None, "condition_number": None}
    if not rows:
        return result
    required = {"model_id", "n", "relation_family", "vocabulary_id", "query_relation",
                "prompt_tokens", "cluster_id", outcome}
    independent_units = any("independent_vocabulary_id" in row for row in rows)
    if independent_units:
        required.add("independent_vocabulary_id")
    if not interaction:
        required.add(predictor)
    missing = Counter()
    converted = []
    for row in rows:
        invalid = []
        for field in required:
            value = row.get(field)
            if field in {"model_id", "relation_family", "vocabulary_id", "independent_vocabulary_id", "cluster_id"}:
                valid = _text(value)
            elif field == "query_relation":
                valid = value in ("before", "after")
            elif field == "n":
                valid = type(value) is int and value > 0
            elif field == "prompt_tokens":
                valid = _number(value) is not None and value > 0
            else:
                valid = _number(value) is not None
            if not valid:
                invalid.append(field)
        missing.update(invalid)
        if not invalid:
            converted.append(dict(row))
    result["n_complete"] = len(converted)
    unit_key = "independent_vocabulary_id" if independent_units else "vocabulary_id"
    result["n_vocabulary"] = len({row[unit_key] for row in rows if _text(row.get(unit_key))})
    result["n_vocabulary_cells"] = len({row["vocabulary_id"] for row in rows if _text(row.get("vocabulary_id"))})
    result["n_latent"] = len({row["cluster_id"] for row in rows if _text(row.get("cluster_id"))})
    result["n_models"] = len({row["model_id"] for row in rows if _text(row.get("model_id"))})
    if result["n_vocabulary"] < 30:
        result["warnings"].append("fewer_than_30_independent_vocabulary_units")
    result["warnings"].append("latent_presentations_and_vocabulary_repetitions_are_dependent")
    if missing:
        result.update(status="missing_values", missing_fields=dict(sorted(missing.items())))
        result["warnings"].append("no_rows_silently_dropped_no_fit")
        return result
    rows = converted
    models = sorted({row["model_id"] for row in rows})
    families = sorted({row["relation_family"] for row in rows})
    vocabulary = sorted({row["vocabulary_id"] for row in rows})
    # One vocabulary identifier denotes one relation/direction lexical unit.
    for identifier in vocabulary:
        attributes = {(row["relation_family"], row["query_relation"])
                      for row in rows if row["vocabulary_id"] == identifier}
        if len(attributes) != 1:
            result.update(status="inconsistent_vocabulary_metadata")
            result["warnings"].append("vocabulary_id_spans_family_or_query_direction")
            return result
        if independent_units and len({row["independent_vocabulary_id"] for row in rows
                                      if row["vocabulary_id"] == identifier}) != 1:
            result.update(status="inconsistent_vocabulary_metadata")
            result["warnings"].append("vocabulary_design_cell_spans_independent_lexical_units")
            return result
    columns = ["Intercept", "log2_n", "log_prompt_tokens"]
    if not interaction:
        columns.append(predictor)
    columns += [f"model[{name}]" for name in models[1:]]
    if interaction:
        columns += [f"vocabulary[{name}]" for name in vocabulary[1:]]
        columns += [f"model[{model}]:vocabulary[{unit}]" for model in models[1:] for unit in vocabulary[1:]]
    else:
        columns += [f"family[{name}]" for name in families[1:]]
        columns.append("query_relation[after]")
    design = {"columns": columns, "intercept": True, "model_reference": models[0],
              "family_reference": None if interaction else families[0],
              "vocabulary_reference": vocabulary[0] if interaction else None,
              "query_relation_reference": None if interaction else "before",
              "frequency_predictor_included": not interaction,
              "vocabulary_design_unit": "vocabulary_id", "independent_vocabulary_unit": unit_key,
              "predictor": None if interaction else predictor,
              "numeric_controls": ["log2_n", "log_prompt_tokens"],
              "formula": ("outcome ~ model * vocabulary + log2(n) + log(prompt_tokens)" if interaction else
                          "outcome ~ predictor + model + family + query_relation + log2(n) + log(prompt_tokens)")}
    result.update(model_design=design, n_parameters=len(columns))
    if interaction:
        result["warnings"].append("frequency_family_direction_absorbed_by_vocabulary_fixed_effects")
        present = {(row["model_id"], row["vocabulary_id"]) for row in rows}
        design["missing_model_vocabulary_cells"] = [[model, unit] for model in models for unit in vocabulary
                                                     if (model, unit) not in present]
    matrix = []
    for row in rows:
        values = [1.0, math.log2(row["n"]), math.log(float(row["prompt_tokens"]))]
        if not interaction:
            values.append(float(row[predictor]))
        values += [float(row["model_id"] == name) for name in models[1:]]
        if interaction:
            values += [float(row["vocabulary_id"] == name) for name in vocabulary[1:]]
            values += [float(row["model_id"] == model and row["vocabulary_id"] == unit)
                       for model in models[1:] for unit in vocabulary[1:]]
        else:
            values += [float(row["relation_family"] == name) for name in families[1:]]
            values.append(float(row["query_relation"] == "after"))
        matrix.append(values)
    try:
        import numpy as np
    except ImportError:
        result.update(status="missing_optional_numpy")
        return result
    x = np.asarray(matrix, dtype=float)
    y = np.asarray([float(row[outcome]) for row in rows], dtype=float)
    singular = np.linalg.svd(x, compute_uv=False)
    tolerance = float(singular[0] * max(x.shape) * np.finfo(float).eps) if len(singular) else 0.0
    rank = int(np.count_nonzero(singular > tolerance))
    condition = float(singular[0] / singular[-1]) if len(singular) and singular[-1] > 0 else None
    result.update(rank=rank, rank_tolerance=tolerance,
                  singular_values=[float(value) for value in singular],
                  condition_number=condition if condition is None or math.isfinite(condition) else None)
    result["constant_columns"] = [columns[index] for index in range(1, x.shape[1])
                                  if np.ptp(x[:, index]) == 0]
    if rank < len(columns):
        result["warnings"].append("rank_deficient_requested_design_no_automatic_column_dropping")
    if len(rows) <= len(columns):
        result.update(status="insufficient_rows")
        return result
    if rank < len(columns):
        result.update(status="rank_deficient")
        return result
    if np.ptp(y) == 0:
        result.update(status="constant_outcome")
        return result
    fitted, _, _, _ = np.linalg.lstsq(x, y, rcond=None)
    residual = y - x @ fitted
    rss = float(residual @ residual)
    centered = y - np.mean(y)
    total = float(centered @ centered)
    result.update(status="ok", coefficients=dict(zip(columns, map(float, fitted))),
                  residual_sum_squares=rss, r_squared=1 - rss / total)
    return result

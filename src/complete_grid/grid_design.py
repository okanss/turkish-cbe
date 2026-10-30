"""Complete, factorized bilingual design without duplicating prompt strings.

Every source query crosses three name controls, nine relation vocabularies and
nine language/presentation conditions. Comparison F measures query complexity;
it is deliberately separate from the same-query equivalence contrasts A--E.
The original 0.4.0 data bank and its narrower contrast validator stay unchanged.
"""

from __future__ import annotations

from collections import Counter
from itertools import product
import re

from data_generation.generator import DEFAULT_DISTANCES, valid_distances
from renderers.turkish import RELATIONS
from experiments.dataset_builder import DEFAULT_PLAN, DatasetPlan, balanced_indices
from experiments.names import NAME_CONTROLS


GRID_VERSION = "0.4.1"
RELATION_IDS = tuple(RELATIONS)
PROMPT_CONDITIONS = (
    ("en", "canonical"), ("en", "equivalent"),
    ("tr", "canonical"), ("tr", "equivalent"),
    ("tr", "P1"), ("tr", "P2"), ("tr", "P3"),
    ("tr", "P4"), ("tr", "P5"),
)
QUERY_TYPES = ("endpoint", "internal", "distance_controlled")


def iter_designs():
    """Yield (name_control, relation_id, language, condition), in stable order."""
    for control, relation, (language, condition) in product(
            NAME_CONTROLS, RELATION_IDS, PROMPT_CONDITIONS):
        yield control, relation, language, condition


def comparison_designs(query_type: str):
    """Yield (arm, left language/condition, right language/condition).

    A--E use the same source query on both sides. F is included only for an
    interior query: its left side must be the endpoint query of the same order.
    Each name/relation cell has nine same-query comparisons and, for interior
    queries, four additional endpoint-to-interior comparisons.
    """
    if type(query_type) is not str or query_type not in QUERY_TYPES:
        raise ValueError("unknown query type")
    yield "A", ("en", "canonical"), ("en", "equivalent")
    yield "B", ("tr", "canonical"), ("tr", "equivalent")
    yield "C", ("en", "canonical"), ("tr", "canonical")
    yield "D", ("en", "equivalent"), ("tr", "equivalent")
    for condition in ("P1", "P2", "P3", "P4", "P5"):
        yield "E", ("tr", "canonical"), ("tr", condition)
    if query_type != "endpoint":
        for language in ("en", "tr"):
            for condition in ("canonical", "equivalent"):
                yield "F", (language, condition), (language, condition)


_TEXT_FIELDS = (
    "latent_instance_id", "query_instance_id", "split", "query_type",
    "query_subject", "query_object", "query_relation", "query_direction",
    "language", "condition", "name_control", "relation_id",
    "presentation_plan_sha256",
)
_SAME_QUERY = (
    "latent_instance_id", "query_instance_id", "split", "n", "query_type",
    "query_distance", "query_subject", "query_object", "query_relation",
    "query_direction", "gold_label", "name_control", "relation_id",
)
_SAME_ORDER_PRESENTATION = (
    "latent_instance_id", "split", "n", "language", "condition",
    "name_control", "relation_id", "presentation_plan_sha256",
)


def _metadata(row: dict, side: str) -> None:
    if not isinstance(row, dict):
        raise ValueError(f"{side} must be a dictionary")
    for key in (*_TEXT_FIELDS, "n", "query_distance", "gold_label"):
        if key not in row:
            raise ValueError(f"{side} missing {key}")
    for key in _TEXT_FIELDS:
        if not isinstance(row[key], str) or not row[key] or row[key] != row[key].strip():
            raise ValueError(f"{side}.{key} must be a nonempty trimmed string")
    if row["split"] not in {"training", "development", "pilot", "test"}:
        raise ValueError(f"{side}.split is unknown")
    if type(row["n"]) is not int or not 4 <= row["n"] <= 1024:
        raise ValueError(f"{side}.n is outside [4, 1024]")
    distance = row["query_distance"]
    if type(distance) is not int or not 1 <= distance < row["n"]:
        raise ValueError(f"{side}.query_distance is outside [1, n-1]")
    if row["query_type"] not in QUERY_TYPES:
        raise ValueError(f"{side}.query_type is unknown")
    if row["query_type"] == "endpoint" and distance != row["n"] - 1:
        raise ValueError(f"{side} endpoint distance must equal n-1")
    if row["query_type"] != "endpoint" and distance > row["n"] - 3:
        raise ValueError(f"{side} interior query includes an endpoint")
    if row["name_control"] not in NAME_CONTROLS or row["relation_id"] not in RELATION_IDS:
        raise ValueError(f"{side} has an unknown name control or relation")
    if (row["language"], row["condition"]) not in PROMPT_CONDITIONS:
        raise ValueError(f"{side} has an unknown language/condition combination")
    if re.fullmatch(r"[0-9a-f]{64}", row["presentation_plan_sha256"]) is None:
        raise ValueError(f"{side}.presentation_plan_sha256 must be a SHA-256 hex digest")
    if row["query_subject"] == row["query_object"]:
        raise ValueError(f"{side} query requires distinct entities")
    if row["query_relation"] not in {"before", "after"}:
        raise ValueError(f"{side}.query_relation is unknown")
    if row["query_direction"] not in {"forward", "reverse"}:
        raise ValueError(f"{side}.query_direction is unknown")
    if type(row["gold_label"]) is not bool:
        raise ValueError(f"{side}.gold_label must be boolean")
    expected_gold = ((row["query_relation"] == "before")
                     == (row["query_direction"] == "forward"))
    if row["gold_label"] is not expected_gold:
        raise ValueError(f"{side} gold contradicts query direction/relation")


def _same(left: dict, right: dict, keys: tuple[str, ...]) -> None:
    for key in keys:
        if left[key] != right[key]:
            raise ValueError("paired metadata differs: " + key)


def validate_grid_pair(left: dict, right: dict, arm: str) -> None:
    """Reject invalid expanded-grid comparisons without interpreting stored text.

    Surface parsing and independent exact oracles must validate prompts before
    this metadata check. E is intentionally valid at every query type, name
    control and relation in this expanded design. F may have different query
    targets and gold answers, but uses an identical presentation plan.
    """
    if type(arm) is not str or arm not in tuple("ABCDEF"):
        raise ValueError("unknown comparison arm")
    _metadata(left, "left")
    _metadata(right, "right")
    if arm == "F":
        _same(left, right, _SAME_ORDER_PRESENTATION)
        if (left["condition"] not in {"canonical", "equivalent"}
                or left["query_type"] != "endpoint"
                or right["query_type"] not in {"internal", "distance_controlled"}
                or left["query_instance_id"] == right["query_instance_id"]):
            raise ValueError("F requires endpoint versus distinct interior query")
        return
    _same(left, right, _SAME_QUERY)
    transition = (left["language"], left["condition"], right["language"], right["condition"])
    expected = {
        "A": ("en", "canonical", "en", "equivalent"),
        "B": ("tr", "canonical", "tr", "equivalent"),
        "C": ("en", "canonical", "tr", "canonical"),
        "D": ("en", "equivalent", "tr", "equivalent"),
    }
    if arm == "E":
        if (transition[:3] != ("tr", "canonical", "tr")
                or right["condition"] not in {"P1", "P2", "P3", "P4", "P5"}):
            raise ValueError("E requires Turkish canonical versus P1--P5")
    elif transition != expected[arm]:
        raise ValueError("incorrect language/condition transition for " + arm)
    if arm in {"C", "D"}:
        _same(left, right, ("presentation_plan_sha256",))


def expected_counts(plan: DatasetPlan = DEFAULT_PLAN) -> dict:
    """Compute complete counts from the plan, without generating prompt text.

    Counts are numbers of materializable prompt identities and comparisons,
    even when the corresponding strings are stored by reusable components.
    Four-cell counts describe source queries before their factor expansion.
    """
    if not isinstance(plan, DatasetPlan):
        raise ValueError("plan must be a DatasetPlan")
    plan.validate()
    result = {"grid_version": GRID_VERSION, "designs_per_query": len(tuple(iter_designs())),
              "name_controls": list(NAME_CONTROLS), "relation_ids": list(RELATION_IDS),
              "prompt_conditions": [list(value) for value in PROMPT_CONDITIONS],
              "splits": {}}
    totals = {"orders": 0, "queries": 0, "prompts": 0, "comparison_cells": 0,
              "cells_by_arm": {arm: 0 for arm in "ABCDEF"}}
    factor_cells = len(NAME_CONTROLS) * len(RELATION_IDS)
    for spec in plan.splits:
        orders = len(spec.sizes) * spec.count
        queries = 0
        blocks, distributions, by_n = {}, {}, {}
        for n in spec.sizes:
            indices = balanced_indices(spec.count, n)
            four_cells = Counter(
                ("forward" if i % 4 < 2 else "reverse") + "|" + str(i % 2 == 0)
                for i in indices)
            query_designs = [("endpoint", None), ("internal", None)] + [
                ("distance_controlled", d) for d in valid_distances(n, DEFAULT_DISTANCES)]
            by_n[str(n)] = {"orders": spec.count,
                            "queries": len(query_designs) * spec.count,
                            "prompts": len(query_designs) * spec.count * result["designs_per_query"]}
            for query_type, distance in query_designs:
                key = f"{n}|{query_type}|{distance}"
                blocks[key] = spec.count
                distributions[key] = dict(sorted(four_cells.items()))
                queries += spec.count
        cells_by_arm = {arm: queries * factor_cells * (5 if arm == "E" else 1)
                        for arm in "ABCDE"}
        cells_by_arm["F"] = (queries - orders) * factor_cells * 4
        summary = {"orders": orders, "queries": queries,
                   "prompts": queries * result["designs_per_query"],
                   "comparison_cells": sum(cells_by_arm.values()),
                   "cells_by_arm": cells_by_arm, "query_blocks": blocks,
                   "query_answer_cells": distributions, "by_n": by_n}
        result["splits"][spec.name] = summary
        for key in ("orders", "queries", "prompts", "comparison_cells"):
            totals[key] += summary[key]
        for arm in "ABCDEF":
            totals["cells_by_arm"][arm] += cells_by_arm[arm]
    result["totals"] = totals
    return result

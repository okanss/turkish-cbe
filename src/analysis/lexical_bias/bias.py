"""Response-share diagnostics that retain planned gold and invalid coverage."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Mapping
import json

from analysis.metrics import summarize_pairs
from analysis.statistics import wilson_interval


MODEL_COLUMNS = ("model_id", "model_family", "model_revision", "model_spec_sha256")
DESIGN_COLUMNS = ("n", "split", "name_control", "relation_id", "relation_family",
                  "query_type", "query_distance", "condition", "language")
PROTECTED_COLUMNS = MODEL_COLUMNS + DESIGN_COLUMNS
_CONDITIONS = {"canonical", "equivalent", "P1", "P2", "P3", "P4", "P5"}


def _json(value):
    return json.dumps(value, sort_keys=True, ensure_ascii=False,
                      separators=(",", ":"), allow_nan=False)


def _groups(rows, columns):
    groups = defaultdict(list)
    for row in rows:
        groups[tuple(row[key] for key in columns)].append(row)
    return [(dict(zip(columns, key)), groups[key]) for key in sorted(groups, key=_json)]


def _validated(observations):
    if not isinstance(observations, list):
        raise ValueError("observations must be the planned observation list")
    required = set(PROTECTED_COLUMNS) | {"gold_label", "prediction", "query_relation",
                                         "query_direction", "cluster_id", "query_id"}
    seen = set()
    for row in observations:
        if not isinstance(row, Mapping) or not required.issubset(row):
            raise ValueError("observation is missing protected design or query fields")
        for name in required - {"gold_label", "prediction", "n", "query_distance", "query_id"}:
            if not isinstance(row[name], str) or not row[name].strip():
                raise ValueError("protected identifiers and conditions must be nonblank strings")
        # Run observations carry the integer grid query identifier; synthetic
        # fixtures may use string identifiers. Both stay exact and nonblank.
        query_id = row["query_id"]
        if type(query_id) is int:
            if query_id < 0:
                raise ValueError("query_id must be a nonnegative exact integer or nonblank string")
        elif not isinstance(query_id, str) or not query_id.strip():
            raise ValueError("query_id must be a nonnegative exact integer or nonblank string")
        if type(row["n"]) is not int or row["n"] < 2:
            raise ValueError("n must be an exact integer >=2")
        if type(row["query_distance"]) is not int or not 1 <= row["query_distance"] < row["n"]:
            raise ValueError("query_distance must be a valid exact integer")
        if type(row["gold_label"]) is not bool or (row["prediction"] is not None and type(row["prediction"]) is not bool):
            raise ValueError("gold must be exact Boolean and prediction Boolean or None")
        if row["query_relation"] not in {"before", "after"} or row["query_direction"] not in {"forward", "reverse"}:
            raise ValueError("lexical query relation and latent query direction must remain distinct axes")
        if row["condition"] not in _CONDITIONS:
            raise ValueError("unregistered presentation condition")
        identity = tuple(row[key] for key in PROTECTED_COLUMNS) + (row["cluster_id"], row["query_id"])
        if identity in seen:
            raise ValueError("duplicate planned observation; repeated runs are not independent evidence")
        seen.add(identity)
    return observations


def _summary(rows):
    total = len(rows)
    valid = [row for row in rows if row["prediction"] is not None]
    yes = sum(row["prediction"] is True for row in valid)
    no = len(valid) - yes
    gold_yes = sum(row["gold_label"] for row in rows)
    gold_yes_valid = sum(row["gold_label"] for row in valid)
    valid_clusters = [row["cluster_id"] for row in valid]
    dependent = len(valid_clusters) != len(set(valid_clusters))
    interval = wilson_interval(yes, len(valid))
    cells = {f"{direction}|{label}": 0 for direction in ("forward", "reverse") for label in ("YES", "NO")}
    for row in rows:
        cells[f"{row['query_direction']}|{'YES' if row['gold_label'] else 'NO'}"] += 1
    q = yes / len(valid) if valid else None
    tendency = ("undefined_no_valid_answers" if q is None else "excessive_yes" if q >= 0.90
                else "excessive_no" if q <= 0.10 else "within_predeclared_thresholds")
    return {"n_planned": total, "n_valid": len(valid), "n_invalid_or_missing": total - len(valid),
            "n_missing": sum(row.get("response_status") == "missing" for row in rows),
            "n_clusters": len({row["cluster_id"] for row in rows}),
            "n_valid_clusters": len(set(valid_clusters)),
            "predicted_yes": yes, "predicted_no": no, "q": q,
            "valid_coverage": len(valid) / total,
            "invalid_response_rate": (total - len(valid)) / total,
            "predicted_yes_over_all_planned": yes / total,
            "all_planned_accuracy": sum(row["prediction"] is not None and row["prediction"] == row["gold_label"]
                                         for row in rows) / total,
            "gold_yes_planned": gold_yes, "gold_no_planned": total - gold_yes,
            "gold_yes_rate_planned": gold_yes / total,
            "gold_balance_status": "balanced" if gold_yes * 2 == total else "unbalanced",
            "gold_yes_valid": gold_yes_valid, "gold_no_valid": len(valid) - gold_yes_valid,
            "gold_yes_rate_valid": gold_yes_valid / len(valid) if valid else None,
            "query_direction_gold_cells": _json(cells),
            "q_wilson_low": interval["low"], "q_wilson_high": interval["high"],
            "q_wilson_status": interval["status"],
            "q_wilson_interpretation": "descriptive_dependent_latent_orders" if dependent else "binomial_independence_assumed",
            "dependent_valid_observations": dependent,
            "excessive_response_tendency": None if q is None else q >= 0.90 or q <= 0.10,
            "response_tendency": tendency, "tendency_threshold_low": 0.10, "tendency_threshold_high": 0.90,
            "threshold_is_significance_test": False}


def summarize_bias(observations) -> list[dict]:
    """Return protected overall and separate lexical/latent direction profiles.

    q and Wilson counts use valid answers only. Planned gold distributions,
    invalid counts and coverage accompany every row.  The predeclared .90/.10
    threshold is a descriptive response-share diagnostic, not an error claim
    or significance test.  Four-cell balance applies to the overall row only;
    fixed-direction rows explicitly retain their conditional gold counts.
    """
    observations = _validated(observations)
    output = []
    for identity, rows in _groups(observations, PROTECTED_COLUMNS):
        summary = _summary(rows)
        cells = json.loads(summary["query_direction_gold_cells"])
        output.append({**identity, "direction_axis": "overall", "query_relation": "all", "query_direction": "all",
                       **summary, "four_cell_balance_status": "balanced" if len(set(cells.values())) == 1 else "unbalanced"})
        for axis in ("query_relation", "query_direction"):
            for value in sorted({row[axis] for row in rows}):
                output.append({**identity, "direction_axis": axis,
                    "query_relation": value if axis == "query_relation" else "all",
                    "query_direction": value if axis == "query_direction" else "all",
                    **_summary([row for row in rows if row[axis] == value]),
                    "four_cell_balance_status": "not_applicable_fixed_direction"})
    return output


def paired_direction_report(observations) -> list[dict]:
    """Compare canonical to P2 or equivalent while preserving the same query.

    Opposite before/after *queries* are never paired: they may change gold.
    P2 is a premise converse intervention; the equivalent condition is a
    separately labelled multiple-transformation intervention.  Missing answers
    remain None in planned pair denominators; missing planned rows are errors.
    """
    observations = _validated(observations)
    if not observations:
        return []
    base_columns = tuple(key for key in PROTECTED_COLUMNS if key != "condition")
    columns = base_columns + ("query_relation", "query_direction")
    p2_designs = {tuple(row[key] for key in base_columns) for row in observations if row["condition"] == "P2"}
    output = []
    for identity, rows in _groups(observations, columns):
        units = defaultdict(dict)
        for row in rows:
            units[(row["cluster_id"], row["query_id"])][row["condition"]] = row
        targets = ["equivalent"] + (["P2"] if tuple(identity[key] for key in base_columns) in p2_designs else [])
        for target in targets:
            pairs = []
            for (cluster_id, _), conditions in sorted(units.items()):
                if "canonical" not in conditions or target not in conditions:
                    raise ValueError("missing planned canonical/equivalent or canonical/P2 half; restore run-plan rows")
                original, equivalent = conditions["canonical"], conditions[target]
                if original["gold_label"] is not equivalent["gold_label"]:
                    raise ValueError("answer-preserving direction intervention changed gold")
                for field in ("query_subject", "query_object", "query_relation", "query_direction"):
                    if original.get(field) != equivalent.get(field):
                        raise ValueError("direction intervention changed the query target or lexical direction")
                pairs.append({"cluster_id": cluster_id, "n": original["n"], "gold": original["gold_label"],
                              "original": original["prediction"], "equivalent": equivalent["prediction"]})
            summary = summarize_pairs(pairs)
            summary["undefined_reasons"] = _json(summary["undefined_reasons"])
            p2 = "P2" in targets
            output.append({**identity, "comparison": "canonical_vs_" + target,
                "original_condition": "canonical", "equivalent_condition": target,
                "transformation_scope": "premise_converse_semantic_preserving" if target == "P2" else "multiple_presentation_changes",
                "query_direction_changed": False, "query_relation_changed": False,
                "p2_available": p2, "p2_unavailable_reason": None if p2 else "P2_not_present_in_planned_observations",
                **summary})
    return output

"""Matched reasoning-arm analysis without pooling prompting and native effort.

Inputs contain every planned response, including missing/invalid predictions.
Base model identity remains in the legacy CBE schema; the effective spec hash
and intervention are additional protected arm metadata. This module makes no
model calls and never generates observations or stand-in research results.
"""

from __future__ import annotations

from collections import Counter, defaultdict
import re

from analysis.io import canonical, digest
from analysis.metrics import summarize_pairs
from analysis.pipeline import DESIGN, MODEL, PROTECTED, build_tables as _cbe_tables, presentation_pairs
from analysis.reporting import export_figures as _cbe_figures
from analysis.statistics import bootstrap, holm_adjust, mcnemar


ANALYSIS_VERSION = "0.8.0"
STRATEGIES = {"direct", "zero_shot_reasoning", "structured_reasoning", "provider_effort"}
_HASH = re.compile(r"[0-9a-f]{64}\Z")
_ARM = ("arm_id", "strategy", "effort", "effective_model_spec_sha256")
_UNIT = DESIGN + ("cluster_id", "query_id", "language", "condition")
_QUERY = ("gold_label", "query_direction", "query_subject", "query_object", "query_relation")
_METRICS = ("original_accuracy", "equivalent_accuracy", "all_planned_mean_accuracy",
            "R", "invariance", "kappa", "valid_pair_coverage", "strict_robust_correctness")
_SENSITIVITY = ("R", "invariance", "kappa")
_EXTRA_TABLES = ("strategy_contrasts", "four_response_sensitivity",
                 "language_strategy_interaction", "arm_usage", "truncation_invalid",
                 "unavailable_strategy_contrasts")


def _key(row, keys):
    return tuple(row[field] for field in keys)


def _groups(rows, keys):
    grouped = defaultdict(list)
    for row in rows:
        grouped[_key(row, keys)].append(row)
    return [(dict(zip(keys, key)), grouped[key]) for key in sorted(grouped, key=canonical)]


def _seed(seed, identity):
    return seed + int(digest(canonical(identity).encode())[:12], 16)


def _validated(observations):
    if not isinstance(observations, list):
        raise ValueError("observations must be a list of planned reasoning records")
    arms = {}
    by_model = defaultdict(lambda: defaultdict(dict))
    metadata = {}
    for row in observations:
        required = MODEL + _UNIT + _ARM + _QUERY + ("prediction", "response_status", "truncated", "finish_reason")
        if not isinstance(row, dict) or any(field not in row for field in required):
            raise ValueError("reasoning records require base-model, arm and planned-query fields")
        if not isinstance(row["arm_id"], str) or not row["arm_id"] or row["strategy"] not in STRATEGIES:
            raise ValueError("unregistered reasoning strategy or empty arm identifier")
        if row["strategy"] == "provider_effort":
            if not isinstance(row["effort"], str) or not row["effort"]:
                raise ValueError("provider_effort requires an explicit effort level")
        elif row["effort"] is not None:
            raise ValueError("explicit prompting/direct arms must not carry native effort labels")
        if not isinstance(row["effective_model_spec_sha256"], str) or not _HASH.fullmatch(row["effective_model_spec_sha256"]):
            raise ValueError("effective model spec hash must be explicit SHA-256")
        if (row.get("prediction") is not None and type(row["prediction"]) is not bool
                or type(row["gold_label"]) is not bool):
            raise ValueError("predictions must be exact Boolean or None; gold must be Boolean")
        if type(row.get("truncated")) is not bool:
            raise ValueError("truncated must be an explicit Boolean for every planned record")
        if row.get("finish_reason") is not None and not isinstance(row["finish_reason"], str):
            raise ValueError("finish_reason must be text or None")
        if row["truncated"] and row.get("prediction") is not None:
            raise ValueError("truncated responses must remain invalid")
        if row.get("response_status") not in {"valid", "invalid", "missing", "provider_failed"}:
            raise ValueError("response_status must preserve valid/invalid/missing/provider_failed")
        if (row["response_status"] == "valid") != (row.get("prediction") is not None):
            raise ValueError("response status and parsed prediction disagree")
        if "response_observed" in row and type(row["response_observed"]) is not bool:
            raise ValueError("response_observed must be Boolean when supplied")
        if row.get("response_observed") is False and row.get("prediction") is not None:
            raise ValueError("unobserved responses cannot contain valid predictions")
        for field in ("input_tokens", "output_tokens", "reasoning_tokens"):
            value = row.get(field)
            if value is not None and (type(value) is not int or value < 0):
                raise ValueError("usage tokens must be reported nonnegative integers or None")
        definition = (row["strategy"], row["effort"])
        if row["arm_id"] in arms and arms[row["arm_id"]] != definition:
            raise ValueError("an arm identifier cannot denote different strategies or effort levels")
        arms[row["arm_id"]] = definition
        model = _key(row, MODEL)
        unit = _key(row, _UNIT)
        if unit in by_model[model][row["arm_id"]]:
            raise ValueError("duplicate planned reasoning coordinate; never pool runs or replicates")
        by_model[model][row["arm_id"]][unit] = row
        meta_key = (model, row["arm_id"])
        current = {field: row[field] for field in _ARM}
        if meta_key in metadata and metadata[meta_key] != current:
            raise ValueError("effective spec changed within a base-model/arm cohort")
        metadata[meta_key] = current
    direct = {}
    for model, choices in by_model.items():
        baselines = [arm for arm in choices if arms[arm][0] == "direct"]
        if len(baselines) != 1:
            raise ValueError("each base model requires exactly one direct baseline arm")
        baseline = baselines[0]
        direct[model] = baseline
        reference = choices[baseline]
        for arm, units in choices.items():
            if units.keys() != reference.keys():
                raise ValueError("reasoning arms have missing or different planned coordinates; restore missing responses")
            for unit, row in units.items():
                if any(row[field] != reference[unit][field] for field in _QUERY):
                    raise ValueError("reasoning intervention changed query target/direction or gold")
    return by_model, metadata, direct


def _pairs(rows, prefix):
    return [{"original": row[prefix + "_original"], "equivalent": row[prefix + "_equivalent"],
             "gold": row["gold"], "cluster_id": row["cluster_id"], "n": row["n"]} for row in rows]


def _summaries(rows, *, common=False):
    selected = [row for row in rows if _four_valid(row)] if common else rows
    summaries = [summarize_pairs(_pairs(selected, side)) for side in ("direct", "strategy")]
    for summary in summaries:
        summary["all_planned_mean_accuracy"] = ((summary["original_accuracy"] + summary["equivalent_accuracy"]) / 2
            if summary["n_pairs"] else None)
    return summaries


def _four_valid(row):
    return all(row[side + "_" + presentation] is not None
               for side in ("direct", "strategy") for presentation in ("original", "equivalent"))


def _difference(left, right):
    return left - right if left is not None and right is not None else None


def _support(rows):
    direct = [row for row in rows if row["direct_original"] is not None and row["direct_equivalent"] is not None]
    strategy = [row for row in rows if row["strategy_original"] is not None and row["strategy_equivalent"] is not None]
    common = [row for row in rows if _four_valid(row)]
    def fingerprint(selected):
        return digest(canonical(sorted([[row["cluster_id"], row["query_id"]] for row in selected])).encode())
    return {"n_pairs": len(rows), "n_clusters": len({row["cluster_id"] for row in rows}),
            "direct_valid_pairs": len(direct), "strategy_valid_pairs": len(strategy),
            "four_response_valid_pairs": len(common), "four_response_coverage": len(common) / len(rows),
            "direct_valid_support_sha256": fingerprint(direct),
            "strategy_valid_support_sha256": fingerprint(strategy),
            "four_response_support_sha256": fingerprint(common),
            "valid_support_equal": fingerprint(direct) == fingerprint(strategy)}


def _binary_test(rows, metric):
    if metric not in {"original_accuracy", "equivalent_accuracy", "strict_robust_correctness"}:
        return {"p_value": None, "method": "matched_cluster_bootstrap_only",
                "status": "not_an_all_planned_binary_contrast"}
    def correct(row, side):
        if metric == "strict_robust_correctness":
            return all(row[side + "_" + presentation] is not None and row[side + "_" + presentation] == row["gold"]
                       for presentation in ("original", "equivalent"))
        prediction = row[side + ("_original" if metric == "original_accuracy" else "_equivalent")]
        return prediction is not None and prediction == row["gold"]
    result = mcnemar([correct(row, "strategy") for row in rows], [correct(row, "direct") for row in rows],
                     clusters=[row["cluster_id"] for row in rows])
    result["mcnemar_n"] = result.pop("n")
    return result


def _contrast_rows(rows, identity, seed, repeats, *, common=False):
    metrics = _SENSITIVITY if common else _METRICS
    def statistic(sample):
        direct, strategy = _summaries(sample, common=common)
        return {metric: _difference(strategy[metric], direct[metric]) for metric in metrics}
    actual_seed = _seed(seed, {**identity, "support": "four_response_common_valid" if common else "primary"})
    intervals = bootstrap(rows, statistic, seed=actual_seed, repeats=repeats)
    direct, strategy = _summaries(rows, common=common)
    support = _support(rows)
    output = []
    for metric in metrics:
        interval = intervals[metric]
        row = {**identity, **support, "metric": metric, "direct_value": direct[metric],
            "strategy_value": strategy[metric], "difference": interval["estimate"],
            "difference_direction": "strategy_minus_direct", "ci_low": interval["low"], "ci_high": interval["high"],
            "bootstrap_status": interval["status"], "bootstrap_defined_replicates": interval["defined_replicates"],
            "bootstrap_replicates": repeats, "bootstrap_seed": actual_seed,
            "ci_method": "matched_latent_cluster_percentile", "confidence": .95,
            "support_scope": ("same_four_response_common_valid_pairs" if common else
                "strategy_specific_joint_valid_pairs" if metric in {"R", "invariance", "kappa"} else "all_planned_pairs"),
            "direct_invalid_responses": sum(row["direct_" + p] is None for row in rows for p in ("original", "equivalent")),
            "strategy_invalid_responses": sum(row["strategy_" + p] is None for row in rows for p in ("original", "equivalent"))}
        row.update({"p_value": None, "method": "matched_cluster_bootstrap_sensitivity",
                    "status": "conditional_common_valid_support"} if common else _binary_test(rows, metric))
        output.append(row)
    return output


def _holm(rows, table):
    families = defaultdict(list)
    for row in rows:
        identity = {field: row[field] for field in DESIGN if field in row}
        identity.update(table=table, language=row.get("language"), comparison=row.get("comparison"),
                        condition=row.get("condition"), metric=row.get("metric"))
        family = digest(canonical(identity).encode())
        row["holm_family"] = family
        families[family].append(row)
    for family in families.values():
        adjusted = holm_adjust({str(i): row["p_value"] for i, row in enumerate(family)})
        for i, row in enumerate(family):
            row.update(p_adjusted=adjusted[str(i)], holm_family_size=len(family))


def _usage(observations):
    usage, contingency = [], []
    keys = PROTECTED + ("language", "condition") + _ARM
    for identity, rows in _groups(observations, keys):
        summary = {**identity, "n_planned_responses": len(rows),
            "n_observed_responses": sum(row.get("response_observed", row["response_status"] in {"valid", "invalid"}) for row in rows),
            "n_invalid_including_missing": sum(row["prediction"] is None for row in rows),
            "n_missing_responses": sum(row["response_status"] == "missing" for row in rows),
            "n_provider_failed": sum(row["response_status"] == "provider_failed"
                or row.get("invalid_reason") in {"provider_error", "ambiguous_call_abandoned"}
                or row.get("response_observed") is False and row["response_status"] != "missing" for row in rows),
            "n_truncated": sum(row["truncated"] for row in rows),
            "invalid_response_rate": sum(row["prediction"] is None for row in rows) / len(rows),
            "truncation_rate": sum(row["truncated"] for row in rows) / len(rows),
            "finish_reason_counts": canonical(dict(sorted(Counter(row["finish_reason"] or "unreported" for row in rows).items()))),
            "token_scope": "provider_reported; reasoning may be included in output, never sum categories"}
        for field in ("input_tokens", "output_tokens", "reasoning_tokens"):
            available = [row[field] for row in rows if row.get(field) is not None]
            summary.update({field + "_reported_responses": len(available),
                            field + "_coverage": len(available) / len(rows),
                            field + "_sum_reported": sum(available) if available else None,
                            field + "_mean_reported": sum(available) / len(available) if available else None})
        usage.append(summary)
        counts = Counter((row["truncated"], row["prediction"] is not None, row["response_status"],
                          row.get("response_observed", row["response_status"] in {"valid", "invalid"})) for row in rows)
        for (truncated, valid, status, observed), count in sorted(counts.items()):
            contingency.append({**identity, "truncated": truncated, "response_valid": valid,
                "response_status": status, "response_observed": observed,
                "responses": count, "n_planned_responses": len(rows),
                "all_planned_fraction": count / len(rows)})
    return usage, contingency


def build_tables(observations, *, seed=20261025, repeats=2000):
    """Analyse complete matched arm cohorts, preserving all invalid denominators.

    Native ``provider_effort`` arms are separate from explicit reasoning prompts.
    Standard CBE tables are computed independently per arm before combination.
    Contrasts are strategy−direct; R/invariance/kappa use each arm's own valid
    pairs in the primary analysis and the same four-answer support in sensitivity.
    Holm families include all models/strategies at a fixed design and endpoint.
    """
    if type(seed) is not int or seed < 0 or type(repeats) is not int or repeats < 1:
        raise ValueError("seed and bootstrap repeats must be valid integers")
    by_model, metadata, direct = _validated(observations)
    tables = {name: [] for name in _cbe_tables([], seed=seed, repeats=repeats)}
    tables.update({name: [] for name in _EXTRA_TABLES})
    for arm in sorted({row["arm_id"] for row in observations}):
        selected = [row for row in observations if row["arm_id"] == arm]
        standard = _cbe_tables(selected, seed=_seed(seed, {"arm_id": arm}), repeats=repeats)
        for name, rows in standard.items():
            for row in rows:
                row.update(metadata[(_key(row, MODEL), arm)])
            tables[name].extend(rows)
    # Arm IDs contain model IDs. Correct across models sharing a scientific
    # intervention rather than resetting multiplicity separately for each ID.
    for name in ("presentation_tests", "language_differences", "language_robustness"):
        families = defaultdict(list)
        for row in tables[name]:
            families[(row["holm_family"], row["strategy"], row["effort"])].append(row)
        for family_key, family in families.items():
            adjusted = holm_adjust({str(i): row["p_value"] for i, row in enumerate(family)})
            for i, row in enumerate(family):
                row.update(p_adjusted=adjusted[str(i)], holm_family_size=len(family),
                           holm_family=digest(canonical(family_key).encode()))
    for model, arms in sorted(by_model.items(), key=lambda item: canonical(item[0])):
        reference_arm = direct[model]
        reference_rows = list(arms[reference_arm].values())
        reference_pairs = presentation_pairs(reference_rows)
        pair_keys = PROTECTED + ("cluster_id", "query_id", "language", "comparison")
        reference_index = {_key(row, pair_keys): row for row in reference_pairs}
        for arm, coordinates in sorted(arms.items()):
            if arm == reference_arm:
                continue
            arm_metadata = {**metadata[(model, arm)], "reference_arm_id": reference_arm,
                "reference_strategy": "direct", "reference_effective_model_spec_sha256": metadata[(model, reference_arm)]["effective_model_spec_sha256"]}
            wide = []
            for pair in presentation_pairs(list(coordinates.values())):
                reference = reference_index[_key(pair, pair_keys)]
                wide.append({**pair, "direct_original": reference["original"], "direct_equivalent": reference["equivalent"],
                             "strategy_original": pair["original"], "strategy_equivalent": pair["equivalent"]})
            for identity, rows in _groups(wide, PROTECTED + ("language", "comparison")):
                identity.update(arm_metadata)
                tables["strategy_contrasts"].extend(_contrast_rows(rows, identity, seed, repeats))
                tables["four_response_sensitivity"].extend(_contrast_rows(rows, identity, seed, repeats, common=True))
            language_rows = []
            for unit, strategy in coordinates.items():
                reference = arms[reference_arm][unit]
                language_rows.append({**strategy, "direct_prediction": reference["prediction"],
                    "improvement": int(strategy["prediction"] is not None and strategy["prediction"] == strategy["gold_label"])
                        - int(reference["prediction"] is not None and reference["prediction"] == reference["gold_label"])})
            for identity, rows in _groups(language_rows, PROTECTED + ("condition",)):
                matched = []
                missing = 0
                for unit, candidates in _groups(rows, ("cluster_id", "query_id")):
                    languages = {row["language"]: row for row in candidates}
                    if set(languages) != {"tr", "en"}:
                        missing += 1
                        continue
                    tr, en = languages["tr"], languages["en"]
                    matched.append({**unit, "n": identity["n"], "difference": tr["improvement"] - en["improvement"],
                        "four_response_valid": all(value is not None for row in (tr, en)
                            for value in (row["prediction"], row["direct_prediction"]))})
                if missing:
                    tables["unavailable_strategy_contrasts"].append({**identity, **arm_metadata,
                        "contrast": "language_by_strategy", "unmatched_language_units": missing,
                        "reason": "counterpart_not_in_planned_selection"})
                if not matched:
                    continue
                actual_seed = _seed(seed, {**identity, **arm_metadata, "contrast": "language_by_strategy"})
                interval = bootstrap(matched, lambda sample: {"difference": sum(row["difference"] for row in sample) / len(sample)},
                                     seed=actual_seed, repeats=repeats)["difference"]
                tables["language_strategy_interaction"].append({**identity, **arm_metadata,
                    "difference": interval["estimate"], "difference_direction": "(TR_strategy-TR_direct)-(EN_strategy-EN_direct)",
                    "ci_low": interval["low"], "ci_high": interval["high"], "p_value": None,
                    "bootstrap_status": interval["status"], "bootstrap_defined_replicates": interval["defined_replicates"],
                    "bootstrap_seed": actual_seed, "bootstrap_replicates": repeats,
                    "n_pairs": len(matched), "n_clusters": len({row["cluster_id"] for row in matched}),
                    "four_response_valid_units": sum(row["four_response_valid"] for row in matched),
                    "support_scope": "all_planned_matched_language_units_invalid_wrong"})
    tables["arm_usage"], tables["truncation_invalid"] = _usage(observations)
    for name in ("strategy_contrasts", "four_response_sensitivity", "language_strategy_interaction"):
        _holm(tables[name], name)
    return tables


def export_figures(tables, outputdir, *, synthetic=False):
    """Export the six CBE figure families with arm/effort protected as facets.

    No data means no figures. Tests must explicitly request synthetic watermarking;
    matched-strategy contrasts remain labeled tables, never TR/EN plot stand-ins.
    """
    return _cbe_figures({"profiles": tables.get("profiles", []),
                         "language_differences": tables.get("language_differences", [])},
                        outputdir, synthetic=synthetic)

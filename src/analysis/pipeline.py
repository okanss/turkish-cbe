"""Matched, explicitly stratified CBE profiles and exploratory contrasts."""

from __future__ import annotations

from collections import defaultdict
import math

from .io import canonical, digest
from .metrics import summarize_pairs
from .statistics import bootstrap, holm_adjust, mcnemar, wilson_interval


MODEL = ("model_id", "model_family", "model_revision", "model_spec_sha256")
DESIGN = ("split", "name_control", "relation_id", "relation_family", "query_type", "query_distance", "n")
PROTECTED = MODEL + DESIGN
METRICS = ("original_accuracy", "equivalent_accuracy", "mean_accuracy", "R", "D", "U",
           "invariance", "kappa", "invalid_response_rate", "valid_pair_coverage", "strict_robust_correctness")


def _key(row, keys):
    return tuple(row[k] for k in keys)


def _groups(rows, keys):
    groups = defaultdict(list)
    for row in rows:
        groups[_key(row, keys)].append(row)
    return [(dict(zip(keys, key)), groups[key]) for key in sorted(groups, key=canonical)]


def _mean(rows, key):
    return sum(row[key] for row in rows) / len(rows) if rows else None


def _seed(seed, identity):
    return seed + int(digest(canonical(identity).encode())[:12], 16)


def _intervals(rows, callback, identity, seed, repeats, *, strata=("n",)):
    return bootstrap(rows, callback, seed=_seed(seed, identity), repeats=repeats,
                     cluster_key="cluster_id", strata_keys=strata)


def _correct(row):
    return row["prediction"] is not None and row["prediction"] == row["gold_label"]


def presentation_pairs(observations):
    """One planned canonical/variant pair per model/query/language/design."""
    if not isinstance(observations, list):
        raise ValueError("observations must be a list")
    unit_keys = PROTECTED + ("cluster_id", "query_id", "language")
    pairs = []
    for identity, rows in _groups(observations, unit_keys):
        variants = {}
        for row in rows:
            prediction = row["prediction"]
            if prediction is not None and type(prediction) is not bool:
                raise ValueError("predictions must be parsed Boolean or None")
            if type(row["gold_label"]) is not bool:
                raise ValueError("gold must be exact Boolean")
            condition = row["condition"]
            if condition in variants:
                raise ValueError("duplicate planned observation; do not pool runs or replicates")
            variants[condition] = row
        if "canonical" not in variants or "equivalent" not in variants:
            raise ValueError("missing planned condition; reconstruct missing responses from run plan")
        original = variants.pop("canonical")
        for condition, equivalent in sorted(variants.items()):
            if condition not in {"equivalent", "P1", "P2", "P3", "P4", "P5"}:
                raise ValueError("unregistered presentation condition")
            if equivalent["gold_label"] != original["gold_label"]:
                raise ValueError("answer-preserving pair changed gold")
            if any(equivalent.get(k) != original.get(k) for k in (
                    "query_direction", "query_subject", "query_object", "query_relation")):
                raise ValueError("pair changed query target/direction")
            pairs.append({**identity, "comparison": condition, "gold": original["gold_label"],
                          "original": original["prediction"], "equivalent": equivalent["prediction"]})
    return pairs


def _flatten_ci(summary, intervals):
    for metric, result in intervals.items():
        summary[metric + "_ci_low"] = result["low"]
        summary[metric + "_ci_high"] = result["high"]
    summary["confidence_intervals"] = canonical(intervals)


def _contrast(rows, identity, seed, repeats, *, name="difference", test=True):
    result = dict(identity)
    intervals = _intervals(rows, lambda sample: {name: _mean(sample, "difference")},
                           identity, seed, repeats)
    result.update(n_pairs=len(rows), n_clusters=len({r["cluster_id"] for r in rows}),
                  difference=_mean(rows, "difference"), ci_low=intervals[name]["low"],
                  ci_high=intervals[name]["high"], ci_method="matched_instance_cluster_percentile",
                  bootstrap_status=intervals[name]["status"],
                  bootstrap_defined_replicates=intervals[name]["defined_replicates"])
    if test:
        paired_test = mcnemar([r["left"] for r in rows], [r["right"] for r in rows],
                              clusters=[r["cluster_id"] for r in rows])
        # Sample size is not the logical number of entities in this profile.
        paired_test["mcnemar_n"] = paired_test.pop("n")
        result.update(paired_test)
    else:
        result.update(p_value=None, method=None, status="exploratory_bootstrap_contrast")
    return result


def _holm(tables):
    """Models share families; no per-model multiple-testing reset."""
    families = defaultdict(list)
    for table in ("presentation_tests", "language_differences", "language_robustness"):
        for row in tables[table]:
            family = {k: row[k] for k in DESIGN if k in row}
            family.update(table=table, condition=row.get("condition"),
                          language=row.get("language"), comparison=row.get("comparison"))
            family_id = digest(canonical(family).encode())
            row["holm_family"] = family_id
            families[family_id].append(row)
    for rows in families.values():
        adjusted = holm_adjust({str(i): r["p_value"] for i, r in enumerate(rows)})
        for i, row in enumerate(rows):
            row["p_adjusted"] = adjusted[str(i)]
            row["holm_family_size"] = len(rows)


def _pearson(xs, ys):
    if len(xs) < 2:
        return None
    mx, my = sum(xs)/len(xs), sum(ys)/len(ys)
    xx = sum((x-mx)**2 for x in xs)
    yy = sum((y-my)**2 for y in ys)
    if not xx or not yy:
        return None
    return sum((x-mx)*(y-my) for x,y in zip(xs,ys))/math.sqrt(xx*yy)


def build_tables(observations, *, seed=20261024, repeats=2000):
    if type(seed) is not int or seed < 0 or type(repeats) is not int or repeats < 1:
        raise ValueError("seed and bootstrap repeats must be valid integers")
    pairs = presentation_pairs(observations)
    tables = {key: [] for key in ("profiles", "relation_family_profiles", "presentation_tests",
        "language_differences", "language_robustness", "language_presentation_interaction",
        "language_complexity_interaction", "endpoint_internal", "invalid_token_relationship", "unavailable_contrasts")}
    profile_keys = PROTECTED + ("language", "comparison")
    for identity, rows in _groups(pairs, profile_keys):
        summary = summarize_pairs(rows)
        intervals = _intervals(rows, lambda sample: {k:v for k,v in summarize_pairs(sample).items()
                                  if k in METRICS}, identity, seed, repeats)
        wilson = wilson_interval(summary["both_correct"], summary["n_valid_pairs"])
        valid_clusters = [r["cluster_id"] for r in rows if r["original"] is not None and r["equivalent"] is not None]
        summary.update(R_wilson_low=wilson["low"], R_wilson_high=wilson["high"],
            R_wilson_interpretation="binomial_iid" if len(valid_clusters)==len(set(valid_clusters))
                else "descriptive_dependent_pairs", R_interval_method="Wilson",
            ci_method="matched_instance_cluster_percentile", bootstrap_replicates=repeats,
            bootstrap_seed=_seed(seed, identity))
        _flatten_ci(summary, intervals)
        summary["undefined_reasons"] = canonical(summary["undefined_reasons"])
        tables["profiles"].append({**identity, **summary})
        contrasts = [{**r, "left": r["original"] is not None and r["original"]==r["gold"],
                     "right": r["equivalent"] is not None and r["equivalent"]==r["gold"],
                     "difference": int(r["equivalent"] is not None and r["equivalent"]==r["gold"])
                         -int(r["original"] is not None and r["original"]==r["gold"])} for r in rows]
        tables["presentation_tests"].append(_contrast(contrasts, identity, seed, repeats))

    # Relation-family summaries preserve n/query/distance and all dependent rows.
    family_keys = tuple(k for k in profile_keys if k != "relation_id")
    for identity, rows in _groups(pairs, family_keys):
        summary = summarize_pairs(rows)
        intervals = _intervals(rows, lambda sample: {k:v for k,v in summarize_pairs(sample).items()
                                  if k in METRICS}, {**identity,"summary":"relation_family"},seed,repeats)
        _flatten_ci(summary, intervals)
        summary["undefined_reasons"] = canonical(summary["undefined_reasons"])
        tables["relation_family_profiles"].append({**identity, **summary,
            "relation_ids": canonical(sorted({r["relation_id"] for r in rows})),
            "interpretation": "descriptive_row_weighted_with_latent_clusters"})

    # Accuracy and strict-R language contrasts on exactly matched planned units.
    language_units, unavailable_units = [], []
    language_keys = PROTECTED + ("cluster_id", "query_id", "condition")
    for identity, rows in _groups(observations, language_keys):
        by_language = {r["language"]: r for r in rows}
        if len(by_language) != len(rows):
            raise ValueError("duplicate language unit")
        if set(by_language) != {"en", "tr"}:
            unavailable_units.append({**identity,"available_languages":canonical(sorted(by_language))})
            continue  # TR-only ablations are not cross-language contrasts.
        tr, en = by_language["tr"], by_language["en"]
        if (tr["gold_label"] != en["gold_label"] or any(tr.get(k)!=en.get(k) for k in
                ("query_direction", "query_subject", "query_object", "query_relation"))):
            raise ValueError("cross-language query/gold mismatch")
        language_units.append({**identity, "left": _correct(tr), "right": _correct(en),
            "difference": int(_correct(tr))-int(_correct(en)), "gold": tr["gold_label"]})
    for identity, rows in _groups(unavailable_units,PROTECTED+("condition","available_languages")):
        tables["unavailable_contrasts"].append({**identity,"contrast":"TR_minus_EN_accuracy",
            "n_planned_units":len(rows),"n_clusters":len({r["cluster_id"] for r in rows}),
            "reason":"counterpart_not_in_planned_selection"})
    for identity, rows in _groups(language_units, PROTECTED + ("condition",)):
        tables["language_differences"].append(_contrast(rows, identity, seed, repeats))

    robust_units = []
    for identity, rows in _groups([p for p in pairs if p["comparison"]=="equivalent"],
                                  PROTECTED + ("cluster_id", "query_id")):
        by_language = {r["language"]:r for r in rows}
        if set(by_language) != {"tr", "en"}:
            continue
        correctness = {lang: r["original"] is not None and r["equivalent"] is not None
                           and r["original"]==r["gold"] and r["equivalent"]==r["gold"]
                       for lang,r in by_language.items()}
        robust_units.append({**identity, "left":correctness["tr"], "right":correctness["en"],
            "difference":int(correctness["tr"])-int(correctness["en"])})
    for identity, rows in _groups(robust_units, PROTECTED):
        tables["language_robustness"].append(_contrast(rows, identity, seed, repeats))

    interactions = []
    for identity, rows in _groups(language_units, PROTECTED + ("cluster_id", "query_id")):
        by_condition = {r["condition"]: r for r in rows}
        if {"canonical","equivalent"} <= by_condition.keys():
            interactions.append({**identity, "difference": by_condition["equivalent"]["difference"]
                                 -by_condition["canonical"]["difference"]})
    for identity, rows in _groups(interactions, PROTECTED):
        tables["language_presentation_interaction"].append(_contrast(rows, identity, seed, repeats, test=False))

    # Endpoint d=n-1 is a structural profile; other query distances stay fixed.
    complexity_keys = tuple(k for k in PROTECTED if k not in {"n","query_distance"}) + ("condition",)
    for identity, candidate_rows in _groups(language_units, complexity_keys):
        distance_groups = {"endpoint":candidate_rows} if identity["query_type"]=="endpoint" else {
            d: [r for r in candidate_rows if r["query_distance"]==d] for d in sorted({r["query_distance"] for r in candidate_rows})}
        for distance, rows in distance_groups.items():
            levels = sorted({r["n"] for r in rows})
            if len(levels) < 2:
                continue
            baseline = levels[0]
            for n in levels[1:]:
                selected = [r for r in rows if r["n"] in {baseline,n}]
                def interaction_stat(sample):
                    first = [r for r in sample if r["n"]==baseline]
                    second = [r for r in sample if r["n"]==n]
                    return {"difference": _mean(second,"difference")-_mean(first,"difference")}
                interval = _intervals(selected, interaction_stat, {**identity,"baseline_n":baseline,"n":n,
                    "distance":distance}, seed,repeats)["difference"]
                tables["language_complexity_interaction"].append({**identity,"baseline_n":baseline,"n":n,
                    "distance_policy":str(distance),"difference":interval["estimate"],
                    "ci_low":interval["low"],"ci_high":interval["high"],"status":interval["status"],
                    "n_clusters":len({r["cluster_id"] for r in selected}), "p_value":None})

    # Match query arms by latent order, never by changed query ID/gold label.
    endpoint_keys = MODEL + ("split","name_control","relation_id","relation_family","n","language","condition","cluster_id")
    arm_rows = []
    for identity, rows in _groups(observations, endpoint_keys):
        endpoints = [r for r in rows if r["query_type"]=="endpoint"]
        if len(endpoints)>1:
            raise ValueError("duplicate endpoint query within latent order")
        if not endpoints:
            continue
        for row in rows:
            if row["query_type"] == "endpoint":
                continue
            arm_rows.append({**identity,"query_type":row["query_type"],"query_distance":row["query_distance"],
                "left":_correct(row),"right":_correct(endpoints[0]),
                "difference":int(_correct(row))-int(_correct(endpoints[0]))})
    for identity, rows in _groups(arm_rows, PROTECTED + ("language","condition")):
        tables["endpoint_internal"].append(_contrast(rows,identity,seed,repeats,test=False))

    # Descriptive point-biserial correlation; missing outcomes have no observed
    # response-format status and are separately counted, never called valid.
    for identity, rows in _groups(observations, PROTECTED + ("language","condition")):
        for field, source in (("input_tokens","provider_reported_input_including_chat"),
                              ("source_prompt_tokens","frozen_old_prompt_bpe_proxy")):
            included = [r for r in rows if r.get(field) is not None and r["response_status"]!="missing"]
            xs = [r[field] for r in included]
            ys = [int(r["prediction"] is None) for r in included]
            correlation = _pearson(xs,ys)
            tables["invalid_token_relationship"].append({**identity,"token_source":source,
                "token_encoding":rows[0].get("token_encoding"),"correlation":correlation,
                "status":"descriptive" if correlation is not None else "undefined_insufficient_or_constant_data",
                "outcome_definition":"invalid_or_provider_failed_record_excluding_missing_responses",
                "n_planned":len(rows),"n_observed_with_tokens":len(included),
                "n_missing_responses":sum(r["response_status"]=="missing" for r in rows),
                "n_clusters":len({r["cluster_id"] for r in included}),"p_value":None})
    _holm(tables)
    return tables

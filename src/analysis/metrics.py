"""CBE metrics with explicit planned-pair and joint-valid denominators."""

from collections.abc import Iterable, Mapping
from typing import Any


def summarize_pairs(pairs: Iterable[Mapping[str, Any]]) -> dict[str, Any]:
    """Summarize matched Boolean answers at one complexity level.

    Each mapping needs ``original``, ``equivalent`` (exact bool or None),
    ``gold`` (exact bool), ``cluster_id`` (nonblank str), and ``n`` (positive
    exact int). None denotes invalid *or missing*: it remains in all planned
    denominators. R/D/U, mean_accuracy, invariance, q, kappa, and the two
    ``*_accuracy_valid`` fields use the same joint-valid support. The latter
    are diagnostic, conditional accuracies, not the marginal main metrics.
    Distinct n levels must be summarized separately, because pooling would
    change the pooled-answer reference used by kappa. Input is not mutated.

    Undefined rates are None, with a field-specific ``undefined_reasons``
    entry. Counts are returned even when the relevant denominator is zero.
    """
    if isinstance(pairs, (str, bytes, Mapping)) or not isinstance(pairs, Iterable):
        raise ValueError("pairs must be an iterable of pair mappings")

    total = valid = invalid_original = invalid_equivalent = 0
    correct_original = correct_equivalent = 0
    both_correct = both_wrong = org_only = eq_only = yes_answers = 0
    clusters: set[str] = set()
    complexity: int | None = None
    required = {"original", "equivalent", "gold", "cluster_id", "n"}

    for index, pair in enumerate(pairs):
        if not isinstance(pair, Mapping):
            raise ValueError(f"pair {index} must be a mapping")
        if not required.issubset(pair):
            missing = ", ".join(sorted(required.difference(pair)))
            raise ValueError(f"pair {index} is missing required fields: {missing}")
        original, equivalent, gold = pair["original"], pair["equivalent"], pair["gold"]
        cluster_id, n = pair["cluster_id"], pair["n"]
        if original is not None and type(original) is not bool:
            raise ValueError(f"pair {index} original must be bool or None")
        if equivalent is not None and type(equivalent) is not bool:
            raise ValueError(f"pair {index} equivalent must be bool or None")
        if type(gold) is not bool:
            raise ValueError(f"pair {index} gold must be bool")
        if type(cluster_id) is not str or not cluster_id.strip():
            raise ValueError(f"pair {index} cluster_id must be a nonblank string")
        if type(n) is not int or n < 1:
            raise ValueError(f"pair {index} n must be a positive integer")
        if complexity is not None and n != complexity:
            raise ValueError("mixed n values must be summarized separately for kappa")
        complexity = n
        total += 1
        clusters.add(cluster_id)
        invalid_original += original is None
        invalid_equivalent += equivalent is None
        c0 = original is not None and original == gold
        c1 = equivalent is not None and equivalent == gold
        correct_original += c0
        correct_equivalent += c1
        if original is None or equivalent is None:
            continue
        valid += 1
        yes_answers += int(original) + int(equivalent)
        both_correct += c0 and c1
        both_wrong += not c0 and not c1
        org_only += c0 and not c1
        eq_only += c1 and not c0

    result: dict[str, Any] = {
        "n_pairs": total,
        "n_valid_pairs": valid,
        "n_clusters": len(clusters),
        "n_invalid_original": invalid_original,
        "n_invalid_equivalent": invalid_equivalent,
        "n_invalid_responses": invalid_original + invalid_equivalent,
        "both_correct": both_correct,
        "both_wrong": both_wrong,
        "org_only": org_only,
        "eq_only": eq_only,
        "undefined_reasons": {},
    }
    all_pair_numerators = {
        "original_accuracy": correct_original,
        "equivalent_accuracy": correct_equivalent,
        "valid_pair_coverage": valid,
        "strict_robust_correctness": both_correct,
    }
    valid_pair_numerators = {
        "original_accuracy_valid": both_correct + org_only,
        "equivalent_accuracy_valid": both_correct + eq_only,
        "mean_accuracy": both_correct + (org_only + eq_only) / 2,
        "R": both_correct,
        "D": both_wrong,
        "U": org_only + eq_only,
        "invariance": both_correct + both_wrong,
    }
    for field, numerator in all_pair_numerators.items():
        result[field] = numerator / total if total else None
        if not total:
            result["undefined_reasons"][field] = "no_pairs"
    result["invalid_response_rate"] = (
        (invalid_original + invalid_equivalent) / (2 * total) if total else None
    )
    if not total:
        result["undefined_reasons"]["invalid_response_rate"] = "no_pairs"
    for field, numerator in valid_pair_numerators.items():
        result[field] = numerator / valid if valid else None
        if not valid:
            result["undefined_reasons"][field] = "no_valid_pairs"
    result["q"] = yes_answers / (2 * valid) if valid else None
    result["I_expected"] = (
        result["q"] ** 2 + (1 - result["q"]) ** 2 if valid else None
    )
    result["kappa"] = None
    if not valid:
        for field in ("q", "I_expected", "kappa"):
            result["undefined_reasons"][field] = "no_valid_pairs"
    elif result["I_expected"] == 1:
        result["undefined_reasons"]["kappa"] = "constant_pooled_answer"
    else:
        result["kappa"] = (
            (result["invariance"] - result["I_expected"])
            / (1 - result["I_expected"])
        )
    return result

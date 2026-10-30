"""Join the two completed 360-item Turkish reviews and apply the predeclared policy.

Inputs are the two reviewers' merged files (annotations only; prompt text comes
from the pristine shipped CSV in both). Outputs, under ``validation/review_results/``:

  review_decisions.csv   per-item: both reviewers' ratings + eligibility decision
  review_summary.json    agreement statistics and aggregate counts

Predeclared inclusion policy (recorded before any annotation existed, with the
2026-10-29 canonical-row clarification): a transformed item is
confirmatory-eligible iff BOTH reviewers rate grammaticality >= 4 AND both say
semantic_equivalence = yes; a canonical item is eligible iff BOTH rate
grammaticality >= 4 AND both say ambiguity = no. Items failing either criterion
for either reviewer are excluded from confirmatory analysis and reported.

Agreement: percent agreement per dimension plus chance-corrected Cohen's kappa
(unweighted, exact categories) and, for the 1-5 scales, agreement on the
policy-relevant binarization (>= 4).

Run:  PYTHONPATH=src python3 -m evaluation.review_adjudication \
          --reviewer1 validation/turkish_linguistic_review.OB.csv \
          --reviewer2 validation/turkish_linguistic_review.MO.csv \
          --output validation/review_results
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from pathlib import Path

DIMS = ("grammaticality", "naturalness", "semantic_equivalence", "ambiguity")


def cohen_kappa(pairs: list[tuple[str, str]]) -> float | None:
    n = len(pairs)
    if n == 0:
        return None
    observed = sum(a == b for a, b in pairs) / n
    left = Counter(a for a, _ in pairs)
    right = Counter(b for _, b in pairs)
    expected = sum(left[c] * right[c] for c in set(left) | set(right)) / (n * n)
    if expected == 1.0:
        return None  # both raters constant; kappa undefined
    return (observed - expected) / (1 - expected)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reviewer1", type=Path, required=True)
    parser.add_argument("--reviewer2", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)

    r1 = list(csv.DictReader(args.reviewer1.open(encoding="utf-8")))
    r2 = list(csv.DictReader(args.reviewer2.open(encoding="utf-8")))
    assert len(r1) == len(r2)

    decisions, agreements = [], {d: [] for d in DIMS}
    binarized = {"grammaticality": [], "naturalness": []}
    excluded_by = Counter()
    for a, b in zip(r1, r2):
        assert a["prompt_id"] == b["prompt_id"]
        canonical = a["presentation_condition"] == "canonical"
        for dim in DIMS:
            if dim == "semantic_equivalence" and canonical:
                continue  # n/a by construction on canonical rows
            agreements[dim].append((a[dim].lower(), b[dim].lower()))
        for dim in binarized:
            binarized[dim].append((str(int(a[dim]) >= 4), str(int(b[dim]) >= 4)))
        grammar_ok = int(a["grammaticality"]) >= 4 and int(b["grammaticality"]) >= 4
        if canonical:
            second_ok = a["ambiguity"] == "no" and b["ambiguity"] == "no"
            criterion = "ambiguity"
        else:
            second_ok = (a["semantic_equivalence"].lower() == "yes"
                         and b["semantic_equivalence"].lower() == "yes")
            criterion = "equivalence"
        eligible = grammar_ok and second_ok
        if not eligible:
            excluded_by["grammaticality" if not grammar_ok else criterion] += 1
        decisions.append({
            "prompt_id": a["prompt_id"],
            "presentation_condition": a["presentation_condition"],
            "relation_id": a["relation_id"], "n": a["n"],
            "r1_grammaticality": a["grammaticality"], "r2_grammaticality": b["grammaticality"],
            "r1_naturalness": a["naturalness"], "r2_naturalness": b["naturalness"],
            "r1_equivalence": a["semantic_equivalence"], "r2_equivalence": b["semantic_equivalence"],
            "r1_ambiguity": a["ambiguity"], "r2_ambiguity": b["ambiguity"],
            "confirmatory_eligible": "yes" if eligible else "no",
            "r1_comment": a["reviewer_comments"], "r2_comment": b["reviewer_comments"],
        })

    args.output.mkdir(parents=True, exist_ok=True)
    with (args.output / "review_decisions.csv").open("w", newline="", encoding="utf-8") as sink:
        writer = csv.DictWriter(sink, fieldnames=list(decisions[0].keys()))
        writer.writeheader()
        writer.writerows(decisions)

    stats = {}
    for dim, pairs in agreements.items():
        stats[dim] = {"n": len(pairs),
                      "percent_agreement": round(sum(a == b for a, b in pairs) / len(pairs), 4),
                      "cohen_kappa": (lambda k: None if k is None else round(k, 4))(cohen_kappa(pairs))}
    for dim, pairs in binarized.items():
        stats[dim + "_binarized_ge4"] = {
            "percent_agreement": round(sum(a == b for a, b in pairs) / len(pairs), 4),
            "cohen_kappa": (lambda k: None if k is None else round(k, 4))(cohen_kappa(pairs))}

    eligible_n = sum(1 for d in decisions if d["confirmatory_eligible"] == "yes")
    by_condition = Counter((d["presentation_condition"], d["confirmatory_eligible"])
                           for d in decisions)
    summary = {
        "items": len(decisions), "confirmatory_eligible": eligible_n,
        "excluded": len(decisions) - eligible_n,
        "exclusion_reason_first_failing": dict(excluded_by),
        "eligible_by_condition": {f"{c}/{e}": v for (c, e), v in sorted(by_condition.items())},
        "agreement": stats,
        "policy": ("predeclared: transformed items need grammaticality>=4 (both) and "
                   "unanimous equivalence=yes; canonical items need grammaticality>=4 "
                   "(both) and unanimous ambiguity=no"),
    }
    (args.output / "review_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=1, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

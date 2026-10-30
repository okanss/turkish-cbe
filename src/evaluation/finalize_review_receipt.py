"""Record a completed two-reviewer Turkish linguistic review as a gate receipt.

This script never judges prompts and never fills ratings. It takes the CSV the
two human reviewers completed, refuses it unless every row carries both
reviewers' four decisions plus a resolution, writes the receipt JSON bound to
the frozen dataset manifest and the exact CSV bytes, and then runs the real
execution-gate validator (``inference.linguistic_review.validate_review``)
against the target run's selections as a self-check. Status is set to
"approved" only when every decision is "pass" and every resolution is
"approved" — i.e., when the humans approved; otherwise the receipt records the
true state and cannot authorize anything.

Run (pinned venv):
    PYTHONPATH=src python3 -m evaluation.finalize_review_receipt \
        --package validation/gate_review/frozen_test \
        --reviewer1 "Full Name" --reviewer2 "Full Name"
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path

from inference.dataset import default_selection
from inference.linguistic_review import RATINGS, validate_review

GRID_MANIFEST = "data/complete_grid/manifest.json"
TRAINING_PLAN = "results/training/plans/pod_pilot_002.json"
ALLOWED_RATINGS = {"pass", "fail"}
ALLOWED_RESOLUTIONS = {"approved", "rejected"}


def target_selections(root: Path, package_name: str) -> list[dict]:
    if package_name == "frozen_test":
        return [default_selection("full")]
    if package_name == "training_eval":
        plan = json.loads((root / TRAINING_PLAN).read_text(encoding="utf-8"))
        return [cohort["selection"] for cohort in plan["data"]["evaluation"].values()]
    raise SystemExit(f"unknown package {package_name!r}")


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--package", type=Path, required=True,
                        help="package directory holding review_sample.csv")
    parser.add_argument("--reviewer1", required=True)
    parser.add_argument("--reviewer2", required=True)
    parser.add_argument("--fixture-only", action="store_true",
                        help="mark the receipt as a synthetic fixture that can "
                             "NEVER authorize a live run (for plumbing tests)")
    args = parser.parse_args(argv)

    root = Path.cwd().resolve()
    package = args.package.resolve()
    csv_path = package / "review_sample.csv"
    reviewers = [args.reviewer1.strip(), args.reviewer2.strip()]
    if not all(reviewers) or reviewers[0] == reviewers[1]:
        raise SystemExit("two distinct, identified reviewers are required")

    raw = csv_path.read_bytes()
    rows = list(csv.DictReader(raw.decode("utf-8").splitlines()))
    problems = []
    all_approved = True
    for index, row in enumerate(rows, 2):  # header is line 1
        for field in (*RATINGS, "resolution"):
            value = (row.get(field) or "").strip()
            allowed = ALLOWED_RESOLUTIONS if field == "resolution" else ALLOWED_RATINGS
            if not value:
                problems.append(f"line {index}: {field} is blank")
            elif value not in allowed:
                problems.append(f"line {index}: {field}={value!r} not in {sorted(allowed)}")
            elif value in ("fail", "rejected"):
                all_approved = False
    if problems:
        print(json.dumps({"status": "incomplete", "rows": len(rows),
                          "problems": problems[:20],
                          "problem_count": len(problems)}, indent=1))
        print("\nThe human review is not complete; no receipt was written.")
        return 1

    manifest_sha = hashlib.sha256((root / GRID_MANIFEST).read_bytes()).hexdigest()
    status = "approved" if all_approved else "completed_with_rejections"
    receipt = {
        "schema_version": 1,
        "artifact_type": "turkish_linguistic_review",
        "status": status,
        "reviewers": reviewers,
        "reviewer_qualification": "two_independent_turkish_proficient_reviewers",
        "dataset_manifest_sha256": manifest_sha,
        "samples": {"path": "review_sample.csv",
                    "sha256": hashlib.sha256(raw).hexdigest(),
                    "rows": len(rows)},
    }
    if args.fixture_only:
        receipt["fixture_only"] = True
    receipt_path = package / ("receipt.fixture.json" if args.fixture_only
                              else "receipt.json")
    receipt_path.write_text(json.dumps(receipt, indent=2) + "\n", encoding="utf-8")

    result = {"status": status, "rows": len(rows), "receipt": str(receipt_path)}
    if status == "approved":
        validated = validate_review(root, GRID_MANIFEST, receipt_path,
                                    target_selections(root, package.name),
                                    allow_fixture=args.fixture_only)
        result["gate_self_check"] = "PASS"
        result["sample_count"] = validated["sample_count"]
    else:
        result["gate_self_check"] = ("not run: rejected rows present; this "
                                     "receipt cannot authorize evaluation")
    print(json.dumps(result, indent=1))
    return 0 if status == "approved" else 2


if __name__ == "__main__":
    raise SystemExit(main())

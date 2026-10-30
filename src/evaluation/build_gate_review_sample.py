"""Build the gate-grade Turkish linguistic review samples.

These packages follow the execution-gate receipt contract
(``inference.linguistic_review.validate_review``): rows are actual frozen grid
prompts identified by (query_id, design_id), re-materialized and hash-checked,
with two-reviewer rating columns left BLANK — human judgments are never
simulated. Two packages are produced, one per gated run family, each sized to
the gate's operational minimum (>=20 distinct Turkish prompts per required
relation x presentation x name-control cell, >=100 overall, every requested n
and query type present in the sample):

  frozen_test    the default confirmatory selection (inference stage "full"):
                 spatial_front x {canonical, equivalent} x native_shared,
                 endpoint queries, sizes 4..1024 on the test split.
  training_eval  the union of the four evaluation cohorts in the frozen
                 training plan (results/training/plans/pod_pilot_002.json):
                 six relations x {canonical, equivalent, P1, P2} x
                 symbolic_shared, three query types, sizes 4..1024.

Sampling is deterministic (seed below), draws from the split the gated run
will actually evaluate (test), and concentrates reviewer effort at small n
while still covering every required size.

Run (pinned venv with vendored tiktoken==0.12.0):
    PYTHONPATH=src python3 -m evaluation.build_gate_review_sample \
        --package frozen_test --output-dir validation/gate_review
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import random
from collections import defaultdict
from pathlib import Path

from inference.dataset import GridDataset, default_selection
from inference.linguistic_review import RATINGS
from complete_grid import dataset_builder as grid

SEED = 20261029
GRID_MANIFEST = "data/complete_grid/manifest.json"
TRAINING_PLAN = "results/training/plans/pod_pilot_002.json"
SMALL_SIZES = (4, 8, 16, 32)
FIELDS = ("query_id", "design_id", "prompt_sha256", "prompt", "relation_id",
          "condition", "name_control", "query_type", "n", *RATINGS, "resolution")


def union_selection(selections: list[dict]) -> dict:
    axes = {"relations": set(), "conditions": set(), "name_controls": set(),
            "sizes": set(), "query_types": set()}
    for selection in selections:
        for key in axes:
            axes[key].update(selection[key])
    return {key: sorted(values) for key, values in axes.items()}


def package_selections(root: Path, package: str) -> list[dict]:
    if package == "frozen_test":
        return [default_selection("full")]
    plan = json.loads((root / TRAINING_PLAN).read_text(encoding="utf-8"))
    return [cohort["selection"] for cohort in plan["data"]["evaluation"].values()]


def candidate_rows(connection, relation, condition, control, sizes, query_types):
    placeholders_s = ",".join("?" * len(sizes))
    placeholders_q = ",".join("?" * len(query_types))
    sql = (
        "SELECT q.id AS query_id, d.id AS design_id, q.query_type, g.n "
        "FROM queries q JOIN groups g ON q.group_id = g.id "
        "JOIN designs d ON d.language='tr' AND d.relation_id=? "
        "AND d.condition=? AND d.name_control=? "
        f"WHERE g.split='test' AND g.n IN ({placeholders_s}) "
        f"AND q.query_type IN ({placeholders_q}) "
        "ORDER BY q.id, d.id")
    return connection.execute(sql, (relation, condition, control,
                                    *sizes, *query_types)).fetchall()


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--package", required=True,
                        choices=("frozen_test", "training_eval"))
    parser.add_argument("--output-dir", type=Path,
                        default=Path("validation/gate_review"))
    parser.add_argument("--per-cell", type=int, default=None,
                        help="reviewed prompts per template cell "
                             "(default: gate minimum, padded to 100 total)")
    args = parser.parse_args(argv)

    root = Path.cwd().resolve()
    selections = package_selections(root, args.package)
    union = union_selection(selections)
    cells = [(r, c, k) for r in union["relations"] for c in union["conditions"]
             for k in union["name_controls"]]
    per_cell = args.per_cell or max(20, -(-100 // len(cells)))  # >=100 overall
    rng = random.Random(SEED)

    out_dir = args.output_dir / args.package
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / "review_sample.csv"
    if csv_path.exists():
        raise SystemExit(f"{csv_path} already exists; review samples are "
                         "write-once — delete it deliberately to regenerate")

    chosen: list[tuple] = []           # (query_id, design_id, cell)
    seen: set[tuple[int, int]] = set()
    sizes_needed = set(union["sizes"])
    qt_needed = set(union["query_types"])

    with GridDataset(root, GRID_MANIFEST) as dataset:
        connection = dataset.connection
        by_cell = {}
        for cell in cells:
            rows = candidate_rows(connection, *cell, union["sizes"],
                                  union["query_types"])
            if not rows:
                raise SystemExit(f"no grid candidates for template cell {cell}")
            by_cell[cell] = rows

        # Per-cell quota, concentrated at small n for reviewer effort.
        for cell, rows in by_cell.items():
            small = [r for r in rows if r["n"] in SMALL_SIZES]
            pool = small if len(small) >= per_cell else rows
            for row in rng.sample(pool, per_cell):
                key = (row["query_id"], row["design_id"])
                if key in seen:
                    continue
                seen.add(key)
                chosen.append((*key, cell))
                sizes_needed.discard(row["n"])
                qt_needed.discard(row["query_type"])

        # Top up coverage: one extra row per still-missing size / query type,
        # spread across cells deterministically.
        for missing, field in ((sorted(sizes_needed), "n"),
                               (sorted(qt_needed), "query_type")):
            for index, value in enumerate(missing):
                cell = cells[index % len(cells)]
                extra = [r for r in by_cell[cell] if r[field] == value
                         and (r["query_id"], r["design_id"]) not in seen]
                if not extra:
                    for cell in cells:
                        extra = [r for r in by_cell[cell] if r[field] == value
                                 and (r["query_id"], r["design_id"]) not in seen]
                        if extra:
                            break
                if not extra:
                    raise SystemExit(f"cannot cover {field}={value}")
                row = rng.choice(extra)
                seen.add((row["query_id"], row["design_id"]))
                chosen.append((row["query_id"], row["design_id"], cell))

        tokenizers = grid.ModelTokenizers(root / "tokenizers")
        records = []
        for query_id, design_id, _ in sorted(chosen):
            sample = grid._materialized(root, connection, query_id, design_id,
                                        tokenizers)
            assert sample["language"] == "tr"
            records.append({
                "query_id": query_id, "design_id": design_id,
                "prompt_sha256": sample["prompt_sha256"],
                "prompt": sample["prompt"],
                "relation_id": sample["relation_id"],
                "condition": sample["condition"],
                "name_control": sample["name_control"],
                "query_type": sample["query_type"], "n": sample["n"],
                **{rating: "" for rating in RATINGS}, "resolution": "",
            })
        manifest_sha = dataset.manifest_sha256

    with csv_path.open("w", newline="", encoding="utf-8") as sink:
        writer = csv.DictWriter(sink, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(records)

    receipt_template = {
        "schema_version": 1,
        "artifact_type": "turkish_linguistic_review",
        "status": "pending_human_review",
        "reviewers": ["<reviewer 1 full name>", "<reviewer 2 full name>"],
        "reviewer_qualification": "two_independent_turkish_proficient_reviewers",
        "dataset_manifest_sha256": manifest_sha,
        "samples": {"path": "review_sample.csv", "sha256": None,
                    "rows": len(records)},
        "note": ("finalize_review_receipt sets status/sha256 only after every "
                 "row carries two complete human decisions; never edit this "
                 "file to 'approved' by hand"),
    }
    (out_dir / "receipt.pending.json").write_text(
        json.dumps(receipt_template, indent=2) + "\n", encoding="utf-8")

    coverage = defaultdict(int)
    for record in records:
        coverage[(record["relation_id"], record["condition"],
                  record["name_control"])] += 1
    summary = {
        "package": args.package, "rows": len(records),
        "cells": {" / ".join(cell): count for cell, count in sorted(coverage.items())},
        "sizes": sorted({record["n"] for record in records}),
        "query_types": sorted({record["query_type"] for record in records}),
        "csv_sha256": hashlib.sha256(csv_path.read_bytes()).hexdigest(),
        "grid_manifest_sha256": manifest_sha,
    }
    (out_dir / "sample_summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({k: summary[k] for k in
                      ("package", "rows", "sizes", "query_types")}, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

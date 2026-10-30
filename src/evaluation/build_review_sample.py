"""Build the stratified Turkish human-review sample for the P1 revision.

Draws from the frozen P1 manifest (never from model outputs), pairs every
transformed Turkish prompt with its canonical counterpart for the same query,
and oversamples P3 (the constituent-order transformation the audit flagged).
Annotation columns are left empty: reviewer judgments are never simulated.

Run:  PYTHONPATH=src python3 -m evaluation.build_review_sample \
          --manifest data/manifests/p1_expanded_manifest.jsonl \
          --output validation/turkish_linguistic_review.csv
"""

from __future__ import annotations

import argparse
import collections
import csv
import json
import random
from pathlib import Path

SEED = 20261028
BASE_PER_CELL = 4          # rows per (relation x TR condition)
P3_EXTRA_PER_RELATION = 12  # oversampling on top of the base quota
GOLD_TEXT = {True: "YES/EVET", False: "NO/HAYIR"}


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    rows = [json.loads(line) for line in args.manifest.read_text(encoding="utf-8").splitlines()
            if line.strip()]
    canonical = {(r["query_instance_id"], r["relation_id"], r["name_control"]): r
                 for r in rows if r["language"] == "tr" and r["condition"] == "canonical"}
    pool = collections.defaultdict(list)
    for r in rows:
        if r["language"] != "tr" or r["name_control"] != "native_shared":
            continue
        key = (r["query_instance_id"], r["relation_id"], r["name_control"])
        if r["condition"] != "canonical" and key not in canonical:
            continue
        pool[(r["relation_id"], r["condition"])].append(r)
    rng = random.Random(SEED)
    sample = []
    for (relation, condition), members in sorted(pool.items()):
        quota = BASE_PER_CELL + (P3_EXTRA_PER_RELATION if condition == "P3" else 0)
        members = sorted(members, key=lambda r: r["prompt_id"])
        # spread over sizes, then gold-balance inside the quota
        by_n = collections.defaultdict(list)
        for r in members:
            by_n[r["n"]].append(r)
        chosen = []
        sizes = sorted(by_n)
        while len(chosen) < min(quota, len(members)):
            for n in sizes:
                if by_n[n] and len(chosen) < quota:
                    chosen.append(by_n[n].pop(rng.randrange(len(by_n[n]))))
        sample.extend(chosen)
    fields = ["prompt_id", "original_sentence", "transformed_sentence",
              "intended_meaning", "presentation_condition", "relation_family",
              "relation_id", "n", "grammaticality", "naturalness",
              "semantic_equivalence", "ambiguity", "reviewer_comments"]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="", encoding="utf-8") as sink:
        writer = csv.DictWriter(sink, fieldnames=fields)
        writer.writeheader()
        for r in sample:
            key = (r["query_instance_id"], r["relation_id"], r["name_control"])
            base = canonical.get(key)
            writer.writerow({
                "prompt_id": r["prompt_id"],
                "original_sentence": (base or r)["prompt"],
                "transformed_sentence": r["prompt"] if r["condition"] != "canonical" else "",
                "intended_meaning": (
                    f"Strict total order; the question asks whether the stated relation holds "
                    f"between the two queried entities. Correct answer: {GOLD_TEXT[r['gold_label']]}."),
                "presentation_condition": r["condition"],
                "relation_family": r["relation_family"],
                "relation_id": r["relation_id"], "n": r["n"],
                "grammaticality": "", "naturalness": "",
                "semantic_equivalence": "", "ambiguity": "", "reviewer_comments": "",
            })
    counts = collections.Counter((r["relation_id"], r["condition"]) for r in sample)
    print(json.dumps({"rows": len(sample),
                      "p3_rows": sum(v for (rel, c), v in counts.items() if c == "P3"),
                      "cells": len(counts)}, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

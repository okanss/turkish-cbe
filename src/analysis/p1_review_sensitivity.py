"""Review-driven sensitivity reanalysis of the P1 P3 contrast.

The completed two-reviewer linguistic review excluded one construction on
grammaticality: the P3 realization of ``temporal_completion``. This script
recomputes the query-matched TR canonical-vs-P3 paired contrast per model with
that relation's pairs removed, using the same audited primitives and pair
construction as ``analysis.p1_statistics`` (latent-cluster bootstrap CI;
ordinary McNemar stays undefined for these pooled pairs).

Run:  PYTHONPATH=src python3 -m analysis.p1_review_sensitivity \
          --manifest data/manifests/p1_expanded_manifest.jsonl \
          --results results/p1/qwen2-5-0-5b.jsonl results/p1/qwen2-5-3b.jsonl \
                    results/p1/qwen2-5-7b.jsonl results/p1/qwen2-5-14b.jsonl \
          --output results/p1/analysis/p3_review_sensitivity.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from analysis.p1_statistics import _pairs, load_results
from analysis.statistics import bootstrap

EXCLUDED_RELATION = "temporal_completion"
LEFT = ("tr", "canonical", "native_shared")
RIGHT = ("tr", "P3", "native_shared")


def contrast(pairs):
    def stat(rows):
        valid = [r for r in rows if r["original"] is not None and r["equivalent"] is not None]
        if not valid:
            return {"delta_accuracy": None}
        return {"delta_accuracy": (sum(r["equivalent"] == r["gold"] for r in valid)
                                   - sum(r["original"] == r["gold"] for r in valid)) / len(valid)}
    intervals = bootstrap(pairs, stat)
    valid = [p for p in pairs if p["original"] is not None and p["equivalent"] is not None]
    point = stat(pairs)["delta_accuracy"]
    return {"n_pairs": len(pairs), "n_valid_pairs": len(valid),
            "delta_accuracy": None if point is None else round(point, 4),
            "boot_low": intervals["delta_accuracy"].get("low"),
            "boot_high": intervals["delta_accuracy"].get("high"),
            "inference": "cluster_bootstrap_contrast_exploratory"}


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--results", type=Path, nargs="+", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    manifest = [json.loads(line) for line in
                args.manifest.read_text(encoding="utf-8").splitlines() if line.strip()]
    kept = [row for row in manifest if row["relation_id"] != EXCLUDED_RELATION]
    report = {"excluded_relation": EXCLUDED_RELATION,
              "reason": "two-reviewer linguistic review: P3 realization failed "
                        "grammaticality for one reviewer; predeclared policy excludes it",
              "comparison": "F2_tr_canonical_vs_P3", "models": {}}
    for path in args.results:
        header, records = load_results(path)
        report["models"][header["model"]] = {
            "all_relations": contrast(_pairs(manifest, records, LEFT, RIGHT)),
            "excluding_temporal_completion": contrast(_pairs(kept, records, LEFT, RIGHT)),
        }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

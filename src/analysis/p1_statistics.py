"""P1 statistical analysis over the supplemental local (Ollama) evaluation.

Reuses the audited primitives — ``analysis.metrics.summarize_pairs`` (R/D/U,
invariance, pooled-marginal kappa with its undefined semantics, all-planned vs
valid-pair estimands, coverage), ``analysis.statistics.mcnemar``,
``holm_adjust``, ``wilson_interval`` and the latent-cluster ``bootstrap`` — so
no metric is redefined for this arm. Prompts sharing a latent order are never
treated as independent: every bootstrap resamples latent clusters.

Inputs: the frozen P1 manifest plus one runner JSONL per model
(``results/p1/<model>.jsonl``). Outputs: CSV tables under
``results/p1/analysis/``. Results are labeled supplemental/quantized and are
never merged with registered-provider runs.

Predeclared Holm families (declared before any expanded outcome is read;
one family per model, pooled over strata, paired at the query level):
  F1 language_gap:    TR canonical vs EN canonical.
  F2 presentation:    TR canonical vs each of {equivalent, P1..P5} (6 tests).
  F3 naming:          native vs {en_transliterated, symbolic_shared} on the
                      naming component's EN+TR canonical pairs (2 tests).

Run:  PYTHONPATH=src python3 -m analysis.p1_statistics \
          --manifest data/manifests/p1_expanded_manifest.jsonl \
          --results results/p1/qwen25-0.5b.jsonl results/p1/qwen25-3b.jsonl \
          --output results/p1/analysis
"""

from __future__ import annotations

import argparse
import collections
import csv
import json
from pathlib import Path

from analysis.metrics import summarize_pairs
from analysis.statistics import bootstrap, holm_adjust, mcnemar, wilson_interval

VERSION = "p1-0.2.0"
TR_CONDITIONS = ("equivalent", "P1", "P2", "P3", "P4", "P5")
AXES = ("language", "condition", "relation_family", "n", "name_control",
        "query_type", "query_distance")


def load_results(path: Path) -> tuple[dict, dict]:
    header, records = None, {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        if row["record_kind"] == "run_header":
            header = row
        elif row["record_kind"] == "inference":
            records[row["prompt_id"]] = row  # resume may duplicate; last wins
    if header is None or not records:
        raise ValueError(f"{path} has no run header or no inference records")
    return header, records


def accuracy_rows(model: str, manifest: list[dict], records: dict) -> list[dict]:
    out = []
    for axis in AXES:
        groups = collections.defaultdict(list)
        for row in manifest:
            record = records.get(row["prompt_id"])
            groups[(row["language"], row[axis])].append(record)
        for (language, value), members in sorted(groups.items(), key=str):
            present = [m for m in members if m is not None]
            valid = [m for m in present if m["parse_valid"]]
            correct = sum(1 for m in present if m["correct"])
            interval = wilson_interval(correct, len(members)) if members else {}
            out.append({
                "model": model, "axis": axis, "language": language, "value": value,
                "n_planned": len(members), "n_missing": len(members) - len(present),
                "n_invalid": len(present) - len(valid),
                "all_planned_accuracy": round(correct / len(members), 4) if members else None,
                "wilson_low": interval.get("low"), "wilson_high": interval.get("high"),
                "wilson_note": "binomial summary; rows share latent orders - see cluster bootstrap",
            })
    return out


def _pairs(manifest, records, left_key, right_key):
    cells = collections.defaultdict(dict)
    for row in manifest:
        record = records.get(row["prompt_id"])
        key = (row["query_instance_id"], row["relation_id"])
        cells[key][(row["language"], row["condition"], row["name_control"])] = (row, record)
    pairs = []
    for key, members in sorted(cells.items(), key=str):
        if left_key in members and right_key in members:
            (lrow, lrec), (rrow, rrec) = members[left_key], members[right_key]
            pairs.append({
                "cluster_id": lrow["latent_instance_id"], "n": lrow["n"],
                "gold": lrow["gold_label"],
                "original": None if lrec is None or not lrec["parse_valid"] else lrec["prediction"],
                "equivalent": None if rrec is None or not rrec["parse_valid"] else rrec["prediction"],
            })
    return pairs


def paired_tables(model: str, manifest, records):
    comparisons = []
    comparisons.append(("F1_language_gap", ("en", "canonical", "native_shared"),
                        ("tr", "canonical", "native_shared")))
    for condition in TR_CONDITIONS:
        comparisons.append((f"F2_tr_canonical_vs_{condition}",
                            ("tr", "canonical", "native_shared"),
                            ("tr", condition, "native_shared")))
    for control in ("en_transliterated", "symbolic_shared"):
        comparisons.append((f"F3_naming_{control}",
                            ("tr", "canonical", "native_shared"),
                            ("tr", "canonical", control)))
    profile_rows, test_rows = [], []
    p_values_by_family = collections.defaultdict(dict)
    for name, left, right in comparisons:
        pairs = _pairs(manifest, records, left, right)
        if not pairs:
            continue
        # kappa is defined per fixed-complexity stratum (manuscript Section 4.2);
        # the audited summarizer enforces this, so profiles are per n, plus one
        # pooled descriptive row that carries no kappa.
        by_n = collections.defaultdict(list)
        for pair in pairs:
            by_n[pair["n"]].append(pair)
        stratum_summaries = {n: summarize_pairs(members) for n, members in sorted(by_n.items())}
        valid_pairs = [p for p in pairs if p["original"] is not None and p["equivalent"] is not None]
        pooled = {
            "n_pairs": len(pairs), "n_valid_pairs": len(valid_pairs),
            "valid_pair_coverage": round(len(valid_pairs) / len(pairs), 4),
            "R": round(sum(p["original"] == p["gold"] == p["equivalent"] for p in valid_pairs)
                       / len(valid_pairs), 4) if valid_pairs else None,
            "D": round(sum(p["original"] != p["gold"] and p["equivalent"] != p["gold"]
                           for p in valid_pairs) / len(valid_pairs), 4) if valid_pairs else None,
            "U": round(sum((p["original"] == p["gold"]) != (p["equivalent"] == p["gold"])
                           for p in valid_pairs) / len(valid_pairs), 4) if valid_pairs else None,
            "invariance": None, "q": None, "kappa": None,
            "mean_accuracy": round(sum((p["original"] == p["gold"]) + (p["equivalent"] == p["gold"])
                                       for p in valid_pairs) / (2 * len(valid_pairs)), 4) if valid_pairs else None,
            "strict_robust_correctness": round(
                sum(p["original"] == p["gold"] == p["equivalent"] for p in valid_pairs)
                / len(pairs), 4) if pairs else None,
            "undefined_reasons": {"kappa": "pooled_across_n; defined per stratum only",
                                  "invariance": "pooled_across_n; defined per stratum only",
                                  "q": "pooled_across_n; defined per stratum only"},
        }

        def pair_stats(rows):
            valid = [r for r in rows if r["original"] is not None and r["equivalent"] is not None]
            if not valid:
                return {"R": None, "delta_accuracy": None}
            return {
                "R": sum(1 for r in valid if r["original"] == r["gold"]
                         and r["equivalent"] == r["gold"]) / len(valid),
                # Paired contrast right minus left; the latent-cluster bootstrap
                # CI on this quantity is the inference when ordinary McNemar is
                # undefined because latent clusters repeat across pairs (same
                # fallback as the audited pipeline's exploratory_bootstrap_contrast).
                "delta_accuracy": (sum(r["equivalent"] == r["gold"] for r in valid)
                                   - sum(r["original"] == r["gold"] for r in valid)) / len(valid),
            }

        intervals = bootstrap(pairs, pair_stats)
        valid = [p for p in pairs if p["original"] is not None and p["equivalent"] is not None]
        left_correct = [p["original"] == p["gold"] for p in valid]
        right_correct = [p["equivalent"] == p["gold"] for p in valid]
        clusters = len({p["cluster_id"] for p in pairs})
        test = mcnemar(left_correct, right_correct,
                       clusters=[p["cluster_id"] for p in valid])
        for stratum, summary in [("pooled", pooled)] + [
                (str(n), s) for n, s in stratum_summaries.items()]:
            profile_rows.append({
                "model": model, "comparison": name, "stratum_n": stratum,
                "left": "/".join(left), "right": "/".join(right),
                "n_pairs": summary["n_pairs"], "n_valid_pairs": summary["n_valid_pairs"],
                "n_clusters": clusters if stratum == "pooled" else None,
                "valid_pair_coverage": summary["valid_pair_coverage"],
                "R": summary["R"], "D": summary["D"], "U": summary["U"],
                "invariance": summary["invariance"], "q": summary["q"],
                "kappa": summary["kappa"],
                "kappa_undefined_reason": summary["undefined_reasons"].get("kappa"),
                "mean_accuracy": summary["mean_accuracy"],
                "strict_robust_correctness": summary["strict_robust_correctness"],
                "R_boot_low": intervals.get("R", {}).get("low") if stratum == "pooled" else None,
                "R_boot_high": intervals.get("R", {}).get("high") if stratum == "pooled" else None,
            })
        family = name.split("_", 1)[0]
        p_values_by_family[(model, family)][name] = test.get("p_value")
        delta = intervals.get("delta_accuracy", {})
        valid_count = len(valid)
        delta_point = (pair_stats(pairs)["delta_accuracy"] if valid_count else None)
        test_rows.append({"model": model, "comparison": name, "family": family,
                          **{k: test.get(k) for k in ("p_value", "b", "c",
                                                      "discordant", "method", "status")},
                          "delta_accuracy": round(delta_point, 4) if delta_point is not None else None,
                          "delta_boot_low": delta.get("low"),
                          "delta_boot_high": delta.get("high"),
                          "delta_boot_status": delta.get("status"),
                          "inference": ("mcnemar_holm" if test.get("p_value") is not None
                                        else "cluster_bootstrap_contrast_exploratory")})
    adjusted = {}
    for (model_name, family), values in p_values_by_family.items():
        adjusted.update(holm_adjust(values))
    for row in test_rows:
        row["holm_adjusted_p"] = adjusted.get(row["comparison"])
    return profile_rows, test_rows


def write_csv(path: Path, rows: list[dict]):
    if not rows:
        return
    fields = sorted({key for row in rows for key in row})
    with path.open("w", newline="", encoding="utf-8") as sink:
        writer = csv.DictWriter(sink, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--results", type=Path, nargs="+", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    manifest = [json.loads(line) for line in
                args.manifest.read_text(encoding="utf-8").splitlines() if line.strip()]
    args.output.mkdir(parents=True, exist_ok=True)
    all_accuracy, all_profiles, all_tests, coverage = [], [], [], []
    for path in args.results:
        header, records = load_results(path)
        model = header["model"]
        missing = [row["prompt_id"] for row in manifest if row["prompt_id"] not in records]
        coverage.append({"model": model, "digest": header["digest"],
                         "quantization": header.get("quantization"),
                         "planned": len(manifest), "answered": len(records),
                         "missing": len(missing),
                         "invalid": sum(1 for r in records.values() if not r["parse_valid"]),
                         "arm_label": header.get("arm_label")})
        all_accuracy.extend(accuracy_rows(model, manifest, records))
        profiles, tests = paired_tables(model, manifest, records)
        all_profiles.extend(profiles)
        all_tests.extend(tests)
    write_csv(args.output / "p1_coverage.csv", coverage)
    write_csv(args.output / "p1_accuracy_by_axis.csv", all_accuracy)
    write_csv(args.output / "p1_paired_profiles.csv", all_profiles)
    write_csv(args.output / "p1_paired_tests.csv", all_tests)
    print(json.dumps({"models": [c["model"] for c in coverage],
                      "tables": ["p1_coverage", "p1_accuracy_by_axis",
                                 "p1_paired_profiles", "p1_paired_tests"],
                      "output": str(args.output)}, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

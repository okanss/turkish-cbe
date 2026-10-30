"""Build the frozen P1 expanded-evaluation manifest (pilot split only).

Sampling is at the matched-GROUP level, never at the prompt level, so every
selected query contributes its complete presentation set and all paired
comparisons remain intact. The frozen test split is never touched; original
partitions are read, not modified. Output is write-once.

Components (all native_shared names, drawn from data/pilot/prompts.jsonl):
  A. nine_way: EN{P0,eq} + TR{P0,eq,P1..P5} complete groups, all 9 relations,
     n in {4..128}; stratified subsample per (relation, n), gold-balanced.
  B. geometry: spatial_front 4-way groups (EN/TR x P0/eq) for internal and
     distance-controlled queries; internal taken fully, distance-controlled
     stratified by (n, distance), gold-balanced.
  C. naming: groups complete across all three name controls for EN/TR
     canonical (6 cells); taken fully if small.

Run:  PYTHONPATH=src python3 -m evaluation.build_p1_manifest \
          --output data/manifests/p1_expanded_manifest.jsonl
"""

from __future__ import annotations

import argparse
import collections
import hashlib
import json
import random
from pathlib import Path

from renderers.turkish import RELATIONS

BUILDER_VERSION = "p1-0.1.0"
SEED = 20261028
NINE = (("en", "canonical"), ("en", "equivalent"), ("tr", "canonical"),
        ("tr", "equivalent"), ("tr", "P1"), ("tr", "P2"), ("tr", "P3"),
        ("tr", "P4"), ("tr", "P5"))
FOUR = NINE[:4]
TWO = (("en", "canonical"), ("tr", "canonical"))
NAME_CONTROLS = ("native_shared", "en_transliterated", "symbolic_shared")

# Per-stratum group quotas (counts of matched groups, not prompts).
NINE_WAY_PER_RELATION_N = 25      # x 9 relations x 6 sizes = 1,350 groups = 12,150 calls
DC_GROUPS = 450                   # distance-controlled 4-way groups = 1,800 calls
# internal 4-way groups (300) and naming groups are taken exhaustively.


def load_bank(root: Path) -> list[dict]:
    rows = []
    for line in (root / "data/pilot/prompts.jsonl").read_text(encoding="utf-8").splitlines():
        if line.strip():
            row = json.loads(line)
            if row["split"] != "pilot":
                raise ValueError("pilot bank contains a non-pilot row")
            rows.append(row)
    return rows


def index_groups(rows, name_control):
    groups = collections.defaultdict(dict)
    for row in rows:
        if row["name_control"] != name_control:
            continue
        key = (row["query_instance_id"], row["relation_id"])
        groups[key][(row["language"], row["condition"])] = row
    return groups


def _balanced_sample(rng, candidates, quota, gold_of):
    """Sample `quota` keys, gold-balanced as far as the pool allows."""
    byg = {True: [], False: []}
    for key in sorted(candidates):
        byg[gold_of(key)].append(key)
    half, rem = divmod(quota, 2)
    take_true = min(half + rem, len(byg[True]))
    take_false = min(quota - take_true, len(byg[False]))
    if take_true + take_false < quota:  # top up from whichever side has spare
        extra = quota - take_true - take_false
        spare_true = len(byg[True]) - take_true
        add_true = min(extra, spare_true)
        take_true += add_true
        take_false += min(extra - add_true, len(byg[False]) - take_false)
    chosen = rng.sample(byg[True], take_true) + rng.sample(byg[False], take_false)
    return chosen, len(candidates)


def build(root: Path) -> tuple[list[dict], dict]:
    bank = load_bank(root)
    native = index_groups(bank, "native_shared")
    rng = random.Random(SEED)
    selected: list[tuple] = []
    protocol: dict = {"seed": SEED, "builder_version": BUILDER_VERSION,
                      "strata": [], "components": {}}

    nine_complete = {k: v for k, v in native.items() if all(c in v for c in NINE)}
    by_rel_n = collections.defaultdict(list)
    for key, cells in nine_complete.items():
        row = cells[("tr", "canonical")]
        by_rel_n[(key[1], row["n"])].append(key)
    a_groups = 0
    for (relation, n), keys in sorted(by_rel_n.items()):
        chosen, pool = _balanced_sample(
            rng, keys, NINE_WAY_PER_RELATION_N,
            lambda k: nine_complete[k][("tr", "canonical")]["gold_label"])
        protocol["strata"].append({"component": "nine_way", "relation": relation,
                                   "n": n, "pool": pool, "sampled": len(chosen),
                                   "probability": round(len(chosen) / pool, 4)})
        for key in chosen:
            selected.append(("nine_way", key, NINE, "native_shared"))
        a_groups += len(chosen)
    protocol["components"]["nine_way"] = {"groups": a_groups, "calls": a_groups * 9}

    four_only = {k: v for k, v in native.items()
                 if all(c in v for c in FOUR) and k not in nine_complete}
    internal = sorted(k for k in four_only
                      if four_only[k][("tr", "canonical")]["query_type"] == "internal")
    for key in internal:
        selected.append(("geometry_internal", key, FOUR, "native_shared"))
    protocol["strata"].append({"component": "geometry_internal", "pool": len(internal),
                               "sampled": len(internal), "probability": 1.0})
    dc = [k for k in four_only
          if four_only[k][("tr", "canonical")]["query_type"] == "distance_controlled"]
    by_nd = collections.defaultdict(list)
    for key in dc:
        row = four_only[key][("tr", "canonical")]
        by_nd[(row["n"], row["query_distance"])].append(key)
    per_cell = max(1, DC_GROUPS // max(1, len(by_nd)))
    dc_taken = 0
    for (n, d), keys in sorted(by_nd.items()):
        chosen, pool = _balanced_sample(
            rng, keys, min(per_cell, len(keys)),
            lambda k: four_only[k][("tr", "canonical")]["gold_label"])
        protocol["strata"].append({"component": "geometry_distance", "n": n,
                                   "distance": d, "pool": pool, "sampled": len(chosen),
                                   "probability": round(len(chosen) / pool, 4)})
        for key in chosen:
            selected.append(("geometry_distance", key, FOUR, "native_shared"))
        dc_taken += len(chosen)
    protocol["components"]["geometry"] = {
        "internal_groups": len(internal), "distance_groups": dc_taken,
        "calls": (len(internal) + dc_taken) * 4}

    per_control = {nc: index_groups(bank, nc) for nc in NAME_CONTROLS}
    naming = sorted(k for k in per_control["native_shared"]
                    if all(all(c in per_control[nc].get(k, {}) for c in TWO)
                           for nc in NAME_CONTROLS))
    for key in naming:
        for nc in NAME_CONTROLS:
            selected.append(("naming", key, TWO, nc))
    protocol["strata"].append({"component": "naming", "pool": len(naming),
                               "sampled": len(naming), "probability": 1.0})
    protocol["components"]["naming"] = {"groups": len(naming),
                                        "calls": len(naming) * len(TWO) * len(NAME_CONTROLS)}

    records, seen = [], set()
    for component, key, cells, name_control in selected:
        group = per_control[name_control][key]
        for cell in cells:
            row = group[cell]
            if row["prompt_id"] in seen:   # naming overlaps nine_way at native_shared
                continue
            seen.add(row["prompt_id"])
            records.append({
                "prompt_id": row["prompt_id"],
                "latent_instance_id": row["latent_instance_id"],
                "query_instance_id": row["query_instance_id"],
                "grid_query_id": row.get("grid_query_id"),
                "split": row["split"], "language": row["language"],
                "condition": row["condition"], "relation_id": row["relation_id"],
                "relation_family": RELATIONS[row["relation_id"]].family,
                "n": row["n"], "name_control": row["name_control"],
                "query_type": row["query_type"],
                "query_distance": row["query_distance"],
                "query_direction": row["query_direction"],
                "gold_label": row["gold_label"],
                "display_query_subject": row["display_query_subject"],
                "display_query_object": row["display_query_object"],
                "sampling_stratum": component,
                "prompt": row["prompt"], "prompt_sha256": row["prompt_sha256"],
            })
    records.sort(key=lambda r: r["prompt_id"])
    protocol["total_prompts"] = len(records)
    gold = collections.Counter(r["gold_label"] for r in records)
    protocol["gold_balance"] = dict(gold)
    return records, protocol


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path.cwd())
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    output = args.output if args.output.is_absolute() else args.root / args.output
    if output.exists():
        raise SystemExit("error: manifest already exists; manifests are write-once")
    records, protocol = build(args.root)
    output.parent.mkdir(parents=True, exist_ok=True)
    body = "".join(json.dumps(r, ensure_ascii=False, sort_keys=True) + "\n" for r in records)
    output.write_text(body, encoding="utf-8")
    protocol["manifest_sha256"] = hashlib.sha256(body.encode("utf-8")).hexdigest()
    protocol_path = output.with_suffix(".protocol.json")
    protocol_path.write_text(json.dumps(protocol, indent=1, ensure_ascii=False) + "\n")
    print(json.dumps({"total_prompts": protocol["total_prompts"],
                      "components": protocol["components"],
                      "gold_balance": protocol["gold_balance"],
                      "manifest_sha256": protocol["manifest_sha256"]}, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""Read-only, manifest-bound cohorts shared by every evaluated model."""

from __future__ import annotations

from collections import Counter
import hashlib
import json
from pathlib import Path

from complete_grid import dataset_builder as grid
from complete_grid.grid_design import RELATION_IDS, QUERY_TYPES


def fingerprint(value):
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True,
                                    separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def default_selection(stage="pilot"):
    if stage not in {"pilot", "full"}:
        raise ValueError("stage must be pilot or full")
    return {"split": "pilot" if stage == "pilot" else "test",
            "sizes": [4, 8] if stage == "pilot" else [4,8,16,32,64,128,256,512,1024],
            "groups_per_size": 4 if stage == "pilot" else None,
            "query_types": ["endpoint"], "distances": None,
            "name_controls": ["native_shared"], "relations": ["spatial_front"],
            "languages": ["en", "tr"], "conditions": ["canonical", "equivalent"]}


def validate_selection(value):
    if not isinstance(value, dict) or set(value) != set(default_selection()):
        raise ValueError("unknown or missing cohort selection fields")
    if value["split"] not in {"pilot", "development", "test"}:
        raise ValueError("training data cannot be evaluated as a held-out cohort")
    allowed = {"sizes": (4,8,16,32,64,128,256,512,1024),
               "query_types": QUERY_TYPES,
               "name_controls": ("native_shared", "en_transliterated", "symbolic_shared"),
               "relations": RELATION_IDS, "languages": ("en", "tr"),
               "conditions": ("canonical", "equivalent", "P1", "P2", "P3", "P4", "P5")}
    for key, choices in allowed.items():
        items = value[key]
        item_type = int if key == "sizes" else str
        if (not isinstance(items, list) or not items
                or any(type(item) is not item_type for item in items)
                or len(set(items)) != len(items) or any(item not in choices for item in items)):
            raise ValueError("invalid selection: " + key)
    count = value["groups_per_size"]
    if count is not None and (type(count) is not int or count < 4 or count % 4):
        raise ValueError("subsample size must be a multiple of four to preserve answer cells")
    distances = value["distances"]
    if distances is not None and (not isinstance(distances, list) or not distances
            or any(type(d) is not int or d < 1 for d in distances)
            or len(set(distances)) != len(distances)):
        raise ValueError("invalid controlled distances")
    if not {"canonical", "equivalent"} <= set(value["conditions"]):
        raise ValueError("canonical and equivalent presentations must stay together")


class GridDataset:
    def __init__(self, root, manifest):
        self.root = Path(root).resolve()
        self.manifest_path = Path(manifest)
        if not self.manifest_path.is_absolute():
            self.manifest_path = self.root / self.manifest_path
        self.connection = None
        self.tokenizers = None

    def __enter__(self):
        self.root, self.manifest, self.connection = grid._load_manifest(
            self.root, self.manifest_path)
        try:
            # Rendering must use the exact sources bound to the frozen bank.
            if (grid._source_hashes() != self.manifest["source_files_sha256"]
                    or grid._base_binding(self.root) != self.manifest["base_binding"]):
                raise ValueError("frozen dataset rendering sources changed")
            grid._check_schema(self.connection)
            self.manifest_sha256 = hashlib.sha256(self.manifest_path.read_bytes()).hexdigest()
        except BaseException:
            self.connection.close()
            self.connection = None
            raise
        return self

    def __exit__(self, *args):
        self.connection.close()

    def select(self, selection):
        validate_selection(selection)
        c = self.connection
        groups = []
        for n in sorted(selection["sizes"]):
            rows = c.execute("SELECT id,latent_instance_id,sample_index FROM groups "
                             "WHERE split=? AND n=? ORDER BY sample_index,id",
                             (selection["split"], n)).fetchall()
            count = selection["groups_per_size"]
            if not rows or (count is not None and count > len(rows)):
                raise ValueError(f"requested groups unavailable for n={n}")
            rows = rows if count is None else rows[:count]
            groups.extend({"id": r[0], "latent_instance_id": r[1], "n": n,
                           "sample_index": r[2]} for r in rows)
        group_ids = {r["id"] for r in groups}
        queries = []
        for row in c.execute("SELECT * FROM queries ORDER BY id"):
            if row["group_id"] not in group_ids or row["query_type"] not in selection["query_types"]:
                continue
            if (row["query_type"] == "distance_controlled" and selection["distances"] is not None
                    and row["query_distance"] not in selection["distances"]):
                continue
            queries.append(dict(row))
        if not queries:
            raise ValueError("selection has no valid queries")
        if {r["group_id"] for r in queries} != group_ids:
            raise ValueError("query filters would silently drop selected latent groups")
        designs = [dict(r) for r in c.execute("SELECT * FROM designs ORDER BY id")
                   if r["name_control"] in selection["name_controls"]
                   and r["relation_id"] in selection["relations"]
                   and r["language"] in selection["languages"]
                   and r["condition"] in selection["conditions"]]
        expected = len(selection["name_controls"]) * len(selection["relations"]) * sum(
            1 for language in selection["languages"] for condition in selection["conditions"]
            if language == "tr" or condition in {"canonical", "equivalent"})
        if len(designs) != expected or not designs:
            raise ValueError("incomplete selected presentation matrix")
        by_group = {g["id"]: g for g in groups}
        strata = Counter()
        cells = Counter()
        for q in queries:
            n = by_group[q["group_id"]]["n"]
            controlled = q["query_distance"] if q["query_type"] == "distance_controlled" else None
            stratum = f"{n}|{q['query_type']}|{controlled}"
            strata[stratum] += 1
            cells[f"{stratum}|{q['query_direction']}|{bool(q['gold_label'])}"] += 1
        answer_cells = {key: {suffix: cells.get(key+"|"+suffix, 0) for suffix in (
            "forward|True", "forward|False", "reverse|True", "reverse|False")} for key in strata}
        if any(max(values.values())-min(values.values()) > 1 for values in answer_cells.values()):
            raise ValueError("selected source query-answer cells are unbalanced")
        cohort = {"dataset_manifest_sha256": self.manifest_sha256,
                  "dataset_version": self.manifest["grid_version"],
                  "selection": selection, "groups": groups,
                  "query_ids": [q["id"] for q in queries], "design_ids": [d["id"] for d in designs],
                  "group_count": len(groups), "query_count": len(queries),
                  "designs_per_query": len(designs), "prompt_count": len(queries)*len(designs),
                  "query_strata": dict(sorted(strata.items())), "answer_cells": answer_cells,
                  "human_review_status": self.manifest["human_review_status"],
                  "eligible_for_confirmatory_evaluation": False}
        cohort["cohort_sha256"] = fingerprint(cohort)
        return cohort

    def _check_cohort(self, cohort):
        if cohort != self.select(cohort["selection"]):
            raise ValueError("cohort does not match frozen source selection")

    def source_lengths(self, cohort):
        """Stored old-prompt lengths are budget proxies, never new chat counts."""
        self._check_cohort(cohort)
        totals = {"prompts": 0, "cl100k_base": 0, "o200k_base": 0,
                  "utf8_bytes": 0, "max_cl100k_base": 0, "max_o200k_base": 0}
        dids = cohort["design_ids"]
        qids = cohort["query_ids"]
        for start in range(0, len(qids), 500):
            batch = qids[start:start+500]
            sql = ("SELECT count(*),sum(cl100k_base),sum(o200k_base),sum(utf8_bytes),"
                   "max(cl100k_base),max(o200k_base) FROM measurements WHERE query_id IN ("
                   + ",".join("?" for _ in batch) + ") AND design_id IN ("
                   + ",".join("?" for _ in dids) + ")")
            row = self.connection.execute(sql, batch+dids).fetchone()
            for key, value in zip(totals, row):
                if key.startswith("max_"):
                    totals[key] = max(totals[key], value or 0)
                else:
                    totals[key] += value or 0
        if totals["prompts"] != cohort["prompt_count"]:
            raise ValueError("missing source measurements in selected cohort")
        return totals

    def iter_rows(self, cohort):
        self._check_cohort(cohort)
        if self.tokenizers is None:
            self.tokenizers = grid.ModelTokenizers(self.root / "tokenizers")
        for qid in cohort["query_ids"]:
            for did in cohort["design_ids"]:
                yield grid._materialized(self.root, self.connection, qid, did, self.tokenizers)

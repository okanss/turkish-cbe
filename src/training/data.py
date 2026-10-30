"""Read-only training cohorts and exact, disjoint supervised-view exports.

The frozen inference selector intentionally rejects training. This module opens
the same fully verified database, selects the existing training partition with
SQL, and calls the frozen renderer without changing its validation rules.
"""

from collections import Counter, defaultdict
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import random
import re
import tempfile
import shutil
import unicodedata

from analysis.lexical_bias.catalog import build_catalog
from complete_grid import dataset_builder as grid
from inference.dataset import GridDataset, fingerprint
from inference.prompts import make_messages
from renderers.morphology import turkish_lower
from renderers.turkish import RELATIONS as TR_RELATIONS
from renderers.english import RELATIONS as EN_RELATIONS


DATA_VERSION = "0.9.0"
TRAIN_RELATIONS = ("comparison_height", "comparison_speed", "comparison_weight")
HELDOUT_RELATIONS = ("spatial_front", "spatial_right", "spatial_above")
CONDITIONS = ("canonical", "equivalent", "P1", "P2")
QUERY_TYPES = ("endpoint", "internal", "distance_controlled")
_SIZES = (4, 8, 16, 32, 64, 128, 256, 512, 1024)
_FIELDS = {"training_sizes", "groups_per_size", "training_relations", "heldout_relations",
    "query_types", "conditions", "evaluation_sizes", "evaluation_groups_per_size", "evaluation_split",
    "grpo_min_n", "grpo_max_instances", "grpo_rollouts_per_view"}
_STOP = {"tr": {"daha", "mı", "mi", "mu", "mü", "mıdır", "midir", "mudur", "müdür",
                "ve", "bir", "için", "bu", "veya", "yalnız", "her"},
         "en": {"in", "of", "the", "than", "is", "to", "on", "a", "an", "and", "are",
                "with", "under", "for", "that", "which", "only", "from", "all", "or", "without", "anyone"}}
_CONTEXT_SCAFFOLD = {"tr": {"kişiler", "kişilerin", "bütün", "aynı", "karşılaştırılır", "sabit", "tek", "göre", "demektir"},
    "en": {"people", "s", "person", "same", "one", "fixed", "every", "refer", "means", "mean", "compared", "measured", "different", "units", "conditions"}}


def _json(value):
    return json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(",", ":"), allow_nan=False)


def _config(config):
    if not isinstance(config, dict) or set(config) != _FIELDS:
        raise ValueError("unknown or missing training data configuration fields")
    for key, allowed in (("training_sizes", _SIZES[:6]), ("evaluation_sizes", _SIZES)):
        values = config[key]
        if (not isinstance(values, list) or not values or any(type(n) is not int or n not in allowed for n in values)
                or len(set(values)) != len(values)):
            raise ValueError("invalid sizes; training must remain n<=128")
    for key, allowed in (("training_relations", TRAIN_RELATIONS), ("heldout_relations", HELDOUT_RELATIONS),
                         ("query_types", QUERY_TYPES), ("conditions", CONDITIONS)):
        if (not isinstance(config[key], list) or any(type(value) is not str for value in config[key])
                or len(config[key]) != len(allowed) or set(config[key]) != set(allowed)):
            raise ValueError("fixed, complete relation/query/view design required: " + key)
    for key, allowed in (("groups_per_size", (4, 50)), ("evaluation_groups_per_size", (4, 200)),
                         ("grpo_min_n", _SIZES[:6]), ("grpo_rollouts_per_view", (4,))):
        if type(config[key]) is not int or config[key] not in allowed:
            raise ValueError("invalid integer data setting: " + key)
    if config["evaluation_split"] != "test":
        raise ValueError("this frozen training plan uses test only for held-out evaluation")
    maximum = config["grpo_max_instances"]
    if type(maximum) is not int or maximum < 4 or maximum > 600 or maximum % 4:
        raise ValueError("GRPO instance cap must be a multiple of four, at most 600")


def _words(text, language):
    text = unicodedata.normalize("NFC", text)
    text = turkish_lower(text) if language == "tr" else text.lower()
    return tuple(re.findall(r"[^\W\d_]+", text, re.UNICODE))


def lexical_overlap_audit(training_relations, heldout_relations, *, catalog=None):
    """Closed renderer-lexicon audit, not general morphological/sense analysis."""
    catalog = build_catalog() if catalog is None else catalog
    index = {row["relation_id"]: row for row in catalog}
    if any(name not in index for name in (*training_relations, *heldout_relations)):
        raise ValueError("unknown relation in lexical overlap audit")
    def collect(relations, language):
        lemmas, phrases, tokens = set(), set(), set()
        for relation in relations:
            for direction in index[relation]["directions"]:
                if language == "tr":
                    lemma = _words(direction["lemma"], language)
                    surfaces = [direction["term"], direction["morphology"]["query_predicate"]]
                else:
                    lemma = _words(direction["english_term"], language)
                    surfaces = [direction["english_term"]]
                lemmas.add(" ".join(lemma))
                for surface in surfaces:
                    words = _words(surface, language)
                    phrases.add(" ".join(words))
                    tokens.update(words)
        return lemmas, phrases, tokens
    report = {"scope": "canonical/equivalent/P1/P2: both converse directions, exact lexical/query surface forms; no P4",
              "entity_and_prompt_scaffold": "excluded; shared names/instructions do not identify a latent instance",
              "normalization": "NFC; Turkish-aware case; exact content words/phrases",
              "morphology_scope": "closed renderer variants, not a stemmer or human linguistic approval",
              "languages": {}, "catalog_prefix_cautions": [], "human_review_status": "pending"}
    for language in ("tr", "en"):
        left, right = collect(training_relations, language), collect(heldout_relations, language)
        content_overlap = (left[2] - _STOP[language]) & (right[2] - _STOP[language])
        report["languages"][language] = {"training_lemmas": sorted(left[0]), "heldout_lemmas": sorted(right[0]),
            "exact_lemma_overlap": sorted(left[0] & right[0]), "exact_phrase_overlap": sorted(left[1] & right[1]),
            "content_token_overlap": sorted(content_overlap), "stopword_overlap": sorted((left[2] & right[2]) & _STOP[language]),
            "stopword_inventory": sorted(_STOP[language])}
        relations = TR_RELATIONS if language == "tr" else EN_RELATIONS
        train_context = set(word for name in training_relations for word in _words(relations[name].context, language))
        heldout_context = set(word for name in heldout_relations for word in _words(relations[name].context, language))
        shared_context = train_context & heldout_context
        report["languages"][language]["context_audit"] = {
            "shared_stopwords": sorted(shared_context & _STOP[language]),
            "shared_task_scaffold": sorted(shared_context & _CONTEXT_SCAFFOLD[language]),
            "shared_content_tokens": sorted(shared_context - _STOP[language] - _CONTEXT_SCAFFOLD[language]),
            "task_scaffold_inventory": sorted(_CONTEXT_SCAFFOLD[language]),
            "scope": "complete frozen relation context text; exact surface words, no inferred stemming"}
    roots = sorted({direction["lemma"] for row in catalog for direction in row["directions"]})
    for i, first in enumerate(roots):
        for second in roots[i + 1:]:
            if min(len(first), len(second)) >= 2 and (first.startswith(second) or second.startswith(first)):
                report["catalog_prefix_cautions"].append({"left": first, "right": second,
                    "scope": "catalog-wide lexical prefix only; not automatic lemma equivalence", "review": "human_required"})
    report["training_families"] = sorted({index[name]["family"] for name in training_relations})
    report["heldout_families"] = sorted({index[name]["family"] for name in heldout_relations})
    report["family_overlap"] = sorted(set(report["training_families"]) & set(report["heldout_families"]))
    report["clean_content_holdout"] = not report["family_overlap"] and all(not layer[key]
        for layer in report["languages"].values() for key in ("exact_lemma_overlap", "exact_phrase_overlap", "content_token_overlap"))
    report["clean_content_holdout_scope"] = "controlled predicate/query lexicon only; full relation contexts audited separately"
    report["clean_full_context_holdout"] = report["clean_content_holdout"] and all(
        not layer["context_audit"]["shared_content_tokens"] for layer in report["languages"].values())
    report["full_context_requires_human_review"] = not report["clean_full_context_holdout"]
    return report


def _training_cohort(data, config):
    c = data.connection
    groups = []
    for n in sorted(config["training_sizes"]):
        rows = c.execute("SELECT id,latent_instance_id,sample_index FROM groups WHERE split='training' AND n=? ORDER BY sample_index,id", (n,)).fetchall()
        if len(rows) < config["groups_per_size"]:
            raise ValueError("requested frozen training groups unavailable")
        groups.extend({"id": row[0], "latent_instance_id": row[1], "sample_index": row[2], "n": n}
                      for row in rows[:config["groups_per_size"]])
    ids = {group["id"] for group in groups}
    queries = [dict(row) for row in c.execute("SELECT * FROM queries ORDER BY id") if row["group_id"] in ids]
    designs = [dict(row) for row in c.execute("SELECT * FROM designs ORDER BY id") if row["language"] == "tr"
               and row["name_control"] == "symbolic_shared" and row["relation_id"] in config["training_relations"]
               and row["condition"] in CONDITIONS]
    if len(designs) != 12 or {row["group_id"] for row in queries} != ids:
        raise ValueError("incomplete frozen training query/view matrix")
    by_group = {group["id"]: group for group in groups}
    strata, cells = Counter(), Counter()
    for query in queries:
        n = by_group[query["group_id"]]["n"]
        distance = query["query_distance"] if query["query_type"] == "distance_controlled" else None
        key = f"{n}|{query['query_type']}|{distance}"
        strata[key] += 1
        cells[f"{key}|{query['query_direction']}|{bool(query['gold_label'])}"] += 1
    answer_cells = {key: {suffix: cells[key + "|" + suffix] for suffix in
        ("forward|True", "forward|False", "reverse|True", "reverse|False")} for key in strata}
    if any(max(cell.values()) - min(cell.values()) > 1 for cell in answer_cells.values()):
        raise ValueError("frozen training query-answer cells are unbalanced")
    cohort = {"split": "training", "groups": groups, "query_ids": [query["id"] for query in queries],
        "design_ids": [row["id"] for row in designs], "group_count": len(groups), "query_count": len(queries),
        "designs_per_query": len(designs), "prompt_count": len(queries) * len(designs),
        "query_strata": dict(sorted(strata.items())), "answer_cells": answer_cells,
        "selection": {"split": "training", "sizes": config["training_sizes"], "groups_per_size": config["groups_per_size"],
            "relations": config["training_relations"], "languages": ["tr"], "name_controls": ["symbolic_shared"], "conditions": list(CONDITIONS)},
        "eligible_for_confirmatory_evaluation": False, "human_review_status": "pending"}
    cohort["cohort_sha256"] = fingerprint(cohort)
    return cohort, queries, designs


def _measurements(data, query_ids, design_ids, *, collect_hashes=False):
    totals = {"prompts": 0, "cl100k_base": 0, "o200k_base": 0, "utf8_bytes": 0,
              "max_cl100k_base": 0, "max_o200k_base": 0}
    hashes = set()
    for start in range(0, len(query_ids), 400):
        batch = query_ids[start:start + 400]
        sql = "SELECT cl100k_base,o200k_base,utf8_bytes,prompt_sha256 FROM measurements WHERE query_id IN (" + ",".join("?" for _ in batch) + ") AND design_id IN (" + ",".join("?" for _ in design_ids) + ")"
        for row in data.connection.execute(sql, batch + design_ids):
            totals["prompts"] += 1
            for i, key in enumerate(("cl100k_base", "o200k_base", "utf8_bytes")):
                totals[key] += row[i]
            for i, key in enumerate(("max_cl100k_base", "max_o200k_base")):
                totals[key] = max(totals[key], row[i])
            if collect_hashes:
                hashes.add(bytes(row[3]))
    if totals["prompts"] != len(query_ids) * len(design_ids):
        raise ValueError("missing stored source-prompt measurements")
    return totals, hashes


def _unit_id(latent_id, query_id, relation):
    return "train-unit-" + fingerprint([latent_id, query_id, relation])


def _units(cohort, queries, relations):
    groups = {row["id"]: row for row in cohort["groups"]}
    return [{"unit_id": _unit_id(groups[query["group_id"]]["latent_instance_id"], query["id"], relation),
        "instance_id": groups[query["group_id"]]["latent_instance_id"], "query_id": query["id"],
        "relation_id": relation, "n": groups[query["group_id"]]["n"], "gold": bool(query["gold_label"]),
        "query_type": query["query_type"], "query_distance": query["query_distance"],
        "query_direction": query["query_direction"], "query_relation": query["query_relation"]}
        for query in queries for relation in relations]


def _grpo(units, config, seed):
    eligible = [unit for unit in units if unit["n"] >= config["grpo_min_n"]]
    if not eligible:
        raise ValueError("GRPO has no eligible training units")
    n_values = sorted({unit["n"] for unit in eligible})
    cells = ("forward|True", "forward|False", "reverse|True", "reverse|False")
    pools = {(n, cell): [unit for unit in eligible if unit["n"] == n and f"{unit['query_direction']}|{unit['gold']}" == cell]
             for n in n_values for cell in cells}
    maximum = min(config["grpo_max_instances"], 4 * sum(min(len(pools[n, cell]) for cell in cells) for n in n_values))
    maximum -= maximum % 4
    rng = random.Random(seed)
    order = list(n_values)
    rng.shuffle(order)
    quota = {n: 0 for n in n_values}
    for i in range(maximum // 4):
        available = [n for n in order if quota[n] < min(len(pools[n, cell]) for cell in cells)]
        if not available:
            raise ValueError("balanced GRPO cap cannot be allocated")
        n = min(available, key=lambda value: (quota[value], order.index(value)))
        quota[n] += 1
    selected = []
    for n in n_values:
        for cell in cells:
            pool = sorted(pools[n, cell], key=lambda unit: unit["unit_id"])
            random.Random(seed + int(fingerprint([n, cell])[:12], 16)).shuffle(pool)
            selected.extend(pool[:quota[n]])
    selected.sort(key=lambda unit: unit["unit_id"])
    return {"eligible_units": len(eligible), "selected_unit_ids": [unit["unit_id"] for unit in selected],
        "selected_units": selected, "selected_count": len(selected),
        "answer_cells": dict(sorted(Counter(f"{unit['query_direction']}|{unit['gold']}" for unit in selected).items())),
        "n_counts": dict(sorted(Counter(str(unit["n"]) for unit in selected).items())),
        "relation_counts": dict(sorted(Counter(unit["relation_id"] for unit in selected).items())),
        "balance_scope": "four query-direction/gold cells exactly; n allocations differ by at most four units; other strata descriptive",
        "rollouts_per_view": config["grpo_rollouts_per_view"]}


def plan_data(root, manifest, configdata, *, seed=42):
    """Plan verified existing training/evaluation IDs; render no full dataset."""
    _config(configdata)
    if type(seed) is not int or seed < 0:
        raise ValueError("seed must be a nonnegative integer")
    config = deepcopy(configdata)
    overlap = lexical_overlap_audit(config["training_relations"], config["heldout_relations"])
    if not overlap["clean_content_holdout"]:
        raise ValueError("held-out family/vocabulary content overlaps training")
    with GridDataset(root, manifest) as data:
        training, queries, designs = _training_cohort(data, config)
        units = _units(training, queries, config["training_relations"])
        grpo = _grpo(units, config, seed)
        evaluation, evaluation_lengths = {}, {}
        train_ids = {group["latent_instance_id"] for group in training["groups"]}
        lengths, train_hashes = _measurements(data, training["query_ids"], training["design_ids"], collect_hashes=True)
        overlaps = {}
        for larger in (False, True):
            sizes = [n for n in config["evaluation_sizes"] if (n > 128) == larger]
            if not sizes:
                continue
            for heldout in (False, True):
                key = ("larger_n" if larger else "in_domain") + ("_heldout_vocabulary" if heldout else "")
                if not larger and heldout:
                    key = "heldout_vocabulary"
                selection = {"split": config["evaluation_split"], "sizes": sizes,
                    "groups_per_size": config["evaluation_groups_per_size"], "query_types": list(QUERY_TYPES),
                    "distances": None, "name_controls": ["symbolic_shared"],
                    "relations": config["heldout_relations"] if heldout else config["training_relations"],
                    "languages": ["en", "tr"], "conditions": list(CONDITIONS)}
                cohort = data.select(selection)
                if train_ids & {group["latent_instance_id"] for group in cohort["groups"]}:
                    raise ValueError("training and evaluation latent identities overlap")
                measured, hashes = _measurements(data, cohort["query_ids"], cohort["design_ids"], collect_hashes=True)
                intersection = train_hashes & hashes
                if intersection:
                    raise ValueError("visible full prompt hash overlaps training and evaluation")
                evaluation[key], evaluation_lengths[key] = cohort, measured
                overlaps[key] = {"latent_id_overlap": 0, "visible_prompt_hash_overlap": 0,
                    "evaluated_unique_prompt_hashes": len(hashes)}
        proxy = {"training_all_views": lengths}
        for name, conditions in (("sft_canonical", {"canonical"}), ("paired_sft", {"canonical", "equivalent"})):
            dids = [row["id"] for row in designs if row["condition"] in conditions]
            proxy[name] = _measurements(data, training["query_ids"], dids)[0]
        selected_by_relation = defaultdict(list)
        for unit in grpo["selected_units"]:
            selected_by_relation[unit["relation_id"]].append(unit["query_id"])
        grpo_lengths = None
        for relation, qids in selected_by_relation.items():
            part = _measurements(data, sorted(set(qids)), [row["id"] for row in designs if row["relation_id"] == relation])[0]
            if grpo_lengths is None:
                grpo_lengths = part
            else:
                for key, value in part.items():
                    grpo_lengths[key] = max(grpo_lengths[key], value) if key.startswith("max_") else grpo_lengths[key] + value
        proxy["grpo_selected_views"] = grpo_lengths
        proxy["evaluation"] = evaluation_lengths
        proxy["scope"] = "frozen old full prompt plain-text BPE proxies; new chat/assistant exposure is unknown until encoded"
        counts = {"training_latent_instances": training["group_count"], "training_queries": len(queries),
            "training_units": len(units), "canonical_rows": len(units), "paired_rows": len(units) * 2,
            "all_view_rows": len(units) * 4, "grpo_eligible_units": grpo["eligible_units"],
            "grpo_selected_units": grpo["selected_count"], "grpo_views": grpo["selected_count"] * 4,
            "grpo_rollout_samples": grpo["selected_count"] * 4 * config["grpo_rollouts_per_view"]}
        result = {"data_version": DATA_VERSION, "config": config, "seed": seed,
            "dataset_manifest": str(data.manifest_path.relative_to(data.root)),
            "dataset_manifest_sha256": data.manifest_sha256, "training": training,
            "training_unit_ids": [unit["unit_id"] for unit in units], "evaluation": evaluation,
            "counts": counts, "source_lengths_proxy": proxy, "grpo": grpo,
            "overlap_audit": {"lexical": overlap, "split_checks": overlaps,
                "training_unique_prompt_hashes": len(train_hashes), "shared_entity_names_are_not_latent_aliases": True},
            "contains_model_results": False, "contains_training_run": False}
        result["data_sha256"] = fingerprint(result)
        return result


def _validate_plan(root, plan):
    if not isinstance(plan, dict) or plan.get("data_version") != DATA_VERSION:
        raise ValueError("invalid training data plan")
    payload = {key: value for key, value in plan.items() if key != "data_sha256"}
    if fingerprint(payload) != plan.get("data_sha256"):
        raise ValueError("training data plan hash mismatch")
    if plan_data(root, plan["dataset_manifest"], plan["config"], seed=plan["seed"]) != plan:
        raise ValueError("training data plan differs from frozen selections")


def iter_training_units(root, plan):
    """Yield four same-query/same-gold training views, preserving source hashes."""
    _validate_plan(root, plan)
    with GridDataset(root, plan["dataset_manifest"]) as data:
        cohort, queries, designs = _training_cohort(data, plan["config"])
        tokenizers = grid.ModelTokenizers(data.root / "tokenizers")
        lookup = {(row["relation_id"], row["condition"]): row["id"] for row in designs}
        for unit in _units(cohort, queries, plan["config"]["training_relations"]):
            views = {condition: grid._materialized(data.root, data.connection, unit["query_id"],
                lookup[unit["relation_id"], condition], tokenizers) for condition in CONDITIONS}
            for row in views.values():
                if (row["split"] != "training" or row["latent_instance_id"] != unit["instance_id"]
                        or row["gold_label"] is not unit["gold"] or row["n"] > 128):
                    raise ValueError("training view changed identity/gold or entered a forbidden split/size")
                make_messages(row)  # Validates the frozen text frame without using gold.
            yield {**unit, "views": views}


def export_data(root, plan, out):
    """Export immutable training units and per-group hashes without inference."""
    out = Path(out).absolute()
    if out.exists():
        raise FileExistsError("training export already exists")
    if any(path.is_symlink() for path in (out, *out.parents)):
        raise ValueError("training export cannot traverse symlinks")
    root = Path(root).resolve()
    for relative in ("src", "tests", "configs", "data", "tokenizers", "results/raw", "results/normalized"):
        if out.is_relative_to(root / relative):
            raise ValueError("training export overlaps protected project inputs")
    _validate_plan(root, plan)
    out.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=".training-data-", dir=out.parent))
    try:
        hashes = []
        path = temporary / "training_units.jsonl"
        with path.open("xb") as handle:
            for unit in iter_training_units(root, plan):
                encoded = (_json(unit) + "\n").encode()
                handle.write(encoded)
                hashes.append({"unit_id": unit["unit_id"], "instance_id": unit["instance_id"],
                    "sha256": hashlib.sha256(encoded).hexdigest(),
                    "view_prompt_hashes": {condition: row["prompt_sha256"] for condition, row in unit["views"].items()}})
        result = {"artifact_type": "training_data_export", "data_version": DATA_VERSION,
            "data_sha256": plan["data_sha256"], "dataset_manifest_sha256": plan["dataset_manifest_sha256"],
            "units": len(hashes), "counts": plan["counts"], "groups": hashes,
            "files": [{"path": path.name, "bytes": path.stat().st_size,
                       "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}],
            "contains_model_results": False, "contains_training_run": False}
        (temporary / "manifest.json").write_text(_json(result) + "\n", encoding="utf-8")
        if out.exists():
            raise FileExistsError("training export appeared concurrently")
        temporary.rename(out)
        return result
    except BaseException:
        shutil.rmtree(temporary, ignore_errors=True)
        raise

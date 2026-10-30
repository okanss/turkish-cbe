"""Deterministic bilingual prompt banks, grouped splits, and comparison cells.

A--D pair identical queries. E changes Turkish presentation. F contrasts query
complexity within an order and MUST NOT be scored as semantic consistency.
All large data are streamed; prompt references avoid duplication in cells.
"""

from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import asdict, dataclass
import hashlib
import json
from pathlib import Path
import platform
import random
import shutil
import sys
import tomllib

from data_generation.generator import (DEFAULT_DISTANCES, DEFAULT_SIZES,
    GenerationPlan, canonical_bytes, generate_record, valid_distances)
from data_generation.oracle import answer_from_edges, answer_from_order
from data_generation.validation import validate_record
from renderers import english, turkish
from . import DATASET_VERSION
from .names import name_maps, validate_name_maps
from .pairing import validate_pair
from .token_lengths import TOKENIZER_SPEC, measure_text
from .surface_reader import TurkishReader

NAME_CONTROLS = ("native_shared", "en_transliterated", "symbolic_shared")
LANGUAGES = ("en", "tr")
CONDITIONS = ("canonical", "equivalent", "P1", "P2", "P3", "P4", "P5")
SCHEMA_VERSION = 1


def digest(value) -> str:
    return hashlib.sha256(canonical_bytes(value)).hexdigest()


def file_hash(path: Path) -> str:
    result = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            result.update(block)
    return result.hexdigest()


@dataclass(frozen=True)
class SplitSpec:
    name: str
    seed: int
    sizes: tuple[int, ...]
    count: int


@dataclass(frozen=True)
class DatasetPlan:
    splits: tuple[SplitSpec, ...]
    presentation_seed: int = 431
    vocabulary_pilot: bool = True

    def validate(self) -> None:
        if type(self.presentation_seed) is not int or not 0 <= self.presentation_seed < 2**64:
            raise ValueError("invalid presentation seed")
        if type(self.vocabulary_pilot) is not bool:
            raise ValueError("vocabulary_pilot must be Boolean")
        if (not isinstance(self.splits, tuple) or len(self.splits) != 4
                or {s.name for s in self.splits} != {"pilot", "development", "training", "test"}):
            raise ValueError("exactly four named splits are required")
        if len({s.seed for s in self.splits}) != 4:
            raise ValueError("split generation seeds must be distinct")
        for spec in self.splits:
            if type(spec.count) is not int or spec.count < 4 or spec.count % 2:
                raise ValueError("split count must be even and >= 4")
            GenerationPlan(seed=spec.seed, sizes=spec.sizes,
                           examples_per_size=4 * ((spec.count + 3) // 4),
                           entity_types=("turkish_names",)).validate()


DEFAULT_PLAN = DatasetPlan((
    SplitSpec("training", 401, DEFAULT_SIZES, 50),
    SplitSpec("development", 402, DEFAULT_SIZES, 50),
    SplitSpec("pilot", 403, DEFAULT_SIZES[:6], 50),
    SplitSpec("test", 404, DEFAULT_SIZES, 200),
))


def load_plan(path: Path) -> DatasetPlan:
    with Path(path).open("rb") as handle:
        value = tomllib.load(handle)
    if (set(value) != {"schema_version", "presentation_seed", "vocabulary_pilot", "splits"}
            or type(value["schema_version"]) is not int or value["schema_version"] != 1):
        raise ValueError("unknown matched dataset configuration schema")
    specs = []
    for item in value["splits"]:
        if set(item) != {"name", "seed", "sizes", "count"}:
            raise ValueError("unknown split configuration fields")
        specs.append(SplitSpec(item["name"], item["seed"], tuple(item["sizes"]), item["count"]))
    plan = DatasetPlan(tuple(specs), value["presentation_seed"], value["vocabulary_pilot"])
    plan.validate()
    return plan


def balanced_indices(count: int, n: int) -> tuple[int, ...]:
    """Preserve generator cells while exactly balancing both binary margins."""
    if type(count) is not int or count < 4 or count % 2 or type(n) is not int or n < 4:
        raise ValueError("count must be even >=4; n must be >=4")
    q, remainder = divmod(count, 4)
    output = list(range(4 * q))
    if remainder:
        extras = (0, 3) if n.bit_length() % 2 else (1, 2)
        output.extend(4 * q + cell for cell in extras)
    return tuple(output)


def _rng(seed: int, ordering_id: str, namespace: str) -> random.Random:
    return random.Random(int(digest([DATASET_VERSION, seed, ordering_id, namespace]), 16))


def _shuffle(items: list, rng: random.Random) -> list:
    result = list(items)
    rng.shuffle(result)
    if len(result) > 1 and result == items:
        result = result[1:] + result[:1]
    return result


def presentation_plans(record: dict, seed: int) -> dict:
    """Geometry is independent of language, aliases, relation and query target."""
    n, group = record["n"], record["ordering_id"]
    names = list(range(n))  # indices into sorted SOURCE names, never latent ranks
    names = _shuffle(names, _rng(seed, group, "canonical_names"))
    edges = list(range(n - 1))  # indices into the source's shuffled premise list
    equivalent_names = _shuffle(names, _rng(seed, group, "equivalent_names"))
    equivalent_edges = _shuffle(edges, _rng(seed, group, "equivalent_edges"))
    rng = _rng(seed, group, "converse_mask")
    m = rng.randint(max(1, (n - 1) // 5), n - 1)
    mask = sorted(rng.sample(range(n - 1), m))  # uniform CHAIN positions, protocol P
    def entry(name_order, edge_order, converse, style="canonical"):
        return {"entity_order": name_order, "premise_order": edge_order,
                "converse_chain_positions": converse, "style": style}
    return {
        "canonical": entry(names, edges, []),
        "equivalent": entry(equivalent_names, equivalent_edges, mask),
        "P1": entry(equivalent_names, equivalent_edges, []),
        "P2": entry(names, edges, list(range(n - 1))),
        "P3": entry(names, edges, [], "syntax"),
        "P4": entry(names, edges, [], "morphology_syntax_lexical"),
        "P5": entry(equivalent_names, equivalent_edges, list(range(n - 1)),
                    "morphology_syntax_lexical"),
    }


def _tr_clause(a, b, relation_id, direction, style):
    spec = turkish.RELATIONS[relation_id]
    if style == "morphology_syntax_lexical":
        return turkish._attribute_statement(a, b, spec, direction)
    subject, compared, predicate = turkish._parts(a, b, spec, direction)
    if style == "syntax":
        return f"{compared} {predicate} {subject}."
    return f"{subject}, {compared} {predicate}."


def _component(record, relation_id, language, aliases, edit_plan):
    source_names = sorted(record["latent_order"])
    names = [aliases[source_names[i]] for i in edit_plan["entity_order"]]
    ranks = {name: i for i, name in enumerate(record["latent_order"])}
    mask = set(edit_plan["converse_chain_positions"])
    premises = []
    for i in edit_plan["premise_order"]:
        a, b = record["premise_edges"][i]
        a, b = aliases[a], aliases[b]
        if ranks[record["premise_edges"][i][0]] in mask:
            a, b, direction = b, a, "after"
        else:
            direction = "before"
        text = (english.statement(a, b, relation_id, direction) if language == "en"
                else _tr_clause(a, b, relation_id, direction, edit_plan["style"]))
        premises.append(text)
    context = (english.context(relation_id, symbolic=all(x.startswith("E") and x[1:].isdigit() for x in names))
               if language == "en" else turkish.expected_context(relation_id, names))
    # Parse actual surface sentences; do not accept a stored equivalence flag.
    inverse = {value: key for key, value in aliases.items()}
    reader = TurkishReader(names, relation_id) if language == "tr" else None
    parsed = ([reader.statement(text) for text in premises] if reader is not None else
              [english.normalize_statement(text, tuple(names), relation_id) for text in premises])
    edges = [(inverse[a], inverse[b]) for a, b in parsed]
    if set(edges) != {tuple(edge) for edge in record["premise_edges"]} or len(set(edges)) != record["n"] - 1:
        raise ValueError("surface premises changed the source chain")
    return {"context": context, "entity_names": names, "premises": premises,
            "edges": edges, "plan_sha256": digest(edit_plan), "reader": reader}


def _prompt(record, split, control, relation_id, language, condition, component, aliases):
    subject, object_ = aliases[record["query_subject"]], aliases[record["query_object"]]
    relation = record["query_relation"]
    query = (english.question(subject, object_, relation_id, relation) if language == "en"
             else turkish._question(subject, object_, turkish.RELATIONS[relation_id], relation))
    parsed_query = (english.normalize_query(query, tuple(component["entity_names"]), relation_id)
                    if language == "en" else component["reader"].query(query))
    if parsed_query != (subject, object_, relation):
        raise ValueError("surface query changed the target or relation")
    gold = answer_from_edges(sorted(record["latent_order"]), component["edges"],
                             record["query_subject"], record["query_object"], relation)
    if gold is not record["gold_label"] or answer_from_order(record["latent_order"],
            record["query_subject"], record["query_object"], relation) is not gold:
        raise ValueError("paired oracle disagreement")
    compose = english.compose_prompt if language == "en" else turkish.compose_prompt
    prompt = compose(component["context"], component["entity_names"], component["premises"], query)
    row = {"schema_version": SCHEMA_VERSION, "dataset_version": DATASET_VERSION,
        "latent_instance_id": record["ordering_id"], "query_instance_id": record["instance_id"],
        "split": split, "n": record["n"], "query_type": record["query_type"],
        "query_distance": record["query_distance"], "query_subject": record["query_subject"],
        "query_object": record["query_object"], "query_relation": relation,
        "query_direction": record["query_direction"], "gold_label": gold,
        "language": language, "condition": condition, "name_control": control,
        "relation_id": relation_id, "display_query_subject": subject,
        "display_query_object": object_, "presentation_plan_sha256": component["plan_sha256"],
        "prompt": prompt, "prompt_sha256": hashlib.sha256(prompt.encode("utf-8")).hexdigest(),
        "lengths": measure_text(prompt), "human_review_status": "pending",
        "eligible_for_confirmatory_evaluation": False}
    row["prompt_id"] = "prompt-" + digest([DATASET_VERSION, split, record["instance_id"],
        control, relation_id, language, condition, row["prompt_sha256"]])
    return row


class _Writer:
    def __init__(self, path):
        self.path = path
        self.handle = path.open("xb")
        self.count = 0
        self.sha = hashlib.sha256()
    def add(self, value):
        payload = canonical_bytes(value) + b"\n"
        self.handle.write(payload)
        self.sha.update(payload)
        self.count += 1
    def close(self):
        self.handle.close()
        return {"path": self.path.name, "records": self.count,
                "bytes": self.path.stat().st_size, "sha256": self.sha.hexdigest()}


def _cell(left, right, arm):
    validate_pair(left, right, arm)
    row = {"arm": arm, "split": left["split"], "n": left["n"],
        "latent_instance_id": left["latent_instance_id"], "name_control": left["name_control"],
        "relation_id": left["relation_id"], "left_prompt_id": left["prompt_id"],
        "right_prompt_id": right["prompt_id"], "left_condition": left["condition"],
        "right_condition": right["condition"], "left_language": left["language"],
        "right_language": right["language"], "left_query_type": left["query_type"],
        "right_query_type": right["query_type"], "query_distance": right["query_distance"],
        "query_direction": right["query_direction"], "gold_label": right["gold_label"],
        "comparison_kind": "query_complexity" if arm == "F" else "same_query",
        "study_track": ("reference_structural" if arm in "ABCD" and
             left["relation_id"] == "spatial_front" else "turkish_extension")}
    row["cell_id"] = "cell-" + digest([arm, left["prompt_id"], right["prompt_id"]])
    return row


def _source_hashes():
    root = Path(__file__).resolve().parents[1]
    paths = []
    for name in ("turkish_cbe", "data_generation", "renderers", "validation", "experiments"):
        package = root / name
        if package.is_symlink() or not package.is_dir():
            raise ValueError("owned source package is missing or symlinked: " + name)
        for path in package.rglob("*"):
            if path.is_symlink():
                raise ValueError("owned source paths must not be symlinks: " + str(path))
            if path.suffix == ".py":
                if not path.is_file():
                    raise ValueError("owned Python source must be a regular file: " + str(path))
                paths.append(path)
    return {p.relative_to(root).as_posix(): file_hash(p) for p in sorted(paths)}


def _legacy_groups(manifest_path):
    if manifest_path is None:
        return {}, None
    manifest_path = Path(manifest_path).resolve()
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    data_root = (manifest_path.parent / manifest["data_root"]).resolve()
    seen = {}
    for entry in manifest["files"]:
        if not entry["path"].startswith("endpoint/"):
            continue
        path = (data_root / entry["path"]).resolve()
        if not path.is_relative_to(data_root) or file_hash(path) != entry["sha256"]:
            raise ValueError("legacy development source failed hash validation")
        for line in path.open(encoding="utf-8"):
            record = json.loads(line)
            seen["id:" + record["ordering_id"]] = "legacy_development"
            seen["view:" + digest(record["latent_order"])] = "legacy_development"
    return seen, {"manifest_sha256": file_hash(manifest_path), "orders": len(seen) // 2,
                  "role": "previously_exposed_development_only"}


def build_dataset(output_root: Path, manifest_path: Path, plan: DatasetPlan = DEFAULT_PLAN,
                  legacy_manifest: Path | None = None) -> dict:
    plan.validate()
    output_root, manifest_path = Path(output_root).resolve(), Path(manifest_path).resolve()
    if manifest_path.exists() or manifest_path.is_symlink():
        raise FileExistsError(manifest_path)
    for spec in plan.splits:
        target = output_root / spec.name
        if target.exists() or target.is_symlink():
            raise FileExistsError(target)
    if not output_root.is_relative_to(manifest_path.parent):
        raise ValueError("data root must be inside the manifest directory")
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    output_root.mkdir(parents=True, exist_ok=True)
    group_seen, legacy = _legacy_groups(legacy_manifest)
    if legacy is not None:
        legacy_path = Path(legacy_manifest).resolve()
        if not legacy_path.is_relative_to(manifest_path.parent):
            raise ValueError("legacy manifest must be inside the manifest directory")
        legacy["manifest_path"] = legacy_path.relative_to(manifest_path.parent).as_posix()
    prompt_seen = {}
    created, writers = [], []
    manifest_owned = False
    summaries, files = {}, []
    try:
        for spec in plan.splits:
            directory = output_root / spec.name
            directory.mkdir()
            created.append(directory)
            outputs = {}
            for name in ("groups", "latent", "prompts", "cells"):
                writer = _Writer(directory / (name + ".jsonl"))
                writers.append(writer)
                outputs[name] = writer
            count_prompts, count_cells, query_cells, query_counts = Counter(), Counter(), Counter(), Counter()
            length_totals, length_counts = Counter(), Counter()
            generation_plan = GenerationPlan(seed=spec.seed, sizes=spec.sizes,
                examples_per_size=4 * ((spec.count + 3) // 4), entity_types=("turkish_names",))
            index_plan = {}
            def write_prompt(row):
                hash_ = row["prompt_sha256"]
                previous = prompt_seen.get(hash_)
                if previous is not None and previous != spec.name:
                    raise ValueError("visible prompt leaked across splits")
                prompt_seen[hash_] = spec.name
                outputs["prompts"].add(row)
                key = "|".join(str(row[k]) for k in ("n", "query_type", "name_control", "relation_id", "language", "condition"))
                count_prompts[key] += 1
                length_key = row["language"] + "|" + row["name_control"]
                length_totals[length_key] += row["lengths"]["reference_token_count"]
                length_counts[length_key] += 1
            def write_cell(left, right, arm):
                row = _cell(left, right, arm)
                outputs["cells"].add(row)
                key = "|".join(str(row[k]) for k in ("arm", "n", "name_control", "relation_id", "left_language", "right_language", "left_condition", "right_condition", "right_query_type", "query_distance", "query_direction", "gold_label"))
                count_cells[key] += 1
            for n in spec.sizes:
                indices = balanced_indices(spec.count, n)
                index_plan[str(n)] = list(indices)
                for index in indices:
                    endpoint = generate_record(generation_plan, n, index, "endpoint")
                    validate_record(endpoint)
                    maps = name_maps(endpoint)
                    validate_name_maps(endpoint, maps)
                    group_id = endpoint["ordering_id"]
                    views = {digest([maps[control][lang][name] for name in endpoint["latent_order"]])
                             for control in NAME_CONTROLS for lang in LANGUAGES}
                    for key in ["id:" + group_id, *("view:" + h for h in sorted(views))]:
                        previous = group_seen.get(key)
                        if previous is not None and previous != spec.name:
                            raise ValueError("latent group or aliased ordering leaked across splits")
                        group_seen[key] = spec.name
                    plans = presentation_plans(endpoint, plan.presentation_seed)
                    outputs["groups"].add({"latent_instance_id": group_id, "split": spec.name,
                        "n": n, "sample_index": index, "seed": spec.seed,
                        "latent_order": endpoint["latent_order"], "name_maps": maps,
                        "presentation_plans": plans, "named_order_sha256": sorted(views)})
                    cache, endpoint_prompts = {}, {}
                    queries = [("endpoint", None), ("internal", None)]
                    queries += [("distance_controlled", d) for d in valid_distances(n, DEFAULT_DISTANCES)]
                    for query_type, distance in queries:
                        record = (endpoint if query_type == "endpoint" else
                                  generate_record(generation_plan, n, index, query_type, distance))
                        validate_record(record)
                        outputs["latent"].add({"latent_instance_id": group_id,
                            "query_instance_id": record["instance_id"], "split": spec.name, "record": record})
                        block = f"{n}|{query_type}|{distance}"
                        query_counts[block] += 1
                        query_cells[block + "|" + record["query_direction"] + "|" + str(record["gold_label"])] += 1
                        controls = NAME_CONTROLS if query_type == "endpoint" else ("native_shared",)
                        relations = tuple(turkish.RELATIONS) if (spec.name == "pilot" and plan.vocabulary_pilot
                                    and query_type == "endpoint") else ("spatial_front",)
                        for relation_id in relations:
                            for control in controls:
                                # Vocabulary extensions do not multiply name controls.
                                if relation_id != "spatial_front" and control != "native_shared":
                                    continue
                                rows = {}
                                conditions = CONDITIONS if query_type == "endpoint" and control == "native_shared" else CONDITIONS[:2]
                                for condition in conditions:
                                    for language in LANGUAGES:
                                        if condition.startswith("P") and language == "en":
                                            continue
                                        key = (relation_id, control, language, condition)
                                        if key not in cache:
                                            cache[key] = _component(endpoint, relation_id, language,
                                                maps[control][language], plans[condition])
                                        row = _prompt(record, spec.name, control, relation_id, language,
                                                      condition, cache[key], maps[control][language])
                                        rows[(language, condition)] = row
                                        write_prompt(row)
                                for arm, a, b in (("A", ("en", "canonical"), ("en", "equivalent")),
                                    ("B", ("tr", "canonical"), ("tr", "equivalent")),
                                    ("C", ("en", "canonical"), ("tr", "canonical")),
                                    ("D", ("en", "equivalent"), ("tr", "equivalent"))):
                                    if query_type == "endpoint":
                                        write_cell(rows[a], rows[b], arm)
                                if query_type == "endpoint":
                                    for language in LANGUAGES:
                                        for condition in CONDITIONS[:2]:
                                            endpoint_prompts[(relation_id, control, language, condition)] = rows[(language, condition)]
                                    if control == "native_shared":
                                        for condition in CONDITIONS[2:]:
                                            write_cell(rows[("tr", "canonical")], rows[("tr", condition)], "E")
                                else:
                                    for (language, condition), row in rows.items():
                                        left = endpoint_prompts[(relation_id, control, language, condition)]
                                        write_cell(left, row, "F")
                    if index == indices[-1]:
                        print(f"{spec.name}: n={n}, {spec.count} matched orders completed", flush=True)
            split_files = []
            for writer in outputs.values():
                entry = writer.close()
                entry["path"] = writer.path.relative_to(manifest_path.parent).as_posix()
                split_files.append(entry)
            summary = {"orders": sum(spec.count for _ in spec.sizes),
                "queries": outputs["latent"].count, "prompts": outputs["prompts"].count,
                "comparison_cells": outputs["cells"].count, "sample_indices": index_plan,
                "query_blocks": dict(sorted(query_counts.items())), "query_cells": dict(sorted(query_cells.items())),
                "prompt_cells": dict(sorted(count_prompts.items())), "comparison_cell_counts": dict(sorted(count_cells.items())),
                "reference_token_totals": dict(sorted(length_totals.items())),
                "reference_token_records": dict(sorted(length_counts.items())),
                "files": split_files}
            summaries[spec.name] = summary
            split_manifest = directory / "split_manifest.json"
            split_manifest.write_bytes(canonical_bytes(summary) + b"\n")
            files.extend(split_files)
            files.append({"path": split_manifest.relative_to(manifest_path.parent).as_posix(),
                          "records": None, "bytes": split_manifest.stat().st_size,
                          "sha256": file_hash(split_manifest)})
        manifest = {"schema_version": SCHEMA_VERSION, "dataset_version": DATASET_VERSION,
            "artifact_type": "matched_bilingual_dataset", "status": "structurally_frozen",
            "contains_model_results": False, "human_review_status": "pending",
            "eligible_for_confirmatory_evaluation": False, "plan": asdict(plan),
            "plan_sha256": digest(asdict(plan)), "tokenizer": TOKENIZER_SPEC,
            "environment": {"python": platform.python_version(), "implementation": platform.python_implementation()},
            "source_files_sha256": _source_hashes(), "legacy_development_guard": legacy,
            "splits": summaries, "files": files,
            "validation": {"paired_queries_match": True, "surface_normal_forms_match": True,
                "independent_oracles_agree": True, "split_groups_disjoint": True,
                "aliased_orders_disjoint": True, "cross_split_prompt_duplicates": 0},
            "hash_algorithm": "SHA-256", "bundle_files_sha256": digest(files)}
        with manifest_path.open("xb") as handle:
            manifest_owned = True
            handle.write(canonical_bytes(manifest) + b"\n")
        return manifest
    except BaseException:
        for writer in writers:
            if not writer.handle.closed:
                writer.handle.close()
        for path in reversed(created):
            shutil.rmtree(path)
        if manifest_owned:
            manifest_path.unlink(missing_ok=True)
        # A concurrently created manifest never belongs to us; never remove it.
        raise


def _read_jsonl(path):
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            yield json.loads(line)


def _verification_plan(value):
    if (not isinstance(value, dict)
            or set(value) != {"splits", "presentation_seed", "vocabulary_pilot"}
            or not isinstance(value["splits"], list)):
        raise ValueError("invalid manifest experiment plan")
    specs = []
    for item in value["splits"]:
        if (not isinstance(item, dict) or set(item) != {"name", "seed", "sizes", "count"}
                or not isinstance(item["name"], str) or not isinstance(item["sizes"], list)):
            raise ValueError("invalid manifest split specification")
        specs.append(SplitSpec(item["name"], item["seed"], tuple(item["sizes"]), item["count"]))
    plan = DatasetPlan(tuple(specs), value["presentation_seed"], value["vocabulary_pilot"])
    plan.validate()
    return plan


def _verification_rows(path, count):
    actual = 0
    with path.open("rb") as handle:
        for line in handle:
            row = json.loads(line)
            if not isinstance(row, dict) or canonical_bytes(row) + b"\n" != line:
                raise ValueError("JSONL row is not a canonical object: " + str(path))
            actual += 1
            yield row
    if actual != count:
        raise ValueError("JSONL record count mismatch: " + str(path))


def _verification_design(record, spec, plan):
    """Enumerate the complete prompt design independently of rows in the bank."""
    endpoint = record["query_type"] == "endpoint"
    relations = tuple(turkish.RELATIONS) if (endpoint and spec.name == "pilot"
                 and plan.vocabulary_pilot) else ("spatial_front",)
    for relation in relations:
        for control in NAME_CONTROLS if endpoint else ("native_shared",):
            if relation != "spatial_front" and control != "native_shared":
                continue
            conditions = CONDITIONS if endpoint and control == "native_shared" else CONDITIONS[:2]
            for condition in conditions:
                for language in LANGUAGES:
                    if not (condition.startswith("P") and language == "en"):
                        yield relation, control, language, condition


def verify_dataset(manifest_path: Path, deep: bool = True) -> dict:
    """Check frozen hashes, then replay sources, surfaces and the complete matrix.

    Hash-only verification is portable. Deep replay requires the frozen source
    code and tokenizer specification, including its Unicode database version.
    """
    if type(deep) is not bool:
        raise ValueError("deep must be Boolean")
    manifest_path = Path(manifest_path).resolve()
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if (not isinstance(manifest, dict) or manifest.get("dataset_version") != DATASET_VERSION
            or type(manifest.get("schema_version")) is not int
            or manifest["schema_version"] != SCHEMA_VERSION):
        raise ValueError("unsupported dataset version")
    plan = _verification_plan(manifest["plan"])
    if digest(manifest["plan"]) != manifest["plan_sha256"]:
        raise ValueError("manifest plan hash changed")
    if (not isinstance(manifest.get("files"), list)
            or digest(manifest["files"]) != manifest["bundle_files_sha256"]):
        raise ValueError("manifest file inventory changed")
    flags = {"artifact_type": "matched_bilingual_dataset", "status": "structurally_frozen",
             "contains_model_results": False, "human_review_status": "pending",
             "eligible_for_confirmatory_evaluation": False, "hash_algorithm": "SHA-256"}
    if any(canonical_bytes(manifest.get(key)) != canonical_bytes(value) for key, value in flags.items()):
        raise ValueError("manifest artifact/provenance flags changed")
    root, inventory = manifest_path.parent, {}
    for entry in manifest["files"]:
        if not isinstance(entry, dict) or set(entry) != {"path", "records", "bytes", "sha256"}:
            raise ValueError("invalid manifest file entry")
        relative = entry["path"]
        if (not isinstance(relative, str) or not relative or Path(relative).is_absolute()
                or Path(relative).as_posix() != relative or ".." in Path(relative).parts
                or relative in inventory or type(entry["bytes"]) is not int or entry["bytes"] < 0
                or not isinstance(entry["sha256"], str) or len(entry["sha256"]) != 64
                or any(c not in "0123456789abcdef" for c in entry["sha256"])):
            raise ValueError("invalid or duplicate manifest path/hash/size")
        path = root / relative
        if (path.resolve() != path or not path.is_file() or path.stat().st_size != entry["bytes"]
                or file_hash(path) != entry["sha256"]):
            raise ValueError("frozen dataset hash/size/path mismatch: " + relative)
        inventory[relative] = entry
    if not isinstance(manifest.get("splits"), dict) or set(manifest["splits"]) != {s.name for s in plan.splits}:
        raise ValueError("manifest split inventory mismatch")
    split_paths, linked_paths = {}, set()
    for spec in plan.splits:
        summary = manifest["splits"][spec.name]
        if not isinstance(summary, dict) or not isinstance(summary.get("files"), list) or len(summary["files"]) != 4:
            raise ValueError("invalid split file inventory")
        entries, directory = {}, None
        for entry in summary["files"]:
            if not isinstance(entry, dict) or inventory.get(entry.get("path")) != entry:
                raise ValueError("nested split file inventory differs from root inventory")
            path = root / entry["path"]
            if (path.suffix != ".jsonl" or path.stem not in {"groups", "latent", "prompts", "cells"}
                    or path.stem in entries or type(entry["records"]) is not int or entry["records"] < 0
                    or path.parent.name != spec.name or (directory is not None and path.parent != directory)):
                raise ValueError("invalid or duplicate split JSONL file")
            directory = path.parent
            entries[path.stem] = (path, entry["records"])
            linked_paths.add(entry["path"])
        split_manifest = directory / "split_manifest.json"
        relative = split_manifest.relative_to(root).as_posix()
        if (relative not in inventory or inventory[relative]["records"] is not None
                or split_manifest.read_bytes() != canonical_bytes(summary) + b"\n"):
            raise ValueError("hashed split manifest differs from root split summary")
        linked_paths.add(relative)
        split_paths[spec.name] = entries
    if linked_paths != set(inventory):
        raise ValueError("root file inventory has unlinked or missing split artifacts")
    if not deep:
        return {"valid": True, "files": len(inventory), "deep": False}
    if manifest.get("tokenizer") != TOKENIZER_SPEC:
        raise ValueError("tokenizer/Unicode database differs from frozen dataset; use the recorded runtime or --hashes-only")
    if manifest.get("source_files_sha256") != _source_hashes():
        raise ValueError("source code differs from frozen dataset; restore recorded sources or use --hashes-only")
    legacy_seen = {}
    legacy = manifest.get("legacy_development_guard")
    if legacy is not None:
        if (not isinstance(legacy, dict)
                or set(legacy) != {"manifest_path", "manifest_sha256", "orders", "role"}
                or not isinstance(legacy["manifest_path"], str)):
            raise ValueError("invalid legacy development guard")
        relative = legacy["manifest_path"]
        legacy_path = root / relative
        if (not relative or Path(relative).is_absolute() or ".." in Path(relative).parts
                or Path(relative).as_posix() != relative or legacy_path.resolve() != legacy_path):
            raise ValueError("invalid legacy development manifest path")
        legacy_seen, expected_legacy = _legacy_groups(legacy_path)
        expected_legacy["manifest_path"] = relative
        if canonical_bytes(legacy) != canonical_bytes(expected_legacy):
            raise ValueError("legacy development manifest or provenance changed")
    group_splits, view_splits, prompt_splits = {}, {}, {}
    total_prompt_records = 0
    for spec in plan.splits:
        name, summary = spec.name, manifest["splits"][spec.name]
        paths = split_paths[name]
        groups, sources = {}, {}
        for row in _verification_rows(*paths["groups"]):
            group_id = row.get("latent_instance_id")
            if not isinstance(group_id, str) or group_id in groups:
                raise ValueError("duplicate or invalid latent group ID")
            groups[group_id] = row
        for row in _verification_rows(*paths["latent"]):
            if (set(row) != {"latent_instance_id", "query_instance_id", "split", "record"}
                    or not isinstance(row["record"], dict)):
                raise ValueError("invalid source-record wrapper schema")
            record, query_id = row["record"], row["query_instance_id"]
            if (not isinstance(query_id, str) or query_id in sources or row["split"] != name
                    or query_id != record.get("instance_id")
                    or row["latent_instance_id"] != record.get("ordering_id")):
                raise ValueError("duplicate source ID or source-wrapper pairing mismatch")
            sources[query_id] = record
        generation_plan = GenerationPlan(seed=spec.seed, sizes=spec.sizes,
            examples_per_size=4 * ((spec.count + 3) // 4), entity_types=("turkish_names",))
        expected_groups, expected_sources, endpoints = [], [], {}
        query_counts, query_cells = Counter(), Counter()
        expected_signatures = set()
        for n in spec.sizes:
            for index in balanced_indices(spec.count, n):
                endpoint = generate_record(generation_plan, n, index, "endpoint")
                group_id = endpoint["ordering_id"]
                expected_groups.append(group_id)
                endpoints[group_id] = endpoint["instance_id"]
                maps = name_maps(endpoint)
                views = sorted({digest([maps[c][lang][entity] for entity in endpoint["latent_order"]])
                                for c in NAME_CONTROLS for lang in LANGUAGES})
                expected_group = {"latent_instance_id": group_id, "split": name, "n": n,
                    "sample_index": index, "seed": spec.seed, "latent_order": endpoint["latent_order"],
                    "name_maps": maps, "presentation_plans": presentation_plans(endpoint, plan.presentation_seed),
                    "named_order_sha256": views}
                group = groups.get(group_id)
                if canonical_bytes(group) != canonical_bytes(expected_group):
                    raise ValueError("group metadata, aliases, presentation plans or view hashes differ from generation plan")
                validate_name_maps(endpoint, group["name_maps"])
                if group_id in group_splits or "id:" + group_id in legacy_seen:
                    raise ValueError("duplicate latent group or split group leakage")
                group_splits[group_id] = name
                for view in views:
                    if ((view in view_splits and view_splits[view] != name)
                            or "view:" + view in legacy_seen):
                        raise ValueError("aliased order leakage")
                    view_splits[view] = name
                queries = [("endpoint", None), ("internal", None)]
                queries += [("distance_controlled", d) for d in valid_distances(n, DEFAULT_DISTANCES)]
                for query_type, distance in queries:
                    expected = endpoint if query_type == "endpoint" else generate_record(
                        generation_plan, n, index, query_type, distance)
                    query_id = expected["instance_id"]
                    expected_sources.append(query_id)
                    if sources.get(query_id) != expected:
                        raise ValueError("source query differs from seeded generator or is missing")
                    validate_record(sources[query_id])
                    block = f"{n}|{query_type}|{distance}"
                    query_counts[block] += 1
                    query_cells[block + "|" + expected["query_direction"] + "|" + str(expected["gold_label"])] += 1
                    for design in _verification_design(expected, spec, plan):
                        expected_signatures.add((query_id, *design))
        if list(groups) != expected_groups or list(sources) != expected_sources:
            raise ValueError("group/source inventory, row order or cardinality differs from generation plan")
        prompts, components, prompt_index = {}, {}, {}
        count_prompts, count_cells, length_totals, length_counts = Counter(), Counter(), Counter(), Counter()
        previous_group = None
        for row in _verification_rows(*paths["prompts"]):
            query_id, group_id = row.get("query_instance_id"), row.get("latent_instance_id")
            if not isinstance(query_id, str) or not isinstance(group_id, str) or query_id not in sources or group_id not in groups:
                raise ValueError("prompt references an unknown source/group")
            source, group = sources[query_id], groups[group_id]
            if source["ordering_id"] != group_id:
                raise ValueError("source/group pairing mismatch")
            key = tuple(row.get(k) for k in ("relation_id", "name_control", "language", "condition"))
            if any(not isinstance(value, str) for value in key):
                raise ValueError("invalid prompt design metadata")
            signature = (query_id, *key)
            if signature not in expected_signatures:
                raise ValueError("duplicate, unexpected or out-of-design prompt")
            expected_signatures.remove(signature)
            if previous_group != group_id:
                components.clear()
                previous_group = group_id
            aliases = group["name_maps"][row["name_control"]][row["language"]]
            if key not in components:
                components[key] = _component(source, row["relation_id"], row["language"], aliases,
                                             group["presentation_plans"][row["condition"]])
            expected = _prompt(source, name, row["name_control"], row["relation_id"], row["language"],
                               row["condition"], components[key], aliases)
            if canonical_bytes(row) != canonical_bytes(expected):
                raise ValueError("prompt schema, text, query, lengths or oracle mismatch")
            prompt_hash = row["prompt_sha256"]
            if prompt_hash in prompt_splits and prompt_splits[prompt_hash] != name:
                raise ValueError("visible prompt leakage")
            prompt_splits[prompt_hash] = name
            if row["prompt_id"] in prompts:
                raise ValueError("duplicate prompt ID")
            prompts[row["prompt_id"]] = {k: v for k, v in row.items() if k not in ("prompt", "lengths")}
            prompt_index[signature] = row["prompt_id"]
            count_prompts["|".join(str(row[k]) for k in ("n", "query_type", "name_control", "relation_id", "language", "condition"))] += 1
            length_key = row["language"] + "|" + row["name_control"]
            length_totals[length_key] += row["lengths"]["reference_token_count"]
            length_counts[length_key] += 1
        if expected_signatures:
            raise ValueError("required prompts are missing from the experiment matrix")
        expected_cells = set()
        for query_id in expected_sources:
            source = sources[query_id]
            designs = {(r, c) for r, c, _, _ in _verification_design(source, spec, plan)}
            for relation, control in designs:
                def prompt_id(q, language, condition):
                    return prompt_index[(q, relation, control, language, condition)]
                if source["query_type"] == "endpoint":
                    contrasts = [("A", "en", "canonical", "en", "equivalent"),
                        ("B", "tr", "canonical", "tr", "equivalent"),
                        ("C", "en", "canonical", "tr", "canonical"),
                        ("D", "en", "equivalent", "tr", "equivalent")]
                    if control == "native_shared":
                        contrasts += [("E", "tr", "canonical", "tr", condition) for condition in CONDITIONS[2:]]
                    for arm, ll, lc, rl, rc in contrasts:
                        expected_cells.add("cell-" + digest([arm, prompt_id(query_id, ll, lc), prompt_id(query_id, rl, rc)]))
                else:
                    for language in LANGUAGES:
                        for condition in CONDITIONS[:2]:
                            expected_cells.add("cell-" + digest(["F", prompt_id(endpoints[source["ordering_id"]], language, condition),
                                                               prompt_id(query_id, language, condition)]))
        expected_cell_count = len(expected_cells)
        for cell in _verification_rows(*paths["cells"]):
            cell_id = cell.get("cell_id")
            if not isinstance(cell_id, str) or cell_id not in expected_cells:
                raise ValueError("duplicate or unexpected comparison cell")
            expected_cells.remove(cell_id)
            if cell.get("left_prompt_id") not in prompts or cell.get("right_prompt_id") not in prompts:
                raise ValueError("comparison cell references an unknown prompt")
            expected = _cell(prompts[cell["left_prompt_id"]], prompts[cell["right_prompt_id"]], cell.get("arm"))
            if canonical_bytes(cell) != canonical_bytes(expected):
                raise ValueError("comparison-cell mismatch")
            key = "|".join(str(cell[k]) for k in ("arm", "n", "name_control", "relation_id", "left_language", "right_language", "left_condition", "right_condition", "right_query_type", "query_distance", "query_direction", "gold_label"))
            count_cells[key] += 1
        if expected_cells:
            raise ValueError("required comparison cells are missing from the experiment matrix")
        expected_summary = {"orders": len(groups), "queries": len(sources), "prompts": len(prompts),
            "comparison_cells": expected_cell_count,
            "sample_indices": {str(n): list(balanced_indices(spec.count, n)) for n in spec.sizes},
            "query_blocks": dict(sorted(query_counts.items())), "query_cells": dict(sorted(query_cells.items())),
            "prompt_cells": dict(sorted(count_prompts.items())), "comparison_cell_counts": dict(sorted(count_cells.items())),
            "reference_token_totals": dict(sorted(length_totals.items())),
            "reference_token_records": dict(sorted(length_counts.items())), "files": summary["files"]}
        if canonical_bytes(summary) != canonical_bytes(expected_summary):
            raise ValueError("split summary counts, cells or length totals differ from actual replay")
        total_prompt_records += len(prompts)
    expected_validation = {"paired_queries_match": True, "surface_normal_forms_match": True,
        "independent_oracles_agree": True, "split_groups_disjoint": True,
        "aliased_orders_disjoint": True, "cross_split_prompt_duplicates": 0}
    if canonical_bytes(manifest.get("validation")) != canonical_bytes(expected_validation):
        raise ValueError("manifest validation claims differ from replay")
    return {"valid": True, "files": len(inventory), "deep": True,
            "orders": len(group_splits), "prompts": len(prompt_splits), "prompt_records": total_prompt_records}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    build = sub.add_parser("build")
    build.add_argument("--config", type=Path, required=True)
    build.add_argument("--output-root", type=Path, required=True)
    build.add_argument("--manifest", type=Path, required=True)
    build.add_argument("--legacy-manifest", type=Path)
    verify = sub.add_parser("verify")
    verify.add_argument("--manifest", type=Path, required=True)
    verify.add_argument("--hashes-only", action="store_true")
    args = parser.parse_args(argv)
    try:
        if args.command == "build":
            result = build_dataset(args.output_root, args.manifest, load_plan(args.config), args.legacy_manifest)
            print(json.dumps({"manifest": str(args.manifest), "splits": {
                key: {field: summary[field] for field in ("orders", "queries", "prompts", "comparison_cells")}
                for key, summary in result["splits"].items()}}, ensure_ascii=False))
        else:
            print(json.dumps(verify_dataset(args.manifest, not args.hashes_only)))
    except (ValueError, OSError, KeyError, TypeError) as error:
        parser.exit(2, f"error: {error}\n")


if __name__ == "__main__":
    main()

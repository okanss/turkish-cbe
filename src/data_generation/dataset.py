"""Streaming JSONL creation, deterministic manifests and independent verification."""

from collections import Counter
import hashlib
import json
import os
from pathlib import Path
import platform
import shutil

from . import GENERATOR_VERSION
from .entities import NAME_RULES_VERSION, entity_pool
from .generator import GenerationPlan, canonical_bytes, iter_block, iter_blocks, valid_distances
from .oracle import DataValidationError
from .validation import BlockValidator, ordering_fingerprint


def _source_hashes() -> dict[str, str]:
    root = Path(__file__).parent
    return {path.name: hashlib.sha256(path.read_bytes()).hexdigest()
            for path in sorted(root.glob("*.py"))}


def _design_metadata(plan: GenerationPlan) -> dict:
    return {
        "name_rules_version": NAME_RULES_VERSION,
        "entity_pools": {kind: {"size": len(entity_pool(kind, 4)),
                               "sha256": hashlib.sha256(canonical_bytes(entity_pool(kind, 4))).hexdigest()}
                         for kind in plan.entity_types},
        "valid_distances_by_n": {str(n): list(valid_distances(n, plan.distances)) for n in plan.sizes},
        "skipped_distances_by_n": {str(n): [d for d in plan.distances if d > n - 3] for n in plan.sizes},
    }


def _read_json(raw: bytes):
    def unique_keys(pairs):
        result = {}
        for key, value in pairs:
            if key in result:
                raise DataValidationError(f"duplicate JSON key: {key}")
            result[key] = value
        return result
    try:
        return json.loads(raw.decode("utf-8"), object_pairs_hook=unique_keys)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise DataValidationError(f"invalid UTF-8 JSON: {exc}") from exc


class _DatasetSummary:
    def __init__(self, plan: GenerationPlan):
        self.plan = plan
        self.ids = set()
        self.orders, self.slots = {}, {}
        self.order_entities = Counter()
        self.branches = {}

    def add(self, record: dict) -> None:
        if (record["seed"] != self.plan.seed or record["entity_type"] not in self.plan.entity_types
                or record["sample_index"] >= self.plan.examples_per_size):
            raise DataValidationError("record metadata disagrees with generation plan")
        if record["instance_id"] in self.ids:
            raise DataValidationError("instance_id must be globally unique")
        self.ids.add(record["instance_id"])
        ordering_id = record["ordering_id"]
        fingerprint = ordering_fingerprint(record)
        slot = (record["n"], record["sample_index"])
        if slot in self.slots and self.slots[slot] != ordering_id:
            raise DataValidationError("query arms disagree on a base ordering id")
        self.slots[slot] = ordering_id
        if ordering_id in self.orders:
            if self.orders[ordering_id] != fingerprint:
                raise DataValidationError("query arms disagree on a shared latent order")
        else:
            self.orders[ordering_id] = fingerprint
            self.order_entities[record["entity_type"]] += 1
        branch = self.branches.setdefault(record["query_type"], {
            "records": 0, "gold_counts": Counter(), "entity_counts": Counter(),
            "query_cells": Counter(), "orders": set(),
        })
        branch["records"] += 1
        branch["gold_counts"][str(record["gold_label"]).lower()] += 1
        branch["entity_counts"][record["entity_type"]] += 1
        branch["query_cells"][f"{record['query_direction']}:{str(record['gold_label']).lower()}"] += 1
        branch["orders"].add(ordering_id)

    def finish(self) -> dict:
        active_sizes = {n for _, n, _, _ in iter_blocks(self.plan)}
        expected_orders = len(active_sizes) * self.plan.examples_per_size
        if len(self.orders) != expected_orders:
            raise DataValidationError("unexpected number of independent base orderings")
        branches = {}
        for name, counts in sorted(self.branches.items()):
            branches[name] = {
                "records": counts["records"], "unique_orderings": len(counts["orders"]),
                **{key: dict(sorted(counts[key].items())) for key in
                   ("gold_counts", "entity_counts", "query_cells")},
            }
        return {"unique_base_orderings": len(self.orders), "records": len(self.ids),
                "base_orderings_by_entity_type": dict(sorted(self.order_entities.items())),
                "branches": branches,
                "validation": {"all_records_valid": True, "independent_oracles_agree": True,
                               "four_cells_balanced_in_every_block": True,
                               "shared_orderings_match_across_arms": True,
                               "globally_unique_instance_ids": True}}


def write_dataset(output_dir: Path, manifest_path: Path, plan: GenerationPlan) -> dict:
    """Create new data only; refuse existing artefacts and roll back failed writes."""
    plan.validate()
    output_dir, manifest_path = Path(output_dir), Path(manifest_path)
    if output_dir.is_symlink() or manifest_path.is_symlink():
        raise FileExistsError("output paths must not be existing symlinks")
    output_dir, manifest_path = output_dir.resolve(), manifest_path.resolve()
    if output_dir.exists() or output_dir.is_symlink():
        raise FileExistsError(f"dataset directory already exists: {output_dir}")
    if manifest_path.exists() or manifest_path.is_symlink():
        raise FileExistsError(f"manifest already exists: {manifest_path}")
    if manifest_path.is_relative_to(output_dir):
        raise ValueError("manifest must be outside the generated data directory")
    output_dir.mkdir(parents=True, exist_ok=False)
    manifest_created = False
    try:
        files, summary = [], _DatasetSummary(plan)
        for relative, n, query_type, distance in iter_blocks(plan):
            path = output_dir / relative
            path.parent.mkdir(parents=True, exist_ok=True)
            digest, size = hashlib.sha256(), 0
            validator = BlockValidator(n, query_type, distance, plan.examples_per_size)
            with path.open("xb") as handle:
                for record in iter_block(plan, n, query_type, distance):
                    validator.add(record)
                    summary.add(record)
                    line = canonical_bytes(record) + b"\n"
                    handle.write(line)
                    digest.update(line)
                    size += len(line)
            files.append({"path": relative, "n": n, "query_type": query_type,
                          "distance": distance, "sha256": digest.hexdigest(), "bytes": size,
                          **validator.finish()})
        manifest = {
            "manifest_schema_version": 1, "artifact_type": "latent_order_dataset",
            "status": "validated_latent_data", "contains_model_results": False,
            "generator_version": GENERATOR_VERSION, "protocol_id": "tr_cbe_latent_v0_2",
            "config": plan.to_dict(),
            "data_root": os.path.relpath(output_dir, manifest_path.parent),
            "environment": {"python": platform.python_version(),
                            "implementation": platform.python_implementation(),
                            "platform": platform.platform()},
            "generator_source_sha256": _source_hashes(),
            **_design_metadata(plan),
            "files": files, "summary": summary.finish(),
            "limitations": ["No natural-language prompts or model outputs are included.",
                            "Query arms and distances share base orders; records are not independent.",
                            "Entity groups are balanced by four-query-cell quartets, not paired aliases.",
                            "This seed-11 material is development data, not a preregistered test set."
                            if plan.seed == 11 else "No train/dev/test split has been assigned."],
        }
        payload = canonical_bytes(manifest) + b"\n"
        manifest_path.parent.mkdir(parents=True, exist_ok=True)
        with manifest_path.open("xb") as handle:
            manifest_created = True
            handle.write(payload)
        return manifest
    except BaseException:
        # Only remove paths created by this invocation, never pre-existing data.
        shutil.rmtree(output_dir)
        if manifest_created:
            manifest_path.unlink(missing_ok=True)
        raise


def _plan_from_manifest(manifest: dict) -> GenerationPlan:
    if (not isinstance(manifest, dict)
            or manifest.get("artifact_type") != "latent_order_dataset"
            or type(manifest.get("manifest_schema_version")) is not int
            or manifest.get("manifest_schema_version") != 1
            or manifest.get("protocol_id") != "tr_cbe_latent_v0_2"
            or manifest.get("generator_version") != GENERATOR_VERSION
            or manifest.get("contains_model_results") is not False
            or manifest.get("status") != "validated_latent_data"):
        raise DataValidationError("invalid or unsupported dataset manifest")
    config = manifest.get("config")
    expected = {"seed", "sizes", "examples_per_size", "entity_types", "query_types", "distances"}
    if not isinstance(config, dict) or set(config) != expected:
        raise DataValidationError("manifest config has missing or unknown fields")
    converted = dict(config)
    for key in ("sizes", "entity_types", "query_types", "distances"):
        if not isinstance(converted[key], list):
            raise DataValidationError(f"manifest config.{key} must be an array")
        converted[key] = tuple(converted[key])
    plan = GenerationPlan(**converted)
    plan.validate()
    return plan


def verify_dataset(manifest_path: Path, replay: bool = False) -> dict:
    """Read every record, verify its semantics and optionally regenerate its bytes."""
    manifest_path = Path(manifest_path).resolve()
    manifest = _read_json(manifest_path.read_bytes())
    plan = _plan_from_manifest(manifest)
    if manifest.get("generator_source_sha256") != _source_hashes():
        raise DataValidationError("generator source hashes differ; use the matching generator release")
    for key, expected in _design_metadata(plan).items():
        if manifest.get(key) != expected:
            raise DataValidationError(f"manifest design metadata mismatch: {key}")
    if not isinstance(manifest.get("data_root"), str) or not manifest["data_root"]:
        raise DataValidationError("manifest data_root must be a nonempty string")
    root_path = manifest_path.parent / manifest["data_root"]
    if root_path.is_symlink():
        raise DataValidationError("dataset root must not be a symlink")
    root = root_path.resolve()
    files = manifest.get("files")
    blocks = list(iter_blocks(plan))
    if not isinstance(files, list) or len(files) != len(blocks):
        raise DataValidationError("manifest file count does not match the design")
    expected_paths = {block[0] for block in blocks}
    if not root.is_dir() or {p.relative_to(root).as_posix() for p in root.rglob("*.jsonl")} != expected_paths:
        raise DataValidationError("dataset has missing or unexpected JSONL files")
    summary = _DatasetSummary(plan)
    for entry, (relative, n, query_type, distance) in zip(files, blocks):
        if (not isinstance(entry, dict) or entry.get("path") != relative
                or entry.get("n") != n or entry.get("query_type") != query_type
                or entry.get("distance") != distance):
            raise DataValidationError("manifest file entry disagrees with its block")
        path = root / relative
        if any(parent.is_symlink() for parent in (path, *path.parents) if parent != root.parent):
            raise DataValidationError("dataset files must not use symlinks")
        digest, size = hashlib.sha256(), 0
        validator = BlockValidator(n, query_type, distance, plan.examples_per_size)
        with path.open("rb") as handle:
            for line in handle:
                if not line.endswith(b"\n") or not line.strip():
                    raise DataValidationError("JSONL needs one nonempty newline-terminated record per line")
                digest.update(line)
                size += len(line)
                record = _read_json(line)
                validator.add(record)
                summary.add(record)
        if digest.hexdigest() != entry.get("sha256") or size != entry.get("bytes"):
            raise DataValidationError(f"file bytes/hash mismatch: {relative}")
        stats = validator.finish()
        if any(entry.get(key) != value for key, value in stats.items()):
            raise DataValidationError(f"file summary mismatch: {relative}")
        if replay:
            regenerated = hashlib.sha256()
            for record in iter_block(plan, n, query_type, distance):
                regenerated.update(canonical_bytes(record) + b"\n")
            if regenerated.hexdigest() != entry["sha256"]:
                raise DataValidationError(f"seed replay mismatch: {relative}")
    result = summary.finish()
    if result != manifest.get("summary"):
        raise DataValidationError("dataset totals or cross-arm grouping mismatch")
    return result

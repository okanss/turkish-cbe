"""Deterministic paired surface samples for human linguistic review.

This creates an unreviewed inspection artefact, never an inference dataset or
a claim that a human has accepted the Turkish templates.
"""

import argparse
from collections import Counter
import csv
import hashlib
import io
import json
import os
from pathlib import Path
import platform
import sys

from data_generation.entities import ENTITY_TYPES
from data_generation.generator import canonical_bytes
from data_generation.validation import validate_record

from . import RENDERER_VERSION
from .turkish import RELATIONS, render

HUMAN_FIELDS = tuple(f"reviewer{number}_{field}" for number in (1, 2) for field in
                     ("grammar", "naturalness", "equivalence", "morphology", "contact_ambiguity")) + ("resolution", "notes")
CSV_FIELDS = (
    "sample_id", "presentation_id", "instance_id", "ordering_id", "n", "entity_type",
    "query_type", "query_distance", "query_direction", "gold_label", "relation_family",
    "relation_id", "variant", "transformations", "factor_axes", "is_multi_transform",
    "canonical_premises", "rendered_premises", "canonical_query", "query", "context",
    "canonical_prompt", "rendered_prompt", "normal_form_sha256", "automatic_equivalence",
    "automatic_gold_unchanged", "morphology_check", "human_review_status", *HUMAN_FIELDS,
)


def _sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _code_hashes() -> dict:
    src = Path(__file__).resolve().parents[1]
    return {path.relative_to(src).as_posix(): _sha(path.read_bytes())
            for folder in ("renderers", "validation") for path in sorted((src / folder).glob("*.py"))}


def _select_sources(dataset_manifest_path: Path):
    manifest_bytes = dataset_manifest_path.read_bytes()
    manifest = json.loads(manifest_bytes)
    if (not isinstance(manifest, dict) or manifest.get("artifact_type") != "latent_order_dataset"
            or manifest.get("contains_model_results") is not False
            or manifest.get("status") != "validated_latent_data"):
        raise ValueError("a validated latent dataset manifest is required")
    config = manifest.get("config", {})
    if (not {4, 8}.issubset(config.get("sizes", []))
            or not set(ENTITY_TYPES).issubset(config.get("entity_types", []))
            or not {"endpoint", "internal", "distance_controlled"}.issubset(config.get("query_types", []))):
        raise ValueError("review sampling needs n=4,8, all entity/query types and distance 1")
    if not isinstance(manifest.get("data_root"), str):
        raise ValueError("invalid dataset data_root")
    root = (dataset_manifest_path.parent / manifest["data_root"]).resolve()
    entries = {item["path"]: item for item in manifest["files"]}
    sources, file_hashes = {}, {}
    query_types = ("endpoint", "internal", "distance_controlled")
    for query_type in query_types:
        for n in (4, 8):
            relative = (f"distance_controlled/d_0001/n_{n:04d}.jsonl" if query_type == "distance_controlled"
                        else f"{query_type}/n_{n:04d}.jsonl")
            if relative not in entries:
                raise ValueError(f"review source block is missing: {relative}")
            path = root / relative
            if not path.resolve().is_relative_to(root) or path.is_symlink():
                raise ValueError("review source must not escape the dataset")
            raw = path.read_bytes()
            if _sha(raw) != entries[relative]["sha256"] or len(raw) != entries[relative]["bytes"]:
                raise ValueError(f"review source file hash/size mismatch: {relative}")
            records = [json.loads(line) for line in raw.splitlines()]
            for record in records:
                validate_record(record)
            sources[(query_type, n)] = sorted(records, key=lambda r: r["sample_index"])
            file_hashes[relative] = _sha(raw)
    used_orderings, selected = set(), []
    cells = (("forward", True), ("forward", False), ("reverse", True), ("reverse", False))
    for relation_index, relation_id in enumerate(RELATIONS):
        for cell_index, (direction, gold) in enumerate(cells):
            entity_type = ENTITY_TYPES[(relation_index + cell_index) % 3]
            query_type = query_types[(2 * relation_index + cell_index) % 3]
            n = 4 if (relation_index + cell_index) % 2 == 0 else 8
            candidates = (r for r in sources[(query_type, n)]
                          if r["entity_type"] == entity_type and r["query_direction"] == direction
                          and r["gold_label"] is gold and r["ordering_id"] not in used_orderings)
            record = next(candidates, None)
            if record is None:
                raise ValueError("insufficient distinct source orders for the fixed review design")
            used_orderings.add(record["ordering_id"])
            selected.append((relation_id, record))
    return manifest_bytes, file_hashes, selected


def generate_review_report(dataset_manifest_path: Path, output_csv: Path,
                           output_manifest: Path, seed: int = 31,
                           include_negation: bool = False) -> dict:
    if type(seed) is not int or not 0 <= seed < 2**64 or type(include_negation) is not bool:
        raise ValueError("invalid review seed or negation option")
    raw_paths = [Path(p) for p in (dataset_manifest_path, output_csv, output_manifest)]
    if any(path.is_symlink() for path in raw_paths):
        raise ValueError("review paths must not be symlinks")
    dataset_manifest_path, output_csv, output_manifest = [path.resolve() for path in raw_paths]
    if output_csv == output_manifest or dataset_manifest_path in (output_csv, output_manifest):
        raise ValueError("review output paths must be distinct from each other and the source")
    if output_csv.exists() or output_manifest.exists():
        raise FileExistsError("review output already exists; preserve existing human annotations")
    manifest_bytes, source_files, selected = _select_sources(dataset_manifest_path)
    variants = tuple(f"P{i}" for i in range(7 if include_negation else 6))
    text = io.StringIO(newline="")
    writer = csv.DictWriter(text, fieldnames=CSV_FIELDS, lineterminator="\n")
    writer.writeheader()
    counts = {key: Counter() for key in ("variant", "relation_id", "entity_type", "gold_label", "query_type", "query_direction")}
    for relation_id, record in selected:
        canonical = render(record, relation_id, seed=seed)
        for variant in variants:
            result = render(record, relation_id, variant, seed=seed, allow_negation=include_negation)
            row = {
                "sample_id": "review-" + result["presentation_id"],
                "presentation_id": result["presentation_id"], "instance_id": record["instance_id"],
                "ordering_id": record["ordering_id"], "n": record["n"], "entity_type": record["entity_type"],
                "query_type": record["query_type"], "query_distance": record["query_distance"],
                "query_direction": record["query_direction"], "gold_label": str(record["gold_label"]).lower(),
                "relation_family": result["family"], "relation_id": relation_id, "variant": variant,
                "transformations": canonical_bytes(result["transformations"]).decode("utf-8"),
                "factor_axes": canonical_bytes(result["factor_axes"]).decode("utf-8"),
                "is_multi_transform": str(result["is_multi_transform"]).lower(),
                "canonical_premises": "\n".join(canonical["premises"]),
                "rendered_premises": "\n".join(result["premises"]),
                "canonical_query": canonical["query"], "query": result["query"], "context": result["context"],
                "canonical_prompt": canonical["prompt"], "rendered_prompt": result["prompt"],
                "normal_form_sha256": result["validation"]["normal_form_sha256"],
                "automatic_equivalence": "true", "automatic_gold_unchanged": "true",
                "morphology_check": "closed_template_checked", "human_review_status": "pending",
                **{field: "" for field in HUMAN_FIELDS},
            }
            writer.writerow(row)
            for key in counts:
                counts[key][row[key]] += 1
    payload = text.getvalue().encode("utf-8")
    manifest = {
        "artifact_type": "turkish_linguistic_review", "report_schema_version": 1,
        "renderer_version": RENDERER_VERSION, "seed": seed, "include_negation": include_negation,
        "row_count": len(selected) * len(variants), "source_contexts": len(selected),
        "human_reviews_completed": 0, "human_review_status": "pending", "contains_model_results": False,
        "csv_file": os.path.relpath(output_csv, output_manifest.parent), "csv_sha256": _sha(payload),
        "source_dataset_manifest_sha256": _sha(manifest_bytes), "source_files_sha256": source_files,
        "source_instance_ids": [record["instance_id"] for _, record in selected],
        "source_ordering_ids": [record["ordering_id"] for _, record in selected],
        "source_code_sha256": _code_hashes(),
        "environment": {"python": platform.python_version(), "platform": platform.platform()},
        "counts": {key: dict(sorted(value.items())) for key, value in counts.items()},
        "automatic_validation": {"all_normal_forms_equal": True, "all_gold_labels_unchanged": True,
                                 "closed_grammar_morphology_checked": True},
        "limitations": ["Automatic checks are not human linguistic approval.",
                        "Samples are a purposive review set, not an inference evaluation.",
                        "P3 changes focus; P4/P5 are mixed interventions; P6 requires strict total order."],
    }
    created = []
    try:
        for path, raw in ((output_csv, payload), (output_manifest, canonical_bytes(manifest) + b"\n")):
            path.parent.mkdir(parents=True, exist_ok=True)
            with path.open("xb") as handle:
                created.append(path)
                handle.write(raw)
    except BaseException:
        for path in created:
            path.unlink(missing_ok=True)
        raise
    return manifest


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Create unreviewed Turkish surface pairs; no inference")
    parser.add_argument("--dataset-manifest", type=Path, default=Path("dataset_manifest.json"))
    parser.add_argument("--output", type=Path, default=Path("data/validation/turkish_samples.csv"))
    parser.add_argument("--manifest", type=Path, default=Path("data/validation/turkish_samples_manifest.json"))
    parser.add_argument("--seed", type=int, default=31)
    parser.add_argument("--include-negation", action="store_true")
    args = parser.parse_args(argv)
    try:
        result = generate_review_report(args.dataset_manifest, args.output, args.manifest,
                                        args.seed, args.include_negation)
        print(json.dumps({"row_count": result["row_count"], "source_contexts": result["source_contexts"],
                          "human_reviews_completed": 0, "contains_model_results": False,
                          "csv_sha256": result["csv_sha256"]}, indent=2))
        return 0
    except (OSError, ValueError, KeyError, TypeError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())

"""Strict record validation with independent position and premise-graph oracles."""

from collections import Counter
from functools import lru_cache
import hashlib

from . import GENERATOR_VERSION
from .entities import ENTITY_TYPES, entity_pool
from .generator import canonical_bytes, QUERY_TYPES
from .oracle import DataValidationError, answer_from_edges, answer_from_order

RECORD_FIELDS = frozenset((
    "instance_id", "ordering_id", "sample_index", "seed", "n", "entity_type",
    "latent_order", "premise_edges", "query_subject", "query_object", "query_relation",
    "query_type", "query_direction", "query_distance", "gold_label", "generator_version",
))


@lru_cache(maxsize=len(ENTITY_TYPES))
def _allowed_entities(entity_type: str) -> frozenset[str]:
    return frozenset(entity_pool(entity_type, 4))


def validate_record(record: dict) -> None:
    if not isinstance(record, dict) or set(record) != RECORD_FIELDS:
        raise DataValidationError("record has missing or unknown fields")
    for name, prefix in (("instance_id", "inst-"), ("ordering_id", "ord-")):
        value = record[name]
        if (not isinstance(value, str) or not value.startswith(prefix)
                or len(value) != len(prefix) + 64
                or any(c not in "0123456789abcdef" for c in value[len(prefix):])):
            raise DataValidationError(f"{name} must be a prefixed SHA-256 identifier")
    for name, minimum, maximum in (("n", 4, 1024), ("seed", 0, 2**64 - 1),
                                    ("sample_index", 0, None), ("query_distance", 1, None)):
        value = record[name]
        if type(value) is not int or value < minimum or (maximum is not None and value > maximum):
            raise DataValidationError(f"{name} is outside the integer domain")
    if record["generator_version"] != GENERATOR_VERSION:
        raise DataValidationError("unsupported generator_version")
    if not isinstance(record["entity_type"], str) or record["entity_type"] not in ENTITY_TYPES:
        raise DataValidationError("unknown entity_type")
    if not isinstance(record["query_type"], str) or record["query_type"] not in QUERY_TYPES:
        raise DataValidationError("unknown query_type")
    if type(record["gold_label"]) is not bool:
        raise DataValidationError("gold_label must be a JSON Boolean")
    order = record["latent_order"]
    if not isinstance(order, list) or len(order) != record["n"]:
        raise DataValidationError("latent_order must contain n entities")
    subject, object_, relation = (record[k] for k in
                                  ("query_subject", "query_object", "query_relation"))
    gold = answer_from_order(order, subject, object_, relation)
    # Inventory is sorted to ensure the graph path does not receive latent ranks.
    graph_gold = answer_from_edges(sorted(order), record["premise_edges"], subject, object_, relation)
    if gold != graph_gold or gold is not record["gold_label"]:
        raise DataValidationError("stored gold disagrees with an independent exact oracle")
    positions = {entity: i for i, entity in enumerate(order)}
    if any(positions[a] + 1 != positions[b] for a, b in record["premise_edges"]):
        raise DataValidationError("every premise must be a forward adjacent latent comparison")
    pool = _allowed_entities(record["entity_type"])
    if any(entity not in pool for entity in order):
        raise DataValidationError("entity is outside its versioned naming pool")
    i, j = positions[subject], positions[object_]
    if abs(i - j) != record["query_distance"]:
        raise DataValidationError("query_distance disagrees with latent positions")
    direction = "forward" if i < j else "reverse"
    if record["query_direction"] != direction:
        raise DataValidationError("query_direction disagrees with latent positions")
    if record["query_type"] == "endpoint":
        if {i, j} != {0, record["n"] - 1}:
            raise DataValidationError("endpoint query must compare both endpoints")
    elif not (0 < i < record["n"] - 1 and 0 < j < record["n"] - 1):
        raise DataValidationError("internal query must exclude both chain endpoints")


class BlockValidator:
    """Streaming validation; repeated queries never inflate independent orders."""

    def __init__(self, n: int, query_type: str, distance: int | None, expected_count: int):
        self.n, self.query_type, self.distance = n, query_type, distance
        self.expected_count = expected_count
        self.ids, self.ordering_ids, self.indices = set(), set(), set()
        self.gold, self.cells, self.entities, self.entity_cells = Counter(), Counter(), Counter(), Counter()

    def add(self, record: dict) -> None:
        validate_record(record)
        if record["n"] != self.n or record["query_type"] != self.query_type:
            raise DataValidationError("record is in the wrong query/size block")
        if self.distance is not None and record["query_distance"] != self.distance:
            raise DataValidationError("record is in the wrong distance block")
        for name, seen in (("instance_id", self.ids), ("ordering_id", self.ordering_ids),
                            ("sample_index", self.indices)):
            if record[name] in seen:
                raise DataValidationError(f"duplicate {name} in block")
            seen.add(record[name])
        label = str(record["gold_label"]).lower()
        cell = f"{record['query_direction']}:{label}"
        self.gold[label] += 1
        self.cells[cell] += 1
        self.entities[record["entity_type"]] += 1
        self.entity_cells[(record["entity_type"], cell)] += 1

    def finish(self) -> dict:
        expected_cells = {f"{d}:{g}" for d in ("forward", "reverse") for g in ("false", "true")}
        if (len(self.ids) != self.expected_count or self.indices != set(range(self.expected_count))
                or set(self.cells) != expected_cells
                or any(v != self.expected_count // 4 for v in self.cells.values())):
            raise DataValidationError("block count, indices or four-cell balance is invalid")
        for entity_type in self.entities:
            if len({self.entity_cells[(entity_type, c)] for c in expected_cells}) != 1:
                raise DataValidationError("query cells are not balanced within an entity type")
        return {"records": len(self.ids), "gold_counts": dict(sorted(self.gold.items())),
                "query_cells": dict(sorted(self.cells.items())),
                "entity_counts": dict(sorted(self.entities.items())),
                "validation": {"all_records_valid": True, "independent_oracles_agree": True,
                               "adjacency_and_acyclicity": True, "four_cells_balanced": True}}


def ordering_fingerprint(record: dict) -> str:
    return hashlib.sha256(canonical_bytes([record["n"], record["entity_type"],
                                          record["latent_order"], record["premise_edges"]])).hexdigest()

"""Parameterised chains and separately identified query arms."""

from dataclasses import asdict, dataclass
import hashlib
import json
import random

from . import GENERATOR_VERSION
from .entities import ENTITY_TYPES, sample_entities
from .oracle import answer_from_order

DEFAULT_SIZES = (4, 8, 16, 32, 64, 128, 256, 512, 1024)
DEFAULT_DISTANCES = (1, 2, 4, 8, 16, 32, 64, 128, 256, 512)
QUERY_TYPES = ("endpoint", "internal", "distance_controlled")


def canonical_bytes(value) -> bytes:
    return json.dumps(value, sort_keys=True, ensure_ascii=False,
                      separators=(",", ":"), allow_nan=False).encode("utf-8")


def _digest(*parts) -> str:
    return hashlib.sha256(canonical_bytes(parts)).hexdigest()


def derive_seed(seed: int, *namespace) -> int:
    """A versioned, stable SHA-256 namespace; never Python's salted hash()."""
    return int(_digest(GENERATOR_VERSION, seed, *namespace), 16)


@dataclass(frozen=True)
class GenerationPlan:
    seed: int = 11
    sizes: tuple[int, ...] = DEFAULT_SIZES
    examples_per_size: int = 200
    entity_types: tuple[str, ...] = ENTITY_TYPES
    query_types: tuple[str, ...] = QUERY_TYPES
    distances: tuple[int, ...] = DEFAULT_DISTANCES

    def validate(self) -> None:
        if type(self.seed) is not int or not 0 <= self.seed < 2**64:
            raise ValueError("seed must be an integer in [0, 2**64)")
        if (type(self.examples_per_size) is not int or self.examples_per_size < 4
                or self.examples_per_size % 4):
            raise ValueError("examples_per_size must be a positive multiple of four")
        for key, minimum, maximum in (("sizes", 4, 1024), ("distances", 1, None)):
            values = getattr(self, key)
            if (not isinstance(values, tuple) or not values
                    or any(type(v) is not int or v < minimum
                           or (maximum is not None and v > maximum) for v in values)
                    or tuple(sorted(set(values))) != values):
                raise ValueError(f"{key} must be a nonempty, strictly increasing integer tuple")
        for key, allowed in (("entity_types", ENTITY_TYPES), ("query_types", QUERY_TYPES)):
            values = getattr(self, key)
            if (not isinstance(values, tuple) or not values
                    or any(not isinstance(v, str) or v not in allowed for v in values)
                    or len(set(values)) != len(values)):
                raise ValueError(f"{key} must be a unique nonempty supported tuple")
        if ("distance_controlled" in self.query_types
                and not any(valid_distances(n, self.distances) for n in self.sizes)):
            raise ValueError("no requested distance is feasible at any requested size")

    def to_dict(self) -> dict:
        return asdict(self)


def valid_distances(n: int, distances: tuple[int, ...]) -> tuple[int, ...]:
    """Internal endpoints must both lie in positions 1..n-2 inclusive."""
    if type(n) is not int or not 4 <= n <= 1024:
        raise ValueError("n must be an integer in [4, 1024]")
    if (not isinstance(distances, tuple) or not distances
            or any(type(d) is not int or d < 1 for d in distances)
            or tuple(sorted(set(distances))) != distances):
        raise ValueError("distances must be a positive increasing integer tuple")
    return tuple(d for d in distances if d <= n - 3)


def generate_record(plan: GenerationPlan, n: int, index: int, query_type: str,
                    distance: int | None = None) -> dict:
    plan.validate()
    if type(n) is not int or n not in plan.sizes:
        raise ValueError("n must belong to the plan")
    if type(index) is not int or not 0 <= index < plan.examples_per_size:
        raise ValueError("sample index is outside the plan")
    if not isinstance(query_type, str) or query_type not in plan.query_types:
        raise ValueError("query_type must belong to the plan")
    if query_type == "distance_controlled":
        if type(distance) is not int or distance not in valid_distances(n, plan.distances):
            raise ValueError("distance must be requested and feasible for internal endpoints")
    elif distance is not None:
        raise ValueError("distance is only accepted for distance_controlled queries")

    # Four consecutive independent draws share a naming condition, not an order.
    # This retains exact direction/gold balance in every entity group.
    entity_type = plan.entity_types[(index // 4 + n.bit_length()) % len(plan.entity_types)]
    names = sample_entities(entity_type, n, random.Random(derive_seed(plan.seed, "names", n, index)))
    permutation = list(range(n))
    random.Random(derive_seed(plan.seed, "order", n, index)).shuffle(permutation)
    order = [names[p] for p in permutation]
    ordering_id = "ord-" + _digest(GENERATOR_VERSION, plan.seed, n, index, permutation)
    edges = [[a, b] for a, b in zip(order, order[1:])]
    random.Random(derive_seed(plan.seed, "premise_order", n, index)).shuffle(edges)
    rng = random.Random(derive_seed(plan.seed, "query", n, index, query_type, distance))
    if query_type == "endpoint":
        low, high = 0, n - 1
    elif query_type == "internal":
        low, high = sorted(rng.sample(range(1, n - 1), 2))
    else:
        low = rng.randint(1, n - distance - 2)
        high = low + distance
    cell = index % 4
    forward, expected_gold = cell < 2, cell % 2 == 0
    subject, object_ = (order[low], order[high]) if forward else (order[high], order[low])
    relation = "before" if forward == expected_gold else "after"
    gold = answer_from_order(order, subject, object_, relation)
    if gold is not expected_gold:
        raise AssertionError("query cell construction disagrees with exact oracle")
    instance_id = "inst-" + _digest(ordering_id, entity_type, query_type, distance,
                                   subject, object_, relation)
    return {
        "instance_id": instance_id, "ordering_id": ordering_id,
        "sample_index": index, "seed": plan.seed, "n": n,
        "entity_type": entity_type, "latent_order": order, "premise_edges": edges,
        "query_subject": subject, "query_object": object_, "query_relation": relation,
        "query_type": query_type, "query_direction": "forward" if forward else "reverse",
        "query_distance": high - low, "gold_label": gold,
        "generator_version": GENERATOR_VERSION,
    }


def iter_block(plan: GenerationPlan, n: int, query_type: str,
               distance: int | None = None):
    """Stable file order; adding other arms or sizes cannot perturb a record."""
    plan.validate()
    indices = list(range(plan.examples_per_size))
    random.Random(derive_seed(plan.seed, "record_order", n, query_type, distance)).shuffle(indices)
    for index in indices:
        yield generate_record(plan, n, index, query_type, distance)


def iter_blocks(plan: GenerationPlan):
    plan.validate()
    for query_type in plan.query_types:
        for n in plan.sizes:
            distances = valid_distances(n, plan.distances) if query_type == "distance_controlled" else (None,)
            for distance in distances:
                relative = (f"distance_controlled/d_{distance:04d}/n_{n:04d}.jsonl"
                            if distance is not None else f"{query_type}/n_{n:04d}.jsonl")
                yield relative, n, query_type, distance

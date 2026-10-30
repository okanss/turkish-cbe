"""Exact, language-independent oracles for strict total-order chains.

``answer_from_order`` checks the recorded latent ordering. The independent
``answer_from_edges`` oracle reconstructs the entire chain using only its
directed premises; the ordering of its ``entities`` argument is irrelevant.
Both paths use O(n) time and memory and never call a language model.
"""

from __future__ import annotations

from collections.abc import Sequence
import unicodedata


class DataValidationError(ValueError):
    """An ordering, premise graph, or query violates the data contract."""


def _validate_entity(value: str, field: str) -> None:
    if not isinstance(value, str) or not value.strip():
        raise DataValidationError(f"{field} must be a nonempty string")
    if unicodedata.normalize("NFC", value) != value:
        raise DataValidationError(f"{field} must be NFC-normalized")


def _entity_positions(entities: Sequence[str]) -> dict[str, int]:
    if not isinstance(entities, (list, tuple)) or len(entities) < 2:
        raise DataValidationError("entities must be a list or tuple of at least two entities")
    positions: dict[str, int] = {}
    for index, entity in enumerate(entities):
        _validate_entity(entity, f"entities[{index}]")
        if entity in positions:
            raise DataValidationError("entities must be unique")
        positions[entity] = index
    return positions


def _validate_query(
    positions: dict[str, int], subject: str, object: str, relation: str,
) -> None:
    _validate_entity(subject, "query_subject")
    _validate_entity(object, "query_object")
    if subject not in positions or object not in positions:
        raise DataValidationError("query entities must belong to the ordering")
    if subject == object:
        raise DataValidationError("a strict ordering query must compare distinct entities")
    if not isinstance(relation, str) or relation not in ("before", "after"):
        raise DataValidationError("query_relation must be 'before' or 'after'")


def answer_from_order(
    latent_order: Sequence[str], subject: str, object: str, relation: str,
) -> bool:
    """Return the exact truth value of a query in a validated latent order.

    The sequence runs from first to last. ``before`` means strictly smaller
    position and ``after`` means strictly greater position; ties are invalid.
    Invalid inputs consistently raise :class:`DataValidationError`.
    """
    positions = _entity_positions(latent_order)
    _validate_query(positions, subject, object, relation)
    if relation == "before":
        return positions[subject] < positions[object]
    return positions[subject] > positions[object]


def answer_from_edges(
    entities: Sequence[str], premise_edges: Sequence[Sequence[str]],
    subject: str, object: str, relation: str,
) -> bool:
    """Solve a query solely from a validated directed chain of premises.

    Each edge ``(a, b)`` asserts that ``a`` is before ``b``. The premises must
    form one connected, acyclic path covering every entity with exactly n-1
    edges. Input entity and edge sequence order have no semantic effect.
    The traversal is iterative, including for chains of 1,024 entities.
    """
    vertices = _entity_positions(entities)
    _validate_query(vertices, subject, object, relation)
    if not isinstance(premise_edges, (list, tuple)):
        raise DataValidationError("premise_edges must be a list or tuple")
    if len(premise_edges) != len(vertices) - 1:
        raise DataValidationError("a chain must contain exactly n-1 premise edges")

    outgoing: dict[str, str] = {}
    incoming: set[str] = set()
    seen_edges: set[tuple[str, str]] = set()
    for index, edge in enumerate(premise_edges):
        if not isinstance(edge, (list, tuple)) or len(edge) != 2:
            raise DataValidationError(f"premise_edges[{index}] must be an entity pair")
        before, after = edge
        _validate_entity(before, f"premise_edges[{index}][0]")
        _validate_entity(after, f"premise_edges[{index}][1]")
        if before not in vertices or after not in vertices:
            raise DataValidationError("premise edges must reference known entities")
        if before == after:
            raise DataValidationError("premise edges must not contain self-loops")
        pair = (before, after)
        if pair in seen_edges:
            raise DataValidationError("premise edges must be unique")
        seen_edges.add(pair)
        if before in outgoing or after in incoming:
            raise DataValidationError("each chain vertex has at most one incoming and outgoing edge")
        outgoing[before] = after
        incoming.add(after)

    sources = [entity for entity in vertices if entity not in incoming]
    sinks = [entity for entity in vertices if entity not in outgoing]
    if len(sources) != 1 or len(sinks) != 1:
        raise DataValidationError("a chain must have exactly one source and one sink")

    # The provided entities sequence is a vertex inventory, not a latent order.
    # Reconstruct all positions by following premise edges from the sole source.
    positions: dict[str, int] = {}
    current = sources[0]
    while True:
        if current in positions:
            raise DataValidationError("premise edges contain a cycle")
        positions[current] = len(positions)
        if current not in outgoing:
            break
        current = outgoing[current]
    if len(positions) != len(vertices):
        raise DataValidationError("premise edges are disconnected or contain an isolated cycle")
    if relation == "before":
        return positions[subject] < positions[object]
    return positions[subject] > positions[object]

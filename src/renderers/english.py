"""Controlled English order statements and their surface-to-logic parser.

The grammar is intentionally closed. An inverse statement swaps the canonical
before edge, whereas a query keeps its displayed argument order and relation.
The templates do not use an LLM or infer truth from an answer label.
"""

from dataclasses import dataclass
from functools import lru_cache
import re
import unicodedata


ENGLISH_RENDERER_VERSION = "0.4.0"


@dataclass(frozen=True)
class EnglishRelation:
    forward: str
    reverse: str
    context: str
    statement_frame: str = "{a} is {relation} {b}."
    question_frame: str = "Is {a} {relation} {b}?"


RELATIONS = {
    "spatial_front": EnglishRelation(
        "in front of", "behind",
        "The people occupy one straight axis with a fixed orientation. Front and "
        "behind refer to that shared axis, regardless of which way anyone faces."),
    "spatial_right": EnglishRelation(
        "to the right of", "to the left of",
        "The people occupy one horizontal line. Right and left use the same "
        "fixed viewing direction for every person."),
    "spatial_above": EnglishRelation(
        "above", "below",
        "The people's positions are compared on one vertical axis. Above and "
        "below mean only higher and lower positions, without physical contact "
        "or standing on another object."),
    "temporal_arrival": EnglishRelation(
        "before", "after",
        "Every person arrived at the same event exactly once. Only arrival "
        "times are compared, and every arrival time is different.",
        "{a} arrived {relation} {b}.", "Did {a} arrive {relation} {b}?"),
    "temporal_completion": EnglishRelation(
        "before", "after",
        "Each entity label names a task, including labels that look like person "
        "names. Every task finished exactly once. Only finishing times are compared.",
        "Task {a} finished {relation} task {b}.",
        "Did task {a} finish {relation} task {b}?"),
    "comparison_height": EnglishRelation(
        "taller than", "shorter than",
        "The people's heights are measured in the same units. Taller and shorter "
        "refer only to height, and every height is different."),
    "comparison_speed": EnglishRelation(
        "faster than", "slower than",
        "The people's constant speeds are measured under the same conditions. "
        "Faster means a higher speed, and slower means a lower speed."),
    "comparison_weight": EnglishRelation(
        "heavier than", "lighter than",
        "The people's weights are measured in the same units. Heavier means "
        "greater weight, and lighter means lower weight."),
    "ranking": EnglishRelation(
        "higher than", "lower than",
        "The people have one performance ranking. A higher rank is better and "
        "a lower rank is worse; no rank is shared.",
        "{a} is ranked {relation} {b}.", "Is {a} ranked {relation} {b}?"),
}


def _relation(relation_id: str) -> EnglishRelation:
    if not isinstance(relation_id, str) or relation_id not in RELATIONS:
        raise ValueError("unknown English relation_id")
    return RELATIONS[relation_id]


def _name(value: str) -> str:
    if (not isinstance(value, str) or not value or value != value.strip()
            or "  " in value or unicodedata.normalize("NFC", value) != value
            or not all(character.isalpha() or character.isdecimal()
                       or character == " " for character in value)):
        raise ValueError("entity names must be nonempty NFC letters, digits and single spaces")
    return value


def _arguments(a: str, b: str, direction: str) -> tuple[str, str]:
    a, b = _name(a), _name(b)
    if a == b:
        raise ValueError("comparison arguments must be distinct")
    if direction not in ("before", "after"):
        raise ValueError("direction must be before or after")
    return a, b


def statement(a: str, b: str, relation_id: str, direction: str = "before",
              style: str = "canonical") -> str:
    """Express ``a relation b``; equivalence uses swapped inverse arguments."""
    spec = _relation(relation_id)
    a, b = _arguments(a, b, direction)
    if style != "canonical":
        raise ValueError("only the canonical English statement style is supported")
    word = spec.forward if direction == "before" else spec.reverse
    return spec.statement_frame.format(a=a, b=b, relation=word)


def question(a: str, b: str, relation_id: str, direction: str = "before") -> str:
    """Ask about exactly the supplied arguments and relation direction."""
    spec = _relation(relation_id)
    a, b = _arguments(a, b, direction)
    word = spec.forward if direction == "before" else spec.reverse
    return spec.question_frame.format(a=a, b=b, relation=word)


def context(relation_id: str, symbolic: bool = False) -> str:
    spec = _relation(relation_id)
    if type(symbolic) is not bool:
        raise ValueError("symbolic must be Boolean")
    output = (spec.context + " In this problem the stated relation is a strict total "
              "order: all values are distinct, with no ties or incomparable pairs. "
              "Every premise uses the same ordering axis.")
    if symbolic:
        output += " Alphanumeric entity names are identifiers and convey no ordering."
    return output


def compose_prompt(context: str, entity_names, premises, query: str) -> str:
    """Build the visible prompt without inserting labels or latent-order data."""
    names = _inventory(entity_names)
    if (not isinstance(context, str) or not context
            or not isinstance(query, str) or not query):
        raise ValueError("context and query must be nonempty strings")
    statements = list(premises)
    if not statements or any(not isinstance(item, str) or not item for item in statements):
        raise ValueError("premises must be nonempty strings")
    return (context + "\nEntities: " + ", ".join(names) + ".\nPremises:\n"
            + "\n".join("- " + item for item in statements) + "\nQuestion: " + query
            + "\nProvide your answer only as YES or NO.")


def _inventory(names) -> tuple[str, ...]:
    if isinstance(names, (str, bytes)):
        raise ValueError("names must be an entity inventory")
    try:
        values = tuple(names)
    except TypeError as exc:
        raise ValueError("names must be an entity inventory") from exc
    if any(not isinstance(name, str) for name in values):
        raise ValueError("entity names must be strings")
    return _checked_inventory(values)


@lru_cache(maxsize=64)
def _checked_inventory(values: tuple[str, ...]) -> tuple[str, ...]:
    for name in values:
        _name(name)
    if len(values) < 2 or len(set(values)) != len(values):
        raise ValueError("entity inventory must contain at least two unique names")
    return values


@lru_cache(maxsize=64)
def _patterns(inventory: tuple[str, ...], relation_id: str, query: bool):
    spec = _relation(relation_id)
    # Longer alternatives first handle inventories such as Ada and Ada Demir.
    alternatives = "(?:" + "|".join(re.escape(name) for name in
                                    sorted(inventory, key=lambda name: (-len(name), name))) + ")"
    frame = spec.question_frame if query else spec.statement_frame
    patterns = []
    for direction, word in (("before", spec.forward), ("after", spec.reverse)):
        # Only the fixed grammar frame is interpolated, never user supplied text.
        pattern = re.escape(frame)
        pattern = pattern.replace(re.escape("{a}"), "(?P<a>" + alternatives + ")")
        pattern = pattern.replace(re.escape("{b}"), "(?P<b>" + alternatives + ")")
        pattern = pattern.replace(re.escape("{relation}"), re.escape(word))
        patterns.append((direction, re.compile(pattern)))
    return tuple(patterns)


def _parse(text: str, names, relation_id: str, *, query: bool) -> tuple[str, str, str]:
    """Recognize literal grammar and entity spans, independent of order/oracle."""
    _relation(relation_id)
    inventory = _inventory(names)
    if not isinstance(text, str):
        raise ValueError("English text must be a string")
    for direction, pattern in _patterns(inventory, relation_id, query):
        match = pattern.fullmatch(text)
        if match is not None:
            a, b = match.group("a"), match.group("b")
            if a == b:
                raise ValueError("self comparisons are not permitted")
            return a, b, direction
    raise ValueError("English text does not match the controlled relation grammar")


def normalize_statement(text: str, names, relation_id: str) -> tuple[str, str]:
    """Return the canonical before edge actually stated by the surface text."""
    a, b, direction = _parse(text, names, relation_id, query=False)
    return (a, b) if direction == "before" else (b, a)


def normalize_query(text: str, names, relation_id: str) -> tuple[str, str, str]:
    """Return displayed query arguments and before/after, without a truth label."""
    return _parse(text, names, relation_id, query=True)

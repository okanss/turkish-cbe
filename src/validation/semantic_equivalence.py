"""Independently parse controlled Turkish back into language-free relations.

This is a finite grammar for the published renderer templates, not a general
Turkish parser or a substitute for human linguistic review. Recognition reads
the actual sentence and checks every inflected entity against morphology;
neither transformation labels nor stored normal forms determine its meaning.

``before`` denotes the declared positive end of each relation family, including
right/above/taller/faster/heavier/higher-ranked. Negative statements are
normalised ONLY under a strict total order over distinct entities: not R(a,b)
is then R(b,a). ``validate_presentation`` requires this assumption in both the
context text and the metadata, and confines negation to the optional P6 arm.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
import hashlib
import random
import re
import unicodedata

from data_generation.generator import canonical_bytes
from data_generation.oracle import DataValidationError, answer_from_edges, answer_from_order
from data_generation.validation import validate_record
from renderers.morphology import MorphologyError, ablative, genitive


class SemanticEquivalenceError(ValueError):
    """A surface form is unsupported or changes the source problem's meaning."""


# These recognition rules are deliberately independent of renderer functions.
# Each pair contains the positive and inverse predicates, respectively.
_SPATIAL = {
    "spatial_front": (("önündedir", "arkasındadır"), ("önünde", "arkasında")),
    "spatial_right": (("sağındadır", "solundadır"), ("sağında", "solunda")),
    "spatial_above": (("üstündedir", "altındadır"), ("üstünde", "altında")),
}
_COMPARISON = {
    "comparison_height": (
        ("uzundur", "kısadır"), ("uzun", "kısa"), ("mudur", "mıdır"),
        "boyu", "boyundan", ("uzundur", "kısadır"),
    ),
    "comparison_speed": (
        ("hızlıdır", "yavaştır"), ("hızlı", "yavaş"), ("mıdır", "mıdır"),
        "hızı", "hızından", ("yüksektir", "düşüktür"),
    ),
    "comparison_weight": (
        ("ağırdır", "hafiftir"), ("ağır", "hafif"), ("mıdır", "midir"),
        "ağırlığı", "ağırlığından", ("fazladır", "azdır"),
    ),
}
_TEMPORAL = ("temporal_arrival", "temporal_completion")
RELATION_IDS = tuple((*_SPATIAL, *_TEMPORAL, *_COMPARISON, "ranking"))
_SUFFIX = r"[a-zçğıöşü]+"
_RENDERER_VERSION = "0.3.0"
_FAMILIES = {
    **dict.fromkeys(_SPATIAL, "spatial"), **dict.fromkeys(_TEMPORAL, "temporal"),
    **dict.fromkeys(_COMPARISON, "comparison"), "ranking": "ranking",
}
# Separate, exact metadata contracts catch mislabeled factors, including claims
# that morphology-only changes occurred when syntax or lexemes changed too.
_FACTOR_CONTRACTS = {
    "P0": ((), ()),
    "P1": (("premise_order", "entity_order"), ("list_order",)),
    "P2": (("argument_swap", "converse_lexeme"), ("expression_direction",)),
    "P3": (("word_order",), ("syntax",)),
    "P4": (("possessive_nominalization", "case_change", "attribute_lexeme"),
           ("morphology", "syntax", "lexical")),
    "P5": (("premise_order", "entity_order", "argument_swap", "converse_lexeme",
            "possessive_nominalization", "case_change", "attribute_lexeme"),
           ("list_order", "expression_direction", "morphology", "syntax", "lexical")),
    "P6": (("controlled_negation", "opposite_lexeme"), ("negation", "lexical")),
}


@dataclass(frozen=True)
class _Rule:
    pattern: re.Pattern[str]
    relation: str
    negated: bool = False
    style: str = "canonical"


@dataclass(frozen=True)
class _Clause:
    subject: str
    object: str
    relation: str
    negated: bool
    style: str


def _inventory(entity_names: tuple[str, ...]) -> tuple[str, ...]:
    if not isinstance(entity_names, (list, tuple)) or len(entity_names) < 2:
        raise SemanticEquivalenceError("entity_names must contain at least two unique names")
    if any(not isinstance(name, str) or not name or name != name.strip()
           or unicodedata.normalize("NFC", name) != name for name in entity_names):
        raise SemanticEquivalenceError("entity_names must be nonempty NFC strings")
    if len(set(entity_names)) != len(entity_names):
        raise SemanticEquivalenceError("entity_names contains duplicates")
    return tuple(sorted(entity_names))


def _text(text: str) -> None:
    if (not isinstance(text, str) or not text
            or unicodedata.normalize("NFC", text) != text):
        raise SemanticEquivalenceError("surface text must be a nonempty NFC string")


def _compile(template: str, names: tuple[str, ...], relation: str,
             negated: bool = False, style: str = "canonical") -> _Rule:
    # Longest-first alternatives safely distinguish names sharing a prefix.
    alternatives = "(?:" + "|".join(re.escape(name) for name in sorted(names, key=lambda n: (-len(n), n))) + ")"
    tokens = {
        "a": f"(?P<a>{alternatives})", "b": f"(?P<b>{alternatives})",
        "ag": f"(?P<a>{alternatives})(?P<ag>'{_SUFFIX})",
        "bg": f"(?P<b>{alternatives})(?P<bg>'{_SUFFIX})",
        "ab": f"(?P<a>{alternatives})(?P<ab>'{_SUFFIX})",
        "bb": f"(?P<b>{alternatives})(?P<bb>'{_SUFFIX})",
    }
    parts = re.split(r"(\{(?:a|b|ag|bg|ab|bb)\})", template)
    expression = "".join(tokens[part[1:-1]] if part.startswith("{") else re.escape(part) for part in parts)
    return _Rule(re.compile(expression), relation, negated, style)


@lru_cache(maxsize=12)
def _rules(names: tuple[str, ...], relation_id: str, query: bool) -> tuple[_Rule, ...]:
    if relation_id not in RELATION_IDS:
        raise SemanticEquivalenceError("unsupported relation_id")
    rules: list[_Rule] = []

    def add(template: str, relation: str, negated: bool = False, style: str = "canonical") -> None:
        rules.append(_compile(template, names, relation, negated, style))

    for index, direction in enumerate(("before", "after")):
        if relation_id in _SPATIAL:
            predicates, positional = _SPATIAL[relation_id]
            if query:
                particle = "midir" if positional[index].endswith("e") else "mıdır"
                add("{a}, {bg} " + positional[index] + " " + particle + "?", direction)
            else:
                add("{a}, {bg} " + predicates[index] + ".", direction)
                add("{bg} " + predicates[index] + " {a}.", direction, style="alternative")
                add("{ag} konumu, {bg} konumunun " + predicates[index] + ".", direction, style="morphological")
                add("{a}, {bg} " + positional[index] + " değildir.", direction, True, "negated")
        elif relation_id in _TEMPORAL:
            adverb = ("önce", "sonra")[index]
            if relation_id == "temporal_arrival":
                if query:
                    add("{a}, {bb} " + adverb + " gelmiş midir?", direction)
                else:
                    add("{a}, {bb} " + adverb + " gelmiştir.", direction)
                    add("{bb} " + adverb + " gelmiştir {a}.", direction, style="alternative")
                    add("{ag} geliş zamanı, {bg} geliş zamanından daha " + ("erkendir", "geçtir")[index] + ".", direction, style="morphological")
                    add("{a}, {bb} " + adverb + " gelmemiştir.", direction, True, "negated")
            elif query:
                add("{a} adlı görev, {b} adlı görevden " + adverb + " tamamlanmış mıdır?", direction)
            else:
                add("{a} adlı görev, {b} adlı görevden " + adverb + " tamamlanmıştır.", direction)
                add("{b} adlı görevden " + adverb + " tamamlanmıştır {a} adlı görev.", direction, style="alternative")
                add("{a} adlı görevin tamamlanma zamanı, {b} adlı görevin tamamlanma zamanından daha " + ("erkendir", "geçtir")[index] + ".", direction, style="morphological")
                add("{a} adlı görev, {b} adlı görevden " + adverb + " tamamlanmamıştır.", direction, True, "negated")
        elif relation_id in _COMPARISON:
            predicates, adjectives, questions, possession, ablative_noun, morphology_predicates = _COMPARISON[relation_id]
            if query:
                add("{a}, {bb} daha " + adjectives[index] + " " + questions[index] + "?", direction)
            else:
                add("{a}, {bb} daha " + predicates[index] + ".", direction)
                add("{bb} daha " + predicates[index] + " {a}.", direction, style="alternative")
                add("{ag} " + possession + ", {bg} " + ablative_noun + " daha " + morphology_predicates[index] + ".", direction, style="morphological")
                add("{a}, {bb} daha " + adjectives[index] + " değildir.", direction, True, "negated")
        else:
            ranking = ("üst", "alt")[index]
            if query:
                add("{a}, {bb} daha " + ranking + " sırada mıdır?", direction)
            else:
                add("{a}, {bb} daha " + ranking + " sıradadır.", direction)
                add("{bb} daha " + ranking + " sıradadır {a}.", direction, style="alternative")
                add("{ag} sırası, {bg} sırasından daha " + ("üsttedir", "alttadır")[index] + ".", direction, style="morphological")
                add("{a}, {bb} daha " + ranking + " sırada değildir.", direction, True, "negated")
    return tuple(rules)


def _parse(text: str, entity_names: tuple[str, ...], relation_id: str, *, query: bool) -> _Clause:
    _text(text)
    names = _inventory(entity_names)
    if not isinstance(relation_id, str) or relation_id not in RELATION_IDS:
        raise SemanticEquivalenceError("unsupported relation_id")
    for rule in _rules(names, relation_id, query):
        match = rule.pattern.fullmatch(text)
        if match is None:
            continue
        groups = match.groupdict()
        for key, function in (("ag", genitive), ("bg", genitive), ("ab", ablative), ("bb", ablative)):
            if groups.get(key) is None:
                continue
            name = groups[key[0]]
            try:
                expected = function(name)
            except MorphologyError as error:
                raise SemanticEquivalenceError(str(error)) from error
            if name + groups[key] != expected:
                raise SemanticEquivalenceError(f"incorrect Turkish suffix for {name!r}")
        if groups["a"] == groups["b"]:
            raise SemanticEquivalenceError("strict comparisons require distinct entities")
        return _Clause(groups["a"], groups["b"], rule.relation, rule.negated, rule.style)
    raise SemanticEquivalenceError("sentence is outside the controlled Turkish grammar")


def _edge(clause: _Clause) -> tuple[str, str]:
    forward = clause.relation == "before"
    if clause.negated:
        forward = not forward
    return ((clause.subject, clause.object) if forward
            else (clause.object, clause.subject))


def _planned_permutation(items: list, seed: int, instance_id: str, namespace: str) -> list:
    """Check the declared list-order plan without regenerating any sentence."""
    payload = [_RENDERER_VERSION, seed, instance_id, namespace]
    permutation_seed = int(hashlib.sha256(canonical_bytes(payload)).hexdigest(), 16)
    output = list(items)
    random.Random(permutation_seed).shuffle(output)
    if len(items) > 1 and output == items:
        output = output[1:] + output[:1]
    return output


def normalize_statement(text: str, entity_names: tuple[str, ...], relation_id: str) -> tuple[str, str]:
    """Map a supported statement to a positive directed edge.

    Callers must establish the strict, tie-free total-order assumptions before
    using this function for negated forms. Full presentations must instead be
    checked with ``validate_presentation``, which enforces those assumptions.
    """
    return _edge(_parse(text, entity_names, relation_id, query=False))


def normalize_query(text: str, entity_names: tuple[str, ...], relation_id: str) -> tuple[str, str, str]:
    """Read the query's original subject, object, and before/after direction."""
    clause = _parse(text, entity_names, relation_id, query=True)
    return clause.subject, clause.object, clause.relation


def validate_presentation(record: dict, presentation: dict) -> dict:
    """Reject changes to source edges, query, answer, context or prompt text.

    Two exact oracles are consulted after independent surface recognition. The
    graph oracle receives a sorted inventory and parsed edges, never latent
    positions. The returned digest covers semantic content and is invariant
    under premise/name-list permutations and the supported equivalent forms.
    """
    try:
        validate_record(record)
    except (DataValidationError, TypeError, KeyError) as error:
        raise SemanticEquivalenceError(f"invalid source record: {error}") from error
    if not isinstance(presentation, dict):
        raise SemanticEquivalenceError("presentation must be a mapping")
    try:
        names = _inventory(presentation["entity_names"])
        relation_id = presentation["relation_id"]
        variant = presentation["variant"]
        premises = presentation["premises"]
        query_text = presentation["query"]
        context = presentation["context"]
        prompt = presentation["prompt"]
    except (KeyError, TypeError) as error:
        raise SemanticEquivalenceError("presentation has missing or invalid fields") from error
    if set(names) != set(record["latent_order"]):
        raise SemanticEquivalenceError("surface entity inventory differs from source")
    if not isinstance(relation_id, str) or relation_id not in RELATION_IDS:
        raise SemanticEquivalenceError("unsupported relation_id")
    if not isinstance(variant, str) or variant not in tuple(f"P{i}" for i in range(7)):
        raise SemanticEquivalenceError("unsupported variation arm")
    if presentation.get("renderer_version") != _RENDERER_VERSION:
        raise SemanticEquivalenceError("unsupported renderer_version")
    if presentation.get("language") != "tr" or presentation.get("family") != _FAMILIES[relation_id]:
        raise SemanticEquivalenceError("language or relation family metadata disagrees")
    seed = presentation.get("seed")
    if type(seed) is not int or not 0 <= seed < 2**64:
        raise SemanticEquivalenceError("renderer seed must be an integer in [0, 2**64)")
    transforms, axes = _FACTOR_CONTRACTS[variant]
    if presentation.get("transformations") != list(transforms) or presentation.get("factor_axes") != list(axes):
        raise SemanticEquivalenceError("factor or transformation metadata disagrees with the variation arm")
    if presentation.get("is_multi_transform") is not (len(axes) > 1):
        raise SemanticEquivalenceError("multiple-transformation flag disagrees with declared axes")
    if (presentation.get("human_review_status") != "pending"
            or presentation.get("eligible_for_confirmatory_evaluation") is not False):
        raise SemanticEquivalenceError("automatic validation cannot certify human linguistic approval")
    if presentation.get("strict_total_order") is not True:
        raise SemanticEquivalenceError("a strict total order must be explicitly declared")
    for key in ("instance_id", "ordering_id", "generator_version"):
        if presentation.get(key) != record[key]:
            raise SemanticEquivalenceError(f"{key} does not identify the source record")
    expected_source_digest = hashlib.sha256(canonical_bytes(record)).hexdigest()
    if presentation.get("source_record_sha256") != expected_source_digest:
        raise SemanticEquivalenceError("source record fingerprint disagrees")
    expected_names = _planned_permutation(sorted(record["latent_order"]), seed, record["instance_id"], "names")
    if variant in ("P1", "P5"):
        expected_names = _planned_permutation(expected_names, seed, record["instance_id"], "P1_names")
    if list(presentation["entity_names"]) != expected_names:
        raise SemanticEquivalenceError("entity-list order disagrees with the declared variation and seed")

    # Context is a whitelist of declared relation semantics, not free-form text.
    # Importing this accessor does not invoke the renderer or recreate clauses.
    from renderers.turkish import expected_context

    if context != expected_context(relation_id, presentation["entity_names"]):
        raise SemanticEquivalenceError("context or strict-order assumptions have changed")
    if not isinstance(premises, (list, tuple)) or len(premises) != record["n"] - 1:
        raise SemanticEquivalenceError("surface premises must contain exactly n-1 statements")
    clauses = [_parse(text, names, relation_id, query=False) for text in premises]
    expected_style = {
        "P0": "canonical", "P1": "canonical", "P2": "canonical", "P3": "alternative",
        "P4": "morphological", "P5": "morphological", "P6": "negated",
    }[variant]
    expected_direction = "after" if variant in ("P2", "P5", "P6") else "before"
    if any(clause.style != expected_style or clause.relation != expected_direction for clause in clauses):
        raise SemanticEquivalenceError("surface grammar does not match the declared variation arm")
    if variant == "P6":
        if any(not clause.negated for clause in clauses):
            raise SemanticEquivalenceError("P6 requires controlled negation in every premise")
    elif any(clause.negated for clause in clauses):
        raise SemanticEquivalenceError("negation is permitted only in the P6 arm")
    actual_edges = sorted(_edge(clause) for clause in clauses)
    expected_edges = sorted(tuple(edge) for edge in record["premise_edges"])
    if actual_edges != expected_edges:
        raise SemanticEquivalenceError("surface premises changed the source normal form")
    expected_edge_order = [tuple(edge) for edge in record["premise_edges"]]
    if variant in ("P1", "P5"):
        indices = _planned_permutation(list(range(len(expected_edge_order))), seed,
                                       record["instance_id"], "P1_premises")
        expected_edge_order = [expected_edge_order[index] for index in indices]
    if [_edge(clause) for clause in clauses] != expected_edge_order:
        raise SemanticEquivalenceError("premise-list order disagrees with the declared variation and seed")
    actual_query = normalize_query(query_text, names, relation_id)
    expected_query = tuple(record[key] for key in ("query_subject", "query_object", "query_relation"))
    if actual_query != expected_query:
        raise SemanticEquivalenceError("surface query changed the source query")
    try:
        graph_gold = answer_from_edges(names, actual_edges, *actual_query)
        order_gold = answer_from_order(record["latent_order"], *actual_query)
    except DataValidationError as error:
        raise SemanticEquivalenceError(str(error)) from error
    if graph_gold is not record["gold_label"] or order_gold is not graph_gold:
        raise SemanticEquivalenceError("surface answer disagrees with an exact oracle")
    if "gold_label" in presentation and presentation["gold_label"] is not graph_gold:
        raise SemanticEquivalenceError("presentation gold metadata disagrees with the oracle")
    expected_prompt = (
        context + "\nVarlıklar: " + ", ".join(presentation["entity_names"])
        + ".\nÖncüller:\n" + "\n".join("- " + text for text in premises)
        + "\nSoru: " + query_text + "\nYanıtınızı yalnız YES veya NO olarak verin."
    )
    if prompt != expected_prompt:
        raise SemanticEquivalenceError("prompt contains missing, extra or changed surface text")
    if presentation.get("prompt_sha256") != hashlib.sha256(prompt.encode("utf-8")).hexdigest():
        raise SemanticEquivalenceError("prompt fingerprint disagrees with actual text")
    presentation_id = "tr-" + hashlib.sha256(canonical_bytes([
        _RENDERER_VERSION, record["instance_id"], relation_id, variant, seed,
        presentation["prompt_sha256"],
    ])).hexdigest()
    if presentation.get("presentation_id") != presentation_id:
        raise SemanticEquivalenceError("presentation identifier does not bind the declared configuration and prompt")
    normal_form = {"entities": list(names), "premise_edges": actual_edges, "query": actual_query}
    return {
        "equivalent": True, "gold_unchanged": True,
        "normal_form_sha256": hashlib.sha256(canonical_bytes(normal_form)).hexdigest(),
        "oracle_gold": graph_gold,
    }

"""Reuse the independent Turkish grammar for an entire prompt inventory.

The stage-three public parser validates and sorts its inventory on every call.
This adapter performs that work once per reader, then applies the same compiled
recognition grammar to the actual statement/query strings. It does not infer
meaning from renderer metadata and does not enable controlled negation.
"""

from __future__ import annotations

from renderers.morphology import MorphologyError, ablative, genitive
from validation import semantic_equivalence as grammar


class TurkishReader:
    """Read non-negated Turkish templates using one validated entity inventory."""

    def __init__(self, names: tuple[str, ...], relation_id: str):
        self._names = grammar._inventory(names)
        if not isinstance(relation_id, str) or relation_id not in grammar.RELATION_IDS:
            raise grammar.SemanticEquivalenceError("unsupported relation_id")
        self._statement_rules = tuple(rule for rule in grammar._rules(
            self._names, relation_id, False) if not rule.negated)
        self._query_rules = tuple(rule for rule in grammar._rules(
            self._names, relation_id, True) if not rule.negated)

    def _parse(self, text: str, *, query: bool):
        grammar._text(text)
        rules = self._query_rules if query else self._statement_rules
        for rule in rules:
            match = rule.pattern.fullmatch(text)
            if match is None:
                continue
            groups = match.groupdict()
            for key, function in (("ag", genitive), ("bg", genitive),
                                  ("ab", ablative), ("bb", ablative)):
                if groups.get(key) is None:
                    continue
                name = groups[key[0]]
                try:
                    expected = function(name)
                except MorphologyError as error:
                    raise grammar.SemanticEquivalenceError(str(error)) from error
                if name + groups[key] != expected:
                    raise grammar.SemanticEquivalenceError(f"incorrect Turkish suffix for {name!r}")
            if groups["a"] == groups["b"]:
                raise grammar.SemanticEquivalenceError("strict comparisons require distinct entities")
            return groups["a"], groups["b"], rule.relation
        raise grammar.SemanticEquivalenceError("sentence is outside the non-negated controlled Turkish grammar")

    def statement(self, text: str) -> tuple[str, str]:
        """Return the actual clause's positive directed edge."""
        subject, object_, relation = self._parse(text, query=False)
        return ((subject, object_) if relation == "before" else (object_, subject))

    def query(self, text: str) -> tuple[str, str, str]:
        """Return the actual question's subject, object and before/after relation."""
        return self._parse(text, query=True)

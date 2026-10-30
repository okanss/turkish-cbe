"""Inventory-independent recognition of the existing controlled grammars.

The grammar stays independent of the sentence renderers. Its compiled rules
capture a restricted proper-name alphabet, then check names against a set.
Consequently each clause lookup does not scan a chain-length name alternation.
This adapter accepts only positive, non-negated templates; P6 remains excluded.
"""

from __future__ import annotations

from functools import lru_cache
import re
import unicodedata

from renderers import english
from renderers.morphology import MorphologyError, ablative, genitive
from validation import semantic_equivalence as turkish_grammar


class SurfaceReadError(ValueError):
    """A clause is outside the declared grammar or entity inventory."""


_PLACEHOLDERS = ("AlphaPlaceholder", "BetaPlaceholder")
_ALTERNATIVES = "(?:" + "|".join(re.escape(name) for name in sorted(
    _PLACEHOLDERS, key=lambda name: (-len(name), name))) + ")"
_NAME_EXPRESSION = r"(?:[A-Za-zÇĞİÖŞÜçğıöşü0-9]+(?: [A-Za-zÇĞİÖŞÜçğıöşü0-9]+)*)"
_NAME = re.compile(_NAME_EXPRESSION)


def _generic(pattern: re.Pattern[str]) -> re.Pattern[str]:
    expression = pattern.pattern
    if expression.count(_ALTERNATIVES) != 2:
        raise SurfaceReadError("controlled grammar name captures changed")
    return re.compile(expression.replace(_ALTERNATIVES, _NAME_EXPRESSION), pattern.flags)


@lru_cache(maxsize=36)
def _rules(language: str, relation_id: str, query: bool):
    """Cache 2 languages x 9 relations x 2 clause kinds, not name inventories."""
    if language == "en":
        return tuple((_generic(pattern), relation) for relation, pattern in
                     english._patterns(_PLACEHOLDERS, relation_id, query))
    return tuple((_generic(rule.pattern), rule.relation) for rule in
                 turkish_grammar._rules(_PLACEHOLDERS, relation_id, query)
                 if not rule.negated)


@lru_cache(maxsize=16)
def _inflections(names: tuple[str, ...]):
    """Reuse exact suffix expectations across relations and presentation styles."""
    try:
        return ({name: genitive(name) for name in names},
                {name: ablative(name) for name in names})
    except MorphologyError as error:
        raise SurfaceReadError(str(error)) from error


class SurfaceReader:
    """Parse actual EN/TR clauses with O(1) inventory membership checks.

    Construction checks the complete inventory once. The allowed alphabet is
    intentionally limited to the generated native/ASCII/symbolic identifiers.
    This is a finite controlled grammar, not a general language parser.
    """

    def __init__(self, names, relation_id: str, language: str):
        if type(language) is not str or language not in {"en", "tr"}:
            raise SurfaceReadError("unsupported language")
        if type(relation_id) is not str or relation_id not in turkish_grammar.RELATION_IDS:
            raise SurfaceReadError("unsupported relation_id")
        if isinstance(names, (str, bytes)):
            raise SurfaceReadError("names must be an entity inventory")
        try:
            values = tuple(names)
        except TypeError as error:
            raise SurfaceReadError("names must be an entity inventory") from error
        if len(values) < 2:
            raise SurfaceReadError("at least two unique names are required")
        for name in values:
            if (not isinstance(name, str) or _NAME.fullmatch(name) is None
                    or unicodedata.normalize("NFC", name) != name):
                raise SurfaceReadError("entity names require the controlled NFC proper-name alphabet")
        self._names = frozenset(values)
        if len(self._names) != len(values):
            raise SurfaceReadError("entity names must be unique")
        self._statement_rules = _rules(language, relation_id, False)
        self._query_rules = _rules(language, relation_id, True)
        self._inflected = _inflections(tuple(sorted(values))) if language == "tr" else None

    def _parse(self, text: str, query: bool):
        if (not isinstance(text, str) or not text
                or unicodedata.normalize("NFC", text) != text):
            raise SurfaceReadError("surface text must be a nonempty NFC string")
        for pattern, relation in (self._query_rules if query else self._statement_rules):
            match = pattern.fullmatch(text)
            if match is None:
                continue
            groups = match.groupdict()
            a, b = groups["a"], groups["b"]
            if a not in self._names or b not in self._names:
                raise SurfaceReadError("sentence refers to an entity outside the inventory")
            if a == b:
                raise SurfaceReadError("strict comparisons require distinct entities")
            if self._inflected is not None:
                genitives, ablatives = self._inflected
                for key, expected in (("ag", genitives), ("bg", genitives),
                                      ("ab", ablatives), ("bb", ablatives)):
                    suffix = groups.get(key)
                    if suffix is not None:
                        name = groups[key[0]]
                        if name + suffix != expected[name]:
                            raise SurfaceReadError("incorrect Turkish suffix for " + repr(name))
            return a, b, relation
        raise SurfaceReadError("sentence is outside the non-negated controlled grammar")

    def statement(self, text: str) -> tuple[str, str]:
        """Return the canonical positive edge actually stated by the clause."""
        a, b, relation = self._parse(text, False)
        return (a, b) if relation == "before" else (b, a)

    def query(self, text: str) -> tuple[str, str, str]:
        """Return actual displayed arguments and before/after query relation."""
        return self._parse(text, True)

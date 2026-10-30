"""Small, deterministic Turkish morphology for the renderer's entity pools.

This module is deliberately not a general Turkish morphological analyser. It
implements proper-name genitive/ablative suffixes and six declared positional
stems. Names retain their spelling (including final consonants); suffixes use
an ASCII apostrophe. Inputs are normalised to NFC without changing their case.

``Edddd`` identifiers are read as the letter E followed by four separate
Turkish digit names. Thus E0004 ends in *dört*, not in *bin dört*. This is an
experimental naming convention that renderers must disclose to readers.

The fixed generator pools use regular Turkish spellings. An explicit phonetic
hint is required for spellings with circumflex vowels. Other exceptional
pronunciations need a caller-supplied hint, except the tiny declared lexicon
below. Automatic checks do not establish that a rendered sentence is natural.
"""

from __future__ import annotations

import re
import unicodedata


class MorphologyError(ValueError):
    """A name, pronunciation or positional stem is outside the supported scope."""


_VOWELS = frozenset("aeıioöuü")
_VOICELESS = frozenset("fstkçşhp")
_PLAIN_LETTERS = frozenset("abcçdefgğhıijklmnoöprsştuüvyz")
_CIRCUMFLEX = frozenset("âîû")
_SYMBOLIC_ID = re.compile(r"E[0-9]{4}\Z", re.ASCII)
_HIGH_VOWEL = {
    "a": "ı", "ı": "ı", "e": "i", "i": "i",
    "o": "u", "u": "u", "ö": "ü", "ü": "ü",
}
_DIGIT_NAMES = (
    "sıfır", "bir", "iki", "üç", "dört", "beş", "altı", "yedi", "sekiz", "dokuz",
)
# Final /l/ in this declared name conditions front suffix vowels. This is a
# local exception, not a comprehensive pronunciation dictionary of names.
_FRONTNESS_EXCEPTIONS = {"alp": "e"}
_POSITIONAL_FORMS = {
    "ön": ("önünde", "önündedir"),
    "arka": ("arkasında", "arkasındadır"),
    "sağ": ("sağında", "sağındadır"),
    "sol": ("solunda", "solundadır"),
    "üst": ("üstünde", "üstündedir"),
    "alt": ("altında", "altındadır"),
}


def turkish_lower(value: str) -> str:
    """Lowercase Turkish dotted/dotless I and normalise Unicode to NFC."""
    if not isinstance(value, str):
        raise MorphologyError("text must be a string")
    normalised = unicodedata.normalize("NFC", value)
    return normalised.translate(str.maketrans({"I": "ı", "İ": "i"})).lower()


def _letters_and_spaces(value: str, *, role: str, circumflex: bool) -> str:
    if not isinstance(value, str) or not value:
        raise MorphologyError(f"{role} must be a nonempty string")
    normalised = unicodedata.normalize("NFC", value)
    lowered = turkish_lower(normalised)
    allowed = _PLAIN_LETTERS | (_CIRCUMFLEX if circumflex else frozenset())
    if (
        normalised != normalised.strip()
        or "  " in normalised
        or any(character != " " and character not in allowed for character in lowered)
    ):
        raise MorphologyError(f"{role} must contain Turkish letters separated by single spaces")
    return normalised


def _name_and_pronunciation(name: str, pronunciation: str | None) -> tuple[str, str]:
    if not isinstance(name, str) or not name:
        raise MorphologyError("name must be a nonempty string")
    name = unicodedata.normalize("NFC", name)
    if _SYMBOLIC_ID.fullmatch(name):
        if pronunciation is not None:
            raise MorphologyError("symbolic identifiers use the fixed digit-by-digit pronunciation")
        return name, _DIGIT_NAMES[int(name[-1])]
    name = _letters_and_spaces(name, role="name", circumflex=True)
    if pronunciation is None:
        if any(character in _CIRCUMFLEX for character in turkish_lower(name)):
            raise MorphologyError("names with circumflex vowels require an explicit pronunciation")
        pronunciation = name
    else:
        pronunciation = _letters_and_spaces(
            pronunciation, role="pronunciation", circumflex=False,
        )
    final_word = turkish_lower(pronunciation).rsplit(" ", 1)[-1]
    if not any(character in _VOWELS for character in final_word):
        raise MorphologyError("the final pronunciation word must contain a Turkish vowel")
    return name, final_word


def _suffix_vowel(pronunciation: str, *, high: bool) -> str:
    last_vowel = next(character for character in reversed(pronunciation) if character in _VOWELS)
    harmony_vowel = _FRONTNESS_EXCEPTIONS.get(pronunciation, last_vowel)
    if high:
        return _HIGH_VOWEL[harmony_vowel]
    return "e" if harmony_vowel in "eiöü" else "a"


def genitive(name: str, pronunciation: str | None = None) -> str:
    """Inflect a complete proper name: ``Ayşe'nin``, ``Ahmet'in``, ``E0004'ün``.

    The optional pronunciation is plain Turkish orthography used only for
    suffix selection; it never replaces or softens the displayed name.
    """
    name, spoken = _name_and_pronunciation(name, pronunciation)
    buffer = "n" if spoken[-1] in _VOWELS else ""
    return f"{name}'{buffer}{_suffix_vowel(spoken, high=True)}n"


def ablative(name: str, pronunciation: str | None = None) -> str:
    """Inflect a complete proper name: ``Ayşe'den``, ``Ahmet'ten``, ``E0004'ten``."""
    name, spoken = _name_and_pronunciation(name, pronunciation)
    consonant = "t" if spoken[-1] in _VOICELESS else "d"
    return f"{name}'{consonant}{_suffix_vowel(spoken, high=False)}n"


def possessive_locative(stem: str, copula: bool = True) -> str:
    """Return a declared third-person possessive + locative positional noun.

    Only ``ön``, ``arka``, ``sağ``, ``sol``, ``üst`` and ``alt`` are supported.
    ``copula=False`` gives forms such as ``önünde``; the default gives
    ``önündedir``. Common-noun suffixes use no apostrophe.
    """
    if not isinstance(stem, str) or stem not in _POSITIONAL_FORMS:
        raise MorphologyError("unsupported positional stem")
    if not isinstance(copula, bool):
        raise MorphologyError("copula must be a bool")
    return _POSITIONAL_FORMS[stem][int(copula)]

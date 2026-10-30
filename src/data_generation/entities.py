"""Versioned deterministic entity pools for language-independent latent chains.

The hand-curated person-name lists are design materials, not corpus samples.
Full names are artificial combinations and do not refer to particular people.
Synthetic strings follow CV.CV vowel harmony; they are not guaranteed to be
unseen names or words. Pools do not depend on the requested chain size.
"""

from functools import lru_cache
import random
import unicodedata

ENTITY_TYPES = ("turkish_names", "synthetic_names", "symbolic_ids")
NAME_RULES_VERSION = "1"

# 48 familiar given names and 32 surnames give 1,536 distinct full names.
# Lists are fixed in this order so their sampling frame is reproducible.
_FIRST_NAMES = (
    "Ahmet", "Mehmet", "Mustafa", "Ali", "Hüseyin", "Hasan", "İbrahim", "İsmail",
    "Osman", "Yusuf", "Murat", "Ömer", "Emre", "Burak", "Can", "Cem", "Deniz",
    "Eren", "Kerem", "Onur", "Selim", "Serkan", "Uğur", "Volkan",
    "Ayşe", "Fatma", "Emine", "Hatice", "Zeynep", "Elif", "Meryem", "Özlem",
    "Esra", "Seda", "Selin", "Ece", "Derya", "Gül", "Gülay", "Gamze", "Pınar",
    "İrem", "Büşra", "Ceren", "Aylin", "Aslı", "Gizem", "Yasemin",
)
_SURNAMES = (
    "Yılmaz", "Kaya", "Demir", "Çelik", "Şahin", "Yıldız", "Yıldırım", "Öztürk",
    "Aydın", "Özdemir", "Arslan", "Doğan", "Kılıç", "Aslan", "Çetin", "Kara",
    "Koç", "Kurt", "Özkan", "Şimşek", "Polat", "Özcan", "Korkmaz", "Çakır",
    "Erdoğan", "Yavuz", "Can", "Acar", "Taş", "Güneş", "Aksoy", "Bozkurt",
)
_CONSONANTS = "bcçdghklmnprsştvyz"
# The second syllable is open, so final vowels obey the stated harmony rule.
_HARMONIC_VOWELS = (
    ("a", "aı"), ("ı", "aı"), ("o", "au"), ("u", "au"),
    ("e", "ei"), ("i", "ei"), ("ö", "eü"), ("ü", "eü"),
)


def _turkish_lower(value: str) -> str:
    return value.translate(str.maketrans({"I": "ı", "İ": "i"})).lower()


def _validate_request(entity_type: str, n: int) -> None:
    if entity_type not in ENTITY_TYPES:
        raise ValueError(f"Unknown entity_type: {entity_type!r}")
    if isinstance(n, bool) or not isinstance(n, int) or not 4 <= n <= 1024:
        raise ValueError("n must be an integer between 4 and 1024")


@lru_cache(maxsize=len(ENTITY_TYPES))
def _fixed_pool(entity_type: str) -> tuple[str, ...]:
    if entity_type == "turkish_names":
        values = tuple(f"{first} {last}" for first in _FIRST_NAMES for last in _SURNAMES)
    elif entity_type == "synthetic_names":
        excluded = {_turkish_lower(name) for name in _FIRST_NAMES}
        values = tuple(
            f"{c1.upper()}{v1}{c2}{v2}"
            for c1 in _CONSONANTS
            for v1, allowed in _HARMONIC_VOWELS
            for c2 in _CONSONANTS
            for v2 in allowed
            if f"{c1}{v1}{c2}{v2}" not in excluded
        )
    else:
        values = tuple(f"E{index:04d}" for index in range(2048))
    return tuple(unicodedata.normalize("NFC", value) for value in values)


def entity_pool(entity_type: str, n: int) -> tuple[str, ...]:
    """Return a fixed pool after validating chain size and available capacity.

    Repeated given names are allowed in ``turkish_names``; the complete
    given-name/surname combination is the entity identifier and is unique.
    """
    _validate_request(entity_type, n)
    pool = _fixed_pool(entity_type)
    if n > len(pool):
        raise ValueError(f"Pool for {entity_type!r} contains fewer than {n} names")
    return pool


def sample_entities(entity_type: str, n: int, rng: random.Random) -> list[str]:
    """Sample ``n`` distinct identifiers using only the supplied RNG state."""
    return rng.sample(entity_pool(entity_type, n), n)

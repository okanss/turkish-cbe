"""Bilingual entity aliases grouped by the original latent ordering identifier.

Transliteration changes only English display labels. Turkish retains the native
spelling and morphology. Shared symbolic labels are sampled from the fixed
generator pool, then assigned to sorted names independently of latent rank.
"""

import hashlib
import random
import re
import unicodedata

from data_generation.entities import entity_pool
from data_generation.generator import canonical_bytes


NAME_CONTROL_VERSION = "0.4.0"
NAME_CONTROLS = ("native_shared", "en_transliterated", "symbolic_shared")
_TRANSLITERATION = str.maketrans("çğıöşüİÇĞÖŞÜ", "cgiosuICGOSU")


def _source_names(record: dict) -> tuple[str, ...]:
    if not isinstance(record, dict):
        raise ValueError("record must be a dictionary")
    n = record.get("n")
    if type(n) is not int or not 4 <= n <= 1024:
        raise ValueError("record n must be an integer in [4, 1024]")
    values = record.get("latent_order")
    if not isinstance(values, list) or len(values) != n:
        raise ValueError("latent_order must contain exactly n source names")
    for name in values:
        if (not isinstance(name, str) or not name or name != name.strip()
                or "  " in name or unicodedata.normalize("NFC", name) != name
                or not all(character.isalpha() or character.isdecimal()
                           or character == " " for character in name)):
            raise ValueError("source names must be nonempty NFC letters, digits and single spaces")
    if len(set(values)) != n:
        raise ValueError("source names must be unique")
    ordering_id = record.get("ordering_id")
    if not isinstance(ordering_id, str) or re.fullmatch(r"ord-[0-9a-f]{64}", ordering_id) is None:
        raise ValueError("record must contain a valid ordering_id")
    return tuple(sorted(values))


def _expected_maps(record: dict, names: tuple[str, ...]) -> dict:
    identity = {name: name for name in names}
    transliterated = {name: name.translate(_TRANSLITERATION) for name in names}
    if any(not name.isascii() for name in transliterated.values()):
        raise ValueError("source name contains unsupported non-ASCII transliteration characters")
    if len(set(transliterated.values())) != len(names):
        raise ValueError("English transliteration is not bijective for this entity inventory")
    seed = int(hashlib.sha256(canonical_bytes([
        NAME_CONTROL_VERSION, record["ordering_id"], "shared-symbolic-aliases"
    ])).hexdigest(), 16)
    labels = random.Random(seed).sample(entity_pool("symbolic_ids", record["n"]), len(names))
    symbolic = dict(zip(names, labels))
    return {
        "native_shared": {"en": dict(identity), "tr": dict(identity)},
        "en_transliterated": {"en": transliterated, "tr": dict(identity)},
        "symbolic_shared": {"en": dict(symbolic), "tr": dict(symbolic)},
    }


def name_maps(record: dict) -> dict:
    """Return all three paired, bijective source-name-to-display-name maps.

    A colliding transliteration is rejected rather than silently disambiguated.
    The mapping namespace deliberately excludes the query, language, relation,
    presentation condition and ordering of the source inventory.
    """
    names = _source_names(record)
    return _expected_maps(record, names)


def validate_name_maps(record: dict, maps: dict) -> None:
    """Reject incomplete, non-bijective, or rule-inconsistent bilingual aliases."""
    names = _source_names(record)
    inventory = set(names)
    if not isinstance(maps, dict) or set(maps) != set(NAME_CONTROLS):
        raise ValueError("name maps must contain exactly the three name controls")
    for control in NAME_CONTROLS:
        languages = maps[control]
        if not isinstance(languages, dict) or set(languages) != {"en", "tr"}:
            raise ValueError("each name control must contain exactly en and tr")
        for language in ("en", "tr"):
            mapping = languages[language]
            if not isinstance(mapping, dict) or set(mapping) != inventory:
                raise ValueError("each alias map must cover exactly the source inventory")
            values = tuple(mapping.values())
            if any(not isinstance(value, str) or not value for value in values):
                raise ValueError("display names must be nonempty strings")
            if len(set(values)) != len(names):
                raise ValueError("each alias map must be bijective")
    if maps != _expected_maps(record, names):
        raise ValueError("name maps do not satisfy the versioned name-control rules")

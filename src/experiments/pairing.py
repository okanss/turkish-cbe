"""Validate the contrasts defined by the bilingual experiment matrix.

This checks pairing metadata, not the linguistic meaning of prompt text. The
rendering and semantic validation layers must establish that meaning first.
Query subjects and objects are canonical entity identifiers, not display names.
"""

from __future__ import annotations


_TEXT_FIELDS = (
    "latent_instance_id", "query_instance_id", "split", "query_type",
    "query_subject", "query_object", "query_relation", "query_direction",
    "language", "condition", "name_control", "relation_id",
    "presentation_plan_sha256",
)

_SAME_QUERY_FIELDS = (
    "latent_instance_id", "query_instance_id", "split", "n", "query_type",
    "query_distance", "query_subject", "query_object", "query_relation",
    "query_direction", "gold_label", "name_control", "relation_id",
)

_QUERY_CONTRAST_FIELDS = (
    "latent_instance_id", "split", "n", "language", "condition",
    "name_control", "relation_id", "presentation_plan_sha256",
)


def _validate_metadata(item: dict, side: str) -> None:
    if not isinstance(item, dict):
        raise ValueError(f"{side} must be a dictionary")
    required = set(_TEXT_FIELDS) | {"n", "query_distance", "gold_label"}
    missing = sorted(required - item.keys())
    if missing:
        raise ValueError(f"{side} is missing required metadata: {', '.join(missing)}")
    for field in _TEXT_FIELDS:
        value = item[field]
        if not isinstance(value, str) or not value.strip():
            raise ValueError(f"{side}.{field} must be a nonempty string")
    if type(item["n"]) is not int or not 4 <= item["n"] <= 1024:
        raise ValueError(f"{side}.n must be an integer between 4 and 1024")
    distance = item["query_distance"]
    if type(distance) is not int or not 1 <= distance < item["n"]:
        raise ValueError(f"{side}.query_distance must be an integer in [1, n-1]")
    if type(item["gold_label"]) is not bool:
        raise ValueError(f"{side}.gold_label must be a boolean")
    if item["language"] not in {"en", "tr"}:
        raise ValueError(f"{side}.language must be en or tr")
    if item["query_type"] not in {"endpoint", "internal", "distance_controlled"}:
        raise ValueError(f"{side}.query_type is not recognized")
    if item["query_relation"] not in {"before", "after"}:
        raise ValueError(f"{side}.query_relation must be before or after")
    if item["query_direction"] not in {"forward", "reverse"}:
        raise ValueError(f"{side}.query_direction must be forward or reverse")
    if item["query_subject"] == item["query_object"]:
        raise ValueError(f"{side} must compare two distinct entities")


def _require_same(left: dict, right: dict, fields: tuple[str, ...]) -> None:
    for field in fields:
        if left[field] != right[field]:
            raise ValueError(f"paired metadata differs: {field}")


def validate_pair(left: dict, right: dict, arm: str) -> None:
    """Reject pairs that do not implement contrast A, B, C, D, E or F.

    A/B compare canonical and equivalent conditions within one language. C/D
    compare languages under the same presentation plan. E compares the native
    Turkish canonical endpoint prompt with a P1--P5 ablation. F compares endpoint
    and internal/distance queries while preserving their common presentation;
    its targets, query identifiers, directions, distances and gold may differ.

    Extra fields are allowed because complete prompt records contain text,
    hashes and provenance in addition to this required metadata.
    """
    if not isinstance(arm, str) or arm not in {"A", "B", "C", "D", "E", "F"}:
        raise ValueError("arm must be one of A, B, C, D, E or F")
    _validate_metadata(left, "left")
    _validate_metadata(right, "right")

    if arm == "F":
        _require_same(left, right, _QUERY_CONTRAST_FIELDS)
        if left["condition"] not in {"canonical", "equivalent"}:
            raise ValueError("arm F requires canonical or equivalent conditions")
        if left["query_type"] != "endpoint" or right["query_type"] not in {
            "internal", "distance_controlled",
        }:
            raise ValueError("arm F requires endpoint versus internal/distance_controlled")
        if left["query_instance_id"] == right["query_instance_id"]:
            raise ValueError("arm F requires distinct query_instance_id values")
        return

    _require_same(left, right, _SAME_QUERY_FIELDS)
    if arm == "A":
        expected = ("en", "canonical", "en", "equivalent")
    elif arm == "B":
        expected = ("tr", "canonical", "tr", "equivalent")
    elif arm == "C":
        expected = ("en", "canonical", "tr", "canonical")
    elif arm == "D":
        expected = ("en", "equivalent", "tr", "equivalent")
    else:
        if left["language"] != "tr" or right["language"] != "tr":
            raise ValueError("arm E requires Turkish prompts")
        if left["condition"] != "canonical" or right["condition"] not in {
            "P1", "P2", "P3", "P4", "P5",
        }:
            raise ValueError("arm E requires canonical versus P1--P5")
        if left["name_control"] != "native_shared" or left["query_type"] != "endpoint":
            raise ValueError("arm E requires native_shared endpoint prompts")
        return
    actual = (left["language"], left["condition"], right["language"], right["condition"])
    if actual != expected:
        raise ValueError(f"arm {arm} has incorrect languages or conditions")
    if arm in {"C", "D"}:
        _require_same(left, right, ("presentation_plan_sha256",))

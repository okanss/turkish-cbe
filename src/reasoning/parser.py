"""Strategy-specific scoring without extracting labels from explanations."""

from __future__ import annotations

import re

from inference.parser import PARSER_VERSION as DIRECT_PARSER_VERSION
from inference.parser import parse_answer as parse_direct_answer


STRATEGIES = ("direct", "zero_shot_reasoning", "structured_reasoning", "provider_effort")
FINAL_PARSER_VERSION = "reasoning-final-answer-v1"
PARSER_VERSIONS = {strategy: (DIRECT_PARSER_VERSION if strategy in {"direct", "provider_effort"}
                             else FINAL_PARSER_VERSION) for strategy in STRATEGIES}
# FINAL: is a reserved marker. Case/spacing variants count as marker occurrences
# for rejecting repetitions, but only the exact uppercase frame can be valid.
_MARKER = re.compile(r"(?<!\w)[Ff][Ii][Nn][Aa][Ll][ \t]*:")


def parse_answer(text, language: str, strategy: str, *, truncated: bool = False,
                 finish_reason: str | None = None, refusal: bool = False) -> dict:
    """Score A/D exactly as before; B/C require one final marker at the end.

    Only the payload of the single last-line ``FINAL: `` marker is passed to
    the existing strict localized-token parser. Explanation words, including
    YES/NO or EVET/HAYIR, are not answer candidates or contradictions. Duplicate
    markers are invalid even if their payloads agree. Outer whitespace is
    allowed; marker capitalization and the single space after its colon are
    exact. Provider refusal/truncation always prevents a valid prediction.
    """
    if type(strategy) is not str or strategy not in STRATEGIES:
        raise ValueError("unknown reasoning strategy")
    flags = {"truncated": truncated, "finish_reason": finish_reason, "refusal": refusal}
    if strategy in {"direct", "provider_effort"}:
        return parse_direct_answer(text, language, **flags)

    # Delegate argument validation and terminal provider conditions. Supplying
    # an empty string avoids scanning explanatory prose for answer vocabulary.
    result = parse_direct_answer("" if isinstance(text, str) else text, language, **flags)
    result["parser_version"] = FINAL_PARSER_VERSION
    marker = {"marker": "FINAL:", "occurrences": 0, "is_final_line": False, "answer_token": None}
    result["final_answer_marker"] = marker
    if result["invalid_reason"] in {"non_text", "refusal", "truncated"}:
        return result
    body = text.strip()
    if not body:
        return result
    occurrences = list(_MARKER.finditer(body))
    marker["occurrences"] = len(occurrences)
    if not occurrences:
        result["invalid_reason"] = "missing_final_marker"
        return result
    if len(occurrences) != 1:
        result["invalid_reason"] = "duplicate_final_marker"
        return result
    last_line = body.splitlines()[-1]
    if not _MARKER.search(last_line):
        result["invalid_reason"] = "final_marker_not_last_line"
        return result
    marker["is_final_line"] = True
    if not last_line.startswith("FINAL: "):
        result["invalid_reason"] = "invalid_final_marker"
        return result
    payload = last_line[len("FINAL: "):]
    marker["answer_token"] = payload
    parsed = parse_direct_answer(payload, language, **flags)
    parsed.update(parser_version=FINAL_PARSER_VERSION, final_answer_marker=marker)
    if payload != payload.strip():
        parsed.update(valid=False, prediction=None, invalid_reason="invalid_final_marker")
    return parsed

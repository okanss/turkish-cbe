"""Strict, versioned parsing of the single requested answer token.

Whitespace surrounding an answer is allowed. Case folding, punctuation,
explanations, Markdown and JSON are never accepted as valid answers. Keyword
inspection is diagnostic only: it cannot turn an invalid response into a label.
"""

from __future__ import annotations

import re


PARSER_VERSION = "strict-answer-v1"
_LABELS = {"tr": {"EVET": True, "HAYIR": False},
           "en": {"YES": True, "NO": False}}
_WORDS = re.compile(r"(?<!\w)(EVET|HAYIR|YES|NO)(?!\w)", re.IGNORECASE)
_TOKEN_LANGUAGE = {token: language for language, tokens in _LABELS.items()
                   for token in tokens}
_TOKEN_VALUE = {token: value for tokens in _LABELS.values()
                for token, value in tokens.items()}
_TRUNCATED_REASONS = frozenset({"length", "max_tokens", "max_output_tokens", "incomplete"})
_REFUSAL_REASONS = frozenset({"content_filter", "refusal"})


def parse_answer(text, language: str, *, truncated: bool = False,
                 finish_reason: str | None = None, refusal: bool = False) -> dict:
    """Return a label only for an exact localized answer, with diagnostics.

``observed_language`` describes recognized answer words, not the language of
arbitrary explanatory prose. ``wrong_language`` is true when at least one
recognized answer word belongs to the other requested answer vocabulary.
Contradiction detection uses case-insensitive Unicode whole-word matches; its
result is diagnostic and never licenses extracting a label from longer text.
Provider refusal or truncation takes precedence over answer-format diagnostics.
The caller retains the original raw response unchanged.
"""
    if type(language) is not str or language not in _LABELS:
        raise ValueError("language must be 'tr' or 'en'")
    if type(truncated) is not bool or type(refusal) is not bool:
        raise ValueError("truncated and refusal must be Boolean")
    if finish_reason is not None and not isinstance(finish_reason, str):
        raise ValueError("finish_reason must be a string or None")

    # Python's Unicode IGNORECASE also equates I and dotted/dotless Turkish I.
    # Canonicalize those diagnostic matches; valid answers still require exact
    # uppercase tokens below and do not receive this normalization.
    words = ([match.group().upper().replace("İ", "I") for match in _WORDS.finditer(text)]
             if isinstance(text, str) else [])
    languages = {_TOKEN_LANGUAGE[word] for word in words}
    observed = (next(iter(languages)) if len(languages) == 1 else
                "mixed" if languages else "unknown")
    result = {"parser_version": PARSER_VERSION, "valid": False,
              "prediction": None, "observed_language": observed,
              "wrong_language": bool(languages - {language}),
              "invalid_reason": None}

    if not isinstance(text, str):
        reason = "non_text"
    elif refusal or finish_reason in _REFUSAL_REASONS:
        reason = "refusal"
    elif truncated or finish_reason in _TRUNCATED_REASONS:
        reason = "truncated"
    elif not text.strip():
        reason = "empty"
    elif len({_TOKEN_VALUE[word] for word in words}) > 1:
        reason = "contradictory_answers"
    elif result["wrong_language"]:
        reason = "wrong_language"
    elif text.strip() not in _LABELS[language]:
        reason = "invalid_format"
    else:
        result.update(valid=True, prediction=_LABELS[language][text.strip()])
        return result
    result["invalid_reason"] = reason
    return result

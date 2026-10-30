"""Portable prompt lengths without inventing model-specific token counts.

The reference tokenizer counts contiguous Unicode letters, combining marks and
numbers as one token. Each remaining non-whitespace code point is one token.
It is deliberately a lexical reference measure, not BPE or an LLM tokenizer.
The Unicode database version is part of its identity. Text is never normalized
or otherwise changed; NFC status is only recorded for inspection.
"""

from __future__ import annotations

import unicodedata


TOKENIZATION_ID = f"unicode-lexical-v1-ucd-{unicodedata.unidata_version}"

TOKENIZER_SPEC = {
    "tokenization_id": TOKENIZATION_ID,
    "algorithm_version": "1",
    "unicode_database_version": unicodedata.unidata_version,
    "normalization": "none; input text is preserved exactly",
    "word_rule": "maximal contiguous span of Unicode categories L, M or N",
    "other_rule": "each other non-whitespace code point is a separate token",
    "whitespace_rule": "Python str.isspace; no token is emitted",
    "is_model_tokenizer": False,
    "model_tokenizer_status": "not_configured",
}


def _reference_count(text: str) -> int:
    count = 0
    in_word = False
    for character in text:
        if character.isspace():
            in_word = False
        elif unicodedata.category(character)[0] in "LMN":
            if not in_word:
                count += 1
            in_word = True
        else:
            count += 1
            in_word = False
    return count


def measure_text(text: str) -> dict:
    """Measure the supplied prompt, including all spaces and line breaks.

    ``reference_token_count`` and ``whitespace_token_count`` are descriptive
    reference measures. ``model_token_count`` remains ``None`` until a specific
    frozen model tokenizer is configured in a later stage. These counts exclude
    any provider chat template, system message or response tokens.

    Unpaired Unicode surrogates are rejected because JSONL artifacts must carry
    lossless UTF-8 text. Empty strings are permitted and have zero length.
    """
    if not isinstance(text, str):
        raise TypeError("text must be a Unicode string")
    try:
        utf8 = text.encode("utf-8", errors="strict")
    except UnicodeEncodeError as error:
        raise ValueError("text must be losslessly encodable as UTF-8") from error
    return {
        "utf8_bytes": len(utf8),
        "unicode_codepoints": len(text),
        "whitespace_token_count": len(text.split()),
        "reference_token_count": _reference_count(text),
        "tokenization_id": TOKENIZATION_ID,
        "unicode_database_version": unicodedata.unidata_version,
        "input_is_nfc": unicodedata.is_normalized("NFC", text),
        "model_token_count": None,
        "model_tokenizer_status": "not_configured",
    }

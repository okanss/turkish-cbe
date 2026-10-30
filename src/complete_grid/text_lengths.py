"""Fast exact reference lengths for the controlled prompt alphabet.

The versioned original counter remains the definition. ASCII plus Turkish
letters can use an equivalent C regex; all other Unicode takes that original
counter, preserving its Unicode database and whitespace semantics exactly.
"""

from __future__ import annotations

import re
import unicodedata

from experiments.token_lengths import TOKENIZATION_ID, measure_text as _original


_UNSUPPORTED = re.compile(r"[^\x00-\x7FÇĞİÖŞÜçğıöşü]")
_TOKENS = re.compile(r"[A-Za-zÇĞİÖŞÜçğıöşü0-9]+|[^\sA-Za-zÇĞİÖŞÜçğıöşü0-9]")


def fastmeasure_text(text: str) -> dict:
    """Return exactly the original measurement dictionary for every input.

    This optimization changes neither tokenization identity nor measured text.
    Surrogates, invalid types and uncommon scripts retain original behavior.
    """
    if not isinstance(text, str) or _UNSUPPORTED.search(text) is not None:
        return _original(text)
    return {
        "utf8_bytes": len(text.encode("utf-8", errors="strict")),
        "unicode_codepoints": len(text),
        "whitespace_token_count": len(text.split()),
        "reference_token_count": len(_TOKENS.findall(text)),
        "tokenization_id": TOKENIZATION_ID,
        "unicode_database_version": unicodedata.unidata_version,
        "input_is_nfc": unicodedata.is_normalized("NFC", text),
        "model_token_count": None,
        "model_tokenizer_status": "not_configured",
    }

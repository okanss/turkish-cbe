"""Versioned reasoning interventions on an unchanged frozen problem frame."""

from __future__ import annotations

from inference import prompts as direct

from .parser import PARSER_VERSIONS, STRATEGIES


PROMPT_VERSIONS = {
    "direct": direct.PROMPT_VERSION,
    "provider_effort": direct.PROMPT_VERSION,
    "zero_shot_reasoning": "reasoning-zero-shot-v1",
    "structured_reasoning": "reasoning-structured-v1",
}
SYSTEM_MESSAGES = {
    "tr": ("Sen mantıksal sıralama sorularını yanıtlayan bir asistansın. "
           "Verilen öncülleri temel al. Açıklamanın ardından istenen biçimde tek bir son cevap satırı üret."),
    "en": ("You are an assistant answering logical ordering questions. "
           "Base your answer on the given premises. After your explanation, produce one final answer line in the requested format."),
}
INSTRUCTIONS = {
    "zero_shot_reasoning": {
        "tr": ("Verilen öncüllerden hareketle soruyu adım adım değerlendir ve açıklamanı yaz. "
               "Yanıtını tek bir son satırda FINAL: EVET veya FINAL: HAYIR biçiminde bitir. "
               "Son cevap satırından sonra başka metin yazma."),
        "en": ("Evaluate the question step by step from the given premises and write your explanation. "
               "End your response with one final line in the form FINAL: YES or FINAL: NO. "
               "Do not write any text after the final answer line."),
    },
    "structured_reasoning": {
        "tr": ("1. Öncüllerdeki ilişkileri, anlamlarını koruyarak ortak bir yön altında normalize et.\n"
               "2. Soruda istenen ilişkinin yönünü belirle.\n"
               "3. Gerekli ardışık ilişkilerden geçişli çıkarımı değerlendir.\n"
               "4. Sonucu tek bir son satırda FINAL: EVET veya FINAL: HAYIR biçiminde ver.\n"
               "Son cevap satırından sonra başka metin yazma."),
        "en": ("1. Normalize the premise relations to a common orientation while preserving their meaning.\n"
               "2. Identify the direction of the relation requested by the question.\n"
               "3. Evaluate the transitive inference through the necessary successive relations.\n"
               "4. Give the conclusion on one final line in the form FINAL: YES or FINAL: NO.\n"
               "Do not write any text after the final answer line."),
    },
}


def _validate(language, strategy):
    if type(language) is not str or language not in SYSTEM_MESSAGES:
        raise ValueError("source language must be 'tr' or 'en'")
    if type(strategy) is not str or strategy not in STRATEGIES:
        raise ValueError("unknown reasoning strategy")


def instruction_overhead_utf8_bytes(language: str, strategy: str) -> int:
    """Nonnegative instruction/system byte increment over the direct messages.

    This is a planning-proxy feature, not a tokenizer measurement, chat framing
    count or certified upper bound on provider context use. The frozen problem
    frame is identical in all strategies and is excluded from the increment.
    """
    _validate(language, strategy)
    if strategy in {"direct", "provider_effort"}:
        return 0
    original = direct.SYSTEM_MESSAGES[language] + direct._FRAMES[language]["instruction"]
    changed = SYSTEM_MESSAGES[language] + INSTRUCTIONS[strategy][language]
    return max(0, len(changed.encode("utf-8")) - len(original.encode("utf-8")))


def make_messages(row: dict, strategy: str) -> dict:
    """Build localized strategy messages without consulting gold/latent metadata.

    A/D retain byte-identical existing direct messages and their version/hash.
    B/C retain the complete existing context, entity inventory, premises and
    question; only the system and final instruction change. Source validation
    remains the frozen direct adapter's responsibility.
    """
    if type(strategy) is not str or strategy not in STRATEGIES:
        raise ValueError("unknown reasoning strategy")
    result = direct.make_messages(row)
    result.update(strategy=strategy, parser_version=PARSER_VERSIONS[strategy])
    if strategy in {"direct", "provider_effort"}:
        return result
    language = row["language"]
    old_user = result["messages"][1]["content"]
    suffix = "\n\n" + direct._FRAMES[language]["instruction"]
    if not old_user.endswith(suffix):
        raise ValueError("direct prompt instruction frame changed")
    user = old_user[:-len(suffix)] + "\n\n" + INSTRUCTIONS[strategy][language]
    messages = [{"role": "system", "content": SYSTEM_MESSAGES[language]},
                {"role": "user", "content": user}]
    result.update(messages=messages, messages_sha256=direct.messages_sha256(messages),
                  prompt_version=PROMPT_VERSIONS[strategy])
    return result

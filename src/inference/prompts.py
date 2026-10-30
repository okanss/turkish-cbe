"""Adapt a frozen dataset prompt to the localized zero-shot message protocol.

The source context, inventory, premise order and question are copied verbatim.
This adapter checks the closed dataset framing and source-text hash; dataset
manifest validation remains responsible for binding that hash to frozen data.
Gold labels and latent orders are never consulted when constructing messages.
Frozen source-token counts do not measure these new system/user messages.
"""

from __future__ import annotations

import hashlib
import json
import re


PROMPT_VERSION = "zero-shot-v1"
SYSTEM_MESSAGES = {
    "tr": ("Sen mantıksal sıralama sorularını yanıtlayan bir asistansın. "
           "Verilen öncülleri temel al ve yalnızca istenen cevabı üret."),
    "en": ("You are an assistant answering logical ordering questions. "
           "Base your answer on the given premises and produce only the requested answer."),
}
_FRAMES = {
    "tr": {"inventory": "\nVarlıklar: ", "premises": "\nÖncüller:\n",
           "question": "\nSoru: ", "question_label": "Soru: ",
           "old_instruction": "\nYanıtınızı yalnız YES veya NO olarak verin.",
           "instruction": "Yalnızca EVET veya HAYIR yaz."},
    "en": {"inventory": "\nEntities: ", "premises": "\nPremises:\n",
           "question": "\nQuestion: ", "question_label": "Question: ",
           "old_instruction": "\nProvide your answer only as YES or NO.",
           "instruction": "Write only YES or NO."},
}
_SHA256 = re.compile(r"[0-9a-f]{64}")


def messages_sha256(messages: list[dict]) -> str:
    """Hash UTF-8 canonical JSON of the actual provider-independent messages."""
    payload = json.dumps(messages, sort_keys=True, ensure_ascii=False,
                         separators=(",", ":"), allow_nan=False).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _parts(prompt: str, language: str, n: int) -> tuple[str, str]:
    frame = _FRAMES[language]
    if "\r" in prompt or not prompt.endswith(frame["old_instruction"]):
        raise ValueError("source prompt has an unsupported answer-instruction frame")
    question_body = prompt[:-len(frame["old_instruction"])]
    if any(question_body.count(item["question"]) != (1 if key == language else 0)
           for key, item in _FRAMES.items()):
        raise ValueError("source prompt must contain exactly one localized question marker")
    premise_block, question = question_body.split(frame["question"])
    if (not question or question != question.strip() or "\n" in question
            or not question.endswith("?")):
        raise ValueError("source question must be one nonempty question line")
    if (premise_block.count(frame["inventory"]) != 1
            or premise_block.count(frame["premises"]) != 1):
        raise ValueError("source prompt has an unsupported inventory/premise frame")
    context, inventory_and_premises = premise_block.split(frame["inventory"])
    if not context or context != context.strip() or "\n" in context:
        raise ValueError("source relation context must be one nonempty line")
    inventory, premises = inventory_and_premises.split(frame["premises"])
    if not inventory.endswith(".") or "\n" in inventory:
        raise ValueError("source entity inventory must be one period-terminated line")
    names = inventory[:-1].split(", ")
    if (len(names) != n or len(set(names)) != n
            or any(not name or name != name.strip() for name in names)):
        raise ValueError("source inventory must contain n unique names")
    lines = premises.split("\n")
    if (len(lines) != n - 1 or any(not line.startswith("- ") or not line[2:]
                                 or line != line.strip() for line in lines)):
        raise ValueError("source prompt must contain exactly n-1 premise lines")
    return premise_block, question


def make_messages(row: dict) -> dict:
    """Build the requested system/user messages from one verified source row.

Required source fields are ``language``, ``n``, ``prompt`` and
``prompt_sha256``. All other metadata, including the gold answer, is ignored.
The returned message hash differs from the frozen source-prompt hash because
the inference instruction and a system message are part of this protocol.
"""
    if not isinstance(row, dict):
        raise ValueError("source row must be a dictionary")
    language, n = row.get("language"), row.get("n")
    if type(language) is not str or language not in _FRAMES:
        raise ValueError("source language must be 'tr' or 'en'")
    if type(n) is not int or not 4 <= n <= 1024:
        raise ValueError("source n must be an integer in [4, 1024]")
    prompt, source_sha = row.get("prompt"), row.get("prompt_sha256")
    if (not isinstance(prompt, str) or not isinstance(source_sha, str)
            or _SHA256.fullmatch(source_sha) is None):
        raise ValueError("source prompt and lowercase SHA-256 digest are required")
    actual_sha = hashlib.sha256(prompt.encode("utf-8")).hexdigest()
    if actual_sha != source_sha:
        raise ValueError("source prompt SHA-256 mismatch")
    premise_block, question = _parts(prompt, language, n)
    frame = _FRAMES[language]
    user = (premise_block + "\n\n" + frame["question_label"] + question
            + "\n\n" + frame["instruction"])
    messages = [{"role": "system", "content": SYSTEM_MESSAGES[language]},
                {"role": "user", "content": user}]
    return {"messages": messages, "prompt_version": PROMPT_VERSION,
            "source_prompt_sha256": source_sha,
            "messages_sha256": messages_sha256(messages)}

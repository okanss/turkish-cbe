"""Exact, offline BPE lengths for two frozen model tokenizer mappings.

Only supplied plain text is counted. These are not billable API token counts:
chat framing, system messages, generated output and provider changes are outside
this measurement. Loading never downloads vocabulary files or opens a network
connection. The optional ``tiktoken`` dependency is imported only on use.
"""

from __future__ import annotations

import base64
from collections import OrderedDict
from copy import deepcopy
from dataclasses import dataclass
import hashlib
from importlib import metadata
import json
from pathlib import Path
import types
import unicodedata


MODEL_TO_ENCODING = {
    "gpt-4-turbo-2024-04-09": "cl100k_base",
    "gpt-4o-2024-08-06": "o200k_base",
}

TIKTOKEN_VERSION = "0.12.0"

VOCABULARY_SHA256 = {
    "cl100k_base": "223921b76ee99bde995b7ff738513eef100fb51d18c93597a113bcffe865b2a7",
    "o200k_base": "446a9538cb6c348e3516120d7c08b09f57c36495e2acfffe59a5bf8b0cfb1a2d",
}


class ModelTokenizationError(ValueError):
    """The pinned local tokenizer bundle cannot be verified or loaded."""


@dataclass(frozen=True)
class PreparedPrefix:
    """Verified prefix counts tied to the tokenizer object that prepared them."""

    text: str
    counts: tuple[tuple[str, int], ...]
    last_pieces: tuple[tuple[str, str], ...]
    owner: object


def _validated_text(text: str) -> None:
    if not isinstance(text, str):
        raise TypeError("text must be a Unicode string")
    try:
        text.encode("utf-8", errors="strict")
    except UnicodeEncodeError as error:
        raise ValueError("text must be losslessly encodable as UTF-8") from error


def _load_ranks(data: bytes) -> dict[bytes, int]:
    ranks = {}
    seen_ranks = set()
    try:
        for line in data.splitlines():
            token_text, rank_text = line.split()
            token = base64.b64decode(token_text, validate=True)
            rank = int(rank_text)
            if not token or token in ranks or rank < 0 or rank in seen_ranks:
                raise ValueError("duplicate or invalid vocabulary entry")
            ranks[token] = rank
            seen_ranks.add(rank)
    except (ValueError, TypeError) as error:
        raise ModelTokenizationError("invalid local BPE vocabulary") from error
    if not ranks or seen_ranks != set(range(len(ranks))):
        raise ModelTokenizationError("BPE ranks must be contiguous from zero")
    return ranks


class ModelTokenizers:
    """Load frozen local encodings and report exact ordinary-text BPE lengths.

    ``resource_dir/encodings.json`` records a schema version, the exact installed
    tiktoken version, official encoding definitions and model mappings. Both BPE
    files are verified against fixed official SHA-256 hashes. Encoding regexes
    and special tokens are independently compared with the installed package's
    constructors using an isolated, local-only vocabulary loader.
    """

    def __init__(self, resource_dir: Path):
        resource_dir = Path(resource_dir).resolve()
        descriptor_path = resource_dir / "encodings.json"
        try:
            descriptor_bytes = descriptor_path.read_bytes()
            descriptor = json.loads(descriptor_bytes)
        except (OSError, UnicodeError, json.JSONDecodeError) as error:
            raise ModelTokenizationError("missing or invalid encodings.json") from error
        if (not isinstance(descriptor, dict)
                or set(descriptor) != {"schema_version", "tiktoken_version", "encodings", "model_to_encoding"}
                or type(descriptor["schema_version"]) is not int
                or descriptor["schema_version"] != 1
                or descriptor["tiktoken_version"] != TIKTOKEN_VERSION
                or descriptor["model_to_encoding"] != MODEL_TO_ENCODING
                or not isinstance(descriptor["encodings"], dict)
                or set(descriptor["encodings"]) != set(VOCABULARY_SHA256)):
            raise ModelTokenizationError("unsupported tokenizer bundle descriptor")
        try:
            import regex
            import tiktoken
            import tiktoken.model
            from tiktoken_ext import openai_public
            installed_version = metadata.version("tiktoken")
        except ImportError as error:
            raise ModelTokenizationError(
                "install the pinned optional tokenizer dependencies to measure model tokens"
            ) from error
        if installed_version != TIKTOKEN_VERSION:
            raise ModelTokenizationError("installed tiktoken version differs from frozen bundle")
        for model, encoding in MODEL_TO_ENCODING.items():
            if tiktoken.model.encoding_name_for_model(model) != encoding:
                raise ModelTokenizationError("installed model-to-encoding mapping differs")

        self._encodings = {}
        self._patterns = {}
        for name, expected_hash in VOCABULARY_SHA256.items():
            entry = descriptor["encodings"][name]
            if (not isinstance(entry, dict)
                    or set(entry) != {"file", "sha256", "pat_str", "special_tokens", "source_url"}
                    or entry["file"] != f"{name}.tiktoken"
                    or entry["source_url"] != f"https://openaipublic.blob.core.windows.net/encodings/{name}.tiktoken"
                    or entry["sha256"] != expected_hash
                    or not isinstance(entry["pat_str"], str)
                    or not isinstance(entry["special_tokens"], dict)):
                raise ModelTokenizationError(f"invalid encoding descriptor: {name}")
            path = (resource_dir / entry["file"]).resolve()
            if path.parent != resource_dir:
                raise ModelTokenizationError("vocabulary must be contained in resource directory")
            try:
                data = path.read_bytes()
            except OSError as error:
                raise ModelTokenizationError(f"missing local vocabulary: {name}") from error
            if hashlib.sha256(data).hexdigest() != expected_hash:
                raise ModelTokenizationError(f"local vocabulary hash mismatch: {name}")
            ranks = _load_ranks(data)

            # Clone the official constructor with isolated globals. Its loader
            # returns only the already verified local vocabulary. No global
            # monkeypatch or package download/cache operation is performed.
            def local_loader(url, expected_hash=None, *, _name=name,
                             _hash=expected_hash, _ranks=ranks):
                if (url != f"https://openaipublic.blob.core.windows.net/encodings/{_name}.tiktoken"
                        or expected_hash != _hash):
                    raise ModelTokenizationError("official encoding vocabulary definition changed")
                return _ranks

            official = getattr(openai_public, name)
            globals_with_local_loader = dict(official.__globals__)
            globals_with_local_loader["load_tiktoken_bpe"] = local_loader
            local_constructor = types.FunctionType(official.__code__, globals_with_local_loader)
            official_definition = local_constructor()
            if (entry["pat_str"] != official_definition["pat_str"]
                    or entry["special_tokens"] != official_definition["special_tokens"]):
                raise ModelTokenizationError(f"encoding rules differ from pinned package: {name}")
            self._encodings[name] = tiktoken.Encoding(
                name=name,
                pat_str=entry["pat_str"],
                mergeable_ranks=ranks,
                special_tokens=entry["special_tokens"],
            )
            self._patterns[name] = regex.compile(entry["pat_str"])

        self._marker_last_pieces={}
        for marker in ("\nQuestion:","\nSoru:"):
            pieces={name:pattern.findall(marker)[-1]
                    for name,pattern in self._patterns.items()}
            if any(piece!=":" for piece in pieces.values()):
                raise ModelTokenizationError("official question marker boundary changed")
            self._marker_last_pieces[marker]=pieces

        self._descriptor = {
            **deepcopy(descriptor),
            "descriptor_sha256": hashlib.sha256(descriptor_bytes).hexdigest(),
            "regex_version": metadata.version("regex"),
            "is_model_tokenizer": True,
            "normalization": "none; input text is preserved exactly",
            "special_token_handling": "encode_ordinary; literal special-token text is ordinary input",
            "count_scope": "exact plain prompt text; excludes chat framing, system messages and output",
            "network_required": False,
        }
        self._prefix_cache = OrderedDict()
        self._short_text_cache = OrderedDict()
        self._owner = object()
        self._safe_joins = 0
        self._full_fallbacks = 0

    def descriptor(self) -> dict:
        """Return a detached, JSON-serializable tokenizer provenance record."""
        return deepcopy(self._descriptor)

    def measure(self, text: str) -> dict[str, int]:
        """Count complete supplied text with both exact model BPE encodings."""
        _validated_text(text)
        if len(text)<=512:
            cached=self._short_text_cache.get(text)
            if cached is not None:
                self._short_text_cache.move_to_end(text)
                return dict(cached)
        counts={name: len(encoding.encode_ordinary(text))
                for name, encoding in self._encodings.items()}
        if len(text)<=512:
            self._short_text_cache[text]=dict(counts)
            if len(self._short_text_cache)>4096:
                self._short_text_cache.popitem(last=False)
        return counts

    def prepare_prefix(self, prefix: str) -> PreparedPrefix:
        """Encode a shared prompt prefix once for subsequent exact joins."""
        _validated_text(prefix)
        cached = self._prefix_cache.get(prefix)
        if cached is None:
            marker=next((m for m in self._marker_last_pieces if prefix.endswith(m)),None)
            if marker is not None:
                # Both verified official regexes end a word piece in letters
                # or marks, never a colon. Their punctuation alternatives
                # cannot consume letters, and their newline alternatives
                # cannot consume the following header word. Therefore this
                # complete newline + word + colon suffix always has the same
                # final ':' piece regardless of any preceding prompt text.
                # Derive/check that piece on the marker once, rather than walk
                # thousands of already BPE-tokenized prefix pieces in Python.
                pieces=self._marker_last_pieces[marker]
            else:
                pieces = {}
                for name, pattern in self._patterns.items():
                    last = None
                    for match in pattern.finditer(prefix):
                        last = match
                    pieces[name] = last.group() if last else ""
            cached = PreparedPrefix(prefix, tuple(self.measure(prefix).items()),
                                    tuple(pieces.items()), self._owner)
            self._prefix_cache[prefix] = cached
            if len(self._prefix_cache) > 32:
                self._prefix_cache.popitem(last=False)
        else:
            self._prefix_cache.move_to_end(prefix)
        return cached

    def count_parts(self, prefix: str | PreparedPrefix, suffix: str) -> dict[str, int]:
        """Count concatenated text exactly, caching a safe question prefix.

        BPE counts are not generally additive. Only the controlled ``Question:``
        or ``Soru:`` boundary followed by a space and a Unicode letter can take
        the cached path, after both official regexes prove the last prefix
        piece and all suffix pieces remain unchanged when joined. Arbitrary
        fragments (including words cut in half) use complete-text encoding.
        """
        _validated_text(suffix)
        if isinstance(prefix, PreparedPrefix):
            if prefix.owner is not self._owner:
                raise ModelTokenizationError("prepared prefix belongs to a different tokenizer object")
            prepared = prefix
            prefix_text = prefix.text
        else:
            _validated_text(prefix)
            prepared = None
            prefix_text = prefix
        candidate = (prefix_text.endswith(("\nQuestion:", "\nSoru:"))
                     and len(suffix) > 1 and suffix[0] == " "
                     and unicodedata.category(suffix[1]).startswith("L"))
        if not candidate:
            self._full_fallbacks += 1
            return self.measure(prefix_text + suffix)
        if prepared is None:
            prepared = self.prepare_prefix(prefix_text)
        prefix_counts, last_pieces = dict(prepared.counts), dict(prepared.last_pieces)
        for name, pattern in self._patterns.items():
            last_piece = last_pieces[name]
            if (last_piece != ":"
                    or pattern.findall(last_piece + suffix) != [last_piece, *pattern.findall(suffix)]):
                self._full_fallbacks += 1
                return self.measure(prefix_text + suffix)
        self._safe_joins += 1
        suffix_counts = self.measure(suffix)
        return {name: prefix_counts[name] + suffix_counts[name] for name in self._encodings}

    def cache_info(self) -> dict[str, int]:
        """Expose cache counters for verification, never part of provenance."""
        return {"cached_prefixes": len(self._prefix_cache),
                "cached_short_texts": len(self._short_text_cache),
                "safe_joins": self._safe_joins, "full_fallbacks": self._full_fallbacks}

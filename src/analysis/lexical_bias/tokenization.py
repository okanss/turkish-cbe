"""Offline, model-bound lexical and prompt token measurements.

No model weights, inference endpoint, tokenizer download, chat heuristic, or
cross-model tokenizer fallback is used. Isolated strings, frozen source prompts,
concatenated message contents, and locally verified chat templates have distinct
count scopes. An injected tokenizer is always labeled as a synthetic fixture.
"""

from __future__ import annotations

from copy import deepcopy
import hashlib
from importlib import metadata
import json
from pathlib import Path
import re
from typing import Any

from complete_grid.model_tokens import MODEL_TO_ENCODING, ModelTokenizers, TIKTOKEN_VERSION
from inference.config import validate_model
from inference.prompts import make_messages
from inference.providers import ProviderError, _chat_options, _template


TOKENIZATION_VERSION = "0.8.0"
_COMMIT = re.compile(r"[0-9a-f]{40}\Z")
_REPOSITORY = re.compile(r"[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+\Z")
_HASH = re.compile(r"[0-9a-f]{64}\Z")
# Every bundle needs the fast-tokenizer pair and a license file; the split BPE
# files exist only in some repositories (e.g. Qwen) and stay verified when present.
_TOKENIZER_REQUIRED = {"tokenizer.json", "tokenizer_config.json"}
_TOKENIZER_LICENSES = {"LICENSE", "LICENSE.txt"}
_TOKENIZER_OPTIONAL = {"vocab.json", "merges.txt", "special_tokens_map.json", "config.json"}
_TOKENIZER_FILES = _TOKENIZER_REQUIRED | _TOKENIZER_LICENSES | _TOKENIZER_OPTIONAL
_MESSAGE_SEPARATOR = "\n\n"
_MATCH_KEYS = ("latent_instance_id", "query_instance_id", "split", "n", "query_type",
               "query_distance", "query_direction", "query_subject", "query_object",
               "query_relation", "condition", "name_control", "relation_id")


def _json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(",", ":"), allow_nan=False)


def _sha(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _text(value: Any, name: str) -> str:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{name} must be a nonempty Unicode string")
    value.encode("utf-8", errors="strict")
    return value


def _specs(models: list[dict]) -> list[dict]:
    if not isinstance(models, list):
        raise ValueError("models must be a list of complete model registry entries")
    seen = set()
    for spec in models:
        validate_model(spec)
        if spec["id"] in seen:
            raise ValueError("duplicate model registry identifier")
        seen.add(spec["id"])
    return deepcopy(models)


def _ids(value: Any) -> int:
    if not isinstance(value, (list, tuple)) or any(type(item) is not int or item < 0 for item in value):
        raise ValueError("tokenizer must return a flat list of nonnegative token IDs")
    return len(value)


class _AssetError(ValueError):
    pass


def _verified_assets(root: Path, spec: dict) -> dict:
    """Verify every bundled cache file before a local-only HF tokenizer load."""
    cache = (root / "tokenizers/hf").resolve()
    try:
        manifest_bytes = (cache / "manifest.json").read_bytes()
    except OSError as exc:
        raise _AssetError("missing_local_asset_manifest") from exc
    try:
        manifest = json.loads(manifest_bytes)
        if (not isinstance(manifest, dict) or type(manifest.get("schema_version")) is not int
                or manifest["schema_version"] != 1 or manifest.get("contains_model_weights") is not False
                or not isinstance(manifest.get("models"), dict)):
            raise ValueError()
    except (ValueError, UnicodeError) as exc:
        raise _AssetError("invalid_local_asset_manifest") from exc
    repo, revision = spec["tokenizer"]["id"], spec["tokenizer"]["revision"]
    entry = manifest["models"].get(repo)
    if entry is None:
        raise _AssetError("missing_local_tokenizer_assets")
    if not isinstance(entry, dict) or entry.get("revision") != revision:
        raise _AssetError("local_asset_revision_mismatch")
    if entry.get("chat_template_sha256") != spec["tokenizer"]["chat_template_sha256"]:
        raise _AssetError("local_asset_template_pin_mismatch")
    files = entry.get("files")
    if not isinstance(files, list):
        raise _AssetError("invalid_local_asset_manifest")
    expected_parent = Path("models--" + repo.replace("/", "--")) / "snapshots" / revision
    names, records = set(), []
    tokenizer_config = None
    for item in files:
        if (not isinstance(item, dict) or not isinstance(item.get("path"), str)
                or type(item.get("bytes")) is not int or item["bytes"] < 0
                or not isinstance(item.get("sha256"), str) or not _HASH.fullmatch(item["sha256"])):
            raise _AssetError("invalid_local_asset_manifest")
        relative = Path(item["path"])
        if (relative.is_absolute() or relative.parent != expected_parent
                or relative.name not in _TOKENIZER_FILES or relative.name in names):
            raise _AssetError("unsafe_or_unexpected_local_asset_path")
        resolved = (cache / relative).resolve()
        if not resolved.is_relative_to(cache):
            raise _AssetError("unsafe_or_unexpected_local_asset_path")
        source = f"https://huggingface.co/{repo}/resolve/{revision}/{relative.name}"
        if item.get("source_url") != source:
            raise _AssetError("local_asset_source_revision_mismatch")
        try:
            data = resolved.read_bytes()
        except OSError as exc:
            raise _AssetError("missing_local_tokenizer_assets") from exc
        actual = hashlib.sha256(data).hexdigest()
        if len(data) != item["bytes"] or actual != item["sha256"]:
            raise _AssetError("local_asset_hash_mismatch")
        upstream = item.get("upstream_lfs_sha256")
        blob = item.get("git_blob_id")
        if upstream is not None:
            if upstream != actual:
                raise _AssetError("local_asset_upstream_hash_mismatch")
        elif (not isinstance(blob, str) or not _COMMIT.fullmatch(blob)
              or hashlib.sha1(f"blob {len(data)}\0".encode() + data).hexdigest() != blob):
            raise _AssetError("local_asset_upstream_hash_mismatch")
        names.add(relative.name)
        records.append({"path": item["path"], "sha256": actual, "bytes": len(data),
                        "git_blob_id": blob, "upstream_lfs_sha256": upstream})
        if relative.name == "tokenizer_config.json":
            try:
                tokenizer_config = json.loads(data)
            except (ValueError, UnicodeError) as exc:
                raise _AssetError("invalid_local_tokenizer_config") from exc
    if not _TOKENIZER_REQUIRED <= names or not names & _TOKENIZER_LICENSES:
        raise _AssetError("missing_required_local_tokenizer_file")
    expected_template = spec["tokenizer"]["chat_template_sha256"]
    if expected_template is not None and (not isinstance(tokenizer_config, dict)
            or not isinstance(tokenizer_config.get("chat_template"), str)
            or _sha(tokenizer_config["chat_template"]) != expected_template):
        raise _AssetError("local_asset_template_pin_mismatch")
    return {"asset_manifest_sha256": hashlib.sha256(manifest_bytes).hexdigest(),
            "asset_revision": revision, "cache_dir": "tokenizers/hf", "assets": records,
            "upstream_verification": "git blob SHA-1 or upstream LFS SHA-256; local bytes SHA-256"}


class _Backend:
    def __init__(self, root: Path, spec: dict, factory):
        self.spec = spec
        self.fixture = factory is not None
        self.tokenizer = None
        self.encoding = spec["tokenizer"]["encoding"]
        self.status = "unavailable"
        self.chat_status = "unavailable"
        self.provenance: dict[str, Any] = {"declared_tokenizer": deepcopy(spec["tokenizer"]),
            "network_used": False, "weights_loaded": False, "normalization": "none",
            "fixture_only": self.fixture}
        self.chat_options = {}
        if spec["provider"] == "openai":
            if (MODEL_TO_ENCODING.get(spec["model_id"]) != self.encoding
                    or spec["tokenizer"]["id"] != "tiktoken"
                    or spec["tokenizer"]["revision"] != f"tiktoken=={TIKTOKEN_VERSION}"):
                self.status = "model_tokenizer_binding_mismatch"
                self.chat_status = "unavailable_without_verified_model_mapping"
                return
            if factory is None:
                try:
                    self.tokenizer = ModelTokenizers(root / "tokenizers")
                    self.provenance["frozen_bundle"] = self.tokenizer.descriptor()
                except (ValueError, OSError, ImportError):
                    self.status = "unavailable_frozen_bpe_bundle"
                    self.chat_status = "unavailable_without_provider_chat_count"
                    return
            else:
                try:
                    self.tokenizer = factory(deepcopy(spec))
                except Exception:
                    self.status = "fixture_tokenizer_load_failed"
                    return
            self.status = "fixture_only" if self.fixture else "measured_frozen_bpe"
            self.chat_status = "unavailable_without_provider_chat_count"
            return
        revision = spec["tokenizer"]["revision"]
        identifier = spec["tokenizer"]["id"]
        if not isinstance(revision, str) or not _COMMIT.fullmatch(revision):
            self.status = "missing_immutable_tokenizer_revision"
            return
        if not _REPOSITORY.fullmatch(identifier) or any(part in {".", ".."} for part in identifier.split("/")):
            self.status = "unsupported_nonrepository_tokenizer_id"
            return
        if factory is None:
            try:
                from transformers import AutoTokenizer
            except ImportError:
                self.status = "missing_dependency"
                return
            try:
                self.provenance["verified_local_assets"] = _verified_assets(root, spec)
            except _AssetError as exc:
                self.status = str(exc)
                return
            try:
                self.tokenizer = AutoTokenizer.from_pretrained(identifier, revision=revision,
                    cache_dir=str((root / "tokenizers/hf").resolve()),
                    local_files_only=True, trust_remote_code=False)
                self.provenance["transformers_version"] = metadata.version("transformers")
            except (OSError, ValueError, ImportError):
                self.status = "missing_or_unusable_local_assets"
                self.tokenizer = None
                return
        else:
            try:
                self.tokenizer = factory(deepcopy(spec))
            except Exception:
                self.status = "fixture_tokenizer_load_failed"
                return
        self.provenance["tokenizer_class"] = type(self.tokenizer).__name__
        kwargs = getattr(self.tokenizer, "init_kwargs", {})
        loaded_revision = kwargs.get("_commit_hash") if isinstance(kwargs, dict) else None
        self.provenance["reported_loaded_commit"] = loaded_revision
        self.provenance["revision_verification"] = ("synthetic_injected_tokenizer; asset provenance unverified"
            if self.fixture else "local_from_pretrained_requested_immutable_commit")
        if loaded_revision is not None and loaded_revision != revision:
            self.status = "tokenizer_revision_mismatch"
            self.tokenizer = None
            return
        self.status = "fixture_only" if self.fixture else "measured_local_pinned_tokenizer"
        expected = spec["tokenizer"]["chat_template_sha256"]
        if expected is None:
            self.chat_status = "missing_chat_template_pin"
            return
        try:
            template = getattr(self.tokenizer, "chat_template", None)
            digest = _template(template, spec)
            self.chat_options = _chat_options(template, spec)
            self.provenance["verified_chat_template_sha256"] = digest
        except ProviderError as exc:
            self.status = exc.code
            self.chat_status = exc.code
            self.tokenizer = None
            return
        self.chat_status = ("fixture_only" if self.fixture else
            "measured_local_template_not_serving_attested" if spec["provider"] == "vllm" else
            "measured_verified_local_chat_template")

    def fields(self) -> dict:
        return {"tokenization_version": TOKENIZATION_VERSION, "model_id": self.spec["id"],
            "model_name": self.spec["model_id"], "model_family": self.spec["family"],
            "model_revision": self.spec["revision"], "provider": self.spec["provider"],
            "tokenizer_id": self.spec["tokenizer"]["id"],
            "tokenizer_revision": self.spec["tokenizer"]["revision"],
            "tokenizer_encoding": self.encoding, "tokenizer_status": self.status,
            "tokenizer_provenance": _json(self.provenance),
            "tokenizer_provenance_sha256": _sha(_json(self.provenance)),
            "fixture_only": self.fixture, "network_used": False, "weights_loaded": False}

    def plain(self, text: str) -> tuple[int | None, str]:
        if self.tokenizer is None:
            return None, self.status
        try:
            if self.spec["provider"] == "openai" and not self.fixture:
                return self.tokenizer.measure(text)[self.encoding], self.status
            return _ids(self.tokenizer.encode(text, add_special_tokens=False)), self.status
        except Exception:
            return None, "tokenizer_count_failed"

    def chat(self, messages: list[dict]) -> tuple[int | None, str]:
        if self.tokenizer is None or self.spec["provider"] == "openai" or self.chat_status == "missing_chat_template_pin":
            return None, self.chat_status
        try:
            count = _ids(self.tokenizer.apply_chat_template(deepcopy(messages), tokenize=True,
                add_generation_prompt=True, **self.chat_options))
            return count, self.chat_status
        except Exception:
            return None, "chat_tokenizer_count_failed"


def _ratio(numerator, denominator):
    return numerator / denominator if numerator is not None and denominator is not None and denominator else None


def _difference(after, before):
    return after - before if after is not None and before is not None else None


def measure_vocabulary(root: str | Path, models: list[dict], catalog: list[dict], *,
                       tokenizer_factory=None) -> list[dict]:
    """Measure exact isolated lexical strings in long bilingual rows.

    Ratios are TR/EN and deltas are after−before within the identical model and
    tokenizer. ``chat_token_count`` stays null for isolated vocabulary. Missing
    dependencies/assets produce explicit null counts; no alternate tokenizer is
    used. Supplying ``tokenizer_factory(spec)`` marks every row fixture-only.
    """
    specs = _specs(models)
    if not isinstance(catalog, list):
        raise ValueError("catalog must be a list of relation dictionaries")
    seen = set()
    for relation in catalog:
        if not isinstance(relation, dict):
            raise ValueError("catalog relations must be dictionaries")
        relation_id = _text(relation.get("relation_id"), "relation_id")
        _text(relation.get("family"), "family")
        if relation_id in seen:
            raise ValueError("duplicate catalog relation")
        seen.add(relation_id)
        directions = relation.get("directions")
        if not isinstance(directions, list) or len(directions) != 2 or {row.get("direction") for row in directions if isinstance(row, dict)} != {"before", "after"}:
            raise ValueError("each relation requires unique before/after directions")
        for direction in directions:
            for key in ("lemma", "term", "english_term", "expression", "english_expression"):
                _text(direction.get(key), key)
    result = []
    for spec in specs:
        backend = _Backend(Path(root).resolve(), spec, tokenizer_factory)
        for relation in catalog:
            group = []
            for direction in relation["directions"]:
                for language in ("tr", "en"):
                    term = direction["term" if language == "tr" else "english_term"]
                    expression = direction["expression" if language == "tr" else "english_expression"]
                    lemma = direction["lemma"] if language == "tr" else term
                    term_count, term_status = backend.plain(term)
                    expression_count, expression_status = backend.plain(expression)
                    lemma_count, lemma_status = backend.plain(lemma)
                    row = {**backend.fields(), "relation_id": relation["relation_id"],
                        "relation_family": relation["family"], "direction": direction["direction"],
                        "language": language, "term": term, "lemma": lemma, "expression": expression,
                        "lemma_scope": "catalog Turkish lemma" if language == "tr" else "English term; no lemmatization applied",
                        "term_sha256": _sha(term), "expression_sha256": _sha(expression),
                        "term_tokens": term_count, "expression_tokens": expression_count,
                        "lemma_token_count": lemma_count, "term_count_status": term_status,
                        "expression_count_status": expression_status, "lemma_count_status": lemma_status,
                        "count_scope": "plain_text_isolated_strings; add_special_tokens=false",
                        "normalization": "none; exact supplied Unicode text",
                        "chat_token_count": None, "chat_count_status": "not_applicable_isolated_text"}
                    group.append(row)
            coordinates = {(row["direction"], row["language"]): row for row in group}
            for row in group:
                tr, en = (coordinates[row["direction"], lang] for lang in ("tr", "en"))
                after, before = (coordinates[direction, row["language"]] for direction in ("after", "before"))
                row.update(term_tr_en_ratio=_ratio(tr["term_tokens"], en["term_tokens"]),
                    expression_tr_en_ratio=_ratio(tr["expression_tokens"], en["expression_tokens"]),
                    term_direction_delta_after_minus_before=_difference(after["term_tokens"], before["term_tokens"]),
                    expression_direction_delta_after_minus_before=_difference(after["expression_tokens"], before["expression_tokens"]))
            result.extend(group)
    return result


def measure_prompts(root: str | Path, models: list[dict], rows: list[dict], *,
                    tokenizer_factory=None) -> list[dict]:
    """Measure frozen prompt text, new message contents, and verified local chat.

    The message-content count encodes the complete system/user contents joined
    with two newlines, rather than summing separately tokenized fragments. It
    excludes role labels/framing and is never advertised as billable chat input.
    Ratios require an exact EN/TR logical/design match within a model. Naming
    deltas compare each language to its *same-language* native_shared source.
    """
    specs = _specs(models)
    if not isinstance(rows, list):
        raise ValueError("rows must be materialized source prompt dictionaries")
    prepared = []
    seen = set()
    for row in rows:
        if not isinstance(row, dict) or any(key not in row for key in _MATCH_KEYS):
            raise ValueError("materialized prompts require all logical/design matching keys")
        key = tuple(row[key] for key in _MATCH_KEYS) + (row.get("language"),)
        if key in seen:
            raise ValueError("duplicate prompt design coordinate; no implicit aggregation")
        seen.add(key)
        messages = make_messages(row)
        prepared.append((row, messages, _MESSAGE_SEPARATOR.join(part["content"] for part in messages["messages"])))
    result = []
    for spec in specs:
        backend = _Backend(Path(root).resolve(), spec, tokenizer_factory)
        measured = []
        for source, adapted, content in prepared:
            source_count, source_status = backend.plain(source["prompt"])
            content_count, content_status = backend.plain(content)
            chat_count, chat_status = backend.chat(adapted["messages"])
            if spec["provider"] == "openai" and not backend.fixture and source_count is not None:
                frozen = source.get("model_token_counts", {}).get(backend.encoding)
                if frozen is not None and frozen != source_count:
                    raise ValueError("frozen source prompt model-token count mismatch")
            row = {**backend.fields(), **{key: source[key] for key in _MATCH_KEYS},
                "language": source["language"], "prompt_id": source.get("prompt_id"),
                "grid_query_id": source.get("grid_query_id"), "grid_design_id": source.get("grid_design_id"),
                "gold_label": source.get("gold_label"), "source_prompt_sha256": adapted["source_prompt_sha256"],
                "messages_sha256": adapted["messages_sha256"], "prompt_version": adapted["prompt_version"],
                "messages_content_sha256": _sha(content),
                "source_prompt_plain_tokens": source_count, "source_prompt_count_status": source_status,
                "messages_content_plain_tokens": content_count, "messages_content_count_status": content_status,
                "messages_content_separator": _MESSAGE_SEPARATOR,
                "plain_text_count_scope": "exact source prompt / joined system-user contents; excludes chat framing",
                "chat_token_count": chat_count, "chat_count_status": chat_status,
                "chat_count_scope": "verified local template including generation prompt; not provider billed usage" if chat_count is not None else "unavailable",
                "name_policy": ("English names transliterated; Turkish names native" if source["name_control"] == "en_transliterated"
                                else "same declared name control in both languages")}
            measured.append(row)
        paired = {tuple(row[key] for key in _MATCH_KEYS) + (row["language"],): row for row in measured}
        for row in measured:
            key = tuple(row[field] for field in _MATCH_KEYS)
            tr, en = (paired.get(key + (language,)) for language in ("tr", "en"))
            if tr is not None and en is not None and tr["gold_label"] != en["gold_label"]:
                raise ValueError("matched language prompts changed gold label")
            for field, prefix in (("source_prompt_plain_tokens", "source_prompt"),
                                  ("messages_content_plain_tokens", "messages_content"),
                                  ("chat_token_count", "chat")):
                row[prefix + "_tr_en_ratio"] = _ratio(tr[field], en[field]) if tr and en else None
            row["language_pair_status"] = "matched_exact_design" if tr and en else "missing_counterpart"
            native_key = tuple("native_shared" if field == "name_control" else row[field] for field in _MATCH_KEYS)
            reference = paired.get(native_key + (row["language"],))
            row["name_control_reference_prompt_id"] = reference["prompt_id"] if reference else None
            row["name_control_comparison_scope"] = "same_language_same_logical_query_and_condition; native_shared reference"
            for field, prefix in (("source_prompt_plain_tokens", "source_prompt"),
                                  ("messages_content_plain_tokens", "messages_content"),
                                  ("chat_token_count", "chat")):
                row[prefix + "_name_delta_from_native_shared"] = _difference(row[field], reference[field]) if reference else None
        result.extend(measured)
    return result

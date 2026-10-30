"""Explicit, single-attempt providers. Importing this module performs no I/O.

HTTP adapters deliberately avoid SDK retries, redirects, fallback model IDs,
automatic prompt truncation, and hidden extraction of a final answer. The run
controller owns retries and durable records; providers return untouched JSON.
"""

from __future__ import annotations

from copy import deepcopy
from datetime import date, datetime, timezone
from email.utils import parsedate_to_datetime
import hashlib
import http.client
import importlib
import json
import math
import os
import re
import socket
from typing import Callable
from urllib.error import HTTPError, URLError
from urllib.parse import quote, urlsplit, urlunsplit
from urllib.request import HTTPRedirectHandler, Request, build_opener


class ProviderError(RuntimeError):
    """Safe diagnostic plus machine-readable recovery information.

    The message never includes the URL, credentials, response body, or an
    underlying exception. ``raw_response`` belongs in the immutable raw log.
    ``ambiguous`` means that generation may already have happened: automatic
    retry must not be inferred from ``retryable`` alone.
    """

    def __init__(self, code: str, *, retryable: bool = False,
                 ambiguous: bool = False, raw_response=None,
                 status_code: int | None = None,
                 retry_after_seconds: float | None = None):
        self.code = code
        self.retryable = retryable
        self.ambiguous = ambiguous
        self.raw_response = raw_response
        self.status_code = status_code
        self.retry_after_seconds = retry_after_seconds
        super().__init__(f"Provider operation failed: {code}")


class _NoRedirect(HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        return None


def _http_transport(method, url, payload, headers, timeout):
    """One HTTP attempt, no retries and no credential-bearing redirects."""
    data = None if payload is None else json.dumps(
        payload, ensure_ascii=False, allow_nan=False).encode("utf-8")
    request = Request(url, data=data, headers=headers, method=method)
    try:
        with build_opener(_NoRedirect()).open(request, timeout=timeout) as response:
            status, response_headers, body = (
                response.status, dict(response.headers), response.read())
    except HTTPError as exc:
        status, response_headers, body = exc.code, dict(exc.headers), exc.read()
    except (URLError, OSError, TimeoutError, socket.timeout,
            http.client.HTTPException):
        raise ProviderError("transport_error", retryable=True,
                            ambiguous=method == "POST") from None
    try:
        raw = json.loads(body.decode("utf-8"))
    except (UnicodeError, ValueError):
        # Preserve the actual failed response; never include it in str(error).
        raw = {"body_text": body.decode("utf-8", errors="replace")}
        if 200 <= status < 300:
            raise ProviderError("invalid_json_response", ambiguous=method == "POST",
                                raw_response=raw, status_code=status) from None
    return status, raw, response_headers


def _retry_after(headers):
    value = next((v for k, v in headers.items()
                  if k.lower() == "retry-after"), None)
    if value is None:
        return None
    try:
        seconds = float(value)
        return max(0.0, seconds) if math.isfinite(seconds) else None
    except (TypeError, ValueError):
        try:
            instant = parsedate_to_datetime(value)
            if instant.tzinfo is None:
                instant = instant.replace(tzinfo=timezone.utc)
            return max(0.0, (instant - datetime.now(timezone.utc)).total_seconds())
        except (TypeError, ValueError, OverflowError):
            return None


def _number(value):
    return value if isinstance(value, int) and not isinstance(value, bool) and value >= 0 else None


def _usage(raw, *, chat=False):
    data = raw.get("usage") or {}
    if not isinstance(data, dict):
        raise ProviderError("invalid_usage", ambiguous=True, raw_response=raw)
    incoming = "prompt_tokens" if chat else "input_tokens"
    outgoing = "completion_tokens" if chat else "output_tokens"
    details_in = data.get(incoming.replace("_tokens", "_tokens_details")) or {}
    details_out = data.get(outgoing.replace("_tokens", "_tokens_details")) or {}
    if not isinstance(details_in, dict) or not isinstance(details_out, dict):
        raise ProviderError("invalid_usage", ambiguous=True, raw_response=raw)
    return {"input_tokens": _number(data.get(incoming)),
            "output_tokens": _number(data.get(outgoing)),
            "reasoning_tokens": _number(details_out.get("reasoning_tokens")),
            "cached_input_tokens": _number(details_in.get("cached_tokens")),
            "total_tokens": _number(data.get("total_tokens"))}


def _messages(messages):
    if not isinstance(messages, list) or not messages:
        raise ProviderError("invalid_messages")
    for item in messages:
        if (not isinstance(item, dict) or set(item) != {"role", "content"}
                or item["role"] not in {"system", "user", "assistant"}
                or not isinstance(item["content"], str)):
            raise ProviderError("invalid_messages")
    return deepcopy(messages)


def _request_id(value):
    if not isinstance(value, str) or not re.fullmatch(r"[A-Za-z0-9_.:-]{1,128}", value):
        raise ProviderError("invalid_request_id")
    return value


def _base_url(value):
    if not isinstance(value, str):
        raise ProviderError("invalid_base_url")
    parsed = urlsplit(value)
    if (parsed.scheme not in {"http", "https"} or not parsed.hostname
            or parsed.username or parsed.password or parsed.query or parsed.fragment):
        raise ProviderError("invalid_base_url")
    if parsed.scheme == "http" and parsed.hostname not in {"localhost", "127.0.0.1", "::1"}:
        raise ProviderError("insecure_nonlocal_endpoint")
    return value.rstrip("/")


def _immutable_revision(value):
    return isinstance(value, str) and re.fullmatch(r"[0-9a-f]{40}", value) is not None


def _template(template, spec):
    if not isinstance(template, str) or not template:
        raise ProviderError("missing_chat_template")
    digest = hashlib.sha256(template.encode("utf-8")).hexdigest()
    if digest != spec["tokenizer"].get("chat_template_sha256"):
        raise ProviderError("chat_template_mismatch")
    thinking = spec["reasoning"].get("enable_thinking")
    if thinking is not None and not re.search(r"\benable_thinking\b", template):
        raise ProviderError("thinking_setting_unsupported_by_template")
    if re.search(r"\bstrftime_now\b", template) and not re.search(r"\bdate_string\b", template):
        raise ProviderError("implicit_template_clock_unsupported")
    if re.search(r"\bdate_string\b", template):
        value = spec["serving"].get("chat_template_date")
        try:
            if not isinstance(value, str) or date.fromisoformat(value).isoformat() != value:
                raise ValueError()
        except (ValueError, TypeError):
            raise ProviderError("frozen_chat_template_date_required") from None
    return digest


def _chat_options(template, spec):
    options = _reasoning(spec) or {}
    if re.search(r"\bdate_string\b", template):
        options["date_string"] = spec["serving"]["chat_template_date"]
    return options


def _reasoning(spec):
    reasoning = spec["reasoning"]
    mode, effort, enabled = (reasoning.get("mode"), reasoning.get("effort"),
                             reasoning.get("enable_thinking"))
    if mode not in {"provider_default", "none", "thinking"}:
        raise ProviderError("unsupported_reasoning_mode")
    if spec["provider"] == "openai":
        if enabled is not None:
            raise ProviderError("unsupported_thinking_setting")
        if mode == "provider_default":
            if effort is not None:
                raise ProviderError("conflicting_reasoning_settings")
            return None
        if mode == "none":
            if effort not in {None, "none"}:
                raise ProviderError("conflicting_reasoning_settings")
            return {"effort": "none"}
        if effort not in {"minimal", "low", "medium", "high", "xhigh", "max"}:
            raise ProviderError("unsupported_reasoning_effort")
        return {"effort": effort}
    if effort is not None:
        raise ProviderError("unsupported_reasoning_effort")
    if ((mode == "provider_default" and enabled is not None)
            or (mode == "none" and enabled is not False)
            or (mode == "thinking" and enabled is not True)):
        raise ProviderError("conflicting_reasoning_settings")
    return None if enabled is None else {"enable_thinking": enabled}


class _HTTPProvider:
    def __init__(self, spec, *, transport: Callable | None = None):
        self.spec = deepcopy(spec)
        self.base_url = _base_url(spec["serving"]["base_url"])
        self.transport = transport or _http_transport
        self.runtime = None
        self._count_cache = None
        _reasoning(spec)

    def _call(self, method, path, payload=None, *, request_id=None, root=False):
        base = self.base_url
        if root:
            parsed = urlsplit(base)
            path_prefix = parsed.path.rstrip("/")
            if path_prefix.endswith("/v1"):
                path_prefix = path_prefix[:-3]
            base = urlunsplit((parsed.scheme, parsed.netloc, path_prefix, "", ""))
        headers = {"Content-Type": "application/json", "Accept": "application/json"}
        env_name = self.spec["serving"].get("api_key_env")
        if env_name:
            key = os.environ.get(env_name)
            if not key:
                raise ProviderError("missing_api_key")
            headers["Authorization"] = f"Bearer {key}"
        elif self.spec["provider"] == "openai":
            raise ProviderError("missing_api_key_environment_name")
        if request_id:
            headers["X-Client-Request-Id"] = _request_id(request_id)
            headers["X-Request-Id"] = request_id
        try:
            status, raw, response_headers = self.transport(
                method, base + path, payload, headers,
                self.spec["serving"].get("timeout_seconds", 60))
        except ProviderError:
            raise
        except Exception:
            raise ProviderError("transport_error", retryable=True,
                                ambiguous=method == "POST") from None
        if not 200 <= status < 300:
            error = raw.get("error", {}) if isinstance(raw, dict) else {}
            code = error.get("code") if isinstance(error, dict) else None
            kind = error.get("type") if isinstance(error, dict) else None
            quota = {"insufficient_quota", "billing_hard_limit_reached", "quota_exceeded"}
            code_name = ("quota_exhausted" if status == 429 and (code in quota or kind in quota)
                         else "rate_limited" if status == 429
                         else "server_error" if status >= 500
                         else "http_error")
            raise ProviderError(code_name, retryable=code_name in {"rate_limited", "server_error"},
                                ambiguous=method == "POST" and (status >= 500 or status == 408),
                                raw_response=raw, status_code=status,
                                retry_after_seconds=_retry_after(response_headers))
        if not isinstance(raw, dict):
            raise ProviderError("invalid_response", ambiguous=method == "POST", raw_response=raw)
        return raw

    def _response_model(self, raw):
        if raw.get("model") != self.spec["model_id"]:
            raise ProviderError("provider_model_mismatch", ambiguous=True, raw_response=raw)

    def _count_call(self, path, payload, *, root=False):
        """Tokenization POSTs cannot have generated a billable model answer."""
        key = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
        if self._count_cache is not None and self._count_cache[0] == key:
            self.runtime["last_count_response"] = deepcopy(self._count_cache[1])
            self.runtime["last_count_payload_sha256"] = hashlib.sha256(key.encode()).hexdigest()
            return deepcopy(self._count_cache[1])
        try:
            raw = self._call("POST", path, payload, root=root)
        except ProviderError as exc:
            raise ProviderError(exc.code, retryable=exc.retryable, ambiguous=False,
                                raw_response=exc.raw_response, status_code=exc.status_code,
                                retry_after_seconds=exc.retry_after_seconds) from None
        self._count_cache = key, deepcopy(raw)
        self.runtime["last_count_response"] = deepcopy(raw)
        self.runtime["last_count_payload_sha256"] = hashlib.sha256(key.encode()).hexdigest()
        return raw


class OpenAIProvider(_HTTPProvider):
    """Responses REST API; model-specific incompatibilities remain explicit errors."""

    def __init__(self, spec, *, transport=None):
        super().__init__(spec, transport=transport)
        if spec.get("seed") is not None:
            raise ProviderError("seed_unsupported_by_responses_api")
        if spec["serving"].get("chat_template_date") is not None:
            raise ProviderError("openai_chat_template_date_unsupported")
        if spec.get("revision") != spec["model_id"]:
            raise ProviderError("openai_revision_must_equal_model_id")
        if spec["sampling"]["max_output_tokens"] < 16:
            raise ProviderError("responses_output_limit_minimum_is_16")

    def preflight(self):
        raw = self._call("GET", "/models/" + quote(self.spec["model_id"], safe=""))
        if raw.get("id") != self.spec["model_id"]:
            raise ProviderError("provider_model_mismatch", raw_response=raw)
        self.runtime = {"api": "openai.responses", "model_metadata": raw,
                        "resolved_revision": raw["id"], "revision_verification": "exact_model_id",
                        "tokenizer": deepcopy(self.spec["tokenizer"]),
                        "input_token_count_scope": "responses.input_tokens_including_chat_framing",
                        "automatic_truncation": False, "sdk_retries": 0}
        return deepcopy(self.runtime)

    def count_input_tokens(self, messages):
        messages = _messages(messages)
        if self.runtime is None:
            self.preflight()
        payload = {"model": self.spec["model_id"], "input": messages}
        reasoning = _reasoning(self.spec)
        if reasoning is not None:
            payload["reasoning"] = reasoning
        raw = self._count_call("/responses/input_tokens", payload)
        count = _number(raw.get("input_tokens"))
        if raw.get("object") != "response.input_tokens" or count is None or count == 0:
            raise ProviderError("invalid_token_count", raw_response=raw)
        return count

    def generate(self, messages, request_id):
        messages, request_id = _messages(messages), _request_id(request_id)
        if self.runtime is None:
            self.preflight()
        count = self.count_input_tokens(messages)
        if count + self.spec["sampling"]["max_output_tokens"] > self.spec["limits"]["context_tokens"]:
            raise ProviderError("context_limit_exceeded")
        payload = {"model": self.spec["model_id"], "input": messages,
                   "max_output_tokens": self.spec["sampling"]["max_output_tokens"],
                   "store": False, "stream": False, "truncation": "disabled",
                   "metadata": {"cbe_request_id": request_id}}
        for key in ("temperature", "top_p"):
            value = self.spec["sampling"].get(key)
            if value is not None:
                payload[key] = value
        reasoning = _reasoning(self.spec)
        if reasoning is not None:
            payload["reasoning"] = reasoning
        raw = self._call("POST", "/responses", payload, request_id=request_id)
        self._response_model(raw)
        if raw.get("status") not in {"completed", "incomplete"}:
            raise ProviderError("response_not_completed", ambiguous=True, raw_response=raw)
        output = raw.get("output")
        if not isinstance(output, list):
            raise ProviderError("invalid_response_output", ambiguous=True, raw_response=raw)
        texts, refusal = [], False
        for item in output:
            if not isinstance(item, dict):
                raise ProviderError("invalid_response_output", ambiguous=True, raw_response=raw)
            if item.get("type") == "message" and item.get("role") == "assistant":
                content = item.get("content")
                if not isinstance(content, list):
                    raise ProviderError("invalid_response_output", ambiguous=True, raw_response=raw)
                for part in content:
                    if not isinstance(part, dict):
                        raise ProviderError("invalid_response_output", ambiguous=True, raw_response=raw)
                    if part.get("type") == "output_text" and isinstance(part.get("text"), str):
                        texts.append(part["text"])
                    elif part.get("type") == "refusal":
                        refusal = True
        details = raw.get("incomplete_details") or {}
        reason = details.get("reason") if isinstance(details, dict) else None
        usage = _usage(raw)
        if usage["input_tokens"] is not None and usage["input_tokens"] != count:
            raise ProviderError("server_token_count_mismatch", ambiguous=True, raw_response=raw)
        return {"text": "\n".join(texts) if texts else None,
                "provider_model": raw["model"], "resolved_revision": raw["model"],
                "response_id": raw.get("id"),
                "finish_reason": reason or ("refusal" if refusal else raw["status"]),
                "truncated": reason == "max_output_tokens",
                "usage": usage, "raw": raw, "runtime": deepcopy(self.runtime)}


class VLLMProvider(_HTTPProvider):
    """vLLM Chat Completions plus explicit version and template verification."""

    def __init__(self, spec, *, transport=None):
        super().__init__(spec, transport=transport)
        if not spec["serving"].get("server_revision"):
            raise ProviderError("missing_server_revision")
        if (not _immutable_revision(spec.get("revision"))
                or not _immutable_revision(spec["tokenizer"].get("revision"))):
            raise ProviderError("immutable_huggingface_revisions_required")

    def preflight(self):
        models = self._call("GET", "/models")
        if not isinstance(models.get("data"), list):
            raise ProviderError("invalid_models_response", raw_response=models)
        found = [item for item in models["data"]
                 if isinstance(item, dict) and item.get("id") == self.spec["model_id"]]
        if len(found) != 1:
            raise ProviderError("provider_model_unavailable", raw_response=models)
        version = self._call("GET", "/version", root=True)
        if version.get("version") != self.spec["serving"]["server_revision"]:
            raise ProviderError("server_revision_mismatch", raw_response=version)
        info = self._call("GET", "/tokenizer_info", root=True)
        digest = _template(info.get("chat_template"), self.spec)
        self.chat_options = _chat_options(info["chat_template"], self.spec)
        actual_revision = info.get("_commit_hash")
        if actual_revision is not None and actual_revision != self.spec["tokenizer"]["revision"]:
            raise ProviderError("tokenizer_revision_mismatch", raw_response=info)
        max_context = found[0].get("max_model_len")
        if isinstance(max_context, int) and self.spec["limits"]["context_tokens"] > max_context:
            raise ProviderError("declared_context_exceeds_server_limit", raw_response=models)
        self.runtime = {"api": "vllm.chat.completions", "server_version": version,
                        "model_metadata": found[0], "resolved_revision": None,
                        "requested_revision": self.spec["revision"],
                        "revision_verification": "served_id_only_weights_revision_declared",
                        "tokenizer_revision_verification": (
                            "reported_commit" if actual_revision else "declared"),
                        "tokenizer": deepcopy(self.spec["tokenizer"]),
                        "chat_template_sha256": digest,
                        "chat_template_options": deepcopy(self.chat_options),
                        "server_generation_defaults": "not_exposed_by_verified_endpoints",
                        "input_token_count_scope": "server_chat_template_including_special_tokens",
                        "automatic_truncation": False, "sdk_retries": 0}
        return deepcopy(self.runtime)

    def count_input_tokens(self, messages):
        messages = _messages(messages)
        if self.runtime is None:
            self.preflight()
        payload = {"model": self.spec["model_id"], "messages": messages,
                   "add_generation_prompt": True, "add_special_tokens": False}
        if self.chat_options:
            payload["chat_template_kwargs"] = deepcopy(self.chat_options)
        raw = self._count_call("/tokenize", payload, root=True)
        count = _number(raw.get("count"))
        if count is None or count == 0:
            raise ProviderError("invalid_token_count", raw_response=raw)
        return count

    def generate(self, messages, request_id):
        messages, request_id = _messages(messages), _request_id(request_id)
        if self.runtime is None:
            self.preflight()
        count = self.count_input_tokens(messages)
        max_output = self.spec["sampling"]["max_output_tokens"]
        if count + max_output > self.spec["limits"]["context_tokens"]:
            raise ProviderError("context_limit_exceeded")
        payload = {"model": self.spec["model_id"], "messages": messages,
                   "max_tokens": max_output, "n": 1, "stream": False,
                   "add_generation_prompt": True, "add_special_tokens": False}
        for key in ("temperature", "top_p"):
            if self.spec["sampling"].get(key) is not None:
                payload[key] = self.spec["sampling"][key]
        if self.spec.get("seed") is not None:
            payload["seed"] = self.spec["seed"]
        if self.chat_options:
            payload["chat_template_kwargs"] = deepcopy(self.chat_options)
        raw = self._call("POST", "/chat/completions", payload, request_id=request_id)
        self._response_model(raw)
        choices = raw.get("choices")
        if not isinstance(choices, list) or len(choices) != 1 or not isinstance(choices[0], dict):
            raise ProviderError("invalid_choices", ambiguous=True, raw_response=raw)
        message = choices[0].get("message")
        if not isinstance(message, dict) or message.get("role") != "assistant":
            raise ProviderError("invalid_assistant_message", ambiguous=True, raw_response=raw)
        text = message.get("content")
        if text is not None and not isinstance(text, str):
            raise ProviderError("invalid_assistant_content", ambiguous=True, raw_response=raw)
        usage = _usage(raw, chat=True)
        if usage["input_tokens"] is not None and usage["input_tokens"] != count:
            raise ProviderError("server_token_count_mismatch", ambiguous=True, raw_response=raw)
        return {"text": text, "provider_model": raw["model"],
                "resolved_revision": None, "response_id": raw.get("id"),
                "finish_reason": choices[0].get("finish_reason") or "unknown",
                "truncated": choices[0].get("finish_reason") == "length",
                "usage": usage, "raw": raw, "runtime": deepcopy(self.runtime)}


class TransformersProvider:
    """Pinned decoder-only HF generation with the tokenizer's verified template."""

    def __init__(self, spec):
        self.spec = deepcopy(spec)
        self.runtime, self.tokenizer, self.model = None, None, None
        self.transformers, self.torch, self.model_config = None, None, None
        self._input_cache = None
        _reasoning(spec)
        if (not _immutable_revision(spec.get("revision"))
                or not _immutable_revision(spec["tokenizer"].get("revision"))):
            raise ProviderError("immutable_huggingface_revisions_required")
        temperature, top_p = (spec["sampling"].get("temperature"),
                              spec["sampling"].get("top_p"))
        if temperature == 0 and top_p not in {None, 1, 1.0}:
            raise ProviderError("top_p_unsupported_with_greedy_decoding")

    def preflight(self):
        try:
            self.transformers = importlib.import_module("transformers")
            self.torch = importlib.import_module("torch")
        except ImportError:
            raise ProviderError("missing_transformers_dependencies") from None
        serving = self.spec["serving"]
        kwargs = {"trust_remote_code": serving.get("trust_remote_code", False),
                  "local_files_only": serving.get("local_files_only", True)}
        try:
            self.model_config = self.transformers.AutoConfig.from_pretrained(
                self.spec["model_id"], revision=self.spec["revision"], **kwargs)
            self.tokenizer = self.transformers.AutoTokenizer.from_pretrained(
                self.spec["tokenizer"]["id"],
                revision=self.spec["tokenizer"]["revision"], **kwargs)
            template = self.tokenizer.get_chat_template()
        except Exception:
            raise ProviderError("pinned_model_or_tokenizer_unavailable") from None
        if getattr(self.model_config, "is_encoder_decoder", False):
            raise ProviderError("encoder_decoder_model_unsupported")
        actual = getattr(self.model_config, "_commit_hash", None)
        if actual is not None and actual != self.spec["revision"]:
            raise ProviderError("model_revision_mismatch")
        actual_tokenizer = getattr(self.tokenizer, "init_kwargs", {}).get("_commit_hash")
        if actual_tokenizer is not None and actual_tokenizer != self.spec["tokenizer"]["revision"]:
            raise ProviderError("tokenizer_revision_mismatch")
        digest = _template(template, self.spec)
        self.chat_options = _chat_options(template, self.spec)
        context = getattr(self.model_config, "max_position_embeddings", None)
        if isinstance(context, int) and self.spec["limits"]["context_tokens"] > context:
            raise ProviderError("declared_context_exceeds_model_limit")
        self.runtime = {"api": "transformers.generate", "resolved_revision": self.spec["revision"],
                        "revision_verification": "immutable_revision_argument" if actual is None else "reported_commit",
                        "tokenizer": deepcopy(self.spec["tokenizer"]),
                        "chat_template_sha256": digest,
                        "chat_template_options": deepcopy(self.chat_options),
                        "transformers_version": getattr(self.transformers, "__version__", "unknown"),
                        "torch_version": getattr(self.torch, "__version__", "unknown"),
                        "device": serving.get("device", "cpu"), "dtype": serving.get("dtype", "auto"),
                        "seed_scope": "torch_rng_best_effort_hardware_and_version_dependent",
                        "input_token_count_scope": "verified_chat_template_including_special_tokens",
                        "automatic_truncation": False, "weights_loaded": False}
        return deepcopy(self.runtime)

    def _tokenize(self, messages, *, tensors=False):
        messages = _messages(messages)
        if self.runtime is None:
            self.preflight()
        try:
            key = json.dumps(messages, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
            if self._input_cache is None or self._input_cache[0] != key:
                rendered = self.tokenizer.apply_chat_template(
                    messages, tokenize=False, add_generation_prompt=True, **self.chat_options)
                if not isinstance(rendered, str):
                    raise ValueError()
                # The verified template already adds its special tokens. Adding
                # another BOS/EOS here would change the actual model input.
                encoded = self.tokenizer(rendered, add_special_tokens=False, truncation=False)
                ids = encoded["input_ids"]
                if not isinstance(ids, list) or not ids or any(not isinstance(i, int) for i in ids):
                    raise ValueError()
                self._input_cache = key, rendered, list(ids)
            _, rendered, ids = self._input_cache
            self.runtime["last_input"] = {
                "rendered_text": rendered,
                "rendered_text_sha256": hashlib.sha256(rendered.encode("utf-8")).hexdigest(),
                "input_token_ids": list(ids),
                "input_token_ids_sha256": hashlib.sha256(json.dumps(ids, separators=(",", ":")).encode()).hexdigest()}
            if not tensors:
                return list(ids)
            encoded = self.tokenizer(rendered, add_special_tokens=False, truncation=False,
                                     return_tensors="pt")
            if encoded["input_ids"][0].tolist() != ids:
                raise ProviderError("chat_tokenization_changed")
            return encoded
        except ProviderError:
            raise
        except Exception:
            raise ProviderError("chat_template_application_failed") from None

    def count_input_tokens(self, messages):
        ids = self._tokenize(messages)
        if not isinstance(ids, list) or any(not isinstance(i, int) for i in ids):
            raise ProviderError("invalid_tokenizer_output")
        return len(ids)

    def _load_weights(self):
        if self.model is not None:
            return
        serving = self.spec["serving"]
        dtype_name = serving.get("dtype", "auto")
        if dtype_name not in {"auto", "float32", "float16", "bfloat16"}:
            raise ProviderError("unsupported_dtype")
        dtype = dtype_name if dtype_name == "auto" else getattr(self.torch, dtype_name)
        try:
            self.model = self.transformers.AutoModelForCausalLM.from_pretrained(
                self.spec["model_id"], revision=self.spec["revision"],
                config=self.model_config, dtype=dtype,
                trust_remote_code=serving.get("trust_remote_code", False),
                local_files_only=serving.get("local_files_only", True))
            self.model.to(serving.get("device", "cpu"))
            self.model.eval()
        except Exception:
            self.model = None
            raise ProviderError("pinned_model_weights_unavailable") from None
        actual = getattr(self.model.config, "_commit_hash", None)
        if actual is not None and actual != self.spec["revision"]:
            self.model = None
            raise ProviderError("model_revision_mismatch")
        self.runtime["weights_loaded"] = True
        self.runtime["generation_defaults"] = deepcopy(self.model.generation_config.to_dict())

    def generate(self, messages, request_id):
        request_id = _request_id(request_id)
        count = self.count_input_tokens(messages)
        limit = self.spec["sampling"]["max_output_tokens"]
        if count + limit > self.spec["limits"]["context_tokens"]:
            raise ProviderError("context_limit_exceeded")
        inputs = self._tokenize(messages, tensors=True)
        self._load_weights()
        try:
            inputs = inputs.to(self.model.device)
        except Exception:
            raise ProviderError("tensor_device_assignment_failed") from None
        sampling = self.spec["sampling"]
        kwargs = {"max_new_tokens": limit, "num_return_sequences": 1,
                  "return_dict_in_generate": True, "output_scores": False}
        if sampling.get("temperature") == 0:
            kwargs["do_sample"] = False
        elif sampling.get("temperature") is not None:
            kwargs.update(do_sample=True, temperature=sampling["temperature"])
        if sampling.get("top_p") is not None and sampling.get("temperature") != 0:
            kwargs["top_p"] = sampling["top_p"]
        try:
            devices = list(range(self.torch.cuda.device_count())) if self.torch.cuda.is_available() else []
            with self.torch.random.fork_rng(devices=devices), self.torch.inference_mode():
                if self.spec.get("seed") is not None:
                    self.torch.manual_seed(self.spec["seed"])
                result = self.model.generate(**inputs, **kwargs)
            generated = result.sequences[0, count:].tolist()
            text = self.tokenizer.decode(generated, skip_special_tokens=True)
            verbatim = self.tokenizer.decode(generated, skip_special_tokens=False)
        except Exception:
            raise ProviderError("generation_failed", ambiguous=True) from None
        eos_ids = self.model.generation_config.eos_token_id
        eos_ids = eos_ids if isinstance(eos_ids, (list, tuple)) else [eos_ids]
        ended = bool(generated) and generated[-1] in eos_ids
        truncated = len(generated) >= limit and not ended
        runtime = deepcopy(self.runtime)
        runtime["generation_arguments"] = kwargs
        return {"text": text, "provider_model": self.spec["model_id"],
                "resolved_revision": self.spec["revision"], "response_id": request_id,
                "finish_reason": "length" if truncated else "eos" if ended else "stopped",
                "truncated": truncated,
                "usage": {"input_tokens": count, "output_tokens": len(generated),
                          "reasoning_tokens": None, "cached_input_tokens": None,
                          "total_tokens": count + len(generated)},
                "raw": {"generated_token_ids": generated, "decoded_text": text,
                        "decoded_with_special_tokens": verbatim}, "runtime": runtime}


def create_provider(spec: dict, *, transport: Callable | None = None):
    """Instantiate only; external access requires explicit preflight/generate."""
    if not isinstance(spec, dict):
        raise ProviderError("invalid_model_spec")
    provider = spec.get("provider")
    if provider == "openai":
        return OpenAIProvider(spec, transport=transport)
    if provider == "vllm":
        return VLLMProvider(spec, transport=transport)
    if provider == "transformers":
        if transport is not None:
            raise ProviderError("http_transport_not_used_by_transformers")
        return TransformersProvider(spec)
    raise ProviderError("unsupported_provider")

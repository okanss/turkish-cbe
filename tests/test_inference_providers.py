"""Provider protocol fixtures only: no network, weights, or model results."""

from contextlib import nullcontext
from copy import deepcopy
import hashlib
import json
import os
import types
import unittest
from unittest.mock import patch
from urllib.error import URLError

import inference.providers as providers_module

from inference.providers import (
    OpenAIProvider, ProviderError, TransformersProvider, VLLMProvider,
    _http_transport, _retry_after, create_provider,
)


MESSAGES = [{"role": "system", "content": "Fixture instruction"},
            {"role": "user", "content": "Fixture question"}]
TEMPLATE = "{{ messages }}{% if enable_thinking %}<think>{% endif %}"
REVISION = "a" * 40


def model_spec(provider="openai"):
    return {"id": "fixture", "provider": provider, "family": "fixture_family",
            "parameters_billion": 1, "model_id": "fixture-model-2026-01-01",
            "revision": "fixture-model-2026-01-01" if provider == "openai" else REVISION,
            "tokenizer": {"id": "fixture-tokenizer", "revision": REVISION,
                          "encoding": None,
                          "chat_template_sha256": hashlib.sha256(TEMPLATE.encode()).hexdigest()},
            "serving": {"base_url": "https://provider.invalid/v1", "api_key_env": "FIXTURE_KEY",
                        "timeout_seconds": 10, "device": "cpu", "dtype": "float32",
                        "trust_remote_code": False, "local_files_only": True,
                        "chat_template_date": None if provider == "openai" else "2026-10-24",
                        "server_revision": "0.fixture"},
            "sampling": {"temperature": 0, "top_p": 1, "max_output_tokens": 32},
            "limits": {"context_tokens": 1024, "max_output_tokens": 64},
            "reasoning": {"mode": "provider_default", "effort": None, "enable_thinking": None},
            "seed": None, "pricing": {}, "enabled": True}


def responses_fixture(**changes):
    raw = {"id": "response-fixture", "model": "fixture-model-2026-01-01",
           "status": "completed", "output": [
               {"type": "reasoning", "summary": [{"text": "NO (not an answer)"}]},
               {"type": "message", "role": "assistant",
                "content": [{"type": "output_text", "text": "YES"}]}],
           "usage": {"input_tokens": 10, "output_tokens": 4, "total_tokens": 14,
                     "input_tokens_details": {"cached_tokens": 3},
                     "output_tokens_details": {"reasoning_tokens": 2}}}
    return {**raw, **changes}


def chat_fixture(**changes):
    return {"id": "chat-fixture", "model": "fixture-model-2026-01-01",
            "choices": [{"message": {"role": "assistant", "content": "EVET",
                                      "reasoning_content": "fixture only"},
                         "finish_reason": "stop"}],
            "usage": {"prompt_tokens": 10, "completion_tokens": 2, "total_tokens": 12},
            **changes}


class FixtureTransport:
    def __init__(self, *, response=None, version="0.fixture", template=TEMPLATE,
                 served_id="fixture-model-2026-01-01", count=10):
        self.calls = []
        self.response = response or responses_fixture()
        self.version, self.template, self.served_id, self.count = version, template, served_id, count

    def __call__(self, method, url, payload, headers, timeout):
        self.calls.append((method, url, deepcopy(payload), dict(headers), timeout))
        if url.endswith("/models/fixture-model-2026-01-01"):
            raw = {"id": self.served_id, "object": "model"}
        elif url.endswith("/models"):
            raw = {"data": [{"id": self.served_id, "max_model_len": 1024}]}
        elif url.endswith("/version"):
            raw = {"version": self.version}
        elif url.endswith("/tokenizer_info"):
            raw = {"chat_template": self.template, "_commit_hash": REVISION}
        elif url.endswith("/tokenize"):
            raw = {"count": self.count}
        elif url.endswith("/responses/input_tokens"):
            raw = {"object": "response.input_tokens", "input_tokens": self.count}
        else:
            raw = deepcopy(self.response)
        return 200, raw, {}


class HTTPProviderTests(unittest.TestCase):
    def setUp(self):
        self.env = patch.dict(os.environ, {"FIXTURE_KEY": "fixture_secret"})
        self.env.start()
        self.addCleanup(self.env.stop)

    def test_factory_instantiates_without_io_or_optional_import(self):
        transport = FixtureTransport()
        for provider, cls in (("openai", OpenAIProvider), ("vllm", VLLMProvider)):
            self.assertIsInstance(create_provider(model_spec(provider), transport=transport), cls)
        self.assertEqual(transport.calls, [])
        self.assertIsInstance(create_provider(model_spec("transformers")), TransformersProvider)
        with self.assertRaisesRegex(ProviderError, "unsupported_provider"):
            create_provider({"provider": "unconfigured"})

    def test_openai_preflight_only_retrieves_exact_model(self):
        transport = FixtureTransport()
        spec = model_spec()
        provider = create_provider(spec, transport=transport)
        result = provider.preflight()
        self.assertEqual([c[0] for c in transport.calls], ["GET"])
        self.assertEqual(result["resolved_revision"], spec["model_id"])
        result["tokenizer"].clear()
        self.assertTrue(provider.runtime["tokenizer"])
        self.assertEqual(provider.count_input_tokens(MESSAGES), 10)
        self.assertEqual(len(transport.calls), 2)
        self.assertEqual(transport.calls[-1][2], {"model": spec["model_id"], "input": MESSAGES})
        self.assertEqual(provider.count_input_tokens(MESSAGES), 10)
        self.assertEqual(len(transport.calls), 2)

    def test_openai_payload_usage_and_visible_content_only(self):
        transport = FixtureTransport()
        provider = create_provider(model_spec(), transport=transport)
        response = provider.generate(MESSAGES, "request-1")
        self.assertEqual(response["text"], "YES")
        self.assertEqual(response["usage"], {"input_tokens": 10, "output_tokens": 4,
                                           "reasoning_tokens": 2, "cached_input_tokens": 3,
                                           "total_tokens": 14})
        method, url, body, headers, timeout = transport.calls[-1]
        self.assertEqual(method, "POST")
        self.assertTrue(url.endswith("/responses"))
        self.assertEqual(body["input"], MESSAGES)
        self.assertEqual(body["metadata"], {"cbe_request_id": "request-1"})
        self.assertEqual(body["truncation"], "disabled")
        self.assertFalse(body["store"])
        self.assertFalse(body["stream"])
        self.assertEqual(body["temperature"], 0)
        self.assertEqual(headers["X-Client-Request-Id"], "request-1")
        self.assertEqual(headers["Authorization"], "Bearer fixture_secret")
        self.assertEqual(response["raw"], responses_fixture())

    def test_openai_omits_only_explicitly_null_sampling(self):
        spec, transport = model_spec(), FixtureTransport()
        spec["sampling"].update(temperature=None, top_p=None)
        create_provider(spec, transport=transport).generate(MESSAGES, "request-2")
        self.assertNotIn("temperature", transport.calls[-1][2])
        self.assertNotIn("top_p", transport.calls[-1][2])

    def test_openai_seed_and_revision_not_silently_replaced(self):
        spec = model_spec()
        spec["seed"] = 1
        with self.assertRaisesRegex(ProviderError, "seed_unsupported"):
            create_provider(spec)
        spec.update(seed=None, revision="latest")
        with self.assertRaisesRegex(ProviderError, "revision_must_equal"):
            create_provider(spec)

    def test_openai_unavailable_model_is_not_substituted(self):
        transport = FixtureTransport(served_id="another-model")
        with self.assertRaisesRegex(ProviderError, "model_mismatch"):
            create_provider(model_spec(), transport=transport).preflight()
        self.assertEqual(len(transport.calls), 1)

    def test_openai_response_model_mismatch_retains_raw_and_is_ambiguous(self):
        raw = responses_fixture(model="another-model")
        with self.assertRaises(ProviderError) as context:
            create_provider(model_spec(), transport=FixtureTransport(response=raw)).generate(MESSAGES, "r")
        self.assertTrue(context.exception.ambiguous)
        self.assertEqual(context.exception.raw_response, raw)

    def test_openai_context_budget_and_exact_count_payload(self):
        spec, transport = model_spec(), FixtureTransport(count=1000)
        spec["reasoning"].update(mode="thinking", effort="high")
        with self.assertRaisesRegex(ProviderError, "context_limit_exceeded"):
            create_provider(spec, transport=transport).generate(MESSAGES, "r")
        self.assertEqual(transport.calls[-1][2], {"model": spec["model_id"], "input": MESSAGES,
                                                "reasoning": {"effort": "high"}})
        self.assertFalse(any(c[1].endswith("/responses") for c in transport.calls))

    def test_failed_count_cannot_be_an_ambiguous_generation(self):
        provider = create_provider(model_spec(), transport=FixtureTransport())
        provider.preflight()
        provider.transport = lambda *args: (_ for _ in ()).throw(TimeoutError())
        with self.assertRaises(ProviderError) as ctx:
            provider.count_input_tokens(MESSAGES)
        self.assertFalse(ctx.exception.ambiguous)
        self.assertTrue(ctx.exception.retryable)

    def test_openai_invalid_counts_outputs_and_usage_are_explicit_errors(self):
        for count in (None, 0, -1, True, "10"):
            provider = create_provider(model_spec(), transport=FixtureTransport(count=count))
            with self.subTest(count=count), self.assertRaisesRegex(ProviderError, "invalid_token_count"):
                provider.count_input_tokens(MESSAGES)
        invalid = responses_fixture(output=[{"type": "message", "role": "assistant", "content": None}])
        mismatch = responses_fixture(usage={"input_tokens": 11})
        for raw in (invalid, mismatch):
            with self.assertRaises(ProviderError) as ctx:
                create_provider(model_spec(), transport=FixtureTransport(response=raw)).generate(MESSAGES, "r")
            self.assertTrue(ctx.exception.ambiguous)
            self.assertEqual(ctx.exception.raw_response, raw)

    def test_openai_output_limit_below_api_minimum_rejected(self):
        spec = model_spec()
        spec["sampling"]["max_output_tokens"] = 4
        with self.assertRaisesRegex(ProviderError, "minimum_is_16"):
            create_provider(spec)

    def test_reasoning_modes_are_explicit(self):
        for mode, effort in (("none", None), ("thinking", "high")):
            spec, transport = model_spec(), FixtureTransport()
            spec["reasoning"].update(mode=mode, effort=effort)
            create_provider(spec, transport=transport).generate(MESSAGES, "r")
            self.assertEqual(transport.calls[-1][2]["reasoning"],
                             {"effort": "none" if mode == "none" else "high"})
        spec = model_spec()
        spec["reasoning"]["enable_thinking"] = False
        with self.assertRaisesRegex(ProviderError, "unsupported_thinking"):
            create_provider(spec)

    def test_refusal_and_incomplete_are_not_guessed_as_answers(self):
        raw = responses_fixture(output=[{"type": "message", "role": "assistant", "content": [
            {"type": "refusal", "refusal": "NO unsupported request"}]}], usage=None)
        result = create_provider(model_spec(), transport=FixtureTransport(response=raw)).generate(MESSAGES, "r")
        self.assertIsNone(result["text"])
        self.assertEqual(result["finish_reason"], "refusal")
        self.assertTrue(all(v is None for v in result["usage"].values()))
        raw = responses_fixture(status="incomplete", incomplete_details={"reason": "max_output_tokens"})
        result = create_provider(model_spec(), transport=FixtureTransport(response=raw)).generate(MESSAGES, "r")
        self.assertTrue(result["truncated"])
        self.assertEqual(result["text"], "YES")

    def test_contradictory_multiple_messages_not_collapsed(self):
        raw = responses_fixture()
        raw["output"].append({"type": "message", "role": "assistant", "content": [
            {"type": "output_text", "text": "NO"}]})
        result = create_provider(model_spec(), transport=FixtureTransport(response=raw)).generate(MESSAGES, "r")
        self.assertEqual(result["text"], "YES\nNO")

    def test_rate_limit_safe_retry_and_quota_no_retry(self):
        for code, retryable in (("rate_limit_exceeded", True), ("insufficient_quota", False)):
            def transport(*args):
                return 429, {"error": {"code": code, "message": "fixture_secret"}}, {"Retry-After": "2"}
            provider = create_provider(model_spec(), transport=transport)
            with self.assertRaises(ProviderError) as ctx:
                provider.preflight()
            self.assertEqual(ctx.exception.retryable, retryable)
            self.assertFalse(ctx.exception.ambiguous)
            self.assertEqual(ctx.exception.retry_after_seconds, 2)
            self.assertNotIn("fixture_secret", str(ctx.exception))

    def test_server_error_after_post_is_ambiguous_and_not_retried(self):
        transport = FixtureTransport()
        provider = create_provider(model_spec(), transport=transport)
        provider.preflight()
        provider.count_input_tokens(MESSAGES)
        calls = []
        def fail(*args):
            calls.append(args)
            return 503, {"error": {"message": "fixture"}}, {}
        provider.transport = fail
        with self.assertRaises(ProviderError) as ctx:
            provider.generate(MESSAGES, "r")
        self.assertTrue(ctx.exception.ambiguous)
        self.assertTrue(ctx.exception.retryable)
        self.assertEqual(len(calls), 1)

    def test_transport_exception_does_not_leak_details(self):
        provider = create_provider(model_spec(), transport=lambda *args: (_ for _ in ()).throw(
            RuntimeError("fixture_secret")))
        with self.assertRaises(ProviderError) as ctx:
            provider.preflight()
        self.assertNotIn("fixture_secret", str(ctx.exception))
        self.assertFalse(ctx.exception.ambiguous)

    def test_missing_key_fails_before_transport(self):
        transport = FixtureTransport()
        with patch.dict(os.environ, {}, clear=True), self.assertRaisesRegex(ProviderError, "missing_api_key"):
            create_provider(model_spec(), transport=transport).preflight()
        self.assertFalse(transport.calls)

    def test_unsafe_endpoints_and_request_ids_rejected(self):
        for url in ("https://user:secret@provider.invalid/v1", "http://provider.invalid/v1",
                    "https://provider.invalid/v1?key=secret", "file:///tmp/model"):
            spec = model_spec()
            spec["serving"]["base_url"] = url
            with self.subTest(url=url), self.assertRaises(ProviderError):
                create_provider(spec)
        provider = create_provider(model_spec(), transport=FixtureTransport())
        with self.assertRaisesRegex(ProviderError, "invalid_request_id"):
            provider.generate(MESSAGES, "injected\r\nheader")
        with self.assertRaisesRegex(ProviderError, "invalid_messages"):
            provider.generate([{"role": "tool", "content": "fixture"}], "r")

    def test_vllm_pins_server_and_template_then_tokenizes_without_generation(self):
        spec, transport = model_spec("vllm"), FixtureTransport(response=chat_fixture())
        provider = create_provider(spec, transport=transport)
        runtime = provider.preflight()
        self.assertEqual([c[0] for c in transport.calls], ["GET", "GET", "GET"])
        self.assertEqual(provider.count_input_tokens(MESSAGES), 10)
        self.assertTrue(transport.calls[-1][1].endswith("/tokenize"))
        self.assertNotIn("/v1/tokenize", transport.calls[-1][1])
        self.assertIsNone(runtime["resolved_revision"])
        self.assertIn("declared", runtime["revision_verification"])
        self.assertEqual(runtime["chat_template_sha256"], spec["tokenizer"]["chat_template_sha256"])

    def test_vllm_exact_usage_seed_and_thinking_payload(self):
        spec, transport = model_spec("vllm"), FixtureTransport(response=chat_fixture())
        spec["seed"] = 42
        spec["reasoning"].update(mode="none", enable_thinking=False)
        result = create_provider(spec, transport=transport).generate(MESSAGES, "request-vllm")
        self.assertEqual(result["text"], "EVET")
        self.assertEqual(result["raw"]["choices"][0]["message"]["reasoning_content"], "fixture only")
        body = transport.calls[-1][2]
        self.assertEqual(body["seed"], 42)
        self.assertEqual(body["chat_template_kwargs"], {"enable_thinking": False})
        self.assertEqual(result["usage"]["input_tokens"], 10)
        self.assertIsNone(result["usage"]["reasoning_tokens"])

    def test_vllm_temporal_template_uses_frozen_date_for_count_and_generation(self):
        temporal = "{% if date_string is not defined %}{{ strftime_now('%Y-%m-%d') }}{% else %}{{ date_string }}{% endif %}"
        spec, transport = model_spec("vllm"), FixtureTransport(response=chat_fixture(), template=temporal)
        spec["tokenizer"]["chat_template_sha256"] = hashlib.sha256(temporal.encode()).hexdigest()
        result = create_provider(spec, transport=transport).generate(MESSAGES, "r")
        for call in transport.calls:
            if call[1].endswith("/tokenize") or call[1].endswith("/chat/completions"):
                self.assertEqual(call[2]["chat_template_kwargs"], {"date_string": "2026-10-24"})
        self.assertEqual(result["runtime"]["chat_template_options"], {"date_string": "2026-10-24"})

    def test_vllm_mismatched_server_or_template_rejected(self):
        for transport in (FixtureTransport(version="wrong"), FixtureTransport(template="wrong")):
            with self.assertRaises(ProviderError):
                create_provider(model_spec("vllm"), transport=transport).preflight()
        spec = model_spec("vllm")
        spec["serving"]["server_revision"] = None
        with self.assertRaisesRegex(ProviderError, "missing_server_revision"):
            create_provider(spec)

    def test_vllm_overlong_inputs_fail_before_generation(self):
        transport = FixtureTransport(response=chat_fixture(), count=1000)
        with self.assertRaisesRegex(ProviderError, "context_limit_exceeded"):
            create_provider(model_spec("vllm"), transport=transport).generate(MESSAGES, "r")
        self.assertFalse(any(c[1].endswith("/chat/completions") for c in transport.calls))

    def test_vllm_rejects_usage_disagreement_and_multiple_choices(self):
        for raw in (chat_fixture(usage={"prompt_tokens": 9}), chat_fixture(choices=[])):
            with self.assertRaises(ProviderError) as ctx:
                create_provider(model_spec("vllm"), transport=FixtureTransport(response=raw)).generate(MESSAGES, "r")
            self.assertTrue(ctx.exception.ambiguous)
            self.assertEqual(ctx.exception.raw_response, raw)

    def test_http_transport_reads_json_once_without_hidden_retry(self):
        class Response:
            status = 200
            headers = {}
            def __enter__(self): return self
            def __exit__(self, *args): return False
            def read(self): return b'{"fixture":true}'
        opener = types.SimpleNamespace(open=lambda *args, **kwargs: Response())
        with patch("inference.providers.build_opener", return_value=opener) as opened:
            self.assertEqual(_http_transport("GET", "https://provider.invalid", None, {}, 5),
                             (200, {"fixture": True}, {}))
            self.assertEqual(opened.call_count, 1)
        opener.open = lambda *args, **kwargs: (_ for _ in ()).throw(URLError("fixture_secret"))
        with patch("inference.providers.build_opener", return_value=opener), self.assertRaises(ProviderError) as ctx:
            _http_transport("POST", "https://provider.invalid", {}, {}, 5)
        self.assertTrue(ctx.exception.ambiguous)
        self.assertNotIn("fixture_secret", str(ctx.exception))

    def test_retry_after_nonfinite_and_invalid_ignored(self):
        for value in ("nan", "inf", "invalid"):
            self.assertIsNone(_retry_after({"retry-after": value}))
        self.assertEqual(_retry_after({"Retry-After": "-1"}), 0)


class FixtureTensor:
    def __init__(self, values): self.values = values
    def __getitem__(self, item):
        if isinstance(item, int):
            return FixtureTensor(self.values[item])
        row, indices = item
        return FixtureTensor(self.values[row][indices])
    def tolist(self): return self.values


class FixtureBatch(dict):
    def to(self, device): return self


class FixtureTokenizer:
    init_kwargs = {"_commit_hash": REVISION}
    def __init__(self): self.calls = []; self.encoding_calls = []; self.template = TEMPLATE
    def get_chat_template(self): return self.template
    def apply_chat_template(self, messages, **kwargs):
        self.calls.append((deepcopy(messages), dict(kwargs)))
        return "<fixture>" + json.dumps(messages) + json.dumps(kwargs, sort_keys=True)
    def __call__(self, rendered, **kwargs):
        self.encoding_calls.append((rendered, dict(kwargs)))
        return FixtureBatch(input_ids=FixtureTensor([[10, 11, 12]])) if kwargs.get("return_tensors") else {"input_ids": [10, 11, 12]}
    def decode(self, ids, *, skip_special_tokens):
        return "EVET" if skip_special_tokens else "EVET<eos>"


class FixtureModel:
    config = types.SimpleNamespace(_commit_hash=REVISION)
    device = "cpu"
    def __init__(self):
        self.calls = []
        self.generation_config = types.SimpleNamespace(eos_token_id=0, to_dict=lambda: {"eos_token_id": 0})
        self.generated = [21, 0]
    def to(self, device): self.device = device; return self
    def eval(self): return self
    def generate(self, **kwargs):
        self.calls.append(kwargs)
        return types.SimpleNamespace(sequences=FixtureTensor([[10, 11, 12] + self.generated]))


class TransformersProviderTests(unittest.TestCase):
    def setUp(self):
        self.tokenizer, self.model = FixtureTokenizer(), FixtureModel()
        self.model_calls, self.tokenizer_calls, self.seeds = [], [], []
        def tokenizer_load(model_id, **kwargs):
            self.tokenizer_calls.append((model_id, kwargs)); return self.tokenizer
        def model_load(model_id, **kwargs):
            self.model_calls.append((model_id, kwargs)); return self.model
        self.config = types.SimpleNamespace(_commit_hash=REVISION, max_position_embeddings=1024,
                                            is_encoder_decoder=False)
        transformers = types.SimpleNamespace(
            __version__="fixture-transformers", AutoConfig=types.SimpleNamespace(from_pretrained=lambda *a, **k: self.config),
            AutoTokenizer=types.SimpleNamespace(from_pretrained=tokenizer_load),
            AutoModelForCausalLM=types.SimpleNamespace(from_pretrained=model_load))
        torch = types.SimpleNamespace(
            __version__="fixture-torch", float32="fixture-float32",
            cuda=types.SimpleNamespace(is_available=lambda: False),
            random=types.SimpleNamespace(fork_rng=lambda **kw: nullcontext()),
            inference_mode=nullcontext, manual_seed=self.seeds.append)
        self.modules = {"transformers": transformers, "torch": torch}
        self.imports = patch("inference.providers.importlib.import_module", side_effect=self.modules.__getitem__)
        self.imports.start()
        self.addCleanup(self.imports.stop)

    def test_preflight_pins_revisions_template_and_does_not_load_weights(self):
        provider = create_provider(model_spec("transformers"))
        runtime = provider.preflight()
        self.assertEqual(self.model_calls, [])
        self.assertEqual(self.tokenizer_calls[0][1]["revision"], REVISION)
        self.assertTrue(self.tokenizer_calls[0][1]["local_files_only"])
        self.assertFalse(runtime["weights_loaded"])
        self.assertEqual(provider.count_input_tokens(MESSAGES), 3)
        self.assertFalse(self.tokenizer.encoding_calls[-1][1]["truncation"])
        self.assertFalse(self.tokenizer.encoding_calls[-1][1]["add_special_tokens"])
        self.assertEqual(provider.runtime["last_input"]["input_token_ids"], [10, 11, 12])

    def test_generate_decodes_new_tokens_only_and_retains_raw_ids(self):
        spec = model_spec("transformers")
        spec["seed"] = 9
        result = create_provider(spec).generate(MESSAGES, "hf-fixture")
        self.assertEqual(result["text"], "EVET")
        self.assertEqual(result["raw"], {"generated_token_ids": [21, 0],
                                       "decoded_text": "EVET", "decoded_with_special_tokens": "EVET<eos>"})
        self.assertEqual(result["usage"]["input_tokens"], 3)
        self.assertEqual(result["usage"]["output_tokens"], 2)
        self.assertEqual(result["usage"]["total_tokens"], 5)
        self.assertIsNone(result["usage"]["reasoning_tokens"])
        self.assertFalse(result["truncated"])
        self.assertEqual(result["finish_reason"], "eos")
        self.assertEqual(self.seeds, [9])
        self.assertFalse(self.model.calls[0]["do_sample"])
        self.assertEqual(self.model_calls[0][1]["revision"], REVISION)
        self.assertEqual(self.model_calls[0][1]["dtype"], "fixture-float32")

    def test_sampling_and_thinking_are_forwarded_explicitly(self):
        spec = model_spec("transformers")
        spec["sampling"].update(temperature=0.6, top_p=0.9)
        spec["reasoning"].update(mode="thinking", enable_thinking=True)
        create_provider(spec).generate(MESSAGES, "r")
        self.assertEqual(self.model.calls[0]["temperature"], 0.6)
        self.assertEqual(self.model.calls[0]["top_p"], 0.9)
        self.assertTrue(self.model.calls[0]["do_sample"])
        self.assertTrue(self.tokenizer.calls[-1][1]["enable_thinking"])

    def test_unpinned_revisions_and_greedy_nucleus_conflict_rejected(self):
        spec = model_spec("transformers")
        spec["revision"] = "main"
        with self.assertRaisesRegex(ProviderError, "immutable_huggingface"):
            create_provider(spec)
        spec = model_spec("transformers")
        spec["sampling"]["top_p"] = 0.9
        with self.assertRaisesRegex(ProviderError, "greedy_decoding"):
            create_provider(spec)

    def test_wrong_template_or_commit_rejected_before_weights(self):
        self.tokenizer.template = "changed"
        with self.assertRaisesRegex(ProviderError, "chat_template_mismatch"):
            create_provider(model_spec("transformers")).preflight()
        self.tokenizer.template = TEMPLATE
        self.config._commit_hash = "b" * 40
        with self.assertRaisesRegex(ProviderError, "model_revision_mismatch"):
            create_provider(model_spec("transformers")).preflight()
        self.assertEqual(self.model_calls, [])

    def test_thinking_flag_must_be_used_by_verified_template(self):
        self.tokenizer.template = "{{ messages }}"
        spec = model_spec("transformers")
        spec["tokenizer"]["chat_template_sha256"] = hashlib.sha256(self.tokenizer.template.encode()).hexdigest()
        spec["reasoning"].update(mode="none", enable_thinking=False)
        with self.assertRaisesRegex(ProviderError, "unsupported_by_template"):
            create_provider(spec).preflight()

    def test_temporal_template_requires_pinned_date_and_rejects_implicit_clock(self):
        for template, frozen_date, code in (
                ("{{ date_string }}", None, "frozen_chat_template_date_required"),
                ("{{ date_string }}", "2026-99-99", "frozen_chat_template_date_required"),
                ("{{ strftime_now('%Y') }}", "2026-10-24", "implicit_template_clock_unsupported")):
            self.tokenizer.template = template
            spec = model_spec("transformers")
            spec["serving"]["chat_template_date"] = frozen_date
            spec["tokenizer"]["chat_template_sha256"] = hashlib.sha256(template.encode()).hexdigest()
            with self.subTest(template=template), self.assertRaisesRegex(ProviderError, code):
                create_provider(spec).preflight()

    def test_rendered_input_and_ids_are_frozen_before_tensor_generation(self):
        self.tokenizer.template = "{{ date_string }} {{ messages }}"
        spec = model_spec("transformers")
        spec["tokenizer"]["chat_template_sha256"] = hashlib.sha256(self.tokenizer.template.encode()).hexdigest()
        provider = create_provider(spec)
        self.assertEqual(provider.count_input_tokens(MESSAGES), 3)
        rendered = provider.runtime["last_input"]["rendered_text"]
        result = provider.generate(MESSAGES, "r")
        self.assertEqual(len(self.tokenizer.calls), 1)
        self.assertEqual(self.tokenizer.calls[0][1]["date_string"], "2026-10-24")
        self.assertEqual(result["runtime"]["last_input"]["rendered_text"], rendered)
        self.assertEqual(result["runtime"]["last_input"]["rendered_text_sha256"], hashlib.sha256(rendered.encode()).hexdigest())
        self.assertEqual(result["runtime"]["last_input"]["input_token_ids"], [10, 11, 12])

    def test_changed_tensor_encoding_rejected_before_model_load(self):
        tokenizer = self.tokenizer
        class ChangedTokenizer(FixtureTokenizer):
            def __call__(self, rendered, **kwargs):
                if kwargs.get("return_tensors"):
                    return FixtureBatch(input_ids=FixtureTensor([[99]]))
                return super().__call__(rendered, **kwargs)
        self.tokenizer = ChangedTokenizer()
        with self.assertRaisesRegex(ProviderError, "chat_tokenization_changed"):
            create_provider(model_spec("transformers")).generate(MESSAGES, "r")
        self.assertEqual(self.model_calls, [])

    def test_context_check_precedes_weights_and_never_truncates(self):
        spec = model_spec("transformers")
        spec["limits"]["context_tokens"] = 34
        with self.assertRaisesRegex(ProviderError, "context_limit_exceeded"):
            create_provider(spec).generate(MESSAGES, "r")
        self.assertEqual(self.model_calls, [])

    def test_length_and_eos_are_distinguished_at_output_limit(self):
        spec = model_spec("transformers")
        spec["sampling"]["max_output_tokens"] = 2
        self.model.generated = [21, 22]
        result = create_provider(spec).generate(MESSAGES, "r")
        self.assertTrue(result["truncated"])
        self.model.generated = [21, 0]
        result = create_provider(spec).generate(MESSAGES, "r")
        self.assertFalse(result["truncated"])
        self.assertEqual(result["finish_reason"], "eos")

    def test_missing_optional_dependencies_no_fallback(self):
        with patch.object(providers_module.importlib, "import_module", side_effect=ImportError("fixture_secret")):
            with self.assertRaisesRegex(ProviderError, "missing_transformers_dependencies") as ctx:
                create_provider(model_spec("transformers")).preflight()
        self.assertNotIn("fixture_secret", str(ctx.exception))


if __name__ == "__main__":
    unittest.main()

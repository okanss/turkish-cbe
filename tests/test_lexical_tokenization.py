"""Exact offline tokenization and explicitly synthetic tokenizer fixtures."""

from copy import deepcopy
import hashlib
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

from analysis.lexical_bias.catalog import build_catalog
from analysis.lexical_bias.tokenization import measure_prompts, measure_vocabulary
from complete_grid.model_tokens import ModelTokenizers
from inference.config import load_models
from inference.prompts import make_messages


ROOT = Path(__file__).resolve().parents[1]
TEMPLATE = "{{ date_string }} {{ enable_thinking }} controlled synthetic template"


class FixtureTokenizer:
    def __init__(self, revision, template=TEMPLATE):
        self.chat_template = template
        self.init_kwargs = {"_commit_hash": revision}
        self.plain_calls = []
        self.chat_calls = []

    def encode(self, text, *, add_special_tokens):
        self.plain_calls.append((text, add_special_tokens))
        return list(range(len(text)))

    def apply_chat_template(self, messages, **options):
        self.chat_calls.append((deepcopy(messages), options))
        return list(range(sum(len(row["content"]) for row in messages) + 7))


class LexicalTokenizationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.models = load_models(ROOT / "configs/models.yaml")["models"]
        # These tests exercise measurement, never execution gating; a recorded
        # live enablement must not leak stricter validation into the fixtures.
        for spec in cls.models:
            spec["enabled"] = False
        cls.openai = next(row for row in cls.models if row["provider"] == "openai")
        cls.hf = next(row for row in cls.models if row["provider"] == "transformers")
        cls.vllm = next(row for row in cls.models if row["provider"] == "vllm")
        cls.catalog = build_catalog()
        cls.source_rows = [json.loads(line) for line in (ROOT / "data/complete_grid/pilot_preview.jsonl").read_text().splitlines()]
        cls.frozen = ModelTokenizers(ROOT / "tokenizers")

    def fixture_spec(self, provider="transformers"):
        spec = deepcopy(self.hf if provider == "transformers" else self.vllm)
        spec["tokenizer"]["chat_template_sha256"] = hashlib.sha256(TEMPLATE.encode()).hexdigest()
        spec["reasoning"].update(mode="none", enable_thinking=False)
        return spec

    def canonical_pair(self, name_control="native_shared"):
        return [deepcopy(row) for row in self.source_rows if row["relation_id"] == self.catalog[0]["relation_id"]
                and row["condition"] == "canonical" and row["name_control"] == name_control]

    def write_assets(self, root, spec):
        cache = Path(root) / "tokenizers/hf"
        parent = Path("models--" + spec["tokenizer"]["id"].replace("/", "--")) / "snapshots" / spec["tokenizer"]["revision"]
        files = []
        for name in ("tokenizer.json", "tokenizer_config.json", "vocab.json", "merges.txt", "LICENSE"):
            data = json.dumps({"chat_template": TEMPLATE}).encode() if name == "tokenizer_config.json" else b"synthetic test assets; never loaded"
            path = cache / parent / name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(data)
            files.append({"path": str(parent / name), "sha256": hashlib.sha256(data).hexdigest(),
                "bytes": len(data), "git_blob_id": hashlib.sha1(f"blob {len(data)}\0".encode() + data).hexdigest(),
                "upstream_lfs_sha256": None,
                "source_url": f"https://huggingface.co/{spec['tokenizer']['id']}/resolve/{spec['tokenizer']['revision']}/{name}"})
        manifest = {"schema_version": 1, "contains_model_weights": False,
                    "models": {spec["tokenizer"]["id"]: {"revision": spec["tokenizer"]["revision"],
                         "chat_template_sha256": spec["tokenizer"]["chat_template_sha256"], "files": files}}}
        (cache / "manifest.json").write_text(json.dumps(manifest))
        return cache, manifest

    def test_real_bpe_catalog_counts_are_model_bound_and_plaintext_only(self):
        rows = measure_vocabulary(ROOT, [self.openai], self.catalog)
        self.assertEqual(len(rows), 36)
        self.assertEqual({row["language"] for row in rows}, {"en", "tr"})
        self.assertEqual({row["direction"] for row in rows}, {"before", "after"})
        for row in rows:
            self.assertEqual(row["term_tokens"], self.frozen.measure(row["term"])["o200k_base"])
            self.assertEqual(row["expression_tokens"], self.frozen.measure(row["expression"])["o200k_base"])
            self.assertEqual(row["tokenizer_status"], "measured_frozen_bpe")
            self.assertFalse(row["fixture_only"])
            self.assertIsNone(row["chat_token_count"])
            self.assertFalse(row["network_used"])
            self.assertFalse(row["weights_loaded"])
            self.assertEqual(row["expression_sha256"], hashlib.sha256(row["expression"].encode()).hexdigest())

    def test_direction_deltas_and_language_ratios_use_the_same_backend(self):
        rows = measure_vocabulary(ROOT, [self.openai], self.catalog[:1])
        coordinate = {(row["language"], row["direction"]): row for row in rows}
        for row in rows:
            before, after = (coordinate[row["language"], direction] for direction in ("before", "after"))
            self.assertEqual(row["term_direction_delta_after_minus_before"], after["term_tokens"] - before["term_tokens"])
            tr, en = (coordinate[language, row["direction"]] for language in ("tr", "en"))
            self.assertEqual(row["term_tr_en_ratio"], tr["term_tokens"] / en["term_tokens"])

    def test_hf_and_vllm_missing_dependencies_never_fall_back_to_bpe(self):
        with patch.dict("sys.modules", {"transformers": None}), patch("analysis.lexical_bias.tokenization.ModelTokenizers", side_effect=AssertionError("forbidden BPE fallback")):
            rows = measure_vocabulary(ROOT, [self.hf, self.vllm], self.catalog[:1])
        self.assertEqual(len(rows), 8)
        self.assertEqual({row["tokenizer_status"] for row in rows}, {"missing_dependency"})
        self.assertTrue(all(row["term_tokens"] is None and row["term_tr_en_ratio"] is None for row in rows))

    def test_local_loader_is_tokenizer_only_offline_and_pinned(self):
        spec = self.fixture_spec()
        loader = Mock(side_effect=OSError("not cached"))
        module = SimpleNamespace(AutoTokenizer=SimpleNamespace(from_pretrained=loader))
        with tempfile.TemporaryDirectory() as folder, patch.dict("sys.modules", {"transformers": module}):
            cache, _ = self.write_assets(folder, spec)
            rows = measure_vocabulary(folder, [spec], self.catalog[:1])
            loader.assert_called_once_with(spec["tokenizer"]["id"], revision=spec["tokenizer"]["revision"],
                                           cache_dir=str(cache.resolve()), local_files_only=True, trust_remote_code=False)
        self.assertEqual({row["tokenizer_status"] for row in rows}, {"missing_or_unusable_local_assets"})

    def test_real_loader_rejects_missing_changed_or_unsafe_assets_before_loading(self):
        spec = self.fixture_spec()
        cases = ("missing_manifest", "changed_bytes", "unsafe_path", "wrong_revision", "upstream_hash", "missing_license")
        for case in cases:
            with self.subTest(case=case), tempfile.TemporaryDirectory() as folder:
                cache, manifest = self.write_assets(folder, spec)
                manifest_path = cache / "manifest.json"
                entry = manifest["models"][spec["tokenizer"]["id"]]
                if case == "missing_manifest":
                    manifest_path.unlink()
                elif case == "changed_bytes":
                    (cache / entry["files"][0]["path"]).write_bytes(b"corrupt")
                elif case == "unsafe_path":
                    entry["files"][0]["path"] = "../../outside/tokenizer.json"
                elif case == "wrong_revision":
                    entry["revision"] = "b" * 40
                elif case == "upstream_hash":
                    entry["files"][0]["git_blob_id"] = "0" * 40
                elif case == "missing_license":
                    entry["files"] = [row for row in entry["files"] if not row["path"].endswith("/LICENSE")]
                if case != "missing_manifest":
                    manifest_path.write_text(json.dumps(manifest))
                loader = Mock(side_effect=AssertionError("unverified assets must not load"))
                module = SimpleNamespace(AutoTokenizer=SimpleNamespace(from_pretrained=loader))
                with patch.dict("sys.modules", {"transformers": module}):
                    rows = measure_vocabulary(folder, [spec], self.catalog[:1])
                loader.assert_not_called()
                self.assertTrue(all(row["term_tokens"] is None for row in rows))
                self.assertTrue(all(row["tokenizer_status"] != "missing_or_unusable_local_assets" for row in rows))

    def test_openai_wrong_mapping_or_revision_is_rejected_before_loading(self):
        for key, value in (("encoding", "cl100k_base"), ("revision", "tiktoken==0.11.0"), ("id", "another-tokenizer")):
            spec = deepcopy(self.openai)
            spec["tokenizer"][key] = value
            factory = Mock()
            rows = measure_vocabulary(ROOT, [spec], self.catalog[:1], tokenizer_factory=factory)
            factory.assert_not_called()
            self.assertEqual({row["tokenizer_status"] for row in rows}, {"model_tokenizer_binding_mismatch"})
            self.assertTrue(all(row["term_tokens"] is None for row in rows))

    def test_unknown_openai_snapshot_never_uses_a_known_model_mapping(self):
        spec = deepcopy(self.openai)
        spec.update(model_id="gpt-4o-2026-01-01", revision="gpt-4o-2026-01-01")
        with patch("analysis.lexical_bias.tokenization.ModelTokenizers", side_effect=AssertionError("no substitution")):
            rows = measure_vocabulary(ROOT, [spec], self.catalog[:1])
        self.assertTrue(all(row["term_tokens"] is None for row in rows))

    def test_hf_unpinned_revision_or_local_path_is_not_silently_accepted(self):
        for key, value in (("revision", None), ("id", "/tmp/arbitrary-local-tokenizer")):
            spec = deepcopy(self.hf)
            spec["tokenizer"][key] = value
            factory = Mock()
            rows = measure_vocabulary(ROOT, [spec], self.catalog[:1], tokenizer_factory=factory)
            factory.assert_not_called()
            self.assertTrue(all(row["term_tokens"] is None for row in rows))

    def test_fixture_counts_are_explicit_and_unicode_text_is_unchanged(self):
        spec = self.fixture_spec()
        tokenizer = FixtureTokenizer(spec["tokenizer"]["revision"])
        catalog = deepcopy(self.catalog[:1])
        catalog[0]["directions"][0]["term"] = "o\u0308nündedir"
        rows = measure_vocabulary(ROOT, [spec], catalog, tokenizer_factory=lambda _: tokenizer)
        self.assertTrue(all(row["fixture_only"] and row["tokenizer_status"] == "fixture_only" for row in rows))
        self.assertIn(("o\u0308nündedir", False), tokenizer.plain_calls)
        self.assertEqual(next(row for row in rows if row["language"] == "tr" and row["direction"] == "before")["term_tokens"], len("o\u0308nündedir"))

    def test_wrong_chat_template_or_loaded_revision_rejects_tokenizer(self):
        spec = self.fixture_spec()
        for tokenizer, expected in ((FixtureTokenizer(spec["tokenizer"]["revision"], "wrong"), "chat_template_mismatch"),
                                     (FixtureTokenizer("b" * 40), "tokenizer_revision_mismatch")):
            rows = measure_vocabulary(ROOT, [spec], self.catalog[:1], tokenizer_factory=lambda _, obj=tokenizer: obj)
            self.assertEqual({row["tokenizer_status"] for row in rows}, {expected})
            self.assertTrue(all(row["term_tokens"] is None for row in rows))

    def test_missing_template_pin_preserves_plaintext_but_never_invents_chat_count(self):
        spec = self.fixture_spec()
        spec["tokenizer"]["chat_template_sha256"] = None
        tokenizer = FixtureTokenizer(spec["tokenizer"]["revision"])
        rows = measure_prompts(ROOT, [spec], self.canonical_pair(), tokenizer_factory=lambda _: tokenizer)
        self.assertTrue(all(row["source_prompt_plain_tokens"] is not None for row in rows))
        self.assertEqual({row["chat_count_status"] for row in rows}, {"missing_chat_template_pin"})
        self.assertEqual(tokenizer.chat_calls, [])

    def test_verified_chat_uses_local_template_and_fixed_reasoning_date(self):
        spec = self.fixture_spec()
        tokenizer = FixtureTokenizer(spec["tokenizer"]["revision"])
        sources = self.canonical_pair()
        rows = measure_prompts(ROOT, [spec], sources, tokenizer_factory=lambda _: tokenizer)
        self.assertEqual(len(tokenizer.chat_calls), 2)
        for (messages, options), source, measured in zip(tokenizer.chat_calls, sources, rows):
            self.assertEqual(messages, make_messages(source)["messages"])
            self.assertEqual(options, {"tokenize": True, "add_generation_prompt": True,
                                      "date_string": spec["serving"]["chat_template_date"], "enable_thinking": False})
            joined = "\n\n".join(message["content"] for message in messages)
            self.assertEqual(measured["messages_content_plain_tokens"], len(joined))
            self.assertEqual(measured["chat_token_count"], sum(len(message["content"]) for message in messages) + 7)
            self.assertEqual(measured["messages_content_sha256"], hashlib.sha256(joined.encode()).hexdigest())
            self.assertTrue(measured["fixture_only"])

    def test_api_plaintext_counts_do_not_claim_chat_framing(self):
        tokenizer = FixtureTokenizer(None)
        rows = measure_prompts(ROOT, [self.openai], self.canonical_pair(), tokenizer_factory=lambda _: tokenizer)
        self.assertEqual(tokenizer.chat_calls, [])
        self.assertTrue(all(row["chat_token_count"] is None for row in rows))
        self.assertEqual({row["chat_count_status"] for row in rows}, {"unavailable_without_provider_chat_count"})

    def test_real_preview_counts_match_frozen_source_and_keep_new_chat_distinct(self):
        rows = measure_prompts(ROOT, [self.openai], self.source_rows)
        self.assertEqual(len(rows), 243)
        sources = {row["prompt_id"]: row for row in self.source_rows}
        for row in rows:
            source = sources[row["prompt_id"]]
            self.assertEqual(row["source_prompt_plain_tokens"], source["model_token_counts"]["o200k_base"])
            content = "\n\n".join(message["content"] for message in make_messages(source)["messages"])
            self.assertEqual(row["messages_content_plain_tokens"], self.frozen.measure(content)["o200k_base"])
            self.assertIsNone(row["chat_token_count"])
            self.assertEqual(row["grid_query_id"], source["grid_query_id"])
            self.assertEqual(row["grid_design_id"], source["grid_design_id"])

    def test_language_ratios_require_identical_controls_and_conditions(self):
        rows = measure_prompts(ROOT, [self.openai], self.source_rows)
        pairs = {}
        for row in rows:
            pairs.setdefault((row["relation_id"], row["condition"], row["name_control"]), {})[row["language"]] = row
        for pair in pairs.values():
            if set(pair) == {"tr", "en"}:
                self.assertEqual(pair["tr"]["source_prompt_tr_en_ratio"], pair["tr"]["source_prompt_plain_tokens"] / pair["en"]["source_prompt_plain_tokens"])
            else:
                row = next(iter(pair.values()))
                self.assertEqual(row["language_pair_status"], "missing_counterpart")
                self.assertIsNone(row["source_prompt_tr_en_ratio"])

    def test_transliteration_delta_uses_english_original_reference(self):
        rows = measure_prompts(ROOT, [self.openai], self.source_rows)
        lookup = {(row["relation_id"], row["condition"], row["name_control"], row["language"]): row for row in rows}
        for row in rows:
            reference = lookup[row["relation_id"], row["condition"], "native_shared", row["language"]]
            self.assertEqual(row["name_control_reference_prompt_id"], reference["prompt_id"])
            self.assertEqual(row["source_prompt_name_delta_from_native_shared"], row["source_prompt_plain_tokens"] - reference["source_prompt_plain_tokens"])
            if row["name_control"] == "en_transliterated" and row["language"] == "tr":
                self.assertEqual(row["source_prompt_name_delta_from_native_shared"], 0)

    def test_duplicate_or_corrupt_source_coordinates_are_rejected(self):
        pair = self.canonical_pair()
        for rows in (pair + [deepcopy(pair[0])], [{**pair[0], "prompt_sha256": "0" * 64}]):
            with self.assertRaises(ValueError):
                measure_prompts(ROOT, [self.openai], rows)

    def test_cross_language_gold_change_and_frozen_count_mismatch_are_rejected(self):
        pair = self.canonical_pair()
        pair[0]["gold_label"] = not pair[0]["gold_label"]
        with self.assertRaisesRegex(ValueError, "gold"):
            measure_prompts(ROOT, [self.openai], pair)
        pair = self.canonical_pair()
        pair[0]["model_token_counts"]["o200k_base"] += 1
        with self.assertRaisesRegex(ValueError, "frozen"):
            measure_prompts(ROOT, [self.openai], pair)

    def test_count_errors_produce_nulls_and_never_substitute_a_backend(self):
        spec = self.fixture_spec()
        tokenizer = FixtureTokenizer(spec["tokenizer"]["revision"])
        tokenizer.encode = Mock(return_value=[[1, 2]])
        rows = measure_vocabulary(ROOT, [spec], self.catalog[:1], tokenizer_factory=lambda _: tokenizer)
        self.assertTrue(all(row["term_tokens"] is None and row["term_count_status"] == "tokenizer_count_failed" for row in rows))

    def test_input_schemas_and_duplicate_ids_are_validated(self):
        for models, catalog in (([self.openai, self.openai], self.catalog), ("all", self.catalog),
                                ([self.openai], self.catalog + [self.catalog[0]]),
                                ([self.openai], [{**self.catalog[0], "directions": []}])):
            with self.assertRaises(ValueError):
                measure_vocabulary(ROOT, models, catalog)
        with self.assertRaises(ValueError):
            measure_prompts(ROOT, [self.openai], [{}])


if __name__ == "__main__":
    unittest.main()

"""Exact offline model lengths and safe shared-prefix optimization checks."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import random
import shutil
import tempfile
import unittest
from unittest.mock import patch

from complete_grid.model_tokens import (
    MODEL_TO_ENCODING, TIKTOKEN_VERSION, ModelTokenizationError, ModelTokenizers,
)


RESOURCE_DIR = Path(__file__).resolve().parents[1] / "tokenizers"


class ModelTokenizerDescriptorTests(unittest.TestCase):
    def test_missing_bundle_rejected_before_optional_dependency_import(self):
        with tempfile.TemporaryDirectory() as temp:
            with self.assertRaisesRegex(ModelTokenizationError, "encodings.json"):
                ModelTokenizers(Path(temp))

    def test_descriptor_rejects_unknown_schema_and_model_mapping(self):
        base = json.loads((RESOURCE_DIR / "encodings.json").read_text())
        changes = ({"schema_version": True}, {"schema_version": 2},
                   {"tiktoken_version": "0.0.0"},
                   {"model_to_encoding": {"unfrozen-model": "cl100k_base"}})
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "encodings.json"
            for change in changes:
                with self.subTest(change=change):
                    path.write_text(json.dumps({**base, **change}))
                    with self.assertRaisesRegex(ModelTokenizationError, "descriptor"):
                        ModelTokenizers(Path(temp))


@unittest.skipUnless(importlib.util.find_spec("tiktoken") is not None,
                     "optional pinned tiktoken dependency is not installed")
class RealModelTokenizerTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tokenizers = ModelTokenizers(RESOURCE_DIR)

    def test_known_ascii_fixture_and_empty_text(self):
        self.assertEqual(self.tokenizers.measure("hello world"),
                         {"cl100k_base": 2, "o200k_base": 2})
        self.assertEqual(self.tokenizers.measure(""),
                         {"cl100k_base": 0, "o200k_base": 0})
        self.assertEqual(self.tokenizers.measure("2 + 2 = 4"),
                         {"cl100k_base": 7, "o200k_base": 7})

    def test_turkish_counts_use_distinct_model_encodings(self):
        measured = self.tokenizers.measure("Ayşe, Bora'nın önündedir.")
        self.assertEqual(measured, {"cl100k_base": 13, "o200k_base": 9})

    def test_special_token_literals_are_ordinary_text(self):
        measured = self.tokenizers.measure("<|endoftext|>")
        self.assertGreater(measured["cl100k_base"], 1)
        self.assertGreater(measured["o200k_base"], 1)

    def test_no_normalization_or_replacement(self):
        self.assertNotEqual(self.tokenizers.measure("İrem"),
                            self.tokenizers.measure("I\u0307rem"))
        for invalid in ("\ud800", "A\udfffB"):
            with self.subTest(invalid=repr(invalid)), self.assertRaises(ValueError):
                self.tokenizers.measure(invalid)
        for invalid in (None, 17, b"hello", True):
            with self.subTest(invalid=invalid), self.assertRaises(TypeError):
                self.tokenizers.measure(invalid)

    def test_provenance_identifies_models_vocabs_and_plaintext_scope(self):
        descriptor = self.tokenizers.descriptor()
        self.assertEqual(descriptor["model_to_encoding"], MODEL_TO_ENCODING)
        self.assertEqual(descriptor["tiktoken_version"], TIKTOKEN_VERSION)
        self.assertTrue(descriptor["is_model_tokenizer"])
        self.assertFalse(descriptor["network_required"])
        self.assertIn("excludes chat framing", descriptor["count_scope"])
        self.assertEqual(len(descriptor["descriptor_sha256"]), 64)
        self.assertEqual(json.loads(json.dumps(descriptor)), descriptor)
        descriptor["model_to_encoding"].clear()
        self.assertEqual(self.tokenizers.descriptor()["model_to_encoding"], MODEL_TO_ENCODING)

    def test_local_constructor_does_not_use_download_loader(self):
        with patch("tiktoken_ext.openai_public.load_tiktoken_bpe",
                   side_effect=AssertionError("network-capable loader must not be called")), \
                patch("socket.socket", side_effect=AssertionError("network is forbidden")):
            loaded = ModelTokenizers(RESOURCE_DIR)
            self.assertEqual(loaded.measure("hello world")["o200k_base"], 2)

    def test_safe_question_boundaries_equal_full_encoding(self):
        before = self.tokenizers.cache_info()["safe_joins"]
        for marker in ("\nQuestion:", "\nSoru:"):
            prefix = "Prelude: Ayşe, Bora'nın önündedir.\n- E0001 E0002." + marker
            prepared = self.tokenizers.prepare_prefix(prefix)
            for suffix in (" Is Ayşe in front of Bora?\nProvide your answer only as YES or NO.",
                           " Ayşe, Bora'nın önünde midir?\nYanıtınızı yalnız YES veya NO olarak verin.",
                           " I\u0307rem, Özgür'den hızlı mıdır?\nYES veya NO."):
                with self.subTest(marker=marker, suffix=suffix):
                    expected = self.tokenizers.measure(prefix + suffix)
                    self.assertEqual(self.tokenizers.count_parts(prefix, suffix), expected)
                    self.assertEqual(self.tokenizers.count_parts(prepared, suffix), expected)
        self.assertGreater(self.tokenizers.cache_info()["safe_joins"], before)

    def test_unsafe_fragments_never_use_additive_counts(self):
        pairs = [("hel", "lo"), (" ", "hello"), ("can", "'t"), ("İ", "rem"),
                 ("A", "\u0307"), ("word\n", "  next"), ("1", "23"),
                 ("\nQuestion:", "\nHello"), ("\nQuestion:", "🙂"),
                 ("\nQuestion:", "  Hello"), ("\nSoru:", "")]
        self.assertNotEqual(self.tokenizers.measure("hel")["cl100k_base"]
                            + self.tokenizers.measure("lo")["cl100k_base"],
                            self.tokenizers.measure("hello")["cl100k_base"])
        before = self.tokenizers.cache_info()["full_fallbacks"]
        for prefix, suffix in pairs:
            with self.subTest(prefix=prefix, suffix=suffix):
                self.assertEqual(self.tokenizers.count_parts(prefix, suffix),
                                 self.tokenizers.measure(prefix + suffix))
        self.assertEqual(self.tokenizers.cache_info()["full_fallbacks"] - before, len(pairs))

    def test_random_multilingual_joins_equal_full_encoding(self):
        rng = random.Random(814)
        alphabet = "abcXYZ0123 çğışöüİ東京مرحبا\u0307\n\t'.,:!?🙂"
        for _ in range(100):
            text = "".join(rng.choice(alphabet) for _ in range(rng.randrange(80)))
            split = rng.randrange(len(text) + 1)
            self.assertEqual(self.tokenizers.count_parts(text[:split], text[split:]),
                             self.tokenizers.measure(text))

    def test_large_1024_entity_prefix_and_all_relation_query_forms(self):
        from renderers import english, turkish
        names = [f"E{i:04d}" for i in range(1024)]
        for language in ("en", "tr"):
            marker = "\nQuestion:" if language == "en" else "\nSoru:"
            prefix = ("\n".join(f"- {names[i]} precedes {names[i + 1]}."
                                 for i in range(1023)) + marker)
            prepared = self.tokenizers.prepare_prefix(prefix)
            for relation_id in english.RELATIONS:
                for direction in ("before", "after"):
                    question = (english.question(names[0], names[-1], relation_id, direction)
                                if language == "en" else turkish._question(
                                    names[0], names[-1], turkish.RELATIONS[relation_id], direction))
                    suffix = " " + question + "\nYES or NO."
                    self.assertEqual(self.tokenizers.count_parts(prepared, suffix),
                                     self.tokenizers.measure(prefix + suffix))

    def test_prepared_prefix_owner_and_bounded_cache(self):
        original = self.tokenizers.prepare_prefix("Prelude.\nQuestion:")
        for index in range(40):
            self.tokenizers.prepare_prefix(f"{index}.\nQuestion:")
        self.assertLessEqual(self.tokenizers.cache_info()["cached_prefixes"], 32)
        self.assertEqual(self.tokenizers.count_parts(original, " Is this true?"),
                         self.tokenizers.measure(original.text + " Is this true?"))
        other = ModelTokenizers(RESOURCE_DIR)
        with self.assertRaisesRegex(ModelTokenizationError, "different tokenizer"):
            other.count_parts(original, " Is this true?")

    def test_short_measurement_cache_is_detached_bounded_and_excludes_long_text(self):
        text="Cache fixture: Ayşe, Bora'nın önündedir."
        measured=self.tokenizers.measure(text)
        expected=dict(measured)
        measured["cl100k_base"]=-1
        self.assertEqual(self.tokenizers.measure(text),expected)
        for index in range(4200):
            self.tokenizers.measure(f"Short cache fixture {index}: Çığ İrem.")
        self.assertEqual(self.tokenizers.cache_info()["cached_short_texts"],4096)
        other=ModelTokenizers(RESOURCE_DIR)
        long_text="Long uncached fixture: Özgür ve İrem. "*100
        other.measure(long_text)
        self.assertEqual(other.cache_info()["cached_short_texts"],0)

    def test_fixed_marker_fast_path_matches_full_regex_without_prefix_walk(self):
        rng=random.Random(193)
        starts=["", "'", "\r", "\n\n", "?!/", "🙂\u0307", "I\u0307rem's",
                "Question:\nSoru:", "\u2003"*20]
        alphabet="abcXYZ0123 çğışöüİ東京مرحبا\u0307\n\t'.,:!?🙂"
        starts.extend("".join(rng.choice(alphabet) for _ in range(500)) for _ in range(15))
        patterns=self.tokenizers._patterns
        walks=[]
        class PatternWalkCounter:
            def __init__(self,pattern):self.pattern=pattern
            def finditer(self,text):
                walks.append(len(text))
                return self.pattern.finditer(text)
        wrappers={name:PatternWalkCounter(pattern) for name,pattern in patterns.items()}
        with patch.object(self.tokenizers,"_patterns",wrappers):
            for start in starts:
                for marker in ("\nQuestion:","\nSoru:"):
                    text=start+marker
                    prepared=self.tokenizers.prepare_prefix(text)
                    expected_last={name:list(pattern.finditer(text))[-1].group()
                                   for name,pattern in patterns.items()}
                    self.assertEqual(dict(prepared.last_pieces),expected_last)
                    full_counts={name:len(encoding.encode_ordinary(text)) for name,encoding
                                 in self.tokenizers._encodings.items()}
                    self.assertEqual(dict(prepared.counts),full_counts)
            self.assertEqual(walks,[])
            self.tokenizers.prepare_prefix("Fallback fixture without a question marker.")
            self.assertEqual(len(walks),2)

    def test_tampered_vocabulary_and_pattern_rejected(self):
        with tempfile.TemporaryDirectory() as temp:
            target = Path(temp)
            for path in RESOURCE_DIR.glob("*.tiktoken"):
                shutil.copyfile(path, target / path.name)
            descriptor = json.loads((RESOURCE_DIR / "encodings.json").read_text())
            descriptor_path = target / "encodings.json"
            descriptor_path.write_text(json.dumps(descriptor))
            vocab = target / "cl100k_base.tiktoken"
            vocab.write_bytes(vocab.read_bytes() + b"\n")
            with self.assertRaisesRegex(ModelTokenizationError, "hash mismatch"):
                ModelTokenizers(target)
            shutil.copyfile(RESOURCE_DIR / vocab.name, vocab)
            descriptor["encodings"]["cl100k_base"]["pat_str"] = ".+"
            descriptor_path.write_text(json.dumps(descriptor))
            with self.assertRaisesRegex(ModelTokenizationError, "encoding rules"):
                ModelTokenizers(target)


if __name__ == "__main__":
    unittest.main()

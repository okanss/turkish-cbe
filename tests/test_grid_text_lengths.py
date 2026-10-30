"""The optimized reference counter is exactly equivalent to its frozen source."""

import random
import unittest
from unittest.mock import patch

from complete_grid.text_lengths import fastmeasure_text
from data_generation.generator import GenerationPlan, generate_record
from experiments.dataset_builder import _tr_clause
from experiments.names import name_maps
from experiments.token_lengths import measure_text
from renderers import english, turkish


class GridTextLengthsTests(unittest.TestCase):
    def test_all_ascii_codepoints_and_ascii_whitespace_match(self):
        texts = [chr(i) for i in range(128)]
        texts += ["".join(chr(i) for i in range(128)), "", "\t\r\n\v\f\x1c\x1d\x1e\x1f "]
        for text in texts:
            with self.subTest(text=repr(text)):
                self.assertEqual(fastmeasure_text(text), measure_text(text))

    def test_one_thousand_seeded_turkish_ascii_and_punctuation_fixtures(self):
        rng = random.Random(4101)
        alphabet = "".join(chr(i) for i in range(128)) + "ÇĞİÖŞÜçğıöşü"
        for index in range(1000):
            text = "".join(rng.choices(alphabet, k=rng.randrange(512)))
            with self.subTest(index=index):
                self.assertEqual(fastmeasure_text(text), measure_text(text))

    def test_all_actual_relation_prompts_at_n4_and_n1024_match_entire_dictionary(self):
        for n in (4, 1024):
            source = generate_record(GenerationPlan(seed=52, sizes=(n,), examples_per_size=4,
                                     entity_types=("turkish_names",)), n, 0, "endpoint")
            for control, maps in name_maps(source).items():
                for relation in turkish.RELATIONS:
                    for language in ("en", "tr"):
                        aliases = maps[language]
                        names = [aliases[name] for name in sorted(source["latent_order"])]
                        a, b = aliases[source["query_subject"]], aliases[source["query_object"]]
                        styles = (("canonical",) if language == "en" else
                                  ("canonical", "syntax", "morphology_syntax_lexical"))
                        for style in styles:
                            premises = []
                            for index, (left, right) in enumerate(source["premise_edges"]):
                                left, right = aliases[left], aliases[right]
                                direction = "before"
                                if index % 2:
                                    left, right, direction = right, left, "after"
                                clause = (english.statement(left, right, relation, direction)
                                          if language == "en" else
                                          _tr_clause(left, right, relation, direction, style))
                                premises.append(clause)
                            if language == "en":
                                prompt = english.compose_prompt(english.context(relation,
                                    symbolic=control == "symbolic_shared"), names, premises,
                                    english.question(a, b, relation, source["query_relation"]))
                            else:
                                prompt = turkish.compose_prompt(turkish.expected_context(relation, names),
                                    names, premises, turkish._question(a, b,
                                    turkish.RELATIONS[relation], source["query_relation"]))
                            with self.subTest(n=n, control=control, relation=relation,
                                              language=language, style=style):
                                self.assertEqual(fastmeasure_text(prompt), measure_text(prompt))

    def test_generic_unicode_scripts_marks_digits_emoji_and_whitespace_use_fallback(self):
        texts = (
            "中文 العربية हिन्दी Ελληνικά", "Cafe\u0301", "Ayşe\u0301", "😀🚀👨‍👩‍👧‍👦",
            "A\u00a0B\u2003C\u2028D\u2029E\u0085F", "١٢٣ １２３ Ⅹ ²", "Türkçe Âdem",
            "\u200b\ufeff\u2060", "a\U0001e4d0b", "\U00010ffff", "\u00ad",
        )
        for text in texts:
            with self.subTest(text=text):
                with patch("complete_grid.text_lengths._original", wraps=measure_text) as original:
                    self.assertEqual(fastmeasure_text(text), measure_text(text))
                    original.assert_called_once_with(text)

    def test_supported_alphabet_uses_fast_path_and_preserves_text(self):
        text = "ÇĞİÖŞÜ çğıöşü ABC xyz 012  _\x00 ' : ?\n"
        expected = measure_text(text)
        with patch("complete_grid.text_lengths._original", side_effect=AssertionError("fallback")):
            self.assertEqual(fastmeasure_text(text), expected)
        self.assertEqual(text, "ÇĞİÖŞÜ çğıöşü ABC xyz 012  _\x00 ' : ?\n")

    def test_invalid_types_and_surrogates_preserve_original_exception_types(self):
        for value in (None, 1, True, b"text", [], {}, "\ud800", "\udfff"):
            with self.subTest(value=repr(value)):
                try:
                    measure_text(value)
                except Exception as error:
                    with self.assertRaises(type(error)):
                        fastmeasure_text(value)
                else:
                    self.fail("invalid fixture was accepted by original measurement")


if __name__ == "__main__":
    unittest.main()

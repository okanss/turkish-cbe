"""Reference token lengths must never masquerade as model token counts."""

import json
import unicodedata
import unittest

from experiments.token_lengths import TOKENIZATION_ID, TOKENIZER_SPEC, measure_text


class TokenLengthTests(unittest.TestCase):
    def test_turkish_punctuation_has_explicit_reference_boundaries(self):
        measured = measure_text("Ayşe, Bora'nın önündedir.")
        self.assertEqual(measured["unicode_codepoints"], 25)
        self.assertEqual(measured["utf8_bytes"], 29)
        self.assertEqual(measured["whitespace_token_count"], 3)
        self.assertEqual(measured["reference_token_count"], 7)
        self.assertTrue(measured["input_is_nfc"])

    def test_english_and_turkish_counts_are_independent_of_gold(self):
        english = measure_text("Ayşe is in front of Bora.")
        turkish = measure_text("Ayşe, Bora'nın önündedir.")
        self.assertEqual(english["reference_token_count"], 7)
        self.assertEqual(english["whitespace_token_count"], 6)
        self.assertEqual(turkish["reference_token_count"], 7)
        self.assertNotEqual(english["whitespace_token_count"], turkish["whitespace_token_count"])

    def test_nfc_is_reported_without_silent_normalization(self):
        composed = measure_text("İrem")
        decomposed = measure_text("I\u0307rem")
        self.assertEqual(composed["reference_token_count"], 1)
        self.assertEqual(decomposed["reference_token_count"], 1)
        self.assertEqual(composed["unicode_codepoints"], 4)
        self.assertEqual(decomposed["unicode_codepoints"], 5)
        self.assertEqual(composed["utf8_bytes"], 5)
        self.assertEqual(decomposed["utf8_bytes"], 6)
        self.assertTrue(composed["input_is_nfc"])
        self.assertFalse(decomposed["input_is_nfc"])

    def test_unicode_whitespace_and_line_breaks_preserve_input_length(self):
        measured = measure_text("A\u00a0B\u2003C\nD\tE\r\n")
        self.assertEqual(measured["whitespace_token_count"], 5)
        self.assertEqual(measured["reference_token_count"], 5)
        self.assertEqual(measured["unicode_codepoints"], 11)
        self.assertEqual(measured["utf8_bytes"], 14)

    def test_numbers_symbols_and_non_bmp_codepoints_follow_the_declared_rule(self):
        measured = measure_text("E0001_A2 ≥ 12.5 🙂")
        # E0001, _, A2, ≥, 12, ., 5, 🙂
        self.assertEqual(measured["reference_token_count"], 8)
        self.assertEqual(measured["whitespace_token_count"], 4)
        self.assertEqual(measure_text("👩\u200d🔬")["reference_token_count"], 3)

    def test_empty_text_has_zero_lengths(self):
        measured = measure_text("")
        for field in ("utf8_bytes", "unicode_codepoints", "whitespace_token_count", "reference_token_count"):
            self.assertEqual(measured[field], 0)
        self.assertTrue(measured["input_is_nfc"])

    def test_model_counts_are_explicitly_unavailable(self):
        measured = measure_text("Do not pretend these are BPE tokens.")
        self.assertIsNone(measured["model_token_count"])
        self.assertEqual(measured["model_tokenizer_status"], "not_configured")
        self.assertFalse(TOKENIZER_SPEC["is_model_tokenizer"])
        self.assertIn(unicodedata.unidata_version, TOKENIZATION_ID)
        self.assertEqual(measured["tokenization_id"], TOKENIZER_SPEC["tokenization_id"])

    def test_measurements_are_deterministic_and_json_serializable(self):
        original = "Çığ, Işıl'ın sağındadır.\nYanıt: YES veya NO."
        first = measure_text(original)
        self.assertEqual(first, measure_text(original))
        self.assertEqual(first, json.loads(json.dumps(first)))
        self.assertEqual(TOKENIZER_SPEC, json.loads(json.dumps(TOKENIZER_SPEC)))

    def test_non_strings_are_rejected(self):
        for invalid in (None, True, 1, b"text", ["text"]):
            with self.subTest(invalid=invalid), self.assertRaises(TypeError):
                measure_text(invalid)

    def test_unpaired_surrogates_are_rejected_without_replacement(self):
        for invalid in ("\ud800", "A\udfffB"):
            with self.subTest(invalid=repr(invalid)), self.assertRaises(ValueError):
                measure_text(invalid)


if __name__ == "__main__":
    unittest.main()

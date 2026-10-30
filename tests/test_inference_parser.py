"""Strict localized answers and diagnostic-only keyword recognition."""

import unittest

from inference.parser import PARSER_VERSION, parse_answer


class InferenceParserTests(unittest.TestCase):
    def test_exact_localized_tokens_produce_boolean_labels(self):
        for language, true, false in (("tr", "EVET", "HAYIR"), ("en", "YES", "NO")):
            for token, label in ((true, True), (false, False)):
                with self.subTest(language=language, token=token):
                    self.assertEqual(parse_answer(token, language), {
                        "parser_version": PARSER_VERSION, "valid": True,
                        "prediction": label, "observed_language": language,
                        "wrong_language": False, "invalid_reason": None})

    def test_surrounding_whitespace_is_the_only_normalization(self):
        self.assertIs(parse_answer(" \n\tEVET\r\n ", "tr")["prediction"], True)
        self.assertIs(parse_answer("\nNO\t", "en")["prediction"], False)

    def test_case_punctuation_markdown_json_and_explanations_are_invalid(self):
        cases = (("tr", ("evet", "hayır", "Hayır", "HAYİR", "EVET.", "**EVET**", '"EVET"',
                          '{"answer":"EVET"}', "Cevap: EVET", "EVET çünkü doğru.",
                          "```\nHAYIR\n```", "EVET EVET")),
                 ("en", ("yes", "No", "NO.", "YES!", "`YES`", "The answer is YES.",
                          "YES YES", "YES\nYES")))
        for language, outputs in cases:
            for output in outputs:
                with self.subTest(output=output):
                    result = parse_answer(output, language)
                    self.assertFalse(result["valid"])
                    self.assertIsNone(result["prediction"])
                    self.assertEqual(result["invalid_reason"], "invalid_format")

    def test_foreign_exact_answers_are_invalid_with_language_diagnostic(self):
        for language, tokens, observed in (("tr", ("YES", "NO"), "en"),
                                            ("en", ("EVET", "HAYIR"), "tr")):
            for token in tokens:
                result = parse_answer(token, language)
                self.assertFalse(result["valid"])
                self.assertIsNone(result["prediction"])
                self.assertTrue(result["wrong_language"])
                self.assertEqual(result["observed_language"], observed)
                self.assertEqual(result["invalid_reason"], "wrong_language")

    def test_conflicting_whole_answer_words_are_invalid_in_all_languages(self):
        for text in ("EVET HAYIR", "EVET. Hayır.", "YES, but NO", "EVET / NO", "YES HAYIR"):
            for language in ("tr", "en"):
                with self.subTest(text=text, language=language):
                    result = parse_answer(text, language)
                    self.assertEqual(result["invalid_reason"], "contradictory_answers")
                    self.assertIsNone(result["prediction"])

    def test_mixed_same_polarity_remains_invalid_and_reports_mixed_language(self):
        for language in ("tr", "en"):
            result = parse_answer("EVET YES", language)
            self.assertEqual(result["observed_language"], "mixed")
            self.assertTrue(result["wrong_language"])
            self.assertEqual(result["invalid_reason"], "wrong_language")
            self.assertIsNone(result["prediction"])

    def test_substrings_and_word_suffixes_do_not_license_keyword_detection(self):
        for text in ("yesterday", "nobody", "EVETLER", "HAYIRLI", "YES_NO", "YES2", "αYESβ"):
            result = parse_answer(text, "en")
            self.assertEqual(result["observed_language"], "unknown")
            self.assertFalse(result["wrong_language"])
            self.assertEqual(result["invalid_reason"], "invalid_format")
            self.assertIsNone(result["prediction"])

    def test_empty_and_non_text_responses_are_never_answers(self):
        for value in (None, True, 1, [], {}, b"YES"):
            self.assertEqual(parse_answer(value, "en")["invalid_reason"], "non_text")
        for value in ("", " \n\t"):
            self.assertEqual(parse_answer(value, "tr")["invalid_reason"], "empty")

    def test_truncated_output_is_invalid_even_when_visible_answer_is_exact(self):
        self.assertEqual(parse_answer("YES", "en", truncated=True)["invalid_reason"], "truncated")
        for reason in ("length", "max_tokens", "max_output_tokens", "incomplete"):
            self.assertEqual(parse_answer("EVET", "tr", finish_reason=reason)["invalid_reason"], "truncated")
        self.assertTrue(parse_answer("NO", "en", finish_reason="stop")["valid"])

    def test_refusal_takes_precedence_over_answer_and_truncation(self):
        self.assertEqual(parse_answer("YES", "en", refusal=True, truncated=True)["invalid_reason"], "refusal")
        for reason in ("content_filter", "refusal"):
            self.assertEqual(parse_answer("HAYIR", "tr", finish_reason=reason)["invalid_reason"], "refusal")

    def test_invalid_arguments_are_rejected(self):
        for language in ("TR", "english", "", None, []):
            with self.assertRaises(ValueError):
                parse_answer("YES", language)
        with self.assertRaises(ValueError):
            parse_answer("YES", "en", truncated=1)
        with self.assertRaises(ValueError):
            parse_answer("YES", "en", finish_reason=[])


if __name__ == "__main__":
    unittest.main()

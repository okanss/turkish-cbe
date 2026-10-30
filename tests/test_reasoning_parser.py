"""Synthetic answer strings for strategy-specific final-answer scoring."""

import copy
import unittest

from reasoning.parser import FINAL_PARSER_VERSION, PARSER_VERSIONS, STRATEGIES, parse_answer
from inference.parser import PARSER_VERSION as DIRECT_PARSER_VERSION
from inference.parser import parse_answer as parse_direct


REASONING = ("zero_shot_reasoning", "structured_reasoning")


class ReasoningParserTests(unittest.TestCase):
    def test_direct_and_provider_effort_delegate_existing_parser_exactly(self):
        values = ("YES", "NO", "EVET", "HAYIR", " yes ", "YES NO", "EVET HAYIR", "YES.",
                  "FINAL: YES", "EVET çünkü doğru.", "", None, True, 1, [], {}, b"YES")
        for strategy in ("direct", "provider_effort"):
            self.assertEqual(PARSER_VERSIONS[strategy], DIRECT_PARSER_VERSION)
            for language in ("tr", "en"):
                for text in values:
                    with self.subTest(strategy=strategy, language=language, text=text):
                        self.assertEqual(parse_answer(text, language, strategy), parse_direct(text, language))

    def test_exact_localized_final_tokens_are_boolean_for_both_reasoning_strategies(self):
        for strategy in REASONING:
            self.assertEqual(PARSER_VERSIONS[strategy], FINAL_PARSER_VERSION)
            for language, tokens in (("tr", (("EVET", True), ("HAYIR", False))),
                                     ("en", (("YES", True), ("NO", False)))):
                for token, expected in tokens:
                    for text in (f"FINAL: {token}", f"Açıklama / explanation.\nFINAL: {token}",
                                 f" \n\tFINAL: {token}\r\n\t "):
                        with self.subTest(strategy=strategy, language=language, text=text):
                            result = parse_answer(text, language, strategy)
                            self.assertTrue(result["valid"])
                            self.assertIs(result["prediction"], expected)
                            self.assertIsNone(result["invalid_reason"])
                            self.assertEqual(result["observed_language"], language)
                            self.assertFalse(result["wrong_language"])
                            self.assertEqual(result["parser_version"], FINAL_PARSER_VERSION)
                            self.assertEqual(result["final_answer_marker"], {"marker": "FINAL:", "occurrences": 1,
                                "is_final_line": True, "answer_token": token})

    def test_explanation_keywords_are_never_candidate_answers_or_contradictions(self):
        explanation = ("YES and NO may each be correct for different questions. "
                       "EVET/HAYIR are Turkish answer words. The target remains unchanged.")
        for strategy in REASONING:
            for language, token in (("tr", "EVET"), ("en", "NO")):
                result = parse_answer(explanation + "\nFINAL: " + token, language, strategy)
                self.assertTrue(result["valid"])
                self.assertEqual(result["observed_language"], language)
                self.assertFalse(result["wrong_language"])
                self.assertIsNone(result["invalid_reason"])

    def test_explanation_is_not_required_or_scored_for_correctness(self):
        for strategy in REASONING:
            for explanation in ("", "This explanation may be logically mistaken.\n", "EVET NO HAYIR YES\n"):
                result = parse_answer(explanation + "FINAL: YES", "en", strategy)
                self.assertTrue(result["valid"])
                self.assertIs(result["prediction"], True)

    def test_duplicate_markers_are_invalid_even_with_identical_labels(self):
        for strategy in REASONING:
            for text in ("FINAL: YES\nFINAL: YES", "FINAL: YES\nFINAL: NO",
                         "FINAL: EVET\nFINAL: YES", "final: YES\nFINAL: YES",
                         "FINAL : YES\nFINAL: YES", "FINAL: YES FINAL: YES",
                         "The reserved string is `FINAL: YES`.\nFINAL: YES"):
                result = parse_answer(text, "en", strategy)
                self.assertFalse(result["valid"])
                self.assertIsNone(result["prediction"])
                self.assertEqual(result["invalid_reason"], "duplicate_final_marker")
                self.assertEqual(result["final_answer_marker"]["occurrences"], 2)

    def test_valid_marker_must_be_on_the_last_nonempty_line(self):
        for strategy in REASONING:
            for text in ("FINAL: YES\nExtra explanation.", "FINAL: YES\r\nNO", "FINAL:\nYES",
                         "FINAL: YES\u2028Tail", "Reason.\nFINAL: YES\nThanks!"):
                result = parse_answer(text, "en", strategy)
                self.assertFalse(result["valid"])
                self.assertIsNone(result["prediction"])
                self.assertEqual(result["invalid_reason"], "final_marker_not_last_line")

    def test_inline_indented_case_and_spacing_marker_forms_are_not_accepted(self):
        for strategy in REASONING:
            for text in ("Reason FINAL: YES", "Reason.\n  FINAL: YES", "final: YES", "Final: YES",
                         "FINAL : YES", "FINAL:\tYES", "FINAL:YES", "FINAL:  YES"):
                with self.subTest(strategy=strategy, text=text):
                    result = parse_answer(text, "en", strategy)
                    self.assertFalse(result["valid"])
                    self.assertIsNone(result["prediction"])
                    self.assertEqual(result["invalid_reason"], "invalid_final_marker")

    def test_no_label_can_be_inferred_without_the_reserved_marker(self):
        for strategy in REASONING:
            for text in ("YES", "The answer is YES.", "EVET", "Answer: YES", "SONUÇ: EVET",
                         "FINAL YES", "FINAL： YES", "preFINAL: YES", "İFINAL: YES", "αFINAL: YES"):
                result = parse_answer(text, "en", strategy)
                self.assertFalse(result["valid"])
                self.assertIsNone(result["prediction"])
                self.assertEqual(result["invalid_reason"], "missing_final_marker")

    def test_wrong_language_is_diagnosed_only_from_final_payload(self):
        for strategy in REASONING:
            for language, text, observed in (("tr", "Reason EVET.\nFINAL: YES", "en"),
                                             ("en", "Reason YES.\nFINAL: HAYIR", "tr")):
                result = parse_answer(text, language, strategy)
                self.assertFalse(result["valid"])
                self.assertIsNone(result["prediction"])
                self.assertEqual(result["invalid_reason"], "wrong_language")
                self.assertEqual(result["observed_language"], observed)
                self.assertTrue(result["wrong_language"])

    def test_conflicting_payload_invalidity_is_distinct_from_explanation_words(self):
        for strategy in REASONING:
            for payload in ("YES NO", "YES / HAYIR", "EVET HAYIR"):
                result = parse_answer("Explanation.\nFINAL: " + payload, "en", strategy)
                self.assertEqual(result["invalid_reason"], "contradictory_answers")
                self.assertFalse(result["valid"])
                self.assertIsNone(result["prediction"])

    def test_payload_case_punctuation_json_markdown_and_extra_tail_remain_invalid(self):
        for strategy in REASONING:
            for payload in ("yes", "Yes", "YES.", "**YES**", '"YES"', '{"answer":"YES"}',
                            "YES because the chain supports it.", "YES YES", "YES\x00"):
                result = parse_answer("FINAL: " + payload, "en", strategy)
                self.assertFalse(result["valid"])
                self.assertIsNone(result["prediction"])
                self.assertEqual(result["invalid_reason"], "invalid_format")
            self.assertFalse(parse_answer("```\nFINAL: YES\n```", "en", strategy)["valid"])

    def test_refusal_and_truncation_override_even_a_valid_final_answer(self):
        for strategy in STRATEGIES:
            text = "YES" if strategy in ("direct", "provider_effort") else "FINAL: YES"
            for kwargs, reason in (({"refusal": True}, "refusal"), ({"truncated": True}, "truncated"),
                                   ({"refusal": True, "truncated": True}, "refusal")):
                result = parse_answer(text, "en", strategy, **kwargs)
                self.assertFalse(result["valid"])
                self.assertIsNone(result["prediction"])
                self.assertEqual(result["invalid_reason"], reason)
            for finish_reason in ("refusal", "content_filter", "length", "max_tokens", "max_output_tokens", "incomplete"):
                result = parse_answer(text, "en", strategy, finish_reason=finish_reason)
                self.assertFalse(result["valid"])
                self.assertIsNone(result["prediction"])
                self.assertEqual(result["invalid_reason"], "refusal" if finish_reason in ("refusal", "content_filter") else "truncated")

    def test_provider_conditions_precede_duplicate_or_missing_marker_diagnostics(self):
        for strategy in REASONING:
            for text in ("FINAL: YES\nFINAL: NO", "No final answer was generated."):
                self.assertEqual(parse_answer(text, "en", strategy, truncated=True)["invalid_reason"], "truncated")
                self.assertEqual(parse_answer(text, "en", strategy, refusal=True)["invalid_reason"], "refusal")

    def test_nontext_and_empty_outputs_stay_invalid_without_prediction(self):
        for strategy in REASONING:
            for text in (None, True, 1, [], {}, b"FINAL: YES"):
                result = parse_answer(text, "en", strategy)
                self.assertEqual(result["invalid_reason"], "non_text")
                self.assertIsNone(result["prediction"])
            for text in ("", " \n\t"):
                result = parse_answer(text, "en", strategy)
                self.assertEqual(result["invalid_reason"], "empty")
                self.assertIsNone(result["prediction"])

    def test_bad_arguments_and_unknown_strategies_are_rejected(self):
        for strategy in ("A", "reasoning", "DIRECT", "", None, [], True):
            with self.subTest(strategy=strategy), self.assertRaises(ValueError):
                parse_answer("YES", "en", strategy)
        for strategy in STRATEGIES:
            for language in (None, "TR", "english", [], True):
                with self.subTest(strategy=strategy, language=language), self.assertRaises(ValueError):
                    parse_answer("FINAL: YES", language, strategy)
            for flags in ({"truncated": 1}, {"refusal": 1}, {"finish_reason": []}):
                with self.subTest(strategy=strategy, flags=flags), self.assertRaises(ValueError):
                    parse_answer("FINAL: YES", "en", strategy, **flags)

    def test_results_are_fresh_and_raw_input_is_not_mutated(self):
        text = "Explanation mentions YES/NO.\nFINAL: EVET"
        first = parse_answer(text, "tr", "structured_reasoning")
        before = copy.deepcopy(first)
        second = parse_answer(text, "tr", "structured_reasoning")
        first["final_answer_marker"]["occurrences"] = 99
        self.assertEqual(second, before)
        self.assertEqual(text, "Explanation mentions YES/NO.\nFINAL: EVET")


if __name__ == "__main__":
    unittest.main()

"""Synthetic checks of denominators and the Boolean CBE identities."""

import copy
import itertools
import unittest

from analysis.metrics import summarize_pairs


def pair(original, equivalent, gold=True, *, cluster_id="latent-0", n=4):
    return {"original": original, "equivalent": equivalent, "gold": gold,
            "cluster_id": cluster_id, "n": n}


class AnalysisMetricTests(unittest.TestCase):
    def test_four_valid_correctness_cells_have_expected_counts(self):
        metrics = summarize_pairs([
            pair(True, True), pair(False, False), pair(True, False), pair(False, True)
        ])
        for field in ("both_correct", "both_wrong", "org_only", "eq_only"):
            self.assertEqual(metrics[field], 1)
        self.assertEqual(metrics["n_pairs"], 4)
        self.assertEqual(metrics["n_valid_pairs"], 4)
        self.assertEqual(metrics["n_clusters"], 1)
        for field in ("original_accuracy", "equivalent_accuracy", "mean_accuracy",
                      "original_accuracy_valid", "equivalent_accuracy_valid"):
            self.assertEqual(metrics[field], 0.5)
        self.assertEqual(metrics["R"], 0.25)
        self.assertEqual(metrics["D"], 0.25)
        self.assertEqual(metrics["U"], 0.5)
        self.assertEqual(metrics["invariance"], 0.5)
        self.assertEqual(metrics["q"], 0.5)
        self.assertEqual(metrics["I_expected"], 0.5)
        self.assertEqual(metrics["kappa"], 0)
        self.assertEqual(metrics["invalid_response_rate"], 0)
        self.assertEqual(metrics["valid_pair_coverage"], 1)
        self.assertEqual(metrics["strict_robust_correctness"], 0.25)
        self.assertEqual(metrics["undefined_reasons"], {})

    def test_false_gold_is_not_confused_with_negative_correctness(self):
        correct = summarize_pairs([pair(False, False, False)])
        self.assertEqual(correct["R"], 1)
        self.assertEqual(correct["D"], 0)
        self.assertEqual(correct["original_accuracy"], 1)
        wrong = summarize_pairs([pair(True, True, False)])
        self.assertEqual(wrong["D"], 1)
        self.assertEqual(wrong["R"], 0)
        self.assertEqual(wrong["original_accuracy"], 0)

    def test_invalid_and_missing_answers_remain_in_main_denominators(self):
        metrics = summarize_pairs([
            pair(True, True), pair(None, True), pair(False, None), pair(None, None)
        ])
        self.assertEqual(metrics["n_pairs"], 4)
        self.assertEqual(metrics["n_valid_pairs"], 1)
        self.assertEqual(metrics["n_invalid_original"], 2)
        self.assertEqual(metrics["n_invalid_equivalent"], 2)
        self.assertEqual(metrics["n_invalid_responses"], 4)
        self.assertEqual(metrics["original_accuracy"], 0.25)
        self.assertEqual(metrics["equivalent_accuracy"], 0.5)
        self.assertEqual(metrics["mean_accuracy"], 1)
        self.assertEqual(metrics["R"], 1)
        self.assertEqual(metrics["strict_robust_correctness"], 0.25)
        self.assertEqual(metrics["valid_pair_coverage"], 0.25)
        self.assertEqual(metrics["invalid_response_rate"], 0.5)
        self.assertEqual(metrics["q"], 1)

    def test_q_uses_common_support_and_is_not_accuracy(self):
        metrics = summarize_pairs([
            pair(True, True, False), pair(True, False, False), pair(None, False, False)
        ])
        self.assertEqual(metrics["q"], 0.75)
        self.assertEqual(metrics["I_expected"], 0.625)
        self.assertEqual(metrics["mean_accuracy"], 0.25)
        self.assertEqual(metrics["invariance"], 0.5)
        self.assertAlmostEqual(metrics["kappa"], -1 / 3)

    def test_all_yes_and_all_no_make_kappa_undefined(self):
        for answer in (True, False):
            with self.subTest(answer=answer):
                metrics = summarize_pairs([pair(answer, answer, True),
                                           pair(answer, answer, False)])
                self.assertEqual(metrics["invariance"], 1)
                self.assertEqual(metrics["q"], int(answer))
                self.assertEqual(metrics["I_expected"], 1)
                self.assertIsNone(metrics["kappa"])
                self.assertEqual(metrics["undefined_reasons"],
                                 {"kappa": "constant_pooled_answer"})

    def test_balanced_polarity_perfect_invariance_has_kappa_one(self):
        metrics = summarize_pairs([pair(True, True), pair(False, False)])
        self.assertEqual(metrics["kappa"], 1)

    def test_opposite_answers_have_negative_one_kappa(self):
        metrics = summarize_pairs([pair(True, False), pair(False, True)])
        self.assertEqual(metrics["kappa"], -1)

    def test_empty_input_does_not_turn_undefined_rates_into_zero(self):
        metrics = summarize_pairs([])
        for field in ("n_pairs", "n_valid_pairs", "n_clusters", "n_invalid_original",
                      "n_invalid_equivalent", "n_invalid_responses", "both_correct", "both_wrong", "org_only", "eq_only"):
            self.assertEqual(metrics[field], 0)
        for field in ("original_accuracy", "equivalent_accuracy", "invalid_response_rate",
                      "valid_pair_coverage", "strict_robust_correctness"):
            self.assertIsNone(metrics[field])
            self.assertEqual(metrics["undefined_reasons"][field], "no_pairs")
        for field in ("R", "D", "U", "mean_accuracy", "invariance", "q", "I_expected",
                      "kappa", "original_accuracy_valid", "equivalent_accuracy_valid"):
            self.assertIsNone(metrics[field])
            self.assertEqual(metrics["undefined_reasons"][field], "no_valid_pairs")

    def test_no_common_valid_pairs_still_has_main_metrics(self):
        metrics = summarize_pairs([pair(None, True), pair(False, None)])
        self.assertEqual(metrics["original_accuracy"], 0)
        self.assertEqual(metrics["equivalent_accuracy"], 0.5)
        self.assertEqual(metrics["strict_robust_correctness"], 0)
        self.assertEqual(metrics["valid_pair_coverage"], 0)
        self.assertEqual(metrics["invalid_response_rate"], 0.5)
        for field in ("R", "D", "U", "mean_accuracy", "invariance", "q", "I_expected", "kappa"):
            self.assertIsNone(metrics[field])
            self.assertEqual(metrics["undefined_reasons"][field], "no_valid_pairs")

    def test_exhaustive_two_pair_outcomes_preserve_cbe_identities(self):
        states = list(itertools.product((True, False, None), (True, False, None),
                                        (True, False)))
        for first, second in itertools.product(states, repeat=2):
            with self.subTest(first=first, second=second):
                records = [pair(*first), pair(*second, cluster_id="latent-1")]
                metrics = summarize_pairs(records)
                common = [row for row in records
                          if row["original"] is not None and row["equivalent"] is not None]
                self.assertEqual(metrics["n_valid_pairs"], len(common))
                self.assertEqual(metrics["n_clusters"], 2)
                self.assertAlmostEqual(metrics["strict_robust_correctness"],
                                       metrics["both_correct"] / 2)
                if not common:
                    continue
                self.assertAlmostEqual(metrics["R"] + metrics["D"] + metrics["U"], 1)
                self.assertAlmostEqual(metrics["mean_accuracy"], metrics["R"] + metrics["U"] / 2)
                self.assertAlmostEqual(metrics["invariance"], metrics["R"] + metrics["D"])
                self.assertAlmostEqual(metrics["mean_accuracy"],
                                       (metrics["original_accuracy_valid"]
                                        + metrics["equivalent_accuracy_valid"]) / 2)
                self.assertAlmostEqual(metrics["strict_robust_correctness"],
                                       metrics["R"] * metrics["valid_pair_coverage"])
                agreement = sum(row["original"] == row["equivalent"] for row in common)
                self.assertAlmostEqual(metrics["invariance"], agreement / len(common))

    def test_iterator_extra_metadata_and_input_nonmutation(self):
        rows = [pair(True, False), pair(None, True, cluster_id="latent-1")]
        rows[0]["metadata"] = {"nested": [1, 2, 3]}
        before = copy.deepcopy(rows)
        self.assertEqual(summarize_pairs(iter(rows)), summarize_pairs(rows))
        self.assertEqual(rows, before)

    def test_pooled_n_is_rejected_even_if_one_level_has_only_invalid_pairs(self):
        with self.assertRaisesRegex(ValueError, "mixed n"):
            summarize_pairs([pair(True, True), pair(None, None, n=8)])

    def test_exact_boolean_types_required(self):
        for key in ("original", "equivalent", "gold"):
            for value in (0, 1, "YES", "EVET", [], {}, 0.0):
                with self.subTest(key=key, value=value):
                    row = pair(True, False)
                    row[key] = value
                    with self.assertRaises(ValueError):
                        summarize_pairs([row])
        with self.assertRaisesRegex(ValueError, "gold"):
            summarize_pairs([pair(True, True, None)])

    def test_invalid_n_and_cluster_ids_rejected(self):
        for value in (0, -1, True, 4.0, "4", None):
            with self.subTest(n=value), self.assertRaises(ValueError):
                summarize_pairs([pair(True, True, n=value)])
        for value in ("", " \n\t", 1, None, []):
            with self.subTest(cluster_id=value), self.assertRaises(ValueError):
                summarize_pairs([pair(True, True, cluster_id=value)])

    def test_missing_fields_bad_rows_and_bad_outer_containers_rejected(self):
        for key in ("original", "equivalent", "gold", "cluster_id", "n"):
            row = pair(True, True)
            del row[key]
            with self.subTest(missing=key), self.assertRaisesRegex(ValueError, "missing"):
                summarize_pairs([row])
        for row in (None, True, "answer", [True, False]):
            with self.subTest(row=row), self.assertRaises(ValueError):
                summarize_pairs([row])
        for container in (None, 1, "", b"", {}, pair(True, False)):
            with self.subTest(container=container), self.assertRaises(ValueError):
                summarize_pairs(container)


if __name__ == "__main__":
    unittest.main()

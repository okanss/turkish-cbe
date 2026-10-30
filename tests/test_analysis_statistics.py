"""Synthetic statistical checks; these are not model evaluation results."""

from collections import Counter
from fractions import Fraction
import math
import random
import unittest

from analysis.statistics import bootstrap, holm_adjust, mcnemar, wilson_interval


class WilsonTests(unittest.TestCase):
    def test_published_score_interval_examples(self):
        examples = ((0, 10, 0.0, 0.2775327998628892),
                    (5, 10, 0.236593090512564, 0.763406909487436),
                    (10, 10, 0.7224672001371107, 1.0))
        for successes, total, low, high in examples:
            with self.subTest(successes=successes):
                interval = wilson_interval(successes, total)
                self.assertEqual(interval["status"], "ok")
                self.assertAlmostEqual(interval["low"], low, places=13)
                self.assertAlmostEqual(interval["high"], high, places=13)
                self.assertEqual(interval["estimate"], successes / total)

    def test_empty_denominator_has_no_interval_or_estimate(self):
        interval = wilson_interval(0, 0)
        self.assertEqual(interval["status"], "undefined_empty_sample")
        self.assertIsNone(interval["low"])
        self.assertIsNone(interval["high"])
        self.assertIsNone(interval["estimate"])

    def test_complement_symmetry(self):
        for total in range(1, 20):
            for successes in range(total + 1):
                a = wilson_interval(successes, total)
                b = wilson_interval(total - successes, total)
                self.assertAlmostEqual(a["low"], 1 - b["high"])
                self.assertAlmostEqual(a["high"], 1 - b["low"])

    def test_higher_confidence_produces_wider_interval(self):
        lower = wilson_interval(20, 100, 0.90)
        higher = wilson_interval(20, 100, 0.99)
        self.assertLess(higher["low"], lower["low"])
        self.assertGreater(higher["high"], lower["high"])

    def test_extreme_finite_confidence_is_supported(self):
        interval = wilson_interval(5, 10, math.nextafter(1.0, 0.0))
        self.assertGreaterEqual(interval["low"], 0)
        self.assertLessEqual(interval["high"], 1)

    def test_large_integer_counts_do_not_require_lossy_float_conversion(self):
        interval = wilson_interval(10 ** 400, 2 * 10 ** 400)
        self.assertEqual(interval["estimate"], 0.5)
        self.assertEqual(interval["status"], "ok")

    def test_invalid_counts_and_confidences_rejected(self):
        for successes, total in ((-1, 1), (2, 1), (0, -1), (True, 1), (1.0, 2), (1, False)):
            with self.subTest(counts=(successes, total)), self.assertRaises(ValueError):
                wilson_interval(successes, total)
        for confidence in (0, 1, -0.1, 1.1, True, None, "0.95", float("nan"), float("inf"), 10 ** 400):
            with self.subTest(confidence=confidence), self.assertRaises(ValueError):
                wilson_interval(1, 2, confidence)


def paired_counts(b, c):
    return [True] * b + [False] * c + [True, False], [False] * b + [True] * c + [True, False]


class McNemarTests(unittest.TestCase):
    def test_exact_values_match_exhaustive_binomial_probability(self):
        for b in range(13):
            for c in range(13):
                with self.subTest(b=b, c=c):
                    left, right = paired_counts(b, c)
                    result = mcnemar(left, right, method="exact")
                    discordant = b + c
                    expected = (min(Fraction(1), 2 * sum(
                        (Fraction(math.comb(discordant, k), 2 ** discordant)
                         for k in range(min(b, c) + 1)), Fraction(0)))
                        if discordant else Fraction(1))
                    self.assertEqual(result["b"], b)
                    self.assertEqual(result["c"], c)
                    self.assertEqual(result["both_correct"], 1)
                    self.assertEqual(result["both_wrong"], 1)
                    self.assertAlmostEqual(result["p_value"], float(expected), places=15)

    def test_known_small_exact_test_and_symmetry(self):
        left, right = paired_counts(1, 9)
        result = mcnemar(left, right)
        self.assertEqual(result["method"], "exact")
        self.assertEqual(result["p_value"], 0.021484375)
        self.assertEqual(mcnemar(right, left)["p_value"], result["p_value"])

    def test_auto_boundary_is_25_discordants(self):
        for count, method in ((25, "exact"), (26, "asymptotic")):
            self.assertEqual(mcnemar(*paired_counts(count, 0))["method"], method)

    def test_continuity_corrected_asymptotic_value(self):
        result = mcnemar(*paired_counts(30, 10))
        self.assertEqual(result["method"], "asymptotic")
        self.assertEqual(result["statistic"], 361 / 40)
        self.assertAlmostEqual(result["p_value"], 0.00266311925913855, places=13)

    def test_explicit_asymptotic_uses_nonnegative_continuity_correction(self):
        result = mcnemar(*paired_counts(1, 1), method="asymptotic")
        self.assertEqual(result["statistic"], 0)
        self.assertEqual(result["p_value"], 1)

    def test_no_discordance_has_p_one_and_empty_sample_is_undefined(self):
        result = mcnemar([True, False], [True, False])
        self.assertEqual(result["p_value"], 1)
        empty = mcnemar([], [])
        self.assertIsNone(empty["p_value"])
        self.assertEqual(empty["status"], "undefined_empty_sample")

    def test_repeated_latent_orders_do_not_receive_ordinary_p_value(self):
        left, right = paired_counts(1, 9)
        result = mcnemar(left, right, clusters=["same-order"] * len(left))
        self.assertEqual(result["b"], 1)
        self.assertEqual(result["c"], 9)
        self.assertIsNone(result["p_value"])
        self.assertEqual(result["status"], "undefined_dependent_clusters")
        self.assertEqual(mcnemar(left, right, clusters=list(range(len(left))))["status"], "ok")

    def test_non_boolean_misaligned_and_invalid_cluster_inputs_rejected(self):
        cases = (([1], [True], {}), ([True], [None], {}), ("YES", "NO", {}),
                 ([True], [], {}), ([True], [False], {"method": "chi-square"}),
                 ([True], [False], {"clusters": []}),
                 ([True], [False], {"clusters": "a"}),
                 ([True], [False], {"clusters": [None]}),
                 ([True], [False], {"clusters": [float("nan")]}))
        for left, right, kwargs in cases:
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                mcnemar(left, right, **kwargs)


class HolmTests(unittest.TestCase):
    def test_known_step_down_example_is_monotonic(self):
        self.assertEqual(holm_adjust({"a": 0.01, "b": 0.04, "c": 0.03}),
                         {"a": 0.03, "b": 0.06, "c": 0.06})

    def test_missing_hypotheses_remain_in_declared_family(self):
        corrected = holm_adjust({"a": 0.01, "b": 0.04, "c": 0.03, "missing": None})
        self.assertEqual(corrected, {"a": 0.04, "b": 0.09, "c": 0.09, "missing": None})

    def test_ties_reordering_and_probability_boundaries(self):
        self.assertEqual(holm_adjust({"z": 0.01, "a": 0.01, "b": 0.03}),
                         {"a": 0.03, "b": 0.03, "z": 0.03})
        self.assertEqual(holm_adjust({"a": 0, "b": 1}), {"a": 0, "b": 1})
        self.assertEqual(holm_adjust({}), {})
        self.assertEqual(holm_adjust({"a": None}), {"a": None})

    def test_invalid_inputs_are_not_silently_treated_as_missing(self):
        for value in (-0.1, 1.1, True, "0.05", float("nan"), float("inf"), 10 ** 400):
            with self.subTest(value=value), self.assertRaises(ValueError):
                holm_adjust({"a": value})
        for values in ([], {"": 0.05}, {1: 0.05}):
            with self.assertRaises(ValueError):
                holm_adjust(values)


def mean_statistic(rows):
    return {"mean": sum(row["value"] for row in rows) / len(rows) if rows else None}


class BootstrapTests(unittest.TestCase):
    def setUp(self):
        self.rows = [{"cluster_id": cluster, "n": n, "language": language, "value": value}
                     for cluster, n, value in (("a", 4, 0), ("b", 4, 1),
                                               ("c", 8, 2), ("d", 8, 3))
                     for language in ("en", "tr")]

    def test_seeded_bootstrap_is_input_order_independent(self):
        a = bootstrap(self.rows, mean_statistic, repeats=101, seed=19)
        shuffled = self.rows[:]
        random.Random(70).shuffle(shuffled)
        b = bootstrap(shuffled, mean_statistic, repeats=101, seed=19)
        self.assertEqual(a, b)
        self.assertEqual(a["mean"]["estimate"], 1.5)
        self.assertEqual(a["mean"]["status"], "ok")

    def test_default_is_2000_replicates_and_records_denominator(self):
        calls = []
        def statistic(rows):
            calls.append(len(rows))
            return mean_statistic(rows)
        interval = bootstrap(self.rows, statistic)["mean"]
        self.assertEqual(len(calls), 2001)
        self.assertEqual(interval["repeats"], 2000)
        self.assertEqual(interval["defined_replicates"], 2000)
        self.assertEqual(interval["independent_clusters"], 4)
        self.assertEqual(interval["strata"], 2)

    def test_all_languages_move_together_strata_fixed_and_duplicate_draws_retained(self):
        saw_duplicate = []
        def statistic(rows):
            self.assertEqual(Counter(row["n"] for row in rows), {4: 4, 8: 4})
            for cluster in {row["cluster_id"] for row in rows}:
                subset = [row for row in rows if row["cluster_id"] == cluster]
                counts = Counter(row["language"] for row in subset)
                self.assertEqual(counts["en"], counts["tr"])
                if counts["en"] > 1:
                    saw_duplicate.append(True)
            return mean_statistic(rows)
        bootstrap(self.rows, statistic, repeats=50)
        self.assertTrue(saw_duplicate)

    def test_variable_sized_clusters_are_resampled_without_row_level_reweighting(self):
        rows = [{"cluster_id": cluster, "n": 4, "value": value}
                for cluster, size, value in (("a", 1, 0), ("b", 3, 1))
                for _ in range(size)]
        sizes = []
        def statistic(current):
            sizes.append(len(current))
            return mean_statistic(current)
        interval = bootstrap(rows, statistic, repeats=100)["mean"]
        self.assertEqual(interval["estimate"], 0.75)
        self.assertEqual(set(sizes), {2, 4, 6})

    def test_percentiles_use_linear_interpolation(self):
        rows = [{"cluster_id": "a", "n": 4, "value": 0},
                {"cluster_id": "b", "n": 4, "value": 1}]
        rng = random.Random(3)
        values = sorted(sum(rng.choice([0, 1]) for _ in range(2)) / 2 for _ in range(7))
        result = bootstrap(rows, mean_statistic, seed=3, repeats=7, confidence=0.80)["mean"]
        self.assertEqual(result["low"], values[0] + 0.6 * (values[1] - values[0]))
        self.assertAlmostEqual(result["high"], values[5] + 0.4 * (values[6] - values[5]))

    def test_partial_undefined_replicates_have_no_silently_conditional_interval(self):
        rows = [{"cluster_id": "a", "n": 4, "value": 0},
                {"cluster_id": "b", "n": 4, "value": 1}]
        def statistic(current):
            values = {row["value"] for row in current}
            return {"metric": 0.5 if len(values) == 2 else None, "always": 1}
        intervals = bootstrap(rows, statistic, repeats=200, seed=5)
        partial = intervals["metric"]
        self.assertEqual(partial["estimate"], 0.5)
        self.assertGreater(partial["defined_replicates"], 0)
        self.assertLess(partial["defined_replicates"], 200)
        self.assertEqual(partial["status"], "partial_undefined")
        self.assertIsNone(partial["low"])
        self.assertIsNone(partial["high"])
        self.assertEqual(intervals["always"]["status"], "ok")

    def test_undefined_original_metric_stays_undefined(self):
        result = bootstrap(self.rows, lambda rows: {"undefined": None}, repeats=3)["undefined"]
        self.assertEqual(result["status"], "undefined_estimate")
        self.assertIsNone(result["estimate"])
        self.assertIsNone(result["low"])
        self.assertEqual(result["defined_replicates"], 0)

    def test_fewer_than_two_latent_clusters_has_no_interval(self):
        for rows in ([], self.rows[:2]):
            interval = bootstrap(rows, mean_statistic, repeats=10)["mean"]
            self.assertEqual(interval["status"], "insufficient_clusters")
            self.assertIsNone(interval["low"])
            self.assertIsNone(interval["high"])
            self.assertEqual(interval["defined_replicates"], 0)
        self.assertEqual(bootstrap(self.rows[:2], mean_statistic)["mean"]["estimate"], 0)

    def test_singleton_stratum_cannot_produce_a_degenerate_confidence_interval(self):
        # Two total clusters do not estimate variation within either n stratum.
        rows = [{"cluster_id": "a", "n": 4, "value": 0},
                {"cluster_id": "b", "n": 8, "value": 1}]
        interval = bootstrap(rows, mean_statistic, repeats=50)["mean"]
        self.assertEqual(interval["estimate"], 0.5)
        self.assertEqual(interval["independent_clusters"], 2)
        self.assertEqual(interval["status"], "insufficient_clusters")
        self.assertIn("within-stratum", interval["reason"])
        self.assertEqual(interval["defined_replicates"], 0)
        self.assertIsNone(interval["low"])
        self.assertIsNone(interval["high"])
        # A single sparse stratum is sufficient to invalidate the global CI.
        mixed = self.rows + [{"cluster_id": "e", "n": 16, "value": 4}]
        self.assertEqual(bootstrap(mixed, mean_statistic, repeats=5)["mean"]["status"],
                         "insufficient_clusters")

    def test_same_cluster_across_strata_is_rejected(self):
        rows = self.rows + [{"cluster_id": "a", "n": 8, "value": 0}]
        with self.assertRaisesRegex(ValueError, "span"):
            bootstrap(rows, mean_statistic)

    def test_multiple_stratum_keys_and_unstratified_mode(self):
        rows = [dict(row, split="test") for row in self.rows]
        a = bootstrap(rows, mean_statistic, repeats=20, strata_keys=("n", "split"))
        b = bootstrap(rows, mean_statistic, repeats=20, strata_keys=())
        self.assertEqual(a["mean"]["strata"], 2)
        self.assertEqual(b["mean"]["strata"], 1)

    def test_invalid_options_are_rejected(self):
        cases = ({"seed": True}, {"seed": -1}, {"seed": 1.0}, {"repeats": 0},
                 {"repeats": True}, {"repeats": 1.0}, {"confidence": 1},
                 {"confidence": float("nan")}, {"cluster_key": ""},
                 {"strata_keys": "n"}, {"strata_keys": ("n", "n")},
                 {"strata_keys": ("cluster_id",)})
        for kwargs in cases:
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                bootstrap(self.rows, mean_statistic, **kwargs)

    def test_invalid_rows_and_statistic_outputs_are_rejected(self):
        cases = ([{}], [{"cluster_id": None, "n": 4}],
                 [{"cluster_id": True, "n": 4}], [{"cluster_id": "a", "n": float("inf")}],
                 [{"cluster_id": "a", "n": 4, "bad": float("nan")}], ["not-row"])
        for rows in cases:
            with self.subTest(rows=rows), self.assertRaises(ValueError):
                bootstrap(rows, mean_statistic)
        for result in ({"x": float("nan")}, {"x": True}, {"x": "0.5"},
                       {"x": 10 ** 400}, {1: 1}, []):
            with self.subTest(result=result), self.assertRaises(ValueError):
                bootstrap(self.rows, lambda rows: result, repeats=2)

    def test_changing_metric_names_is_an_error(self):
        calls = []
        def statistic(rows):
            calls.append(True)
            return {"estimate" if len(calls) == 1 else "changed": 1}
        with self.assertRaisesRegex(ValueError, "same metric names"):
            bootstrap(self.rows, statistic, repeats=2)


if __name__ == "__main__":
    unittest.main()

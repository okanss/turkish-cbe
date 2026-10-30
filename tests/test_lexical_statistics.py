"""Synthetic lexical-statistics checks, never research performance results."""

from copy import deepcopy
import importlib.util
import math
import random
import unittest
from unittest.mock import patch

from analysis.lexical_bias.statistics import fit_controlled, spearman


class SpearmanTests(unittest.TestCase):
    def test_monotone_and_reversed_values_have_unit_rank_correlation(self):
        self.assertEqual(spearman([1, 2, 4, 8], [2, 4, 16, 64])["coefficient"], 1)
        self.assertEqual(spearman([1, 2, 4, 8], [64, 16, 4, 2])["coefficient"], -1)

    def test_ties_use_average_ranks(self):
        result = spearman([1, 1, 2, 3], [1, 2, 2, 4])
        self.assertAlmostEqual(result["coefficient"], 5 / 6)
        self.assertEqual(result["tie_method"], "average_rank")
        self.assertEqual(result["status"], "ok")
        self.assertEqual(spearman([1, 2, 2, 3], [4, 1, 1, 0])["coefficient"], -1)

    def test_constant_and_small_samples_are_explicitly_undefined(self):
        for x, y, status in (([1, 1], [2, 3], "constant_input"),
                             ([1, 2], [3, 3], "constant_input"),
                             ([], [], "insufficient_pairs"), ([1], [2], "insufficient_pairs")):
            with self.subTest(status=status):
                result = spearman(x, y)
                self.assertIsNone(result["coefficient"])
                self.assertEqual(result["status"], status)

    def test_missing_and_nonfinite_pairs_are_not_silently_dropped(self):
        for value in (None, True, "2", float("nan"), float("inf"), 10 ** 400):
            with self.subTest(value=value):
                result = spearman([1, value, 3], [1, 2, 3])
                self.assertEqual(result["status"], "invalid_values")
                self.assertEqual(result["n"], 3)
                self.assertEqual(result["n_complete"], 2)
                self.assertIsNone(result["coefficient"])

    def test_input_and_length_errors_have_null_coefficient(self):
        for x, y, status in (([1, 2], [1], "length_mismatch"),
                             ("12", [1, 2], "invalid_input"),
                             ((i for i in range(2)), [1, 2], "invalid_input")):
            result = spearman(x, y)
            self.assertEqual(result["status"], status)
            self.assertIsNone(result["coefficient"])

    def test_repeated_presentations_never_obtain_an_iid_p_value(self):
        result = spearman([1, 1, 2, 2] * 100, [1, 2, 3, 4] * 100)
        self.assertIsNone(result["p_value"])
        self.assertIn("no_independent_observation_p_value", result["inference"])


def regression_rows():
    rows = []
    for model_index, model in enumerate(("model-a", "model-b")):
        for unit in range(8):
            family = "spatial" if unit < 4 else "temporal"
            direction = "before" if unit % 2 == 0 else "after"
            frequency = 0.4 * unit + 0.11 * (unit % 3)
            for replicate in range(6):
                n = 4 * 2 ** (replicate % 3)
                tokens = 30 + 7 * replicate + 3 * unit + 5 * model_index
                outcome = (1.25 + 0.6 * frequency + 0.2 * math.log2(n) - 0.3 * math.log(tokens)
                           + 0.7 * model_index + 0.4 * (family == "temporal") - 0.2 * (direction == "after"))
                rows.append({"model_id": model, "n": n, "relation_family": family,
                             "vocabulary_id": f"unit-{unit}", "query_relation": direction,
                             "log_frequency": frequency, "prompt_tokens": tokens, "accuracy": outcome,
                             "cluster_id": f"latent-{unit}-{replicate}"})
    return rows


@unittest.skipUnless(importlib.util.find_spec("numpy") is not None, "optional pinned NumPy is required")
class ControlledRegressionTests(unittest.TestCase):
    def test_known_full_rank_controlled_coefficients_are_recovered(self):
        rows = regression_rows()
        original = deepcopy(rows)
        result = fit_controlled(rows, outcome="accuracy")
        self.assertEqual(rows, original)
        self.assertEqual(result["status"], "ok")
        expected = {"Intercept": 1.25, "log2_n": 0.2, "log_prompt_tokens": -0.3,
                    "log_frequency": 0.6, "model[model-b]": 0.7,
                    "family[temporal]": 0.4, "query_relation[after]": -0.2}
        for column, coefficient in expected.items():
            self.assertAlmostEqual(result["coefficients"][column], coefficient, places=11)
        self.assertEqual(result["rank"], result["n_parameters"])
        self.assertEqual((result["n_rows"], result["n_vocabulary"], result["n_latent"]), (96, 8, 48))
        self.assertAlmostEqual(result["r_squared"], 1)
        self.assertIn("fewer_than_30_independent_vocabulary_units", result["warnings"])

    def test_coefficient_order_is_stable_under_row_reordering(self):
        rows = regression_rows()
        a = fit_controlled(rows, outcome="accuracy")
        random.Random(7).shuffle(rows)
        b = fit_controlled(rows, outcome="accuracy")
        self.assertEqual(a["model_design"], b["model_design"])
        for name in a["coefficients"]:
            self.assertAlmostEqual(a["coefficients"][name], b["coefficients"][name], places=11)

    def test_vocabulary_interaction_excludes_frequency_and_recovers_known_terms(self):
        rows = regression_rows()
        for row in rows:
            unit = int(row["vocabulary_id"].split("-")[1])
            model_b = row["model_id"] == "model-b"
            row["accuracy"] = (2 + 0.2 * math.log2(row["n"]) + 0.1 * math.log(row["prompt_tokens"])
                               + 0.4 * model_b + 0.3 * unit + 0.05 * unit * model_b)
            del row["log_frequency"]  # Not required or spuriously reinserted.
        result = fit_controlled(rows, outcome="accuracy", interaction=True)
        self.assertEqual(result["status"], "ok")
        self.assertFalse(result["model_design"]["frequency_predictor_included"])
        self.assertNotIn("log_frequency", result["coefficients"])
        self.assertNotIn("query_relation[after]", result["coefficients"])
        self.assertAlmostEqual(result["coefficients"]["Intercept"], 2, places=10)
        self.assertAlmostEqual(result["coefficients"]["model[model-b]"], 0.4, places=10)
        self.assertAlmostEqual(result["coefficients"]["vocabulary[unit-3]"], 0.9, places=10)
        self.assertAlmostEqual(result["coefficients"]["model[model-b]:vocabulary[unit-3]"], 0.15, places=10)

    def test_rank_deficiency_returns_diagnostics_without_minimum_norm_coefficients(self):
        rows = regression_rows()
        for row in rows:
            row["log_frequency"] = -2
        result = fit_controlled(rows, outcome="accuracy")
        self.assertEqual(result["status"], "rank_deficient")
        self.assertIsNone(result["coefficients"])
        self.assertLess(result["rank"], result["n_parameters"])
        self.assertIn("log_frequency", result["constant_columns"])
        self.assertTrue(result["singular_values"])

    def test_single_complexity_control_is_not_silently_removed(self):
        rows = regression_rows()
        for row in rows:
            row["n"] = 8
        result = fit_controlled(rows, outcome="accuracy")
        self.assertEqual(result["status"], "rank_deficient")
        self.assertIn("log2_n", result["constant_columns"])
        self.assertIsNone(result["coefficients"])

    def test_missing_model_vocabulary_cell_is_explicit_and_unestimable(self):
        rows = [row for row in regression_rows() if not (
            row["model_id"] == "model-b" and row["vocabulary_id"] == "unit-7")]
        result = fit_controlled(rows, outcome="accuracy", interaction=True)
        self.assertEqual(result["status"], "rank_deficient")
        self.assertEqual(result["model_design"]["missing_model_vocabulary_cells"], [["model-b", "unit-7"]])
        self.assertIsNone(result["coefficients"])

    def test_missing_values_return_original_denominator_and_no_hidden_completecase_fit(self):
        for field, value in (("log_frequency", None), ("prompt_tokens", 0),
                             ("accuracy", float("nan")), ("n", True), ("model_id", "")):
            with self.subTest(field=field):
                rows = regression_rows()
                rows[0][field] = value
                result = fit_controlled(rows, outcome="accuracy")
                self.assertEqual(result["status"], "missing_values")
                self.assertEqual(result["n_rows"], 96)
                self.assertEqual(result["n_complete"], 95)
                self.assertEqual(result["missing_fields"][field], 1)
                self.assertIsNone(result["coefficients"])

    def test_vocab_identifiers_cannot_span_incompatible_relation_metadata(self):
        rows = regression_rows()
        rows[0]["query_relation"] = "after"
        result = fit_controlled(rows, outcome="accuracy")
        self.assertEqual(result["status"], "inconsistent_vocabulary_metadata")
        self.assertIsNone(result["coefficients"])

    def test_small_n_and_constant_outcome_are_explicit(self):
        self.assertEqual(fit_controlled(regression_rows()[:3], outcome="accuracy")["status"], "insufficient_rows")
        rows = regression_rows()
        for row in rows:
            row["accuracy"] = 1
        result = fit_controlled(rows, outcome="accuracy")
        self.assertEqual(result["status"], "constant_outcome")
        self.assertIsNone(result["coefficients"])

    def test_no_number_of_repeated_rows_creates_iid_inference(self):
        rows = regression_rows() * 3
        result = fit_controlled(rows, outcome="accuracy")
        self.assertEqual(result["status"], "ok")
        self.assertIsNone(result["p_values"])
        self.assertIsNone(result["standard_errors"])
        self.assertIsNone(result["confidence_intervals"])
        self.assertEqual(result["n_latent"], 48)
        self.assertEqual(result["n_vocabulary"], 8)

    def test_eighteen_design_cells_can_contain_only_fourteen_independent_roots(self):
        rows = []
        prototype = regression_rows()[0]
        for model_index, model in enumerate(("model-a", "model-b")):
            for unit in range(18):
                root = unit if unit < 14 else unit - 14
                for replicate in range(6):
                    row = deepcopy(prototype)
                    row.update(model_id=model, vocabulary_id=f"cell-{unit}",
                        independent_vocabulary_id=f"root-{root}",
                        relation_family="spatial" if unit < 9 else "temporal",
                        query_relation="before" if unit % 2 == 0 else "after",
                        log_frequency=root * 0.1, n=4 * 2 ** (replicate % 3),
                        prompt_tokens=30 + unit * 3 + replicate * 7 + model_index * 5,
                        cluster_id=f"latent-{unit}-{replicate}", accuracy=root * 0.1 + replicate)
                    rows.append(row)
        result = fit_controlled(rows, outcome="accuracy")
        self.assertEqual(result["n_vocabulary_cells"], 18)
        self.assertEqual(result["n_vocabulary"], 14)
        self.assertEqual(result["model_design"]["independent_vocabulary_unit"], "independent_vocabulary_id")
        self.assertIn("fewer_than_30_independent_vocabulary_units", result["warnings"])
        self.assertNotEqual(result["status"], "inconsistent_vocabulary_metadata")

    def test_optional_independent_unit_identifiers_must_be_complete_and_nonblank(self):
        for value in (None, ""):
            rows = regression_rows()
            for row in rows:
                row["independent_vocabulary_id"] = row["vocabulary_id"]
            rows[0]["independent_vocabulary_id"] = value
            result = fit_controlled(rows, outcome="accuracy")
            self.assertEqual(result["status"], "missing_values")
            self.assertEqual(result["missing_fields"]["independent_vocabulary_id"], 1)
        rows = regression_rows()
        rows[0]["independent_vocabulary_id"] = "only-one-row-labelled"
        result = fit_controlled(rows, outcome="accuracy")
        self.assertEqual(result["status"], "missing_values")
        self.assertEqual(result["n_complete"], 1)

    def test_missing_optional_numpy_has_no_substitute_solver(self):
        import builtins
        original = builtins.__import__
        def without_numpy(name, *args, **kwargs):
            if name == "numpy":
                raise ImportError("synthetic optional dependency absence")
            return original(name, *args, **kwargs)
        with patch("builtins.__import__", side_effect=without_numpy):
            result = fit_controlled(regression_rows(), outcome="accuracy")
        self.assertEqual(result["status"], "missing_optional_numpy")
        self.assertIsNone(result["coefficients"])


class RegressionValidationTests(unittest.TestCase):
    def test_empty_and_invalid_api_inputs_are_explicit(self):
        self.assertEqual(fit_controlled([], outcome="accuracy")["status"], "no_rows")
        for rows, kwargs in (({}, {"outcome": "accuracy"}), ([1], {"outcome": "accuracy"}),
                             ([], {"outcome": ""}), ([], {"outcome": "n"}),
                             ([], {"outcome": "log_frequency"}),
                             ([], {"outcome": "accuracy", "predictor": "log2_n"}),
                             ([], {"outcome": "accuracy", "interaction": 1})):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                fit_controlled(rows, **kwargs)


if __name__ == "__main__":
    unittest.main()

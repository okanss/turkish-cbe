"""Synthetic bias diagnostics preserve gold, missing answers, and query axes."""

from copy import deepcopy
import json
import unittest

from analysis.lexical_bias.bias import paired_direction_report, summarize_bias


def observations(conditions=("canonical",)):
    rows = []
    cells = (("forward", "before", True), ("forward", "after", False),
             ("reverse", "before", False), ("reverse", "after", True))
    for index, (latent_direction, lexical_direction, gold) in enumerate(cells):
        for condition in conditions:
            rows.append({"model_id": "synthetic-model", "model_family": "fixture-family",
                "model_revision": "fixture-revision", "model_spec_sha256": "f" * 64,
                "n": 8, "split": "test", "name_control": "symbolic", "relation_id": "front_back",
                "relation_family": "spatial", "query_type": "endpoint", "query_distance": 7,
                "condition": condition, "language": "tr", "query_direction": latent_direction,
                "query_relation": lexical_direction, "gold_label": gold, "prediction": gold,
                "cluster_id": "cluster-" + str(index), "query_id": "query-" + str(index),
                "query_subject": "X" if latent_direction == "forward" else "Y",
                "query_object": "Y" if latent_direction == "forward" else "X",
                "response_status": "valid"})
    return rows


def overall(rows):
    return next(row for row in summarize_bias(rows) if row["direction_axis"] == "overall")


class BiasSummaryTests(unittest.TestCase):
    def test_balanced_planned_design_and_axes_are_explicit(self):
        source = observations()
        original = deepcopy(source)
        rows = summarize_bias(source)
        self.assertEqual(source, original)
        self.assertEqual(len(rows), 5)
        main = next(row for row in rows if row["direction_axis"] == "overall")
        self.assertEqual(main["gold_balance_status"], "balanced")
        self.assertEqual(main["four_cell_balance_status"], "balanced")
        self.assertEqual(main["n_planned"], 4)
        self.assertEqual(main["q"], 0.5)
        self.assertEqual((main["predicted_yes"], main["predicted_no"]), (2, 2))
        self.assertEqual(set(json.loads(main["query_direction_gold_cells"]).values()), {1})
        lexical = [row for row in rows if row["direction_axis"] == "query_relation"]
        latent = [row for row in rows if row["direction_axis"] == "query_direction"]
        self.assertEqual({row["query_relation"] for row in lexical}, {"before", "after"})
        self.assertTrue(all(row["query_direction"] == "all" for row in lexical))
        self.assertEqual({row["query_direction"] for row in latent}, {"forward", "reverse"})
        self.assertTrue(all(row["query_relation"] == "all" for row in latent))
        self.assertTrue(all(row["four_cell_balance_status"] == "not_applicable_fixed_direction" for row in lexical + latent))

    def test_selective_invalid_answers_do_not_erase_planned_gold_balance(self):
        rows = observations()
        for row in rows:
            if not row["gold_label"]:
                row.update(prediction=None, response_status="missing")
        result = overall(rows)
        self.assertEqual(result["gold_yes_rate_planned"], 0.5)
        self.assertEqual(result["gold_yes_rate_valid"], 1)
        self.assertEqual(result["valid_coverage"], 0.5)
        self.assertEqual(result["invalid_response_rate"], 0.5)
        self.assertEqual(result["n_missing"], 2)
        self.assertEqual(result["all_planned_accuracy"], 0.5)
        self.assertEqual(result["q"], 1)
        self.assertTrue(result["excessive_response_tendency"])
        self.assertEqual(result["response_tendency"], "excessive_yes")

    def test_no_valid_answers_have_null_q_and_null_interval(self):
        rows = observations()
        for row in rows:
            row["prediction"] = None
        result = overall(rows)
        self.assertIsNone(result["q"])
        self.assertIsNone(result["q_wilson_low"])
        self.assertIsNone(result["q_wilson_high"])
        self.assertIsNone(result["gold_yes_rate_valid"])
        self.assertIsNone(result["excessive_response_tendency"])
        self.assertEqual(result["valid_coverage"], 0)
        self.assertEqual(result["n_planned"], 4)

    def test_dependent_latent_orders_label_wilson_descriptive(self):
        rows = observations()
        rows[1]["cluster_id"] = rows[0]["cluster_id"]
        result = overall(rows)
        self.assertTrue(result["dependent_valid_observations"])
        self.assertEqual(result["n_valid_clusters"], 3)
        self.assertEqual(result["q_wilson_interpretation"], "descriptive_dependent_latent_orders")

    def test_predeclared_extreme_thresholds_include_point_nine_and_point_one(self):
        for yes, expected in ((18, True), (2, True), (17, False), (3, False)):
            rows = []
            for index in range(20):
                row = deepcopy(observations()[index % 4])
                row.update(cluster_id=f"latent-{index}", query_id=f"query-{index}", prediction=index < yes)
                rows.append(row)
            result = overall(rows)
            self.assertEqual(result["q"], yes / 20)
            self.assertIs(result["excessive_response_tendency"], expected)
            self.assertFalse(result["threshold_is_significance_test"])

    def test_imbalance_is_reported_instead_of_rebalanced_or_dropped(self):
        rows = observations()[:-1]
        result = overall(rows)
        self.assertEqual(result["gold_balance_status"], "unbalanced")
        self.assertEqual(result["four_cell_balance_status"], "unbalanced")
        self.assertEqual(result["n_planned"], 3)
        self.assertEqual(result["gold_yes_rate_planned"], 1 / 3)

    def test_protected_model_language_and_condition_cells_are_not_pooled(self):
        rows = observations(("canonical", "equivalent"))
        second = deepcopy(rows)
        for row in second:
            row.update(model_id="other-model", language="en")
        reports = [row for row in summarize_bias(rows + second) if row["direction_axis"] == "overall"]
        self.assertEqual(len(reports), 4)
        self.assertEqual({row["n_planned"] for row in reports}, {4})
        self.assertEqual({row["condition"] for row in reports}, {"canonical", "equivalent"})
        self.assertEqual({row["model_id"] for row in reports}, {"synthetic-model", "other-model"})

    def test_outputs_are_flat_and_empty_input_creates_no_result(self):
        self.assertEqual(summarize_bias([]), [])
        for row in summarize_bias(observations()):
            self.assertTrue(all(not isinstance(value, (dict, list)) for value in row.values()))

    def test_invalid_fields_and_duplicates_are_rejected(self):
        for field, value in (("prediction", 1), ("gold_label", 0), ("query_direction", "before"),
                             ("query_relation", "forward"), ("query_distance", 8), ("n", True)):
            rows = observations()
            rows[0][field] = value
            with self.subTest(field=field), self.assertRaises(ValueError):
                summarize_bias(rows)
        rows = observations()
        with self.assertRaisesRegex(ValueError, "duplicate"):
            summarize_bias(rows + [rows[0]])
        with self.assertRaises(ValueError):
            summarize_bias([{}])


class DirectionPairTests(unittest.TestCase):
    def test_p2_and_multi_transformation_equivalent_are_separately_labelled(self):
        rows = paired_direction_report(observations(("canonical", "equivalent", "P2")))
        self.assertEqual(len(rows), 8)
        p2 = [row for row in rows if row["equivalent_condition"] == "P2"]
        equivalent = [row for row in rows if row["equivalent_condition"] == "equivalent"]
        self.assertEqual(len(p2), 4)
        self.assertEqual({row["transformation_scope"] for row in p2}, {"premise_converse_semantic_preserving"})
        self.assertEqual({row["transformation_scope"] for row in equivalent}, {"multiple_presentation_changes"})
        self.assertTrue(all(row["p2_available"] for row in rows))
        self.assertEqual({row["query_relation"] for row in rows}, {"before", "after"})
        self.assertEqual({row["query_direction"] for row in rows}, {"forward", "reverse"})

    def test_missing_p2_plan_is_explicit_without_inventing_a_p2_result(self):
        reports = paired_direction_report(observations(("canonical", "equivalent")))
        self.assertEqual(len(reports), 4)
        self.assertEqual({row["equivalent_condition"] for row in reports}, {"equivalent"})
        self.assertTrue(all(row["p2_available"] is False for row in reports))
        self.assertTrue(all(row["p2_unavailable_reason"] == "P2_not_present_in_planned_observations" for row in reports))

    def test_invalid_response_stays_in_all_planned_accuracy_denominator(self):
        rows = observations(("canonical", "equivalent", "P2"))
        for row in rows:
            if row["condition"] == "P2":
                row["prediction"] = None
        p2 = [row for row in paired_direction_report(rows) if row["equivalent_condition"] == "P2"]
        self.assertTrue(all(row["n_pairs"] == 1 and row["n_valid_pairs"] == 0 for row in p2))
        self.assertTrue(all(row["original_accuracy"] == 1 and row["equivalent_accuracy"] == 0 for row in p2))
        self.assertTrue(all(row["valid_pair_coverage"] == 0 and row["R"] is None for row in p2))

    def test_cbe_components_use_shared_valid_support(self):
        rows = observations(("canonical", "equivalent"))
        for row in rows:
            if row["condition"] == "equivalent":
                row["prediction"] = not row["gold_label"]
        reports = paired_direction_report(rows)
        self.assertTrue(all(row["R"] == 0 and row["D"] == 0 and row["U"] == 1 for row in reports))
        self.assertTrue(all(row["mean_accuracy"] == 0.5 and row["invariance"] == 0 for row in reports))

    def test_changed_gold_or_query_target_is_rejected(self):
        for field in ("gold_label", "query_subject"):
            rows = observations(("canonical", "equivalent", "P2"))
            altered = next(row for row in rows if row["condition"] == "P2")
            altered[field] = not altered[field] if field == "gold_label" else "different-entity"
            with self.subTest(field=field), self.assertRaises(ValueError):
                paired_direction_report(rows)

    def test_incomplete_planned_counterpart_is_not_silently_omitted(self):
        for missing in ("canonical", "equivalent", "P2"):
            rows = observations(("canonical", "equivalent", "P2"))
            index = next(i for i, row in enumerate(rows) if row["condition"] == missing)
            rows.pop(index)
            with self.subTest(missing=missing), self.assertRaisesRegex(ValueError, "missing planned"):
                paired_direction_report(rows)

    def test_opposite_lexical_queries_are_not_used_as_equivalent_presentations(self):
        rows = observations(("canonical", "equivalent"))
        altered = next(row for row in rows if row["condition"] == "equivalent")
        altered["query_relation"] = "after"
        altered["gold_label"] = not altered["gold_label"]
        with self.assertRaises(ValueError):
            paired_direction_report(rows)
        self.assertEqual(paired_direction_report([]), [])


if __name__ == "__main__":
    unittest.main()

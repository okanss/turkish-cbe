"""Synthetic, temporary-only tests of matched reasoning-arm evaluation."""

from copy import deepcopy
import json
from pathlib import Path
import tempfile
import unittest

from reasoning.analysis import build_tables, export_figures
from test_analysis_pipeline import cohort


def arm(rows, strategy="direct", *, arm_id=None, effort=None, model_id=None):
    identifiers = {"direct": "a", "zero_shot_reasoning": "b", "structured_reasoning": "c", "provider_effort": "d"}
    output = []
    for source in rows:
        row = deepcopy(source)
        if model_id is not None:
            row.update(model_id=model_id, model_family="synthetic-second-family", model_spec_sha256="e" * 64)
        row.update(arm_id=arm_id or row["model_id"] + "--" + strategy,
                   strategy=strategy, effort=effort,
                   effective_model_spec_sha256=identifiers[strategy] * 64,
                   output_tokens=8 if strategy == "direct" else 64, reasoning_tokens=None,
                   truncated=False, finish_reason="stop", fixture_only=True,
                   response_observed=row["response_status"] != "missing")
        output.append(row)
    return output


def selected(rows, **criteria):
    matches = [row for row in rows if all(row.get(key) == value for key, value in criteria.items())]
    if len(matches) != 1:
        raise AssertionError(f"Expected one row for {criteria}; got {len(matches)}")
    return matches[0]


class ReasoningAnalysisTests(unittest.TestCase):
    def test_arm_partition_prevents_duplicate_legacy_observation_pooling(self):
        observations = []
        for strategy in ("direct", "zero_shot_reasoning", "structured_reasoning", "provider_effort"):
            observations += arm(cohort(), strategy, effort="low" if strategy == "provider_effort" else None)
        tables = build_tables(observations, repeats=50)
        self.assertEqual(len(tables["profiles"]), 8)
        self.assertEqual({row["strategy"] for row in tables["profiles"]},
                         {"direct", "zero_shot_reasoning", "structured_reasoning", "provider_effort"})
        self.assertEqual({row["n_pairs"] for row in tables["profiles"]}, {2})
        self.assertEqual(len(tables["strategy_contrasts"]), 3 * 2 * 8)
        for row in tables["profiles"]:
            self.assertIn("effective_model_spec_sha256", row)
            self.assertEqual(row["model_spec_sha256"], "a" * 64)

    def test_invalid_is_wrong_for_accuracy_strict_r_and_coverage(self):
        observations = arm(cohort()) + arm(cohort(), "zero_shot_reasoning")
        target = next(row for row in observations if row["strategy"] == "zero_shot_reasoning"
                      and row["language"] == "tr" and row["condition"] == "canonical" and row["cluster_id"].endswith("-1"))
        target.update(prediction=None, response_status="invalid", truncated=True, finish_reason="length")
        tables = build_tables(observations, repeats=50)
        def contrast(metric):
            return selected(tables["strategy_contrasts"], language="tr", metric=metric)
        self.assertEqual(contrast("original_accuracy")["difference"], -.5)
        self.assertEqual(contrast("equivalent_accuracy")["difference"], 0)
        self.assertEqual(contrast("all_planned_mean_accuracy")["difference"], -.25)
        self.assertEqual(contrast("strict_robust_correctness")["difference"], -.5)
        self.assertEqual(contrast("valid_pair_coverage")["difference"], -.5)
        self.assertEqual(contrast("R")["difference"], 0)
        self.assertEqual(contrast("R")["strategy_invalid_responses"], 1)
        self.assertEqual(contrast("R")["four_response_valid_pairs"], 1)

    def support_fixture(self, direct_first=False):
        original = cohort()
        strategy = cohort()
        for row in original:
            if row["cluster_id"].endswith("-1"):
                row["prediction"] = direct_first if row["condition"] == "canonical" else False
        for row in strategy:
            if row["cluster_id"].endswith("-1") and row["condition"] == "canonical":
                row.update(prediction=None, response_status="invalid")
        return arm(original) + arm(strategy, "zero_shot_reasoning")

    def test_four_response_sensitivity_exposes_valid_support_selection(self):
        tables = build_tables(self.support_fixture(), repeats=50)
        primary = selected(tables["strategy_contrasts"], language="tr", metric="R")
        common = selected(tables["four_response_sensitivity"], language="tr", metric="R")
        self.assertEqual(primary["direct_value"], .5)
        self.assertEqual(primary["strategy_value"], 1)
        self.assertEqual(primary["difference"], .5)
        self.assertEqual(common["difference"], 0)
        self.assertEqual(primary["direct_valid_pairs"], 2)
        self.assertEqual(primary["strategy_valid_pairs"], 1)
        self.assertEqual(common["four_response_coverage"], .5)
        self.assertFalse(primary["valid_support_equal"])
        self.assertNotEqual(primary["direct_valid_support_sha256"], primary["strategy_valid_support_sha256"])
        self.assertEqual(common["support_scope"], "same_four_response_common_valid_pairs")

    def test_common_four_answer_invariance_uses_identical_latent_pairs(self):
        tables = build_tables(self.support_fixture(direct_first=True), repeats=50)
        primary = selected(tables["strategy_contrasts"], language="en", metric="invariance")
        common = selected(tables["four_response_sensitivity"], language="en", metric="invariance")
        self.assertEqual(primary["difference"], .5)
        self.assertEqual(common["difference"], 0)

    def test_undefined_kappa_and_sparse_bootstrap_are_not_zero_filled(self):
        observations = arm(cohort(clusters=1)) + arm(cohort(clusters=1), "zero_shot_reasoning")
        tables = build_tables(observations)
        kappa = selected(tables["strategy_contrasts"], language="tr", metric="kappa")
        self.assertIsNone(kappa["difference"])
        self.assertIsNone(kappa["ci_low"])
        self.assertEqual(kappa["bootstrap_replicates"], 2000)
        self.assertEqual(kappa["bootstrap_status"], "insufficient_clusters")
        self.assertIsNone(kappa["p_value"])

    def test_matched_kappa_difference_is_computed_from_each_arm_q(self):
        direct = cohort(clusters=4)
        reasoning = cohort(clusters=4)
        patterns = ((True, True), (False, False), (True, False), (False, True))
        for row in direct:
            index = int(row["cluster_id"].rsplit("-", 1)[1])
            row["prediction"] = patterns[index][row["condition"] == "equivalent"]
        for row in reasoning:
            row["prediction"] = int(row["cluster_id"].rsplit("-", 1)[1]) % 2 == 0
        tables = build_tables(arm(direct) + arm(reasoning, "zero_shot_reasoning"), repeats=50)
        result = selected(tables["strategy_contrasts"], language="tr", metric="kappa")
        self.assertEqual(result["direct_value"], 0)
        self.assertEqual(result["strategy_value"], 1)
        self.assertEqual(result["difference"], 1)

    def test_language_strategy_interaction_and_paired_language_sign(self):
        direct = cohort({("tr", "canonical"): False, ("tr", "equivalent"): False,
                         ("en", "canonical"): False, ("en", "equivalent"): False})
        reasoning = cohort({("en", "canonical"): False, ("en", "equivalent"): False})
        tables = build_tables(arm(direct) + arm(reasoning, "structured_reasoning"), repeats=50)
        self.assertEqual(len(tables["language_strategy_interaction"]), 2)
        for row in tables["language_strategy_interaction"]:
            self.assertEqual(row["difference"], 1)
            self.assertEqual(row["n_pairs"], 2)
        self.assertEqual(selected(tables["language_differences"], strategy="direct", condition="canonical")["difference"], 0)
        self.assertEqual(selected(tables["language_differences"], strategy="structured_reasoning", condition="canonical")["difference"], 1)

    def test_tr_only_planned_ablation_is_explicitly_unavailable_cross_language(self):
        rows = cohort()
        rows += [{**row, "condition": "P4"} for row in rows if row["language"] == "tr" and row["condition"] == "equivalent"]
        tables = build_tables(arm(rows) + arm(rows, "zero_shot_reasoning"), repeats=50)
        unavailable = selected(tables["unavailable_strategy_contrasts"], condition="P4")
        self.assertEqual(unavailable["unmatched_language_units"], 2)
        self.assertFalse(any(row["condition"] == "P4" for row in tables["language_strategy_interaction"]))

    def test_holm_covers_models_and_strategies_without_per_model_reset(self):
        direct = cohort(clusters=8)
        changed = cohort({("tr", "equivalent"): False, ("en", "equivalent"): False}, clusters=8)
        observations = []
        for model in (None, "synthetic-model-b"):
            observations += arm(direct, model_id=model)
            observations += arm(changed, "zero_shot_reasoning", model_id=model)
            observations += arm(changed, "structured_reasoning", model_id=model)
        tables = build_tables(observations, repeats=50)
        main = [row for row in tables["strategy_contrasts"] if row["language"] == "tr" and row["metric"] == "equivalent_accuracy"]
        self.assertEqual(len(main), 4)
        self.assertEqual({row["holm_family_size"] for row in main}, {4})
        self.assertEqual({row["p_adjusted"] for row in main}, {8 / 256})
        presentation = [row for row in tables["presentation_tests"] if row["language"] == "tr" and row["strategy"] == "zero_shot_reasoning"]
        self.assertEqual({row["holm_family_size"] for row in presentation}, {2})
        self.assertEqual({row["p_adjusted"] for row in presentation}, {4 / 256})

    def test_repeated_queries_keep_cluster_bootstrap_and_disable_iid_mcnemar(self):
        rows = cohort(n=8, query_type="internal", distance=2)
        rows += [{**row, "query_id": str(row["query_id"]) + "-second"} for row in rows]
        tables = build_tables(arm(rows) + arm(rows, "zero_shot_reasoning"), repeats=50)
        result = selected(tables["strategy_contrasts"], language="tr", metric="original_accuracy")
        self.assertEqual(result["n_pairs"], 4)
        self.assertEqual(result["n_clusters"], 2)
        self.assertEqual(result["status"], "undefined_dependent_clusters")
        self.assertIsNone(result["p_value"])
        self.assertEqual(result["bootstrap_defined_replicates"], 50)

    def test_complexity_query_distance_and_endpoint_internal_arms_are_never_pooled(self):
        rows = cohort(n=4) + cohort(n=8)
        rows += cohort(n=8, query_type="internal", distance=1)
        rows += cohort(n=8, query_type="internal", distance=2)
        tables = build_tables(arm(rows) + arm(rows, "zero_shot_reasoning"), repeats=50)
        self.assertEqual({(row["n"], row["query_type"], row["query_distance"]) for row in tables["strategy_contrasts"]},
                         {(4, "endpoint", 3), (8, "endpoint", 7), (8, "internal", 1), (8, "internal", 2)})
        self.assertEqual(len(tables["strategy_contrasts"]), 64)

    def test_token_usage_null_coverage_and_transport_failures_are_visible(self):
        rows = arm(cohort()) + arm(cohort(), "zero_shot_reasoning")
        target = next(row for row in rows if row["strategy"] == "zero_shot_reasoning" and row["language"] == "tr"
                      and row["condition"] == "canonical" and row["cluster_id"].endswith("-1"))
        target.update(prediction=None, response_status="invalid", response_observed=False,
                      invalid_reason="provider_error", input_tokens=None, output_tokens=None)
        tables = build_tables(rows, repeats=50)
        usage = selected(tables["arm_usage"], strategy="zero_shot_reasoning", language="tr", condition="canonical")
        self.assertEqual(usage["n_provider_failed"], 1)
        self.assertEqual(usage["n_observed_responses"], 1)
        self.assertEqual(usage["input_tokens_coverage"], .5)
        self.assertIsNone(usage["reasoning_tokens_sum_reported"])
        self.assertEqual(usage["output_tokens_sum_reported"], 64)
        self.assertEqual(usage["output_tokens_mean_reported"], 64)
        contingency = [row for row in tables["truncation_invalid"] if row["strategy"] == "zero_shot_reasoning"
                       and row["language"] == "tr" and row["condition"] == "canonical"]
        self.assertEqual(sum(row["responses"] for row in contingency), 2)
        self.assertEqual({row["response_observed"] for row in contingency}, {True, False})

    def test_truncation_is_not_confused_with_other_invalid_or_missing_outputs(self):
        rows = arm(cohort()) + arm(cohort(), "zero_shot_reasoning")
        candidates = [row for row in rows if row["strategy"] == "zero_shot_reasoning" and row["language"] == "tr"]
        candidates[0].update(prediction=None, response_status="invalid", truncated=True, finish_reason="length")
        candidates[1].update(prediction=None, response_status="missing", response_observed=False,
                             input_tokens=None, output_tokens=None, finish_reason=None)
        tables = build_tables(rows, repeats=50)
        usage = [row for row in tables["arm_usage"] if row["strategy"] == "zero_shot_reasoning" and row["language"] == "tr"]
        self.assertEqual(sum(row["n_truncated"] for row in usage), 1)
        self.assertEqual(sum(row["n_missing_responses"] for row in usage), 1)
        self.assertEqual(sum(row["n_invalid_including_missing"] for row in usage), 2)

    def test_missing_arm_coordinate_is_rejected_instead_of_dropping_it(self):
        rows = arm(cohort()) + arm(cohort(), "zero_shot_reasoning")
        rows.pop()
        with self.assertRaisesRegex(ValueError, "planned coordinates"):
            build_tables(rows, repeats=50)

    def test_direct_baseline_is_required_and_unique(self):
        for rows in (arm(cohort(), "zero_shot_reasoning"),
                     arm(cohort()) + arm(cohort(), arm_id="second-direct")):
            with self.assertRaisesRegex(ValueError, "exactly one direct"):
                build_tables(rows, repeats=50)

    def test_duplicate_coordinate_or_changed_query_and_gold_is_rejected(self):
        baseline = arm(cohort()) + arm(cohort(), "zero_shot_reasoning")
        with self.assertRaisesRegex(ValueError, "duplicate"):
            build_tables(baseline + [deepcopy(baseline[-1])], repeats=50)
        for field, value in (("query_subject", "changed"), ("query_relation", "after"), ("gold_label", False)):
            rows = deepcopy(baseline)
            rows[-1][field] = value
            with self.subTest(field=field), self.assertRaisesRegex(ValueError, "changed query"):
                build_tables(rows, repeats=50)

    def test_arm_definitions_and_token_metadata_are_not_silently_coerced(self):
        baseline = arm(cohort()) + arm(cohort(), "zero_shot_reasoning")
        cases = (("truncated", 1), ("truncated", True), ("output_tokens", True),
                 ("reasoning_tokens", -1), ("effective_model_spec_sha256", "bad"),
                 ("effort", "low"), ("strategy", "unregistered"), ("response_observed", False))
        for field, value in cases:
            rows = deepcopy(baseline)
            rows[-1][field] = value
            with self.subTest(field=field, value=value), self.assertRaises(ValueError):
                build_tables(rows, repeats=50)

    def test_effective_spec_cannot_change_inside_a_model_arm(self):
        rows = arm(cohort()) + arm(cohort(), "zero_shot_reasoning")
        rows[-1]["effective_model_spec_sha256"] = "f" * 64
        with self.assertRaisesRegex(ValueError, "effective spec"):
            build_tables(rows, repeats=50)

    def test_input_order_does_not_change_paired_bootstrap_tables(self):
        rows = self.support_fixture()
        first = build_tables(rows, seed=19, repeats=50)
        second = build_tables(list(reversed(rows)), seed=19, repeats=50)
        self.assertEqual(first, second)

    def test_empty_results_create_no_tables_with_observations_or_figures(self):
        tables = build_tables([])
        self.assertTrue(all(not rows for rows in tables.values()))
        with tempfile.TemporaryDirectory() as folder:
            destination = Path(folder) / "figures"
            self.assertEqual(export_figures(tables, destination), [])
            self.assertFalse(destination.exists())

    def test_actual_pdf_svg_png_exports_keep_arms_separate_and_watermarked(self):
        tables = build_tables(arm(cohort()) + arm(cohort(), "zero_shot_reasoning"), repeats=50)
        with tempfile.TemporaryDirectory() as folder:
            paths = export_figures(tables, folder, synthetic=True)
            self.assertEqual({path.suffix for path in paths}, {".pdf", ".svg", ".png"})
            self.assertEqual({path.name.split("_")[1] for path in paths}, {"1", "2", "3", "4", "5", "6"})
            svg = [path for path in paths if path.suffix == ".svg"]
            self.assertTrue(all("SYNTHETIC TEST DATA" in path.read_text() for path in svg))
            self.assertTrue(any("zero_shot_reasoning" in path.read_text() for path in svg))
            self.assertTrue(any("strategy=direct" in path.read_text() for path in svg))
            with self.assertRaises(FileExistsError):
                export_figures(tables, folder, synthetic=True)

    def test_seed_and_bootstrap_repeats_are_validated(self):
        for kwargs in ({"seed": True}, {"seed": -1}, {"repeats": True}, {"repeats": 0}):
            with self.assertRaises(ValueError):
                build_tables([], **kwargs)


if __name__ == "__main__":
    unittest.main()

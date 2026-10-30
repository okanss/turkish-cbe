"""Synthetic-only integration tests; no research outputs are persisted."""

import copy
import json
import unittest

from analysis.pipeline import build_tables, presentation_pairs


def observation(cluster, language, condition, prediction=True, *, gold=True,
                n=4, query_type="endpoint", distance=None, query_id=None, **overrides):
    distance = n - 1 if distance is None else distance
    result = {
        "model_id": "synthetic-model-a", "model_family": "synthetic-family-a",
        "model_revision": "synthetic-revision", "model_spec_sha256": "a" * 64,
        "split": "test", "name_control": "symbolic_shared",
        "relation_id": "spatial_front", "relation_family": "spatial",
        "query_type": query_type, "query_distance": distance, "n": n,
        "cluster_id": cluster, "query_id": query_id or f"{cluster}-{query_type}-{distance}",
        "language": language, "condition": condition, "gold_label": gold,
        "prediction": prediction, "query_direction": "forward",
        "query_subject": "E001", "query_object": "E004", "query_relation": "forward",
        "response_status": "invalid" if prediction is None else "valid",
        "input_tokens": 100, "source_prompt_tokens": 90,
        "token_encoding": "synthetic-encoding",
    }
    result.update(overrides)
    return result


def cohort(predictions=None, *, clusters=2, n=4, prefix="synthetic", **kwargs):
    """Complete four-cell planned fixtures; None is preserved as a missing answer."""
    predictions = predictions or {}
    return [observation(f"{prefix}-n{n}-{i}", language, condition,
                        predictions.get((language, condition), True), n=n, **kwargs)
            for i in range(clusters) for language in ("en", "tr")
            for condition in ("canonical", "equivalent")]


def find(rows, **criteria):
    matches = [row for row in rows if all(row.get(k) == v for k, v in criteria.items())]
    if len(matches) != 1:
        raise AssertionError(f"Expected one row for {criteria}; found {len(matches)}")
    return matches[0]


class AnalysisPipelineTests(unittest.TestCase):
    def test_profiles_retain_invalid_and_missing_main_denominators(self):
        rows = cohort()
        invalid = next(row for row in rows if row["language"] == "tr"
                       and row["condition"] == "canonical"
                       and row["cluster_id"].endswith("-1"))
        invalid.update(prediction=None, response_status="missing", input_tokens=None)
        tables = build_tables(rows, repeats=50)
        profile = find(tables["profiles"], language="tr")
        self.assertEqual(profile["n_pairs"], 2)
        self.assertEqual(profile["n_valid_pairs"], 1)
        self.assertEqual(profile["n_invalid_responses"], 1)
        self.assertEqual(profile["original_accuracy"], .5)
        self.assertEqual(profile["equivalent_accuracy"], 1)
        self.assertEqual(profile["valid_pair_coverage"], .5)
        self.assertEqual(profile["invalid_response_rate"], .25)
        self.assertEqual(profile["strict_robust_correctness"], .5)
        self.assertEqual(profile["R"], 1)
        self.assertEqual(profile["mean_accuracy"], 1)
        self.assertEqual(profile["q"], 1)
        self.assertEqual(json.loads(profile["undefined_reasons"])["kappa"],
                         "constant_pooled_answer")

    def test_all_cbe_counts_and_identities_reach_profiles(self):
        rows = []
        for i, answers in enumerate(((True, True), (False, False), (True, False), (False, True))):
            for language in ("tr", "en"):
                for condition, prediction in zip(("canonical", "equivalent"), answers):
                    rows.append(observation(f"synthetic-latent-{i}", language, condition, prediction))
        for row in build_tables(rows, repeats=50)["profiles"]:
            for key in ("both_correct", "both_wrong", "org_only", "eq_only"):
                self.assertEqual(row[key], 1)
            self.assertEqual(row["R"] + row["D"] + row["U"], 1)
            self.assertEqual(row["mean_accuracy"], row["R"] + row["U"] / 2)
            self.assertEqual(row["invariance"], row["R"] + row["D"])
            self.assertEqual(row["q"], .5)
            self.assertEqual(row["kappa"], 0)
            self.assertLessEqual(row["R_wilson_low"], row["R"])
            self.assertGreaterEqual(row["R_wilson_high"], row["R"])

    def test_language_difference_and_presentation_interaction_have_correct_sign(self):
        rows = cohort({("tr", "canonical"): True, ("en", "canonical"): False,
                       ("tr", "equivalent"): None, ("en", "equivalent"): True})
        tables = build_tables(rows, repeats=50)
        self.assertEqual(find(tables["language_differences"], condition="canonical")["difference"], 1)
        self.assertEqual(find(tables["language_differences"], condition="equivalent")["difference"], -1)
        self.assertEqual(tables["language_presentation_interaction"][0]["difference"], -2)
        self.assertEqual(tables["language_robustness"][0]["difference"], 0)
        self.assertEqual(find(tables["presentation_tests"], language="tr")["difference"], -1)

    def test_strict_robust_language_difference_counts_invalid_as_failure(self):
        rows = cohort({("tr", "canonical"): True, ("tr", "equivalent"): True,
                       ("en", "canonical"): True, ("en", "equivalent"): None})
        result = build_tables(rows, repeats=50)["language_robustness"][0]
        self.assertEqual(result["difference"], 1)
        self.assertEqual(result["b"], 2)
        self.assertEqual(result["c"], 0)

    def test_endpoint_complexity_interaction_respects_structural_distance(self):
        rows = cohort(n=4) + cohort({("en", "canonical"): False,
                                   ("en", "equivalent"): False}, n=8)
        tables = build_tables(rows, repeats=50)
        self.assertEqual({(row["n"], row["query_distance"]) for row in tables["profiles"]},
                         {(4, 3), (8, 7)})
        self.assertEqual(len(tables["language_complexity_interaction"]), 2)
        for row in tables["language_complexity_interaction"]:
            self.assertEqual(row["baseline_n"], 4)
            self.assertEqual(row["n"], 8)
            self.assertEqual(row["difference"], 1)
            self.assertEqual(row["distance_policy"], "endpoint")

    def test_distance_complexity_interactions_hold_distance_fixed(self):
        rows = []
        for distance in (1, 2):
            rows += cohort(n=8, prefix=f"distance-{distance}",
                           query_type="distance_controlled", distance=distance)
            wrong_language = "en" if distance == 1 else "tr"
            rows += cohort({(wrong_language, "canonical"): False,
                            (wrong_language, "equivalent"): False}, n=16,
                           prefix=f"distance-{distance}", query_type="distance_controlled",
                           distance=distance)
        results = build_tables(rows, repeats=50)["language_complexity_interaction"]
        self.assertEqual(len(results), 4)
        for row in results:
            self.assertEqual(row["difference"], 1 if row["distance_policy"] == "1" else -1)

    def test_models_share_holm_family_without_resetting_by_family(self):
        first = cohort({("tr", "equivalent"): False}, clusters=8)
        second = [{**row, "model_id": "synthetic-model-b", "model_family": "synthetic-family-b",
                   "model_spec_sha256": "b" * 64} for row in first]
        results = build_tables(first + second, repeats=50)["presentation_tests"]
        tr = [row for row in results if row["language"] == "tr"]
        self.assertEqual(len(tr), 2)
        self.assertEqual(len({row["holm_family"] for row in tr}), 1)
        for row in tr:
            self.assertEqual(row["holm_family_size"], 2)
            self.assertEqual(row["p_value"], 2 / 256)
            self.assertEqual(row["p_adjusted"], 4 / 256)
        self.assertEqual(len({row["model_id"] for row in results}), 2)

    def test_design_axes_are_protected_and_relation_families_are_explicit(self):
        base = cohort()
        changed = [{**row, "name_control": "native_shared", "relation_id": "spatial_right"}
                   for row in base]
        results = build_tables(base + changed, repeats=50)
        self.assertEqual(len(results["profiles"]), 4)
        self.assertEqual({row["name_control"] for row in results["profiles"]},
                         {"symbolic_shared", "native_shared"})
        self.assertEqual(len(results["relation_family_profiles"]), 4)

    def test_relation_family_summary_retains_dependent_cluster_count(self):
        base = cohort()
        another_relation = [{**row, "relation_id": "spatial_right"} for row in base]
        results = build_tables(base + another_relation, repeats=50)
        self.assertEqual(len(results["profiles"]), 4)
        self.assertEqual(len(results["relation_family_profiles"]), 2)
        for row in results["relation_family_profiles"]:
            self.assertEqual(row["n_pairs"], 4)
            self.assertEqual(row["n_clusters"], 2)
            self.assertEqual(json.loads(row["relation_ids"]), ["spatial_front", "spatial_right"])

    def test_repeated_queries_are_clustered_and_disable_ordinary_mcnemar(self):
        first = cohort({("tr", "equivalent"): False}, n=8, query_type="internal", distance=2)
        second = [{**row, "query_id": row["query_id"] + "-second"} for row in first]
        results = build_tables(first + second, repeats=50)
        for row in results["profiles"]:
            self.assertEqual(row["n_pairs"], 4)
            self.assertEqual(row["n_clusters"], 2)
            self.assertEqual(row["R_wilson_interpretation"], "descriptive_dependent_pairs")
        for row in results["presentation_tests"]:
            self.assertIsNone(row["p_value"])
            self.assertEqual(row["status"], "undefined_dependent_clusters")
            self.assertEqual(row["bootstrap_defined_replicates"], 50)

    def test_endpoint_and_internal_arms_are_separate_but_matched_by_order(self):
        rows = cohort(n=8) + cohort({("tr", "canonical"): False,
                                    ("tr", "equivalent"): False}, n=8,
                                   query_type="internal", distance=2, gold=False)
        # Different questions have different gold labels; keep endpoint correct,
        # and make both languages' internal answers explicitly wrong.
        for row in rows:
            if row["query_type"] == "internal":
                row["prediction"] = True
        results = build_tables(rows, repeats=50)
        self.assertEqual(len(results["profiles"]), 4)
        self.assertEqual({row["query_type"] for row in results["profiles"]}, {"endpoint", "internal"})
        self.assertEqual(len(results["endpoint_internal"]), 4)
        for row in results["endpoint_internal"]:
            self.assertEqual(row["difference"], -1)
            self.assertEqual(row["query_distance"], 2)
            self.assertEqual(row["n_clusters"], 2)

    def test_invalid_token_relation_reports_token_and_response_coverage(self):
        rows = cohort(clusters=3)
        for row in rows:
            index = int(row["cluster_id"].rsplit("-", 1)[1])
            row["source_prompt_tokens"] = (10, 20, 30)[index]
            row["input_tokens"] = (20, 40, None)[index]
            if index:
                row.update(prediction=None, response_status="invalid" if index == 1 else "missing")
        result = build_tables(rows, repeats=50)["invalid_token_relationship"]
        self.assertEqual(len(result), 8)
        for row in result:
            self.assertEqual(row["n_planned"], 3)
            self.assertEqual(row["n_observed_with_tokens"], 2)
            self.assertEqual(row["n_missing_responses"], 1)
            self.assertEqual(row["correlation"], 1)
            self.assertEqual(row["p_value"], None)

    def test_tr_only_ablation_is_not_silently_cross_language_matched(self):
        rows = [row for row in cohort() if row["language"] == "tr"]
        rows += [{**row, "condition": "P4"} for row in rows if row["condition"] == "equivalent"]
        results = build_tables(rows, repeats=50)
        self.assertEqual({row["comparison"] for row in results["profiles"]}, {"equivalent", "P4"})
        self.assertEqual(results["language_differences"], [])
        self.assertEqual(results["language_robustness"], [])

    def test_missing_planned_condition_requires_plan_reconstruction(self):
        rows = cohort()
        rows.pop()
        with self.assertRaisesRegex(ValueError, "missing planned condition"):
            build_tables(rows, repeats=50)

    def test_duplicate_run_or_replicate_observation_is_rejected(self):
        rows = cohort()
        rows.append(dict(rows[0]))
        with self.assertRaisesRegex(ValueError, "duplicate planned observation"):
            build_tables(rows, repeats=50)

    def test_gold_or_query_change_is_rejected_for_equivalent_and_language_pairs(self):
        for key, value in (("gold_label", False), ("query_subject", "E999"),
                           ("query_direction", "reverse"), ("query_relation", "reverse")):
            with self.subTest(key=key):
                rows = cohort()
                rows[1][key] = value
                with self.assertRaisesRegex(ValueError, "changed"):
                    build_tables(rows, repeats=50)
                rows = cohort()
                for row in rows:
                    if row["language"] == "tr":
                        row[key] = value
                with self.assertRaisesRegex(ValueError, "cross-language"):
                    build_tables(rows, repeats=50)

    def test_unsupported_presentation_and_nonboolean_prediction_rejected(self):
        rows = cohort()
        rows += [{**rows[0], "condition": "unregistered-synthetic"}]
        with self.assertRaisesRegex(ValueError, "unregistered"):
            presentation_pairs(rows)
        for key in ("prediction", "gold_label"):
            rows = cohort()
            rows[0][key] = 1
            with self.subTest(key=key), self.assertRaisesRegex(ValueError, "Boolean"):
                presentation_pairs(rows)

    def test_bootstrap_is_reproducible_and_input_is_not_mutated(self):
        rows = cohort({("tr", "equivalent"): False})
        before = copy.deepcopy(rows)
        first = build_tables(rows, seed=42, repeats=50)
        second = build_tables(list(reversed(rows)), seed=42, repeats=50)
        self.assertEqual(first, second)
        self.assertEqual(rows, before)

    def test_default_2000_bootstrap_replicates_are_used(self):
        tables = build_tables(cohort(), seed=20261024)
        self.assertTrue(tables["profiles"])
        for row in tables["profiles"]:
            self.assertEqual(row["bootstrap_replicates"], 2000)
            intervals = json.loads(row["confidence_intervals"])
            self.assertEqual(intervals["R"]["repeats"], 2000)
            self.assertEqual(intervals["R"]["defined_replicates"], 2000)

    def test_empty_and_malformed_execution_controls(self):
        self.assertTrue(all(not rows for rows in build_tables([], repeats=50).values()))
        for kwargs in ({"seed": True}, {"seed": -1}, {"repeats": 0}, {"repeats": True}):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                build_tables([], **kwargs)


if __name__ == "__main__":
    unittest.main()

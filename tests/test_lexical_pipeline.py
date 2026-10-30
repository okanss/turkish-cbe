"""Synthetic joins and denominator checks for lexical model-analysis assembly.

No runner, provider, model response artifact or research report is created.
Regression input-inspection fixtures deliberately do not simulate fitted effects.
"""

from copy import deepcopy
import hashlib
import math
import unittest
from unittest.mock import patch

from analysis.lexical_bias.__main__ import model_analysis
from analysis.lexical_bias.catalog import build_catalog
import test_lexical_bias as bias_fixtures


def frequency_rows(relations=None):
    output = []
    catalog = build_catalog()
    lemmas = list(dict.fromkeys(direction["lemma"] for relation in catalog for direction in relation["directions"]))
    by_lemma = {lemma: 13 + 124 * index for index, lemma in enumerate(lemmas)}
    for relation in catalog:
        if relations is not None and relation["relation_id"] not in relations:
            continue
        for direction in relation["directions"]:
            value = by_lemma[direction["lemma"]]
            output.append({"relation_id": relation["relation_id"], "direction": direction["direction"],
                "lemma": direction["lemma"], "lemma_frequency_per_million": value,
                "lemma_log10_per_million": math.log10(value), "status": "synthetic_counted_fixture"})
    return output


def source_rows(*, conditions=("canonical", "equivalent"), models=("model-a",), sizes=(8,),
                relations=("spatial_front",), languages=("tr",)):
    catalog = {row["relation_id"]: row for row in build_catalog()}
    output = []
    for model in models:
        for n in sizes:
            for relation in relations:
                for language in languages:
                    for row in bias_fixtures.observations(conditions):
                        cell = row["query_id"].split("-")[-1]
                        row.update(model_id=model, model_spec_sha256=hashlib.sha256(model.encode()).hexdigest(),
                            n=n, query_distance=n - 1, relation_id=relation,
                            relation_family=catalog[relation]["family"], language=language,
                            cluster_id=f"latent-n{n}-cell{cell}", query_id=f"query-n{n}-{relation}-cell{cell}",
                            input_tokens=100 + n * 2 + int(cell) + (0 if row["condition"] == "canonical" else 10),
                            response_observed=True)
                        output.append(row)
    return output


class LexicalPipelineTests(unittest.TestCase):
    def inspect(self, observations, frequencies):
        calls = []
        def inspection(rows, **kwargs):
            calls.append((deepcopy(rows), dict(kwargs)))
            return {"status": "synthetic_input_inspection_only", "n_rows": len(rows),
                    "outcome": kwargs["outcome"], "coefficients": None}
        with patch("analysis.lexical_bias.__main__.fit_controlled", side_effect=inspection):
            result = model_analysis(observations, frequencies)
        return result, calls

    def test_before_after_frequency_join_uses_the_unchanged_query_direction(self):
        rows = source_rows(conditions=("canonical", "equivalent", "P2"))
        original = deepcopy(rows)
        frequencies = frequency_rows(("spatial_front",))
        result, _ = self.inspect(rows, frequencies)
        self.assertEqual(rows, original)
        by_direction = {row["direction"]: row for row in frequencies}
        self.assertEqual(len(result["lexical_profiles"]), 4)
        for row in result["lexical_profiles"]:
            reference = by_direction[row["query_relation"]]
            self.assertEqual(row["frequency"], reference["lemma_frequency_per_million"])
            self.assertEqual(row["log_frequency"], reference["lemma_log10_per_million"])
            self.assertEqual(row["independent_vocabulary_id"], reference["lemma"])
            self.assertEqual(row["vocabulary_id"], "spatial_front|" + row["query_relation"])

    def test_selective_invalids_keep_conditional_and_planned_gold_visible(self):
        rows = source_rows()
        for row in rows:
            if row["condition"] == "canonical" and not row["gold_label"]:
                row.update(prediction=None, response_status="missing", response_observed=False)
        result, _ = self.inspect(rows, frequency_rows())
        canonical = next(row for row in result["answer_bias"] if row["direction_axis"] == "overall"
                         and row["condition"] == "canonical")
        self.assertEqual(canonical["gold_yes_rate_planned"], 0.5)
        self.assertEqual(canonical["gold_yes_rate_valid"], 1)
        self.assertEqual(canonical["valid_coverage"], 0.5)
        self.assertEqual(canonical["q"], 1)
        self.assertEqual(canonical["all_planned_accuracy"], 0.5)

    def test_joint_valid_cbe_support_is_not_marginal_accuracy_support(self):
        rows = source_rows()
        for row in rows:
            if row["query_relation"] == "before" and row["query_direction"] == "reverse" and row["condition"] == "equivalent":
                row["prediction"] = True  # Correct original / wrong equivalent.
            if row["query_relation"] == "after" and row["query_direction"] == "forward" and row["condition"] == "canonical":
                row["prediction"] = None
            if row["query_relation"] == "after" and row["query_direction"] == "reverse":
                row["prediction"] = False  # Shared wrong on the only valid after pair.
        result, calls = self.inspect(rows, frequency_rows())
        before = next(row for row in result["lexical_profiles"] if row["query_relation"] == "before")
        after = next(row for row in result["lexical_profiles"] if row["query_relation"] == "after")
        self.assertEqual((before["R"], before["D"], before["U"]), (0.5, 0, 0.5))
        self.assertEqual((before["original_accuracy"], before["equivalent_accuracy"]), (1, 0.5))
        self.assertEqual((after["R"], after["D"], after["U"]), (0, 1, 0))
        self.assertEqual(after["n_pairs"], 2)
        self.assertEqual(after["n_valid_pairs"], 1)
        self.assertEqual(after["valid_pair_coverage"], 0.5)
        self.assertEqual(after["mean_accuracy"], 0)
        self.assertEqual((after["original_accuracy"], after["equivalent_accuracy"]), (0, 0.5))
        accuracy = [row for sample, kwargs in calls if kwargs["outcome"] == "accuracy" and not kwargs.get("interaction")
                    for row in sample]
        self.assertEqual(len(accuracy), len(rows))
        self.assertTrue(all(row["accuracy"] == 0 for row in accuracy if row["prediction"] is None))
        robust = next(sample for sample, kwargs in calls if kwargs["outcome"] == "robust_correct")
        self.assertEqual(len(robust), 3)
        self.assertTrue(all(row["joint_valid"] for row in robust))
        report = next(row for row in result["controlled_regressions"] if row["outcome"] == "robust_correct")
        self.assertEqual(report["analysis_support"], "joint_valid_pairs_only")
        self.assertEqual((report["n_planned_pairs"], report["n_excluded_invalid_pairs"]), (4, 1))

    def test_p2_is_separate_from_multiple_change_equivalent(self):
        result, _ = self.inspect(source_rows(conditions=("canonical", "equivalent", "P2")), frequency_rows())
        self.assertEqual({row["comparison"] for row in result["lexical_profiles"]}, {"equivalent", "P2"})
        p2 = [row for row in result["paired_direction"] if row["equivalent_condition"] == "P2"]
        equivalent = [row for row in result["paired_direction"] if row["equivalent_condition"] == "equivalent"]
        self.assertEqual(len(p2), 4)
        self.assertEqual(len(equivalent), 4)
        self.assertEqual({row["transformation_scope"] for row in p2}, {"premise_converse_semantic_preserving"})
        self.assertEqual({row["transformation_scope"] for row in equivalent}, {"multiple_presentation_changes"})

    def test_endpoint_regression_keeps_n_and_model_variation_inside_controls(self):
        rows = source_rows(models=("model-a", "model-b"), sizes=(4, 8),
                           relations=("spatial_front", "temporal_arrival"))
        result, calls = self.inspect(rows, frequency_rows())
        accuracy = [(sample, kwargs) for sample, kwargs in calls if kwargs["outcome"] == "accuracy"]
        self.assertEqual(len(accuracy), 4)  # canonical/equivalent, main/interaction.
        for sample, _ in accuracy:
            self.assertEqual({row["n"] for row in sample}, {4, 8})
            self.assertEqual({row["model_id"] for row in sample}, {"model-a", "model-b"})
            self.assertEqual({row["relation_family"] for row in sample}, {"spatial", "temporal"})
            self.assertEqual({row["query_relation"] for row in sample}, {"before", "after"})
            self.assertEqual({row["query_distance"] for row in sample}, {"n-1"})
        self.assertTrue(all(row["query_distance"] == "n-1" for row in result["controlled_regressions"]))
        self.assertTrue(all(row["query_distance"] in (3, 7) for row in rows))  # Source not mutated.

    def test_internal_query_distances_are_preserved_as_separate_controls(self):
        rows = source_rows(sizes=(8, 16))
        for row in rows:
            row.update(query_type="distance", query_distance=2)
        result, calls = self.inspect(rows, frequency_rows())
        self.assertTrue(all(row["query_distance"] == 2 for row in result["controlled_regressions"]))
        for sample, kwargs in calls:
            if kwargs["outcome"] == "accuracy":
                self.assertEqual({row["n"] for row in sample}, {8, 16})

    def test_spearman_sample_size_never_overwrites_problem_complexity(self):
        relations = tuple(row["relation_id"] for row in build_catalog())
        result, _ = self.inspect(source_rows(sizes=(4, 8), relations=relations), frequency_rows())
        self.assertEqual({row["n"] for row in result["correlations"]}, {4, 8})
        self.assertEqual({row["spearman_n"] for row in result["correlations"]}, {18})
        self.assertEqual({row["n_lexical_profiles"] for row in result["correlations"]}, {18})
        self.assertEqual({row["n_unique_lemma_units"] for row in result["correlations"]}, {14})
        self.assertTrue(all(row["p_value"] is None for row in result["correlations"]))

    def test_repeated_root_frequency_is_not_reported_as_a_new_independent_lexeme(self):
        relations = ("spatial_above", "ranking", "temporal_arrival", "temporal_completion")
        result, calls = self.inspect(source_rows(relations=relations), frequency_rows())
        profiles = result["lexical_profiles"]
        self.assertEqual(len(profiles), 8)
        self.assertEqual(len({row["vocabulary_id"] for row in profiles}), 8)
        self.assertEqual(len({row["independent_vocabulary_id"] for row in profiles}), 4)
        self.assertEqual(len({row["frequency"] for row in profiles}), 4)
        self.assertTrue(all(row["spearman_n"] == 8 and row["n_unique_lemma_units"] == 4
                            and row["p_value"] is None for row in result["correlations"]))
        for sample, kwargs in calls:
            if kwargs["outcome"] == "accuracy":
                roots = {}
                for row in sample:
                    roots.setdefault(row["independent_vocabulary_id"], set()).add(row["log_frequency"])
                self.assertTrue(all(len(values) == 1 for values in roots.values()))

    def test_frequency_missing_or_zero_is_not_replaced_with_another_direction(self):
        frequencies = frequency_rows(("spatial_front",))
        before = next(row for row in frequencies if row["direction"] == "before")
        before.update(lemma_frequency_per_million=0, lemma_log10_per_million=None)
        frequencies = [before]  # after genuinely unprovided, not measured zero.
        result, calls = self.inspect(source_rows(), frequencies)
        before_profile = next(row for row in result["lexical_profiles"] if row["query_relation"] == "before")
        after_profile = next(row for row in result["lexical_profiles"] if row["query_relation"] == "after")
        self.assertEqual(before_profile["frequency"], 0)
        self.assertIsNone(before_profile["log_frequency"])
        self.assertIsNone(after_profile["frequency"])
        self.assertEqual(after_profile["frequency_status"], "missing_corpus")
        for sample, _ in calls:
            self.assertTrue(all(row["log_frequency"] is None for row in sample))

    def test_english_observations_never_borrow_turkish_frequency_evidence(self):
        result, calls = self.inspect(source_rows(languages=("tr", "en")), frequency_rows())
        self.assertEqual({row["language"] for row in result["answer_bias"]}, {"tr", "en"})
        self.assertEqual({row["language"] for row in result["paired_direction"]}, {"tr", "en"})
        self.assertEqual({row["language"] for row in result["lexical_profiles"]}, {"tr"})
        self.assertEqual({row["language"] for row in result["correlations"]}, {"tr"})
        self.assertTrue(all(row["language"] == "tr" for sample, _ in calls for row in sample))
        english, calls = self.inspect(source_rows(languages=("en",)), frequency_rows())
        self.assertEqual(english["lexical_profiles"], [])
        self.assertEqual(english["correlations"], [])
        self.assertEqual(english["controlled_regressions"], [])
        self.assertEqual(calls, [])

    def test_p4_frequency_claim_is_explicitly_query_lemma_not_changed_attribute(self):
        # Temporal P4 changes the attribute lexeme to erken/geç, while the query
        # still uses önce/sonra. Their corpus frequencies must not be conflated.
        result, calls = self.inspect(source_rows(conditions=("canonical", "equivalent", "P4"),
            relations=("temporal_arrival",)), frequency_rows())
        p4 = [row for row in result["lexical_profiles"] if row["comparison"] == "P4"]
        self.assertEqual({row["independent_vocabulary_id"] for row in p4}, {"önce", "sonra"})
        self.assertTrue(all(row["frequency_scope"] == "canonical_query_lemma_all_senses" for row in p4))
        self.assertTrue(all(row["frequency_scope"] == "canonical_query_lemma_all_senses" for sample, _ in calls for row in sample))

    def test_missing_reported_prompt_tokens_are_not_replaced_by_proxy_counts(self):
        rows = source_rows()
        rows[0].update(input_tokens=None, source_prompt_tokens=999)
        result, calls = self.inspect(rows, frequency_rows())
        before = next(row for row in result["lexical_profiles"] if row["query_relation"] == rows[0]["query_relation"])
        self.assertIsNone(before["prompt_tokens"])
        self.assertEqual(before["n_missing_prompt_tokens"], 1)
        self.assertIn("no proxy substitution", before["prompt_token_scope"])
        accuracy = [row for sample, kwargs in calls if kwargs["outcome"] == "accuracy" for row in sample]
        self.assertTrue(any(row["prompt_tokens"] is None for row in accuracy))

    def test_empty_input_has_no_model_analysis_results(self):
        result, calls = self.inspect([], frequency_rows())
        self.assertTrue(all(rows == [] for rows in result.values()))
        self.assertEqual(calls, [])


if __name__ == "__main__":
    unittest.main()

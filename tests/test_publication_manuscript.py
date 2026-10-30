"""Reporting contracts use temporary synthetic fixtures, never research outputs."""

from copy import deepcopy
from pathlib import Path
import tempfile
import unittest

from reports.publication.manuscript import TABLES, TITLE, write_manuscript


def fixture():
    evidence = {
        "contains_model_results": False,
        "eligible_for_confirmatory_evaluation": False,
        "limitations": ["Independent human linguistic review is pending."],
    }
    tables = {stem: [{
        "status": "not_run", "sample_size": None,
        "ci_low": None, "ci_high": None, "p_adjusted": None,
    }] for stem, _ in TABLES}
    tables[TABLES[0][0]] = [{
        "status": "dataset_statistics", "split": "all", "n": None,
        "orders": 3000, "queries": 23200, "prompts": 5637600,
        "comparison_cells": 7819200,
    }]
    return evidence, tables


class PublicationManuscriptTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.paper = Path(self.temp.name) / "paper"
        self.evidence, self.tables = fixture()

    def write(self):
        return write_manuscript(self.evidence, self.tables, self.paper)

    def read(self, name):
        return (self.paper / name).read_text(encoding="utf-8")

    def test_writes_ten_artifacts_with_exact_title_and_all_sections(self):
        paths = self.write()
        self.assertEqual(len(paths), 10)
        self.assertTrue(all(path.is_file() for path in paths))
        draft = self.read("manuscript.md")
        self.assertTrue(draft.startswith("# " + TITLE + "\n"))
        for section in ("Introduction", "Related Work", "Methodology", "Experimental Setup",
                        "Results", "Discussion", "Limitations", "Conclusion"):
            self.assertIn("# " + section + "\n", draft)

    def test_missing_model_results_remain_pending_for_all_five_questions(self):
        self.write()
        result = self.read("results.md")
        self.assertIn("No verified model inference or training outcomes", result)
        self.assertEqual(result.count("the hypothesis is neither supported nor refuted"), 5)
        self.assertEqual(result.count("Pending verified model outcomes."), 6)
        self.assertNotIn("statistically significant improvement", result)
        self.assertIn("All five research questions remain", self.read("conclusion.md"))

    def test_every_research_question_links_its_prespecified_table(self):
        self.write()
        result = self.read("results.md")
        for rq, number in (("RQ1", 3), ("RQ2", 4), ("RQ3", 6), ("RQ4", 5), ("RQ5", 7)):
            row = next(line for line in result.splitlines() if line.startswith(f"| {rq} |"))
            self.assertIn(f"Table {number}:", row)

    def test_dataset_counts_are_retained_at_their_actual_units(self):
        self.write()
        setup = self.read("experimental_setup.md")
        self.assertIn("| Stratum | Split | n | Orders | Queries | Prompts |", setup)
        self.assertIn("| all | — | 3000 | 23200 | 5637600 |", setup)
        self.assertIn("independent reasoning-trial count", setup)

    def test_verified_observations_preserve_estimates_uncertainty_and_conditions(self):
        self.evidence["contains_model_results"] = True
        self.tables[TABLES[2][0]] = [{
            "status": "observed", "model_id": "unit-test-fixture-only",
            "difference": -0.125, "sample_size": 16, "n_clusters": 8,
            "ci_low": -0.25, "ci_high": 0.05, "p_adjusted": 0.25,
            "experiment_conditions": {"language_pair": "tr-en", "n": 8},
            "statistical_test": "exact McNemar",
        }]
        self.write()
        result = self.read("results.md")
        for token in ("-0.125", "-0.25", "0.05", "0.25", "exact McNemar", "t3-r00001"):
            self.assertIn(token, result)
        self.assertIn("CSV, which retains full experiment conditions", result)
        self.assertNotIn('"language_pair":', result)
        self.assertIn("1 verified result row(s)", result)
        self.assertIn("No automated support/refutation decision", result)
        self.assertNotIn("All five research questions remain unanswered", self.read("conclusion.md"))
        self.assertNotIn("At the current stage, no empirical Turkish-English gap", self.read("discussion.md"))

    def test_contradictory_no_results_flag_is_rejected_before_writing(self):
        self.tables[TABLES[1][0]][0]["status"] = "observed"
        with self.assertRaisesRegex(ValueError, "contradict"):
            self.write()
        self.assertFalse(self.paper.exists())

    def test_null_statistics_are_missing_and_not_imputed_zero(self):
        self.write()
        result = self.read("results.md")
        self.assertIn("| not_available | — |", result)
        self.assertNotIn("| not_available | 0 |", result)
        self.assertIn("not replaced with zero", result)

    def test_numeric_feature_rows_do_not_become_model_findings(self):
        self.tables[TABLES[4][0]] = [{
            "status": "preprocessing_only", "relation_token_count": 7,
            "surface_frequency": 0.001, "accuracy": None,
        }]
        self.write()
        result = self.read("results.md")
        self.assertEqual(result.count("the hypothesis is neither supported nor refuted"), 5)
        self.assertIn("preprocessing_only", result)
        self.assertNotIn("surface_frequency", result)
        self.assertNotIn("relation_token_count", result)

    def test_wide_native_metadata_does_not_expand_paper_to_eighty_columns(self):
        self.evidence["contains_model_results"] = True
        self.tables[TABLES[1][0]] = [{
            "status": "observed", "original_accuracy": 0.5,
            "sample_size": 8, "n_clusters": 4,
            **{f"unused_metadata_{i}": f"value-{i}" for i in range(80)},
        }]
        self.write()
        result = self.read("results.md")
        header = "| Stratum | Measure | Estimate | N planned / valid / clusters | 95% CI | Test | p / Holm p | Status |"
        self.assertIn(header, result)
        self.assertNotIn("unused_metadata_", result)
        self.assertIn("t2-r00001 | original_accuracy | 0.5", result)
        self.assertIn("retains full experiment conditions", result)

    def test_documented_q_support_matches_native_metric_with_an_orphan_valid_answer(self):
        from analysis.metrics import summarize_pairs
        value = summarize_pairs([
            {"original": True, "equivalent": False, "gold": True, "cluster_id": "a", "n": 4},
            {"original": True, "equivalent": None, "gold": True, "cluster_id": "b", "n": 4},
        ])
        self.assertEqual(value["q"], 0.5)
        self.write()
        self.assertIn("A lone valid answer from a pair with", self.read("methodology.md"))

    def test_metric_denominators_clustering_and_compound_interventions_are_explicit(self):
        self.write()
        methods = self.read("methodology.md")
        for text in ("invalid answer counts as wrong", "both answers are parsed",
                     "R+D+U=1", "2,000 resamples", "latent instance", "Wilson",
                     "zero denominator", "compound interventions", "P6 controlled negation",
                     "two responses of shared-valid pairs", "at a fixed complexity n"):
            self.assertIn(text, methods)

    def test_relation_direction_avoids_equating_invariance_with_accuracy(self):
        self.write()
        result = self.read("results.md")
        self.assertIn("Equal marginal accuracy can coexist", result)
        self.assertIn("high invariance can include", result)
        self.assertIn("gold balance, invalid rates, R, D, U, and kappa", result)

    def test_existing_tables_and_figures_are_accepted_but_target_files_are_immutable(self):
        (self.paper / "tables").mkdir(parents=True)
        (self.paper / "figures").mkdir()
        marker = self.paper / "tables/existing.csv"
        marker.write_text("preserved\n", encoding="utf-8")
        paths = self.write()
        original = {path: path.read_bytes() for path in paths}
        with self.assertRaises(FileExistsError): self.write()
        self.assertEqual({path: path.read_bytes() for path in paths}, original)
        self.assertEqual(marker.read_text(), "preserved\n")

    def test_preflight_existing_one_target_prevents_partial_other_outputs(self):
        self.paper.mkdir()
        marker = self.paper / "references.bib"
        marker.write_text("preserve", encoding="utf-8")
        with self.assertRaises(FileExistsError): self.write()
        self.assertEqual(list(self.paper.iterdir()), [marker])

    def test_empty_tables_and_extra_evidence_do_not_invent_observations(self):
        self.tables = {}
        self.evidence["untrusted_summary"] = "accuracy improved by 100 percent"
        self.write()
        draft = self.read("manuscript.md")
        self.assertIn("No verified table rows are available", draft)
        self.assertNotIn("accuracy improved by 100 percent", draft)

    def test_text_metadata_cannot_inject_markup_or_new_table_rows(self):
        self.tables[TABLES[0][0]][0]["split"] = "<script>bad</script>|split\nrow"
        self.write()
        result = self.read("experimental_setup.md")
        self.assertNotIn("<script>", result)
        self.assertIn("&lt;script&gt;bad&lt;/script&gt;&#124;split row", result)

    def test_large_table_excerpt_is_explicit_and_retains_row_order(self):
        self.tables[TABLES[0][0]] = [{"split": f"split-{i}", "orders": i} for i in range(20)]
        self.write()
        setup = self.read("experimental_setup.md")
        self.assertIn("the first 12 of 20 publication-view rows", setup)
        self.assertIn("No result is selected by magnitude or significance", setup)
        self.assertLess(setup.index("split-0 |"), setup.index("split-11 |"))
        self.assertNotIn("split-12 |", setup)

    def test_bibliography_uses_verified_primary_sources_and_correct_authors(self):
        self.write()
        refs = self.read("references.bib")
        self.assertIn("Başar, Ezgi and Padovani, Francesca", refs)
        for url in ("2610.22894v1", "D19-1458", "2020.acl-main.442",
                    "2025.naacl-long.59", "2025.emnlp-main.834", "2026.sigturk-1.9"):
            self.assertIn(url, refs)
        self.assertIn("not a claim to be the", self.read("introduction.md"))

    def test_non_mapping_rows_are_rejected_without_partial_files(self):
        self.tables[TABLES[1][0]] = ["not a table row"]
        with self.assertRaisesRegex(ValueError, "non-mapping"): self.write()
        self.assertFalse(self.paper.exists())

    def test_inputs_are_not_mutated(self):
        evidence, tables = deepcopy(self.evidence), deepcopy(self.tables)
        self.write()
        self.assertEqual(self.evidence, evidence)
        self.assertEqual(self.tables, tables)


if __name__ == "__main__":
    unittest.main()

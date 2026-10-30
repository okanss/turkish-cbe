"""Temporary, explicitly synthetic evidence fixtures; no research results."""

import csv
from pathlib import Path
import tempfile
import unittest

from publication.tables import TABLES, METRICS, build_paper_tables, export_paper_tables, publication_view


def evidence(**changes):
    fixture = {"contains_model_results":False, "synthetic_test_fixture":True,
        "dataset_rows":[{"split":"pilot","n":4,"orders":50,"latent_instances":50,"queries":100,"prompts":200,"comparison_cells":50,"cells_by_arm":{"endpoint":50}}],
        "dataset_metadata":{}, "conditions":[], "inference_tables":{}, "reasoning_tables":{},
        "training_tables":{}, "lexical_tables":{}}
    fixture.update(changes)
    return fixture


def profile(**changes):
    row = {"model_id":"fixture/model","model_revision":"revision-1","model_family":"fixture",
        "model_spec_sha256":"a"*64,"n":4,"query_type":"internal","query_distance":1,
        "relation_id":"front","relation_family":"spatial","name_control":"symbolic_shared",
        "language":"tr","comparison":"P4","split":"test","n_pairs":10,"n_valid_pairs":8,
        "n_clusters":10,"original_accuracy":.6,"equivalent_accuracy":.7,"R":.625,"D":.125,"U":.25,
        "invariance":.75,"R_ci_low":.3,"R_ci_high":.8,"ci_method":"synthetic_test_only",
        "invalid_response_rate":.1}
    row.update(changes)
    return row


class PublicationTableTests(unittest.TestCase):
    def test_seven_tables_and_explicit_null_not_run_outcomes(self):
        tables=build_paper_tables(evidence())
        self.assertEqual(tuple(tables),TABLES)
        self.assertEqual(tables[TABLES[0]][0]["status"],"dataset_statistics")
        for name in TABLES[1:]:
            self.assertEqual(len(tables[name]),1)
            row=tables[name][0]
            self.assertEqual(row["status"],"not_run")
            self.assertIsNone(row["sample_size"])
            self.assertIsNone(row["ci_low"])
            self.assertIsNone(row["p_value"])
            self.assertTrue(all(row[key] is None for key in METRICS))

    def test_dataset_counts_are_not_model_sample_sizes_or_scores(self):
        row=build_paper_tables(evidence())[TABLES[0]][0]
        self.assertEqual((row["n"],row["orders"],row["prompts"]),(4,50,200))
        self.assertEqual(row["cells_by_arm"],'{"endpoint":50}')
        self.assertIsNone(row["sample_size"])
        self.assertIsNone(row["R"])

    def test_empirical_rows_keep_all_protected_design_and_metric_fields(self):
        native=profile()
        row=build_paper_tables(evidence(contains_model_results=True,inference_tables={"profiles":[native]}))[TABLES[1]][0]
        self.assertEqual(row["status"],"observed")
        for field,value in native.items():self.assertEqual(row[field],value)
        self.assertEqual(row["sample_size"],10)
        self.assertEqual(row["sample_size_valid_pairs"],8)
        self.assertEqual(row["n"],4)
        self.assertIn("joint-valid",row["denominator_scope"])

    def test_rq_mapping_and_compound_morphology_limit(self):
        tables=build_paper_tables(evidence(contains_model_results=True,inference_tables={"profiles":[profile()]}))
        self.assertEqual(tables[TABLES[3]][0]["research_question"],"RQ2")
        self.assertIn("P4/P5",tables[TABLES[3]][0]["interpretation_limit"])
        self.assertEqual(tables[TABLES[5]][0]["research_question"],"RQ3")
        self.assertEqual(tables[TABLES[5]][0]["query_type"],"internal")

    def test_paired_tests_cis_adjusted_p_and_native_undefined_status_preserved(self):
        contrast=profile(difference=.1,ci_low=-.2,ci_high=.3,p_value=.08,p_adjusted=.16,
            method="exact_mcnemar",status="cluster_dependence_no_exact_test",holm_family_size=2)
        row=build_paper_tables(evidence(contains_model_results=True,
            inference_tables={"language_differences":[contrast]}))[TABLES[2]][0]
        self.assertEqual(row["statistical_test"],"exact_mcnemar")
        self.assertEqual((row["ci_low"],row["ci_high"],row["p_adjusted"]),(-.2,.3,.16))
        self.assertEqual(row["analysis_status"],"cluster_dependence_no_exact_test")

    def test_unavailable_native_contrasts_do_not_become_observed_evidence(self):
        tables=build_paper_tables(evidence(contains_model_results=True,
            inference_tables={"unavailable_contrasts":[{"n":4,"n_planned_units":10,"reason":"counterpart_not_planned"}]}))
        rows=tables[TABLES[2]]
        self.assertEqual([row["status"] for row in rows],["unavailable","not_run"])
        self.assertTrue(all(row["difference"] is None and row["p_value"] is None for row in rows))

    def test_conditions_are_canonical_json_not_an_unrecorded_default(self):
        conditions=[{"model_revision":"pin","decoding":{"temperature":0}}]
        row=build_paper_tables(evidence(conditions=conditions))[TABLES[0]][0]
        self.assertEqual(row["experiment_conditions"],'[{"decoding":{"temperature":0},"model_revision":"pin"}]')

    def test_training_cohorts_and_reasoning_arms_never_pool(self):
        tables=build_paper_tables(evidence(contains_model_results=True,
            reasoning_tables={"profiles":[profile(arm_id="reasoning")]},
            training_tables={"in_domain":{"profiles":[profile(model_id="fixture/sft")]},
                "heldout_vocabulary":{"profiles":[profile(model_id="fixture/sft")]}}))
        rows=tables[TABLES[6]]
        self.assertEqual(len(rows),3)
        self.assertEqual({row.get("transfer_cohort") for row in rows},{None,"in_domain","heldout_vocabulary"})
        self.assertEqual(rows[0]["arm_id"],"reasoning")

    def test_lexical_features_remain_preprocessing_and_outcomes_not_run(self):
        tables=build_paper_tables(evidence(dataset_metadata={"lexical_preprocessing":{
            "status":"preprocessing_only_no_model_results","corpus":{"version":"fixture"}}}))
        rows=tables[TABLES[4]]
        self.assertEqual([row["status"] for row in rows],["preprocessing_only","not_run"])
        self.assertTrue(all(row["R"] is None and row["p_value"] is None for row in rows))

    def test_native_frequency_and_token_features_never_become_model_outcomes(self):
        tables=build_paper_tables(evidence(dataset_metadata={"lexical_preprocessing":{
            "corpus_frequencies":[{"lemma_frequency_per_million":2.5,"lemma_count":3}],
            "tokenization_vocabulary":[{"model_id":"fixture","term_tokens":4,"fixture_only":False}]}}))
        rows=tables[TABLES[4]]
        self.assertEqual([row["status"] for row in rows],["preprocessing_only"]*3+["not_run"])
        self.assertEqual(rows[1]["lemma_count"],3)
        self.assertEqual(rows[2]["term_tokens"],4)
        self.assertTrue(all(row["sample_size"] is None and row["R"] is None for row in rows))
        view=publication_view(TABLES[4],rows)
        self.assertIn("lemma_count",[row["Measure"] for row in view])
        self.assertEqual(view[0]["Stratum"],"t5-r00001")

    def test_public_review_view_matches_export_row_ids(self):
        tables=build_paper_tables(evidence(contains_model_results=True,inference_tables={"profiles":[profile()]}))
        view=publication_view(TABLES[1],tables[TABLES[1]])
        self.assertTrue(all(row["Stratum"]=="t2-r00001" for row in view))
        with self.assertRaises(ValueError):publication_view("../invalid",[])

    def test_undefined_lexical_associations_and_real_coefficient_denominators(self):
        fixture=evidence(contains_model_results=True,lexical_tables={
            "correlations":[{"status":"constant_input","coefficient":None,"spearman_n":3}],
            "controlled_regressions":[{"status":"ok","coefficients":{"log_frequency":.125},"n_rows":20}]})
        rows=build_paper_tables(fixture)[TABLES[4]]
        self.assertEqual(rows[0]["status"],"unavailable")
        self.assertEqual(rows[0]["sample_size"],3)
        self.assertEqual(rows[1]["sample_size"],20)
        view=publication_view(TABLES[4],rows)
        fit=[row for row in view if row["Measure"]=="OLS: log_frequency"]
        self.assertEqual(fit[0]["Estimate"],.125)
        self.assertIsNone(fit[0]["95% CI"])

    def test_inconsistent_evidence_and_malformed_tables_are_rejected(self):
        cases=[{}, evidence(contains_model_results=1),evidence(inference_tables={"profiles":[profile()]}),
            evidence(inference_tables=[]),evidence(contains_model_results=True,training_tables={"bad":[]}),
            evidence(dataset_rows="rows"),evidence(contains_model_results=True,inference_tables={"profiles":"bad"})]
        for case in cases:
            with self.subTest(case=case), self.assertRaises(ValueError):build_paper_tables(case)

    def test_exports_all_formats_full_precision_utf8_and_scalar_escape(self):
        tables=build_paper_tables(evidence())
        row=tables[TABLES[0]][0]
        row.update(label="Ayşe & ı_|<tag>\\%\nÇığ",split="Ayşe & ı_|<tag>\\%\nÇığ",precise=.1234567890123456)
        with tempfile.TemporaryDirectory() as folder:
            paths=export_paper_tables(tables,folder)
            self.assertEqual(len(paths),21)
            self.assertEqual({path.suffix for path in paths},{".csv",".tex",".md"})
            with paths[0].open(newline="",encoding="utf-8") as handle:csvrows=list(csv.DictReader(handle))
            self.assertEqual(csvrows[0]["label"],row["label"])
            self.assertEqual(csvrows[0]["precise"],"0.1234567890123456")
            self.assertIn(r"Ayşe \& ı\_",paths[1].read_text())
            self.assertIn("&#124;&lt;tag&gt;&#92;%<br>Çığ",paths[2].read_text())
            self.assertIn("SYNTHETIC UNIT TEST",paths[1].read_text())
            self.assertIn(r"\linewidth",paths[1].read_text())
            self.assertIn(r"\allowbreak{}",paths[1].read_text())
            self.assertNotIn("experiment_conditions",paths[1].read_text())
            self.assertIn("publication_row_id",paths[0].read_text())

    def test_compact_views_use_metric_specific_intervals_not_a_generic_profile_ci(self):
        fixture=evidence(contains_model_results=True,inference_tables={"profiles":[profile(ci_low=.01,ci_high=.99)]})
        with tempfile.TemporaryDirectory() as folder:
            export_paper_tables(build_paper_tables(fixture),folder)
            text=(Path(folder)/(TABLES[1]+".md")).read_text()
            self.assertIn("[0.3, 0.8]",text)
            self.assertNotIn("[0.01, 0.99]",text)

    def test_compact_wilson_interval_labels_dependent_pair_restriction(self):
        fixture=evidence(contains_model_results=True,inference_tables={"profiles":[profile(
            R_wilson_low=.2,R_wilson_high=.9,R_wilson_interpretation="descriptive_dependent_pairs")]})
        rows=build_paper_tables(fixture)[TABLES[1]]
        r=next(row for row in publication_view(TABLES[1],rows) if row["Measure"]=="R")
        self.assertIn("Wilson",r["95% CI"])
        self.assertIn("descriptive dependent pairs",r["95% CI"])
        self.assertEqual((rows[0]["R_ci_low"],rows[0]["R_ci_high"]),(.3,.8))

    def test_export_determinism_and_collision_leaves_existing_bytes(self):
        tables=build_paper_tables(evidence())
        with tempfile.TemporaryDirectory() as folder:
            first=export_paper_tables(tables,Path(folder)/"first")
            second=export_paper_tables(tables,Path(folder)/"second")
            self.assertEqual([p.read_bytes() for p in first],[p.read_bytes() for p in second])
            saved=first[0].read_bytes()
            with self.assertRaises(FileExistsError):export_paper_tables(tables,Path(folder)/"first")
            self.assertEqual(first[0].read_bytes(),saved)

    def test_invalid_export_values_and_symlink_paths_write_nothing(self):
        for value in (float("nan"),float("inf"),[],{}):
            tables=build_paper_tables(evidence());tables[TABLES[0]][0]["bad"]=value
            with tempfile.TemporaryDirectory() as folder, self.assertRaises(ValueError):
                export_paper_tables(tables,Path(folder)/"absent")
                self.assertFalse((Path(folder)/"absent").exists())
        with tempfile.TemporaryDirectory() as folder:
            target=Path(folder)/"target";target.mkdir();link=Path(folder)/"link";link.symlink_to(target,target_is_directory=True)
            with self.assertRaises(ValueError):export_paper_tables(build_paper_tables(evidence()),link)
            self.assertEqual(list(target.iterdir()),[])


if __name__ == "__main__":unittest.main()

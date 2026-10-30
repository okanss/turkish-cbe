"""Publication evidence tests; all model observations below are temporary fixtures.

Fixture rows are never written to publication deliverables. Native journal tests
retain the test_fixture flag and prove the public collector rejects them.
"""
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from publication import evidence
from test_analysis_pipeline import cohort
import test_inference_runner as fixtures


PROJECT = Path(__file__).resolve().parents[1]


def empty_dataset(root, sources):
    return [], {"totals":{"orders":0,"queries":0,"prompts":0,"comparison_cells":0},
                "human_review_status":"pending"}


def fake_run(run_id="unit-a", *, observations=None, cohort_sha="a"*64, synthetic=False):
    rows = cohort(clusters=2) if observations is None else deepcopy(observations)
    for index,row in enumerate(rows):
        row.update(query_relation="before",response_observed=True,task_id=f"unit-task-{index}",run_id=run_id)
    return {"run_id":run_id,"synthetic":synthetic,"observations":rows,"input_files_sha256":{},
        "plan":{"source_files_sha256":{},"cohort":{"cohort_sha256":cohort_sha},
                "plan_sha256":"b"*64,"models":[],"policy":{},"environment":{}}}


class PublicationEvidenceUnitTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name).resolve()
        self.patch_dataset = patch.object(evidence,"_dataset",side_effect=empty_dataset)
        self.patch_dataset.start();self.addCleanup(self.patch_dataset.stop)

    def collect(self, **kwargs):
        options = {"inference_runs":[],"reasoning_runs":[],"training_evaluations":[],"training_runs":[],"repeats":2}
        options.update(kwargs)
        return evidence.collect_evidence(self.root,**options)

    def test_no_result_does_not_compute_success_from_gold(self):
        result = self.collect()
        self.assertFalse(result["contains_model_results"])
        self.assertFalse(result["eligible_for_confirmatory_evaluation"])
        for name in ("inference_tables","reasoning_tables","training_tables","lexical_tables"):
            self.assertEqual(result[name],{})
        self.assertEqual(result["availability"]["inference"]["observed_responses"],0)
        self.assertEqual(result["availability"]["training"]["optimizer_steps"],0)
        self.assertTrue(any("No verified model responses" in s for s in result["limitations"]))
        json.dumps(result,allow_nan=False)

    def test_explicit_empty_selection_does_not_discover_runs(self):
        with patch.object(evidence,"discover_runs",side_effect=AssertionError("must not discover")):
            self.collect(inference_runs=[])

    def test_none_selection_discovers_runs(self):
        with patch.object(evidence,"discover_runs",return_value=[]) as discover:
            self.collect(inference_runs=None)
        discover.assert_called_once_with(self.root)

    def test_duplicate_run_ids_rejected(self):
        with self.assertRaisesRegex(ValueError,"duplicate"):
            self.collect(inference_runs=["a","a"])

    def test_arbitrary_csv_path_is_not_an_observation_api(self):
        with self.assertRaisesRegex(ValueError,"lists"):
            self.collect(inference_runs="results.csv")

    def test_dictionary_observation_injection_rejected(self):
        with self.assertRaisesRegex(ValueError,"strings"):
            self.collect(inference_runs=[{"prediction":True}])

    def test_invalid_analysis_settings_rejected(self):
        for options in ({"repeats":0},{"seed":-1},{"seed":True},{"repeats":False}):
            with self.subTest(options=options), self.assertRaises(ValueError): self.collect(**options)

    def test_synthetic_reader_result_is_rejected_defensively(self):
        with patch.object(evidence,"load_run",return_value=fake_run(synthetic=True)):
            with self.assertRaisesRegex(ValueError,"synthetic"): self.collect(inference_runs=["a"])

    def test_native_reader_errors_are_not_silently_excluded(self):
        with patch.object(evidence,"load_run",side_effect=ValueError("immutable raw hash mismatch")):
            with self.assertRaisesRegex(ValueError,"hash"): self.collect(inference_runs=["a"])

    def test_mismatched_cohort_is_rejected_before_statistics(self):
        first = fake_run()
        second = fake_run("unit-b",cohort_sha="c"*64)
        for row in second["observations"]: row["model_id"]="synthetic-model-b"
        with patch.object(evidence,"load_run",side_effect=[first,second]):
            with self.assertRaisesRegex(ValueError,"cohorts differ"):
                self.collect(inference_runs=["a","b"])

    def test_same_model_coordinate_cannot_be_counted_twice_across_runs(self):
        with patch.object(evidence,"load_run",side_effect=[fake_run("a"),fake_run("b")]):
            with self.assertRaisesRegex(ValueError,"overlapping"):
                self.collect(inference_runs=["a","b"])

    def test_loaded_without_observed_responses_has_no_performance_table(self):
        item = fake_run()
        for row in item["observations"]:
            row.update(response_observed=False,prediction=None,response_status="missing")
        with patch.object(evidence,"load_run",return_value=item):
            result = self.collect(inference_runs=["a"])
        self.assertEqual(result["availability"]["inference"]["planned_responses"],8)
        self.assertFalse(result["contains_model_results"])
        self.assertEqual(result["inference_tables"],{})

    def test_planned_missing_answers_remain_in_native_statistical_denominator(self):
        item = fake_run()
        tr_original = next(row for row in item["observations"] if row["language"]=="tr" and row["condition"]=="canonical")
        tr_original.update(response_observed=False,prediction=None,response_status="missing",input_tokens=None)
        with patch.object(evidence,"load_run",return_value=item) as reader:
            result = self.collect(inference_runs=["unit-a"])
        reader.assert_called_once_with(self.root,"unit-a")  # No synthetic override.
        profile = next(r for r in result["inference_tables"]["profiles"] if r["language"]=="tr")
        self.assertEqual(profile["n_pairs"],2)
        self.assertEqual(profile["original_accuracy"],.5)
        self.assertEqual(profile["valid_pair_coverage"],.5)
        self.assertEqual(result["availability"]["inference"]["observed_responses"],7)
        self.assertTrue(result["contains_model_results"])
        json.dumps(result,allow_nan=False)

    def test_hash_mutation_is_detected_and_file_never_repaired(self):
        path = self.root/"raw.json"
        path.write_text('{"original":true}\n')
        sha = hashlib.sha256(path.read_bytes()).hexdigest()
        sources = {}
        evidence._hash(self.root,"raw.json",sources,expected=sha)
        path.write_text('{"original":false}\n')
        changed = path.read_bytes()
        with self.assertRaisesRegex(ValueError,"hash"):
            evidence._hash(self.root,"raw.json",{},expected=sha)
        self.assertEqual(path.read_bytes(),changed)
        with self.assertRaisesRegex(ValueError,"changed"):
            evidence._hash(self.root,"raw.json",sources)

    def test_symlink_input_is_rejected(self):
        path = self.root/"original.json";path.write_text('{}')
        link = self.root/"link.json";link.symlink_to(path)
        with self.assertRaisesRegex(ValueError,"symlink"):
            evidence._hash(self.root,link,{})

    def test_traversal_and_outside_training_paths_rejected(self):
        for value in ("../outside","results/../outside"):
            with self.subTest(value=value),self.assertRaises(ValueError):
                self.collect(training_runs=[value])

    def test_corrupt_lexical_baseline_never_becomes_missing_corpus(self):
        path = self.root/evidence.LEXICAL;path.parent.mkdir(parents=True)
        path.write_text(json.dumps({"artifact_type":"lexical_preprocessing_and_analysis",
            "status":"preprocessing_only_no_model_results","contains_model_results":True}))
        with self.assertRaisesRegex(ValueError,"baseline"):
            self.collect()

    def test_fixture_row_cannot_hide_inside_a_nonsynthetic_reader_result(self):
        item = fake_run()
        item["observations"][0]["fixture_only"] = True
        with patch.object(evidence,"load_run",return_value=item):
            with self.assertRaisesRegex(ValueError,"fixture row"):
                self.collect(inference_runs=["a"])

    def test_reasoning_uses_native_arm_analysis_without_merging_strategies(self):
        from test_reasoning_analysis import arm
        rows = arm(cohort(),"direct") + arm(cohort(),"structured_reasoning")
        for row in rows: row.pop("fixture_only")
        item = fake_run(observations=rows)
        with patch.object(evidence,"load_reasoning_run",return_value=item) as reader:
            result = self.collect(reasoning_runs=["unit-a"])
        reader.assert_called_once_with(self.root,"unit-a")
        self.assertEqual(len(result["reasoning_tables"]["profiles"]),4)
        self.assertEqual({r["strategy"] for r in result["reasoning_tables"]["profiles"]},{"direct","structured_reasoning"})
        self.assertEqual(result["availability"]["reasoning"]["observed_responses"],16)
        self.assertEqual(result["inference_tables"],{})

    def test_native_training_evaluation_reader_rejects_fixture_manifest(self):
        path = self.root/"results/training/evaluations/fixture";path.mkdir(parents=True)
        (path/"evaluation_manifest.json").write_text(json.dumps({"status":"completed",
            "execution_kind":"test_fixture","contains_model_results":False}))
        with self.assertRaisesRegex(ValueError,"real evaluations"):
            self.collect(training_evaluations=[path])

    def test_training_summary_without_native_verified_plan_rejected(self):
        directory = self.root/"results/training/runs/fake";directory.mkdir(parents=True)
        (directory/"training_summary.json").write_text('{"optimizer_steps":600}')
        with self.assertRaises(FileNotFoundError): self.collect(training_runs=[directory])


class PublicationEvidenceFrozenDataTests(unittest.TestCase):
    def test_actual_frozen_counts_and_preprocessing_are_not_model_results(self):
        result = evidence.collect_evidence(PROJECT,inference_runs=[],reasoning_runs=[],
            training_evaluations=[],training_runs=[],repeats=2)
        self.assertEqual(result["dataset_metadata"]["totals"]["orders"],3000)
        self.assertEqual(result["dataset_metadata"]["totals"]["queries"],23200)
        self.assertEqual(result["dataset_metadata"]["totals"]["prompts"],5637600)
        self.assertEqual(sum(r["orders"] for r in result["dataset_rows"]),3000)
        self.assertFalse(result["contains_model_results"])
        lexical = result["dataset_metadata"]["lexical_preprocessing"]
        self.assertEqual(lexical["status"],"preprocessing_only_no_model_results")
        self.assertEqual(lexical["corpus"]["coverage"]["sentences"],9761)
        self.assertEqual(len(lexical["corpus_frequencies"]),18)
        self.assertEqual(len(lexical["tokenization_vocabulary"]),216)
        self.assertTrue(all(not r["fixture_only"] for r in lexical["tokenization_vocabulary"]))
        self.assertIn("data/complete_grid/grid.sqlite",result["sources"])
        self.assertIn("data/corpora/ud_turkish_boun/r2.18/tr_boun-ud-train.conllu",result["sources"])
        self.assertTrue(any("tokenizer.json" in p for p in result["sources"]))


class PublicationEvidenceNativeFixtureTests(unittest.TestCase):
    setUpClass = classmethod(fixtures.InferenceRunnerTests.setUpClass.__func__)
    setUp = fixtures.InferenceRunnerTests.setUp
    plan = fixtures.InferenceRunnerTests.plan
    execute = fixtures.InferenceRunnerTests.execute

    def test_public_collector_rejects_native_synthetic_journal(self):
        self.execute(self.plan(),fixtures.FixtureFactory(),"fixture-publication")
        with patch.object(evidence,"_dataset",side_effect=empty_dataset):
            with self.assertRaisesRegex(ValueError,"synthetic"):
                evidence.collect_evidence(self.root,inference_runs=["fixture-publication"],
                    reasoning_runs=[],training_evaluations=[],training_runs=[],repeats=2)

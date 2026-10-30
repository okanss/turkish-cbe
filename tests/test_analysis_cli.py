"""CLI orchestration tests; all made-up responses stay in disposable folders."""

from contextlib import redirect_stdout, redirect_stderr
from copy import deepcopy
import io
import json
from pathlib import Path
import shutil
import tempfile
import unittest
from unittest.mock import patch

from analysis.__main__ import analyse, main
from test_analysis_pipeline import cohort


PROJECT = Path(__file__).resolve().parents[1]
WORK = PROJECT.parents[1] / "work"


class AnalysisCLITests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(dir=WORK)
        self.addCleanup(self.temporary.cleanup)
        self.base = Path(self.temporary.name)
        self.root = self.base / "project"
        shutil.copytree(PROJECT / "src/analysis", self.root / "src/analysis",
                        ignore=shutil.ignore_patterns("__pycache__"))
        (self.root/"results/raw").mkdir(parents=True)
        (self.root/"results/raw/README.md").write_text("No model results.")
        self.output = self.base / "analysis"

    def fixture(self, name="synthetic-run-a", *, rows=None):
        rows = deepcopy(cohort() if rows is None else rows)
        for i,row in enumerate(rows):
            row.update(task_id=f"test-only-task-{i}", response_observed=True,
                       human_review_status="pending",eligible_for_confirmatory_evaluation=False)
        return {"run_id":name,"synthetic":False,"observations":rows,"input_files_sha256":{},
            "ignored_replicates":0,"plan":{"plan_sha256":"test-only-plan",
                "cohort":{"cohort_sha256":"test-only-cohort","dataset_manifest_sha256":"test-only-dataset"}}}

    def test_empty_real_directory_produces_only_truthful_status_manifest(self):
        result=analyse(self.root,self.output,figures=True)
        self.assertEqual(result["status"],"no_model_results")
        self.assertFalse(result["contains_model_results"])
        self.assertEqual(result["n_planned_responses"],0)
        self.assertEqual(result["files"],[])
        self.assertEqual([p.name for p in self.output.iterdir()],["analysis_manifest.json"])

    def test_empty_status_is_deterministic_including_source_and_config_hashes(self):
        first=analyse(self.root,self.output)
        second=analyse(self.root,self.base/"another-output")
        self.assertEqual(first,second)
        self.assertEqual(len(first["source_files_sha256"]),7)
        self.assertEqual(first["config"]["bootstrap_repeats"],2000)

    def test_existing_output_is_preserved(self):
        self.output.mkdir()
        original=self.output/"keep.txt"
        original.write_text("original")
        with self.assertRaises(FileExistsError):
            analyse(self.root,self.output)
        self.assertEqual(original.read_text(),"original")

    def test_invalid_config_and_duplicate_runs_rejected_without_outputs(self):
        for kwargs in ({"seed":True},{"repeats":0},{"seed":-1},{"run_ids":["a","a"]}):
            with self.subTest(kwargs=kwargs),self.assertRaises(ValueError):
                analyse(self.root,self.output,**kwargs)
        self.assertFalse(self.output.exists())

    def test_protected_input_destinations_are_rejected(self):
        for name in ("data","src","tests","configs","logs","results/raw","results/normalized"):
            with self.subTest(name=name),self.assertRaises(ValueError):
                analyse(self.root,self.root/name/"new-analysis")

    def test_nonempty_synthetic_data_are_rejected_by_cli_even_if_reader_is_mocked(self):
        run=self.fixture()
        run["synthetic"]=True
        with patch("analysis.__main__.load_run",return_value=run),self.assertRaisesRegex(ValueError,"synthetic"):
            analyse(self.root,self.output,run_ids=[run["run_id"]])
        self.assertFalse(self.output.exists())

    def test_no_response_failure_journal_has_counts_but_no_performance_tables(self):
        run=self.fixture()
        for row in run["observations"]:
            row.update(response_observed=False,prediction=None,response_status="missing")
        with patch("analysis.__main__.load_run",return_value=run):
            result=analyse(self.root,self.output,run_ids=[run["run_id"]])
        self.assertEqual(result["status"],"no_model_results")
        self.assertEqual(result["n_planned_responses"],8)
        self.assertEqual(result["n_missing_responses"],8)
        self.assertEqual(result["n_invalid_responses"],8)
        self.assertEqual(result["files"],[])

    def test_one_missing_planned_response_is_retained_in_report(self):
        run=self.fixture()
        run["observations"][0].update(response_observed=False,prediction=None,response_status="missing")
        with patch("analysis.__main__.load_run",return_value=run):
            result=analyse(self.root,self.output,run_ids=[run["run_id"]],repeats=20)
        self.assertEqual(result["n_missing_responses"],1)
        self.assertEqual(result["n_planned_responses"],8)
        self.assertEqual(result["table_rows"]["profiles"],2)
        self.assertFalse(result["eligible_for_confirmatory_evaluation"])
        self.assertTrue(all((self.output/f["path"]).is_file() for f in result["files"]))
        self.assertEqual({Path(f["path"]).suffix for f in result["files"]},{".csv",".tex"})

    def test_overlapping_runs_are_rejected(self):
        one=self.fixture("synthetic-one")
        two=self.fixture("synthetic-two")
        with patch("analysis.__main__.load_run",side_effect=[one,two]),self.assertRaisesRegex(ValueError,"overlapping"):
            analyse(self.root,self.output,run_ids=[one["run_id"],two["run_id"]])

    def test_mismatched_cohorts_are_rejected(self):
        one=self.fixture("synthetic-one")
        two=self.fixture("synthetic-two")
        two["plan"]["cohort"]["cohort_sha256"]="different"
        with patch("analysis.__main__.load_run",side_effect=[one,two]),self.assertRaisesRegex(ValueError,"cohorts"):
            analyse(self.root,self.output,run_ids=[one["run_id"],two["run_id"]])

    def test_export_failure_is_atomic_and_inputs_stay_unchanged(self):
        run=self.fixture()
        before=deepcopy(run)
        with patch("analysis.__main__.load_run",return_value=run),patch("analysis.__main__.export_tables",side_effect=ValueError("test-only export failure")),self.assertRaisesRegex(ValueError,"export failure"):
            analyse(self.root,self.output,run_ids=[run["run_id"]],repeats=10)
        self.assertFalse(self.output.exists())
        self.assertEqual(list(self.output.parent.glob(".analysis-*")),[])
        self.assertEqual(run,before)

    def test_analysis_code_drift_is_rejected(self):
        (self.root/"src/analysis/__init__.py").write_text("# different implementation")
        with self.assertRaisesRegex(ValueError,"analysis code differs"):
            analyse(self.root,self.output)

    def test_cli_empty_directory_success_and_error_exit_codes(self):
        with redirect_stdout(io.StringIO()) as output:
            status=main(["--root",str(self.root),"--output",str(self.output)])
        self.assertEqual(status,0)
        self.assertEqual(json.loads(output.getvalue())["status"],"no_model_results")
        with redirect_stderr(io.StringIO()) as error:
            status=main(["--root",str(self.root),"--output",str(self.output)])
        self.assertEqual(status,2)
        self.assertIn("already exists",error.getvalue())

    def test_symlink_destination_is_rejected(self):
        (self.base/"alias").symlink_to(self.root,target_is_directory=True)
        with self.assertRaisesRegex(ValueError,"symlinks"):
            analyse(self.root,self.base/"alias/new-output")


if __name__=="__main__":
    unittest.main()

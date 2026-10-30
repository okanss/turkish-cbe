"""Publication boundary tests use temporary evidence and never model results."""

from contextlib import ExitStack
import hashlib
import json
from pathlib import Path
import shutil
import tempfile
import unittest
from unittest.mock import patch

from publication.__main__ import build_report, main, rq_coverage
from publication.tables import TABLES


class PublicationBoundaryTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name).resolve()
        project = Path(__file__).resolve().parents[1]
        shutil.copytree(project / "reports/publication", self.root / "reports/publication",
                        ignore=shutil.ignore_patterns("__pycache__"))
        (self.root / "src").mkdir()
        (self.root / "src/core.py").write_text("# fixture source\n")
        (self.root / "pyproject.toml").write_text('[project]\nversion="0.10.0"\n')
        (self.root / "input.json").write_text('{"fixture":"metadata only"}\n')
        self.evidence = {
            "dataset_rows": [{"split":"pilot", "n":4, "orders":4, "queries":4, "prompts":972, "comparison_cells":1404}],
            "dataset_metadata": {}, "inference_tables":{}, "reasoning_tables":{},
            "training_tables":{}, "lexical_tables":{},
            "sources":{"input.json":hashlib.sha256((self.root / "input.json").read_bytes()).hexdigest()},
            "availability":{"inference":{"runs":0,"observed_responses":0}}, "conditions":[],
            "limitations":["Fixture metadata only; no model responses."],
            "contains_model_results":False, "eligible_for_confirmatory_evaluation":False}

    def fake_tables(self, tables, directory):
        directory.mkdir(parents=True)
        path = directory / "fixture.csv"; path.write_text("status\nnot_run\n")
        return [path]

    def fake_figures(self, evidence, tables, directory):
        directory.mkdir(parents=True)
        path = directory / "fixture.md"; path.write_text("No empirical figure.\n")
        return [path]

    def fake_manuscript(self, evidence, tables, directory):
        path = directory / "results.md"; path.write_text("Results pending.\n")
        return [path]

    def mocks(self, *, mutate=False):
        stack = ExitStack()
        stack.enter_context(patch("publication.__main__.collect_evidence", return_value=self.evidence))
        stack.enter_context(patch("publication.__main__.export_paper_tables", side_effect=self.fake_tables))
        stack.enter_context(patch("publication.__main__.export_paper_figures", side_effect=self.fake_figures))
        def manuscript(*args):
            result = self.fake_manuscript(*args)
            if mutate: (self.root / "input.json").write_text("changed input\n")
            return result
        stack.enter_context(patch("publication.__main__.write_manuscript", side_effect=manuscript))
        return stack

    def test_new_report_binds_sources_inputs_outputs_and_pending_questions(self):
        original = (self.root / "input.json").read_bytes()
        with self.mocks(): manifest = build_report(self.root)
        self.assertEqual(manifest["status"], "results_pending")
        self.assertFalse(manifest["contains_model_results"])
        self.assertFalse(manifest["submission_ready"])
        self.assertEqual(len(manifest["rq_coverage"]), 5)
        self.assertTrue(all(row["status"] == "results_pending" for row in manifest["rq_coverage"]))
        self.assertTrue(all(row["hypothesis_supported"] is None for row in manifest["rq_coverage"]))
        self.assertEqual(manifest["actual_generation_calls_by_reporting"], 0)
        self.assertEqual(manifest["actual_optimizer_steps_by_reporting"], 0)
        self.assertEqual((self.root / "input.json").read_bytes(), original)
        self.assertTrue((self.root / "paper/results.md").is_file())
        for item in manifest["files"]:
            self.assertEqual(hashlib.sha256((self.root / item["path"]).read_bytes()).hexdigest(), item["sha256"])
        self.assertEqual(json.loads((self.root / "reports/current/report_manifest.json").read_bytes()), manifest)

    def test_existing_output_is_never_overwritten_or_collected(self):
        (self.root / "paper").mkdir()
        with patch("publication.__main__.collect_evidence") as collect:
            with self.assertRaises(FileExistsError): build_report(self.root)
            collect.assert_not_called()

    def test_protected_code_data_and_report_source_destinations_rejected(self):
        for target in ("src/new", "data/new", "reports/publication/new", "reports", "../escape"):
            with self.subTest(target=target), self.assertRaises(ValueError):
                build_report(self.root, paper_dir=target)
        with self.assertRaises(ValueError): build_report(self.root, report_dir="other_reports")

    def test_symlink_destination_rejected_before_collection(self):
        (self.root / "link").symlink_to(self.root / "src", target_is_directory=True)
        with self.assertRaisesRegex(ValueError, "symlink"):
            build_report(self.root, paper_dir="link/new")

    def test_paper_and_status_directory_overlap_rejected(self):
        with self.assertRaises(ValueError):
            build_report(self.root, paper_dir="reports/new", report_dir="reports/new/nested")

    def test_input_change_aborts_without_publishing_partial_paper(self):
        with self.mocks(mutate=True), self.assertRaisesRegex(ValueError, "input changed"):
            build_report(self.root)
        self.assertFalse((self.root / "paper").exists())
        self.assertFalse((self.root / "reports/current").exists())
        self.assertFalse(list((self.root / "reports").glob(".publication-*")))

    def test_reporting_uses_required_bootstrap_protocol(self):
        for repeats in (0, 10, 1999, True):
            with self.assertRaises(ValueError): build_report(self.root, repeats=repeats)
        with self.assertRaises(ValueError): build_report(self.root, seed=-1)

    def test_changed_installed_source_cannot_be_silently_used(self):
        (self.root / "reports/publication/__init__.py").write_text("changed=1\n")
        with self.assertRaisesRegex(ValueError, "installed publication source"):
            build_report(self.root)

    def test_preprocessing_only_rows_do_not_answer_a_research_question(self):
        tables = {name:[{"status":"not_run"}] for name in TABLES}
        tables[TABLES[4]].insert(0,{"status":"preprocessing_only"})
        coverage = rq_coverage(self.evidence,tables)
        self.assertTrue(all(row["status"] == "results_pending" for row in coverage))

    def test_cli_has_no_arbitrary_result_csv_or_fixture_override(self):
        with self.assertRaises(SystemExit): main(["--csv", "unverified.csv"])
        with self.assertRaises(SystemExit): main(["--allow-synthetic"])


if __name__ == "__main__":
    unittest.main()

"""Offline CLI/provenance checks, with all derived fixtures in temporary work."""

from contextlib import redirect_stdout, redirect_stderr
from copy import deepcopy
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from reasoning.__main__ import main, analyse
from reasoning.planning import DEFAULT_SETTINGS
import test_reasoning_runner as fixture


class ReasoningCliTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        fixture.ReasoningRunnerTests.setUpClass.__func__(cls)

    def setUp(self):
        fixture.ReasoningRunnerTests.setUp(self)
        configs = self.root / "configs"
        configs.mkdir()
        # Original example registry is copied, never enabled by these tests.
        # Live model selections must not leak into the fixture, so every
        # enabled flag is normalized back to the disabled example state.
        registry = (fixture.PROJECT / "configs/models.yaml").read_bytes()
        (configs / "models.yaml").write_bytes(registry.replace(b"enabled: true", b"enabled: false"))
        (configs / "reasoning.toml").write_bytes((fixture.PROJECT / "configs/reasoning.toml").read_bytes())

    def invoke(self, args):
        out = io.StringIO()
        with redirect_stdout(out), patch("urllib.request.urlopen", side_effect=AssertionError("no network")):
            status = main(args)
        return status, json.loads(out.getvalue())

    def test_pilot_is_dry_run_and_requires_explicit_model(self):
        args = ["pilot", "--project-root", str(self.root), "--model", "qwen25-05b-hf",
                "--sizes", "4", "--groups-per-size", "4", "--output", "plan.json"]
        status, report = self.invoke(args)
        self.assertEqual(status, 0)
        self.assertTrue(report["dry_run"])
        self.assertEqual(report["actual_generation_calls"], 0)
        self.assertEqual(report["planned_generation_calls"], 144)  # 3 query strata ×16×3 arms.
        self.assertEqual(len(report["unavailable_arms"]), 4)
        self.assertFalse((self.root / "results/reasoning/raw").exists())
        self.assertTrue((self.root / "plan.json").exists())

    def test_saved_plan_replays_without_model_or_config_override(self):
        self.invoke(["pilot", "--project-root", str(self.root), "--model", "qwen25-05b-hf",
                     "--sizes", "4", "--output", "plan.json"])
        status, report = self.invoke(["pilot", "--project-root", str(self.root), "--plan", "plan.json"])
        self.assertEqual(status, 0)
        self.assertEqual(report["actual_generation_calls"], 0)
        for flag in ("--settings", "--models", "--dataset-manifest"):
            with redirect_stderr(io.StringIO()), self.assertRaises(SystemExit) as caught:
                main(["pilot", "--project-root", str(self.root), "--plan", "plan.json", flag, "ignored"])
            self.assertEqual(caught.exception.code, 2)

    def test_execute_without_plan_and_disabled_plan_are_blocked(self):
        with redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
            main(["pilot", "--project-root", str(self.root), "--execute", "--model", "qwen25-05b-hf"])
        self.invoke(["pilot", "--project-root", str(self.root), "--model", "qwen25-05b-hf", "--output", "plan.json"])
        with redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
            main(["pilot", "--project-root", str(self.root), "--plan", "plan.json", "--execute"])
        self.assertFalse((self.root / "results/reasoning/raw").exists())

    def test_plan_and_analysis_never_overwrite(self):
        args = ["pilot", "--project-root", str(self.root), "--model", "qwen25-05b-hf", "--output", "plan.json"]
        self.invoke(args)
        before = (self.root / "plan.json").read_bytes()
        with redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
            main(args)
        self.assertEqual((self.root / "plan.json").read_bytes(), before)
        output = self.root / "results/reasoning/empty"
        analyse(self.root, output)
        with self.assertRaises(FileExistsError):
            analyse(self.root, output)

    def test_empty_analysis_contains_no_metrics_or_figures(self):
        output = self.root / "results/reasoning/empty"
        status, report = self.invoke(["analyse", "--project-root", str(self.root),
                                     "--output", str(output), "--figures"])
        self.assertEqual(status, 0)
        self.assertEqual(report["status"], "no_model_results")
        self.assertFalse(report["contains_model_results"])
        self.assertEqual(report["files"], [])
        self.assertEqual(report["table_rows"], {})
        self.assertEqual(report["n_planned_responses"], 0)
        self.assertEqual([p.name for p in output.iterdir()], ["analysis_manifest.json"])

    def test_analysis_refuses_protected_output_and_repeated_run_ids(self):
        for destination in ("data/new", "src/new", "results/reasoning/raw/new", "results/reasoning/plans/new"):
            with self.assertRaisesRegex(ValueError, "protected"):
                analyse(self.root, self.root / destination)
        with self.assertRaisesRegex(ValueError, "duplicates"):
            analyse(self.root, self.root / "fresh", run_ids=["same", "same"])

    def test_fixture_results_cannot_be_published_by_cli(self):
        # Run the same explicitly synthetic fixture used in the runner suite.
        instance = fixture.ReasoningRunnerTests(methodName="test_dry_run_has_no_provider_or_journal")
        instance.__dict__.update(self.__dict__)
        instance.execute()
        with self.assertRaisesRegex(ValueError, "synthetic"):
            analyse(self.root, self.root / "cannot-publish")
        self.assertFalse((self.root / "cannot-publish").exists())


if __name__ == "__main__":
    unittest.main()

"""Command-line tests never send a provider request or download weights."""

import contextlib
import io
import json
from pathlib import Path
import tempfile
import unittest

from inference.cli import stage_main, main
from inference.planning import make_plan
from inference.config import load_models, select_models


ROOT = Path(__file__).resolve().parents[1]


class InferenceCliTests(unittest.TestCase):
    def test_explicit_execution_cannot_skip_saved_dry_run(self):
        with contextlib.redirect_stderr(io.StringIO()) as stream:
            with self.assertRaises(SystemExit) as error:
                stage_main("pilot", ["--project-root", str(ROOT), "--execute"])
        self.assertEqual(error.exception.code, 2)
        self.assertIn("requires --plan", stream.getvalue())

    def test_model_selection_is_explicit(self):
        with contextlib.redirect_stderr(io.StringIO()):
            with self.assertRaises(SystemExit) as error:
                stage_main("pilot", ["--project-root", str(ROOT)])
        self.assertEqual(error.exception.code, 2)

    def test_fixture_controls_are_not_public_cli_flags(self):
        with contextlib.redirect_stderr(io.StringIO()):
            with self.assertRaises(SystemExit) as error:
                stage_main("pilot", ["--test-fixture"])
        self.assertEqual(error.exception.code, 2)

    def test_only_dry_run_output_is_created_and_never_overwritten(self):
        try:
            registry = load_models(ROOT/"configs/models.yaml")
        except RuntimeError as error:
            if "PyYAML" in str(error):
                self.skipTest("optional PyYAML is absent")
            raise
        with tempfile.TemporaryDirectory() as temp:
            output = Path(temp)/"plan.json"
            arguments = ["--project-root", str(ROOT), "--model", "qwen25-05b-hf", "llama32-1b-hf",
                         "--output", str(output)]
            with contextlib.redirect_stdout(io.StringIO()) as stream:
                code = stage_main("pilot", arguments)
            result = json.loads(stream.getvalue())
            self.assertEqual(code, 0)
            self.assertEqual(result["actual_generation_calls"], 0)
            self.assertEqual(result["planned_generation_calls"], 64)
            original = output.read_bytes()
            with contextlib.redirect_stderr(io.StringIO()):
                with self.assertRaises(SystemExit):
                    stage_main("pilot", arguments)
            self.assertEqual(output.read_bytes(), original)

    def test_approval_does_not_exist_for_an_unrun_pilot(self):
        with tempfile.TemporaryDirectory() as temp, contextlib.redirect_stderr(io.StringIO()):
            output = Path(temp)/"approval.json"
            with self.assertRaises(SystemExit) as error:
                main(["approve-pilot", "--project-root", temp, "--run-id", "no-real-run",
                      "--approved-by", "Fixture reviewer", "--linguistic-review", str(Path(temp)/"review.json"),
                      "--output", str(output)])
            self.assertEqual(error.exception.code, 2)
            self.assertFalse(output.exists())

    def test_unknown_flags_fail_before_any_work(self):
        with contextlib.redirect_stderr(io.StringIO()):
            with self.assertRaises(SystemExit) as error:
                stage_main("full", ["--unknown-provider-fallback", "yes"])
        self.assertEqual(error.exception.code, 2)

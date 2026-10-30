"""CLI checks stay offline and expose no experiment execution command."""

from contextlib import redirect_stderr, redirect_stdout
from io import StringIO
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from turkish_cbe.__main__ import main


PROJECT_ROOT = Path(__file__).resolve().parents[1]


class CliTests(unittest.TestCase):
    def setUp(self):
        self.folder = tempfile.TemporaryDirectory()
        self.addCleanup(self.folder.cleanup)
        self.root = Path(self.folder.name)
        self.config = self.root / "pilot.toml"
        self.config.write_bytes((PROJECT_ROOT / "configs" / "pilot.toml").read_bytes())

    def invoke(self, args):
        stdout, stderr = StringIO(), StringIO()
        with redirect_stdout(stdout), redirect_stderr(stderr):
            code = main([str(arg) for arg in args])
        return code, stdout.getvalue(), stderr.getvalue()

    def test_validate_reports_planning_budget_without_executing_experiment(self):
        code, stdout, stderr = self.invoke(["validate", self.config])
        self.assertEqual(code, 0)
        self.assertEqual(stderr, "")
        result = json.loads(stdout)
        self.assertEqual(result["stage"], "pilot")
        self.assertEqual(result["status"], "planning_only")
        self.assertIs(result["execution_ready"], False)
        self.assertEqual(result["budget"]["total_planned_requests"], 240)
        self.assertTrue(result["pending_gates"])
        self.assertEqual(set(self.root.iterdir()), {self.config})

    def test_missing_config_and_invalid_toml_return_two_with_no_success_json(self):
        malformed = self.root / "malformed.toml"
        malformed.write_text("stage = [\n", encoding="utf-8")
        for path in (self.root / "missing.toml", malformed):
            with self.subTest(path=path):
                code, stdout, stderr = self.invoke(["validate", path])
                self.assertEqual(code, 2)
                self.assertEqual(stdout, "")
                self.assertIn("error:", stderr)

    def test_run_command_unknown_arguments_and_missing_command_are_rejected(self):
        for args in (
            [], ["run", str(self.config)],
            ["validate", str(self.config), "--unknown"],
        ):
            with self.subTest(args=args):
                stdout, stderr = StringIO(), StringIO()
                with redirect_stdout(stdout), redirect_stderr(stderr):
                    with self.assertRaises(SystemExit) as raised:
                        main(args)
                self.assertEqual(raised.exception.code, 2)
                self.assertEqual(stdout.getvalue(), "")
                self.assertIn("error:", stderr.getvalue())

    def test_manifest_requires_a_project_root_containing_pyproject(self):
        output = self.root / "plan.json"
        code, stdout, stderr = self.invoke([
            "manifest", self.config, "--output", output, "--project-root", self.root,
        ])
        self.assertEqual(code, 2)
        self.assertEqual(stdout, "")
        self.assertIn("--project-root must contain pyproject.toml", stderr)
        self.assertFalse(output.exists())

    def test_manifest_defaults_to_current_directory_as_project_root(self):
        output = self.root / "plan.json"
        with patch("turkish_cbe.__main__.Path.cwd", return_value=self.root):
            code, stdout, stderr = self.invoke([
                "manifest", self.config, "--output", output,
            ])
        self.assertEqual(code, 2)
        self.assertEqual(stdout, "")
        self.assertIn("--project-root", stderr)
        self.assertFalse(output.exists())

    def test_manifest_writes_a_plan_and_existing_output_is_preserved(self):
        (self.root / "pyproject.toml").write_text(
            "[project]\nname = 'fixture'\n", encoding="utf-8",
        )
        output = self.root / "outputs" / "plan.json"
        args = ["manifest", self.config, "--output", output, "--project-root", self.root]
        with patch("turkish_cbe.provenance.importlib.metadata.distributions", return_value=[]):
            with patch("turkish_cbe.provenance._git_state", return_value={"status": "fixture"}):
                code, stdout, stderr = self.invoke(args)
                self.assertEqual(code, 0)
                self.assertEqual(stderr, "")
                result = json.loads(stdout)
                self.assertEqual(result["status"], "planning_only")
                self.assertEqual(result["manifest"], str(output))
                original = output.read_bytes()
                record = json.loads(original)
                self.assertEqual(record["status"], "planning_only")
                self.assertIs(record["contains_model_results"], False)
                self.assertIsNone(record["dataset"])
                code, stdout, stderr = self.invoke(args)
        self.assertEqual(code, 2)
        self.assertEqual(stdout, "")
        self.assertIn("error:", stderr)
        self.assertEqual(output.read_bytes(), original)


if __name__ == "__main__":
    unittest.main()

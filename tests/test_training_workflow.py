"""Frozen-data planning, no-results reports and execution gates; no model calls."""

from contextlib import redirect_stdout, redirect_stderr
from copy import deepcopy
import io
import csv
import json
from pathlib import Path
import shutil
import unittest
from unittest.mock import patch

import test_inference_runner as fixtures
from inference.dataset import fingerprint
from training.__main__ import main
from training.config import load_config
from training.engine import run_training, execution_gate, TrainingBlocked
from training.evaluation import evaluate, planned_calls, adapter_identity, load_evaluation
from training.planning import make_plan, validate_plan, write_plan
from training.reporting import report


PROJECT = Path(__file__).resolve().parents[1]


class TrainingWorkflowTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        fixtures.InferenceRunnerTests.setUpClass.__func__(cls)

    def setUp(self):
        fixtures.InferenceRunnerTests.setUp(self)
        shutil.copytree(PROJECT / "configs", self.root / "configs")
        self.config = load_config(self.root / "configs/training/pilot.toml")
        self.plan = make_plan(self.root, self.config)

    def invoke(self, args):
        out, err = io.StringIO(), io.StringIO()
        with redirect_stdout(out), redirect_stderr(err):
            status = main(["--project-root", str(self.root), *args])
        return status, json.loads(out.getvalue()) if out.getvalue() else err.getvalue()

    def test_plan_has_six_unrun_arms_and_unknown_measured_costs(self):
        plan = self.plan
        self.assertEqual(len(plan["arm_budgets"]), 6)
        self.assertFalse(plan["contains_training_run"])
        self.assertEqual(plan["actual_optimizer_steps"], 0)
        self.assertIsNone(plan["estimated_gpu_hours"])
        self.assertIsNone(plan["estimated_cost"])
        self.assertIn("execution_not_enabled", plan["blockers"])
        self.assertIn("model_registry_entry_disabled", plan["blockers"])
        self.assertTrue(all(row["actual_gpu_seconds"] is None for row in plan["arm_budgets"]))

    def test_full_reference_and_step_control_preserve_different_budgets(self):
        cfg = load_config(self.root / "configs/training/reference_inspired.toml")
        reference = make_plan(self.root, cfg)
        values = {row["arm"]: row for row in reference["arm_budgets"]}
        self.assertEqual(values["sft_lora"]["planned_optimizer_steps"], 357)
        self.assertEqual(values["paired_sft"]["planned_optimizer_steps"], 713)
        self.assertEqual(values["grpo_consistency"]["planned_optimizer_steps"], 600)
        self.assertEqual(reference["data"]["counts"]["training_units"], 5700)
        self.assertFalse(reference["fairness"]["matched_optimizer_steps"])
        matched = make_plan(self.root, load_config(self.root / "configs/training/matched_steps.toml"))
        self.assertEqual({row["planned_optimizer_steps"] for row in matched["arm_budgets"] if row["arm"] != "base"}, {600})
        self.assertTrue(matched["fairness"]["matched_optimizer_steps"])
        self.assertFalse(matched["fairness"]["matched_token_budget"])

    def test_rehashed_derived_budget_is_not_trusted(self):
        value = deepcopy(self.plan); value["arm_budgets"][1]["planned_optimizer_steps"] = 0
        value["plan_sha256"] = fingerprint({k:v for k,v in value.items() if k != "plan_sha256"})
        with self.assertRaisesRegex(ValueError, "budget"): validate_plan(value)

    def test_plans_are_immutable(self):
        path = self.root / "plan.json"
        write_plan(self.plan, path)
        before = path.read_bytes()
        with self.assertRaises(FileExistsError): write_plan(self.plan, path)
        self.assertEqual(path.read_bytes(), before)

    def test_dry_run_does_not_import_a_backend_or_create_a_run(self):
        with patch("training.engine.load_runtime", side_effect=AssertionError("no model")):
            result = run_training(self.root, self.plan, "sft_lora", self.root / "unused")
        self.assertEqual(result["status"], "dry_run")
        self.assertEqual(result["actual_optimizer_steps"], 0)
        self.assertFalse((self.root / "unused").exists())

    def test_training_is_blocked_before_loading_weights(self):
        with patch("training.engine.load_runtime", side_effect=AssertionError("no model")):
            with self.assertRaises(TrainingBlocked):
                run_training(self.root, self.plan, "grpo_consistency", self.root / "unused", execute=True)
        self.assertFalse((self.root / "unused").exists())

    def test_base_never_becomes_a_training_run(self):
        with self.assertRaisesRegex(TrainingBlocked, "base has no training"):
            run_training(self.root, self.plan, "base", self.root / "unused", execute=True)

    def test_evaluation_dry_run_preserves_full_heldout_call_budget(self):
        value = evaluate(self.root, self.plan, "base", self.root / "unused")
        self.assertEqual(value["planned_generation_calls"], planned_calls(self.plan))
        self.assertTrue(value["over_call_budget"])
        self.assertEqual(value["actual_generation_calls"], 0)
        self.assertFalse(value["contains_model_results"])
        with self.assertRaisesRegex(TrainingBlocked, "budget"):
            evaluate(self.root, self.plan, "base", self.root / "unused", execute=True)

    def test_adapter_must_be_completed_and_match_full_plan(self):
        self.assertIsNone(adapter_identity(self.plan, "base")["adapter_sha256"])
        with self.assertRaises(ValueError): adapter_identity(self.plan, "base", self.root)
        with self.assertRaises(ValueError): adapter_identity(self.plan, "paired_sft")
        run = self.root / "fake"; run.mkdir()
        (run / "plan.json").write_text(json.dumps(self.plan))
        (run / "training_summary.json").write_text(json.dumps({"status": "planned", "contains_training_run": False}))
        with self.assertRaisesRegex(ValueError, "completed"):
            adapter_identity(self.plan, "paired_sft", run)

    def test_partial_or_fixture_evaluations_are_rejected(self):
        path = self.root / "fixture"; path.mkdir()
        (path / "evaluation_manifest.json").write_text(json.dumps({"status": "completed", "execution_kind": "test_fixture", "contains_model_results": False}))
        with self.assertRaisesRegex(ValueError, "real"):
            load_evaluation(self.root, path)
        (path / "evaluation_manifest.json").unlink()
        with self.assertRaises(FileNotFoundError): load_evaluation(self.root, path)

    def test_empty_report_has_no_performance_tables_or_completed_arms(self):
        output = self.root / "results/training/reports/empty"
        result = report(self.root, self.plan, output)
        self.assertEqual(result["status"], "no_training_or_model_results")
        self.assertFalse(result["contains_model_results"])
        self.assertFalse(result["contains_training_run"])
        self.assertEqual(result["table_rows"], {})
        self.assertEqual(result["files"], [])
        self.assertEqual(result["actual_optimizer_steps"], 0)
        self.assertEqual([p.name for p in output.iterdir()], ["training_report.json"])
        self.assertFalse(result["lexical_overlap_audit"]["clean_full_context_holdout"])
        with self.assertRaises(ValueError): report(self.root, self.plan, output)

    def test_reports_cannot_write_into_sources(self):
        with self.assertRaises(ValueError): report(self.root, self.plan, self.root / "src/new")

    def test_synthetic_multiarm_report_keeps_profiles_separate_and_holm_shared(self):
        # Mocked reader observations exist only in this temporary unit fixture.
        # No provider/model is called and no fixture is saved to deliverables.
        from test_analysis_pipeline import observation
        loaded = []
        for arm in ("base", "sft_lora"):
            rows = [observation("synthetic-"+str(i), language, condition,
                prediction=(condition == "canonical"), transfer="in_domain",
                model_id="synthetic-model--"+arm, model_spec_sha256=("a" if arm == "base" else "b")*64)
                for i in range(6) for language in ("en", "tr") for condition in ("canonical", "equivalent")]
            loaded.append({"plan": self.plan, "manifest": {"identity": {"arm": arm}},
                           "manifest_sha256": "c"*64, "observations": rows})
        out = self.root / "results/training/reports/synthetic-multiplicity"
        # Adapter evaluation now requires the exact verified training identity.
        # Mock that boundary only inside this explicitly synthetic fixture.
        training = self.root / "synthetic-training"; training.mkdir()
        (training / "training_summary.json").write_text(json.dumps({"arm":"sft_lora","stats":{"optimizer_steps":1}}))
        with patch("training.reporting.load_evaluation", side_effect=loaded), patch(
                "training.reporting.adapter_identity", return_value=loaded[1]["manifest"]["identity"]):
            report(self.root, self.plan, out, evaluations=["synthetic-base", "synthetic-sft"],
                   training_runs=[training], repeats=20)
        with (out / "tables/presentation_tests.csv").open() as handle:
            rows = list(csv.DictReader(handle))
        self.assertEqual(len(rows), 4)
        self.assertEqual({row["training_arm"] for row in rows}, {"base", "sft_lora"})
        self.assertTrue(all(int(row["holm_family_size"]) == 2 for row in rows))
        self.assertTrue(all(float(row["p_value"]) == .03125 and float(row["p_adjusted"]) == .0625 for row in rows))

    def test_cli_plan_run_evaluate_and_report_stay_offline(self):
        status, plan = self.invoke(["plan", "--output", "plan.json"])
        self.assertEqual(status, 0)
        self.assertFalse(plan["contains_training_run"])
        status, result = self.invoke(["run", "--plan", "plan.json", "--arm", "sft_lora", "--output", "unused"])
        self.assertEqual(status, 0); self.assertEqual(result["actual_optimizer_steps"], 0)
        status, result = self.invoke(["evaluate", "--plan", "plan.json", "--arm", "base", "--output", "unused"])
        self.assertEqual(status, 0); self.assertEqual(result["actual_generation_calls"], 0)
        status, result = self.invoke(["report", "--plan", "plan.json", "--output", "results/training/reports/cli"])
        self.assertEqual(status, 0); self.assertEqual(result["status"], "no_training_or_model_results")

    def test_cli_execute_without_saved_plan_is_rejected(self):
        status, error = self.invoke(["run", "--arm", "sft_lora", "--execute", "--output", "unused"])
        self.assertEqual(status, 2)
        self.assertIn("explicit --plan", error)

    def test_cli_saved_plan_cannot_hide_config_or_seed_overrides(self):
        write_plan(self.plan, self.root / "plan.json")
        status, error = self.invoke(["run", "--plan", "plan.json", "--seed", "43", "--arm", "sft_lora", "--output", "unused"])
        self.assertEqual(status, 2)
        self.assertIn("overrides", error)

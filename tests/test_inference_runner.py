"""Execution fixtures on frozen real prompts, without contacting any model.

Fixture answers are fixed by output language and never inspect a gold label.
Every generated journal is confined to a disposable project and explicitly
marked test_fixture / contains_model_results=false.
"""

from copy import deepcopy
import importlib.util
import json
import os
from pathlib import Path
import shutil
import tempfile
import unittest
from unittest.mock import patch

from complete_grid import dataset_builder as grid
from inference.dataset import GridDataset
from inference.planning import DEFAULT_POLICY, make_plan
from inference.prompts import make_messages
from inference.providers import ProviderError
from inference.runner import (
    ExecutionBlocked, approve_pilot, request_for, run_plan, task_for,
)
from inference.storage import RunStore
from test_inference_planning import fixture_model


PROJECT_ROOT = Path(__file__).resolve().parents[1]
WORK_ROOT = PROJECT_ROOT.parents[1] / "work"
MANIFEST = Path("data/complete_grid/manifest.json")
HAS_TIKTOKEN = importlib.util.find_spec("tiktoken") is not None


class FakeClock:
    def __init__(self):
        self.now = 1_800_000_000.0
        self.sleeps = []

    def __call__(self):
        return self.now

    def sleep(self, duration):
        self.sleeps.append(duration)
        self.now += duration


class FixtureFactory:
    """Adapter-shaped deterministic fixtures, without a source/gold interface."""
    def __init__(self, actions=(), *, always=None, count=100, preflight_error=None):
        self.actions = list(actions)
        self.always, self.count, self.preflight_error = always, count, preflight_error
        self.instances, self.calls, self.count_calls, self.preflights = [], [], [], []

    def __call__(self, spec):
        self.instances.append(deepcopy(spec))
        factory = self

        class FixtureProvider:
            def preflight(self):
                factory.preflights.append(spec["id"])
                if factory.preflight_error:
                    raise factory.preflight_error
                return {"api": "test_fixture", "resolved_revision": spec["revision"],
                        "model_id": spec["model_id"], "contains_model_results": False}

            def count_input_tokens(self, messages):
                factory.count_calls.append(deepcopy(messages))
                return factory.count

            def generate(self, messages, request_id):
                factory.calls.append({"model_id": spec["id"], "request_id": request_id,
                                      "messages": deepcopy(messages)})
                action = factory.actions.pop(0) if factory.actions else factory.always
                if isinstance(action, BaseException):
                    raise action
                language = "tr" if messages[1]["content"].endswith("Yalnızca EVET veya HAYIR yaz.") else "en"
                text = "EVET" if language == "tr" else "YES"
                response = {"text": text, "provider_model": spec["model_id"],
                            "resolved_revision": spec["revision"], "response_id": request_id,
                            "finish_reason": "stop", "truncated": False, "refusal": False,
                            "usage": {"input_tokens": factory.count, "output_tokens": 2,
                                      "total_tokens": factory.count + 2, "cached_input_tokens": 0,
                                      "reasoning_tokens": 0},
                            "raw": {"fixture": True, "fixed_answer": text}}
                if isinstance(action, dict):
                    response.update(deepcopy(action))
                return response

        return FixtureProvider()


@unittest.skipUnless(HAS_TIKTOKEN, "pinned tiktoken dependency is required for real prompt reconstruction")
class InferenceRunnerTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        with GridDataset(PROJECT_ROOT, MANIFEST) as data:
            cls.database_hash = next(item["sha256"] for item in data.manifest["files"]
                                     if item["path"] == "grid.sqlite")
        cls.original_database = PROJECT_ROOT / "data/complete_grid/grid.sqlite"
        cls.database_identity = (cls.original_database.stat().st_dev, cls.original_database.stat().st_ino,
                                 cls.original_database.stat().st_size, cls.original_database.stat().st_mtime_ns)
        WORK_ROOT.mkdir(exist_ok=True)

    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(prefix="runner-tests-", dir=WORK_ROOT)
        self.root = Path(self.temporary.name) / "project"
        shutil.copytree(PROJECT_ROOT / "src", self.root / "src",
                        ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
        if (PROJECT_ROOT / "scripts").exists():
            shutil.copytree(PROJECT_ROOT / "scripts", self.root / "scripts")
        shutil.copyfile(PROJECT_ROOT / "experiment_dataset_manifest.json",
                        self.root / "experiment_dataset_manifest.json")
        for directory, names in (
            ("data/complete_grid", ("manifest.json", "coverage.json", "grid.sqlite")),
            ("tokenizers", tuple(path.name for path in (PROJECT_ROOT / "tokenizers").iterdir() if path.is_file())),
        ):
            target = self.root / directory
            target.mkdir(parents=True)
            for name in names:
                os.link(PROJECT_ROOT / directory / name, target / name)
        # The full bank was hash-validated above. Every test uses that same
        # immutable inode through a read-only SQLite connection; only this
        # repeated large-file checksum is elided, not cohort/source validation.
        original_hash = grid.file_hash

        def fixture_hash(path):
            path = Path(path)
            if path == self.root / "data/complete_grid/grid.sqlite":
                stat = path.stat()
                self.assertEqual((stat.st_dev, stat.st_ino, stat.st_size, stat.st_mtime_ns), self.database_identity)
                return self.database_hash
            return original_hash(path)

        self.hash_patch = patch("complete_grid.dataset_builder.file_hash", side_effect=fixture_hash)
        self.hash_patch.start()
        self.addCleanup(self.hash_patch.stop)
        self.addCleanup(self.temporary.cleanup)
        self.clock = FakeClock()
        self.model = fixture_model()
        self.model["enabled"] = True
        self.policy = {**DEFAULT_POLICY, "requests_per_minute": 2000, "tokens_per_minute": 1_000_000}

    def plan(self, models=None, stage="pilot", policy=None):
        return make_plan(self.root, MANIFEST, [self.model] if models is None else models,
                         stage=stage, policy=self.policy if policy is None else policy)

    def execute(self, plan, factory, run_id="fixture-run", **kwargs):
        return run_plan(self.root, plan, run_id, execute=True, provider_factory=factory,
                        test_fixture=True, sleep=self.clock.sleep, clock=self.clock, **kwargs)

    def normalized(self, run_id="fixture-run"):
        return [json.loads(line) for line in
                (self.root / "results/normalized" / (run_id + ".jsonl")).read_text().splitlines()]

    def test_default_is_dry_run_and_never_instantiates_provider_or_creates_raw_results(self):
        factory = FixtureFactory()
        report = run_plan(self.root, self.plan(), provider_factory=factory)
        self.assertTrue(report["dry_run"])
        self.assertEqual(report["actual_generation_calls"], 0)
        self.assertEqual(factory.instances, [])
        self.assertFalse((self.root / "results").exists())

    def test_disabled_model_execution_is_blocked_before_provider_creation(self):
        model = deepcopy(self.model); model["enabled"] = False
        factory = FixtureFactory()
        with self.assertRaisesRegex(ExecutionBlocked, "disabled"):
            self.execute(self.plan(models=[model]), factory)
        self.assertEqual(factory.instances, [])

    def test_injected_providers_cannot_be_published_as_live_results_or_use_live_run_ids(self):
        plan, factory = self.plan(), FixtureFactory()
        with self.assertRaises(ExecutionBlocked):
            run_plan(self.root, plan, execute=True, provider_factory=factory)
        with self.assertRaises(ExecutionBlocked):
            run_plan(self.root, plan, execute=True, test_fixture=True)
        with self.assertRaises(ExecutionBlocked):
            self.execute(plan, factory, "ordinary-live-looking-run")
        self.assertEqual(factory.calls, [])

    def test_small_fixture_run_has_32_unique_calls_and_explicit_non_model_evidence(self):
        plan, factory = self.plan(), FixtureFactory()
        report = self.execute(plan, factory)
        self.assertTrue(report["complete"])
        self.assertEqual(report["actual_generation_calls_this_invocation"], 32)
        self.assertEqual(len(factory.calls), 32)
        self.assertEqual(len({call["request_id"] for call in factory.calls}), 32)
        rows = self.normalized()
        self.assertEqual(len(rows), 32)
        self.assertTrue(all(row["execution_kind"] == "test_fixture" and not row["contains_model_results"] for row in rows))
        evidence = json.loads((self.root / "results/raw/fixture-run/execution.json").read_bytes())
        self.assertFalse(evidence["contains_model_results"])
        self.assertEqual(evidence["execution_kind"], "test_fixture")
        self.assertFalse(report["contains_model_results"])

    def test_every_model_receives_the_same_latent_query_and_design_cohort(self):
        second = deepcopy(self.model); second.update(id="fixture-large", parameters_billion=1.5)
        factory = FixtureFactory()
        report = self.execute(self.plan(models=[self.model, second]), factory)
        self.assertTrue(report["complete"])
        rows = self.normalized()
        by_model = {}
        for row in rows:
            source = row["source"]
            by_model.setdefault(row["model_id"], set()).add((source["latent_instance_id"],
                                                           source["query_instance_id"], source["grid_design_id"]))
        self.assertEqual(len(by_model[self.model["id"]]), 32)
        self.assertEqual(by_model[self.model["id"]], by_model[second["id"]])
        self.assertEqual(report["actual_generation_calls_this_invocation"], 64)

    def test_task_ids_are_stable_and_only_localized_messages_reach_provider(self):
        with GridDataset(self.root, MANIFEST) as data:
            row = next(data.iter_rows(data.select(self.plan()["cohort"]["selection"])))
        built = make_messages(row)
        first = task_for(row, self.model, built)
        self.assertEqual(first, task_for(deepcopy(row), deepcopy(self.model), deepcopy(built)))
        self.assertNotEqual(first["task_id"], task_for(row, self.model, built, replicate=1)["task_id"])
        request = request_for(first, self.model, built)
        self.assertEqual(request["messages"], built["messages"])
        self.assertNotIn("gold_label", json.dumps(request["messages"]))
        self.assertNotIn("latent_order", json.dumps(request["messages"]))
        changed = deepcopy(row); changed["gold_label"] = not changed["gold_label"]
        self.assertEqual(make_messages(changed), built)
        factory = FixtureFactory()
        self.execute(self.plan(), factory)
        self.assertTrue(all(set(message) == {"role", "content"} for call in factory.calls for message in call["messages"]))

    def test_checkpoint_resume_makes_zero_duplicate_provider_calls_and_preserves_raw_bytes(self):
        plan, factory = self.plan(), FixtureFactory()
        self.execute(plan, factory)
        paths = sorted((self.root / "results/raw/fixture-run/attempts").glob("*/response.json"))
        original = {path: path.read_bytes() for path in paths}
        resumed_factory = FixtureFactory()
        report = self.execute(plan, resumed_factory)
        self.assertTrue(report["complete"])
        self.assertEqual(report["actual_generation_calls_this_invocation"], 0)
        self.assertEqual(resumed_factory.instances, [])
        self.assertEqual({path: path.read_bytes() for path in paths}, original)
        self.assertEqual(len(self.normalized()), 32)
        self.assertEqual(report["summary"]["attempts"], 32)

    def test_invalid_and_contradictory_outputs_are_recorded_without_retrying(self):
        factory = FixtureFactory(always={"text": "YES and NO"})
        report = self.execute(self.plan(), factory)
        self.assertTrue(report["complete"])
        self.assertEqual(len(factory.calls), 32)
        rows = self.normalized()
        self.assertTrue(all(not row["parse"]["valid"] and row["parse"]["prediction"] is None for row in rows))
        self.assertTrue(all(row["parse"]["invalid_reason"] == "contradictory_answers" for row in rows))

    def test_foreign_answer_language_is_reported_separately(self):
        report = self.execute(self.plan(), FixtureFactory(always={"text": "YES"}))
        self.assertTrue(report["complete"])
        tr = [row for row in self.normalized() if row["source"]["language"] == "tr"]
        en = [row for row in self.normalized() if row["source"]["language"] == "en"]
        self.assertEqual((len(tr), len(en)), (16, 16))
        self.assertTrue(all(row["parse"]["wrong_language"] and not row["parse"]["valid"] for row in tr))
        self.assertTrue(all(row["parse"]["valid"] for row in en))

    def test_truncation_and_explicit_refusal_flags_invalidate_even_exact_answers(self):
        factory = FixtureFactory(actions=[{"truncated": True}, {"refusal": True}])
        self.execute(self.plan(), factory)
        rows = self.normalized()
        self.assertEqual(rows[0]["parse"]["invalid_reason"], "truncated")
        self.assertEqual(rows[1]["parse"]["invalid_reason"], "refusal")
        self.assertEqual(len(factory.calls), 32)

    def test_safe_429_rejections_back_off_exponentially_then_finish_once(self):
        rejection = ProviderError("rate_limited", retryable=True, ambiguous=False, status_code=429)
        factory = FixtureFactory(actions=[rejection, rejection])
        report = self.execute(self.plan(), factory)
        self.assertTrue(report["complete"])
        self.assertEqual(report["actual_generation_calls_this_invocation"], 34)
        self.assertEqual(report["summary"]["attempts"], 34)
        self.assertEqual(report["summary"]["raw_errors"], 2)
        self.assertEqual(report["summary"]["normalized_records"], 32)
        self.assertEqual(self.clock.sleeps[:2], [1.0, 2.0])

    def test_maximum_retry_attempts_stop_without_silently_dropping_planned_cohort(self):
        factory = FixtureFactory(always=ProviderError("rate_limited", retryable=True, ambiguous=False))
        plan = self.plan()
        report = self.execute(plan, factory)
        self.assertFalse(report["complete"])
        self.assertEqual(report["aborted_reason"], "maximum_attempts_exhausted")
        self.assertEqual(report["actual_generation_calls_this_invocation"], 3)
        self.assertEqual(report["planned_generation_calls"], 32)
        self.assertEqual(report["summary"]["statuses"], {"permanent_failed": 1})
        self.assertEqual(plan["cohort"]["group_count"], 8)

    def test_uncertain_transport_failure_blocks_resume_until_audited_resolution(self):
        factory = FixtureFactory(actions=[ProviderError("transport_error", retryable=True, ambiguous=True)])
        plan = self.plan()
        initial = self.execute(plan, factory)
        self.assertEqual(initial["actual_generation_calls_this_invocation"], 1)
        self.assertEqual(initial["summary"]["statuses"], {"ambiguous": 1})
        resumed = self.execute(plan, factory)
        self.assertEqual(resumed["actual_generation_calls_this_invocation"], 0)
        self.assertEqual(resumed["aborted_reason"], "ambiguous_call_requires_explicit_resolution")
        with RunStore(self.root, "fixture-run", plan) as store:
            task = store.pending_tasks()[0]
            store.resolve_ambiguous(task["task_id"], "retry", "Test explicitly accepts possible duplicate", True)
        completed = self.execute(plan, factory)
        self.assertTrue(completed["complete"])
        self.assertEqual(completed["actual_generation_calls_this_invocation"], 32)
        self.assertEqual(completed["summary"]["resolutions"], 1)

    def test_complete_raw_response_after_crash_is_recovered_without_calling_that_task_again(self):
        plan, factory = self.plan(), FixtureFactory()
        with patch("inference.runner.RunStore.finish", side_effect=OSError("simulated crash after raw response")):
            with self.assertRaises(OSError):
                self.execute(plan, factory)
        self.assertEqual(len(factory.calls), 1)
        report = self.execute(plan, factory)
        self.assertTrue(report["complete"])
        self.assertEqual(report["recovered_responses_this_invocation"], 1)
        self.assertEqual(report["actual_generation_calls_this_invocation"], 31)
        self.assertEqual(len(factory.calls), 32)
        self.assertEqual(len(self.normalized()), 32)

    def test_cache_reuse_has_attribution_and_is_never_an_independent_observation(self):
        plan, first_factory = self.plan(policy={**self.policy, "cache": True}), FixtureFactory()
        first = self.execute(plan, first_factory, "fixture-origin")
        second_factory = FixtureFactory()
        reused = self.execute(plan, second_factory, "fixture-reuse")
        self.assertTrue(first["complete"] and reused["complete"])
        self.assertEqual(reused["actual_generation_calls_this_invocation"], 0)
        self.assertEqual(reused["cache_hits_this_invocation"], 32)
        self.assertEqual(second_factory.instances, [])
        rows = self.normalized("fixture-reuse")
        self.assertTrue(all(row["cache_origin"]["run_id"] == "fixture-origin" for row in rows))
        self.assertTrue(all(not row["independent_observation"] and row["newly_billed_usage"] is None for row in rows))
        self.assertTrue(all(row["cost"]["status"] == "cache_reuse_no_new_generation" for row in rows))
        self.assertTrue((self.root / "results/cache/fixtures").is_dir())
        self.assertFalse((self.root / "results/cache/live").exists())

    def test_fixture_pilot_cannot_authorize_full_evaluation(self):
        plan = self.plan()
        self.execute(plan, FixtureFactory())
        review = self.root / "fixture-review.json"
        review.write_text(json.dumps({"status": "approved", "reviewers": ["Test fixture"],
                                      "dataset_manifest_sha256": plan["cohort"]["dataset_manifest_sha256"]}))
        approval = self.root / "must-not-create-approval.json"
        with self.assertRaises(ExecutionBlocked):
            approve_pilot(self.root, "fixture-run", approval, approved_by="Fixture", linguistic_review=review)
        self.assertFalse(approval.exists())

    def test_full_evaluation_without_real_pilot_approval_never_calls_provider(self):
        factory = FixtureFactory()
        plan = self.plan(stage="full", policy={**self.policy, "max_calls": 10000})
        with self.assertRaisesRegex(ExecutionBlocked, "approval"):
            self.execute(plan, factory)
        self.assertEqual(factory.instances, [])

    def test_budget_cap_blocks_whole_unchanged_cohort_before_provider_creation(self):
        plan, factory = self.plan(policy={**self.policy, "max_calls": 1}), FixtureFactory()
        with self.assertRaisesRegex(ExecutionBlocked, "call_budget_exceeded"):
            self.execute(plan, factory)
        self.assertEqual(plan["cohort"]["prompt_count"], 32)
        self.assertEqual(factory.instances, [])

    def test_changed_inference_sources_invalidate_plan_before_any_provider_call(self):
        plan, factory = self.plan(), FixtureFactory()
        path = self.root / "src/inference/parser.py"
        path.write_bytes(path.read_bytes() + b"\n# fixture source changed after planning\n")
        with self.assertRaisesRegex(ValueError, "sources?"):
            self.execute(plan, factory)
        self.assertEqual(factory.calls, [])

    def test_exact_context_preflight_rejects_instead_of_truncating_or_shrinking_cohort(self):
        model = deepcopy(self.model)
        model["limits"].update(context_tokens=64, max_output_tokens=32)
        plan, factory = self.plan(models=[model]), FixtureFactory(count=100)
        report = self.execute(plan, factory)
        self.assertEqual(report["aborted_reason"], "context_limit_exceeded")
        self.assertEqual(report["actual_generation_calls_this_invocation"], 0)
        self.assertEqual(factory.calls, [])
        self.assertEqual(report["planned_generation_calls"], 32)

    def test_unavailable_model_identity_is_a_failure_without_fallback(self):
        factory = FixtureFactory(preflight_error=ProviderError("model_unavailable"))
        report = self.execute(self.plan(), factory)
        self.assertEqual(report["aborted_reason"], "model_unavailable")
        self.assertEqual(report["actual_generation_calls_this_invocation"], 0)
        self.assertEqual(factory.calls, [])
        self.assertEqual([spec["model_id"] for spec in factory.instances], [self.model["model_id"]])

    def test_rate_limiter_uses_injected_clock_and_never_actual_long_sleeps(self):
        plan = self.plan(policy={**self.policy, "requests_per_minute": 2})
        factory = FixtureFactory()
        report = self.execute(plan, factory)
        self.assertTrue(report["complete"])
        self.assertEqual(len(factory.calls), 32)
        self.assertTrue(self.clock.sleeps)
        self.assertTrue(all(0 < duration <= 60 for duration in self.clock.sleeps))
        self.assertGreaterEqual(sum(self.clock.sleeps), 15 * 60)


if __name__ == "__main__":
    unittest.main()

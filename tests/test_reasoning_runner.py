"""Reasoning execution fixtures never inspect gold or contact a model."""

from copy import deepcopy
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import test_inference_runner as fixtures
from inference.dataset import GridDataset, default_selection
from inference.planning import DEFAULT_POLICY
from inference.providers import ProviderError
from reasoning.io import discover_runs, expected_tasks, load_run
from reasoning.planning import make_plan
from reasoning.runner import ReasoningStore, run_plan, approve_pilot, ExecutionBlocked


PROJECT = Path(__file__).resolve().parents[1]
SETTINGS = {"schema_version": 1, "output_budget": 2048,
            "strategies": ["direct", "zero_shot_reasoning", "structured_reasoning"],
            "native_efforts": ["none", "low", "medium", "high"], "capabilities": []}


class ReasoningFactory(fixtures.FixtureFactory):
    def __call__(self, spec):
        provider = super().__call__(spec)
        original = provider.generate

        def generate(messages, request_id):
            response = original(messages, request_id)
            # This fixture sees messages only, never latent order or gold.
            text = messages[-1]["content"]
            answer = "EVET" if "Soru:" in text else "YES"
            response["text"] = "Fixture explanation.\nFINAL: " + answer if "FINAL:" in text else answer
            if self.force is not None:
                response.update(deepcopy(self.force))
            return response

        provider.generate = generate
        return provider

    def __init__(self, *args, force=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.force = force


class ReasoningRunnerTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        fixtures.InferenceRunnerTests.setUpClass.__func__(cls)

    def setUp(self):
        fixtures.InferenceRunnerTests.setUp(self)

    def plan(self, *, model=None, settings=None, stage="pilot", policy=None):
        selection = default_selection(stage)
        selection.update(sizes=[4], groups_per_size=4)
        return make_plan(self.root, fixtures.MANIFEST, [model or self.model], stage,
                         selection, policy or DEFAULT_POLICY, settings or SETTINGS)

    def execute(self, plan=None, factory=None, run_id="fixture-reasoning", **kwargs):
        return run_plan(self.root, plan or self.plan(), run_id, execute=True,
                        provider_factory=factory or ReasoningFactory(), test_fixture=True,
                        sleep=self.clock.sleep, clock=self.clock, **kwargs)

    def normalized(self, run_id="fixture-reasoning"):
        return [json.loads(line) for line in
                (self.root / "results/reasoning/normalized" / (run_id+".jsonl")).read_text().splitlines()]

    def test_dry_run_has_no_provider_or_journal(self):
        factory = ReasoningFactory()
        report = run_plan(self.root, self.plan(), provider_factory=factory)
        self.assertEqual(report["actual_generation_calls"], 0)
        self.assertFalse(report["contains_model_results"])
        self.assertEqual(factory.instances, [])
        self.assertFalse((self.root / "results").exists())

    def test_three_arms_keep_exact_cohort_and_final_answer_only(self):
        plan, factory = self.plan(), ReasoningFactory()
        result = self.execute(plan, factory)
        self.assertTrue(result["complete"])
        self.assertEqual(len(factory.calls), plan["generation_calls"])
        rows = self.normalized()
        self.assertTrue(all(r["parse"]["valid"] for r in rows))
        self.assertTrue(all(r["contains_model_results"] is False for r in rows))
        self.assertEqual({r["strategy"] for r in rows}, set(SETTINGS["strategies"]))
        coordinates = {}
        for row in rows:
            coordinates.setdefault(row["strategy"], set()).add(json.dumps(row["source"], sort_keys=True))
        self.assertEqual(len({tuple(sorted(c)) for c in coordinates.values()}), 1)
        self.assertEqual(len({r["task_id"] for r in rows}), len(rows))
        self.assertEqual(len({r["attempt_id"] for r in rows}), len(rows))
        self.assertFalse((self.root / "results/raw").exists())
        self.assertEqual(discover_runs(self.root), ["fixture-reasoning"])

    def test_resume_has_no_calls_and_raw_bytes_unchanged(self):
        plan, factory = self.plan(), ReasoningFactory()
        self.execute(plan, factory)
        raw = self.root / "results/reasoning/raw/fixture-reasoning/attempts"
        before = {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in raw.rglob("*.json")}
        other = ReasoningFactory()
        report = self.execute(plan, other)
        self.assertTrue(report["complete"])
        self.assertEqual(other.calls, [])
        self.assertEqual(before, {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in raw.rglob("*.json")})

    def test_reader_rebuilds_every_arm_and_rejects_fixture_by_default(self):
        plan = self.plan()
        self.execute(plan)
        with self.assertRaisesRegex(ValueError, "synthetic"):
            load_run(self.root, "fixture-reasoning")
        loaded = load_run(self.root, "fixture-reasoning", allow_synthetic=True)
        self.assertTrue(loaded["synthetic"])
        self.assertEqual(len(loaded["observations"]), plan["generation_calls"])
        self.assertTrue(all(row["prediction"] is True for row in loaded["observations"]))
        self.assertEqual(len({row["model_spec_sha256"] for row in loaded["observations"]}), 1)
        self.assertEqual(len({row["effective_model_spec_sha256"] for row in loaded["observations"]}), 1)

    def test_missing_calls_are_restored_for_all_arms_as_invalid(self):
        plan = self.plan()
        factory = ReasoningFactory(preflight_error=ProviderError("fixture_unavailable"))
        report = self.execute(plan, factory)
        self.assertFalse(report["complete"])
        loaded = load_run(self.root, "fixture-reasoning", allow_synthetic=True)
        rows = loaded["observations"]
        self.assertEqual(len(rows), plan["generation_calls"])
        self.assertTrue(all(row["prediction"] is None for row in rows))
        self.assertEqual(sum(row["response_status"] == "missing" for row in rows), len(rows)-1)

    def test_length_finish_reason_is_separate_truncation_diagnostic(self):
        factory = ReasoningFactory(force={"finish_reason": "length", "truncated": False})
        self.execute(factory=factory)
        rows = self.normalized()
        self.assertTrue(all(row["truncated"] and not row["parse"]["valid"] for row in rows))

    def test_context_overflow_does_not_generate_or_truncate_problem(self):
        factory = ReasoningFactory(count=200000)
        report = self.execute(factory=factory)
        self.assertFalse(report["complete"])
        self.assertEqual(factory.calls, [])
        self.assertEqual(report["aborted_reason"], "context_limit_exceeded")

    def test_retry_uses_backoff_but_ambiguous_response_is_not_repeated(self):
        factory = ReasoningFactory(actions=[ProviderError("rate_limit", retryable=True)])
        report = self.execute(factory=factory)
        self.assertTrue(report["complete"])
        self.assertEqual(len(factory.calls), report["planned_generation_calls"]+1)
        self.assertTrue(self.clock.sleeps)
        second = ReasoningFactory(actions=[ProviderError("timeout", ambiguous=True)])
        plan = self.plan()
        report = self.execute(plan, second, "fixture-timeout")
        self.assertFalse(report["complete"])
        again = ReasoningFactory()
        report = self.execute(plan, again, "fixture-timeout")
        self.assertEqual(report["aborted_reason"], "ambiguous_call_requires_explicit_resolution")
        self.assertEqual(again.calls, [])

    def test_disabled_model_and_injected_live_factory_are_blocked(self):
        model = deepcopy(self.model); model["enabled"] = False
        factory = ReasoningFactory()
        with self.assertRaisesRegex(ExecutionBlocked, "disabled"):
            self.execute(self.plan(model=model), factory)
        with self.assertRaises(ExecutionBlocked):
            run_plan(self.root, self.plan(), execute=True, provider_factory=factory)
        with self.assertRaises(ExecutionBlocked):
            run_plan(self.root, self.plan(), execute=True, test_fixture=True)
        with self.assertRaises(ExecutionBlocked):
            self.execute(factory=factory, run_id="live-looking")
        self.assertEqual(factory.calls, [])

    def test_changed_rehashed_budget_is_recomputed(self):
        from inference.dataset import fingerprint
        plan = self.plan(); plan["generation_calls"] = 1
        plan["plan_sha256"] = fingerprint({k: v for k, v in plan.items() if k != "plan_sha256"})
        with self.assertRaises((ValueError, ExecutionBlocked)):
            self.execute(plan)

    def test_cache_is_never_independent_research_evidence(self):
        plan = self.plan(policy={**DEFAULT_POLICY, "cache": True})
        self.execute(plan)
        other = ReasoningFactory()
        report = self.execute(plan, other, "fixture-cached")
        self.assertEqual(other.calls, [])
        self.assertEqual(report["cache_hits_this_invocation"], plan["generation_calls"])
        with self.assertRaisesRegex(ValueError, "cached"):
            load_run(self.root, "fixture-cached", allow_synthetic=True)

    def test_raw_tampering_is_detected_without_mutation(self):
        self.execute()
        raw = next((self.root / "results/reasoning/raw/fixture-reasoning/attempts").glob("*/response.json"))
        raw.write_bytes(raw.read_bytes()+b" ")
        before = raw.read_bytes()
        with self.assertRaisesRegex(ValueError, "hash"):
            load_run(self.root, "fixture-reasoning", allow_synthetic=True)
        self.assertEqual(raw.read_bytes(), before)

    def test_full_requires_reasoning_pilot_receipt(self):
        with self.assertRaisesRegex(ExecutionBlocked, "pilot"):
            self.execute(self.plan(stage="full"))

    def test_fixture_pilot_cannot_authorize_full(self):
        self.execute()
        with self.assertRaisesRegex(ValueError, "synthetic"):
            approve_pilot(self.root, "fixture-reasoning", self.root / "approval.json",
                          approved_by="Fixture", linguistic_review=self.root / "missing.json")
        self.assertFalse((self.root / "approval.json").exists())

    def test_namespace_cannot_be_redirected_into_old_result_files(self):
        plan = self.plan()
        old = self.root / "results/raw"; old.mkdir(parents=True)
        target = self.root / "results/reasoning"; target.mkdir()
        (target / "raw").symlink_to(old, target_is_directory=True)
        factory = ReasoningFactory()
        with self.assertRaisesRegex(ValueError, "symlinks"):
            self.execute(plan, factory)
        self.assertEqual(factory.instances, [])
        self.assertEqual(list(old.iterdir()), [])

    def test_refusal_finish_reason_is_reported_without_a_response_flag(self):
        self.execute(factory=ReasoningFactory(force={"finish_reason": "refusal", "refusal": False}))
        self.assertTrue(all(r["refusal"] and not r["parse"]["valid"] for r in self.normalized()))

    def test_supported_native_arms_share_frozen_coordinates_and_distinct_tasks(self):
        model = fixtures.fixture_model("openai", "fixture-api")
        model["enabled"] = True
        settings = deepcopy(SETTINGS)
        settings["capabilities"] = [{"provider": "openai", "model_id": model["model_id"],
            "revision": model["revision"], "evidence_url": "https://example.invalid/fixture",
            "verified_on": "2026-10-25", "supported_efforts": settings["native_efforts"],
            "sampling_restrictions": {level: {"temperature": "allowed", "top_p": "allowed"}
                                      for level in settings["native_efforts"]}}]
        plan = self.plan(model=model, settings=settings)
        self.assertEqual(len(plan["arms"]), 7)
        self.execute(plan)
        loaded = load_run(self.root, "fixture-reasoning", allow_synthetic=True)
        observations = loaded["observations"]
        self.assertEqual(len({r["model_spec_sha256"] for r in observations}), 1)
        self.assertEqual({r["effort"] for r in observations}, {None, "none", "low", "medium", "high"})
        self.assertEqual(len({r["task_id"] for r in observations}), plan["generation_calls"])
        self.assertEqual({r["query_id"] for r in observations if r["strategy"] == "provider_effort"},
                         {r["query_id"] for r in observations if r["strategy"] == "direct"})

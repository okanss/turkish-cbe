"""Frozen real cohorts and synthetic capability declarations; no provider calls."""

from contextlib import contextmanager
from copy import deepcopy
import json
from pathlib import Path
import shutil
import tempfile
import unittest
from unittest.mock import patch

from inference.dataset import GridDataset, default_selection, fingerprint
from inference.planning import DEFAULT_POLICY
from reasoning.planning import (DEFAULT_SETTINGS, load_settings, make_plan,
    source_fingerprints, validate_plan, validate_settings, write_plan)
from test_inference_planning import fixture_model


PROJECT = Path(__file__).resolve().parents[1]
WORK = PROJECT.parents[1] / "work"
MANIFEST = Path("data/complete_grid/manifest.json")


def declaration(model, efforts=("none", "low"), rule="allowed"):
    return {"provider": model["provider"], "model_id": model["model_id"], "revision": model["revision"],
            "evidence_url": "https://example.org/explicit-synthetic-capability", "verified_on": "2026-10-25",
            "supported_efforts": list(efforts), "sampling_restrictions": {
                effort: {"temperature": rule, "top_p": rule} for effort in efforts}}


def rehash(plan):
    plan["plan_sha256"] = fingerprint({key: value for key, value in plan.items() if key != "plan_sha256"})


class ReasoningPlanningTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.data = GridDataset(PROJECT, MANIFEST).__enter__()

    @classmethod
    def tearDownClass(cls):
        cls.data.__exit__(None, None, None)

    def setUp(self):
        self.model = fixture_model()
        self.api = fixture_model("openai", "fixture-api")

    def plan(self, models=None, **kwargs):
        @contextmanager
        def frozen(root, manifest):
            self.assertEqual(Path(root), PROJECT)
            self.assertEqual(Path(manifest), MANIFEST)
            yield self.data
        with patch("reasoning.planning.GridDataset", frozen):
            return make_plan(PROJECT, MANIFEST, [self.model] if models is None else models, **kwargs)

    def test_default_128_prompt_cohort_has_all_three_query_branches(self):
        plan = self.plan()
        self.assertEqual(plan["cohort"]["prompt_count"], 128)
        self.assertEqual(plan["cohort"]["group_count"], 8)
        self.assertEqual(plan["cohort"]["selection"]["query_types"], ["endpoint", "internal", "distance_controlled"])
        self.assertEqual(plan["generation_calls"], 384)
        self.assertEqual({row["generation_calls"] for row in plan["arm_budgets"]}, {128})
        self.assertTrue(any("|distance_controlled|1" in key for key in plan["cohort"]["query_strata"]))
        for counts in plan["cohort"]["answer_cells"].values():
            self.assertEqual(len(set(counts.values())), 1)

    def test_base_specs_unchanged_and_common_cap_applied_to_every_active_arm(self):
        original = deepcopy(self.model)
        plan = self.plan()
        self.assertEqual(self.model, original)
        self.assertEqual(plan["models"], [original])
        self.assertEqual(original["sampling"]["max_output_tokens"], 32)
        for arm in plan["arms"]:
            self.assertEqual(arm["effective_model"]["sampling"]["max_output_tokens"], 2048)
            self.assertEqual(arm["effective_model"]["seed"], original["seed"])
            self.assertEqual(arm["effective_model"]["reasoning"], original["reasoning"])
        self.assertEqual({b["maximum_generated_tokens"] for b in plan["arm_budgets"]}, {128 * 2048})

    def test_disabled_models_are_counted_and_blocked_without_dropping(self):
        plan = self.plan(models=[self.model, self.api])
        self.assertEqual(len(plan["arms"]), 6)
        self.assertEqual(plan["generation_calls"], 768)
        self.assertEqual(len(plan["unavailable_arms"]), 8)
        self.assertTrue(all(arm["generation_calls"] == 0 for arm in plan["unavailable_arms"]))
        self.assertIn("selected_models_disabled", plan["execution_blockers"])
        self.assertFalse(plan["contains_model_results"])
        self.assertEqual(plan["actual_generation_calls"], 0)

    def test_missing_capability_never_assumes_none_or_selects_another_model(self):
        plan = self.plan(models=[self.api])
        self.assertEqual({arm["strategy"] for arm in plan["arms"]}, set(DEFAULT_SETTINGS["strategies"]))
        self.assertEqual({arm["effort"] for arm in plan["unavailable_arms"]}, {"none", "low", "medium", "high"})
        self.assertTrue(all("no_verified" in arm["reason"] for arm in plan["unavailable_arms"]))
        self.assertTrue(all(arm["effective_model"]["model_id"] == self.api["model_id"] for arm in plan["arms"]))

    def test_native_support_is_bound_to_exact_full_name_revision_and_provider(self):
        for changed in ("model_id", "revision"):
            settings = deepcopy(DEFAULT_SETTINGS)
            cap = declaration(self.api)
            cap[changed] += "-different"
            settings["capabilities"] = [cap]
            plan = self.plan(models=[self.api], settings=settings)
            self.assertEqual(len(plan["arms"]), 3)
            self.assertEqual(len(plan["unavailable_arms"]), 4)

    def test_explicit_native_none_is_distinct_from_provider_default(self):
        settings = deepcopy(DEFAULT_SETTINGS)
        settings["capabilities"] = [declaration(self.api)]
        plan = self.plan(models=[self.api], settings=settings)
        self.assertEqual(plan["generation_calls"], 5 * 128)
        native = {arm["effort"]: arm for arm in plan["arms"] if arm["strategy"] == "provider_effort"}
        self.assertEqual(set(native), {"none", "low"})
        self.assertEqual(native["none"]["effective_model"]["reasoning"], {"mode": "none", "effort": "none", "enable_thinking": None})
        self.assertEqual(native["low"]["effective_model"]["reasoning"]["mode"], "thinking")
        direct = next(arm for arm in plan["arms"] if arm["strategy"] == "direct")
        self.assertEqual(direct["effective_model"]["reasoning"]["mode"], "provider_default")
        self.assertIsNone(direct["effort"])
        self.assertEqual({arm["effort"] for arm in plan["unavailable_arms"]}, {"medium", "high"})

    def test_open_weight_effort_declaration_cannot_be_mapped_to_boolean_thinking(self):
        settings = deepcopy(DEFAULT_SETTINGS)
        settings["capabilities"] = [declaration(self.model)]
        plan = self.plan(settings=settings)
        self.assertEqual(len(plan["arms"]), 3)
        self.assertTrue(all("adapter" in arm["reason"] for arm in plan["unavailable_arms"]))
        self.assertTrue(all(arm["effective_model"]["reasoning"]["enable_thinking"] is None for arm in plan["arms"]))

    def test_sampling_restrictions_block_whole_plan_without_removing_arms_or_parameters(self):
        settings = deepcopy(DEFAULT_SETTINGS)
        settings["capabilities"] = [declaration(self.api, ("low",), rule="must_be_null")]
        plan = self.plan(models=[self.api], settings=settings)
        self.assertEqual(len(plan["arms"]), 4)
        self.assertEqual(plan["generation_calls"], 512)
        self.assertEqual(len([b for b in plan["execution_blockers"] if "sampling_incompatible" in b]), 2)
        for arm in plan["arms"]:
            self.assertEqual(arm["effective_model"]["sampling"]["temperature"], 0)
            self.assertEqual(arm["effective_model"]["sampling"]["top_p"], 1)
        model = deepcopy(self.api)
        model["sampling"].update(temperature=None, top_p=None)
        explicit = self.plan(models=[model], settings=settings)
        self.assertFalse(any("sampling_incompatible" in b for b in explicit["execution_blockers"]))

    def test_instruction_bytes_are_an_explicit_proxy_and_do_not_claim_hf_chat_counts(self):
        plan = self.plan()
        budgets = {row["strategy"]: row for row in plan["arm_budgets"]}
        direct = budgets["direct"]
        self.assertEqual(direct["instruction_bytes_used_as_proxy_tokens"], 0)
        self.assertEqual(direct["source_prompt_counts_by_language"], {"en": 64, "tr": 64})
        for strategy in ("zero_shot_reasoning", "structured_reasoning"):
            row = budgets[strategy]
            self.assertGreater(row["estimated_input_tokens"], direct["estimated_input_tokens"])
            self.assertEqual(row["estimated_input_tokens"] - direct["estimated_input_tokens"], row["instruction_bytes_used_as_proxy_tokens"])
            self.assertFalse(row["chat_count_exact"])
            self.assertIsNone(row["actual_new_message_token_count"])
        self.assertTrue(any("planning proxies" in note for note in plan["assumptions"]))

    def test_common_output_cap_violation_fails_before_dataset_access(self):
        model = deepcopy(self.model)
        model["limits"].update(context_tokens=1024, max_output_tokens=1024)
        with patch("reasoning.planning.GridDataset", side_effect=AssertionError("must fail first")):
            with self.assertRaisesRegex(ValueError, "common output"):
                make_plan(PROJECT, MANIFEST, [model])

    def test_auxiliary_http_counts_distinguish_cached_method_calls_and_preflight(self):
        vllm = deepcopy(self.model)
        vllm.update(id="fixture-vllm", provider="vllm")
        vllm["serving"].update(base_url="http://localhost:8000/v1", device=None, dtype=None, server_revision="0.13.0")
        plan = self.plan(models=[self.model, self.api, vllm])
        for row in plan["arm_budgets"]:
            if row["model_id"] == self.model["id"]:
                self.assertEqual(row["auxiliary_token_count_requests"], 0)
                self.assertEqual(row["count_input_tokens_method_invocations_per_generation"], 1)
                self.assertEqual(row["nominal_preflight_http_requests_per_arm_invocation"], 0)
            else:
                self.assertEqual(row["auxiliary_token_count_requests"], 128)
                self.assertEqual(row["maximum_auxiliary_token_count_requests"], 384)
                self.assertEqual(row["count_input_tokens_method_invocations_per_generation"], 2)
                self.assertIn("provider cache", row["auxiliary_http_count_assumption"])
                expected = 1 if row["model_id"] == self.api["id"] else 3
                self.assertEqual(row["nominal_preflight_http_requests_per_arm_invocation"], expected)
                self.assertEqual(row["maximum_preflight_http_requests_per_arm_invocation"], expected * 3)

    def test_replicates_and_explicit_selection_are_counted_without_changing_group_support(self):
        one = self.plan()
        two = self.plan(policy={**DEFAULT_POLICY, "replicates": 2})
        self.assertEqual(one["cohort"], two["cohort"])
        self.assertEqual(two["generation_calls"], one["generation_calls"] * 2)
        selected = self.plan(selection=default_selection("pilot"))
        self.assertEqual(selected["cohort"]["prompt_count"], 32)
        self.assertEqual(selected["generation_calls"], 96)

    def test_budget_context_unknown_cost_and_full_approval_blockers_are_explicit(self):
        capped = self.plan(policy={**DEFAULT_POLICY, "max_calls": 1, "max_estimated_cost": 0})
        self.assertIn("call_budget_exceeded", capped["execution_blockers"])
        self.assertIn("cost_cap_cannot_be_verified", capped["execution_blockers"])
        full = self.plan(stage="full")
        self.assertIn("completed_real_pilot_and_explicit_approval_required", full["execution_blockers"])
        self.assertEqual(full["cohort"]["selection"]["split"], "test")
        model = deepcopy(self.model)
        model["limits"].update(context_tokens=2100, max_output_tokens=2048)
        self.assertIn("possible_context_overflow_requires_exact_preflight", self.plan(models=[model])["execution_blockers"])

    def test_invalid_stage_duplicate_model_and_nonheldout_split_are_rejected(self):
        for kwargs in ({"stage": "training"}, {"models": []}, {"models": [self.model, self.model]}):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                self.plan(**kwargs)
        selection = default_selection("full")
        with self.assertRaisesRegex(ValueError, "partitions"):
            self.plan(selection=selection)

    def test_plan_and_settings_hashes_and_rehashed_derived_fields_reject_tampering(self):
        original = self.plan()
        validate_plan(original, PROJECT)
        mutations = (lambda p: p.update(generation_calls=0), lambda p: p["arms"].pop(),
                     lambda p: p["arm_budgets"][0].update(estimated_input_tokens=0),
                     lambda p: p["unavailable_arms"][0].update(generation_calls=128),
                     lambda p: p.update(estimated_cost_total=0),
                     lambda p: p["settings"].update(output_budget=1024),
                     lambda p: p.update(contains_model_results=True),
                     lambda p: p.update(actual_generation_calls=True))
        for mutate in mutations:
            with self.subTest(mutate=mutate):
                value = deepcopy(original)
                mutate(value)
                rehash(value)
                with self.assertRaises(ValueError):
                    validate_plan(value)
        value = deepcopy(original); value["generation_calls"] = 0
        with self.assertRaisesRegex(ValueError, "fingerprint"):
            validate_plan(value)

    def test_old_inference_plan_schema_is_not_accepted(self):
        value = {"schema_version": 1, "artifact_type": "inference_dry_run_plan", "inference_version": "0.5.0"}
        value["plan_sha256"] = fingerprint(value)
        with self.assertRaisesRegex(ValueError, "schema"):
            validate_plan(value)

    def test_source_binding_includes_reused_analysis_and_all_new_reasoning_modules(self):
        fingerprints = source_fingerprints(PROJECT)
        self.assertIn("src/inference/runner.py", fingerprints)
        self.assertIn("src/analysis/statistics.py", fingerprints)
        self.assertIn("src/reasoning/planning.py", fingerprints)
        with tempfile.TemporaryDirectory(prefix="reasoning-source-fixture-", dir=WORK) as directory:
            root = Path(directory)
            shutil.copytree(PROJECT / "src", root / "src", ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
            shutil.copytree(PROJECT / "scripts", root / "scripts")
            self.assertEqual(source_fingerprints(root), fingerprints)
            target = root / "src/reasoning/planning.py"
            target.write_bytes(target.read_bytes() + b"\n# synthetic changed installed source\n")
            with self.assertRaisesRegex(ValueError, "source differs"):
                source_fingerprints(root)

    def test_write_plan_is_exclusive_and_offline(self):
        with patch("socket.create_connection", side_effect=AssertionError("No network")) as network:
            plan = self.plan()
        self.assertEqual(network.call_count, 0)
        with tempfile.TemporaryDirectory(prefix="reasoning-plan-fixture-", dir=WORK) as directory:
            path = Path(directory) / "plan.json"
            write_plan(plan, path)
            self.assertEqual(json.loads(path.read_bytes()), plan)
            with self.assertRaises(FileExistsError):
                write_plan(plan, path)


class ReasoningSettingsTests(unittest.TestCase):
    def test_delivered_toml_is_explicit_and_has_no_assumed_capabilities(self):
        self.assertEqual(load_settings(PROJECT / "configs/reasoning.toml"), DEFAULT_SETTINGS)

    def test_strict_settings_reject_unknown_fields_types_dates_urls_and_duplicate_evidence(self):
        api = fixture_model("openai", "fixture-api")
        for field, value in (("schema_version", True), ("output_budget", True), ("output_budget", 0),
                             ("strategies", ["zero_shot_reasoning"]), ("native_efforts", ["minimal"]),
                             ("native_efforts", ["none", "none"])):
            settings = deepcopy(DEFAULT_SETTINGS); settings[field] = value
            with self.subTest(field=field), self.assertRaises(ValueError):
                validate_settings(settings)
        settings = deepcopy(DEFAULT_SETTINGS); settings["unknown"] = 1
        with self.assertRaises(ValueError): validate_settings(settings)
        for field, value in (("verified_on", "2026-2-3"), ("evidence_url", "http://example.org"),
                             ("evidence_url", "https://user:secret@example.org"), ("provider", [])):
            settings = deepcopy(DEFAULT_SETTINGS); cap = declaration(api); cap[field] = value
            settings["capabilities"] = [cap]
            with self.subTest(field=field), self.assertRaises(ValueError): validate_settings(settings)
        settings = deepcopy(DEFAULT_SETTINGS); cap = declaration(api)
        cap["sampling_restrictions"]["none"]["temperature"] = []
        settings["capabilities"] = [cap]
        with self.assertRaises(ValueError): validate_settings(settings)
        settings["capabilities"] = [declaration(api), declaration(api)]
        with self.assertRaisesRegex(ValueError, "duplicate"): validate_settings(settings)

    def test_duplicate_toml_keys_unquoted_dates_and_oversize_file_are_rejected(self):
        with tempfile.TemporaryDirectory(prefix="reasoning-settings-fixture-", dir=WORK) as directory:
            path = Path(directory) / "reasoning.toml"
            path.write_text("schema_version=1\nschema_version=1\n")
            with self.assertRaisesRegex(ValueError, "TOML"): load_settings(path)
            path.write_bytes(b"#" * (1_048_576 + 1))
            with self.assertRaisesRegex(ValueError, "1 MiB"): load_settings(path)


if __name__ == "__main__":
    unittest.main()

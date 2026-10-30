"""Actual frozen cohorts and offline budgets; no API, GPU or model download."""

from contextlib import contextmanager
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import sqlite3
import tempfile
import unittest
from unittest.mock import patch

from complete_grid.grid_design import RELATION_IDS
from inference.config import validate_model
from inference.dataset import GridDataset, default_selection, validate_selection
from inference.planning import (
    DEFAULT_POLICY, make_plan, price_usage, validate_plan, validate_policy, write_plan,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
MANIFEST = Path("data/complete_grid/manifest.json")


def fixture_model(provider="transformers", identifier="fixture-small", size=0.5):
    """A disabled but schema-valid identity; it is never loaded or contacted."""
    api = provider == "openai"
    return {
        "id": identifier, "provider": provider, "family": "fixture-api" if api else "fixture-open",
        "parameters_billion": None if api else size,
        "model_id": "fixture-model-2026-01-01" if api else "fixture/model",
        "revision": "fixture-model-2026-01-01" if api else "a" * 40,
        "tokenizer": {"id": "tiktoken" if api else "fixture/tokenizer",
                      "revision": "tiktoken==0.12.0" if api else "a" * 40,
                      "encoding": "o200k_base" if api else None,
                      "chat_template_sha256": None if api else "b" * 64},
        "serving": {"base_url": "https://api.invalid/v1" if api else None,
                    "api_key_env": "ABSENT_FIXTURE_KEY" if api else None,
                    "timeout_seconds": 60, "device": None if api else "cpu",
                    "dtype": None if api else "float32", "trust_remote_code": False,
                    "local_files_only": True, "server_revision": None,
                    "chat_template_date": None if api else "2026-10-24"},
        "sampling": {"temperature": 0, "top_p": 1, "max_output_tokens": 32},
        "limits": {"context_tokens": 131072, "max_output_tokens": 8192},
        "reasoning": {"mode": "provider_default", "effort": None, "enable_thinking": None},
        "seed": None if api else 11,
        "pricing": {"currency": None, "input_per_million": None, "output_per_million": None,
                    "cached_input_per_million": None, "as_of": None, "source": None},
        "enabled": False,
    }


class InferencePlanningTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # This is a real full-file/hash/source/schema validation, once for the
        # class. Repeated planning uses the same validated read-only bank.
        cls.data = GridDataset(PROJECT_ROOT, MANIFEST).__enter__()

    @classmethod
    def tearDownClass(cls):
        cls.data.__exit__(None, None, None)

    def setUp(self):
        self.small = fixture_model()
        self.large = fixture_model(identifier="fixture-large", size=1.5)
        self.api = fixture_model("openai", "fixture-api")
        for model in (self.small, self.large, self.api):
            validate_model(model)

    def plan(self, models=None, stage="pilot", selection=None, policy=None):
        @contextmanager
        def already_validated(root, manifest):
            self.assertEqual(Path(root), PROJECT_ROOT)
            self.assertEqual(Path(manifest), MANIFEST)
            yield self.data

        with patch("inference.planning.GridDataset", already_validated):
            return make_plan(PROJECT_ROOT, MANIFEST, models or [self.small, self.large],
                             stage=stage, selection=selection, policy=policy)

    def test_small_pilot_has_eight_shared_groups_and_four_presentations_per_query(self):
        plan = self.plan()
        cohort = plan["cohort"]
        self.assertEqual((cohort["group_count"], cohort["query_count"], cohort["designs_per_query"],
                          cohort["prompt_count"]), (8, 8, 4, 32))
        self.assertEqual(sorted(group["n"] for group in cohort["groups"]), [4] * 4 + [8] * 4)
        self.assertEqual(plan["generation_calls"], 64)
        self.assertEqual([budget["generation_calls"] for budget in plan["model_budgets"]], [32, 32])
        self.assertEqual(len({group["latent_instance_id"] for group in cohort["groups"]}), 8)
        for values in cohort["answer_cells"].values():
            self.assertEqual(set(values.values()), {1})

    def test_model_identity_size_order_or_context_does_not_change_source_cohort(self):
        baseline = self.plan()
        constrained = deepcopy(self.api)
        constrained["limits"]["context_tokens"] = 64
        constrained["limits"]["max_output_tokens"] = 32
        comparison = self.plan(models=[constrained, self.large, self.small])
        self.assertEqual(comparison["cohort"], baseline["cohort"])
        self.assertIn("possible_context_overflow_requires_exact_preflight", comparison["execution_blockers"])
        self.assertEqual(comparison["generation_calls"], 96)

    def test_full_heldout_baseline_has_1800_groups_and_7200_calls_per_model_without_rendering(self):
        with patch("complete_grid.dataset_builder._materialized", side_effect=AssertionError("must remain SQL-only")), \
                patch("complete_grid.dataset_builder.ModelTokenizers", side_effect=AssertionError("no tokenizer load")):
            plan = self.plan(stage="full")
        cohort = plan["cohort"]
        self.assertEqual(cohort["selection"]["split"], "test")
        self.assertEqual((cohort["group_count"], cohort["query_count"], cohort["prompt_count"]), (1800, 1800, 7200))
        self.assertEqual(plan["generation_calls"], 14400)
        self.assertEqual([budget["generation_calls"] for budget in plan["model_budgets"]], [7200, 7200])
        self.assertEqual(len(cohort["answer_cells"]), 9)
        for values in cohort["answer_cells"].values():
            self.assertEqual(set(values.values()), {50})
        self.assertIn("call_budget_exceeded", plan["execution_blockers"])
        self.assertIn("completed_real_pilot_and_explicit_approval_required", plan["execution_blockers"])

    def test_endpoint_internal_and_controlled_distances_remain_separate_balanced_strata(self):
        selection = default_selection()
        selection["query_types"] = ["endpoint", "internal", "distance_controlled"]
        cohort = self.data.select(selection)
        self.assertEqual((cohort["group_count"], cohort["query_count"], cohort["prompt_count"]), (8, 32, 128))
        self.assertEqual(set(cohort["query_strata"]), {
            "4|endpoint|None", "4|internal|None", "4|distance_controlled|1",
            "8|endpoint|None", "8|internal|None", "8|distance_controlled|1",
            "8|distance_controlled|2", "8|distance_controlled|4"})
        for values in cohort["answer_cells"].values():
            self.assertEqual(set(values.values()), {1})

    def test_distance_filter_cannot_silently_remove_selected_latent_groups(self):
        selection = default_selection()
        selection.update(query_types=["distance_controlled"], distances=[4])
        with self.assertRaisesRegex(ValueError, "silently drop"):
            self.data.select(selection)

    def test_full_relation_name_and_valid_language_conditions_cross_without_inventing_english_ablations(self):
        selection = default_selection()
        selection.update(sizes=[4], name_controls=["native_shared", "en_transliterated", "symbolic_shared"],
                         relations=list(RELATION_IDS),
                         conditions=["canonical", "equivalent", "P1", "P2", "P3", "P4", "P5"])
        cohort = self.data.select(selection)
        self.assertEqual((cohort["designs_per_query"], cohort["prompt_count"]), (243, 972))

    def test_disabled_registry_examples_are_plannable_but_not_executable(self):
        with patch("urllib.request.urlopen", side_effect=AssertionError("no HTTP")), \
                patch("inference.providers.create_provider", side_effect=AssertionError("no provider")), \
                patch("complete_grid.dataset_builder.ModelTokenizers", side_effect=AssertionError("no tokenizer")):
            plan = self.plan(models=[self.api, self.small])
        self.assertIn("selected_models_disabled", plan["execution_blockers"])
        self.assertFalse(plan["contains_model_results"])
        self.assertEqual(plan["actual_generation_calls"], 0)
        self.assertTrue(all(model["enabled"] is False for model in plan["models"]))

    def test_offline_hf_counts_and_prices_are_unknown_and_proxy_is_explicit(self):
        plan = self.plan(models=[self.small])
        budget = plan["model_budgets"][0]
        self.assertIsNone(budget["actual_new_message_token_count"])
        self.assertFalse(budget["chat_count_exact"])
        self.assertIn("proxy", budget["token_estimate_method"])
        self.assertGreater(budget["estimated_input_tokens"], 0)
        self.assertIsNone(budget["cost"]["amount"])
        self.assertEqual(budget["cost"]["status"], "unknown")
        self.assertIsNone(plan["estimated_cost_total"])
        self.assertEqual(budget["auxiliary_token_count_requests"], 0)

    def test_http_auxiliary_count_requests_and_retry_upper_bound_are_reported(self):
        plan = self.plan(models=[self.api])
        budget = plan["model_budgets"][0]
        self.assertEqual(budget["generation_calls"], 32)
        self.assertEqual(budget["auxiliary_token_count_requests"], 32)
        self.assertEqual(budget["maximum_generation_attempts"], 96)
        self.assertEqual(budget["maximum_generated_tokens"], 1024)

    def test_over_call_budget_retains_entire_cohort_and_returns_blocker(self):
        baseline = self.plan()
        policy = {**DEFAULT_POLICY, "max_calls": 1}
        constrained = self.plan(policy=policy)
        self.assertEqual(constrained["cohort"], baseline["cohort"])
        self.assertEqual(constrained["generation_calls"], 64)
        self.assertIn("call_budget_exceeded", constrained["execution_blockers"])

    def test_unknown_price_blocks_cost_capped_execution_without_zero_cost_substitution(self):
        plan = self.plan(policy={**DEFAULT_POLICY, "max_estimated_cost": 100})
        self.assertIn("cost_cap_cannot_be_verified", plan["execution_blockers"])
        self.assertIsNone(plan["estimated_cost_total"])
        self.assertEqual(plan["cohort"]["prompt_count"], 32)

    def test_priced_budget_uses_explicit_price_snapshot_and_enforces_cap(self):
        model = deepcopy(self.api)
        model["pricing"].update(currency="USD", input_per_million=2.5, output_per_million=10,
                                as_of="2026-10-24", source="https://provider.invalid/prices")
        plan = self.plan(models=[model], policy={**DEFAULT_POLICY, "max_estimated_cost": 0})
        self.assertGreater(plan["estimated_cost_total"], 0)
        self.assertIn("estimated_cost_cap_exceeded", plan["execution_blockers"])
        self.assertEqual(plan["model_budgets"][0]["cost"]["pricing_as_of"], "2026-10-24")

    def test_price_usage_honors_cached_tokens_and_requires_known_cached_rate(self):
        model = deepcopy(self.api)
        model["pricing"].update(currency="USD", input_per_million=2, output_per_million=4,
                                cached_input_per_million=1, as_of="2026-10-24",
                                source="https://provider.invalid/prices")
        cost = price_usage(model, {"input_tokens": 1000000, "output_tokens": 1000000,
                                  "cached_input_tokens": 500000})
        self.assertEqual(cost["amount"], 5.5)
        model["pricing"]["cached_input_per_million"] = None
        self.assertEqual(price_usage(model, {"input_tokens": 100, "output_tokens": 1,
                                            "cached_input_tokens": 1})["status"], "unknown_cached_rate")
        with self.assertRaises(ValueError):
            price_usage(model, {"input_tokens": 1, "output_tokens": 1, "cached_input_tokens": 2})

    def test_replicates_multiply_every_model_budget_and_cache_cannot_fake_independence(self):
        plan = self.plan(policy={**DEFAULT_POLICY, "replicates": 3})
        self.assertEqual(plan["generation_calls"], 192)
        self.assertEqual(plan["cohort"]["prompt_count"], 32)
        self.assertEqual([b["generation_calls"] for b in plan["model_budgets"]], [96, 96])
        with self.assertRaises(ValueError):
            validate_policy({**DEFAULT_POLICY, "cache": True, "replicates": 2})

    def test_cohort_and_plan_hashes_are_stable_and_bind_original_manifest(self):
        first, second = self.plan(), self.plan()
        self.assertEqual(first, second)
        self.assertEqual(first["cohort"]["dataset_manifest_sha256"],
                         hashlib.sha256((PROJECT_ROOT / MANIFEST).read_bytes()).hexdigest())
        self.assertEqual(len(first["source_files_sha256"]),
                         len(list((PROJECT_ROOT / "src/inference").glob("*.py")))
                         + len(list((PROJECT_ROOT / "scripts").glob("run_*.py"))))
        validate_plan(first, PROJECT_ROOT)
        tampered = deepcopy(first)
        tampered["cohort"]["query_ids"][0] += 1
        with self.assertRaisesRegex(ValueError, "fingerprint"):
            validate_plan(tampered)

    def test_cohort_tampering_is_rejected_before_any_lengths_or_rendering(self):
        cohort = self.data.select(default_selection())
        for key, value in (("query_ids", []), ("design_ids", [1]), ("prompt_count", 1)):
            tampered = deepcopy(cohort)
            tampered[key] = value
            with self.assertRaisesRegex(ValueError, "frozen source"):
                self.data.source_lengths(tampered)

    def test_changed_inference_source_binding_invalidates_previously_created_plan(self):
        plan = self.plan()
        with patch("inference.planning.source_fingerprints", return_value={}):
            with self.assertRaisesRegex(ValueError, "source files changed"):
                validate_plan(plan, PROJECT_ROOT)

    def test_changed_bound_renderer_sources_are_rejected_before_cohort_selection(self):
        # Reuse only the already hash-validated file/metadata binding. The source
        # mismatch and read-only DB schema checks remain real on the new reader.
        def validated_files(root, manifest_path):
            connection = sqlite3.connect("file:" + str(PROJECT_ROOT / "data/complete_grid/grid.sqlite") + "?mode=ro", uri=True)
            connection.row_factory = sqlite3.Row
            return PROJECT_ROOT, deepcopy(self.data.manifest), connection

        with patch("complete_grid.dataset_builder._load_manifest", side_effect=validated_files), \
                patch("complete_grid.dataset_builder._source_hashes", return_value={}):
            with self.assertRaisesRegex(ValueError, "sources changed"):
                with GridDataset(PROJECT_ROOT, MANIFEST):
                    pass

    def test_eval_partitions_and_presentation_pairs_cannot_be_silently_changed(self):
        selection = default_selection()
        selection["split"] = "training"
        with self.assertRaises(ValueError):
            validate_selection(selection)
        selection = default_selection(); selection["conditions"] = ["canonical"]
        with self.assertRaises(ValueError):
            validate_selection(selection)
        selection = default_selection(); selection["split"] = "development"
        with self.assertRaisesRegex(ValueError, "disjoint"):
            self.plan(selection=selection)

    def test_unavailable_groups_and_imbalanced_subsamples_are_rejected(self):
        for count in (2, 6, 52):
            selection = default_selection(); selection["groups_per_size"] = count
            with self.assertRaises(ValueError):
                self.data.select(selection)

    def test_written_plan_is_exclusive_and_reloaded_plan_passes_hash_validation(self):
        plan = self.plan()
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "plan.json"
            write_plan(plan, path)
            content = path.read_bytes()
            validate_plan(json.loads(content))
            with self.assertRaises(FileExistsError):
                write_plan(plan, path)
            self.assertEqual(path.read_bytes(), content)


if __name__ == "__main__":
    unittest.main()

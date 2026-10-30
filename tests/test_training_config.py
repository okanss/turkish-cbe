"""Strict disabled training configuration checks; no backend is imported."""

from copy import deepcopy
import importlib
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from training.config import ARMS, DEPENDENCY_PINS, LORA_TARGETS, load_config, validate_config


PROJECT = Path(__file__).resolve().parents[1]
WORK = PROJECT.parents[1] / "work"


class TrainingConfigTests(unittest.TestCase):
    def setUp(self):
        self.config = load_config(PROJECT / "configs/training/pilot.toml")

    def test_three_delivered_profiles_are_disabled_and_have_exact_six_arms(self):
        for profile in ("pilot", "reference_inspired", "matched_steps"):
            with self.subTest(profile=profile):
                cfg = load_config(PROJECT / "configs/training" / (profile + ".toml"))
                self.assertEqual(cfg["profile"], profile)
                self.assertFalse(cfg["enable_execution"])
                self.assertEqual(cfg["seed"], 42)
                self.assertEqual(tuple(cfg["arms"]), ARMS)
                self.assertEqual(cfg["model_id"], "qwen25-15b-hf")
                self.assertEqual(cfg["dependencies"], DEPENDENCY_PINS)
                self.assertEqual(tuple(cfg["lora"]["target_modules"]), LORA_TARGETS)
                self.assertIsNone(cfg["budget"]["cost_per_gpu_hour"])
                self.assertIsNone(cfg["budget"]["currency"])

    def test_pilot_and_reference_data_and_step_budgets_are_distinct(self):
        pilot = self.config
        reference = load_config(PROJECT / "configs/training/reference_inspired.toml")
        matched = load_config(PROJECT / "configs/training/matched_steps.toml")
        self.assertEqual(pilot["data"]["training_sizes"], [4, 8])
        self.assertEqual(pilot["data"]["groups_per_size"], 4)
        self.assertEqual(pilot["data"]["grpo_max_instances"], 8)
        self.assertEqual(pilot["sft"]["max_steps"], 2)
        self.assertEqual(reference["data"]["groups_per_size"], 50)
        self.assertEqual(reference["data"]["training_sizes"], [4, 8, 16, 32, 64, 128])
        self.assertEqual(reference["data"]["grpo_min_n"], 16)
        self.assertEqual(reference["sft"]["max_steps"], 0)
        self.assertEqual(reference["grpo"]["max_steps"], 600)
        self.assertEqual(matched["sft"]["max_steps"], matched["grpo"]["max_steps"])
        self.assertEqual(matched["sft"]["max_steps"], 600)

    def test_validation_does_not_mutate_configuration(self):
        before = deepcopy(self.config)
        validate_config(self.config)
        self.assertEqual(self.config, before)

    def test_loading_and_validation_never_import_or_download_a_backend(self):
        original = importlib.import_module
        def guarded(name, *args, **kwargs):
            if name in {"torch", "transformers", "peft"}:
                raise AssertionError("training backend must not be imported")
            return original(name, *args, **kwargs)
        with patch("importlib.import_module", side_effect=guarded), patch("socket.create_connection", side_effect=AssertionError("no network")):
            cfg = load_config(PROJECT / "configs/training/pilot.toml")
            validate_config(cfg)

    def test_unknown_missing_and_misspelled_fields_are_rejected(self):
        for table, field in ((None, "enable_executon"), ("data", "train_sizes"), ("lora", "r"),
                             ("sft", "lr"), ("runtime", "allow_download"), ("dependencies", "trl")):
            cfg = deepcopy(self.config)
            target = cfg if table is None else cfg[table]
            target[field] = 1
            with self.subTest(table=table), self.assertRaises(ValueError): validate_config(cfg)
        cfg = deepcopy(self.config); del cfg["seed"]
        with self.assertRaises(ValueError): validate_config(cfg)

    def test_boolean_numeric_and_nonfinite_values_are_not_accepted(self):
        for table, field, value in ((None, "seed", True), ("lora", "rank", True),
             ("sft", "learning_rate", float("nan")), ("grpo", "temperature", float("inf")), ("grpo", "top_p", .95),
             ("runtime", "minimum_free_gpu_memory_gib", True), ("budget", "max_length", 10**400)):
            cfg = deepcopy(self.config)
            (cfg if table is None else cfg[table])[field] = value
            with self.subTest(field=field), self.assertRaises(ValueError): validate_config(cfg)

    def test_training_sizes_cannot_include_large_transfer_evaluation_sizes(self):
        for sizes in ([4, 256], [4, 1024], [4, 4], [4.0, 8]):
            cfg = deepcopy(self.config); cfg["data"]["training_sizes"] = sizes
            with self.subTest(sizes=sizes), self.assertRaises(ValueError): validate_config(cfg)

    def test_relation_and_condition_splits_remain_explicit(self):
        cfg = deepcopy(self.config); cfg["data"]["training_relations"][0] = "spatial_front"
        with self.assertRaises(ValueError): validate_config(cfg)
        cfg = deepcopy(self.config); cfg["data"]["evaluation_split"] = "training"
        with self.assertRaises(ValueError): validate_config(cfg)
        cfg = deepcopy(self.config); cfg["data"]["conditions"].remove("P2")
        with self.assertRaises(ValueError): validate_config(cfg)

    def test_groups_are_explicit_and_grpo_instances_are_multiple_four(self):
        cfg = deepcopy(self.config); cfg["data"]["groups_per_size"] = None
        with self.assertRaises(ValueError): validate_config(cfg)
        cfg = deepcopy(self.config); cfg["data"]["grpo_max_instances"] = 7
        with self.assertRaises(ValueError): validate_config(cfg)
        cfg = deepcopy(self.config); cfg["data"]["grpo_rollouts_per_view"] = 8
        with self.assertRaises(ValueError): validate_config(cfg)

    def test_data_budget_bounds_match_the_frozen_protocol(self):
        for field, value in (("groups_per_size", 8), ("evaluation_groups_per_size", 50),
                             ("grpo_max_instances", 604)):
            cfg = deepcopy(self.config); cfg["data"][field] = value
            with self.subTest(field=field), self.assertRaises(ValueError): validate_config(cfg)

    def test_beta_nonzero_is_rejected_instead_of_claiming_an_unimplemented_kl_penalty(self):
        cfg = deepcopy(self.config); cfg["grpo"]["beta"] = 0.01
        with self.assertRaises(ValueError): validate_config(cfg)
        cfg["grpo"]["beta"] = 0
        cfg["grpo"]["advantage_mode"] = "instance_centered"
        validate_config(cfg)

    def test_unregistered_guard_optimizer_and_scheduler_are_rejected(self):
        for table, field, value in (("consistency", "guard", "paper_guard"),
            ("budget", "optimizer", "sgd"), ("budget", "scheduler", "constant")):
            cfg = deepcopy(self.config); cfg[table][field] = value
            with self.subTest(field=field), self.assertRaises(ValueError): validate_config(cfg)

    def test_gpu_backend_has_no_cpu_mps_fp16_or_download_fallback(self):
        for field, value in (("device", "mps"), ("device", "cpu"), ("dtype", "float16"), ("local_files_only", False)):
            cfg = deepcopy(self.config); cfg["runtime"][field] = value
            with self.subTest(field=field), self.assertRaises(ValueError): validate_config(cfg)

    def test_paths_do_not_interpolate_secrets_or_escape_project(self):
        for field, value in (("model_registry", "../private.yaml"), ("dataset_manifest", "/tmp/data.json"),
                             ("model_registry", "${SECRET}/models.yaml")):
            cfg = deepcopy(self.config); cfg[field] = value
            with self.subTest(value=value), self.assertRaises(ValueError): validate_config(cfg)
        cfg = deepcopy(self.config); cfg["runtime"]["weights_path"] = "relative/weights"
        with self.assertRaises(ValueError): validate_config(cfg)

    def test_step_matching_and_global_step_budget_cannot_be_silently_overridden(self):
        cfg = load_config(PROJECT / "configs/training/matched_steps.toml")
        cfg["sft"]["max_steps"] = 599
        with self.assertRaisesRegex(ValueError, "matched_steps"): validate_config(cfg)
        cfg = deepcopy(self.config); cfg["grpo"]["max_steps"] = 3
        with self.assertRaisesRegex(ValueError, "step budget"): validate_config(cfg)

    def test_price_currency_and_exact_dependency_versions_are_validated(self):
        cfg = deepcopy(self.config); cfg["budget"].update(cost_per_gpu_hour=2.5, currency="USD")
        validate_config(cfg)
        cfg["budget"]["currency"] = None
        with self.assertRaises(ValueError): validate_config(cfg)
        for version in ("latest", ">=2.9", "2.9.*"):
            cfg = deepcopy(self.config); cfg["dependencies"]["torch"] = version
            with self.subTest(version=version), self.assertRaises(ValueError): validate_config(cfg)

    def test_toml_duplicate_unknown_types_and_size_limit_are_enforced(self):
        with tempfile.TemporaryDirectory(prefix="synthetic-training-config-", dir=WORK) as directory:
            path = Path(directory) / "config.toml"
            for data in (b"schema_version=1\nschema_version=1\n", b"\xff"):
                path.write_bytes(data)
                with self.assertRaisesRegex(ValueError, "TOML"): load_config(path)
            path.write_bytes(b"#" * (1_048_576 + 1))
            with self.assertRaisesRegex(ValueError, "1 MiB"): load_config(path)


if __name__ == "__main__":
    unittest.main()

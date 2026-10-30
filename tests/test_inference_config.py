"""Model selection must never silently coerce, substitute or lose settings."""

from copy import deepcopy
import importlib.util
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from inference.config import ConfigError, load_models, select_models, validate_model


PROJECT_ROOT = Path(__file__).resolve().parents[1]
HAS_YAML = importlib.util.find_spec("yaml") is not None


@unittest.skipUnless(HAS_YAML, "optional PyYAML dependency is not installed")
class ModelRegistryTests(unittest.TestCase):
    def setUp(self):
        self.registry = load_models(PROJECT_ROOT / "configs/models.yaml")
        self.openai = deepcopy(self.registry["models"][0])
        self.hf = next(deepcopy(x) for x in self.registry["models"]
                       if x["provider"] == "transformers" and x["family"] == "qwen2.5")
        self.vllm = next(deepcopy(x) for x in self.registry["models"] if x["provider"] == "vllm")

    def load_text(self, source):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "models.yaml"
            path.write_text(source, encoding="utf-8")
            return load_models(path)

    def test_examples_cover_providers_families_sizes_without_enabling(self):
        specs = self.registry["models"]
        self.assertEqual({x["provider"] for x in specs}, {"openai", "transformers", "vllm"})
        families = {x["family"] for x in specs if x["provider"] == "transformers"}
        self.assertGreaterEqual(len(families), 2)
        for family in families:
            self.assertGreaterEqual(len({x["parameters_billion"] for x in specs
                                         if x["family"] == family}), 2)
        # Enablement is an explicit, recorded user selection (2026-10-27, for
        # the executed pilot runs); any further enablement must update this pin.
        self.assertEqual({x["id"] for x in specs if x["enabled"]},
                         {"qwen25-05b-hf", "openai-gpt4o-snapshot"})
        self.assertTrue(all(x["pricing"]["input_per_million"] is None for x in specs if x["provider"] != "openai"))
        self.assertEqual(self.openai["pricing"]["currency"], "USD")
        self.assertEqual(self.openai["pricing"]["as_of"], "2026-10-24")
        self.assertEqual(self.openai["pricing"]["input_per_million"], 2.5)

    def test_exact_selection_order_and_detached_values(self):
        ids = [self.registry["models"][-1]["id"], self.registry["models"][0]["id"]]
        selected = select_models(self.registry, ids)
        self.assertEqual([x["id"] for x in selected], ids)
        selected[0]["sampling"]["max_output_tokens"] = 1
        self.assertEqual(self.registry["models"][-1]["sampling"]["max_output_tokens"], 32)

    def test_missing_and_duplicate_selection_never_substitute(self):
        identifier = self.openai["id"]
        for ids in ([], ["absent"], [identifier, identifier], "all", [True]):
            with self.subTest(ids=ids), self.assertRaises(ConfigError):
                select_models(self.registry, ids)

    def test_duplicate_yaml_and_registry_identifiers_rejected(self):
        with self.assertRaises(ConfigError):
            self.load_text("schema_version: 1\nschema_version: 1\nmodels: []\n")
        self.registry["models"].append(deepcopy(self.registry["models"][0]))
        with self.assertRaises(ConfigError):
            select_models(self.registry, [self.openai["id"]])

    def test_unknown_and_missing_fields_rejected_at_every_level(self):
        for section in (None, "tokenizer", "serving", "sampling", "limits", "reasoning", "pricing"):
            for action in ("unknown", "missing"):
                spec = deepcopy(self.openai)
                target = spec if section is None else spec[section]
                if action == "unknown":
                    target["silent_capability"] = True
                else:
                    del target[next(iter(target))]
                with self.subTest(section=section, action=action), self.assertRaises(ConfigError):
                    validate_model(spec)

    def test_yaml_aliases_recursive_objects_and_tags_rejected(self):
        for source in ("a: &a [*a]", "a: &a 1\nb: *a", "!!python/object:os.system {}",
                       "schema_version: 1\nmodels: []\n---\nmodels: []"):
            with self.subTest(source=source), self.assertRaises(ConfigError):
                self.load_text(source)

    def test_schema_version_is_exact_integer_one(self):
        for value in (True, 1.0, 2, "1"):
            registry = deepcopy(self.registry)
            registry["schema_version"] = value
            with self.subTest(value=value), self.assertRaises(ConfigError):
                select_models(registry, [self.openai["id"]])

    def test_boolean_is_never_integer_numeric_setting(self):
        for section, key in ((None, "seed"), (None, "parameters_billion"),
                             ("limits", "context_tokens"), ("sampling", "temperature"),
                             ("sampling", "max_output_tokens"), ("serving", "timeout_seconds"),
                             ("pricing", "input_per_million")):
            spec = deepcopy(self.openai)
            (spec if section is None else spec[section])[key] = True
            with self.subTest(section=section, key=key), self.assertRaises(ConfigError):
                validate_model(spec)

    def test_nonfinite_extreme_and_negative_numbers_rejected(self):
        for value in (float("inf"), float("-inf"), float("nan"), -1, 10**400):
            spec = deepcopy(self.hf)
            spec["sampling"]["temperature"] = value
            with self.subTest(value=str(value)), self.assertRaises(ConfigError):
                validate_model(spec)

    def test_sampling_omission_is_preserved(self):
        self.openai["sampling"]["temperature"] = None
        self.openai["sampling"]["top_p"] = None
        before = deepcopy(self.openai)
        validate_model(self.openai)
        self.assertEqual(self.openai, before)

    def test_context_output_and_sampling_bounds(self):
        for section, key, value in (("sampling", "top_p", 0), ("sampling", "top_p", 1.01),
                                    ("sampling", "max_output_tokens", 8),
                                    ("sampling", "max_output_tokens", 20000),
                                    ("limits", "max_output_tokens", 200000),
                                    ("limits", "context_tokens", 0)):
            spec = deepcopy(self.openai)
            spec[section][key] = value
            with self.subTest(section=section, key=key), self.assertRaises(ConfigError):
                validate_model(spec)

    def test_openai_requires_exact_snapshot_and_no_invented_seed(self):
        self.openai["enabled"] = True
        validate_model(self.openai)
        for key, value in (("model_id", "gpt-4o"), ("revision", "some-other-model"), ("seed", 11),
                           ("parameters_billion", 123)):
            spec = deepcopy(self.openai)
            spec[key] = value
            with self.subTest(key=key), self.assertRaises(ConfigError):
                validate_model(spec)

    def test_open_weight_revision_and_template_pins(self):
        self.hf["enabled"] = True
        validate_model(self.hf)
        for section, key, value in ((None, "revision", "main"),
                                   ("tokenizer", "revision", "v1"),
                                   ("tokenizer", "chat_template_sha256", None)):
            spec = deepcopy(self.hf)
            (spec if section is None else spec[section])[key] = value
            with self.subTest(section=section, key=key), self.assertRaises(ConfigError):
                validate_model(spec)

    def test_disabled_missing_pin_is_pending_not_enabled(self):
        # The live registry may carry a recorded user enablement; this test is
        # about the disabled pending state, so start from it explicitly.
        self.hf["enabled"] = False
        self.hf["revision"] = None
        self.hf["tokenizer"]["revision"] = None
        self.hf["tokenizer"]["chat_template_sha256"] = None
        validate_model(self.hf)
        self.hf["enabled"] = True
        with self.assertRaises(ConfigError):
            validate_model(self.hf)

    def test_vllm_requires_actual_engine_pin_to_enable(self):
        for version in (None, "UNSELECTED", "latest", "main"):
            spec = deepcopy(self.vllm)
            spec["enabled"] = True
            spec["serving"]["server_revision"] = version
            with self.subTest(version=version), self.assertRaises(ConfigError):
                validate_model(spec)
        self.vllm["enabled"] = True
        self.vllm["serving"]["server_revision"] = "0.11.0"
        validate_model(self.vllm)

    def test_provider_default_never_claims_disabled_thinking(self):
        self.openai["reasoning"]["effort"] = "low"
        with self.assertRaises(ConfigError):
            validate_model(self.openai)
        self.hf["reasoning"]["enable_thinking"] = False
        with self.assertRaises(ConfigError):
            validate_model(self.hf)

    def test_reasoning_controls_are_explicit_and_provider_specific(self):
        self.openai["reasoning"] = {"mode": "thinking", "effort": "low", "enable_thinking": None}
        validate_model(self.openai)
        self.openai["reasoning"]["effort"] = "max"
        validate_model(self.openai)
        self.openai["reasoning"]["effort"] = None
        with self.assertRaises(ConfigError):
            validate_model(self.openai)
        self.hf["reasoning"] = {"mode": "none", "effort": None, "enable_thinking": False}
        validate_model(self.hf)
        self.hf["reasoning"]["effort"] = "low"
        with self.assertRaises(ConfigError):
            validate_model(self.hf)

    def test_raw_keys_interpolation_and_credential_urls_rejected(self):
        for field, value in (("api_key_env", "sk-secret-value"),
                             ("api_key_env", "${OPENAI_API_KEY}"),
                             ("base_url", "https://username:secret@example.com/v1"),
                             ("base_url", "https://example.com/v1?key=secret"),
                             ("base_url", "https://[")):
            spec = deepcopy(self.openai)
            spec["serving"][field] = value
            with self.subTest(field=field, value=value), self.assertRaises(ConfigError):
                validate_model(spec)

    def test_prices_are_unknown_or_dated_sourced_values(self):
        validate_model(self.openai)
        self.openai["pricing"] = {key: None for key in self.openai["pricing"]}
        validate_model(self.openai)
        self.openai["pricing"]["input_per_million"] = 1.0
        with self.assertRaises(ConfigError):
            validate_model(self.openai)
        self.openai["pricing"].update(currency="USD", as_of="2026-10-24",
                                      source="https://example.com/pricing", output_per_million=4)
        validate_model(self.openai)
        self.openai["pricing"]["as_of"] = "2026-02-30"
        with self.assertRaises(ConfigError):
            validate_model(self.openai)

    def test_chat_template_date_is_frozen_literal_not_wall_clock(self):
        self.hf["serving"]["chat_template_date"] = "2026-10-24"
        validate_model(self.hf)
        for value in ("now", "20261024", "2026-02-30", "2026-10-24T00:00:00", 20261024):
            spec = deepcopy(self.hf)
            spec["serving"]["chat_template_date"] = value
            with self.subTest(value=value), self.assertRaises(ConfigError):
                validate_model(spec)
        self.openai["serving"]["chat_template_date"] = "2026-10-24"
        with self.assertRaises(ConfigError):
            validate_model(self.openai)

    def test_bad_container_values_do_not_escape_as_type_errors(self):
        for section, field in (("serving", "dtype"), ("reasoning", "mode"), ("reasoning", "effort")):
            spec = deepcopy(self.openai)
            spec[section][field] = []
            with self.subTest(section=section, field=field), self.assertRaises(ConfigError):
                validate_model(spec)

    def test_yaml_timestamp_remains_string(self):
        import yaml
        registry = deepcopy(self.registry)
        registry["models"][0]["pricing"].update(
            currency="USD", input_per_million=1, output_per_million=2,
            cached_input_per_million=0.5,
            as_of="2026-10-24", source="https://example.com/pricing")
        source = yaml.safe_dump(registry, sort_keys=False).replace("'2026-10-24'", "2026-10-24")
        restored = self.load_text(source)
        self.assertIsInstance(restored["models"][0]["pricing"]["as_of"], str)

    def test_yaml_yes_on_and_numeric_booleans_are_not_coerced(self):
        import yaml
        source = yaml.safe_dump(self.registry, sort_keys=False)
        for spelling in ("yes", "on", "1", "True"):
            with self.subTest(spelling=spelling), self.assertRaises(ConfigError):
                self.load_text(source.replace("enabled: false", f"enabled: {spelling}", 1))

    def test_size_limit_before_parsing(self):
        with self.assertRaises(ConfigError):
            self.load_text("#" * (1_048_576 + 1))


class OptionalYamlDependencyTests(unittest.TestCase):
    def test_missing_optional_dependency_is_actionable(self):
        with patch.dict("sys.modules", {"yaml": None}):
            with self.assertRaisesRegex(RuntimeError, "optional PyYAML"):
                load_models("not-read-without-dependency.yaml")


if __name__ == "__main__":
    unittest.main()

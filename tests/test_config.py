"""Configuration invariants; no datasets or model requests are generated."""

from copy import deepcopy
from pathlib import Path
import tempfile
import unittest

from turkish_cbe.config import ConfigError, load_config, parse_config


PROJECT_ROOT = Path(__file__).resolve().parents[1]


class ConfigTests(unittest.TestCase):
    def setUp(self):
        self.raw = {
            "schema_version": 1,
            "protocol_id": "tr_cbe_v0_1",
            "stage": "pilot",
            "status": "planning_only",
            "task": "ordering_endpoint",
            "relation": "front_behind",
            "seeds": [11],
            "sizes": [4, 8, 16],
            "instances_per_size_per_seed": 16,
            "languages": ["tr", "en"],
            "presentations": [
                {"language": language, "variant": variant}
                for language in ("tr", "en")
                for variant in ("canonical", "converse")
            ],
            "models": [{
                "slot": "model_a", "provider": "unresolved",
                "model_id": "unresolved", "revision": "unresolved",
                "tokenizer_revision": "unresolved",
            }],
            "prompting": ["direct"],
            "repeats": {"fraction": 0.25, "total_observations": 2, "seed": 401},
            "decoding": {
                "temperature": 0.0, "top_p": 1.0,
                "max_output_tokens": 256, "reasoning_effort": "none",
                "seed_policy": "record_if_supported",
            },
            "analysis": {
                "bootstrap_resamples": 2000, "seed": 701,
                "alpha": 0.05, "multiplicity": "holm",
            },
            "training": {"enabled": False},
            "pending_gates": ["source_verification", "linguistic_review"],
        }

    def assert_invalid(self, path, value):
        raw = deepcopy(self.raw)
        target = raw
        for key in path[:-1]:
            target = target[key]
        target[path[-1]] = value
        with self.assertRaises(ConfigError):
            parse_config(raw)

    def test_pilot_budget_counts_requests_and_stratified_repeats(self):
        config = parse_config(self.raw)
        self.assertEqual(config.budget(), {
            "latent_instances": 48,
            "first_observation_requests": 192,
            "repeat_selected_instances": 12,
            "additional_repeat_requests": 48,
            "total_planned_requests": 240,
        })
        self.assertEqual(config.seeds, (11,))
        self.assertFalse(config.to_dict()["training"]["enabled"])

    def test_baseline_budget_includes_all_presentations_and_models(self):
        self.raw.update(
            stage="baseline", sizes=[4, 8, 16, 32, 64, 128],
            seeds=[101, 102, 103], instances_per_size_per_seed=200,
        )
        self.raw["repeats"]["fraction"] = 0.1
        self.raw["presentations"].extend([
            {"language": "tr", "variant": variant}
            for variant in ("premise_shuffle", "syntax", "morph_joint")
        ])
        second_model = dict(self.raw["models"][0], slot="model_b")
        self.raw["models"].append(second_model)
        self.assertEqual(parse_config(self.raw).budget(), {
            "latent_instances": 3600,
            "first_observation_requests": 50400,
            "repeat_selected_instances": 360,
            "additional_repeat_requests": 5040,
            "total_planned_requests": 55440,
        })

    def test_four_query_cells_must_have_equal_integer_counts(self):
        for count in (0, 3, 5, 17):
            with self.subTest(count=count):
                self.assert_invalid(("instances_per_size_per_seed",), count)

    def test_repeat_selection_must_be_integral_within_each_query_cell(self):
        self.assert_invalid(("repeats", "fraction"), 0.1)
        # Total selection is integer (48 * .125 = 6), but each cell is not.
        self.assert_invalid(("repeats", "fraction"), 0.125)

    def test_duplicates_rejected_in_plan_arrays(self):
        replacements = {
            "seeds": [11, 11], "sizes": [4, 4],
            "languages": ["tr", "en", "tr"],
            "prompting": ["direct", "direct"],
            "pending_gates": ["review", "review"],
        }
        for key, value in replacements.items():
            with self.subTest(key=key):
                self.assert_invalid((key,), value)

    def test_duplicate_presentation_and_model_slot_rejected(self):
        for key in ("presentations", "models"):
            with self.subTest(key=key):
                self.assert_invalid((key,), self.raw[key] + [deepcopy(self.raw[key][0])])

    def test_missing_and_unknown_root_keys_rejected(self):
        for key in self.raw:
            with self.subTest(missing=key):
                raw = deepcopy(self.raw)
                del raw[key]
                with self.assertRaises(ConfigError):
                    parse_config(raw)
        self.assert_invalid(("unrecognized",), True)

    def test_missing_and_unknown_nested_keys_rejected(self):
        paths = [
            ("presentations", 0), ("models", 0), ("repeats",),
            ("decoding",), ("analysis",), ("training",),
        ]
        for path in paths:
            original = self.raw
            for key in path:
                original = original[key]
            for key in original:
                with self.subTest(path=path, missing=key):
                    replacement = dict(original)
                    del replacement[key]
                    self.assert_invalid(path, replacement)
            with self.subTest(path=path, unknown=True):
                self.assert_invalid(path, dict(original, unrecognized=True))

    def test_malformed_root_and_nested_containers_raise_config_error(self):
        for raw in (None, [], "plan", 1, True):
            with self.subTest(root=raw), self.assertRaises(ConfigError):
                parse_config(raw)
        nested_paths = [
            ("seeds",), ("sizes",), ("languages",), ("presentations",),
            ("models",), ("prompting",), ("pending_gates",),
            ("repeats",), ("decoding",), ("analysis",), ("training",),
            ("presentations", 0), ("models", 0),
        ]
        for path in nested_paths:
            for malformed in (None, 1, True, "text", []):
                with self.subTest(path=path, malformed=malformed):
                    self.assert_invalid(path, malformed)

    def test_malformed_array_elements_raise_config_error(self):
        for key in ("seeds", "sizes", "languages", "prompting", "pending_gates"):
            for element in (None, [], {}, True, ""):
                with self.subTest(key=key, element=element):
                    self.assert_invalid((key,), [element])
        for key in ("presentations", "models"):
            for element in (None, [], 1, "text"):
                with self.subTest(key=key, element=element):
                    self.assert_invalid((key,), [element])

    def test_presentation_fields_reject_nonstrings_without_type_error(self):
        for field in ("language", "variant"):
            for value in (None, [], {}, 1, True):
                with self.subTest(field=field, value=value):
                    self.assert_invalid(("presentations", 0, field), value)

    def test_only_supported_language_variant_pairs_accepted(self):
        for languages in (["tr"], ["en"], ["tr", "en", "de"], ["TR", "en"]):
            with self.subTest(languages=languages):
                self.assert_invalid(("languages",), languages)
        for entry in (
            {"language": "de", "variant": "canonical"},
            {"language": "tr", "variant": "unimplemented"},
            {"language": "en", "variant": "syntax"},
            {"language": "en", "variant": "morph_joint"},
            {"language": "en", "variant": "premise_shuffle"},
        ):
            with self.subTest(entry=entry):
                self.assert_invalid(("presentations", 0), entry)

    def test_canonical_and_converse_required_in_both_languages(self):
        for index in range(4):
            with self.subTest(index=index):
                self.assert_invalid(("presentations",), [
                    entry for i, entry in enumerate(self.raw["presentations"])
                    if i != index
                ])

    def test_boolean_values_are_not_integers(self):
        integer_paths = [
            ("schema_version",), ("seeds", 0), ("sizes", 0),
            ("instances_per_size_per_seed",),
            ("repeats", "total_observations"), ("repeats", "seed"),
            ("decoding", "max_output_tokens"),
            ("analysis", "bootstrap_resamples"), ("analysis", "seed"),
        ]
        for path in integer_paths:
            for value in (True, False):
                with self.subTest(path=path, value=value):
                    self.assert_invalid(path, value)

    def test_integer_bounds_and_increasing_sizes(self):
        for path, value in (
            (("seeds", 0), -1), (("sizes", 0), 3),
            (("sizes",), [8, 4]), (("repeats", "seed"), -1),
            (("repeats", "total_observations"), 1),
            (("decoding", "max_output_tokens"), 0),
            (("analysis", "seed"), -1),
            (("analysis", "bootstrap_resamples"), 0),
        ):
            with self.subTest(path=path, value=value):
                self.assert_invalid(path, value)

    def test_numeric_ranges_reject_nan_infinity_booleans_and_strings(self):
        paths = [
            ("repeats", "fraction"), ("decoding", "temperature"),
            ("decoding", "top_p"), ("analysis", "alpha"),
        ]
        for path in paths:
            for value in (float("nan"), float("inf"), -float("inf"), True, False, "0.5"):
                with self.subTest(path=path, value=value):
                    self.assert_invalid(path, value)

    def test_numeric_range_boundaries(self):
        for path, values in (
            (("repeats", "fraction"), (0, -0.1, 1.1)),
            (("decoding", "temperature"), (-0.1, 2.1)),
            (("decoding", "top_p"), (0, -0.1, 1.1)),
            (("analysis", "alpha"), (0, 1, -0.1)),
        ):
            for value in values:
                with self.subTest(path=path, value=value):
                    self.assert_invalid(path, value)
        self.raw["repeats"]["fraction"] = 1
        self.raw["decoding"].update(temperature=2, top_p=1)
        self.assertEqual(parse_config(self.raw).repeats.fraction, 1)

    def test_training_must_be_disabled_with_literal_false(self):
        for value in (True, 0, 1, None, "false", []):
            with self.subTest(value=value):
                self.assert_invalid(("training", "enabled"), value)

    def test_reasoning_requires_advanced_stage_and_direct_control(self):
        for stage in ("pilot", "baseline"):
            self.raw["stage"] = stage
            self.assert_invalid(("prompting",), ["direct", "reasoning"])
        self.raw["stage"] = "advanced"
        self.raw["prompting"] = ["direct", "reasoning"]
        self.assertEqual(parse_config(self.raw).prompting, ("direct", "reasoning"))
        self.assert_invalid(("prompting",), ["reasoning"])
        self.assert_invalid(("prompting",), ["direct", "chain_of_thought"])

    def test_unsupported_scope_and_decoding_controls_rejected(self):
        for key, value in (
            ("schema_version", 2), ("protocol_id", "other"),
            ("status", "ready"), ("stage", "training"),
            ("task", "other"), ("relation", "other"),
        ):
            with self.subTest(key=key):
                self.assert_invalid((key,), value)
        self.assert_invalid(("decoding", "reasoning_effort"), "high")
        self.assert_invalid(("decoding", "seed_policy"), "assume_deterministic")
        self.assert_invalid(("analysis", "multiplicity"), "none")

    def test_model_fields_must_be_nonempty_strings(self):
        for field in self.raw["models"][0]:
            for value in (None, [], {}, True, "", " "):
                with self.subTest(field=field, value=value):
                    self.assert_invalid(("models", 0, field), value)

    def test_normalized_config_does_not_alias_mutable_input(self):
        config = parse_config(self.raw)
        self.raw["seeds"].append(12)
        self.raw["models"][0]["slot"] = "changed"
        exported = config.to_dict()
        exported["repeats"]["seed"] = 999
        self.assertEqual(config.seeds, (11,))
        self.assertEqual(config.models[0].slot, "model_a")
        self.assertEqual(config.repeats.seed, 401)

    def test_invalid_toml_is_reported_as_config_error(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "malformed.toml"
            path.write_text("stage = [\n", encoding="utf-8")
            with self.assertRaises(ConfigError):
                load_config(path)

    def test_committed_plans_load_with_disjoint_generation_seeds(self):
        plans = {
            stage: load_config(PROJECT_ROOT / "configs" / f"{stage}.toml")
            for stage in ("pilot", "baseline", "advanced")
        }
        for stage, config in plans.items():
            with self.subTest(stage=stage):
                self.assertEqual(config.stage, stage)
                self.assertEqual(config.status, "planning_only")
                self.assertFalse(config.training.enabled)
        self.assertEqual(plans["pilot"].budget()["total_planned_requests"], 240)
        self.assertEqual(plans["baseline"].budget()["total_planned_requests"], 55440)
        stages = list(plans)
        for index, stage in enumerate(stages):
            for other in stages[index + 1:]:
                with self.subTest(stage=stage, other=other):
                    self.assertTrue(set(plans[stage].seeds).isdisjoint(plans[other].seeds))


if __name__ == "__main__":
    unittest.main()

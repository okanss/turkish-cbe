"""Frozen SQL cohorts and temporary data-only exports; no model training."""

from copy import deepcopy
import hashlib
import json
from pathlib import Path
import socket
import sqlite3
import tempfile
import unittest
from unittest.mock import patch

from inference.dataset import GridDataset, fingerprint
from inference.prompts import make_messages
from training.data import export_data, iter_training_units, lexical_overlap_audit, plan_data


ROOT = Path(__file__).resolve().parents[1]
MANIFEST = "data/complete_grid/manifest.json"


def config(full=False):
    return {"training_sizes": [4, 8, 16, 32, 64, 128] if full else [4, 8],
        "groups_per_size": 50 if full else 4,
        "training_relations": ["comparison_height", "comparison_speed", "comparison_weight"],
        "heldout_relations": ["spatial_front", "spatial_right", "spatial_above"],
        "query_types": ["endpoint", "internal", "distance_controlled"],
        "conditions": ["canonical", "equivalent", "P1", "P2"],
        "evaluation_sizes": [4, 8, 16, 32, 64, 128, 256, 512, 1024],
        "evaluation_groups_per_size": 200 if full else 4, "evaluation_split": "test",
        "grpo_min_n": 16 if full else 4, "grpo_max_instances": 600 if full else 8,
        "grpo_rollouts_per_view": 4}


class TrainingDataTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.pilot = plan_data(ROOT, MANIFEST, config(), seed=42)
        cls.full = plan_data(ROOT, MANIFEST, config(True), seed=42)
        cls.units = list(iter_training_units(ROOT, cls.pilot))

    def test_pilot_counts_and_frozen_training_split(self):
        counts = self.pilot["counts"]
        self.assertEqual((counts["training_latent_instances"], counts["training_queries"], counts["training_units"]), (8, 32, 96))
        self.assertEqual((counts["canonical_rows"], counts["paired_rows"], counts["all_view_rows"]), (96, 192, 384))
        self.assertEqual((counts["grpo_selected_units"], counts["grpo_views"], counts["grpo_rollout_samples"]), (8, 32, 128))
        self.assertEqual(self.pilot["training"]["split"], "training")
        self.assertFalse(self.pilot["contains_training_run"])
        self.assertFalse(self.pilot["contains_model_results"])

    def test_full_sql_counts_match_all_training_sizes_and_queries(self):
        counts = self.full["counts"]
        self.assertEqual((counts["training_latent_instances"], counts["training_queries"], counts["training_units"]), (300, 1900, 5700))
        self.assertEqual((counts["canonical_rows"], counts["paired_rows"]), (5700, 11400))
        self.assertEqual((counts["grpo_eligible_units"], counts["grpo_selected_units"], counts["grpo_rollout_samples"]), (4500, 600, 9600))
        self.assertEqual({group["n"] for group in self.full["training"]["groups"]}, {4, 8, 16, 32, 64, 128})

    def test_training_cannot_enter_existing_evaluation_selector(self):
        with GridDataset(ROOT, MANIFEST) as data:
            selection = deepcopy(next(iter(self.pilot["evaluation"].values()))["selection"])
            selection["split"] = "training"
            with self.assertRaisesRegex(ValueError, "training data"):
                data.select(selection)
            with self.assertRaises(sqlite3.OperationalError):
                data.connection.execute("CREATE TABLE forbidden_training_write(value INTEGER)")

    def test_train_test_latents_and_visible_prompt_hashes_are_disjoint(self):
        for plan in (self.pilot, self.full):
            training = {row["latent_instance_id"] for row in plan["training"]["groups"]}
            for key, cohort in plan["evaluation"].items():
                self.assertFalse(training & {row["latent_instance_id"] for row in cohort["groups"]})
                self.assertEqual(cohort["selection"]["split"], "test")
                self.assertEqual(plan["overlap_audit"]["split_checks"][key]["visible_prompt_hash_overlap"], 0)
            self.assertTrue(plan["overlap_audit"]["shared_entity_names_are_not_latent_aliases"])

    def test_transfer_cohorts_keep_size_and_vocabulary_axes_separate(self):
        self.assertEqual(set(self.full["evaluation"]), {"in_domain", "heldout_vocabulary", "larger_n", "larger_n_heldout_vocabulary"})
        for key, cohort in self.full["evaluation"].items():
            sizes = cohort["selection"]["sizes"]
            self.assertTrue(all(n > 128 for n in sizes) if key.startswith("larger_n") else all(n <= 128 for n in sizes))
            expected = config(True)["heldout_relations"] if "heldout_vocabulary" in key else config(True)["training_relations"]
            self.assertEqual(cohort["selection"]["relations"], expected)

    def test_grpo_units_are_balanced_deterministic_and_instance_bound(self):
        for plan, expected in ((self.pilot, 2), (self.full, 150)):
            grpo = plan["grpo"]
            self.assertEqual(set(grpo["answer_cells"].values()), {expected})
            self.assertEqual(len(set(grpo["selected_unit_ids"])), grpo["selected_count"])
            self.assertTrue(set(grpo["selected_unit_ids"]) <= set(plan["training_unit_ids"]))
            self.assertTrue(all(unit["n"] >= plan["config"]["grpo_min_n"] for unit in grpo["selected_units"]))
            self.assertLessEqual(max(grpo["n_counts"].values()) - min(grpo["n_counts"].values()), 4)

    def test_seed_changes_only_explicit_grpo_subselection(self):
        another = plan_data(ROOT, MANIFEST, config(), seed=43)
        self.assertEqual(another["training"], self.pilot["training"])
        self.assertEqual(another["training_unit_ids"], self.pilot["training_unit_ids"])
        self.assertNotEqual(another["grpo"]["selected_unit_ids"], self.pilot["grpo"]["selected_unit_ids"])
        self.assertEqual(another["grpo"]["answer_cells"], self.pilot["grpo"]["answer_cells"])
        self.assertEqual(plan_data(ROOT, MANIFEST, config(), seed=42), self.pilot)

    def test_every_materialized_unit_has_same_query_gold_and_four_valid_views(self):
        self.assertEqual(len(self.units), 96)
        self.assertEqual({unit["unit_id"] for unit in self.units}, set(self.pilot["training_unit_ids"]))
        for unit in self.units:
            self.assertEqual(set(unit["views"]), {"canonical", "equivalent", "P1", "P2"})
            for condition, row in unit["views"].items():
                self.assertEqual(row["condition"], condition)
                self.assertEqual(row["split"], "training")
                self.assertEqual(row["grid_query_id"], unit["query_id"])
                self.assertEqual(row["latent_instance_id"], unit["instance_id"])
                self.assertEqual(row["gold_label"], unit["gold"])
                self.assertEqual(row["query_direction"], unit["query_direction"])
                self.assertEqual(row["relation_id"], unit["relation_id"])
                self.assertEqual((row["language"], row["name_control"]), ("tr", "symbolic_shared"))
                self.assertLessEqual(row["n"], 128)

    def test_gold_metadata_cannot_leak_into_formatted_problem_messages(self):
        row = self.units[0]["views"]["canonical"]
        baseline = make_messages(row)
        changed = deepcopy(row)
        changed.update(gold_label=not row["gold_label"], latent_order=["SECRET_GOLD_ORDER"])
        self.assertEqual(make_messages(changed), baseline)
        self.assertNotIn("SECRET_GOLD_ORDER", json.dumps(baseline))

    def test_content_and_stopwords_are_separate_and_prefix_is_not_lemma_identity(self):
        audit = self.pilot["overlap_audit"]["lexical"]
        self.assertTrue(audit["clean_content_holdout"])
        self.assertEqual(audit["training_families"], ["comparison"])
        self.assertEqual(audit["heldout_families"], ["spatial"])
        for layer in audit["languages"].values():
            self.assertEqual(layer["exact_lemma_overlap"], [])
            self.assertEqual(layer["exact_phrase_overlap"], [])
            self.assertEqual(layer["content_token_overlap"], [])
            self.assertIn("stopword_inventory", layer)
        self.assertTrue(any({row["left"], row["right"]} == {"ön", "önce"} for row in audit["catalog_prefix_cautions"]))
        overlap = lexical_overlap_audit(["comparison_height"], ["comparison_height"])
        self.assertFalse(overlap["clean_content_holdout"])
        self.assertTrue(overlap["languages"]["tr"]["content_token_overlap"])
        self.assertNotIn("P4", audit["scope"].split(";")[0])

    def test_old_full_prompt_lengths_are_explicit_proxies_and_counts_match_views(self):
        proxy = self.pilot["source_lengths_proxy"]
        self.assertEqual(proxy["sft_canonical"]["prompts"], 96)
        self.assertEqual(proxy["paired_sft"]["prompts"], 192)
        self.assertEqual(proxy["training_all_views"]["prompts"], 384)
        self.assertEqual(proxy["grpo_selected_views"]["prompts"], 32)
        self.assertIn("unknown until encoded", proxy["scope"])

    def test_full_context_shared_content_is_not_claimed_clean_vocabulary(self):
        audit = self.pilot["overlap_audit"]["lexical"]
        self.assertTrue(audit["clean_content_holdout"])
        self.assertIn("predicate/query lexicon only", audit["clean_content_holdout_scope"])
        self.assertFalse(audit["clean_full_context_holdout"])
        self.assertTrue(audit["full_context_requires_human_review"])
        self.assertTrue({"higher", "lower"} <= set(audit["languages"]["en"]["context_audit"]["shared_content_tokens"]))
        self.assertNotIn("higher", audit["languages"]["en"]["stopword_inventory"])

    def test_types_size_bounds_splits_and_views_are_strict(self):
        cases = (("training_sizes", [4, 256]), ("training_sizes", [True]),
                 ("groups_per_size", True), ("groups_per_size", 4.0),
                 ("evaluation_split", "training"), ("conditions", ["canonical", "P4"]),
                 ("grpo_max_instances", 601), ("grpo_rollouts_per_view", True),
                 ("heldout_relations", ["comparison_height"]))
        for key, value in cases:
            value_config = config()
            value_config[key] = value
            with self.subTest(key=key), self.assertRaises(ValueError):
                plan_data(ROOT, MANIFEST, value_config)
        with self.assertRaises(ValueError):
            plan_data(ROOT, MANIFEST, config(), seed=True)

    def test_complete_config_sets_allow_reordering_without_losing_views(self):
        reordered = config()
        for field in ("training_relations", "heldout_relations", "query_types", "conditions"):
            reordered[field].reverse()
        plan = plan_data(ROOT, MANIFEST, reordered)
        self.assertEqual(plan["counts"], self.pilot["counts"])
        self.assertEqual(set(plan["training_unit_ids"]), set(self.pilot["training_unit_ids"]))
        self.assertEqual(plan["grpo"]["selected_unit_ids"], self.pilot["grpo"]["selected_unit_ids"])

    def test_tampered_and_manually_rehashed_plans_are_rejected(self):
        corrupted = deepcopy(self.pilot)
        corrupted["counts"]["canonical_rows"] += 1
        with self.assertRaisesRegex(ValueError, "hash"):
            next(iter_training_units(ROOT, corrupted))
        corrupted["data_sha256"] = fingerprint({key: value for key, value in corrupted.items() if key != "data_sha256"})
        with self.assertRaisesRegex(ValueError, "frozen selections"):
            next(iter_training_units(ROOT, corrupted))

    def test_immutable_jsonl_export_hashes_and_unit_membership(self):
        with tempfile.TemporaryDirectory() as folder:
            output = Path(folder).resolve() / "training-export"
            with patch.object(socket.socket, "connect", side_effect=AssertionError("no network")):
                result = export_data(ROOT, self.pilot, output)
            payload = (output / "training_units.jsonl").read_bytes()
            lines = payload.splitlines(keepends=True)
            self.assertEqual(len(lines), 96)
            self.assertEqual(result["files"][0]["sha256"], hashlib.sha256(payload).hexdigest())
            for line, group in zip(lines, result["groups"]):
                unit = json.loads(line)
                self.assertEqual(hashlib.sha256(line).hexdigest(), group["sha256"])
                self.assertEqual(unit["unit_id"], group["unit_id"])
                self.assertTrue(all(row["split"] == "training" for row in unit["views"].values()))
            with self.assertRaises(FileExistsError):
                export_data(ROOT, self.pilot, output)
            self.assertEqual((output / "training_units.jsonl").read_bytes(), payload)

    def test_export_cannot_overwrite_input_tree(self):
        with self.assertRaises(ValueError):
            export_data(ROOT, self.pilot, ROOT / "src/forbidden-training-export")


if __name__ == "__main__":
    unittest.main()

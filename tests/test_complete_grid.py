"""Full-factor coverage, exact tokenizer measurements and frozen integrity."""

from collections import Counter
from copy import deepcopy
import hashlib
import importlib.util
import json
from pathlib import Path
import shutil
import sqlite3
import tempfile
import unittest
from unittest.mock import patch

from data_generation.generator import canonical_bytes
from experiments.dataset_builder import (
    DatasetPlan, SplitSpec, build_dataset, _source_hashes as base_source_hashes,
)
from complete_grid.dataset_builder import (
    build_grid, export_grid, reconstruct_prompt, verify_grid,
)
from complete_grid.model_tokens import ModelTokenizers


PROJECT_ROOT = Path(__file__).resolve().parents[1]
RESOURCES = PROJECT_ROOT / "tokenizers"


@unittest.skipUnless(importlib.util.find_spec("tiktoken") is not None,
                     "optional pinned tiktoken dependency is not installed")
class CompleteGridTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temporary = tempfile.TemporaryDirectory()
        cls.root = Path(cls.temporary.name) / "project"
        cls.root.mkdir()
        cls.plan = DatasetPlan(tuple(
            SplitSpec(name, seed, (4,), 4)
            for name, seed in zip(
                ("training", "development", "pilot", "test"), range(401, 405)
            )
        ), presentation_seed=431, vocabulary_pilot=False)
        cls.base_manifest = cls.root / "experiment_dataset_manifest.json"
        build_dataset(cls.root / "data", cls.base_manifest, cls.plan)
        cls.base_bytes = cls.base_manifest.read_bytes()
        cls.output = cls.root / "data" / "complete_grid"
        cls.manifest = build_grid(cls.root, cls.output, cls.plan, RESOURCES)
        cls.manifest_path = cls.output / "manifest.json"
        cls.database = cls.output / "grid.sqlite"
        cls.tokenizers = ModelTokenizers(RESOURCES)
        cls.export = cls.root / "first-query.jsonl"
        cls.export_result = export_grid(
            cls.root, cls.manifest_path, cls.export,
            limit=243, resource_dir=RESOURCES,
        )
        cls.rows = [json.loads(line) for line in cls.export.read_bytes().splitlines()]

    @classmethod
    def tearDownClass(cls):
        cls.temporary.cleanup()

    def clone(self, name):
        target = self.root / name
        shutil.copytree(self.output, target)
        return target

    @staticmethod
    def update_file_hash(directory, name="grid.sqlite"):
        path = directory / "manifest.json"
        manifest = json.loads(path.read_bytes())
        item = next(entry for entry in manifest["files"] if entry["path"] == name)
        payload = (directory / name).read_bytes()
        item.update(bytes=len(payload), sha256=hashlib.sha256(payload).hexdigest())
        path.write_bytes(canonical_bytes(manifest) + b"\n")

    def test_every_query_has_all_243_unique_factor_combinations(self):
        with sqlite3.connect(self.database) as connection:
            expected = {
                "groups": 16, "queries": 48, "designs": 243,
                "components": 3888, "measurements": 11664,
                "comparison_cells": 15120,
            }
            for table, count in expected.items():
                with self.subTest(table=table):
                    self.assertEqual(connection.execute(
                        "SELECT count(*) FROM " + table).fetchone()[0], count)
            counts = connection.execute(
                "SELECT count(*) FROM measurements GROUP BY query_id"
            ).fetchall()
            self.assertEqual(counts, [(243,)] * 48)
            self.assertEqual(connection.execute("PRAGMA foreign_key_check").fetchall(), [])
            self.assertEqual(connection.execute(
                "SELECT count(*) FROM measurements m JOIN queries q ON q.id=m.query_id "
                "WHERE q.group_id != m.group_id").fetchone()[0], 0)

    def test_export_covers_all_relations_names_and_turkish_ablation_coordinates(self):
        self.assertEqual(self.export_result["records"], 243)
        self.assertEqual(self.export_result["sha256"],
                         hashlib.sha256(self.export.read_bytes()).hexdigest())
        self.assertEqual(len({r["query_instance_id"] for r in self.rows}), 1)
        self.assertEqual(len({r["latent_instance_id"] for r in self.rows}), 1)
        expected_relations = {
            "spatial_front", "spatial_right", "spatial_above",
            "temporal_arrival", "temporal_completion", "comparison_height",
            "comparison_speed", "comparison_weight", "ranking",
        }
        expected_conditions = {
            ("en", "canonical"), ("en", "equivalent"),
            ("tr", "canonical"), ("tr", "equivalent"),
            *(("tr", "P" + str(i)) for i in range(1, 6)),
        }
        self.assertEqual({r["relation_id"] for r in self.rows}, expected_relations)
        self.assertEqual({(r["language"], r["condition"]) for r in self.rows},
                         expected_conditions)
        self.assertEqual(Counter(r["name_control"] for r in self.rows), {
            "native_shared": 81, "en_transliterated": 81, "symbolic_shared": 81,
        })
        self.assertEqual(len({(r["name_control"], r["relation_id"], r["language"],
                               r["condition"]) for r in self.rows}), 243)
        self.assertTrue(all(r["human_review_status"] == "pending" for r in self.rows))
        self.assertTrue(all(not r["eligible_for_confirmatory_evaluation"] for r in self.rows))

    def test_each_exported_measurement_equals_full_text_bpe_encoding(self):
        with sqlite3.connect(self.database) as connection:
            measured_lengths = {
                row[0]: row[1:] for row in connection.execute(
                    "SELECT design_id,utf8_bytes,unicode_codepoints,whitespace_tokens,reference_tokens "
                    "FROM measurements WHERE query_id=?", (self.rows[0]["grid_query_id"],)
                )
            }
        for row in self.rows:
            with self.subTest(design=row["grid_design_id"]):
                self.assertEqual(row["model_token_counts"],
                                 self.tokenizers.measure(row["prompt"]))
                self.assertEqual(row["prompt_sha256"], hashlib.sha256(
                    row["prompt"].encode("utf-8")).hexdigest())
                self.assertEqual(row["lengths"]["model_token_count"],
                                 row["model_token_counts"]["o200k_base"])
                self.assertEqual(row["lengths"]["model_tokenizer_status"], "measured")
                self.assertEqual(measured_lengths[row["grid_design_id"]], tuple(
                    row["lengths"][key] for key in (
                        "utf8_bytes", "unicode_codepoints", "whitespace_token_count",
                        "reference_token_count",
                    )
                ))

    def test_previous_frozen_source_and_overlapping_prompts_remain_unchanged(self):
        self.assertEqual(self.base_manifest.read_bytes(), self.base_bytes)
        original = json.loads(self.base_bytes)
        self.assertEqual(original["source_files_sha256"], base_source_hashes())
        self.assertEqual(self.manifest["base_binding"]["source_files_sha256"],
                         original["source_files_sha256"])
        query = self.rows[0]["query_instance_id"]
        new = {(r["name_control"], r["relation_id"], r["language"], r["condition"]): r
               for r in self.rows}
        overlap = 0
        for line in (self.root / "data/training/prompts.jsonl").read_bytes().splitlines():
            row = json.loads(line)
            if row["query_instance_id"] == query:
                key = tuple(row[k] for k in ("name_control", "relation_id", "language", "condition"))
                self.assertEqual(new[key]["prompt"], row["prompt"])
                self.assertEqual(new[key]["gold_label"], row["gold_label"])
                overlap += 1
        self.assertEqual(overlap, 17)

    def test_internal_and_fixed_distance_queries_have_the_expanded_ablations(self):
        with sqlite3.connect(self.database) as connection:
            for query_type in ("internal", "distance_controlled"):
                for control in ("native_shared", "en_transliterated", "symbolic_shared"):
                    count = connection.execute(
                        "SELECT count(*) FROM measurements m JOIN queries q ON q.id=m.query_id "
                        "JOIN designs d ON d.id=m.design_id WHERE q.query_type=? "
                        "AND d.name_control=? AND d.language='tr' AND d.condition='P4'",
                        (query_type, control),
                    ).fetchone()[0]
                    self.assertEqual(count, 16 * 9)
            query_id, design_id = connection.execute(
                "SELECT m.query_id,m.design_id FROM measurements m JOIN queries q ON q.id=m.query_id "
                "JOIN designs d ON d.id=m.design_id WHERE q.query_type='distance_controlled' "
                "AND d.name_control='symbolic_shared' AND d.relation_id='ranking' "
                "AND d.condition='P4' LIMIT 1"
            ).fetchone()
        row = reconstruct_prompt(self.root, self.database, query_id, design_id, RESOURCES)
        self.assertEqual(row["query_distance"], 1)
        self.assertEqual(row["query_type"], "distance_controlled")
        self.assertTrue(row["is_multi_transform"])
        self.assertEqual(row["model_token_counts"], self.tokenizers.measure(row["prompt"]))

    def test_split_groups_and_four_query_answer_cells_are_disjoint_and_balanced(self):
        with sqlite3.connect(self.database) as connection:
            groups = connection.execute(
                "SELECT latent_instance_id,count(DISTINCT split) FROM groups GROUP BY latent_instance_id"
            ).fetchall()
            self.assertEqual(len(groups), 16)
            self.assertTrue(all(count == 1 for _, count in groups))
            cells = connection.execute(
                "SELECT g.split,q.query_type,q.query_direction,q.gold_label,count(*) "
                "FROM queries q JOIN groups g ON g.id=q.group_id GROUP BY 1,2,3,4"
            ).fetchall()
            self.assertEqual(len(cells), 4 * 3 * 4)
            self.assertTrue(all(count == 1 for *_, count in cells))

    def test_cross_group_measurements_are_rejected_by_sql_foreign_keys(self):
        directory = self.clone("wrong-group")
        with sqlite3.connect(directory / "grid.sqlite") as connection:
            connection.execute("PRAGMA foreign_keys=ON")
            with self.assertRaises(sqlite3.IntegrityError):
                connection.execute(
                    "UPDATE measurements SET group_id=2 WHERE query_id=1 AND design_id=1")

    def test_full_replay_verifies_sources_surfaces_and_actual_token_counts(self):
        result = verify_grid(self.root, self.manifest_path, resource_dir=RESOURCES)
        self.assertTrue(result["valid"])
        self.assertTrue(result["deep"])
        self.assertEqual((result["orders"], result["queries"], result["prompts"]),
                         (16, 48, 11664))

    def test_missing_measurement_is_rejected_after_outer_hash_is_repaired(self):
        directory = self.clone("missing-measurement")
        with sqlite3.connect(directory / "grid.sqlite") as connection:
            connection.execute("DELETE FROM measurements WHERE query_id=1 AND design_id=1")
        self.update_file_hash(directory)
        with self.assertRaises(ValueError):
            verify_grid(self.root, directory / "manifest.json", resource_dir=RESOURCES)

    def test_positive_forged_token_count_is_rejected_after_outer_hash_is_repaired(self):
        directory = self.clone("forged-token")
        with sqlite3.connect(directory / "grid.sqlite") as connection:
            connection.execute(
                "UPDATE measurements SET cl100k_base=cl100k_base+1 "
                "WHERE query_id=1 AND design_id=1")
        self.update_file_hash(directory)
        with self.assertRaisesRegex(ValueError, "BPE|token|replay"):
            verify_grid(self.root, directory / "manifest.json", resource_dir=RESOURCES)

    def test_comparison_view_cannot_swap_endpoint_and_interior_queries(self):
        directory = self.clone("reversed-comparison-view")
        with sqlite3.connect(directory / "grid.sqlite") as connection:
            original_sql = connection.execute(
                "SELECT sql FROM sqlite_master WHERE type='view' AND name='comparison_cells'"
            ).fetchone()[0]
            original_count = connection.execute(
                "SELECT count(*) FROM comparison_cells").fetchone()[0]
            expected_fragment = "SELECT r.id,r.arm,q.group_id,e.id,r.left_design_id,q.id,r.right_design_id"
            tampered_sql = original_sql.replace(
                expected_fragment,
                "SELECT r.id,r.arm,q.group_id,q.id,r.left_design_id,e.id,r.right_design_id",
            )
            self.assertNotEqual(original_sql, tampered_sql)
            connection.execute("DROP VIEW comparison_cells")
            connection.execute(tampered_sql)
            self.assertEqual(connection.execute(
                "SELECT count(*) FROM comparison_cells").fetchone()[0], original_count)
            # The cell count is unchanged, while F now gives the endpoint to
            # the right side. Metadata counts alone cannot catch this change.
            wrong = connection.execute(
                "SELECT count(*) FROM comparison_cells c "
                "JOIN queries q ON q.id=c.right_query_id "
                "WHERE c.arm='F' AND q.query_type='endpoint'"
            ).fetchone()[0]
            self.assertGreater(wrong, 0)
        self.update_file_hash(directory)
        with self.assertRaisesRegex(ValueError, "schema|view"):
            verify_grid(self.root, directory / "manifest.json", resource_dir=RESOURCES)

    def test_direct_reconstruction_rejects_gold_corruption(self):
        directory = self.clone("forged-gold")
        with sqlite3.connect(directory / "grid.sqlite") as connection:
            connection.execute("UPDATE queries SET gold_label=1-gold_label WHERE id=1")
        with self.assertRaises(ValueError):
            reconstruct_prompt(self.root, directory / "grid.sqlite", 1, 1, RESOURCES)

    def test_manifest_cannot_drop_file_inventory_or_claim_human_approval(self):
        for name, mutate in (
            ("omitted-inventory", lambda m: m.update(files=[])),
            ("claimed-human-approval", lambda m: m.update(human_review_status="approved")),
        ):
            with self.subTest(change=name):
                directory = self.clone(name)
                manifest = deepcopy(self.manifest)
                mutate(manifest)
                (directory / "manifest.json").write_bytes(canonical_bytes(manifest) + b"\n")
                with self.assertRaises(ValueError):
                    verify_grid(self.root, directory / "manifest.json", resource_dir=RESOURCES)

    def test_export_filters_and_exclusive_writes(self):
        target = self.root / "filtered.jsonl"
        result = export_grid(self.root, self.manifest_path, target, split="test", n=4,
                             relation_id="comparison_weight", limit=7, resource_dir=RESOURCES)
        rows = [json.loads(line) for line in target.read_bytes().splitlines()]
        self.assertEqual(result["records"], 7)
        self.assertEqual(len(rows), 7)
        self.assertTrue(all((r["split"], r["n"], r["relation_id"]) ==
                            ("test", 4, "comparison_weight") for r in rows))
        before = target.read_bytes()
        with self.assertRaises(FileExistsError):
            export_grid(self.root, self.manifest_path, target, resource_dir=RESOURCES)
        self.assertEqual(target.read_bytes(), before)
        with self.assertRaises(FileExistsError):
            build_grid(self.root, self.output, self.plan, RESOURCES)

    def test_export_keeps_a_file_created_concurrently_before_exclusive_open(self):
        target = self.root / "concurrent-export.jsonl"
        real_open = Path.open
        external_payload = b"created by a concurrent writer\n"

        def racing_open(path, mode="r", *args, **kwargs):
            if path == target and mode == "xb":
                # This happens after the exporter's existence check. The
                # exclusive open must fail without claiming ownership.
                with real_open(path, "wb") as handle:
                    handle.write(external_payload)
            return real_open(path, mode, *args, **kwargs)

        with patch.object(Path, "open", new=racing_open):
            with self.assertRaises(FileExistsError):
                export_grid(self.root, self.manifest_path, target, limit=1,
                            resource_dir=RESOURCES)
        self.assertEqual(target.read_bytes(), external_payload)


if __name__ == "__main__":
    unittest.main()

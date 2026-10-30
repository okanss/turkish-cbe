"""Dataset export, independent verification, replay, and CLI integration."""

from collections import Counter
from contextlib import redirect_stderr, redirect_stdout
from dataclasses import replace
import hashlib
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from data_generation.dataset import verify_dataset, write_dataset
from data_generation.generator import GenerationPlan, canonical_bytes, iter_block


class DatasetTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.plan = GenerationPlan(
            sizes=(4, 8), examples_per_size=12, distances=(1, 2, 4),
        )

    def export(self, name="a", plan=None):
        folder = self.root / name
        folder.mkdir()
        data_root = folder / "latent"
        manifest = folder / "dataset_manifest.json"
        result = write_dataset(data_root, manifest, plan or self.plan)
        return data_root, manifest, result

    @staticmethod
    def data_bytes(data_root):
        return {
            str(path.relative_to(data_root)): path.read_bytes()
            for path in sorted(data_root.rglob("*.jsonl"))
        }

    @staticmethod
    def rewrite_first_record(path, mutate):
        lines = path.read_bytes().splitlines()
        record = json.loads(lines[0])
        mutate(record)
        lines[0] = canonical_bytes(record)
        path.write_bytes(b"\n".join(lines) + b"\n")

    @staticmethod
    def update_file_hash(manifest_path, data_root, path):
        manifest = json.loads(manifest_path.read_bytes())
        relative = path.relative_to(data_root).as_posix()
        entry = next(entry for entry in manifest["files"] if entry["path"] == relative)
        payload = path.read_bytes()
        entry.update(sha256=hashlib.sha256(payload).hexdigest(), bytes=len(payload))
        manifest_path.write_bytes(canonical_bytes(manifest) + b"\n")

    def test_export_counts_balance_shared_orders_and_replay(self):
        data_root, manifest, result = self.export()
        files = sorted(data_root.rglob("*.jsonl"))
        self.assertEqual(len(files), 8)
        records = []
        orders = {}
        arm_counts = Counter()
        for path in files:
            block = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
            self.assertEqual(len(block), 12)
            self.assertEqual(Counter((r["query_direction"], r["gold_label"]) for r in block), {
                ("forward", True): 3, ("forward", False): 3,
                ("reverse", True): 3, ("reverse", False): 3,
            })
            self.assertEqual(sorted(Counter(r["entity_type"] for r in block).values()), [4, 4, 4])
            for record in block:
                arm_counts[record["query_type"]] += 1
                ordering_id = record["ordering_id"]
                structure = (record["latent_order"], record["premise_edges"], record["entity_type"])
                if ordering_id in orders:
                    self.assertEqual(orders[ordering_id], structure)
                orders[ordering_id] = structure
            records.extend(block)
        self.assertEqual(len(records), 96)
        self.assertEqual(len(orders), 24)
        self.assertEqual(arm_counts, {"endpoint": 24, "internal": 24, "distance_controlled": 48})
        self.assertEqual(Counter(r["gold_label"] for r in records), {True: 48, False: 48})
        self.assertIsInstance(result, dict)
        self.assertFalse(result["contains_model_results"])
        self.assertEqual(result["summary"]["records"], 96)
        self.assertEqual(result["summary"]["unique_base_orderings"], 24)
        verified = verify_dataset(manifest, replay=True)
        self.assertEqual(verified, result["summary"])

    def test_same_plan_produces_identical_bytes_in_different_locations(self):
        first_root, first_manifest, _ = self.export("a")
        second_root, second_manifest, _ = self.export("b")
        self.assertEqual(self.data_bytes(first_root), self.data_bytes(second_root))
        # Both manifests sit at the same depth relative to their data folder.
        self.assertEqual(first_manifest.read_bytes(), second_manifest.read_bytes())
        self.assertEqual(verify_dataset(first_manifest, replay=True), verify_dataset(second_manifest, replay=True))

    def test_changed_seed_changes_data(self):
        first_root, _, _ = self.export("a")
        second_root, second_manifest, _ = self.export("b", replace(self.plan, seed=12))
        self.assertNotEqual(self.data_bytes(first_root), self.data_bytes(second_root))
        verify_dataset(second_manifest, replay=True)

    def test_existing_output_is_never_overwritten(self):
        data_root, manifest, _ = self.export()
        original_data = self.data_bytes(data_root)
        original_manifest = manifest.read_bytes()
        with self.assertRaises((ValueError, FileExistsError)):
            write_dataset(data_root, manifest, self.plan)
        self.assertEqual(self.data_bytes(data_root), original_data)
        self.assertEqual(manifest.read_bytes(), original_manifest)

        other_root = self.root / "new_data"
        with self.assertRaises((ValueError, FileExistsError)):
            write_dataset(other_root, manifest, self.plan)
        self.assertFalse(other_root.exists())
        self.assertEqual(manifest.read_bytes(), original_manifest)

        occupied = self.root / "occupied"
        occupied.mkdir()
        sentinel = occupied / "keep.txt"
        sentinel.write_bytes(b"preserve this content")
        unused_manifest = self.root / "unused_manifest.json"
        with self.assertRaises((ValueError, FileExistsError)):
            write_dataset(occupied, unused_manifest, self.plan)
        self.assertEqual(sentinel.read_bytes(), b"preserve this content")
        self.assertFalse(unused_manifest.exists())

    def test_manifest_alias_inside_output_is_rejected_before_creation(self):
        output = self.root / "latent"
        # A lexical prefix check misses this path, which resolves into output.
        manifest = self.root / "alias" / ".." / "latent" / "manifest.json"
        with self.assertRaisesRegex(ValueError, "manifest.*outside"):
            write_dataset(output, manifest, self.plan)
        self.assertFalse(output.exists())
        self.assertFalse(manifest.resolve().exists())
        self.assertFalse((self.root / "alias").exists())

    def test_replay_rejects_modified_naming_and_distance_metadata(self):
        _, manifest_path, _ = self.export()
        original = manifest_path.read_bytes()
        changes = {
            "name_rules_version": lambda value: value.update(name_rules_version="unsupported"),
            "entity_pool_hash": lambda value: value["entity_pools"]["turkish_names"].update(
                sha256="0" * 64),
            "entity_pool_size": lambda value: value["entity_pools"]["turkish_names"].update(
                size=1),
            "valid_distances": lambda value: value["valid_distances_by_n"].update({"4": [999]}),
            "skipped_distances": lambda value: value["skipped_distances_by_n"].update({"4": []}),
        }
        for name, mutate in changes.items():
            with self.subTest(metadata=name):
                altered = json.loads(original)
                mutate(altered)
                manifest_path.write_bytes(canonical_bytes(altered) + b"\n")
                with self.assertRaisesRegex(ValueError, "design metadata"):
                    verify_dataset(manifest_path, replay=True)
        manifest_path.write_bytes(original)
        verify_dataset(manifest_path, replay=True)

    def test_failed_generation_rolls_back_its_partial_output(self):
        output = self.root / "incomplete"
        manifest = self.root / "dataset_manifest.json"

        def interrupted_block(*args, **kwargs):
            yield next(iter_block(*args, **kwargs))
            raise RuntimeError("deliberately interrupted export")

        with patch("data_generation.dataset.iter_block", side_effect=interrupted_block):
            with self.assertRaisesRegex(RuntimeError, "deliberately interrupted"):
                write_dataset(output, manifest, self.plan)
        self.assertFalse(output.exists())
        self.assertFalse(manifest.exists())

    def test_file_corruption_missing_and_extra_files_are_rejected(self):
        mutations = {
            "changed_gold": lambda root, path: self.rewrite_first_record(
                path, lambda record: record.update(gold_label=not record["gold_label"])),
            "truncated_json": lambda root, path: path.write_bytes(path.read_bytes()[:-9]),
            "missing_file": lambda root, path: path.unlink(),
            "extra_file": lambda root, path: (root / "unlisted.jsonl").write_bytes(b"{}\n"),
        }
        for name, mutate in mutations.items():
            with self.subTest(name=name):
                data_root, manifest, _ = self.export(name)
                path = next(data_root.rglob("*.jsonl"))
                mutate(data_root, path)
                with self.assertRaises((ValueError, OSError)):
                    verify_dataset(manifest)

    def test_invalid_manifest_is_rejected(self):
        for index, raw in enumerate((b"{", b"null", b"[]", b"{}")):
            with self.subTest(raw=raw):
                manifest = self.root / f"bad_{index}.json"
                manifest.write_bytes(raw)
                with self.assertRaises((ValueError, OSError)):
                    verify_dataset(manifest)

    def test_semantic_corruption_is_rejected_even_after_updating_file_hash(self):
        data_root, manifest, _ = self.export()
        path = next(data_root.rglob("*.jsonl"))
        self.rewrite_first_record(path, lambda record: record.update(gold_label=not record["gold_label"]))
        self.update_file_hash(manifest, data_root, path)
        with self.assertRaisesRegex(ValueError, "gold|oracle"):
            verify_dataset(manifest)

    def test_shared_order_mismatch_is_rejected_after_rehashing_a_valid_record(self):
        data_root, manifest, _ = self.export()
        path = data_root / "internal" / "n_0008.jsonl"

        def rename_entities(record):
            order = record["latent_order"]
            renamed = dict(zip(order, order[1:] + order[:1]))
            record["latent_order"] = [renamed[name] for name in order]
            record["premise_edges"] = [[renamed[a], renamed[b]] for a, b in record["premise_edges"]]
            record["query_subject"] = renamed[record["query_subject"]]
            record["query_object"] = renamed[record["query_object"]]

        # Logical truth, distance, entity pool, cell counts, and identifiers stay
        # valid individually, but this arm changes a shared base ordering.
        self.rewrite_first_record(path, rename_entities)
        self.update_file_hash(manifest, data_root, path)
        with self.assertRaisesRegex(ValueError, "shared latent order|cross-arm"):
            verify_dataset(manifest)

    def test_replay_detects_valid_reordered_bytes_after_hash_update(self):
        data_root, manifest, _ = self.export()
        path = next(data_root.rglob("*.jsonl"))
        lines = path.read_bytes().splitlines()
        path.write_bytes(b"\n".join(reversed(lines)) + b"\n")
        self.update_file_hash(manifest, data_root, path)
        # Semantic verification deliberately permits record permutation; replay
        # additionally enforces the versioned generator's exact byte sequence.
        verify_dataset(manifest)
        with self.assertRaisesRegex(ValueError, "replay"):
            verify_dataset(manifest, replay=True)

    def test_cli_generate_and_verify_report_json_summary(self):
        from data_generation.__main__ import main

        output = self.root / "cli_data"
        manifest = self.root / "cli_manifest.json"
        stdout, stderr = io.StringIO(), io.StringIO()
        with redirect_stdout(stdout), redirect_stderr(stderr):
            code = main([
                "generate", "--output", str(output), "--manifest", str(manifest),
                "--seed", "11", "--sizes", "4", "8", "--examples-per-size", "12",
                "--entity-types", *self.plan.entity_types,
                "--query-types", *self.plan.query_types,
                "--distances", "1", "2", "4",
            ])
        self.assertEqual(code, 0, stderr.getvalue())
        generated = json.loads(stdout.getvalue())
        self.assertFalse(generated["contains_model_results"])
        self.assertEqual(generated["summary"]["records"], 96)
        self.assertEqual(len(list(output.rglob("*.jsonl"))), 8)
        stdout, stderr = io.StringIO(), io.StringIO()
        with redirect_stdout(stdout), redirect_stderr(stderr):
            code = main(["verify", str(manifest), "--replay"])
        self.assertEqual(code, 0, stderr.getvalue())
        verified = json.loads(stdout.getvalue())
        self.assertTrue(verified["replayed"])
        self.assertEqual(verified["summary"], generated["summary"])


if __name__ == "__main__":
    unittest.main()

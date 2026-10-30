"""Matched bilingual datasets, split isolation, and frozen artifact integrity."""

from collections import Counter, defaultdict
from copy import deepcopy
from dataclasses import replace
import hashlib
import json
from pathlib import Path
import shutil
import tempfile
import unittest
from unittest.mock import patch

from data_generation.generator import canonical_bytes
from experiments import dataset_builder as builder
from experiments.dataset_builder import (
    DatasetPlan, SplitSpec, balanced_indices, build_dataset, load_plan,
    validate_pair, verify_dataset,
)
from experiments.token_lengths import measure_text


class SourceFingerprintTests(unittest.TestCase):
    def test_installed_fingerprints_include_only_owned_packages_and_reject_symlinks(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / "site-packages"
            expected = {}
            for name in ("turkish_cbe", "data_generation", "renderers", "validation", "experiments"):
                package = root / name
                package.mkdir(parents=True)
                payload = ("# owned package " + name + "\n").encode()
                (package / "__init__.py").write_bytes(payload)
                expected[name + "/__init__.py"] = hashlib.sha256(payload).hexdigest()
            module = root / "experiments/dataset_builder.py"
            module.write_bytes(b"# owned builder\n")
            expected["experiments/dataset_builder.py"] = hashlib.sha256(module.read_bytes()).hexdigest()
            unrelated = root / "pip/unrelated.py"
            unrelated.parent.mkdir()
            unrelated.write_bytes(b"# unrelated dependency\n")
            (root / "unrelated.py").write_bytes(b"# unrelated top-level module\n")
            with patch.object(builder, "__file__", str(module)):
                self.assertEqual(builder._source_hashes(), expected)
                alias = root / "renderers/alias.py"
                alias.symlink_to(unrelated)
                with self.assertRaisesRegex(ValueError, "symlink"):
                    builder._source_hashes()
                alias.unlink()
                saved = root / "external_validation"
                (root / "validation").rename(saved)
                with self.assertRaisesRegex(ValueError, "missing"):
                    builder._source_hashes()
                (root / "validation").symlink_to(saved, target_is_directory=True)
                with self.assertRaisesRegex(ValueError, "symlink"):
                    builder._source_hashes()


class BalancedIndexTests(unittest.TestCase):
    def test_fifty_has_near_equal_cells_and_exact_gold_margins(self):
        for n in (4, 8, 16, 32, 64, 128):
            with self.subTest(n=n):
                indices = balanced_indices(50, n)
                self.assertIsInstance(indices, tuple)
                self.assertEqual(len(indices), 50)
                self.assertEqual(len(set(indices)), 50)
                self.assertTrue(all(type(index) is int and index >= 0 for index in indices))
                counts = Counter(index % 4 for index in indices)
                self.assertEqual(sorted(counts.values()), [12, 12, 13, 13])
                self.assertEqual(sum(count for cell, count in counts.items() if cell % 2 == 0), 25)
                self.assertEqual(indices, balanced_indices(50, n))

    def test_two_hundred_has_exact_fifty_per_cell(self):
        for n in (4, 8, 16, 32, 64, 128, 256, 512, 1024):
            indices = balanced_indices(200, n)
            self.assertEqual(len(set(indices)), 200)
            self.assertEqual(Counter(index % 4 for index in indices), dict.fromkeys(range(4), 50))


class DatasetBuilderTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temporary = tempfile.TemporaryDirectory()
        cls.folder = Path(cls.temporary.name)
        cls.plan = DatasetPlan(
            splits=tuple(
                SplitSpec(name=name, seed=seed, sizes=(4,), count=4)
                for name, seed in zip(
                    ("pilot", "development", "test", "training"), range(401, 405)
                )
            ),
            presentation_seed=431,
            vocabulary_pilot=False,
        )
        cls.output = cls.folder / "original" / "data"
        cls.manifest_path = cls.folder / "original" / "experiment_dataset_manifest.json"
        cls.manifest = build_dataset(cls.output, cls.manifest_path, cls.plan)

    @classmethod
    def tearDownClass(cls):
        cls.temporary.cleanup()

    @staticmethod
    def rows(path):
        return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]

    def prompts(self, split="pilot"):
        return self.rows(self.output / split / "prompts.jsonl")

    def copy_export(self, name):
        parent = self.folder / name
        shutil.copytree(self.output.parent, parent)
        return parent / "data", parent / self.manifest_path.name

    def update_hash(self, manifest_path, output, path):
        manifest = json.loads(manifest_path.read_bytes())
        relative = path.relative_to(manifest_path.parent).as_posix()
        entry = next(item for item in manifest["files"] if item["path"] == relative)
        payload = path.read_bytes()
        entry["sha256"] = hashlib.sha256(payload).hexdigest()
        if "bytes" in entry:
            entry["bytes"] = len(payload)
        if path.suffix == ".jsonl":
            entry["records"] = len(payload.splitlines())
        # Repair every nested file checksum too, so semantic corruption cannot
        # pass this test merely because an outer metadata checksum catches it.
        split = path.relative_to(output).parts[0]
        summary = manifest["splits"][split]
        for nested in summary["files"]:
            if nested["path"] == relative:
                nested.update(entry)
        split_manifest = output / split / "split_manifest.json"
        split_manifest.write_bytes(canonical_bytes(summary) + b"\n")
        split_entry = next(item for item in manifest["files"]
                           if item["path"] == split_manifest.relative_to(manifest_path.parent).as_posix())
        split_entry.update(bytes=split_manifest.stat().st_size,
                           sha256=hashlib.sha256(split_manifest.read_bytes()).hexdigest())
        manifest["bundle_files_sha256"] = hashlib.sha256(canonical_bytes(manifest["files"])).hexdigest()
        manifest_path.write_bytes(canonical_bytes(manifest) + b"\n")

    def test_minimal_matrix_counts_and_disjoint_split_groups(self):
        all_groups = set()
        for spec in self.plan.splits:
            with self.subTest(split=spec.name):
                groups = self.rows(self.output / spec.name / "groups.jsonl")
                latent = self.rows(self.output / spec.name / "latent.jsonl")
                prompts = self.prompts(spec.name)
                cells = self.rows(self.output / spec.name / "cells.jsonl")
                self.assertEqual(len(groups), 4)
                self.assertEqual(len(latent), 12)
                self.assertEqual(len(prompts), 100)
                self.assertEqual(len(cells), 100)
                ids = {row["latent_instance_id"] for row in prompts}
                self.assertEqual(len(ids), 4)
                self.assertFalse(ids & all_groups)
                all_groups.update(ids)
                queries = {}
                for row in prompts:
                    self.assertEqual(row["split"], spec.name)
                    self.assertEqual(row["n"], 4)
                    queries.setdefault(row["query_instance_id"], row)
                self.assertEqual(len(queries), 12)
                self.assertEqual(Counter(row["query_type"] for row in queries.values()), {
                    "endpoint": 4, "internal": 4, "distance_controlled": 4,
                })
                self.assertEqual(Counter(row["gold_label"] for row in queries.values()), {
                    True: 6, False: 6,
                })
                self.assertTrue(all(row["query_distance"] == 1 for row in queries.values()
                                    if row["query_type"] == "distance_controlled"))

    def test_baseline_and_ablation_conditions_remain_separate(self):
        rows = self.prompts()
        self.assertEqual(Counter(row["condition"] for row in rows), {
            "canonical": 40, "equivalent": 40,
            "P1": 4, "P2": 4, "P3": 4, "P4": 4, "P5": 4,
        })
        for row in rows:
            if row["condition"].startswith("P"):
                self.assertEqual(row["language"], "tr")
                self.assertEqual(row["query_type"], "endpoint")
                self.assertEqual(row["name_control"], "native_shared")
            if row["query_type"] != "endpoint":
                self.assertEqual(row["name_control"], "native_shared")
        endpoint = [row for row in rows if row["query_type"] == "endpoint"
                    and row["condition"] in ("canonical", "equivalent")]
        self.assertEqual(Counter(row["name_control"] for row in endpoint), {
            "native_shared": 16, "en_transliterated": 16, "symbolic_shared": 16,
        })

    def test_each_prompt_hash_and_length_measure_exact_visible_text(self):
        for row in self.prompts():
            self.assertEqual(row["prompt_sha256"], hashlib.sha256(row["prompt"].encode("utf-8")).hexdigest())
            self.assertEqual(row["lengths"], measure_text(row["prompt"]))
            self.assertIsNone(row["lengths"]["model_token_count"])
            self.assertEqual(row["lengths"]["model_tokenizer_status"], "not_configured")
            self.assertIs(type(row["gold_label"]), bool)
            self.assertNotIn("model_response", row)
        turkish = "\n".join(row["prompt"] for row in self.prompts() if row["language"] == "tr")
        self.assertTrue(any(character in turkish for character in "çğıöşüÇĞİÖŞÜ"))

    def test_canonical_cross_language_pairs_preserve_target_and_edit_plan(self):
        grouped = defaultdict(dict)
        for row in self.prompts():
            if row["condition"] == "canonical":
                key = (row["query_instance_id"], row["name_control"], row["relation_id"])
                grouped[key][row["language"]] = row
        self.assertEqual(len(grouped), 20)
        for pair in grouped.values():
            self.assertEqual(set(pair), {"en", "tr"})
            english, turkish = pair["en"], pair["tr"]
            self.assertIsNone(validate_pair(english, turkish, "C"))
            for field in ("latent_instance_id", "n", "query_type", "query_distance",
                          "query_subject", "query_object", "query_relation", "query_direction",
                          "gold_label", "presentation_plan_sha256"):
                self.assertEqual(english[field], turkish[field], field)

    def test_pair_validator_accepts_all_defined_equivalence_arms(self):
        rows = self.prompts()
        canonical = next(row for row in rows if row["language"] == "en"
                         and row["condition"] == "canonical"
                         and row["name_control"] == "native_shared"
                         and row["query_type"] == "endpoint")
        corresponding = {
            (row["language"], row["condition"]): row for row in rows
            if row["query_instance_id"] == canonical["query_instance_id"]
            and row["name_control"] == "native_shared"
            and row["relation_id"] == canonical["relation_id"]
        }
        for arm, left_key, right_key in (
            ("A", ("en", "canonical"), ("en", "equivalent")),
            ("B", ("tr", "canonical"), ("tr", "equivalent")),
            ("C", ("en", "canonical"), ("tr", "canonical")),
            ("D", ("en", "equivalent"), ("tr", "equivalent")),
            ("E", ("tr", "canonical"), ("tr", "P3")),
        ):
            with self.subTest(arm=arm):
                self.assertIsNone(validate_pair(corresponding[left_key], corresponding[right_key], arm))

    def test_pair_validator_rejects_mismatched_problem_and_condition(self):
        rows = self.prompts()
        left = next(row for row in rows if row["language"] == "en" and row["condition"] == "canonical")
        right = next(row for row in rows if row["language"] == "tr"
                     and row["query_instance_id"] == left["query_instance_id"]
                     and row["condition"] == left["condition"]
                     and row["name_control"] == left["name_control"]
                     and row["relation_id"] == left["relation_id"])
        for field, value in {
            "latent_instance_id": "ord-" + "0" * 64,
            "query_instance_id": "inst-" + "0" * 64,
            "query_subject": "different target",
            "query_direction": "reverse" if left["query_direction"] == "forward" else "forward",
            "gold_label": not left["gold_label"],
            "presentation_plan_sha256": "0" * 64,
            "condition": "equivalent",
        }.items():
            with self.subTest(field=field):
                altered = deepcopy(right)
                altered[field] = value
                with self.assertRaises(ValueError):
                    validate_pair(left, altered, "C")

    def test_manifest_lists_every_artifact_with_counts_and_hashes(self):
        expected = {path.relative_to(self.manifest_path.parent).as_posix()
                    for path in self.output.rglob("*") if path.is_file()}
        actual = {entry["path"] for entry in self.manifest["files"]}
        self.assertEqual(actual, expected)
        for entry in self.manifest["files"]:
            path = self.manifest_path.parent / entry["path"]
            self.assertEqual(entry["sha256"], hashlib.sha256(path.read_bytes()).hexdigest())
            if path.suffix == ".jsonl":
                self.assertEqual(entry["records"], len(self.rows(path)))
            else:
                self.assertIsNone(entry["records"])
        self.assertIsInstance(verify_dataset(self.manifest_path, deep=True), dict)

    def test_identical_plan_reproduces_artifacts_and_manifest(self):
        other_parent = self.folder / "replayed"
        other_output = other_parent / "data"
        other_manifest = other_parent / self.manifest_path.name
        build_dataset(other_output, other_manifest, self.plan)
        for path in self.output.rglob("*.jsonl"):
            self.assertEqual(path.read_bytes(), (other_output / path.relative_to(self.output)).read_bytes())
        self.assertEqual(self.manifest_path.read_bytes(), other_manifest.read_bytes())

    def test_changed_seed_changes_only_its_split(self):
        changed = replace(self.plan, splits=(replace(self.plan.splits[0], seed=405), *self.plan.splits[1:]))
        other_output = self.folder / "changed" / "data"
        build_dataset(other_output, self.folder / "changed" / self.manifest_path.name, changed)
        self.assertNotEqual((self.output / "pilot" / "prompts.jsonl").read_bytes(),
                            (other_output / "pilot" / "prompts.jsonl").read_bytes())
        for name in ("development", "test", "training"):
            for file_name in ("groups.jsonl", "latent.jsonl", "prompts.jsonl", "cells.jsonl"):
                self.assertEqual((self.output / name / file_name).read_bytes(),
                                 (other_output / name / file_name).read_bytes())

    def test_existing_export_is_preserved(self):
        original_manifest = self.manifest_path.read_bytes()
        original_prompt = (self.output / "test" / "prompts.jsonl").read_bytes()
        with self.assertRaises((ValueError, FileExistsError)):
            build_dataset(self.output, self.manifest_path, self.plan)
        self.assertEqual(self.manifest_path.read_bytes(), original_manifest)
        self.assertEqual((self.output / "test" / "prompts.jsonl").read_bytes(), original_prompt)

    def test_invalid_output_containment_has_no_filesystem_side_effects(self):
        output = self.folder / "outside_dataset"
        manifest = self.folder / "separate_parent" / "manifest.json"
        with self.assertRaises(ValueError):
            build_dataset(output, manifest, self.plan)
        self.assertFalse(output.exists())
        self.assertFalse(manifest.parent.exists())

    def test_failed_final_manifest_write_removes_owned_partial_artifacts(self):
        parent = self.folder / "write_failure"
        output = parent / "data"
        output.mkdir(parents=True)
        sentinel = output / "keep.txt"
        sentinel.write_bytes(b"preexisting research data")
        manifest = parent / "manifest.json"
        original_open = Path.open

        class InterruptedManifest:
            def __init__(self, handle):
                self.handle = handle

            def __enter__(self):
                return self

            def write(self, payload):
                self.handle.write(payload[:17])
                raise OSError("deliberately interrupted manifest write")

            def __exit__(self, *args):
                self.handle.close()

        def interrupt_final_open(path, *args, **kwargs):
            mode = args[0] if args else kwargs.get("mode", "r")
            handle = original_open(path, *args, **kwargs)
            if path.resolve() == manifest.resolve() and mode == "xb":
                return InterruptedManifest(handle)
            return handle

        with patch.object(Path, "open", new=interrupt_final_open):
            with self.assertRaisesRegex(OSError, "interrupted manifest"):
                build_dataset(output, manifest, self.plan)
        self.assertFalse(manifest.exists())
        self.assertTrue(all(not (output / spec.name).exists() for spec in self.plan.splits))
        self.assertEqual(sentinel.read_bytes(), b"preexisting research data")

    def test_concurrent_manifest_is_preserved_when_exclusive_open_fails(self):
        parent = self.folder / "manifest_race"
        output = parent / "data"
        manifest = parent / "manifest.json"
        original_open = Path.open
        concurrent_bytes = b"another process owns this manifest\n"

        def race_final_open(path, *args, **kwargs):
            mode = args[0] if args else kwargs.get("mode", "r")
            if path.resolve() == manifest.resolve() and mode == "xb":
                with original_open(path, "wb") as handle:
                    handle.write(concurrent_bytes)
            return original_open(path, *args, **kwargs)

        with patch.object(Path, "open", new=race_final_open):
            with self.assertRaises(FileExistsError):
                build_dataset(output, manifest, self.plan)
        self.assertEqual(manifest.read_bytes(), concurrent_bytes)
        self.assertTrue(all(not (output / spec.name).exists() for spec in self.plan.splits))

    def test_repeated_seed_cannot_put_the_same_orders_in_training_and_test(self):
        leaked = replace(self.plan, splits=(
            self.plan.splits[0], self.plan.splits[1], self.plan.splits[2],
            replace(self.plan.splits[3], seed=self.plan.splits[2].seed),
        ))
        with self.assertRaises(ValueError):
            build_dataset(self.folder / "leaked" / "data",
                          self.folder / "leaked" / self.manifest_path.name, leaked)

    def test_file_tampering_is_rejected(self):
        output, manifest = self.copy_export("tampered")
        path = output / "test" / "prompts.jsonl"
        path.write_bytes(path.read_bytes() + b"{}\n")
        with self.assertRaises((ValueError, OSError)):
            verify_dataset(manifest, deep=True)

    def test_semantic_tampering_is_rejected_after_rehashing(self):
        output, manifest = self.copy_export("rehashed")
        path = output / "test" / "prompts.jsonl"
        rows = self.rows(path)
        rows[0]["gold_label"] = not rows[0]["gold_label"]
        path.write_bytes(b"".join(canonical_bytes(row) + b"\n" for row in rows))
        self.update_hash(manifest, output, path)
        with self.assertRaisesRegex(ValueError, "prompt schema.*oracle"):
            verify_dataset(manifest, deep=True)

    def test_missing_comparison_is_rejected_after_rehashing(self):
        output, manifest = self.copy_export("missing_cell")
        path = output / "test" / "cells.jsonl"
        lines = path.read_bytes().splitlines(keepends=True)
        self.assertEqual(len(lines), 100)
        path.write_bytes(b"".join(lines[1:]))
        self.update_hash(manifest, output, path)
        with self.assertRaises(ValueError):
            verify_dataset(manifest, deep=True)

    def test_group_metadata_aliases_and_plans_are_verified_after_rehashing(self):
        def change_sample(group):
            group["sample_index"] += 4

        def change_aliases(group):
            aliases = group["name_maps"]["symbolic_shared"]["en"]
            first, second = list(aliases)[:2]
            aliases[first], aliases[second] = aliases[second], aliases[first]

        def change_plan(group):
            plan = group["presentation_plans"]["canonical"]
            plan["entity_order"] = plan["entity_order"][1:] + plan["entity_order"][:1]

        for name, mutate in (("sample", change_sample), ("alias", change_aliases),
                             ("plan", change_plan)):
            with self.subTest(mutation=name):
                output, manifest = self.copy_export("group_" + name)
                path = output / "test" / "groups.jsonl"
                rows = self.rows(path)
                mutate(rows[0])
                path.write_bytes(b"".join(canonical_bytes(row) + b"\n" for row in rows))
                self.update_hash(manifest, output, path)
                with self.assertRaises(ValueError):
                    verify_dataset(manifest, deep=True)

    def test_load_plan_round_trip_from_explicit_toml(self):
        config = self.folder / "mini_plan.toml"
        lines = ["schema_version = 1", "presentation_seed = 431", "vocabulary_pilot = false"]
        for split in self.plan.splits:
            lines.extend(("", "[[splits]]", f'name = "{split.name}"',
                          f"seed = {split.seed}", "sizes = [4]", "count = 4"))
        config.write_text("\n".join(lines) + "\n", encoding="utf-8")
        self.assertEqual(load_plan(config), self.plan)


if __name__ == "__main__":
    unittest.main()

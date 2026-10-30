"""Planning provenance remains stable, bounded and distinct from model results."""

import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from turkish_cbe.provenance import (
    build_manifest, canonical_json, sha256_bytes, source_hashes, write_manifest,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]


class ProvenanceTests(unittest.TestCase):
    def setUp(self):
        self.folder = tempfile.TemporaryDirectory()
        self.addCleanup(self.folder.cleanup)
        self.root = Path(self.folder.name) / "project"
        (self.root / "configs").mkdir(parents=True)
        (self.root / "src" / "turkish_cbe").mkdir(parents=True)
        self.config = self.root / "configs" / "pilot.toml"
        self.config.write_bytes((PROJECT_ROOT / "configs" / "pilot.toml").read_bytes())
        self.source = self.root / "src" / "turkish_cbe" / "example.py"
        self.source.write_text("VALUE = 1\n", encoding="utf-8")
        (self.root / "pyproject.toml").write_text("[project]\nname = 'fixture'\n", encoding="utf-8")
        # Host package inventory and Git subprocesses are unrelated to these invariants.
        self.addCleanup(patch.stopall)
        patch("turkish_cbe.provenance.importlib.metadata.distributions", return_value=[]).start()
        patch("turkish_cbe.provenance._git_state", return_value={
            "commit": None, "dirty": None, "status": "unversioned_project",
        }).start()

    def test_canonical_json_normalizes_key_order_and_retains_unicode(self):
        first = {"z": [3, {"b": 2, "a": "Türkçe"}], "a": 1}
        second = {"a": 1, "z": [3, {"a": "Türkçe", "b": 2}]}
        self.assertEqual(canonical_json(first), canonical_json(second))
        self.assertIn("Türkçe", canonical_json(first))
        self.assertEqual(
            sha256_bytes(canonical_json(first).encode("utf-8")),
            sha256_bytes(canonical_json(second).encode("utf-8")),
        )

    def test_same_config_has_stable_raw_resolved_and_source_fingerprints(self):
        first = build_manifest(self.config, self.root)
        second = build_manifest(self.config, self.root)
        for field in ("config_sha256", "resolved_config_sha256", "source_tree_sha256"):
            with self.subTest(field=field):
                self.assertEqual(first[field], second[field])
        self.assertEqual(
            first["source_files_sha256"]["configs/pilot.toml"], first["config_sha256"],
        )
        self.assertEqual(first["resolved_config_sha256"], sha256_bytes(
            canonical_json(first["resolved_config"]).encode("utf-8"),
        ))

    def test_comment_changes_raw_hash_but_not_resolved_config_hash(self):
        before = build_manifest(self.config, self.root)
        with self.config.open("a", encoding="utf-8") as handle:
            handle.write("\n# Harmless planning comment.\n")
        after = build_manifest(self.config, self.root)
        self.assertNotEqual(before["config_sha256"], after["config_sha256"])
        self.assertEqual(before["resolved_config_sha256"], after["resolved_config_sha256"])
        self.assertEqual(before["resolved_config"], after["resolved_config"])

    def test_source_mutation_changes_file_and_tree_fingerprints(self):
        before = build_manifest(self.config, self.root)
        self.source.write_text("VALUE = 2\n", encoding="utf-8")
        after = build_manifest(self.config, self.root)
        self.assertNotEqual(before["source_tree_sha256"], after["source_tree_sha256"])
        self.assertNotEqual(
            before["source_files_sha256"]["src/turkish_cbe/example.py"],
            after["source_files_sha256"]["src/turkish_cbe/example.py"],
        )
        self.assertEqual(before["resolved_config_sha256"], after["resolved_config_sha256"])

    def test_secrets_environments_and_generated_files_are_excluded(self):
        before = source_hashes(self.root)
        excluded = (
            ".env", ".env.local", "secrets.txt", "configs/api_key.env",
            "data/generated/instances.jsonl", "results/responses.jsonl",
            "outputs/manifest.json", "work/scratch.py",
            ".venv/lib/site-packages/dependency.py",
        )
        for relative in excluded:
            path = self.root / relative
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text("irrelevant fixture content\n", encoding="utf-8")
        self.assertEqual(source_hashes(self.root), before)

    def test_source_directory_symlinks_to_external_trees_are_rejected(self):
        for directory in ("src", "configs"):
            with self.subTest(directory=directory), tempfile.TemporaryDirectory() as folder:
                root = Path(folder) / "root"
                external = Path(folder) / "external"
                root.mkdir()
                external.mkdir()
                (external / "outside.py").write_text("external = True\n", encoding="utf-8")
                (root / directory).symlink_to(external, target_is_directory=True)
                with self.assertRaises(ValueError):
                    source_hashes(root)

    def test_source_and_config_file_symlinks_to_external_files_are_rejected(self):
        for directory, filename in (("src", "external.py"), ("configs", "external.toml")):
            with self.subTest(directory=directory), tempfile.TemporaryDirectory() as folder:
                external = Path(folder) / filename
                external.write_text("external fixture\n", encoding="utf-8")
                link = self.root / directory / filename
                link.symlink_to(external)
                try:
                    with self.assertRaises(ValueError):
                        source_hashes(self.root)
                finally:
                    link.unlink()

    def test_empty_source_tree_is_not_given_a_misleading_fingerprint(self):
        with tempfile.TemporaryDirectory() as folder:
            with self.assertRaises(ValueError):
                source_hashes(Path(folder))

    def test_manifest_explicitly_contains_only_a_nonexecutable_plan(self):
        manifest = build_manifest(self.config, self.root)
        self.assertEqual(manifest["artifact_type"], "planning_manifest")
        self.assertEqual(manifest["status"], "planning_only")
        self.assertIs(manifest["contains_model_results"], False)
        self.assertIs(manifest["execution_ready"], False)
        self.assertIsNone(manifest["dataset"])
        self.assertIsNone(manifest["template_bundle"])
        self.assertEqual(manifest["budget"]["total_planned_requests"], 240)
        self.assertEqual(manifest["pending_gates"], list(manifest["resolved_config"]["pending_gates"]))
        self.assertNotIn("results", manifest)
        self.assertNotIn("accuracy", manifest)

    def test_write_manifest_roundtrips_and_refuses_to_overwrite_existing_file(self):
        output = self.root / "outputs" / "plan.json"
        manifest = {"status": "planning_only", "note": "Türkçe"}
        write_manifest(manifest, output)
        original = output.read_bytes()
        self.assertEqual(json.loads(original), manifest)
        with self.assertRaises(FileExistsError):
            write_manifest({"status": "changed"}, output)
        self.assertEqual(output.read_bytes(), original)


if __name__ == "__main__":
    unittest.main()

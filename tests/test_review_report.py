"""The human-review CSV must be traceable, balanced and truthful about status."""

from collections import Counter, defaultdict
from contextlib import redirect_stderr, redirect_stdout
import csv
import hashlib
import io
import json
from pathlib import Path
import tempfile
import unittest

from data_generation.dataset import write_dataset
from data_generation.generator import GenerationPlan
from data_generation.oracle import answer_from_edges
from renderers import RENDERER_VERSION
from renderers.review import generate_review_report, main
from renderers.turkish import RELATIONS
from validation.semantic_equivalence import normalize_query, normalize_statement


class ReviewReportTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.fixture = tempfile.TemporaryDirectory()
        cls.addClassCleanup(cls.fixture.cleanup)
        cls.source_root = Path(cls.fixture.name)
        cls.source_manifest = cls.source_root / "dataset_manifest.json"
        cls.source_metadata = write_dataset(
            cls.source_root / "latent", cls.source_manifest,
            GenerationPlan(sizes=(4, 8), examples_per_size=48, distances=(1,)),
        )
        cls.sources = {}
        for entry in cls.source_metadata["files"]:
            source = cls.source_root / "latent" / entry["path"]
            for line in source.read_text(encoding="utf-8").splitlines():
                record = json.loads(line)
                cls.sources[record["instance_id"]] = record

    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)

    def report(self, name="a", *, include_negation=False, seed=31):
        folder = self.root / name
        folder.mkdir()
        output = folder / "turkish_samples.csv"
        manifest = folder / "turkish_samples_manifest.json"
        metadata = generate_review_report(
            self.source_manifest, output, manifest,
            seed=seed, include_negation=include_negation,
        )
        with output.open(encoding="utf-8", newline="") as handle:
            rows = list(csv.DictReader(handle))
        return output, manifest, metadata, rows

    def test_review_material_has_all_required_audit_columns(self):
        _, _, _, rows = self.report()
        required = {
            "sample_id", "presentation_id", "instance_id", "ordering_id", "n",
            "entity_type", "query_type", "query_distance", "query_direction", "gold_label",
            "relation_family", "relation_id", "variant", "transformations", "factor_axes",
            "is_multi_transform", "canonical_premises", "rendered_premises",
            "canonical_query", "query", "context", "canonical_prompt", "rendered_prompt",
            "normal_form_sha256", "automatic_equivalence", "automatic_gold_unchanged",
            "morphology_check", "human_review_status", "resolution", "notes",
        }
        for reviewer in (1, 2):
            required.update(f"reviewer{reviewer}_{dimension}" for dimension in (
                "grammar", "naturalness", "equivalence", "morphology", "contact_ambiguity",
            ))
        self.assertTrue(required <= set(rows[0]))

    def test_default_report_covers_balanced_distinct_contexts_and_six_variants(self):
        _, _, metadata, rows = self.report()
        self.assertEqual(len(rows), 216)
        self.assertGreaterEqual(len(rows), 100)
        self.assertEqual(metadata["row_count"], len(rows))
        self.assertEqual(metadata["source_contexts"], 36)
        self.assertEqual(len({row["sample_id"] for row in rows}), len(rows))
        self.assertEqual(len({row["presentation_id"] for row in rows}), len(rows))
        contexts = defaultdict(list)
        for row in rows:
            contexts[row["ordering_id"]].append(row)
        self.assertEqual(len(contexts), 36)
        self.assertEqual(Counter(row["variant"] for row in rows), {f"P{i}": 36 for i in range(6)})
        self.assertEqual(Counter(row["relation_id"] for row in rows), {key: 24 for key in RELATIONS})
        self.assertEqual({row["relation_family"] for row in rows}, {
            "spatial", "temporal", "comparison", "ranking",
        })
        self.assertEqual(Counter(row["entity_type"] for row in rows), {
            "turkish_names": 72, "synthetic_names": 72, "symbolic_ids": 72,
        })
        self.assertEqual(Counter(row["query_type"] for row in rows), {
            "endpoint": 72, "internal": 72, "distance_controlled": 72,
        })
        self.assertEqual(Counter(row["gold_label"] for row in rows), {"true": 108, "false": 108})
        self.assertEqual({row["n"] for row in rows}, {"4", "8"})
        self.assertEqual(Counter((row["query_direction"], row["gold_label"]) for row in rows), {
            (direction, gold): 54
            for direction in ("forward", "reverse") for gold in ("true", "false")
        })
        for context in contexts.values():
            self.assertEqual(len(context), 6)
            self.assertEqual(len({row["instance_id"] for row in context}), 1)
            self.assertEqual(len({row["relation_id"] for row in context}), 1)
            self.assertEqual({row["variant"] for row in context}, {f"P{i}" for i in range(6)})

    def test_csv_surface_text_reconstructs_source_logic_and_gold(self):
        _, _, _, rows = self.report()
        canonical = {}
        for row in rows:
            source = self.sources[row["instance_id"]]
            names = tuple(source["latent_order"])
            premises = row["rendered_premises"].splitlines()
            canonical_premises = row["canonical_premises"].splitlines()
            self.assertEqual(len(premises), source["n"] - 1)
            parsed = [normalize_statement(text, names, row["relation_id"]) for text in premises]
            expected = [tuple(edge) for edge in source["premise_edges"]]
            self.assertEqual(sorted(parsed), sorted(expected))
            canonical_parsed = [normalize_statement(text, names, row["relation_id"])
                                for text in canonical_premises]
            self.assertEqual(sorted(canonical_parsed), sorted(expected))
            query = normalize_query(row["query"], names, row["relation_id"])
            self.assertEqual(query, tuple(source[key] for key in (
                "query_subject", "query_object", "query_relation",
            )))
            self.assertEqual(normalize_query(row["canonical_query"], names, row["relation_id"]), query)
            self.assertEqual(answer_from_edges(names, parsed, *query), row["gold_label"] == "true")
            self.assertEqual(int(row["query_distance"]), source["query_distance"])
            for text in premises:
                self.assertIn("- " + text, row["rendered_prompt"])
            self.assertIn(row["query"], row["rendered_prompt"])
            self.assertIn(row["context"], row["rendered_prompt"])
            context_id = (row["instance_id"], row["relation_id"])
            value = (row["normal_form_sha256"], row["canonical_prompt"],
                     row["canonical_premises"], row["canonical_query"])
            if context_id in canonical:
                self.assertEqual(value, canonical[context_id])
            else:
                canonical[context_id] = value
            self.assertRegex(row["normal_form_sha256"], r"\A[0-9a-f]{64}\Z")
            self.assertEqual(row["automatic_equivalence"], "true")
            self.assertEqual(row["automatic_gold_unchanged"], "true")

    def test_human_assessment_is_pending_and_all_human_judgements_are_blank(self):
        _, _, metadata, rows = self.report()
        self.assertEqual(metadata["human_reviews_completed"], 0)
        self.assertIs(metadata["contains_model_results"], False)
        for row in rows:
            self.assertEqual(row["human_review_status"], "pending")
            fields = [key for key in row if key.startswith(("reviewer1_", "reviewer2_"))]
            self.assertEqual(len(fields), 10)
            for field in (*fields, "resolution", "notes"):
                self.assertEqual(row[field], "")

    def test_multi_factor_flags_disclose_joint_morphological_transformations(self):
        _, _, _, rows = self.report()
        for row in rows:
            axes = json.loads(row["factor_axes"])
            transforms = json.loads(row["transformations"])
            self.assertIsInstance(axes, list)
            self.assertIsInstance(transforms, list)
            self.assertEqual(row["is_multi_transform"], str(len(axes) > 1).lower())
            if row["variant"] in ("P4", "P5"):
                self.assertEqual(row["is_multi_transform"], "true")
                self.assertTrue({"morphology", "syntax", "lexical"} <= set(axes))
            if row["variant"] == "P0":
                self.assertEqual(axes, [])
                self.assertEqual(transforms, [])

    def test_optional_negation_requires_explicit_request_and_adds_one_paired_arm(self):
        _, _, _, baseline = self.report("default")
        _, _, metadata, with_negation = self.report("negation", include_negation=True)
        self.assertNotIn("P6", {row["variant"] for row in baseline})
        self.assertEqual(len(with_negation), 252)
        self.assertEqual(metadata["row_count"], 252)
        self.assertEqual(Counter(row["variant"] for row in with_negation),
                         {f"P{i}": 36 for i in range(7)})
        self.assertEqual([row for row in with_negation if row["variant"] != "P6"], baseline)
        for row in with_negation:
            if row["variant"] == "P6":
                self.assertEqual(row["automatic_equivalence"], "true")
                self.assertEqual(row["automatic_gold_unchanged"], "true")
                self.assertIn("negation", json.loads(row["factor_axes"]))
                self.assertIn("katı bir tam sıralamadır", row["context"])

    def test_same_seed_produces_identical_csv_in_different_directories(self):
        first_csv, _, first, first_rows = self.report("first")
        second_csv, _, second, second_rows = self.report("second")
        self.assertEqual(first_csv.read_bytes(), second_csv.read_bytes())
        self.assertEqual(first_rows, second_rows)
        self.assertEqual(first["csv_sha256"], second["csv_sha256"])

    def test_manifest_hashes_bind_csv_source_manifest_and_each_latent_file(self):
        output, manifest_path, metadata, rows = self.report()
        self.assertEqual(json.loads(manifest_path.read_text(encoding="utf-8")), metadata)
        self.assertEqual(metadata["renderer_version"], RENDERER_VERSION)
        self.assertEqual(metadata["csv_sha256"], hashlib.sha256(output.read_bytes()).hexdigest())
        self.assertEqual(metadata["source_dataset_manifest_sha256"],
                         hashlib.sha256(self.source_manifest.read_bytes()).hexdigest())
        self.assertEqual(metadata["source_files_sha256"], {
            entry["path"]: entry["sha256"] for entry in self.source_metadata["files"]
        })
        for key, column in (
            ("variant", "variant"), ("relation_id", "relation_id"),
            ("entity_type", "entity_type"), ("gold_label", "gold_label"),
            ("query_type", "query_type"), ("query_direction", "query_direction"),
        ):
            self.assertEqual(metadata["counts"][key], dict(Counter(row[column] for row in rows)))
        self.assertTrue(metadata["source_code_sha256"])
        project = Path(__file__).resolve().parents[1]
        for relative, expected in metadata["source_code_sha256"].items():
            self.assertEqual(hashlib.sha256((project / "src" / relative).read_bytes()).hexdigest(), expected)

    def test_existing_output_files_are_refused_without_overwriting(self):
        output, manifest, _, _ = self.report()
        original_output, original_manifest = output.read_bytes(), manifest.read_bytes()
        with self.assertRaises((ValueError, FileExistsError)):
            generate_review_report(self.source_manifest, output, manifest)
        self.assertEqual(output.read_bytes(), original_output)
        self.assertEqual(manifest.read_bytes(), original_manifest)
        another_csv = self.root / "unused.csv"
        with self.assertRaises((ValueError, FileExistsError)):
            generate_review_report(self.source_manifest, another_csv, manifest)
        self.assertFalse(another_csv.exists())
        self.assertEqual(manifest.read_bytes(), original_manifest)

    def test_tampered_source_bytes_are_rejected_before_creating_review_outputs(self):
        source_folder = self.root / "corrupted"
        manifest = source_folder / "dataset_manifest.json"
        data = source_folder / "latent"
        metadata = write_dataset(
            data, manifest,
            GenerationPlan(sizes=(4, 8), examples_per_size=48, distances=(1,)),
        )
        path = data / metadata["files"][0]["path"]
        content = path.read_bytes()
        # Keep records logically valid but break their published byte/hash bond.
        path.write_bytes(content.replace(b'"entity_type":', b'"entity_type": ', 1))
        output, output_manifest = self.root / "bad.csv", self.root / "bad_manifest.json"
        with self.assertRaises(ValueError):
            generate_review_report(manifest, output, output_manifest)
        self.assertFalse(output.exists())
        self.assertFalse(output_manifest.exists())

    def test_cli_exports_review_material_and_machine_readable_summary(self):
        output, manifest = self.root / "cli.csv", self.root / "cli_manifest.json"
        stdout, stderr = io.StringIO(), io.StringIO()
        with redirect_stdout(stdout), redirect_stderr(stderr):
            code = main([
                "--dataset-manifest", str(self.source_manifest),
                "--output", str(output), "--manifest", str(manifest),
                "--seed", "31", "--include-negation",
            ])
        self.assertEqual(code, 0, stderr.getvalue())
        self.assertTrue(output.is_file())
        self.assertTrue(manifest.is_file())
        summary = json.loads(stdout.getvalue())
        self.assertEqual(summary["row_count"], 252)
        self.assertFalse(summary["contains_model_results"])
        with output.open(encoding="utf-8", newline="") as handle:
            rows = list(csv.DictReader(handle))
        self.assertEqual(len(rows), 252)


if __name__ == "__main__":
    unittest.main()

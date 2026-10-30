"""Offline lexical CLI checks; synthetic fixtures stay in disposable folders."""

from contextlib import ExitStack, redirect_stderr, redirect_stdout
from copy import deepcopy
import csv
import hashlib
import io
import json
import os
import sys
from pathlib import Path
import shutil
import tempfile
import unittest
from unittest.mock import patch

from analysis.lexical_bias import __main__ as cli
from analysis.lexical_bias.catalog import build_catalog
from inference.config import load_models


PROJECT = Path(__file__).resolve().parents[1]
WORK = PROJECT.parents[1] / "work"


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def protected_snapshot():
    """The 50 pre-lexical sources, the reasoning extension, and model results."""
    lexical = PROJECT / "src/analysis/lexical_bias"
    reasoning = PROJECT / "src/reasoning"
    training = PROJECT / "src/training"
    review = PROJECT / "src/inference/linguistic_review.py"
    sources = [str(path.relative_to(PROJECT)) for path in (PROJECT / "src").rglob("*.py")
               if not path.is_relative_to(lexical) and not path.is_relative_to(reasoning) and not path.is_relative_to(training) and path != review]
    if len(sources) != 50:
        raise AssertionError("Expected the preceding 50 source modules")
    names = set(sources)
    names.add(str(review.relative_to(PROJECT)))
    names.update(str(path.relative_to(PROJECT)) for path in reasoning.rglob("*.py"))
    names.update(str(path.relative_to(PROJECT)) for path in training.rglob("*.py"))
    for relative in ("results/raw", "results/normalized"):
        names.update(str(path.relative_to(PROJECT)) for path in (PROJECT / relative).rglob("*") if path.is_file())
    return {name: sha(PROJECT / name) for name in sorted(names)}


class LexicalCLITests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(prefix="synthetic-lexical-cli-", dir=WORK)
        self.addCleanup(self.temporary.cleanup)
        self.base = Path(self.temporary.name)
        self.root = self.base / "project"
        shutil.copytree(PROJECT / "src/analysis/lexical_bias", self.root / "src/analysis/lexical_bias",
                        ignore=shutil.ignore_patterns("__pycache__"))
        (self.root / "configs").mkdir()
        shutil.copyfile(PROJECT / "configs/models.yaml", self.root / "configs/models.yaml")
        (self.root / "data/complete_grid").mkdir(parents=True)
        (self.root / "data/complete_grid/pilot_preview.jsonl").write_text("")
        (self.root / "results/raw").mkdir(parents=True)
        (self.root / "results/normalized").mkdir(parents=True)
        (self.root / "results/raw/README.md").write_text("No synthetic research results.\n")
        self.output = self.base / "output"
        self.model_id = load_models(PROJECT / "configs/models.yaml")["models"][0]["id"]

    def fast_preprocessing(self):
        """Only setup is mocked; argument checks and publication remain real."""
        stack = ExitStack()
        stack.enter_context(patch.object(cli, "count_corpus", return_value={
            "rows": [], "coverage": {"status": "missing_corpus"}, "input_files_sha256": {}}))
        stack.enter_context(patch.object(cli, "measure_vocabulary", return_value=[]))
        stack.enter_context(patch.object(cli, "measure_prompts", return_value=[]))
        return stack

    def recorded(self, name="synthetic-run-a", *, observations=None, synthetic=False):
        return {"run_id": name, "synthetic": synthetic,
            "observations": deepcopy(observations or []), "input_files_sha256": {}, "ignored_replicates": 0,
            "plan": {"plan_sha256": "synthetic-plan", "cohort": {"cohort_sha256": "synthetic-cohort"}}}

    def test_actual_full_corpus_and_preview_complete_offline_without_model_results(self):
        before = protected_snapshot()
        actual_output = self.base / "actual-reference-preprocessing"
        with ExitStack() as stack:
            guards = {}
            for name in ("socket.create_connection", "socket.socket.connect", "urllib.request.urlopen"):
                guards[name] = stack.enter_context(patch(name, side_effect=AssertionError("network access forbidden")))
            stack.enter_context(patch.dict(os.environ, {"USE_TORCH": "0", "USE_TF": "0", "USE_FLAX": "0",
                "HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1"}))
            # huggingface_hub freezes HF_HUB_OFFLINE at import time, so a copy
            # cached by an earlier test defeats the environment patch above;
            # flip every already-imported copy for the duration of this test.
            for module in list(sys.modules.values()):
                if module is not None and getattr(module, "HF_HUB_OFFLINE", None) is False:
                    stack.enter_context(patch.object(module, "HF_HUB_OFFLINE", True))
            with redirect_stdout(io.StringIO()) as output:
                # The project now retains completed pilot runs; a bare "--run"
                # selects none of them, keeping this a preprocessing-only check.
                code = cli.main(["--root", str(PROJECT), "--output", str(actual_output), "--run"])
        self.assertEqual(code, 0)
        summary = json.loads(output.getvalue())
        result = json.loads((actual_output / "lexical_manifest.json").read_text())
        self.assertEqual(summary["status"], "preprocessing_only_no_model_results")
        self.assertFalse(result["contains_model_results"])
        self.assertFalse(result["contains_model_weights"])
        self.assertEqual(result["model_api_calls"], 0)
        self.assertEqual(result["n_planned_model_responses"], 0)
        self.assertEqual(result["n_observed_model_responses"], 0)
        self.assertEqual(result["input_runs"], [])
        self.assertEqual(result["table_rows"], {"corpus_frequencies": 18,
            "tokenization_vocabulary": 216, "tokenization_prompt_preview": 1458})
        self.assertEqual(result["prompt_measurement_scope"]["n_source_prompts"], 243)
        self.assertEqual(len(result["selected_model_ids"]), 6)
        self.assertEqual(len(result["unavailable_analyses"]), 5)
        self.assertEqual(set(result["unavailable_analyses"]), {"Frequency_vs_Accuracy", "Frequency_vs_R",
            "Token_Count_vs_Invariance", "Lexical_Direction_vs_Answer_Bias", "Model_x_Vocabulary"})
        self.assertEqual(result["corpus"]["coverage"]["sentences"], 9761)
        self.assertEqual(result["corpus"]["coverage"]["syntactic_words"], 125212)
        self.assertEqual(result["corpus"]["coverage"]["surface_tokens"]
                         + result["corpus"]["coverage"]["surface_punctuation"], 121835)
        self.assertEqual(json.loads((actual_output / "relation_vocabulary.json").read_text()), build_catalog())
        self.assertFalse((actual_output / "figures").exists())
        forbidden = {"answer_bias", "paired_direction", "lexical_profiles", "correlations", "controlled_regressions"}
        self.assertFalse(forbidden.intersection(result["table_rows"]))
        self.assertEqual(len(result["files"]), 7)
        provenance = result["tokenizer_provenance_records"]
        self.assertTrue(provenance)
        for fingerprint, record in provenance.items():
            serialized = json.dumps(record, sort_keys=True, ensure_ascii=False, separators=(",", ":"), allow_nan=False)
            self.assertEqual(hashlib.sha256(serialized.encode()).hexdigest(), fingerprint)
        for entry in result["files"]:
            path = actual_output / entry["path"]
            self.assertEqual(path.stat().st_size, entry["bytes"])
            self.assertEqual(sha(path), entry["sha256"])
        for name, rows in result["table_rows"].items():
            csv_path, tex_path = actual_output / "tables" / (name + ".csv"), actual_output / "tables" / (name + ".tex")
            self.assertTrue(csv_path.is_file())
            self.assertTrue(tex_path.is_file())
            with csv_path.open(newline="", encoding="utf-8") as handle:
                entries = list(csv.DictReader(handle))
                self.assertEqual(len(entries), rows)
                if name.startswith("tokenization_"):
                    self.assertTrue(all("tokenizer_provenance" not in entry for entry in entries))
                    self.assertTrue(all(entry["tokenizer_provenance_sha256"] in provenance for entry in entries))
        self.assertTrue(all(guard.call_count == 0 for guard in guards.values()))
        after = protected_snapshot()
        self.assertEqual(before, after)
        proof = {"proof_kind": "actual_offline_reference_preprocessing_not_llm_results",
            "python": result["environment"]["python"], "status": result["status"],
            "contains_model_results": False, "model_api_calls": 0,
            "table_rows": result["table_rows"], "unavailable_analyses": result["unavailable_analyses"],
            "corpus_coverage": result["corpus"]["coverage"], "tokenizer_coverage": result["tokenizer_coverage"],
            "network_guard_calls": {name: guard.call_count for name, guard in guards.items()},
            "preceding_source_files_unchanged": 50, "raw_and_normalized_results_unchanged": True,
            "protected_files_sha256": after, "output_manifest_sha256": sha(actual_output / "lexical_manifest.json"),
            "verified_artifacts": result["files"], "temporary_output_removed_after_test": True}
        (WORK / "stage7-cli-verification.json").write_text(json.dumps(proof, ensure_ascii=False,
            sort_keys=True, indent=2, allow_nan=False) + "\n", encoding="utf-8")

    def test_existing_output_is_preserved_and_cli_returns_error(self):
        self.output.mkdir()
        keep = self.output / "keep.txt"
        keep.write_text("original")
        with redirect_stderr(io.StringIO()) as error:
            status = cli.main(["--root", str(self.root), "--output", str(self.output)])
        self.assertEqual(status, 2)
        self.assertIn("already exists", error.getvalue())
        self.assertEqual(keep.read_text(), "original")
        self.assertEqual(list(self.output.iterdir()), [keep])

    def test_protected_input_destinations_and_symlinks_are_rejected(self):
        for relative in ("src", "tests", "configs", "logs", "data", "tokenizers", "results/raw", "results/normalized"):
            with self.subTest(relative=relative), self.assertRaisesRegex(ValueError, "protected"):
                cli.run(self.root, self.root / relative / "synthetic-new-output")
        (self.base / "alias").symlink_to(self.root, target_is_directory=True)
        with self.assertRaisesRegex(ValueError, "symlinks"):
            cli.run(self.root, self.base / "alias/new-output")

    def test_installed_source_drift_rejected_without_partial_output(self):
        (self.root / "src/analysis/lexical_bias/__init__.py").write_text("# synthetic altered code\n")
        with self.fast_preprocessing(), self.assertRaisesRegex(ValueError, "lexical code differs"):
            cli.run(self.root, self.output)
        self.assertFalse(self.output.exists())
        self.assertFalse(list(self.base.glob(".lexical-*")))

    def test_duplicate_or_malformed_run_selection_is_rejected(self):
        with self.fast_preprocessing():
            for ids in (["synthetic-a", "synthetic-a"], "synthetic-run"):
                with self.subTest(ids=ids), self.assertRaisesRegex(ValueError, "duplicate|list"):
                    cli.run(self.root, self.output, run_ids=ids)
        self.assertFalse(self.output.exists())

    def test_duplicate_tasks_across_runs_cannot_become_independent_evidence(self):
        rows = [{"model_id": self.model_id, "task_id": "synthetic-same-call"}]
        records = [self.recorded(name, observations=rows) for name in ("synthetic-a", "synthetic-b")]
        with self.fast_preprocessing(), patch.object(cli, "load_run", side_effect=records), \
                self.assertRaisesRegex(ValueError, "overlapping run tasks"):
            cli.run(self.root, self.output, run_ids=[record["run_id"] for record in records])
        self.assertFalse(self.output.exists())

    def test_mismatched_cohorts_are_not_pooled(self):
        records = [self.recorded(name) for name in ("synthetic-a", "synthetic-b")]
        records[1]["plan"]["cohort"]["cohort_sha256"] = "synthetic-other-cohort"
        with self.fast_preprocessing(), patch.object(cli, "load_run", side_effect=records), \
                self.assertRaisesRegex(ValueError, "cohorts differ"):
            cli.run(self.root, self.output, run_ids=[record["run_id"] for record in records])

    def test_fixture_runs_and_fixture_tokenizer_counts_cannot_be_published(self):
        record = self.recorded(synthetic=True)
        with self.fast_preprocessing(), patch.object(cli, "load_run", return_value=record), \
                self.assertRaisesRegex(ValueError, "fixture result"):
            cli.run(self.root, self.output, run_ids=[record["run_id"]])
        with self.fast_preprocessing(), patch.object(cli, "measure_vocabulary", return_value=[{"fixture_only": True}]), \
                self.assertRaisesRegex(ValueError, "synthetic tokenizer"):
            cli.run(self.root, self.output)
        self.assertFalse(self.output.exists())

    def test_reader_cache_rejection_propagates_without_reinterpreting_evidence(self):
        # Immutable journal/cache validation belongs to analysis.io.load_run;
        # this orchestration check verifies it is not bypassed or swallowed.
        with self.fast_preprocessing(), patch.object(cli, "load_run",
                side_effect=ValueError("cached responses cannot be independent research observations")) as reader, \
                self.assertRaisesRegex(ValueError, "cached responses"):
            cli.run(self.root, self.output, run_ids=["synthetic-cached-run"])
        reader.assert_called_once_with(self.root, "synthetic-cached-run")
        self.assertFalse(self.output.exists())

    def test_tokenizer_provenance_hash_mismatch_is_rejected_before_export(self):
        row = {"fixture_only": False, "tokenizer_provenance": "{}", "tokenizer_provenance_sha256": "0" * 64}
        with self.fast_preprocessing(), patch.object(cli, "measure_vocabulary", return_value=[row]), \
                self.assertRaisesRegex(ValueError, "provenance fingerprint mismatch"):
            cli.run(self.root, self.output)
        self.assertFalse(self.output.exists())

    def test_failed_export_leaves_no_partial_output_and_preserves_inputs(self):
        original = sha(self.root / "configs/models.yaml")
        with self.fast_preprocessing(), patch.object(cli, "export_tables", side_effect=ValueError("synthetic export failure")), \
                self.assertRaisesRegex(ValueError, "export failure"):
            cli.run(self.root, self.output)
        self.assertFalse(self.output.exists())
        self.assertFalse(list(self.base.glob(".lexical-*")))
        self.assertEqual(sha(self.root / "configs/models.yaml"), original)

    def test_fast_empty_run_keeps_performance_unavailable_and_deterministic(self):
        with self.fast_preprocessing():
            first = cli.run(self.root, self.output)
            second = cli.run(self.root, self.base / "other-output")
        self.assertEqual(first, second)
        self.assertEqual(first["status"], "preprocessing_only_no_model_results")
        self.assertEqual(first["model_api_calls"], 0)
        self.assertEqual(first["n_observed_model_responses"], 0)
        self.assertEqual(len(first["unavailable_analyses"]), 5)
        self.assertEqual(first["table_rows"], {"corpus_frequencies": 0,
            "tokenization_vocabulary": 0, "tokenization_prompt_preview": 0})

    def test_model_selection_is_explicit_and_no_fallback_occurs(self):
        with self.fast_preprocessing():
            for ids in (["synthetic-missing-model"], [self.model_id, self.model_id]):
                with self.subTest(ids=ids), self.assertRaises(ValueError):
                    cli.run(self.root, self.output, model_ids=ids)
        self.assertFalse(self.output.exists())


if __name__ == "__main__":
    unittest.main()

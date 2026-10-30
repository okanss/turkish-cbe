"""Read-only analysis of explicit synthetic journals on real frozen prompts.

All runner outputs below are fixtures in disposable projects.  No test contacts
a provider, downloads model weights, or creates research-result deliverables.
Mutations intentionally corrupt only those disposable journals.
"""

from contextlib import closing
import hashlib
import importlib.util
import json
from pathlib import Path
import sqlite3
import unittest

from analysis.io import canonical, decode, discover_runs, load_run
from inference.dataset import GridDataset
from inference.prompts import make_messages
from inference.providers import ProviderError
from inference.runner import task_for
from inference.storage import RunStore
import test_inference_runner as fixtures


HAS_TIKTOKEN = importlib.util.find_spec("tiktoken") is not None


@unittest.skipUnless(HAS_TIKTOKEN, "pinned tiktoken is needed for frozen prompt reconstruction")
class AnalysisIOTests(unittest.TestCase):
    # Reuse the already-established real frozen-bank fixture setup, without
    # inheriting its tests or mocking the actual prompt reconstruction.
    setUpClass = classmethod(fixtures.InferenceRunnerTests.setUpClass.__func__)
    setUp = fixtures.InferenceRunnerTests.setUp
    plan = fixtures.InferenceRunnerTests.plan
    execute = fixtures.InferenceRunnerTests.execute
    normalized = fixtures.InferenceRunnerTests.normalized

    def complete_run(self, *, factory=None, policy=None, run_id="fixture-run"):
        plan = self.plan(policy=policy)
        factory = fixtures.FixtureFactory() if factory is None else factory
        result = self.execute(plan, factory, run_id)
        self.assertTrue(result["complete"])
        return plan, factory

    def connection(self):
        return closing(sqlite3.connect(self.root / "logs/fixture-run.sqlite"))

    def rewrite_normalized(self, transform):
        """Coherently rehash one test row to exercise semantic validation."""
        with self.connection() as connection:
            row = connection.execute("SELECT seq,record_json FROM normalized_outbox ORDER BY seq LIMIT 1").fetchone()
            record = json.loads(row[1])
            transform(record)
            encoded = canonical(record)
            sha = hashlib.sha256((encoded + "\n").encode()).hexdigest()
            connection.execute("UPDATE normalized_outbox SET record_json=?,record_sha256=? WHERE seq=?",
                               (encoded, sha, row[0]))
            connection.commit()
            output = b"".join((item[0] + "\n").encode() for item in
                              connection.execute("SELECT record_json FROM normalized_outbox ORDER BY seq"))
        (self.root / "results/normalized/fixture-run.jsonl").write_bytes(output)

    def first_attempt(self):
        with self.connection() as connection:
            return connection.execute("SELECT attempt_id FROM attempts ORDER BY created_at LIMIT 1").fetchone()[0]

    def alter_raw(self, kind, transform, *, rehash=True, attempt=None):
        attempt = self.first_attempt() if attempt is None else attempt
        path = self.root / "results/raw/fixture-run/attempts" / attempt / (kind + ".json")
        data = json.loads(path.read_bytes())
        transform(data)
        payload = (canonical(data) + "\n").encode()
        path.write_bytes(payload)
        if rehash:
            with self.connection() as connection:
                connection.execute("UPDATE attempts SET " + kind + "_sha256=? WHERE attempt_id=?",
                                   (hashlib.sha256(payload).hexdigest(), attempt))
                connection.commit()

    def assert_rejected(self, message=None):
        with self.assertRaisesRegex(ValueError, message or "."):
            load_run(self.root, "fixture-run", allow_synthetic=True)

    def test_fixture_requires_explicit_synthetic_library_override(self):
        self.complete_run()
        with self.assertRaisesRegex(ValueError, "synthetic"):
            load_run(self.root, "fixture-run")
        result = load_run(self.root, "fixture-run", allow_synthetic=True)
        self.assertTrue(result["synthetic"])
        self.assertEqual(len(result["observations"]), 32)

    def test_frozen_grid_sample_is_reconstructed_and_all_planned_identities_retained(self):
        plan, factory = self.complete_run()
        result = load_run(self.root, "fixture-run", allow_synthetic=True)
        with GridDataset(self.root, fixtures.MANIFEST) as data:
            row = next(data.iter_rows(plan["cohort"]))
        task = task_for(row, self.model, make_messages(row))
        observation = next(item for item in result["observations"] if item["task_id"] == task["task_id"])
        for field in ("latent_instance_id", "grid_query_id", "grid_design_id", "gold_label",
                      "language", "condition", "n", "query_type", "query_distance"):
            self.assertEqual(observation[field], row[field])
        self.assertEqual(observation["cluster_id"], row["latent_instance_id"])
        self.assertEqual(observation["prediction"], True)  # Fixed fixture answer, not gold-derived.
        self.assertEqual(observation["input_tokens"], 100)
        self.assertEqual(observation["response_status"], "valid")
        self.assertEqual(len(factory.calls), plan["generation_calls"])
        self.assertEqual(len({item["task_id"] for item in result["observations"]}), 32)
        self.assertEqual(len({item["cluster_id"] for item in result["observations"]}), 8)

    def test_analysis_is_read_only_and_hashes_its_actual_evidence(self):
        self.complete_run()
        def snapshot():
            return {str(path.relative_to(self.root)): hashlib.sha256(path.read_bytes()).hexdigest()
                    for directory in ("results/raw", "results/normalized", "logs")
                    for path in (self.root / directory).rglob("*") if path.is_file()}
        before = snapshot()
        result = load_run(self.root, "fixture-run", allow_synthetic=True)
        self.assertEqual(snapshot(), before)
        for path, sha in result["input_files_sha256"].items():
            self.assertEqual(before[path], sha)
        self.assertIn("results/raw/fixture-run/manifest.json", result["input_files_sha256"])
        self.assertIn("logs/fixture-run.sqlite", result["input_files_sha256"])
        self.assertEqual(discover_runs(self.root), ["fixture-run"])

    def test_never_started_tasks_remain_in_denominator_after_provider_failure(self):
        plan = self.plan()
        report = self.execute(plan, fixtures.FixtureFactory(preflight_error=ProviderError("fixture_unavailable")))
        self.assertFalse(report["complete"])
        self.assertEqual(report["actual_generation_calls_this_invocation"], 0)
        result = load_run(self.root, "fixture-run", allow_synthetic=True)
        observations = result["observations"]
        self.assertEqual(len(observations), plan["generation_calls"])
        self.assertEqual(sum(item["response_status"] == "invalid" for item in observations), 1)
        self.assertEqual(sum(item["response_status"] == "missing" for item in observations), 31)
        self.assertTrue(all(item["prediction"] is None for item in observations))

    def test_unresolved_ambiguous_call_keeps_every_missing_response(self):
        plan = self.plan()
        self.execute(plan, fixtures.FixtureFactory(actions=[
            ProviderError("fixture_timeout", retryable=True, ambiguous=True)]))
        result = load_run(self.root, "fixture-run", allow_synthetic=True)
        self.assertEqual(len(result["observations"]), 32)
        self.assertTrue(all(item["response_status"] == "missing" for item in result["observations"]))
        self.assertTrue(all(item["invalid_reason"] == "missing_response" for item in result["observations"]))

    def test_wrong_language_outputs_remain_invalid_without_reparsing_leniently(self):
        self.complete_run(factory=fixtures.FixtureFactory(always={"text": "YES"}))
        observations = load_run(self.root, "fixture-run", allow_synthetic=True)["observations"]
        turkish = [item for item in observations if item["language"] == "tr"]
        english = [item for item in observations if item["language"] == "en"]
        self.assertEqual((len(turkish), len(english)), (16, 16))
        self.assertTrue(all(item["response_status"] == "invalid" and item["prediction"] is None
                            and item["invalid_reason"] == "wrong_language" for item in turkish))
        self.assertTrue(all(item["response_status"] == "valid" for item in english))

    def test_replicates_are_verified_but_only_replicate_zero_is_observed(self):
        plan, factory = self.complete_run(policy={**self.policy, "replicates": 2})
        result = load_run(self.root, "fixture-run", allow_synthetic=True)
        self.assertEqual(len(factory.calls), 64)
        self.assertEqual(plan["generation_calls"], 64)
        self.assertEqual(result["ignored_replicates"], 32)
        self.assertEqual(len(result["observations"]), 32)
        self.assertEqual(len({item["task_id"] for item in result["observations"]}), 32)

    def test_cache_reuse_is_not_accepted_as_an_independent_observation(self):
        plan = self.plan(policy={**self.policy, "cache": True})
        self.execute(plan, fixtures.FixtureFactory(), "fixture-origin")
        self.execute(plan, fixtures.FixtureFactory(), "fixture-run")
        self.assert_rejected("cached")

    def test_unplanned_registered_task_is_rejected(self):
        plan, _ = self.complete_run()
        with self.connection() as connection:
            task = json.loads(connection.execute("SELECT task_json FROM tasks LIMIT 1").fetchone()[0])
        task.update(task_id="unplanned-task", request_key="unplanned-request")
        with RunStore(self.root, "fixture-run", plan) as store:
            store.register_tasks([task])
        self.assert_rejected("unplanned")

    def test_unplanned_normalized_task_is_rejected_with_coherent_record_hash(self):
        self.complete_run()
        unexpected = "call-" + "f" * 64
        self.rewrite_normalized(lambda value: value.update(task_id=unexpected))
        with self.connection() as connection:
            connection.execute("UPDATE normalized_outbox SET task_id=? WHERE seq=(SELECT MIN(seq) FROM normalized_outbox)",
                               (unexpected,))
            connection.commit()
        self.assert_rejected("unplanned")

    def test_duplicate_checkpointed_task_record_is_rejected(self):
        self.complete_run()
        with self.connection() as connection:
            connection.execute("ALTER TABLE normalized_outbox RENAME TO original_outbox")
            connection.execute("CREATE TABLE normalized_outbox AS SELECT * FROM original_outbox")
            connection.execute("INSERT INTO normalized_outbox SELECT seq+1000,task_id,attempt_id,record_json,record_sha256 FROM original_outbox ORDER BY seq LIMIT 1")
            connection.commit()
            payload = b"".join((item[0] + "\n").encode() for item in connection.execute(
                "SELECT record_json FROM normalized_outbox ORDER BY seq"))
        (self.root / "results/normalized/fixture-run.jsonl").write_bytes(payload)
        self.assert_rejected("identity/hash")

    def test_uncheckpointed_duplicate_jsonl_is_rejected_without_repair(self):
        self.complete_run()
        path = self.root / "results/normalized/fixture-run.jsonl"
        payload = path.read_bytes()
        path.write_bytes(payload + payload.splitlines(keepends=True)[0])
        self.assert_rejected("outbox")
        self.assertEqual(path.read_bytes(), payload + payload.splitlines(keepends=True)[0])

    def test_raw_byte_corruption_is_rejected(self):
        self.complete_run()
        self.alter_raw("response", lambda value: value.update(text="NO"), rehash=False)
        self.assert_rejected("raw hash")

    def test_raw_evidence_of_an_earlier_rejected_retry_is_also_verified(self):
        rejected = ProviderError("rate_limited", retryable=True, ambiguous=False, status_code=429)
        self.complete_run(factory=fixtures.FixtureFactory(actions=[rejected]))
        self.alter_raw("error", lambda value: value.update(code="altered-fixture-code"), rehash=False)
        self.assert_rejected("raw hash")

    def test_raw_evidence_without_any_normalized_response_is_still_verified(self):
        self.execute(self.plan(), fixtures.FixtureFactory(actions=[ProviderError("fixture_timeout", ambiguous=True)]))
        self.alter_raw("request", lambda value: value.update(prompt_version="altered-fixture-version"), rehash=False)
        self.assert_rejected("raw hash")

    def test_checkpointed_normalized_hash_itself_is_verified(self):
        self.complete_run()
        with self.connection() as connection:
            connection.execute("UPDATE normalized_outbox SET record_sha256=? WHERE seq=(SELECT MIN(seq) FROM normalized_outbox)",
                               ("0" * 64,))
            connection.commit()
        self.assert_rejected("identity/hash")

    def test_plan_fingerprint_is_not_bypassed_by_a_changed_call_denominator(self):
        self.complete_run()
        path = self.root / "results/raw/fixture-run/manifest.json"
        manifest = json.loads(path.read_bytes())
        manifest["plan"]["generation_calls"] += 1
        path.write_text(canonical(manifest) + "\n")
        self.assert_rejected("fingerprint")

    def test_changed_inference_source_bytes_invalidate_analysis_of_old_plan(self):
        self.complete_run()
        path = self.root / "src/inference/parser.py"
        path.write_bytes(path.read_bytes() + b"\n# Synthetic changed source\n")
        self.assert_rejected("source")

    def test_reported_input_token_counts_require_nonnegative_exact_integers(self):
        self.complete_run(factory=fixtures.FixtureFactory(actions=[{"usage": {"input_tokens": True}}]))
        self.assert_rejected("input token")

    def test_normalized_correctness_cannot_be_changed_even_with_matching_outbox_hash(self):
        self.complete_run()
        self.rewrite_normalized(lambda value: value.update(correct=not value["correct"]))
        self.assert_rejected("raw/parser/source")

    def test_normalized_schema_version_cannot_be_changed_with_coherent_hash(self):
        self.complete_run()
        self.rewrite_normalized(lambda value: value.update(schema_version=2))
        self.assert_rejected("raw/parser/source")

    def test_bool_integer_schema_confusion_is_rejected(self):
        self.complete_run()
        self.rewrite_normalized(lambda value: value.update(schema_version=True))
        self.assert_rejected("raw/parser/source")

    def test_gold_source_type_confusion_is_rejected(self):
        self.complete_run()
        self.rewrite_normalized(lambda value: value["source"].update(gold_label=int(value["source"]["gold_label"])))
        self.assert_rejected("raw/parser/source")

    def test_registered_source_must_match_the_frozen_task(self):
        self.complete_run()
        with self.connection() as connection:
            task_id, encoded = connection.execute("SELECT task_id,task_json FROM tasks LIMIT 1").fetchone()
            task = json.loads(encoded)
            task["source"]["gold_label"] = not task["source"]["gold_label"]
            connection.execute("UPDATE tasks SET task_json=? WHERE task_id=?", (canonical(task), task_id))
            connection.commit()
        self.assert_rejected("frozen source")

    def test_raw_request_messages_must_match_frozen_messages_after_rehash(self):
        self.complete_run()
        self.alter_raw("request", lambda value: value["messages"][1].update(content="A different logical problem"))
        self.assert_rejected("request")

    def test_raw_request_source_must_match_frozen_source_after_rehash(self):
        self.complete_run()
        self.alter_raw("request", lambda value: value["source"].update(gold_label=not value["source"]["gold_label"]))
        self.assert_rejected("request")

    def test_normalized_response_must_belong_to_terminal_current_attempt(self):
        self.complete_run()
        with self.connection() as connection:
            connection.execute("UPDATE tasks SET status='ambiguous' WHERE task_id=(SELECT task_id FROM normalized_outbox ORDER BY seq LIMIT 1)")
            connection.commit()
        self.assert_rejected("terminal")

    def test_checkpoint_metadata_and_manifest_identity_are_checked(self):
        self.complete_run()
        with self.connection() as connection:
            connection.execute("UPDATE metadata SET value='other-run' WHERE key='run_id'")
            connection.commit()
        self.assert_rejected("checkpoint identity")

    def test_run_with_exclusive_writer_lock_is_not_analysed(self):
        plan, _ = self.complete_run()
        with RunStore(self.root, "fixture-run", plan):
            self.assert_rejected("active")

    def test_symlinked_raw_evidence_is_rejected(self):
        self.complete_run()
        path = self.root / "results/raw/fixture-run/attempts" / self.first_attempt() / "response.json"
        outside = self.root.parent / "outside-response.json"
        outside.write_bytes(path.read_bytes())
        path.unlink()
        path.symlink_to(outside)
        self.assert_rejected("symlink")

    def test_abandoned_ambiguous_call_requires_intact_resolution_evidence(self):
        plan = self.plan()
        self.execute(plan, fixtures.FixtureFactory(actions=[ProviderError("fixture_timeout", ambiguous=True)]))
        with RunStore(self.root, "fixture-run", plan) as store:
            task = store.pending_tasks()[0]
            store.resolve_ambiguous(task["task_id"], "abandon", "Synthetic auditor resolution")
        result = load_run(self.root, "fixture-run", allow_synthetic=True)
        self.assertEqual(sum(item["invalid_reason"] == "ambiguous_call_abandoned"
                             for item in result["observations"]), 1)
        resolution = next((self.root / "results/raw/fixture-run/resolutions").glob("*.json"))
        resolution.write_bytes(resolution.read_bytes() + b" ")
        self.assert_rejected("resolution")


class StrictJSONTests(unittest.TestCase):
    def test_duplicate_and_nonfinite_json_are_rejected(self):
        for text in ('{"key":1,"key":2}', '{"n":NaN}', '{"n":Infinity}', '{"n":-Infinity}'):
            with self.subTest(text=text), self.assertRaises(ValueError):
                decode(text)

    def test_unsafe_run_ids_are_rejected_before_any_file_access(self):
        for run_id in ("../run", "", "a/b", ".", "..", None, "has space"):
            with self.subTest(run_id=run_id), self.assertRaises(ValueError):
                load_run(Path("/does-not-exist"), run_id)


if __name__ == "__main__":
    unittest.main()

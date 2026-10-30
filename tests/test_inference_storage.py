"""Fault-injected call-journal durability, conservative resume and immutability."""

from copy import deepcopy
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

from inference.storage import (
    AmbiguousCallError, ImmutableConflictError, IntegrityError, RunLockedError,
    RunStore, StorageError,
)


class InferenceStorageTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.plan = {"phase": "pilot", "model": "frozen-fixture", "seed": 71}
        self.tasks = [{"task_id": "task-a", "request_key": "key-a", "source_id": "source-a"}]
        self.request = {"model": "frozen-fixture", "messages": [{"role": "user", "content": "Örnek?"}]}
        self.response = {"text": "EVET", "usage": {"input_tokens": 6, "output_tokens": 2}}
        self.rejection = {"code": "rate_limited", "retryable": True, "ambiguous": False}

    def tearDown(self):
        self.temporary.cleanup()

    def store(self, plan=None, run_id="run-71"):
        return RunStore(self.root, run_id, self.plan if plan is None else plan)

    def setup_call(self, store):
        store.register_tasks(deepcopy(self.tasks))
        return store.claim("task-a", deepcopy(self.request))

    def complete(self, store, status="succeeded"):
        attempt = self.setup_call(store)
        store.write_response(attempt, self.response)
        store.finish("task-a", attempt, {"valid": True, "prediction": True}, status=status)
        return attempt

    def test_manifest_is_exclusive_and_plan_changes_are_rejected_without_overwrite(self):
        with self.store() as store:
            raw = store.manifest_path.read_bytes()
            self.assertEqual(json.loads(raw)["plan"], self.plan)
        with self.store() as store:
            self.assertEqual(store.manifest_path.read_bytes(), raw)
        with self.assertRaises(ImmutableConflictError):
            with self.store({**self.plan, "seed": 72}):
                pass
        self.assertEqual((self.root / "results/raw/run-71/manifest.json").read_bytes(), raw)

    def test_methods_require_context_and_run_id_cannot_escape_paths(self):
        store = self.store()
        with self.assertRaises(StorageError):
            store.summary()
        for run_id in ("../escape", ".", "", "/absolute", "a/b", "a\n", "x" * 97):
            with self.assertRaises(ValueError):
                self.store(run_id=run_id)

    def test_lifetime_lock_blocks_second_owner_and_releases_after_close(self):
        with self.store():
            with self.assertRaises(RunLockedError):
                with self.store():
                    pass
        with self.store() as store:
            self.assertEqual(store.summary()["tasks"], 0)

    def test_lifetime_lock_also_blocks_a_separate_process(self):
        source = ("from pathlib import Path\nfrom inference.storage import RunStore, RunLockedError\n"
                  "import json,sys\ntry:\n with RunStore(Path(sys.argv[1]),'run-71',json.loads(sys.argv[2])):\n"
                  "  sys.exit(3)\nexcept RunLockedError:\n sys.exit(0)\n")
        with self.store():
            child = subprocess.run([sys.executable, "-c", source, str(self.root), json.dumps(self.plan)],
                                   env=os.environ.copy(), capture_output=True, text=True, timeout=10)
            self.assertEqual(child.returncode, 0, child.stderr)

    def test_registration_is_idempotent_and_rejects_changed_tasks_or_duplicate_requests(self):
        with self.store() as store:
            store.register_tasks(self.tasks)
            store.register_tasks(self.tasks)
            self.assertEqual(store.summary()["tasks"], 1)
            with self.assertRaises(ImmutableConflictError):
                store.register_tasks([{**self.tasks[0], "source_id": "different"}])
            with self.assertRaises(ImmutableConflictError):
                store.register_tasks([{**self.tasks[0], "task_id": "task-b"}])
            with self.assertRaises(ValueError):
                store.register_tasks(self.tasks * 2)
            self.assertEqual(store.summary()["tasks"], 1)

    def test_registration_transaction_does_not_keep_half_a_batch(self):
        with self.store() as store:
            with self.assertRaises(ImmutableConflictError):
                store.register_tasks([self.tasks[0], {"task_id": "task-b", "request_key": "key-a"}])
            self.assertEqual(store.summary()["tasks"], 0)

    def test_claim_durably_publishes_request_before_returning(self):
        with self.store() as store:
            attempt = self.setup_call(store)
            self.assertTrue(attempt.startswith("call-"))
            self.assertEqual(store.get_request(attempt), self.request)
            self.assertTrue(store.attempt_path(attempt, "request").is_file())
            self.assertEqual(store.get_task("task-a")["status"], "inflight")
            with self.assertRaises(AmbiguousCallError):
                store.claim("task-a", self.request)
            with self.assertRaises(ImmutableConflictError):
                store.claim("task-a", {**self.request, "model": "different"})

    def test_raw_response_is_immutable_and_identical_replay_is_idempotent(self):
        with self.store() as store:
            attempt = self.setup_call(store)
            first = store.write_response(attempt, self.response)
            payload = Path(first["path"]).read_bytes()
            self.assertEqual(first["sha256"], hashlib.sha256(payload).hexdigest())
            self.assertEqual(first["bytes"], len(payload))
            self.assertEqual(store.write_response(attempt, self.response), first)
            with self.assertRaises(ImmutableConflictError):
                store.write_response(attempt, {"text": "HAYIR"})
            self.assertEqual(Path(first["path"]).read_bytes(), payload)

    def test_normalized_completion_is_once_per_task_and_raw_output_stays_unchanged(self):
        with self.store() as store:
            attempt = self.complete(store)
            raw = store.attempt_path(attempt).read_bytes()
            normalized = store.normalized_path.read_bytes()
            store.finish("task-a", attempt, {"valid": True, "prediction": True})
            self.assertEqual(store.normalized_path.read_bytes(), normalized)
            self.assertIsNone(store.claim("task-a", self.request))
            self.assertEqual(store.pending_tasks(), [])
            with self.assertRaises(ImmutableConflictError):
                store.finish("task-a", attempt, {"valid": True, "prediction": False})
            self.assertEqual(store.attempt_path(attempt).read_bytes(), raw)
            row = json.loads(normalized)
            self.assertEqual((row["task_id"], row["attempt_id"], row["run_id"]), ("task-a", attempt, "run-71"))

    def test_cached_completion_has_a_new_local_attempt_and_cache_reference(self):
        with self.store() as store:
            attempt = self.setup_call(store)
            response = {**self.response, "cache_reference": {"run_id": "prior", "attempt_id": "old-call"}}
            store.write_response(attempt, response)
            store.finish("task-a", attempt, {"cached": True}, status="cached")
            self.assertEqual(store.get_task("task-a")["status"], "cached")
            self.assertIsNone(store.claim("task-a", self.request))
            self.assertEqual(store.get_response(attempt), response)

    def test_permanent_failure_requires_raw_error_and_cannot_turn_into_success(self):
        with self.store() as store:
            attempt = self.setup_call(store)
            with self.assertRaises(StorageError):
                store.finish("task-a", attempt, {}, status="permanent_failed")
            store.write_error(attempt, {"code": "unknown_model", "retryable": False, "ambiguous": False})
            store.finish("task-a", attempt, {"error": "unknown_model"}, status="permanent_failed")
            self.assertIsNone(store.claim("task-a", self.request))
            self.assertEqual(store.summary()["statuses"], {"permanent_failed": 1})
            with self.assertRaises(StorageError):
                store.finish("task-a", attempt, {})

    def test_finished_raw_response_is_recovered_without_a_second_claim(self):
        with self.store() as store:
            attempt = self.setup_call(store)
            store.write_response(attempt, self.response)
        with self.store() as store:
            task = store.pending_tasks()[0]
            self.assertEqual(task["status"], "inflight")
            self.assertTrue(task["recoverable"])
            self.assertEqual(store.get_response(task["attempt_id"]), self.response)
            with self.assertRaises(AmbiguousCallError):
                store.claim("task-a", self.request)
            store.finish("task-a", attempt, {"recovered": True})
            self.assertEqual(store.summary()["attempts"], 1)

    def test_inflight_without_response_becomes_ambiguous_and_never_auto_retries(self):
        with self.store() as store:
            attempt = self.setup_call(store)
        with self.store() as store:
            task = store.pending_tasks()[0]
            self.assertEqual(task["status"], "ambiguous")
            self.assertFalse(task["recoverable"])
            self.assertEqual(task["attempt_id"], attempt)
            with self.assertRaises(AmbiguousCallError):
                store.claim("task-a", self.request)
            self.assertEqual(store.summary()["attempts"], 1)

    def test_known_rejected_errors_can_retry_with_unique_preserved_attempts(self):
        with self.store() as store:
            first = self.setup_call(store)
            store.set_retry("task-a", first, self.rejection)
            self.assertEqual(store.normalized_path.read_bytes(), b"")
            second = store.claim("task-a", self.request)
            self.assertNotEqual(first, second)
            self.assertEqual(store.get_error(first), self.rejection)
            store.write_response(second, self.response)
            store.finish("task-a", second, {"valid": True})
            summary = store.summary()
            self.assertEqual((summary["attempts"], summary["raw_requests"], summary["raw_errors"], summary["raw_responses"]), (2, 2, 1, 1))
            self.assertEqual(summary["normalized_records"], 1)

    def test_crash_after_known_rejection_persistence_recovers_retryable_state(self):
        with self.store() as store:
            attempt = self.setup_call(store)
            store.write_error(attempt, self.rejection)
        with self.store() as store:
            self.assertEqual(store.get_task("task-a")["status"], "retryable")
            self.assertNotEqual(store.claim("task-a", self.request), attempt)

    def test_retry_classifier_refuses_uncertain_or_unclassified_errors(self):
        with self.store() as store:
            attempt = self.setup_call(store)
            for error in ({"code": "timeout"}, {"retryable": True, "ambiguous": True},
                          {"retryable": False, "ambiguous": False}):
                with self.assertRaises(StorageError):
                    store.set_retry("task-a", attempt, error)
            self.assertIsNone(store.get_error(attempt))
            store.mark_ambiguous("task-a", attempt, {"code": "timeout", "ambiguous": True})
            self.assertEqual(store.get_task("task-a")["status"], "ambiguous")

    def test_ambiguous_retry_requires_acknowledgment_and_retains_immutable_audit(self):
        with self.store() as store:
            first = self.setup_call(store)
            store.mark_ambiguous("task-a", first, {"code": "transport_timeout", "ambiguous": True})
            with self.assertRaises(StorageError):
                store.resolve_ambiguous("task-a", "retry", "Provider logs could not resolve receipt")
            resolution = store.resolve_ambiguous("task-a", "retry", "Explicit duplicate-call risk accepted", True)
            audit = store.get_resolution(resolution)
            self.assertTrue(audit["ack_duplicate_risk"])
            self.assertEqual(audit["attempt_id"], first)
            self.assertNotEqual(store.claim("task-a", self.request), first)
            self.assertEqual(store.summary()["resolutions"], 1)
            self.assertEqual(store.get_error(first)["code"], "transport_timeout")

    def test_abandoning_ambiguous_call_is_terminal_and_emits_only_failure_record(self):
        with self.store() as store:
            attempt = self.setup_call(store)
            store.mark_ambiguous("task-a", attempt, {"code": "unknown_outcome"})
            resolution = store.resolve_ambiguous("task-a", "abandon", "Avoid a possible duplicate")
            task = store.get_task("task-a")
            self.assertEqual(task["status"], "permanent_failed")
            self.assertEqual(task["normalized"]["error"]["code"], "ambiguous_call_abandoned")
            self.assertEqual(task["normalized"]["resolution_id"], resolution)
            self.assertIsNone(store.claim("task-a", self.request))
        with self.store() as store:
            self.assertEqual(store.summary()["normalized_records"], 1)

    def test_a_recoverable_response_cannot_be_resolved_by_starting_another_call(self):
        with self.store() as store:
            attempt = self.setup_call(store)
            store.mark_ambiguous("task-a", attempt, {"code": "late_response"})
            store.write_response(attempt, self.response)
            with self.assertRaises(StorageError):
                store.resolve_ambiguous("task-a", "retry", "Retry", True)
            store.finish("task-a", attempt, {"recovered": True})

    def test_raw_hash_tampering_is_detected_on_response_recovery(self):
        with self.store() as store:
            attempt = self.setup_call(store)
            store.write_response(attempt, self.response)
            path = store.attempt_path(attempt)
        path.write_text('{"text":"HAYIR"}\n')
        with self.assertRaises(IntegrityError):
            with self.store():
                pass

    def test_raw_request_tampering_and_missing_terminal_files_are_detected(self):
        with self.store() as store:
            attempt = self.complete(store)
            request_path = store.attempt_path(attempt, "request")
            response_path = store.attempt_path(attempt)
            request_path.write_text("{}\n")
            with self.assertRaises(IntegrityError):
                store.get_request(attempt)
            response_path.unlink()
            with self.assertRaises(IntegrityError):
                store.get_response(attempt)

    def test_response_publication_failure_leaves_no_valid_partial_raw_response(self):
        with self.store() as store:
            attempt = self.setup_call(store)
            with patch("inference.storage.os.link", side_effect=OSError("simulated interrupted publication")):
                with self.assertRaises(OSError):
                    store.write_response(attempt, self.response)
            self.assertIsNone(store.get_response(attempt))
            self.assertEqual(store.summary()["raw_responses"], 0)
            self.assertEqual(list(store.attempt_path(attempt).parent.glob(".*.tmp")), [])
        with self.store() as store:
            self.assertEqual(store.get_task("task-a")["status"], "ambiguous")

    def test_failure_before_request_publication_never_returns_a_claim_for_calling_provider(self):
        with self.store() as store:
            store.register_tasks(self.tasks)
            with patch("inference.storage._publish", side_effect=OSError("disk error")):
                with self.assertRaises(OSError):
                    store.claim("task-a", self.request)
            task = store.get_task("task-a")
            self.assertEqual(task["status"], "ambiguous")
            self.assertFalse(task["recoverable"])

    def test_checkpointed_completion_is_replayed_to_normalized_after_crash_before_append(self):
        with self.store() as store:
            attempt = self.setup_call(store)
            store.write_response(attempt, self.response)
            with patch.object(store, "_append_normalized", side_effect=OSError("simulated crash")):
                with self.assertRaises(OSError):
                    store.finish("task-a", attempt, {"valid": True})
            self.assertEqual(store.normalized_path.read_bytes(), b"")
            self.assertEqual(store.get_task("task-a")["status"], "succeeded")
        with self.store() as store:
            rows = store.normalized_path.read_text().splitlines()
            self.assertEqual(len(rows), 1)
            self.assertEqual(json.loads(rows[0])["attempt_id"], attempt)
            self.assertEqual(store.summary()["attempts"], 1)

    def test_partial_last_normalized_line_is_completed_only_by_appending_suffix(self):
        with self.store() as store:
            attempt = self.setup_call(store)
            store.write_response(attempt, self.response)

            def interrupted_append(payload):
                with store.normalized_path.open("ab") as handle:
                    handle.write(payload[:19])
                    handle.flush()
                    os.fsync(handle.fileno())
                raise OSError("simulated crash mid-append")

            with patch.object(store, "_append_bytes", side_effect=interrupted_append):
                with self.assertRaises(OSError):
                    store.finish("task-a", attempt, {"valid": True})
            prefix = store.normalized_path.read_bytes()
            self.assertEqual(len(prefix), 19)
        with self.store() as store:
            repaired = store.normalized_path.read_bytes()
            self.assertTrue(repaired.startswith(prefix))
            self.assertEqual(json.loads(repaired)["attempt_id"], attempt)
            self.assertTrue(repaired.endswith(b"\n"))

    def test_normalized_tampering_and_uncheckpointed_rows_are_rejected(self):
        with self.store() as store:
            self.complete(store)
            path = store.normalized_path
            original = path.read_bytes()
        path.write_bytes(b"X" + original[1:])
        with self.assertRaises(IntegrityError):
            with self.store():
                pass
        path.write_bytes(original + b"{}\n")
        with self.assertRaises(IntegrityError):
            with self.store():
                pass

    def test_ambiguous_resolution_audit_tampering_is_rejected_on_resume(self):
        with self.store() as store:
            attempt = self.setup_call(store)
            store.mark_ambiguous("task-a", attempt, {"code": "timeout"})
            resolution = store.resolve_ambiguous("task-a", "abandon", "No duplicate")
            path = store.raw_dir / "resolutions" / (resolution + ".json")
        path.write_text("{}\n")
        with self.assertRaises(IntegrityError):
            with self.store():
                pass

    def test_unknown_attempts_foreign_task_ids_and_normalized_identity_conflicts_are_rejected(self):
        with self.store() as store:
            attempt = self.setup_call(store)
            store.write_response(attempt, self.response)
            with self.assertRaises(StorageError):
                store.get_response("unknown")
            with self.assertRaises(ImmutableConflictError):
                store.finish("task-b", attempt, {})
            with self.assertRaises(ImmutableConflictError):
                store.finish("task-a", attempt, {"task_id": "different"})
            with self.assertRaises(ValueError):
                store.finish("task-a", attempt, {}, status="retryable")

    def test_closed_attempt_cannot_gain_a_new_raw_record_after_completion(self):
        with self.store() as store:
            attempt = self.complete(store)
            with self.assertRaises(StorageError):
                store.write_error(attempt, {"code": "late_error"})
            self.assertFalse(store.attempt_path(attempt, "error").exists())
            self.assertEqual(store.get_response(attempt), self.response)

    def test_exhausted_known_rejections_can_finish_permanently_from_retryable(self):
        with self.store() as store:
            attempt = self.setup_call(store)
            store.set_retry("task-a", attempt, self.rejection)
            with self.assertRaises(StorageError):
                store.finish("task-a", attempt, {"valid": True})
            store.finish("task-a", attempt, {"error": "maximum_attempts_exhausted"}, status="permanent_failed")
            self.assertEqual(store.get_task("task-a")["status"], "permanent_failed")
            self.assertEqual(store.summary()["normalized_records"], 1)
            self.assertIsNone(store.claim("task-a", self.request))


if __name__ == "__main__":
    unittest.main()

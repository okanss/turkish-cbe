"""Offline rate-window and immutable cache checks; no inference fixtures are results."""

from copy import deepcopy
from contextlib import closing
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
import sqlite3
import tempfile
import threading
import unittest
from unittest.mock import patch

from inference.runtime import RateLimiter, ResponseCache
from inference.storage import IntegrityError, _publish


class Clock:
    def __init__(self, now=1000.0):
        self.now = now
        self.sleeps = []

    def time(self):
        return self.now

    def sleep(self, seconds):
        if not 0 < seconds <= 60:
            raise AssertionError("rate limiter must make bounded progress")
        self.sleeps.append(seconds)
        self.now += seconds


class RateLimiterTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.path = Path(self.temp.name) / "rates.sqlite"
        self.clock = Clock()

    def limiter(self, *, requests=2, tokens=100):
        return RateLimiter(self.path, {"requests_per_minute": requests,
                                      "tokens_per_minute": tokens},
                           clock=self.clock.time, sleep=self.clock.sleep)

    def test_first_reservations_fit_without_sleep_and_persist(self):
        limiter = self.limiter()
        limiter.reserve(40, 1)
        limiter.reserve(60, 1)
        self.assertEqual(self.clock.sleeps, [])
        with closing(sqlite3.connect(self.path)) as db:
            self.assertEqual(db.execute("SELECT SUM(tokens),SUM(requests) FROM reservations").fetchone(),
                             (100, 2))

    def test_new_instance_shares_request_window(self):
        self.limiter(requests=1).reserve(1, 1)
        self.limiter(requests=1).reserve(1, 1)
        self.assertEqual(self.clock.sleeps, [60.0])
        self.assertEqual(self.clock.now, 1060.0)

    def test_tokens_independently_limit_requests(self):
        limiter = self.limiter(requests=20)
        limiter.reserve(90, 1)
        limiter.reserve(20, 1)
        self.assertEqual(self.clock.sleeps, [60.0])

    def test_auxiliary_requests_and_generation_token_scope_not_double_counted(self):
        limiter = self.limiter(requests=1)
        limiter.reserve(tokens=0, requests=1)
        limiter.reserve(tokens=100, requests=0)
        self.assertEqual(self.clock.sleeps, [])
        with closing(sqlite3.connect(self.path)) as db:
            self.assertEqual(db.execute("SELECT SUM(tokens),SUM(requests) FROM reservations").fetchone(),
                             (100, 1))
        limiter.reserve(tokens=0, requests=1)
        self.assertEqual(self.clock.sleeps, [60.0])

    def test_window_expires_exactly_at_sixty_seconds(self):
        limiter = self.limiter(requests=1)
        limiter.reserve(100, 1)
        self.clock.now += 60
        limiter.reserve(100, 1)
        self.assertEqual(self.clock.sleeps, [])
        with closing(sqlite3.connect(self.path)) as db:
            self.assertEqual(db.execute("SELECT COUNT(*) FROM reservations").fetchone()[0], 1)

    def test_staggered_reservations_wait_only_until_budget_available(self):
        limiter = self.limiter(requests=2)
        limiter.reserve(40, 1)
        self.clock.now += 10
        limiter.reserve(40, 1)
        limiter.reserve(40, 1)
        self.assertEqual(self.clock.sleeps, [50.0])
        self.assertEqual(self.clock.now, 1060.0)

    def test_invalid_or_oversized_single_request_fails_without_waiting(self):
        limiter = self.limiter()
        for tokens, requests in ((101, 1), (0, 3), (-1, 1), (0, -1), (True, 1), (1.0, 1)):
            with self.subTest(tokens=tokens, requests=requests), self.assertRaises(ValueError):
                limiter.reserve(tokens=tokens, requests=requests)
        self.assertEqual(self.clock.sleeps, [])
        with closing(sqlite3.connect(self.path)) as db:
            self.assertEqual(db.execute("SELECT COUNT(*) FROM reservations").fetchone()[0], 0)


class ResponseCacheTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.cache = ResponseCache(self.root / "live")
        self.spec = {"model_id": "fixture", "revision": "a"*40, "seed": 1,
                     "serving": {"chat_template_date": "2026-10-24"}}
        self.key = self.cache.key(self.spec, "messages-sha-fixture")
        self.response = {"text": "YES", "raw": {"fixture_only": True}, "usage": {"input_tokens": 10}}
        self.origin = {"run_id": "live-fixture-protocol", "task_id": "task-1", "attempt_id": "attempt-1"}

    def test_model_version_seed_template_date_and_messages_separate_cache_keys(self):
        for field, value in (("revision", "b"*40), ("seed", 2)):
            spec = deepcopy(self.spec)
            spec[field] = value
            self.assertNotEqual(self.key, self.cache.key(spec, "messages-sha-fixture"))
        spec = deepcopy(self.spec)
        spec["serving"]["chat_template_date"] = "2026-10-25"
        self.assertNotEqual(self.key, self.cache.key(spec, "messages-sha-fixture"))
        self.assertNotEqual(self.key, self.cache.key(self.spec, "other-messages-sha"))

    def test_first_cached_response_is_immutable_and_attributed(self):
        self.assertIsNone(self.cache.get(self.key))
        self.assertTrue(self.cache.put(self.key, self.response, self.origin))
        path = self.cache.root / (self.key+".json")
        frozen = path.read_bytes()
        self.assertFalse(self.cache.put(self.key, {"text": "NO"}, {"run_id": "another-run"}))
        self.assertEqual(path.read_bytes(), frozen)
        result = self.cache.get(self.key)
        self.assertEqual(result["response"], self.response)
        self.assertEqual(result["origin"], self.origin)
        self.assertFalse(result["independent_observation"])
        result["response"]["text"] = "mutated caller copy"
        self.assertEqual(self.cache.get(self.key)["response"]["text"], "YES")

    def test_fixture_and_live_namespace_paths_cannot_reuse_each_other(self):
        fixture = ResponseCache(self.root / "test_fixture")
        fixture.put(self.key, self.response, self.origin)
        self.assertIsNone(self.cache.get(self.key))
        self.assertIsNotNone(fixture.get(self.key))

    def test_response_or_provenance_tamper_is_rejected(self):
        self.cache.put(self.key, self.response, self.origin)
        path = self.cache.root / (self.key+".json")
        original = json.loads(path.read_bytes())
        for mutate in (
                lambda v: v["response"].update(text="NO"),
                lambda v: v["origin"].update(run_id="forged-origin"),
                lambda v: v.update(cache_key="wrong-binding"),
                lambda v: v.update(independent_observation=True)):
            value = deepcopy(original)
            mutate(value)
            path.write_text(json.dumps(value))
            with self.assertRaises(IntegrityError):
                self.cache.get(self.key)

    def test_symlink_cache_entry_is_rejected(self):
        self.cache.put(self.key, self.response, self.origin)
        path = self.cache.root / (self.key+".json")
        moved = path.with_suffix(".target")
        path.rename(moved)
        path.symlink_to(moved)
        with self.assertRaises(IntegrityError):
            self.cache.get(self.key)

    def test_concurrent_publication_keeps_one_complete_attributed_response(self):
        barrier = threading.Barrier(2)
        def simultaneous_publish(path, payload):
            barrier.wait(timeout=5)
            return _publish(path, payload)
        def put(text):
            return self.cache.put(self.key, {"text": text}, {"run_id": "fixture-"+text})
        with patch("inference.runtime._publish", side_effect=simultaneous_publish):
            with ThreadPoolExecutor(max_workers=2) as executor:
                results = list(executor.map(put, ("YES", "NO")))
        self.assertEqual(sorted(results), [False, True])
        frozen = self.cache.get(self.key)
        self.assertIn(frozen["response"]["text"], {"YES", "NO"})
        self.assertEqual(frozen["origin"]["run_id"], "fixture-"+frozen["response"]["text"])


if __name__ == "__main__":
    unittest.main()

"""Parallel execution preserves deterministic source order and exact results."""

import hashlib
import importlib.util
import json
from pathlib import Path
import unittest

from complete_grid.parallel import MAX_WORKERS, _jobs, iter_groups
from experiments.dataset_builder import DEFAULT_PLAN, DatasetPlan, SplitSpec


RESOURCE_DIR = Path(__file__).resolve().parents[1] / "tokenizers"


def _small_plan(sizes=(4,)):
    return DatasetPlan(tuple(SplitSpec(name, seed, sizes, 4) for name, seed in zip(
        ("training", "development", "pilot", "test"), range(401, 405))))


def _hash(value):
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True,
        separators=(",", ":"), default=lambda item: {"bytes": item.hex()}).encode()).hexdigest()


class ParallelJobTests(unittest.TestCase):
    def test_jobs_preserve_split_size_index_and_contiguous_query_ids(self):
        jobs = tuple(_jobs(_small_plan((4, 8))))
        self.assertEqual(len(jobs), 32)
        self.assertEqual([j[0] for j in jobs], list(range(1, 33)))
        next_query = 1
        for ordinal, job in enumerate(jobs):
            group_id, query_start, spec, presentation_seed, n, index = job
            self.assertEqual(query_start, next_query)
            next_query += 3 if n == 4 else 5
            self.assertEqual(group_id, ordinal + 1)
            self.assertEqual(spec["sizes"], (4, 8))
            self.assertEqual(presentation_seed, 431)
            self.assertEqual(index, ordinal % 4)
            self.assertEqual(n, 4 if (ordinal // 4) % 2 == 0 else 8)
        self.assertEqual(next_query, 129)
        self.assertEqual([j[2]["name"] for j in jobs[::8]],
                         ["training", "development", "pilot", "test"])

    def test_default_job_count_matches_3000_groups_and_23200_queries(self):
        jobs = tuple(_jobs(DEFAULT_PLAN))
        self.assertEqual(len(jobs), 3000)
        self.assertEqual(jobs[-1][0], 3000)
        self.assertEqual(jobs[-1][1], 23200 - 12 + 1)
        self.assertEqual(jobs[-1][2]["name"], "test")
        self.assertEqual(jobs[-1][4], 1024)
        self.assertEqual(jobs[-1][5], 199)

    def test_invalid_worker_counts_and_plans_rejected_before_loading_tokenizer(self):
        self.assertEqual(MAX_WORKERS, 8)
        for workers in (True, False, 0, -1, 9, 1.5, "2", None):
            with self.subTest(workers=workers), self.assertRaises(ValueError):
                list(iter_groups(_small_plan(), Path("missing-resource"), workers))
        for plan in (None, {}, "plan"):
            with self.subTest(plan=plan), self.assertRaises(ValueError):
                list(iter_groups(plan, Path("missing-resource"), 1))


@unittest.skipUnless(importlib.util.find_spec("tiktoken") is not None,
                     "optional pinned tiktoken dependency is not installed")
class RealParallelTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.serial = list(iter_groups(_small_plan(), RESOURCE_DIR, workers=1))
        cls.parallel = list(iter_groups(_small_plan(), RESOURCE_DIR, workers=2))

    def test_one_and_two_workers_produce_identical_group_result_hashes(self):
        self.assertEqual([_hash(row) for row in self.serial],
                         [_hash(row) for row in self.parallel])
        self.assertEqual(len(self.parallel), 16)
        self.assertEqual([row["group_id"] for row in self.parallel], list(range(1, 17)))

    def test_query_and_measurement_foreign_keys_and_design_order_are_complete(self):
        query_ids, measurements = [], 0
        for result in self.parallel:
            group_id = result["group_id"]
            self.assertEqual(result["group_row"][:5], (group_id, result["ordering_id"],
                result["split"], result["n"], result["index"]))
            self.assertEqual(json.loads(result["group_row"][5]), result["source"])
            query_rows = result["query_rows"]
            group_queries = [row[0] for row in query_rows]
            query_ids.extend(group_queries)
            self.assertEqual(len(query_rows), 3)
            self.assertEqual({row[1] for row in query_rows}, {group_id})
            self.assertEqual([row[3] for row in query_rows],
                             ["endpoint", "internal", "distance_controlled"])
            self.assertEqual(len(result["component_rows"]), 243)
            self.assertEqual([(row[0], row[1]) for row in result["component_rows"]],
                             [(group_id, design_id) for design_id in range(1, 244)])
            self.assertEqual(len(result["logical_digests"]), 243)
            self.assertTrue(all(type(value) is bytes and len(value) == 32
                                for value in result["logical_digests"]))
            rows = result["measurement_rows"]
            self.assertEqual(len(rows), 729)
            self.assertEqual([(row[0], row[1], row[2]) for row in rows],
                [(query_id, design_id, group_id) for design_id in range(1, 244)
                 for query_id in group_queries])
            self.assertTrue(all(type(row[3]) is bytes and len(row[3]) == 32 for row in rows))
            self.assertTrue(all(row[4] > 0 and row[5] > 0 for row in rows))
            self.assertEqual(result["views"], sorted(set(result["views"])))
            measurements += len(rows)
        self.assertEqual(query_ids, list(range(1, 49)))
        self.assertEqual(measurements, 11664)


if __name__ == "__main__":
    unittest.main()

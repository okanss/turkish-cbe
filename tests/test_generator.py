"""Deterministic latent-order generation, experimental balance and contracts."""

from collections import Counter
from copy import deepcopy
from dataclasses import replace
import json
import os
from pathlib import Path
import random
import subprocess
import sys
import unittest

from data_generation.entities import ENTITY_TYPES
from data_generation.generator import (
    GenerationPlan,
    generate_record,
    iter_block,
    valid_distances,
)
from data_generation.oracle import DataValidationError, answer_from_edges, answer_from_order
from data_generation.validation import validate_record


REQUIRED_FIELDS = {
    "instance_id", "seed", "n", "entity_type", "latent_order", "premise_edges",
    "query_subject", "query_object", "query_relation", "query_type",
    "query_distance", "gold_label", "generator_version",
}
CELLS = {(direction, gold) for direction in ("forward", "reverse") for gold in (True, False)}


def canonical_bytes(records):
    """The explicit serialization used for byte-level repeatability checks."""
    return ("\n".join(json.dumps(record, ensure_ascii=False, sort_keys=True,
                                separators=(",", ":")) for record in records) + "\n").encode("utf-8")


class GeneratorTests(unittest.TestCase):
    def setUp(self):
        self.plan = GenerationPlan(seed=11, sizes=(4, 8, 16, 32, 64, 128, 256, 512, 1024),
                                   examples_per_size=200, entity_types=ENTITY_TYPES,
                                   query_types=("endpoint", "internal", "distance_controlled"),
                                   distances=(1, 2, 4, 8, 16, 32, 64, 128, 256, 512))

    def test_default_plan_is_valid(self):
        self.plan.validate()

    def test_invalid_plans_are_rejected(self):
        cases = (
            {"seed": True}, {"seed": "11"},
            {"sizes": ()}, {"sizes": (3,)}, {"sizes": (1025,)},
            {"sizes": (True,)}, {"sizes": (4.0,)},
            {"sizes": (8, 4)}, {"sizes": (4, 4)},
            {"examples_per_size": True}, {"examples_per_size": 0},
            {"examples_per_size": -4}, {"examples_per_size": 6},
            {"examples_per_size": 4.0},
            {"entity_types": ()}, {"entity_types": ("unknown",)},
            {"entity_types": (ENTITY_TYPES[0], ENTITY_TYPES[0])},
            {"query_types": ()}, {"query_types": ("unknown",)},
            {"query_types": ("endpoint", "endpoint")},
            {"distances": (0,)}, {"distances": (-1,)},
            {"distances": (True,)}, {"distances": (1.0,)},
            {"distances": (1, 1)},
        )
        for changes in cases:
            with self.subTest(changes=changes), self.assertRaises(ValueError):
                replace(self.plan, **changes).validate()

    def test_distance_filter_uses_strictly_interior_positions(self):
        self.assertEqual(valid_distances(4, self.plan.distances), (1,))
        self.assertEqual(valid_distances(8, self.plan.distances), (1, 2, 4))
        self.assertEqual(valid_distances(16, self.plan.distances), (1, 2, 4, 8))
        self.assertEqual(valid_distances(1024, self.plan.distances), self.plan.distances)

    def test_non_power_of_two_distances_and_largest_interior_distance(self):
        distances = (1, 3, 7, 1021, 1022)
        self.assertEqual(valid_distances(4, distances), (1,))
        self.assertEqual(valid_distances(8, distances), (1, 3))
        self.assertEqual(valid_distances(16, distances), (1, 3, 7))
        self.assertEqual(valid_distances(1024, distances), (1, 3, 7, 1021))
        plan = replace(self.plan, distances=distances)
        plan.validate()
        record = generate_record(plan, 1024, 0, "distance_controlled", distance=1021)
        validate_record(record)
        positions = {entity: index for index, entity in enumerate(record["latent_order"])}
        pair = {positions[record["query_subject"]], positions[record["query_object"]]}
        self.assertEqual(pair, {1, 1022})
        self.assertEqual(record["query_distance"], 1021)

    def test_each_arm_has_exact_four_cell_and_entity_stratified_balance(self):
        for n, query_type, distance in (
            (4, "endpoint", None),
            (8, "internal", None),
            (16, "distance_controlled", 4),
        ):
            with self.subTest(n=n, query_type=query_type, distance=distance):
                records = list(iter_block(self.plan, n, query_type, distance=distance))
                self.assertEqual(len(records), 200)
                self.assertEqual(len({record["instance_id"] for record in records}), 200)
                cells = Counter((record["query_direction"], record["gold_label"]) for record in records)
                self.assertEqual(cells, Counter({cell: 50 for cell in CELLS}))
                self.assertEqual(Counter(record["gold_label"] for record in records),
                                 Counter({True: 100, False: 100}))
                groups = Counter(record["entity_type"] for record in records)
                self.assertEqual(sorted(groups.values()), [64, 68, 68])
                self.assertEqual(set(groups), set(ENTITY_TYPES))
                for entity_type, count in groups.items():
                    group_cells = Counter((record["query_direction"], record["gold_label"])
                                          for record in records if record["entity_type"] == entity_type)
                    self.assertEqual(group_cells, Counter({cell: count // 4 for cell in CELLS}))
                for record in records:
                    validate_record(record)

    def test_record_fields_order_and_exact_oracles_agree(self):
        for index in range(12):
            with self.subTest(index=index):
                record = generate_record(self.plan, 16, index, "endpoint")
                self.assertTrue(REQUIRED_FIELDS <= record.keys())
                self.assertEqual(record["generator_version"], "0.2.0")
                self.assertEqual(record["seed"], self.plan.seed)
                self.assertEqual(record["n"], 16)
                self.assertEqual(record["sample_index"], index)
                self.assertIsInstance(record["gold_label"], bool)
                order = record["latent_order"]
                self.assertEqual(len(order), 16)
                self.assertEqual(len(set(order)), 16)
                self.assertEqual(len(record["premise_edges"]), 15)
                expected_edges = {tuple(pair) for pair in zip(order, order[1:])}
                self.assertEqual({tuple(edge) for edge in record["premise_edges"]}, expected_edges)
                subject, object = record["query_subject"], record["query_object"]
                self.assertEqual({subject, object}, {order[0], order[-1]})
                args = (subject, object, record["query_relation"])
                self.assertEqual(answer_from_order(order, *args), record["gold_label"])
                self.assertEqual(answer_from_edges(list(reversed(order)), record["premise_edges"], *args),
                                 record["gold_label"])
                self.assertEqual(record["query_distance"], 15)
                validate_record(record)

    def test_direction_is_subject_position_and_is_independent_of_truth(self):
        cells = set()
        for index in range(4):
            record = generate_record(self.plan, 8, index, "endpoint")
            order = record["latent_order"]
            forward = order.index(record["query_subject"]) < order.index(record["query_object"])
            self.assertEqual(record["query_direction"], "forward" if forward else "reverse")
            expected = forward if record["query_relation"] == "before" else not forward
            self.assertEqual(record["gold_label"], expected)
            cells.add((record["query_direction"], record["gold_label"]))
        self.assertEqual(cells, CELLS)

    def test_query_arms_share_base_order_but_keep_distinct_instance_ids(self):
        for index in range(8):
            records = [
                generate_record(self.plan, 16, index, "endpoint"),
                generate_record(self.plan, 16, index, "internal"),
                generate_record(self.plan, 16, index, "distance_controlled", distance=1),
                generate_record(self.plan, 16, index, "distance_controlled", distance=8),
            ]
            with self.subTest(index=index):
                self.assertEqual(len({record["ordering_id"] for record in records}), 1)
                self.assertEqual(len({record["instance_id"] for record in records}), 4)
                for record in records[1:]:
                    self.assertEqual(record["latent_order"], records[0]["latent_order"])
                    self.assertEqual(record["premise_edges"], records[0]["premise_edges"])
                    self.assertEqual(record["entity_type"], records[0]["entity_type"])

    def test_internal_queries_exclude_both_endpoints(self):
        for n in self.plan.sizes:
            for index in range(4):
                with self.subTest(n=n, index=index):
                    record = generate_record(self.plan, n, index, "internal")
                    positions = {entity: i for i, entity in enumerate(record["latent_order"])}
                    i, j = positions[record["query_subject"]], positions[record["query_object"]]
                    self.assertTrue(1 <= i < n - 1)
                    self.assertTrue(1 <= j < n - 1)
                    self.assertNotEqual(i, j)
                    self.assertEqual(record["query_distance"], abs(i - j))
                    validate_record(record)

    def test_every_requested_valid_distance_is_realized(self):
        for n in self.plan.sizes:
            for distance in valid_distances(n, self.plan.distances):
                for index in range(4):
                    with self.subTest(n=n, distance=distance, index=index):
                        record = generate_record(self.plan, n, index, "distance_controlled", distance=distance)
                        positions = {entity: i for i, entity in enumerate(record["latent_order"])}
                        i, j = positions[record["query_subject"]], positions[record["query_object"]]
                        self.assertTrue(1 <= i < n - 1)
                        self.assertTrue(1 <= j < n - 1)
                        self.assertEqual(abs(i - j), distance)
                        self.assertEqual(record["query_distance"], distance)
                        validate_record(record)

    def test_same_seed_repeats_records_and_serialized_bytes(self):
        first = list(iter_block(self.plan, 8, "endpoint"))
        second = list(iter_block(self.plan, 8, "endpoint"))
        self.assertEqual(first, second)
        self.assertEqual(canonical_bytes(first), canonical_bytes(second))
        self.assertEqual({record["sample_index"] for record in first}, set(range(200)))
        # A shuffle is deterministic but must not disclose the four-cell cycle.
        self.assertNotEqual([record["sample_index"] for record in first], list(range(200)))

    def test_other_seed_changes_ordering_and_identifiers(self):
        first = generate_record(self.plan, 32, 0, "endpoint")
        second = generate_record(replace(self.plan, seed=12), 32, 0, "endpoint")
        self.assertNotEqual(first["latent_order"], second["latent_order"])
        self.assertNotEqual(first["ordering_id"], second["ordering_id"])
        self.assertNotEqual(first["instance_id"], second["instance_id"])

    def test_generation_does_not_consume_global_rng(self):
        state = random.getstate()
        generate_record(self.plan, 16, 3, "internal")
        list(iter_block(replace(self.plan, examples_per_size=4), 8, "endpoint"))
        self.assertEqual(random.getstate(), state)

    def test_hash_randomization_and_fresh_processes_do_not_change_output(self):
        source = (
            "import json\n"
            "from data_generation.generator import GenerationPlan, iter_block\n"
            "plan = GenerationPlan(seed=11, sizes=(8,), examples_per_size=8)\n"
            "print(json.dumps(list(iter_block(plan,8,'endpoint')),ensure_ascii=False,sort_keys=True))\n"
        )
        src = str(Path(__file__).resolve().parents[1] / "src")
        outputs = []
        for hash_seed in ("0", "171"):
            env = os.environ.copy()
            env["PYTHONHASHSEED"] = hash_seed
            env["PYTHONPATH"] = src + os.pathsep + env.get("PYTHONPATH", "")
            result = subprocess.run([sys.executable, "-c", source], env=env, capture_output=True,
                                    check=True, timeout=30)
            outputs.append(result.stdout)
        self.assertEqual(outputs[0], outputs[1])
        self.assertEqual(len(json.loads(outputs[0])), 8)

    def test_maximum_size_all_entity_groups_validate(self):
        for entity_type in ENTITY_TYPES:
            plan = replace(self.plan, entity_types=(entity_type,))
            record = generate_record(plan, 1024, 0, "endpoint")
            with self.subTest(entity_type=entity_type):
                self.assertEqual(len(set(record["latent_order"])), 1024)
                self.assertEqual(len(record["premise_edges"]), 1023)
                self.assertEqual(record["query_distance"], 1023)
                validate_record(record)

    def test_record_validation_rejects_semantic_corruption(self):
        valid = generate_record(self.plan, 8, 0, "endpoint")
        cases = []
        changed = deepcopy(valid)
        changed["gold_label"] = not valid["gold_label"]
        cases.append(("wrong oracle label", changed))
        changed = deepcopy(valid)
        changed["query_distance"] -= 1
        cases.append(("wrong query distance", changed))
        changed = deepcopy(valid)
        changed["query_subject"] = "NOT_AN_ENTITY"
        cases.append(("unknown query entity", changed))
        changed = deepcopy(valid)
        changed["premise_edges"][0] = list(reversed(changed["premise_edges"][0]))
        cases.append(("contradictory premise", changed))
        changed = deepcopy(valid)
        changed["premise_edges"].append([changed["latent_order"][-1], changed["latent_order"][0]])
        cases.append(("cycle", changed))
        changed = deepcopy(valid)
        changed["latent_order"][1] = changed["latent_order"][0]
        cases.append(("duplicate entity", changed))
        for reason, changed in cases:
            with self.subTest(reason=reason), self.assertRaises(DataValidationError):
                validate_record(changed)

    def test_record_validation_enforces_query_arm_boundaries(self):
        for query_type in ("internal", "distance_controlled"):
            record = generate_record(self.plan, 8, 0, "endpoint")
            record["query_type"] = query_type
            with self.subTest(query_type=query_type), self.assertRaises(DataValidationError):
                validate_record(record)


if __name__ == "__main__":
    unittest.main()

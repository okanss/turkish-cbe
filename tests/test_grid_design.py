"""Expanded-factor coverage, independent count arithmetic and pairing guards."""

import copy
from collections import Counter
import unittest

from data_generation.generator import GenerationPlan, generate_record
from experiments.dataset_builder import (DEFAULT_PLAN, DatasetPlan, SplitSpec,
    digest, presentation_plans)
from complete_grid.grid_design import (GRID_VERSION, NAME_CONTROLS,
    PROMPT_CONDITIONS, RELATION_IDS, comparison_designs, expected_counts,
    iter_designs, validate_grid_pair)


def _record(query_type="endpoint", distance=None):
    return generate_record(GenerationPlan(seed=19, sizes=(8,), examples_per_size=4,
                           entity_types=("turkish_names",)), 8, 0, query_type, distance)


def _metadata(record, language, condition, control="native_shared", relation="spatial_front"):
    return {
        "latent_instance_id": record["ordering_id"],
        "query_instance_id": record["instance_id"], "split": "test", "n": record["n"],
        "query_type": record["query_type"], "query_distance": record["query_distance"],
        "query_subject": record["query_subject"], "query_object": record["query_object"],
        "query_relation": record["query_relation"], "query_direction": record["query_direction"],
        "gold_label": record["gold_label"], "language": language, "condition": condition,
        "name_control": control, "relation_id": relation,
        "presentation_plan_sha256": digest(presentation_plans(record, 431)[condition]),
    }


class GridDesignTests(unittest.TestCase):
    def test_complete_factorial_has_exactly_243_unique_designs_per_query(self):
        designs = tuple(iter_designs())
        self.assertEqual(GRID_VERSION, "0.4.1")
        self.assertEqual(len(designs), 243)
        self.assertEqual(len(set(designs)), 243)
        self.assertEqual(tuple(iter_designs()), designs)
        self.assertEqual(len(RELATION_IDS), 9)
        self.assertEqual(len(NAME_CONTROLS), 3)
        for control in NAME_CONTROLS:
            for relation in RELATION_IDS:
                observed = {(lang, condition) for c, r, lang, condition in designs
                            if (c, r) == (control, relation)}
                self.assertEqual(observed, set(PROMPT_CONDITIONS))
        self.assertEqual(designs[0], ("native_shared", "spatial_front", "en", "canonical"))
        self.assertEqual(designs[-1], ("symbolic_shared", "ranking", "tr", "P5"))

    def test_each_query_gets_all_same_query_contrasts_and_interiors_get_f(self):
        for query_type in ("endpoint", "internal", "distance_controlled"):
            designs = tuple(comparison_designs(query_type))
            self.assertEqual(len(designs), 9 if query_type == "endpoint" else 13)
            counts = Counter(arm for arm, _, _ in designs)
            self.assertEqual({k: counts[k] for k in "ABCDE"},
                             {"A": 1, "B": 1, "C": 1, "D": 1, "E": 5})
            self.assertEqual(counts["F"], 0 if query_type == "endpoint" else 4)
            self.assertEqual(len(set(designs)), len(designs))
            for _, left, right in designs:
                self.assertIn(left, PROMPT_CONDITIONS)
                self.assertIn(right, PROMPT_CONDITIONS)
        for invalid in (None, False, [], "random"):
            with self.subTest(invalid=invalid), self.assertRaises(ValueError):
                tuple(comparison_designs(invalid))

    def test_default_counts_match_independent_fixed_totals(self):
        counts = expected_counts()
        expected = {
            "training": (450, 3550, 862650, 1197450),
            "development": (450, 3550, 862650, 1197450),
            "pilot": (300, 1900, 461700, 634500),
            "test": (1800, 14200, 3450600, 4789800),
        }
        for split, values in expected.items():
            with self.subTest(split=split):
                row = counts["splits"][split]
                self.assertEqual(tuple(row[k] for k in
                    ("orders", "queries", "prompts", "comparison_cells")), values)
                self.assertEqual(sum(row["cells_by_arm"].values()), row["comparison_cells"])
                self.assertEqual(sum(row["query_blocks"].values()), row["queries"])
        self.assertEqual(counts["totals"], {
            "orders": 3000, "queries": 23200, "prompts": 5637600,
            "comparison_cells": 7819200,
            "cells_by_arm": {"A": 626400, "B": 626400, "C": 626400,
                             "D": 626400, "E": 3132000, "F": 2181600}})
        self.assertEqual(expected_counts(DEFAULT_PLAN), counts)

    def test_query_answer_cells_remain_exact_or_approximately_balanced(self):
        result = expected_counts()
        for name, split in result["splits"].items():
            for block, counts in split["query_answer_cells"].items():
                with self.subTest(split=name, block=block):
                    self.assertEqual(set(counts), {"forward|True", "forward|False",
                                                  "reverse|True", "reverse|False"})
                    if name == "test":
                        self.assertEqual(set(counts.values()), {50})
                    else:
                        self.assertEqual(sorted(counts.values()), [12, 12, 13, 13])
                    self.assertEqual(counts["forward|True"] + counts["forward|False"],
                                     counts["reverse|True"] + counts["reverse|False"])
                    self.assertEqual(counts["forward|True"] + counts["reverse|True"],
                                     counts["forward|False"] + counts["reverse|False"])

    def test_small_plan_counts_match_explicit_design_enumeration(self):
        plan = DatasetPlan(tuple(SplitSpec(name, seed, (4, 8), 4) for name, seed in
                           zip(("training", "development", "pilot", "test"), range(41, 45))))
        expected = expected_counts(plan)
        # Known distance grid: n=4 has d=1; n=8 has d=1,2,4.
        query_types = ["endpoint", "internal", "distance_controlled"] * 4
        query_types += ["endpoint", "internal", "distance_controlled",
                        "distance_controlled", "distance_controlled"] * 4
        prompt_count, cell_count = 0, Counter()
        for query_type in query_types:
            prompt_count += sum(1 for _ in iter_designs())
            for _control in NAME_CONTROLS:
                for _relation in RELATION_IDS:
                    cell_count.update(arm for arm, _, _ in comparison_designs(query_type))
        for summary in expected["splits"].values():
            self.assertEqual(summary["orders"], 8)
            self.assertEqual(summary["queries"], len(query_types))
            self.assertEqual(summary["prompts"], prompt_count)
            self.assertEqual(summary["cells_by_arm"], cell_count)
        for invalid in (None, {}, "plan"):
            with self.subTest(invalid=invalid), self.assertRaises(ValueError):
                expected_counts(invalid)

    def test_all_comparisons_validate_real_query_metadata_for_all_factors(self):
        endpoint = _record()
        for query_type, distance in (("endpoint", None), ("internal", None),
                                      ("distance_controlled", 2)):
            source = endpoint if query_type == "endpoint" else _record(query_type, distance)
            for control in NAME_CONTROLS:
                for relation in RELATION_IDS:
                    for arm, left_design, right_design in comparison_designs(query_type):
                        left_source = endpoint if arm == "F" else source
                        left = _metadata(left_source, *left_design, control, relation)
                        right = _metadata(source, *right_design, control, relation)
                        with self.subTest(query=query_type, control=control, relation=relation,
                                          arm=arm, conditions=(left_design, right_design)):
                            before = copy.deepcopy((left, right))
                            self.assertIsNone(validate_grid_pair(left, right, arm))
                            self.assertEqual((left, right), before)

    def test_a_through_e_reject_query_or_factor_mismatch(self):
        source = _record("internal")
        for arm, a, b in comparison_designs("endpoint"):
            for key, value in {
                "latent_instance_id": "different-order", "query_instance_id": "different-query",
                "split": "training", "n": 16, "query_subject": "different-subject",
                "query_object": "different-object", "query_distance": 1,
                "name_control": "symbolic_shared", "relation_id": "ranking",
            }.items():
                left, right = _metadata(source, *a), _metadata(source, *b)
                if right[key] == value:
                    value = 2 if key == "query_distance" else "different"
                right[key] = value
                with self.subTest(arm=arm, key=key), self.assertRaises(ValueError):
                    validate_grid_pair(left, right, arm)

    def test_cross_language_pairs_require_identical_presentation_plan(self):
        source = _record()
        for arm, a, b in comparison_designs("endpoint"):
            left, right = _metadata(source, *a), _metadata(source, *b)
            right["presentation_plan_sha256"] = "f" * 64
            if arm in {"C", "D"}:
                with self.subTest(arm=arm), self.assertRaises(ValueError):
                    validate_grid_pair(left, right, arm)
            else:
                validate_grid_pair(left, right, arm)

    def test_f_allows_different_query_gold_but_rejects_different_common_context(self):
        endpoint, interior = _record(), _record("distance_controlled", 2)
        left = _metadata(endpoint, "tr", "equivalent", "symbolic_shared", "ranking")
        right = _metadata(interior, "tr", "equivalent", "symbolic_shared", "ranking")
        right.update(query_relation="after", query_direction="forward", gold_label=False)
        self.assertIsNone(validate_grid_pair(left, right, "F"))
        for key, value in {"latent_instance_id": "another-order", "split": "pilot",
                           "language": "en", "condition": "canonical",
                           "name_control": "native_shared", "relation_id": "spatial_front",
                           "presentation_plan_sha256": "f" * 64,
                           "query_instance_id": left["query_instance_id"]}.items():
            changed = dict(right, **{key: value})
            with self.subTest(key=key), self.assertRaises(ValueError):
                validate_grid_pair(left, changed, "F")
        left["query_type"] = "internal"
        left["query_distance"] = 2
        with self.assertRaises(ValueError):
            validate_grid_pair(left, right, "F")

    def test_required_metadata_types_and_unknown_factor_values_are_rejected(self):
        source = _record()
        for key in _metadata(source, "en", "canonical"):
            left, right = _metadata(source, "en", "canonical"), _metadata(source, "tr", "canonical")
            del right[key]
            with self.subTest(missing=key), self.assertRaises(ValueError):
                validate_grid_pair(left, right, "C")
        invalids = {
            "n": (True, 3, 1025, "8"), "query_distance": (False, 0, 8, 2, "7"),
            "gold_label": (1, "YES", None, False), "split": ("frozen",),
            "language": ("de",), "condition": ("P6",), "query_type": ("random",),
            "query_relation": ("left",), "query_direction": ("sideways",),
            "name_control": ("arbitrary",), "relation_id": ("comparison_age",),
            "query_subject": ("", None, 1), "presentation_plan_sha256": ("hash", "z" * 64),
        }
        for key, values in invalids.items():
            for value in values:
                left, right = _metadata(source, "en", "canonical"), _metadata(source, "tr", "canonical")
                right[key] = value
                with self.subTest(key=key, value=value), self.assertRaises(ValueError):
                    validate_grid_pair(left, right, "C")
        for invalid in (None, [], 1, "G"):
            with self.subTest(arm=invalid), self.assertRaises(ValueError):
                validate_grid_pair(_metadata(source, "en", "canonical"),
                                   _metadata(source, "tr", "canonical"), invalid)


if __name__ == "__main__":
    unittest.main()

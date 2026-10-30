"""Pair validity follows the intended contrast rather than gold coincidence."""

import copy
import unittest

from experiments.pairing import validate_pair


def _base():
    return {
        "latent_instance_id": "latent-1", "query_instance_id": "query-1",
        "split": "test", "n": 8, "query_type": "endpoint", "query_distance": 7,
        "query_subject": "entity-A", "query_object": "entity-B",
        "query_relation": "before", "query_direction": "forward", "gold_label": True,
        "language": "en", "condition": "canonical", "name_control": "native_shared",
        "relation_id": "spatial_front", "presentation_plan_sha256": "a" * 64,
    }


def _pair(arm):
    left = _base()
    right = copy.deepcopy(left)
    if arm == "A":
        right.update(condition="equivalent", presentation_plan_sha256="b" * 64)
    elif arm == "B":
        left["language"] = "tr"
        right.update(language="tr", condition="equivalent", presentation_plan_sha256="b" * 64)
    elif arm == "C":
        right["language"] = "tr"
    elif arm == "D":
        left["condition"] = "equivalent"
        right.update(language="tr", condition="equivalent")
    elif arm == "E":
        left["language"] = "tr"
        right.update(language="tr", condition="P3", presentation_plan_sha256="b" * 64)
    elif arm == "F":
        right.update(query_instance_id="query-2", query_type="internal", query_distance=2,
                     query_subject="entity-C", query_object="entity-D", query_relation="after",
                     query_direction="reverse", gold_label=False)
    return left, right


class PairingTests(unittest.TestCase):
    def test_all_six_intended_contrasts_are_accepted(self):
        for arm in "ABCDEF":
            with self.subTest(arm=arm):
                self.assertIsNone(validate_pair(*_pair(arm), arm))

    def test_f_allows_different_gold_targets_direction_relation_and_distance(self):
        left, right = _pair("F")
        validate_pair(left, right, "F")
        right["query_type"] = "distance_controlled"
        validate_pair(left, right, "F")
        left["condition"] = right["condition"] = "equivalent"
        left["language"] = right["language"] = "tr"
        validate_pair(left, right, "F")

    def test_a_through_e_reject_each_mismatched_query_or_group_field(self):
        changes = {
            "latent_instance_id": "latent-2", "query_instance_id": "query-2",
            "split": "training", "n": 16, "query_type": "internal", "query_distance": 2,
            "query_subject": "entity-C", "query_object": "entity-D",
            "query_relation": "after", "query_direction": "reverse", "gold_label": False,
            "name_control": "symbolic_shared", "relation_id": "comparison_height",
        }
        for arm in "ABCDE":
            for field, value in changes.items():
                left, right = _pair(arm)
                right[field] = value
                with self.subTest(arm=arm, field=field), self.assertRaises(ValueError):
                    validate_pair(left, right, arm)

    def test_c_and_d_require_the_identical_presentation_plan(self):
        for arm in "CD":
            left, right = _pair(arm)
            right["presentation_plan_sha256"] = "c" * 64
            with self.subTest(arm=arm), self.assertRaises(ValueError):
                validate_pair(left, right, arm)

    def test_a_b_e_allow_presentation_plans_to_differ(self):
        for arm in "ABE":
            with self.subTest(arm=arm):
                validate_pair(*_pair(arm), arm)

    def test_language_and_condition_transitions_are_not_interchangeable(self):
        for arm in "ABCDE":
            for field, value in (("language", "en" if arm != "A" else "tr"),
                                 ("condition", "P6")):
                left, right = _pair(arm)
                right[field] = value
                with self.subTest(arm=arm, field=field), self.assertRaises(ValueError):
                    validate_pair(left, right, arm)

    def test_e_only_allows_native_turkish_endpoint_ablations(self):
        for condition in ("P1", "P2", "P3", "P4", "P5"):
            left, right = _pair("E")
            right["condition"] = condition
            validate_pair(left, right, "E")
        for updates in ({"name_control": "symbolic_shared"}, {"query_type": "internal"}):
            left, right = _pair("E")
            left.update(updates)
            right.update(updates)
            with self.subTest(updates=updates), self.assertRaises(ValueError):
                validate_pair(left, right, "E")

    def test_f_rejects_changes_to_every_common_field(self):
        changes = {
            "latent_instance_id": "latent-2", "split": "training", "n": 16,
            "language": "tr", "condition": "equivalent", "name_control": "symbolic_shared",
            "relation_id": "comparison_height", "presentation_plan_sha256": "c" * 64,
        }
        for field, value in changes.items():
            left, right = _pair("F")
            right[field] = value
            with self.subTest(field=field), self.assertRaises(ValueError):
                validate_pair(left, right, "F")

    def test_f_requires_distinct_queries_and_endpoint_to_interior_contrast(self):
        for updates in ({"query_instance_id": "query-1"}, {"query_type": "endpoint"}):
            left, right = _pair("F")
            right.update(updates)
            with self.subTest(updates=updates), self.assertRaises(ValueError):
                validate_pair(left, right, "F")
        left, right = _pair("F")
        left["condition"] = right["condition"] = "P3"
        with self.assertRaises(ValueError):
            validate_pair(left, right, "F")
        left, right = _pair("F")
        left["query_type"] = "internal"
        with self.assertRaises(ValueError):
            validate_pair(left, right, "F")

    def test_required_fields_are_required_on_both_sides(self):
        for side in (0, 1):
            for field in _base():
                pair = list(_pair("C"))
                del pair[side][field]
                with self.subTest(side=side, field=field), self.assertRaises(ValueError):
                    validate_pair(*pair, "C")

    def test_invalid_types_and_unknown_labels_are_rejected(self):
        changes = {
            "n": (True, 3, 1025, "8"), "query_distance": (False, 0, 8, "7"),
            "gold_label": (1, "YES", None), "query_subject": ("", None, 5),
            "language": ("de",), "query_relation": ("left",),
            "query_direction": ("sideways",), "query_type": ("random",),
            "presentation_plan_sha256": (" ", 3),
        }
        for field, values in changes.items():
            for value in values:
                left, right = _pair("C")
                right[field] = value
                with self.subTest(field=field, value=value), self.assertRaises(ValueError):
                    validate_pair(left, right, "C")
        for invalid in (None, "G", "", [], 4):
            with self.subTest(arm=invalid), self.assertRaises(ValueError):
                validate_pair(*_pair("C"), invalid)

    def test_distinct_entities_and_dictionary_inputs_are_required(self):
        left, right = _pair("C")
        right["query_object"] = right["query_subject"]
        with self.assertRaises(ValueError):
            validate_pair(left, right, "C")
        for invalid in (None, [], "metadata"):
            with self.subTest(invalid=invalid), self.assertRaises(ValueError):
                validate_pair(invalid, right, "C")
            with self.subTest(invalid=invalid), self.assertRaises(ValueError):
                validate_pair(left, invalid, "C")

    def test_full_prompt_records_can_include_extra_provenance(self):
        left, right = _pair("C")
        left.update(prompt="English text", prompt_sha256="c" * 64)
        right.update(prompt="Türkçe metin", source_record_sha256="d" * 64)
        before = copy.deepcopy((left, right))
        validate_pair(left, right, "C")
        self.assertEqual((left, right), before)


if __name__ == "__main__":
    unittest.main()

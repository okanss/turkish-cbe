"""English fixtures, semantic inverses, and adversarial closed-grammar checks."""

import unittest

from renderers import english
from renderers.turkish import RELATIONS as TURKISH_RELATIONS


class EnglishRendererTests(unittest.TestCase):
    def setUp(self):
        self.names = ["Ayşe Yılmaz", "Bora Çelik", "İdil Şahin", "E0004"]

    def test_relation_inventory_matches_turkish(self):
        self.assertEqual(set(english.RELATIONS), set(TURKISH_RELATIONS))
        self.assertEqual(english.ENGLISH_RENDERER_VERSION, "0.4.0")

    def test_independent_statement_fixtures(self):
        fixtures = {
            "spatial_front": ("Ayşe Yılmaz is in front of Bora Çelik.",
                              "Bora Çelik is behind Ayşe Yılmaz."),
            "spatial_right": ("Ayşe Yılmaz is to the right of Bora Çelik.",
                              "Bora Çelik is to the left of Ayşe Yılmaz."),
            "spatial_above": ("Ayşe Yılmaz is above Bora Çelik.",
                              "Bora Çelik is below Ayşe Yılmaz."),
            "temporal_arrival": ("Ayşe Yılmaz arrived before Bora Çelik.",
                                 "Bora Çelik arrived after Ayşe Yılmaz."),
            "temporal_completion": ("Task Ayşe Yılmaz finished before task Bora Çelik.",
                                    "Task Bora Çelik finished after task Ayşe Yılmaz."),
            "comparison_height": ("Ayşe Yılmaz is taller than Bora Çelik.",
                                  "Bora Çelik is shorter than Ayşe Yılmaz."),
            "comparison_speed": ("Ayşe Yılmaz is faster than Bora Çelik.",
                                 "Bora Çelik is slower than Ayşe Yılmaz."),
            "comparison_weight": ("Ayşe Yılmaz is heavier than Bora Çelik.",
                                  "Bora Çelik is lighter than Ayşe Yılmaz."),
            "ranking": ("Ayşe Yılmaz is ranked higher than Bora Çelik.",
                        "Bora Çelik is ranked lower than Ayşe Yılmaz."),
        }
        for relation, (forward, converse) in fixtures.items():
            with self.subTest(relation=relation):
                self.assertEqual(english.statement(self.names[0], self.names[1], relation), forward)
                self.assertEqual(english.statement(self.names[1], self.names[0], relation, "after"), converse)
                self.assertEqual(english.normalize_statement(forward, self.names, relation), tuple(self.names[:2]))
                self.assertEqual(english.normalize_statement(converse, self.names, relation), tuple(self.names[:2]))

    def test_independent_query_fixtures(self):
        fixtures = {
            "spatial_front": "Is Bora Çelik behind Ayşe Yılmaz?",
            "spatial_right": "Is Bora Çelik to the left of Ayşe Yılmaz?",
            "spatial_above": "Is Bora Çelik below Ayşe Yılmaz?",
            "temporal_arrival": "Did Bora Çelik arrive after Ayşe Yılmaz?",
            "temporal_completion": "Did task Bora Çelik finish after task Ayşe Yılmaz?",
            "comparison_height": "Is Bora Çelik shorter than Ayşe Yılmaz?",
            "comparison_speed": "Is Bora Çelik slower than Ayşe Yılmaz?",
            "comparison_weight": "Is Bora Çelik lighter than Ayşe Yılmaz?",
            "ranking": "Is Bora Çelik ranked lower than Ayşe Yılmaz?",
        }
        for relation, query in fixtures.items():
            with self.subTest(relation=relation):
                self.assertEqual(english.question(self.names[1], self.names[0], relation, "after"), query)
                self.assertEqual(english.normalize_query(query, self.names, relation),
                                 (self.names[1], self.names[0], "after"))

    def test_all_relations_arguments_and_query_directions(self):
        for relation in english.RELATIONS:
            for a in self.names:
                for b in self.names:
                    if a == b:
                        continue
                    for direction in ("before", "after"):
                        text = english.statement(a, b, relation, direction)
                        expected = (a, b) if direction == "before" else (b, a)
                        self.assertEqual(english.normalize_statement(text, self.names, relation), expected)
                        text = english.question(a, b, relation, direction)
                        self.assertEqual(english.normalize_query(text, self.names, relation), (a, b, direction))

    def test_parser_detects_changed_direction_and_target(self):
        original = ("Ayşe Yılmaz", "Bora Çelik")
        self.assertNotEqual(english.normalize_statement("Ayşe Yılmaz is behind Bora Çelik.", self.names,
                                                       "spatial_front"), original)
        self.assertNotEqual(english.normalize_statement("Ayşe Yılmaz is in front of İdil Şahin.", self.names,
                                                       "spatial_front"), original)
        self.assertNotEqual(english.normalize_query("Is Ayşe Yılmaz behind Bora Çelik?", self.names,
                                                   "spatial_front"), (*original, "before"))

    def test_parser_rejects_malformed_unknown_and_ambiguous_text(self):
        invalid = ["Ayşe Yılmaz is in front of Nobody.", "Ayşe Yılmaz is in front of Ayşe Yılmaz.",
                   "Ayşe Yılmaz is not in front of Bora Çelik.", "Ayşe Yılmaz is above Bora Çelik.",
                   "Ayşe Yılmaz is in front of Bora Çelik. YES", " Ayşe Yılmaz is in front of Bora Çelik.",
                   "Ayşe Yılmaz  is in front of Bora Çelik.", "Ayşe Yılmaz is in front of Bora Çelik.\n", None]
        for text in invalid:
            with self.subTest(text=text), self.assertRaises(ValueError):
                english.normalize_statement(text, self.names, "spatial_front")
        with self.assertRaises(ValueError):
            english.normalize_query("Is Ayşe Yılmaz in front of Bora Çelik? YES", self.names, "spatial_front")

    def test_overlapping_entity_spans_are_recognized_exactly(self):
        names = ["Ada", "Ada Demir", "Demir", "E0010"]
        self.assertEqual(english.normalize_statement("Ada Demir is behind Ada.", names, "spatial_front"),
                         ("Ada", "Ada Demir"))
        self.assertEqual(english.normalize_query("Did Ada Demir arrive before Demir?", names, "temporal_arrival"),
                         ("Ada Demir", "Demir", "before"))

    def test_context_communicates_total_order_and_relation_assumptions(self):
        for relation in english.RELATIONS:
            self.assertIn("strict total order", english.context(relation))
            self.assertIn("no ties or incomparable pairs", english.context(relation))
        self.assertIn("without physical contact", english.context("spatial_above"))
        self.assertIn("exactly once", english.context("temporal_arrival"))
        self.assertIn("names a task", english.context("temporal_completion"))
        self.assertIn("convey no ordering", english.context("spatial_front", symbolic=True))

    def test_prompt_keeps_supplied_list_orders_and_contains_no_gold(self):
        names = self.names[:2][::-1]
        premises = ["Bora Çelik is behind Ayşe Yılmaz."]
        text = english.compose_prompt("Fixed context.", names, premises, "Is Ayşe Yılmaz in front of Bora Çelik?")
        self.assertEqual(text, "Fixed context.\nEntities: Bora Çelik, Ayşe Yılmaz.\nPremises:\n"
                         "- Bora Çelik is behind Ayşe Yılmaz.\nQuestion: Is Ayşe Yılmaz in front of Bora Çelik?\n"
                         "Provide your answer only as YES or NO.")
        self.assertNotIn("gold", text)
        self.assertNotIn("latent", text)

    def test_invalid_function_arguments_fail_explicitly(self):
        for relation in (None, "unknown", [], True):
            with self.subTest(relation=relation), self.assertRaises(ValueError):
                english.statement("Ada", "Bora", relation)
        for name in ("", " Ada", "Ada ", "Ada  Demir", "Ada\nDemir", "Ada.", "A\u0308da", None):
            with self.subTest(name=name), self.assertRaises(ValueError):
                english.question(name, "Bora", "spatial_front")
        with self.assertRaises(ValueError):
            english.statement("Ada", "Ada", "spatial_front")
        with self.assertRaises(ValueError):
            english.statement("Ada", "Bora", "spatial_front", direction="front")
        with self.assertRaises(ValueError):
            english.statement("Ada", "Bora", "spatial_front", style="paraphrase")
        with self.assertRaises(ValueError):
            english.context("spatial_front", symbolic=1)
        for names in ([], ["Ada"], ["Ada", "Ada"], "Ada", None):
            with self.subTest(names=names), self.assertRaises(ValueError):
                english.normalize_statement("Ada is in front of Bora.", names, "spatial_front")


if __name__ == "__main__":
    unittest.main()

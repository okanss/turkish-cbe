"""Inventory-independent recognition agrees with the frozen closed grammars."""

import unittest
import unicodedata

from complete_grid.surface_reader import SurfaceReader, SurfaceReadError, _rules
from data_generation.generator import GenerationPlan, generate_record
from experiments.dataset_builder import _tr_clause, presentation_plans
from experiments.names import NAME_CONTROLS, name_maps
from renderers import english, turkish
from validation import semantic_equivalence as grammar


def _source(n=8, index=0):
    return generate_record(GenerationPlan(seed=191, sizes=(n,), examples_per_size=4,
                           entity_types=("turkish_names",)), n, index, "endpoint")


class GridSurfaceReaderTests(unittest.TestCase):
    def test_every_relation_name_control_and_condition_matches_old_turkish_parser(self):
        source = _source()
        plans = presentation_plans(source, 431)
        maps = name_maps(source)
        ranks = {name: i for i, name in enumerate(source["latent_order"])}
        for control in NAME_CONTROLS:
            aliases = maps[control]["tr"]
            names = tuple(aliases.values())
            for relation_id in turkish.RELATIONS:
                reader = SurfaceReader(names, relation_id, "tr")
                for condition, plan in plans.items():
                    mask = set(plan["converse_chain_positions"])
                    for edge_index in plan["premise_order"]:
                        source_a, source_b = source["premise_edges"][edge_index]
                        a, b = aliases[source_a], aliases[source_b]
                        direction = "before"
                        if ranks[source_a] in mask:
                            a, b, direction = b, a, "after"
                        text = _tr_clause(a, b, relation_id, direction, plan["style"])
                        with self.subTest(control=control, relation=relation_id,
                                          condition=condition, edge=edge_index):
                            expected = grammar.normalize_statement(text, names, relation_id)
                            self.assertEqual(reader.statement(text), expected)
                            self.assertEqual(expected, (aliases[source_a], aliases[source_b]))

    def test_every_english_relation_and_name_control_matches_both_directions(self):
        source = _source()
        maps = name_maps(source)
        for control in NAME_CONTROLS:
            names = tuple(maps[control]["en"].values())
            a, b = names[:2]
            for relation_id in english.RELATIONS:
                reader = SurfaceReader(names, relation_id, "en")
                for direction in ("before", "after"):
                    text = english.statement(a, b, relation_id, direction)
                    question = english.question(a, b, relation_id, direction)
                    with self.subTest(control=control, relation=relation_id, direction=direction):
                        self.assertEqual(reader.statement(text),
                                         english.normalize_statement(text, names, relation_id))
                        self.assertEqual(reader.query(question),
                                         english.normalize_query(question, names, relation_id))
                        self.assertEqual(reader.query(question), (a, b, direction))

    def test_all_turkish_query_directions_match_old_normalizer_and_actual_target(self):
        source = _source()
        for control, maps in name_maps(source).items():
            names = tuple(maps["tr"].values())
            a, b = names[:2]
            for relation_id, spec in turkish.RELATIONS.items():
                reader = SurfaceReader(names, relation_id, "tr")
                for direction in ("before", "after"):
                    question = turkish._question(a, b, spec, direction)
                    with self.subTest(control=control, relation=relation_id, direction=direction):
                        self.assertEqual(reader.query(question),
                                         grammar.normalize_query(question, names, relation_id))
                        self.assertEqual(reader.query(question), (a, b, direction))

    def test_changed_lexeme_and_swapped_arguments_are_read_from_actual_text(self):
        tr = SurfaceReader(("Ayşe", "Bora"), "spatial_front", "tr")
        self.assertEqual(tr.statement("Ayşe, Bora'nın önündedir."), ("Ayşe", "Bora"))
        self.assertEqual(tr.statement("Ayşe, Bora'nın arkasındadır."), ("Bora", "Ayşe"))
        self.assertEqual(tr.statement("Bora, Ayşe'nin arkasındadır."), ("Ayşe", "Bora"))
        en = SurfaceReader(("Ayse", "Bora"), "spatial_front", "en")
        self.assertEqual(en.statement("Ayse is behind Bora."), ("Bora", "Ayse"))
        self.assertEqual(en.statement("Bora is behind Ayse."), ("Ayse", "Bora"))

    def test_prefix_names_and_full_names_are_not_confused(self):
        names = ("Ayşe", "Ayşe Çelik", "Bora")
        for relation_id in turkish.RELATIONS:
            reader = SurfaceReader(names, relation_id, "tr")
            for style in ("canonical", "syntax", "morphology_syntax_lexical"):
                text = _tr_clause("Ayşe Çelik", "Bora", relation_id, "before", style)
                with self.subTest(relation=relation_id, style=style):
                    self.assertEqual(reader.statement(text), ("Ayşe Çelik", "Bora"))
        en = SurfaceReader(("Ada", "Ada Demir", "Bora"), "spatial_front", "en")
        self.assertEqual(en.statement("Ada Demir is in front of Bora."), ("Ada Demir", "Bora"))

    def test_wrong_suffix_unknown_name_self_comparison_and_poison_text_rejected(self):
        tr = SurfaceReader(("Ayşe", "Ahmet", "Bora"), "comparison_height", "tr")
        bad = (
            "Ayşe, Ahmet'den daha uzundur.", "Ayşe, Bora'den daha uzundur.",
            "Ayşe'nin boyu, Bora'nun boyundan daha uzundur.",
            "Ayşe, Osman'dan daha uzundur.", "Ayşe, Ayşe'den daha uzundur.",
            "Ayşe, Bora'dan daha kısa değildir.", "Ayşe, Bora'dan daha uzundur. YES",
            "Ayşe\n, Bora'dan daha uzundur.", "Ayşe, Bora'dan daha uzundur.\n",
        )
        for text in bad:
            with self.subTest(text=text), self.assertRaises(SurfaceReadError):
                tr.statement(text)
        en = SurfaceReader(("Ayse", "Bora"), "spatial_front", "en")
        for text in ("Ayse is in front of Osman.", "Ayse is in front of Ayse.",
                     "Ayse is not behind Bora.", "Ayse is in front of Bora. YES",
                     "Ayse is in front of Bora.\n", "Ayse is in front of Bora; drop table."):
            with self.subTest(text=text), self.assertRaises(SurfaceReadError):
                en.statement(text)

    def test_non_nfc_inputs_invalid_inventories_and_invalid_grammar_rejected(self):
        inventories = (None, "Ayşe", (), ("Ayşe",), ("Ayşe", "Ayşe"),
                       (" Ayşe", "Bora"), ("Ayşe  Çelik", "Bora"),
                       ("Ayşe'", "Bora"), ("Ayşe,", "Bora"),
                       (unicodedata.normalize("NFD", "İrem"), "Bora"), (5, "Bora"))
        for names in inventories:
            with self.subTest(names=names), self.assertRaises(SurfaceReadError):
                SurfaceReader(names, "spatial_front", "en")
        for language, relation in (("de", "spatial_front"), (None, "spatial_front"),
                                    ("en", "unknown"), ("tr", None)):
            with self.subTest(language=language, relation=relation), self.assertRaises(SurfaceReadError):
                SurfaceReader(("Ayşe", "Bora"), relation, language)
        reader = SurfaceReader(("İrem", "Bora"), "spatial_front", "en")
        for text in (None, 5, "", unicodedata.normalize("NFD", "İrem is in front of Bora.")):
            with self.subTest(text=text), self.assertRaises(SurfaceReadError):
                reader.statement(text)

    def test_compiled_rules_are_shared_across_inventory_sizes(self):
        _rules.cache_clear()
        a = SurfaceReader(("Ayşe", "Bora"), "spatial_front", "en")
        b = SurfaceReader(("Ada", "Ada Demir", "Bora"), "spatial_front", "en")
        self.assertIs(a._statement_rules, b._statement_rules)
        self.assertIs(a._query_rules, b._query_rules)
        self.assertEqual(_rules.cache_info().misses, 2)
        self.assertNotIn("Ayşe", a._statement_rules[0][0].pattern)
        self.assertNotIn("AlphaPlaceholder", a._statement_rules[0][0].pattern)

    def test_n1024_all_adjacent_edges_and_queries_have_exact_normal_form(self):
        source = _source(1024)
        maps = name_maps(source)["symbolic_shared"]
        for language in ("en", "tr"):
            aliases = maps[language]
            names = tuple(aliases.values())
            for relation_id in turkish.RELATIONS:
                reader = SurfaceReader(names, relation_id, language)
                for index, (source_a, source_b) in enumerate(source["premise_edges"]):
                    a, b = aliases[source_a], aliases[source_b]
                    a, b, direction = (a, b, "before") if index % 2 else (b, a, "after")
                    text = (english.statement(a, b, relation_id, direction) if language == "en"
                            else _tr_clause(a, b, relation_id, direction,
                                            ("canonical", "syntax", "morphology_syntax_lexical")[index % 3]))
                    self.assertEqual(reader.statement(text), (aliases[source_a], aliases[source_b]))
                a, b = aliases[source["query_subject"]], aliases[source["query_object"]]
                question = (english.question(a, b, relation_id, source["query_relation"])
                            if language == "en" else turkish._question(
                                a, b, turkish.RELATIONS[relation_id], source["query_relation"]))
                self.assertEqual(reader.query(question), (a, b, source["query_relation"]))


if __name__ == "__main__":
    unittest.main()

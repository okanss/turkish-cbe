"""Inventory reuse must retain the independent parser's semantic checks."""

import unittest
from unittest.mock import patch

from data_generation.entities import entity_pool
from data_generation.generator import GenerationPlan, generate_record
from experiments.surface_reader import TurkishReader
from renderers import turkish
from validation import semantic_equivalence as grammar


class TurkishReaderTests(unittest.TestCase):
    def test_all_relations_and_positive_styles_agree_with_the_public_parser(self):
        plan = GenerationPlan(seed=71, sizes=(4,), examples_per_size=4,
                              entity_types=("turkish_names",))
        for index in (0, 1):
            record = generate_record(plan, 4, index, "endpoint")
            names = tuple(record["latent_order"])
            for relation_id in turkish.RELATIONS:
                reader = TurkishReader(names, relation_id)
                for variant in ("P0", "P1", "P2", "P3", "P4", "P5"):
                    presentation = turkish.render(record, relation_id=relation_id, variant=variant)
                    for text in presentation["premises"]:
                        with self.subTest(index=index, relation=relation_id, variant=variant):
                            self.assertEqual(reader.statement(text),
                                grammar.normalize_statement(text, names, relation_id))
                    self.assertEqual(reader.query(presentation["query"]),
                        grammar.normalize_query(presentation["query"], names, relation_id))

    def test_inventory_is_validated_once_for_repeated_statement_and_query_reads(self):
        names = ("Ayşe", "Bora")
        with patch.object(grammar, "_inventory", wraps=grammar._inventory) as inventory:
            reader = TurkishReader(names, "spatial_front")
            for _ in range(10):
                self.assertEqual(reader.statement("Ayşe, Bora'nın önündedir."), ("Ayşe", "Bora"))
                self.assertEqual(reader.query("Ayşe, Bora'nın önünde midir?"),
                                 ("Ayşe", "Bora", "before"))
            self.assertEqual(inventory.call_count, 1)

    def test_actual_surface_reversal_and_inverse_predicate_are_read(self):
        reader = TurkishReader(("Ayşe", "Bora"), "spatial_front")
        self.assertEqual(reader.statement("Ayşe, Bora'nın önündedir."), ("Ayşe", "Bora"))
        self.assertEqual(reader.statement("Bora, Ayşe'nin önündedir."), ("Bora", "Ayşe"))
        self.assertEqual(reader.statement("Ayşe, Bora'nın arkasındadır."), ("Bora", "Ayşe"))
        self.assertEqual(reader.query("Bora, Ayşe'nin arkasında mıdır?"),
                         ("Bora", "Ayşe", "after"))

    def test_malformed_suffix_unknown_names_and_self_comparison_are_rejected(self):
        reader = TurkishReader(("Ayşe", "Bora"), "spatial_front")
        for text in ("Ayşe, Bora'nin önündedir.", "Ayşe, Bora'dan önündedir.",
                     "Can, Bora'nın önündedir.", "Ayşe, Ayşe'nin önündedir.",
                     "Ayşe, Bora'nın önündedir.\n"):
            with self.subTest(text=text), self.assertRaises(grammar.SemanticEquivalenceError):
                reader.statement(text)
        with self.assertRaises(grammar.SemanticEquivalenceError):
            reader.query("Ayşe, Bora'nin önünde midir?")

    def test_ablative_suffix_validation_is_preserved(self):
        reader = TurkishReader(("Ayşe", "Zeynep"), "comparison_height")
        self.assertEqual(reader.statement("Ayşe, Zeynep'ten daha uzundur."), ("Ayşe", "Zeynep"))
        with self.assertRaises(grammar.SemanticEquivalenceError):
            reader.statement("Ayşe, Zeynep'den daha uzundur.")

    def test_negation_is_rejected_even_though_the_source_grammar_supports_it(self):
        reader = TurkishReader(("Ayşe", "Bora"), "spatial_front")
        text = "Ayşe, Bora'nın arkasında değildir."
        self.assertEqual(grammar.normalize_statement(text, ("Ayşe", "Bora"), "spatial_front"),
                         ("Ayşe", "Bora"))
        with self.assertRaises(grammar.SemanticEquivalenceError):
            reader.statement(text)

    def test_nfc_and_inventory_checks_are_preserved(self):
        for names, relation in ((("Ayşe", "Ayşe"), "spatial_front"),
                                (("Ayşe", "Bora"), "unsupported"),
                                (("Ayşe", "Bora"), None),
                                (("Ayşe", "I\u0307rem"), "spatial_front")):
            with self.subTest(names=names, relation=relation), self.assertRaises(grammar.SemanticEquivalenceError):
                TurkishReader(names, relation)
        reader = TurkishReader(("Ayşe", "İrem"), "spatial_front")
        with self.assertRaises(grammar.SemanticEquivalenceError):
            reader.statement("Ayşe, I\u0307rem'in önündedir.")
        for text in (None, 1, ""):
            with self.subTest(text=text), self.assertRaises(grammar.SemanticEquivalenceError):
                reader.statement(text)

    def test_one_thousand_twenty_four_entities_use_the_same_semantic_contract(self):
        names = entity_pool("symbolic_ids", 1024)[:1024]
        reader = TurkishReader(names, "spatial_front")
        self.assertEqual(reader.statement("E0000, E1023'ün önündedir."), ("E0000", "E1023"))
        self.assertEqual(reader.query("E0000, E1023'ün önünde midir?"),
                         ("E0000", "E1023", "before"))


if __name__ == "__main__":
    unittest.main()

"""The lexical catalog is a literal view of the frozen renderer vocabulary."""

import copy
import unicodedata
import unittest

from analysis.lexical_bias.catalog import build_catalog
from renderers import english, turkish
from renderers.morphology import ablative, genitive, possessive_locative


class LexicalCatalogTests(unittest.TestCase):
    def test_all_nine_relations_and_eighteen_directed_roots_match_renderers(self):
        catalog = build_catalog()
        self.assertEqual(len(catalog), 9)
        self.assertEqual({row["relation_id"] for row in catalog}, set(turkish.RELATIONS))
        self.assertEqual(sum(len(row["directions"]) for row in catalog), 18)
        for row in catalog:
            spec = turkish.RELATIONS[row["relation_id"]]
            self.assertEqual(row["family"], spec.family)
            self.assertEqual([item["direction"] for item in row["directions"]], ["before", "after"])
            self.assertEqual([item["lemma"] for item in row["directions"]], [spec.forward, spec.reverse])
            self.assertEqual(row["human_review_status"], "pending")

    def test_literal_canonical_query_and_attribute_templates_are_preserved(self):
        for relation in build_catalog():
            relation_id = relation["relation_id"]
            spec = turkish.RELATIONS[relation_id]
            for item in relation["directions"]:
                direction = item["direction"]
                subject, compared, predicate = turkish._parts("Ayşe", "Bora", spec, direction)
                self.assertEqual(item["expression"], f"{subject}, {compared} {predicate}.")
                self.assertEqual(item["term"], predicate)
                self.assertEqual(item["morphology"]["query_expression"],
                                 turkish._question("Ayşe", "Bora", spec, direction))
                self.assertEqual(item["morphology"]["attribute_expression"],
                                 turkish._attribute_statement("Ayşe", "Bora", spec, direction))
                for key in ("canonical_predicate", "query_predicate", "attribute_predicate"):
                    self.assertIn(item["morphology"][key], item["surface_forms"])

    def test_english_lexical_terms_and_examples_are_actual_renderer_outputs(self):
        for relation in build_catalog():
            relation_id = relation["relation_id"]
            spec = english.RELATIONS[relation_id]
            for item in relation["directions"]:
                direction = item["direction"]
                self.assertEqual(item["english_term"], spec.forward if direction == "before" else spec.reverse)
                self.assertEqual(item["english_expression"], english.statement("Ayşe", "Bora", relation_id, direction))
                self.assertEqual(item["english_question"], english.question("Ayşe", "Bora", relation_id, direction))

    def test_inflection_examples_reuse_validated_suffix_helpers(self):
        for relation in build_catalog():
            for item in relation["directions"]:
                for name, forms in item["morphology"]["proper_name_suffixes"].items():
                    self.assertEqual(forms["genitive"], genitive(name))
                    self.assertEqual(forms["ablative"], ablative(name))
                    self.assertIn("'", forms["genitive"])
                    self.assertIn("'", forms["ablative"])
                if relation["family"] == "spatial":
                    self.assertEqual(item["term"], possessive_locative(item["lemma"]))
                    self.assertEqual(item["morphology"]["possessive_locative_without_copula"],
                                     possessive_locative(item["lemma"], copula=False))

    def test_query_and_attribute_lexical_changes_are_explicit(self):
        by_id = {row["relation_id"]: row for row in build_catalog()}
        temporal = by_id["temporal_arrival"]["directions"]
        self.assertEqual(temporal[0]["term"], "önce gelmiştir")
        self.assertEqual(temporal[0]["morphology"]["attribute_predicate"], "daha erkendir")
        self.assertEqual(temporal[1]["morphology"]["attribute_predicate"], "daha geçtir")
        speed = by_id["comparison_speed"]["directions"][0]
        self.assertEqual(speed["term"], "daha hızlıdır")
        self.assertEqual(speed["morphology"]["query_predicate"], "daha hızlı mıdır")
        self.assertEqual(speed["morphology"]["attribute_predicate"], "daha yüksektir")
        self.assertIn("lexical", speed["morphology"]["attribute_variant"])

    def test_unicode_and_surface_variants_are_deterministic_and_fresh(self):
        first, second = build_catalog(), build_catalog()
        self.assertEqual(first, second)
        for relation in first:
            for item in relation["directions"]:
                self.assertEqual(item["lemma_tokens"], [item["lemma"]])
                self.assertEqual(len(item["surface_forms"]), len(set(item["surface_forms"])))
                for text in (item["lemma"], item["term"], item["expression"], *item["surface_forms"]):
                    self.assertEqual(unicodedata.normalize("NFC", text), text)
        baseline = copy.deepcopy(second)
        first[0]["directions"][0]["surface_forms"].append("synthetic-corruption")
        self.assertEqual(second, baseline)
        self.assertEqual(build_catalog(), baseline)

    def test_closed_p4_root_mapping_distinguishes_changed_lexemes(self):
        expected = {"temporal_arrival": ("erken", "geç"), "temporal_completion": ("erken", "geç"),
                    "comparison_height": ("uzun", "kısa"), "comparison_speed": ("yüksek", "düşük"),
                    "comparison_weight": ("fazla", "az"), "ranking": ("üst", "alt")}
        for relation in build_catalog():
            for index, item in enumerate(relation["directions"]):
                morphology = item["morphology"]
                self.assertEqual(morphology["query_lemma_tokens"], item["lemma_tokens"])
                attribute_root = (expected[relation["relation_id"]][index]
                                  if relation["relation_id"] in expected else item["lemma"])
                self.assertEqual(morphology["attribute_lemma_tokens"], [attribute_root])
                self.assertEqual(morphology["attribute_lexeme_changed"], attribute_root != item["lemma"])
                self.assertIn("excludes P4", morphology["canonical_lemma_scope"])

    def test_literal_suffix_segments_and_sound_rule_metadata_do_not_create_forms(self):
        for relation in build_catalog():
            for item in relation["directions"]:
                morphology = item["morphology"]
                self.assertIn("harmony", morphology["sound_rules"]["vowel_harmony"])
                inventory = {form.split("'", 1)[1] for forms in morphology["proper_name_suffixes"].values()
                             for form in forms.values()}
                self.assertEqual(morphology["proper_name_suffix_inventory"], sorted(inventory))
                if relation["family"] == "spatial":
                    self.assertEqual("".join(morphology["canonical_morphemes"]), item["term"])
                    self.assertEqual("".join(morphology["canonical_morphemes"][:-1]),
                                     morphology["possessive_locative_without_copula"])


if __name__ == "__main__":
    unittest.main()

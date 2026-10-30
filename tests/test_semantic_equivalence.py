"""Independent sentence-recognition and rejection tests, not only round trips."""

from copy import deepcopy
import hashlib
import random
import unittest

from data_generation.generator import GenerationPlan, canonical_bytes, generate_record
from renderers.morphology import ablative, genitive
from validation.semantic_equivalence import (
    SemanticEquivalenceError,
    normalize_query,
    normalize_statement,
    validate_presentation,
)


NAMES = ("Ayşe", "Ahmet", "Işıl", "Gökçe")
# Handwritten sentences supply independent anchors for every family and style.
# A positive normal form is always (Ayşe, Ahmet), regardless of word order.
STATEMENTS = {
    "spatial_front": (
        "Ayşe, Ahmet'in önündedir.",
        "Ahmet, Ayşe'nin arkasındadır.",
        "Ahmet'in önündedir Ayşe.",
        "Ayşe'nin konumu, Ahmet'in konumunun önündedir.",
        "Ayşe, Ahmet'in arkasında değildir.",
    ),
    "spatial_right": (
        "Ayşe, Ahmet'in sağındadır.",
        "Ahmet, Ayşe'nin solundadır.",
        "Ahmet'in sağındadır Ayşe.",
        "Ayşe'nin konumu, Ahmet'in konumunun sağındadır.",
        "Ayşe, Ahmet'in solunda değildir.",
    ),
    "spatial_above": (
        "Ayşe, Ahmet'in üstündedir.",
        "Ahmet, Ayşe'nin altındadır.",
        "Ahmet'in üstündedir Ayşe.",
        "Ayşe'nin konumu, Ahmet'in konumunun üstündedir.",
        "Ayşe, Ahmet'in altında değildir.",
    ),
    "temporal_arrival": (
        "Ayşe, Ahmet'ten önce gelmiştir.",
        "Ahmet, Ayşe'den sonra gelmiştir.",
        "Ahmet'ten önce gelmiştir Ayşe.",
        "Ayşe'nin geliş zamanı, Ahmet'in geliş zamanından daha erkendir.",
        "Ayşe, Ahmet'ten sonra gelmemiştir.",
    ),
    "temporal_completion": (
        "Ayşe adlı görev, Ahmet adlı görevden önce tamamlanmıştır.",
        "Ahmet adlı görev, Ayşe adlı görevden sonra tamamlanmıştır.",
        "Ahmet adlı görevden önce tamamlanmıştır Ayşe adlı görev.",
        "Ayşe adlı görevin tamamlanma zamanı, Ahmet adlı görevin tamamlanma zamanından daha erkendir.",
        "Ayşe adlı görev, Ahmet adlı görevden sonra tamamlanmamıştır.",
    ),
    "comparison_height": (
        "Ayşe, Ahmet'ten daha uzundur.",
        "Ahmet, Ayşe'den daha kısadır.",
        "Ahmet'ten daha uzundur Ayşe.",
        "Ayşe'nin boyu, Ahmet'in boyundan daha uzundur.",
        "Ayşe, Ahmet'ten daha kısa değildir.",
    ),
    "comparison_speed": (
        "Ayşe, Ahmet'ten daha hızlıdır.",
        "Ahmet, Ayşe'den daha yavaştır.",
        "Ahmet'ten daha hızlıdır Ayşe.",
        "Ayşe'nin hızı, Ahmet'in hızından daha yüksektir.",
        "Ayşe, Ahmet'ten daha yavaş değildir.",
    ),
    "comparison_weight": (
        "Ayşe, Ahmet'ten daha ağırdır.",
        "Ahmet, Ayşe'den daha hafiftir.",
        "Ahmet'ten daha ağırdır Ayşe.",
        "Ayşe'nin ağırlığı, Ahmet'in ağırlığından daha fazladır.",
        "Ayşe, Ahmet'ten daha hafif değildir.",
    ),
    "ranking": (
        "Ayşe, Ahmet'ten daha üst sıradadır.",
        "Ahmet, Ayşe'den daha alt sıradadır.",
        "Ahmet'ten daha üst sıradadır Ayşe.",
        "Ayşe'nin sırası, Ahmet'in sırasından daha üsttedir.",
        "Ayşe, Ahmet'ten daha alt sırada değildir.",
    ),
}
QUERIES = {
    "spatial_front": ("önünde midir", "arkasında mıdır"),
    "spatial_right": ("sağında mıdır", "solunda mıdır"),
    "spatial_above": ("üstünde midir", "altında mıdır"),
    "temporal_arrival": ("önce gelmiş midir", "sonra gelmiş midir"),
    "comparison_height": ("daha uzun mudur", "daha kısa mıdır"),
    "comparison_speed": ("daha hızlı mıdır", "daha yavaş mıdır"),
    "comparison_weight": ("daha ağır mıdır", "daha hafif midir"),
    "ranking": ("daha üst sırada mıdır", "daha alt sırada mıdır"),
}


class SentenceNormalizationTests(unittest.TestCase):
    def test_handwritten_equivalence_across_nine_families_and_five_styles(self):
        for relation_id, sentences in STATEMENTS.items():
            for sentence in sentences:
                with self.subTest(relation_id=relation_id, sentence=sentence):
                    self.assertEqual(normalize_statement(sentence, NAMES, relation_id), ("Ayşe", "Ahmet"))

    def test_inverting_a_predicate_reverses_the_normal_form(self):
        for relation_id, sentences in STATEMENTS.items():
            sentence = sentences[1].replace("Ahmet", "PLACEHOLDER").replace("Ayşe", "Ahmet").replace("PLACEHOLDER", "Ayşe")
            # Repair suffixes after the name substitution independently.
            sentence = sentence.replace("Ahmet'nin", "Ahmet'in").replace("Ayşe'in", "Ayşe'nin")
            sentence = sentence.replace("Ahmet'den", "Ahmet'ten").replace("Ayşe'ten", "Ayşe'den")
            with self.subTest(relation_id=relation_id):
                self.assertEqual(normalize_statement(sentence, NAMES, relation_id), ("Ahmet", "Ayşe"))

    def test_query_subject_and_direction_are_preserved(self):
        for relation_id, predicates in QUERIES.items():
            for index, predicate in enumerate(predicates):
                object_form = genitive("Ahmet") if relation_id.startswith("spatial_") else ablative("Ahmet")
                text = f"Ayşe, {object_form} {predicate}?"
                with self.subTest(relation_id=relation_id, index=index):
                    self.assertEqual(normalize_query(text, NAMES, relation_id), ("Ayşe", "Ahmet", ("before", "after")[index]))
        for index, predicate in enumerate(("önce", "sonra")):
            text = f"Ayşe adlı görev, Ahmet adlı görevden {predicate} tamamlanmış mıdır?"
            self.assertEqual(normalize_query(text, NAMES, "temporal_completion"), ("Ayşe", "Ahmet", ("before", "after")[index]))

    def test_wrong_affixes_apostrophes_and_nfc_are_rejected(self):
        cases = (
            "Ayşe, Ahmet'nin önündedir.", "Ayşe, Ahmetin önündedir.",
            "Ayşe, Ahmet’in önündedir.", "Ayşe, Ahmet'ın önündedir.",
            "Ays\u0327e, Ahmet'in önündedir.", "Ayşe, Ahmet'in ö\u0308nündedir.",
        )
        for sentence in cases:
            with self.subTest(sentence=sentence), self.assertRaises(SemanticEquivalenceError):
                normalize_statement(sentence, NAMES, "spatial_front")
        with self.assertRaises(SemanticEquivalenceError):
            normalize_statement("Ayşe, Ahmet'den önce gelmiştir.", NAMES, "temporal_arrival")

    def test_rejects_unknown_entities_reflexive_comparisons_and_extra_words(self):
        for sentence in (
            "Ayşe, Mehmet'in önündedir.", "Ayşe, Ayşe'nin önündedir.",
            "Bütün Ayşe, Ahmet'in önündedir.", "Ayşe, Ahmet'in hemen önündedir.",
            "Ayşe, Ahmet'in önündedir. Fakat sıra tersidir.",
            "Ayşe, Ahmet'in önündedir.\n", "Ayşe,  Ahmet'in önündedir.",
        ):
            with self.subTest(sentence=sentence), self.assertRaises(SemanticEquivalenceError):
                normalize_statement(sentence, NAMES, "spatial_front")

    def test_family_context_is_required_for_lexical_normalization(self):
        with self.assertRaises(SemanticEquivalenceError):
            normalize_statement(STATEMENTS["spatial_front"][0], NAMES, "spatial_right")
        with self.assertRaises(SemanticEquivalenceError):
            normalize_statement(STATEMENTS["spatial_front"][0], NAMES, "unknown")

    def test_inventory_errors_are_explicit(self):
        for names in ((), ("Ayşe",), ("Ayşe", "Ayşe"), ("Ayşe", None)):
            with self.subTest(names=names), self.assertRaises(SemanticEquivalenceError):
                normalize_statement(STATEMENTS["spatial_front"][0], names, "spatial_front")

    def test_prefix_names_and_1024_entity_inventory(self):
        names = tuple(f"E{i:04d}" for i in range(1024))
        self.assertEqual(normalize_statement("E0001, E1023'ün önündedir.", names, "spatial_front"), ("E0001", "E1023"))
        self.assertEqual(normalize_query("E1023, E0004'ten daha hafif midir?", names, "comparison_weight"), ("E1023", "E0004", "after"))
        self.assertEqual(normalize_statement("Ayşe, Ayşen'in önündedir.", ("Ayşe", "Ayşen"), "spatial_front"), ("Ayşe", "Ayşen"))


class PresentationTamperTests(unittest.TestCase):
    """Build an independent canonical presentation instead of re-rendering it."""

    def setUp(self):
        from renderers.turkish import expected_context
        plan = GenerationPlan(sizes=(4,), examples_per_size=4, entity_types=("symbolic_ids",))
        self.record = generate_record(plan, 4, 0, "endpoint")
        record = self.record
        names = self.planned_permutation(sorted(record["latent_order"]), "names")
        premises = [f"{a}, {genitive(b)} önündedir." for a, b in record["premise_edges"]]
        q = record["query_subject"], record["query_object"]
        relation = "önünde" if record["query_relation"] == "before" else "arkasında"
        self.presentation = {
            "instance_id": record["instance_id"], "ordering_id": record["ordering_id"],
            "generator_version": record["generator_version"], "renderer_version": "0.3.0",
            "source_record_sha256": hashlib.sha256(canonical_bytes(record)).hexdigest(),
            "relation_id": "spatial_front", "variant": "P0", "entity_names": names,
            "seed": 31, "language": "tr", "family": "spatial", "transformations": [],
            "factor_axes": [], "is_multi_transform": False, "human_review_status": "pending",
            "eligible_for_confirmatory_evaluation": False,
            "premises": premises, "query": f"{q[0]}, {genitive(q[1])} {relation} {'midir' if relation.endswith('e') else 'mıdır'}?",
            "context": expected_context("spatial_front", names), "strict_total_order": True,
        }
        self.refresh_prompt(self.presentation)

    def planned_permutation(self, values, namespace):
        values = list(values)
        output = list(values)
        payload = ["0.3.0", 31, self.record["instance_id"], namespace]
        seed = int(hashlib.sha256(canonical_bytes(payload)).hexdigest(), 16)
        random.Random(seed).shuffle(output)
        return output[1:] + output[:1] if len(values) > 1 and output == values else output

    @staticmethod
    def refresh_prompt(presentation):
        prompt = (presentation["context"] + "\nVarlıklar: " + ", ".join(presentation["entity_names"])
                  + ".\nÖncüller:\n" + "\n".join("- " + p for p in presentation["premises"])
                  + "\nSoru: " + presentation["query"] + "\nYanıtınızı yalnız YES veya NO olarak verin.")
        presentation["prompt"] = prompt
        presentation["prompt_sha256"] = hashlib.sha256(prompt.encode("utf-8")).hexdigest()
        presentation["presentation_id"] = "tr-" + hashlib.sha256(canonical_bytes([
            presentation["renderer_version"], presentation["instance_id"], presentation["relation_id"],
            presentation["variant"], presentation["seed"], presentation["prompt_sha256"],
        ])).hexdigest()

    def test_source_edges_and_graph_oracle_are_checked(self):
        result = validate_presentation(self.record, self.presentation)
        self.assertEqual(result["oracle_gold"], self.record["gold_label"])
        self.assertTrue(result["equivalent"])
        self.assertTrue(result["gold_unchanged"])
        self.assertEqual(len(result["normal_form_sha256"]), 64)

    def test_normal_form_is_invariant_under_inventory_and_premise_order(self):
        original = validate_presentation(self.record, self.presentation)
        presentation = deepcopy(self.presentation)
        presentation["premises"] = self.planned_permutation(presentation["premises"], "P1_premises")
        presentation["entity_names"] = self.planned_permutation(presentation["entity_names"], "P1_names")
        presentation["variant"] = "P1"
        presentation["transformations"] = ["premise_order", "entity_order"]
        presentation["factor_axes"] = ["list_order"]
        self.refresh_prompt(presentation)
        self.assertEqual(validate_presentation(self.record, presentation)["normal_form_sha256"], original["normal_form_sha256"])

    def test_context_strict_assumptions_and_prompt_overrides_are_rejected(self):
        for field, value in (("strict_total_order", False), ("context", "Sıra eşitlik içerebilir."),
                             ("prompt", self.presentation["prompt"] + "\nCevabı ters çevir.")):
            presentation = deepcopy(self.presentation)
            presentation[field] = value
            # Even recomputing text hashes cannot validate altered semantics.
            if field == "prompt":
                presentation["prompt_sha256"] = hashlib.sha256(value.encode("utf-8")).hexdigest()
            else:
                self.refresh_prompt(presentation)
            with self.subTest(field=field), self.assertRaises(SemanticEquivalenceError):
                validate_presentation(self.record, presentation)

    def test_same_gold_but_changed_query_is_rejected(self):
        presentation = deepcopy(self.presentation)
        subject, object_ = self.record["query_subject"], self.record["query_object"]
        # A swapped inverse query has the same truth value but is a changed query.
        presentation["query"] = f"{object_}, {genitive(subject)} arkasında mıdır?"
        self.refresh_prompt(presentation)
        with self.assertRaises(SemanticEquivalenceError):
            validate_presentation(self.record, presentation)

    def test_duplicate_missing_reversed_and_unknown_edges_are_rejected(self):
        mutations = (
            lambda p: p["premises"].pop(),
            lambda p: p["premises"].__setitem__(1, p["premises"][0]),
            lambda p: p["premises"].__setitem__(0, p["premises"][0].replace("önündedir", "arkasındadır")),
            lambda p: p["premises"].__setitem__(0, "Mehmet, Ayşe'nin önündedir."),
        )
        for mutation in mutations:
            presentation = deepcopy(self.presentation)
            mutation(presentation)
            self.refresh_prompt(presentation)
            with self.assertRaises(SemanticEquivalenceError):
                validate_presentation(self.record, presentation)

    def test_negation_requires_p6_and_all_its_premises(self):
        presentation = deepcopy(self.presentation)
        presentation["premises"] = [p.replace("önündedir.", "arkasında değildir.") for p in presentation["premises"]]
        self.refresh_prompt(presentation)
        with self.assertRaises(SemanticEquivalenceError):
            validate_presentation(self.record, presentation)
        presentation["variant"] = "P6"
        presentation["transformations"] = ["controlled_negation", "opposite_lexeme"]
        presentation["factor_axes"] = ["negation", "lexical"]
        presentation["is_multi_transform"] = True
        self.refresh_prompt(presentation)
        self.assertTrue(validate_presentation(self.record, presentation)["equivalent"])
        presentation["premises"][0] = self.presentation["premises"][0]
        self.refresh_prompt(presentation)
        with self.assertRaises(SemanticEquivalenceError):
            validate_presentation(self.record, presentation)

    def test_variation_labels_cannot_hide_changed_syntax_or_list_order(self):
        presentation = deepcopy(self.presentation)
        # A morphological premise has identical logic but belongs to another arm.
        a, b = self.record["premise_edges"][0]
        presentation["premises"][0] = f"{genitive(a)} konumu, {genitive(b)} konumunun önündedir."
        self.refresh_prompt(presentation)
        with self.assertRaises(SemanticEquivalenceError):
            validate_presentation(self.record, presentation)
        presentation = deepcopy(self.presentation)
        presentation["premises"].reverse()
        self.refresh_prompt(presentation)
        with self.assertRaises(SemanticEquivalenceError):
            validate_presentation(self.record, presentation)

    def test_mutated_gold_source_fingerprints_and_surface_gold_are_rejected(self):
        record = deepcopy(self.record)
        record["gold_label"] = not record["gold_label"]
        with self.assertRaises(SemanticEquivalenceError):
            validate_presentation(record, self.presentation)
        for field, value in (("source_record_sha256", "0" * 64), ("instance_id", "bad-id"),
                             ("ordering_id", "bad-id"), ("prompt_sha256", "0" * 64),
                             ("gold_label", not self.record["gold_label"])):
            presentation = deepcopy(self.presentation)
            presentation[field] = value
            with self.subTest(field=field), self.assertRaises(SemanticEquivalenceError):
                validate_presentation(self.record, presentation)

    def test_metadata_cannot_misidentify_the_renderer_language_family_or_factors(self):
        for field, value in (
            ("renderer_version", "9.9.9"), ("language", "en"), ("family", "temporal"),
            ("transformations", ["word_order"]), ("factor_axes", ["syntax"]),
            ("is_multi_transform", True), ("is_multi_transform", 0),
            ("presentation_id", "tr-" + "0" * 64),
        ):
            presentation = deepcopy(self.presentation)
            presentation[field] = value
            with self.subTest(field=field), self.assertRaises(SemanticEquivalenceError):
                validate_presentation(self.record, presentation)

    def test_renderer_seed_must_be_a_bounded_integer(self):
        for seed in (True, -1, 2**64, 1.0, "31", None):
            presentation = deepcopy(self.presentation)
            presentation["seed"] = seed
            with self.subTest(seed=seed), self.assertRaises(SemanticEquivalenceError):
                validate_presentation(self.record, presentation)

    def test_automatic_semantic_checks_do_not_create_human_approval(self):
        for field, value in (("human_review_status", "approved"),
                             ("eligible_for_confirmatory_evaluation", True)):
            presentation = deepcopy(self.presentation)
            presentation[field] = value
            with self.subTest(field=field), self.assertRaises(SemanticEquivalenceError):
                validate_presentation(self.record, presentation)

    def test_entity_list_permutation_is_confined_to_its_declared_factor_plan(self):
        presentation = deepcopy(self.presentation)
        presentation["entity_names"].reverse()
        self.refresh_prompt(presentation)
        with self.assertRaises(SemanticEquivalenceError):
            validate_presentation(self.record, presentation)

    def test_p1_requires_both_planned_list_changes(self):
        original = deepcopy(self.presentation)
        presentation = deepcopy(original)
        presentation["variant"] = "P1"
        presentation["transformations"] = ["premise_order", "entity_order"]
        presentation["factor_axes"] = ["list_order"]
        expected_names = self.planned_permutation(original["entity_names"], "P1_names")
        expected_premises = self.planned_permutation(original["premises"], "P1_premises")
        # Missing both, just the premise change, or just the name change must fail.
        for names, premises in ((original["entity_names"], original["premises"]),
                                (expected_names, original["premises"]),
                                (original["entity_names"], expected_premises)):
            presentation["entity_names"] = names
            presentation["premises"] = premises
            self.refresh_prompt(presentation)
            with self.subTest(names_changed=names == expected_names,
                              premises_changed=premises == expected_premises):
                with self.assertRaises(SemanticEquivalenceError):
                    validate_presentation(self.record, presentation)
        presentation["entity_names"] = expected_names
        presentation["premises"] = expected_premises
        self.refresh_prompt(presentation)
        self.assertTrue(validate_presentation(self.record, presentation)["equivalent"])

    def test_p5_requires_its_own_exact_premise_and_name_permutations(self):
        from renderers.turkish import render
        original = render(self.record, "spatial_front", "P5", seed=31)
        unpermuted_names = self.presentation["entity_names"]
        unpermuted_premises = [f"{genitive(b)} konumu, {genitive(a)} konumunun arkasındadır."
                              for a, b in self.record["premise_edges"]]
        for field, value in (("entity_names", unpermuted_names),
                             ("premises", unpermuted_premises)):
            presentation = deepcopy(original)
            presentation[field] = value
            self.refresh_prompt(presentation)
            with self.subTest(field=field), self.assertRaises(SemanticEquivalenceError):
                validate_presentation(self.record, presentation)


if __name__ == "__main__":
    unittest.main()

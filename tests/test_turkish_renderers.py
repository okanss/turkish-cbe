"""Surface regressions and oracle checks for the constrained Turkish renderer.

Successful mechanical checks do not certify naturalness; a human review of
these templates and the separate sample report is still required.
"""

from collections import Counter
from copy import deepcopy
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import unittest

from data_generation.entities import ENTITY_TYPES
from data_generation.generator import GenerationPlan, canonical_bytes, generate_record
from data_generation.oracle import answer_from_edges
from renderers.turkish import RELATIONS, VARIANTS, render
from validation.semantic_equivalence import (
    normalize_query,
    normalize_statement,
    validate_presentation,
)


RELATION_IDS = (
    "spatial_front", "spatial_right", "spatial_above", "temporal_arrival",
    "temporal_completion", "comparison_height", "comparison_speed",
    "comparison_weight", "ranking",
)


def make_record(entity_type="turkish_names", n=8, index=0, query_type="endpoint"):
    plan = GenerationPlan(seed=11, sizes=(n,), examples_per_size=4,
                          entity_types=(entity_type,), distances=(1, 2, 4))
    return generate_record(plan, n, index, query_type,
                           1 if query_type == "distance_controlled" else None)


def words(sentence):
    """Ignore punctuation, retain every word and inflection verbatim."""
    return Counter(re.findall(r"[\w']+", sentence, flags=re.UNICODE))


class TurkishRendererTests(unittest.TestCase):
    def test_declared_relation_and_variant_inventory(self):
        self.assertEqual(set(RELATIONS), set(RELATION_IDS))
        self.assertEqual(set(VARIANTS), {f"P{i}" for i in range(7)})

    def test_all_families_variants_names_and_gold_preserve_the_oracle(self):
        query_types = ("endpoint", "internal", "distance_controlled")
        cases = 0
        for entity_type, query_type in zip(ENTITY_TYPES, query_types):
            for index, n in ((0, 4), (1, 8)):
                record = make_record(entity_type, n, index, query_type)
                original = deepcopy(record)
                for relation_id in RELATION_IDS:
                    canonical = render(record, relation_id=relation_id)
                    for variant in VARIANTS:
                        with self.subTest(entity=entity_type, n=n, relation=relation_id,
                                          variant=variant, gold=record["gold_label"]):
                            result = render(record, relation_id=relation_id, variant=variant,
                                            allow_negation=variant == "P6")
                            report = validate_presentation(record, result)
                            self.assertIs(report["equivalent"], True)
                            self.assertIs(report["gold_unchanged"], True)
                            self.assertIs(report["oracle_gold"], record["gold_label"])
                            self.assertEqual(result["validation"], report)
                            self.assertEqual(report["normal_form_sha256"],
                                             canonical["validation"]["normal_form_sha256"])
                            normal_edges = [normalize_statement(text, result["entity_names"], relation_id)
                                            for text in result["premises"]]
                            self.assertCountEqual(normal_edges,
                                                  [tuple(edge) for edge in record["premise_edges"]])
                            self.assertEqual(normalize_query(result["query"], result["entity_names"], relation_id),
                                             (record["query_subject"], record["query_object"],
                                              record["query_relation"]))
                            self.assertEqual(result["query"], canonical["query"])
                            self.assertEqual(result["context"], canonical["context"])
                            self.assertEqual(set(result["entity_names"]), set(record["latent_order"]))
                            self.assertEqual(len(result["entity_names"]), n)
                            self.assertEqual(len(result["premises"]), n - 1)
                            self.assertIs(result["strict_total_order"], True)
                            self.assertEqual(result["instance_id"], record["instance_id"])
                            self.assertEqual(result["ordering_id"], record["ordering_id"])
                            self.assertEqual(result["renderer_version"], "0.3.0")
                            self.assertEqual(result["human_review_status"], "pending")
                            self.assertIs(result["eligible_for_confirmatory_evaluation"], False)
                            self.assertEqual(record, original)
                            cases += 1
        self.assertEqual(cases, 378)

    def test_all_four_query_cells_and_all_query_arms(self):
        for query_type in ("endpoint", "internal", "distance_controlled"):
            observed = Counter()
            for index in range(4):
                record = make_record(index=index, query_type=query_type)
                result = render(record, relation_id="comparison_height", variant="P2")
                validate_presentation(record, result)
                observed[(record["query_direction"], record["gold_label"])] += 1
                self.assertIs(answer_from_edges(sorted(record["latent_order"]),
                                                record["premise_edges"],
                                                record["query_subject"], record["query_object"],
                                                record["query_relation"]), record["gold_label"])
            self.assertEqual(observed, Counter({("forward", True): 1, ("forward", False): 1,
                                                ("reverse", True): 1, ("reverse", False): 1}))

    def test_question_particles_use_the_correct_vowel_harmony(self):
        # Independent expected endings prevent a shared renderer/parser typo
        # from passing a round-trip equivalence test.
        endings = {
            "spatial_front": ("önünde midir?", "arkasında mıdır?"),
            "spatial_right": ("sağında mıdır?", "solunda mıdır?"),
            "spatial_above": ("üstünde midir?", "altında mıdır?"),
            "temporal_arrival": ("önce gelmiş midir?", "sonra gelmiş midir?"),
            "temporal_completion": ("önce tamamlanmış mıdır?", "sonra tamamlanmış mıdır?"),
            "comparison_height": ("daha uzun mudur?", "daha kısa mıdır?"),
            "comparison_speed": ("daha hızlı mıdır?", "daha yavaş mıdır?"),
            "comparison_weight": ("daha ağır mıdır?", "daha hafif midir?"),
            "ranking": ("daha üst sırada mıdır?", "daha alt sırada mıdır?"),
        }
        for relation_id, pair in endings.items():
            for index, ending in enumerate(pair):
                with self.subTest(relation=relation_id, index=index):
                    result = render(make_record(index=index), relation_id=relation_id)
                    self.assertTrue(result["query"].endswith(ending), result["query"])

    def test_p1_only_permutates_lists_and_does_not_change_query(self):
        record = make_record(n=16)
        canonical = render(record, variant="P0")
        shuffled = render(record, variant="P1")
        self.assertCountEqual(shuffled["premises"], canonical["premises"])
        self.assertCountEqual(shuffled["entity_names"], canonical["entity_names"])
        self.assertNotEqual(shuffled["premises"], canonical["premises"])
        self.assertNotEqual(shuffled["entity_names"], canonical["entity_names"])
        self.assertEqual(shuffled["query"], canonical["query"])
        self.assertEqual(shuffled["context"], canonical["context"])
        self.assertIs(shuffled["is_multi_transform"], False)
        validate_presentation(record, shuffled)

    def test_p2_keeps_inventory_and_premise_order_but_inverts_every_relation(self):
        record = make_record()
        for relation_id in RELATION_IDS:
            with self.subTest(relation=relation_id):
                canonical = render(record, relation_id=relation_id)
                inverse = render(record, relation_id=relation_id, variant="P2")
                self.assertEqual(canonical["entity_names"], inverse["entity_names"])
                self.assertTrue(all(a != b for a, b in zip(canonical["premises"], inverse["premises"])))
                self.assertIs(inverse["is_multi_transform"], False)
                self.assertEqual(
                    [normalize_statement(text, inverse["entity_names"], relation_id)
                     for text in inverse["premises"]],
                    [tuple(edge) for edge in record["premise_edges"]],
                )
                validate_presentation(record, inverse)

    def test_p3_preserves_words_and_inflections_of_each_premise(self):
        record = make_record()
        for relation_id in RELATION_IDS:
            with self.subTest(relation=relation_id):
                canonical = render(record, relation_id=relation_id)
                alternate = render(record, relation_id=relation_id, variant="P3")
                self.assertEqual(canonical["entity_names"], alternate["entity_names"])
                for a, b in zip(canonical["premises"], alternate["premises"]):
                    self.assertNotEqual(a, b)
                    self.assertEqual(words(a), words(b))
                self.assertIs(alternate["is_multi_transform"], False)
                self.assertEqual(
                    [normalize_statement(text, alternate["entity_names"], relation_id)
                     for text in alternate["premises"]],
                    [tuple(edge) for edge in record["premise_edges"]],
                )
                validate_presentation(record, alternate)

    def test_p4_and_p5_are_explicit_multiple_transformations(self):
        record = make_record(n=16)
        for relation_id in RELATION_IDS:
            with self.subTest(relation=relation_id):
                canonical = render(record, relation_id=relation_id)
                morphological = render(record, relation_id=relation_id, variant="P4")
                combined = render(record, relation_id=relation_id, variant="P5")
                self.assertEqual(morphological["entity_names"], canonical["entity_names"])
                self.assertNotEqual(combined["entity_names"], canonical["entity_names"])
                original_edges = [tuple(edge) for edge in record["premise_edges"]]
                self.assertEqual(
                    [normalize_statement(text, morphological["entity_names"], relation_id)
                     for text in morphological["premises"]], original_edges,
                )
                self.assertNotEqual(
                    [normalize_statement(text, combined["entity_names"], relation_id)
                     for text in combined["premises"]], original_edges,
                )
                self.assertNotEqual(set(morphological["premises"]), set(combined["premises"]))
                for result in (morphological, combined):
                    self.assertIs(result["is_multi_transform"], True)
                    self.assertGreater(len(set(result["factor_axes"])), 1)
                    self.assertGreater(len(result["transformations"]), 1)
                    validate_presentation(record, result)

    def test_p6_requires_explicit_opt_in_and_strict_order(self):
        record = make_record()
        for relation_id in RELATION_IDS:
            with self.subTest(relation=relation_id):
                with self.assertRaises(ValueError):
                    render(record, relation_id=relation_id, variant="P6")
                negated = render(record, relation_id=relation_id, variant="P6", allow_negation=True)
                self.assertTrue(all(re.search(r"değil|gelmemiş|tamamlanmamış", sentence)
                                    for sentence in negated["premises"]))
                self.assertIs(negated["strict_total_order"], True)
                self.assertIsInstance(negated["negation_proof"], str)
                self.assertTrue(negated["negation_proof"].strip())
                validate_presentation(record, negated)

    def test_render_is_repeatable_and_does_not_mutate_source(self):
        record = make_record(n=16)
        source = deepcopy(record)
        for variant in VARIANTS:
            kwargs = {"variant": variant, "seed": 31, "allow_negation": variant == "P6"}
            with self.subTest(variant=variant):
                first = render(record, **kwargs)
                self.assertEqual(first, render(record, **kwargs))
                self.assertEqual(record, source)
        first = render(record, variant="P1", seed=31)
        second = render(record, variant="P1", seed=32)
        self.assertNotEqual(first["presentation_id"], second["presentation_id"])
        self.assertNotEqual(first["prompt"], second["prompt"])
        self.assertEqual(first["query"], second["query"])

    def test_hashes_identify_the_exact_source_and_visible_prompt(self):
        record = make_record()
        result = render(record)
        self.assertEqual(result["source_record_sha256"], hashlib.sha256(canonical_bytes(record)).hexdigest())
        self.assertEqual(result["prompt_sha256"], hashlib.sha256(result["prompt"].encode("utf-8")).hexdigest())
        self.assertIn(result["query"], result["prompt"])
        self.assertIn(result["context"], result["prompt"])
        for premise in result["premises"]:
            self.assertIn(premise, result["prompt"])
        for field in ("latent_order", "premise_edges", "gold_label", "query_distance",
                      "query_direction", "sample_index", "ordering_id", "instance_id"):
            self.assertNotIn(field, result["prompt"])
        self.assertNotIn(record["instance_id"], result["prompt"])
        self.assertNotIn(record["ordering_id"], result["prompt"])

    def test_symbolic_inventory_does_not_expose_latent_rank(self):
        record = make_record("symbolic_ids", n=16)
        result = render(record)
        self.assertCountEqual(result["entity_names"], record["latent_order"])
        self.assertNotEqual(result["entity_names"], record["latent_order"])
        self.assertNotEqual(result["entity_names"], sorted(record["latent_order"]))

    def test_invalid_options_are_rejected(self):
        record = make_record()
        for relation_id in (None, "", "spatial_touching", 1, [], {}):
            with self.subTest(relation_id=relation_id), self.assertRaises(ValueError):
                render(record, relation_id=relation_id)
        for variant in (None, "", "p0", "P7", 1, [], {}):
            with self.subTest(variant=variant), self.assertRaises(ValueError):
                render(record, variant=variant)
        for seed in (None, True, -1, 2**64, 1.5, "31", [], {}):
            with self.subTest(seed=seed), self.assertRaises(ValueError):
                render(record, seed=seed)

    def test_malformed_or_inconsistent_source_is_rejected(self):
        record = make_record()
        bad_records = [None, [], {}, {**record, "entity_type": "unknown"},
                       {**record, "gold_label": not record["gold_label"]},
                       {**record, "query_distance": 100},
                       {**record, "latent_order": record["latent_order"][:-1]},
                       {**record, "premise_edges": record["premise_edges"][:-1]}]
        for broken in bad_records:
            with self.subTest(record=broken), self.assertRaises(ValueError):
                render(broken)

    def test_maximum_size_chain_can_be_rendered_and_validated(self):
        record = make_record("symbolic_ids", n=1024, index=1)
        for variant in ("P0", "P2"):
            with self.subTest(variant=variant):
                result = render(record, relation_id="spatial_front", variant=variant)
                self.assertEqual(len(result["premises"]), 1023)
                self.assertEqual(len(result["entity_names"]), 1024)
                validate_presentation(record, result)

    def test_salted_python_hash_cannot_change_a_presentation(self):
        project = Path(__file__).resolve().parents[1]
        script = """import json
from data_generation.generator import GenerationPlan, generate_record
from renderers.turkish import render
p = GenerationPlan(seed=11, sizes=(8,), examples_per_size=4)
r = generate_record(p, 8, 1, 'internal')
print(json.dumps(render(r, variant='P5', seed=31), sort_keys=True, ensure_ascii=False))
"""
        results = []
        for seed in ("1", "976543"):
            env = {**os.environ, "PYTHONPATH": str(project / "src"), "PYTHONHASHSEED": seed}
            result = subprocess.run([sys.executable, "-c", script], env=env, cwd=project,
                                    text=True, capture_output=True, check=True)
            results.append(json.loads(result.stdout))
        self.assertEqual(results[0], results[1])


if __name__ == "__main__":
    unittest.main()

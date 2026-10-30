"""Independent exact-oracle checks, including exhaustive small orderings."""

from itertools import permutations
import random
import unittest

from data_generation.oracle import (
    DataValidationError,
    answer_from_edges,
    answer_from_order,
)


class OracleTests(unittest.TestCase):
    def test_all_small_permutations_pairs_and_relations(self):
        # Exhaustively checks 2! + 3! + 4! + 5! orderings against a direct
        # position-based expectation, including internal and endpoint queries.
        for n in range(2, 6):
            entities = [f"E{index}" for index in range(n)]
            for order in permutations(entities):
                edges = list(zip(order, order[1:]))
                for i, subject in enumerate(order):
                    for j, object in enumerate(order):
                        if i == j:
                            continue
                        for relation in ("before", "after"):
                            expected = i < j if relation == "before" else i > j
                            self.assertIs(answer_from_order(order, subject, object, relation), expected)
                            self.assertIs(answer_from_edges(entities, edges, subject, object, relation), expected)

    def test_edge_and_entity_inventory_order_do_not_affect_answers(self):
        order = ["Çağla", "Öykü", "Şule", "İlke", "Efe"]
        edges = [[a, b] for a, b in zip(order, order[1:])]
        rng = random.Random(834)
        for _ in range(12):
            inventory = order.copy()
            rng.shuffle(inventory)
            rng.shuffle(edges)
            self.assertTrue(answer_from_edges(inventory, edges, "Öykü", "İlke", "before"))
            self.assertFalse(answer_from_edges(inventory, edges, "Öykü", "İlke", "after"))

    def test_large_chain_is_iterative_and_deterministic(self):
        order = [f"E{index:04d}" for index in range(1024)]
        edges = tuple(zip(order, order[1:]))
        for _ in range(3):
            self.assertTrue(answer_from_order(order, order[17], order[1000], "before"))
            self.assertTrue(answer_from_edges(tuple(reversed(order)), edges, order[17], order[1000], "before"))
            self.assertFalse(answer_from_edges(order, tuple(reversed(edges)), order[-1], order[0], "before"))

    def test_broken_premise_graphs_are_rejected(self):
        entities = ["A", "B", "C", "D"]
        broken = {
            "missing edge": [("A", "B"), ("B", "C")],
            "extra edge": [("A", "B"), ("B", "C"), ("C", "D"), ("A", "D")],
            "unknown entity": [("A", "B"), ("B", "C"), ("C", "Z")],
            "reversed internal edge": [("A", "B"), ("C", "B"), ("C", "D")],
            "self loop": [("A", "B"), ("B", "B"), ("C", "D")],
            "duplicate edge": [("A", "B"), ("A", "B"), ("C", "D")],
            "isolated cycle": [("A", "B"), ("C", "D"), ("D", "C")],
            "cycle and isolated entity": [("A", "B"), ("B", "C"), ("C", "A")],
            "outgoing fork": [("A", "B"), ("A", "C"), ("C", "D")],
            "incoming fork": [("A", "C"), ("B", "C"), ("C", "D")],
            "disconnected paths": [("A", "B"), ("C", "D")],
        }
        for name, edges in broken.items():
            with self.subTest(name=name), self.assertRaises(DataValidationError):
                answer_from_edges(entities, edges, "A", "D", "before")

    def test_reverse_total_order_is_a_valid_different_chain(self):
        # Reversing every edge is valid; a single accidental reversal is not.
        self.assertFalse(answer_from_edges(["A", "B", "C"], [("C", "B"), ("B", "A")], "A", "C", "before"))

    def test_invalid_entity_inventories_are_rejected_by_both_oracles(self):
        invalid = [
            None, True, False, 1, "AB", {"A", "B"}, {"A": 0, "B": 1},
            [], ["A"], ["A", "A"], ["A", ""], ["A", " "],
            ["A", True], ["A", False], ["A", 1], ["A", []],
            ["A", {}], ["A", None], ["A", "C\u0327ag\u0306la"],
        ]
        for entities in invalid:
            with self.subTest(entities=entities):
                with self.assertRaises(DataValidationError):
                    answer_from_order(entities, "A", "B", "before")
                with self.assertRaises(DataValidationError):
                    answer_from_edges(entities, [("A", "B")], "A", "B", "before")

    def test_invalid_queries_are_rejected_by_both_oracles(self):
        order = ["A", "B", "Çağla"]
        edges = [("A", "B"), ("B", "Çağla")]
        queries = [("A", "A", "before"), ("A", "Z", "before")]
        for malformed in (None, [], {}, 1, True, False, "", " ", "C\u0327ag\u0306la"):
            queries.extend([(malformed, "B", "before"), ("A", malformed, "after")])
        for malformed in (None, [], {}, 1, True, False, "", "<", "equal", "BEFORE"):
            queries.append(("A", "B", malformed))
        for subject, object, relation in queries:
            with self.subTest(query=(subject, object, relation)):
                with self.assertRaises(DataValidationError):
                    answer_from_order(order, subject, object, relation)
                with self.assertRaises(DataValidationError):
                    answer_from_edges(order, edges, subject, object, relation)

    def test_malformed_edge_containers_and_pairs_are_rejected(self):
        invalid = [
            None, True, False, 1, "AB", {("A", "B")},
            [None], [True], [1], ["AB"], [[]], [["A"]], [["A", "B", "C"]],
            [["A", True]], [["A", []]], [["A", {}]], [["A", None]],
            [["A", ""]], [["A", " "]], [["A", "C\u0327ag\u0306la"]],
        ]
        for edges in invalid:
            with self.subTest(edges=edges), self.assertRaises(DataValidationError):
                answer_from_edges(["A", "B"], edges, "A", "B", "before")


if __name__ == "__main__":
    unittest.main()

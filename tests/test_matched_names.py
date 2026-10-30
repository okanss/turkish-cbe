"""Bilingual labels preserve entity identity across query and surface factors."""

from copy import deepcopy
import unicodedata
import unittest

from data_generation.entities import entity_pool
from data_generation.generator import GenerationPlan, generate_record
from experiments.names import NAME_CONTROLS, NAME_CONTROL_VERSION, name_maps, validate_name_maps


class MatchedNamesTests(unittest.TestCase):
    def setUp(self):
        self.plan = GenerationPlan(seed=409, sizes=(4, 8, 1024), examples_per_size=4,
                                   entity_types=("turkish_names",))
        self.record = generate_record(self.plan, 8, 0, "endpoint")

    def test_all_controls_are_complete_and_bijective(self):
        maps = name_maps(self.record)
        self.assertEqual(NAME_CONTROL_VERSION, "0.4.0")
        self.assertEqual(set(maps), set(NAME_CONTROLS))
        for control in NAME_CONTROLS:
            self.assertEqual(set(maps[control]), {"en", "tr"})
            for mapping in maps[control].values():
                self.assertEqual(set(mapping), set(self.record["latent_order"]))
                self.assertEqual(len(set(mapping.values())), self.record["n"])
        self.assertIsNone(validate_name_maps(self.record, maps))

    def test_native_names_remain_identical_in_both_languages(self):
        maps = name_maps(self.record)
        identity = {name: name for name in self.record["latent_order"]}
        self.assertEqual(maps["native_shared"], {"en": identity, "tr": identity})
        self.assertEqual(maps["en_transliterated"]["tr"], identity)

    def test_explicit_turkish_transliteration_preserves_case_and_spaces(self):
        record = deepcopy(self.record)
        record["n"] = 4
        record["latent_order"] = ["Çığ Öşü", "İÇĞÖŞÜ", "Ayşe Yılmaz", "İrem Güneş"]
        aliases = name_maps(record)["en_transliterated"]
        self.assertEqual(aliases["en"], {"Çığ Öşü": "Cig Osu", "İÇĞÖŞÜ": "ICGOSU",
                                        "Ayşe Yılmaz": "Ayse Yilmaz", "İrem Güneş": "Irem Gunes"})
        self.assertTrue(all(value.isascii() for value in aliases["en"].values()))
        self.assertEqual(aliases["tr"], {name: name for name in record["latent_order"]})

    def test_full_turkish_person_pool_has_no_transliteration_collisions(self):
        pool = entity_pool("turkish_names", 1024)
        # The fixed generator pool has 1,536 names; inspect it in valid-size batches.
        all_aliases = {}
        for start in (0, len(pool) - 1024):
            record = deepcopy(self.record)
            record["n"] = 1024
            record["latent_order"] = list(pool[start:start + 1024])
            all_aliases.update(name_maps(record)["en_transliterated"]["en"])
        self.assertEqual(len(all_aliases), len(pool))
        self.assertEqual(len(set(all_aliases.values())), len(pool))

    def test_symbolic_labels_are_shared_fixed_pool_samples(self):
        maps = name_maps(self.record)["symbolic_shared"]
        self.assertEqual(maps["en"], maps["tr"])
        self.assertTrue(set(maps["en"].values()).issubset(entity_pool("symbolic_ids", 8)))
        self.assertNotEqual(set(maps["en"].values()), {f"E{index:04d}" for index in range(8)})
        self.assertEqual(name_maps(self.record), name_maps(deepcopy(self.record)))

    def test_aliases_are_stable_across_queries_on_one_ordering(self):
        endpoint = generate_record(self.plan, 8, 0, "endpoint")
        internal = generate_record(self.plan, 8, 0, "internal")
        distance = generate_record(self.plan, 8, 0, "distance_controlled", distance=2)
        self.assertNotEqual(endpoint["instance_id"], internal["instance_id"])
        self.assertEqual(endpoint["ordering_id"], distance["ordering_id"])
        self.assertEqual(name_maps(endpoint), name_maps(internal))
        self.assertEqual(name_maps(endpoint), name_maps(distance))
        changed = deepcopy(endpoint)
        changed.update(query_subject=endpoint["query_object"], query_object=endpoint["query_subject"],
                       query_relation="after", instance_id="arbitrary query identifier",
                       language="en", relation_id="comparison_height", presentation="equivalent")
        self.assertEqual(name_maps(endpoint), name_maps(changed))

    def test_mapping_assignment_is_independent_of_inventory_or_latent_rank_order(self):
        original = name_maps(self.record)
        changed = deepcopy(self.record)
        changed["latent_order"] = list(reversed(changed["latent_order"]))
        self.assertEqual(name_maps(changed), original)
        # A new ordering group has its own random label assignment, not rank labels.
        changed["ordering_id"] = "ord-" + "a" * 64
        self.assertNotEqual(name_maps(changed)["symbolic_shared"], original["symbolic_shared"])

    def test_language_maps_are_independent_mutable_copies(self):
        maps = name_maps(self.record)
        name = self.record["latent_order"][0]
        maps["symbolic_shared"]["en"][name] = "E9999"
        self.assertNotEqual(maps["symbolic_shared"]["en"], maps["symbolic_shared"]["tr"])
        self.assertNotEqual(maps["native_shared"]["en"][name], "E9999")
        self.assertNotEqual(name_maps(self.record), maps)

    def test_transliteration_collisions_and_unsupported_characters_are_rejected(self):
        for names in (["Ipek", "İpek", "Ada", "Bora"], ["Âdem", "İpek", "Ada", "Bora"]):
            record = deepcopy(self.record)
            record["n"] = 4
            record["latent_order"] = names
            with self.subTest(names=names), self.assertRaises(ValueError):
                name_maps(record)

    def test_invalid_source_inventories_fail_explicitly(self):
        mutations = [None, {}, {**self.record, "n": True}, {**self.record, "n": 3},
                     {**self.record, "latent_order": self.record["latent_order"][:-1]},
                     {**self.record, "latent_order": ["Ada"] * 8},
                     {**self.record, "ordering_id": "not-an-ordering-id"}]
        for name in ("", " Ada", "Ada  Demir", "Ada\nDemir", "Ada.", "A\u0308da", None):
            record = deepcopy(self.record)
            record["latent_order"][0] = name
            mutations.append(record)
        for record in mutations:
            with self.subTest(record=record), self.assertRaises(ValueError):
                name_maps(record)
        self.assertTrue(all(unicodedata.normalize("NFC", name) == name
                            for name in self.record["latent_order"]))

    def test_validator_rejects_incomplete_nonbijective_and_wrong_rule_maps(self):
        good = name_maps(self.record)
        mutations = [None, {}, {**good, "extra": {}}]
        bad = deepcopy(good)
        del bad["native_shared"]["tr"]
        mutations.append(bad)
        bad = deepcopy(good)
        bad["native_shared"]["xx"] = bad["native_shared"]["en"]
        mutations.append(bad)
        first, second = self.record["latent_order"][:2]
        for control, language, value in (("native_shared", "en", "Changed"),
                                         ("en_transliterated", "tr", "Changed"),
                                         ("symbolic_shared", "tr", "E9999"),
                                         ("symbolic_shared", "en", []),
                                         ("symbolic_shared", "en", "")):
            bad = deepcopy(good)
            bad[control][language][first] = value
            mutations.append(bad)
        bad = deepcopy(good)
        bad["symbolic_shared"]["en"][first] = bad["symbolic_shared"]["en"][second]
        mutations.append(bad)
        bad = deepcopy(good)
        del bad["native_shared"]["en"][first]
        mutations.append(bad)
        bad = deepcopy(good)
        # Even a bijection of valid codes must follow the stable versioned mapping.
        for language in ("en", "tr"):
            mapping = bad["symbolic_shared"][language]
            mapping[first], mapping[second] = mapping[second], mapping[first]
        mutations.append(bad)
        for maps in mutations:
            with self.subTest(maps=maps), self.assertRaises(ValueError):
                validate_name_maps(self.record, maps)


if __name__ == "__main__":
    unittest.main()

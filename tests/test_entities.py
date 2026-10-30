"""Entity naming and sampling rules, including the largest supported chain."""

import random
import unicodedata
import unittest

from data_generation.entities import (
    ENTITY_TYPES,
    NAME_RULES_VERSION,
    entity_pool,
    sample_entities,
)


class EntityTests(unittest.TestCase):
    def test_declared_rules_version_and_types(self):
        self.assertEqual(NAME_RULES_VERSION, "1")
        self.assertEqual(ENTITY_TYPES, ("turkish_names", "synthetic_names", "symbolic_ids"))

    def test_pools_are_fixed_unique_nfc_and_large_enough(self):
        for entity_type in ENTITY_TYPES:
            with self.subTest(entity_type=entity_type):
                pool = entity_pool(entity_type, 1024)
                self.assertIsInstance(pool, tuple)
                self.assertGreaterEqual(len(pool), 1024)
                self.assertEqual(pool, entity_pool(entity_type, 4))
                self.assertEqual(len(pool), len(set(pool)))
                self.assertTrue(all(value == unicodedata.normalize("NFC", value) for value in pool))

    def test_maximum_size_samples_are_reproducible_unique_nfc(self):
        for entity_type in ENTITY_TYPES:
            with self.subTest(entity_type=entity_type):
                first = sample_entities(entity_type, 1024, random.Random(90210))
                second = sample_entities(entity_type, 1024, random.Random(90210))
                self.assertEqual(first, second)
                self.assertEqual(len(first), 1024)
                self.assertEqual(len(set(first)), 1024)
                self.assertTrue(all(value == unicodedata.normalize("NFC", value) for value in first))
                self.assertTrue(set(first) <= set(entity_pool(entity_type, 4)))
                self.assertNotEqual(first, sample_entities(entity_type, 1024, random.Random(90211)))

    def test_sampling_does_not_touch_global_random_state(self):
        before = random.getstate()
        sample_entities("synthetic_names", 32, random.Random(33))
        self.assertEqual(before, random.getstate())

    def test_person_names_use_complete_unique_full_names(self):
        pool = entity_pool("turkish_names", 1024)
        pairs = [name.split(" ") for name in pool]
        self.assertTrue(all(len(pair) == 2 for pair in pairs))
        self.assertEqual(len({pair[0] for pair in pairs}), 48)
        self.assertEqual(len({pair[1] for pair in pairs}), 32)
        self.assertEqual(len(pool), 48 * 32)

    def test_symbolic_pool_is_fixed_and_formatted(self):
        self.assertEqual(
            entity_pool("symbolic_ids", 1024),
            tuple(f"E{index:04d}" for index in range(2048)),
        )

    def test_synthetic_names_are_cv_cv_and_harmonic(self):
        consonants = set("bcçdghklmnprsştvyz")
        harmony = {
            "a": "aı", "ı": "aı", "o": "au", "u": "au",
            "e": "ei", "i": "ei", "ö": "eü", "ü": "eü",
        }
        first_names = {name.split()[0].translate(str.maketrans({"I": "ı", "İ": "i"})).lower()
                       for name in entity_pool("turkish_names", 4)}
        for name in entity_pool("synthetic_names", 1024):
            self.assertEqual(len(name), 4)
            self.assertIn(name[0].lower(), consonants)
            self.assertEqual(name[0], name[0].upper())
            self.assertIn(name[2], consonants)
            self.assertIn(name[1], harmony)
            self.assertIn(name[3], harmony[name[1]])
            self.assertNotIn(name.lower(), first_names)

    def test_invalid_sizes_raise_value_error_at_both_interfaces(self):
        for entity_type in ENTITY_TYPES:
            for bad in (True, False, None, "4", 4.0, -1, 0, 3, 1025):
                with self.subTest(entity_type=entity_type, n=bad):
                    with self.assertRaises(ValueError):
                        entity_pool(entity_type, bad)
                    with self.assertRaises(ValueError):
                        sample_entities(entity_type, bad, random.Random(1))

    def test_invalid_entity_types_raise_value_error(self):
        for bad in (None, "", "names", 3, True, [], {}):
            with self.subTest(entity_type=bad):
                with self.assertRaises(ValueError):
                    entity_pool(bad, 4)
                with self.assertRaises(ValueError):
                    sample_entities(bad, 4, random.Random(1))


if __name__ == "__main__":
    unittest.main()

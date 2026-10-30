"""Suffix correctness at explicit Turkish morphology boundaries.

These are mechanical fixtures, not a substitute for human linguistic review.
"""

import unicodedata
import unittest

from data_generation.entities import ENTITY_TYPES, entity_pool
from renderers.morphology import (
    MorphologyError,
    ablative,
    genitive,
    possessive_locative,
    turkish_lower,
)


class MorphologyTests(unittest.TestCase):
    def test_turkish_lower_handles_dotted_and_dotless_i(self):
        self.assertEqual(turkish_lower("IİÇĞÖŞÜ"), "ıiçğöşü")
        self.assertEqual(turkish_lower("İREM IŞIL"), "irem ışıl")
        self.assertEqual(turkish_lower("I\u0307rem"), "irem")
        self.assertEqual(turkish_lower(""), "")
        for bad in (None, 1, b"I", True):
            with self.subTest(bad=bad), self.assertRaises(MorphologyError):
                turkish_lower(bad)

    def test_genitive_four_way_harmony_and_vowel_buffer(self):
        pairs = {
            "Can": "Can'ın", "Işıl": "Işıl'ın", "Eren": "Eren'in", "İrem": "İrem'in",
            "Onur": "Onur'un", "Gül": "Gül'ün", "Ömür": "Ömür'ün",
            "Aslı": "Aslı'nın", "Kaya": "Kaya'nın", "Ayşe": "Ayşe'nin",
            "Ali": "Ali'nin", "Banu": "Banu'nun", "Öykü": "Öykü'nün",
            "Bado": "Bado'nun", "Bögö": "Bögö'nün",
        }
        for name, expected in pairs.items():
            with self.subTest(name=name):
                self.assertEqual(genitive(name), expected)

    def test_ablative_two_way_harmony_and_consonant_assimilation(self):
        pairs = {
            "Can": "Can'dan", "Ömer": "Ömer'den", "Aslı": "Aslı'dan",
            "Ayşe": "Ayşe'den", "Zeynep": "Zeynep'ten", "Ahmet": "Ahmet'ten",
            "Çelik": "Çelik'ten", "Kılıç": "Kılıç'tan", "Barış": "Barış'tan",
            "Elif": "Elif'ten", "Enes": "Enes'ten", "Fatih": "Fatih'ten",
            "Koç": "Koç'tan", "Öztürk": "Öztürk'ten", "Dağ": "Dağ'dan",
        }
        for name, expected in pairs.items():
            with self.subTest(name=name):
                self.assertEqual(ablative(name), expected)

    def test_proper_name_roots_never_soften(self):
        for name, expected in (
            ("Ahmet", "Ahmet'in"), ("Çelik", "Çelik'in"),
            ("Öztürk", "Öztürk'ün"), ("Zeynep", "Zeynep'in"),
        ):
            with self.subTest(name=name):
                self.assertEqual(genitive(name), expected)
                self.assertTrue(ablative(name).startswith(name + "'"))

    def test_full_name_uses_the_last_word_and_keeps_casing(self):
        self.assertEqual(genitive("Hüseyin Kaya"), "Hüseyin Kaya'nın")
        self.assertEqual(ablative("Ayşe Öztürk"), "Ayşe Öztürk'ten")
        self.assertEqual(genitive("AYŞE ÇELİK"), "AYŞE ÇELİK'in")
        self.assertEqual(genitive("IŞIL"), "IŞIL'ın")
        self.assertEqual(genitive("İREM"), "İREM'in")

    def test_digit_by_digit_symbolic_convention_all_ten_endings(self):
        expected_genitives = ("ın", "in", "nin", "ün", "ün", "in", "nın", "nin", "in", "un")
        expected_ablatives = ("dan", "den", "den", "ten", "ten", "ten", "dan", "den", "den", "dan")
        for digit, (gen, abl) in enumerate(zip(expected_genitives, expected_ablatives)):
            name = f"E000{digit}"
            with self.subTest(name=name):
                self.assertEqual(genitive(name), f"{name}'{gen}")
                self.assertEqual(ablative(name), f"{name}'{abl}")
        # Whole-number reading of 0010 would yield a different suffix.
        self.assertEqual(genitive("E0010"), "E0010'ın")
        self.assertEqual(ablative("E0010"), "E0010'dan")

    def test_declared_frontness_exception_and_explicit_pronunciation(self):
        self.assertEqual(genitive("Alp"), "Alp'in")
        self.assertEqual(ablative("Alp"), "Alp'ten")
        self.assertEqual(genitive("Ayşe Alp"), "Ayşe Alp'in")
        self.assertEqual(genitive("Kâmil", pronunciation="kamil"), "Kâmil'in")
        self.assertEqual(ablative("Rüzgâr", pronunciation="rüzgar"), "Rüzgâr'dan")
        for function in (genitive, ablative):
            with self.subTest(function=function.__name__), self.assertRaises(MorphologyError):
                function("Kâmil")

    def test_unicode_normalisation_preserves_the_name_except_nfc(self):
        original = "I\u0307rem O\u0308ztu\u0308rk"
        self.assertEqual(genitive(original), "İrem Öztürk'ün")
        self.assertEqual(ablative(original), "İrem Öztürk'ten")
        self.assertEqual(genitive(original), unicodedata.normalize("NFC", genitive(original)))

    def test_declared_common_noun_forms_with_optional_copula(self):
        pairs = {
            "ön": ("önünde", "önündedir"), "arka": ("arkasında", "arkasındadır"),
            "sağ": ("sağında", "sağındadır"), "sol": ("solunda", "solundadır"),
            "üst": ("üstünde", "üstündedir"), "alt": ("altında", "altındadır"),
        }
        for stem, (short, full) in pairs.items():
            with self.subTest(stem=stem):
                self.assertEqual(possessive_locative(stem, copula=False), short)
                self.assertEqual(possessive_locative(stem), full)
                self.assertEqual(possessive_locative(stem, copula=True), full)
                self.assertNotIn("'", full)

    def test_all_fixed_entity_pools_have_deterministic_suffixes(self):
        for entity_type in ENTITY_TYPES:
            for name in entity_pool(entity_type, 1024):
                with self.subTest(entity_type=entity_type, name=name):
                    for function in (genitive, ablative):
                        inflected = function(name)
                        self.assertEqual(inflected, function(name))
                        self.assertTrue(inflected.startswith(name + "'"))
                        self.assertEqual(inflected.count("'"), 1)
                        self.assertEqual(inflected, unicodedata.normalize("NFC", inflected))

    def test_malformed_names_and_unknown_identifier_formats_are_rejected(self):
        invalid = (
            None, 1, True, [], {}, "", " ", " Ayşe", "Ayşe ", "Ayşe  Kaya",
            "Ayşe\tKaya", "Ayşe\nKaya", "Ayşe'nin", "Ayşe’nin", "Ayşe-Kaya",
            "E4", "E00004", "e0004", "E０００４", "E٠٠٠٤", "A0004", "1234",
            "BC", "Bç", "Qwerty", "Max", "Nora!", "Deniz\u200b", "AYŞE🙂",
        )
        for function in (genitive, ablative):
            for bad in invalid:
                with self.subTest(function=function.__name__, bad=bad), self.assertRaises(MorphologyError):
                    function(bad)

    def test_malformed_pronunciations_and_symbolic_overrides_are_rejected(self):
        for function in (genitive, ablative):
            for hint in ("", " ", " alp", "alp ", "al  p", "abc1", "xyz", "hâle", 1, False):
                with self.subTest(function=function.__name__, hint=hint), self.assertRaises(MorphologyError):
                    function("Alp", pronunciation=hint)
            with self.subTest(function=function.__name__), self.assertRaises(MorphologyError):
                function("E0004", pronunciation="dört")

    def test_unknown_stems_and_non_bool_copula_are_rejected(self):
        for bad in (None, 1, [], {}, "", "ÖN", "ön ", "yan", "üstünde", "o\u0308n"):
            with self.subTest(stem=bad), self.assertRaises(MorphologyError):
                possessive_locative(bad)
        for bad in (None, 0, 1, "True", [], {}):
            with self.subTest(copula=bad), self.assertRaises(MorphologyError):
                possessive_locative("ön", copula=bad)


if __name__ == "__main__":
    unittest.main()

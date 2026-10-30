"""Zero-shot adaptation preserves every frozen linguistic design coordinate."""

from copy import deepcopy
import hashlib
import json
from pathlib import Path
import unittest

from inference.prompts import PROMPT_VERSION, SYSTEM_MESSAGES, make_messages, messages_sha256


PROJECT_ROOT = Path(__file__).resolve().parents[1]


class InferencePromptTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.rows = [json.loads(line) for line in
                    (PROJECT_ROOT / "data/complete_grid/pilot_preview.jsonl").read_text().splitlines()]

    @staticmethod
    def reseal(row):
        row["prompt_sha256"] = hashlib.sha256(row["prompt"].encode("utf-8")).hexdigest()
        return row

    def test_all_243_real_coordinates_preserve_context_premises_and_question(self):
        self.assertEqual(len(self.rows), 243)
        self.assertEqual(len({r["relation_id"] for r in self.rows}), 9)
        self.assertEqual(len({(r["language"], r["condition"]) for r in self.rows}), 9)
        self.assertEqual(len({r["name_control"] for r in self.rows}), 3)
        for row in self.rows:
            with self.subTest(design=row["grid_design_id"]):
                before = deepcopy(row)
                result = make_messages(row)
                language = row["language"]
                marker = "\nSoru: " if language == "tr" else "\nQuestion: "
                old_body, old_question = row["prompt"].split(marker)
                old_question = old_question.split("\n")[0]
                user = result["messages"][1]["content"]
                self.assertEqual(user.split("\n\n")[0], old_body)
                self.assertEqual(user.split("\n\n")[1], marker[1:] + old_question)
                self.assertEqual(result["messages"][0], {"role": "system", "content": SYSTEM_MESSAGES[language]})
                self.assertEqual(result["messages"][1]["role"], "user")
                self.assertEqual(result["source_prompt_sha256"], row["prompt_sha256"])
                self.assertEqual(result["prompt_version"], PROMPT_VERSION)
                self.assertEqual(result["messages_sha256"], messages_sha256(result["messages"]))
                self.assertNotEqual(result["messages_sha256"], row["prompt_sha256"])
                self.assertEqual(row, before)

    def test_exact_requested_turkish_instruction_and_equivalent_english_instruction(self):
        tr = next(row for row in self.rows if row["language"] == "tr")
        en = next(row for row in self.rows if row["language"] == "en")
        self.assertEqual(make_messages(tr)["messages"][0]["content"],
                         "Sen mantıksal sıralama sorularını yanıtlayan bir asistansın. "
                         "Verilen öncülleri temel al ve yalnızca istenen cevabı üret.")
        self.assertTrue(make_messages(tr)["messages"][1]["content"].endswith("\n\nYalnızca EVET veya HAYIR yaz."))
        self.assertTrue(make_messages(en)["messages"][1]["content"].endswith("\n\nWrite only YES or NO."))
        self.assertNotIn("Yanıtınızı yalnız YES", make_messages(tr)["messages"][1]["content"])

    def test_gold_label_and_latent_metadata_cannot_leak_into_message_construction(self):
        row = deepcopy(self.rows[0])
        baseline = make_messages(row)
        row.update(gold_label=not row["gold_label"], latent_order=["SECRET_GOLD_ORDER"],
                   query_subject="SECRET_TARGET", query_object="SECRET_TARGET_2",
                   query_relation="after", latent_instance_id="SECRET_ID")
        self.assertEqual(make_messages(row), baseline)
        for key in ("gold_label", "latent_order", "query_subject", "query_object"):
            row.pop(key)
        self.assertEqual(make_messages(row), baseline)

    def test_message_hash_uses_stable_utf8_canonical_json(self):
        messages = [{"role": "user", "content": "Çığ öşü"}]
        expected = hashlib.sha256(b'[{"content":"' + "Çığ öşü".encode() + b'","role":"user"}]').hexdigest()
        self.assertEqual(messages_sha256(messages), expected)
        self.assertEqual(messages_sha256([{"content": "Çığ öşü", "role": "user"}]), expected)

    def test_missing_corrupted_and_uppercase_source_hashes_are_rejected(self):
        for value in (None, "0" * 64, self.rows[0]["prompt_sha256"].upper()):
            row = deepcopy(self.rows[0])
            row["prompt_sha256"] = value
            with self.assertRaises(ValueError):
                make_messages(row)
        row = deepcopy(self.rows[0]); row["prompt"] += " "
        with self.assertRaises(ValueError):
            make_messages(row)

    def test_extra_or_wrong_language_question_markers_are_rejected_after_resealing(self):
        for suffix in ("\nQuestion: Is this a second question?", "\nSoru: İkinci soru mu?"):
            row = deepcopy(self.rows[0])
            row["prompt"] = row["prompt"].replace("\nProvide your answer", suffix + "\nProvide your answer")
            with self.assertRaises(ValueError):
                make_messages(self.reseal(row))

    def test_different_old_instruction_extra_output_and_crlf_are_rejected(self):
        source = self.rows[0]
        for text in (source["prompt"] + "\nextra", source["prompt"].replace("YES or NO.", "TRUE or FALSE."),
                     source["prompt"].replace("\n", "\r\n"), source["prompt"] + "\n"):
            row = deepcopy(source); row["prompt"] = text
            with self.assertRaises(ValueError):
                make_messages(self.reseal(row))

    def test_inventory_and_premise_cardinalities_are_checked(self):
        for n in (True, None, 3, 1025, 5):
            row = deepcopy(self.rows[0]); row["n"] = n
            with self.assertRaises(ValueError):
                make_messages(row)
        row = deepcopy(self.rows[0]); lines = row["prompt"].splitlines()
        first_premise = next(i for i, line in enumerate(lines) if line.startswith("- "))
        lines[first_premise] = "An unframed premise."
        row["prompt"] = "\n".join(lines)
        with self.assertRaises(ValueError):
            make_messages(self.reseal(row))

    def test_unknown_languages_or_wrong_input_types_are_rejected(self):
        for value in (None, [], "prompt"):
            with self.assertRaises(ValueError):
                make_messages(value)
        for language in ("TR", "fr", None, []):
            row = deepcopy(self.rows[0]); row["language"] = language
            with self.assertRaises(ValueError):
                make_messages(row)


if __name__ == "__main__":
    unittest.main()

"""Reasoning instructions never change the frozen logical problem surface."""

from copy import deepcopy
import hashlib
import json
from pathlib import Path
import unittest

from reasoning.parser import PARSER_VERSIONS, STRATEGIES
from reasoning.prompts import INSTRUCTIONS, PROMPT_VERSIONS, SYSTEM_MESSAGES, instruction_overhead_utf8_bytes, make_messages
from inference import prompts as direct


PROJECT = Path(__file__).resolve().parents[1]


class ReasoningPromptTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.rows = [json.loads(line) for line in
                    (PROJECT / "data/complete_grid/pilot_preview.jsonl").read_text().splitlines()]

    def test_all_243_source_coordinates_preserve_identical_problem_frames(self):
        self.assertEqual(len(self.rows), 243)
        for row in self.rows:
            before = deepcopy(row)
            baseline = direct.make_messages(row)
            problem = baseline["messages"][1]["content"].rsplit("\n\n", 1)[0]
            for strategy in STRATEGIES:
                with self.subTest(design=row["grid_design_id"], strategy=strategy):
                    built = make_messages(row, strategy)
                    self.assertEqual(built["strategy"], strategy)
                    self.assertEqual(built["parser_version"], PARSER_VERSIONS[strategy])
                    self.assertEqual(built["source_prompt_sha256"], baseline["source_prompt_sha256"])
                    self.assertEqual(built["messages_sha256"], direct.messages_sha256(built["messages"]))
                    self.assertEqual(built["prompt_version"], PROMPT_VERSIONS[strategy])
                    if strategy in ("direct", "provider_effort"):
                        for key, value in baseline.items():
                            self.assertEqual(built[key], value)
                    else:
                        self.assertEqual(built["messages"][1]["content"], problem + "\n\n" + INSTRUCTIONS[strategy][row["language"]])
                        self.assertEqual(built["messages"][0]["content"], SYSTEM_MESSAGES[row["language"]])
                        self.assertNotEqual(built["messages_sha256"], baseline["messages_sha256"])
                    self.assertEqual(row, before)

    def test_direct_and_native_effort_messages_are_byte_identical(self):
        for row in self.rows:
            direct_built = make_messages(row, "direct")
            effort_built = make_messages(row, "provider_effort")
            for key in ("messages", "messages_sha256", "source_prompt_sha256", "prompt_version", "parser_version"):
                self.assertEqual(direct_built[key], effort_built[key])
            self.assertEqual(direct_built["prompt_version"], direct.PROMPT_VERSION)

    def test_localized_final_marker_and_structured_steps_are_explicit(self):
        for language, yes, no in (("tr", "EVET", "HAYIR"), ("en", "YES", "NO")):
            row = next(row for row in self.rows if row["language"] == language)
            for strategy in ("zero_shot_reasoning", "structured_reasoning"):
                instruction = INSTRUCTIONS[strategy][language]
                built = make_messages(row, strategy)
                self.assertIn("FINAL: " + yes, instruction)
                self.assertIn("FINAL: " + no, instruction)
                self.assertTrue(built["messages"][1]["content"].endswith(instruction))
                self.assertNotEqual(built["messages"][0]["content"], direct.SYSTEM_MESSAGES[language])
            structured = INSTRUCTIONS["structured_reasoning"][language]
            for step in range(1, 5):
                self.assertIn(str(step) + ". ", structured)
            self.assertNotEqual(INSTRUCTIONS["zero_shot_reasoning"][language], structured)

    def test_gold_latent_and_query_metadata_are_never_used_or_leaked(self):
        for strategy in STRATEGIES:
            row = deepcopy(self.rows[0])
            original = make_messages(row, strategy)
            row.update(gold_label="SECRET_GOLD", latent_order=["SECRET_LATENT"],
                       query_subject="SECRET_SUBJECT", query_object="SECRET_OBJECT", query_relation="SECRET_RELATION",
                       latent_instance_id="SECRET_ID", n_hidden="SECRET_N")
            self.assertEqual(make_messages(row, strategy), original)
            for key in ("gold_label", "latent_order", "query_subject", "query_object", "query_relation"):
                row.pop(key)
            self.assertEqual(make_messages(row, strategy), original)
            self.assertNotIn("SECRET", json.dumps(original, ensure_ascii=False))

    def test_only_the_four_source_fields_are_required(self):
        for source in self.rows[:3]:
            minimal = {key: source[key] for key in ("language", "n", "prompt", "prompt_sha256")}
            for strategy in STRATEGIES:
                self.assertEqual(make_messages(minimal, strategy), make_messages(source, strategy))

    def test_source_validation_is_preserved_in_every_strategy(self):
        for strategy in STRATEGIES:
            for key, value in (("prompt_sha256", "0" * 64), ("language", "TR"), ("n", True), ("n", 3), ("prompt", "corrupted")):
                row = deepcopy(self.rows[0])
                row[key] = value
                with self.subTest(strategy=strategy, key=key), self.assertRaises(ValueError):
                    make_messages(row, strategy)
            row = deepcopy(self.rows[0])
            row["prompt"] = row["prompt"].replace("\n", "\r\n")
            row["prompt_sha256"] = hashlib.sha256(row["prompt"].encode()).hexdigest()
            with self.assertRaises(ValueError):
                make_messages(row, strategy)

    def test_bad_types_and_unregistered_strategy_names_are_rejected(self):
        for strategy in ("A", "B", "reasoning", "DIRECT", None, [], True, ""):
            with self.subTest(strategy=strategy), self.assertRaises(ValueError):
                make_messages(self.rows[0], strategy)
        for strategy in STRATEGIES:
            for row in (None, [], "text"):
                with self.subTest(strategy=strategy, row=row), self.assertRaises(ValueError):
                    make_messages(row, strategy)

    def test_utf8_instruction_overhead_matches_actual_changed_message_bytes(self):
        for language in ("tr", "en"):
            row = next(row for row in self.rows if row["language"] == language)
            old = direct.make_messages(row)
            old_size = sum(len(message["content"].encode()) for message in old["messages"])
            for strategy in STRATEGIES:
                built = make_messages(row, strategy)
                size = sum(len(message["content"].encode()) for message in built["messages"])
                overhead = instruction_overhead_utf8_bytes(language, strategy)
                self.assertEqual(overhead, max(0, size - old_size))
                if strategy in ("direct", "provider_effort"):
                    self.assertEqual(overhead, 0)
                else:
                    self.assertGreater(overhead, 0)

    def test_overhead_helper_validates_language_and_strategy(self):
        for language, strategy in (("TR", "direct"), (None, "direct"), ("en", "B"), ("tr", [])):
            with self.subTest(language=language, strategy=strategy), self.assertRaises(ValueError):
                instruction_overhead_utf8_bytes(language, strategy)

    def test_repeated_construction_is_deterministic_and_outputs_are_fresh(self):
        row = self.rows[0]
        for strategy in STRATEGIES:
            first = make_messages(row, strategy)
            second = make_messages(row, strategy)
            self.assertEqual(first, second)
            first["messages"][0]["content"] = "synthetic corruption"
            self.assertNotEqual(first, second)
            self.assertEqual(make_messages(row, strategy), second)


if __name__ == "__main__":
    unittest.main()

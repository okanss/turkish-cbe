"""Pure encoding/scheduling checks; no model or optimization is executed."""

import unittest

from training.engine import encode_supervision, sft_batches, TrainingBlocked


class CharacterTokenizer:
    def apply_chat_template(self, messages, *, tokenize, add_generation_prompt, **options):
        text = "".join(row["role"]+":"+row["content"]+"|" for row in messages)
        if add_generation_prompt: text += "assistant:"
        return [ord(character) for character in text]


class TrainingEncodingTests(unittest.TestCase):
    def setUp(self):
        self.tokenizer = CharacterTokenizer()
        self.messages = [{"role": "user", "content": "fixture"}]

    def test_complete_multitoken_answer_and_terminal_frame_are_supervised(self):
        encoded = encode_supervision(self.tokenizer, self.messages, "HAYIR", 100)
        prefix = encoded["prompt_tokens"]
        self.assertEqual(encoded["labels"][:prefix], [-100]*prefix)
        self.assertEqual(encoded["labels"][prefix:], list(map(ord, "HAYIR|")))
        self.assertEqual(encoded["completion_tokens"], 6)
        self.assertEqual(len(encoded["input_ids"]), len(encoded["labels"]))

    def test_invalid_tokenizer_prefix_is_rejected(self):
        class Broken:
            def apply_chat_template(self, messages, *, add_generation_prompt, **kwargs):
                return [1, 2] if add_generation_prompt else [3, 4, 5]
        with self.assertRaisesRegex(ValueError, "prefix"):
            encode_supervision(Broken(), self.messages, "EVET", 100)

    def test_long_input_is_never_truncated_or_dropped(self):
        with self.assertRaisesRegex(TrainingBlocked, "no truncation"):
            encode_supervision(self.tokenizer, self.messages, "EVET", 5)

    def test_bad_answer_and_sequence_budget_are_rejected(self):
        for answer, length in (("", 100), (None, 100), ("EVET", True), ("EVET", 0)):
            with self.assertRaises(ValueError): encode_supervision(self.tokenizer, self.messages, answer, length)


class TrainingScheduleTests(unittest.TestCase):
    def test_epoch_mode_keeps_short_last_batch_and_exact_unique_pool(self):
        batches = list(sft_batches(list(range(19)), paired=False, effective_batch_size=16, epochs=1, max_steps=0, seed=42))
        self.assertEqual([len(batch) for batch in batches], [16, 3])
        self.assertEqual(sorted(value for batch in batches for value in batch), list(range(19)))

    def test_paired_units_never_split(self):
        batches = list(sft_batches(list(range(19)), paired=True, effective_batch_size=16, epochs=1, max_steps=0, seed=42))
        self.assertEqual([len(batch) for batch in batches], [8, 8, 3])
        self.assertEqual(sum(len(batch)*2 for batch in batches), 38)

    def test_step_control_has_explicit_repeated_exposures(self):
        batches = list(sft_batches(list(range(4)), paired=True, effective_batch_size=16, epochs=1, max_steps=3, seed=42))
        self.assertEqual(len(batches), 3)
        self.assertEqual(sum(len(batch)*2 for batch in batches), 48)
        self.assertEqual(len(set(value for batch in batches for value in batch)), 4)

    def test_schedule_is_deterministic_by_seed(self):
        def schedule(seed):
            return list(sft_batches(list(range(40)), paired=False, effective_batch_size=16, epochs=2, max_steps=0, seed=seed))
        self.assertEqual(schedule(42), schedule(42))
        self.assertNotEqual(schedule(42), schedule(43))

    def test_empty_duplicate_and_pair_splitting_inputs_rejected(self):
        for ids, size in (([], 16), ([1, 1], 16), ([1, 2], 3)):
            with self.assertRaises(ValueError):
                list(sft_batches(ids, paired=True, effective_batch_size=size, epochs=1, max_steps=0, seed=42))


class TrainingPolicyContractTests(unittest.TestCase):
    def setUp(self):
        self.config = {"grpo": {"temperature": .6, "top_p": 1.0},
                       "budget": {"output_tokens": 32, "max_wall_seconds": 10}}

    def test_full_support_sampling_disables_top_k_and_fixes_eos(self):
        from training.engine import grpo_sampling_options
        settings = grpo_sampling_options(self.config, {9, 3}, 3)
        self.assertEqual(settings["temperature"], .6)
        self.assertEqual(settings["top_p"], 1)
        self.assertEqual(settings["top_k"], 0)
        self.assertEqual(settings["eos_token_id"], [3, 9])
        self.assertEqual(settings["max_new_tokens"], 32)

    def test_nucleus_filter_or_bad_temperature_cannot_change_behavior_policy(self):
        from copy import deepcopy
        from training.engine import grpo_sampling_options
        for key, value in (("top_p", .95), ("temperature", 0), ("temperature", float("nan"))):
            config = deepcopy(self.config); config["grpo"][key] = value
            with self.assertRaises(ValueError): grpo_sampling_options(config, {3}, 3)

    def test_generation_config_starts_with_neutral_penalties_and_no_forced_tokens(self):
        from transformers import GenerationConfig
        from training.engine import grpo_sampling_options
        generation = GenerationConfig(**grpo_sampling_options(self.config, {3}, 3))
        # transformers>=5 reports unset neutral options as None instead of
        # their neutral numeric values; both spellings disable the option.
        self.assertIn(generation.repetition_penalty, (None, 1))
        self.assertIn(generation.typical_p, (None, 1))
        self.assertIn(generation.epsilon_cutoff, (None, 0))
        self.assertIn(generation.eta_cutoff, (None, 0))
        for field in ("min_p", "forced_bos_token_id", "forced_eos_token_id", "bad_words_ids", "suppress_tokens", "sequence_bias"):
            self.assertIsNone(getattr(generation, field))

    def test_missing_terminal_token_contract_is_rejected(self):
        from training.engine import grpo_sampling_options
        for eos, pad in ((set(), 3), ({3}, None)):
            with self.assertRaises(ValueError): grpo_sampling_options(self.config, eos, pad)

    def test_expired_step_boundary_raises_instead_of_completing(self):
        from unittest.mock import patch
        from training.engine import _check_wall_time
        with patch("training.engine.time.monotonic", return_value=111):
            with self.assertRaisesRegex(TrainingBlocked, "wall_time_budget"):
                _check_wall_time(100, self.config)

    def test_exact_deadline_boundary_is_allowed(self):
        from unittest.mock import patch
        from training.engine import _check_wall_time
        with patch("training.engine.time.monotonic", return_value=110):
            _check_wall_time(100, self.config)


import importlib.util
if importlib.util.find_spec("torch") is not None:
    class TorchPolicyProbabilityTests(unittest.TestCase):
        def test_temperature_scaled_token_likelihood_matches_behavior_policy_and_gradients(self):
            import torch
            from types import SimpleNamespace
            from training.engine import sequence_logps
            class TinyPolicy(torch.nn.Module):
                def __init__(self):
                    super().__init__()
                    self.scores = torch.nn.Parameter(torch.tensor([[[0., 1.], [2., 0.], [0., 0.]]]))
                def forward(self, **kwargs):
                    return SimpleNamespace(logits=self.scores)
            model = TinyPolicy()
            actual = sequence_logps(model, torch, [0, 1, 0], 1, temperature=.6)
            expected = torch.log_softmax(model.scores[:, :2] / .6, -1)[0, [0, 1], [1, 0]]
            self.assertTrue(torch.allclose(actual[0], expected))
            self.assertFalse(torch.allclose(actual, sequence_logps(model, torch, [0, 1, 0], 1)))
            actual.sum().backward()
            self.assertGreater(model.scores.grad.abs().sum().item(), 0)

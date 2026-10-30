"""Synthetic objective audits. No model, optimizer, weights or network is used.

Pure-Python reference tests always run. Torch numerical/gradient fixtures are
defined only when that optional backend is present; its absence is explicitly
tested as an error, never substituted with another backend or reported as a
successful gradient test. The delivery report records backend coverage.
"""

from copy import deepcopy
import importlib.util
import math
import unittest
from unittest.mock import patch

from training.objectives import (NORMALIZATION, advantages, agreement_diagnostics,
    binary_js_loss, binary_js_reference, grpo_loss, grpo_reference_loss, reward_components)


TORCH_NUMERICAL_TESTS_AVAILABLE = importlib.util.find_spec("torch") is not None


def sample(prediction, *, presentation="p0", instance="i0", query="q0", gold=True, truncated=False):
    return {"instance_id": instance, "query_id": query, "presentation_id": presentation,
            "prediction": prediction, "gold": gold, "truncated": truncated}


def binary_group(wrong_majority=False):
    first = [True, False, False, False] if wrong_majority else [True, True, True, False]
    second = [False] * 4 if wrong_majority else [True] * 4
    return [sample(value) for value in first] + [sample(value, presentation="p1") for value in second]


def logs(values):
    return [math.log(value) for value in values]


class TrainingRewardTests(unittest.TestCase):
    def test_pooled_q_and_positive_clipped_bonus_match_registered_formula(self):
        rows = binary_group()
        result = reward_components(rows)
        for row in result:
            self.assertEqual(row["q"], 7 / 8)
            self.assertEqual(row["chance"], (7 / 8) ** 2 + (1 / 8) ** 2)
            self.assertEqual(row["group_valid"], 8)
            self.assertEqual(row["other_valid"], 4)
        self.assertEqual(result[0]["agreement"], 1)
        self.assertEqual(result[0]["bonus"], .21875)
        self.assertEqual(result[0]["correctness"], 1)
        self.assertEqual(result[0]["reward"], 1.109375)
        self.assertEqual(result[3]["bonus"], 0)
        self.assertEqual(result[3]["reward"], 0)
        self.assertEqual(result[4]["agreement"], .75)
        self.assertEqual(result[4]["bonus"], 0)

    def test_same_presentation_repetitions_do_not_count_as_cross_presentation_agreement(self):
        rows = [sample(True), sample(True), sample(False, presentation="p1")]
        result = reward_components(rows)
        self.assertEqual(result[0]["other_valid"], 1)
        self.assertEqual(result[0]["agreement"], 0)
        self.assertEqual(result[2]["other_valid"], 2)
        self.assertEqual(result[2]["agreement"], 0)

    def test_invalid_and_truncated_are_zero_and_excluded_from_pooled_bias(self):
        rows = [sample(True), sample(None, presentation="p1"), sample(False, presentation="p1", truncated=True)]
        result = reward_components(rows)
        self.assertEqual(result[0]["q"], 1)
        self.assertEqual(result[0]["other_valid"], 0)
        self.assertIsNone(result[0]["agreement"])
        for row in result[1:]:
            self.assertFalse(row["valid"])
            self.assertEqual((row["correctness"], row["bonus"], row["reward"]), (0, 0, 0))

    def test_no_valid_group_has_undefined_q_and_chance_not_fake_zero(self):
        result = reward_components([sample(None), sample(True, presentation="p1", truncated=True)])
        for row in result:
            self.assertIsNone(row["q"])
            self.assertIsNone(row["chance"])
            self.assertEqual(row["reward"], 0)
            self.assertEqual(row["group_valid"], 0)

    def test_single_presentation_or_no_valid_counterpart_cannot_earn_bonus(self):
        result = reward_components([sample(True), sample(False)])
        self.assertTrue(all(row["other_valid"] == 0 and row["bonus"] == 0 for row in result))

    def test_instances_and_queries_have_independent_bias_groups(self):
        rows = [sample(True), sample(True, presentation="p1"),
                sample(False, query="q1"), sample(False, presentation="p1", query="q1"),
                sample(False, instance="i1", gold=False), sample(False, presentation="p1", instance="i1", gold=False)]
        result = reward_components(rows)
        self.assertEqual([row["q"] for row in result], [1, 1, 0, 0, 0, 0])
        self.assertEqual([row["correctness"] for row in result], [1, 1, 0, 0, 1, 1])

    def test_shared_wrong_constant_answers_have_no_bonus_or_update(self):
        rows = [sample(False), sample(False, presentation="p1"), sample(False, presentation="p2")]
        result = reward_components(rows)
        self.assertTrue(all(row["q"] == 0 and row["chance"] == 1 and row["reward"] == 0 for row in result))
        diagnostic = agreement_diagnostics(rows)
        self.assertTrue(diagnostic["all_zero_correctness_update"])
        self.assertTrue(diagnostic["all_zero_agreement_update"])
        self.assertIsNone(diagnostic["cosine"])
        self.assertFalse(diagnostic["accuracy_improvement_measured"])

    def test_agreement_weight_zero_retains_correctness_rewards(self):
        result = reward_components(binary_group(), agreement_weight=0)
        self.assertTrue(any(row["bonus"] > 0 for row in result))
        self.assertTrue(all(row["reward"] == row["correctness"] for row in result))

    def test_mixed_gold_or_malformed_samples_fail_before_rewards(self):
        with self.assertRaisesRegex(ValueError, "mix gold"):
            reward_components([sample(True), sample(False, gold=False)])
        for key, value in (("gold", 1), ("truncated", 0), ("prediction", 1),
                           ("instance_id", ""), ("query_id", True), ("presentation_id", [])):
            row = sample(True)
            row[key] = value
            with self.subTest(key=key), self.assertRaises(ValueError):
                reward_components([row])
        for value in (None, "rows", {}, [None], [{"prediction": True}]):
            with self.subTest(value=value), self.assertRaises(ValueError):
                reward_components(value)
        for weight in (True, -1, float("nan"), float("inf"), "0.5"):
            with self.subTest(weight=weight), self.assertRaises(ValueError):
                reward_components([], agreement_weight=weight)

    def test_input_and_results_are_fresh_without_backend_import(self):
        rows = binary_group()
        before = deepcopy(rows)
        with patch("training.objectives.importlib.import_module", side_effect=AssertionError("no backend import")):
            first = reward_components(rows)
            second = reward_components(rows)
            advantages(rows, [row["reward"] for row in first])
            agreement_diagnostics(rows)
        self.assertEqual(rows, before)
        self.assertEqual(first, second)
        first[0]["reward"] = 99
        self.assertNotEqual(first, second)


class TrainingAdvantageTests(unittest.TestCase):
    def test_prompt_normalization_uses_population_std_plus_epsilon(self):
        rows = [sample(True), sample(False), sample(False), sample(False)]
        values = advantages(rows, [1, 0, 0, 0], epsilon=0)
        self.assertAlmostEqual(values[0], math.sqrt(3))
        for value in values[1:]:
            self.assertAlmostEqual(value, -1 / math.sqrt(3))
        self.assertAlmostEqual(math.fsum(values), 0)
        self.assertIn("population_std", NORMALIZATION)

    def test_invalid_zero_reward_is_retained_in_advantage_denominator(self):
        rows = [sample(True), sample(False), sample(None)]
        result = advantages(rows, [1, 0, 0], epsilon=0)
        self.assertAlmostEqual(result[0], math.sqrt(2))
        self.assertAlmostEqual(result[1], -1 / math.sqrt(2))
        self.assertEqual(result[1], result[2])
        with self.assertRaisesRegex(ValueError, "zero rewards"):
            advantages(rows, [1, 0, .1])

    def test_instance_centered_is_explicitly_unscaled_and_cross_presentation(self):
        rows = [sample(True), sample(False), sample(True, presentation="p1"), sample(True, presentation="p1")]
        self.assertEqual(advantages(rows, [1, 0, 1, 1], mode="instance_centered"), [.25, -.75, .25, .25])
        self.assertEqual(advantages(rows, [1, 0, 1, 1], epsilon=0), [1, -1, 0, 0])

    def test_appendix_g_positive_bonus_is_inert_under_binary_prompt_normalization(self):
        for wrong_majority in (False, True):
            rows = binary_group(wrong_majority)
            baseline = [r["reward"] for r in reward_components(rows, agreement_weight=0)]
            agreement = [r["reward"] for r in reward_components(rows)]
            self.assertNotEqual(baseline, agreement)
            left = advantages(rows, baseline, epsilon=0)
            right = advantages(rows, agreement, epsilon=0)
            for a, b in zip(left, right):
                self.assertAlmostEqual(a, b)
            diagnostic = agreement_diagnostics(rows, epsilon=0)
            self.assertGreater(diagnostic["n_positive_bonuses"], 0)
            self.assertTrue(diagnostic["prompt_advantage_direction_invariant"])
            self.assertAlmostEqual(diagnostic["cosine"], 1)
            self.assertLess(diagnostic["max_absolute_advantage_difference"], 1e-12)
            self.assertFalse(diagnostic["accuracy_improvement_measured"])

    def test_nonzero_epsilon_exposes_small_rescaling_not_new_binary_direction(self):
        result = agreement_diagnostics(binary_group())
        self.assertTrue(result["epsilon_only_rescaling"])
        self.assertTrue(result["prompt_advantage_direction_invariant"])
        self.assertGreater(result["max_absolute_advantage_difference"], 0)
        self.assertLess(result["max_absolute_advantage_difference"], .001)
        self.assertAlmostEqual(result["cosine"], 1)

    def test_instance_centering_can_change_signal_and_is_a_new_control(self):
        rows = binary_group()
        result = agreement_diagnostics(rows, mode="instance_centered")
        self.assertFalse(result["binary_prompt_invariance_scope"])
        self.assertGreater(result["max_absolute_advantage_difference"], .01)
        self.assertLess(result["cosine"], 1)
        self.assertIn("unscaled", result["normalization"])

    def test_invalid_outputs_break_binary_invariance_scope_explicitly(self):
        rows = [sample(True), sample(False), sample(None)] + [sample(False, presentation="p1")] * 4
        result = agreement_diagnostics(rows)
        self.assertFalse(result["binary_prompt_invariance_scope"])
        self.assertFalse(result["prompt_advantage_direction_invariant"])

    def test_constant_singleton_and_empty_groups_have_no_update(self):
        self.assertEqual(advantages([], []), [])
        self.assertEqual(advantages([sample(True)], [1]), [0])
        self.assertEqual(advantages([sample(True), sample(True)], [1, 1], epsilon=0), [0, 0])

    def test_bad_advantage_inputs_and_nonfinite_values_are_rejected(self):
        rows = [sample(True)]
        for kwargs in ({"mode": "instance"}, {"mode": []}, {"epsilon": -1}, {"epsilon": float("nan")}, {"epsilon": True}):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                advantages(rows, [1], **kwargs)
        for values in ([], [True], [float("inf")], [float("nan")], "rewards"):
            with self.subTest(values=values), self.assertRaises(ValueError):
                advantages(rows, values)


class TrainingReferenceLossTests(unittest.TestCase):
    def test_binary_js_reference_matches_manual_information_formula(self):
        left, right = [.2, .8], [.4, .6]
        midpoint = [(a + b) / 2 for a, b in zip(left, right)]
        expected = .5 * sum(p * math.log(p / m) for p, m in zip(left, midpoint))
        expected += .5 * sum(q * math.log(q / m) for q, m in zip(right, midpoint))
        value, diagnostic = binary_js_reference([logs(left)], [logs(right)], [1], .7)
        self.assertAlmostEqual(value, .7 * expected)
        self.assertEqual(diagnostic["active_pairs"], 1)
        self.assertIn("whole_answer_sequences", diagnostic["probability_scope"])

    def test_js_guard_is_strict_and_zero_guard_preserves_primary_ce_scope(self):
        for left, right in (([.5, .5], [.2, .8]), ([.1, .9], [.8, .2])):
            value, diagnostic = binary_js_reference([logs(left)], [logs(right)], [1])
            self.assertEqual(value, 0)
            self.assertEqual(diagnostic["active_pairs"], 0)

    def test_js_is_batch_mean_guarded_not_active_pair_mean(self):
        first, _ = binary_js_reference([logs([.2, .8])], [logs([.4, .6])], [1])
        batch, diagnostic = binary_js_reference([logs([.2, .8]), logs([.9, .1])],
                                              [logs([.4, .6]), logs([.1, .9])], [1, 1])
        self.assertAlmostEqual(batch, first / 2)
        self.assertEqual(diagnostic["active_pairs"], 1)

    def test_complete_answer_normalization_does_not_measure_invalid_mass(self):
        value, _ = binary_js_reference([[-7, -8]], [[-1007, -1008]], [0])
        self.assertAlmostEqual(value, 0, places=12)
        value, _ = binary_js_reference([[0, -1000]], [[0, -1001]], [0])
        self.assertTrue(math.isfinite(value))

    def test_js_bad_shapes_labels_and_nonfinite_inputs_abort(self):
        for left, right, gold in (([], [], []), ([[1, 2, 3]], [[1, 2]], [1]),
                                  ([[1, 2]], [[1, 2]], [2]), ([[1, 2]], [[1, 2]], [1.0]),
                                  ([[float("nan"), 0]], [[1, 2]], [1])):
            with self.subTest(left=left, gold=gold), self.assertRaises(ValueError):
                binary_js_reference(left, right, gold)

    def test_clipped_pg_both_advantage_signs_match_reference(self):
        value, diagnostic = grpo_reference_loss([[math.log(1.5)], [math.log(.5)]], [[0], [0]], [[1], [1]], [1, -1])
        self.assertAlmostEqual(value, -.2)
        self.assertEqual(diagnostic["clip_fraction"], 1)
        self.assertEqual(diagnostic["beta"], 0)

    def test_per_sequence_masked_mean_is_not_token_pooled_mean(self):
        value, diagnostic = grpo_reference_loss([[0, 0, 0], [0, 0, 0]], [[0, 0, 0], [0, 0, 0]],
                                               [[1, 1, 1], [1, 0, 0]], [1, 3])
        self.assertEqual(value, -2)
        self.assertEqual(diagnostic["n_tokens"], 4)
        self.assertEqual(diagnostic["reduction"], "mean_of_per_sequence_masked_token_means")

    def test_zero_advantage_has_zero_loss_and_masked_tokens_have_no_contribution(self):
        self.assertEqual(grpo_reference_loss([[.3, .8]], [[.2, -.2]], [[1, 0]], [0])[0], 0)
        left = grpo_reference_loss([[0, 0]], [[0, 0]], [[1, 0]], [1])[0]
        right = grpo_reference_loss([[0, 1]], [[0, -1]], [[1, 0]], [1])[0]
        self.assertEqual(left, right)

    def test_bad_masks_shapes_nonfinite_and_overflow_abort(self):
        for new, old, mask, advantage in (([[0]], [[0]], [[0]], [1]), ([[0]], [[0]], [[.5]], [1]),
            ([[0]], [[0]], [[1]], []), ([[0, 0]], [[0]], [[1]], [1]),
            ([[float("inf")]], [[0]], [[1]], [1]), ([[1000]], [[0]], [[1]], [1])):
            with self.subTest(new=new, mask=mask), self.assertRaises(ValueError):
                grpo_reference_loss(new, old, mask, advantage)
        with self.assertRaises(ValueError):
            grpo_reference_loss([[0]], [[0]], [[1]], [1], epsilon=-.2)

    def test_missing_torch_backend_is_an_explicit_error_without_math_fallback(self):
        with patch("training.objectives.importlib.import_module", side_effect=ImportError("synthetic missing Torch")):
            with self.assertRaisesRegex(RuntimeError, "Torch objective backend unavailable"):
                binary_js_loss(None, None, None, .5)
            with self.assertRaisesRegex(RuntimeError, "Torch objective backend unavailable"):
                grpo_loss(None, None, None, None)


if TORCH_NUMERICAL_TESTS_AVAILABLE:
    class TorchTrainingObjectiveTests(unittest.TestCase):
        @classmethod
        def setUpClass(cls):
            import torch
            cls.torch = torch

        def tensor(self, values, *, grad=False):
            return self.torch.tensor(values, dtype=self.torch.float64, requires_grad=grad)

        def test_binary_js_matches_reference_and_backpropagates_both_presentations(self):
            left, right = self.tensor([logs([.2, .8])], grad=True), self.tensor([logs([.4, .6])], grad=True)
            gold = self.torch.tensor([1])
            loss, diag = binary_js_loss(left, right, gold, .7)
            expected, _ = binary_js_reference(left.detach().tolist(), right.detach().tolist(), [1], .7)
            self.assertAlmostEqual(loss.item(), expected)
            self.assertEqual(diag["active_pairs"], 1)
            loss.backward()
            self.assertGreater(left.grad.abs().sum().item(), 0)
            self.assertGreater(right.grad.abs().sum().item(), 0)

        def test_js_detached_correctness_guard_zeroes_inactive_pair_gradients(self):
            left, right = self.tensor([logs([.8, .2])], grad=True), self.tensor([logs([.2, .8])], grad=True)
            loss, diag = binary_js_loss(left, right, self.torch.tensor([1]), .5)
            self.assertEqual(loss.item(), 0)
            self.assertEqual(diag["active_pairs"], 0)
            loss.backward()
            self.assertEqual(left.grad.abs().sum().item(), 0)
            self.assertEqual(right.grad.abs().sum().item(), 0)

        def test_grpo_matches_reference_detaches_old_and_advantages(self):
            new, old = self.tensor([[0, 0], [0, 0]], grad=True), self.tensor([[0, 0], [0, 0]], grad=True)
            adv, mask = self.tensor([1, -1], grad=True), self.torch.tensor([[1, 1], [1, 0]])
            loss, diag = grpo_loss(new, old, mask, adv)
            expected, _ = grpo_reference_loss(new.detach().tolist(), old.detach().tolist(), mask.tolist(), adv.detach().tolist())
            self.assertAlmostEqual(loss.item(), expected)
            self.assertEqual(diag["n_tokens"], 3)
            loss.backward()
            self.assertIsNone(old.grad)
            self.assertIsNone(adv.grad)
            self.assertEqual(new.grad[1, 1].item(), 0)
            self.assertGreater(new.grad.abs().sum().item(), 0)

        def test_clipped_regions_have_zero_new_policy_gradient(self):
            new = self.tensor([[math.log(1.5)], [math.log(.5)]], grad=True)
            loss, _ = grpo_loss(new, self.tensor([[0], [0]]), self.torch.tensor([[1], [1]]), self.tensor([1, -1]))
            loss.backward()
            self.assertEqual(new.grad.abs().sum().item(), 0)

        def test_torch_nonfinite_or_empty_masks_abort(self):
            with self.assertRaises(ValueError):
                grpo_loss(self.tensor([[0]]), self.tensor([[0]]), self.torch.tensor([[0]]), self.tensor([1]))
            with self.assertRaises(ValueError):
                grpo_loss(self.tensor([[float("nan")]]), self.tensor([[0]]), self.torch.tensor([[1]]), self.tensor([1]))
            with self.assertRaises(ValueError):
                binary_js_loss(self.tensor([[0, 0]]), self.tensor([[0, 0]]), self.tensor([1]), .5)

        def test_js_gradcheck_for_active_guard_region(self):
            left, right = self.tensor([logs([.2, .8])], grad=True), self.tensor([logs([.4, .6])], grad=True)
            gold = self.torch.tensor([1])
            self.assertTrue(self.torch.autograd.gradcheck(lambda a, b: binary_js_loss(a, b, gold, .7)[0], (left, right)))


if __name__ == "__main__":
    unittest.main()

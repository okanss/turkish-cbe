"""Auditable rewards and losses; importing this module never loads Torch.

The correctness guard for binary JS is a Turkish-CBE extension. Binary JS uses
probabilities conditional on the two *whole answer sequences*, so it does not
measure invalid-output probability. GRPO implements clipped policy gradients
with beta=0 only. Population standard deviation plus epsilon is an explicit
normalization choice, not a claimed resolution of an unspecified paper detail.
"""

from __future__ import annotations

from collections import defaultdict
import importlib
import math


NORMALIZATION = "population_std_plus_epsilon; zero-variance groups map to zero"


def _number(value, name, *, minimum=None, maximum=None):
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(name + " must be a finite number")
    try:
        value = float(value)
    except OverflowError as error:
        raise ValueError(name + " must be finite") from error
    if not math.isfinite(value) or (minimum is not None and value < minimum) or (maximum is not None and value > maximum):
        raise ValueError(name + " is nonfinite or outside its permitted range")
    return value


def _identifier(value, name):
    if type(value) is str and value.strip():
        return ("str", value)
    if type(value) is int and value >= 0:
        return ("int", value)
    raise ValueError(name + " must be a nonblank string or nonnegative exact integer")


def _samples(samples):
    if not isinstance(samples, list):
        raise ValueError("samples must be a list")
    grouped = defaultdict(list)
    golds = {}
    for index, sample in enumerate(samples):
        if not isinstance(sample, dict) or not {"instance_id", "query_id", "presentation_id", "prediction", "gold", "truncated"} <= sample.keys():
            raise ValueError("each sample needs instance/query/presentation, prediction, gold and truncated")
        key = (_identifier(sample["instance_id"], "instance_id"), _identifier(sample["query_id"], "query_id"))
        _identifier(sample["presentation_id"], "presentation_id")
        if type(sample["gold"]) is not bool or type(sample["truncated"]) is not bool:
            raise ValueError("gold and truncated must be exact Boolean values")
        if sample["prediction"] is not None and type(sample["prediction"]) is not bool:
            raise ValueError("prediction must be exact Boolean or None")
        if key in golds and golds[key] != sample["gold"]:
            raise ValueError("one instance/query group cannot mix gold labels")
        golds[key] = sample["gold"]
        grouped[key].append(index)
    return grouped


def _valid(sample):
    return sample["prediction"] is not None and not sample["truncated"]


def reward_components(samples, *, agreement_weight=0.5):
    """Return correctness plus positively clipped, cross-presentation bonus.

    q pools valid, untruncated Boolean answers within one instance/query. The
    agreement fraction excludes every repetition of the current presentation.
    Invalid/truncated samples receive exactly zero correctness, bonus and reward.
    Missing valid alternative presentations cannot earn an agreement bonus.
    """
    weight = _number(agreement_weight, "agreement_weight", minimum=0)
    groups = _samples(samples)
    output = [None] * len(samples)
    for indexes in groups.values():
        valid = [i for i in indexes if _valid(samples[i])]
        q = sum(samples[i]["prediction"] for i in valid) / len(valid) if valid else None
        chance = q * q + (1 - q) * (1 - q) if q is not None else None
        for index in indexes:
            sample = samples[index]
            others = [i for i in valid if _identifier(samples[i]["presentation_id"], "presentation_id")
                      != _identifier(sample["presentation_id"], "presentation_id")]
            matching = sum(samples[i]["prediction"] == sample["prediction"] for i in others) if _valid(sample) else 0
            agreement = matching / len(others) if others and _valid(sample) else None
            correctness = float(sample["prediction"] == sample["gold"]) if _valid(sample) else 0.0
            bonus = max(0.0, agreement - chance) if agreement is not None else 0.0
            output[index] = {"correctness": correctness, "bonus": bonus,
                "reward": correctness + weight * bonus, "q": q, "chance": chance,
                "agreement": agreement, "other_valid": len(others), "other_matching": matching,
                "group_valid": len(valid), "valid": _valid(sample)}
    return output


def _advantage_groups(samples, mode):
    if type(mode) is not str or mode not in {"prompt_normalized", "instance_centered"}:
        raise ValueError("unknown advantage mode")
    groups = _samples(samples)
    if mode == "instance_centered":
        return groups
    prompts = defaultdict(list)
    for key, indexes in groups.items():
        for index in indexes:
            prompts[key + (_identifier(samples[index]["presentation_id"], "presentation_id"),)].append(index)
    return prompts


def advantages(samples, rewards, *, mode="prompt_normalized", epsilon=1e-4):
    """Center all planned samples, including zero-reward invalid generations.

    Prompt-normalized mode divides by population_std + epsilon. Epsilon=0 is
    supported for algebraic audits; zero-variance groups still produce zeros.
    Instance-centered mode is unscaled and is a distinct control intervention.
    Invalid samples may have negative *advantages* after centering; their input
    rewards must nevertheless be zero.
    """
    epsilon = _number(epsilon, "epsilon", minimum=0)
    groups = _advantage_groups(samples, mode)
    if not isinstance(rewards, list) or len(rewards) != len(samples):
        raise ValueError("rewards must be a list aligned with samples")
    values = [_number(value, "reward") for value in rewards]
    if any(not _valid(sample) and values[i] != 0 for i, sample in enumerate(samples)):
        raise ValueError("invalid/truncated samples must have zero rewards")
    output = [0.0] * len(samples)
    for indexes in groups.values():
        try:
            mean = math.fsum(values[i] / len(indexes) for i in indexes)
        except OverflowError as error:
            raise ValueError("reward mean is nonfinite") from error
        centered = [values[i] - mean for i in indexes]
        std = math.sqrt(math.fsum(value * value for value in centered) / len(centered))
        if not math.isfinite(std):
            raise ValueError("reward variance is nonfinite")
        denominator = std + epsilon if mode == "prompt_normalized" else 1.0
        for i, value in zip(indexes, centered):
            output[i] = value / denominator if denominator and std else 0.0
    return output


def _cosine(left, right):
    norm0 = math.sqrt(math.fsum(value * value for value in left))
    norm1 = math.sqrt(math.fsum(value * value for value in right))
    return max(-1.0, min(1.0, math.fsum(a * b for a, b in zip(left, right)) / (norm0 * norm1))) if norm0 and norm1 else None


def agreement_diagnostics(samples, *, agreement_weight=0.5, mode="prompt_normalized", epsilon=1e-4):
    """Compare agreement versus correctness-only advantages without efficacy claims.

    With valid binary answers, prompt-wise normalization removes a positive
    affine reward change. At nonzero epsilon, tiny prompt-wise scale changes can
    remain. This diagnostic exposes those changes rather than claiming extra
    training signal or improved reasoning from a nonzero raw bonus.
    """
    baseline = reward_components(samples, agreement_weight=0)
    changed = reward_components(samples, agreement_weight=agreement_weight)
    first = advantages(samples, [r["reward"] for r in baseline], mode=mode, epsilon=epsilon)
    second = advantages(samples, [r["reward"] for r in changed], mode=mode, epsilon=epsilon)
    group_cosines = []
    same_directions = True
    for indexes in _advantage_groups(samples, mode).values():
        left, right = [first[i] for i in indexes], [second[i] for i in indexes]
        cosine = _cosine(left, right)
        equal_zero = all(value == 0 for value in left + right)
        same_directions &= equal_zero or (cosine is not None and math.isclose(cosine, 1.0, abs_tol=1e-12))
        group_cosines.append(cosine)
    delta = max((abs(a - b) for a, b in zip(first, second)), default=0.0)
    binary_scope = mode == "prompt_normalized" and bool(samples) and all(_valid(s) for s in samples)
    return {"mode": mode, "epsilon": epsilon, "normalization": NORMALIZATION if mode == "prompt_normalized" else "instance_query_mean_centered_unscaled",
        "n_samples": len(samples), "n_positive_bonuses": sum(r["bonus"] > 0 for r in changed),
        "cosine": _cosine(first, second), "max_absolute_advantage_difference": delta,
        "active_correctness_advantages": sum(abs(v) > 1e-12 for v in first),
        "active_agreement_advantages": sum(abs(v) > 1e-12 for v in second),
        "all_zero_correctness_update": all(v == 0 for v in first),
        "all_zero_agreement_update": all(v == 0 for v in second),
        "binary_prompt_invariance_scope": binary_scope,
        "prompt_advantage_direction_invariant": bool(binary_scope and same_directions),
        "epsilon_only_rescaling": bool(binary_scope and same_directions and delta > 1e-12),
        "group_cosines": group_cosines, "accuracy_improvement_measured": False}


def _matrix(values, name):
    if not isinstance(values, list) or not values or not all(isinstance(row, list) and row for row in values):
        raise ValueError(name + " must be a nonempty rectangular list matrix")
    width = len(values[0])
    if any(len(row) != width for row in values):
        raise ValueError(name + " is ragged")
    return [[_number(value, name) for value in row] for row in values]


def _binary_logdist(values):
    maximum = max(values)
    norm = maximum + math.log(math.fsum(math.exp(value - maximum) for value in values))
    output = [value - norm for value in values]
    if not all(math.isfinite(value) for value in output):
        raise ValueError("normalized answer log probabilities are nonfinite")
    return output


def _logaddexp(left, right):
    maximum = max(left, right)
    return maximum + math.log1p(math.exp(min(left, right) - maximum))


def binary_js_reference(logp0, logp1, gold, weight=1.0):
    """Pure-Python reference of guarded binary JS over whole-answer log scores."""
    weight = _number(weight, "weight", minimum=0)
    left, right = _matrix(logp0, "logp0"), _matrix(logp1, "logp1")
    if len(left) != len(right) or any(len(row) != 2 for row in left + right):
        raise ValueError("answer log probabilities must have matching [batch,2] shape")
    if not isinstance(gold, list) or len(gold) != len(left) or any(type(value) not in (int, bool) or value not in (0, 1) for value in gold):
        raise ValueError("gold must be aligned 0=negative,1=positive class indexes")
    values, active = [], []
    for a, b, target in zip(left, right, gold):
        a, b = _binary_logdist(a), _binary_logdist(b)
        p, q = [math.exp(x) for x in a], [math.exp(x) for x in b]
        middle = [_logaddexp(x, y) - math.log(2) for x, y in zip(a, b)]
        js = .5 * math.fsum(x * (lp - lm) for x, lp, lm in zip(p, a, middle))
        js += .5 * math.fsum(x * (lp - lm) for x, lp, lm in zip(q, b, middle))
        values.append(max(0.0, js))
        active.append(p[int(target)] > .5 and q[int(target)] > .5)
    loss = weight * math.fsum(value for value, enabled in zip(values, active) if enabled) / len(values)
    return loss, {"backend": "python_reference", "mean_js": math.fsum(values) / len(values),
        "active_pairs": sum(active), "n_pairs": len(values), "weight": weight,
        "guard": "detached_both_gold_probability_strictly_gt_0.5", "probability_scope": "conditional_on_two_whole_answer_sequences"}


def grpo_reference_loss(new_logprobs, old_logprobs, mask, advantages, epsilon=0.2):
    """Pure-Python reference: mean of per-sequence masked-token clipped losses."""
    epsilon = _number(epsilon, "clip epsilon", minimum=0, maximum=1)
    new, old = _matrix(new_logprobs, "new_logprobs"), _matrix(old_logprobs, "old_logprobs")
    if len(new) != len(old) or any(len(a) != len(b) for a, b in zip(new, old)):
        raise ValueError("new/old log probability shapes differ")
    if (not isinstance(mask, list) or len(mask) != len(new) or any(not isinstance(row, list) or len(row) != len(new[i])
            or any(type(v) not in (bool, int, float) or v not in (0, 1) for v in row) for i, row in enumerate(mask))):
        raise ValueError("mask must be aligned binary token rows")
    if not isinstance(advantages, list) or len(advantages) != len(new):
        raise ValueError("advantages must have one value per sequence")
    adv = [_number(value, "advantage") for value in advantages]
    losses, ratios, clipped, lengths = [], [], [], []
    for current, previous, active, advantage in zip(new, old, mask, adv):
        count = sum(active)
        if not count:
            raise ValueError("each sequence requires at least one unmasked token")
        terms = []
        for value, before, include in zip(current, previous, active):
            try:
                ratio = math.exp(value - before)
            except OverflowError as error:
                raise ValueError("policy ratio is nonfinite") from error
            if not math.isfinite(ratio):
                raise ValueError("policy ratio is nonfinite")
            if include:
                terms.append(-min(ratio * advantage, min(1 + epsilon, max(1 - epsilon, ratio)) * advantage))
                ratios.append(ratio)
                clipped.append(abs(ratio - 1) > epsilon)
        losses.append(math.fsum(terms) / count)
        lengths.append(count)
    loss = math.fsum(losses) / len(losses)
    if not math.isfinite(loss):
        raise ValueError("clipped policy gradient loss is nonfinite")
    return loss, {"backend": "python_reference", "beta": 0.0, "n_sequences": len(new), "n_tokens": sum(lengths),
        "clip_fraction": sum(clipped) / len(clipped), "ratio_mean": math.fsum(ratios) / len(ratios),
        "reduction": "mean_of_per_sequence_masked_token_means"}


def _torch():
    try:
        return importlib.import_module("torch")
    except ImportError as error:
        raise RuntimeError("Torch objective backend unavailable; install the pinned training dependencies") from error


def _tensor(torch, value, name, shape=None):
    if not isinstance(value, torch.Tensor) or not value.is_floating_point():
        raise ValueError(name + " must be a floating Torch tensor")
    if shape is not None and value.shape != shape:
        raise ValueError(name + " has a mismatched shape")
    if not bool(torch.isfinite(value).all().item()):
        raise ValueError(name + " is nonfinite")


def binary_js_loss(logp0, logp1, gold_tensor, weight):
    """Return (weighted_loss, diagnostics); label order is [NO/HAYIR,YES/EVET].

    Inputs are [batch,2] *complete answer sequence* log likelihoods. Caller adds
    this consistency loss to assistant-only CE. The detached guard is our
    explicit extension, not an attributed missing paper formula.
    """
    torch = _torch()
    weight = _number(weight, "weight", minimum=0)
    _tensor(torch, logp0, "logp0")
    _tensor(torch, logp1, "logp1", logp0.shape)
    if logp0.ndim != 2 or logp0.shape[0] == 0 or logp0.shape[1] != 2 or logp0.device != logp1.device or logp0.dtype != logp1.dtype:
        raise ValueError("answer log likelihoods require matching [batch,2] dtype/device")
    if (not isinstance(gold_tensor, torch.Tensor) or gold_tensor.ndim != 1 or len(gold_tensor) != len(logp0)
            or gold_tensor.device != logp0.device or gold_tensor.is_floating_point() or gold_tensor.is_complex()
            or not bool(((gold_tensor == 0) | (gold_tensor == 1)).all().item())):
        raise ValueError("gold tensor must contain aligned integer/bool class indexes 0 or 1")
    left, right = torch.log_softmax(logp0, -1), torch.log_softmax(logp1, -1)
    _tensor(torch, left, "normalized logp0")
    _tensor(torch, right, "normalized logp1")
    p, q = left.exp(), right.exp()
    middle = torch.logaddexp(left, right) - math.log(2)
    js = (.5 * ((p * (left - middle)).sum(-1) + (q * (right - middle)).sum(-1))).clamp_min(0)
    target = gold_tensor.long().unsqueeze(-1)
    guard = ((p.gather(-1, target).squeeze(-1) > .5) & (q.gather(-1, target).squeeze(-1) > .5)).detach()
    loss = weight * (js * guard.to(js.dtype)).mean()
    _tensor(torch, loss, "binary JS loss")
    return loss, {"backend": "torch", "mean_js": float(js.detach().mean().item()), "active_pairs": int(guard.sum().item()),
        "n_pairs": len(logp0), "weight": weight, "guard": "detached_both_gold_probability_strictly_gt_0.5",
        "probability_scope": "conditional_on_two_whole_answer_sequences"}


def grpo_loss(new_logprobs, old_logprobs, mask, advantages, epsilon=0.2):
    """Return clipped beta=0 PG loss; detach old policy, rewards and mask.

    Each sequence contributes its own masked-token mean, then sequences are
    averaged. A zero-token sequence or any nonfinite value aborts the loss.
    """
    torch = _torch()
    epsilon = _number(epsilon, "clip epsilon", minimum=0, maximum=1)
    _tensor(torch, new_logprobs, "new_logprobs")
    _tensor(torch, old_logprobs, "old_logprobs", new_logprobs.shape)
    if (new_logprobs.ndim != 2 or not all(new_logprobs.shape)
            or new_logprobs.device != old_logprobs.device or new_logprobs.dtype != old_logprobs.dtype):
        raise ValueError("new/old log probabilities need matching nonempty [batch,tokens] dtype/device")
    if (not isinstance(mask, torch.Tensor) or mask.shape != new_logprobs.shape or mask.device != new_logprobs.device
            or not bool(((mask == 0) | (mask == 1)).all().item())):
        raise ValueError("mask must be an aligned binary Torch tensor")
    if isinstance(mask, torch.Tensor) and mask.is_complex():
        raise ValueError("mask cannot be complex")
    _tensor(torch, advantages, "advantages", new_logprobs.shape[:1])
    if advantages.device != new_logprobs.device:
        raise ValueError("advantages device differs from policy")
    active = mask.detach().to(new_logprobs.dtype)
    lengths = active.sum(-1)
    if bool((lengths == 0).any().item()):
        raise ValueError("each sequence requires at least one unmasked token")
    ratio = (new_logprobs - old_logprobs.detach()).exp()
    _tensor(torch, ratio, "policy ratio")
    adv = advantages.detach().unsqueeze(-1)
    unclipped, clipped = ratio * adv, ratio.clamp(1 - epsilon, 1 + epsilon) * adv
    loss = (-(torch.minimum(unclipped, clipped) * active).sum(-1) / lengths).mean()
    _tensor(torch, loss, "GRPO loss")
    tokens = active.sum()
    return loss, {"backend": "torch", "beta": 0.0, "n_sequences": new_logprobs.shape[0], "n_tokens": int(tokens.item()),
        "clip_fraction": float((((ratio - 1).abs() > epsilon).to(active.dtype) * active).sum().detach().item() / tokens.item()),
        "ratio_mean": float((ratio * active).sum().detach().item() / tokens.item()),
        "reduction": "mean_of_per_sequence_masked_token_means"}

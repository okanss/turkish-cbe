"""Optional local Torch/PEFT loops. Nothing trains or downloads on import.

This deliberately small backend uses one optimizer update per on-policy GRPO
rollout group (beta=0). It is not a claim of reproducing the original trainer.
Public execution requires a reviewed plan and verified CUDA/local assets.
"""

from copy import deepcopy
import hashlib
import json
import math
import os
from pathlib import Path
import random
import time

from inference.dataset import fingerprint
from inference.parser import parse_answer
from inference.prompts import make_messages
from inference.providers import _chat_options, _template
from inference.runner import _immutable
from .data import iter_training_units
from .hardware import inspect_hardware, check_resources
from .objectives import reward_components, advantages, agreement_diagnostics, binary_js_loss, grpo_loss
from .planning import make_plan, validate_plan


class TrainingBlocked(ValueError):
    pass


def encode_supervision(tokenizer, messages, answer, max_length, *, options=None):
    """Mask the entire prompt, supervise the complete assistant sequence/EOS."""
    if not isinstance(answer, str) or not answer or type(max_length) is not int or max_length < 1:
        raise ValueError("nonempty answer and positive max_length required")
    options = options or {}
    prefix = list(tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=True, **options))
    full = list(tokenizer.apply_chat_template(messages + [{"role": "assistant", "content": answer}],
                                              tokenize=True, add_generation_prompt=False, **options))
    if not prefix or full[:len(prefix)] != prefix or len(full) <= len(prefix):
        raise ValueError("chat template does not preserve the assistant-generation prefix")
    if len(full) > max_length:
        raise TrainingBlocked("encoded_sequence_exceeds_max_length; no truncation or row dropping")
    if any(type(token) is not int or token < 0 for token in full):
        raise ValueError("tokenizer must return integer IDs")
    return {"input_ids": full, "labels": [-100] * len(prefix) + full[len(prefix):],
            "prompt_tokens": len(prefix), "completion_tokens": len(full)-len(prefix)}


def sft_batches(unit_ids, *, paired, effective_batch_size, epochs, max_steps, seed):
    """Deterministic unit batches; pairs never split across optimizer steps.

    Positive max_steps explicitly cycles the pool; epoch mode keeps the short
    final batch and never invents extra examples. Repetitions are exposures.
    """
    if not unit_ids or len(set(unit_ids)) != len(unit_ids):
        raise ValueError("unique nonempty unit IDs required")
    if type(paired) is not bool or type(effective_batch_size) is not int or effective_batch_size < 1:
        raise ValueError("invalid batch settings")
    per_unit = 2 if paired else 1
    if effective_batch_size % per_unit:
        raise ValueError("paired batch must contain whole presentation pairs")
    if type(epochs) is not int or epochs < 1 or type(max_steps) is not int or max_steps < 0:
        raise ValueError("invalid epoch/step settings")
    width = effective_batch_size // per_unit
    rng = random.Random(seed)
    step, cycle = 0, 0
    while max_steps and step < max_steps or not max_steps and cycle < epochs:
        ordered = list(unit_ids); rng.shuffle(ordered)
        cycle += 1
        for start in range(0, len(ordered), width):
            if max_steps and step >= max_steps:
                return
            batch = ordered[start:start+width]
            # A matched-step override uses a complete effective batch even at
            # cycle boundaries; this is explicit repetition, not a new unit.
            if max_steps and len(batch) < width:
                batch += [ordered[i % len(ordered)] for i in range(width-len(batch))]
            step += 1
            yield batch


def load_runtime(plan, arm, *, adapter=None):
    """Load only a hash-verified local snapshot; no Hub/network fallback."""
    import torch
    from transformers import AutoTokenizer, AutoModelForCausalLM
    from peft import LoraConfig, get_peft_model, PeftModel
    config, spec = plan["config"], plan["model"]
    path = config["runtime"]["weights_path"]
    tokenizer = AutoTokenizer.from_pretrained(path, local_files_only=True, trust_remote_code=False)
    template = tokenizer.chat_template
    _template(template, spec)
    options = _chat_options(template, spec)
    model = AutoModelForCausalLM.from_pretrained(path, local_files_only=True, trust_remote_code=False,
        torch_dtype=torch.bfloat16, attn_implementation="eager")
    if int(model.config.max_position_embeddings) < config["budget"]["max_length"]:
        raise TrainingBlocked("declared sequence budget exceeds loaded model context")
    model.to("cuda:0")
    if adapter is not None:
        model = PeftModel.from_pretrained(model, str(adapter), is_trainable=False, local_files_only=True)
    elif arm != "base":
        missing = [target for target in config["lora"]["target_modules"]
                   if not any(name.endswith("."+target) for name, _ in model.named_modules())]
        if missing:
            raise TrainingBlocked("LoRA target modules missing: " + ",".join(missing))
        lora = config["lora"]
        model = get_peft_model(model, LoraConfig(task_type="CAUSAL_LM", r=lora["rank"],
            lora_alpha=lora["alpha"], lora_dropout=lora["dropout"],
            target_modules=lora["target_modules"], bias="none"))
        model.config.use_cache = False
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    return model, tokenizer, torch, options


def _inputs(encoded, model, torch):
    ids = torch.tensor([encoded["input_ids"]], dtype=torch.long, device=next(model.parameters()).device)
    return {"input_ids": ids, "attention_mask": torch.ones_like(ids),
            "labels": torch.tensor([encoded["labels"]], dtype=torch.long, device=ids.device)}


def sequence_logps(model, torch, input_ids, prompt_tokens, *, temperature=1.0):
    """Log probabilities for every generated token, including EOS if present."""
    if not 0 < prompt_tokens < len(input_ids):
        raise ValueError("completion must contain at least one token")
    if not math.isfinite(temperature) or temperature <= 0:
        raise ValueError("positive finite policy temperature required")
    ids = torch.tensor([input_ids], dtype=torch.long, device=next(model.parameters()).device)
    logits = model(input_ids=ids, attention_mask=torch.ones_like(ids)).logits.float()
    selected = torch.log_softmax(logits[:, :-1] / temperature, -1).gather(-1, ids[:, 1:, None]).squeeze(-1)
    return selected[:, prompt_tokens-1:]


def answer_logps(model, tokenizer, torch, messages, max_length, options):
    answers = []
    for answer in ("HAYIR", "EVET"):
        encoded = encode_supervision(tokenizer, messages, answer, max_length, options=options)
        answers.append(sequence_logps(model, torch, encoded["input_ids"], encoded["prompt_tokens"]).sum(-1))
    return torch.stack(answers, dim=-1)


def _finite(loss, torch):
    if loss.ndim != 0 or not bool(torch.isfinite(loss).item()):
        raise TrainingBlocked("nonfinite_or_nonscalar_training_loss")


def _eos_ids(model, tokenizer):
    result = set()
    for value in (getattr(model.generation_config, "eos_token_id", None), tokenizer.eos_token_id):
        if isinstance(value, list): result.update(value)
        elif type(value) is int: result.add(value)
    return result


def grpo_sampling_options(config, eos_ids, pad_token_id):
    """Full-support temperature policy; never inherit model decoding warpers."""
    temperature = config["grpo"]["temperature"]
    if not math.isfinite(temperature) or temperature <= 0 or config["grpo"]["top_p"] != 1:
        raise ValueError("GRPO requires positive temperature and top_p=1 for matched full-support log probabilities")
    if not eos_ids or type(pad_token_id) is not int:
        raise ValueError("explicit EOS and pad token IDs required")
    return {"do_sample": True, "temperature": temperature, "top_p": 1.0, "top_k": 0,
            "max_new_tokens": config["budget"]["output_tokens"],
            "eos_token_id": sorted(eos_ids), "pad_token_id": pad_token_id}


def _check_wall_time(started, config):
    if time.monotonic()-started > config["budget"]["max_wall_seconds"]:
        raise TrainingBlocked("training_wall_time_budget_exceeded")


def rollout(model, tokenizer, torch, messages, config, options):
    from transformers import GenerationConfig
    prefix = list(tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=True, **options))
    cap = config["budget"]["output_tokens"]
    if len(prefix)+cap > config["budget"]["max_length"]:
        raise TrainingBlocked("rollout_context_exceeds_training_budget")
    ids = torch.tensor([prefix], dtype=torch.long, device=next(model.parameters()).device)
    # A fresh config and use_model_defaults=False prevent top-k, repetition
    # penalties or forced tokens in the snapshot from changing the policy.
    generation = GenerationConfig(**grpo_sampling_options(config, _eos_ids(model, tokenizer), tokenizer.eos_token_id))
    with torch.no_grad():
        generated = model.generate(input_ids=ids, attention_mask=torch.ones_like(ids),
            generation_config=generation, use_model_defaults=False)
    completion = generated[0, len(prefix):].tolist()
    if not completion: raise TrainingBlocked("empty_generation_sequence")
    truncated = len(completion) >= cap and completion[-1] not in _eos_ids(model, tokenizer)
    text = tokenizer.decode(completion, skip_special_tokens=True)
    parsed = parse_answer(text, "tr", truncated=truncated)
    return {"input_ids": prefix+completion, "prompt_tokens": len(prefix),
        "completion_tokens": len(completion), "text": text, "truncated": truncated, "parse": parsed}


def _schedule(optimizer, steps, warmup_ratio, torch):
    warmup = math.ceil(steps * warmup_ratio)
    def factor(index):
        if index < warmup: return (index+1) / max(1, warmup)
        return .5 * (1 + math.cos(math.pi * min(1, (index-warmup) / max(1, steps-warmup))))
    return torch.optim.lr_scheduler.LambdaLR(optimizer, factor)


def train_loaded(model, tokenizer, torch, options, units, config, arm, *, log_step=None):
    """Execute on an already gated model. Tests may supply tiny local fixtures.

    No test fixture is accepted by the public CLI as real training evidence.
    No evaluator/test row is accessible through this function's unit interface.
    """
    if arm not in config["arms"] or arm == "base": raise ValueError("select a trained arm")
    index = {unit["unit_id"]: unit for unit in units}
    if len(index) != len(units) or not index: raise ValueError("nonempty unique training units required")
    encoded, messages = {}, {}
    grpo = arm.startswith("grpo_")
    views = ("canonical", "equivalent", "P1", "P2") if grpo else (
        ("canonical", "equivalent") if arm in {"paired_sft", "sft_consistency"} else ("canonical",))
    # Exact token preflight over every selected row and both answer candidates.
    # No optimizer is created until all selected sequences fit without truncation.
    for unit in units:
        if unit["n"] > 128 or any(unit["views"][v]["split"] != "training" for v in views):
            raise TrainingBlocked("nontraining_or_large_n_row_in_training")
        for view in views:
            row = unit["views"][view]
            if row["latent_instance_id"] != unit["instance_id"] or row["gold_label"] is not unit["gold"]:
                raise TrainingBlocked("training_identity_or_gold_changed")
            built = make_messages(row)
            messages[(unit["unit_id"], view)] = built["messages"]
            for answer in ("HAYIR", "EVET"):
                item = encode_supervision(tokenizer, built["messages"], answer, config["budget"]["max_length"], options=options)
                if grpo and item["prompt_tokens"]+config["budget"]["output_tokens"] > config["budget"]["max_length"]:
                    raise TrainingBlocked("rollout_budget_exceeds_exact_context")
                if answer == ("EVET" if unit["gold"] else "HAYIR"):
                    encoded[(unit["unit_id"], view)] = item
    if grpo:
        total_steps = config["grpo"]["max_steps"]
        rng = random.Random(config["seed"])
        order = list(index); rng.shuffle(order)
        batches = ([order[step % len(order)]] for step in range(total_steps))
    else:
        paired = len(views) == 2
        batches = list(sft_batches(list(index), paired=paired,
            effective_batch_size=config["sft"]["effective_batch_size"], epochs=config["sft"]["epochs"],
            max_steps=config["sft"]["max_steps"], seed=config["seed"]))
        total_steps = len(batches)
    if total_steps > config["budget"]["max_optimizer_steps"]:
        raise TrainingBlocked("optimizer_step_budget_exceeded")
    parameters = [value for value in model.parameters() if value.requires_grad]
    count = sum(value.numel() for value in parameters)
    if not count: raise TrainingBlocked("no_trainable_adapter_parameters")
    optimizer = torch.optim.AdamW(parameters, lr=config["grpo" if grpo else "sft"]["learning_rate"], weight_decay=0.0)
    scheduler = _schedule(optimizer, total_steps, config["budget"]["warmup_ratio"], torch)
    stats = {"optimizer_steps": 0, "sequence_exposures": 0, "training_token_exposures": 0,
        "assistant_token_exposures": 0, "rollout_generation_input_tokens": 0,
        "rollout_generation_output_tokens": 0, "auxiliary_answer_scoring_tokens": 0,
        "old_policy_scoring_token_exposures": 0,
        "trainable_parameters": count, "parameter_update_budget": 0,
        "unique_training_units_seen": 0, "unique_training_latents_seen": 0, "zero_gradient_steps": 0}
    seen, latents = set(), set()
    started = time.monotonic()
    for step, batch in enumerate(batches, 1):
        _check_wall_time(started, config)
        optimizer.zero_grad(set_to_none=True)
        diagnostics, raw_rollouts = {}, []
        if not grpo:
            model.train()
            loss_value = 0.0
            for uid in batch:
                unit = index[uid]
                losses = []
                for view in views:
                    item = encoded[(uid, view)]
                    losses.append(model(**_inputs(item, model, torch)).loss.float())
                    stats["sequence_exposures"] += 1
                    stats["training_token_exposures"] += len(item["input_ids"])
                    stats["assistant_token_exposures"] += item["completion_tokens"]
                loss = torch.stack(losses).mean()
                if arm == "sft_consistency":
                    pair = [answer_logps(model, tokenizer, torch, messages[(uid, v)],
                        config["budget"]["max_length"], options) for v in views]
                    penalty, diagnostics = binary_js_loss(pair[0], pair[1],
                        torch.tensor([int(unit["gold"])], device=pair[0].device), config["consistency"]["js_weight"])
                    loss = loss+penalty
                    for v in views:
                        for answer in ("HAYIR", "EVET"):
                            stats["auxiliary_answer_scoring_tokens"] += len(encode_supervision(tokenizer,
                                messages[(uid, v)], answer, config["budget"]["max_length"], options=options)["input_ids"])
                _finite(loss, torch)
                (loss / len(batch)).backward()
                loss_value += float(loss.detach().item()) / len(batch)
                seen.add(uid); latents.add(unit["instance_id"])
        else:
            # Dropout is disabled for sampling and both old/current likelihoods;
            # this avoids pretending dropout noise is a policy update.
            model.eval()
            uid = batch[0]; unit = index[uid]; samples, old = [], []
            for view in views:
                for _ in range(config["data"]["grpo_rollouts_per_view"]):
                    _check_wall_time(started, config)
                    generated = rollout(model, tokenizer, torch, messages[(uid, view)], config, options)
                    with torch.no_grad():
                        logp = sequence_logps(model, torch, generated["input_ids"], generated["prompt_tokens"],
                                              temperature=config["grpo"]["temperature"])
                    old.append(logp.detach())
                    stats["old_policy_scoring_token_exposures"] += len(generated["input_ids"])
                    samples.append({"instance_id": unit["instance_id"], "query_id": unit["query_id"],
                        "presentation_id": view, "prediction": generated["parse"]["prediction"],
                        "gold": unit["gold"], "truncated": generated["truncated"]})
                    raw_rollouts.append(generated)
                    stats["rollout_generation_input_tokens"] += generated["prompt_tokens"]
                    stats["rollout_generation_output_tokens"] += generated["completion_tokens"]
            weight = config["consistency"]["agreement_bonus_weight"] if arm == "grpo_consistency" else 0.0
            components = reward_components(samples, agreement_weight=weight)
            adv = advantages(samples, [row["reward"] for row in components], mode=config["grpo"]["advantage_mode"])
            diagnostics = agreement_diagnostics(samples, agreement_weight=weight, mode=config["grpo"]["advantage_mode"])
            diagnostics.update(correctness=sum(row["correctness"] for row in components)/len(samples),
                agreement_bonus=sum(row["bonus"] for row in components)/len(samples),
                invalid_rate=sum(row["prediction"] is None or row["truncated"] for row in samples)/len(samples))
            loss_value = 0.0
            for generated, old_logp, value in zip(raw_rollouts, old, adv):
                current = sequence_logps(model, torch, generated["input_ids"], generated["prompt_tokens"],
                                         temperature=config["grpo"]["temperature"])
                loss, _ = grpo_loss(current, old_logp, torch.ones_like(current),
                    torch.tensor([value], dtype=current.dtype, device=current.device))
                _finite(loss, torch)
                (loss / len(samples)).backward()
                loss_value += float(loss.detach().item()) / len(samples)
                stats["sequence_exposures"] += 1
                stats["training_token_exposures"] += len(generated["input_ids"])
                stats["assistant_token_exposures"] += generated["completion_tokens"]
            seen.add(uid); latents.add(unit["instance_id"])
        norm = torch.nn.utils.clip_grad_norm_(parameters, 1.0, error_if_nonfinite=True)
        if float(norm) == 0: stats["zero_gradient_steps"] += 1
        optimizer.step(); scheduler.step()
        stats.update(optimizer_steps=step, parameter_update_budget=step*count,
                     unique_training_units_seen=len(seen), unique_training_latents_seen=len(latents))
        if log_step:
            log_step({"step": step, "loss": loss_value, "gradient_norm_before_clip": float(norm),
                "diagnostics": diagnostics, "stats": deepcopy(stats), "rollouts": raw_rollouts})
        # A final optimizer step that overruns must not become a completed run.
        _check_wall_time(started, config)
    stats["training_loop_wall_seconds"] = time.monotonic()-started
    stats["total_accounted_forward_token_exposures"] = sum(stats[key] for key in (
        "training_token_exposures", "auxiliary_answer_scoring_tokens", "old_policy_scoring_token_exposures",
        "rollout_generation_input_tokens", "rollout_generation_output_tokens"))
    stats["token_accounting_scope"] = "sequence inputs across CE/policy/auxiliary scoring plus rollout input/output; not FLOPs or equal GPU work"
    return stats


def execution_gate(root, plan):
    validate_plan(plan, root)
    fresh = make_plan(root, plan["config"], hardware=plan["hardware"])
    if fresh != plan:
        raise TrainingBlocked("plan/data/model/environment changed; create a new reviewed plan")
    if plan["blockers"]:
        raise TrainingBlocked(", ".join(plan["blockers"]))
    hardware = inspect_hardware(probe_torch=True)
    resources = check_resources(plan["config"], plan["model"], hardware)
    if resources["blockers"]: raise TrainingBlocked(", ".join(resources["blockers"]))
    return hardware, resources


def run_training(root, plan, arm, output, *, execute=False):
    validate_plan(plan, root)
    if arm not in plan["config"]["arms"]: raise ValueError("unknown training arm")
    if not execute:
        return {"status": "dry_run", "arm": arm, "contains_training_run": False,
                "contains_model_results": False, "actual_optimizer_steps": 0, "blockers": plan["blockers"]}
    if arm == "base": raise TrainingBlocked("base has no training; use the evaluation command")
    hardware, resources = execution_gate(root, plan)
    output = Path(output).absolute()
    if output.exists() or any(p.is_symlink() for p in (output, *output.parents)):
        raise ValueError("use a new nonsymlink run output directory")
    root = Path(root).resolve()
    if not output.is_relative_to(root / "results/training/runs"):
        raise ValueError("live training outputs must stay in results/training/runs")
    config = plan["config"]
    units = list(iter_training_units(root, plan["data"]))
    if arm.startswith("grpo_"):
        wanted = set(plan["data"]["grpo"]["selected_unit_ids"])
        units = [unit for unit in units if unit["unit_id"] in wanted]
    output.mkdir(parents=True)
    _immutable(output / "plan.json", plan)
    _immutable(output / "execution.json", {"arm": arm, "execution_kind": "live",
        "hardware": hardware, "resources": resources, "seed": config["seed"]})
    try:
        import torch
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
        random.seed(config["seed"]); torch.manual_seed(config["seed"])
        torch.cuda.manual_seed_all(config["seed"])
        torch.use_deterministic_algorithms(True)
        model, tokenizer, torch, options = load_runtime(plan, arm)
        torch.cuda.synchronize(); start_event, end_event = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        start_event.record()
        with (output / "steps.jsonl").open("x", encoding="utf-8") as handle:
            def record(value):
                handle.write(json.dumps(value, sort_keys=True, ensure_ascii=False, allow_nan=False)+"\n")
                handle.flush()
            stats = train_loaded(model, tokenizer, torch, options, units, config, arm, log_step=record)
        end_event.record(); torch.cuda.synchronize()
        stats["gpu_event_seconds"] = start_event.elapsed_time(end_event)/1000
        model.save_pretrained(output / "adapter", safe_serialization=True)
        files = [{"path": str(p.relative_to(output)), "bytes": p.stat().st_size,
                  "sha256": hashlib.sha256(p.read_bytes()).hexdigest()} for p in sorted((output / "adapter").rglob("*")) if p.is_file()]
        evidence = [{"path": name, "bytes": (output / name).stat().st_size,
                     "sha256": hashlib.sha256((output / name).read_bytes()).hexdigest()}
                    for name in ("plan.json", "execution.json", "steps.jsonl")]
        rate = config["budget"].get("cost_per_gpu_hour")
        summary = {"training_version": plan["training_version"], "arm": arm, "status": "completed",
            "contains_training_run": True, "contains_model_results": False, "plan_sha256": plan["plan_sha256"],
            "base_model": plan["model"], "stats": stats, "files": files, "evidence_files": evidence,
            "measured_cost_estimate": None if rate is None else stats["training_loop_wall_seconds"]/3600*rate,
            "currency": config["budget"].get("currency"),
            "cost_scope": "training-loop wall time times declared GPU rate; not an invoice or total loading/evaluation cost"}
        _immutable(output / "training_summary.json", summary)
        return summary
    except BaseException as error:
        _immutable(output / "failure.json", {"status": "aborted_not_completed", "error_type": type(error).__name__,
            "message": str(error), "contains_model_results": False})
        raise

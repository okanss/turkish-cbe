# RQ5 training backend validation — first executed training runs

Date: 2026-10-27. Hardware: RunPod RTX 6000 Ada (48 GB), CUDA 12.8, bf16.
Stack: the config's declared pins, now **validated on CUDA for the first time** —
torch 2.9.0+cu128, transformers 4.57.1, peft 0.17.1, accelerate 1.11.0, tiktoken 0.12.0.
Plan `results/training/plans/pod_pilot_002.json` (pilot profile, Qwen2.5-1.5B-Instruct
@ `989aa798…`, LoRA r16, hash-verified local weights bundle incl. upstream LFS SHA-256):
**zero blockers**.

## Executed arms (all `status: completed`, training_version 0.9.0)

| Arm | Wall (s) | Token exposures | Trainable params | Notes |
|---|---|---|---|---|
| sft_lora          | 12.9 | 12,253 | 18,464,768 | 2 optimizer steps, 0 zero-gradient steps |
| paired_sft        | 13.3 | 12,062 | 18,464,768 | paired canonical/equivalent supervision |
| sft_consistency   | 39.1 | 12,062 (36,170 fwd) | 18,464,768 | JS consistency term triple-scores sequences |
| grpo_correctness  | 19.2 | 11,357 (34,071 fwd) | 18,464,768 | 1/2 steps zero-gradient (binary prompt-normalized advantages; predicted by plan warning) |
| grpo_consistency  | 20.8 | 11,357 (34,071 fwd) | 18,464,768 | same zero-gradient diagnostic |

Run directories: `results/training/runs/pod-<arm>-001` (immutable plan + execution
records + adapters). Export: `results/training/data/pod_pilot_002`. Total GPU cost ≈ $2.

## What this establishes / does not establish

Establishes: the training backend (data units, LoRA, SFT/GRPO objectives, consistency
term, budgets, determinism seeds, hash-gated weights) executes end to end on the
declared pinned CUDA stack. The limitations statement "real training smoke validation
remains unexecuted" is superseded.

Does **not** establish any adaptation effect: the pilot profile is smoke-scale
(2 optimizer steps), and no evaluation was run.

## Evaluation remains gated — by design

The protocol's evaluation design is frozen by the config validator: all nine sizes
(4–1024), test split only, 4 or 200 groups. That yields 10,224 generation calls per
arm, above the 2,000-call threshold, so execution requires the dataset-bound
two-reviewer Turkish linguistic review receipt. A "reduced review-free evaluation"
is not expressible in the framework, deliberately; no gate was loosened. The trained
adapters are retained and can be evaluated as soon as the review receipt exists.

Registry note: `qwen25-15b-hf` was enabled in the **pod copy** of `configs/models.yaml`
(user instruction, 2026-10-27) for these runs; the local registry is unchanged.

# Turkish-CBE

A reproducible framework for evaluating logical-ordering competence of language
models across Turkish and English and across answer-preserving Turkish
presentations. Companion repository of the manuscript
(*Turkish-CBE: Evaluating Logical Reasoning Accuracy and Morphosyntactic
Robustness in Large Language Models*, Oluçoğlu & Bursa; submission source
`paper/main_GPTreview.tex`, with the earlier pre-revision version retained as
`paper/main.tex` for provenance).

Frozen bank: 3,000 latent orders, 23,200 source queries, 5,637,600 measured
prompt identities (not independent problem counts). Executed evidence, all
exploratory by design: two registered pilot runs (Qwen2.5-0.5B, gpt-4o), a
supplemental quantized local sweep of four Qwen2.5 scales over 16,618 prompts
each (manuscript §7.7; see [README_P1_REPRODUCIBILITY.md](README_P1_REPRODUCIBILITY.md)),
and a completed two-reviewer Turkish linguistic review of 360 sampled stimuli
(§7.8; artifacts in `validation/review_results/`). The frozen-test confirmatory
run and the training-adapter evaluation are gated and unexecuted; no
confirmatory claim is made. Methods and source/extension separation:
[EXPERIMENT_PROTOCOL.md](EXPERIMENT_PROTOCOL.md); data coverage:
[EXPERIMENT_MATRIX.md](EXPERIMENT_MATRIX.md); schema:
[DATASET_CARD.md](DATASET_CARD.md); change history and evidence:
[PROGRESS.md](PROGRESS.md).

## What is NOT in this repository

- `data/{test,training,development,pilot,latent}/` and
  `data/complete_grid/grid.sqlite` — multi-GB materializations of the frozen
  grid, rebuilt deterministically from the tracked sources, plans, and binding
  manifests (which hash-verify the rebuild); see
  [REPRODUCIBILITY.md](REPRODUCIBILITY.md).
- `tokenizers/hf/` — license-gated `meta-llama` tokenizer assets that may not
  be redistributed; fetch them with your own Hugging Face account as described
  in [REPRODUCIBILITY.md](REPRODUCIBILITY.md).
- `results/training/runs/*/adapter/` — trained LoRA adapter weights (71 MB
  each); plans, summaries, and step logs are tracked.

## Quality audit

[RESEARCH_AUDIT.md](RESEARCH_AUDIT.md) reports the 18-scope audit, the four
corrected Major software issues, and the open scientific prerequisites. The
execution gates it defines (two-reviewer receipts for >2,000-call runs, pilot
approval for frozen-test execution) remain in force and unissued.

```sh
PYTHONPATH=src:reports python3 -m research_audit --project-root . --output reports/audit/replay_001
```

The installed command is `turkish-cbe-audit`. A fresh output name is required;
the audit makes no model calls. Audit evidence lives in
`reports/audit/current/`, current-source dry-run plans in
`reports/audit/plans/`. Any pilot or training evaluation above 2,000 calls
additionally requires a genuine, scope-checked linguistic review receipt.
Historical Prompt-10 paper/plan records are preserved as immutable artifacts.

## Manuscript and reporting

`paper/main_GPTreview.tex` is the submission manuscript (compiles with
XeLaTeX + BibTeX from its own `paper/sections_gptreview/`); `paper/main.tex`
is the retained pre-revision version compiling from `paper/sections/`. All
quantitative figures are regenerated from retained measurements by
`paper/figures/manuscript/generate_figures.py` and `generate_p1_figures.py`,
with source and asset hashes recorded in the plot manifests.

[reports/README.md](reports/README.md) describes the reporting commands and
evidence contract. The publication exporter never aggregates input rows:

```sh
PYTHONPATH=src:reports python3 -m publication --paper-dir paper_next --report-dir reports/next
```

Key analysis reports: `reports/p1_statistical_analysis.md` (supplemental sweep),
`reports/inference_integrity_report.md` (PASS across all four arms),
`reports/pilot/PILOT_RESULTS.md` (registered pilots), and the review
adjudication in `validation/review_results/`.

## Optional training (RQ5)

[TRAINING_COMPARISON.md](TRAINING_COMPARISON.md) defines the six arms, budget
profiles, and reward/consistency separation. All five adaptation arms were
executed at smoke scale on rented CUDA hardware under the declared dependency
pins (torch 2.9.0, transformers 4.57.1, peft 0.17.1, accelerate 1.11.0);
plans, summaries, and step logs are tracked under `results/training/`, while
adapter weight files are excluded from the repository. These runs validate the
training machinery and its accounting only. The matched base-versus-adapter
evaluation is frozen at 10,224 calls per arm on the test split and requires
the two-reviewer execution-gate receipt; it has not been run, so no adaptation
effect of any direction is established.

```sh
PYTHONPATH=src python3 -m training plan --config configs/training/pilot.toml --output results/training/plans/my_pilot.json
PYTHONPATH=src python3 -m training run --plan results/training/plans/my_pilot.json --arm sft_lora --output results/training/runs/my_sft
```

Both commands are plan/dry-run first by default. The installed CLI is
`turkish-cbe-train`.

## Reasoning experiments

[REASONING_EXPERIMENTS.md](REASONING_EXPERIMENTS.md) defines the direct,
zero-shot reasoning, structured reasoning, and native-effort arms over shared
latent/query/presentation/language coordinates with a shared output budget;
only the final answer is scored. No reasoning-arm model calls have been
executed. The installed entry point is `turkish-cbe-reasoning`.

```sh
PYTHONPATH=src .venv/bin/python -m reasoning pilot --model qwen25-05b-hf --output results/reasoning/plans/my_pilot.json
PYTHONPATH=src .venv/bin/python -m reasoning analyse --output results/reasoning/my_analysis --figures
```

## Lexical and tokenizer analysis (RQ4 features)

[LEXICAL_ANALYSIS.md](LEXICAL_ANALYSIS.md) is the scope and method contract.
Counts use the locked UD Turkish-BOUN r2.18 corpus (CC BY-SA 4.0; 9,761
sentences). The catalog covers nine relations / 18 directions / 14 unique
lemmas. All 216 isolated-vocabulary tokenizer records are measured across the
six registered configurations — the two Llama tokenizers were vendored at
their registered immutable revisions after upstream license acceptance, with
every file hash-verified (an earlier snapshot recording them as unavailable is
retained as provenance). Write-once measurement snapshots live under
`results/lexical/` (`baseline`, `with_llama`, `pilot_joined`,
`preprocessing_current`).

```sh
.venv/bin/python -m pip install --require-hashes -r requirements-lexical.lock
PYTHONPATH=src USE_TORCH=0 USE_TF=0 USE_FLAX=0 HF_HUB_OFFLINE=1 .venv/bin/python -m analysis.lexical_bias --root . --output results/lexical/my_measurement
```

The pilot-joined snapshot provides the direction-specific diagnostics of
manuscript §7.5; with one relation family in the pilot cohort, the association
specifications correctly return recorded refusals rather than coefficients.

## CBE analysis

[ANALYSIS.md](ANALYSIS.md) defines metric denominators, record validation,
McNemar/Holm, the 2,000-replicate latent-cluster bootstrap, and Wilson
intervals. Models/revisions, name controls, n, relations, and query arms stay
in declared strata. Analyses of both registered pilot runs are retained under
`results/analysis/`; the supplemental sweep's audited-primitive analysis is
under `results/p1/analysis/` (produced by `analysis.p1_statistics`).

```sh
.venv/bin/python -m pip install --require-hashes -r requirements-analysis.lock
PYTHONPATH=src .venv/bin/python -m analysis --root . --run RUN_ID --output results/analysis/RUN_ID --figures
```

## Inference layer

- OpenAI Responses REST, local Hugging Face Transformers, and vLLM HTTP
  adapters share one execution contract; no model name is hard-coded.
- [configs/models.yaml](configs/models.yaml) pins weight/tokenizer commits,
  chat-template hashes, sampling/reasoning/seed settings, context/output
  limits, and dated price metadata. Models are enabled only by explicit user
  selection; the registry never substitutes a replacement.
- Strict parsing: exact Turkish `EVET`/`HAYIR`, English `YES`/`NO`.
  Explanations, punctuation, JSON, contradictions, wrong-language answers,
  refusals, and truncated output are preserved as invalid — never retried to
  "improve" an answer.
- Raw records are written atomically and append-only; checkpoint/resume never
  resends a completed call; ambiguous timeout/crash calls require an explicit,
  logged retry/abandon resolution.
- The supplemental quantized arm (`src/inference/ollama_runner.py`) reuses the
  audited message constructor (run-time prompt SHA-256 re-verification) and the
  strict parser against a local Ollama server; it is labeled on every record
  and never pooled with registered-provider results. See
  [README_P1_REPRODUCIBILITY.md](README_P1_REPRODUCIBILITY.md).

## Installation

Python >=3.11,<3.14 (tested on 3.12/3.13). On CPython 3.12 / macOS arm64 the
offline dependency install below works as-is; on other platforms use the same
lock files without `--no-index/--find-links`. The default configuration
downloads no model weights; API keys are read from environment variables only.

```sh
python3.12 -m venv .venv
.venv/bin/python -m pip install --no-index --find-links vendor/inference --require-hashes -r requirements-inference.lock
.venv/bin/python -m pip install --no-index --find-links vendor/tokenizers --require-hashes -r requirements-tokenizers.lock
.venv/bin/python -m pip install --require-hashes -r requirements-build.lock
.venv/bin/python -m pip install --no-build-isolation --no-deps -e .
PYTHONPATH=src .venv/bin/python -m unittest discover -s tests -v
```

The pinned `tiktoken==0.12.0` from `vendor/tokenizers` is required by the grid
layer; a mismatched version refuses to plan or execute.

## Planning and execution

Plans are write-once, dry-run-first JSON artifacts; the executor re-derives
each plan from the registry and environment at run time and compares
byte-for-byte:

```sh
.venv/bin/python scripts/run_pilot.py --model <model-id> --output logs/my_pilot_plan.json
.venv/bin/python scripts/run_pilot.py --plan logs/my_pilot_plan.json --execute --run-id <run-id>
```

The default small pilot is 32 calls/model; the default full (frozen-test)
cohort is 7,200 calls/model and additionally requires a completed real pilot,
an explicit approval artifact, and the dataset-bound two-reviewer linguistic
review receipt — none of which is bypassable from the command line. The
executed pilot runs are retained under run ids `pilot-qwen05b-main-001` and
`pilot-gpt4o-main-003` (failed attempts kept as provenance); the frozen-test
plan `logs/frozen_test_gpt4o_plan.json` is generated but unexecuted.

## Layer versions and verification

Distribution/lexical 0.7.0; CBE analysis 0.6.0; inference 0.5.0; frozen grid
0.4.1; matched core 0.4.0; TR renderer 0.3.0; generator 0.2.0. The owned
source packages are hash-bound to the frozen grid manifest — adding or editing
any `.py` under `src/{turkish_cbe,data_generation,renderers,validation,experiments}`
invalidates the binding by design.

```sh
PYTHONPATH=src python3 -m complete_grid.dataset_builder verify --manifest data/complete_grid/manifest.json --hashes-only
PYTHONPATH=src python3 -m data_generation verify dataset_manifest.json --replay
PYTHONPATH=src python3 -m experiments.dataset_builder verify --manifest experiment_dataset_manifest.json --hashes-only
```

The full factored data schema and JSONL export commands are in
[DATASET_CARD.md](DATASET_CARD.md) and [REPRODUCIBILITY.md](REPRODUCIBILITY.md).

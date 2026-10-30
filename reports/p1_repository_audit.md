# P1 repository and manuscript audit

Date: 2026-10-28. Scope: MDPI Information P1 revision program, Stage 1.

## 1. Component inventory (P1 phase → existing asset)

| P1 ask | Existing component | Status |
|---|---|---|
| Benchmark generation | `src/data_generation/`, `src/complete_grid/` (frozen grid in `data/complete_grid/grid.sqlite`) | complete, frozen |
| Dataset partitions/manifests | `data/{training,development,pilot,test}/` + `dataset_manifest.json`, `experiment_dataset_manifest.json` | complete, immutable |
| Prompt construction/transformations | `src/renderers/{turkish,english,morphology}.py`, `src/inference/prompts.py` | complete |
| Model inference | `src/inference/` (providers: **openai, transformers, vllm** — no Ollama), gated plans, checkpointed SQLite runner | complete for registered providers |
| Completed pilot experiments | `pilot-qwen05b-main-001`, `pilot-gpt4o-main-003` (1,512 calls each; raw+normalized retained) | complete |
| Statistical analysis | `src/analysis/` (paired R/D/U/κ_CBE, exact McNemar, Holm families, 2,000-rep latent-cluster bootstrap) | complete |
| Evaluation metrics | `src/analysis/metrics.py` (all-planned vs valid-pair estimands, coverage, undefined-κ semantics) | complete |
| Semantic validation | `src/validation/semantic_equivalence.py` + grid round-trip checks | complete (automatic layer) |
| Human linguistic review | contract in `src/inference/linguistic_review.py` (dataset-bound, two reviewers, ≥100 prompts, per-cell minimums); `data/validation/turkish_samples.csv` | prepared, **no human annotations exist** |
| Training arms | `src/training/` — five arms executed at smoke scale 2026-10-27 (RunPod CUDA; `results/training/runs/pod-*-001`) | backend validated; adapters unevaluated |
| GPT-4o runner + cost estimation (P1 Phase 8) | already provided by the audited inference CLI: dated-snapshot provider, token/cost estimates in every plan, budget caps, checkpoint/resume, resolve protocol | complete — **do not duplicate** |

## 2. Verified quantities

- 3,000 latent orders; 23,200 source queries; 5,637,600 prompt identities; 7,819,200
  comparison cells (Table 3 of the manuscript; matches `dataset_manifest.json`).
- 3,024 completed pilot inference calls (2 × 1,512; recomputed from the normalized
  records, not quoted from the paper).
- Pilot split: 300 orders, 1,900 queries, sizes 4–128 only. Frozen test: 1,800 orders,
  14,200 queries, sizes 4–1,024. **Stress sizes 256–1,024 exist only in the frozen test
  split** — a material constraint on P1 Phase 5 (see §5).

## 3. Status separation (current manuscript, verified consistent)

| Item | Status |
|---|---|
| Benchmark construction | complete (manuscript §6) |
| Pilot evaluation (2 models, 3,024 calls) | complete, exploratory (§7, Tables 7–9, Figure 5) |
| Expanded evaluation | pending (this P1 program) |
| Frozen-test evaluation | pending; gated on human review + approved pilot |
| Training smoke runs | complete (§7.6); five arms, declared pins validated on CUDA |
| Fine-tuned adapters | trained, **unevaluated** (evaluation design frozen at 10,224 calls/arm → review-gated) |

Contradiction check: the P1 prompt anticipates contradictions between "Section 5.2,
Table 2, Section 7". Under the current numbering (5 = experimental design, Table 2 =
RQ/readout table, 7 = pilot evaluation) no contradiction exists: the availability table
(Table 6) was updated to "Pilot (exploratory)" / "Trained; evaluation gated" when the
pilot and training sections were added, and a full status sweep on 2026-10-27 found no
stale "not run"/"unperformed" claims. A re-scan today found none either.

## 4. Reproducibility issues found

1. **Environment pins vs installed stack**: lexical/grid layers need `tiktoken==0.12.0`
   (vendored wheel is macOS-arm64; PyPI used elsewhere); test suite requires
   `PYTHONPATH=src:reports`. Documented in `reports/test_suite_status.md`; suite is
   1,034/1,034 green.
2. **No Ollama provider**: P1's local expanded evaluation runs *outside* the audited
   provider set. Quantized GGUF weights (Q4_K_M) are not the registered fp/bf16
   snapshots, so Ollama results are a **separate, supplemental, quantized arm** and must
   never be merged with registered-model results (P1 non-negotiable #8).
3. **Registry divergence**: the RunPod copy of `configs/models.yaml` enables
   `qwen25-15b-hf`; the local registry does not (tests pin the local enabled set).
4. Llama registry entries lack `chat_template_sha256` pins (blocker only if enabled).

## 5. Protocol conflicts the P1 design must respect (declared up front)

1. **The >2,000-call review gate.** The manuscript commits to a dataset-bound
   two-reviewer Turkish linguistic review before any evaluation above 2,000 calls.
   A ~15,000-call/model local run therefore cannot be presented as protocol-compliant
   until the review receipt exists. Resolution adopted here: the expanded run is
   designed, benchmarked, and **held at the approval stop**; if executed before the
   review, it must be labeled *supplemental exploratory evaluation, human review
   pending* in every artifact and manuscript mention — never as the audited framework's
   gated evaluation. Phase 4 prepares the actual review package so the receipt can
   exist before confirmatory claims.
2. **Frozen-test protection.** Primary expanded evaluation samples the **pilot split
   only** (sizes 4–128 available there). The optional 256–1,024 stress sizes would
   consume frozen-test items; they are designed but DEFERRED, labeled supplemental, and
   excluded from the default manifest.
3. **Pilot preservation.** Original pilot results are retained untouched; all P1
   outputs go to new write-once paths (`results/p1/…`).

## 6. Missing components to be built in this program

- `src/hardware_profile.py` (Phase 2) — M1/unified-memory/Ollama detection.
- `src/inference/ollama_runner.py` (Phase 3) — checkpointed local runner reusing the
  frozen prompt bank (byte-identical prompts from `grid.sqlite`), strict parser
  semantics, JSONL + raw preservation.
- `src/evaluation/semantic_validator.py` (Phase 4) — orchestrates the existing
  structural checks over the P1 sample; no LLM-as-judge.
- `validation/turkish_linguistic_review.csv` + reviewer instructions (Phase 4) —
  stratified, P3-oversampled, schema-compatible with `linguistic_review.py`.
- `src/evaluation/build_p1_manifest.py` (Phase 5) — paired, seeded, immutable manifest.
- `src/benchmark/ollama_speed_test.py` (Phase 6) — throughput/memory benchmark.

GPT-4o (Phase 8) maps onto the existing audited CLI and plan-level cost estimates; a
new runner will not be built. Statistics (Phase 9) reuse `src/analysis` definitions
via a thin P1 driver so no metric is redefined.

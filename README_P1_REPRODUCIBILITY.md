# P1 reproducibility — supplemental quantized local evaluation

End-to-end recipe for the supplemental quantized arm reported in manuscript
§7.7 and Appendix C.2. Everything runs locally against Ollama; no CUDA, cloud,
or API key is required (the 3B/7B/14B arms *may* run on any machine with
Ollama — ours ran on a rented A100 via `colab/P1_colab_runner.ipynb` — and the
results are verified identical in protocol by the integrity report).

## What this arm is, and is not

- **Is:** Qwen2.5-instruct 0.5B/3B/7B/14B, Ollama Q4_K_M GGUF, `/api/chat`,
  temperature 0, top_k 1, seed 42, `num_ctx` 4096, `num_predict` 32; the
  audited message constructor (run-time prompt SHA-256 re-verification) and
  the registered strict parser; all-planned estimand (invalid responses kept,
  flagged, scored as errors).
- **Is not:** the registered fp/bf16 provider pipeline. Results are labeled
  `supplemental_quantized_local_evaluation` on every record and are never
  pooled with registered runs. Exploratory pending the two-reviewer Turkish
  linguistic review (the run exceeds the 2,000-call threshold).

## Pinned inputs

| Artifact | Identity |
|---|---|
| Manifest `data/manifests/p1_expanded_manifest.jsonl` | SHA-256 `34f76f726e75f954e8326a491953ad5486b3cec51be3ad95a3aeb5debdd1242e` (16,618 prompts, pilot split, n ≤ 128) |
| qwen2.5:0.5b | digest `a8b0c51577010a279d933d14c2a8ab4b268079d44c5c8830c0a93900f1827c67` |
| qwen2.5:3b | digest `357c53fb659c5076de1d65ccb0b397446227b71a42be9d1603d46168015c9e4b` |
| qwen2.5:7b | digest `845dbda0ea48ed749caafd9e6037047aa19acfcfd82e704d7ca97d631a0b697e` |
| qwen2.5:14b | digest `7cdf5a0187d5c58cc5d369b255592f7841d1c4696d45a8c8a9489440385b22f6` |

(The canonical manifest SHA is also asserted inside every run header and by
the integrity report; trust those over this table if they ever disagree.)

## Steps

```bash
# 0. Ollama running locally with the four models pulled (digests above).

# 1. Manifest (write-once; refuses to overwrite) + semantic validation.
PYTHONPATH=src python3 -m evaluation.build_p1_manifest
PYTHONPATH=src python3 -m evaluation.semantic_validator \
    --manifest data/manifests/p1_expanded_manifest.jsonl

# 2. Execute each arm (sequential recommended; resumable and pausable).
./scripts/p1_run_all.sh          # all four, checkpointed, caffeinated
./scripts/p1_pause.sh            # graceful pause (exit 75 within one call)
./scripts/p1_resume.sh           # resume exactly, zero duplicates

# 3. Verify, analyze, plot.
PYTHONPATH=src python3 -m evaluation.p1_integrity_report \
    --manifest data/manifests/p1_expanded_manifest.jsonl \
    --results results/p1/qwen2-5-0-5b.jsonl results/p1/qwen2-5-3b.jsonl \
              results/p1/qwen2-5-7b.jsonl results/p1/qwen2-5-14b.jsonl \
    --expected-digests results/p1/expected_digests.json \
    --output reports/inference_integrity_report.md
PYTHONPATH=src python3 -m analysis.p1_statistics \
    --manifest data/manifests/p1_expanded_manifest.jsonl \
    --results results/p1/qwen2-5-0-5b.jsonl results/p1/qwen2-5-3b.jsonl \
              results/p1/qwen2-5-7b.jsonl results/p1/qwen2-5-14b.jsonl \
    --output results/p1/analysis
python3 paper/figures/manuscript/generate_p1_figures.py
```

Do not glob `results/p1/*.jsonl` into the analysis: the directory also holds
the ctx-8192 sensitivity artifacts.

## Executed-run provenance

- Integrity verdict: **PASS** (`reports/inference_integrity_report.md`) — all
  four arms 16,618/16,618, zero duplicates/missing/transport failures, one
  pinned digest per arm, protocol options echoed on every record.
- Context check: 11 TR n=128 prompts sit within 8 tokens of `num_ctx`; a
  `num_ctx=8192` sensitivity rerun returned identical token counts and
  predictions (`results/p1/sensitivity_qwen2-5-0-5b_ctx8192.jsonl`).
- Analysis summary: `reports/p1_statistical_analysis.md`. Statistics reuse the
  audited primitives only (`summarize_pairs`, `mcnemar`, `holm_adjust`,
  `wilson_interval`, latent-cluster `bootstrap`); where pooled pairs share
  latent clusters, McNemar is reported as undefined and inference is the
  cluster-bootstrap contrast, labeled exploratory.
- Manuscript integration: `reports/p1_manuscript_change_log.md`.
- Determinism note: options are requested and echoed, but Ollama backends
  differ across hardware; bit-identical outputs across machines are not
  claimed. Reproduction is at the protocol level (same prompts, options,
  parser, estimands, digests).

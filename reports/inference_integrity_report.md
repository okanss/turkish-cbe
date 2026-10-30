# P1 inference integrity report

Generated 2026-10-29T09:38:52+00:00 by `evaluation.p1_integrity_report`.

Manifest: `data/manifests/p1_expanded_manifest.jsonl` — SHA-256 `34f76f726e75f954e8326a491953ad5486b3cec51be3ad95a3aeb5debdd1242e`, 16,618 prompts. Protocol options on every call: `{"temperature": 0, "top_k": 1, "seed": 42, "num_ctx": 4096, "num_predict": 32}` via `/api/chat`.

All four arms are the **supplemental quantized local evaluation** (Ollama Q4_K_M GGUF); they are never pooled with registered-provider results. Invalid responses are parsed, flagged, and kept — none are discarded.

## Verdict

**PASS** — no missing or duplicate prompt IDs, single pinned weight digest per arm, manifest SHA verified in every run header, protocol options echoed on all calls, zero transport failures.

## Per-arm summary

| Model | Digest (Q4_K_M) | Records | Unique | Missing | Dup | Parse-valid | Invalid | Output-truncated | Transport err | All-planned acc. | Wall (h) |
|---|---|---|---|---|---|---|---|---|---|---|---|
| qwen2.5:0.5b | `a8b0c5157701…` | 16,618 | 16,618 | 0 | 0 | 16,618 | 0 | 0 | 0 | 0.4927 | 3.4 |
| qwen2.5:3b | `357c53fb659c…` | 16,618 | 16,618 | 0 | 0 | 16,593 | 25 | 9 | 0 | 0.5016 | 2.63 |
| qwen2.5:7b | `845dbda0ea48…` | 16,618 | 16,618 | 0 | 0 | 16,434 | 184 | 0 | 0 | 0.5873 | 5.02 |
| qwen2.5:14b | `7cdf5a0187d5…` | 16,618 | 16,618 | 0 | 0 | 16,371 | 247 | 247 | 0 | 0.6915 | 11.26 |

## Invalid responses and truncation

### qwen2.5:0.5b

No invalid responses, no output truncations.
- Response distribution (language, prediction): {'en/False': 9, 'en/True': 4925, 'tr/False': 11190, 'tr/True': 494}

### qwen2.5:3b

- Invalid (unparseable) responses: 25 — tr/P2: 2, tr/P3: 5, tr/P4: 3, tr/canonical: 7, tr/equivalent: 8.
- Output hit `num_predict` limit (done_reason=length): tr n=128: 9.
- Response distribution (language, prediction): {'en/False': 4726, 'en/True': 208, 'tr/False': 11414, 'tr/None': 25, 'tr/True': 245}

### qwen2.5:7b

- Invalid (unparseable) responses: 184 — tr/P1: 31, tr/P2: 5, tr/P3: 20, tr/P4: 42, tr/P5: 15, tr/canonical: 57, tr/equivalent: 14.
- Response distribution (language, prediction): {'en/False': 2907, 'en/True': 2027, 'tr/False': 972, 'tr/None': 184, 'tr/True': 10528}

### qwen2.5:14b

- Invalid (unparseable) responses: 247 — tr/P1: 1, tr/P2: 1, tr/P3: 1, tr/P4: 108, tr/P5: 136.
- Output hit `num_predict` limit (done_reason=length): tr n=128: 245, tr n=32: 1, tr n=64: 1.
- Response distribution (language, prediction): {'en/False': 3348, 'en/True': 1586, 'tr/False': 3606, 'tr/None': 247, 'tr/True': 7831}

## Context-window check

Eleven Turkish n=128 (P4/P5) prompts have prompt token counts within 8 of `num_ctx=4096` and are flagged `possible_context_truncation` in every arm (same 11 prompt IDs). A sensitivity rerun of these prompts at `num_ctx=8192` (`results/p1/sensitivity_qwen2-5-0-5b_ctx8192.jsonl`) returned byte-identical prompt token counts (4,088–4,095 — the prompts genuinely fit) and identical predictions, so no input truncation occurred under the protocol setting.

## Execution provenance

- qwen2.5:0.5b ran locally on the Apple M1 host (`results/p1/RUN_STATUS.txt`); qwen2.5:3b/7b/14b ran on a Colab A100 via `colab/P1_colab_runner.ipynb` against the byte-identical bundle (`colab/p1_colab_bundle.zip`) and the same frozen manifest; run headers above verify the manifest SHA and weight digests per arm.
- Caveat (predeclared): this expanded run exceeds the 2,000-call human-review threshold; the two-reviewer Turkish linguistic review of the 360-item sample is pending, so all P1 results are exploratory pending that receipt.

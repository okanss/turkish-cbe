# P1 expanded-evaluation sampling protocol

Builder: `src/evaluation/build_p1_manifest.py` (version p1-0.1.0, seed 20261028).
Manifest: `data/manifests/p1_expanded_manifest.jsonl` —
SHA-256 `34f76f726e75f954e8326a491953ad5486b3cec51be3ad95a3aeb5debdd1242e`,
**16,618 prompts**, write-once; per-stratum pools, quotas and inclusion
probabilities in `p1_expanded_manifest.protocol.json`.

## Source and partition policy

All prompts come from the frozen, pre-materialized **pilot split** bank
(`data/pilot/prompts.jsonl`; 33,100 prompts over 300 latent orders, 1,900 query
instances, sizes 4–128). Prompt text and SHA-256 are copied byte-identically; the
runner re-verifies each SHA at inference time. The frozen test split is untouched:
the optional 256–1,024 stress sizes exist only there and are **deferred**, not
silently substituted. Original partitions are read-only. This evaluation is
therefore a **supplemental exploratory evaluation on the pilot split** — not the
frozen-test protocol, and (pending the two-reviewer linguistic review) not
confirmatory; every derived artifact carries that label.

## Unit of sampling: matched groups, never prompts

The sampled unit is a complete matched group — one (query instance × relation ×
name control) with every presentation cell present — so all paired comparisons stay
intact and one model response serves every comparison it is eligible for.

| Component | Cells per group | Pool | Sampled | Calls | Covers |
|---|---|---|---|---|---|
| nine_way | EN{P0,eq} + TR{P0,eq,P1–P5} | 2,700 (300 × 9 relations; 450 per n ∈ {4,8,16,32,64,128}; endpoint) | 1,350 (25 per relation × n stratum, gold-balanced) | 12,150 | language gap, P1–P5 robustness, relation families, entity-count scaling |
| geometry_internal | EN/TR × {P0,eq} | 300 (spatial_front, internal) | 300 (exhaustive) | 1,200 | endpoint-shortcut control |
| geometry_distance | EN/TR × {P0,eq} | 1,300 (spatial_front, distance-controlled) | 442 (~per (n, d) cell, gold-balanced) | 1,768 | query-distance effects |
| naming | EN/TR canonical × 3 name controls | 300 queries complete in all three controls | 300 (exhaustive) | 1,800 | naming-condition sensitivity |

Deduplicated total: 16,618 prompts per model (naming overlaps nine_way at
native_shared). Gold balance 8,598 TRUE / 8,020 FALSE (exact balance is impossible
because the exhaustive components are taken whole; per-stratum balance is recorded).

## Deviations from the original protocol (declared)

1. Pilot split instead of frozen test: deliberate, to protect the frozen set and
   stay consistent with the manuscript's gate structure.
2. Quantized Ollama arm (Q4_K_M GGUF) instead of the registered fp/bf16 snapshots:
   results form a separate supplemental arm; never pooled with registered-provider
   results and never presented as equivalent to full-precision inference.
3. ~16.6k calls/model exceeds the 2,000-call threshold that the audited framework
   couples to the human linguistic review. Execution before the review receipt is
   possible only as *exploratory, human review pending* — the same epistemic label
   as the existing pilot — and the review package (Phase 4) exists so the receipt
   can follow.
4. P-conditions are evaluated on all nine relations at endpoint queries (the bank's
   complete-group coverage); distance/internal geometry is spatial_front only, as in
   the bank. Cross-relation geometry requires a future grid materialization.

# P1 statistical analysis — supplemental quantized local evaluation

Generated 2026-10-29 from `results/p1/analysis/` (produced by
`analysis.p1_statistics`, version p1-0.2.0, which reuses the audited
primitives: `summarize_pairs`, `mcnemar`, `holm_adjust`, `wilson_interval`,
and the latent-cluster `bootstrap`). Integrity: `reports/inference_integrity_report.md`
(verdict PASS).

**Arm label (applies to every number below):** supplemental quantized local
evaluation — Ollama Q4_K_M GGUF weights, `/api/chat`, temperature 0, top_k 1,
seed 42, `num_ctx` 4096, `num_predict` 32. These are *not* the registered
fp/bf16 snapshots; results are never pooled with registered-provider runs and
quantized accuracy is never claimed equivalent to full precision.
**Status: exploratory.** The run exceeds the predeclared 2,000-call threshold and
no execution-gate receipt was issued. The two-reviewer 360-item stimulus review
was completed 2026-10-30 (unanimous equivalence; 16 P3×temporal_completion items
excluded on grammaticality; naturalness contested — see
`validation/review_results/` and `results/p1/analysis/p3_review_sensitivity.json`:
the 14B P3 contrast deepens to −0.408 with the excluded relation removed).

## 1. Coverage

Manifest: `data/manifests/p1_expanded_manifest.jsonl` — 16,618 prompts
(pilot split only, sizes 4–128; SHA-256 `34f76f72…`). Every arm answered all
16,618 prompts; no prompt was discarded.

| Model | Digest | Answered | Invalid (kept, flagged) | All-planned acc. |
|---|---|---|---|---|
| qwen2.5:0.5b | `a8b0c515…` | 16,618 | 0 | 0.4927 |
| qwen2.5:3b | `357c53fb…` | 16,618 | 25 | 0.5016 |
| qwen2.5:7b | `845dbda0…` | 16,618 | 184 | 0.5873 |
| qwen2.5:14b | `7cdf5a01…` | 16,618 | 247 | 0.6915 |

Invalid responses are concentrated in Turkish (all 25/184/247) and, for 14b,
coincide with `num_predict`-length truncations (247). All-planned accuracy
counts them as errors (the predeclared primary estimand).

## 2. Scale and language (descriptive + F1)

All-planned accuracy by language:

| Model | EN | TR | Gap |
|---|---|---|---|
| 0.5b | 0.5174 | 0.4822 | 0.035 |
| 3b | 0.5241 | 0.4921 | 0.032 |
| 7b | 0.7144 | 0.5336 | 0.181 |
| 14b | 0.7641 | 0.6608 | 0.103 |

0.5b and 3b are near-constant responders (0.5b: TR→HAYIR on 95.8% of TR calls,
EN→YES on 99.8% of EN calls; 3b: both languages→NO), so their accuracies sit at
the base-rate. Discrimination emerges at 7b in English only, and in both
languages at 14b.

F1 paired contrast (EN canonical vs TR canonical, same queries; Δ = TR − EN,
latent-cluster bootstrap 95% CI; ordinary McNemar is undefined here because
latent clusters repeat across pairs, so inference is the audited pipeline's
exploratory cluster-bootstrap contrast):

| Model | Δ paired accuracy | 95% CI | Credible? |
|---|---|---|---|
| 0.5b | −0.036 | [−0.160, +0.079] | no |
| 3b | −0.027 | [−0.039, −0.017] | yes |
| 7b | −0.176 | [−0.241, −0.106] | yes |
| 14b | −0.075 | [−0.126, −0.020] | yes |

The 0.5b F1 profile is the degenerate signature in its purest form:
R = 0.017, U = 0.970 — on nearly every matched pair exactly one language is
"correct", purely because the two constant responses disagree.

## 3. Presentation robustness (F2)

Paired Δ (condition − TR canonical) on query-matched pairs, cluster-bootstrap
95% CI. Starred rows have CIs excluding zero:

| Contrast | 0.5b | 3b | 7b | 14b |
|---|---|---|---|---|
| Equivalent | +0.003 | +0.004 | −0.021\* | −0.081\* |
| P1 | −0.002 | −0.009 | +0.017 | −0.011 |
| P2 | +0.003 | −0.012 | −0.045\* | −0.095\* |
| P3 | −0.002 | −0.010 | **−0.079\*** | **−0.381\*** |
| P4 | −0.005 | +0.010\* | −0.024\* | −0.059\* |
| P5 | +0.007 | −0.004 | −0.048\* | −0.129\* |

The gpt-4o presentation cliff **replicates in open weights and deepens with
scale**: at 14b, P3 costs 38.1 points of paired accuracy
(CI [−0.418, −0.344]; discordant pairs 554 vs 40), taking TR P3 accuracy to
0.431 — below the 0.5 base rate. The pooled by-condition accuracies in
`p1_accuracy_by_axis.csv` are *not* corrected for strata mix (e.g., pooled P1
accuracy exceeds canonical at 14b purely compositionally); the paired deltas
above are the estimand.

Per-stratum κ_CBE (14b, canonical vs equivalent) decays with complexity:
0.573 (n=4) → 0.327 (8) → 0.351 (16) → 0.215 (32) → 0.127 (64) → 0.118 (128).
For canonical vs P3 at 14b, κ_CBE turns *negative* at n=16 and n=32 (−0.218,
−0.195): systematic answer flips rather than noise. κ is reported per fixed-n
stratum only; pooled rows carry no κ by design (manuscript §4.2).

## 4. Naming controls (F3)

Naming-component pairs are cluster-unique, so exact/asymptotic McNemar applies
(Holm within family):

| Model | native vs en_transliterated | native vs symbolic_shared |
|---|---|---|
| 0.5b | Δ 0.000, Holm p = 1.0 | Δ +0.013, Holm p = 1.0 |
| 3b | Δ 0.000, Holm p = 1.0 | Δ −0.037, Holm p = 0.272 |
| 7b | Δ −0.003, Holm p = 1.0 | Δ +0.011, Holm p = 1.0 |
| 14b | Δ −0.013, Holm p = 0.388 | **Δ −0.117, Holm p = 0.0004** |

At 14b, replacing native Turkish names with shared symbolic labels costs 11.7
points (b = 59, c = 24, McNemar p = 1.9×10⁻⁴) — entity-name surface form
matters once competence exists, while transliteration alone does not.

## 5. Complexity

14b all-planned accuracy decays monotonically with n in Turkish
(0.839 at n=4 → 0.508 at n=128, i.e., to chance) and more slowly in English
(0.941 → 0.670). 7b shows the same decay in English only. This mirrors the
registered gpt-4o pilot pattern at larger scale.

## 6. Interpretation boundaries

1. Quantized (Q4_K_M) weights: no claim of equivalence to full-precision
   registered snapshots; cross-arm comparisons are qualitative replications.
2. Exploratory pending the two-reviewer Turkish linguistic review receipt.
3. Pilot split only; frozen-test sizes (256–1,024) untouched.
4. F1/F2 inference is cluster-bootstrap CIs (exploratory label), not McNemar,
   because latent clusters repeat across pooled pairs; F3 uses McNemar + Holm.
5. Figures: `paper/figures/manuscript/fig6_p1_quantized_scaling.pdf`
   (provenance in `p1_plot_manifest.json`).

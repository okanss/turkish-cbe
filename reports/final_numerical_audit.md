# Final numerical audit — main_GPTreview (2026-10-30)

Every value below was recomputed or read directly from the retained analysis
outputs (`results/p1/analysis/`, `results/p1/analysis/p3_review_sensitivity.json`,
`validation/review_results/review_summary.json`) by script in this session and
compared against the manuscript. **Result: 34/34 automated checks PASS; zero
unresolved discrepancies.**

## Verified equations (§4)

- R + D + U = 1 on jointly valid support — holds in every reported profile row
  (e.g., Table 8 P3 row: 0.298 + 0.250 + 0.452 = 1.000; column 4 of Table 8 is
  the pooled affirmative rate q, not I).
- A_valid = R + U/2, I = R + D, R_strict = v·R — definitional identities of
  Eqs. (3)–(7); denominators and validity conditions stated at each use.
- κ_CBE per Eq. (8), defined per fixed-n stratum only; pooled rows carry no κ
  (enforced by the audited summarizer).

## Verified model statistics (supplemental sweep; 16,618 prompts/model)

| Quantity | Manuscript | Source value | Status |
|---|---|---|---|
| All-planned accuracy 0.5B/3B/7B/14B | 49.3/50.2/58.7/69.2% | 0.4927/0.5016/0.5873/0.6915 | PASS |
| Answered per arm / invalid | 16,618 each; 0/25/184/247 | coverage table | PASS |
| 14B EN / TR | 76.4% / 66.1% | 0.7641 / 0.6608 | PASS |
| 7B EN / TR | 71.4% / 53.4% | 0.7144 / 0.5336 | PASS |
| 14B language gap | −0.075 [−0.126, −0.020] | −0.0749 [−0.1261, −0.0201] | PASS |
| 7B language gap | −0.176 [−0.241, −0.106] | −0.1761 [−0.2408, −0.1061] | PASS |
| 14B P3 effect | −0.381 [−0.418, −0.344]; b=554, c=40 | exact | PASS |
| 7B P3 effect | −0.079 | −0.0787 | PASS |
| P3 sensitivity (excl. temporal_completion) | −0.408 [−0.446, −0.370] | −0.4078 [−0.4456, −0.3695] | PASS |
| 14B TR P3 accuracy | 43.1% | 0.4311 | PASS |
| 14B naming (symbolic) | −11.7 pp; b=59, c=24; Holm p=3.8×10⁻⁴ | −0.1167; 0.00038 | PASS |
| 14B complexity TR n=4→128 | 83.9% → 50.8% | 0.8391 → 0.5083 | PASS |
| 14B complexity EN n=4→128 | 94.1% → 67.0% | 0.9408 → 0.6703 | PASS |
| 0.5B F1 degenerate profile | R=0.017, U=0.970 | 0.0165, 0.9697 | PASS |
| 14B κ_CBE (can. vs equiv.) n=4→128 | 0.57 → 0.12 | 0.5733 → 0.1177 | PASS |
| 0.5B response bias | TR hayır 95.8%, EN yes 99.8% | 11,190/11,684; 4,925/4,934 | PASS |

## Verified pilot statistics (registered runs)

Verified at integration time against `results/normalized/pilot-*-003.jsonl`
and the released pilot analysis tables; internally re-checked this session:
1,512/1,512 calls per run, zero invalid; GPT-4o EN 72.0% vs TR 57.8%; P3
36.9%; Table 8 profiles sum to 1 per row; κ_CBE(P3) = −0.027; direction
diagnostics (Table 9) unchanged.

## Verified dataset and review counts

- Inventory: 3,000 orders / 23,200 queries / 5,637,600 prompt identities /
  7,819,200 cells; splits 450/450/300/1,800 (Table 3) — matches the frozen
  manifest-backed Table 1 audit.
- Sweep manifest: 16,618 prompts, SHA-bound; integrity report verdict PASS.
- Review: 360 items; 344 eligible; 16 excluded, all P3×temporal_completion on
  grammaticality; equivalence agreement 1.0 (324 transformed pairs);
  ambiguity agreement 1.0 — matches `review_summary.json`. PASS.

## Confidence-interval conventions

All signed differences now carry signed CIs matching their estimand direction
(condition − canonical; TR − EN); percentage-point reductions are stated
separately from proportions. Holm-adjusted p-values appear only for the
cluster-unique naming family; all other families use latent-order
cluster-bootstrap intervals labeled exploratory.

## Unresolved discrepancies

None.

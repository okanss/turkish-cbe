# GPTreview revision report — Turkish-CBE manuscript

Revision executed 2026-10-30 on git branch `gptreview` (base: `main`,
manuscript of 30 October 2026). Scope: scientific positioning, results
presentation, consistency, and submission readiness. **No experiment was rerun,
no number was changed, no gate was loosened, and all results remain labeled
exploratory.** All numerical values in revised text were verified against
`results/p1/analysis/`, `results/p1/analysis/p3_review_sensitivity.json`, and
`validation/review_results/` before reproduction.

## Deliverable 2 — Revision summary

| Section | Modification | Scientific rationale | Status |
|---|---|---|---|
| Title | "Beyond Translation…: A Reproducible Framework, Construction Audit, and Pilot Evaluation" → **"Turkish-CBE: Evaluating Logical Reasoning Accuracy and Morphosyntactic Robustness in Large Language Models — A Controlled Turkish–English Benchmark with Correctness-Aware Robustness Evaluation"** | Names the benchmark, reflects the expanded 5-model empirical evaluation, drops the subtitle that undersold completed work; avoids causal morphology claims | done |
| Abstract | Rewritten, 230 words: problem → method → experiments → findings (constant responders, scale, P3 −38.1 pp → −40.8 pp after exclusion) → review → joint-evaluation conclusion | Leads with contributions instead of dataset inventory; every number verified | done |
| §1 Introduction | Restructured into the six required paragraph groups; **three principal contributions** (A controlled bilingual benchmark, B correctness-aware robustness, C empirical characterization) replace the old four-item list; RQ evidence status split (exploratory RQ1–RQ3/RQ4-diagnostics vs. future-confirmatory) | Positions contributions once, without repetition; explicitly disclaims eliminating all confounders and generalizing beyond evaluated models | done |
| §5.1 Table 2 (tab:rqs) | Caption "All model-outcome estimands are pending" → distinguishes completed exploratory estimates from pending confirmatory/adapter estimands | Outdated status corrected | done |
| §5.2 | "Current model registries remain disabled and no real model run is reported" → accurate description of the two registered runs + labeled supplemental sweep; gate receipts noted as unissued | Factual correction; protocol text retained as methodology | done |
| §5.3 | "No adapter training, CUDA smoke run… is claimed" → executed smoke-scale runs of all five arms acknowledged as feasibility/accounting evidence only | Conflict with §7.6 removed; no adaptation benefit implied | done |
| §6.4 + Table (tab:availability) | "unreviewed stimuli" removed; statuses now Pilot + sweep (exploratory); new row: stimulus linguistic validity = Completed (sampled) | Review completed; availability table current | done |
| §7 title + intro | "Pilot Model Evaluation" → "Exploratory Model Evaluation and Linguistic Validation"; "human review remains pending" → chronology of runs then review; gate-receipt status stated | Section now spans pilot + sweep + review | done |
| §7.9 (closing subsection) | Condensed; review/P3 synthesis moved to Discussion to remove duplication | De-duplication (Task 14) | done |
| §8 Discussion | Restructured to the required 8.1–8.6 outline; new consolidated findings table (Table 10) with exact numbers; scaling subsection adds the key claim "improved canonical accuracy does not guarantee presentation robustness" with 14B as the example; P3 subsection integrates review + sensitivity bounds | Tasks 6, 7, 8, 11 | done |
| §9 Limitations | Reorganized: five limitation paragraphs (completed work no longer described as missing) + new **Future work** subsection (frozen test, gate approvals, model families, adapter contrast, prompting, coverage, lexical analyses) | Task 12; gates framed as future procedures, not blockers | done |
| §10 Conclusion | Rewritten around the three contributions; ends with the required takeaway sentence; no new numbers | Task 13 | done |
| Figures | Figure 6 retained unchanged (already panels scale/presentation-effects/complexity with paired estimands); optional accuracy-vs-R scatter **not added** (see Deliverable 4) | Task 9 | done |
| Terminology | GPT-4o standardized in prose (literal snapshot `gpt-4o-2024-08-06` kept in IDs); P0–P5, Q4_K_M, "supplemental quantized" usage checked; new labels `sec:statistics`, `sec:discussion`, `sec:limitations`, `sec:future` | Task 14 | done |
| Typography | New-content overfull boxes fixed (findings table widths, appendix paths); 0 undefined references; 30 pp. | Task 14 | done |

## Deliverable 3 — Consistency audit (contradictions found and resolved)

| # | Location | Original statement | Resolution |
|---|---|---|---|
| 1 | §5.1 Table 2 caption | "All model-outcome estimands are pending." | Exploratory estimates exist (pilot + sweep); pending = confirmatory + adapter only |
| 2 | §5.2 (end) | "Current model registries remain disabled and no real model run is reported." | Two registered runs executed by explicit selection; sweep labeled supplemental; receipts unissued |
| 3 | §5.3 (end) | "No adapter training, CUDA smoke run, or optional Torch numerical/gradient validation is claimed." | Smoke runs of all five arms executed on rented CUDA; feasibility only, no performance claim |
| 4 | §6.4 | "…one relation family, and unreviewed stimuli" | Review completed (§7.8); sweep covers nine relations |
| 5 | §6.4 Table | RQ1–RQ3 status "Pilot (exploratory)" only; no review row | "Pilot + sweep (exploratory)"; review row "Completed (sampled)" added |
| 6 | §7 intro | "…human linguistic review of the Turkish stimuli remains pending" | Chronology stated: runs first, review completed after, on the same prompt population |
| 7 | §7.6 (RQ5) | (previously fixed) "requires the pending human linguistic review" | Already distinguishes the completed stimulus review from the unissued execution-gate receipt |
| 8 | §8.5 (old) | "gaps that must be filled: human review, frozen-test model calls…" | Review no longer a gap; restructured §8.6 lists only genuinely open items |
| 9 | §7.9 vs new §8 | Review/P3 synthesis stated in both | §7.9 condensed to protocol scope; synthesis lives in §8.2/§8.6 |
| 10 | Prose naming | "gpt-4o" in prose vs "GPT-4o" | GPT-4o in prose; lowercase reserved for the literal snapshot identifier |
| 11 | Title page date | "27 October 2026" | "30 October 2026 (GPTreview revision)" |

Phrase sweep for "no real model run", "estimands are pending", "review remains
pending", "unreviewed", "no CUDA", "No adapter training": zero remaining
occurrences outside correct contexts (the one hit, §5.3 "the local machine has
no CUDA device", is a true hardware statement).

## Deliverable 4 — Remaining limitations (not resolvable by editing)

1. **Optional accuracy-vs-robustness figure not added.** The required paired
   measurements exist (canonical accuracy; pooled R from canonical-vs-equivalent
   pairs), but only four comparable points exist in the sweep cohort, and the
   GPT-4o pilot cohort (one relation, n ≤ 32, full precision) may not be pooled
   with it. A four-point scatter adds nothing beyond Table 10 and §8.3's prose,
   so per Task 9's own criterion it was omitted. The R values are reported in
   text instead.
2. **Naturalness disagreement** (P3 medians 4 vs. 2) is documented variation;
   only naturalness-controlled stimuli (future work item 6) can remove it.
3. **Single quantized model family** in the sweep; wording is scoped, but only
   new experiments can widen family coverage.
4. **Confirmatory status** of RQ1–RQ5 requires the frozen test and gate
   receipts; by design this cannot be edited into existence.
5. **MDPI template conversion** is deliberately deferred: MDPI accepts
   free-format first submissions, and the back-matter sections (contributions,
   availability, funding, COI) are present.
6. Two `%% TODO(submission)` markers remain in source by prior decision:
   archival DOI + repo flip to public at acceptance; CRediT split confirmation.

## Deliverable 5 — Submission-readiness assessment (MDPI Information)

| Criterion | Assessment |
|---|---|
| Scientific contribution | Clear and now explicitly structured: controlled bilingual benchmark + correctness-aware robustness evaluation + 5-model empirical characterization with a completed human linguistic review. The P3 finding (−38.1 pp, persisting at −40.8 pp after the review exclusion) is a substantial, well-bounded empirical result. |
| Methodological rigor | Strong: deterministic generation, exact oracles, frozen manifests, declared denominators, predeclared families and inclusion policy, documented execution gates. |
| Empirical support | Substantial and honestly scoped: 2 registered runs + 4×16,618-call sweep + 360-item two-reviewer review; every claim carries its cohort and exploratory label. |
| Statistical reporting | Consistent estimand definitions; cluster-bootstrap CIs where McNemar is undefined; Holm within predeclared families; prompt counts never presented as independent n. |
| Linguistic validity | Sampled two-reviewer review, unanimous equivalence, one grammaticality exclusion with a sensitivity reanalysis; naturalness disagreement reported as qualification, not suppressed. |
| Reproducibility | Code/data repository cited; appendices give exact rebuild/run commands; hash-bound artifacts. |
| Organization & readability | Discussion now synthesis-structured (8.1–8.6); consolidated Table 10; de-duplicated; 30 pp., clean compile. |
| Suitability for MDPI Information | Good fit (AI/information-systems scope, reproducibility emphasis); free-format submission acceptable. |

**Genuine remaining blockers:** none. Pre-submission checklist (non-blocking):
confirm CRediT split with the co-author; flip the GitHub repository public and
(optionally) mint a Zenodo DOI at or before acceptance; verify the journal's
current abstract word limit at upload.

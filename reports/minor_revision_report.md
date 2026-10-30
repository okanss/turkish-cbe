# Minor revision report — main_GPTreview (2026-10-30, final)

Scope: three scientific reporting issues + verification passes. Applied to
`paper/main_GPTreview.tex` + `paper/sections_gptreview/` only. No experiment
run, no number changed, no gate loosened.

| Priority | Location | Identified issue | Correction | Evidence |
|---|---|---|---|---|
| P1 | §7.7 (language gap) | 14B marginals (76.4%/66.1% → −10.3 pp) appear inconsistent with the reported paired gap (−7.5 pp) | **Case A — both correct, different estimands.** Added passage defining both: marginals pool all conditions/name controls (4,934 EN / 11,684 TR prompts); paired gap is the mean TR−EN difference on 2,242 query-matched canonical/native pairs (EN 78.2%, TR 70.7%); explicit statement that the paired estimate is not the marginal subtraction | Recomputed from raw records: marginals 0.7641/0.6608 exact; matched-pair accuracies 0.7823/0.7074 → −0.0749 exact; 7B analogously verified (−0.1761). Full audit: `reports/language_gap_verification.md` |
| P1 | §8 Table 10 (Language row) | Evidence cell juxtaposed pooled marginals with the paired gap without flagging the different populations | Evidence now leads with matched-pair accuracies (78.2% vs. 70.7%) and labels 76.4%/66.1% as all-condition marginals | Same verification |
| P2 | §8 Table 10 (Invariance row) | Perfect-invariance claim was supported by cross-language statistics (R = 0.017, U = 0.970), which show disagreement, not invariance | Split into two rows: **Invariance** (within-language: pilot 0.5B I = 1.00, κ undefined; sweep 0.5B TR canonical-vs-equivalent I = 0.96) and **Response bias** (cross-language: R = 0.017, U = 0.970 from constant *yes*/EN, *hayır*/TR) | I = R + D from `p1_paired_profiles.csv` (0.4627 + 0.5000 = 0.9627); pilot I = 1.00 from the registered run's constant responses (§7.2); cross-language values from the F1 pooled profile |
| P2 | §8.1 | Prose ran the two observations together ("attains perfect raw invariance … and its English–Turkish pairs show …") | Rewritten to present them as complementary, explicitly named observations: within-language invariance ≠ cross-language agreement; disagreement attributed to language-conditioned bias, not instability | Text + same sources |
| P2 | §7.8 | "a regime in which a competent human reader does not lose the ability to evaluate the stated facts" — the review measured linguistic judgments, not human task accuracy | Replaced with the evidence-aligned formulation (equivalence + grammaticality supported; naturalness differences documented; P3 persistence after exclusion means the sensitivity is not solely attributable to the excluded construction) and an explicit sentence that no claim about human reasoning performance is made | Review artifacts contain only linguistic ratings (`validation/review_results/`) |
| P2 | §8.2 | Same human-reader implication at the subsection's close | Replaced: the review "does not license a comparison with human reasoning performance on the same prompts" | Same |
| P2 | Availability | (Verification) | Re-verified: statement says the repository "will be made publicly accessible upon publication" — matches actual status (private, exists, all 17 Appendix-C paths present and tracked). No DOI invented | Repo check this session |
| P3 | Whole MS | Consistency verification of prior corrections | Confirmed: κ_CBE never read as independence; Δ_{P3−P0} = −0.381 [−0.418, −0.344] consistent in abstract/§7.7/§8.2/Table 10/figure caption; McNemar wording correct in §7.7/§9; experimental-status distinctions intact; no universal scale-threshold claim; R+D+U identities hold in every reported profile row | grep sweeps + numerical audit (34/34 from the previous pass still valid; new rows re-checked) |
| P3 | Figures/tables | Final check | Figure/table numbering and cross-references resolve (0 undefined); Table 10 (now 7 rows) renders without overflow; Figures 5–6 captions accurate; no internal draft labels | Compile log + rendered-page inspection |

Compile status: `main_GPTreview.pdf`, **31 pages**, 0 undefined
references/citations; the only remaining overfull boxes (≤29 pt) predate these
revisions and sit in unrelated paragraphs.

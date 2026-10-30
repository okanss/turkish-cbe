# Final submission check — main_GPTreview.pdf (MDPI Information)

Date: 2026-10-30, after the minor revision
(`reports/minor_revision_report.md`, `reports/language_gap_verification.md`).

| Criterion | Status | Notes |
|---|---|---|
| Numerical consistency | ✅ | The 14B language-gap "discrepancy" resolved as two verified estimands (marginals vs. matched-pair contrast), now explained in §7.7 and Table 10; all previously audited values (34/34) unchanged and still matching the retained outputs; new numbers (78.2%, 70.7%, I = 0.96, I = 1.00) recomputed from raw records/profiles this session. |
| Statistical interpretation | ✅ | κ_CBE ≈ 0 not read as independence; signed effects with signed CIs; cluster-bootstrap inference where rows share latent orders; within-language invariance vs. cross-language agreement now explicitly distinct (Table 10 split rows + §8.1). |
| Linguistic validation reporting | ✅ | §7.8 claims only what was measured: unanimous semantic equivalence, grammaticality outside one excluded construction, contested naturalness, P3 persistence after exclusion; explicit disclaimer that human reasoning accuracy was not measured. |
| Figure & table consistency | ✅ | 31 pages; 0 undefined references; Table 10 (7 rows) renders cleanly; Figure 6 annotated variant generated from retained CSVs; captions name estimand directions and exploratory status. |
| Reproducibility | ✅ | All Appendix-C artifact paths verified present and tracked in the repository; availability statement matches actual status (private repo, public upon publication); no DOI or release invented. |
| Manuscript metadata | ✅ | No internal draft labels; title, authors, ORCIDs, corresponding author, keywords, funding (none), COI (none), contributions, availability present. |
| Experimental scope | ✅ | Exploratory status preserved throughout; frozen test, gate approvals, adapter evaluation, added model families remain future work; no new inference, training, or human evaluation performed. |

## Remaining submission blockers

**None.**

## Non-blocking checklist at upload time

1. Flip the GitHub repository public (± Zenodo DOI) at or before acceptance;
   update the availability sentence to the live form.
2. Confirm the CRediT contribution split with the co-author (single
   `%% TODO(submission)` marker remains at that line).
3. Check the journal's current abstract word limit (~235 words as written).
4. Submit `paper/main_GPTreview.pdf`; `paper/main.pdf` and `paper/versions/`
   retain the pre-revision lineage for provenance.

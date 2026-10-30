# Submission readiness — main_GPTreview.pdf (MDPI Information)

Assessed 2026-10-30 after the final scientific revision
(`reports/final_revision_log.md`) and numerical audit
(`reports/final_numerical_audit.md`, 34/34 PASS).

| Criterion | Status | Notes |
|---|---|---|
| Scientific consistency | ✅ | The four P1 issues corrected: κ_CBE no longer read as independence; signed-CI conventions uniform; §7.1 protocol claim reconciled with unissued gate receipts; McNemar language distinguishes definability from inferential validity. Exploratory status unambiguous throughout. |
| Statistical reporting | ✅ | Declared estimands with directions; cluster-bootstrap CIs where rows share latent orders; Holm only in the cluster-unique family; prompt counts never presented as independent n. |
| Experimental transparency | ✅ | Registered pilots vs. supplemental quantized sweep vs. completed review vs. prospective gates clearly separated (§5.2, §7.1, §7.7–7.8, App. C); invalid responses retained and counted as errors. |
| Linguistic validation | ✅ | Completed two-reviewer review reported without overclaiming: unanimous equivalence, one grammaticality exclusion with persisting effect (−0.408), naturalness disagreement documented as qualification. |
| Reproducibility | ✅ | All 17 Appendix-C artifact paths verified present in the repository this session; commands match the code; availability statement now accurately says the repository will be made publicly accessible upon publication (it is currently private). |
| Figure quality | ✅ | Figure 6 (annotated variant) regenerated from retained CSVs only; direction, zero-reference, and CI identified in caption; colors/labels legible; quantized arm labeled. Figures 1–5 unchanged. |
| Reference consistency | ✅ | BibTeX resolves; 0 undefined citations/references; no new references needed. |
| Submission metadata | ✅ | Internal draft label removed; title, authors, ORCIDs, corresponding author, keywords, funding (none), COI (none), contributions, availability present. |
| Compilation | ✅ | XeLaTeX, 30 pages, 0 undefined references; two pre-existing ≤29 pt overfull boxes in unrelated paragraphs, content not clipped. |

## Remaining blockers

**None for submission.** MDPI accepts free-format first submissions, and all
required back-matter statements are present.

## Non-blocking pre-submission checklist

1. Flip the GitHub repository to public (and optionally mint a Zenodo DOI) at
   or before acceptance; then replace "will be made publicly accessible upon
   publication" with the live statement.
2. Confirm the CRediT contribution split with the co-author (currently "both
   authors"; a `%% TODO(submission)` marker remains at that line only).
3. At upload, check the journal's current abstract word limit (abstract is
   ~235 words) and keyword count (10 provided).
4. Submit `main_GPTreview.pdf`; the pre-revision version remains available as
   `paper/main.pdf` and in `paper/versions/` for provenance.

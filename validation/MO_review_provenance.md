# Provenance — reviewer MO's completed 360-item review

- Source artifact: `validation/turkish_linguistic_review_MO.csv` (UTF-8 CSV,
  2026-10-30 13:06), completed by reviewer MO (Müge Oluçoğlu), working
  independently of reviewer OB (no access to OB's scores or distributions,
  which were deliberately kept unsummarized until this file arrived).
- Annotation columns were merged onto the pristine shipped CSV row-by-row,
  keyed and verified by `prompt_id` (360/360 matched, order unchanged).
- Canonical rows: MO marked `semantic_equivalence` as `-` on all 36 canonical
  rows; normalized to `n/a` per the predeclared canonical-row convention.
  Mixed-case `Yes`/`No` values normalized to lowercase. No other changes.
- Final file: `validation/turkish_linguistic_review.MO.csv`, 360 rows, zero
  blank rating cells, SHA-256
  `bd96ed11d4d62549d3e2ee14c37a2d20412326e53863898b71477fdcc7c6aa83`.

# Note on `~/Downloads/turkish_linguistic_review_V1.csv`

That file (2026-10-29 14:29, a renamed Apple Numbers document) was identified
by reviewer OB, in writing on 2026-10-30, as **OB's own earlier draft**, not a
second reviewer's work. It was not used for anything: not as reviewer 2, not to
amend OB's final file. Recorded here because its existence and timing could
otherwise be misread as a third annotation source.

# Adjudication

Joined decisions, the predeclared inclusion policy, and agreement statistics:
`validation/review_results/` (produced by `evaluation.review_adjudication`).

# Provenance — reviewer OB's completed 360-item review

- Source artifact: `turkish_linguistic_review_OB.numbers` (Apple Numbers,
  2026-10-29 17:05), completed by reviewer OB (Okan Bursa).
- Annotation columns were extracted from the Numbers table and merged onto the
  pristine shipped CSV row-by-row, keyed and verified by `prompt_id` (360/360
  matched, order unchanged); prompt text and metadata come from the original
  file, so no spreadsheet mangling can have entered the reviewed text.
- Canonical rows: the reviewer recorded `semantic_equivalence = yes` on 32
  canonical rows (self-equivalence reading) and `n/a` on 4; values kept as
  recorded. Analysis treats equivalence on canonical rows as n/a per the
  clarified predeclared rule (grammar >= 4 and ambiguity = no from both
  reviewers decide canonical eligibility).
- Naturalness on 32 canonical rows was left "N/a" in the Numbers file
  (completed before the 2026-10-29 instruction clarification). The reviewer
  subsequently directed, in writing: "N/A is 5 for empty ones" — i.e., those 32
  canonical naturalness scores are 5. Applied as a batch value on the
  reviewer's explicit instruction; recorded here because it was supplied as a
  batch decision rather than per item.
- Final file: `validation/turkish_linguistic_review.OB.csv`, 360 rows, zero
  blank rating cells, SHA-256
  `929f0087062fabff2813f6c4e09e44b8b280d78db1ac2a5cdc0bfa8ed30a3ace`.
- Judgment distributions are deliberately not summarized anywhere until
  reviewer 2 completes their independent copy.

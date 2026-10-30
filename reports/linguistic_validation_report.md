# P1 linguistic validation report

Date: 2026-10-28. Two layers, kept strictly separate: completed **automatic**
re-verification, and a prepared — **not yet performed** — human review.

## 1. Automatic semantic re-verification (completed)

Tool: `src/evaluation/semantic_validator.py` (v p1-0.2.0), built on the project's
closed-grammar normalizers (`validation.semantic_equivalence`) and the exact graph
oracle (`data_generation.oracle`). No LLM judgment is used anywhere. Full output:
`reports/p1_semantic_validation.json`.

Scope: every prompt of the frozen P1 manifest (16,618; SHA-256
`34f76f72…1242e`). Checks per prompt: entity-inventory size/uniqueness from the
prompt's own inventory line; premise count = n−1; for Turkish, every premise parsed
by the closed grammar into a directed edge; edge set equal to the latent adjacent
chain (native-name arms) or forming a strict total order over the surface inventory
(relabeled name-control arms, certified by the oracle); parsed query entities equal
to the recorded display query pair; oracle answer equal to the stored gold label.
Per matched group: single gold label and a single normalized semantic digest across
all presentation cells.

Results:

| Status | Count | Meaning |
|---|---|---|
| pass (Turkish, fully normalized) | 11,684 | parses under the closed grammar; edges, query, oracle answer and gold all agree |
| en_structural_only (English) | 4,934 | the normalization grammar is Turkish-only by design; structural and group-level checks pass |
| semantic failure | 0 | — |
| not normalizable | 0 | every sampled Turkish surface form, including all P1–P5 and the transliterated/symbolic naming arms, is inside the implemented grammar |
| group consistency failures | 0 of 2,842 matched groups | — |

Interpretation limit (unchanged from the manuscript): these checks certify
agreement with the implemented grammar and the latent logic. They cannot certify
grammatical naturalness, discourse acceptability, or unrestricted Turkish
equivalence — that is precisely the human review's job, and P3 remains the condition
most in need of it.

## 2. Human review package (prepared; judgments NOT simulated)

- `validation/turkish_linguistic_review.csv` — 360 Turkish items: 4 per
  (relation × condition) cell across all 9 relations and 7 conditions (63 cells),
  plus 12 extra P3 items per relation (**144 P3 items total**), spread over entity
  counts and gold labels, each paired with its canonical counterpart
  (seeded sampler: `src/evaluation/build_review_sample.py`, seed 20261028).
- `validation/REVIEWER_INSTRUCTIONS.md` — independent-review protocol, 1–5
  grammaticality/naturalness scales, equivalence and ambiguity judgments.
- Inclusion/exclusion policy **predeclared before any expanded model outcome is
  examined**: confirmatory-eligible items require grammaticality ≥ 4 from both
  reviewers and unanimous `semantic_equivalence = yes`; all exclusions are reported.
  Clarification (2026-10-29, still before any annotation exists): canonical
  baseline rows carry no transformed sentence, so their `semantic_equivalence`
  is `n/a` by construction; for them eligibility requires grammaticality ≥ 4
  and `ambiguity = no` from both reviewers.
- Inter-annotator agreement (percent agreement and chance-corrected) is computed
  only after both reviewers' files exist. No annotation has been entered; all
  annotation columns are empty in the shipped CSV.

Relation to the framework's gate: this 360-item package is sized against the
framework's review contract (≥100 distinct Turkish prompts, ≥20 per selected
relation–presentation cell for the chosen cells, coverage of requested sizes), so a
completed review can be converted into the dataset-bound receipt the >2,000-call
gate requires.

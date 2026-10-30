# Turkish linguistic review — reviewer instructions (P1)

You are one of (at least) two independent Turkish-proficient reviewers. Work alone;
do not discuss items with the other reviewer before both of you have submitted your
annotations. Disagreements are resolved afterwards in a recorded adjudication pass.

## What you are judging

Each row of `turkish_linguistic_review.csv` contains a full Turkish prompt
(`original_sentence`) and, for transformed conditions, its answer-preserving variant
(`transformed_sentence`). The `intended_meaning` column states the intended logic
and the correct answer. Conditions: P0/canonical (baseline), equivalent (general
matched paraphrase), P1 (premise/name-list reorder), P2 (converse construction),
P3 (alternative constituent order, e.g. postverbal subject), P4 (case/possessive
alternatives), P5 (combined transformations). P3 is deliberately oversampled.

Judge the TRANSFORMED sentence when present, otherwise the original. Rows with
`presentation_condition = canonical` are the baseline: their `transformed_sentence`
is empty on purpose — judge the original for grammaticality, naturalness, and
ambiguity, and write `n/a` in `semantic_equivalence` (there is nothing to compare).
Fill in:

| Column | Scale |
|---|---|
| grammaticality | 1–5 (1 = ungrammatical, 5 = fully grammatical) |
| naturalness | 1–5 (1 = no native speaker would produce this, 5 = fully natural) |
| semantic_equivalence | `yes` / `no` / `uncertain` — checked by aligning statements pairwise between original and transformed: does each transformed premise (and the question) assert the same relation between the same entities as its counterpart? No answer computation is involved. |
| ambiguity | `yes` / `no` — judged clause by clause, never globally: for any single premise (or the question), could a cooperative reader reverse or reassign WHO fills WHICH role in that one relation (e.g., unclear case marking leaves open who is taller/in front)? You never need to derive the full ordering or the answer — solving the puzzle is the model's task, not yours. If `yes`, name the clause and the alternative reading in comments. |
| reviewer_comments | free text (Turkish or English) |

## Ground rules

- Judge language, not logic puzzles: you are not asked to solve the ordering task.
- A sentence can be grammatical yet unnatural; score the two separately.
- Low frequency of a construction is not ungrammaticality; note it under naturalness.
- Do not consult LLMs for judgments; this review exists because automatic checks
  cannot certify naturalness.
- Observations that do not fit any rating column — e.g., discomfort with the use
  of (synthetic) person names in ranking tasks, or a family-wide vocabulary
  choice — go into `reviewer_comments`, marked "systematic" if they apply to a
  whole relation family or condition rather than this item. They are reported
  with the review but do not by themselves affect any score or item eligibility;
  score columns record only item-level linguistic judgments.
- Save your file as `turkish_linguistic_review.<your-initials>.csv` without changing
  row order or non-annotation columns.

Inter-annotator agreement and the inclusion/exclusion policy are computed only after
both files exist; the policy (drafted before any expanded model outcome is examined)
is: items where both reviewers rate grammaticality ≥ 4 and semantic_equivalence =
`yes` are confirmatory-eligible; items failing either criterion for either reviewer
are excluded from confirmatory analysis and reported separately. For canonical
(baseline) rows, where semantic_equivalence is `n/a` by construction, eligibility
requires grammaticality ≥ 4 from both reviewers and ambiguity = `no` from both.
(Clarification recorded 2026-10-29, before any reviewer annotations exist; it
specializes the predeclared rule to the baseline rows rather than changing it.)

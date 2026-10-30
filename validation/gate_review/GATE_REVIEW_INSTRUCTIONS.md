# Gate-grade Turkish linguistic review — reviewer instructions

Two packages, each a `review_sample.csv` of real frozen-grid Turkish prompts.
Completing one produces the execution-gate receipt that authorizes the
corresponding >2,000-call run. These are **not** the same instrument as
`validation/turkish_linguistic_review.csv` (the 360-item manuscript package
with 1–5 scales); the same two reviewers may complete both.

| Package | Rows | Unlocks |
|---|---|---|
| `frozen_test/` | 105 | the confirmatory frozen-test run (gpt-4o, test split) |
| `training_eval/` | 485 | the RQ5 base-vs-adapter evaluation (10,224 calls/arm) |

## Who may review

Two **independent, Turkish-proficient** reviewers, identified by full name.
Independent means: each fills their own four columns without seeing the
other's judgments; the shared `resolution` column is set together afterwards.
Judgments must be real human decisions — the receipt tooling refuses blanks
but cannot detect fabrication; do not let anyone (including an AI) fill these.

## What to judge, per row

Read the `prompt` column (the full Turkish prompt as the model receives it)
with its metadata (`relation_id`, `condition`, `name_control`, `n`). Each
reviewer fills their four columns with exactly `pass` or `fail`:

- `reviewerN_grammar` — every Turkish sentence is grammatical (case marking,
  suffix phonology, agreement).
- `reviewerN_naturalness` — phrasing a native speaker could produce; awkward
  but grammatical text that no competent speaker would misread still passes;
  text you had to re-read to parse fails.
- `reviewerN_equivalence` — judged clause by clause, never by solving: each
  premise and the question admit exactly one relational reading (who fills
  which role), and for transformed `condition`s each statement asserts the
  same relation as its canonical phrasing would. You never derive the full
  ordering or the answer — logical determinacy is machine-checked by
  independent oracles; you certify only that the language points one way.
- `reviewerN_morphology` — suffix choices on entity names are correct for the
  name's phonology (vowel harmony, consonant alternation), including on
  `symbolic_shared` / transliterated names.

`resolution`: `approved` or `rejected`, set jointly after both reviewers
finish. Large-`n` prompts (256–1024) list many facts of identical shape; read
the instruction, the question, and a representative sample of fact lines plus
any line that looks anomalous — then judge.

Be strict: a `fail` anywhere simply means that run stays blocked until the
stimulus issue is understood. **Never** change `prompt`, `prompt_sha256`,
`query_id`, `design_id`, or the metadata columns; the gate re-materializes
every prompt from the frozen grid and rejects any edited row. Edit the CSV
with a plain-text/CSV editor that preserves UTF-8 and quoting (not Excel's
default save-as).

## When both packages' columns are complete

```bash
PYTHONPATH=src python3 -m evaluation.finalize_review_receipt \
    --package validation/gate_review/frozen_test \
    --reviewer1 "Full Name" --reviewer2 "Full Name"
```

(and the same with `--package validation/gate_review/training_eval`). The
script verifies completeness, writes `receipt.json` bound to the frozen
dataset manifest and the exact CSV bytes, and self-checks it against the real
execution gate. It sets `status: approved` only if every decision passed; it
never fills a judgment. `receipt.pending.json` is a template for reference —
do not hand-edit a receipt to "approved".

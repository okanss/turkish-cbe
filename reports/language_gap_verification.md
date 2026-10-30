# Language-gap verification — Qwen2.5-14B (minor revision, 2026-10-30)

**Question.** The manuscript reports 14B English accuracy 76.4%, Turkish
accuracy 66.1%, and a paired language difference of −7.5 pp. The marginal
subtraction gives −10.3 pp. Which is right?

**Decision: Case A — both values are correct; they are different estimands.**
No number was changed. An explanatory passage was added to §7.7 and the
Table 10 evidence cell now names both populations.

## 1–3. Originally reported values

- English marginal accuracy: 76.4%
- Turkish marginal accuracy: 66.1%
- Paired contrast (TR − EN): −0.075, 95% cluster-bootstrap CI [−0.126, −0.020]

## 4–6. Estimand definitions, populations, denominators

| Quantity | Population | Denominator / weighting |
|---|---|---|
| Marginal accuracies | ALL planned prompts of that language in the sweep manifest: EN = canonical + equivalent × 3 name controls = **4,934** prompts; TR = 7 conditions × 3 name controls = **11,684** prompts | All-planned accuracy (invalid counted wrong); each prompt weighted equally, so the TR pool carries the transformed presentations (P1–P5, equivalent) and all name controls |
| Paired gap (F1) | Query-matched **canonical × native_shared** pairs, keyed by (query_instance_id, relation_id): **2,242** pairs, 2,242 jointly valid at 14B | Mean of per-pair (TR correct − EN correct) on jointly valid pairs; equal weight per matched pair |

The two populations differ by construction: the paired contrast deliberately
holds presentation and naming fixed (canonical, native names) so that only
language varies; the marginals summarize each language's full planned pool.

## 7. Authoritative source files

- `results/p1/qwen2-5-14b.jsonl` (raw inference records)
- `data/manifests/p1_expanded_manifest.jsonl` (frozen manifest)
- `results/p1/analysis/p1_accuracy_by_axis.csv`, `p1_paired_tests.csv`
- Pair construction identical to `analysis.p1_statistics._pairs`

## 8. Verification calculations (recomputed from raw records this session)

14B:
- EN marginal: 3,770/4,934 = **0.7641** ✓ matches manuscript 76.4%
- TR marginal: 7,721/11,684 = **0.6608** ✓ matches manuscript 66.1%
- Marginal subtraction: −0.1033 (not a reported estimand)
- Matched canonical/native pairs: 2,242; EN on pairs **0.7823**, TR on pairs
  **0.7074**; paired difference **−0.0749** ✓ matches reported −0.075

7B (same-type check, step 2.4):
- Marginals 0.7144/0.5336 (subtraction −0.1808, not reported)
- Matched pairs: 2,203 jointly valid; EN 0.7245, TR 0.5483; paired difference
  **−0.1761** ✓ matches reported −0.176

0.5B/3B: near-constant responders; same estimand structure applies; no
marginal-vs-paired juxtaposition is made in the text for them.

## 9. Final decision and justification

Both reported values are verified exact under their own definitions; the
apparent discrepancy is a composition effect (the Turkish marginal pool
contains the transformed presentations that reduce Turkish accuracy, and the
pools differ in size and mix). Manuscript changes:

- §7.7: added the estimand-distinction passage, including the matched-pair
  accuracies (78.2% vs. 70.7%) and the explicit statement that the paired
  estimate is not the difference of the displayed marginals.
- Table 10 (Language row): evidence cell now reports the matched-pair
  accuracies first and labels the pooled marginals as all-condition values.

Unresolved items: none.

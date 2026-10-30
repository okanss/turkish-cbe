# Research audit — Turkish-CBE, Prompt 11

Date: 2026-10-25. Distribution: 0.11.0. Decision: **not ready for empirical publication**.

This audit inspected the frozen dataset, renderers, inference/reasoning/training execution controls, statistics and publication evidence. Independent read-only reviews preceded the fixes. No live model request, weight download, optimizer step or qualified human linguistic approval was performed. Synthetic counterexamples below test software contracts and are not research findings.

## Severity and disposition

### Critical

No concrete Critical defect was found in the audited frozen instance separation, oracle labels or CBE arithmetic. This is a bounded audit conclusion, not proof of absence of leakage, universal semantic equivalence or population competence. The four Major software defects below could have affected future experiments; no genuine experiment has used them in this project.

### Major — corrected

| ID | Before correction | Verified correction | Regression evidence |
|---|---|---|---|
| M01 | Training evaluation could reach model loading for 511,200 planned calls without linguistic review if the GPU and call gates were satisfied. A pilot label could also bypass the full-run review gate. | Inference, reasoning and training evaluation require dataset-bound, scope-checked two-reviewer evidence above 2,000 calls, before loading a provider/model or creating a live run. Full-run pilot approval uses the same validator. | `test_audit_execution_gates.py`: pre-GPU/provider refusal, large-pilot checks, malformed/pending/tampered/fixture receipt rejection and coverage checks. Baseline sentinel: `reports/audit/baseline/execution_before.json`. |
| M02 | Greedy training evaluation could inherit model-repository generation settings (beams, penalties or forced tokens), confounding base/adapter comparison. | A fresh neutral greedy GenerationConfig is supplied with use_model_defaults=False. Full sampling configuration, seed and hash are recorded and checked in each response/evaluation. | Neutral-generation regression in `test_audit_execution_gates.py`; no weights loaded. |
| M03 | Training reports supplied separate arm profiles but no matched base-versus-trained intervention effect or confidence interval. | `training.comparison.training_contrasts` adds trained-minus-base accuracy, strict R and coverage on all planned coordinates; R/invariance/kappa sensitivity uses common-four-valid support. Latent-cluster bootstrap (2,000 default), support counts, exact/appropriate McNemar and declared-arm Holm are reported. | `test_audit_training_comparisons.py`: invalids, unequal/missing coordinates, query/gold changes, cluster dependence, missing arms, undefined kappa and deterministic resampling. Before-fix synthetic counterexample is retained. |
| M04 | Native training reports could accept a non-base evaluation without the exact completed adapter training identity; publication collection was stricter. | Native reporting now requires an exact full adapter identity matching a supplied completed training run. | Missing run, different plan/adapter identity and matching identity regressions in `test_audit_training_comparisons.py`. |

The dependent-profile R plot also now prefers cluster-bootstrap intervals; Wilson remains descriptive when several rows share a latent instance. Frozen dataset generation/rendering packages and all existing data, raw-result directories and historical output bytes were preserved.

### Major — open prerequisites

1. **Human linguistic review is pending.** The 252-row report contains no supplied human ratings. Automated normal-form checks and an AI reading of 216 short cases cannot establish Turkish naturalness. Two independent Turkish-proficient reviewers must inspect the actual sampled prompts and resolve all four rating dimensions. No approved receipt was fabricated.
2. **No genuine model evidence exists.** Research questions RQ1–RQ5 remain pending; accuracy, language effects, reasoning gains and transfer cannot be established. Model selection and the real small pilot remain the user's deferred decision. No full evaluation was launched.
3. **Optional Torch/CUDA execution remains unverified.** Apple M1/16 GiB is not the required CUDA training environment. Seven conditional Torch numerical/gradient tests were not registered or executed here. Pure-Python reference checks do not validate GPU training, optimizer behavior or an actual adapter. There is no complete transitive CUDA dependency lock or measured training budget.

### Minor and interpretation limits

- Endpoint questions admit a degree/signature shortcut: the audit solved all 3,000 frozen endpoint cases without traversing the chain. Entity count is input load; latent distance is graph distance; neither is demonstrated model inference depth. Endpoint, internal and distance-controlled arms must remain separate.
- The strict total-order family is structurally isomorphic under renaming. Exact instance separation is established, not independence of abstract reasoning algorithms or contamination of external model training corpora.
- Morphological P4/P5 conditions change multiple linguistic features. Shared renderer/parser morphology code can reproduce its own mistakes. P3 postverbal subjects and spatial P4 naturalness particularly require human judgment.
- Prompt length, tokenizer, name normalization, response language and English scaffold vocabulary remain potential confounders. English higher/lower context overlap weakens claims of a completely unseen lexical family. Lexical associations are correlational.
- Holm correction is within the declared model/design/metric contrast family (five predeclared trained arms for training contrasts), including missing hypothesis slots; it is not study-wide familywise control across every n, relation or distance. Results are exploratory until confirmatory families and tests are fixed before results are observed.
- Conditional R/invariance and marginal accuracy have different denominators. Common-four-valid training contrasts are labelled sensitivity analyses; their support and all-planned strict R/coverage must accompany interpretation. Undefined kappa and unsupported tests remain null.
- Checks establish local, hash-bound provenance, not external provider attestation or reviewer identity authentication. Pointwise intervals are not simultaneous intervals.

## All 18 requested checks

| # | Check | Finding and scope |
|---|---|---|
| 1 | Dataset leakage | Zero cross-split exact source-order, alias-order or visible-prompt-hash overlaps. Abstract-order isomorphism is not excluded. |
| 2 | Latent train/test split | All 3,000 source orders and 23,200 query seeds replayed; split membership is by latent source. |
| 3 | Turkish equivalence | 729,000 component normal forms agree; stratified actual surface replay passes. Human naturalness pending. |
| 4 | Morphology | 12 independent suffix fixtures and renderer regressions pass; not native-speaker sign-off. |
| 5 | Direction/gold balance | 251 strata checked; frozen-test four cells are 50 each. Other strata have 12/13, maximum difference one. |
| 6 | Query arms | Endpoint/internal/distance coordinates and statistics are separated; endpoint shortcut explicitly demonstrated. |
| 7 | Matched models/languages | Frozen design and EN/TR plan hashes match. Genuine multi-model output matching remains untested because no runs exist. |
| 8 | Invalid answers | Strict parsers, wrong-in-marginals convention and planned response denominators tested. |
| 9 | R+D+U=1 | Independent exhaustive check of 1,458 Boolean/invalid/gold cases on joint-valid support. |
| 10 | Estimand support | Marginal/all-planned and joint-valid metrics labelled; common-four training sensitivity and coverage added. |
| 11 | Truncation, cost, completion | Context failures/invalids remain in denominators; sampling and usage contracts tested. Unknown costs remain null; genuine backend measurement pending. |
| 12 | Paired statistics | 960 exact McNemar tables match integer-binomial oracle; ordinary McNemar rejected for dependent clusters. 2,000-replicate cluster bootstrap tested. |
| 13 | Multiplicity | Holm retains missing slots and declares scope. No study-wide FWER claim. |
| 14 | Linguistic confounding | Controls and limitations documented; causal linguistic effect cannot be claimed. |
| 15 | Reproducibility | Full seed replay, current source hashes, dual-Python suite and installed-wheel checks; GPU execution excluded. |
| 16 | Call lineage | Request/task/attempt/plan/raw-normalized lineage and exact adapters tested; no actual calls to trace. |
| 17 | Paper numerical provenance | All 44 published output hashes/sizes checked; seven tables reconstructed from evidence. Performance remains NA. |
| 18 | Inherited/new methods | CBE latent generator/equivalent presentation/oracle metrics separated from Turkish, internal/distance, lexical and intervention extensions. |

## Verified coverage and evidence

The bank contains 3,000 source orders (training450, development450, pilot300, test1,800), 23,200 source queries, 5,637,600 measured prompt records and 7,819,200 comparison cells. These are not independent problem counts. All 18,000 aliased-order views were checked, and every query has 243 designs. The stratified actual replay covers 33 split×n groups ×243 designs = 8,019 components, 1,543,293 clause instances and 24,057 prompt hashes. Replay digest: `aae42adbcb6ffa630a70a2a924b59530efa2ad31b2a3a8fbb53532a5ca228c15`.

All 252 linguistic CSV rows reparse and preserve their intended normal form; they cover 36 contexts at n4/8. The initial AI screen read 216 examples across nine relations, three name types and P0/P2/P3/P4/P5/P6, without finding a definite grammar error. This is explicitly not human approval and is not repeated by the automatic CLI.

Machine-readable evidence is in `reports/audit/current/`: `audit_manifest.json` binds 87 packaged Python sources and 51 input files, `data_checks.json` records detailed strata and replay, `statistics_checks.json` records arithmetic probes, and `checklist.csv` records all 18 scopes. The read-only CLI refuses existing output directories and protected destinations. Before-fix evidence is in `reports/audit/baseline/`.

## Linguistic review execution contract

A review JSON must specify schema_version1, artifact_type=turkish_linguistic_review, status=approved, two distinct identified reviewers, reviewer_qualification=two_independent_turkish_proficient_reviewers, the exact frozen dataset manifest SHA-256, and samples={path,sha256,rows}. Do not set approved until the human review is actually complete.

The CSV must include query_id, design_id, full prompt text, prompt_sha256, relation_id, condition, name_control, query_type, n, reviewer1/reviewer2 grammar/naturalness/equivalence/morphology ratings and resolution. Every rating must pass and resolution must be approved. At least 100 distinct actual Turkish grid prompts are required, with at least 20 per required relation×presentation×name template cell and coverage of each requested n and query type. This operational sample rule is not exhaustive review of every n×query×template cross-cell. Every text is re-materialized and matched, not merely trusted from metadata. Fixture-only receipts cannot authorize live runs.

The 2,000-call threshold applies to total planned generation calls, including large pilots and training evaluation. Full evaluation additionally requires a genuine approved pilot. Raising max-calls or naming a run pilot does not waive review. Smaller exploratory pilots still do not establish confirmatory eligibility.

## Reproduction and historical outputs

```sh
PYTHONPATH=src:reports python3 -m unittest discover -s tests -v
PYTHONPATH=src:reports python3 -m research_audit --project-root . --output reports/audit/replay_001
```

Install analysis/inference/lexical extras and pinned tokenizer assets as described in REPRODUCIBILITY.md; the audit makes no network calls. Use a new output name for each audit. Fresh current-source dry-run plans are stored separately under `reports/audit/plans/`; they perform no model calls and do not enable disabled model entries.

The Prompt10 `paper/` and `reports/current/`, and Prompt5/8/9 plans remain immutable historical0.10 artifacts. Their numeric construction evidence and 44 output hashes still verify; their old source bindings do not describe the modified0.11 execution code. Old plans must be regenerated before execution. `source_migration.json` and `migration_from_0_10.patch` preserve the exact before/after code inventory and reversible source delta, allowing historical fingerprints to be recovered in a separate checkout. Do not weaken source validation or relabel old plans as current.

The source article [Jiang and Zhai, arXiv:2610.22894v1](https://arxiv.org/html/2610.22894v1) explicitly discusses the endpoint shortcut in §4.1. Its published numbers and training recipe are not claimed to have been replicated here. The CPU audit and synthetic unit fixtures establish implementation properties, not empirical competence or training effectiveness.

## Completion and readiness

- Full unit suite: **1,027/1,027 passed** on Python3.12.14 (139.846s) and Python3.13.1 (159.984s); no registered skips. Seven optional Torch tests were not registered/executed and are not counted as passes.
- Thirty new regression cases cover review gates, neutral decoding, intervention contrasts, adapter lineage and audit/endpoint contracts. A first-pass audit test fixture used a wrong generator keyword; it was corrected before the successful full runs.
- Wheel0.11.0 built and installed offline: **87/87 packaged Python files match**, all **12 CLI help commands work**, seven fresh plans are saved, inference/reasoning plans reload, and all six training arms stay in dry-run, and installed audit statistics/checklist match the source audit. No network attempt, model request or optimizer step occurred.
- Source migration reverse-application in a temporary copy restored all previous Python files and pyproject exactly (81 old packaged sources); frozen generator/rendering source bytes are unchanged.
- Complete logs and hashes: `reports/audit/test_verification.json`, `tests_python312.log`, `tests_python313.log`, `wheel_verification.json` and `source_migration.json`.

Software audit and necessary fixes are complete when the verification record above passes. External Major prerequisites remain open. The project is ready to collect genuine reviewed pilot evidence after model/access selection; it is not ready to submit an empirical results paper. No next-prompt work was started. All new/modified file paths and decisions are listed in PROGRESS.md.

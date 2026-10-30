# Scientific reporting (0.10.0)

The English draft is **results pending**, not a completed empirical study.
`../paper/` contains eight article sections, an assembled manuscript, seven
CSV/LaTeX/Markdown table bundles, architecture/pipeline diagrams, and explicit
placeholders for missing outcome figures. `current/` contains the input/output
hash manifest, verified evidence inventory and five-RQ traceability ledger.

## Run a new report

Use the existing inference/tokenizer/analysis environments and locks. There
are no new third-party dependencies. From the project root:

```sh
PYTHONPATH=src:reports python3 -m publication --paper-dir paper_next --report-dir reports/next
# Installed equivalent:
turkish-cbe-paper --paper-dir paper_next --report-dir reports/next
```

Both destinations must be new. Existing raw responses, normalized files,
frozen data, source, plans and previous papers are never overwritten. The
reporter performs no model calls, model downloads or optimization. It does
perform read-only hashing and native statistical analysis of selected genuine
runs. Expensive full evaluations remain a separate authorized workflow.

By default the reporter discovers native inference/reasoning journals and
completed training/evaluation records. Select specific runs when cohorts differ:

```sh
PYTHONPATH=src:reports python3 -m publication --paper-dir paper_selected --report-dir reports/selected --inference-run RUN_ID --reasoning-run REASONING_RUN_ID
```

Training inputs use `--training-run results/training/runs/RUN` and
`--training-evaluation results/training/evaluations/EVAL`. A non-base adapter
evaluation requires its matching verified completed training record. Selections
are never arbitrary CSV files or user-supplied prediction rows. The library's
`collect_evidence` accepts explicit empty lists to select none; CLI defaults
auto-discover unmentioned source kinds. Replications or incompatible cohorts
require separate reporting calls, not uncontrolled pooling.

## Evidence and statistics

Dataset counts come from the frozen SQLite bank, coverage and manifest, with
source/file SHA checks. Latent orders, queries, surfaces and comparison cells
are different units. Model outcomes come only from native journals, whose raw,
normalized, checkpoint, model, prompt, split and gold identities are checked by
the established readers. Native exceptions stop publication rather than silently
excluding an invalid run. Runner provenance is not independent server attestation.

CBE, reasoning and training tables retain their native model/revision, language,
relation, name-control, n, query type/distance, presentation and intervention
strata. The reporter does not pool, re-estimate, impute or re-adjust p-values.
The configured matched-instance bootstrap is 2000 repeats, seed 20261025 by
default, 95% confidence, latent-instance clusters and n strata. Native Holm,
McNemar/exact McNemar and Wilson limitations are retained. Invalids are wrong
in marginal accuracy; R/D/U and their identities use common valid pairs.
Coverage and strict R remain alongside conditional measures. Undefined kappa,
intervals and tests stay unavailable. Lexical correlations/regressions remain
exploratory, with no causal or model-parameter claims.

Table 1 is construction statistics. Tables 2–7 contain explicit `not_run`
rows while genuine outcomes are absent; estimates, model sample sizes, CIs and
p-values are null/NA, not fabricated zeroes. Table 5 distinguishes verified
lexical preprocessing features from missing performance associations. Lossless
CSV exports retain all native metadata and full precision. Compact Markdown/
LaTeX views use row identifiers to reference that record rather than dropping
conditions. No hypothesis is automatically endorsed from a p-value or a table.

RQ1→Table 3; RQ2→Table 4; RQ3→Table 6; RQ4→Table 5; RQ5→Table 7.
Each pending RQ states that neither support nor refutation is established.
Architecture and generation-pipeline PDF/SVG/PNG files describe methods.
Performance/difference/RDU/vocabulary-bias figures are not drawn without genuine
corresponding observations. Synthetic fixture plots exist only in temporary
test directories, explicitly labelled as fixtures.

## Preservation and readiness

The reporting package is `reports/publication/`, discovered separately from
`src/`. This keeps all prior 75 source files and Prompt 9's full-src plan
fingerprints intact. Native inference, analysis and training code is reused.
The publication manifest hashes both code roots, configuration/environment,
verified inputs, tables, text and figures. A new report is required after any
input or source change; input changes during rendering abort publication.

Human linguistic review, actual model selection/inference, and GPU validation
remain pending. Seven optional Torch tests and CUDA training were not run.
P4/P5 combine transformations; n and endpoint distance are coupled; the corpus
is a limited frequency proxy; model-specific chat token overhead differs from
plaintext BPE; English training/held-out contexts share higher/lower. Optional
training is not a replication of the paper's recipe. These constraints are
stated in the English limitations and prevent a submission-ready claim.


## Prompt11 audit integration

`research_audit` is a separate installed package and CLI. Evidence under `reports/audit/current/` is a CPU/software audit, not model data. The historical `reports/current/` and paper remain the immutable Prompt10 snapshot; source changes in0.11 require a new plan/report destination.

Future verified training evidence includes `training_arm_contrasts`: trained-minus-base effects on exact planned coordinates, explicit common-four-valid sensitivity for conditional CBE measures, all-planned invalid-as-wrong accuracy/strictR/coverage, latent-cluster intervals and declared five-arm Holm families. Exact completed adapter identities are required by both native training reporting and publication collection. Missing arms are unavailable rather than empirical rows. No intervention result was generated here.

# Beyond Translation: Evaluating Logical Reasoning Competence in Large Language Models under Turkish Morphosyntactic Variations

**Scientific reporting draft.** The availability statement and table statuses distinguish verified empirical observations from pending model outcomes. This draft is not a claim of completed model evaluation or publication readiness.

# Introduction

Logical ordering tasks provide a controlled setting in which to distinguish success on one
presentation from reliable success across meaning-preserving presentations. A model can answer
one surface realization correctly and fail on another realization of the same latent problem.
Competence-Based Evaluation (CBE) therefore links a parameterized problem family, an exact
oracle, and equivalent presentations to a requirement of correct answers across presentations
([Jiang and Zhai, 2026](https://arxiv.org/html/2610.22894v1)). A finite benchmark can provide
evidence about this behavioral requirement; it cannot establish competence over an entire
unbounded problem family.

Turkish-CBE extends this experimental framing to matched Turkish and English ordering problems.
Turkish case marking, genitive/possessive constructions, vowel harmony, and word-order choices
motivate controlled surface interventions. Our renderer implements a closed set of constructions,
not a complete model of Turkish grammar; the [Turkish Universal Dependencies documentation](https://universaldependencies.org/tr/)
provides linguistic context. Matching the latent order, query, gold label, and presentation plan
reduces several translation-related differences, while name controls, token lengths, and explicit
intervention labels expose remaining confounders.

The proposed contribution is a reproducible framework connecting latent ordering, morphosyntactic
presentations, exact semantic checks, and paired reliability measures. It is not a claim to be the
first Turkish linguistic benchmark. The study asks five empirical questions about language,
presentation, complexity/distance, lexical features, and reasoning/training interventions. Their
answers require model outcomes and human-reviewed stimuli.

No verified model inference or training outcomes are available. This document reports the implemented protocol and verified dataset/preprocessing inventory; model-performance cells are pending. Missing estimates, sample sizes for model outcomes, confidence intervals, and p-values are not replaced with zero. Software tests and dry-runs are not research results.

| Research question | Planned target | Result table | Evidence status |
| --- | --- | --- | --- |
| RQ1 | Does performance differ for matched Turkish and English problems? | [Table 3: Turkish-English Comparison](tables/table_3_turkish_english_comparison.md) | Pending model outcomes; the hypothesis is neither supported nor refuted. |
| RQ2 | Do meaning-preserving Turkish presentations affect correctness and consistency? | [Table 4: Morphosyntactic Robustness](tables/table_4_morphosyntactic_robustness.md) | Pending model outcomes; the hypothesis is neither supported nor refuted. |
| RQ3 | How do entity count and inference distance relate to performance? | [Table 6: Complexity and Inference Distance](tables/table_6_complexity_inference_distance.md) | Pending model outcomes; the hypothesis is neither supported nor refuted. |
| RQ4 | Are lexical frequency, tokenization, and relation direction associated with performance? | [Table 5: Lexical Relation Analysis](tables/table_5_lexical_relation_analysis.md) | Pending model outcomes; the hypothesis is neither supported nor refuted. |
| RQ5 | Do reasoning prompting and optional training improve reliable inference? | [Table 7: Reasoning and Fine-Tuning](tables/table_7_reasoning_training.md) | Pending model outcomes; the hypothesis is neither supported nor refuted. |


# Related Work

**Logical reasoning and competence-based evaluation.**
[CLUTRR](https://aclanthology.org/D19-1458/) provides a diagnostic setting for relational
reasoning over natural language (Sinha et al., 2019). Turkish-CBE instead uses strict total
orders and adjacent comparisons with independently checked exact answers. The CBE framework
of [Jiang and Zhai (2026)](https://arxiv.org/html/2610.22894v1) motivates evaluation across
equivalent presentations. Our cross-language matching, Turkish constructions, distance controls,
and lexical analyses are extensions; the current implementation is not a direct replication of
all source protocols or training settings.

**Multilingual reasoning.**
[Shi et al. (2023)](https://openreview.net/forum?id=fR3wGCk-IXp) study multilingual arithmetic
reasoning through MGSM. This motivates a separation of language and reasoning conditions, but
its arithmetic setting does not validate Turkish ordering templates or establish the expected
direction of a Turkish-English effect here.

**Morphological and syntactic robustness.**
[Ismayilzada et al. (2025)](https://aclanthology.org/2025.naacl-long.59/) examine morphological
compositional generalization, including Turkish; [TurBLiMP](https://aclanthology.org/2025.emnlp-main.834/)
(Başar et al., 2025) evaluates Turkish linguistic contrasts. [Başar and Bisazza (2026)](https://aclanthology.org/2026.sigturk-1.9/)
address morphology-aware evaluation of Turkish syntax. These studies establish relevant prior
work and motivate care with morphology, length, and segmentation. Their tasks differ from
gold-preserving ordering, so their findings are not imported as Turkish-CBE results.

**Metamorphic testing and consistency.**
Meaning-preserving input transformations provide a metamorphic relation between expected outputs
([Segura et al., 2016](https://www.isa.us.es/publications/type/article-journal/2016/survey-metamorphic-testing)).
[CheckList](https://aclanthology.org/2020.acl-main.442/) (Ribeiro et al., 2020) illustrates
behavioral testing beyond aggregate benchmark scores. Turkish-CBE applies this logic to exact
ordering answers and distinguishes correct consistency from shared errors: invariance alone is
not evidence of correct reasoning.


# Methodology

## Latent generator and exact answers

Each latent example contains a strict total order over n unique entities. The n−1 adjacent
comparisons specify its chain independently of the surface language. The exact oracle computes
answers from latent positions without a language model; a graph-based check independently
validates the chain. Query targets, relation direction, latent distance d=|i−j|, and gold labels
are retained in every derived presentation. Endpoint, internal, and distance-controlled queries
remain separate analysis strata. Entity count n measures input size; it is not by itself a
measure of the amount of inference performed by a model.

All derivatives of a latent instance stay in the same pilot, development/validation, training,
or frozen-test split. They are matched observations, not independent additional problems.
The dataset table reports source orders, queries, and presentation records at distinct units.

## Equivalent presentations and Turkish rendering

The renderer covers explicitly linear spatial, temporal, comparative, and rank relations.
Forward and inverse terms are mapped back to the same normalized relation. P0 is canonical;
P1 changes surface list order; P2 applies a converse relation with swapped arguments; P3 uses
an approved alternative sentence construction. P4/P5 include multiple changes and are labeled
compound interventions; their results cannot isolate a pure morphological causal effect.
P6 controlled negation is optional and requires a strict-order equivalence proof; it is not
silently pooled with the implemented core conditions.

Turkish suffix generation checks vowel harmony, case and possessive endings, apostrophes on
proper names, selected consonant alternations, and Turkish Unicode characters. Automated
round-trip normalization and gold preservation validate the closed grammar only. Two independent
Turkish-proficient reviewers must evaluate naturalness and meaning, with adjudication of
disagreements, before the stimuli support confirmatory linguistic claims.

## Paired correctness and invariance

Let C0 and C1 indicate correctness of the original and equivalent answers. Marginal original
and equivalent accuracy use every planned matched example; an invalid answer counts as wrong.
The following paired quantities use only the common set on which both answers are parsed:

R=P(C0=1,C1=1), D=P(C0=0,C1=0), U=P(C0≠C1).

MeanAccuracy=R+U/2; RawInvariance=R+D; and R+D+U=1 on this shared-valid denominator.
For the binary task, D captures shared wrong answers. It is not robust correctness. Valid Pair
Coverage reports the shared-valid denominator divided by all planned pairs. Strict Robust
Correctness divides the count correct on both presentations by all planned pairs, retaining
invalid pairs in the denominator. These quantities are always reported with Invalid Response Rate.

For each specified model/condition stratum at a fixed complexity n, q is the pooled YES/EVET
proportion over the two responses of shared-valid pairs. A lone valid answer from a pair with
an invalid counterpart is excluded from q, just as that pair is excluded from R/D/U.
I_expected=q²+(1−q)² and
kappa=(RawInvariance−I_expected)/(1−I_expected). A zero denominator produces an undefined/null
kappa, not zero. The paired Turkish-English accuracy difference retains matched instances and
the same missing-response policy; no score is inferred from a response fragment.

## Statistical analysis

The analysis module supports paired McNemar tests, exact inference where applicable, and Holm
correction within explicitly declared comparison families. Matched-instance bootstrap confidence
intervals use 2,000 resamples, with the latent instance retained as the cluster and relevant
complexity strata retained. Multiple queries and surface forms within an instance are not
resampled as independent observations. Wilson intervals for R are reported as descriptive
binomial intervals when clustering violates their independent-Bernoulli assumption; cluster
bootstrap intervals provide the dependence-aware uncertainty assessment. Empty or undefined
quantities remain null.

Analyses separate model family, language, relation family, n, query type, and query distance.
Language×presentation and language×complexity interactions require identifiable, sufficiently
populated data; unavailable or unsupported fits are labeled accordingly. Lexical frequency and
tokenization associations are correlational, with model, size, relation family, and prompt length
controlled where supported. Neither these associations nor regression coefficients reveal causal
frequency effects or internal model parameter weights.


[Table 1: Dataset Statistics](tables/table_1_dataset_statistics.md) provides the verified inventory.


# Experimental Setup

No verified model inference or training outcomes are available. This document reports the implemented protocol and verified dataset/preprocessing inventory; model-performance cells are pending. Missing estimates, sample sizes for model outcomes, confidence intervals, and p-values are not replaced with zero. Software tests and dry-runs are not research results.

This bundle is not eligible for confirmatory evaluation. Human linguistic review and the protocol's remaining execution gates must be completed before confirmatory claims.

## Models, prompts, and decoding

Models are selected through configuration rather than hard-coded identities. Every executed
condition must retain its exact model and tokenizer revisions, provider/serving system, prompt
and dataset hashes, sampling parameters, context/output budgets, reasoning setting, and
applicable seeds. Availability failures must not trigger an unreported model substitution.
The disabled Qwen2.5-1.5B-Instruct training candidate is not an evaluated model; Turkish ordering
competence for this size has not been measured. Family-level multilingual evidence does not
replace that measurement.

The Turkish direct system instruction is: “Sen mantıksal sıralama sorularını yanıtlayan bir
asistansın. Verilen öncülleri temel al ve yalnızca istenen cevabı üret.” The user provides
premises and a question, followed by “Yalnızca EVET veya HAYIR yaz.” The English control has
an equivalent instruction and YES/NO outputs. Exact system/user messages and their hashes,
rather than this prose description, define an executed condition. A strict parser rejects
extra or contradictory answer forms and separately records response-language errors.

Explicit reasoning prompting, structured reasoning, and provider reasoning-effort settings
are distinct interventions. Reasoning explanations are not scored as evidence of correctness;
the final marked answer is scored. Output budgets, truncation, invalid responses, and token
usage accompany comparisons. Optional SFT/GRPO arms remain unrun unless their execution and
hardware gates pass. An equal optimizer-step control does not establish equal token, GPU-time,
or parameter-update budgets.

## Data, hardware, and reproducibility

The following table is populated from verified dataset metadata rather than model outcomes.
The complete-grid plain-text tokenizer counts exclude chat-template, role, and system-message
overhead; executed provider usage and model-specific chat counts are separate measurements.
Neither prompt count nor token count is an independent reasoning-trial count.

**Table 1. Dataset Statistics.** [CSV](tables/table_1_dataset_statistics.csv), [LaTeX](tables/table_1_dataset_statistics.tex), [complete Markdown table](tables/table_1_dataset_statistics.md).

Stratum IDs resolve to publication_row_id in the CSV, which retains full experiment conditions, model revisions, denominator annotations, and native statistics. The compact view does not aggregate or fit another analysis.

Navigation excerpt: the first 12 of 33 publication-view rows are shown in their original order. No result is selected by magnitude or significance; the linked CSV contains all native rows and columns, and the complete Markdown table contains every compact-view row.

| Stratum | Split | n | Orders | Queries | Prompts | Comparison cells | Status |
| --- | --- | --- | --- | --- | --- | --- | --- |
| t1-r00001 | development | 4 | 50 | 150 | 36450 | 47250 | dataset_statistics |
| t1-r00002 | development | 8 | 50 | 250 | 60750 | 82350 | dataset_statistics |
| t1-r00003 | development | 16 | 50 | 300 | 72900 | 99900 | dataset_statistics |
| t1-r00004 | development | 32 | 50 | 350 | 85050 | 117450 | dataset_statistics |
| t1-r00005 | development | 64 | 50 | 400 | 97200 | 135000 | dataset_statistics |
| t1-r00006 | development | 128 | 50 | 450 | 109350 | 152550 | dataset_statistics |
| t1-r00007 | development | 256 | 50 | 500 | 121500 | 170100 | dataset_statistics |
| t1-r00008 | development | 512 | 50 | 550 | 133650 | 187650 | dataset_statistics |
| t1-r00009 | development | 1024 | 50 | 600 | 145800 | 205200 | dataset_statistics |
| t1-r00010 | pilot | 4 | 50 | 150 | 36450 | 47250 | dataset_statistics |
| t1-r00011 | pilot | 8 | 50 | 250 | 60750 | 82350 | dataset_statistics |
| t1-r00012 | pilot | 16 | 50 | 300 | 72900 | 99900 | dataset_statistics |


The recorded local preparation environment is an Apple M1 machine with 16 GiB unified memory
and no CUDA device. Unified memory is not CUDA VRAM. No CUDA training fit or speed claim is
therefore supported. CPU software tests and dry-runs do not validate autograd behavior, CUDA
kernels, or training convergence; the optional Torch backend tests remain a separate execution
requirement. Reproducibility records preserve configuration, source/data hashes, environment,
and append-only raw outputs. Remote API seeds do not imply bit-identical regenerated answers.


# Results

No verified model inference or training outcomes are available. This document reports the implemented protocol and verified dataset/preprocessing inventory; model-performance cells are pending. Missing estimates, sample sizes for model outcomes, confidence intervals, and p-values are not replaced with zero. Software tests and dry-runs are not research results.

This bundle is not eligible for confirmatory evaluation. Human linguistic review and the protocol's remaining execution gates must be completed before confirmatory claims.

| Research question | Planned target | Result table | Evidence status |
| --- | --- | --- | --- |
| RQ1 | Does performance differ for matched Turkish and English problems? | [Table 3: Turkish-English Comparison](tables/table_3_turkish_english_comparison.md) | Pending model outcomes; the hypothesis is neither supported nor refuted. |
| RQ2 | Do meaning-preserving Turkish presentations affect correctness and consistency? | [Table 4: Morphosyntactic Robustness](tables/table_4_morphosyntactic_robustness.md) | Pending model outcomes; the hypothesis is neither supported nor refuted. |
| RQ3 | How do entity count and inference distance relate to performance? | [Table 6: Complexity and Inference Distance](tables/table_6_complexity_inference_distance.md) | Pending model outcomes; the hypothesis is neither supported nor refuted. |
| RQ4 | Are lexical frequency, tokenization, and relation direction associated with performance? | [Table 5: Lexical Relation Analysis](tables/table_5_lexical_relation_analysis.md) | Pending model outcomes; the hypothesis is neither supported nor refuted. |
| RQ5 | Do reasoning prompting and optional training improve reliable inference? | [Table 7: Reasoning and Fine-Tuning](tables/table_7_reasoning_training.md) | Pending model outcomes; the hypothesis is neither supported nor refuted. |

## Table 2: Overall Model Performance

Pending verified model outcomes. Preprocessing rows, where present, describe features or availability only; they do not estimate accuracy, robustness, invariance, treatment effects, or hypothesis support.

**Table 2. Overall Model Performance.** [CSV](tables/table_2_overall_model_performance.csv), [LaTeX](tables/table_2_overall_model_performance.tex), [complete Markdown table](tables/table_2_overall_model_performance.md).

Stratum IDs resolve to publication_row_id in the CSV, which retains full experiment conditions, model revisions, denominator annotations, and native statistics. The compact view does not aggregate or fit another analysis.

| Stratum | Measure | Estimate | N planned / valid / clusters | 95% CI | Test | p / Holm p | Status |
| --- | --- | --- | --- | --- | --- | --- | --- |
| t2-r00001 | not_available | — | NA / NA / NA | — | — | — | not_run |


## Table 3: Turkish-English Comparison

Pending verified model outcomes. Preprocessing rows, where present, describe features or availability only; they do not estimate accuracy, robustness, invariance, treatment effects, or hypothesis support.

**Table 3. Turkish-English Comparison.** [CSV](tables/table_3_turkish_english_comparison.csv), [LaTeX](tables/table_3_turkish_english_comparison.tex), [complete Markdown table](tables/table_3_turkish_english_comparison.md).

Stratum IDs resolve to publication_row_id in the CSV, which retains full experiment conditions, model revisions, denominator annotations, and native statistics. The compact view does not aggregate or fit another analysis.

| Stratum | Measure | Estimate | N planned / valid / clusters | 95% CI | Test | p / Holm p | Status |
| --- | --- | --- | --- | --- | --- | --- | --- |
| t3-r00001 | not_available | — | NA / NA / NA | — | — | — | not_run |


## Table 4: Morphosyntactic Robustness

Pending verified model outcomes. Preprocessing rows, where present, describe features or availability only; they do not estimate accuracy, robustness, invariance, treatment effects, or hypothesis support.

**Table 4. Morphosyntactic Robustness.** [CSV](tables/table_4_morphosyntactic_robustness.csv), [LaTeX](tables/table_4_morphosyntactic_robustness.tex), [complete Markdown table](tables/table_4_morphosyntactic_robustness.md).

Stratum IDs resolve to publication_row_id in the CSV, which retains full experiment conditions, model revisions, denominator annotations, and native statistics. The compact view does not aggregate or fit another analysis.

| Stratum | Measure | Estimate | N planned / valid / clusters | 95% CI | Test | p / Holm p | Status |
| --- | --- | --- | --- | --- | --- | --- | --- |
| t4-r00001 | not_available | — | NA / NA / NA | — | — | — | not_run |


## Table 5: Lexical Relation Analysis

Pending verified model outcomes. Preprocessing rows, where present, describe features or availability only; they do not estimate accuracy, robustness, invariance, treatment effects, or hypothesis support.

**Table 5. Lexical Relation Analysis.** [CSV](tables/table_5_lexical_relation_analysis.csv), [LaTeX](tables/table_5_lexical_relation_analysis.tex), [complete Markdown table](tables/table_5_lexical_relation_analysis.md).

Stratum IDs resolve to publication_row_id in the CSV, which retains full experiment conditions, model revisions, denominator annotations, and native statistics. The compact view does not aggregate or fit another analysis.

Navigation excerpt: the first 12 of 290 publication-view rows are shown in their original order. No result is selected by magnitude or significance; the linked CSV contains all native rows and columns, and the complete Markdown table contains every compact-view row.

| Stratum | Measure | Estimate | N planned / valid / clusters | 95% CI | Test | p / Holm p | Status |
| --- | --- | --- | --- | --- | --- | --- | --- |
| t5-r00001 | not_available | — | NA / NA / NA | — | — | — | preprocessing_only |
| t5-r00002 | surface_count | 0 | NA / NA / NA | — | — | — | preprocessing_only |
| t5-r00002 | lemma_count | 124 | NA / NA / NA | — | — | — | preprocessing_only |
| t5-r00002 | surface_frequency_per_million | 0.0 | NA / NA / NA | — | — | — | preprocessing_only |
| t5-r00002 | lemma_frequency_per_million | 1187.4664828008888 | NA / NA / NA | — | — | — | preprocessing_only |
| t5-r00003 | surface_count | 0 | NA / NA / NA | — | — | — | preprocessing_only |
| t5-r00003 | lemma_count | 53 | NA / NA / NA | — | — | — | preprocessing_only |
| t5-r00003 | surface_frequency_per_million | 0.0 | NA / NA / NA | — | — | — | preprocessing_only |
| t5-r00003 | lemma_frequency_per_million | 507.5461579713476 | NA / NA / NA | — | — | — | preprocessing_only |
| t5-r00004 | surface_count | 0 | NA / NA / NA | — | — | — | preprocessing_only |
| t5-r00004 | lemma_count | 25 | NA / NA / NA | — | — | — | preprocessing_only |
| t5-r00004 | surface_frequency_per_million | 0.0 | NA / NA / NA | — | — | — | preprocessing_only |


## Table 6: Complexity and Inference Distance

Pending verified model outcomes. Preprocessing rows, where present, describe features or availability only; they do not estimate accuracy, robustness, invariance, treatment effects, or hypothesis support.

**Table 6. Complexity and Inference Distance.** [CSV](tables/table_6_complexity_inference_distance.csv), [LaTeX](tables/table_6_complexity_inference_distance.tex), [complete Markdown table](tables/table_6_complexity_inference_distance.md).

Stratum IDs resolve to publication_row_id in the CSV, which retains full experiment conditions, model revisions, denominator annotations, and native statistics. The compact view does not aggregate or fit another analysis.

| Stratum | Measure | Estimate | N planned / valid / clusters | 95% CI | Test | p / Holm p | Status |
| --- | --- | --- | --- | --- | --- | --- | --- |
| t6-r00001 | not_available | — | NA / NA / NA | — | — | — | not_run |


## Table 7: Reasoning and Fine-Tuning

Pending verified model outcomes. Preprocessing rows, where present, describe features or availability only; they do not estimate accuracy, robustness, invariance, treatment effects, or hypothesis support.

**Table 7. Reasoning and Fine-Tuning.** [CSV](tables/table_7_reasoning_training.csv), [LaTeX](tables/table_7_reasoning_training.tex), [complete Markdown table](tables/table_7_reasoning_training.md).

Stratum IDs resolve to publication_row_id in the CSV, which retains full experiment conditions, model revisions, denominator annotations, and native statistics. The compact view does not aggregate or fit another analysis.

| Stratum | Measure | Estimate | N planned / valid / clusters | 95% CI | Test | p / Holm p | Status |
| --- | --- | --- | --- | --- | --- | --- | --- |
| t7-r00001 | not_available | — | NA / NA / NA | — | — | — | not_run |


## Interpretation of relation direction

Forward/inverse accuracy and paired invariance answer different questions. Equal marginal accuracy can coexist with errors on different instances; high invariance can include shared wrong answers. Direction-specific predicted EVET/YES and HAYIR/NO proportions, gold balance, invalid rates, R, D, U, and kappa must therefore be read together. Missing direction-specific outcomes cannot be supplied by token or frequency measurements.


# Discussion

No verified model inference or training outcomes are available. This document reports the implemented protocol and verified dataset/preprocessing inventory; model-performance cells are pending. Missing estimates, sample sizes for model outcomes, confidence intervals, and p-values are not replaced with zero. Software tests and dry-runs are not research results.

The study's primary distinction is between correct performance and stable correct performance
across equivalent formulations. A future increase in mean accuracy without a corresponding
increase in R would not by itself show improved correct consistency. Conversely, an increase
in raw invariance driven by D would reflect shared errors. Coverage, invalid responses, and
strict robustness are required to detect selective removal of difficult pairs.

Language comparisons may reflect instruction/answer-label differences, naming conventions,
token segmentation, chat formatting, or corpus familiarity alongside linguistic structure.
Morphological presentations can also change syntax and vocabulary; P4/P5 must remain compound
interventions. Increasing n changes input length and distractor load, whereas query distance
changes a distinct property. Endpoint shortcuts offer an alternative explanation for strong
endpoint results and motivate the separately retained internal/distance arms.

Lexical analyses do not identify training-data exposure: an independent corpus is a limited
frequency proxy. A frequency association may be confounded by relation meaning, morphology,
length, and model identity. Reasoning prompts may improve performance partly through a larger
output budget; provider reasoning effort cannot be treated as the same mechanism. A generated
explanation does not establish the model's actual internal computation.

Turkish-CBE extends CBE through matched languages, Turkish constructions, and explicit distance
and lexical controls. These extensions alter the experimental estimand. Unless model identity,
presentation distribution, task family, and training recipe are aligned and verified, numerical
differences from the source paper are not direct replication differences. No original-paper
result or training success is imported into this report.

At the current stage, no empirical Turkish-English gap, perturbation effect, lexical bias, reasoning benefit, or training improvement is established. Future discussion must be rewritten around verified condition-specific estimates and uncertainty, including null and adverse results.


# Limitations

1. **Empirical availability and eligibility.** Planned configurations, software tests, dry-runs,
   and tokenizer/corpus measurements are not model-performance evidence. Human linguistic review
   is pending; automatic semantic normalization validates a closed grammar rather than naturalness
   across unrestricted Turkish. The study is not externally preregistered by virtue of these files.
2. **Scope and dependence.** Synthetic strict total orders, a finite relation inventory, templated
   language, and repeated presentations limit generalization. Queries and presentations share
   latent instances. A large number of prompt records is not a large independent sample of
   reasoning problems, and finite benchmark success does not prove family-wide competence.
3. **Intervention identification.** P4/P5 jointly alter several linguistic features. Language,
   names, answer labels, tokenization, and input/output budgets can be confounded. Pure morphology
   or causal language effects are not identified by the current template design alone.
4. **Frequency and segmentation.** BOUN r2.18 coverage is restricted and its frequencies are not
   measurements of a model's training exposure. Surface, lemma, and multiword counts require
   distinct definitions; unavailable lemma data remain unavailable. Plain-text BPE counts and
   complete provider chat usage are different quantities. Lexical regressions remain correlational.
5. **Training and hardware.** The M1/16-GiB preparation machine has no CUDA device. Optional Torch
   numerical/gradient checks and CUDA training smoke validation were not executed in this local
   environment. The candidate 1.5B model's Turkish ordering ability is unmeasured. Direct dependency
   pins are not a resolved transitive CUDA environment lock. Configurations do not establish
   convergence, feasibility, equal compute, or reproduction of the reference training procedure.
6. **Transfer and lexical overlap.** Training/test latent instances are separated, but English
   context words “higher” and “lower” overlap the intended vocabulary holdout. Clean relation-term
   separation is not clean full-context lexical separation. Transfer claims must retain this caveat.
7. **Inference and uncertainty.** Provider revisions, remote nondeterminism, context overflow,
   truncation, invalid-answer mechanisms, and limited cluster counts affect comparisons. Marginal
   scores retain invalid answers as wrong; valid-pair-only CBE metrics require coverage and strict
   robustness. Wilson intervals alone do not resolve dependence between clustered presentations.


Additional limitations retained from the validated evidence inventory:

- Human linguistic review remains pending; no output is eligible for confirmatory evaluation.
- Runner journals establish recorded provenance, not independent attestation of provider execution.
- Dataset prompts/comparison cells share latent structures and must not be counted as independent samples.
- Corpus frequency associations are correlational, corpus-specific, and do not identify causal effects or model parameter weights.
- Held-out training predicate vocabulary is disjoint, but English presentation contexts share higher/lower; full-context lexical novelty is not established.
- The recorded environment did not run 7 Torch numerical/gradient tests; pure-Python tests do not establish autograd correctness.
- CUDA/model training smoke validation remains unexecuted in the recorded environment; no training compatibility or training success is established.
- No verified model responses are available. Model-performance tables, confidence intervals, significance tests and outcome figures are placeholders only.


# Conclusion

Turkish-CBE operationalizes a controlled link between latent ordering problems, equivalent
Turkish/English presentations, and paired measures of correctness and consistency. Its protocol
and reporting infrastructure make language, presentation, complexity, lexical, and intervention
conditions explicit. This is an implemented research framework, not evidence that any model
has acquired reliable logical reasoning competence.

No model-performance finding is currently available. All five research questions remain unanswered empirically; no hypothesis is reported as supported or refuted. Human stimulus review and approved real-model evaluation are necessary before an empirical conclusion.


# References

Primary sources are linked in context. Machine-readable citations: [references.bib](references.bib).

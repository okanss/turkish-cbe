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

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

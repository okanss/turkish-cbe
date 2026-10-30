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

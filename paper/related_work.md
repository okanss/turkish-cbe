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

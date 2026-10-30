"""English article drafts derived from validated publication evidence.

This module writes reporting artifacts only.  It does not fit models, select a
favourable comparison, infer missing statistics, or treat software tests as
empirical model results.  Input validation and provenance checks belong to the
publication evidence reader; the generated text retains their availability
and eligibility decisions.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
import json
from pathlib import Path


TITLE = (
    "Beyond Translation: Evaluating Logical Reasoning Competence in Large "
    "Language Models under Turkish Morphosyntactic Variations"
)

TABLES = (
    ("table_1_dataset_statistics", "Dataset Statistics"),
    ("table_2_overall_model_performance", "Overall Model Performance"),
    ("table_3_turkish_english_comparison", "Turkish-English Comparison"),
    ("table_4_morphosyntactic_robustness", "Morphosyntactic Robustness"),
    ("table_5_lexical_relation_analysis", "Lexical Relation Analysis"),
    ("table_6_complexity_inference_distance", "Complexity and Inference Distance"),
    ("table_7_reasoning_training", "Reasoning and Fine-Tuning"),
)

RQ_TABLES = (
    ("RQ1", "Does performance differ for matched Turkish and English problems?", 3),
    ("RQ2", "Do meaning-preserving Turkish presentations affect correctness and consistency?", 4),
    ("RQ3", "How do entity count and inference distance relate to performance?", 6),
    ("RQ4", "Are lexical frequency, tokenization, and relation direction associated with performance?", 5),
    ("RQ5", "Do reasoning prompting and optional training improve reliable inference?", 7),
)


def _cell(value: object) -> str:
    """Render generated metadata as text, without interpreting it as Markdown."""
    if value is None:
        return "—"
    if isinstance(value, bool):
        text = "true" if value else "false"
    elif isinstance(value, (dict, list, tuple)):
        text = json.dumps(value, ensure_ascii=False, sort_keys=True, allow_nan=False)
    else:
        text = str(value)
    return (text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
            .replace("|", "&#124;").replace("\r", " ").replace("\n", " "))


def _rows(tables: Mapping[str, object], stem: str) -> list[Mapping[str, object]]:
    value = tables.get(stem, [])
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        raise ValueError(f"{stem} must contain a sequence of table rows")
    result = list(value)
    if any(not isinstance(row, Mapping) for row in result):
        raise ValueError(f"{stem} contains a non-mapping row")
    return result


def _table_link(number: int) -> str:
    stem, label = TABLES[number - 1]
    return f"[Table {number}: {label}](tables/{stem}.md)"


def _table_excerpt(number: int, rows: Sequence[Mapping[str, object]]) -> str:
    """Reuse the readable export view, with traceable native CSV stratum IDs."""
    stem, label = TABLES[number - 1]
    links = (
        f"[CSV](tables/{stem}.csv), [LaTeX](tables/{stem}.tex), "
        f"[complete Markdown table](tables/{stem}.md)"
    )
    if not rows:
        return f"**Table {number}. {label}.** No verified table rows are available. {links}.\n"
    from .tables import publication_view
    compact_rows = publication_view(stem, list(rows))
    columns = list(dict.fromkeys(key for row in compact_rows for key in row))
    selected = compact_rows[:12]
    lines = [f"**Table {number}. {label}.** {links}.", ""]
    lines.extend([
        "Stratum IDs resolve to publication_row_id in the CSV, which retains full experiment "
        "conditions, model revisions, denominator annotations, and native statistics. "
        "The compact view does not aggregate or fit another analysis.", "",
    ])
    if len(compact_rows) > len(selected):
        lines.extend([
            f"Navigation excerpt: the first {len(selected)} of {len(compact_rows)} publication-view rows "
            "are shown in their original order. No result is selected by magnitude or significance; "
            "the linked CSV contains all native rows and columns, and the complete Markdown "
            "table contains every compact-view row.", "",
        ])
    lines.extend([
        "| " + " | ".join(_cell(key) for key in columns) + " |",
        "| " + " | ".join("---" for _ in columns) + " |",
    ])
    lines.extend("| " + " | ".join(_cell(row.get(key)) for key in columns) + " |" for row in selected)
    return "\n".join(lines) + "\n"


def _availability(evidence: Mapping[str, object]) -> str:
    if evidence.get("contains_model_results") is True:
        return (
            "Verified model-result artifacts are available for the conditions explicitly listed "
            "in the tables. Availability does not establish confirmatory eligibility, statistical "
            "significance, or support for any research hypothesis. Unrun conditions remain missing."
        )
    return (
        "No verified model inference or training outcomes are available. This document reports "
        "the implemented protocol and verified dataset/preprocessing inventory; model-performance "
        "cells are pending. Missing estimates, sample sizes for model outcomes, confidence intervals, "
        "and p-values are not replaced with zero. Software tests and dry-runs are not research results."
    )


def _rq_status(tables: Mapping[str, object]) -> str:
    lines = [
        "| Research question | Planned target | Result table | Evidence status |",
        "| --- | --- | --- | --- |",
    ]
    for rq, question, number in RQ_TABLES:
        rows = _rows(tables, TABLES[number - 1][0])
        observed = sum(row.get("status") == "observed" for row in rows)
        status = (
            f"{observed} verified result row(s); interpret the listed conditions and uncertainty only. "
            "No automated support/refutation decision is made."
            if observed else
            "Pending model outcomes; the hypothesis is neither supported nor refuted."
        )
        lines.append(f"| {rq} | {question} | {_table_link(number)} | {status} |")
    return "\n".join(lines)


def _extra_limitations(evidence: Mapping[str, object]) -> str:
    value = evidence.get("limitations", [])
    if isinstance(value, Mapping):
        entries = [f"{key}: {_cell(item)}" for key, item in value.items()]
    elif isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        entries = [_cell(item) for item in value]
    elif value:
        entries = [_cell(value)]
    else:
        return ""
    return "\n\nAdditional limitations retained from the validated evidence inventory:\n\n" + "\n".join(
        f"- {entry}" for entry in entries
    )


def _sections(evidence: Mapping[str, object], tables: Mapping[str, object]) -> dict[str, str]:
    status = _availability(evidence)
    eligible = evidence.get("eligible_for_confirmatory_evaluation") is True
    eligibility = (
        "The evidence inventory marks this bundle as eligible for confirmatory evaluation. "
        "That flag is an eligibility check, not an empirical finding."
        if eligible else
        "This bundle is not eligible for confirmatory evaluation. Human linguistic review and "
        "the protocol's remaining execution gates must be completed before confirmatory claims."
    )
    sections = {}
    sections["introduction"] = """# Introduction

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

""" + status + "\n\n" + _rq_status(tables) + "\n"

    sections["related_work"] = """# Related Work

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
"""

    sections["methodology"] = r"""# Methodology

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
""" + "\n\n" + _table_link(1) + " provides the verified inventory.\n"

    sections["experimental_setup"] = """# Experimental Setup

""" + status + "\n\n" + eligibility + """

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

""" + _table_excerpt(1, _rows(tables, TABLES[0][0])) + """

The recorded local preparation environment is an Apple M1 machine with 16 GiB unified memory
and no CUDA device. Unified memory is not CUDA VRAM. No CUDA training fit or speed claim is
therefore supported. CPU software tests and dry-runs do not validate autograd behavior, CUDA
kernels, or training convergence; the optional Torch backend tests remain a separate execution
requirement. Reproducibility records preserve configuration, source/data hashes, environment,
and append-only raw outputs. Remote API seeds do not imply bit-identical regenerated answers.
"""

    outcome_parts = ["# Results", "", status, "", eligibility, "", _rq_status(tables), ""]
    for number, (_, label) in enumerate(TABLES[1:], 2):
        rows = _rows(tables, TABLES[number - 1][0])
        observed = [row for row in rows if row.get("status") == "observed"]
        outcome_parts.extend([f"## Table {number}: {label}", ""])
        if observed:
            outcome_parts.append(
                "The table contains verified observations for its explicitly recorded experiment "
                "conditions. Read sample size, valid-pair coverage, cluster count, confidence "
                "interval method, and multiplicity-adjusted test together. An unavailable interval "
                "or test is not evidence of significance. No hypothesis decision is generated "
                "from an unadjusted p-value or a descriptive difference alone."
            )
        else:
            outcome_parts.append(
                "Pending verified model outcomes. Preprocessing rows, where present, describe "
                "features or availability only; they do not estimate accuracy, robustness, "
                "invariance, treatment effects, or hypothesis support."
            )
        outcome_parts.extend(["", _table_excerpt(number, rows), ""])
    outcome_parts.extend([
        "## Interpretation of relation direction", "",
        "Forward/inverse accuracy and paired invariance answer different questions. Equal marginal "
        "accuracy can coexist with errors on different instances; high invariance can include "
        "shared wrong answers. Direction-specific predicted EVET/YES and HAYIR/NO proportions, "
        "gold balance, invalid rates, R, D, U, and kappa must therefore be read together. Missing "
        "direction-specific outcomes cannot be supplied by token or frequency measurements.", "",
    ])
    sections["results"] = "\n".join(outcome_parts)

    sections["discussion"] = """# Discussion

""" + status + """

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

""" + (
        "The available rows establish observations only for their recorded conditions. An empirical "
        "claim requires the corresponding matched comparison, adequate coverage, appropriate "
        "uncertainty, and a declared multiplicity family; this report does not automatically infer "
        "a Turkish-English gap or an intervention benefit. Null and adverse results require the "
        "same reporting as favorable estimates.\n"
        if evidence.get("contains_model_results") is True else
        "At the current stage, no empirical Turkish-English gap, perturbation effect, lexical bias, "
        "reasoning benefit, or training improvement is established. Future discussion must be "
        "rewritten around verified condition-specific estimates and uncertainty, including null "
        "and adverse results.\n"
    )

    sections["limitations"] = """# Limitations

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
""" + _extra_limitations(evidence) + "\n"

    sections["conclusion"] = """# Conclusion

Turkish-CBE operationalizes a controlled link between latent ordering problems, equivalent
Turkish/English presentations, and paired measures of correctness and consistency. Its protocol
and reporting infrastructure make language, presentation, complexity, lexical, and intervention
conditions explicit. This is an implemented research framework, not evidence that any model
has acquired reliable logical reasoning competence.

""" + (
        "Verified model outcomes are documented only for the conditions in the exported tables. "
        "Unrun conditions remain pending, and no global hypothesis or competence claim follows "
        "automatically from the available estimates.\n"
        if evidence.get("contains_model_results") is True else
        "No model-performance finding is currently available. All five research questions remain "
        "unanswered empirically; no hypothesis is reported as supported or refuted. Human stimulus "
        "review and approved real-model evaluation are necessary before an empirical conclusion.\n"
    )
    return sections


REFERENCES = r"""% Primary references supporting the protocol and research context, not model results.
@article{jiang2026cbe,
  title={Competence-Based Evaluation of Logical Reasoning in Large Language Models},
  author={Jiang, Jize and Zhai, Chengxiang},
  year={2026}, journal={arXiv preprint arXiv:2610.22894v1},
  url={https://arxiv.org/html/2610.22894v1}
}
@inproceedings{sinha2019clutrr,
  title={CLUTRR: A Diagnostic Benchmark for Inductive Reasoning from Text},
  author={Sinha, Koustuv and others}, year={2019},
  booktitle={EMNLP-IJCNLP}, url={https://aclanthology.org/D19-1458/}
}
@inproceedings{ribeiro2020checklist,
  title={Beyond Accuracy: Behavioral Testing of NLP Models with CheckList},
  author={Ribeiro, Marco Tulio and Wu, Tongshuang and Guestrin, Carlos and Singh, Sameer},
  year={2020}, booktitle={ACL}, url={https://aclanthology.org/2020.acl-main.442/}
}
@inproceedings{shi2023mgsm,
  title={Language Models are Multilingual Chain-of-Thought Reasoners},
  author={Shi, Freda and others}, year={2023}, booktitle={ICLR},
  url={https://openreview.net/forum?id=fR3wGCk-IXp}
}
@inproceedings{ismayilzada2025morphology,
  title={Evaluating Morphological Compositional Generalization in Large Language Models},
  author={Ismayilzada, Mete and others}, year={2025}, booktitle={NAACL},
  doi={10.18653/v1/2025.naacl-long.59}, url={https://aclanthology.org/2025.naacl-long.59/}
}
@inproceedings{basar2025turblimp,
  title={TurBLiMP: A Turkish Benchmark of Linguistic Minimal Pairs},
  author={Başar, Ezgi and Padovani, Francesca and Jumelet, Jaap and Bisazza, Arianna},
  year={2025}, booktitle={EMNLP}, doi={10.18653/v1/2025.emnlp-main.834},
  url={https://aclanthology.org/2025.emnlp-main.834/}
}
@inproceedings{basar2026morphologyaware,
  title={A Morphology-Aware Evaluation of Turkish Syntax in Large Language Models},
  author={Başar, Ezgi and Bisazza, Arianna}, year={2026}, booktitle={SIGTURK},
  doi={10.18653/v1/2026.sigturk-1.9}, url={https://aclanthology.org/2026.sigturk-1.9/}
}
@article{segura2016metamorphic,
  title={A Survey on Metamorphic Testing},
  author={Segura, Sergio and Fraser, Gordon and Sánchez, Ana B. and Ruiz-Cortés, Antonio},
  year={2016}, journal={IEEE Transactions on Software Engineering},
  doi={10.1109/TSE.2016.2532875},
  url={https://www.isa.us.es/publications/type/article-journal/2016/survey-metamorphic-testing}
}
@misc{turkishud,
  title={Turkish Universal Dependencies: Language Documentation},
  author={{Universal Dependencies contributors}},
  url={https://universaldependencies.org/tr/}
}
"""


def write_manuscript(
    evidence: Mapping[str, object],
    tables: Mapping[str, object],
    output_dir: str | Path,
) -> list[Path]:
    """Write the eight requested sections, assembled draft, and bibliography.

    Table figures and CSV/LaTeX exports are produced by the separate publication
    table module.  This function never reads arbitrary experiment files itself.
    Its caller must supply the validated evidence bundle, not simulated results.
    """
    if not isinstance(evidence, Mapping) or not isinstance(tables, Mapping):
        raise TypeError("evidence and tables must be mappings")
    for stem, _ in TABLES:
        _rows(tables, stem)
    if evidence.get("contains_model_results") is not True and any(
        row.get("status") == "observed"
        for stem, _ in TABLES[1:] for row in _rows(tables, stem)
    ):
        raise ValueError("observed result rows contradict the no-model-results evidence flag")
    directory = Path(output_dir)
    if directory.is_symlink():
        raise ValueError("manuscript output directory must not be a symlink")
    directory.mkdir(parents=True, exist_ok=True)
    sections = _sections(evidence, tables)
    expected = [directory / f"{name}.md" for name in sections]
    expected.extend([directory / "manuscript.md", directory / "references.bib"])
    if any(path.exists() or path.is_symlink() for path in expected):
        raise FileExistsError("manuscript output files already exist; choose a new bundle")
    paths = []
    for name, content in sections.items():
        path = directory / f"{name}.md"
        with path.open("x", encoding="utf-8") as handle:
            handle.write(content.rstrip() + "\n")
        paths.append(path)
    draft = (
        f"# {TITLE}\n\n"
        "**Scientific reporting draft.** The availability statement and table statuses distinguish "
        "verified empirical observations from pending model outcomes. This draft is not a claim "
        "of completed model evaluation or publication readiness.\n\n"
        + "\n\n".join(sections.values())
        + "\n\n# References\n\nPrimary sources are linked in context. Machine-readable citations: "
        "[references.bib](references.bib).\n"
    )
    manuscript = directory / "manuscript.md"
    with manuscript.open("x", encoding="utf-8") as handle:
        handle.write(draft.rstrip() + "\n")
    paths.append(manuscript)
    bibliography = directory / "references.bib"
    with bibliography.open("x", encoding="utf-8") as handle:
        handle.write(REFERENCES)
    paths.append(bibliography)
    return paths

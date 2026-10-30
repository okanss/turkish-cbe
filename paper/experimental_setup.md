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

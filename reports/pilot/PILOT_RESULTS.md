# Turkish-CBE pilot results — first genuine model evidence

Date: 2026-10-27. Status: **exploratory pilot** (pilot split, ≤2,000 calls per run,
below the human-linguistic-review threshold). Human review of the Turkish stimuli
remains pending, so none of this is confirmatory evidence; frozen-test evaluation
still requires the two-reviewer receipt and an approved pilot.

## Runs

| Run id | Model | Provider | Calls | Errors | Invalid answers | Cost |
|---|---|---|---|---|---|---|
| `pilot-qwen05b-main-001` | Qwen2.5-0.5B-Instruct @ `7ae55760…` | transformers, CPU/float32, greedy | 1,512/1,512 | 0 | 0 | local compute |
| `pilot-gpt4o-main-003` | gpt-4o-2024-08-06 | OpenAI Responses API, temp 0 | 1,512/1,512 | 1 transient (resolved `resolution-85751ee7…`, retried) | 0 | $2.07 actual (807,761 in / 4,576 out tokens) |

Shared immutable plan cohort: pilot split, n ∈ {4, 8, 16, 32}, 8 latent orders per
size, endpoint + internal + distance-controlled queries, relation `spatial_front`,
`native_shared` names, EN {canonical, equivalent} and TR {canonical, equivalent,
P1–P5}. Identical prompts, strict parser, append-only raw records with full lineage.
Analysis: `results/analysis/<run id>/` (2,000-replicate latent-cluster bootstrap,
exact McNemar, Holm within declared families).

Failed run ids `pilot-gpt4o-main-001/-002` are retained as provenance: `-001` failed
on local TLS certificates before any billable call; `-002` reached the API but every
generation was refused with `organization_spend_limit_exceeded` ($0 billed).

## Headline findings

### 1. Qwen2.5-0.5B: perfectly invariant — and incompetent (the framework's showcase)

The model emitted a constant answer per language on **all 1,512 prompts**: `YES` to
every English prompt, `HAYIR` to every Turkish prompt, at every size, query type,
and presentation. Consequences, exactly as the protocol predicts:

- Marginal accuracy 0.500 everywhere (balanced gold): Δ(TR−EN) = 0, p = 1.
- Raw invariance 1.00, but R = 0.50, D = 0.50, U = 0 — half the "consistency" is
  shared wrong answers.
- κ_CBE **undefined** (`constant_pooled_answer`): the pooled chance correction
  refuses to credit a constant responder. A raw-agreement metric would have scored
  this model 100% "robust".

### 2. gpt-4o: real competence signal, graded failure, and a presentation cliff

All answers were valid and in the requested language. Descriptive pooled accuracy
(formal per-stratum tables under `results/analysis/pilot-gpt4o-main-003/tables/`):

| Cut | Accuracy |
|---|---|
| English (N=336) | 72.0% |
| Turkish (N=1,176) | 57.8% |
| TR canonical / equivalent | 67.9% / 57.7% |
| TR P1 / P2 / **P3** / P4 / P5 | 68.5% / 58.3% / **36.9%** / 61.9% / 53.6% |
| n = 4 → 32 (EN) | 100% → 58.9% |
| n = 4 → 32 (TR) | 77.4% → 51.5% |
| endpoint / internal / distance-controlled | 81.2% / 73.3% / 64.4% (pooled) |
| distance d = 1 → 16 | 70.5% → 36.1% |

Paired CBE profiles (pooled, descriptive; 168 pairs each):

| Comparison | R | D | U | pooled κ_CBE |
|---|---|---|---|---|
| TR canonical → TR equivalent | 0.458 | 0.202 | 0.339 | 0.253 |
| TR canonical → TR P1 | 0.565 | 0.202 | 0.232 | — |
| TR canonical → TR P2 | 0.500 | 0.238 | 0.262 | — |
| TR canonical → **TR P3** | **0.298** | 0.250 | **0.452** | **−0.027** |
| TR canonical → TR P4 | 0.542 | 0.244 | 0.214 | — |
| TR canonical → TR P5 | 0.446 | 0.232 | 0.321 | — |
| EN canonical → TR canonical | 0.565 | 0.173 | 0.262 | — |
| EN equivalent → TR equivalent | 0.476 | 0.173 | 0.351 | — |

Observations (exploratory):

- **Turkish–English gap.** The stratum-level TR−EN difference is negative in 31 of
  70 strata (positive in 10), mean −0.098, but no single 8-pair stratum survives
  Holm. Direction is consistent; confirmatory inference needs the frozen-test run.
- **P3 is a cliff.** The alternative constituent construction (postverbal subject)
  drops Turkish accuracy from 67.9% to 36.9%; cross-presentation agreement falls to
  chance (pooled κ ≈ −0.03), and P3 is the only condition with strata that survive
  Holm even at pilot power (two strata at n=32, Holm p = 0.031, diff −0.75). This is
  also precisely the condition the audit flagged as most needing human naturalness
  review — the linguistic review must precede any confirmatory claim.
- **Complexity and distance.** Accuracy declines monotonically in n and in latent
  distance d, reaching 36.1% at d=16 (below the 50% balanced baseline, i.e. a
  systematic directional error, not noise). Endpoint ≫ internal ≫ distance-controlled
  ordering is consistent with the endpoint-shortcut analysis.
- **Answer bias.** Against balanced gold labels, gpt-4o's predicted affirmative rate
  is 0.476 in English but 0.680 in Turkish — Turkish errors skew toward EVET —
  unlike Qwen's 1.00/0.00 degeneracy.

## Interpretation limits

Pilot split only; one relation family (`spatial_front`); shared latent orders make
prompt records dependent (clusters = 8 latent orders per size stratum); human
linguistic review pending, so Turkish naturalness is unverified — the P3 result in
particular could reflect unnatural stimuli rather than model failure. Training arms
were later executed at smoke scale (see reports/pilot/TRAINING_SMOKE.md), but no
adapter has been evaluated. These results are exploratory and do not answer RQ1–RQ5
confirmatorily.

## Reproduction

```sh
PYTHONPATH=src python -m inference pilot --plan logs/pilot_main_plan.json  --execute --run-id <new id>   # Qwen (local)
PYTHONPATH=src python -m inference pilot --plan logs/pilot_gpt4o_plan.json --execute --run-id <new id>   # gpt-4o (needs OPENAI_API_KEY)
PYTHONPATH=src python -m analysis --root . --run <run id> --output results/analysis/<run id> --figures
```

Environment notes: pinned `tiktoken==0.12.0` (vendored wheel) is required by the
grid layer; on macOS python.org builds set `SSL_CERT_FILE` to certifi's bundle for
the OpenAI provider. Registry change: `qwen25-05b-hf` and `openai-gpt4o-snapshot`
were enabled in `configs/models.yaml` (user model selection, 2026-10-27).

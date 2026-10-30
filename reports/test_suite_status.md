# Test suite status — 2026-10-27

Final run: **1,034 passed, 0 failed** (`PYTHONPATH=src:reports python -m pytest tests/ -q`,
pinned venv with vendored `tiktoken==0.12.0`; `reports/` must be on the path for the
publication/audit test modules).

## Version-compatibility adaptations (last 4 failures → green)

This machine has `transformers 5.3.0`, newer than the project's recorded environment.
Four tests were adapted so the suite passes under both the recorded and current versions:

| Test | Adaptation |
|---|---|
| `test_audit_execution_gates …decoding_penalties` | transformers ≥5 reports an unset neutral `repetition_penalty` as `None` instead of `1`; the assertion now accepts both spellings of "disabled" |
| `test_training_engine …neutral_penalties…` | same for `repetition_penalty`, `typical_p`, `epsilon_cutoff`, `eta_cutoff` |
| `test_lexical_tokenization …unpinned_revision…` | not version drift after all: the fixture copied the live (now enabled) Qwen spec, and `validate_model` rightly rejects an enabled spec with nulled pins; fixtures now normalize `enabled: false` |
| `test_lexical_tokenization …missing_template_pin…` | same fixture normalization |

## Fixed earlier on 2026-10-27 (9 failures → 4)

The pilot runs and the user's model selection legitimately changed project state that
several tests/manifests had pinned. Fixes, all intent-preserving:

1. **Publication evidence stale-snapshot rejection** — the evidence collector pinned the
   `results/lexical/baseline` manifest, whose input hashes predate the registry change.
   A current preprocessing-only snapshot (`results/lexical/preprocessing_current`,
   tables byte-identical to `with_llama`) is now the `LEXICAL` pointer in
   `reports/publication/evidence.py`.
2. **Run discovery vs. retained failed attempts** — `analysis.lexical_bias` discovery now
   skips runs with zero observed responses (failed attempts kept for provenance), so they
   are never pooled with their completed retries; an explicit `--run` selection stays
   strict, and a bare `--run` selects no runs (`nargs="*"`).
3. **`query_id` type mismatch** — run observations carry the integer grid query id; the
   bias validator now accepts exact nonnegative integers alongside nonblank strings.
4. **Registry-enablement assumptions in tests** — the reasoning CLI fixture normalizes
   copied registries to all-disabled; `test_inference_config` pins the recorded 2026-10-27
   enablement (`qwen25-05b-hf`, `openai-gpt4o-snapshot`) and starts the pending-state test
   from `enabled: false`.
5. **Order-dependent offline flag** — `huggingface_hub` freezes `HF_HUB_OFFLINE` at import;
   the lexical CLI network-isolation test now also patches every imported copy of the flag.

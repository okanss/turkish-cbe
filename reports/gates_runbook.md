# Gates runbook — what unlocks what, and the exact commands

State as of 2026-10-29. Four open items chain off ONE human prerequisite: the
two-reviewer Turkish linguistic review. Nothing below can be started by
automation alone, by design; the commands are staged so each step is a single
action once its human input exists. Pinned venv (vendored `tiktoken==0.12.0`)
required throughout; `PYTHONPATH=src`.

> **DECISION (2026-10-29, project owner):** the gate review packages will not
> be completed. The frozen-test run (§1) and the RQ5 evaluation (§2) therefore
> remain unexecuted, exactly as the manuscript's conditional language already
> states; no manuscript claim depends on them. The packages, receipt tooling,
> frozen plan, and local adapters are all retained unchanged, so this decision
> is reversible at any time by simply doing the reviews. The 360-item
> manuscript review (a separate instrument, not a gate) continues: reviewer OB
> is complete, reviewer 2 pending.

## 0. The one blocking prerequisite: human review

Two independent Turkish-proficient reviewers complete, independently:

- `validation/gate_review/frozen_test/review_sample.csv` — 105 rows → unlocks
  the frozen-test run.
- `validation/gate_review/training_eval/review_sample.csv` — 485 rows →
  unlocks the RQ5 evaluation.
- (separate instrument) `validation/turkish_linguistic_review.csv` — 360 rows,
  1–5 scales → the manuscript's P1/pilot review reporting, per
  `validation/REVIEWER_INSTRUCTIONS.md`.

Then, per package:

```bash
PYTHONPATH=src python3 -m evaluation.finalize_review_receipt \
    --package validation/gate_review/frozen_test \
    --reviewer1 "Full Name" --reviewer2 "Full Name"
```

This writes `receipt.json` (status approved only if every human decision
passed) and self-checks it against the real execution gate. Instructions:
`validation/gate_review/GATE_REVIEW_INSTRUCTIONS.md`.

## 1. Frozen-test confirmatory run (gpt-4o)

Plan is already generated and immutable: `logs/frozen_test_gpt4o_plan.json`
(SHA `1b8ea52d…`): default full selection (test split, spatial_front,
canonical+equivalent, native_shared, endpoint, sizes 4–1024), **7,200 calls**,
estimated **$86.96** (configured 2026-10-24 gpt-4o prices). Expected blockers
recorded in the plan: call budget and pilot approval — both resolve at
execution time with the receipt:

```bash
# (a) pilot approval receipt — requires the frozen-test review receipt and an
#     identified human approver; validates the real, uncached completed pilot.
PYTHONPATH=src python3 -m inference approve-pilot \
    --run-id pilot-gpt4o-main-003 --approved-by "<name>" \
    --linguistic-review validation/gate_review/frozen_test/receipt.json \
    --output logs/pilot_approval_gpt4o.json

# (b) execute the saved plan (needs OPENAI_API_KEY in the environment; on
#     macOS python.org builds also SSL_CERT_FILE at certifi's bundle).
PYTHONPATH=src python3 -m inference full \
    --plan logs/frozen_test_gpt4o_plan.json --execute \
    --run-id frozen-gpt4o-main-001 \
    --pilot-approval logs/pilot_approval_gpt4o.json \
    --linguistic-review validation/gate_review/frozen_test/receipt.json \
    --max-calls 7200
```

## 2. RQ5 base-vs-adapter evaluation (RunPod)

All five adapters and the frozen plan are safe LOCALLY
(`results/training/runs/pod-*-001/`, `results/training/plans/pod_pilot_002.json`);
the pod's repo copy was deleted on 2026-10-29 and is not needed. The evaluation
host (RunPod pod, or a Colab A100 — any Linux box) must be rebuilt: rsync this
project up, recreate the venv, and restore the hash-pinned base weights at the
plan's absolute paths `/workspace/qwen15b-weights` +
`/workspace/qwen15b-weights-manifest.json` (if the network volume survived,
they may still be there; otherwise re-download the pinned Qwen2.5-1.5B-Instruct
revision and re-verify — the plan requires the manifest bytes to hash to
`bc4de801…`). Per arm, 10,224 calls; the frozen design is not reducible. Then:

```bash
ssh root@<pod-ip> -p <port> -i ~/.ssh/id_ed25519
python3 -m venv /root/tcbe-venv            # container disk, NOT /workspace,
                                           # NOT --system-site-packages
/root/tcbe-venv/bin/pip install torch==2.9.0 transformers==4.57.1 \
    peft==0.17.1 accelerate==1.11.0 tiktoken==0.12.0 pyyaml
# rsync the completed training_eval receipt + CSV to the pod copy, then per arm:
PYTHONPATH=src /root/tcbe-venv/bin/python -m training evaluate \
    --plan results/training/plans/pod_pilot_002.json --arm <arm> --execute \
    --max-calls 10224 \
    --linguistic-review validation/gate_review/training_eval/receipt.json \
    --adapter-run results/training/runs/pod-<arm>-001 \
    --output results/training/eval/pod-<arm>-eval-001
```

(Also run `--arm base` once, without `--adapter-run`. Verify exact CLI flag
spellings with `python -m training evaluate --help` on the pod; the gate and
budget semantics above are from `src/training/evaluation.py`.)

## 3. Optional gpt-4o P1 arm (~$51)

16,618 calls over the P1 manifest (estimate $40–65). Two prerequisites, both
user-held: the OPENAI key (never stored in this repo) and explicit go for the
spend. Note it exceeds the 2,000-call threshold, so like the local quantized
arms it is exploratory pending review, and its runner (a thin OpenAI
counterpart to `inference.ollama_runner`, same checkpointing and strict
parser) is written only on that go. It adds a registered-precision
reference line to Figure 6; it is not required for any present claim.

## 4. Submission mechanics (no gate)

MDPI Information: keywords, author metadata (ORCIDs, corresponding author),
availability / funding / COI / acknowledgments statements. Drafted directly in
`paper/main.tex` with placeholders the authors must fill; see the manuscript's
`%% TODO(submission)` comments.

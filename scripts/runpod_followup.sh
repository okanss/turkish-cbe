#!/usr/bin/env bash
set -euo pipefail

# Connectionless RunPod runner for Turkish-CBE Revision R1 (Reviewer 2, Comment 5a).
# Safe to re-run after pod restart/crash: resumes from existing JSONL records.

REPO_URL="${REPO_URL:-https://github.com/okanss/turkish-cbe.git}"
BRANCH="${BRANCH:-revision-r1-followup}"
WORKDIR="${WORKDIR:-/workspace/turkish-cbe}"
MODEL="${MODEL:-qwen2.5:14b}"
EXPECTED_MANIFEST_SHA="89dde862db84447ec23a20302f1c4449f9504aed3f681abc04c92681b185deee"
EXPECTED_DIGEST_PREFIX="7cdf5a01"

export OLLAMA_MODELS="${OLLAMA_MODELS:-/workspace/ollama_models}"
export OLLAMA_HOST="127.0.0.1:11434"
mkdir -p "$OLLAMA_MODELS"

echo "=== [$(date -u '+%Y-%m-%d %H:%M:%S UTC')] Starting Turkish-CBE R1 followup on RunPod ==="
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv || true

if [ ! -d "$WORKDIR/.git" ]; then
    echo "Cloning $REPO_URL (branch $BRANCH) into $WORKDIR..."
    git clone --depth 1 --branch "$BRANCH" "$REPO_URL" "$WORKDIR"
else
    echo "Repository already exists in $WORKDIR; preserving local results."
fi

cd "$WORKDIR"
mkdir -p results/followup

MANIFEST="data/manifests/followup_reasoning_manifest.jsonl"
OUTPUT_JSONL="results/followup/qwen2-5-14b_structured.jsonl"
PROGRESS_MD="results/followup/PROGRESS.md"
SUMMARY_JSON="results/followup/followup_summary.json"

if [ ! -f "$MANIFEST" ]; then
    python3 scripts/build_followup_manifest.py
fi

ACTUAL_SHA="$(sha256sum "$MANIFEST" | awk '{print $1}')"
if [ "$ACTUAL_SHA" != "$EXPECTED_MANIFEST_SHA" ]; then
    echo "ERROR: Manifest SHA-256 mismatch!"
    echo "  Expected: $EXPECTED_MANIFEST_SHA"
    echo "  Actual:   $ACTUAL_SHA"
    exit 1
fi
echo "Manifest SHA-256 verified: $ACTUAL_SHA"

if ! command -v ollama >/dev/null 2>&1; then
    echo "Installing Ollama..."
    curl -fsSL https://ollama.com/install.sh | sh
fi

if ! curl -sf http://127.0.0.1:11434/api/tags >/dev/null 2>&1; then
    echo "Starting ollama serve (models in $OLLAMA_MODELS)..."
    nohup ollama serve > /workspace/ollama.log 2>&1 < /dev/null &
    for _ in $(seq 1 30); do
        if curl -sf http://127.0.0.1:11434/api/tags >/dev/null 2>&1; then
            break
        fi
        sleep 2
    done
fi

curl -sf http://127.0.0.1:11434/api/tags >/dev/null 2>&1 || {
    echo "ERROR: Ollama server failed to start. Check /workspace/ollama.log"
    exit 1
}

echo "Pulling $MODEL..."
ollama pull "$MODEL"

DIGEST="$(curl -s http://127.0.0.1:11434/api/tags | python3 -c "import sys,json; ms=[m['digest'] for m in json.load(sys.stdin)['models'] if m['name']=='$MODEL']; print(ms[0] if ms else '')")"
echo "Installed $MODEL digest: $DIGEST"
if [[ "$DIGEST" != "${EXPECTED_DIGEST_PREFIX}"* ]]; then
    echo "WARNING: Digest ($DIGEST) does not start with expected prefix $EXPECTED_DIGEST_PREFIX; recording actual digest in run header."
fi

# Smoke check first 5 prompts if starting fresh
if [ ! -f "$OUTPUT_JSONL" ]; then
    echo "Running 5-prompt smoke test..."
    PYTHONPATH=src python3 -m inference.ollama_reasoning_runner \
        --manifest "$MANIFEST" \
        --model "$MODEL" \
        --strategy structured_reasoning \
        --output "$OUTPUT_JSONL" \
        --limit 5 \
        --progress-every 5
    python3 scripts/followup_progress.py \
        --manifest "$MANIFEST" \
        --output-jsonl "$OUTPUT_JSONL" \
        --progress-md "$PROGRESS_MD" \
        --status "SMOKE_OK_RUNNING"
fi

# Periodic local progress writer (every 5 minutes)
(
    while true; do
        sleep 300
        python3 scripts/followup_progress.py \
            --manifest "$MANIFEST" \
            --output-jsonl "$OUTPUT_JSONL" \
            --progress-md "$PROGRESS_MD" \
            --status "RUNNING" >/dev/null 2>&1 || true
    done
) &
MONITOR_PID=$!
trap 'kill "$MONITOR_PID" 2>/dev/null || true' EXIT

echo "Running full structured-reasoning follow-up (resumable)..."
PYTHONPATH=src python3 -m inference.ollama_reasoning_runner \
    --manifest "$MANIFEST" \
    --model "$MODEL" \
    --strategy structured_reasoning \
    --output "$OUTPUT_JSONL" \
    --progress-every 25

kill "$MONITOR_PID" 2>/dev/null || true

echo "Running final paired analysis (Table 14)..."
PYTHONPATH=src python3 scripts/analyze_followup.py \
    --direct results/p1/qwen2-5-14b.jsonl \
    --structured "$OUTPUT_JSONL" \
    --output "$SUMMARY_JSON"

python3 scripts/followup_progress.py \
    --manifest "$MANIFEST" \
    --output-jsonl "$OUTPUT_JSONL" \
    --progress-md "$PROGRESS_MD" \
    --status "DONE"

echo "=== [$(date -u '+%Y-%m-%d %H:%M:%S UTC')] Follow-up completed cleanly! ==="

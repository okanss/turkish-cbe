#!/bin/zsh
# Resume the P1 run exactly where it stopped (completed prompts are skipped
# via the per-model JSONL checkpoint; duplicates are impossible).
cd "$(dirname "$0")/.."
rm -f results/p1/PAUSE
if pgrep -f "inference.ollama_runner" > /dev/null; then
  echo "A runner is already active; nothing to resume."
  exit 1
fi
mkdir -p results/p1
nohup ./scripts/p1_run_all.sh >> results/p1/run_all.log 2>&1 &
echo "resumed $(date -u +%FT%TZ) pid $!" >> results/p1/RUN_STATUS.txt
echo "Resumed in background (pid $!). Progress: tail -f results/p1/run_all.log"

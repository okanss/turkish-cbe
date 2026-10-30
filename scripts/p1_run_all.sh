#!/bin/zsh
# P1 full local evaluation: all four Qwen models, sequential, checkpointed.
# Pause:  ./scripts/p1_pause.sh   (takes effect within one call, ~seconds)
# Resume: ./scripts/p1_resume.sh  (skips every completed prompt exactly)
set -u
cd "$(dirname "$0")/.."
MANIFEST=data/manifests/p1_expanded_manifest.jsonl
STATUS=results/p1/RUN_STATUS.txt
PAUSE=results/p1/PAUSE
mkdir -p results/p1
rm -f "$PAUSE"
echo "launched $(date -u +%FT%TZ) pid $$" >> "$STATUS"
for MODEL in qwen2.5:0.5b qwen2.5:3b qwen2.5:7b qwen2.5:14b; do
  SAFE=${MODEL//[:.]/-}
  echo "start $MODEL $(date -u +%FT%TZ)" >> "$STATUS"
  caffeinate -is env PYTHONPATH=src python3 -m inference.ollama_runner \
    --manifest "$MANIFEST" --model "$MODEL" \
    --output "results/p1/${SAFE}.jsonl" --progress-every 200 \
    --pause-file "$PAUSE"
  CODE=$?
  if [ "$CODE" -eq 75 ]; then
    echo "paused_during $MODEL $(date -u +%FT%TZ)" >> "$STATUS"
    exit 75
  fi
  echo "done $MODEL exit=$CODE $(date -u +%FT%TZ)" >> "$STATUS"
done
echo "all_models_complete $(date -u +%FT%TZ)" >> "$STATUS"

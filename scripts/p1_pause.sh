#!/bin/zsh
# Request a graceful pause of the P1 run: the runner finishes the call in
# flight, checkpoints, and the whole pipeline exits within seconds.
cd "$(dirname "$0")/.."
touch results/p1/PAUSE
echo "pause requested $(date -u +%FT%TZ)" >> results/p1/RUN_STATUS.txt
echo "Pause flag set. The run will stop within one model call."
echo "Resume anytime with: ./scripts/p1_resume.sh"

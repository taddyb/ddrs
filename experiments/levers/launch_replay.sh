#!/usr/bin/env bash
# Launch one zero-step training-years replay, fully detached. Usage: launch_replay.sh <seed>
set -euo pipefail
SEED="$1"
HERE=/home/tbindas/projects/ddrs/.claude/worktrees/agent-a92e512a7c47c97b4/experiments/levers
BIN=/home/tbindas/.claude/jobs/dacd6d8c/tmp/bin/ddrs-v6-77aa4a0
cd /home/tbindas/projects/ddrs/.claude/worktrees/agent-a92e512a7c47c97b4
DDRS_JOURNAL_OFF=1 RAYON_NUM_THREADS=8 setsid nohup "$BIN" --config "$HERE/replay_trainyears_s${SEED}.yaml" \
  --workspace /home/tbindas/projects/ddrs/.ddrs run --workflow train-and-test --backend cpu \
  > "$HERE/replay_s${SEED}.log" 2>&1 < /dev/null &
echo "pid $!"

#!/usr/bin/env bash
# Sequential replay queue: one ddrs run at a time, each only when >= 20 GB memory is available.
# Usage: queue.sh <pid to wait for first> <job> [<job> ...]   (job = replay_<job>.yaml in this directory)
HERE=/home/tbindas/projects/ddrs/.claude/worktrees/agent-a92e512a7c47c97b4/experiments/levers
BIN=/home/tbindas/.claude/jobs/dacd6d8c/tmp/bin/ddrs-v6-77aa4a0
WAITPID="$1"; shift
cd /home/tbindas/projects/ddrs/.claude/worktrees/agent-a92e512a7c47c97b4
while kill -0 "$WAITPID" 2>/dev/null; do sleep 20; done
for JOB in "$@"; do
  while [ "$(free -g | awk '/^Mem:/{print $7}')" -lt 20 ]; do
    echo "$(date -u +%FT%TZ) waiting for memory before $JOB" >> "$HERE/queue.log"; sleep 60
  done
  echo "$(date -u +%FT%TZ) start $JOB" >> "$HERE/queue.log"
  DDRS_JOURNAL_OFF=1 RAYON_NUM_THREADS=8 "$BIN" --config "$HERE/replay_${JOB}.yaml" \
    --workspace /home/tbindas/projects/ddrs/.ddrs run --workflow train-and-test --backend cpu \
    > "$HERE/replay_${JOB}.log" 2>&1 < /dev/null
  echo "$(date -u +%FT%TZ) done $JOB exit=$?" >> "$HERE/queue.log"
done
echo "$(date -u +%FT%TZ) queue finished" >> "$HERE/queue.log"

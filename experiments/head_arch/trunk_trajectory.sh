#!/usr/bin/env bash
# When during training does the trunk collapse?
#
# §32.2 measured the H=21 trunk at effective rank 4.36 of 21 at initialisation.
# The trained 500-update head measures 1.38, with PC1 holding 84.6% instead of
# 41.2%. So training destroys three of the four directions the trunk starts
# with. This walks the checkpoint series to find out when.
#
# Usage:  experiments/head_arch/trunk_trajectory.sh <run-id> [epochs...]
set -u

RUN=${1:?usage: trunk_trajectory.sh <run-id> [epochs...]}
shift
EPOCHS=${*:-"1 2 3 5 8 10 15 20 25 30 40 50"}

ROOT=/home/tbindas/projects/ddrs/.claude/worktrees/kan-split-heads
RUNS=/home/tbindas/projects/ddrs/.ddrs/runs
OUT=/home/tbindas/projects/ddrs/.ddrs/experiments/head-arch/trajectory-$RUN
CFG=$RUNS/$RUN/config.yaml

for e in $EPOCHS; do
  # Highest mini-batch present for that epoch = end of the epoch.
  ck=$(ls -d "$RUNS/$RUN/checkpoints/epoch_${e}_mb_"* 2>/dev/null \
       | sed "s/.*_mb_//" | sort -n | tail -1)
  [ -z "$ck" ] && { echo "epoch $e: no checkpoint"; continue; }
  d=$OUT/epoch_$e
  mkdir -p "$d"
  "$ROOT/target/release/head_arch_screen" --config "$CFG" --out-dir "$d" \
    --sample 4000 \
    --checkpoint "$RUNS/$RUN/checkpoints/epoch_${e}_mb_${ck}" >/dev/null 2>&1 \
    && echo "epoch $e (mb $ck): dumped" || echo "epoch $e: FAILED"
done
echo "now: experiments/head_arch/trunk_trajectory.py $OUT"

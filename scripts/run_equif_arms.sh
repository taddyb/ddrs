#!/usr/bin/env bash
# Sequential CPU training of the three selective-equifinality arms (R1-R3).
# Detach with: nohup scripts/run_equif_arms.sh > output/equif_runs.log 2>&1 &
# Survives agent/session death. Projected (Task 4 smoke): R1 ~12 min,
# R2 ~80 min, R3 ~91 min + eval phases.
set -uo pipefail
cd "$(dirname "$0")/.."

WS=/home/tbindas/projects/ddrs/.ddrs
ARMS=(equif_daily_lstm_flat equif_daily_lstm_disagg equif_hourly_lstm)
STATUS_FILE=output/equif_runs.status
mkdir -p output
: > "$STATUS_FILE"

for c in "${ARMS[@]}"; do
  echo "[$(date -u +%FT%TZ)] START $c" | tee -a "$STATUS_FILE"
  if ddrs --workspace "$WS" --config "config/experiments/$c.yaml" \
       run --backend cpu --workflow train-and-test; then
    echo "[$(date -u +%FT%TZ)] OK    $c" | tee -a "$STATUS_FILE"
  else
    echo "[$(date -u +%FT%TZ)] FAIL  $c (exit $?)" | tee -a "$STATUS_FILE"
    # keep going — arms are independent; a failed arm is diagnosed from
    # its run.log, the others still produce science
  fi
done
echo "[$(date -u +%FT%TZ)] ALL DONE" | tee -a "$STATUS_FILE"

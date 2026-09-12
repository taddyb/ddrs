#!/usr/bin/env bash
# Finish the tau=9 CPU set after the 2026-08-09 driver kill: hourly_lstm
# eval-only resume (epoch-30 checkpoint, zero training batches) then the
# aorc2f_lumped full run. Launch with setsid so it survives the parent.
set -uo pipefail
cd /home/tbindas/projects/ddrs
DDRS=target/release/ddrs
for cfg in tau9_train_hourly_lstm_evalresume tau9_train_aorc2f_lumped; do
  echo "=== ${cfg} start $(date -u +%FT%TZ) ==="
  $DDRS --config config/experiments/${cfg}.yaml --workspace .ddrs \
    run --workflow train-and-test --backend cpu \
    > output/tau_sweep/train9_${cfg}.launch.log 2>&1 \
    || { echo "=== ${cfg} FAILED $(date -u +%FT%TZ) ==="; continue; }
  echo "=== ${cfg} done $(date -u +%FT%TZ) ==="
done
echo "=== remaining arms done $(date -u +%FT%TZ) ==="

#!/usr/bin/env bash
# After run_tau9_remaining.sh (the aorc2f_lumped run) exits, evaluate the
# hourly_lstm arm's final checkpoint with the legacy eval binary — the
# train-and-test zero-batch resume cannot re-enter Phase 2 (it requires
# checkpoints written by its own Phase 1), so this is the completion path
# for the arm whose eval was killed mid-run on 2026-08-09.
set -uo pipefail
cd /home/tbindas/projects/ddrs
while pgrep -f "run_tau9_remaining.sh" > /dev/null; do sleep 60; done
echo "=== hourly_lstm legacy eval start $(date -u +%FT%TZ) ==="
target/release/eval \
  --config config/experiments/tau9_train_hourly_lstm.yaml \
  --checkpoint .ddrs/runs/2026-08-09T14-55-05Z-train-and-test/checkpoints/epoch_30_mb_1 \
  --backend cpu \
  --output "$PWD/output/tau_sweep/train9_hourly_lstm_eval.zarr" \
  > output/tau_sweep/train9_hourly_lstm_eval.log 2>&1
echo "=== hourly_lstm legacy eval done $(date -u +%FT%TZ), exit $? ==="

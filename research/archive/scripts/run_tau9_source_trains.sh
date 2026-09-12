#!/usr/bin/env bash
# tau=9 cross-source retrain: five train-and-test runs, one per streamflow
# store, all on CPU (user directive 2026-08-08). Each config is the epoch-30
# run's snapshot with only {streamflow, sparse_solver: cpu} changed and
# params.tau left to the new default 9 (2026-08-08 convention == old 20).
#
# Uses target/release/ddrs (NOT the installed ~/.cargo/bin copy) to dodge the
# stale-binary trap after the tau convention change.
# --workspace pins .ddrs to the repo root (configs live under config/experiments/).
set -uo pipefail
cd /home/tbindas/projects/ddrs

DDRS=target/release/ddrs
for src in aorc2f_dist uh_retro daily_lstm hourly_lstm aorc2f_lumped; do
  echo "=== train ${src} start $(date -u +%FT%TZ) ==="
  $DDRS --config config/experiments/tau9_train_${src}.yaml --workspace .ddrs \
    run --workflow train-and-test --backend cpu \
    > output/tau_sweep/train9_${src}.launch.log 2>&1 \
    || { echo "=== train ${src} FAILED $(date -u +%FT%TZ), continuing ==="; continue; }
  echo "=== train ${src} done $(date -u +%FT%TZ) ==="
done
echo "=== all arms done $(date -u +%FT%TZ) ==="

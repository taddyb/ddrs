#!/usr/bin/env bash
# Option C sandbox on the dam benchmark: eval the sr_n0_gamma head on the 121 benchmark gauges with
# reservoirs off, with ResOpsUS-fitted T (44 dams), and with fitted + median-prior T (121 dams).
# Sequential on CPU so the wall times are comparable. Inputs from prepare_option_c_eval.py.
#   bash experiments/reservoir/benchmark/run_option_c_eval.sh [arm ...]   (default: off fit prior)
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/../../.." && pwd)"
OUT="$ROOT/output/reservoir_benchmark"
CKPT=/home/tbindas/projects/ddrs/.ddrs/runs/2026-09-12T23-39-03Z-train-and-test/checkpoints/epoch_50_mb_9
BIN="$ROOT/target/release/eval"
cd "$ROOT"
for arm in "${@:-off fit prior}"; do
  for a in $arm; do
    rm -rf "$OUT/pred_$a.zarr"
    start=$(date +%s.%N)
    "$BIN" --config "$OUT/eval_$a.yaml" --checkpoint "$CKPT" --output "$OUT/pred_$a.zarr" \
      --backend cpu > "$OUT/eval_$a.log" 2>&1
    end=$(date +%s.%N)
    echo "$a wall_seconds $(awk -v s="$start" -v e="$end" 'BEGIN { printf "%.1f", e - s }')" | tee -a "$OUT/timing.txt"
  done
done

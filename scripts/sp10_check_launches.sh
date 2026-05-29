#!/usr/bin/env bash
# SP-10 V10: gate cuLaunchKernel total call count against the SP-9 baseline.
#
# Forward CUDA graphs are active per commit afb7754 — they should reduce the
# per-batch launch count materially. Backward still issues direct launches
# (SP-9 path), so the realistic target is forward-only: ≥40% drop.
#
# SP-9 baseline (commit a67a7e2, 3 mini-batches, sparse_solver=cuda,
# use_cuda_graphs=false): 7,684,365 cuLaunchKernel calls.
set -euo pipefail

NSYS_DIR="${NSYS_DIR:-$HOME/nsys_out}"
mkdir -p "$NSYS_DIR"
REPORT="$NSYS_DIR/sp10_v10"
CKPT="/tmp/sp10_v10_ckpt"
SP9_BASELINE="${SP9_BASELINE:-7684365}"
THRESHOLD_PCT="${SP10_LAUNCH_DROP_PCT:-40.0}"

# Inject both sparse_solver: cuda and use_cuda_graphs: true under params:.
TMP_CFG="/tmp/sp10_v10.yaml"
awk '
    /^params:/ {
        print;
        print "  sparse_solver: cuda";
        print "  use_cuda_graphs: true";
        next
    }
    /sparse_solver:/ { next }
    /use_cuda_graphs:/ { next }
    { print }
' config/merit_training.yaml > "$TMP_CFG"

rm -rf "$CKPT"
nsys profile --trace=cuda --sample=none --cpuctxsw=none \
    --output="$REPORT" --force-overwrite=true \
    target/release/train --config "$TMP_CFG" \
                          --checkpoint-dir "$CKPT" \
                          --max-mini-batches 3

STATS="$NSYS_DIR/sp10_v10_stats.txt"
nsys stats --force-export=true "$REPORT.nsys-rep" --report cuda_api_sum > "$STATS"

# cuda_api_sum columns: "Time (%)", "Total Time (ns)", "Num Calls", "Avg",
# "Med", "Min", "Max", "StdDev", "Name". We want column 3 ("Num Calls") for
# the cuLaunchKernel row.
CALLS=$(awk '
    /cuLaunchKernel/ {
        gsub(",", "", $3);
        print $3;
        exit;
    }
' "$STATS")

if [ -z "$CALLS" ]; then
    echo "V10 ERROR: cuLaunchKernel not found in $STATS"
    cat "$STATS" | head -40
    exit 1
fi

DROP_PCT=$(awk -v c="$CALLS" -v b="$SP9_BASELINE" \
    'BEGIN { printf "%.2f", (1.0 - c/b) * 100.0 }')

echo "V10: cuLaunchKernel calls = $CALLS (SP-9 baseline = $SP9_BASELINE)"
echo "V10: drop = $DROP_PCT% (threshold $THRESHOLD_PCT%)"
awk -v p="$DROP_PCT" -v t="$THRESHOLD_PCT" 'BEGIN { exit (p+0 >= t+0) ? 0 : 1 }'

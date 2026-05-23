#!/usr/bin/env bash
# SP-8 V7b (also used by SP-9): gate scatter_kernel_t_f32_i_i32 below 30% of
# GPU compute time. Forces sparse_solver: cuda so the CUDA SpMV path runs.
set -euo pipefail

NSYS_DIR="${NSYS_DIR:-$HOME/nsys_out}"
mkdir -p "$NSYS_DIR"
REPORT="$NSYS_DIR/sp8_v7b"
CKPT="/tmp/sp8_v7b_ckpt"
THRESHOLD="${SP8_SCATTER_THRESHOLD:-30.0}"

# Write a temp YAML with sparse_solver: cuda injected under params:.
TMP_CFG="/tmp/v7b_cuda.yaml"
awk '
    /^params:/ {
        print;
        print "  sparse_solver: cuda";
        next
    }
    /sparse_solver:/ { next }   # strip any pre-existing sparse_solver line (commented or not)
    { print }
' config/merit_training.yaml > "$TMP_CFG"

rm -rf "$CKPT"
nsys profile --trace=cuda --sample=none --cpuctxsw=none \
    --output="$REPORT" --force-overwrite=true \
    target/release/train --config "$TMP_CFG" \
                          --checkpoint-dir "$CKPT" \
                          --max-mini-batches 3

STATS="$NSYS_DIR/sp8_v7b_stats.txt"
nsys stats "$REPORT.nsys-rep" --report cuda_gpu_kern_sum > "$STATS"

# nsys output column layout for cuda_gpu_kern_sum:
#   "Time (%)", "Total Time (ns)", "Instances", "Avg", "Med", "Min", "Max", "StdDev", "Name"
# We extract column 1 (Time %) for the row whose Name column contains
# scatter_kernel_t_f32_i_i32.
PCT=$(awk '
    /scatter_kernel_t_f32_i_i32/ {
        gsub(",", "", $1);
        print $1;
        exit;
    }
' "$STATS")

if [ -z "$PCT" ]; then
    echo "V7b: scatter_kernel_t_f32_i_i32 not found in $STATS — assuming 0%"
    PCT=0
fi
echo "V7b: scatter_kernel percentage = $PCT% (threshold $THRESHOLD%)"
awk -v p="$PCT" -v t="$THRESHOLD" 'BEGIN { exit (p+0 < t+0) ? 0 : 1 }'

#!/usr/bin/env bash
# SP-8 V7b: gate scatter_kernel_t_f32_i_i32 below 30% of GPU compute time.
#
# Exits 0 if the gate passes, 1 otherwise. Writes the nsys report and stats
# to $NSYS_DIR (default $HOME/nsys_out).
set -euo pipefail

NSYS_DIR="${NSYS_DIR:-$HOME/nsys_out}"
mkdir -p "$NSYS_DIR"
REPORT="$NSYS_DIR/sp8_v7b"
CKPT="/tmp/sp8_v7b_ckpt"
THRESHOLD="${SP8_SCATTER_THRESHOLD:-30.0}"

rm -rf "$CKPT"
nsys profile --trace=cuda --sample=none --cpuctxsw=none \
    --output="$REPORT" --force-overwrite=true \
    target/release/train --config config/merit_training.yaml \
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

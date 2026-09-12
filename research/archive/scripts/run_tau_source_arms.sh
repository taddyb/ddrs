#!/usr/bin/env bash
# Cross-source tau sweep: same epoch-30 checkpoint, same 2,365-gauge network,
# only data_sources.streamflow changes. Nearest (repeat-24) upsampling for
# daily stores; hourly_lstm is hourly-native (no upsampling). The aorc2f
# distributed store is already covered by output/tau_sweep/g3000_nearest/.
set -euo pipefail
cd /home/tbindas/projects/ddrs

CKPT=.ddrs/runs/2026-08-05T04-58-58Z-conus-experimental-train-and-test/checkpoints/epoch_30_mb_1

for src in uh_retro daily_lstm hourly_lstm aorc2f_lumped; do
  out=output/tau_sweep/src_${src}
  mkdir -p "$out"
  echo "=== source: ${src} $(date -u +%FT%TZ) ==="
  # CPU by default (user preference 2026-08-06): deterministic NdArray, keeps
  # the GPU free. NOTE: never mix backends WITHIN one comparison set — the
  # 2026-08-06 cross-source set ran entirely on cuda.
  DDRS_HOURLY_DUMP=$PWD/${out}/hourly.f32 \
  target/release/eval \
    --config config/experiments/tau_src_${src}.yaml \
    --checkpoint "$CKPT" \
    --backend cpu \
    --output "$PWD/${out}/eval.zarr" \
    > "${out}/eval.log" 2>&1 || { echo "=== source ${src} FAILED, continuing ==="; continue; }
  echo "=== source ${src} done $(date -u +%FT%TZ) ==="
done

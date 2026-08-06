#!/usr/bin/env bash
# 3-arm q'-interpolation experiment: nearest / linear / quadratic upsampling,
# gages_3000 population, epoch-30 checkpoint, full eval window + hourly dump.
# Spec: docs/superpowers/specs/2026-08-05-per-gauge-tau-sweep-design.md (Phase 2 prep)
set -euo pipefail
cd /home/tbindas/projects/ddrs

CKPT=.ddrs/runs/2026-08-05T04-58-58Z-conus-experimental-train-and-test/checkpoints/epoch_30_mb_1
CFG=config/experiments/tau_interp_g3000.yaml

for mode in nearest linear quadratic; do
  out=output/tau_sweep/g3000_${mode}
  mkdir -p "$out"
  echo "=== arm: ${mode} $(date -u +%FT%TZ) ==="
  # CPU by default (user preference 2026-08-06): deterministic NdArray, keeps
  # the GPU free. NOTE: never mix backends WITHIN one comparison set — the
  # 2026-08-06 interp arms ran entirely on cuda.
  DDRS_QPRIME_INTERP=${mode} \
  DDRS_HOURLY_DUMP=$PWD/${out}/hourly.f32 \
  target/release/eval \
    --config "$CFG" \
    --checkpoint "$CKPT" \
    --backend cpu \
    --output "$PWD/${out}/eval.zarr" \
    > "${out}/eval.log" 2>&1
  echo "=== arm ${mode} done $(date -u +%FT%TZ), exit $? ==="
done

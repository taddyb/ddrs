#!/usr/bin/env bash
# Train the head-topology arms serially on the single GPU.
#
# Arms differ from each other ONLY in `kan_head`; everything else comes from the
# 500-update nse-batch baseline 2026-09-10T21-21-48Z-conus-train-and-test.
# Order puts the matched control first so a crash later still leaves a
# comparable pair, and the discriminating arm (kan_readout) last.
#
# `--plot` is not optional here: the whole measurement is rho(n, q) over the
# full CONUS parameter field, which only `plot/kan_parameters.nc` carries.
#
# `--workspace` is likewise not optional: without it `.ddrs` lands beside the
# config in config/experiments/ (ddrs-dev fact 2).
#
# `--backend cpu` is NOT a diagnostic convention here, it is the fast path.
# Measured 2026-09-11: CUDA runs this model at ~20 s per micro-batch with the
# GPU only 7% utilised, against ~3.1 s on CPU. The sparse triangular solve is
# sequential in the reach ordering, so the GPU pays kernel-launch overhead per
# timestep with no parallelism to recover it. The baseline these arms are
# compared against (2026-09-10T21-21-48Z, 50 epochs in 97 min, NSE 0.7376 /
# KGE 0.7600) also ran on CPU, so this keeps the comparison like for like.
# Pausing the six concurrent landscape shards changed the CUDA rate only from
# 22 s to 20 s, which is how the GPU rather than contention was identified.
#
# Run detached so it survives the session:
#   systemd-run --user --unit=ddrs-head-arms --collect \
#     --working-directory=$PWD experiments/head_arch/train_arms.sh
set -u

ROOT=/home/tbindas/projects/ddrs/.claude/worktrees/kan-split-heads
WS=/home/tbindas/projects/ddrs/.ddrs
BIN=$ROOT/target/release/ddrs
LOG=$WS/experiments/head-arch/train_arms.log

cd "$ROOT" || exit 1
mkdir -p "$(dirname "$LOG")"

for arm in head_shared_linear head_split_trunk head_wider head_kan_readout; do
  echo "[$(date -u +%FT%TZ)] START $arm" | tee -a "$LOG"
  "$BIN" --config "config/experiments/$arm.yaml" --workspace "$WS" \
    run --workflow train-and-test --plot --backend cpu
  rc=$?
  echo "[$(date -u +%FT%TZ)] END $arm rc=$rc" | tee -a "$LOG"
  # Keep going on failure: one arm dying should not cost the rest of the night.
done
echo "[$(date -u +%FT%TZ)] ALL ARMS DONE" | tee -a "$LOG"

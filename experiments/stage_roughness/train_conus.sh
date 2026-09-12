#!/usr/bin/env bash
# Matched CONUS pair for stage-dependent roughness.
#
# Both arms are `config/experiments/head_shared_linear.yaml` (the 500-update
# nse-batch baseline with n + p_spatial + q_spatial learnable), differing ONLY
# in `params.stage_roughness`. The control is re-run rather than reused from the
# head-topology chain, because that ran on an earlier binary and a matched pair
# on one binary is worth the two hours.
#
# gamma = 0.35 is the Juniata optimum, not the literature value. The
# literature-motivated 0.183 reproduces observed at-a-station exponents when
# paired with q ~ 0.65; the Juniata sweep peaked higher, which probably means
# gamma is also absorbing the model's too-deep flood geometry. Recorded as a
# caveat, not hidden: see experiments/stage_roughness/juniata/gamma_sweep.log.
#
#   gamma   Juniata NSE   Juniata KGE
#   0.00       0.7903        0.8810
#   0.183      0.8630        0.9256
#   0.25       0.8767        0.9326
#   0.35       0.8813        0.9344   <- peak
#   0.50       0.8762        0.9299
#
# --backend cpu: CUDA runs this model ~6x slower (traps.md T13).
set -u

ROOT=/home/tbindas/projects/ddrs/.claude/worktrees/kan-split-heads
WS=/home/tbindas/projects/ddrs/.ddrs
BIN=$ROOT/target/release/ddrs
SRC=$ROOT/config/experiments/head_shared_linear.yaml
OUT=$ROOT/config/experiments
LOG=$WS/experiments/stage-roughness-conus.log

cd "$ROOT" || exit 1
mkdir -p "$(dirname "$LOG")"

for g in 0.0 0.35; do
  name=$(echo "sr_gamma_$g" | tr '.' 'p')
  cfg="$OUT/$name.yaml"
  python3 - "$SRC" "$cfg" "$g" "$name" <<'PY'
import sys, pathlib
src, dst, g, name = sys.argv[1:5]
t = pathlib.Path(src).read_text()
banner = (
    f"# {name}\n#\n"
    f"# Stage-dependent roughness arm, gamma = {g}. Identical to\n"
    "# head_shared_linear.yaml in every other respect, so the pair is matched.\n"
    "# n(d) = n_0*(d/d_ref)^(-gamma); gamma = 0 is byte-identical to the\n"
    "# historical solver (tests/stage_roughness.rs::gamma_zero_is_bit_identical).\n"
    "# Design: docs/superpowers/specs/2026-09-12-stage-dependent-roughness-design.md\n#\n"
)
if float(g) != 0.0:
    t = t.replace("params:\n  parameter_ranges:",
                  f"params:\n  stage_roughness:\n    gamma: {g}\n    d_ref: 1.0\n  parameter_ranges:", 1)
pathlib.Path(dst).write_text(banner + t)
PY
  echo "[$(date -u +%FT%TZ)] START $name" | tee -a "$LOG"
  "$BIN" --config "$cfg" --workspace "$WS" run --workflow train-and-test --plot --backend cpu
  echo "[$(date -u +%FT%TZ)] END $name rc=$?" | tee -a "$LOG"
done
echo "[$(date -u +%FT%TZ)] ALL DONE" | tee -a "$LOG"

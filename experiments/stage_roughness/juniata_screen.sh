#!/usr/bin/env bash
# Cheap 2x2 screen for stage-dependent roughness on the Juniata sample.
#
# A CONUS train costs ~2 h; Juniata costs ~21 s, so the value of gamma gets
# chosen here rather than guessed.
#
# WHY BOTH AXES. At-a-station the model gives
#   m = (2 + 3·gamma) / (5 + 3q + 3·gamma)   against an observed m ~ 0.34.
# At the trained q ~ 0.084 the model is ALREADY at m = 0.381, above the
# observation, so gamma alone pushes it to 0.439 and makes the geometry worse.
# gamma only helps near q ~ 0.65, where (q=0.65, gamma=0.183) reproduces the
# observed (b, f, m) = (0.26, 0.40, 0.34) exactly. So gamma must be screened
# together with q, not on its own.
#
# q cannot simply be fixed: `training/forward.rs` requires `q_spatial` in
# `kan_head.learnable_parameters`. The config-only equivalent is to narrow its
# RANGE so the sigmoid midpoint lands on the target, which also makes the
# untrained prior Leopold & Maddock instead of arbitrary.
#
#   arm            gamma   q range        sigmoid-centre q   implied m
#   control        0.0     [0.0, 1.0]     0.50               0.348
#   gamma-only     0.183   [0.0, 1.0]     0.50               0.404
#   q-only         0.0     [0.4, 0.9]     0.65               0.288
#   at-a-station   0.183   [0.4, 0.9]     0.65               0.340  <- target
#
# Run from the repo root. The Juniata workspace intentionally lands beside its
# config (the one sanctioned exception to the --workspace rule).
set -u

ROOT=/home/tbindas/projects/ddrs/.claude/worktrees/kan-split-heads
BIN=$ROOT/target/release/ddrs
SRC=$ROOT/examples/juniata/ddrs.yaml
OUT=$ROOT/experiments/stage_roughness/juniata
LOG=$OUT/screen.log

cd "$ROOT" || exit 1
mkdir -p "$OUT"
: > "$LOG"

run_arm() {
  local name=$1 gamma=$2 qlo=$3 qhi=$4
  local cfg="$OUT/juniata_$name.yaml"
  python3 - "$SRC" "$cfg" "$gamma" "$qlo" "$qhi" <<'PY'
import sys, re, pathlib
src, dst, gamma, qlo, qhi = sys.argv[1:6]
t = pathlib.Path(src).read_text()
t = re.sub(r"^(\s*)q_spatial: \[.*\]$",
           rf"\g<1>q_spatial: [{qlo}, {qhi}]", t, count=1, flags=re.M)
if float(gamma) != 0.0:
    t = t.replace("params:\n  parameter_ranges:",
                  f"params:\n  stage_roughness:\n    gamma: {gamma}\n    d_ref: 1.0\n  parameter_ranges:",
                  1)
pathlib.Path(dst).write_text(t)
PY
  echo "=== $name  gamma=$gamma  q=[$qlo,$qhi]" | tee -a "$LOG"
  "$BIN" --config "$cfg" run --workflow train-and-test --backend cpu 2>&1 \
    | grep -E "median NSE|median KGE|error|panic" | tee -a "$LOG"
}

run_arm control      0.0   0.0 1.0
run_arm gamma-only   0.183 0.0 1.0
run_arm q-only       0.0   0.4 0.9
run_arm at-a-station 0.183 0.4 0.9

echo "=== DONE" | tee -a "$LOG"

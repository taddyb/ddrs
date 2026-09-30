#!/usr/bin/env bash
# Expected-output run for the dam-release smoke set: the no-dam trained head (sr_n0_gamma,
# 2026-09-12T23-39-03Z @ epoch_50_mb_9) routed over 1981-10-01..2010-09-30 in one continuous eval on the
# smoke gauges (gages_smoke.csv), CPU. One window covering training and test years, so the routed inflow for the
# offline release fit has no restart between them and a single gauge set (trap T19).
# Output: output/reservoir_smoke/pred_1981_2010.zarr, eval.log. Then run expected_release_fit.py.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/../../.." && pwd)"
OUT="$ROOT/output/reservoir_smoke"
RUN=/home/tbindas/projects/ddrs/.ddrs/runs/2026-09-12T23-39-03Z-train-and-test
mkdir -p "$OUT"
cd "$ROOT"
~/projects/ddr/.venv/bin/python - "$RUN/config.yaml" "$ROOT/experiments/reservoir/smoke/gages_smoke.csv" "$OUT/eval_smoke.yaml" <<'PY'
import re, sys
src, gages, dst = sys.argv[1:]
c = open(src).read()
c, n1 = re.subn(r"(?m)^(\s*gages:\s*)\S+", lambda m: m.group(1) + gages, c)
t = c.index("\ntesting:")
head, tail = c[:t], c[t:]
tail, n2 = re.subn(r"(?m)^(\s*start_time:\s*)\S+", r"\g<1>1981/10/01", tail, count=1)
tail, n3 = re.subn(r"(?m)^(\s*end_time:\s*)\S+", r"\g<1>2010/09/30", tail, count=1)
assert (n1, n2, n3) == (1, 1, 1), (n1, n2, n3)
open(dst, "w").write(head + tail)
PY
rm -rf "$OUT/pred_1981_2010.zarr"
start=$(date +%s)
"$ROOT/target/release/eval" --config "$OUT/eval_smoke.yaml" --checkpoint "$RUN/checkpoints/epoch_50_mb_9" \
  --output "$OUT/pred_1981_2010.zarr" --backend cpu > "$OUT/eval.log" 2>&1
echo "wall_seconds $(( $(date +%s) - start ))" | tee "$OUT/timing.txt"

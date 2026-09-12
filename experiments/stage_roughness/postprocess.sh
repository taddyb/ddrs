#!/usr/bin/env bash
# Everything that should happen the moment the CONUS stage-roughness pair lands.
#
# Chained after `ddrs-sr-conus` so it runs unattended:
#   systemd-run --user --unit=ddrs-sr-post --collect \
#     --working-directory=$PWD experiments/stage_roughness/postprocess.sh
#
# Produces, for the gamma = 0 control and the gamma = 0.35 arm:
#   1. skill, side by side, and against the summed-Q' baseline
#   2. the DOWNSTREAM WIDTH EXPONENT b — §33.3 requires this next to every NSE,
#      because freeing p improved skill while driving b from 0.099 to 0.004 and
#      skill alone would have logged that as a clean win
#   3. output collapse: rho(n,q), affine R^2, trained trunk rank
#   4. the n(d) animation over a water year
set -u

ROOT=/home/tbindas/projects/ddrs/.claude/worktrees/kan-split-heads
WS=/home/tbindas/projects/ddrs/.ddrs
OUT=$WS/experiments/stage-roughness-report.txt
cd "$ROOT" || exit 1

# Wait for the training chain, however it ends.
while systemctl --user is-active -q ddrs-sr-conus; do sleep 60; done

: > "$OUT"
say() { echo "$@" | tee -a "$OUT"; }

say "stage-dependent roughness — CONUS pair"
say "generated $(date -u +%FT%TZ)"
say ""

# Newest run per arm config, identified by the banner the config carries.
runs=()
for cfg in sr_gamma_0p0 sr_gamma_0p35; do
  r=$(grep -l "^# $cfg\$" "$WS"/runs/*/config.yaml 2>/dev/null \
      | sed 's|.*/runs/||; s|/config.yaml||' | sort | tail -1)
  [ -n "$r" ] && runs+=("$r") && say "$cfg -> $r"
done
say ""
[ ${#runs[@]} -eq 0 ] && { say "no arm runs found"; exit 1; }

say "== SKILL =="
for r in "${runs[@]}"; do
  m=$(python3 -c "
import json,sys
d=json.load(open('$WS/runs/$r/manifest.json'))
out={}
def w(o):
    if isinstance(o,dict):
        for k,v in o.items():
            if isinstance(v,(int,float)) and 'median' in k.lower() and ('nse' in k.lower() or 'kge' in k.lower()): out[k]=v
            w(v)
    elif isinstance(o,list):
        [w(v) for v in o]
w(d); print(' '.join(f'{k}={v:.4f}' for k,v in sorted(out.items())))
" 2>/dev/null)
  say "  $r  $m"
done
say ""

say "== DOWNSTREAM WIDTH EXPONENT (§33.3: report this next to NSE, always) =="
./experiments/head_arch/downstream_geometry.py "${runs[@]}" 2>&1 | tee -a "$OUT"
say ""

say "== OUTPUT COLLAPSE =="
./experiments/head_arch/compare_arms.py "${runs[@]}" 2>&1 | tail -60 | tee -a "$OUT"
say ""

say "== TRAINED TRUNK RANK =="
for r in "${runs[@]}"; do
  ck=$(ls -d "$WS/runs/$r/checkpoints/epoch_"* 2>/dev/null \
       | sed 's/.*epoch_//' | tr '_' ' ' | awk '{print $1, $3}' \
       | sort -k1,1n -k2,2n | tail -1 | tr ' ' '_')
  [ -z "$ck" ] && continue
  d=$WS/experiments/head-arch/trunk-$r
  mkdir -p "$d"
  ./target/release/head_arch_screen --config "$WS/runs/$r/config.yaml" \
    --out-dir "$d" --sample 4000 \
    --checkpoint "$WS/runs/$r/checkpoints/epoch_$ck" >/dev/null 2>&1 \
    && ./experiments/head_arch/analyze.py "$d" 2>/dev/null \
       | grep -A 3 "TRUNK RANK" | tee -a "$OUT"
done
say ""

say "== n(d) ANIMATION =="
for r in "${runs[@]}"; do
  say "  $r"
  ./experiments/head_arch/../stage_roughness/animate_n_of_d.py "$WS/runs/$r" \
    --water-year 1996 --traces 2>&1 | grep -E "gamma|n\(d\)|ratio|wrote|!" | tee -a "$OUT"
  ./experiments/stage_roughness/animate_n_of_d.py "$WS/runs/$r" \
    --water-year 1996 2>&1 | grep -E "wrote|!" | tee -a "$OUT"
done

say ""
say "report: $OUT"

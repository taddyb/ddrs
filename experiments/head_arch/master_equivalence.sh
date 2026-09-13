#!/usr/bin/env bash
# Is head_shared_linear's +0.008 NSE an artifact of the kan_head restructure?
#
# The run used master + commit 1bbc3b7 (parameter_groups / input_layer_kan /
# output_layer_kan / the legacy checkpoint loader). It sets none of those keys,
# so it should take the default path, but "should" is an argument. This runs the
# IDENTICAL config on both binaries for two micro-batches and compares the
# per-micro-batch losses. Identical losses means identical training.
set -u

BRANCH=/home/tbindas/projects/ddrs/.claude/worktrees/kan-split-heads/target/release/ddrs
MASTER=/home/tbindas/.claude/jobs/d68b0903/tmp/master-tree/target/release/ddrs
CFG=/home/tbindas/projects/ddrs/.claude/worktrees/kan-split-heads/config/experiments/head_shared_linear.yaml
TMP=/home/tbindas/.claude/jobs/d68b0903/tmp/equiv
rm -rf "$TMP"; mkdir -p "$TMP/a" "$TMP/b"

run_one() {
  local bin=$1 ws=$2 tag=$3
  "$bin" --config "$CFG" --workspace "$ws" run --workflow train \
    --max-mini-batches 2 --backend cpu 2>&1 \
    | grep -E "micro [0-9]+/[0-9]+|mb=" \
    | sed 's/^\[[^]]*\] *//' > "$TMP/$tag.txt"
  echo "--- $tag ($bin)"
  cat "$TMP/$tag.txt"
}

run_one "$MASTER" "$TMP/a" master
run_one "$BRANCH" "$TMP/b" branch

echo
if diff -q "$TMP/master.txt" "$TMP/branch.txt" >/dev/null 2>&1; then
  echo "IDENTICAL — the kan_head restructure is numerically inert for this config."
else
  echo "DIFFERENT — losses diverge; the +0.008 cannot be attributed to p_spatial alone."
  diff "$TMP/master.txt" "$TMP/branch.txt" | head -20
fi

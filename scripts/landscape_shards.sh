#!/usr/bin/env bash
# Launch K parallel landscape-census shards as transient systemd --user units
# so they survive this session exiting.
#
# Usage: scripts/landscape_shards.sh <bundle> <K> [backend]
#
# Starts K processes:
#   target/release/ddrs --workspace .ddrs experiment <bundle> --backend <backend> --shard i/K
# for i in 0..K-1, each as its own `systemd-run --user --unit=ddrs-<bundle>-shard-i
# --collect --quiet` unit. Each shard's stdout+stderr is logged to
# /home/tbindas/.claude/jobs/d68b0903/tmp/<bundle>.shard-i.log; once a shard's
# run directory appears in its log, this script prints it and moves on --
# it does not wait for the shard to finish (each covers ~1/K of the
# population at ~16 s/gauge on a single core).
set -euo pipefail

bundle="${1:?usage: landscape_shards.sh <bundle> <K> [backend]}"
K="${2:?usage: landscape_shards.sh <bundle> <K> [backend]}"
backend="${3:-cpu}"

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ddrs_bin="${repo_root}/target/release/ddrs"
log_dir="/home/tbindas/.claude/jobs/d68b0903/tmp"
mkdir -p "$log_dir"

if [[ ! -x "$ddrs_bin" ]]; then
    echo "error: ${ddrs_bin} not found or not executable; run \`cargo build --release --bin ddrs\` first" >&2
    exit 1
fi
if [[ ! "$K" =~ ^[0-9]+$ ]] || (( K < 1 )); then
    echo "error: K must be a positive integer, got \`${K}\`" >&2
    exit 1
fi

echo "launching ${K} shard(s) of bundle '${bundle}' (backend ${backend})..." >&2
for ((i = 0; i < K; i++)); do
    unit="ddrs-${bundle}-shard-${i}"
    log="${log_dir}/${bundle}.shard-${i}.log"
    : >"$log"
    systemd-run --user --unit="$unit" --collect --quiet \
        --working-directory="$repo_root" \
        -- bash -c "exec '${ddrs_bin}' --workspace .ddrs experiment '${bundle}' --backend '${backend}' --shard '${i}/${K}' >'${log}' 2>&1"
    echo "  shard ${i}/${K}: unit ${unit}, log ${log}" >&2
done

echo "waiting for each shard's run directory to appear in its log (up to 60s)..." >&2
for ((i = 0; i < K; i++)); do
    log="${log_dir}/${bundle}.shard-${i}.log"
    dir=""
    for _ in $(seq 1 60); do
        dir="$(grep -m1 -oP "(?<=experiment \`${bundle}\` → ).*" "$log" 2>/dev/null || true)"
        [[ -n "$dir" ]] && break
        sleep 1
    done
    if [[ -n "$dir" ]]; then
        echo "shard ${i}/${K}: ${dir}"
    else
        echo "shard ${i}/${K}: run directory not seen in ${log} after 60s -- check the log (systemctl --user status ddrs-${bundle}-shard-${i})" >&2
    fi
done

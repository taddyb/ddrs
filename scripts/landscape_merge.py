#!/usr/bin/env python3
"""Merge `ddrs experiment` landscape-census shard run directories into one
directory that census.py, conus_map.py, and plots.py can read like an
ordinary (unsharded) run.

Each input is a run directory produced by one `--shard I/K` invocation
against the same bundle (see scripts/landscape_shards.sh) -- same arms, same
experiment.yaml, a disjoint slice of the gauge population. This:

  - concatenates each arm's summary.csv (one header)
  - hard-links (falls back to copying across filesystems) each arm's
    gauges/*.nc into the merged directory
  - writes a merged gauges.csv (union of all shards' rows)
  - writes a merged manifest.json: the first shard's manifest, with `shard`
    cleared and a `shards` list recording the source directories

Only stdlib -- no ddr venv needed.

Usage:
    scripts/landscape_merge.py <run-dir-1> [<run-dir-2> ...] --out <merged-dir>
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import shutil
from pathlib import Path


def load_manifest(run_dir: Path) -> dict:
    path = run_dir / "manifest.json"
    return json.loads(path.read_text())


def link_or_copy(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists():
        dst.unlink()
    try:
        os.link(src, dst)
    except OSError:
        shutil.copy2(src, dst)


def merge_gauges_csv(run_dirs: list[Path], out_dir: Path) -> int:
    header: list[str] | None = None
    rows: list[list[str]] = []
    seen: set[str] = set()
    for d in run_dirs:
        path = d / "gauges.csv"
        with path.open(newline="") as f:
            reader = csv.reader(f)
            h = next(reader)
            if header is None:
                header = h
            elif h != header:
                raise SystemExit(f"{path}: header {h} does not match {header} from {run_dirs[0]}")
            for row in reader:
                staid = row[0]
                if staid in seen:
                    print(f"warning: staid {staid!r} appears in more than one shard's gauges.csv ({path}); keeping the first occurrence")
                    continue
                seen.add(staid)
                rows.append(row)
    assert header is not None
    with (out_dir / "gauges.csv").open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(header)
        writer.writerows(rows)
    return len(rows)


def merge_arm(arm: str, run_dirs: list[Path], out_dir: Path) -> tuple[int, int]:
    header: list[str] | None = None
    rows: list[list[str]] = []
    n_nc = 0
    for d in run_dirs:
        summary_path = d / arm / "summary.csv"
        if summary_path.exists():
            with summary_path.open(newline="") as f:
                reader = csv.reader(f)
                h = next(reader)
                if header is None:
                    header = h
                elif h != header:
                    raise SystemExit(f"{summary_path}: header {h} does not match {header}")
                rows.extend(reader)
        gauges_dir = d / arm / "gauges"
        if gauges_dir.is_dir():
            for nc in sorted(gauges_dir.glob("*.nc")):
                link_or_copy(nc, out_dir / arm / "gauges" / nc.name)
                n_nc += 1
    if header is not None:
        (out_dir / arm).mkdir(parents=True, exist_ok=True)
        with (out_dir / arm / "summary.csv").open("w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(header)
            writer.writerows(rows)
    return len(rows), n_nc


def merge(run_dirs: list[Path], out_dir: Path) -> None:
    if not run_dirs:
        raise SystemExit("no shard run directories given")
    out_dir.mkdir(parents=True, exist_ok=True)

    manifests = [load_manifest(d) for d in run_dirs]
    base = manifests[0]
    arm_names = [a["name"] for a in base["arms"]]
    for m, d in zip(manifests[1:], run_dirs[1:]):
        names = [a["name"] for a in m["arms"]]
        if names != arm_names:
            raise SystemExit(f"{d}: arm set {names} does not match {run_dirs[0]}'s {arm_names}")

    n_gauges = merge_gauges_csv(run_dirs, out_dir)
    print(f"gauges.csv: {n_gauges} gauge(s) from {len(run_dirs)} shard(s)")

    for arm in arm_names:
        n_rows, n_nc = merge_arm(arm, run_dirs, out_dir)
        print(f"arm {arm}: {n_rows} summary row(s), {n_nc} gauge netCDF(s)")

    merged_manifest = dict(base)
    merged_manifest["shard"] = None
    merged_manifest["shards"] = [str(d.resolve()) for d in run_dirs]
    merged_manifest["status"] = "merged"
    (out_dir / "manifest.json").write_text(json.dumps(merged_manifest, indent=2))
    print(f"manifest.json: merged from {len(run_dirs)} shard(s) -> {out_dir}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("run_dirs", nargs="+", type=Path, help="shard run directories (each a `ddrs experiment` output)")
    ap.add_argument("--out", required=True, type=Path, help="merged output directory")
    args = ap.parse_args()
    merge([d.resolve() for d in args.run_dirs], args.out.resolve())


if __name__ == "__main__":
    main()

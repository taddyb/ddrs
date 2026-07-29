"""Consensus geometry for the synthetic-n recoverability experiment
(docs/superpowers/specs/2026-07-22-synthetic-n-recoverability-design.md).

Runs `dump_parameters` against the 4 already-converged real-Q'-source
checkpoints from this campaign, then computes the per-COMID MEDIAN
q_spatial/p_spatial across them — the "most common trained value trend",
used as the FIXED geometry truth for every synthetic-n student.

Run from ddrs-py's venv:
    cd ddrs-py && uv run python ../scripts/synthetic_n_consensus_geometry.py
"""
from __future__ import annotations

import subprocess
from pathlib import Path

import numpy as np
import xarray as xr

REPO = Path(__file__).resolve().parent.parent
OUT_DIR = REPO / "output/synthetic_n"
OUT_DIR.mkdir(parents=True, exist_ok=True)

CHECKPOINTS = [
    {
        "label": "aorc2f_distributed",
        "config": REPO / ".ddrs/runs/2026-07-16T02-22-14Z-train-and-test/config.yaml",
        "checkpoint": REPO / ".ddrs/runs/2026-07-16T02-22-14Z-train-and-test/checkpoints/epoch_5_mb_35/head",
    },
    {
        "label": "aorc2f_lumped",
        "config": REPO / ".ddrs/runs/2026-07-16T02-23-20Z-train-and-test/config.yaml",
        "checkpoint": REPO / ".ddrs/runs/2026-07-16T02-23-20Z-train-and-test/checkpoints/epoch_5_mb_35/head",
    },
    {
        "label": "daily_lstm",
        "config": REPO / ".ddrs/runs/2026-07-16T11-31-50Z-train-and-test/config.yaml",
        "checkpoint": REPO / ".ddrs/runs/2026-07-16T11-31-50Z-train-and-test/checkpoints/epoch_5_mb_35/head",
    },
    {
        "label": "hourly_lstm",
        "config": REPO / ".ddrs/runs/2026-07-16T11-31-52Z-train-and-test/config.yaml",
        "checkpoint": REPO / ".ddrs/runs/2026-07-16T11-31-52Z-train-and-test/checkpoints/epoch_5_mb_35/head",
    },
]


def _is_intact(path: Path) -> bool:
    """A prior run killed mid-write (OOM, disk full, Ctrl-C) can leave a
    `.nc` file that exists but is truncated or missing variables — silently
    reusing it would poison the "fixed geometry truth" every downstream
    synthetic-n student depends on. Require both q_spatial and p_spatial to
    actually be readable before trusting an existing dump."""
    try:
        with xr.open_dataset(path) as ds:
            return "q_spatial" in ds and "p_spatial" in ds
    except Exception:
        return False


def dump_one(ckpt: dict) -> Path:
    out = OUT_DIR / f"{ckpt['label']}_kan_parameters.nc"
    if out.exists():
        if _is_intact(out):
            print(f"{out} already exists, skipping dump_parameters re-run")
            return out
        print(f"{out} exists but is truncated/corrupt — removing and re-running dump_parameters")
        out.unlink()
    cmd = [
        "cargo", "run", "--release", "--bin", "dump_parameters", "--",
        "--backend", "cpu",
        "--config", str(ckpt["config"]),
        "--checkpoint", str(ckpt["checkpoint"]),
        "--output", str(out),
    ]
    print("running:", " ".join(cmd))
    subprocess.run(cmd, cwd=REPO, check=True)
    return out


def main() -> None:
    dumps = [dump_one(c) for c in CHECKPOINTS]

    datasets = [xr.open_dataset(d) for d in dumps]
    comids_0 = datasets[0]["COMID"].values
    for d, ds in zip(dumps, datasets):
        if not np.array_equal(np.sort(ds["COMID"].values), np.sort(comids_0)):
            raise SystemExit(
                f"{d}: COMID set differs from {dumps[0]} — cannot take a per-COMID "
                "median across checkpoints with different networks"
            )

    # Re-index every dump to dumps[0]'s COMID order before stacking, since
    # dump_parameters row order isn't guaranteed identical across runs.
    order = comids_0
    q_stack = np.stack([ds.set_index(COMID="COMID").sel(COMID=order)["q_spatial"].values for ds in datasets])
    p_stack = np.stack([ds.set_index(COMID="COMID").sel(COMID=order)["p_spatial"].values for ds in datasets])

    q_median = np.median(q_stack, axis=0).astype(np.float32)
    p_median = np.median(p_stack, axis=0).astype(np.float32)

    out = OUT_DIR / "consensus_geometry.nc"
    xr.Dataset(
        {
            "q_spatial": ("COMID", q_median),
            "p_spatial": ("COMID", p_median),
        },
        coords={"COMID": order},
    ).to_netcdf(out)
    print(f"consensus geometry ({len(order)} reaches) -> {out}")


if __name__ == "__main__":
    main()

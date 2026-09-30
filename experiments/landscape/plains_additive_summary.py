#!/usr/bin/env python
"""Read-out of the additive d_gw sweep on the plains gauges
(experiments/landscape-plains-dgw-additive) against the multiplicative census
(experiments/landscape-huc-leakance) at the same gauges.

Per gauge, on the K_D x d_gw plane through the trained point:
  - d_gw offset at the loss minimum, in metres (alpha * additive_scale)
  - best NSE reachable on the plane with n fixed, vs the census value
  - loss response on each side of the trained water table: the largest
    fractional loss drop with d_gw raised (toward gaining) and with d_gw
    lowered (toward more losing), K_D free

Usage:
    python plains_additive_summary.py <additive-run-glob> <census-run-glob> --out <dir>
"""
from __future__ import annotations

import argparse
import glob
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr


def read(run_glob: str) -> pd.DataFrame:
    rows = []
    for p in sorted(glob.glob(f"{run_glob}/*/gauges/*.nc")):
        ds = xr.open_dataset(p, decode_timedelta=False)
        names = ds.attrs["plane_names"].split(",")
        k = names.index("kd-dgw")
        scale = [float(s) for s in ds.attrs.get("additive_scale", "0,0,0").split(",")]
        L = ds["grid_loss"].values[k].astype(float)
        N = ds["grid_nse"].values[k].astype(float)
        cl = ds["grid_clamped"].values[k].astype(float)
        L = np.where(cl > 0.05, np.nan, L)
        N = np.where(cl > 0.05, np.nan, N)
        a = ds["grid_axis_a"].values[k].astype(float)  # K_D slot (log mult)
        b = ds["grid_axis_b"].values[k].astype(float)  # d_gw slot
        loss0 = float(ds["loss0"])
        i, j = np.unravel_index(np.nanargmin(L), L.shape)
        raised = b > 0
        lowered = b < 0
        drop = 1.0 - L / loss0  # fractional loss drop, positive = better
        rows.append(dict(
            staid=ds.attrs["staid"], n_reach=int(ds.attrs["n_reach"]), nse0=float(ds["nse0"]),
            additive=scale[2] > 0,
            dgw_at_min=float(b[j]) * (scale[2] if scale[2] > 0 else 1.0),
            kd_at_min_logmult=float(a[i]),
            nse_best_plane=float(np.nanmax(N)),
            drop_raised=float(np.nanmax(drop[:, raised])) if raised.any() else np.nan,
            drop_lowered=float(np.nanmax(drop[:, lowered])) if lowered.any() else np.nan,
            plateau=float(np.nanmean(np.abs(L - loss0) <= 0.05 * loss0)),
            hess_dgw_trained=float(ds["hess0"].values[2, 2]),
        ))
        ds.close()
    return pd.DataFrame(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("additive_glob")
    ap.add_argument("census_glob")
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    add = read(args.additive_glob)
    cen = read(args.census_glob)
    cen = cen[cen.staid.isin(add.staid)]
    df = add.merge(cen[["staid", "nse_best_plane", "plateau", "dgw_at_min"]], on="staid", suffixes=("", "_census"))
    df["dnse_leak_additive"] = df.nse_best_plane - df.nse0
    df["dnse_leak_census"] = df.nse_best_plane_census - df.nse0
    df["prefers"] = np.where(df.drop_raised > df.drop_lowered, "raise d_gw (less losing / gaining)", "lower d_gw (more losing)")
    df.to_csv(args.out / "plains_additive_gauges.csv", index=False)
    cols = ["staid", "n_reach", "nse0", "dgw_at_min", "drop_raised", "drop_lowered", "prefers",
            "dnse_leak_additive", "dnse_leak_census", "plateau", "plateau_census"]
    lines = ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
    for r in df.sort_values("nse0").itertuples():
        vals = [getattr(r, c) for c in cols]
        lines.append("| " + " | ".join(v if isinstance(v, str) else (f"{v:.3g}" if isinstance(v, float) else str(v)) for v in vals) + " |")
    (args.out / "PLAINS_ADDITIVE_TABLE.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))
    print(f"\n{len(df)} gauges; prefer raising d_gw: {(df.drop_raised > df.drop_lowered).sum()}, lowering: {(df.drop_raised <= df.drop_lowered).sum()}")
    print(f"median leak-only dNSE: additive {df.dnse_leak_additive.median():+.3f} vs census {df.dnse_leak_census.median():+.3f}")
    print(f"median plateau: additive {df.plateau.median():.2f} vs census {df.plateau_census.median():.2f}")
    print(f"median d_gw offset at minimum: {df.dgw_at_min.median():+.2f} m")


if __name__ == "__main__":
    main()

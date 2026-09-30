#!/usr/bin/env python
"""Cross-gauge comparison of the JRB nested-cascade landscapes.

Reads every per-gauge netCDF written by the `landscape-jrb-all` bundle and
answers one question: does the daily hydrograph constrain the channel
parameters equally well at every gauged location in one basin, or does the
constraint degrade as the contributing network grows?

The per-gauge 3-D pictures come from `plot3d.py`; this script is the table and
the two summary panels that make the differences comparable.

Usage:
    ~/projects/ddr/.venv/bin/python experiments/landscape/jrb_compare.py \
        <dir-with-gauge-ncs> --out <dir>
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import xarray as xr  # noqa: E402

# Drainage area (km2) and station name, from gages_3000.csv, for labelling only.
META = {
    "01564500": (445.6, "Aughwick Ck nr Three Springs"),
    "01558000": (572.5, "Little Juniata R at Spruce Ck"),
    "01556000": (749.6, "Frankstown Br at Williamsburg"),
    "01562000": (1944.0, "Raystown Br at Saxton"),
    "01559000": (2111.3, "Juniata R at Huntingdon"),
    "01563200": (2485.1, "Raystown Br bl Raystown Dam"),
    "01563500": (5261.9, "Juniata R at Mapleton Depot"),
    "01567000": (8657.3, "Juniata R at Newport"),
}
ALPHA_NAMES = ["n", "p", "q"]


def collect(nc_dir: Path) -> list[dict]:
    rows = []
    for path in sorted(nc_dir.glob("*.nc")):
        staid = path.stem
        ds = xr.open_dataset(path, decode_timedelta=False)
        active = np.asarray(ds["active"].values).astype(bool)
        eig = np.asarray(ds["eigval_star"].values, dtype=float)
        finite = eig[np.isfinite(eig)]
        # eigenvalues are descending; the fixed component is a trailing NaN
        stiff = finite[0] if finite.size else np.nan
        sloppy = finite[-1] if finite.size else np.nan
        hw = np.asarray(ds["half_width"].values, dtype=float)  # (tol, k)
        tols = np.asarray(ds["tolerances"].values, dtype=float)
        i05 = int(np.argmin(np.abs(tols - 0.05)))
        da, name = META.get(staid, (np.nan, ""))
        rows.append(
            dict(
                staid=staid,
                name=name,
                da_km2=da,
                n_reach=int(ds.sizes["reach"]),
                n_active=int(active.sum()),
                nse0=float(ds["nse0"].values),
                nse_star=float(ds["nse_star"].values),
                kge0=float(ds["kge0"].values),
                loss0=float(ds["loss0"].values),
                loss_star=float(ds["loss_star"].values),
                alpha_star=np.asarray(ds["alpha_star"].values, dtype=float),
                grad0=np.asarray(ds["grad0"].values, dtype=float),
                eig=eig,
                stiff=stiff,
                sloppy=sloppy,
                cond=stiff / sloppy if sloppy and np.isfinite(sloppy) and sloppy > 0 else np.nan,
                hw05=hw[i05],
                coord_trained=np.asarray(ds["coord_trained"].values, dtype=float),
                v_stiff=np.asarray(ds["eigvec_star"].values, dtype=float)[:, 0],
                v_sloppy=np.asarray(ds["eigvec_star"].values, dtype=float)[:, 2],
            )
        )
        ds.close()
    rows.sort(key=lambda r: r["n_reach"])
    return rows


def write_table(rows: list[dict], out: Path) -> None:
    lines = [
        "| STAID | station | DA km2 | reaches | NSE trained | NSE at optimum | ΔNSE | "
        "α* (n, p, q) | λ stiff | λ sloppy | λs/λn | 5 % half-width (stiff, sloppy) |",
        "|---|---|---:|---:|---:|---:|---:|---|---:|---:|---:|---|",
    ]
    for r in rows:
        a = ", ".join(f"{v:+.2f}" for v in r["alpha_star"])
        hw = np.asarray(r["hw05"], dtype=float)
        hwf = hw[np.isfinite(hw)]
        hws = f"{hwf[0]:.2f}, {hwf[-1]:.2f}" if hwf.size else "n/a"
        lines.append(
            f"| {r['staid']} | {r['name']} | {r['da_km2']:.0f} | {r['n_reach']} | "
            f"{r['nse0']:.3f} | {r['nse_star']:.3f} | {r['nse_star'] - r['nse0']:+.3f} | "
            f"{a} | {r['stiff']:.3g} | {r['sloppy']:.3g} | {r['cond']:.1f} | {hws} |"
        )
    out.write_text("\n".join(lines) + "\n")


def figure(rows: list[dict], out_png: Path) -> None:
    n = np.array([r["n_reach"] for r in rows], dtype=float)
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.6))

    ax = axes[0]
    for k, (marker, lab) in enumerate([("o", "λ1 (stiff)"), ("s", "λ2"), ("^", "λ3 (sloppy)")]):
        y = np.array([r["eig"][k] for r in rows], dtype=float)
        ax.plot(n, np.abs(y), marker + "-", label=lab)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("reaches upstream of the gauge")
    ax.set_ylabel("|eigenvalue| of H(α*)")
    ax.set_title("Curvature spectrum:\nhow many directions the gauge constrains")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3, which="both")

    ax = axes[1]
    ax.plot(n, [r["cond"] for r in rows], "o-", color="firebrick")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("reaches upstream of the gauge")
    ax.set_ylabel("λ_stiff / λ_sloppy")
    ax.set_title("Sloppiness:\nlarger means a flatter valley")
    ax.grid(alpha=0.3, which="both")

    ax = axes[2]
    width = 0.11
    idx = np.arange(3)
    for j, r in enumerate(rows):
        v1 = np.asarray(r["v_stiff"], dtype=float)
        if v1[0] < 0:            # sign of an eigenvector is arbitrary; fix on n > 0
            v1 = -v1
        ax.bar(idx + (j - len(rows) / 2) * width, v1, width, label=r["staid"][-5:])
    ax.axhline(0, color="k", lw=0.8)
    ax.set_xticks(idx)
    ax.set_xticklabels(["n", "p", "q"])
    ax.set_ylabel("loading of the stiff eigenvector")
    ax.set_title("The one constrained direction,\nat every gauge in the basin")
    ax.legend(fontsize=6, ncol=2)
    ax.grid(alpha=0.3, axis="y")

    for ax, rowlab in zip(axes, [rows, rows, rows]):
        for r in rowlab:
            ax.annotate(r["staid"][-5:], (r["n_reach"], ax.get_ylim()[0]), fontsize=6, alpha=0)
    fig.suptitle(
        "Juniata River Basin: per-gauge loss landscape as the contributing network grows "
        "(arm uh-seed42, WY2000, nse-batch)",
        fontsize=11,
    )
    fig.tight_layout()
    fig.savefig(out_png, dpi=150)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("nc_dir", type=Path, help="directory holding the per-gauge .nc files")
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    rows = collect(args.nc_dir)
    if not rows:
        raise SystemExit(f"no .nc files in {args.nc_dir}")
    write_table(rows, args.out / "JRB_LANDSCAPE_TABLE.md")
    figure(rows, args.out / "jrb_landscape_summary.png")
    print(f"{len(rows)} gauges -> {args.out}")
    print((args.out / "JRB_LANDSCAPE_TABLE.md").read_text())


if __name__ == "__main__":
    main()

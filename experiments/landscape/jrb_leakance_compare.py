#!/usr/bin/env python
"""Cross-gauge summary of the JRB cascade in the leakance frame (n, K_D, d_gw).

Three questions, one panel each: does d_gw's curvature at the local optimum
grow with network size like a summed mass term; is d_gw orthogonal to n at
the trained point; does K_D run to the sweep wall everywhere.

Usage:
    ~/projects/ddr/.venv/bin/python experiments/landscape/jrb_leakance_compare.py \
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

NR = {"01557500": 3, "01564500": 9, "01558000": 11, "01560000": 13, "01556000": 23,
      "01559000": 47, "01562000": 53, "01563200": 63, "01563500": 127, "01567000": 213}
BOX = 2.3025851


def collect(nc_dir: Path):
    rows = []
    for p in sorted(nc_dir.glob("*.nc")):
        s = p.stem
        ds = xr.open_dataset(p, decode_timedelta=False)
        H0 = ds["hess0"].values.astype(float)
        Hs = ds["hess_star"].values.astype(float)
        d = np.sqrt(np.abs(np.diag(H0)))
        C = H0 / np.outer(d, d)
        rows.append(dict(
            staid=s, n=NR[s], nse0=float(ds["nse0"]), nse_star=float(ds["nse_star"]),
            a=ds["alpha_star"].values.astype(float), h0=np.diag(H0), hs=np.diag(Hs),
            eig=ds["eigval_star"].values.astype(float), c_ndgw=C[0, 2],
        ))
        ds.close()
    rows.sort(key=lambda r: r["n"])
    return rows


def table(rows, out: Path):
    L = ["| STAID | reaches | NSE trained | NSE opt | α* n | α* K_D | α* d_gw | H n (trained→opt) | H K_D (trained→opt) | H d_gw (trained→opt) | corr(n, d_gw) |",
         "|---|---:|---:|---:|---:|---:|---:|---|---|---|---:|"]
    for r in rows:
        f = lambda a, b: f"{a:.1e} → {b:.1e}"
        wall = " (wall)" if abs(abs(r["a"][1]) - BOX) < 0.01 else ""
        L.append(f"| {r['staid']} | {r['n']} | {r['nse0']:.3f} | {r['nse_star']:.3f} | {r['a'][0]:+.2f} | {r['a'][1]:+.2f}{wall} | {r['a'][2]:+.2f} | "
                 f"{f(r['h0'][0], r['hs'][0])} | {f(r['h0'][1], r['hs'][1])} | {f(r['h0'][2], r['hs'][2])} | {r['c_ndgw']:+.3f} |")
    out.write_text("\n".join(L) + "\n")


def figure(rows, out_png: Path):
    n = np.array([r["n"] for r in rows], float)
    fig, ax = plt.subplots(1, 3, figsize=(15, 4.6))

    a = ax[0]
    a.plot(n, [abs(r["hs"][0]) for r in rows], "o-", label="n, at optimum")
    a.plot(n, [abs(r["hs"][2]) for r in rows], "s-", label="d_gw, at optimum")
    a.plot(n, [abs(r["h0"][2]) for r in rows], "s--", color="C1", alpha=0.5, label="d_gw, at trained point")
    a.set_xscale("log"); a.set_yscale("log")
    a.set_xlabel("reaches upstream of the gauge"); a.set_ylabel("|diagonal curvature|")
    a.set_title("d_gw curvature grows with the network,\nbut only at the local optimum")
    a.legend(fontsize=8); a.grid(alpha=0.3, which="both")

    a = ax[1]
    c = np.array([r["c_ndgw"] for r in rows])
    a.axhline(0, color="k", lw=0.8)
    a.plot(n, c, "o", color="C2")
    for r in rows:
        a.annotate(r["staid"][-5:], (r["n"], r["c_ndgw"]), fontsize=6, xytext=(3, 3), textcoords="offset points")
    a.set_xscale("log"); a.set_ylim(-1, 1)
    a.set_xlabel("reaches upstream of the gauge"); a.set_ylabel("corr(n, d_gw) at trained point")
    a.set_title("Is the groundwater offset orthogonal\nto roughness?")
    a.grid(alpha=0.3)

    a = ax[2]
    akd = np.array([r["a"][1] for r in rows])
    hkd = np.array([r["hs"][1] for r in rows])
    col = ["firebrick" if h < 0 else "C0" for h in hkd]
    a.axhline(BOX, color="grey", ls="--", lw=1); a.axhline(-BOX, color="grey", ls="--", lw=1)
    a.axhline(0, color="k", lw=0.8)
    a.scatter(n, akd, c=col, s=50, zorder=3)
    a.set_xscale("log"); a.set_ylim(-2.7, 2.7)
    a.set_xlabel("reaches upstream of the gauge"); a.set_ylabel("α* for K_D (log multiplier)")
    a.set_title("Where the optimizer sends conductance\n(red: negative curvature there)")
    a.text(3.2, BOX + 0.08, "sweep ceiling ×10", fontsize=7, color="grey")
    a.text(3.2, -BOX - 0.25, "sweep floor ÷10", fontsize=7, color="grey")
    a.grid(alpha=0.3)

    fig.suptitle("Juniata River Basin in the leakance frame (n, K_D, d_gw): arm twoway-ep30, WY2000, nse-batch", fontsize=11)
    fig.tight_layout(); fig.savefig(out_png, dpi=150)


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("nc_dir", type=Path); ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args(); a.out.mkdir(parents=True, exist_ok=True)
    rows = collect(a.nc_dir)
    table(rows, a.out / "JRB_LEAKANCE_TABLE.md"); figure(rows, a.out / "jrb_leakance_summary.png")
    print(f"{len(rows)} gauges -> {a.out}")


if __name__ == "__main__":
    main()

#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.11"
# dependencies = ["numpy", "pandas", "netCDF4", "zarr>=3", "scipy", "matplotlib", "geopandas", "pyogrio"]
# ///
"""Do dams explain where the learned roughness is high?

Hypothesis (user, 2026-09-13, looking at the low/high-flow roughness maps of
the n_0 + gamma arm): the slow, rough parts of the Mississippi basin in the
Midwest coincide with dams. Mechanism: a reach impounded behind a dam runs deep
and slow at every discharge, and a model with prescribed channel shape can only
express that through a large n_0.

Test: reaches hosting a HydroLAKES/GRanD reservoir outlet (DDR's
merit_reservoir_params.csv, 2,178 COMIDs), plus their neighbours 1-3 hops
upstream (impounded) and downstream (regulated), against every other live reach
in the same log10-drainage-area bin. Reports the per-bin median n_0 by group,
the pooled residual (n_0 minus the bin median of unaffected reaches), and a
Mann-Whitney p per bin, CONUS-wide and for the Midwest box; draws a figure.

    experiments/stage_roughness/dam_roughness.py <run-id> [<run-id> ...]
"""
import re
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
from netCDF4 import Dataset
from scipy.stats import mannwhitneyu

RUNS = Path("/home/tbindas/projects/ddrs/.ddrs/runs")
DDR = Path("/home/tbindas/projects/ddr/data")
FABRIC = DDR / "merit/riv_pfaf_7_MERIT_Hydro_v07_Basins_v01_bugfix1.shp"
MIDWEST = (-98.0, 36.0, -84.0, 47.0)  # lon_min, lat_min, lon_max, lat_max: Upper Mississippi / Ohio-Missouri confluence region
HOPS = 3


def hop_sets(adj_path, seeds):
    g = __import__("zarr").open_group(adj_path, mode="r")
    order = np.asarray(g["order"][:])
    rows, cols = np.asarray(g["indices_0"][:]), np.asarray(g["indices_1"][:])
    pos = {int(c): i for i, c in enumerate(order)}
    up, down = defaultdict(list), {}
    for r, c in zip(rows, cols):  # row = downstream, col = upstream
        up[r].append(c)
        down[c] = r
    seed_idx = [pos[c] for c in seeds if int(c) in pos]
    up_hops, down_hops = {}, {}
    for s in seed_idx:
        frontier = [s]
        for h in range(1, HOPS + 1):
            nxt = []
            for x in frontier:
                for y in up[x]:
                    if y not in up_hops and y not in seed_idx:
                        up_hops[y] = h
                    nxt.append(y)
            frontier = nxt
        x = s
        for h in range(1, HOPS + 1):
            if x not in down:
                break
            x = down[x]
            if x not in down_hops and x not in seed_idx:
                down_hops[x] = h
    return order, up_hops, down_hops


def main() -> int:
    res = pd.read_csv(DDR / "merit_reservoir_params.csv")
    lakes = pd.read_csv(DDR / "hydrolakes_rfc_da.csv")
    # join on the (exact) lake area to recover the lake type and GRanD id
    lk = lakes.drop_duplicates("LkArea").set_index("LkArea")
    res["Lake_type"] = lk["Lake_type"].reindex(res["lake_area_m2"]).values
    res["Grand_id"] = lk["Grand_id"].reindex(res["lake_area_m2"]).values
    dam_comids = res["COMID"].astype(np.int64).values
    print(f"{len(res)} reservoir reaches; lake types: {res['Lake_type'].value_counts(dropna=False).to_dict()} "
          "(HydroLAKES: 1 natural, 2 reservoir, 3 natural lake with dam control)")

    for rid in sys.argv[1:]:
        run = RUNS / rid
        cfg = (run / "config.yaml").read_text()
        ds = Dataset(run / "plot" / "kan_parameters.nc")
        comid = np.asarray(ds["COMID"][:], dtype=np.int64)
        n0 = np.asarray(ds["n"][:], dtype=np.float64)
        gamma = np.asarray(ds["gamma"][:], dtype=np.float64) if "gamma" in ds.variables else np.zeros_like(n0)
        at = Dataset(re.search(r"^\s+attributes:\s*(\S+)", cfg, re.M).group(1))
        ac = np.asarray(at["COMID"][:], dtype=np.int64)
        ua = np.asarray(at["log10_uparea"][:], dtype=np.float64)
        la = pd.Series(ua, index=ac).reindex(comid).values
        adj = re.search(r"^\s+conus_adjacency:\s*(\S+)", cfg, re.M).group(1)
        order, up_h, down_h = hop_sets(adj, dam_comids)
        pos = {int(c): i for i, c in enumerate(order)}
        idx_in_order = np.array([pos.get(int(c), -1) for c in comid])
        group = np.full(comid.size, "other", dtype=object)
        for k, i in enumerate(idx_in_order):
            if i in up_h:
                group[k] = f"upstream {up_h[i]} hop"
            elif i in down_h:
                group[k] = f"downstream {down_h[i]} hop"
        group[np.isin(comid, dam_comids)] = "dam reach"

        df = pd.DataFrame({"comid": comid, "n0": n0, "gamma": gamma, "la": la, "group": group})
        df = df[np.isfinite(df.la)]
        # coordinates for the region filter and the map
        import pyogrio
        gdf = pyogrio.read_dataframe(str(FABRIC), columns=["COMID"]).set_index("COMID")
        pts = gdf.geometry.representative_point()
        df["lon"] = pts.x.reindex(df.comid).values
        df["lat"] = pts.y.reindex(df.comid).values
        lon0, lat0, lon1, lat1 = MIDWEST
        df["midwest"] = (df.lon >= lon0) & (df.lon <= lon1) & (df.lat >= lat0) & (df.lat <= lat1)

        df["bin"] = np.floor(df.la / 0.25) * 0.25
        base = df[df.group == "other"].groupby("bin")["n0"].median()
        df["resid"] = df.n0 - base.reindex(df.bin).values
        base_g = df[df.group == "other"].groupby("bin")["gamma"].median()
        df["gresid"] = df.gamma - base_g.reindex(df.bin).values
        print(f"\n##### {rid}")
        for label, sub in [("CONUS", df), ("Midwest box", df[df.midwest])]:
            print(f"\n{label}: n_0 residual vs same-size unaffected reaches (median [IQR]), n, and Mann-Whitney p vs 'other'")
            other = sub[sub.group == "other"]
            for gname in ["dam reach"] + [f"upstream {h} hop" for h in range(1, HOPS + 1)] + [f"downstream {h} hop" for h in range(1, HOPS + 1)] + ["other"]:
                s = sub[sub.group == gname]
                if s.empty:
                    continue
                p = mannwhitneyu(s.resid.dropna(), other.resid.dropna()).pvalue if gname != "other" else float("nan")
                print(f"  {gname:>18}: n0 {s.n0.median():.4f}  resid {s.resid.median():+.4f} [{s.resid.quantile(.25):+.4f}, {s.resid.quantile(.75):+.4f}]  "
                      f"gamma resid {s.gresid.median():+.4f}  n={len(s):>7,}  p={p:.1e}")
            print(f"  by size, dam reaches vs other (median n_0):")
            for b, s in sub[sub.group == "dam reach"].groupby("bin"):
                o = other[other.bin == b]
                if len(s) >= 15 and len(o) >= 50:
                    print(f"    10^{b:.2f} km2: dam {s.n0.median():.4f} (n={len(s)})  other {o.n0.median():.4f} (n={len(o):,})  ratio {s.n0.median() / o.n0.median():.2f}")

        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(1, 3, figsize=(20, 6))
        # (a) n0 vs size: binned medians per group
        for gname, c in [("other", "0.5"), ("dam reach", "tab:red"), ("upstream 1 hop", "tab:orange"), ("downstream 1 hop", "tab:blue")]:
            s = df[df.group == gname]
            m = s.groupby("bin")["n0"].median()
            cnt = s.groupby("bin")["n0"].size()
            m = m[cnt >= 15]
            ax[0].plot(m.index + 0.125, m.values, "-o", color=c, ms=3, label=f"{gname} (n={len(s):,})")
        ax[0].set_xlabel("log10 drainage area (km²)"); ax[0].set_ylabel("median n_0"); ax[0].legend(fontsize=8); ax[0].grid(alpha=.3)
        ax[0].set_title("n_0 by river size and dam proximity")
        # (b) residual box by group
        order_g = ["dam reach"] + [f"upstream {h} hop" for h in (1, 2, 3)] + [f"downstream {h} hop" for h in (1, 2, 3)]
        data = [df[df.group == g].resid.dropna().values for g in order_g]
        ax[1].boxplot(data, showfliers=False)
        ax[1].set_xticks(range(1, len(order_g) + 1)); ax[1].set_xticklabels([g.replace(" hop", "").replace("stream ", "") for g in order_g], rotation=30, fontsize=8)
        ax[1].axhline(0, color="k", lw=0.8); ax[1].set_ylabel("n_0 minus same-size median of unaffected reaches"); ax[1].grid(alpha=.3)
        ax[1].set_title("roughness excess near dams (0 = no effect)")
        # (c) map of dam reaches coloured by residual, Midwest box drawn
        oth = df[df.group == "other"]
        ax[2].scatter(oth.lon, oth.lat, s=0.2, c="0.85", rasterized=True)
        d = df[df.group == "dam reach"]
        sc = ax[2].scatter(d.lon, d.lat, s=8, c=d.resid, cmap="coolwarm", vmin=-0.1, vmax=0.1, edgecolor="k", linewidth=0.2)
        ax[2].add_patch(plt.Rectangle((lon0, lat0), lon1 - lon0, lat1 - lat0, fill=False, ec="k", ls="--"))
        ax[2].set_title("dam reaches: n_0 residual vs same-size reaches (box = Midwest)"); ax[2].set_aspect("equal")
        fig.colorbar(sc, ax=ax[2], shrink=0.7, label="n_0 residual")
        fig.suptitle(f"{rid}: does the learned roughness track dams?")
        fig.tight_layout()
        out = run / "plots" / "dam_roughness.png"
        fig.savefig(out, dpi=150, facecolor="white"); print(f"wrote {out}")
        df.to_csv(run / "plots" / "dam_roughness.csv", index=False)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python
"""Validate a leakance arm's learned fields against measured physical ranges, and
map every parameter over the CONUS river network.

Three validations, each against an external reference rather than internal
consistency:
  roughness n     -> Chow (1959) tabulated Manning's n for natural channels
  leakance K_D    -> MODFLOW streambed leakance K/M, against measured streambed Kv
  offset d_gw     -> depth of the water table below the channel

usage: validate_params.py <params.nc> [--zeta <kan_parameters.nc>] --out-prefix P
"""
import argparse, json
import numpy as np, netCDF4 as nc
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt, matplotlib.colors as mcolors
from scipy.stats import spearmanr

FAB = "/home/tbindas/projects/ddr/data/merit/riv_pfaf_7_MERIT_Hydro_v07_Basins_v01_bugfix1.shp"
ATTR = "/home/tbindas/projects/ddr/data/merit_global_attributes_v2.nc"

# --- external reference ranges -------------------------------------------------
# Chow, V.T. (1959) Open-Channel Hydraulics, Table 5-6, natural streams.
CHOW = {
    "clean straight, full stage":        (0.025, 0.033),
    "clean winding, pools and shoals":   (0.033, 0.045),
    "mountain, gravel and cobbles":      (0.030, 0.050),
    "mountain, cobbles large boulders":  (0.040, 0.070),
    "sluggish, weedy, deep pools":       (0.050, 0.080),
    "very weedy reaches":                (0.075, 0.150),
}
# Streambed vertical hydraulic conductivity, measured, four orders of magnitude
# (Nature Scientific Reports 2020, 10:3333): 7.57e-3 .. 1.81 m/day.
KV_MEAS_MDAY = (7.57e-3, 1.81)
BED_THICK_M = 1.0   # leakance = Kv / M; state the assumption rather than hide it

# The leakance term computes its plan-view width as (p*d)^q, while the routing
# geometry computes the SAME reach's width as p*d^q (src/geometry.rs, Leopold and
# Maddock). Those differ by the depth-independent factor p^(q-1), which is 0.3445
# at the prescribed p = 21, q = 0.65. So the area multiplying K_D is 0.3445 of the
# routing's plan area, and a learned K_D silently absorbs that factor. To compare
# K_D against a measured streambed leakance we must divide it out. This is
# inherited from DDR (tests/leakance_reference_match.rs matches its formula
# exactly), so it is a property of the formulation, not a port error. The term's
# area is also dimensionally not m^2 unless q = 1, which makes this comparison
# approximate rather than exact. See research/specs/2026-09-17-model-equations-reference.md.
AREA_FORM_FACTOR = 0.3445

def band(v, lo, hi):
    return float(((v >= lo) & (v <= hi)).mean())

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("params"); ap.add_argument("--zeta", default=None)
    ap.add_argument("--out-prefix", required=True)
    ap.add_argument("--fabric", default=FAB)
    a = ap.parse_args()

    d = nc.Dataset(a.params)
    have = lambda v: v in d.variables
    g = lambda v: np.asarray(d[v][:], dtype=float)
    cid = np.asarray(d["COMID"][:], dtype=np.int64)
    F = {"n": g("n")}
    for v in ["K_D", "d_gw", "leakance_factor", "gamma", "q_spatial", "p_spatial", "slope"]:
        if have(v): F[v] = g(v)
    if "K_D" in F and "leakance_factor" in F:
        F["conductance"] = F["leakance_factor"] * F["K_D"]

    report = {"n_reaches": int(cid.size), "source": a.params}

    # ---- 1. roughness against Chow -------------------------------------------
    n = F["n"]
    report["roughness"] = {
        "p2": float(np.percentile(n, 2)), "median": float(np.median(n)),
        "p98": float(np.percentile(n, 98)),
        "frac inside Chow natural-channel envelope 0.025-0.150": band(n, 0.025, 0.150),
        "frac below 0.025 (smoother than any tabulated natural channel)": float((n < 0.025).mean()),
        "frac above 0.150 (rougher than very weedy)": float((n > 0.150).mean()),
        "by Chow class": {k: band(n, lo, hi) for k, (lo, hi) in CHOW.items()},
    }

    # ---- 2. leakance against measured streambeds -----------------------------
    if "conductance" in F:
        C = F["conductance"]
        # divide out the area-form factor so C is comparable to a real leakance
        kv_mday = (C / AREA_FORM_FACTOR) * BED_THICK_M * 86400.0
        lo, hi = KV_MEAS_MDAY
        report["leakance"] = {
            "assumed streambed thickness m": BED_THICK_M,
            "area-form factor divided out": AREA_FORM_FACTOR,
            "conductance 1/s p10/median/p90": [float(np.percentile(C,10)), float(np.median(C)), float(np.percentile(C,90))],
            "implied Kv m/day p10/median/p90": [float(np.percentile(kv_mday,10)), float(np.median(kv_mday)), float(np.percentile(kv_mday,90))],
            "measured Kv range m/day": list(KV_MEAS_MDAY),
            "frac inside measured range": band(kv_mday, lo, hi),
            "frac below measured minimum": float((kv_mday < lo).mean()),
            "position in measured range (0=min,1=max, log scale)":
                float(np.clip((np.log10(np.median(kv_mday)) - np.log10(lo)) / (np.log10(hi) - np.log10(lo)), 0, 1)),
        }

    # ---- 3. the offset as a water table --------------------------------------
    if "d_gw" in F:
        dg = F["d_gw"]
        rec = {"p10": float(np.percentile(dg,10)), "median": float(np.median(dg)), "p90": float(np.percentile(dg,90)),
               "frac positive (offset ABOVE the bed, so the losing-only clamp gates the reach off)": float((dg > 0).mean())}
        if a.zeta:
            z = nc.Dataset(a.zeta)
            dep = np.asarray(z["depth_mean"][:], dtype=float)
            # offsets are on the full network, depths on the eval network: join by COMID
            zc = np.asarray(z["COMID_eval"][:], dtype=np.int64)
            import pandas as pd
            pos = pd.Index(cid).get_indexer(zc)
            ok = pos >= 0
            rec["frac of eval reaches with offset above mean depth (leakance switched off)"] = \
                float((dg[pos[ok]] > dep[ok]).mean())
        report["offset"] = rec

    # ---- attribute correlations: is any of this reading the landscape? -------
    A = nc.Dataset(ATTR); acid = np.asarray(A["COMID"][:], dtype=np.int64)
    idx = {c: i for i, c in enumerate(acid)}
    sel = np.array([idx.get(c, -1) for c in cid]); ok = sel >= 0
    attrs = {k: np.asarray(A[k][:], dtype=float)[sel[ok]]
             for k in ["aridity", "meanslope", "permeability", "meanP", "meanelevation", "HWSD_sand"]}
    corr = {}
    for name, v in F.items():
        vv = v[ok]; row = {}
        for k, av in attrs.items():
            m = np.isfinite(vv) & np.isfinite(av)
            row[k] = round(float(spearmanr(vv[m], av[m]).statistic), 3)
        corr[name] = row
    report["attribute_correlations"] = corr

    # ---- are the outputs independent? ---------------------------------------
    keys = [k for k in ["n", "gamma", "K_D", "d_gw", "leakance_factor"] if k in F]
    if len(keys) > 1:
        X = np.column_stack([F[k] for k in keys]); X = (X - X.mean(0)) / X.std(0)
        s = np.linalg.svd(X, compute_uv=False); p = s**2 / (s**2).sum()
        report["output_independence"] = {
            "fields": keys, "variance_shares": [round(float(x), 4) for x in p],
            "effective_rank": round(float(np.exp(-(p * np.log(p)).sum())), 3),
            "pairwise_spearman": {f"{keys[i]}|{keys[j]}": round(float(spearmanr(F[keys[i]], F[keys[j]]).statistic), 4)
                                  for i in range(len(keys)) for j in range(i+1, len(keys))},
        }

    with open(f"{a.out_prefix}_validation.json", "w") as fh:
        json.dump(report, fh, indent=2)
    print(json.dumps(report, indent=2))

    # ---- maps ----------------------------------------------------------------
    import geopandas as gpd, pandas as pd, pyogrio
    gdf = pyogrio.read_dataframe(a.fabric, columns=["COMID", "uparea"]).set_index("COMID")
    gdf = gdf.loc[gdf.index.intersection(cid)]
    pos = pd.Index(cid).get_indexer(gdf.index.values)
    for k, v in F.items(): gdf[k] = v[pos]
    gdf["lw"] = np.clip(0.15 + 0.35*np.log10(np.maximum(gdf["uparea"].values, 1.0))/5.0, 0.15, 1.2)
    gdf = gdf.sort_values("uparea")

    panels = [k for k in ["n", "conductance", "d_gw", "leakance_factor", "gamma", "q_spatial"] if k in gdf]
    titles = {"n": "Manning's $n$", "conductance": "effective conductance $g\\cdot K_D$ (1/s)",
              "d_gw": "groundwater offset $d_{gw}$ (m)", "leakance_factor": "leakance gate $g$",
              "gamma": "stage exponent $\\gamma$", "q_spatial": "width exponent $q$"}
    ncol = 2; nrow = int(np.ceil(len(panels)/ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(9.5*ncol, 5.5*nrow), squeeze=False)
    for ax, col in zip(axes.ravel(), panels):
        v = gdf[col].values; lo, hi = np.percentile(v, [2, 98])
        logc = col == "conductance"
        cmap = {"n": "plasma_r", "conductance": "viridis", "d_gw": "coolwarm",
                "leakance_factor": "RdYlBu_r", "gamma": "magma", "q_spatial": "cividis"}.get(col, "viridis")
        norm = mcolors.LogNorm(vmin=max(lo,1e-12), vmax=hi) if logc else mcolors.Normalize(vmin=lo, vmax=hi)
        gdf.plot(ax=ax, column=col, cmap=cmap, norm=norm, linewidth=gdf["lw"].values, rasterized=True)
        ax.set_title(titles.get(col, col), fontsize=12); ax.set_axis_off()
        fig.colorbar(plt.cm.ScalarMappable(norm=norm, cmap=cmap), ax=ax, fraction=0.03, pad=0.01)
    for ax in axes.ravel()[len(panels):]: ax.set_axis_off()
    fig.suptitle(f"Learned parameter fields, CONUS ({cid.size:,} reaches)\n{a.params}", fontsize=12)
    fig.tight_layout()
    out = f"{a.out_prefix}_maps.png"; fig.savefig(out, dpi=115, bbox_inches="tight")
    print("wrote", out)

if __name__ == "__main__":
    main()

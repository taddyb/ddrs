"""Per-gauge attribute table for the 2,365-gauge population: gauges_table.csv.

Columns: landscape covariates (HUC02, ecoregion, CLASS, network length, outlet attributes), upstream
catchsize-weighted means of the routing head's 10 inputs plus snow/aridity/seasonality/temperature (MERIT attributes),
longest flow path (km, from the CONUS adjacency length_m), dam fields from paired_full_run.csv.
"""
import numpy as np
import pandas as pd
import xarray as xr
import zarr

import common as C

HEAD = ["SoilGrids1km_clay", "aridity", "meanelevation", "meanP", "NDVI", "meanslope", "log10_uparea",
        "SoilGrids1km_sand", "ETPOT_Hargr", "Porosity"]
EXTRA = ["snow_fraction", "snowfall_fraction", "seasonality_P", "seasonality_PET", "meanTa", "permeability", "FW",
         "glaciers", "permafrost"]

ids, t, P, O = C.load_preds(C.S42)
cov = pd.read_csv("/home/tbindas/projects/ddrs/.ddrs/experiments/landscape-p21-all-5yr/merged/figures/covariates.csv",
                  dtype={"staid": str, "HUC02": str}).set_index("staid")
keep = ["n_reach", "total_length_km", "mean_reach_length_km", "DRAIN_SQKM", "HUC02", "AGGECOREGI", "CLASS", "STATE",
        "LAT_GAGE", "LNG_GAGE", "obs_mean_q_m3s"]
T = cov.reindex(ids)[keep].copy()
T["HUC02"] = T.HUC02.astype(str).str.zfill(2)
print("covariates matched", T.n_reach.notna().sum(), "of", len(ids))

A = xr.open_dataset("/home/tbindas/projects/ddr/data/merit_global_attributes_v2.nc")
gz = zarr.open_group("/home/tbindas/projects/ddr/data/merit_gages_conus_adjacency.zarr", mode="r")
cz = zarr.open_group("/home/tbindas/projects/ddr/data/merit_conus_adjacency.zarr", mode="r")
corder = np.asarray(cz["order"][:], dtype=np.int64)
clen = np.asarray(cz["length_m"][:], dtype=float)
cslope = np.asarray(cz["slope"][:], dtype=float)

sub = {}
allc = set()
for s in ids:
    g = gz[s]
    sub[s] = (np.asarray(g["order"][:], dtype=np.int64), np.asarray(g["indices_0"][:], dtype=np.int64),
              np.asarray(g["indices_1"][:], dtype=np.int64), int(g.attrs["gage_idx"]))
    allc.update(sub[s][0].tolist())
allc = np.array(sorted(allc))
Asub = A.sel(COMID=allc)
AV = {v: pd.Series(Asub[v].values, index=allc) for v in HEAD + EXTRA + ["catchsize"]}

rows = []
for s in ids:
    order, i0, i1, gidx = sub[s]
    w = AV["catchsize"].reindex(order).values
    w = np.where(np.isfinite(w), w, 0.0)
    r = {"STAID": s}
    for v in HEAD + EXTRA:
        x = AV[v].reindex(order).values
        ok = np.isfinite(x)
        r["up_" + v] = float((x[ok] * w[ok]).sum() / w[ok].sum()) if ok.any() and w[ok].sum() > 0 else np.nan
    r["out_log10_uparea"] = float(AV["log10_uparea"].get(order[-1], np.nan)) if len(order) else np.nan
    # longest flow path to the gauge reach: DP over global indices, edges i1 (upstream) -> i0 (downstream)
    L = {}
    ups = {}
    for a, b in zip(i1, i0):
        ups.setdefault(int(b), []).append(int(a))
    # nodes in topological order: global conus index order is topological (rows >= cols)
    nodes = sorted(set(i0.tolist()) | set(i1.tolist()) | {gidx})
    for n_ in nodes:
        up = ups.get(n_, [])
        L[n_] = clen[n_] / 1000.0 + (max(L[u] for u in up) if up else 0.0)
    r["longest_path_km"] = float(L.get(gidx, np.nan))
    r["gauge_slope"] = float(cslope[gidx])
    rows.append(r)
U = pd.DataFrame(rows).set_index("STAID")
T = T.join(U)
# travel time proxy at a nominal 1 m/s celerity, days
T["tt_days_1ms"] = T.longest_path_km * 1000.0 / 86400.0

pf = pd.read_csv("/home/tbindas/projects/ddrs/.claude/worktrees/reservoir-options/experiments/reservoir/full_run/"
                 "paired_full_run.csv", dtype={"STAID": str}).set_index("STAID")
T = T.join(pf[["n_nid_ge10mcm", "nid_dor", "nid_on_gauge_reach", "nse_base", "kge_base"]])
T.index.name = "STAID"
T.to_csv(C.HERE / "gauges_table.csv")
print(T.describe().T[["count", "50%"]].to_string())

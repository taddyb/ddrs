#!/usr/bin/env python
"""Per-gauge scores, paired comparisons and page data for one seed pair of the learned dam release.

    analysis.py --off <run-id> --learned <run-id> --tag s42 [--series]

Scores every gauge on the test window (1995-10-02..2010-09-29, the days both arms and the summed-Q' baseline share)
with the same NSE / KGE code as ../paired_full_run.py, then writes

  output/reservoir_full_run/web/build/results_<tag>.json   paired medians with bootstrap intervals, up/down counts
  output/reservoir_full_run/web/build/gauges_<tag>.csv     one row per gauge
  output/reservoir_full_run/web/publish/data_<tag>.json    what the page reads (gauges, dams, groups)
  output/reservoir_full_run/web/publish/series/*.b64.txt   (--series) daily series for the hydrograph viewer

Series format: per HUC2 region, for each gauge, observed / no dam / dam release as log-spaced codes (step 0.5 %,
lo 1e-3 m3/s; 0 = at or below lo, 65535 = missing), first-differenced in time (dam release stored as its difference
from no dam), bytes split low/high, gzip, base64. The page inflates with DecompressionStream.
"""
from __future__ import annotations

import argparse
import base64
import gzip
import json
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
import zarr
from scipy.stats import binomtest, spearmanr

HERE = Path(__file__).resolve().parent
EXP = HERE.parents[1]                      # experiments/reservoir
WT = HERE.parents[3]                       # worktree root
OUT = WT / "output/reservoir_full_run/web"
RUNS = Path("/home/tbindas/projects/ddrs/.ddrs/runs")
GAGES = Path("/home/tbindas/projects/ddr/references/gage_info/gages_3000.csv")
HUC_ZIP = "/vsizip//mnt/ssd1/data/camels/basin_dataset_public_v1p2/shapefiles/huc_02.zip/huc_02.shp"
DAM_FEATURES = Path("/home/tbindas/projects/ddrs/.claude/worktrees/agent-a92e512a7c47c97b4/experiments/reservoir/release_head/dam_features.csv")
SMOKE_FIT = EXP / "smoke/expected_release_fit.csv"
MAPW, PAD = 960.0, 8.0
LO, STEP = -3.0, float(np.log10(1.005))
DOR_EDGES, DOR_NAMES = [0, 0.1, 0.5, 1, 2, np.inf], ["<=0.1", "0.1-0.5", "0.5-1", "1-2", ">2"]

ap = argparse.ArgumentParser()
ap.add_argument("--off", required=True)
ap.add_argument("--learned", required=True)
ap.add_argument("--tag", required=True)
ap.add_argument("--series", action="store_true")
args = ap.parse_args()
(OUT / "build").mkdir(parents=True, exist_ok=True)
(OUT / "publish/series").mkdir(parents=True, exist_ok=True)


# ---------- load and score ----------
def load(run):
    z = zarr.open(str(RUNS / run / "eval/predictions.zarr"), mode="r")
    ids = [bytes(r).decode().strip("\x00") for r in z["gage_ids"][:]]
    t = z["time"][:].astype("datetime64[ns]").astype("datetime64[D]")
    return pd.Index(ids), t, z["predictions"][:].astype(float), z["observations"][:].astype(float)


def metrics(p, o):
    m = np.isfinite(p) & np.isfinite(o)
    if m.sum() < 365:
        return np.nan, np.nan
    p, o = p[m], o[m]
    nse = 1 - ((p - o) ** 2).sum() / ((o - o.mean()) ** 2).sum()
    r = np.corrcoef(p, o)[0, 1] if p.std() > 0 else 0.0
    kge = 1 - np.sqrt((r - 1) ** 2 + (p.std() / o.std() - 1) ** 2 + (p.mean() / o.mean() - 1) ** 2)
    return nse, kge


def paired(d, seed=42):
    d = np.asarray(d, float)
    d = d[np.isfinite(d)]
    if not len(d):
        return dict(n=0)
    nz = d[np.abs(d) > 1e-9]
    up = int((nz > 0).sum())
    b = np.median(np.random.default_rng(seed).choice(d, (2000, len(d))), axis=1)
    return dict(n=int(len(d)), median=float(np.median(d)), ci=[float(np.percentile(b, 2.5)), float(np.percentile(b, 97.5))],
                n_up=up, n_down=int(len(nz) - up), sign_p=float(binomtest(up, len(nz)).pvalue) if len(nz) else None)


ids, t, P_off, O = load(args.off)
ids_l, t_l, P_dam, O_l = load(args.learned)
assert (ids == ids_l).all() and (t == t_l).all()
assert np.array_equal(np.isnan(O), np.isnan(O_l)) and np.nanmax(np.abs(O - O_l)) == 0
bm = json.load(open(RUNS / args.off / "baseline/manifest.json"))
B_all = np.fromfile(RUNS / args.off / "baseline/predictions.f32", dtype=np.float32).reshape(bm["n_gauges"], bm["n_days"])
bidx = {str(g): i for i, g in enumerate(bm["gage_ids"])}
off0 = int((t[0] - np.datetime64(str(bm["time_range_daily"][0])[:10])).astype(int))
B = np.full(P_off.shape, np.nan)
for i, s in enumerate(ids):
    if s in bidx:
        B[i] = B_all[bidx[s], off0:off0 + len(t)]
print(f"{len(ids)} gauges x {len(t)} days, {t[0]}..{t[-1]}; baseline rows found {sum(s in bidx for s in ids)}")

rows = []
for i, s in enumerate(ids):
    n_o, k_o = metrics(P_off[i], O[i])
    n_l, k_l = metrics(P_dam[i], O[i])
    n_b, k_b = metrics(B[i], O[i])
    rows.append(dict(STAID=s, nse_off=n_o, kge_off=k_o, nse_dam=n_l, kge_dam=k_l, nse_base=n_b, kge_base=k_b,
                     obs_days=int(np.isfinite(O[i]).sum())))
df = pd.DataFrame(rows).set_index("STAID")
nid = pd.read_csv(EXP / "nid/nid_dams_by_gauge.csv", dtype={"STAID": str}).set_index("STAID")
df = df.join(nid[["area_km2", "qmean", "n_nid", "n_nid_ge10mcm", "nid_storage_mcm", "nid_dor", "nid_on_gauge_reach",
                  "largest_dam", "largest_storage_mcm"]])
gm = pd.read_csv(GAGES, dtype={"STAID": str}).set_index("STAID")
df = df.join(gm[["STANAME", "LAT_GAGE", "LNG_GAGE", "COMID"]])
df["dammed"] = df.n_nid_ge10mcm > 0
df["on_reach"] = df.dammed & df.nid_on_gauge_reach.astype(bool)
df["dnse"], df["dkge"] = df.nse_dam - df.nse_off, df.kge_dam - df.kge_off
df["dor_bin"] = pd.cut(df.nid_dor.where(df.dammed), DOR_EDGES, labels=DOR_NAMES).astype(str)

# ---------- map frame: same as ../../smoke/page/build_map.py (CONUS Albers, HUC2 bounds) ----------
h = gpd.read_file(HUC_ZIP).set_crs(26915)
h = h[h.HUC2.astype(int) <= 18].to_crs(5070)
h["geometry"] = h.geometry.buffer(0)
minx, miny, maxx, maxy = h.total_bounds
sc = (MAPW - 2 * PAD) / (maxx - minx)
mapjson = json.loads((EXP / "smoke/page/map.json").read_text())
assert abs(round((maxy - miny) * sc + 2 * PAD, 1) - mapjson["h"]) < 0.05, "map frame differs from the smoke basemap"


def xy(x, y):
    return round(PAD + (x - minx) * sc, 1), round(PAD + (maxy - y) * sc, 1)


gp = gpd.GeoDataFrame(df.reset_index(), geometry=gpd.points_from_xy(df.LNG_GAGE, df.LAT_GAGE), crs=4326).to_crs(5070)
j = gpd.sjoin(gp, h[["HUC2", "geometry"]], how="left", predicate="within")
j = j[~j.index.duplicated()]
miss = j.HUC2.isna()
if miss.any():  # coastal gauges just outside the simplified polygons: nearest region
    nn = gpd.sjoin_nearest(gp[miss.values], h[["HUC2", "geometry"]], how="left")
    nn = nn[~nn.index.duplicated()]
    j.loc[miss, "HUC2"] = nn.HUC2.values
print("gauges assigned to HUC2 by nearest polygon:", int(miss.sum()))
df["huc2"] = [f"{int(v):02d}" for v in j.HUC2.values]
df["x"], df["y"] = zip(*[xy(g.x, g.y) for g in gp.geometry])

# ---------- groups ----------
groups = {
    "all": np.ones(len(df), bool), "dammed": df.dammed.values, "undammed": ~df.dammed.values,
    "on_reach": df.on_reach.values, "further_up": (df.dammed & ~df.on_reach).values,
}
for b in DOR_NAMES:
    groups["dor_" + b] = (df.dammed & (df.dor_bin == b)).values
res = dict(runs=dict(off=args.off, learned=args.learned), tag=args.tag, n=len(df),
           window=[str(t[0]), str(t[-1])], n_days=int(len(t)))
res["group_n"] = {k: int(m.sum()) for k, m in groups.items()}
res["median"] = {k: {f"{met}_{arm}": float(np.nanmedian(df[f"{met}_{arm}"].values[m])) for met in ("nse", "kge")
                     for arm in ("base", "off", "dam")} for k, m in groups.items()}
res["paired"] = {}
for name, (a, b) in {"dam_minus_off": ("dam", "off"), "off_minus_base": ("off", "base"), "dam_minus_base": ("dam", "base")}.items():
    res["paired"][name] = {met: {k: paired(df[f"{met}_{a}"].values[m] - df[f"{met}_{b}"].values[m]) for k, m in groups.items()}
                           for met in ("nse", "kge")}
huc = {}
for hc in sorted(df.huc2.unique()):
    for grp, m in [("dammed", df.dammed), ("undammed", ~df.dammed)]:
        mm = (df.huc2 == hc) & m
        if mm.sum():
            d = df.dnse[mm].dropna()
            huc.setdefault(hc, {})[grp] = dict(n=int(mm.sum()), median=float(d.median()), n_up=int((d > 0).sum()),
                                               vs_base=float((df.nse_dam - df.nse_base)[mm].median()))
res["huc2"] = huc

# ---------- dams ----------
rp = pd.read_csv(RUNS / args.learned / "release_params.csv")
feat = pd.read_csv(DAM_FEATURES)[["COMID", "largest_nid_id", "largest_name", "storage_mcm", "n_dams"]]
nd = pd.read_csv(EXP / "nid/nid_dams_in_eval_network.csv")
dm = rp.merge(feat, on="COMID", how="left").merge(
    nd[["nid_id", "lat", "lon", "primary_purpose", "purposes", "year", "state", "river", "n_eval_gauges_downstream"]],
    left_on="largest_nid_id", right_on="nid_id", how="left")
assert dm.lat.notna().all(), "dam without coordinates"
dp = gpd.GeoDataFrame(dm, geometry=gpd.points_from_xy(dm.lon, dm.lat), crs=4326).to_crs(5070)
dm["x"], dm["y"] = zip(*[xy(g.x, g.y) for g in dp.geometry])
dm["amp"] = np.hypot(dm.a, dm.b)
dm["peak_doy"] = (np.arctan2(dm.a, dm.b) % (2 * np.pi)) * 365.25 / (2 * np.pi)
dm["flood"] = dm.purposes.fillna("").str.contains("Flood")
# residence time where the dam sits on a gauge's reach: NID storage / mean flow at that gauge
onr = df[df.dammed][["COMID", "qmean", "nid_dor"]].reset_index()
onr = onr[onr.COMID.isin(dm.COMID)].drop_duplicates("COMID")
dm = dm.merge(onr.rename(columns={"STAID": "gauge"}), on="COMID", how="left")
dm["res_days"] = dm.storage_mcm * 1e6 / (dm.qmean * 86400)
sf = pd.read_csv(SMOKE_FIT, dtype={"STAID": str})
sf = sf[(sf.role == "dam") & sf.on_reach.astype(bool)][["dam_COMID", "seas_T0"]].rename(columns={"dam_COMID": "COMID", "seas_T0": "fit_T0"})
dm = dm.merge(sf.drop_duplicates("COMID"), on="COMID", how="left")


def sp(a, b):
    m = np.isfinite(a) & np.isfinite(b)
    r = spearmanr(a[m], b[m])
    return dict(n=int(m.sum()), rho=float(r.statistic), p=float(r.pvalue))


res["release"] = dict(
    n=len(dm), T0=dm.T0_days.quantile([.1, .25, .5, .75, .9]).round(4).tolist(), T0_max=float(dm.T0_days.max()),
    ratio=(dm.T_max_days / dm.T_min_days).quantile([.1, .25, .5, .75, .9]).round(3).tolist(),
    peak_month_counts=np.histogram(dm.peak_doy, bins=np.linspace(0, 365.25, 13))[0].tolist(),
    T0_flood=float(dm.T0_days[dm.flood].median()), T0_notflood=float(dm.T0_days[~dm.flood].median()),
    n_flood=int(dm.flood.sum()),
    rho_storage=sp(np.log(dm.storage_mcm.values), np.log(dm.T0_days.values)),
    rho_residence=sp(np.log(dm.res_days.values), np.log(dm.T0_days.values)),
    rho_fit=sp(np.log(dm.fit_T0.values), np.log(dm.T0_days.values)),
    res_days_median=float(dm.res_days.median()), T0_median_onreach=float(dm.T0_days[dm.res_days.notna()].median()),
)

# ---------- write ----------
json.dump(res, open(OUT / f"build/results_{args.tag}.json", "w"), indent=1)
df.to_csv(OUT / f"build/gauges_{args.tag}.csv")
dm.drop(columns="geometry", errors="ignore").to_csv(OUT / f"build/dams_{args.tag}.csv", index=False)


def r4(v):
    return None if not np.isfinite(v) else round(float(v), 4)


def r6(v):
    return None if not np.isfinite(v) else round(float(v), 6)


gauges = [dict(id=s, nm=r.STANAME, h=r.huc2, x=r.x, y=r.y, a=round(float(r.area_km2), 1), d=int(r.dammed), o=int(r.on_reach),
               dor=r4(r.nid_dor) if r.dammed else 0, dam=(r.largest_dam if isinstance(r.largest_dam, str) else ""),
               st=round(float(r.largest_storage_mcm), 1) if r.dammed else 0, nb=r6(r.nse_base), no=r6(r.nse_off), nd=r6(r.nse_dam),
               kb=r6(r.kge_base), ko=r6(r.kge_off), kd=r6(r.kge_dam)) for s, r in df.iterrows()]
dams = [dict(c=int(r.COMID), nm=r.largest_name, x=r.x, y=r.y, st=round(float(r.storage_mcm), 1), yr=None if pd.isna(r.year) else int(r.year),
             pu=r.primary_purpose if isinstance(r.primary_purpose, str) else "", ps=r.purposes if isinstance(r.purposes, str) else "",
             T0=round(float(r.T0_days), 4), a=round(float(r.a), 4), b=round(float(r.b), 4), tmin=round(float(r.T_min_days), 4),
             tmax=round(float(r.T_max_days), 4), res=r4(r.res_days), fit=r4(r.fit_T0), g=r.gauge if isinstance(r.gauge, str) else "",
             nd=int(r.n_dams)) for _, r in dm.iterrows()]
json.dump(dict(tag=args.tag, runs=res["runs"], window=res["window"], gauges=gauges, dams=dams),
          open(OUT / f"publish/data_{args.tag}.json", "w"), separators=(",", ":"))
print("wrote", OUT / f"publish/data_{args.tag}.json", (OUT / f"publish/data_{args.tag}.json").stat().st_size)


# ---------- series ----------
def codes(a):
    c = np.full(a.shape, 65535, np.int32)
    f = np.isfinite(a)
    c[f & (a <= 10 ** LO)] = 0
    p = f & (a > 10 ** LO)
    c[p] = np.clip(np.round((np.log10(a[p]) - LO) / STEP) + 1, 1, 65534)
    return c


def pack(rows_codes):
    """rows_codes: list of int32 arrays, each (n_days,), first one per gauge absolute; returns gz+b64 text."""
    arr = np.stack(rows_codes).astype(np.int64)
    d = np.diff(arr, axis=1, prepend=0)
    u = (d % 65536).astype(np.uint16)
    b = u.view(np.uint8).reshape(u.shape + (2,))
    raw = np.concatenate([b[..., 0].ravel(), b[..., 1].ravel()]).tobytes()
    return base64.b64encode(gzip.compress(raw, 9)).decode("ascii")


if args.series:
    layout, tot = {}, 0
    for hc in sorted(df.huc2.unique()):
        gi = [i for i, s in enumerate(ids) if df.huc2.iloc[i] == hc]
        rows_c = []
        for i in gi:
            co, c0, c1 = codes(O[i]), codes(P_off[i]), codes(P_dam[i])
            dd = np.where((c0 == 65535) | (c1 == 65535), 0, c1 - c0)
            rows_c += [co, c0, dd]
        txt = pack(rows_c)
        f = OUT / f"publish/series/huc{hc}.b64.txt"
        f.write_text(txt)
        tot += len(txt)
        layout[hc] = dict(file=f"series/huc{hc}.b64.txt", gauges=[ids[i] for i in gi])
    # picks: add the summed Q' as a fourth row
    picks = json.loads((HERE / "picks.json").read_text())
    rows_c, pl = [], []
    for pk in picks["gauges"]:
        i = ids.get_loc(pk["id"])
        c0 = codes(P_off[i])
        c1 = codes(P_dam[i])
        rows_c += [codes(O[i]), c0, np.where((c0 == 65535) | (c1 == 65535), 0, c1 - c0), codes(B[i])]
        pl.append(pk["id"])
    (OUT / "publish/series/picks.b64.txt").write_text(pack(rows_c))
    meta = dict(lo=LO, step=STEP, n_days=int(len(t)), start=str(t[0]), layout=layout, picks=pl)
    json.dump(meta, open(OUT / "publish/series/index.json", "w"), separators=(",", ":"))
    # self-check: decode one gauge and compare to the source within the quantisation step
    txt = (OUT / f"publish/series/huc{df.huc2.iloc[0]}.b64.txt").read_text()
    raw = gzip.decompress(base64.b64decode(txt))
    n = len(raw) // 2
    u = (np.frombuffer(raw[:n], np.uint8).astype(np.uint16) | (np.frombuffer(raw[n:], np.uint8).astype(np.uint16) << 8))
    u = u.reshape(-1, len(t)).astype(np.int64)
    c = np.cumsum(u, axis=1) % 65536
    g0 = layout[df.huc2.iloc[0]]["gauges"][0]
    i0 = ids.get_loc(g0)
    dec = np.where(c[1] == 65535, np.nan, np.where(c[1] == 0, 0, 10 ** (LO + (c[1] - 1) * STEP)))
    ok = np.isfinite(P_off[i0]) & (P_off[i0] > 10 ** LO)
    print("series total b64 MB", round(tot / 1e6, 2), "max rel err no-dam", float(np.nanmax(np.abs(dec[ok] / P_off[i0][ok] - 1))))
print(json.dumps({k: res[k] for k in ("group_n",)}, indent=0))

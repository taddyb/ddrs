"""Build an area-balanced ~2,000-gauge CSV from GAGES-II.

Follow-up to /tmp/experiment-handoff-small-basin-domination.md (2026-08-02):
the gages_3000 population is 87.5% <5,000 km2 and its DA_VALID column is an
absolute-difference criterion that preferentially deletes large basins.

This script:
  (a) starts from GAGES-II.csv (8,931 gauges, same schema as gages_3000.csv);
  (b) recomputes DA_VALID as ABS_DIFF / DRAIN_SQKM <= 0.10 (relative);
  (c) keeps only gauges present in the USGS observations icechunk store with
      >= COVERAGE_MIN non-NaN daily coverage in BOTH the configured training
      window (1981-10-01..1995-09-30) and eval window (1995-10-01..2010-09-30),
      and present in merit_gages_conus_adjacency.zarr as a non-headwater
      subgraph (order length > 1) -- mirroring the dataset.rs filter so the
      CSV population equals the population training actually uses;
  (d) subsamples to ~2,000: ALL basins >= 5,000 km2, topped up from
      [1,000, 5,000) to 1,000 gauges >= 1,000 km2, plus a random 1,000 from
      < 1,000 km2 -- ~50/50 either side of 1,000 km2. Small basins are kept
      deliberately (they teach the identity-routing regime); the goal is to
      reduce their gradient share, not remove them.

Run under DDR's venv:
  cd ~/projects/ddr && uv run python ~/projects/ddrs/scripts/build_gages_2000_area_balanced.py
"""

import numpy as np
import pandas as pd
import xarray as xr
import zarr
import icechunk

GAGES_II = "/home/tbindas/projects/ddr/references/gage_info/GAGES-II.csv"
OBS_STORE = "/mnt/ssd1/data/icechunk/usgs_daily_observations"
ADJ_STORE = "/home/tbindas/projects/ddr/data/merit_gages_conus_adjacency.zarr"
OUT_CSV = "/home/tbindas/projects/ddr/references/gage_info/gages_2000_area_balanced.csv"

REL_TOL = 0.10
COVERAGE_MIN = 0.80
TRAIN_WINDOW = ("1981-10-01", "1995-09-30")  # config/merit_training.yaml experiment window
EVAL_WINDOW = ("1995-10-01", "2010-09-30")   # testing window
SEED = 42
N_SMALL = 1000   # target below 1,000 km2
N_LARGE = 1000   # target at/above 1,000 km2 (all >=5,000 kept, rest from [1k,5k))

# ---- (a) + (b): load GAGES-II, recompute DA_VALID relatively -------------
df = pd.read_csv(GAGES_II, dtype={"STAID": str})
df["STAID"] = df["STAID"].str.zfill(8)
assert not df["STAID"].duplicated().any(), "duplicate STAIDs in GAGES-II.csv"
n_total = len(df)

df["REL_DIFF"] = df["ABS_DIFF"] / df["DRAIN_SQKM"]
old_valid = df["DA_VALID"].astype(str).str.lower().eq("true")
df["DA_VALID"] = df["REL_DIFF"] <= REL_TOL
print(f"GAGES-II rows: {n_total}")
print(f"DA_VALID old (absolute criterion): {old_valid.sum()}")
print(f"DA_VALID new (rel <= {REL_TOL:.0%}):   {df['DA_VALID'].sum()}"
      f"  (recovered {(df['DA_VALID'] & ~old_valid).sum()},"
      f" dropped {(~df['DA_VALID'] & old_valid).sum()})")

pool = df[df["DA_VALID"]].copy()

# ---- (c) part 1: observation coverage in both windows --------------------
storage = icechunk.local_filesystem_storage(OBS_STORE)
repo = icechunk.Repository.open(storage)
sess = repo.readonly_session("main")
ds = xr.open_zarr(sess.store, consolidated=False)

in_obs = pool["STAID"].isin(set(ds["gage_id"].values.astype(str)))
print(f"after DA_VALID, in observation store: {in_obs.sum()} / {len(pool)}")
pool = pool[in_obs].copy()

sf = ds["streamflow"].sel(gage_id=pool["STAID"].values)
frac_train = sf.sel(time=slice(*TRAIN_WINDOW)).notnull().mean("time").compute()
frac_eval = sf.sel(time=slice(*EVAL_WINDOW)).notnull().mean("time").compute()
pool["COV_TRAIN"] = frac_train.values
pool["COV_EVAL"] = frac_eval.values

print("\ncoverage sensitivity (gauges meeting the bar in BOTH windows):")
for thr in (0.0, 0.5, 0.8, 0.9, 0.95, 1.0):
    n = ((pool["COV_TRAIN"] > thr) & (pool["COV_EVAL"] > thr)).sum() if thr == 0.0 else \
        ((pool["COV_TRAIN"] >= thr) & (pool["COV_EVAL"] >= thr)).sum()
    print(f"  >= {thr:>4.0%}: {n}")

cov_ok = (pool["COV_TRAIN"] >= COVERAGE_MIN) & (pool["COV_EVAL"] >= COVERAGE_MIN)
print(f"applying bar {COVERAGE_MIN:.0%}: {cov_ok.sum()} / {len(pool)}")
pool = pool[cov_ok].copy()

# ---- (c) part 2: non-headwater subgraph exists ---------------------------
adj = zarr.open_group(ADJ_STORE, mode="r")
keep = []
n_missing = n_headwater = 0
for staid in pool["STAID"]:
    try:
        sub = adj[staid]
    except KeyError:
        n_missing += 1
        keep.append(False)
        continue
    if sub["order"].shape[0] <= 1:  # GageSubgraph::is_headwater (zarr.rs:128)
        n_headwater += 1
        keep.append(False)
    else:
        keep.append(True)
print(f"adjacency filter: kept {sum(keep)}"
      f" (dropped {n_missing} missing, {n_headwater} headwater)")
pool = pool[keep].copy()

# ---- (d): area-balanced subsample ----------------------------------------
rng = np.random.default_rng(SEED)
small = pool[pool["DRAIN_SQKM"] < 1000]
mid = pool[(pool["DRAIN_SQKM"] >= 1000) & (pool["DRAIN_SQKM"] < 5000)]
big = pool[pool["DRAIN_SQKM"] >= 5000]
print(f"\neligible pool: {len(pool)}  (<1k: {len(small)},"
      f" 1k-5k: {len(mid)}, >=5k: {len(big)})")

n_mid = min(len(mid), max(0, N_LARGE - len(big)))
n_small = min(len(small), N_SMALL)
sel = pd.concat([
    big,
    mid.iloc[rng.permutation(len(mid))[:n_mid]],
    small.iloc[rng.permutation(len(small))[:n_small]],
]).sort_values("STAID")

cols = ["STAID", "STANAME", "DRAIN_SQKM", "LAT_GAGE", "LNG_GAGE", "COMID",
        "COMID_DRAIN_SQKM", "COMID_UNITAREA_SQKM", "ABS_DIFF", "DA_VALID",
        "FLOW_SCALE", "REL_DIFF", "COV_TRAIN", "COV_EVAL"]
out = sel[cols].copy()
out["DA_VALID"] = "True"
out.to_csv(OUT_CSV, index=False)
print(f"\nwrote {len(out)} gauges -> {OUT_CSV}")

# ---- report --------------------------------------------------------------
bins = [0, 100, 500, 1000, 5000, 10000, 30000, np.inf]
labels = ["<100", "100-500", "500-1k", "1k-5k", "5k-10k", "10k-30k", ">=30k"]
hist = pd.cut(out["DRAIN_SQKM"], bins=bins, labels=labels).value_counts().reindex(labels)
print("\ndrainage-area histogram (km2):")
for lab, n in hist.items():
    print(f"  {lab:>8}: {n:4d}  ({n / len(out):.1%})")
n_below = (out["DRAIN_SQKM"] < 1000).sum()
print(f"\n<1,000 km2: {n_below} ({n_below/len(out):.1%})"
      f"  >=1,000 km2: {len(out)-n_below} ({(len(out)-n_below)/len(out):.1%})")
print(f">=5,000 km2 kept in full: {len(big)}")
print(f"all {len(out)} gauges have >= {COVERAGE_MIN:.0%} obs coverage in both"
      f" {TRAIN_WINDOW} and {EVAL_WINDOW}")

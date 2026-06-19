# Reference: parameter map plot

Spatial map of a learned KAN parameter (`n`, `q_spatial`, `p_spatial`, or `slope`) colored over MERIT-Hydro catchment polygons with a basemap. Direct port of DDR's `param_plot` from `~/projects/ddr/examples/merit/plot_parameter_map.ipynb`, adapted to ddrs's `dump_parameters` output schema.

## Inputs

### Parameter NetCDF (from `cargo run --bin dump_parameters`)

Schema is intentionally identical to the KAN subset of DDR's `merit_geometry_predictions.nc` (see `src/bin/dump_parameters.rs:1-32`):

- dim `COMID` — int64 reach identifier (MERIT-Hydro)
- var `n` — Manning's n (m⁻¹/³ s, range [0.015, 0.25])
- var `q_spatial` — Leopold & Maddock exponent (range [0, 1])
- var `p_spatial` — Leopold & Maddock coefficient (range [1, 200])
- var `slope` — channel bed slope (m/m, clamped ≥0.001)

### MERIT-Hydro fabric (external)

**Pick the fabric that matches the run's `data_sources` (read `config.yaml` in the run dir).** Two cases:

**CONUS shapefile** (`geodataset: merit` CONUS-only runs) — per `~/projects/ddr/examples/merit/README.md`:
- Path: `~/projects/ddr/data/merit/cat_pfaf_*_MERIT_Hydro_v07_Basins_v01_bugfix1.shp` (catchment polygons)
- The CONUS subset covers COMIDs 71000001-78028489 (346,321 reaches).
- Multiple Pfafstetter L2 shapefiles tile CONUS. For a small area subset, load the one(s) covering the bounding box.
- Source: <https://www.reachhydro.org/home/params/merit-basins>
- Load with `gpd.read_file(...)`, join with `np.intersect1d` (template below).

**Global GeoPackage** (runs whose `config.yaml` sets `geospatial_fabric: .../global_merit_riv.gpkg`) — use the **global template** in §"Global-fabric runs", NOT the CONUS template:
- Path: `/projects/mhpi/data/MERIT/raw/global_merit_riv.gpkg`, layer `flowlines` (LINESTRING, EPSG:4326), 2,939,408 reaches.
- It's ~6.4 GB, so **read with `pyogrio.read_dataframe(GPKG, layer="flowlines", columns=["COMID", "uparea"])`** — column pushdown loads only what you join + filter on (a few minutes, once). `gpd.read_file` on the whole file is far slower.
- These are flowlines (lines), not catchment polygons — plot with thin `linewidth`, not filled polygons.
- `dump_parameters` on a global run writes ~2.94M COMIDs; the join covers the whole planet, so always produce a CONUS-bbox map **and** a `uparea`-filtered global map (headwater reaches are sub-pixel at world scale).

### User-supplied selection

- **variable** — one of `n`, `q_spatial`, `p_spatial`, `slope`. Default: `n` (Manning's, the most physically interpretable).
- **region**: one of
  - Bounding box `(min_lon, min_lat, max_lon, max_lat)`
  - Named region (CONUS, Northeast, Pacific Northwest, etc.) — translate to bounding box
  - List of COMIDs (e.g., a single basin or HUC)
  - Single gauge STAID — pull contributing COMIDs from `~/projects/ddr/data/merit_gages_conus_adjacency.zarr` (group `<STAID>/comids`)

## Notebook template

```python
from pathlib import Path

import contextily as cx
import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
import xarray as xr
from mpl_toolkits.axes_grid1 import make_axes_locatable

# --- USER INPUTS ---------------------------------------------------------
PARAMS_NC = Path("/home/tbindas/projects/ddrs/output/kan_parameters_latest.nc")
CKPT_DIR  = Path("/home/tbindas/projects/ddrs/output/saved_models_1")
SHAPEFILE = Path("/home/tbindas/projects/ddr/data/merit/cat_pfaf_7_MERIT_Hydro_v07_Basins_v01_bugfix1.shp")
VARIABLE  = "n"
BBOX      = (-125, 24, -66, 53)   # CONUS; set tighter for a region
REGION_LABEL = "CONUS"
# -------------------------------------------------------------------------

PLOT_DIR = CKPT_DIR / "plots"
PLOT_DIR.mkdir(exist_ok=True)

# Load and join
ds  = xr.open_dataset(PARAMS_NC)
gdf = gpd.read_file(SHAPEFILE).set_index("COMID")
shared = np.intersect1d(gdf.index.values, ds.COMID.values)
ds_subset = ds.sel(COMID=shared)
gdf.loc[shared, VARIABLE] = ds_subset[VARIABLE].values
gdf = gdf.set_crs(epsg=4326)

# Region filter (bbox)
xmin, ymin, xmax, ymax = BBOX
gdf = gdf.cx[xmin:xmax, ymin:ymax]
gdf_clean = gdf.dropna(subset=[VARIABLE]).sort_values(VARIABLE, ascending=True)
if gdf_clean.empty:
    raise ValueError(f"No reaches with {VARIABLE} in bbox {BBOX}")

# Plot
PLOT_CONFIGS = {
    "n":         {"title": "Manning's Roughness", "unit": "m⁻¹/³ s", "cmap": "plasma_r", "vmax": 0.2},
    "q_spatial": {"title": "Width-Depth Exponent (q)", "unit": "–", "cmap": "viridis"},
    "p_spatial": {"title": "Width Coefficient (p)",   "unit": "–", "cmap": "viridis"},
    "slope":     {"title": "Channel Bed Slope",       "unit": "m/m", "cmap": "magma"},
}
cfg = PLOT_CONFIGS[VARIABLE]

fig, ax = plt.subplots(figsize=(10, 6), dpi=150)
vmin = float(np.nanmin(gdf_clean[VARIABLE]))
vmax = cfg.get("vmax", float(np.nanmax(gdf_clean[VARIABLE])))

gdf_clean.plot(ax=ax, column=VARIABLE, cmap=cfg["cmap"],
               linewidth=0.3, vmin=vmin, vmax=vmax, zorder=1)
cx.add_basemap(ax, crs=gdf_clean.crs, source=cx.providers.CartoDB.Positron,
               alpha=0.6, zorder=0, attribution=False)

ax.set_xlim(xmin, xmax); ax.set_ylim(ymin, ymax)
ax.set_xticks([]); ax.set_yticks([])
ax.set_title(f"{cfg['title']} — {REGION_LABEL}", fontsize=14)

divider = make_axes_locatable(ax)
cax = divider.append_axes("right", size="3%", pad=0.1)
sm = plt.cm.ScalarMappable(cmap=cfg["cmap"])
sm.set_array([])
sm.set_clim(vmin, vmax)
cbar = fig.colorbar(sm, cax=cax)
cbar.set_label(f"{VARIABLE} ({cfg['unit']})")

plt.tight_layout()
out = PLOT_DIR / f"parameter_map_{VARIABLE}_{REGION_LABEL.lower().replace(' ', '_')}.png"
fig.savefig(out, dpi=300, bbox_inches="tight", facecolor="white")
print(f"saved {out}")
```

## Global-fabric runs

When `config.yaml` points at the global gpkg (`geospatial_fabric: .../global_merit_riv.gpkg`), the CONUS shapefile + `gpd.read_file` + `np.intersect1d` recipe above does NOT apply — use this instead. A working, executed instance lives at
`.ddrs/runs/2026-06-12T01-22-03Z-train-and-test/plots/parameter_map_n_global_conus.ipynb`; the generator that produced the same notebook for a later run is the canonical source. Key differences from the CONUS template:

- **`pyogrio.read_dataframe(GPKG, layer="flowlines", columns=["COMID", "uparea"])`** with column pushdown — never `gpd.read_file` the whole 6.4 GB file.
- **Join by `pd.Series(...).reindex(gdf.index)`**, not `np.intersect1d` + `.sel` + `.loc` — both the NetCDF `COMID` axis and the gpkg index are unique COMID labels, so a reindex is exact and avoids the O(n) intersect over 2.94M reaches.
- **Subset the GeoDataFrame BEFORE plotting** — `gdf.cx[xmin:xmax, ymin:ymax]` for the CONUS bbox, `gdf[gdf["uparea"] >= 100]` for the global map. Setting axis limits alone still renders all 2.94M geometries (minutes per plot, or OOM). Subset once, reuse across variables.
- **Two maps per variable**: CONUS bbox + global (uparea-filtered). Headwater reaches are sub-pixel at world scale.
- **Map color ceiling**: use the YAML `vmax` for `n` (0.2, matches DDR); for `q_spatial`/`p_spatial`, whose realized values sit well inside the declared range (e.g. log-space `p_spatial` realizes ~1.4–14 vs range [1, 200]), drive `vmax` off the data 98th percentile so structure is visible. **Histogram x-axis still anchors to the YAML `parameter_ranges`** so a collapsed-at-the-bound pathology stays legible.

### Producing `kan_parameters.nc` for a managed-adjacency run

Global runs almost always use **managed adjacency** — `config.yaml` sets
`geospatial_fabric:` instead of explicit `conus_adjacency:`/`gages_adjacency:`
zarr paths, and `ddrs plan` builds the stores into `.ddrs/adjacency/<key>/`.
The standalone `dump_parameters` binary does **not** resolve managed adjacency,
so pointing it at such a config fails before writing anything:

```
Error: ConfigInvalid { ... "conus_adjacency not resolved — invoke via
`ddrs run --plot` (which resolves adjacency), or set
conus_adjacency/gages_adjacency explicitly" }
```

`ddrs run --plot` is the wrong fix here — it kicks off a **fresh full workflow**
(train-and-test = train + eval) just to dump params, and dumps from the *new*
run's checkpoints, not the existing one. Instead, take the adjacency paths the
original run already resolved and feed them to `dump_parameters` via a throwaway
config copy:

```bash
RUN=.ddrs/runs/<run-id>
# 1. Pull the resolved adjacency zarr paths the run recorded at plan time.
python - "$RUN/manifest.json" <<'PY'
import json, sys
ra = json.load(open(sys.argv[1]))["resolved_adjacency"]
print(ra["conus"]); print(ra["gages"])
PY
# 2. Copy the run's config and append the two keys under data_sources:
#    (any indented key inside the data_sources: block works — e.g. after gages:)
#      conus_adjacency: <ra.conus>
#      gages_adjacency: <ra.gages>
cp "$RUN/config.yaml" /tmp/dump_cfg.yaml   # then edit in the two lines
# 3. dump_parameters now resolves adjacency from the explicit paths.
cargo run --release --bin dump_parameters -- \
  --config /tmp/dump_cfg.yaml \
  --checkpoint "$RUN/checkpoints/<epoch_E_mb_M>/head" \
  --output "$RUN/kan_parameters.nc"
```

The checkpoint base is the predictions zarr's `model` attr without `.mpk`
(`<run-id>/checkpoints/epoch_E_mb_M/head`). The dump streams the fabric in
50k-reach batches (~2.94M reaches for the global fabric, ~1 min on GPU once the
binary is built) and writes `n`, `q_spatial`, `p_spatial`, `slope`
(plus `x_storage` once that learnable parameter lands). The patched config is
disposable — never commit explicit adjacency paths back into the run's
`config.yaml`.

### Global-fabric notebook template

```python
from pathlib import Path
import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pyogrio
import xarray as xr
from mpl_toolkits.axes_grid1 import make_axes_locatable

RUN_DIR   = Path("/projects/mhpi/tbindas/ddrs/.ddrs/runs/<run-id>")
PARAMS_NC = RUN_DIR / "kan_parameters.nc"
GPKG      = Path("/projects/mhpi/data/MERIT/raw/global_merit_riv.gpkg")
PLOT_DIR  = RUN_DIR / "plots"; PLOT_DIR.mkdir(exist_ok=True)

CONUS_BBOX, GLOBAL_BBOX = (-125, 24, -66, 53), (-180, -60, 180, 85)
GLOBAL_MIN_UPAREA_KM2 = 100.0
PARAM_CFG = {
    "n":         {"title": "Manning's Roughness", "unit": "m$^{-1/3}$ s", "range": (0.015, 0.25), "cmap": "plasma_r", "vmax_map": 0.2},
    "q_spatial": {"title": "Width-Depth Exponent (q)", "unit": "–", "range": (0.0, 1.0),   "cmap": "viridis", "vmax_map": None},
    "p_spatial": {"title": "Width Coefficient (p)",     "unit": "–", "range": (1.0, 200.0), "cmap": "viridis", "vmax_map": None},
}
VARIABLES = ["n", "q_spatial", "p_spatial"]   # the run's learnable_parameters

ds = xr.open_dataset(PARAMS_NC)
gdf = pyogrio.read_dataframe(GPKG, layer="flowlines", columns=["COMID", "uparea"]).set_index("COMID")
for v in VARIABLES:
    gdf[v] = pd.Series(ds[v].values, index=ds["COMID"].values).reindex(gdf.index).values

def _clim(g, var, cfg):
    lo = cfg["range"][0]
    if cfg["vmax_map"] is not None: return lo, cfg["vmax_map"]
    finite = g[var].dropna().values
    return (float(np.nanmin(finite)) if finite.size else lo,
            float(np.nanpercentile(finite, 98)) if finite.size else cfg["range"][1])

def plot_param_map(g, var, bbox, label, linewidth, fname):  # g is pre-subset
    cfg = PARAM_CFG[var]; g = g.dropna(subset=[var]).sort_values(var)
    vmin, vmax = _clim(g, var, cfg)
    fig, ax = plt.subplots(figsize=(14, 8), dpi=150)
    g.plot(ax=ax, column=var, cmap=cfg["cmap"], linewidth=linewidth, vmin=vmin, vmax=vmax, zorder=1)
    ax.set_xlim(bbox[0], bbox[2]); ax.set_ylim(bbox[1], bbox[3])
    try:
        import contextily as cx
        cx.add_basemap(ax, crs=g.crs, source=cx.providers.CartoDB.Positron, alpha=0.6, zorder=0, attribution=False)
    except Exception as e:
        print(f"basemap skipped ({type(e).__name__}: {e})"); ax.set_facecolor("#f0f0f0")
    ax.set_xticks([]); ax.set_yticks([]); ax.set_title(f"{cfg['title']} - {label}", fontsize=14)
    cax = make_axes_locatable(ax).append_axes("right", size="3%", pad=0.1)
    sm = plt.cm.ScalarMappable(cmap=cfg["cmap"]); sm.set_array([]); sm.set_clim(vmin, vmax)
    fig.colorbar(sm, cax=cax).set_label(f"{var} ({cfg['unit']})")
    fig.savefig(PLOT_DIR / fname, dpi=300, bbox_inches="tight", facecolor="white"); plt.close(fig)

conus = gdf.cx[CONUS_BBOX[0]:CONUS_BBOX[2], CONUS_BBOX[1]:CONUS_BBOX[3]]
big   = gdf[gdf["uparea"] >= GLOBAL_MIN_UPAREA_KM2]
for var in VARIABLES:
    plot_param_map(conus, var, CONUS_BBOX, "CONUS", 0.3, f"parameter_map_{var}_conus.png")
    plot_param_map(big, var, GLOBAL_BBOX, f"Global (uparea >= {GLOBAL_MIN_UPAREA_KM2:g} km² shown)", 0.15, f"parameter_map_{var}_global.png")
    # histogram over the FULL population (ds[var]), x-axis anchored to PARAM_CFG[var]["range"] — see Companion cells below
```

## Companion cells

Emit these alongside the map whenever the user wants a "full look" at a learned parameter — they live in the same notebook as the map so all three views travel together. Both reuse `ds`, `VARIABLE`, `REGION_LABEL`, and `PLOT_DIR` from the cells above.

### Distribution histogram

x-axis = parameter value over the full reach population (NOT just polygons that survived the bbox/shapefile filter — we want the model's learned distribution, not a regional slice). Vertical lines mark median and mean so reviewers can eyeball skew.

```python
# --- Histogram of learned parameter across the full reach population ----
v_all = ds[VARIABLE].values
v_finite = v_all[np.isfinite(v_all)]
# Use the YAML-declared range for the x-axis so under-trained runs make
# their "all at the lower bound" pathology visible.
PARAM_RANGES = {"n": (0.015, 0.25), "q_spatial": (0.0, 1.0), "p_spatial": (1.0, 200.0)}
vmin_hist, vmax_hist = PARAM_RANGES.get(VARIABLE, (float(v_finite.min()), float(v_finite.max())))

fig, ax = plt.subplots(figsize=(10, 5), dpi=150)
ax.hist(v_finite, bins=80, range=(vmin_hist, vmax_hist),
        color="#6c2178", edgecolor="white", linewidth=0.3)
ax.axvline(float(np.nanmedian(v_finite)), color="black", linestyle="--",
           linewidth=1.5, label=f"median = {float(np.nanmedian(v_finite)):.4f}")
ax.axvline(float(np.nanmean(v_finite)),   color="#c63",  linestyle=":",
           linewidth=1.5, label=f"mean   = {float(np.nanmean(v_finite)):.4f}")
ax.set_xlabel(f"{VARIABLE} ({cfg['unit']})")
ax.set_ylabel(f"reach count  (total = {len(v_finite):,})")
ax.set_title(f"Distribution of learned {VARIABLE} - {REGION_LABEL}")
ax.set_xlim(vmin_hist, vmax_hist)
ax.legend(loc="upper right", frameon=True)
ax.grid(axis="y", alpha=0.3)
out_hist = PLOT_DIR / f"parameter_hist_{VARIABLE}_{REGION_LABEL.lower().replace(' ', '_')}.png"
fig.savefig(out_hist, dpi=300, bbox_inches="tight", facecolor="white")
print(f"saved {out_hist}")
```

### Parameter vs log10(drainage area) hexbin

The KAN takes `log10_uparea` as input, so this scatter is the obvious sanity check — does the learned parameter actually depend on drainage area? `log10_uparea` lives in `merit_global_attributes_v2.nc` already log-transformed. With ~300k reaches a raw scatter is just black, so use hexbin density with a median-per-bin overlay.

```python
# --- Scatter: learned parameter vs log10(drainage area) ----------------
ATTRS_NC = Path("/home/tbindas/projects/ddr/data/merit_global_attributes_v2.nc")
attrs = xr.open_dataset(ATTRS_NC)
shared = np.intersect1d(attrs.COMID.values, ds.COMID.values)
print(f"joined {len(shared):,} COMIDs (of {ds.sizes['COMID']:,} predicted, "
      f"{attrs.sizes['COMID']:,} in attributes)")

attrs_s = attrs.sel(COMID=shared)
ds_s    = ds.sel(COMID=shared)
log_da  = attrs_s["log10_uparea"].values
y_vals  = ds_s[VARIABLE].values
mask    = np.isfinite(log_da) & np.isfinite(y_vals)
log_da, y_vals = log_da[mask], y_vals[mask]

fig, ax = plt.subplots(figsize=(10, 6), dpi=150)
hb = ax.hexbin(log_da, y_vals, gridsize=80, cmap="viridis", mincnt=1,
               extent=(log_da.min(), log_da.max(), vmin_hist, vmax_hist))
fig.colorbar(hb, ax=ax, label="reach count per hex")
ax.set_xlabel(r"$\log_{10}$(drainage area, km$^2$)")
ax.set_ylabel(f"learned {VARIABLE} ({cfg['unit']})")
ax.set_title(f"{VARIABLE} vs drainage area - {REGION_LABEL}  ({len(y_vals):,} reaches)")
ax.set_ylim(vmin_hist, vmax_hist)
ax.grid(alpha=0.3)

# Median-per-bin overlay so the trend reads through the density.
bin_edges = np.linspace(log_da.min(), log_da.max(), 21)
bin_idx   = np.digitize(log_da, bin_edges) - 1
med = np.array([
    np.nanmedian(y_vals[bin_idx == b]) if np.any(bin_idx == b) else np.nan
    for b in range(len(bin_edges) - 1)
])
bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])
ax.plot(bin_centers, med, color="#ff4500", lw=2.0, label=f"median {VARIABLE} per bin")
ax.legend(loc="upper right", frameon=True)
out_sc = PLOT_DIR / f"parameter_scatter_{VARIABLE}_vs_log10_uparea_{REGION_LABEL.lower().replace(' ', '_')}.png"
fig.savefig(out_sc, dpi=300, bbox_inches="tight", facecolor="white")
print(f"saved {out_sc}")
```

## Notes

- **Shapefile is large.** A full Pfafstetter-7 shapefile is several GB. For small-region plots, filter before plotting (`.cx[xmin:xmax, ymin:ymax]` is fast — uses the spatial index).
- **Multiple Pfafstetter tiles for full CONUS.** If the bbox crosses Pfafstetter boundaries, load and concatenate the relevant tiles. For pure CONUS, `cat_pfaf_7_*` covers most of it; check the DDR data dir for which tiles are available.
- **Why `plasma_r` for Manning's n?** High n = rough = darker. DDR uses the reversed plasma in `plot_parameter_map.ipynb` for this reason.
- **`gdf_clean.sort_values(ascending=True)`** — geopandas draws in row order. Sorting ascending puts high-value polygons on top, so outliers stand out.
- **CRS**: MERIT shapefiles are EPSG:4326 (lat/lon). Don't reproject before plotting — `contextily` will fetch tiles matching the geo coords.
- **Joining to a gauge**: use `~/projects/ddr/data/merit_gages_conus_adjacency.zarr/<STAID>/comids` to get the contributing COMIDs for a specific gauge, then filter `ds.sel(COMID=...)` before plotting. Compute the bbox from the polygons' total extent.
- **Histogram x-axis uses the YAML-declared parameter range, not data min/max.** If an under-trained model collapses every reach near the lower bound (n ≈ 0.015), a data-driven x-axis would zoom in and hide the pathology. Anchoring the x-axis to `parameter_ranges` makes "the model hasn't learned much yet" immediately legible.
- **Scatter pulls drainage area from the global attributes NetCDF**, not from the predictions NetCDF. `dump_parameters` writes `n / q_spatial / p_spatial / slope` only — `log10_uparea` is a model INPUT and lives in `merit_global_attributes_v2.nc`. Join on COMID; ddrs is a subset (CONUS only), the attributes file is global, so use `np.intersect1d`.
- **Hexbin not scatter** for the parameter-vs-area plot. 300k points as a raw `ax.scatter` is solid black even at `alpha=0.01`; hexbin with `mincnt=1` reveals the density structure and an overlaid `median-per-bin` line shows the trend. If a user has a small region (<5k reaches) and asks for a scatter explicitly, falling back to `ax.scatter(..., s=4, alpha=0.3)` is fine.

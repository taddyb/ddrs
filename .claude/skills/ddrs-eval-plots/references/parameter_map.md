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

### MERIT-Hydro shapefile (external)

Per `~/projects/ddr/examples/merit/README.md`:
- Path: `~/projects/ddr/data/merit/cat_pfaf_*_MERIT_Hydro_v07_Basins_v01_bugfix1.shp`
- The CONUS subset covers COMIDs 71000001-78028489 (346,321 reaches).
- Multiple Pfafstetter L2 shapefiles tile CONUS. For a small area subset, load the one(s) covering the bounding box.
- Source: <https://www.reachhydro.org/home/params/merit-basins>

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

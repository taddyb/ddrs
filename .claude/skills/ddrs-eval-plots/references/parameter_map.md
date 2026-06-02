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

## Notes

- **Shapefile is large.** A full Pfafstetter-7 shapefile is several GB. For small-region plots, filter before plotting (`.cx[xmin:xmax, ymin:ymax]` is fast — uses the spatial index).
- **Multiple Pfafstetter tiles for full CONUS.** If the bbox crosses Pfafstetter boundaries, load and concatenate the relevant tiles. For pure CONUS, `cat_pfaf_7_*` covers most of it; check the DDR data dir for which tiles are available.
- **Why `plasma_r` for Manning's n?** High n = rough = darker. DDR uses the reversed plasma in `plot_parameter_map.ipynb` for this reason.
- **`gdf_clean.sort_values(ascending=True)`** — geopandas draws in row order. Sorting ascending puts high-value polygons on top, so outliers stand out.
- **CRS**: MERIT shapefiles are EPSG:4326 (lat/lon). Don't reproject before plotting — `contextily` will fetch tiles matching the geo coords.
- **Joining to a gauge**: use `~/projects/ddr/data/merit_gages_conus_adjacency.zarr/<STAID>/comids` to get the contributing COMIDs for a specific gauge, then filter `ds.sel(COMID=...)` before plotting. Compute the bbox from the polygons' total extent.

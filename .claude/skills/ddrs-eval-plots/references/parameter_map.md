# Reference: parameter map plot

Spatial map of a learned KAN parameter (`n`, `q_spatial`, `p_spatial`, `x_storage`, or `slope`) colored over MERIT-Hydro catchment polygons with a basemap. Direct port of DDR's `param_plot` from `~/projects/ddr/examples/merit/plot_parameter_map.ipynb`, adapted to ddrs's `dump_parameters` output schema.

## Contents

| Section | Use it when |
|---|---|
| §Inputs | Always — parameter NetCDF schema, MERIT fabric, what you must choose |
| §Notebook template | CONUS run, standard case |
| §Global-fabric runs | The run used `geospatial_fabric` / managed adjacency, or a non-CONUS domain |
| §Global-fabric runs → Producing `kan_parameters.nc` | `dump_parameters` failed with `conus_adjacency not resolved` |
| §Companion cells → Distribution histogram | "Is the parameter pinned at a bound?" |
| §Companion cells → Parameter vs log10(drainage area) | "Did the model learn a scale relationship?" |
| **§Convergence** | **"Have the parameters converged?" / "Is this run under-trained?"** |
| §Notes | Colormaps, projection, joins, performance |

## Inputs

### Parameter NetCDF — `<RUN_DIR>/plot/kan_parameters.nc`

Written by `dump_parameters` (`src/dump_parameters.rs::write_netcdf`), either via `ddrs run --plot` or the standalone binary. `dump_parameters` **always** writes these five variables, whatever `kan_head.learnable_parameters` contains — a non-learnable entry falls back to its `params.defaults` / sigmoid-init value rather than being omitted:

- dim `COMID` — int64 reach identifier (MERIT-Hydro), 346,321 for CONUS
- var `n` — Manning's n (s/m^(1/3), range [0.015, 0.25])
- var `q_spatial` — Leopold & Maddock exponent (range [0, 1])
- var `p_spatial` — Leopold & Maddock coefficient (range [1, 200])
- var `x_storage` — Muskingum X storage weight (range [0, 0.5]; 0 = attenuation, 0.5 = pure lag; sigmoid init ≈ 0.25)
- var `slope` — channel bed slope (m/m, clamped to `attribute_minimums.slope`, ≥0.001)

Plus `K_D` (1/s), `d_gw` (m), and `leakance_factor` (dimensionless) **only** when the run has `params.use_leakance: true`.

Root attrs: `checkpoint` (the exact head base that was dumped), `ddrs_version`, `n_reaches`, `note`.

> **The zeta variables are in a different file on a different dimension.** `zeta`, `zeta_net`, `depth_mean`, `area_z_mean`, `q_mean` live in `<RUN_DIR>/kan_parameters.nc` (no `plot/`) on dim `COMID_eval` — 64,892 reaches, the gauge-subgraph union, **not** the 346,321-reach CONUS network. They are written by `eval --zeta-output` (or `train-and-test` Phase 2), never by `dump_parameters`: zeta needs the routed per-timestep depth, which only exists during eval. Do not `xr.open_dataset` one expecting the other's variables, and never plot the two on a shared COMID axis.

### MERIT-Hydro fabric (external)

**Pick the fabric that matches the run's `data_sources` (read `config.yaml` in the run dir).** Two cases:

**CONUS shapefile** (`geodataset: merit` CONUS-only runs) — per `~/projects/ddr/examples/merit/README.md`:
- Path: `~/projects/ddr/data/merit/cat_pfaf_*_MERIT_Hydro_v07_Basins_v01_bugfix1.shp` (catchment polygons)
- The CONUS subset covers COMIDs 71000001-78028489 (346,321 reaches).
- Multiple Pfafstetter L2 shapefiles tile CONUS. For a small area subset, load the one(s) covering the bounding box.
- Source: <https://www.reachhydro.org/home/params/merit-basins>
- Load with `gpd.read_file(...)`, join with `np.intersect1d` (template below).

**Global GeoPackage** (runs whose `config.yaml` sets `geospatial_fabric: .../global_merit_riv.gpkg`) — use the **global template** in §"Global-fabric runs", NOT the CONUS template:
- Path: `/projects/mhpi/data/MERIT/raw/global_merit_riv.gpkg`, layer `flowlines` (LINESTRING, EPSG:4326), 2,939,408 reaches. **HPC-only — this file is not present on this workstation.** Global-fabric plots have to run where the gpkg lives; don't offer them locally.
- It's ~6.4 GB, so **read with `pyogrio.read_dataframe(GPKG, layer="flowlines", columns=["COMID", "uparea"])`** — column pushdown loads only what you join + filter on (a few minutes, once). `gpd.read_file` on the whole file is far slower.
- These are flowlines (lines), not catchment polygons — plot with thin `linewidth`, not filled polygons.
- `dump_parameters` on a global run writes ~2.94M COMIDs; the join covers the whole planet, so always produce a CONUS-bbox map **and** a `uparea`-filtered global map (headwater reaches are sub-pixel at world scale).

### User-supplied selection

- **variable** — one of `n`, `q_spatial`, `p_spatial`, `x_storage`, `slope`. Default: `n` (Manning's, the most physically interpretable).
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
RUN_DIR   = Path("/home/tbindas/projects/ddrs/.ddrs/runs/<id>")
PARAMS_NC = RUN_DIR / "plot" / "kan_parameters.nc"
PLOT_DIR  = RUN_DIR / "plots"
SHAPEFILE = Path("/home/tbindas/projects/ddr/data/merit/cat_pfaf_7_MERIT_Hydro_v07_Basins_v01_bugfix1.shp")
VARIABLE  = "n"
BBOX      = (-125, 24, -66, 53)   # CONUS; set tighter for a region
REGION_LABEL = "CONUS"
# -------------------------------------------------------------------------

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
    "x_storage": {"title": "Muskingum X Storage Weight", "unit": "–", "cmap": "coolwarm", "vmax": 0.5},
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
  --output "$RUN/plot/kan_parameters.nc"
```

The checkpoint base is the predictions zarr's `model` attr without `.mpk`
(`<run-id>/checkpoints/epoch_E_mb_M/head`). The dump streams the fabric in
50k-reach batches (~2.94M reaches for the global fabric, ~1 min on GPU once the
binary is built) and writes `n`, `q_spatial`, `p_spatial`, `x_storage`, `slope`
(plus `K_D`, `d_gw`, `leakance_factor` for a leakance run). The patched config is
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

RUN_DIR   = Path("/projects/mhpi/tbindas/ddrs/.ddrs/runs/<run-id>")   # HPC path
PARAMS_NC = RUN_DIR / "plot" / "kan_parameters.nc"
GPKG      = Path("/projects/mhpi/data/MERIT/raw/global_merit_riv.gpkg")  # HPC-only
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

## Convergence: has training actually moved the parameters?

A single-epoch map answers "what did it learn". This family answers "was it done learning". It is the standard follow-up to any parameter map, and the honest answer for ddrs runs so far has been *no*.

### Step 1 — produce per-epoch dumps

`dump_parameters` reads one checkpoint at a time, so pick three or more epochs spanning the run (first, middle, last) and dump each into its own NetCDF. `--checkpoint` takes the **head base** (`.../head`, no `.mpk`), and the dumped file records it in the `checkpoint` root attr, so the provenance is self-describing.

```bash
RUN=/home/tbindas/projects/ddrs/.ddrs/runs/<id>
mkdir -p "$RUN/plot"
for E in 1 15 30; do
  cargo run --release --bin dump_parameters -- \
    --config "$RUN/config.yaml" \
    --checkpoint "$RUN/checkpoints/epoch_${E}_mb_0/head" \
    --output "$RUN/plot/kan_parameters_epoch${E}.nc"
done
```

The mini-batch suffix is whatever the run wrote — `ls "$RUN/checkpoints"` first; all-basin accumulated runs end each epoch at `mb_0`. If the config uses managed adjacency, apply the throwaway-config patch from §"Producing `kan_parameters.nc` for a managed-adjacency run" to every dump. Each dump is a GPU job over the full fabric (~1 min, ~10 MB out for CONUS).

> The 2026-07-30 CONUS run (`2026-07-30T01-58-07Z-train-and-test`) wrote its per-epoch dumps at the **run root** — `kan_parameters_epoch1.nc`, `kan_parameters_epoch15.nc`, and epoch 30 as `kan_parameters.nc`. That collides with the eval zeta diagnostic's filename. Write new dumps under `plot/` and read the older run's from the root.

### Step 2 — the four diagnostics

Reported by the template below, and executed for real in `.ddrs/runs/2026-07-30T01-58-07Z-train-and-test/plots/parameter_maps.ipynb` (final cell) → `parameter_convergence_{n,q_spatial,p_spatial}.png` + `parameter_convergence_stats.json`.

1. **Late-half movement as a fraction of total movement** — `median|e_last − e_mid| / median|e_last − e_first|`. A converged run spends its late half barely moving, so this should be well under 0.5. Near or above 0.5 means the trajectory is still going.
2. **IQR across epochs** — contracting means reaches are agreeing on a value; **expanding** means the head is still spreading reaches apart, i.e. still differentiating them.
3. **Realized span as a % of the declared `parameter_ranges` span** — take p1–p99 of the last epoch over `hi − lo` from the run's own `config.yaml`. A few percent means the sigmoid output has barely left its initialization plateau.
4. **Fraction of reaches pinned at a bound** — within 1% of `lo` or `hi`. This is the *opposite* failure: saturation. Diagnostics 3 and 4 are both bad, and they are mutually exclusive, so always report both.

**Interpretation.** Expanding IQR *plus* a realized span of a few percent of the declared range means the parameters sit **near initialization** — the model is **UNDER-TRAINED, not converged**. Do not read "the distribution barely moved" as "the optimizer found its optimum". Measured on the 2026-07-30 30-epoch CONUS run: `n` late-half fraction 0.59, IQR 0.0011 → 0.0027 (expanding 2.4×), realized span 4.5% of [0.015, 0.25], 0.0% at a bound. `q_spatial` 3.1% span, `p_spatial` 1.7%. That run had not converged.

### Template

Add this as a final cell to the parameter-map notebook so all views travel together. It reuses `RUN_DIR`, `VARIABLES`, and `PLOT_DIR`.

```python
import json
import numpy as np
import matplotlib.pyplot as plt
import xarray as xr
import yaml

# Per-epoch dumps: first, middle, last. Point at wherever Step 1 wrote them.
EPOCH_NCS = {
    1:  RUN_DIR / "plot" / "kan_parameters_epoch1.nc",
    15: RUN_DIR / "plot" / "kan_parameters_epoch15.nc",
    30: RUN_DIR / "plot" / "kan_parameters.nc",
}
BOUND_TOL = 0.01   # "at a bound" = within 1% of the declared range

# Declared ranges come from the run's OWN config, not a hardcoded dict.
ranges = yaml.safe_load((RUN_DIR / "config.yaml").read_text())["params"]["parameter_ranges"]

eds    = {ep: xr.open_dataset(p) for ep, p in EPOCH_NCS.items()}
epochs = sorted(eds)
e_first, e_mid, e_last = epochs[0], epochs[len(epochs) // 2], epochs[-1]

stats = {}
for var in VARIABLES:
    lo, hi = ranges[var]
    span   = hi - lo
    vals   = {ep: eds[ep][var].values for ep in epochs}

    d_total = np.abs(vals[e_last] - vals[e_first])
    d_late  = np.abs(vals[e_last] - vals[e_mid])
    iqr = {ep: float(np.nanpercentile(vals[ep], 75) - np.nanpercentile(vals[ep], 25))
           for ep in epochs}

    v        = vals[e_last][np.isfinite(vals[e_last])]
    realized = float(np.nanpercentile(v, 99) - np.nanpercentile(v, 1))
    at_bound = float(np.mean((v <= lo + BOUND_TOL * span) | (v >= hi - BOUND_TOL * span)))

    stats[var] = {
        # 1. late-half movement / total movement
        "late-half movement fraction": float(np.nanmedian(d_late) / np.nanmedian(d_total)),
        f"median |e{e_last}-e{e_first}| (% of range)": float(np.nanmedian(d_total)) / span * 100,
        f"median |e{e_last}-e{e_mid}| (% of range)":   float(np.nanmedian(d_late))  / span * 100,
        f"p95 |e{e_last}-e{e_mid}| (% of range)":      float(np.nanpercentile(d_late, 95)) / span * 100,
        # 2. IQR contracting or expanding
        "IQR by epoch": iqr,
        "IQR trend": "expanding" if iqr[e_last] > iqr[e_first] else "contracting",
        # 3. realized span vs declared range
        "realized span (p1-p99, % of declared)": realized / span * 100,
        # 4. pinned at a bound
        "fraction at a bound": at_bound,
    }

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 5), dpi=150)
    shades = plt.cm.viridis(np.linspace(0.15, 0.85, len(epochs)))
    for c, ep in zip(shades, epochs):
        vv = vals[ep][np.isfinite(vals[ep])]
        ax1.hist(vv, bins=80, range=(lo, hi), histtype="step", linewidth=1.8, color=c,
                 label=f"epoch {ep} (median {np.nanmedian(vv):.4f}, IQR {iqr[ep]:.4f})")
    ax1.set_xlim(lo, hi)                      # anchored to the DECLARED range
    ax1.set_xlabel(var); ax1.set_ylabel("reach count")
    ax1.set_title(f"{var}: distribution by epoch")
    ax1.legend(frameon=True); ax1.grid(axis="y", alpha=0.3)

    ax2.hist(d_late[np.isfinite(d_late)] / span * 100, bins=80,
             color="#aa3333", edgecolor="white", linewidth=0.3)
    ax2.set_xlabel(f"|epoch {e_last} - epoch {e_mid}| as % of declared {var} range")
    ax2.set_ylabel("reach count")
    ax2.set_title(f"{var}: late-training per-reach movement")
    ax2.grid(axis="y", alpha=0.3)

    fig.savefig(PLOT_DIR / f"parameter_convergence_{var}.png",
                dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)

print(json.dumps(stats, indent=2))
(PLOT_DIR / "parameter_convergence_stats.json").write_text(json.dumps(stats, indent=2))
```

**Why the left panel's x-axis is the declared range, not the data range.** Auto-scaling to the data zooms into a 4%-wide sliver and makes three near-identical epoch curves look like meaningful separation. Anchoring to `parameter_ranges` shows the collapse at a glance — that is the whole point of the plot.

## Notes

- **Shapefile is large.** A full Pfafstetter-7 shapefile is several GB. For small-region plots, filter before plotting (`.cx[xmin:xmax, ymin:ymax]` is fast — uses the spatial index).
- **Multiple Pfafstetter tiles for full CONUS.** If the bbox crosses Pfafstetter boundaries, load and concatenate the relevant tiles. For pure CONUS, `cat_pfaf_7_*` covers most of it; check the DDR data dir for which tiles are available.
- **Why `plasma_r` for Manning's n?** High n = rough = darker. DDR uses the reversed plasma in `plot_parameter_map.ipynb` for this reason.
- **`gdf_clean.sort_values(ascending=True)`** — geopandas draws in row order. Sorting ascending puts high-value polygons on top, so outliers stand out.
- **CRS**: MERIT shapefiles are EPSG:4326 (lat/lon). Don't reproject before plotting — `contextily` will fetch tiles matching the geo coords.
- **Joining to a gauge**: use `~/projects/ddr/data/merit_gages_conus_adjacency.zarr/<STAID>/comids` to get the contributing COMIDs for a specific gauge, then filter `ds.sel(COMID=...)` before plotting. Compute the bbox from the polygons' total extent.
- **Histogram x-axis uses the YAML-declared parameter range, not data min/max.** If an under-trained model collapses every reach near the lower bound (n ≈ 0.015), a data-driven x-axis would zoom in and hide the pathology. Anchoring the x-axis to `parameter_ranges` makes "the model hasn't learned much yet" immediately legible.
- **Scatter pulls drainage area from the global attributes NetCDF**, not from the parameter NetCDF. `dump_parameters` writes `n / q_spatial / p_spatial / x_storage / slope` only — `log10_uparea` is a model INPUT and lives in `merit_global_attributes_v2.nc`. Join on COMID; ddrs is a subset (CONUS only), the attributes file is global, so use `np.intersect1d`.
- **Hexbin not scatter** for the parameter-vs-area plot. 300k points as a raw `ax.scatter` is solid black even at `alpha=0.01`; hexbin with `mincnt=1` reveals the density structure and an overlaid `median-per-bin` line shows the trend. If a user has a small region (<5k reaches) and asks for a scatter explicitly, falling back to `ax.scatter(..., s=4, alpha=0.3)` is fine.

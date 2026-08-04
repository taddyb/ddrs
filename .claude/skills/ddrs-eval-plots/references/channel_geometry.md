# Reference: baseflow channel geometry (width & depth) from learned parameters

Turns the learned KAN parameters into **physical channel width and depth at
baseflow** for all 346,321 CONUS MERIT reaches, maps them, and — the reason this
exists — tests whether the channel-geometry parameterisation reproduces observed
**downstream hydraulic geometry**. That test is the only internal check we have
on whether `p_spatial` / `q_spatial` are physically sensible, because the
attribute NetCDF carries **no width or depth variable** to validate against
(29 vars, none of them width/depth — checked 2026-08-03).

## The equations (mirrors `src/geometry.rs:37-67`)

Given Manning's `n`, Leopold-Maddock `p` and `q`, bed slope `S`, and discharge `Q`:

```
depth = ((Q · n · (q+1)) / (p · √S))^(3 / (5 + 3q))      geometry.rs:40-44
top_width  = p · depth^q                                  geometry.rs:47
side_slope = clamp(top_width · q / (2·depth), 0.5, 50)     geometry.rs:50
bottom_width = clamp(top_width − 2·side_slope·depth, 0.01) geometry.rs:53
area = (top_width + bottom_width) · depth / 2              geometry.rs:57
wetted_perimeter = bottom_width + 2·depth·√(1+z²)          geometry.rs:60
R = area / wetted_perimeter                                geometry.rs:64
velocity = n⁻¹ · R^(2/3) · √S                              geometry.rs:67
```

The depth relation is the exact inversion of Manning + the power-law width
profile `w(y) = p·y^q`, whose area is `A = p·d^(q+1)/(q+1)` — the `(q+1)` is that
normalisation, not a fudge factor.

**Use the post-clamp slope** from `plot/kan_parameters.nc`'s `slope` variable,
not the raw fabric slope: `attribute_minimums.slope` (default 1e-3) is what the
solver actually applied, and 33.2% of CONUS reaches are pinned at it.

## Choosing a baseflow discharge

There is no per-reach baseflow in the parameter NetCDF, so pick one explicitly
and **state the assumption in the notebook**:

```python
Q_SPEC = 0.005          # m3/s per km2 — CONUS-ish baseflow specific discharge
Q = Q_SPEC * 10**attrs["log10_uparea"]
```

`0.005` is roughly half the ~0.01 m³/s/km² CONUS *mean* specific discharge. Sweep
it (0.002 / 0.005 / 0.01) — the hydraulic-geometry **exponents below are
invariant to `Q_SPEC`** (it is a constant multiplier inside a log-log fit), so
only the absolute widths and depths move. That invariance is itself a useful
sanity check that the fit is wired correctly.

## The diagnostic that matters: downstream hydraulic geometry

Leopold & Maddock (1953) established that, moving *downstream* through a network:

```
w ∝ Q^b    b ≈ 0.50
d ∝ Q^f    f ≈ 0.40
v ∝ Q^m    m ≈ 0.10        with b + f + m = 1 identically
```

Fit `b` and `f` by least squares on `log10(w)` vs `log10(Q)` and `log10(d)` vs
`log10(Q)` across all reaches, then compare.

### Measured on run `2026-08-03T13-11-00Z` (346,321 reaches, `Q_SPEC = 0.005`)

```
                     fitted    Leopold & Maddock
width exponent  b     0.226           ~0.50
depth exponent  f     0.600           ~0.40
velocity     m=1-b-f  0.174           ~0.10

depth (m)   p5 0.058   p50 0.250   p95  3.733
width (m)   p5 4.560   p50 6.497   p95 21.997
w/d         p5 5.41    p50 26.6    p95 79.2
```

**The parameterisation is structurally incapable of reaching `b ≈ 0.50`.** With `p`
held constant, substituting `d ∝ Q^(3/(5+3q))` into `w = p·d^q` gives:

```
d exponent = 3/(5+3q)        w exponent = 3q/(5+3q)

  q = 0.3 :  d 0.508   w 0.153
  q = 0.5 :  d 0.462   w 0.231
  q = 1.0 :  d 0.375   w 0.375      <- q's upper bound
```

The width exponent **maxes out at 0.375 when `q = 1`**, below L&M's 0.50, and the
depth exponent bottoms out at 0.375, above L&M's 0.40 only for `q > 1`. Spatial
variation in `p` (learned, ρ ≈ +0.33 with `log10_uparea`) shifts the realised fit
but did not close the gap: 0.226 against 0.50.

**Interpretation.** Channels are modelled as too narrow and too deep, and
increasingly so downstream. That is not cosmetic — depth drives `R`, `R` drives
velocity, velocity drives celerity and hence `K = L/c` and the whole routing
timescale. A deep-narrow bias inflates `R` for a given area and therefore
inflates velocity.

**Before concluding the KAN mis-learned `q`:** the cap is a property of the
`w = p·d^q` form itself, not of the fitted values. Reaching `b ≈ 0.5` requires
either `q > 1` (outside `parameter_ranges.q_spatial: [0, 1]`) or a `p` that
grows with `Q` strongly enough to make up the difference. Widening the `q` range
is the cheap experiment; changing the width law is the real fix.

## Notebook template

```python
from pathlib import Path
import geopandas as gpd, matplotlib.pyplot as plt, numpy as np, pyogrio, xarray as xr
from mpl_toolkits.axes_grid1 import make_axes_locatable

RUN_DIR   = Path("/home/tbindas/projects/ddrs/.ddrs/runs/<id>")
PARAMS_NC = RUN_DIR / "plot" / "kan_parameters.nc"
ATTRS_NC  = Path("/home/tbindas/projects/ddr/data/merit_global_attributes_v2.nc")
SHAPEFILE = Path("/home/tbindas/projects/ddr/data/merit/"
                 "cat_pfaf_7_MERIT_Hydro_v07_Basins_v01_bugfix1.shp")
PLOT_DIR  = RUN_DIR / "plots"; PLOT_DIR.mkdir(exist_ok=True)
Q_SPEC    = 0.005      # m3/s per km2 — STATE THIS ASSUMPTION IN THE WRITEUP
CONUS_BBOX = (-125, 24, -66, 53)

ds, attrs = xr.open_dataset(PARAMS_NC), xr.open_dataset(ATTRS_NC)
sh, ia, ib = np.intersect1d(attrs.COMID.values, ds.COMID.values, return_indices=True)
A  = 10.0 ** attrs["log10_uparea"].values[ia]          # km2
Q  = Q_SPEC * A
n  = ds["n"].values[ib]
p  = ds["p_spatial"].values[ib]
q  = ds["q_spatial"].values[ib]
S  = np.maximum(ds["slope"].values[ib], 1e-3)          # POST-clamp slope
comid = ds.COMID.values[ib]

# --- geometry, mirroring src/geometry.rs exactly ---------------------------
qe    = q + 1e-6
depth = np.maximum(((Q * n * (qe + 1.0)) / (p * np.sqrt(S) + 1e-8)) ** (3.0 / (5.0 + 3.0 * qe)), 0.01)
width = p * depth ** qe
z     = np.clip(width * qe / (depth * 2.0), 0.5, 50.0)
bw    = np.maximum(width - z * depth * 2.0, 0.01)
area  = (width + bw) * depth / 2.0
wp    = bw + depth * np.sqrt(z ** 2 + 1.0) * 2.0
R     = area / wp
v     = (1.0 / n) * R ** (2.0 / 3.0) * np.sqrt(S)

# --- hydraulic-geometry exponents ------------------------------------------
ok = np.isfinite(depth) & np.isfinite(width) & (Q > 0)
lq = np.log10(Q[ok])
b_fit = np.polyfit(lq, np.log10(width[ok]), 1)[0]
f_fit = np.polyfit(lq, np.log10(depth[ok]), 1)[0]
print(f"b (width) {b_fit:.3f} vs L&M 0.50 | f (depth) {f_fit:.3f} vs 0.40 "
      f"| m (velocity) {1-b_fit-f_fit:.3f} vs 0.10")
print(f"structural cap at q=1: b_max = {3*1.0/(5+3*1.0):.3f}")

# --- log-log panels with L&M reference slopes ------------------------------
fig, axes = plt.subplots(1, 2, figsize=(14, 6), dpi=150)
for ax, y, lab, fit, ref in ((axes[0], width[ok], "width (m)",  b_fit, 0.50),
                             (axes[1], depth[ok], "depth (m)",  f_fit, 0.40)):
    hb = ax.hexbin(lq, np.log10(y), gridsize=80, cmap="viridis", mincnt=1, bins="log")
    fig.colorbar(hb, ax=ax, label="reach count")
    xs = np.linspace(lq.min(), lq.max(), 10)
    c  = np.median(np.log10(y)) - fit * np.median(lq)
    ax.plot(xs, fit * xs + c, "r-",  lw=2, label=f"fitted  {fit:.3f}")
    ax.plot(xs, ref * xs + c, "w--", lw=2, label=f"L&M     {ref:.2f}")
    ax.set_xlabel(r"$\log_{10}$ Q (m$^3$/s)"); ax.set_ylabel(f"log10 {lab}")
    ax.legend(); ax.grid(alpha=0.3)
fig.suptitle(f"Downstream hydraulic geometry at baseflow (Q_spec={Q_SPEC})", fontsize=14)
fig.tight_layout()
fig.savefig(PLOT_DIR / "geometry_hydraulic_exponents.png", dpi=250,
            bbox_inches="tight", facecolor="white")

# --- CONUS maps ------------------------------------------------------------
gdf = pyogrio.read_dataframe(SHAPEFILE, columns=["COMID"]).set_index("COMID")
if gdf.crs is None:
    gdf = gdf.set_crs(epsg=4326)            # cat_pfaf_7 ships without a .prj
import pandas as pd
for name, arr, cmap, unit in (("width", width, "Blues",  "m"),
                              ("depth", depth, "Blues",  "m"),
                              ("wd_ratio", width/depth, "magma", "-")):
    gdf[name] = pd.Series(arr, index=comid).reindex(gdf.index).values
conus = gdf.cx[CONUS_BBOX[0]:CONUS_BBOX[2], CONUS_BBOX[1]:CONUS_BBOX[3]]
for name, unit in (("width", "m"), ("depth", "m"), ("wd_ratio", "-")):
    g = conus.dropna(subset=[name]).sort_values(name)
    lo, hi = np.nanpercentile(g[name], [2, 98])
    fig, ax = plt.subplots(figsize=(14, 8), dpi=150)
    g.plot(ax=ax, column=name, cmap="Blues" if unit == "m" else "magma",
           linewidth=0.0, vmin=lo, vmax=hi, zorder=1)
    try:
        import contextily as cx
        cx.add_basemap(ax, crs=g.crs, source=cx.providers.CartoDB.Positron,
                       alpha=0.6, zorder=0, attribution=False)
    except Exception as e:
        print(f"basemap skipped ({type(e).__name__})"); ax.set_facecolor("#f0f0f0")
    ax.set_xlim(CONUS_BBOX[0], CONUS_BBOX[2]); ax.set_ylim(CONUS_BBOX[1], CONUS_BBOX[3])
    ax.set_xticks([]); ax.set_yticks([])
    ax.set_title(f"Baseflow {name} — CONUS (2nd-98th pct colour)", fontsize=13)
    cax = make_axes_locatable(ax).append_axes("right", size="3%", pad=0.1)
    sm = plt.cm.ScalarMappable(cmap="Blues" if unit == "m" else "magma")
    sm.set_array([]); sm.set_clim(lo, hi)
    fig.colorbar(sm, cax=cax).set_label(f"{name} ({unit})")
    fig.savefig(PLOT_DIR / f"geometry_map_{name}_conus.png", dpi=250,
                bbox_inches="tight", facecolor="white")
    plt.close(fig)
```

## Plausibility bands

No ground-truth widths exist in the attributes, so sanity-check against
literature ranges rather than data:

| quantity | plausible | run `2026-08-03T13-11-00Z` |
|---|---|---|
| `w/d` natural channels | 10–50 (up to ~100 for braided) | p50 **26.6**, p95 79.2 |
| baseflow depth, headwaters | 0.05–0.5 m | p5 0.058, p50 **0.250** |
| baseflow width, headwaters | 1–15 m | p5 4.56, p50 **6.50** |
| `b + f + m` | **exactly 1** | 1.000 by construction |

`b + f + m = 1` is an identity, not a test — it holds regardless of whether the
parameters are any good. Do not report it as validation.

## Notes

- **Exponents are `Q_SPEC`-invariant; absolute widths and depths are not.** Sweep
  `Q_SPEC` and confirm `b`/`f` do not move — if they do, the fit is broken.
- **Use post-clamp `slope`** from the parameter NetCDF. Using raw fabric slope
  gives geometry the solver never saw.
- **This is baseflow, not bankfull.** L&M's downstream exponents are usually
  quoted at a consistent frequency (often bankfull or mean annual). Comparing a
  low-flow geometry to bankfull exponents is defensible for the *exponents*
  (which are scale-free) but not for absolute widths.
- **The `w = p·d^q` form caps `b` at 0.375.** If the goal is matching observed
  downstream hydraulic geometry, that is a parameterisation change, not a
  training problem. Widening `parameter_ranges.q_spatial` past 1.0 is the cheap
  probe; note it also changes the celerity β, which depends on `q` through the
  trapezoid closure (`side_slope = T·q/(2d)`).
- **Cross-check against the routing diagnostics.** A deep-narrow bias inflates
  `R` and hence velocity and celerity; if `median_n` is also low, the two
  compound. See `references/parameter_map.md` §Convergence.

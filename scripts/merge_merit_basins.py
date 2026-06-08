"""Merge MERIT pfaf-region shapefiles into one GeoPackage layer.

Usage (run under DDR's venv, /projects/mhpi/tbindas/ddr/.venv/bin/python):

  # catchment polygons (the original invocation; defaults)
  python merge_merit_basins.py

  # river flowlines
  python merge_merit_basins.py \
      '/projects/mhpi/data/MERIT/raw/flowlines/riv_pfaf_*.shp' \
      /projects/mhpi/data/MERIT/raw/global_merit_riv.gpkg \
      flowlines

Streams file-by-file (append mode) so memory stays flat. Files with no
declared CRS are *assigned* EPSG:4326 (MERIT Hydro v0.7 is WGS84 globally
— verified against the .prj files that do exist); files declaring any
other CRS are reprojected with a warning.
"""

import glob
import sys
import time

import geopandas as gpd

SRC_GLOB = sys.argv[1] if len(sys.argv) > 1 else "/projects/mhpi/data/MERIT/raw/basins/cat_pfaf_*.shp"
OUT = sys.argv[2] if len(sys.argv) > 2 else "/projects/mhpi/data/MERIT/raw/global_merit.gpkg"
LAYER = sys.argv[3] if len(sys.argv) > 3 else "catchments"
CRS = "EPSG:4326"

files = sorted(glob.glob(SRC_GLOB))
if not files:
    sys.exit(f"no shapefiles matched {SRC_GLOB}")

total = 0
t0 = time.time()
for i, f in enumerate(files):
    gdf = gpd.read_file(f)
    if gdf.crs is None:
        gdf = gdf.set_crs(CRS)
    elif gdf.crs.to_epsg() != 4326:
        # Defensive: none expected, but reproject rather than mix coordinates.
        print(f"  WARNING: {f} declares {gdf.crs}; reprojecting", flush=True)
        gdf = gdf.to_crs(CRS)
    mode = "w" if i == 0 else "a"
    gdf.to_file(OUT, layer=LAYER, driver="GPKG", mode=mode)
    total += len(gdf)
    name = f.rsplit("/", 1)[-1]
    print(
        f"[{i + 1:2d}/{len(files)}] {name}: {len(gdf):>7,} features "
        f"(running total {total:,}, {time.time() - t0:.0f}s)",
        flush=True,
    )

# Verify: feature count in the gpkg must equal the sum of inputs.
import pyogrio

info = pyogrio.read_info(OUT, layer=LAYER)
print(f"\nwrote {OUT}")
print(f"layer={LAYER} crs={info['crs']} features={info['features']:,}")
if info["features"] != total:
    sys.exit(f"MISMATCH: gpkg has {info['features']:,}, inputs sum to {total:,}")
print("feature count verified: gpkg matches sum of inputs")

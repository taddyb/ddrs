"""Basemap + point coordinates for the smoke page: HUC2 regions (USGS 1:2M, via CAMELS huc_02.zip, UTM 15N NAD83)
projected to CONUS Albers (EPSG:5070), simplified, written as SVG path strings in a fixed viewBox."""
import json
from pathlib import Path

import geopandas as gpd
import pandas as pd
from shapely.geometry import Polygon, MultiPolygon

HERE = Path(__file__).resolve().parent
SMOKE = HERE.parent
W = 960.0
PAD = 8.0

h = gpd.read_file("/vsizip//mnt/ssd1/data/camels/basin_dataset_public_v1p2/shapefiles/huc_02.zip/huc_02.shp").set_crs(26915)
h = h[h.HUC2.astype(int) <= 18].to_crs(5070)
h["geometry"] = h.geometry.buffer(0)
nat = h.dissolve().geometry.iloc[0]
minx, miny, maxx, maxy = h.total_bounds
s = (W - 2 * PAD) / (maxx - minx)
H = round((maxy - miny) * s + 2 * PAD, 1)
print("viewBox", W, H, "km per unit", 1 / s / 1000)


def xy(x, y):
    return round(PAD + (x - minx) * s, 1), round(PAD + (maxy - y) * s, 1)


def ring(coords):
    pts = [xy(x, y) for x, y in coords]
    out, last = [], None
    for p in pts:
        if p != last:
            out.append(p)
            last = p
    if len(out) < 4:
        return ""
    return "M" + "L".join(f"{a:g} {b:g}" for a, b in out[:-1]) + "Z"


def path(geom, tol, min_area):
    geom = geom.simplify(tol, preserve_topology=True)
    polys = [geom] if isinstance(geom, Polygon) else list(geom.geoms) if isinstance(geom, MultiPolygon) else []
    d = []
    for p in polys:
        if p.area < min_area:
            continue
        d.append(ring(p.exterior.coords))
        for r in p.interiors:
            if Polygon(r).area >= min_area * 4:
                d.append(ring(r.coords))
    return "".join(d)


TOL = 3500.0
regions = []
for _, r in h.sort_values("HUC2").iterrows():
    code = f"{int(r.HUC2):02d}"
    lp = r.geometry.representative_point()
    regions.append(dict(huc=code, d=path(r.geometry, TOL, 2.5e8), lx=xy(lp.x, lp.y)[0], ly=xy(lp.x, lp.y)[1]))
outline = path(nat, TOL, 2.5e8)

g = pd.read_csv(SMOKE / "gages_smoke.csv", dtype={"STAID": str})
gp = gpd.GeoDataFrame(g, geometry=gpd.points_from_xy(g.LNG_GAGE, g.LAT_GAGE), crs=4326).to_crs(5070)
sm = pd.read_csv(SMOKE / "smoke_gauges.csv", dtype={"STAID": str, "huc2": str}).set_index("STAID")
# sanity: gauge falls in its HUC2 polygon
j = gpd.sjoin(gp, h[["HUC2", "geometry"]], how="left", predicate="within")
j["h_poly"] = j.HUC2.map(lambda v: f"{int(v):02d}" if pd.notna(v) else None)
j["h_tab"] = j.STAID.map(sm.huc2)
mism = j[j.h_poly != j.h_tab]
print("gauges", len(gp), "HUC2 polygon != table huc2:", len(mism))
print(mism[["STAID", "h_tab", "h_poly", "LAT_GAGE", "LNG_GAGE"]].head(20).to_string())
pts = {r.STAID: list(xy(r.geometry.x, r.geometry.y)) for r in gp.itertuples()}

d = pd.read_csv(SMOKE / "smoke_dams.csv")
dp = gpd.GeoDataFrame(d, geometry=gpd.points_from_xy(d.lon, d.lat), crs=4326).to_crs(5070)
dams = [[*xy(r.geometry.x, r.geometry.y), str(r.name), round(float(r.storage_mcm), 1), int(r.year) if pd.notna(r.year) else None,
         str(r.primary_purpose) if pd.notna(r.primary_purpose) else ""] for r in dp.itertuples()]

out = dict(w=W, h=H, regions=regions, outline=outline, gauges=pts, dams=dams,
           source="USGS 1:2,000,000 Hydrologic Units (Water Resources Regions), huc_02 from the CAMELS basin dataset")
(HERE / "map.json").write_text(json.dumps(out, separators=(",", ":")))
print("map.json bytes", (HERE / "map.json").stat().st_size, "paths", sum(len(r["d"]) for r in regions) + len(outline))

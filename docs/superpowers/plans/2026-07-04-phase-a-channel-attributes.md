# Phase A: Channel-Corridor Attribute Extraction Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Produce `merit_channel_attributes_v1.nc` — per-COMID channel-corridor and groundwater attributes in the exact `global.nc` schema — plus normalization statistics, so ddrs's existing `AttributesStore` can feed them to the KAN unchanged.

**Architecture:** Precomputed per-reach products (StreamCat, Zarrabi, SWORD) are transferred onto MERIT COMIDs via ID-crosswalk tables (one downloaded — MERIT↔SWORD; one built — NHDPlus→MERIT, length-weighted buffered intersection). Only the channel water-table field is extracted from rasters, using nearest-channel-cell sampling (coarse WTD grids) and 100 m corridors (fine rasters) per the literature-grounded buffer strategy. A final assembly script emits the netCDF + stats JSON.

**Tech Stack:** Python under `uv` in `/home/tbindas/projects/extractrs` (extractrs, geopandas, pyogrio, rioxarray, xarray, netCDF4, pandas, pytest). No Rust changes to extractrs expected (categorical/threshold stats reduce to boolean-raster preprocessing + coverage-weighted `mean`).

**Spec:** `ddrs docs/superpowers/specs/2026-07-04-leakance-gate-program-design.md` §3 (Phase A). Read its §A0 ID-space clarification before starting: MERIT COMID and NHDPlus "COMID" are UNRELATED ID spaces sharing a name.

**Working directory:** ALL code lives in `/home/tbindas/projects/extractrs` under a new `pipelines/channel_attrs/` directory. Data staging: `/mnt/ssd1/data/channel_attrs/{raw,derived}/`. Final outputs: `/home/tbindas/projects/ddr/data/merit_channel_attributes_v1.nc` (+ stats JSON next to it). One validation test lives in the ddrs worktree (Task 11).

**Key inputs that already exist:**
- MERIT CONUS flowlines (geometry + attrs incl. `lengthkm`, `uparea`): `/home/tbindas/projects/ddr/data/riv_pfaf_7_MERIT_Hydro_v07_Basins_v01_bugfix1.shp` (346,321 reaches)
- Global attributes schema reference: `/home/tbindas/projects/ddr/data/merit_global_attributes_v2.nc` (dim `COMID`=2,939,404 int64; one f64 var per attribute; `catchsize` variable exists — used for drainage density)
- ddrs stats-JSON convention: inspect `/home/tbindas/projects/ddr/data/statistics/` before Task 10 (Task 0 records the exact format).

**Download manifest (Task 0 stages; all verified downloadable 2026-07-04):**

| Dataset | Source | Target |
|---|---|---|
| MERIT-SWORD translation tables | Zenodo 13152826 (`ms_translate.zip`) | `raw/merit_sword/` |
| SWORD v2 reaches (widths) | Zenodo 10013982 (NA netCDF) | `raw/sword/` |
| NHDPlusV2 national seamless flowlines | EPA NHDPlusV21 (per-VPU or national GDB) | `raw/nhdplusv2/` |
| StreamCat `PctImp2019Rp100Cat` | EPA StreamCat FTP/API (per-region CSVs) | `raw/streamcat/` |
| Zarrabi bankfull geometry | Zenodo 13883263 (`Bankfull_Meanflow_CONUS.txt`) | `raw/zarrabi/` |
| Zell & Sanford 2020 WTD | ScienceBase 10.5066/P91LFFN1 (identify the depth-to-water raster in the release) | `raw/zs_wtd/` |
| Fan 2013 WTD (cross-check) | THREDDS `GLOBALWTDFTP` North America annual mean | `raw/fan_wtd/` |
| USGS Wolock BFI grid | USGS `bfi48grd` | `raw/bfi/` |
| USGS Principal Aquifers polygons | ScienceBase 10.5066/P9Y2HOUJ | `raw/aquifers/` |
| GFPLAIN250m NA tile | figshare (Nardi et al. 2019) | `raw/gfplain/` |

---

### Task 0: Environment, staging, and schema reconnaissance

**Files:**
- Create: `pipelines/channel_attrs/README.md`
- Create: `pipelines/channel_attrs/paths.py`
- Create: `pipelines/channel_attrs/tests/test_paths.py`

- [ ] **Step 1: Set up the uv environment** in `/home/tbindas/projects/extractrs`:

```bash
cd /home/tbindas/projects/extractrs
uv venv --python 3.13 .venv-pipelines
uv pip install --python .venv-pipelines/bin/python \
  extractrs geopandas pyogrio rioxarray xarray netCDF4 pandas pyarrow \
  requests tqdm pytest shapely
```

(If `extractrs` isn't on PyPI for py313, fall back to the repo's own build: `uv pip install --python .venv-pipelines/bin/python maturin && cd extractrs-python && maturin develop --release`. Record which path was used in the README.)

- [ ] **Step 2: Write `paths.py`** — single source of truth for every path:

```python
"""Canonical paths for the channel-attributes pipeline (leakance gate, Phase A)."""
from pathlib import Path

RAW = Path("/mnt/ssd1/data/channel_attrs/raw")
DERIVED = Path("/mnt/ssd1/data/channel_attrs/derived")

MERIT_RIV = Path(
    "/home/tbindas/projects/ddr/data/riv_pfaf_7_MERIT_Hydro_v07_Basins_v01_bugfix1.shp"
)
GLOBAL_NC = Path("/home/tbindas/projects/ddr/data/merit_global_attributes_v2.nc")
STATS_DIR = Path("/home/tbindas/projects/ddr/data/statistics")

OUT_NC = Path("/home/tbindas/projects/ddr/data/merit_channel_attributes_v1.nc")
OUT_STATS = Path("/home/tbindas/projects/ddr/data/statistics/merit_channel_attributes_v1.json")

# Equal-area CRS for all buffering/length math (CONUS Albers).
CRS_EQUAL_AREA = "EPSG:5070"

CORRIDOR_HALF_WIDTH_M = 100.0   # StreamCat precedent (Hill et al. 2016)
CORRIDOR_WIDE_M = 200.0         # flat-valley widening (Amatulli 2022 error tail)
CROSSWALK_BUFFER_M = 300.0      # NHD->MERIT matching envelope (MERIT lateral error 100-300 m)
CROSSWALK_TOP_K = 40            # mirror Wade et al. 2025 table shape
```

- [ ] **Step 3: Failing test** `pipelines/channel_attrs/tests/test_paths.py`:

```python
from pipelines.channel_attrs import paths

def test_inputs_exist():
    assert paths.MERIT_RIV.exists(), "MERIT CONUS flowlines missing"
    assert paths.GLOBAL_NC.exists(), "global attributes nc missing"

def test_staging_dirs():
    assert paths.RAW.is_dir() and paths.DERIVED.is_dir()
```

Run: `cd /home/tbindas/projects/extractrs && .venv-pipelines/bin/python -m pytest pipelines/channel_attrs/tests/test_paths.py -v` → FAIL (dirs missing). Create `mkdir -p /mnt/ssd1/data/channel_attrs/{raw,derived}` and an empty `pipelines/channel_attrs/__init__.py` + `tests/__init__.py`; rerun → PASS.

- [ ] **Step 4: Schema reconnaissance** (record in README — later tasks depend on these facts):

```bash
.venv-pipelines/bin/python - <<'EOF'
import netCDF4, json, pathlib
ds = netCDF4.Dataset("/home/tbindas/projects/ddr/data/merit_global_attributes_v2.nc")
print("dims:", {d: len(ds.dimensions[d]) for d in ds.dimensions})
print("vars:", {v: (ds[v].dimensions, str(ds[v].dtype)) for v in list(ds.variables)[:4]})
stats = sorted(pathlib.Path("/home/tbindas/projects/ddr/data/statistics").glob("*.json"))
print("stats files:", [s.name for s in stats])
if stats:
    print(json.loads(stats[0].read_text()) if stats[0].stat().st_size < 50000 else "large")
EOF
```

Record the EXACT stats-JSON structure (keys, per-variable fields) in the README — Task 10 must emit the same shape. Also `ogrinfo -so` (or pyogrio `read_info`) the MERIT shapefile and record its field names (`COMID`, `lengthkm`, `uparea`, ...).

- [ ] **Step 5: Download the manifest.** For each row of the download manifest write the fetch into `pipelines/channel_attrs/README.md` as you execute it (exact URL used, file sizes, checksums via `md5sum`). Use `wget -c` into the `raw/` subdirs. Items needing discovery: (a) Zell & Sanford — list the ScienceBase item's files and identify the CONUS depth-to-water raster (record the filename); if no directly usable raster exists in the release, note it and designate Fan 2013 as primary (decision recorded in README + surfaced in the task report). (b) NHDPlusV2 — the national seamless geodatabase is ~7 GB; per-VPU flowline shapefiles are acceptable (record which). This step is bandwidth-bound — run downloads in the background and continue with Task 1 (which needs only local inputs) while they complete.

- [ ] **Step 6: Commit** (in extractrs):

```bash
cd /home/tbindas/projects/extractrs
git checkout -b channel-attrs-pipeline
git add pipelines/channel_attrs
git commit -m "feat(pipelines): channel_attrs scaffolding — paths, env, staging, schema recon"
```

---

### Task 1: Corridor geometries

**Files:**
- Create: `pipelines/channel_attrs/corridors.py`
- Create: `pipelines/channel_attrs/tests/test_corridors.py`
- Output: `derived/corridors_100m.parquet`, `derived/corridors_wide.parquet` (GeoParquet, EPSG:5070)

- [ ] **Step 1: Failing test** (synthetic mini-network — no big data needed):

```python
import geopandas as gpd
from shapely.geometry import LineString
from pipelines.channel_attrs.corridors import build_corridors

def test_build_corridors_buffers_and_preserves_ids():
    gdf = gpd.GeoDataFrame(
        {"COMID": [1, 2]},
        geometry=[LineString([(0, 0), (1000, 0)]), LineString([(1000, 0), (1000, 800)])],
        crs="EPSG:5070",
    )
    out = build_corridors(gdf, half_width_m=100.0)
    assert list(out["COMID"]) == [1, 2]
    assert out.crs.to_epsg() == 5070
    # 1000 m line buffered 100 m: area ~ 1000*200 + pi*100^2 (round caps)
    assert abs(out.geometry.iloc[0].area - (1000 * 200 + 3.14159 * 100**2)) < 500
```

Run: `.venv-pipelines/bin/python -m pytest pipelines/channel_attrs/tests/test_corridors.py -v` → FAIL (module missing).

- [ ] **Step 2: Implement `corridors.py`:**

```python
"""Buffered channel corridors from MERIT flowlines.

Buffer rationale (spec §3): MERIT lateral positional error is typically
100-300 m (Amatulli et al. 2022); 100 m half-width matches the StreamCat
precedent (Hill et al. 2016); reaches under broad GFPLAIN floodplains widen
to 200 m (flat-valley error tail).
"""
import geopandas as gpd

from . import paths


def build_corridors(riv: gpd.GeoDataFrame, half_width_m: float) -> gpd.GeoDataFrame:
    riv = riv.to_crs(paths.CRS_EQUAL_AREA)
    out = riv[["COMID"]].copy()
    out["geometry"] = riv.geometry.buffer(half_width_m)
    return gpd.GeoDataFrame(out, geometry="geometry", crs=paths.CRS_EQUAL_AREA)


def main() -> None:
    riv = gpd.read_file(paths.MERIT_RIV, columns=["COMID"])
    for name, hw in [("corridors_100m", paths.CORRIDOR_HALF_WIDTH_M),
                     ("corridors_wide", paths.CORRIDOR_WIDE_M)]:
        corr = build_corridors(riv, hw)
        corr.to_parquet(paths.DERIVED / f"{name}.parquet")
        print(f"{name}: {len(corr)} corridors")


if __name__ == "__main__":
    main()
```

- [ ] **Step 3:** Test passes; then run `main()` on the real fabric (~346k reaches, minutes). Verify: `len == 346321` both files.

- [ ] **Step 4: Commit** `feat(pipelines): MERIT corridor geometries (100m + wide)`.

Note: the GFPLAIN-conditional selection between the two corridor widths happens in Task 6 (the only consumer of fine-raster corridors); Task 1 just materializes both. If GFPLAIN download failed in Task 0, the wide corridor is still produced and the selection falls back to 100 m everywhere (recorded, not silent).

---

### Task 2: SWORD widths onto MERIT (published crosswalk)

**Files:**
- Create: `pipelines/channel_attrs/transfer.py` (the weighted-join helper — shared by Tasks 2, 4, 5)
- Create: `pipelines/channel_attrs/sword_width.py`
- Create: `pipelines/channel_attrs/tests/test_transfer.py`
- Output: `derived/channel_width_obs.parquet` (columns `COMID`, `channel_width_obs`)

- [ ] **Step 1: Failing test for the weighted join** (pure function — this helper is the backbone of every ID transfer):

```python
import pandas as pd
from pipelines.channel_attrs.transfer import weighted_transfer

def test_weighted_transfer_length_weights():
    # COMID 1 maps to foreign reaches A (3000 m) and B (1000 m).
    xwalk = pd.DataFrame({
        "COMID": [1, 1, 2],
        "foreign_id": ["A", "B", "C"],
        "part_len": [3000.0, 1000.0, 500.0],
    })
    attrs = pd.DataFrame({"foreign_id": ["A", "B", "C"], "width": [100.0, 20.0, 7.0]})
    out = weighted_transfer(xwalk, attrs, value_col="width")
    # (3000*100 + 1000*20) / 4000 = 80
    assert out.loc[out.COMID == 1, "width"].item() == 80.0
    assert out.loc[out.COMID == 2, "width"].item() == 7.0

def test_weighted_transfer_drops_unmatched_foreign():
    xwalk = pd.DataFrame({"COMID": [1], "foreign_id": ["Z"], "part_len": [100.0]})
    attrs = pd.DataFrame({"foreign_id": ["A"], "width": [5.0]})
    out = weighted_transfer(xwalk, attrs, value_col="width")
    assert out.loc[out.COMID == 1, "width"].isna().all()
```

- [ ] **Step 2: Implement `transfer.py`:**

```python
"""Length-weighted attribute transfer over reach-ID crosswalk tables.

A crosswalk row says: `part_len` meters of MERIT reach COMID run along
foreign reach `foreign_id`. Transfer = sum(part_len * value) / sum(part_len)
over matched rows with non-null values.
"""
import numpy as np
import pandas as pd


def weighted_transfer(
    xwalk: pd.DataFrame, attrs: pd.DataFrame, value_col: str
) -> pd.DataFrame:
    m = xwalk.merge(attrs[["foreign_id", value_col]], on="foreign_id", how="left")
    m = m[m[value_col].notna() & (m["part_len"] > 0)]
    if m.empty:
        return pd.DataFrame({"COMID": xwalk["COMID"].unique(), value_col: np.nan})
    m["_wv"] = m["part_len"] * m[value_col]
    g = m.groupby("COMID").agg(_wv=("_wv", "sum"), _w=("part_len", "sum"))
    out = (g["_wv"] / g["_w"]).rename(value_col).reset_index()
    return out.merge(
        pd.DataFrame({"COMID": xwalk["COMID"].unique()}), on="COMID", how="right"
    )
```

Tests pass.

- [ ] **Step 3: Implement `sword_width.py`** — melt the Wade et al. tables into the long `(COMID, foreign_id, part_len)` shape and apply:

```python
"""SWORD/GRWL observed widths -> MERIT COMIDs via the published MERIT-SWORD
translation tables (Wade et al. 2025, Zenodo 13152826).

Table shape per pfaf-2 region: variables sword_1..sword_40 (reach_id) and
part_len_1..part_len_40 (m) on a MERIT-reach dimension. CONUS = pfaf regions
71..78 (region 7x files; confirm exact file names on disk and record).
"""
import xarray as xr
import pandas as pd

from . import paths
from .transfer import weighted_transfer


def melt_translation(ds: xr.Dataset, comid_var: str = "COMID") -> pd.DataFrame:
    frames = []
    for k in range(1, 41):
        s, p = f"sword_{k}", f"part_len_{k}"
        if s not in ds or p not in ds:
            break
        df = pd.DataFrame({
            "COMID": ds[comid_var].values,
            "foreign_id": ds[s].values,
            "part_len": ds[p].values,
        })
        frames.append(df[(df.foreign_id > 0) & (df.part_len > 0)])
    return pd.concat(frames, ignore_index=True)
```

`main()`: open each CONUS-region translation file + the SWORD NA reach netCDF (variable `width` per `reach_id`), rename SWORD's id to `foreign_id`, run `weighted_transfer`, write parquet. ADAPT variable names to what's actually in the downloaded files (the melt/table shape is verified from the Zenodo record; per-file naming may differ — record any adaptation).

- [ ] **Step 4: Run + sanity:** widths present for large rivers only (SWORD covers ≥30 m rivers): expect coverage on the order of 5–15% of CONUS reaches, median width of covered reaches > 30 m, Mississippi mainstem reaches > 500 m. Print coverage stats into the report.

- [ ] **Step 5: Commit** `feat(pipelines): SWORD observed widths via MERIT-SWORD weighted transfer`.

---

### Task 3: NHDPlus→MERIT crosswalk (built once)

**Files:**
- Create: `pipelines/channel_attrs/nhd_crosswalk.py`
- Create: `pipelines/channel_attrs/tests/test_nhd_crosswalk.py`
- Output: `derived/nhd_merit_crosswalk.parquet` (`COMID`, `foreign_id` (NHD COMID), `part_len`, plus per-COMID `match_frac` quality flag)

- [ ] **Step 1: Failing test** (synthetic geometries):

```python
import geopandas as gpd
from shapely.geometry import LineString
from pipelines.channel_attrs.nhd_crosswalk import build_crosswalk

def test_crosswalk_length_weighted_matching():
    merit = gpd.GeoDataFrame(
        {"COMID": [10]},
        geometry=[LineString([(0, 0), (2000, 0)])], crs="EPSG:5070")
    # NHD reach A runs along the first 1500 m (offset 50 m north);
    # NHD reach B is 5 km away (no match).
    nhd = gpd.GeoDataFrame(
        {"nhd_comid": [900, 901]},
        geometry=[LineString([(0, 50), (1500, 50)]),
                  LineString([(0, 5000), (1500, 5000)])], crs="EPSG:5070")
    xw = build_crosswalk(merit, nhd, buffer_m=300.0, top_k=40)
    assert set(xw["foreign_id"]) == {900}
    row = xw.iloc[0]
    assert row["COMID"] == 10
    assert abs(row["part_len"] - 1500.0) < 1.0   # clipped NHD length inside the buffer
    # quality: 1500 m of a 2000 m MERIT reach matched
    assert abs(row["match_frac"] - 0.75) < 0.01
```

- [ ] **Step 2: Implement `build_crosswalk`:**

```python
"""NHDPlusV2 -> MERIT crosswalk by buffered-geometry length-weighted matching
(mirrors Wade et al. 2025's MERIT-SWORD construction; spec §A0).

part_len = length of the NHD flowline clipped to the MERIT reach's matching
buffer (CROSSWALK_BUFFER_M = 300 m — the MERIT positional-error envelope).
match_frac = sum(part_len) / merit reach length, a per-COMID quality flag
(low values = headwater divergence between the networks; consumers should
NaN-out transfers below a threshold, default 0.3).
"""
import geopandas as gpd
import pandas as pd

from . import paths


def build_crosswalk(
    merit: gpd.GeoDataFrame, nhd: gpd.GeoDataFrame, buffer_m: float, top_k: int
) -> pd.DataFrame:
    merit = merit.to_crs(paths.CRS_EQUAL_AREA)
    nhd = nhd.to_crs(paths.CRS_EQUAL_AREA)
    buf = merit.copy()
    buf["merit_len"] = merit.geometry.length
    buf["geometry"] = merit.geometry.buffer(buffer_m)

    joined = gpd.sjoin(nhd, buf[["COMID", "merit_len", "geometry"]],
                       how="inner", predicate="intersects")
    if joined.empty:
        return pd.DataFrame(columns=["COMID", "foreign_id", "part_len", "match_frac"])

    # Exact clipped length per (nhd, merit) candidate pair.
    buf_geo = buf.set_index("COMID").geometry
    joined["part_len"] = [
        geom.intersection(buf_geo.loc[c]).length
        for geom, c in zip(joined.geometry, joined["COMID"])
    ]
    joined = joined[joined["part_len"] > 0]

    out = (joined.rename(columns={"nhd_comid": "foreign_id"})
                 [["COMID", "foreign_id", "part_len", "merit_len"]]
                 .sort_values(["COMID", "part_len"], ascending=[True, False])
                 .groupby("COMID").head(top_k))
    frac = out.groupby("COMID").apply(
        lambda g: min(1.0, g["part_len"].sum() / g["merit_len"].iloc[0]),
        include_groups=False).rename("match_frac")
    return out.drop(columns="merit_len").merge(frac, on="COMID")
```

Test passes.

- [ ] **Step 3: `main()` — run at CONUS scale, chunked.** Load MERIT riv (346k), load NHD flowlines per VPU (or in ~200k-row chunks from the national file with pyogrio's `bbox` filter per MERIT pfaf-4 tile), run `build_crosswalk` per chunk against the spatially-overlapping MERIT subset (use a coarse bbox prefilter), concatenate, dedupe on (COMID, foreign_id) keeping max part_len, re-rank/top-k, write parquet. Print: total pairs, COMIDs with ≥1 match, `match_frac` quartiles. Expected: >90% of non-headwater MERIT reaches matched; `match_frac` median > 0.7; headwaters worse (known networks divergence — that's what the flag is for). This is the heaviest compute task of the plan (hours); run under `nohup`/tee with progress prints per chunk.

- [ ] **Step 4: Spot-verification:** pick 3 known rivers (Mississippi at Vicksburg, Platte, Gila), confirm the matched NHD COMIDs' GNIS names (NHD attribute) agree with the river. Include in the report.

- [ ] **Step 5: Commit** `feat(pipelines): NHDPlus->MERIT length-weighted crosswalk (+match_frac quality flags)`.

---

### Task 4: StreamCat imperviousness transfer

**Files:**
- Create: `pipelines/channel_attrs/streamcat_transfer.py`
- Output: `derived/corridor_impervious.parquet`

- [ ] **Step 1:** Load StreamCat regional CSVs; concatenate `COMID` (NHD) + `PctImp2019Rp100Cat` → rename to `foreign_id`/value. Load the Task 3 crosswalk, NaN-out rows with `match_frac < 0.3`, run `weighted_transfer` (Task 2's helper — already tested), divide by 100 to a 0–1 fraction, write parquet.

```python
"""StreamCat PctImp2019Rp100Cat (NLCD imperviousness in the 100 m riparian
buffer, precomputed on NHDPlusV2) -> MERIT via the Task-3 crosswalk."""
import pandas as pd

from . import paths
from .transfer import weighted_transfer

MATCH_FRAC_MIN = 0.3


def main() -> None:
    sc = pd.concat(
        [pd.read_csv(p, usecols=["COMID", "PctImp2019Rp100Cat"])
         for p in sorted((paths.RAW / "streamcat").glob("*.csv"))],
        ignore_index=True,
    ).rename(columns={"COMID": "foreign_id", "PctImp2019Rp100Cat": "corridor_impervious"})
    xw = pd.read_parquet(paths.DERIVED / "nhd_merit_crosswalk.parquet")
    xw = xw[xw["match_frac"] >= MATCH_FRAC_MIN]
    out = weighted_transfer(xw, sc, value_col="corridor_impervious")
    out["corridor_impervious"] /= 100.0
    out.to_parquet(paths.DERIVED / "corridor_impervious.parquet")
    print(out["corridor_impervious"].describe())


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Run + the LA-River spot check** (this is a spec-mandated validation): identify the MERIT COMIDs along the Los Angeles River (bbox query on the riv shapefile around 34.0N, −118.2E), assert their `corridor_impervious` > 0.6; check a rural Montana reach < 0.05. Record both numbers in the report.

- [ ] **Step 3: Commit** `feat(pipelines): corridor imperviousness via StreamCat Rp100 transfer`.

---

### Task 5: Zarrabi bankfull geometry + McManamay confinement transfer

**Files:**
- Create: `pipelines/channel_attrs/bankfull_transfer.py`
- Output: `derived/bankfull.parquet` (`COMID`, `bankfull_depth`, `bankfull_width`), `derived/confinement.parquet` (optional — drop gracefully if the McManamay download stalls; it's a nice-to-have proxy, record the decision)

- [ ] **Step 1:** Same pattern as Task 4 verbatim (load `Bankfull_Meanflow_CONUS.txt` — inspect its delimiter/columns first and record; expect NHD `COMID`, bankfull width/depth columns), crosswalk-filter, `weighted_transfer` per value column, write parquet. Confinement is categorical — transfer as per-class fractions (one `weighted_transfer` per one-hot class column, argmax at the end) and store the majority class as an integer code + the unconfined fraction as a float (floats are what the netCDF schema wants; store `unconfined_frac`).

- [ ] **Step 2: Run + sanity:** `bankfull_depth` median in 0.3–3 m, increasing with `uparea` (spearman ρ > 0.5 against the riv shapefile's uparea); widths consistent with Task 2's SWORD widths on covered reaches (ρ > 0.5, SWORD systematically wider on big rivers is fine — different definitions). Record.

- [ ] **Step 3: Commit** `feat(pipelines): bankfull geometry (Zarrabi) + confinement transfer`.

---

### Task 6: Channel water-table sampling (the novel extraction)

**Files:**
- Create: `pipelines/channel_attrs/wtd_sample.py`
- Create: `pipelines/channel_attrs/tests/test_wtd_sample.py`
- Output: `derived/wtd_channel.parquet` (`COMID`, `wtd_channel_zs`, `wtd_channel_fan`, `wtd_corridor100_zs` (sensitivity), `wtd_channel_n` (sample count))

- [ ] **Step 1: Failing test for the point-sampling core** (synthetic raster + line):

```python
import numpy as np
import xarray as xr
import geopandas as gpd
from shapely.geometry import LineString
from pipelines.channel_attrs.wtd_sample import sample_along_lines

def test_sample_along_lines_nearest_cell_mean():
    # 1 km grid: WTD deepens eastward: columns 0,1,2 -> 5, 10, 15 m.
    da = xr.DataArray(
        np.array([[5.0, 10.0, 15.0]] * 3),
        coords={"y": [2500.0, 1500.0, 500.0], "x": [500.0, 1500.0, 2500.0]},
        dims=("y", "x"),
    ).rio.write_crs("EPSG:5070")
    gdf = gpd.GeoDataFrame(
        {"COMID": [1]},
        geometry=[LineString([(200, 1500), (2800, 1500)])], crs="EPSG:5070")
    out = sample_along_lines(da, gdf, spacing_m=200.0)
    row = out.loc[out.COMID == 1].iloc[0]
    # densified points cross all three cells; mean of nearest-cell values
    assert 5.0 < row["value_mean"] < 15.0
    assert row["n"] >= 10
```

- [ ] **Step 2: Implement:**

```python
"""Nearest-channel-cell water-table sampling along MERIT flowlines.

Coarse WTD grids (~500 m - 1 km) get NO fine buffer: the cell already spans
MERIT's positional error (spec §3 buffer strategy; RiverATLAS-style
native-grid association). Lines are densified to `spacing_m` vertices and the
nearest cell value is taken per vertex; per-reach mean + count are returned.
"""
import numpy as np
import pandas as pd
import geopandas as gpd
import xarray as xr


def _densify(line, spacing_m: float):
    n = max(2, int(line.length // spacing_m) + 1)
    return [line.interpolate(d) for d in np.linspace(0.0, line.length, n)]


def sample_along_lines(
    da: xr.DataArray, gdf: gpd.GeoDataFrame, spacing_m: float = 200.0
) -> pd.DataFrame:
    gdf = gdf.to_crs(da.rio.crs)
    rows = []
    for comid, line in zip(gdf["COMID"], gdf.geometry):
        pts = _densify(line, spacing_m)
        xs = xr.DataArray([p.x for p in pts], dims="pt")
        ys = xr.DataArray([p.y for p in pts], dims="pt")
        vals = da.sel(x=xs, y=ys, method="nearest").values.astype(float)
        vals = vals[np.isfinite(vals)]
        rows.append({"COMID": comid,
                     "value_mean": float(np.mean(vals)) if len(vals) else np.nan,
                     "n": int(len(vals))})
    return pd.DataFrame(rows)
```

Test passes. (Coordinate naming: real rasters may use `lon`/`lat` or band dims — `main()` normalizes with `rio.reproject` to EPSG:5070 and renames to x/y before calling this; if the Zell & Sanford raster is in a projected CRS already, only rename.)

- [ ] **Step 3: `main()`** — run for BOTH WTD sources: open Zell & Sanford raster and Fan NA annual-mean netCDF via rioxarray, reproject to 5070 (coarse grids — cheap), `sample_along_lines` over the MERIT riv lines (chunk by pfaf-4 tile, progress prints), plus one extractrs corridor-mean pass as sensitivity (`corridors_100m.parquet` on the Zell & Sanford grid — and where GFPLAIN marks floodplain, substitute the wide corridor; if GFPLAIN absent, 100 m everywhere, recorded):

```python
import extractrs  # noqa: F401  (registers the .extrs accessor)
corr = gpd.read_parquet(paths.DERIVED / "corridors_100m.parquet")
zs_corr = zs_da.to_dataset(name="wtd").extrs.zonal_stats(corr, stat="mean", id_col="COMID")
```

Merge the three columns + count, write parquet.

- [ ] **Step 4: Run + cross-check (spec-mandated):** `spearman(wtd_channel_zs, wtd_channel_fan)` over reaches with both — expect ρ > 0.4 (positive, not identical); nearest-cell vs corridor-mean ρ > 0.8 (the sensitivity check — if these diverge wildly the buffer strategy matters more than the literature suggested: STOP and surface). Record both.

- [ ] **Step 5: Commit** `feat(pipelines): channel water-table sampling (nearest-cell + corridor sensitivity, ZS + Fan)`.

---

### Task 7: Bed-relative head + losing fraction

**Files:**
- Create: `pipelines/channel_attrs/wtd_bedrel.py`
- Create: `pipelines/channel_attrs/tests/test_wtd_bedrel.py`
- Output: `derived/wtd_bedrel.parquet` (`COMID`, `channel_wtd_bed_rel`, `losing_fraction`)

- [ ] **Step 1: Failing test:**

```python
import pandas as pd
from pipelines.channel_attrs.wtd_bedrel import bed_relative

def test_bed_relative_sign_convention():
    wtd = pd.DataFrame({"COMID": [1, 2], "wtd_channel_zs": [10.0, 0.5]})
    bank = pd.DataFrame({"COMID": [1, 2], "bankfull_depth": [2.0, 2.0]})
    out = bed_relative(wtd, bank)
    # channel_wtd_bed_rel = WTD below land surface - bankfull depth
    # positive = water table BELOW the bed = losing-possible
    assert out.loc[out.COMID == 1, "channel_wtd_bed_rel"].item() == 8.0
    assert out.loc[out.COMID == 2, "channel_wtd_bed_rel"].item() == -1.5  # WT above bed
```

- [ ] **Step 2: Implement** (`bed_relative` merges and subtracts; `losing_fraction` comes from re-running Task 6's per-vertex samples against the per-reach bed depth — extend `sample_along_lines` usage in `main()` to also return `frac_below = mean(vals > bankfull_depth)` per reach; NaN-safe). Sign convention documented in the module docstring: **positive `channel_wtd_bed_rel` = water table below bed = losing-possible**; this is the orientation `d_gw` supervision/validation uses in Phase C.

- [ ] **Step 3: Run + the three spec-mandated regime checks:** an Ogallala/High-Plains reach (expect strongly positive bed-relative WTD), an Appalachian perennial reach (expect negative), LA River (positive WTD but Task 4 says impervious — the falsification pair for Phase C's Leg 3). Record all three COMIDs + values in the report — the findings doc and Phase C reuse them.

- [ ] **Step 4: Commit** `feat(pipelines): bed-relative channel head + losing fraction`.

---

### Task 8: Alluvium fraction (polygon overlay)

**Files:**
- Create: `pipelines/channel_attrs/alluvium.py`
- Output: `derived/alluvium_fraction.parquet`

- [ ] **Step 1:** Principal Aquifers polygons → filter alluvial classes (inspect the layer's ROCK_TYPE/AQ_NAME fields at execution; record the class list chosen) → `gpd.overlay(corridors_100m, alluvial_polys, how="intersection")` in EPSG:5070 → `alluvium_fraction = intersected_area / corridor_area` per COMID (0 where no intersection — explicit fill, not NaN: absence of mapped alluvium is information). Chunk by pfaf-4 tile.

- [ ] **Step 2: Run + sanity:** Mississippi alluvial valley reaches ≈ 1.0; Rocky Mountain headwaters ≈ 0. Coverage: every CONUS COMID has a value (0 default).

- [ ] **Step 3: Commit** `feat(pipelines): corridor alluvium fraction from Principal Aquifers overlay`.

---

### Task 9: Basin-scale extras (BFI, drainage density)

**Files:**
- Create: `pipelines/channel_attrs/basin_extras.py`
- Output: `derived/basin_extras.parquet` (`COMID`, `bfi`, `drainage_density`)

- [ ] **Step 1:** BFI: Wolock `bfi48grd` via rioxarray → corridor-mean with extractrs over `corridors_100m` (BFI is a smooth ~1 km field; corridor vs catchment mean differ little, and corridors avoid needing catchment polygons — note recorded). Drainage density: `lengthkm` from the riv shapefile ÷ `catchsize` from `global.nc` (km²) per COMID → 1/km; guard `catchsize <= 0` → NaN.

- [ ] **Step 2: Run + sanity:** BFI ∈ [0,1] (grid is %, divide by 100); drainage_density median in 0.1–2 km⁻¹; humid East > arid Southwest for BFI.

- [ ] **Step 3: Commit** `feat(pipelines): BFI + drainage density extras`.

---

### Task 10: Assemble the netCDF + normalization statistics

**Files:**
- Create: `pipelines/channel_attrs/assemble.py`
- Create: `pipelines/channel_attrs/tests/test_assemble.py`
- Output: `/home/tbindas/projects/ddr/data/merit_channel_attributes_v1.nc`, `.../statistics/merit_channel_attributes_v1.json`

- [ ] **Step 1: Failing test** (synthetic mini-assembly):

```python
import numpy as np
import pandas as pd
import netCDF4
from pipelines.channel_attrs.assemble import assemble

def test_assemble_matches_global_schema(tmp_path):
    global_comids = np.array([10, 20, 30, 40], dtype="int64")
    frames = {
        "corridor_impervious": pd.DataFrame({"COMID": [10, 30], "corridor_impervious": [0.9, 0.1]}),
        "bfi": pd.DataFrame({"COMID": [10, 20, 30], "bfi": [0.5, 0.6, 0.7]}),
    }
    out = tmp_path / "test.nc"
    assemble(global_comids, frames, out)
    ds = netCDF4.Dataset(out)
    assert len(ds.dimensions["COMID"]) == 4
    assert str(ds["COMID"].dtype) == "int64"
    assert str(ds["corridor_impervious"].dtype) == "float64"
    v = ds["corridor_impervious"][:]
    assert v[0] == 0.9 and np.isnan(v[1]) and v[2] == 0.1 and np.isnan(v[3])
```

- [ ] **Step 2: Implement `assemble`:** read the FULL global COMID vector from `GLOBAL_NC` (2,939,404 int64 — index compatibility with the same reader, NaN outside CONUS), left-align every parquet's values onto it (pandas reindex on COMID), write one f64 variable per attribute with dim `COMID` — byte-schema-identical to `global.nc` (no extra dims, no groups). `main()` assembles ALL columns: `channel_width_obs`, `corridor_impervious`, `bankfull_depth`, `bankfull_width`, `unconfined_frac`, `channel_wtd_bed_rel`, `losing_fraction`, `alluvium_fraction`, `bfi`, `drainage_density`, plus `wtd_channel_zs`/`wtd_channel_fan` (kept for Leg-3 validation even though only derived fields feed the KAN).

- [ ] **Step 3: Stats JSON:** compute per-variable normalization statistics over FINITE values only, in the EXACT format recorded in Task 0's reconnaissance (match keys/fields of the existing ddrs statistics files verbatim — this is a hard requirement; if the recon found per-variable mean/std, emit that; if quantiles, emit those). Write next to the existing stats.

- [ ] **Step 4: Run assembly; validate:**

```bash
.venv-pipelines/bin/python -m pipelines.channel_attrs.assemble
.venv-pipelines/bin/python - <<'EOF'
import netCDF4, numpy as np
a = netCDF4.Dataset("/home/tbindas/projects/ddr/data/merit_global_attributes_v2.nc")
b = netCDF4.Dataset("/home/tbindas/projects/ddr/data/merit_channel_attributes_v1.nc")
assert len(a.dimensions["COMID"]) == len(b.dimensions["COMID"])
assert (a["COMID"][:100] == b["COMID"][:100]).all()
for v in b.variables:
    if v != "COMID":
        assert b[v].dimensions == ("COMID",) and str(b[v].dtype) == "float64", v
        finite = np.isfinite(b[v][:]).sum()
        print(f"{v}: finite={finite} ({finite/len(b.dimensions['COMID']):.1%})")
EOF
```

Expected: identical dim length + COMID ordering; every variable f64 on (COMID,); finite coverage ≈ CONUS fraction (~12%) for corridor vars, less for SWORD width.

- [ ] **Step 5: Commit** `feat(pipelines): assemble merit_channel_attributes_v1.nc + normalization stats`, then push the branch and open a PR in extractrs per that repo's conventions.

---

### Task 11: ddrs reader roundtrip (in the ddrs worktree)

**Files:**
- Create (ddrs worktree): `tests/channel_attrs_store.rs`

- [ ] **Step 1: Write the path-gated test** (ddrs convention: skip when the data file is absent — same pattern as the `/mnt/ssd1` icechunk tests):

```rust
//! Phase-A deliverable gate: merit_channel_attributes_v1.nc must open through
//! the SAME AttributesStore reader as merit_global_attributes_v2.nc (spec
//! §3: identical schema, zero reader changes).

use std::path::Path;

use ddrs::data::store::AttributesStore;

const CHANNEL_NC: &str = "/home/tbindas/projects/ddr/data/merit_channel_attributes_v1.nc";

#[test]
fn channel_attributes_open_through_attributes_store() {
    if !Path::new(CHANNEL_NC).exists() {
        eprintln!("skipping: {CHANNEL_NC} not present (Phase A not yet run)");
        return;
    }
    let store = AttributesStore::open(CHANNEL_NC).expect("opens like global.nc");
    for name in [
        "channel_wtd_bed_rel",
        "losing_fraction",
        "corridor_impervious",
        "alluvium_fraction",
        "bfi",
        "drainage_density",
        "bankfull_depth",
    ] {
        assert!(
            store.has_variable(name),
            "missing Phase-A attribute '{name}'"
        );
    }
}
```

ADAPT to `AttributesStore`'s real constructor/method names (read `src/data/store/netcdf.rs` first — if there's no `has_variable`, read one variable's values for a few COMIDs instead and assert finite values exist). The test must exercise the exact code path `MeritGagesDataset::open` uses for attributes.

- [ ] **Step 2:** Run `cargo test --test channel_attrs_store` — passes (skips gracefully before Phase A's data lands; passes for real after Task 10).

- [ ] **Step 3: Commit** (ddrs worktree) `test(data): gate Phase-A channel attributes through AttributesStore`.

---

## Self-review (done at write time)

- **Spec coverage:** §A0 crosswalks → Tasks 2 (published) + 3 (built, with match_frac flags per the spec's quality-flag deliverable); §A1 transfers → Tasks 2/4/5/9; §A2 corridor extraction → Tasks 1/6/7/8 (two-tier buffers: nearest-cell for coarse WTD + 100 m corridors, GFPLAIN widening with recorded fallback); §A "global.nc schema + stats + AttributesStore roundtrip" → Tasks 10/11; spec validation reaches (LA River / Ogallala / Appalachian, ZS-vs-Fan correlation) → Tasks 4/6/7.
- **Known unknowns flagged inline with resolution paths** (not placeholders): Zell & Sanford raster identification (Task 0, Fan fallback recorded), MERIT-SWORD per-file variable naming (Task 2), Zarrabi column names (Task 5), Principal Aquifers class fields (Task 8), stats-JSON exact format (Task 0 recon → Task 10), `AttributesStore` method names (Task 11).
- **Type consistency:** `weighted_transfer(xwalk, attrs, value_col)` with `(COMID, foreign_id, part_len)` shape used identically in Tasks 2/4/5; `paths.py` constants referenced by name throughout; output parquet column names match Task 10's assembly list.

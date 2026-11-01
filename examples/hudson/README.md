# Hudson River tidal example

A committed bundle that runs the stage block on the tidal Hudson. The
boundary is the measured water level at The Battery (NOAA 8518750). The
evaluation point is the USGS gauge below Poughkeepsie (01372058).

Run from the repo root. Give `--workspace` explicitly.

```bash
ddrs --config examples/hudson/ddrs_mc_only.yaml --workspace /path/to/.ddrs plan
ddrs --config examples/hudson/ddrs.yaml --workspace /path/to/.ddrs plan
```

`ddrs_mc_only.yaml` is the control: same data, no stage block.

## Domain

756 MERIT reaches upstream of the New York Harbor outlet (COMID 73005067), one
outlet. ddrs routes the union of the gauge subgraphs, so a fourth row in
`hudson_gage.csv` is a pseudo-gauge at the outlet (STAID and COMID 73005067,
`DA_VALID True`, no observations). It makes every batch the whole domain.
`batch_size: 4` is the gauge count. The bundle holds its observation series as
all NaN. The pseudo-gauge passes the dataset gauge filter (tested in
`tests/hudson_bundle.rs::harbor_pseudo_gauge_survives_the_gauge_filter`). It is
never a target: the loss drops it, and its baseline NSE is null.

The stage set is the 41-reach tidal mainstem from COMID 73001655 (the reach
below Green Island) to the outlet. Other reaches run MC.

Targets: Poughkeepsie 01372058 (USGS parameter 72137, tidally filtered
discharge, daily, 6,159 days from 2002-03-15 to 2019-09-04), Green Island
01358000, Croton 01375000.

## Record

All stores and the Battery series run 2001-05-01 to 2019-12-31 (6,819 days,
163,656 hours). The hourly Battery product has a 4-month outage from
2001-01-01 to 2001-04-30, so the stores start 2001-05-01, 17 months before the training start
(2002-10-01). Training: 2002-10-01 to 2012-09-30. Test: 2012-10-01 to
2019-08-31.

## Battery series

`data/hudson_battery_stage_navd88.csv`: columns `time,stage_m`, hourly, UTC,
NAVD88 metres. Source: CO-OPS `hourly_height`, `datum=NAVD`, `units=metric`,
`time_zone=gmt`. Built with `scripts/fetch_noaa_hourly.py`. Every filled hour
is in `data/hudson_battery_stage_navd88.csv.gaps.txt` with its method.
3,713 of 163,656 hours (2.27 %) are filled. The share must stay under 3 %.

| Method | Hours | Where |
|---|---|---|
| `6min_mean` | 501 | 2003-11-01 to 11-26 and 2008-09-24 to 10-02, hours the hourly product flagged L=1 |
| `predictions_plus_residual` | 3,121 | 2004-02-01 to 05-31 (2,904 h outage) and 217 hours of the 2003-11 span, in fallback runs longer than 6 h |
| `linear` | 91 | one 3 h gap in 2018, and 88 hours of the 2003-11 span, in fallback runs of 6 h or less |

- `6min_mean`: CO-OPS 6-minute `water_level` (same datum). Only samples with
  all flags 0 count. The hour value is the mean of the samples within 24
  minutes of the hour (up to 9, symmetric about the hour). At least 5 samples
  are needed. This approximates, but is not, the CO-OPS hourly filter.
- `predictions_plus_residual`: CO-OPS hourly `predictions` plus a residual. For
  the 2004 outage the residual runs linearly across the span between the 7-day
  mean of (observed minus predicted) before the span and the same after it. In
  the `--fill-6min` spans, hours with too few clean 6-minute samples form
  fallback runs. A run of 6 h or less is filled linearly between its real
  neighbours (`linear`). A longer run uses predictions plus a residual that
  runs linearly between the residual at the filled or observed hour just
  before the run and the one just after it. This keeps the fill from switching
  back and forth between real values and a span-wide ramp.
- Step check: the script stops if an hour-to-hour step at a `linear` or
  `predictions_plus_residual` hour exceeds the 99.9th percentile of the steps
  between two observed hours (0.601 m). Measured maximum 0.553 m (2004-05-05,
  in the tide-only span). Steps at `6min_mean` hours are real measurements and
  reach 0.702 m.
- Limit: for February to May 2004 the series is tide only plus a smooth
  residual. Storm surge in that span is missing. That span is in the training
  window.
- A CO-OPS value flagged `L=1` counts as missing before the fills above.

## Channel geometry

`data/hudson_channel_geometry.csv` is a uniform placeholder: top width 1000 m,
bed -10.0 m NAVD88, `n` 0.022, `source` = `fallback_uniform`, for all 41 rows.
It is not a survey. Real bathymetry (NOAA ENC soundings or the Hudson River
Estuary multibeam survey) replaces it later. Chart soundings are MLLW; the
Battery MLLW datum is 0.844 m below NAVD88. `anchor_elevation_m` in `ddrs.yaml`
(-10.0) equals the outlet row's `bed_z_m`; change both together. The MLLW to
NAVD88 offset differs along the river (Battery 0.84 m, Haverstraw 0.50 m, Albany
0.55 m), so use the nearest station's offset or VDatum, not the Battery value
for all reaches. The tide-shape
check stays off until the placeholder is replaced.

## Poughkeepsie series

USGS NWIS `dv`, parameter 72137, converted from cfs to m3/s with
`scripts/fetch_nwis_daily.py`. 245 days are negative (minimum -705 m3/s; a
tidally filtered flow can reverse). 289 days are estimated (`A:e`). About 13 %
of 90-day training windows have a NaN after warm-up, so Poughkeepsie leaves the
loss in those windows. This is not a defect.

## Results

`train-and-test` on CPU, seed 42, 30 epochs, 2026-11-01. The test is
`tests/hudson_acceptance.rs` (release, local only, not in CI). Metrics are per
gauge on the daily series after the 5-day warm-up. "Stage" is `ddrs.yaml`,
"MC" is `ddrs_mc_only.yaml`.

| Gauge | MC NSE / KGE | Stage NSE / KGE | Baseline NSE / KGE |
|---|---|---|---|
| Poughkeepsie 01372058 | 0.145 / 0.604 | 0.308 / 0.625 | 0.144 / 0.5605 |
| Green Island 01358000 | 0.781 / 0.876 | 0.780 / 0.876 | 0.688 / 0.801 |
| Croton 01375000 | 0.820 / 0.856 | 0.819 / 0.856 | 0.835 / 0.867 |
| Harbor pseudo-gauge 73005067 | NaN | NaN | NaN |

Control row, MC with the Godin filter applied at eval only (an offline replay
of the MC hourly series through the same filter, tau trim and warm-up; a
filter-trained MC head is not possible, because the filter needs stage rows):

| Gauge | MC + Godin at eval only, NSE / KGE |
|---|---|
| Poughkeepsie 01372058 | 0.174 / 0.616 |
| Green Island 01358000 | 0.793 / 0.886 |
| Croton 01375000 | 0.819 / 0.849 |

Both runs report `n_gauges_total` 4 and `n_gauges_finite_nse` 3. The stage
block lifts Poughkeepsie NSE by 0.16 and leaves the other gauges alone. About
0.13 of that 0.16 is the stage block and about 0.03 is the filter (control:
Godin applied at eval only). The Battery fix in this version (filled hours in
Nov 2003) did not move the stage result at the printed precision.

Limits of this evidence: one seed (42) and one tidal gauge. 30 epochs with 4
gauges and `batch_size` 4 is 30 optimizer steps, so the head is not converged.
The February to May 2004 boundary is filled (tide plus an interpolated
residual, no surge).

Spring-neap check: the amplitude of the 14.77 day cycle in the filtered daily
Poughkeepsie series, model over observed, is 0.96 for the stage run and 0.54 for
MC (maximum over periods of 13.5 to 16 days: 1.11 and 0.78).

Wall time: MC 27 s, stage 123 s (ratio 4.5). The one-chunk eval peaks at about
2 GB of memory in the test process.

Acceptance floors (measured minus 0.04): stage Poughkeepsie NSE >= 0.26 and
KGE >= 0.58, plus NSE above the summed-Q' baseline, Green Island NSE within 0.02
of MC, and a stage mass balance of at most 1e-4 (measured 5.9e-11).

### Caveats

- Fallback geometry. All 41 stage rows use the uniform placeholder (width
  1000 m, bed -10 m). The tide shape is not asserted. Real bathymetry is the next
  step.
- 2004 filled boundary. The Battery series from 2004-02-01 to 2004-05-31 is tide
  plus a smooth residual, with no storm surge. That span is in the training window.
- One-chunk eval. Both configs set `testing.batch_size: 2600`, which covers the
  whole test period (2,526 days) in one eval chunk. Chunked eval repeats or skips
  one hour at each chunk seam, and that error leaks into the 71 h Godin filter at
  Poughkeepsie. With 15-day chunks the daily Poughkeepsie error reached 125 m3/s
  (35 %), against 1.9 m3/s for MC. This is a known limitation of chunked eval on
  tidal reaches. Keep eval as one chunk for stage runs with `tidal_filter: godin`.

## Rebuilding the bundle

The fabric and the Q' and observation stores are workstation paths. Run steps 1,
2 and 4 under the DDR venv (`cd ~/projects/ddr && uv run python ...`), with
`S` a scratch directory outside the repo. Raw downloads go into their own
subdirectory of `S`.

1. Fetch the Battery series (CO-OPS station 8518750). Responses are cached in
   `--cache-dir`. The three fill spans are the 2003-11 and 2008-09 6-minute
   fills and the 2004 outage:

   ```bash
   python ~/projects/ddrs/scripts/fetch_noaa_hourly.py --station 8518750 \
       --start 2001-05-01 --end 2019-12-31 --cache-dir $S/noaa_raw \
       --fill-6min 2003-11-01:2003-11-26 --fill-6min 2008-09-24:2008-10-02 \
       --fill-pred 2004-02-01:2004-05-31 \
       --out ~/projects/ddrs/examples/hudson/data/hudson_battery_stage_navd88.csv
   ```

2. Fetch Poughkeepsie (USGS 01372058, parameter 72137):

   ```bash
   python ~/projects/ddrs/scripts/fetch_nwis_daily.py --site 01372058 \
       --param 72137 --start 2001-01-01 --end 2019-12-31 \
       --raw $S/nwis_raw/pk_72137.rdb --out $S/pk_72137.csv
   ```

3. Build the full-CONUS adjacency and the gauge subgraphs with `ddrs plan`. Copy
   `examples/hudson/ddrs_mc_only.yaml` to `$S/build.yaml`. In its
   `data_sources:` block keep `gages: examples/hudson/data/hudson_gage.csv` and
   `attributes`, `streamflow`, `observations` pointing at the workstation
   stores, and replace `conus_adjacency` and `gages_adjacency` with
   `geospatial_fabric: /mnt/ssd1/data/merit/riv_pfaf_7_MERIT_Hydro_v07_Basins_v01_bugfix1.shp`.
   Then:

   ```bash
   ddrs --config $S/build.yaml --workspace $S/ws plan
   ```

   The stores land in `$S/ws/adjacency/<key>/` as
   `riv_pfaf_7_MERIT_Hydro_v07_Basins_v01_bugfix1_adjacency.zarr` (`--conus-zarr`)
   and `..._gages_adjacency.zarr` (`--gages-zarr`).

4. Extract the bundle:

   ```bash
   python ~/projects/ddrs/scripts/extract_merit_bundle.py --out $S/hudson_out \
       --name hudson --gages 01372058,01358000,01375000,73005067 \
       --outlet-gage 73005067 --gage-csv examples/hudson/data/hudson_gage.csv \
       --conus-zarr $S/ws/adjacency/<key>/riv_pfaf_7_MERIT_Hydro_v07_Basins_v01_bugfix1_adjacency.zarr \
       --gages-zarr $S/ws/adjacency/<key>/riv_pfaf_7_MERIT_Hydro_v07_Basins_v01_bugfix1_gages_adjacency.zarr \
       --domain-from-comid 73005067 --time 2001-05-01:2019-12-31 \
       --extra-obs $S/pk_72137.csv --nan-obs-gages 73005067
   ```

   Copy the generated `hudson_*` stores, the attribute file and the statistics
   JSON into `examples/hudson/data/`. The channel geometry table is
   hand-written and is not generated.

## Baseline (summed Q', test window)

Poughkeepsie NSE 0.144, KGE 0.5605. Green Island NSE 0.688, KGE 0.801. Croton
NSE 0.835, KGE 0.867.

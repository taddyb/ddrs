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
163,656 hours). The Battery has no usable hourly data before 2001-05-01 (a
4-month outage), so the stores start there, 17 months before the training start
(2002-10-01). Training: 2002-10-01 to 2012-09-30. Test: 2012-10-01 to
2019-08-31.

## Battery series

`data/hudson_battery_stage_navd88.csv`: columns `time,stage_m`, hourly, UTC,
NAVD88 metres. Source: CO-OPS `hourly_height`, `datum=NAVD`, `units=metric`,
`time_zone=gmt`. Built with `scripts/fetch_noaa_hourly.py`. Every filled hour
is in `data/hudson_battery_stage_navd88.csv.gaps.txt` with its method.
3,713 of 163,656 hours (2.27 %) are filled. This replaces the plan's earlier
0.1 % target by controller ruling.

| Method | Hours | Where |
|---|---|---|
| `6min_mean` | 501 | 2003-11-01 to 11-26 and 2008-09-24 to 10-02, hours the hourly product flagged L=1 |
| `predictions_plus_residual` | 3,209 | 2004-02-01 to 05-31 (2,904 h outage) and 305 hours of the 2003-11 span with too few clean 6-minute samples |
| `linear` | 3 | one 3 h gap in 2018 |

- `6min_mean`: CO-OPS 6-minute `water_level` (same datum). Only samples with
  all flags 0 count. The hour value is the mean of the samples within 24
  minutes of the hour (up to 9, symmetric about the hour). At least 5 samples
  are needed. This approximates, but is not, the CO-OPS hourly filter.
- `predictions_plus_residual`: CO-OPS hourly `predictions` plus a residual. The
  residual runs linearly across the span between the 7-day mean of
  (observed minus predicted) before the span and the same after it.
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
(-10.0) equals the outlet row's `bed_z_m`; change both together. The tide-shape
check stays off until the placeholder is replaced.

## Poughkeepsie series

USGS NWIS `dv`, parameter 72137, converted from cfs to m3/s with
`scripts/fetch_nwis_daily.py`. 245 days are negative (minimum -705 m3/s; a
tidally filtered flow can reverse). 289 days are estimated (`A:e`). About 13 %
of 90-day training windows have a NaN after warm-up, so Poughkeepsie leaves the
loss in those windows. This is not a defect.

## Rebuilding the bundle

`scripts/extract_merit_bundle.py` with `--domain-from-comid 73005067
--time 2001-05-01:2019-12-31 --extra-obs <72137 csv> --nan-obs-gages 73005067`.
The fabric and the Q' and observation stores are workstation paths.

## Baseline (summed Q', test window)

Poughkeepsie NSE 0.144, KGE 0.560. Green Island NSE 0.688, KGE 0.801. Croton
NSE 0.835, KGE 0.867.

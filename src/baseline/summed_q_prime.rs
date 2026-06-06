//! Summed Q' baseline — ports `~/projects/ddr/scripts/summed_q_prime.py`.
//!
//! Pipeline:
//!   1. Open CONUS / per-gauge adjacency + streamflow + observations
//!   2. For each valid gauge, derive upstream COMID set via the COO indices
//!   3. Bulk-read daily Qr for the union of upstream divides
//!   4. Per-gauge `nansum` across that gauge's upstream slice → predictions
//!   5. Bulk-read daily USGS observations for the gauge set
//!   6. `Metrics::compute` → NSE/RMSE/KGE/bias/FHV/FLV per gauge
//!
//! The work splits cleanly into `compute` (opens stores, drives icechunk
//! reads) and `assemble_from_arrays` (pure reduction over already-loaded
//! arrays). Tests target the pure half.

use std::collections::{BTreeMap, BTreeSet};
use std::path::PathBuf;

use chrono::{Duration, NaiveDate};
use ndarray::Array2;

use crate::config::Config;
use crate::data::error::DataError;
use crate::data::ids::{Comid, Staid};
use crate::data::store::{
    ConusAdjacencyStore, GageMetadata, GagesAdjacencyStore, StreamflowStore,
    UsgsObservationsStore,
};
use crate::training::metrics::Metrics;

#[derive(Debug, thiserror::Error)]
pub enum BaselineError {
    #[error("config missing field: {0}")]
    ConfigMissing(&'static str),
    #[error("invalid date {value:?}: {source}")]
    BadDate {
        value: String,
        source: chrono::ParseError,
    },
    #[error("end_time {end} is before start_time {start}")]
    InvertedWindow { start: NaiveDate, end: NaiveDate },
    #[error("no evaluation gauges found in {gages:?} or all were missing from {adj:?}")]
    NoGauges { gages: PathBuf, adj: PathBuf },
    #[error(transparent)]
    Data(#[from] DataError),
}

pub struct SummedQPrime {
    /// (n_gauges, n_days), m³/s — per-gauge summed upstream Qr.
    pub predictions: Array2<f32>,
    /// (n_gauges, n_days), m³/s — USGS daily observations, NaN where missing.
    pub observations: Array2<f32>,
    /// Gauges present in `gages_adjacency` (subset of the gauges CSV).
    pub gage_ids: Vec<Staid>,
    /// Daily timestamps covering `[start_time, end_time]` inclusive.
    pub time_range_daily: Vec<NaiveDate>,
    pub metrics: Metrics,
}

/// Compute the summed Q' baseline against the testing-mode eval window
/// in `test_cfg`. Opens all four icechunk/zarr stores; intended to run
/// from `ddrs plan` or `ddrs run`.
pub fn compute(test_cfg: &Config) -> Result<SummedQPrime, BaselineError> {
    let ds = test_cfg
        .data_sources
        .as_ref()
        .ok_or(BaselineError::ConfigMissing("data_sources"))?;
    let exp = test_cfg
        .experiment
        .as_ref()
        .ok_or(BaselineError::ConfigMissing("experiment"))?;

    let (start, end, n_days) = parse_window(&exp.start_time, &exp.end_time)?;

    // TODO(managed-adjacency Task 7): replace with resolved paths from adjacency cache.
    let conus_path = ds
        .conus_adjacency
        .as_ref()
        .ok_or(BaselineError::ConfigMissing(
            "conus_adjacency — adjacency paths not configured; set conus_adjacency/gages_adjacency \
             explicitly or wait for managed adjacency build (Task 7)",
        ))?;
    let gages_adj_path = ds
        .gages_adjacency
        .as_ref()
        .ok_or(BaselineError::ConfigMissing(
            "gages_adjacency — adjacency paths not configured; set conus_adjacency/gages_adjacency \
             explicitly or wait for managed adjacency build (Task 7)",
        ))?;

    let conus = ConusAdjacencyStore::open(conus_path)?;
    let gage_meta = GageMetadata::open(&ds.gages)?;
    let all_staids: Vec<Staid> = gage_meta.rows.iter().map(|r| r.staid.clone()).collect();

    let gages_adj = GagesAdjacencyStore::open(gages_adj_path, &all_staids)?;
    // Preserve CSV order; drop gauges without a subgraph.
    let valid_staids: Vec<Staid> = all_staids
        .iter()
        .filter(|s| gages_adj.get(s).is_some())
        .cloned()
        .collect();
    if valid_staids.is_empty() {
        return Err(BaselineError::NoGauges {
            gages: ds.gages.clone(),
            adj: gages_adj_path.clone(),
        });
    }

    // Per-gauge upstream COMIDs + union over all gauges.
    let mut gauge_basins: BTreeMap<Staid, Vec<Comid>> = BTreeMap::new();
    let mut all_needed: BTreeSet<Comid> = BTreeSet::new();
    for staid in &valid_staids {
        let sg = gages_adj
            .get(staid)
            .expect("valid_staids filtered for presence above");
        let upstream = sg.upstream_comids(&conus);
        all_needed.extend(upstream.iter().copied());
        gauge_basins.insert(staid.clone(), upstream);
    }
    let all_needed_sorted: Vec<Comid> = all_needed.into_iter().collect();

    eprintln!(
        "summed Q': {} gauges × {} unique upstream divides × {} days",
        valid_staids.len(),
        all_needed_sorted.len(),
        n_days,
    );

    let streamflow = StreamflowStore::open(&ds.streamflow)?;
    let observations = UsgsObservationsStore::open(&ds.observations)?;

    let qr_daily = streamflow.read_window_daily(start, n_days, &all_needed_sorted)?;
    let obs_daily = observations.read_window_daily(start, n_days, &valid_staids)?;

    Ok(assemble_from_arrays(
        &qr_daily,
        &obs_daily,
        &all_needed_sorted,
        &gauge_basins,
        valid_staids,
        start,
        n_days,
    ))
}

/// Pure reducer over already-loaded daily arrays. Split from `compute` so
/// tests can drive it without opening real icechunk stores.
///
/// `qr_daily`: `(n_days, n_needed)` — daily Qr per upstream divide.
/// `obs_daily`: `(n_days, n_gauges)` — daily observations per gauge,
///   columns aligned to `gage_ids`.
/// `all_needed_sorted`: column order of `qr_daily`.
/// `gauge_basins`: COMIDs upstream of each gauge.
pub fn assemble_from_arrays(
    qr_daily: &Array2<f32>,
    obs_daily: &Array2<f32>,
    all_needed_sorted: &[Comid],
    gauge_basins: &BTreeMap<Staid, Vec<Comid>>,
    gage_ids: Vec<Staid>,
    start: NaiveDate,
    n_days: usize,
) -> SummedQPrime {
    debug_assert_eq!(qr_daily.shape()[0], n_days);
    debug_assert_eq!(qr_daily.shape()[1], all_needed_sorted.len());
    debug_assert_eq!(obs_daily.shape()[0], n_days);
    debug_assert_eq!(obs_daily.shape()[1], gage_ids.len());

    // COMID → column index in qr_daily, for O(1) lookup per gauge.
    let comid_pos: BTreeMap<Comid, usize> = all_needed_sorted
        .iter()
        .enumerate()
        .map(|(i, c)| (*c, i))
        .collect();

    let mut predictions = Array2::<f32>::zeros((gage_ids.len(), n_days));
    for (g_i, staid) in gage_ids.iter().enumerate() {
        let upstream = gauge_basins
            .get(staid)
            .expect("gauge_basins must contain every gage_ids entry");
        let positions: Vec<usize> = upstream
            .iter()
            .filter_map(|c| comid_pos.get(c).copied())
            .collect();
        for t in 0..n_days {
            let s: f32 = positions
                .iter()
                .map(|&p| qr_daily[(t, p)])
                .filter(|v| v.is_finite())
                .sum();
            predictions[(g_i, t)] = s;
        }
    }

    // Transpose observations to (n_gauges, n_days) so Metrics::compute is happy.
    let observations = obs_daily.t().to_owned();

    let metrics = Metrics::compute(&predictions, &observations);

    let time_range_daily: Vec<NaiveDate> = (0..n_days)
        .map(|d| start + Duration::days(d as i64))
        .collect();

    SummedQPrime {
        predictions,
        observations,
        gage_ids,
        time_range_daily,
        metrics,
    }
}

fn parse_window(start: &str, end: &str) -> Result<(NaiveDate, NaiveDate, usize), BaselineError> {
    let start_date = NaiveDate::parse_from_str(start, "%Y/%m/%d").map_err(|e| {
        BaselineError::BadDate {
            value: start.to_string(),
            source: e,
        }
    })?;
    let end_date = NaiveDate::parse_from_str(end, "%Y/%m/%d").map_err(|e| BaselineError::BadDate {
        value: end.to_string(),
        source: e,
    })?;
    if end_date < start_date {
        return Err(BaselineError::InvertedWindow {
            start: start_date,
            end: end_date,
        });
    }
    // Inclusive of both endpoints, matching pandas date_range(inclusive="both").
    let n_days = (end_date - start_date).num_days() as usize + 1;
    Ok((start_date, end_date, n_days))
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::array;

    #[test]
    fn assemble_sums_upstream_divides_per_gauge() {
        // 2 gauges, 3 days. Upstream sets overlap.
        //   gauge A: COMIDs 100, 200       (positions 0, 1 in needed)
        //   gauge B: COMIDs 200, 300, 400  (positions 1, 2, 3)
        let all_needed = vec![Comid(100), Comid(200), Comid(300), Comid(400)];
        // (n_days=3, n_needed=4): each row = one day, each col = one divide.
        let qr = array![
            [1.0_f32, 2.0, 3.0, 4.0],
            [10.0, 20.0, 30.0, 40.0],
            [100.0, 200.0, 300.0, 400.0],
        ];
        // (n_days=3, n_gauges=2)
        let obs = array![[3.0_f32, 9.0], [30.0, 90.0], [300.0, 900.0]];

        let mut basins = BTreeMap::new();
        basins.insert(
            Staid::from("A"),
            vec![Comid(100), Comid(200)],
        );
        basins.insert(
            Staid::from("B"),
            vec![Comid(200), Comid(300), Comid(400)],
        );

        let gage_ids = vec![Staid::from("A"), Staid::from("B")];
        let start = NaiveDate::from_ymd_opt(2000, 1, 1).unwrap();
        let q = assemble_from_arrays(&qr, &obs, &all_needed, &basins, gage_ids, start, 3);

        // gauge A: row sums of cols 0,1 → 3, 30, 300
        assert_eq!(q.predictions.row(0).to_vec(), vec![3.0, 30.0, 300.0]);
        // gauge B: row sums of cols 1,2,3 → 9, 90, 900
        assert_eq!(q.predictions.row(1).to_vec(), vec![9.0, 90.0, 900.0]);

        // Observations transposed.
        assert_eq!(q.observations.row(0).to_vec(), vec![3.0, 30.0, 300.0]);
        assert_eq!(q.observations.row(1).to_vec(), vec![9.0, 90.0, 900.0]);

        // Perfect-prediction sanity — NSE should be 1.0 on both gauges
        // (zero residual, non-zero observed variance).
        for (g, &nse) in q.metrics.nse.iter().enumerate() {
            assert!(
                (nse - 1.0).abs() < 1e-5,
                "gauge {g}: expected NSE=1.0, got {nse}"
            );
        }

        // Time axis covers the requested span.
        assert_eq!(q.time_range_daily.len(), 3);
        assert_eq!(q.time_range_daily[0], start);
        assert_eq!(
            q.time_range_daily[2],
            NaiveDate::from_ymd_opt(2000, 1, 3).unwrap()
        );
    }

    #[test]
    fn assemble_treats_nan_qr_as_zero() {
        let all_needed = vec![Comid(1), Comid(2)];
        // Day 0 col 0 is NaN; should be skipped, sum is just col 1.
        let qr = array![[f32::NAN, 5.0], [3.0, 7.0]];
        let obs = array![[5.0_f32], [10.0]];
        let mut basins = BTreeMap::new();
        basins.insert(Staid::from("G"), vec![Comid(1), Comid(2)]);
        let q = assemble_from_arrays(
            &qr,
            &obs,
            &all_needed,
            &basins,
            vec![Staid::from("G")],
            NaiveDate::from_ymd_opt(2000, 1, 1).unwrap(),
            2,
        );
        assert_eq!(q.predictions.row(0).to_vec(), vec![5.0, 10.0]);
    }

    #[test]
    fn parse_window_inclusive_endpoints() {
        let (s, e, n) = parse_window("2000/01/01", "2000/01/03").unwrap();
        assert_eq!(s, NaiveDate::from_ymd_opt(2000, 1, 1).unwrap());
        assert_eq!(e, NaiveDate::from_ymd_opt(2000, 1, 3).unwrap());
        assert_eq!(n, 3);
    }

    #[test]
    fn parse_window_rejects_inverted() {
        assert!(parse_window("2000/01/03", "2000/01/01").is_err());
    }
}

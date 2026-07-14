//! Real-data pretraining pipeline for the disaggregation head
//! (`src/nn/disagg_head.rs`). Trains DIRECTLY against real, measured USGS
//! hourly streamflow (`src/data/store/camels_hourly.rs`) + real NLDAS hourly
//! precip (bundled in the same file), with zero second-model bias, before
//! any weights are brought into the production routing pipeline. See
//! `docs/2026-07-1x-disagg-real-pretrain-*.md` for the campaign writeup.
//!
//! Not part of the production training/eval data path — `ddrs run` never
//! touches this module. Entry point is `src/bin/pretrain_disagg.rs`.
//!
//! ## Mass-balance invariant (load-bearing, tested at every level)
//!
//! `daily_q` is ALWAYS derived as the mean of the same 24 real hourly values
//! used as the target — never sourced from a different record — so the
//! target is reachable by construction (`DisaggHead`'s softmax output's
//! daily mean must equal its `daily_q` input exactly). `extract_complete_days`
//! enforces this by construction (not by convention): it computes
//! `daily_q_m3s` FROM the 24-hour block it returns, so the invariant cannot
//! drift even if this module is refactored later.

use chrono::NaiveDate;
use ndarray::{Array1, Array2};
use rand::rngs::StdRng;
use rand::seq::SliceRandom;
use rand::SeedableRng;

use crate::data::dataset::normalize_precip;
use crate::data::error::{DataError, Result};
use crate::data::ids::Staid;
use crate::data::store::gage_csv::GageRow;
use crate::data::store::icechunk::UsgsObservationsStore;

/// `Q[m³/s] = qobs[mm/hr] · area[km²] / 3.6`
/// (mm/hr·km² = 1e-3 m · 1e6 m² / 3600 s = 1e3/3600 m³/s = area/3.6 m³/s).
pub fn qobs_mm_hr_to_m3s(qobs_mm_hr: f32, drain_sqkm: f64) -> f32 {
    qobs_mm_hr * (drain_sqkm / 3.6) as f32
}

/// One pretraining sample: a single (gauge, day) with the REAL 24-hour
/// target and the `daily_q` feature derived from it. `daily_q_m3s` is
/// GUARANTEED to equal `mean(target_hourly_m3s)` — see module docs and
/// [`extract_complete_days`].
#[derive(Clone, Debug)]
pub struct PretrainRow {
    pub day_index: usize,
    pub daily_q_m3s: f32,
    pub target_hourly_m3s: [f32; 24],
}

/// Scan an hourly m³/s series in 24-hour blocks (block `d` = hours
/// `[d*24, d*24+24)`), keeping only blocks where all 24 hours are finite.
/// `daily_q_m3s` is computed as the mean of exactly the 24 values returned —
/// this is what makes the mass-balance invariant hold by construction, not
/// by convention.
pub fn extract_complete_days(qobs_m3s: &Array1<f32>) -> Vec<PretrainRow> {
    let n_hours = qobs_m3s.len();
    let n_days = n_hours / 24;
    let mut rows = Vec::new();
    for d in 0..n_days {
        let block = &qobs_m3s.as_slice().unwrap()[d * 24..d * 24 + 24];
        if block.iter().any(|v| !v.is_finite()) {
            continue;
        }
        let mut target = [0f32; 24];
        target.copy_from_slice(block);
        let daily_q_m3s = target.iter().sum::<f32>() / 24.0;
        rows.push(PretrainRow {
            day_index: d,
            daily_q_m3s,
            target_hourly_m3s: target,
        });
    }
    rows
}

/// Scan for maximal runs of CONSECUTIVE complete calendar days (day `d`,
/// `d+1`, `d+2`, ... all present with no gap and no incomplete hours) and
/// slice each run into non-overlapping windows of exactly `window_days`
/// consecutive [`PretrainRow`]s. This is what makes `DisaggHead`'s
/// `boundary_blend` meaningful at all: it only has an effect when `d_use`
/// (the number of days in a single `forward()` call) is `> 1`, which
/// requires TRUE calendar adjacency between the rows in a window —
/// [`extract_complete_days`] alone does not guarantee day `d` and `d+1` are
/// both present (either could be missing due to a real data gap), so
/// stitching arbitrary "nearby" rows together would silently pair
/// non-adjacent days and corrupt the boundary-blend continuity semantics.
pub fn extract_complete_day_windows(qobs_m3s: &Array1<f32>, window_days: usize) -> Vec<Vec<PretrainRow>> {
    let all_days = extract_complete_days(qobs_m3s);
    let mut windows = Vec::new();
    let mut run_start = 0usize;
    while run_start < all_days.len() {
        // Extend the run while day indices stay consecutive.
        let mut run_end = run_start + 1;
        while run_end < all_days.len() && all_days[run_end].day_index == all_days[run_end - 1].day_index + 1 {
            run_end += 1;
        }
        let run = &all_days[run_start..run_end];
        for chunk in run.chunks_exact(window_days) {
            windows.push(chunk.to_vec());
        }
        run_start = run_end;
    }
    windows
}

/// Mass-balance invariant check (explicit, not just structural) — asserts
/// every row's `daily_q_m3s` equals the mean of its own `target_hourly_m3s`
/// to float32 tolerance. Call this in tests AND periodically during the
/// Phase 3 training loop so a future refactor that breaks the invariant is
/// caught immediately, not discovered after a full pretrain run.
pub fn assert_mass_balance(rows: &[PretrainRow], tol: f32) {
    for r in rows {
        let mean = r.target_hourly_m3s.iter().sum::<f32>() / 24.0;
        assert!(
            (mean - r.daily_q_m3s).abs() <= tol,
            "mass-balance invariant violated at day {}: mean(target)={mean} != daily_q={}",
            r.day_index,
            r.daily_q_m3s
        );
    }
}

/// Build precip features for a set of complete-day rows from an ALREADY
/// per-window-z-scored precip array (i.e. the output of
/// [`normalize_precip`] applied to the whole window at once — matching
/// production's exact per-window statistics, not a per-day recomputation).
/// `precip_norm_window` is `(n_hours, 1)` for one gauge.
pub fn slice_precip_features(precip_norm_window: &Array2<f32>, day_indices: &[usize]) -> Vec<[f32; 24]> {
    day_indices
        .iter()
        .map(|&d| {
            let mut feat = [0f32; 24];
            for h in 0..24 {
                feat[h] = precip_norm_window[(d * 24 + h, 0)];
            }
            feat
        })
        .collect()
}

/// Normalize a single gauge's raw hourly precip `(n_hours,)` the SAME way
/// production does (`normalize_precip`, which operates on `(T, N)` — here
/// `N=1`), reused verbatim to avoid pretrain/production feature drift.
pub fn normalize_gauge_precip(raw_precip: Array1<f32>) -> Array2<f32> {
    let n = raw_precip.len();
    let as_2d = raw_precip.into_shape_with_order((n, 1)).unwrap();
    normalize_precip(as_2d)
}

// ---------------------------------------------------------------------------
// Train/val/test gauge split
// ---------------------------------------------------------------------------

#[derive(Debug, Clone)]
pub struct SplitManifest {
    pub seed: u64,
    pub train: Vec<Staid>,
    pub val: Vec<Staid>,
    pub test: Vec<Staid>,
}

/// Area-stratified (drainage-area tercile), seeded, deterministic gauge
/// split. Proportions ~70% train / 10% val / 20% test (matches the
/// pre-registered plan's 355/50/100 out of 505 overlap gauges — computed
/// proportionally here so it stays correct if the overlap set size changes).
pub fn build_split_manifest(gauges: &[GageRow], seed: u64) -> SplitManifest {
    let mut sorted: Vec<&GageRow> = gauges.iter().collect();
    sorted.sort_by(|a, b| a.drain_sqkm.partial_cmp(&b.drain_sqkm).unwrap());
    let n = sorted.len();
    let tercile_size = n.div_ceil(3);

    let mut train = Vec::new();
    let mut val = Vec::new();
    let mut test = Vec::new();
    let mut rng = StdRng::seed_from_u64(seed);

    for tercile in sorted.chunks(tercile_size) {
        let mut idx: Vec<&GageRow> = tercile.to_vec();
        idx.shuffle(&mut rng);
        let n_t = idx.len();
        let n_val = (n_t as f64 * 0.10).round() as usize;
        let n_test = (n_t as f64 * 0.20).round() as usize;
        for (i, row) in idx.into_iter().enumerate() {
            if i < n_val {
                val.push(row.staid.clone());
            } else if i < n_val + n_test {
                test.push(row.staid.clone());
            } else {
                train.push(row.staid.clone());
            }
        }
    }

    SplitManifest {
        seed,
        train,
        val,
        test,
    }
}

// ---------------------------------------------------------------------------
// Reconciliation QA: camels_hourly daily-mean-of-hourly vs production's
// own usgs_daily_observations store, for the SAME gauge.
// ---------------------------------------------------------------------------

#[derive(Debug, Clone)]
pub struct ReconciliationResult {
    pub staid: Staid,
    /// median(camels_daily / production_daily) over overlapping finite days.
    pub median_ratio: f32,
    pub correlation: f32,
    pub n_overlap_days: usize,
    /// True iff median_ratio in [0.9, 1.1] and correlation >= 0.90.
    pub keep: bool,
}

/// Compare this module's daily-mean-of-real-hourly (m³/s, already unit
/// converted) against production's `UsgsObservationsStore` daily values for
/// the same gauge/period. A gauge failing this check means the two REAL
/// records disagree — evidence of a units/timezone/gauge-mismatch bug in
/// OUR pipeline (or a genuinely untrustworthy record), not something to
/// silently paper over.
///
/// Gate thresholds calibrated empirically against the real 505-gauge
/// overlap (`examples/pretrain_reconciliation_check.rs`, 1998-2013): an
/// initial `correlation >= 0.98` bar excluded 280/505 gauges, but 263 of
/// those had a perfectly healthy ratio (median 1.015) and failed purely on
/// correlation in the 0.92-0.98 range -- normal day-to-day noise between
/// two independently-processed real daily records, not a bug. Only 16/505
/// (3%, an expected rate for regulated/impaired basins) had a genuinely bad
/// ratio. Relaxed to correlation >= 0.90, which keeps ~489/505 and still
/// excludes real outliers.
pub fn reconcile_gauge(
    staid: &Staid,
    camels_daily_m3s: &[(NaiveDate, f32)],
    obs_store: &UsgsObservationsStore,
) -> Result<ReconciliationResult> {
    if camels_daily_m3s.is_empty() {
        return Err(DataError::Malformed {
            path: obs_store.path.clone(),
            message: format!("reconcile_gauge({staid}): no camels daily values supplied"),
        });
    }
    let start = camels_daily_m3s.iter().map(|(d, _)| *d).min().unwrap();
    let end = camels_daily_m3s.iter().map(|(d, _)| *d).max().unwrap();
    let n_days = (end - start).num_days() as usize + 1;

    let obs = obs_store.read_window_daily(start, n_days, std::slice::from_ref(staid))?;

    let mut ratios = Vec::new();
    let mut xs = Vec::new();
    let mut ys = Vec::new();
    for &(date, camels_val) in camels_daily_m3s {
        if !camels_val.is_finite() || camels_val <= 0.0 {
            continue;
        }
        let idx = (date - start).num_days() as usize;
        let obs_val = obs[(idx, 0)];
        if !obs_val.is_finite() || obs_val <= 0.0 {
            continue;
        }
        ratios.push(camels_val / obs_val);
        xs.push(camels_val as f64);
        ys.push(obs_val as f64);
    }

    let n_overlap_days = ratios.len();
    if n_overlap_days < 30 {
        return Ok(ReconciliationResult {
            staid: staid.clone(),
            median_ratio: f32::NAN,
            correlation: f32::NAN,
            n_overlap_days,
            keep: false,
        });
    }

    ratios.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let median_ratio = ratios[ratios.len() / 2];
    let correlation = pearson(&xs, &ys) as f32;

    let keep = (0.9..=1.1).contains(&median_ratio) && correlation >= 0.90;

    Ok(ReconciliationResult {
        staid: staid.clone(),
        median_ratio,
        correlation,
        n_overlap_days,
        keep,
    })
}

fn pearson(xs: &[f64], ys: &[f64]) -> f64 {
    let n = xs.len() as f64;
    let mx = xs.iter().sum::<f64>() / n;
    let my = ys.iter().sum::<f64>() / n;
    let mut cov = 0.0;
    let mut vx = 0.0;
    let mut vy = 0.0;
    for i in 0..xs.len() {
        let dx = xs[i] - mx;
        let dy = ys[i] - my;
        cov += dx * dy;
        vx += dx * dx;
        vy += dy * dy;
    }
    if vx <= 0.0 || vy <= 0.0 {
        return 0.0;
    }
    cov / (vx.sqrt() * vy.sqrt())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn qobs_conversion_matches_hand_calc() {
        // 1 mm/hr over 36 km^2 = 36e6 m^2 * 1e-3 m = 36000 m^3 in 1 hour
        // = 36000/3600 = 10 m^3/s.
        let q = qobs_mm_hr_to_m3s(1.0, 36.0);
        assert!((q - 10.0).abs() < 1e-4, "got {q}");
    }

    #[test]
    fn extract_complete_days_drops_incomplete_blocks_and_conserves_mass() {
        // 3 days: day0 complete, day1 has one NaN (dropped), day2 complete.
        let mut data = vec![1.0f32; 24]; // day0: all 1.0, mean 1.0
        data.extend(vec![2.0f32; 24]);
        data[24 + 5] = f32::NAN; // corrupt day1
        data.extend((0..24).map(|h| h as f32)); // day2: 0..23, mean 11.5

        let arr = Array1::from_vec(data);
        let rows = extract_complete_days(&arr);
        assert_eq!(rows.len(), 2, "day1 (has a NaN) must be dropped");
        assert_eq!(rows[0].day_index, 0);
        assert!((rows[0].daily_q_m3s - 1.0).abs() < 1e-6);
        assert_eq!(rows[1].day_index, 2);
        assert!((rows[1].daily_q_m3s - 11.5).abs() < 1e-6);

        // Mass-balance invariant: must hold for every emitted row.
        assert_mass_balance(&rows, 1e-5);
    }

    #[test]
    fn extract_complete_day_windows_only_pairs_true_calendar_adjacency() {
        // 5 days: 0,1 complete+consecutive; 2 has a gap (dropped); 3,4
        // complete+consecutive. window_days=2 must yield exactly ONE window
        // (day0+day1) -- day2's absence must NOT let day1 pair with day3
        // (they are not adjacent calendar days).
        let mut data = vec![1.0f32; 24]; // day0
        data.extend(vec![2.0f32; 24]); // day1
        data.extend(vec![3.0f32; 24]);
        data[2 * 24 + 3] = f32::NAN; // corrupt day2
        data.extend(vec![4.0f32; 24]); // day3
        data.extend(vec![5.0f32; 24]); // day4

        let arr = Array1::from_vec(data);
        let windows = extract_complete_day_windows(&arr, 2);
        assert_eq!(windows.len(), 2, "day0+day1 and day3+day4 -- day2's gap must not bridge a fake pair");
        assert_eq!(windows[0][0].day_index, 0);
        assert_eq!(windows[0][1].day_index, 1);
        assert_eq!(windows[1][0].day_index, 3);
        assert_eq!(windows[1][1].day_index, 4);
    }

    #[test]
    fn extract_complete_day_windows_drops_odd_remainder() {
        // 3 consecutive complete days, window_days=2 -> exactly 1 window
        // (days 0+1); day2 has no partner within this run and is dropped,
        // not silently paired with a non-adjacent day from elsewhere.
        let mut data = vec![1.0f32; 24];
        data.extend(vec![2.0f32; 24]);
        data.extend(vec![3.0f32; 24]);
        let arr = Array1::from_vec(data);
        let windows = extract_complete_day_windows(&arr, 2);
        assert_eq!(windows.len(), 1);
        assert_eq!(windows[0][0].day_index, 0);
        assert_eq!(windows[0][1].day_index, 1);
    }

    #[test]
    #[should_panic(expected = "mass-balance invariant violated")]
    fn assert_mass_balance_catches_a_corrupted_row() {
        let bad_row = PretrainRow {
            day_index: 0,
            daily_q_m3s: 999.0, // deliberately wrong -- not mean(target)
            target_hourly_m3s: [1.0; 24],
        };
        assert_mass_balance(&[bad_row], 1e-5);
    }

    #[test]
    fn normalize_gauge_precip_matches_production_normalize_precip() {
        // Single-column reuse must be byte-identical to calling
        // normalize_precip directly on a (T,1) array.
        let raw = Array1::from_vec(vec![0.0f32, 0.0, 5.0, 0.0, 0.0]);
        let via_helper = normalize_gauge_precip(raw.clone());
        let direct = normalize_precip(raw.into_shape_with_order((5, 1)).unwrap());
        assert_eq!(via_helper, direct);
    }

    #[test]
    fn split_manifest_is_deterministic_and_covers_every_gauge() {
        let gauges: Vec<GageRow> = (0..30)
            .map(|i| GageRow {
                staid: Staid::new(&format!("{:08}", i)),
                staname: String::new(),
                drain_sqkm: (i as f64) * 10.0 + 1.0,
                lat_gage: 0.0,
                lng_gage: 0.0,
                comid: None,
                comid_drain_sqkm: None,
                comid_unitarea_sqkm: None,
                abs_diff: None,
                da_valid: None,
                flow_scale: None,
            })
            .collect();

        let m1 = build_split_manifest(&gauges, 42);
        let m2 = build_split_manifest(&gauges, 42);
        assert_eq!(m1.train, m2.train, "same seed must be deterministic");
        assert_eq!(m1.val, m2.val);
        assert_eq!(m1.test, m2.test);

        let total = m1.train.len() + m1.val.len() + m1.test.len();
        assert_eq!(total, 30, "every gauge must land in exactly one split");

        // No gauge appears in more than one split.
        let mut all: Vec<&Staid> = m1.train.iter().chain(&m1.val).chain(&m1.test).collect();
        all.sort();
        all.dedup();
        assert_eq!(all.len(), 30);

        let m3 = build_split_manifest(&gauges, 7);
        assert_ne!(m1.train, m3.train, "different seed should (almost certainly) differ");
    }

    #[test]
    fn pearson_matches_known_values() {
        let xs = [1.0, 2.0, 3.0, 4.0, 5.0];
        let ys = [2.0, 4.0, 6.0, 8.0, 10.0];
        assert!((pearson(&xs, &ys) - 1.0).abs() < 1e-9, "perfect positive correlation");
        let ys_neg = [10.0, 8.0, 6.0, 4.0, 2.0];
        assert!((pearson(&xs, &ys_neg) + 1.0).abs() < 1e-9, "perfect negative correlation");
    }
}

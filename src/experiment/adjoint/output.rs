//! netCDF + tidy-CSV writers for the adjoint study.

use std::io::Write;
use std::path::Path;

use super::super::BoxError;

/// One anchor of the kernel functional.
#[derive(Debug, Clone, serde::Serialize)]
pub struct AnchorRecord {
    pub day_idx: usize,
    pub date: chrono::NaiveDate,
    pub obs: f32,
    /// `"high"` or `"low"`.
    pub kind: String,
    pub window_start_day: usize,
    /// Hour index of the anchor within the window.
    pub t0: usize,
}

/// One residual window.
#[derive(Debug, Clone, serde::Serialize)]
pub struct WindowRecord {
    pub start_day: usize,
    pub start_date: chrono::NaiveDate,
    pub gauge_mean_residual: f32,
    pub gauge_mse: f32,
    pub n_valid: usize,
}

/// Everything the study computed for one gauge under one arm.
pub struct GaugeResult {
    pub staid: String,
    pub arm: String,
    pub run_id: String,
    pub checkpoint: String,
    pub comids: Vec<i64>,
    pub dist_to_gauge_m: Vec<f32>,
    pub q_prime_mean: Vec<f32>,
    pub gauge_row: usize,
    pub window_days: usize,
    pub lag_days: usize,
    pub tail_days: usize,
    pub tau: u32,
    pub warmup: usize,
    // kernel
    pub anchors: Vec<AnchorRecord>,
    /// `[anchor][reach][lag_day]`
    pub kernel: Vec<Vec<Vec<f32>>>,
    /// `[anchor][lag_hour]`
    pub kernel_hourly: Vec<Vec<f32>>,
    /// `[anchor][reach]`
    pub kernel_mass: Vec<Vec<f32>>,
    /// `[anchor][reach]`
    pub kernel_mean_lag_days: Vec<Vec<f32>>,
    /// `[anchor][reach]` window-mean hourly inflow in each anchor's window —
    /// reaches at the `discharge` clamp floor have a zero gradient by
    /// construction, so this is how to tell a dead reach from a dry one.
    pub q_prime_mean_by_anchor: Vec<Vec<f32>>,
    /// `[anchor][reach]` path-sum of Muskingum K (window-mean over the lag
    /// horizon) from reach to gauge, in days — the hydraulic travel time the
    /// kernel mean lag should reproduce.
    pub hydraulic_lag_days: Vec<Vec<f32>>,
    /// `[anchor][reach]` same, evaluated at the anchor hour's discharge only.
    pub hydraulic_lag_t0_days: Vec<Vec<f32>>,
    // volume
    pub volume_window_start_day: Option<usize>,
    pub volume_sens: Vec<f32>,
    /// `[hour]`
    pub volume_profile: Vec<f32>,
    // residual
    pub residual_windows: Vec<WindowRecord>,
    /// `[window][reach]`
    pub residual_attr_by_window: Vec<Vec<f32>>,
    pub residual_attr: Vec<f32>,
}

pub fn write_gauge_netcdf(path: &Path, r: &GaugeResult) -> Result<(), BoxError> {
    if path.exists() {
        std::fs::remove_file(path)?;
    }
    let mut f = netcdf::create(path)?;
    let n = r.comids.len();
    let n_anchor = r.anchors.len();
    let n_lag_day = r.lag_days + 1;
    let n_lag_hour = 24 * r.lag_days;
    let n_window = r.residual_windows.len();
    let n_hour = r.volume_profile.len();

    f.add_attribute("staid", r.staid.as_str())?;
    f.add_attribute("arm", r.arm.as_str())?;
    f.add_attribute("run_id", r.run_id.as_str())?;
    f.add_attribute("checkpoint", r.checkpoint.as_str())?;
    f.add_attribute("window_days", r.window_days as i64)?;
    f.add_attribute("lag_days", r.lag_days as i64)?;
    f.add_attribute("tail_days", r.tail_days as i64)?;
    f.add_attribute("tau", r.tau as i64)?;
    f.add_attribute("warmup_days", r.warmup as i64)?;
    f.add_attribute("gauge_row", r.gauge_row as i64)?;
    f.add_attribute("ddrs_version", env!("CARGO_PKG_VERSION"))?;
    f.add_attribute(
        "kernel_definition",
        "d Q_gauge(t0) / d q'(reach, hour), t0 = noon of the anchor day; kernel(anchor, reach, lag_day) \
         sums the 24 source hours t0 - (24*lag_day + h); kernel_hourly sums reaches at each hourly lag",
    )?;
    f.add_attribute(
        "volume_definition",
        "d [sum_{t >= warmup} Q_gauge(t)] / d q'(reach, hour), mean over source hours in \
         [warmup*24, T - tail_days*24); ~1 upstream by mass conservation",
    )?;
    f.add_attribute(
        "residual_definition",
        "d [mean over valid days d >= warmup of (Qbar_gauge(d) - obs(d))^2] / d q'(reach, hour) \
         = (2/n) sum_d (Qbar_d - obs_d) dQbar_d/dq'; Qbar via tau_trim_and_downsample(tau), pooled day d \
         scored against obs day d; mean over source hours in [warmup*24, T - tail_days*24), averaged over \
         windows. Positive = this reach's inflow arrives when the gauge over-predicts (reduce it to cut error).",
    )?;

    f.add_dimension("reach", n)?;
    f.add_dimension("anchor", n_anchor.max(1))?;
    f.add_dimension("lag_day", n_lag_day)?;
    f.add_dimension("lag_hour", n_lag_hour)?;
    f.add_dimension("window", n_window.max(1))?;
    f.add_dimension("hour", n_hour.max(1))?;

    {
        let mut v = f.add_variable::<i64>("COMID", &["reach"])?;
        v.put_values(&r.comids, ..)?;
        v.put_attribute("long_name", "MERIT reach identifier (gauge subgraph, topological order)")?;
    }
    put_f32(&mut f, "dist_to_gauge_m", &["reach"], &r.dist_to_gauge_m, "along-channel distance from reach outlet to gauge outlet", "m")?;
    put_f32(&mut f, "q_prime_mean", &["reach"], &r.q_prime_mean, "window-mean hourly lateral inflow (first kernel window)", "m3 s-1")?;
    {
        let is_gauge: Vec<i32> = (0..n).map(|i| (i == r.gauge_row) as i32).collect();
        let mut v = f.add_variable::<i32>("is_gauge_reach", &["reach"])?;
        v.put_values(&is_gauge, ..)?;
    }

    if n_anchor > 0 {
        let flat: Vec<f32> = r.kernel.iter().flat_map(|a| a.iter().flat_map(|reach| reach.iter().copied())).collect();
        put_f32(&mut f, "kernel", &["anchor", "reach", "lag_day"], &flat, "kernel by daily lag", "dimensionless")?;
        let flat: Vec<f32> = r.kernel_hourly.iter().flat_map(|a| a.iter().copied()).collect();
        put_f32(&mut f, "kernel_hourly", &["anchor", "lag_hour"], &flat, "kernel summed over reaches by hourly lag", "dimensionless")?;
        let flat: Vec<f32> = r.kernel_mass.iter().flat_map(|a| a.iter().copied()).collect();
        put_f32(&mut f, "kernel_mass", &["anchor", "reach"], &flat, "sum of kernel over lags 0..24*lag_days", "dimensionless")?;
        let flat: Vec<f32> = r.kernel_mean_lag_days.iter().flat_map(|a| a.iter().copied()).collect();
        put_f32(&mut f, "kernel_mean_lag_days", &["anchor", "reach"], &flat, "kernel-weighted mean lag", "days")?;
        let flat: Vec<f32> = r.q_prime_mean_by_anchor.iter().flat_map(|a| a.iter().copied()).collect();
        put_f32(&mut f, "q_prime_mean_by_anchor", &["anchor", "reach"], &flat, "window-mean hourly lateral inflow in the anchor's window", "m3 s-1")?;
        if !r.hydraulic_lag_days.is_empty() {
            let flat: Vec<f32> = r.hydraulic_lag_days.iter().flat_map(|a| a.iter().copied()).collect();
            put_f32(&mut f, "hydraulic_lag_days", &["anchor", "reach"], &flat, "path sum of Muskingum K = L/c from reach to gauge, mean over the lag horizon", "days")?;
            let flat: Vec<f32> = r.hydraulic_lag_t0_days.iter().flat_map(|a| a.iter().copied()).collect();
            put_f32(&mut f, "hydraulic_lag_t0_days", &["anchor", "reach"], &flat, "path sum of Muskingum K = L/c from reach to gauge at the anchor hour", "days")?;
        }
        let days: Vec<i32> = r.anchors.iter().map(|a| a.day_idx as i32).collect();
        put_i32(&mut f, "anchor_day_idx", &["anchor"], &days, "anchor day index on the eval axis")?;
        let epoch: Vec<i32> = r
            .anchors
            .iter()
            .map(|a| (a.date - chrono::NaiveDate::from_ymd_opt(1970, 1, 1).unwrap()).num_days() as i32)
            .collect();
        put_i32(&mut f, "anchor_date", &["anchor"], &epoch, "anchor date, days since 1970-01-01")?;
        let obs: Vec<f32> = r.anchors.iter().map(|a| a.obs).collect();
        put_f32(&mut f, "anchor_obs", &["anchor"], &obs, "observed discharge on the anchor day", "m3 s-1")?;
        let kind: Vec<i32> = r.anchors.iter().map(|a| (a.kind == "high") as i32).collect();
        put_i32(&mut f, "anchor_is_high", &["anchor"], &kind, "1 = high-flow anchor, 0 = low-flow anchor")?;
        let ws: Vec<i32> = r.anchors.iter().map(|a| a.window_start_day as i32).collect();
        put_i32(&mut f, "anchor_window_start_day", &["anchor"], &ws, "window start day index")?;
        let t0: Vec<i32> = r.anchors.iter().map(|a| a.t0 as i32).collect();
        put_i32(&mut f, "anchor_t0_hour", &["anchor"], &t0, "anchor hour index within the window")?;
    }

    if let Some(start) = r.volume_window_start_day {
        f.add_attribute("volume_window_start_day", start as i64)?;
        put_f32(&mut f, "volume_sens", &["reach"], &r.volume_sens, "volume sensitivity (time-mean)", "dimensionless")?;
        put_f32(&mut f, "volume_profile", &["hour"], &r.volume_profile, "volume sensitivity, reach-mean by source hour", "dimensionless")?;
    }

    if n_window > 0 {
        put_f32(&mut f, "residual_attr", &["reach"], &r.residual_attr, "squared-error sensitivity to inflow, mean over windows", "m3 s-1")?;
        let flat: Vec<f32> = r.residual_attr_by_window.iter().flat_map(|w| w.iter().copied()).collect();
        put_f32(&mut f, "residual_attr_by_window", &["window", "reach"], &flat, "squared-error sensitivity to inflow per window", "m3 s-1")?;
        let m: Vec<f32> = r.residual_windows.iter().map(|w| w.gauge_mean_residual).collect();
        put_f32(&mut f, "residual_gauge_mean", &["window"], &m, "gauge mean (pred - obs) over valid post-warmup days", "m3 s-1")?;
        let mse: Vec<f32> = r.residual_windows.iter().map(|w| w.gauge_mse).collect();
        put_f32(&mut f, "residual_gauge_mse", &["window"], &mse, "gauge mean squared error over valid post-warmup days", "m6 s-2")?;
        let nv: Vec<i32> = r.residual_windows.iter().map(|w| w.n_valid as i32).collect();
        put_i32(&mut f, "residual_n_valid", &["window"], &nv, "valid observation days in the window")?;
        let ws: Vec<i32> = r.residual_windows.iter().map(|w| w.start_day as i32).collect();
        put_i32(&mut f, "residual_window_start_day", &["window"], &ws, "window start day index")?;
        let epoch: Vec<i32> = r
            .residual_windows
            .iter()
            .map(|w| (w.start_date - chrono::NaiveDate::from_ymd_opt(1970, 1, 1).unwrap()).num_days() as i32)
            .collect();
        put_i32(&mut f, "residual_window_start_date", &["window"], &epoch, "window start date, days since 1970-01-01")?;
    }
    Ok(())
}

fn put_f32(
    f: &mut netcdf::FileMut,
    name: &str,
    dims: &[&str],
    vals: &[f32],
    long_name: &str,
    units: &str,
) -> Result<(), BoxError> {
    let mut v = f.add_variable::<f32>(name, dims)?;
    v.put_values(vals, ..)?;
    v.put_attribute("long_name", long_name)?;
    v.put_attribute("units", units)?;
    Ok(())
}

fn put_i32(f: &mut netcdf::FileMut, name: &str, dims: &[&str], vals: &[i32], long_name: &str) -> Result<(), BoxError> {
    let mut v = f.add_variable::<i32>(name, dims)?;
    v.put_values(vals, ..)?;
    v.put_attribute("long_name", long_name)?;
    Ok(())
}

pub const SUMMARY_HEADER: &str = "arm,staid,comid,reach_row,is_gauge_reach,dist_to_gauge_m,q_prime_mean,volume_sens,residual_attr,kernel_mass_high,kernel_mass_low,kernel_mean_lag_high_days,kernel_mean_lag_low_days";

/// Append one row per reach to `<arm>/summary.csv` (header written when new).
pub fn append_summary(path: &Path, r: &GaugeResult) -> Result<(), BoxError> {
    let new = !path.exists();
    let mut w = std::fs::OpenOptions::new().create(true).append(true).open(path)?;
    if new {
        writeln!(w, "{SUMMARY_HEADER}")?;
    }
    let n = r.comids.len();
    let mean_over = |kind: &str, table: &Vec<Vec<f32>>, reach: usize| -> f32 {
        let vals: Vec<f32> = r
            .anchors
            .iter()
            .enumerate()
            .filter(|(_, a)| a.kind == kind)
            .map(|(i, _)| table[i][reach])
            .filter(|v| v.is_finite())
            .collect();
        if vals.is_empty() { f32::NAN } else { vals.iter().sum::<f32>() / vals.len() as f32 }
    };
    for i in 0..n {
        writeln!(
            w,
            "{},{},{},{},{},{},{},{},{},{},{},{},{}",
            r.arm,
            r.staid,
            r.comids[i],
            i,
            (i == r.gauge_row) as i32,
            r.dist_to_gauge_m[i],
            r.q_prime_mean[i],
            r.volume_sens.get(i).copied().unwrap_or(f32::NAN),
            r.residual_attr.get(i).copied().unwrap_or(f32::NAN),
            mean_over("high", &r.kernel_mass, i),
            mean_over("low", &r.kernel_mass, i),
            mean_over("high", &r.kernel_mean_lag_days, i),
            mean_over("low", &r.kernel_mean_lag_days, i),
        )?;
    }
    Ok(())
}

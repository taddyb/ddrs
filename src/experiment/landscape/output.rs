//! netCDF + CSV writers for the landscape study.

use std::io::Write;
use std::path::Path;

use crate::experiment::BoxError;

#[derive(Debug, Clone, serde::Serialize)]
pub struct NewtonStep {
    pub alpha: [f32; 3],
    pub loss: f32,
    pub grad_norm: f32,
}

pub struct Slice {
    pub name: String,
    pub axis_a: Vec<f32>,
    pub axis_b: Vec<f32>,
    /// Unit vectors in (α_n, α_p, α_q) spanned by the two grid axes.
    pub basis_a: [f32; 3],
    pub basis_b: [f32; 3],
    /// Row-major (grid_a, grid_b).
    pub loss: Vec<f32>,
    pub nse: Vec<f32>,
    pub clamped: Vec<f32>,
}

/// Window-0 daily series, present only when `landscape.series: true`. See
/// `LandscapeResult::series`.
pub struct SeriesData {
    /// Observed discharge, m3/s (NaN = missing), length `d`.
    pub obs_daily: Vec<f32>,
    /// Routed discharge at the gauge, alpha = 0 (trained point), m3/s, length `d`.
    pub routed_daily_trained: Vec<f32>,
    /// Routed discharge at the gauge, alpha = alpha_star, m3/s, length `d`.
    pub routed_daily_star: Vec<f32>,
    /// No-routing baseline: sum over the gauge's subgraph reaches of the
    /// daily q_prime inflow (`WindowData.tensors.q_prime_daily`, already
    /// daily resolution -- not pooled from the inner hourly `q_prime`),
    /// m3/s, length `d`.
    pub summed_qprime_daily: Vec<f32>,
}

pub struct LandscapeResult {
    pub staid: String,
    pub arm: String,
    pub run_id: String,
    pub checkpoint: String,
    pub n_reach: usize,
    pub window_days: usize,
    pub window_starts: Vec<usize>,
    pub sigma: f32,
    pub loss0: f32,
    pub nse0: f32,
    pub grad0: [f32; 3],
    pub hess0: [[f32; 3]; 3],
    pub loss_star: f32,
    pub nse_star: f32,
    pub nse_star_windows: Vec<f32>,
    pub alpha_star: [f32; 3],
    pub grad_star: [f32; 3],
    pub hess_star: [[f32; 3]; 3],
    pub eigval_star: [f32; 3],
    /// Columns are eigenvectors (row index = α component n, p, q).
    pub eigvec_star: [[f32; 3]; 3],
    /// Trained point (α = 0) in the eigenbasis of H(α*): c_k = v_kᵀ(0 − α*).
    pub coord_trained: [f32; 3],
    pub tolerances: Vec<f32>,
    /// Half-widths w_k = sqrt(2 ε_L / λ_k) per tolerance.
    pub half_widths: Vec<[f32; 3]>,
    /// Unit direction in α that most increases hydraulic path travel time.
    pub celerity_dir: [f32; 3],
    pub newton_path: Vec<NewtonStep>,
    pub slices: Vec<Slice>,
    /// Slice-centering convention used to compute `slices`: "optimum"
    /// (default) or "trained". See `LandscapeSpec::slice_center`.
    pub slice_center: String,
    pub clamped_frac_star: f32,
    /// Per-gauge effective `max_clamped` bound used by `run_gauge`'s Newton
    /// acceptance test: `max(spec.max_clamped, spec.max_clamped_min_reaches
    /// / (n_reach * n_active_params))`. Equal to `spec.max_clamped` except
    /// for small basins where the floor widens it. See
    /// `LandscapeSpec::max_clamped_min_reaches`.
    pub max_clamped_effective: f32,
    /// True if the final accepted point has `clamped_frac > max_clamped`, or
    /// the descent stopped because every Newton trial step exceeded it.
    pub hit_range_bound: bool,
    /// True if at least one Newton iteration used the steepest-descent
    /// fallback (the damped Newton step was not a descent direction). See
    /// `objective::newton_step_capped`.
    pub used_gradient_fallback: bool,
    /// Trained (alpha = 0) per-reach physical fields, length n_reach.
    pub n0: Vec<f32>,
    pub p0: Vec<f32>,
    pub q0: Vec<f32>,
    /// Reach COMIDs, same order/length as n0/p0/q0.
    pub comid: Vec<i64>,
    /// Trained (window 0) per-reach channel slope (dimensionless, m/m),
    /// clamped to `params.attribute_minimums.slope`. Same order as n0/p0/q0.
    pub slope: Vec<f32>,
    /// Trained (window 0) per-reach channel length, meters. Same order as
    /// n0/p0/q0.
    pub length: Vec<f32>,
    /// Index of the gauge's own reach in the `reach` dimension (window 0).
    pub gauge_reach_row: usize,
    /// Mean observed daily discharge (m3/s) over all valid days (finite,
    /// >= 0, day >= warmup) across all landscape windows.
    pub obs_mean_q_m3s: f32,
    /// Number of valid observed days that fed `obs_mean_q_m3s`.
    pub obs_n_valid_days: usize,
    /// Per-reach `dL/d ln x` at alpha = 0, `[n, p, q]`; `Some` only when
    /// `landscape.reach_grad: true`.
    pub reach_grad0: Option<[Vec<f32>; 3]>,
    /// Per-reach `dL/d ln x` at alpha*, `[n, p, q]`; `Some` only when
    /// `landscape.reach_grad: true`.
    pub reach_grad_star: Option<[Vec<f32>; 3]>,
    /// Per-reach along-channel distance to the gauge outlet, meters; empty
    /// unless `landscape.reach_grad: true`.
    pub dist_to_gauge_m: Vec<f32>,
    /// `active[k]` for `k` in `(n, p_spatial, q_spatial)`: true when the
    /// parameter is a real model parameter (in the head's
    /// `learnable_parameters`), false when it's fixed at `params.defaults`.
    /// See `Objective::active`.
    pub active: [bool; 3],
    /// `LandscapeSpec::objective` used to drive the Newton search for this
    /// gauge: "nse-batch" or "kge".
    pub objective: String,
    /// KGE at the trained point (mean over windows). Always computed,
    /// regardless of `objective`.
    pub kge0: f32,
    /// KGE at the per-gauge optimum (mean over windows).
    pub kge_star: f32,
    pub kge_star_windows: Vec<f32>,
    /// Window-0 daily series; `Some` only when `landscape.series: true`.
    pub series: Option<SeriesData>,
    /// `LandscapeSpec::period` resolved for this run: "testing" or "training".
    pub period: String,
    /// Eval-axis start date (`InfluenceContext.axis.start`), "%Y-%m-%d" --
    /// the Testing- or Training-window axis selected by `period`. Combined
    /// with `window_start_day[0]` and (when `series` is set) the `day`
    /// index, this dates every window and every series entry.
    pub axis_start_date: String,
}

pub fn write_landscape_netcdf(path: &Path, r: &LandscapeResult) -> Result<(), BoxError> {
    if path.exists() {
        std::fs::remove_file(path)?;
    }
    let mut f = netcdf::create(path)?;
    f.add_attribute("staid", r.staid.as_str())?;
    f.add_attribute("arm", r.arm.as_str())?;
    f.add_attribute("run_id", r.run_id.as_str())?;
    f.add_attribute("checkpoint", r.checkpoint.as_str())?;
    f.add_attribute("n_reach", r.n_reach as i64)?;
    f.add_attribute("window_days", r.window_days as i64)?;
    f.add_attribute("sigma_obs_training", r.sigma as f64)?;
    f.add_attribute("alpha_components", "log-multipliers on (n, p_spatial, q_spatial) applied to the trained physical fields; alpha = 0 is the trained point")?;
    f.add_attribute("objective", r.objective.as_str())?;
    f.add_attribute("eigvec_layout", "eigvec[component, k]: column k is the k-th eigenvector (descending eigenvalue) of the Hessian at alpha_star")?;
    f.add_attribute("coord_trained_definition", "c_k = v_k^T (0 - alpha_star): trained point in the eigenbasis of H(alpha_star)")?;
    f.add_attribute("half_width_definition", "w_k = sqrt(2 * tol * L(alpha_star) / lambda_k): behavioural half-width along eigenvector k (quadratic approximation)")?;
    f.add_attribute("celerity_dir_definition", "unit alpha direction that most increases the hydraulic mean path travel time (sum L/c) at alpha = 0")?;
    f.add_attribute("clamped_frac_star", r.clamped_frac_star as f64)?;
    f.add_attribute("max_clamped_effective", r.max_clamped_effective as f64)?;
    f.add_attribute("hit_range_bound", r.hit_range_bound as i32)?;
    f.add_attribute("used_gradient_fallback", r.used_gradient_fallback as i32)?;
    f.add_attribute("slice_center", r.slice_center.as_str())?;
    f.add_attribute("gauge_reach_row", r.gauge_reach_row as i64)?;
    f.add_attribute("obs_mean_q_m3s", r.obs_mean_q_m3s as f64)?;
    f.add_attribute("obs_n_valid_days", r.obs_n_valid_days as i64)?;
    f.add_attribute("period", r.period.as_str())?;
    f.add_attribute("axis_start_date", r.axis_start_date.as_str())?;
    const PARAM_NAMES: [&str; 3] = ["n", "p_spatial", "q_spatial"];
    let active_params: Vec<&str> = PARAM_NAMES.iter().zip(r.active.iter()).filter(|(_, &a)| a).map(|(&n, _)| n).collect();
    f.add_attribute("active_params", active_params.join(","))?;

    f.add_dimension("alpha", 3)?;
    f.add_dimension("k", 3)?;
    f.add_dimension("window", r.window_starts.len())?;
    f.add_dimension("tol", r.tolerances.len())?;
    f.add_dimension("newton", r.newton_path.len())?;
    f.add_dimension("reach", r.n0.len())?;
    // `grid: 0` produces no slices, so skip the plane/ga/gb dims and grid variables.
    if !r.slices.is_empty() {
        f.add_dimension("plane", r.slices.len())?;
        f.add_dimension("ga", r.slices[0].axis_a.len())?;
        f.add_dimension("gb", r.slices[0].axis_b.len())?;
    }
    if let Some(s) = &r.series {
        f.add_dimension("day", s.obs_daily.len())?;
    }

    let active_i: Vec<i32> = r.active.iter().map(|&a| a as i32).collect();
    let mut active_var = f.add_variable::<i32>("active", &["alpha"])?;
    active_var.put_values(&active_i, ..)?;
    active_var.put_attribute("long_name", "1 if the alpha component is a learned model parameter, 0 if fixed at params.defaults")?;

    let mut put = |name: &str, dims: &[&str], vals: &[f32], long: &str| -> Result<(), BoxError> {
        let mut v = f.add_variable::<f32>(name, dims)?;
        v.put_values(vals, ..)?;
        v.put_attribute("long_name", long)?;
        Ok(())
    };
    put("window_start_day", &["window"], &r.window_starts.iter().map(|s| *s as f32).collect::<Vec<_>>(), "window start day index on the eval axis")?;
    put("loss0", &[], &[r.loss0], "L at trained point")?;
    put("nse0", &[], &[r.nse0], "NSE at trained point (mean over windows)")?;
    put("loss_star", &[], &[r.loss_star], "L at per-gauge optimum")?;
    put("nse_star", &[], &[r.nse_star], "NSE at per-gauge optimum")?;
    put("nse_star_windows", &["window"], &r.nse_star_windows, "NSE per window at optimum")?;
    put("kge0", &[], &[r.kge0], "KGE at trained point (mean over windows)")?;
    put("kge_star", &[], &[r.kge_star], "KGE at per-gauge optimum (mean over windows)")?;
    put("kge_star_windows", &["window"], &r.kge_star_windows, "KGE per window at optimum")?;
    put("alpha_star", &["alpha"], &r.alpha_star, "per-gauge optimum in log-multiplier space")?;
    put("grad0", &["alpha"], &r.grad0, "dL/dalpha at trained point")?;
    put("grad_star", &["alpha"], &r.grad_star, "dL/dalpha at optimum")?;
    let flat = |m: &[[f32; 3]; 3]| -> Vec<f32> { m.iter().flat_map(|row| row.iter().copied()).collect() };
    put("hess0", &["alpha", "k"], &flat(&r.hess0), "Hessian (FD of gradient) at trained point")?;
    put("hess_star", &["alpha", "k"], &flat(&r.hess_star), "Hessian at optimum")?;
    put("eigval_star", &["k"], &r.eigval_star, "eigenvalues of hess_star, descending")?;
    put("eigvec_star", &["alpha", "k"], &flat(&r.eigvec_star), "eigenvectors of hess_star (columns)")?;
    put("coord_trained", &["k"], &r.coord_trained, "trained point in eigenbasis of hess_star")?;
    put("tolerances", &["tol"], &r.tolerances, "loss tolerance as fraction of L(alpha_star)")?;
    let hw: Vec<f32> = r.half_widths.iter().flat_map(|w| w.iter().copied()).collect();
    put("half_width", &["tol", "k"], &hw, "behavioural half-width along eigenvector k")?;
    put("celerity_dir", &["alpha"], &r.celerity_dir, "hydraulic travel-time gradient direction (unit)")?;
    let na: Vec<f32> = r.newton_path.iter().flat_map(|s| s.alpha.iter().copied()).collect();
    put("newton_alpha", &["newton", "alpha"], &na, "Newton iterate")?;
    put("newton_loss", &["newton"], &r.newton_path.iter().map(|s| s.loss).collect::<Vec<_>>(), "L at iterate")?;
    put("newton_grad_norm", &["newton"], &r.newton_path.iter().map(|s| s.grad_norm).collect::<Vec<_>>(), "|dL/dalpha| at iterate")?;
    if !r.slices.is_empty() {
        let cat = |sel: &dyn Fn(&Slice) -> &Vec<f32>| -> Vec<f32> { r.slices.iter().flat_map(|s| sel(s).iter().copied()).collect() };
        put("grid_axis_a", &["plane", "ga"], &cat(&|s| &s.axis_a), "grid coordinate along basis_a")?;
        put("grid_axis_b", &["plane", "gb"], &cat(&|s| &s.axis_b), "grid coordinate along basis_b")?;
        put("grid_loss", &["plane", "ga", "gb"], &cat(&|s| &s.loss), "L on the slice")?;
        put("grid_nse", &["plane", "ga", "gb"], &cat(&|s| &s.nse), "NSE on the slice")?;
        put("grid_clamped", &["plane", "ga", "gb"], &cat(&|s| &s.clamped), "fraction of reach-parameters clamped at a range edge")?;
        let ba: Vec<f32> = r.slices.iter().flat_map(|s| s.basis_a.iter().copied()).collect();
        let bb: Vec<f32> = r.slices.iter().flat_map(|s| s.basis_b.iter().copied()).collect();
        put("basis_a", &["plane", "alpha"], &ba, "unit alpha vector of grid axis a")?;
        put("basis_b", &["plane", "alpha"], &bb, "unit alpha vector of grid axis b")?;
    }
    put("n0", &["reach"], &r.n0, "trained (alpha = 0) per-reach Manning's n")?;
    put("p0", &["reach"], &r.p0, "trained (alpha = 0) per-reach Leopold-Maddock p")?;
    put("q0", &["reach"], &r.q0, "trained (alpha = 0) per-reach Leopold-Maddock q")?;
    put("slope", &["reach"], &r.slope, "per-reach channel slope (dimensionless, m/m), clamped to params.attribute_minimums.slope")?;
    put("length", &["reach"], &r.length, "per-reach channel length (m)")?;
    if let Some(rg) = &r.reach_grad0 {
        put("reach_grad0_n", &["reach"], &rg[0], "dL/d ln(n) at alpha = 0, per reach (loss per unit ln parameter)")?;
        put("reach_grad0_p", &["reach"], &rg[1], "dL/d ln(p_spatial) at alpha = 0, per reach (loss per unit ln parameter)")?;
        put("reach_grad0_q", &["reach"], &rg[2], "dL/d ln(q_spatial) at alpha = 0, per reach (loss per unit ln parameter)")?;
    }
    if let Some(rg) = &r.reach_grad_star {
        put("reach_grad_star_n", &["reach"], &rg[0], "dL/d ln(n) at alpha*, per reach (loss per unit ln parameter)")?;
        put("reach_grad_star_p", &["reach"], &rg[1], "dL/d ln(p_spatial) at alpha*, per reach (loss per unit ln parameter)")?;
        put("reach_grad_star_q", &["reach"], &rg[2], "dL/d ln(q_spatial) at alpha*, per reach (loss per unit ln parameter)")?;
    }
    if !r.dist_to_gauge_m.is_empty() {
        put("dist_to_gauge_m", &["reach"], &r.dist_to_gauge_m, "along-channel distance from reach outlet to gauge outlet, meters")?;
    }
    if let Some(s) = &r.series {
        put("obs_daily", &["day"], &s.obs_daily, "observed discharge at the gauge, window 0, m3/s (NaN = missing)")?;
        put("routed_daily_trained", &["day"], &s.routed_daily_trained, "routed discharge at the gauge, window 0, alpha = 0 (trained point), m3/s")?;
        put("routed_daily_star", &["day"], &s.routed_daily_star, "routed discharge at the gauge, window 0, alpha = alpha_star (per-gauge optimum), m3/s")?;
        put(
            "summed_qprime_daily",
            &["day"],
            &s.summed_qprime_daily,
            "no-routing baseline: sum over the gauge's subgraph reaches of the daily q_prime inflow (WindowData.tensors.q_prime_daily), window 0, m3/s",
        )?;
    }
    // `put`'s last use is above; from here on `f` is borrowed directly (NLL
    // releases the closure's borrow once it's no longer called).
    f.add_attribute("plane_names", r.slices.iter().map(|s| s.name.as_str()).collect::<Vec<_>>().join(","))?;
    let mut comid_var = f.add_variable::<i64>("comid", &["reach"])?;
    comid_var.put_values(&r.comid, ..)?;
    comid_var.put_attribute("long_name", "reach COMID, same order as n0/p0/q0")?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn minimal_result(slices: Vec<Slice>) -> LandscapeResult {
        LandscapeResult {
            staid: "test".into(),
            arm: "arm".into(),
            run_id: "run".into(),
            checkpoint: "ckpt".into(),
            n_reach: 2,
            window_days: 90,
            window_starts: vec![0],
            sigma: 1.0,
            loss0: 1.0,
            nse0: 0.5,
            grad0: [0.0; 3],
            hess0: [[0.0; 3]; 3],
            loss_star: 0.5,
            nse_star: 0.6,
            nse_star_windows: vec![0.6],
            alpha_star: [0.0; 3],
            grad_star: [0.0; 3],
            hess_star: [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
            eigval_star: [1.0, 1.0, 1.0],
            eigvec_star: [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
            coord_trained: [0.0; 3],
            tolerances: vec![0.05],
            half_widths: vec![[1.0; 3]],
            celerity_dir: [1.0, 0.0, 0.0],
            newton_path: vec![NewtonStep { alpha: [0.0; 3], loss: 1.0, grad_norm: 0.0 }],
            slices,
            slice_center: "optimum".into(),
            clamped_frac_star: 0.0,
            max_clamped_effective: 0.05,
            hit_range_bound: false,
            used_gradient_fallback: false,
            n0: vec![0.03, 0.04],
            p0: vec![21.0, 21.0],
            q0: vec![0.5, 0.5],
            comid: vec![1, 2],
            reach_grad0: None,
            reach_grad_star: None,
            dist_to_gauge_m: Vec::new(),
            active: [true, true, true],
            slope: vec![1e-3, 2e-3],
            length: vec![500.0, 750.0],
            gauge_reach_row: 1,
            obs_mean_q_m3s: 12.5,
            obs_n_valid_days: 360,
            objective: "nse-batch".into(),
            kge0: 0.4,
            kge_star: 0.7,
            kge_star_windows: vec![0.7],
            series: None,
            period: "testing".into(),
            axis_start_date: "1995-10-01".into(),
        }
    }

    #[test]
    fn grid_zero_writes_no_slice_dims_or_variables() {
        let path = std::env::temp_dir().join(format!("ddrs-landscape-grid0-{}.nc", std::process::id()));
        let r = minimal_result(Vec::new());
        write_landscape_netcdf(&path, &r).unwrap();

        let f = netcdf::open(&path).unwrap();
        assert!(f.dimension("plane").is_none());
        assert!(f.dimension("ga").is_none());
        assert!(f.dimension("gb").is_none());
        assert!(f.variable("grid_nse").is_none());
        assert!(f.variable("n0").is_some());
        let plane_names: String = f.attribute("plane_names").unwrap().value().unwrap().try_into().unwrap();
        assert_eq!(plane_names, "");
        let hit: i32 = f.attribute("hit_range_bound").unwrap().value().unwrap().try_into().unwrap();
        assert_eq!(hit, 0);

        std::fs::remove_file(&path).unwrap();
    }

    #[test]
    fn active_mask_writes_attribute_and_variable() {
        let path = std::env::temp_dir().join(format!("ddrs-landscape-active-{}.nc", std::process::id()));
        let mut r = minimal_result(Vec::new());
        r.active = [true, false, true]; // p_spatial fixed
        write_landscape_netcdf(&path, &r).unwrap();

        let f = netcdf::open(&path).unwrap();
        let active_params: String = f.attribute("active_params").unwrap().value().unwrap().try_into().unwrap();
        assert_eq!(active_params, "n,q_spatial");
        let active_var = f.variable("active").unwrap();
        let vals: Vec<i32> = active_var.get_values(..).unwrap();
        assert_eq!(vals, vec![1, 0, 1]);

        std::fs::remove_file(&path).unwrap();
    }

    #[test]
    fn slope_length_and_obs_mean_q_are_written() {
        let path = std::env::temp_dir().join(format!("ddrs-landscape-slope-{}.nc", std::process::id()));
        let r = minimal_result(Vec::new());
        write_landscape_netcdf(&path, &r).unwrap();

        let f = netcdf::open(&path).unwrap();
        let slope: Vec<f32> = f.variable("slope").unwrap().get_values(..).unwrap();
        assert_eq!(slope, vec![1e-3, 2e-3]);
        let length: Vec<f32> = f.variable("length").unwrap().get_values(..).unwrap();
        assert_eq!(length, vec![500.0, 750.0]);
        let gauge_row: i64 = f.attribute("gauge_reach_row").unwrap().value().unwrap().try_into().unwrap();
        assert_eq!(gauge_row, 1);
        let mean_q: f64 = f.attribute("obs_mean_q_m3s").unwrap().value().unwrap().try_into().unwrap();
        assert!((mean_q - 12.5).abs() < 1e-6);
        let n_valid: i64 = f.attribute("obs_n_valid_days").unwrap().value().unwrap().try_into().unwrap();
        assert_eq!(n_valid, 360);

        std::fs::remove_file(&path).unwrap();
    }

    #[test]
    fn objective_and_kge_are_written() {
        let path = std::env::temp_dir().join(format!("ddrs-landscape-objective-{}.nc", std::process::id()));
        let mut r = minimal_result(Vec::new());
        r.objective = "kge".into();
        r.kge0 = 0.4;
        r.kge_star = 0.7;
        r.kge_star_windows = vec![0.7];
        write_landscape_netcdf(&path, &r).unwrap();

        let f = netcdf::open(&path).unwrap();
        let objective: String = f.attribute("objective").unwrap().value().unwrap().try_into().unwrap();
        assert_eq!(objective, "kge");
        let kge0: Vec<f32> = f.variable("kge0").unwrap().get_values(..).unwrap();
        assert_eq!(kge0, vec![0.4]);
        let kge_star: Vec<f32> = f.variable("kge_star").unwrap().get_values(..).unwrap();
        assert_eq!(kge_star, vec![0.7]);
        let kge_star_windows: Vec<f32> = f.variable("kge_star_windows").unwrap().get_values(..).unwrap();
        assert_eq!(kge_star_windows, vec![0.7]);

        std::fs::remove_file(&path).unwrap();
    }

    #[test]
    fn series_off_by_default_writes_no_day_dim_or_variables() {
        let path = std::env::temp_dir().join(format!("ddrs-landscape-noseries-{}.nc", std::process::id()));
        let r = minimal_result(Vec::new());
        write_landscape_netcdf(&path, &r).unwrap();

        let f = netcdf::open(&path).unwrap();
        assert!(f.dimension("day").is_none());
        assert!(f.variable("obs_daily").is_none());
        // `axis_start_date` and `period` are always written (the resolved
        // eval window), independent of `series`.
        let axis_start_date: String = f.attribute("axis_start_date").unwrap().value().unwrap().try_into().unwrap();
        assert_eq!(axis_start_date, "1995-10-01");
        let period: String = f.attribute("period").unwrap().value().unwrap().try_into().unwrap();
        assert_eq!(period, "testing");

        std::fs::remove_file(&path).unwrap();
    }

    #[test]
    fn series_writes_day_dim_and_four_series() {
        let path = std::env::temp_dir().join(format!("ddrs-landscape-series-{}.nc", std::process::id()));
        let mut r = minimal_result(Vec::new());
        r.axis_start_date = "1980-01-01".into();
        r.series = Some(SeriesData {
            obs_daily: vec![1.0, f32::NAN, 3.0],
            routed_daily_trained: vec![1.1, 2.1, 2.9],
            routed_daily_star: vec![1.05, 2.05, 2.95],
            summed_qprime_daily: vec![0.9, 1.9, 2.8],
        });
        write_landscape_netcdf(&path, &r).unwrap();

        let f = netcdf::open(&path).unwrap();
        assert_eq!(f.dimension("day").unwrap().len(), 3);
        let axis_start_date: String = f.attribute("axis_start_date").unwrap().value().unwrap().try_into().unwrap();
        assert_eq!(axis_start_date, "1980-01-01");
        let obs_daily: Vec<f32> = f.variable("obs_daily").unwrap().get_values(..).unwrap();
        assert_eq!(obs_daily[0], 1.0);
        assert!(obs_daily[1].is_nan());
        let routed_trained: Vec<f32> = f.variable("routed_daily_trained").unwrap().get_values(..).unwrap();
        assert_eq!(routed_trained, vec![1.1, 2.1, 2.9]);
        let routed_star: Vec<f32> = f.variable("routed_daily_star").unwrap().get_values(..).unwrap();
        assert_eq!(routed_star, vec![1.05, 2.05, 2.95]);
        let summed_qprime: Vec<f32> = f.variable("summed_qprime_daily").unwrap().get_values(..).unwrap();
        assert_eq!(summed_qprime, vec![0.9, 1.9, 2.8]);

        std::fs::remove_file(&path).unwrap();
    }

    #[test]
    fn reach_grad_writes_reach_dimensioned_variables() {
        let path = std::env::temp_dir().join(format!("ddrs-landscape-reachgrad-{}.nc", std::process::id()));
        let mut r = minimal_result(Vec::new());
        r.reach_grad0 = Some([vec![1.0, 2.0], vec![3.0, 4.0], vec![5.0, 6.0]]);
        r.reach_grad_star = Some([vec![0.1, 0.2], vec![0.3, 0.4], vec![0.5, 0.6]]);
        r.dist_to_gauge_m = vec![0.0, 1500.0];
        write_landscape_netcdf(&path, &r).unwrap();

        let f = netcdf::open(&path).unwrap();
        assert!(f.variable("reach_grad0_n").is_some());
        assert!(f.variable("reach_grad_star_q").is_some());
        assert!(f.variable("dist_to_gauge_m").is_some());

        std::fs::remove_file(&path).unwrap();
    }
}

pub fn append_summary(path: &Path, r: &LandscapeResult) -> Result<(), BoxError> {
    let new = !path.exists();
    let mut w = std::fs::OpenOptions::new().create(true).append(true).open(path)?;
    if new {
        writeln!(w, "arm,staid,n_reach,sigma,loss0,nse0,loss_star,nse_star,alpha_n_star,alpha_p_star,alpha_q_star,mult_n,mult_p,mult_q,lambda1,lambda2,lambda3,v1_n,v1_p,v1_q,v3_n,v3_p,v3_q,c1,c2,c3,hw1_tol0,hw2_tol0,hw3_tol0,cel_n,cel_p,cel_q,cos_v1_cel,grad0_norm,grad_star_norm,newton_iters,clamped_frac_star,hit_range_bound,used_gradient_fallback")?;
    }
    let v = &r.eigvec_star;
    let cos = (0..3).map(|i| v[i][0] * r.celerity_dir[i]).sum::<f32>();
    let hw = r.half_widths.first().copied().unwrap_or([f32::NAN; 3]);
    let n0 = (r.grad0[0].powi(2) + r.grad0[1].powi(2) + r.grad0[2].powi(2)).sqrt();
    let ns = (r.grad_star[0].powi(2) + r.grad_star[1].powi(2) + r.grad_star[2].powi(2)).sqrt();
    writeln!(
        w,
        "{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{}",
        r.arm, r.staid, r.n_reach, r.sigma, r.loss0, r.nse0, r.loss_star, r.nse_star,
        r.alpha_star[0], r.alpha_star[1], r.alpha_star[2], r.alpha_star[0].exp(), r.alpha_star[1].exp(), r.alpha_star[2].exp(),
        r.eigval_star[0], r.eigval_star[1], r.eigval_star[2],
        v[0][0], v[1][0], v[2][0], v[0][2], v[1][2], v[2][2],
        r.coord_trained[0], r.coord_trained[1], r.coord_trained[2], hw[0], hw[1], hw[2],
        r.celerity_dir[0], r.celerity_dir[1], r.celerity_dir[2], cos, n0, ns, r.newton_path.len() - 1, r.clamped_frac_star,
        r.hit_range_bound as i32, r.used_gradient_fallback as i32
    )?;
    Ok(())
}

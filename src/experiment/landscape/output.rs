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
    pub clamped_frac_star: f32,
    /// Trained (alpha = 0) per-reach physical fields, length n_reach.
    pub n0: Vec<f32>,
    pub p0: Vec<f32>,
    pub q0: Vec<f32>,
    /// Reach COMIDs, same order/length as n0/p0/q0.
    pub comid: Vec<i64>,
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
    f.add_attribute("objective", "NSE-batch loss (training objective) restricted to this gauge, mean over windows")?;
    f.add_attribute("eigvec_layout", "eigvec[component, k]: column k is the k-th eigenvector (descending eigenvalue) of the Hessian at alpha_star")?;
    f.add_attribute("coord_trained_definition", "c_k = v_k^T (0 - alpha_star): trained point in the eigenbasis of H(alpha_star)")?;
    f.add_attribute("half_width_definition", "w_k = sqrt(2 * tol * L(alpha_star) / lambda_k): behavioural half-width along eigenvector k (quadratic approximation)")?;
    f.add_attribute("celerity_dir_definition", "unit alpha direction that most increases the hydraulic mean path travel time (sum L/c) at alpha = 0")?;
    f.add_attribute("clamped_frac_star", r.clamped_frac_star as f64)?;

    let g = r.slices[0].axis_a.len();
    f.add_dimension("alpha", 3)?;
    f.add_dimension("k", 3)?;
    f.add_dimension("window", r.window_starts.len())?;
    f.add_dimension("tol", r.tolerances.len())?;
    f.add_dimension("newton", r.newton_path.len())?;
    f.add_dimension("plane", r.slices.len())?;
    f.add_dimension("ga", g)?;
    f.add_dimension("gb", g)?;
    f.add_dimension("reach", r.n0.len())?;

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
    put("n0", &["reach"], &r.n0, "trained (alpha = 0) per-reach Manning's n")?;
    put("p0", &["reach"], &r.p0, "trained (alpha = 0) per-reach Leopold-Maddock p")?;
    put("q0", &["reach"], &r.q0, "trained (alpha = 0) per-reach Leopold-Maddock q")?;
    f.add_attribute("plane_names", r.slices.iter().map(|s| s.name.as_str()).collect::<Vec<_>>().join(","))?;
    let mut comid_var = f.add_variable::<i64>("comid", &["reach"])?;
    comid_var.put_values(&r.comid, ..)?;
    comid_var.put_attribute("long_name", "reach COMID, same order as n0/p0/q0")?;
    Ok(())
}

pub fn append_summary(path: &Path, r: &LandscapeResult) -> Result<(), BoxError> {
    let new = !path.exists();
    let mut w = std::fs::OpenOptions::new().create(true).append(true).open(path)?;
    if new {
        writeln!(w, "arm,staid,n_reach,sigma,loss0,nse0,loss_star,nse_star,alpha_n_star,alpha_p_star,alpha_q_star,mult_n,mult_p,mult_q,lambda1,lambda2,lambda3,v1_n,v1_p,v1_q,v3_n,v3_p,v3_q,c1,c2,c3,hw1_tol0,hw2_tol0,hw3_tol0,cel_n,cel_p,cel_q,cos_v1_cel,grad0_norm,grad_star_norm,newton_iters,clamped_frac_star")?;
    }
    let v = &r.eigvec_star;
    let cos = (0..3).map(|i| v[i][0] * r.celerity_dir[i]).sum::<f32>();
    let hw = r.half_widths.first().copied().unwrap_or([f32::NAN; 3]);
    let n0 = (r.grad0[0].powi(2) + r.grad0[1].powi(2) + r.grad0[2].powi(2)).sqrt();
    let ns = (r.grad_star[0].powi(2) + r.grad_star[1].powi(2) + r.grad_star[2].powi(2)).sqrt();
    writeln!(
        w,
        "{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{}",
        r.arm, r.staid, r.n_reach, r.sigma, r.loss0, r.nse0, r.loss_star, r.nse_star,
        r.alpha_star[0], r.alpha_star[1], r.alpha_star[2], r.alpha_star[0].exp(), r.alpha_star[1].exp(), r.alpha_star[2].exp(),
        r.eigval_star[0], r.eigval_star[1], r.eigval_star[2],
        v[0][0], v[1][0], v[2][0], v[0][2], v[1][2], v[2][2],
        r.coord_trained[0], r.coord_trained[1], r.coord_trained[2], hw[0], hw[1], hw[2],
        r.celerity_dir[0], r.celerity_dir[1], r.celerity_dir[2], cos, n0, ns, r.newton_path.len() - 1, r.clamped_frac_star
    )?;
    Ok(())
}

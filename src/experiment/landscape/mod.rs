//! Per-gauge loss landscape in channel-parameter (log-multiplier) space.
//! Spec: docs/superpowers/specs/2026-09-07-adjoint-landscape-design.md

pub mod objective;
pub mod output;

use std::path::Path;
use std::time::Instant;

use burn::tensor::backend::Backend;
use serde::{Deserialize, Serialize};

use crate::data::ids::Staid;
use crate::experiment::adjoint::gauges::{gauge_list_from_pairs, nested_reference_selection, read_gages_ii_class, write_gauges_csv, GaugeEntry};
use crate::experiment::adjoint::influence::{dist_to_gauge, InfluenceContext};
use crate::experiment::adjoint::{seasonal_window_starts, GaugeSource, GaugeSpec};
use crate::experiment::{BoxError, ExperimentManifest, ResolvedArm};

use self::objective::{eig3, solve3, Objective};
use self::output::{write_landscape_netcdf, LandscapeResult, NewtonStep, Slice};

#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct LandscapeSpec {
    pub gauges: GaugeSpec,
    #[serde(default = "d_window_days")]
    pub window_days: usize,
    #[serde(default = "d_water_year")]
    pub water_year: i32,
    /// How many of the four seasonal windows to use (1–4).
    #[serde(default = "d_n_windows")]
    pub n_windows: usize,
    /// Half-range of the log-multiplier domain (default ln 3).
    #[serde(default = "d_alpha_max")]
    pub alpha_max: f32,
    #[serde(default = "d_grid")]
    pub grid: usize,
    #[serde(default = "d_h")]
    pub fd_step: f32,
    #[serde(default = "d_newton_iters")]
    pub newton_iters: usize,
    /// Loss tolerances (fractions of L(α*)) for behavioural half-widths.
    #[serde(default = "d_tols")]
    pub tolerances: Vec<f32>,
    /// Max acceptable `clamped_frac` for a Newton trial point (else treated
    /// like a non-decrease: halve the step and retry).
    #[serde(default = "d_max_clamped")]
    pub max_clamped: f32,
    /// Which landscape slices through α* to compute. Subset of
    /// `VALID_PLANES`: "n-p", "n-q", "p-q" (parameter-plane slices) and
    /// "stiff-sloppy" (the eigen-plane of H(α*)). Checked against
    /// `VALID_PLANES` at the top of `run_landscape`.
    #[serde(default = "d_planes")]
    pub planes: Vec<String>,
    /// Also compute per-reach `g_i = dL/d ln x_i` (x in n, p, q) at α = 0 and
    /// α*, plus `dist_to_gauge_m`, for the "perturbations in a watershed"
    /// study: where in the network the gauge still constrains parameters,
    /// even where the basin-uniform gradient has gone to zero. Off by
    /// default: it doubles the per-gauge netCDF's reach-dimensioned
    /// variables and costs two extra backward passes.
    #[serde(default)]
    pub reach_grad: bool,
}
fn d_window_days() -> usize { 90 }
fn d_water_year() -> i32 { 2000 }
fn d_n_windows() -> usize { 4 }
fn d_alpha_max() -> f32 { 1.0986123 }
fn d_grid() -> usize { 11 }
fn d_h() -> f32 { 0.05 }
fn d_newton_iters() -> usize { 10 }
fn d_tols() -> Vec<f32> { vec![0.05, 0.10] }
fn d_max_clamped() -> f32 { 0.05 }
fn d_planes() -> Vec<String> {
    VALID_PLANES.iter().map(|s| s.to_string()).collect()
}

/// The four slices `run_gauge` knows how to compute.
pub const VALID_PLANES: [&str; 4] = ["n-p", "n-q", "p-q", "stiff-sloppy"];

impl LandscapeSpec {
    /// Reject any `planes` entry that isn't one of `VALID_PLANES`.
    pub fn validate_planes(&self) -> Result<(), BoxError> {
        for p in &self.planes {
            if !VALID_PLANES.contains(&p.as_str()) {
                return Err(format!(
                    "unknown landscape plane `{p}`; valid planes are {VALID_PLANES:?}"
                )
                .into());
            }
        }
        Ok(())
    }
}

pub struct LandscapeOptions {
    pub max_gauges: Option<usize>,
    pub force_cpu: bool,
    pub jobs: usize,
    pub dry_run: bool,
}

pub fn run_landscape<I: Backend + 'static>(
    spec: &LandscapeSpec,
    arms: &[ResolvedArm],
    out_dir: &Path,
    opts: &LandscapeOptions,
    device: &I::Device,
    manifest: &mut ExperimentManifest,
) -> Result<(), BoxError>
where
    I::FloatTensorPrimitive: 'static,
    I::Device: 'static + Send + Sync,
{
    spec.validate_planes()?;
    if arms.is_empty() {
        return Err("no arms selected".into());
    }
    let mut gauges: Vec<GaugeEntry> = match spec.gauges.source {
        GaugeSource::Explicit => gauge_list_from_pairs(&spec.gauges.pairs),
        GaugeSource::NestedReference => {
            let dbf = spec.gauges.gages_ii_dbf.as_ref().ok_or("nested-reference requires gages_ii_dbf")?;
            let class = read_gages_ii_class(dbf)?;
            let ctx = InfluenceContext::<I>::open(&arms[0], device, opts.force_cpu)?;
            nested_reference_selection(&ctx.dataset, &class, spec.gauges.max_downstream)?
        }
    };
    if let Some(k) = opts.max_gauges {
        gauges.truncate(k);
    }
    write_gauges_csv(&out_dir.join("gauges.csv"), &gauges)?;
    println!(
        "landscape: {} arm(s), {} gauge(s), {} window(s) × {} d, grid {}², α ∈ ±{:.3}, fd_step {}",
        arms.len(), gauges.len(), spec.n_windows, spec.window_days, spec.grid, spec.alpha_max, spec.fd_step
    );
    if opts.dry_run {
        return Ok(());
    }
    let jobs = opts.jobs.max(1);
    let mut all_notes: Vec<String> = Vec::new();
    for chunk in arms.chunks(jobs) {
        let reports: Vec<Result<Vec<String>, BoxError>> = std::thread::scope(|s| {
            let handles: Vec<_> = chunk
                .iter()
                .map(|arm| {
                    let gauges = &gauges;
                    let device = device.clone();
                    s.spawn(move || run_arm::<I>(spec, arm, gauges, out_dir, opts, &device))
                })
                .collect();
            handles.into_iter().map(|h| h.join().unwrap_or_else(|_| Err("arm thread panicked".into()))).collect()
        });
        for r in reports {
            all_notes.extend(r?);
        }
    }
    manifest.notes.extend(all_notes);
    Ok(())
}

fn run_arm<I: Backend + 'static>(
    spec: &LandscapeSpec,
    arm: &ResolvedArm,
    gauges: &[GaugeEntry],
    out_dir: &Path,
    opts: &LandscapeOptions,
    device: &I::Device,
) -> Result<Vec<String>, BoxError>
where
    I::FloatTensorPrimitive: 'static,
    I::Device: 'static,
{
    let t_arm = Instant::now();
    println!("=== arm {} (run {}, checkpoint {}) start ===", arm.name, arm.run_id, arm.checkpoint_label);
    let ctx = InfluenceContext::<I>::open(arm, device, opts.force_cpu)?;
    let arm_dir = out_dir.join(&arm.name);
    std::fs::create_dir_all(arm_dir.join("gauges"))?;
    let seasonal = seasonal_window_starts(&ctx.axis, spec.water_year, spec.window_days)?;
    let starts: Vec<usize> = seasonal.iter().take(spec.n_windows.clamp(1, 4)).map(|(s, _)| *s).collect();
    let mut notes = Vec::new();
    for (gi, g) in gauges.iter().enumerate() {
        let t_g = Instant::now();
        match run_gauge::<I>(&ctx, spec, arm, g, &arm_dir, &starts) {
            Ok(r) => {
                println!(
                    "  [{}] {} done in {:.0} s ({} of {}): L0 {:.4} → L* {:.4} (NSE {:.3} → {:.3}), α* = [{:+.3} {:+.3} {:+.3}], λ = [{:.3e} {:.3e} {:.3e}]",
                    arm.name, g.staid, t_g.elapsed().as_secs_f32(), gi + 1, gauges.len(),
                    r.loss0, r.loss_star, r.nse0, r.nse_star, r.alpha_star[0], r.alpha_star[1], r.alpha_star[2],
                    r.eigval_star[0], r.eigval_star[1], r.eigval_star[2]
                );
            }
            Err(e) => {
                let msg = format!("{}/{}: FAILED — {e}", arm.name, g.staid);
                eprintln!("  {msg}");
                notes.push(msg);
            }
        }
    }
    println!("=== arm {} done in {:.1} s ===", arm.name, t_arm.elapsed().as_secs_f32());
    Ok(notes)
}

fn run_gauge<I: Backend + 'static>(
    ctx: &InfluenceContext<I>,
    spec: &LandscapeSpec,
    arm: &ResolvedArm,
    g: &GaugeEntry,
    arm_dir: &Path,
    starts: &[usize],
) -> Result<LandscapeResult, BoxError>
where
    I::FloatTensorPrimitive: 'static,
    I::Device: 'static,
{
    let staid = Staid::new(&g.staid);
    let obj = Objective::<I>::build(ctx, &staid, starts, spec.window_days)?;
    let h = spec.fd_step;

    // 1. trained point
    let e0 = obj.eval([0.0; 3], true);
    let grad0 = e0.grad.unwrap();
    let hess0 = obj.hessian([0.0; 3], h);
    println!("  [{}] {} trained point: L0 {:.5} NSE {:.3} grad {:?} clamped {:.3}", arm.name, g.staid, e0.loss, e0.nse_mean, grad0, e0.clamped_frac);

    // 2. damped Newton to the per-gauge optimum
    let mut alpha = [0.0f32; 3];
    let mut cur = e0.clone();
    let mut hess = hess0;
    let mut path: Vec<NewtonStep> = vec![NewtonStep { alpha, loss: cur.loss, grad_norm: norm(&grad0) }];
    let mut newton_hit_bound = false;
    for it in 0..spec.newton_iters {
        let gcur = cur.grad.unwrap();
        let (vals, _) = eig3(hess);
        let mu = (-vals[2]).max(0.0) + 1e-3 * vals[0].abs().max(1e-6);
        let mut hd = hess;
        for k in 0..3 {
            hd[k][k] += mu;
        }
        let Some(d) = solve3(hd, [-gcur[0], -gcur[1], -gcur[2]]) else { break };
        // backtracking line search within the domain; a trial whose
        // clamped_frac exceeds max_clamped is unacceptable exactly like a
        // non-decrease: halve the step and retry.
        let mut t = 1.0f32;
        let mut accepted = None;
        let mut saw_within_bound = false;
        for _ in 0..8 {
            let mut a = alpha;
            for k in 0..3 {
                a[k] = (alpha[k] + t * d[k]).clamp(-spec.alpha_max, spec.alpha_max);
            }
            let e = obj.eval(a, true);
            if e.clamped_frac > spec.max_clamped {
                t *= 0.5;
                continue;
            }
            saw_within_bound = true;
            if e.loss < cur.loss {
                accepted = Some((a, e));
                break;
            }
            t *= 0.5;
        }
        let Some((a, e)) = accepted else {
            if saw_within_bound {
                println!("  [{}] {} newton: no decrease at iter {it}, stopping", arm.name, g.staid);
            } else {
                newton_hit_bound = true;
            }
            break;
        };
        let step = ((a[0] - alpha[0]).powi(2) + (a[1] - alpha[1]).powi(2) + (a[2] - alpha[2]).powi(2)).sqrt();
        alpha = a;
        cur = e;
        path.push(NewtonStep { alpha, loss: cur.loss, grad_norm: norm(&cur.grad.unwrap()) });
        hess = obj.hessian(alpha, h);
        if step < 1e-3 || norm(&cur.grad.unwrap()) < 1e-3 * norm(&grad0).max(1e-12) {
            break;
        }
    }
    let alpha_star = alpha;
    let e_star = cur;
    let hess_star = hess;
    let (eigval_star, eigvec_star) = eig3(hess_star);
    let grad_star = e_star.grad.unwrap();
    let hit_range_bound = newton_hit_bound || e_star.clamped_frac > spec.max_clamped;
    if hit_range_bound {
        println!(
            "  [{}] {} landscape: hit_range_bound (clamped_frac_star {:.3}, max_clamped {:.3})",
            arm.name, g.staid, e_star.clamped_frac, spec.max_clamped
        );
    }

    // 3. behavioural half-widths and trained-point coordinates in the eigenbasis of H(α*)
    let mut coord_trained = [0.0f32; 3];
    for k in 0..3 {
        coord_trained[k] = (0..3).map(|r| eigvec_star[r][k] * (0.0 - alpha_star[r])).sum();
    }
    let half_widths: Vec<[f32; 3]> = spec
        .tolerances
        .iter()
        .map(|tol| {
            let eps_l = tol * e_star.loss.max(1e-12);
            let mut w = [0.0f32; 3];
            for k in 0..3 {
                w[k] = if eigval_star[k] > 0.0 { (2.0 * eps_l / eigval_star[k]).sqrt() } else { f32::INFINITY };
            }
            w
        })
        .collect();

    // 4. analytic celerity direction: d(mean path travel time)/dα at α = 0 (hydraulic K), for pass criterion (iii)
    let celerity_dir = celerity_direction(&obj, h);

    // 5. landscape slices through α*, named in `spec.planes` (default all
    // four: (n,p), (n,q), (p,q), and (v1, v3)). `grid: 0` skips every plane
    // (used for cheap studies over many arms or checkpoints where only the
    // Newton optimum is needed). Plane names are validated against
    // `VALID_PLANES` in `run_landscape`, so an unrecognized name here would
    // already have errored before any gauge ran.
    let mut slices = Vec::new();
    if spec.grid > 0 {
        let gsz = spec.grid.max(3);
        let axis: Vec<f32> = (0..gsz).map(|i| -spec.alpha_max + 2.0 * spec.alpha_max * i as f32 / (gsz - 1) as f32).collect();
        let axis_planes: [(&str, [usize; 2]); 3] = [("n-p", [0, 1]), ("n-q", [0, 2]), ("p-q", [1, 2])];
        for name in &spec.planes {
            if let Some(&(_, [i, j])) = axis_planes.iter().find(|(n, _)| *n == name) {
                let mut loss = vec![f32::NAN; gsz * gsz];
                let mut nse = vec![f32::NAN; gsz * gsz];
                let mut clamped = vec![f32::NAN; gsz * gsz];
                for (a_i, &va) in axis.iter().enumerate() {
                    for (b_i, &vb) in axis.iter().enumerate() {
                        let mut a = alpha_star;
                        a[i] = va;
                        a[j] = vb;
                        let e = obj.eval(a, false);
                        loss[a_i * gsz + b_i] = e.loss;
                        nse[a_i * gsz + b_i] = e.nse_mean;
                        clamped[a_i * gsz + b_i] = e.clamped_frac;
                    }
                }
                slices.push(Slice { name: name.clone(), axis_a: axis.clone(), axis_b: axis.clone(), basis_a: unit(i), basis_b: unit(j), loss, nse, clamped });
            } else if name == "stiff-sloppy" {
                // eigen-plane: alpha = alpha* + s*v1 + t*v3, s,t in [-alpha_max, alpha_max]
                let v1 = [eigvec_star[0][0], eigvec_star[1][0], eigvec_star[2][0]];
                let v3 = [eigvec_star[0][2], eigvec_star[1][2], eigvec_star[2][2]];
                let mut loss = vec![f32::NAN; gsz * gsz];
                let mut nse = vec![f32::NAN; gsz * gsz];
                let mut clamped = vec![f32::NAN; gsz * gsz];
                for (a_i, &s) in axis.iter().enumerate() {
                    for (b_i, &t) in axis.iter().enumerate() {
                        let mut a = alpha_star;
                        for k in 0..3 {
                            a[k] = (alpha_star[k] + s * v1[k] + t * v3[k]).clamp(-2.0 * spec.alpha_max, 2.0 * spec.alpha_max);
                        }
                        let e = obj.eval(a, false);
                        loss[a_i * gsz + b_i] = e.loss;
                        nse[a_i * gsz + b_i] = e.nse_mean;
                        clamped[a_i * gsz + b_i] = e.clamped_frac;
                    }
                }
                slices.push(Slice { name: "stiff-sloppy".into(), axis_a: axis.clone(), axis_b: axis.clone(), basis_a: v1, basis_b: v3, loss, nse, clamped });
            }
        }
    }

    // 6. per-reach gradients ("perturbations in a watershed"): where the
    // gauge still constrains parameters, even at alpha* where the
    // basin-uniform gradient is zero. Logs (not asserts) the chain-rule
    // consistency check that would otherwise only be a `#[ignore]`d unit
    // test: the sum over reaches of reach_grad must equal the
    // basin-uniform grad, since both differentiate the same
    // `x_i = x0_i * exp(leaf_i)` multiply.
    let (reach_grad0, reach_grad_star, dist_to_gauge_m) = if spec.reach_grad {
        let rg0 = obj.reach_grad([0.0; 3]);
        let rgs = obj.reach_grad(alpha_star);
        let sum = |v: &[f32]| v.iter().sum::<f32>();
        for (label, rg, uniform) in [("alpha=0", &rg0, &grad0), ("alpha*", &rgs, &grad_star)] {
            for (k, name) in ["n", "p", "q"].into_iter().enumerate() {
                let per_reach = [&rg.n, &rg.p, &rg.q][k];
                let s = sum(per_reach);
                let u = uniform[k];
                let rel = (s - u).abs() / u.abs().max(1e-8);
                println!(
                    "  [{}] {} reach_grad consistency ({label}, {name}): sum {s:.6e} vs uniform {u:.6e} (rel diff {rel:.2e})",
                    arm.name, g.staid
                );
            }
        }
        let dist = dist_to_gauge(&obj.windows[0].tensors.adjacency, obj.windows[0].gauge_row);
        (Some([rg0.n, rg0.p, rg0.q]), Some([rgs.n, rgs.p, rgs.q]), dist)
    } else {
        (None, None, Vec::new())
    };

    let r = LandscapeResult {
        staid: g.staid.clone(),
        arm: arm.name.clone(),
        run_id: arm.run_id.clone(),
        checkpoint: arm.checkpoint_dir.display().to_string(),
        n_reach: obj.n_reach(),
        window_days: spec.window_days,
        window_starts: starts.to_vec(),
        sigma: obj.sigma,
        loss0: e0.loss,
        nse0: e0.nse_mean,
        grad0,
        hess0,
        loss_star: e_star.loss,
        nse_star: e_star.nse_mean,
        nse_star_windows: e_star.nse.clone(),
        alpha_star,
        grad_star,
        hess_star,
        eigval_star,
        eigvec_star,
        coord_trained,
        tolerances: spec.tolerances.clone(),
        half_widths,
        celerity_dir,
        newton_path: path,
        slices,
        clamped_frac_star: e_star.clamped_frac,
        hit_range_bound,
        n0: obj.windows[0].n0.clone().into_data().to_vec::<f32>().unwrap(),
        p0: obj.windows[0].p0.clone().into_data().to_vec::<f32>().unwrap(),
        q0: obj.windows[0].q0.clone().into_data().to_vec::<f32>().unwrap(),
        comid: obj.windows[0].comids.clone(),
        reach_grad0,
        reach_grad_star,
        dist_to_gauge_m,
    };
    write_landscape_netcdf(&arm_dir.join("gauges").join(format!("{}.nc", g.staid)), &r)?;
    output::append_summary(&arm_dir.join("summary.csv"), &r)?;
    Ok(r)
}

fn norm(v: &[f32; 3]) -> f32 {
    (v[0] * v[0] + v[1] * v[1] + v[2] * v[2]).sqrt()
}
fn unit(i: usize) -> [f32; 3] {
    let mut u = [0.0f32; 3];
    u[i] = 1.0;
    u
}

/// Unit vector in α-space along which the mean hydraulic path travel time to the
/// gauge (Σ K = Σ L/c at the window-mean discharge of the first window) increases
/// fastest at α = 0. The stiff eigenvector should be close to ± this direction.
fn celerity_direction<I: Backend + 'static>(obj: &Objective<I>, h: f32) -> [f32; 3]
where
    I::FloatTensorPrimitive: 'static,
    I::Device: 'static,
{
    use crate::experiment::adjoint::hydraulics::{path_travel_time_hours, reach_k_hours};
    let w = &obj.windows[0];
    let n = obj.n_reach();
    // Window-mean routed discharge at α = 0: approximate by mean inflow accumulated? Use q' mean per reach
    // routed once at α=0 would need runoff; use the engine forward at α = 0.
    let e = obj.eval([0.0; 3], false);
    let _ = e;
    // Fall back to the window-mean inflow as the discharge proxy for K (cheap; direction only).
    let t = w.q_prime.dims()[0];
    let qp: Vec<f32> = w.q_prime.clone().into_data().to_vec::<f32>().unwrap();
    let mut qmean = vec![0.0f32; n];
    for hh in 0..t {
        for i in 0..n {
            qmean[i] += qp[hh * n + i];
        }
    }
    qmean.iter_mut().for_each(|v| *v /= t as f32);
    // accumulate downstream so Q at a reach includes upstream inflow (rough steady-state)
    let adj = &w.tensors.adjacency;
    let mut down = vec![-1i32; n];
    for (r, c) in adj.rows.iter().zip(&adj.cols) {
        if r != c {
            down[*c as usize] = *r;
        }
    }
    let mut q_acc = qmean.clone();
    for i in 0..n {
        // topological order: upstream first
        if down[i] >= 0 {
            q_acc[down[i] as usize] += q_acc[i];
        }
    }
    let q_t = burn::tensor::Tensor::<I, 1>::from_floats(q_acc.as_slice(), &obj.ctx.device);
    let mean_tt = |alpha: [f32; 3]| -> f32 {
        let (nn, pp, qq) = obj.fields_at(w, alpha);
        let k = reach_k_hours::<I>(&obj.ctx.cfg, &nn, &pp, &qq, q_t.clone(), &w.slope, &w.length);
        let tt = path_travel_time_hours(adj, w.gauge_row, &k);
        let f: Vec<f32> = tt.into_iter().filter(|v| v.is_finite()).collect();
        f.iter().sum::<f32>() / f.len().max(1) as f32
    };
    let mut gvec = [0.0f32; 3];
    for k in 0..3 {
        let mut ap = [0.0f32; 3];
        ap[k] = h;
        let mut am = [0.0f32; 3];
        am[k] = -h;
        gvec[k] = (mean_tt(ap) - mean_tt(am)) / (2.0 * h);
    }
    let nrm = norm(&gvec).max(1e-12);
    [gvec[0] / nrm, gvec[1] / nrm, gvec[2] / nrm]
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn default_planes_are_the_four_valid_names() {
        let spec: LandscapeSpec = serde_yaml::from_str("gauges: {}\n").unwrap();
        assert_eq!(spec.planes, vec!["n-p", "n-q", "p-q", "stiff-sloppy"]);
        assert!(spec.validate_planes().is_ok());
    }

    #[test]
    fn unknown_plane_name_errors_with_valid_list() {
        let spec: LandscapeSpec =
            serde_yaml::from_str("gauges: {}\nplanes: [\"n-p\", \"bogus\"]\n").unwrap();
        let err = spec.validate_planes().unwrap_err();
        assert!(err.to_string().contains("bogus"));
        assert!(err.to_string().contains("n-p"));
        assert!(err.to_string().contains("stiff-sloppy"));
    }
}

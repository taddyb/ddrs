//! Per-gauge loss landscape in channel-parameter (log-multiplier) space.
//! Spec: docs/superpowers/specs/2026-09-07-adjoint-landscape-design.md

pub mod objective;
pub mod output;

use std::path::Path;
use std::time::Instant;

use burn::tensor::backend::Backend;
use chrono::Duration;
use serde::{Deserialize, Serialize};

use crate::data::ids::Staid;
use crate::experiment::adjoint::gauges::{all_gauges_selection, gauge_list_from_pairs, gauge_list_from_staids, nested_reference_selection, read_gages_ii_class, write_gauges_csv, GaugeEntry};
use crate::experiment::adjoint::influence::{dist_to_gauge, InfluenceContext, PERIOD_TESTING, PERIOD_TRAINING};
use crate::experiment::adjoint::{resolve_window_days, seasonal_window_starts, GaugeSource, GaugeSpec};
use crate::experiment::{shard_gauges, BoxError, ExperimentManifest, ResolvedArm, Shard};

use self::objective::{eig3, eig_active, newton_step_capped, Objective};
use self::output::{write_landscape_netcdf, LandscapeResult, NewtonStep, SeriesData, Slice};

#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct LandscapeSpec {
    pub gauges: GaugeSpec,
    /// Length of each evaluation window, in days. `0` means "from the
    /// window start to the end of the eval axis" (`resolve_window_days`),
    /// resolved once per arm from the first window's start. Only valid with
    /// `n_windows: 1` — with more than one seasonal window a single shared
    /// length can't span every window to the axis end (later starts leave
    /// less axis remaining), so `run_landscape` rejects the combination via
    /// `validate_window_days`.
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
    /// Max length (log units, active components only) of the damped Newton
    /// step before backtracking. Small basins have a nearly flat or
    /// indefinite Hessian, so the raw Newton step can be hundreds of log
    /// units long; every backtracking trial then lands on the same clamped
    /// box corner and the search reports `hit_range_bound` with zero
    /// iterations even though the gradient is not small. See
    /// `objective::newton_direction`.
    #[serde(default = "d_newton_step_cap")]
    pub newton_step_cap: f32,
    /// Loss tolerances (fractions of L(α*)) for behavioural half-widths.
    #[serde(default = "d_tols")]
    pub tolerances: Vec<f32>,
    /// Max acceptable `clamped_frac` for a Newton trial point (else treated
    /// like a non-decrease: halve the step and retry).
    #[serde(default = "d_max_clamped")]
    pub max_clamped: f32,
    /// Minimum number of clamped (reach, parameter) entries always
    /// tolerated, regardless of `max_clamped`. For small basins (3-5
    /// reaches), a single clamped entry can already exceed `max_clamped`
    /// as a fraction, starving the damped Newton search of any acceptable
    /// step. `run_gauge` computes the per-gauge effective bound as
    /// `max(max_clamped, max_clamped_min_reaches / (n_reach *
    /// n_active_params))`. See `effective_max_clamped`.
    #[serde(default = "d_max_clamped_min_reaches")]
    pub max_clamped_min_reaches: usize,
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
    /// Where the slice planes are centred. "optimum" (default): axis planes
    /// hold the third component at alpha_star and sweep absolute alpha in
    /// ±alpha_max; the stiff-sloppy plane is centred at alpha_star. "trained":
    /// axis planes hold the third component at 0 (the trained value) instead
    /// of alpha_star; the stiff-sloppy plane is centred at alpha = 0 instead
    /// of alpha_star (still spanned by the eigenvectors of H(alpha_star)).
    /// Checked against `VALID_SLICE_CENTERS` at the top of `run_landscape`.
    #[serde(default = "d_slice_center")]
    pub slice_center: String,
    /// Which per-window loss the Newton search optimizes: "nse-batch"
    /// (default; the training objective restricted to this gauge) or "kge"
    /// (`1 - KGE` over the window's valid days). `nse` and `kge`
    /// diagnostics are always both computed regardless of which one drives
    /// the search. Checked against `VALID_OBJECTIVES` at the top of
    /// `run_landscape`.
    #[serde(default = "d_objective")]
    pub objective: String,
    /// When true, write each gauge's window-0 daily series (observed,
    /// routed at alpha = 0 and alpha*, and the no-routing summed-q'
    /// baseline) to the netCDF on a new `day` dimension. Off by default: it
    /// triples the per-gauge netCDF's day-dimensioned payload and needs two
    /// extra forward passes (`Objective::daily_series`).
    #[serde(default)]
    pub series: bool,
    /// Which of the config's two eval windows to score on: "testing"
    /// (default — the config's `testing:` block, 1995-10-01..2010-09-30 for
    /// the p21 arm) or "training" (the un-overlaid `experiment:` block).
    /// "training" lets a study ask whether the aggregate gradient has
    /// vanished on the data the model was actually trained on, rather than
    /// on held-out data. Checked against `VALID_PERIODS` at the top of
    /// `run_landscape`. Threaded to `InfluenceContext::open` via
    /// `LandscapeOptions::period`.
    #[serde(default = "d_period")]
    pub period: String,
}
fn d_window_days() -> usize { 365 }
fn d_water_year() -> i32 { 2000 }
fn d_n_windows() -> usize { 1 }
fn d_alpha_max() -> f32 { 1.0986123 }
fn d_grid() -> usize { 11 }
fn d_h() -> f32 { 0.05 }
fn d_newton_iters() -> usize { 10 }
fn d_newton_step_cap() -> f32 { 1.0 }
fn d_tols() -> Vec<f32> { vec![0.05, 0.10] }
fn d_max_clamped() -> f32 { 0.05 }
fn d_max_clamped_min_reaches() -> usize { 2 }

/// Effective `max_clamped` bound for a gauge: the configured `spec_value`,
/// widened so at least `min_reaches` clamped (reach, parameter) entries are
/// always tolerated out of `n_reach * n_active` total entries. Small basins
/// (few reaches) would otherwise reject every Newton trial point on a
/// single clamped entry.
fn effective_max_clamped(spec_value: f32, min_reaches: usize, n_reach: usize, n_active: usize) -> f32 {
    let total = (n_reach * n_active).max(1) as f32;
    spec_value.max(min_reaches as f32 / total)
}
fn d_planes() -> Vec<String> {
    VALID_PLANES.iter().map(|s| s.to_string()).collect()
}
fn d_slice_center() -> String {
    "optimum".to_string()
}
fn d_objective() -> String {
    "nse-batch".to_string()
}
fn d_period() -> String {
    PERIOD_TESTING.to_string()
}

/// The four slices `run_gauge` knows how to compute.
pub const VALID_PLANES: [&str; 4] = ["n-p", "n-q", "p-q", "stiff-sloppy"];
/// The two slice-centering conventions accepted by `slice_center`.
pub const VALID_SLICE_CENTERS: [&str; 2] = ["optimum", "trained"];
/// The two per-window losses accepted by `LandscapeSpec::objective`.
pub const VALID_OBJECTIVES: [&str; 2] = ["nse-batch", "kge"];
/// The two eval windows accepted by `LandscapeSpec::period`.
pub const VALID_PERIODS: [&str; 2] = [PERIOD_TESTING, PERIOD_TRAINING];

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

    /// Reject a `slice_center` that isn't one of `VALID_SLICE_CENTERS`.
    pub fn validate_slice_center(&self) -> Result<(), BoxError> {
        if !VALID_SLICE_CENTERS.contains(&self.slice_center.as_str()) {
            return Err(format!(
                "unknown landscape slice_center `{}`; valid values are {VALID_SLICE_CENTERS:?}",
                self.slice_center
            )
            .into());
        }
        Ok(())
    }

    /// Reject `window_days: 0` ("span to the eval axis end") unless
    /// `n_windows == 1`. A single resolved length applied to more than one
    /// seasonal window would give each window a different actual duration
    /// (later starts leave less axis remaining) while `Objective::build`
    /// assumes all windows share the same length.
    pub fn validate_window_days(&self) -> Result<(), BoxError> {
        if self.window_days == 0 && self.n_windows != 1 {
            return Err(format!(
                "landscape.window_days: 0 (span to the eval axis end) requires n_windows: 1, got {}",
                self.n_windows
            )
            .into());
        }
        Ok(())
    }

    /// Reject an `objective` that isn't one of `VALID_OBJECTIVES`.
    pub fn validate_objective(&self) -> Result<(), BoxError> {
        if !VALID_OBJECTIVES.contains(&self.objective.as_str()) {
            return Err(format!(
                "unknown landscape objective `{}`; valid objectives are {VALID_OBJECTIVES:?}",
                self.objective
            )
            .into());
        }
        Ok(())
    }

    /// Reject a `period` that isn't one of `VALID_PERIODS`.
    pub fn validate_period(&self) -> Result<(), BoxError> {
        if !VALID_PERIODS.contains(&self.period.as_str()) {
            return Err(format!(
                "unknown landscape period `{}`; valid periods are {VALID_PERIODS:?}",
                self.period
            )
            .into());
        }
        Ok(())
    }
}

pub struct LandscapeOptions {
    pub max_gauges: Option<usize>,
    pub force_cpu: bool,
    pub jobs: usize,
    pub dry_run: bool,
    /// Keep only every K-th gauge (sorted by staid), for parallel sharding
    /// across processes. `None` runs the whole selected population.
    pub shard: Option<Shard>,
    /// `LandscapeSpec::period`, threaded from the bundle spec to every
    /// `InfluenceContext::open` call this study makes.
    pub period: String,
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
    spec.validate_slice_center()?;
    spec.validate_window_days()?;
    spec.validate_objective()?;
    spec.validate_period()?;
    if arms.is_empty() {
        return Err("no arms selected".into());
    }
    let mut gauges: Vec<GaugeEntry> = match spec.gauges.source {
        GaugeSource::Explicit => gauge_list_from_pairs(&spec.gauges.pairs),
        GaugeSource::NestedReference => {
            let dbf = spec.gauges.gages_ii_dbf.as_ref().ok_or("nested-reference requires gages_ii_dbf")?;
            let class = read_gages_ii_class(dbf)?;
            let ctx = InfluenceContext::<I>::open(&arms[0], device, opts.force_cpu, &opts.period)?;
            nested_reference_selection(&ctx.dataset, &class, spec.gauges.max_downstream)?
        }
        GaugeSource::All => {
            let ctx = InfluenceContext::<I>::open(&arms[0], device, opts.force_cpu, &opts.period)?;
            let list = all_gauges_selection(&ctx.dataset);
            println!("gauge source `all`: {} gauges the dataset can evaluate (subgraph + observations present)", list.len());
            list
        }
        GaugeSource::List => gauge_list_from_staids(&spec.gauges.staids),
    };
    let population_n = gauges.len();
    gauges = shard_gauges(gauges, opts.shard);
    if let Some(s) = opts.shard {
        println!("shard {s}: kept {} of {population_n} gauges (sorted by staid, index % {} == {})", gauges.len(), s.count, s.index);
    }
    if let Some(k) = opts.max_gauges {
        gauges.truncate(k);
    }
    write_gauges_csv(&out_dir.join("gauges.csv"), &gauges)?;
    println!(
        "landscape: {} arm(s), {} gauge(s) (of {population_n} population), {} window(s) × {} d, grid {}², α ∈ ±{:.3}, fd_step {}",
        arms.len(), gauges.len(), spec.n_windows, spec.window_days, spec.grid, spec.alpha_max, spec.fd_step
    );
    // Dry runs still enter `run_arm` (one per arm): it opens the arm's
    // context, resolves and logs the actual window length, then stops
    // before the per-gauge loop. That's needed so `window_days: 0`'s
    // resolved length is visible in a dry-run log, not just a real run's.
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
    let ctx = InfluenceContext::<I>::open(arm, device, opts.force_cpu, &opts.period)?;
    let arm_dir = out_dir.join(&arm.name);
    std::fs::create_dir_all(arm_dir.join("gauges"))?;
    let n_windows = spec.n_windows.clamp(1, 4);
    // Only the first `n_windows` seasonal offsets are actually used below
    // (`starts`), so only those need the fixed-length overflow check — see
    // `seasonal_window_starts`'s `n_check` doc.
    let seasonal = seasonal_window_starts(&ctx.axis, spec.water_year, spec.window_days, n_windows)?;
    let starts: Vec<usize> = seasonal.iter().take(n_windows).map(|(s, _)| *s).collect();
    // Resolve `window_days: 0` from the (only, per `validate_window_days`)
    // window's start.
    let window_days = resolve_window_days(spec.window_days, ctx.axis.num_days, starts[0])?;
    // `seasonal[0].1` is the first window's calendar start date on this
    // arm's axis (Training or Testing, per `opts.period`); the last day of
    // the window is inclusive, so `window_days - 1` days after that.
    let window_start_date = seasonal[0].1;
    let window_end_date = window_start_date + Duration::days(window_days.saturating_sub(1) as i64);
    let window_note = format!(
        "[{}] period {}: resolved window_days {window_days}{} — {window_start_date} .. {window_end_date}",
        arm.name,
        opts.period,
        if spec.window_days == 0 { format!(" (spec 0 -> full axis from day {})", starts[0]) } else { String::new() }
    );
    println!("  {window_note}");
    let mut notes = vec![window_note];
    if opts.dry_run {
        return Ok(notes);
    }
    for (gi, g) in gauges.iter().enumerate() {
        let t_g = Instant::now();
        match run_gauge::<I>(&ctx, spec, arm, g, &arm_dir, &starts, window_days) {
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
    window_days: usize,
) -> Result<LandscapeResult, BoxError>
where
    I::FloatTensorPrimitive: 'static,
    I::Device: 'static,
{
    let staid = Staid::new(&g.staid);
    let obj = Objective::<I>::build(ctx, &staid, starts, window_days, &spec.objective)?;
    let h = spec.fd_step;
    let active = obj.active();
    let n_active_params = active.iter().filter(|&&a| a).count();
    let max_clamped = effective_max_clamped(spec.max_clamped, spec.max_clamped_min_reaches, obj.n_reach(), n_active_params);
    if max_clamped != spec.max_clamped {
        println!(
            "  [{}] {} landscape: effective max_clamped {:.4} (base {:.4}, n_reach {}, n_active {})",
            arm.name, g.staid, max_clamped, spec.max_clamped, obj.n_reach(), n_active_params
        );
    }

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
    let mut used_gradient_fallback = false;
    for it in 0..spec.newton_iters {
        let gcur = cur.grad.unwrap();
        let (vals, _) = eig3(hess);
        let mu = (-vals[2]).max(0.0) + 1e-3 * vals[0].abs().max(1e-6);
        let (d, fallback) = newton_step_capped(hess, gcur, mu, active, spec.newton_step_cap);
        if fallback {
            println!("  [{}] {} newton direction not descent; using gradient step", arm.name, g.staid);
            used_gradient_fallback = true;
        }
        for k in 0..3 {
            if !active[k] {
                assert_eq!(d[k], 0.0, "Newton step must not move fixed alpha component {k}");
            }
        }
        // backtracking line search within the domain; a trial whose
        // clamped_frac exceeds max_clamped is unacceptable exactly like a
        // non-decrease: halve the step and retry.
        let mut t = 1.0f32;
        let mut accepted = None;
        let mut saw_within_bound = false;
        for _ in 0..16 {
            let mut a = alpha;
            for k in 0..3 {
                a[k] = (alpha[k] + t * d[k]).clamp(-spec.alpha_max, spec.alpha_max);
            }
            let e = obj.eval(a, true);
            if e.clamped_frac > max_clamped {
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
    let (eigval_star, eigvec_star) = eig_active(hess_star, active);
    let grad_star = e_star.grad.unwrap();
    let hit_range_bound = newton_hit_bound || e_star.clamped_frac > max_clamped;
    if hit_range_bound {
        println!(
            "  [{}] {} landscape: hit_range_bound (clamped_frac_star {:.3}, max_clamped {:.3})",
            arm.name, g.staid, e_star.clamped_frac, max_clamped
        );
    }

    // 3. behavioural half-widths and trained-point coordinates in the
    // eigenbasis of H(α*), over the active sub-block only. `eig_active`
    // marks the dropped (fixed-parameter) eigen-slot with `eigval_star[k] ==
    // NaN`; both quantities are NaN there too rather than a computed 0 or
    // INFINITY, since that slot is not a real eigen-direction.
    let mut coord_trained = [0.0f32; 3];
    for k in 0..3 {
        coord_trained[k] = if eigval_star[k].is_nan() {
            f32::NAN
        } else {
            (0..3).map(|r| eigvec_star[r][k] * (0.0 - alpha_star[r])).sum()
        };
    }
    let half_widths: Vec<[f32; 3]> = spec
        .tolerances
        .iter()
        .map(|tol| {
            let eps_l = tol * e_star.loss.max(1e-12);
            let mut w = [0.0f32; 3];
            for k in 0..3 {
                w[k] = if eigval_star[k].is_nan() {
                    f32::NAN
                } else if eigval_star[k] > 0.0 {
                    (2.0 * eps_l / eigval_star[k]).sqrt()
                } else {
                    f32::INFINITY
                };
            }
            w
        })
        .collect();

    // 4. analytic celerity direction: d(mean path travel time)/dα at α = 0 (hydraulic K), for pass criterion (iii)
    let celerity_dir = celerity_direction(&obj, h, active);

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
        // "optimum" (default): axis planes pin the third component at
        // alpha_star, the eigen plane is centred at alpha_star. "trained":
        // axis planes pin the third component at 0, the eigen plane is
        // centred at alpha = 0. Validated in `run_landscape`.
        let trained_center = spec.slice_center == "trained";
        let axis_third = if trained_center { [0.0f32; 3] } else { alpha_star };
        let eigen_center = if trained_center { [0.0f32; 3] } else { alpha_star };
        for name in &spec.planes {
            if let Some(fixed) = plane_skip_reason(name, active) {
                println!("  [{}] {} plane {name} skipped: {fixed} fixed", arm.name, g.staid);
                continue;
            }
            if let Some(&(_, [i, j])) = axis_planes.iter().find(|(n, _)| *n == name) {
                let mut loss = vec![f32::NAN; gsz * gsz];
                let mut nse = vec![f32::NAN; gsz * gsz];
                let mut clamped = vec![f32::NAN; gsz * gsz];
                for (a_i, &va) in axis.iter().enumerate() {
                    for (b_i, &vb) in axis.iter().enumerate() {
                        let mut a = axis_third;
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
                // eigen-plane: alpha = center + s*v1 + t*v3, s,t in [-alpha_max, alpha_max].
                // v1/v3 are the first and last ACTIVE eigenvectors (the
                // dropped fixed-parameter slot from `eig_active` is never
                // used as a plane axis).
                let active_cols: Vec<usize> = (0..3).filter(|&k| !eigval_star[k].is_nan()).collect();
                if active_cols.len() < 2 {
                    println!("  [{}] {} plane stiff-sloppy skipped: fewer than 2 active parameters", arm.name, g.staid);
                    continue;
                }
                let v1_col = active_cols[0];
                let v3_col = *active_cols.last().unwrap();
                let v1 = [eigvec_star[0][v1_col], eigvec_star[1][v1_col], eigvec_star[2][v1_col]];
                let v3 = [eigvec_star[0][v3_col], eigvec_star[1][v3_col], eigvec_star[2][v3_col]];
                let mut loss = vec![f32::NAN; gsz * gsz];
                let mut nse = vec![f32::NAN; gsz * gsz];
                let mut clamped = vec![f32::NAN; gsz * gsz];
                for (a_i, &s) in axis.iter().enumerate() {
                    for (b_i, &t) in axis.iter().enumerate() {
                        let mut a = [0.0f32; 3];
                        for k in 0..3 {
                            a[k] = (eigen_center[k] + s * v1[k] + t * v3[k]).clamp(-2.0 * spec.alpha_max, 2.0 * spec.alpha_max);
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

    // Per-reach slope/length (window 0; the network is the same reach order
    // across windows) and the mean observed daily discharge over all valid
    // days (finite, >= 0, day >= warmup) across all windows, for the
    // depth-at-gauge landscape axis (surface.py --depth-axis).
    let slope: Vec<f32> = obj.windows[0].slope.clone().into_data().to_vec::<f32>().unwrap();
    let length: Vec<f32> = obj.windows[0].length.clone().into_data().to_vec::<f32>().unwrap();
    let gauge_reach_row = obj.windows[0].gauge_row;
    let mut obs_sum = 0.0f64;
    let mut obs_n_valid_days = 0usize;
    for w in &obj.windows {
        for i in ctx.warmup..w.obs.len() {
            let v = w.obs[i];
            if v.is_finite() && v >= 0.0 {
                obs_sum += v as f64;
                obs_n_valid_days += 1;
            }
        }
    }
    let obs_mean_q_m3s = if obs_n_valid_days > 0 { (obs_sum / obs_n_valid_days as f64) as f32 } else { f32::NAN };

    // 7. Optional daily series (window 0 only -- `day` is a single window's
    // timeline, so this doesn't generalize past `n_windows: 1`): observed,
    // routed at alpha = 0 and alpha*, and the no-routing baseline (summed
    // q_prime over the gauge's subgraph reaches). `daily_series` re-runs the
    // forward at each alpha (not cached from steps 1-2 above) since `eval`
    // doesn't carry the daily tensor out. The no-routing baseline sums
    // `WindowData.tensors.q_prime_daily` -- the already-daily-resolution
    // inflow tensor (not the inner hourly `q_prime`, which would need
    // pooling) -- over reaches (dim 1), truncated to the routed series'
    // length (`q_prime_daily` is one day longer: the tau-trim in
    // `InfluenceContext::daily` drops the last day, see
    // `training::loss::tau_trim_and_downsample`).
    let series = if spec.series {
        let routed_daily_trained = obj.daily_series([0.0; 3]);
        let routed_daily_star = obj.daily_series(alpha_star);
        let d = routed_daily_trained.len();
        let obs_daily: Vec<f32> = obj.windows[0].obs.iter().take(d).copied().collect();
        let mut summed_qprime_daily: Vec<f32> = obj.windows[0]
            .tensors
            .q_prime_daily
            .clone()
            .inner()
            .sum_dim(1)
            .into_data()
            .to_vec::<f32>()
            .unwrap();
        summed_qprime_daily.truncate(d);
        Some(SeriesData { obs_daily, routed_daily_trained, routed_daily_star, summed_qprime_daily })
    } else {
        None
    };

    let r = LandscapeResult {
        period: spec.period.clone(),
        axis_start_date: ctx.axis.start.format("%Y-%m-%d").to_string(),
        staid: g.staid.clone(),
        arm: arm.name.clone(),
        run_id: arm.run_id.clone(),
        checkpoint: arm.checkpoint_dir.display().to_string(),
        n_reach: obj.n_reach(),
        window_days,
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
        slice_center: spec.slice_center.clone(),
        clamped_frac_star: e_star.clamped_frac,
        max_clamped_effective: max_clamped,
        hit_range_bound,
        used_gradient_fallback,
        n0: obj.windows[0].n0.clone().into_data().to_vec::<f32>().unwrap(),
        p0: obj.windows[0].p0.clone().into_data().to_vec::<f32>().unwrap(),
        q0: obj.windows[0].q0.clone().into_data().to_vec::<f32>().unwrap(),
        comid: obj.windows[0].comids.clone(),
        reach_grad0,
        reach_grad_star,
        dist_to_gauge_m,
        active,
        slope,
        length,
        gauge_reach_row,
        obs_mean_q_m3s,
        obs_n_valid_days,
        objective: spec.objective.clone(),
        kge0: e0.kge_mean,
        kge_star: e_star.kge_mean,
        kge_star_windows: e_star.kge.clone(),
        series,
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

const PARAM_NAMES: [&str; 3] = ["n", "p_spatial", "q_spatial"];

/// Which two alpha indices an axis plane's grid spans; `None` for
/// `"stiff-sloppy"` (it is never skipped for naming a fixed parameter --
/// it re-slices onto the active eigenvectors instead).
fn axis_plane_indices(name: &str) -> Option<[usize; 2]> {
    match name {
        "n-p" => Some([0, 1]),
        "n-q" => Some([0, 2]),
        "p-q" => Some([1, 2]),
        _ => None,
    }
}

/// Name of the fixed parameter that makes axis plane `name` unavailable, or
/// `None` when both its axes are active (or `name` isn't an axis plane at
/// all, e.g. `"stiff-sloppy"`). Pure, so it's unit-tested without a trained
/// run.
fn plane_skip_reason(name: &str, active: [bool; 3]) -> Option<&'static str> {
    let [i, j] = axis_plane_indices(name)?;
    if !active[i] {
        Some(PARAM_NAMES[i])
    } else if !active[j] {
        Some(PARAM_NAMES[j])
    } else {
        None
    }
}

/// Unit vector in α-space along which the mean hydraulic path travel time to the
/// gauge (Σ K = Σ L/c at the window-mean discharge of the first window) increases
/// fastest at α = 0. The stiff eigenvector should be close to ± this direction.
/// Computed and normalized over `active` components only; a fixed
/// parameter's slot is `NaN`, not 0: it is not part of the direction.
fn celerity_direction<I: Backend + 'static>(obj: &Objective<I>, h: f32, active: [bool; 3]) -> [f32; 3]
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
        if !active[k] {
            continue;
        }
        let mut ap = [0.0f32; 3];
        ap[k] = h;
        let mut am = [0.0f32; 3];
        am[k] = -h;
        gvec[k] = (mean_tt(ap) - mean_tt(am)) / (2.0 * h);
    }
    let nrm = (0..3).filter(|&k| active[k]).map(|k| gvec[k] * gvec[k]).sum::<f32>().sqrt().max(1e-12);
    let mut dir = [f32::NAN; 3];
    for k in 0..3 {
        if active[k] {
            dir[k] = gvec[k] / nrm;
        }
    }
    dir
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

    #[test]
    fn default_slice_center_is_optimum() {
        let spec: LandscapeSpec = serde_yaml::from_str("gauges: {}\n").unwrap();
        assert_eq!(spec.slice_center, "optimum");
        assert!(spec.validate_slice_center().is_ok());
    }

    #[test]
    fn window_days_zero_requires_n_windows_one() {
        let ok: LandscapeSpec = serde_yaml::from_str("gauges: {}\nwindow_days: 0\nn_windows: 1\n").unwrap();
        assert!(ok.validate_window_days().is_ok());
        let bad: LandscapeSpec = serde_yaml::from_str("gauges: {}\nwindow_days: 0\nn_windows: 4\n").unwrap();
        let err = bad.validate_window_days().unwrap_err().to_string();
        assert!(err.contains("window_days: 0"), "{err}");
        assert!(err.contains("n_windows: 1"), "{err}");
    }

    #[test]
    fn nonzero_window_days_never_requires_n_windows_one() {
        let spec: LandscapeSpec = serde_yaml::from_str("gauges: {}\nwindow_days: 90\nn_windows: 4\n").unwrap();
        assert!(spec.validate_window_days().is_ok());
    }

    #[test]
    fn plane_skip_reason_names_the_fixed_parameter() {
        // p_spatial (index 1) fixed.
        let active = [true, false, true];
        assert_eq!(plane_skip_reason("n-p", active), Some("p_spatial"));
        assert_eq!(plane_skip_reason("p-q", active), Some("p_spatial"));
        assert_eq!(plane_skip_reason("n-q", active), None);
        assert_eq!(plane_skip_reason("stiff-sloppy", active), None);
    }

    #[test]
    fn plane_skip_reason_all_active_never_skips() {
        let active = [true, true, true];
        for name in VALID_PLANES {
            assert_eq!(plane_skip_reason(name, active), None);
        }
    }

    #[test]
    fn default_max_clamped_min_reaches_is_two() {
        let spec: LandscapeSpec = serde_yaml::from_str("gauges: {}\n").unwrap();
        assert_eq!(spec.max_clamped_min_reaches, 2);
    }

    #[test]
    fn effective_max_clamped_widens_the_bound_for_small_basins() {
        // 3-reach basin, 2 active params: floor 2 / (3*2) = 0.333... dominates.
        let eff = effective_max_clamped(0.05, 2, 3, 2);
        assert!((eff - 2.0 / 6.0).abs() < 1e-6, "eff = {eff}");
    }

    #[test]
    fn effective_max_clamped_keeps_base_for_large_basins() {
        // 213-reach basin, 2 active params: floor 2 / (213*2) << 0.05.
        let eff = effective_max_clamped(0.05, 2, 213, 2);
        assert!((eff - 0.05).abs() < 1e-6, "eff = {eff}");
    }

    #[test]
    fn default_objective_is_nse_batch() {
        let spec: LandscapeSpec = serde_yaml::from_str("gauges: {}\n").unwrap();
        assert_eq!(spec.objective, "nse-batch");
        assert!(spec.validate_objective().is_ok());
        assert!(!spec.series);
    }

    #[test]
    fn unknown_objective_errors_with_valid_list() {
        let spec: LandscapeSpec = serde_yaml::from_str("gauges: {}\nobjective: bogus\n").unwrap();
        let err = spec.validate_objective().unwrap_err();
        assert!(err.to_string().contains("bogus"));
        assert!(err.to_string().contains("nse-batch"));
        assert!(err.to_string().contains("kge"));
    }

    #[test]
    fn gauge_source_list_parses() {
        let spec: LandscapeSpec =
            serde_yaml::from_str("gauges:\n  source: list\n  staids: [\"01563500\", \"01567000\"]\nobjective: kge\nseries: true\n").unwrap();
        assert_eq!(spec.gauges.source, GaugeSource::List);
        assert_eq!(spec.gauges.staids, vec!["01563500".to_string(), "01567000".to_string()]);
        assert_eq!(spec.objective, "kge");
        assert!(spec.series);
        let gauges = gauge_list_from_staids(&spec.gauges.staids);
        assert_eq!(gauges.len(), 2);
        assert!(gauges.iter().all(|g| g.role == "gauge" && g.upstream.is_empty()));
    }

    #[test]
    fn unknown_slice_center_errors_with_valid_list() {
        let spec: LandscapeSpec =
            serde_yaml::from_str("gauges: {}\nslice_center: bogus\n").unwrap();
        let err = spec.validate_slice_center().unwrap_err();
        assert!(err.to_string().contains("bogus"));
        assert!(err.to_string().contains("optimum"));
        assert!(err.to_string().contains("trained"));
    }

    #[test]
    fn default_period_is_testing() {
        let spec: LandscapeSpec = serde_yaml::from_str("gauges: {}\n").unwrap();
        assert_eq!(spec.period, "testing");
        assert!(spec.validate_period().is_ok());
    }

    #[test]
    fn training_period_parses_and_validates() {
        let spec: LandscapeSpec = serde_yaml::from_str("gauges: {}\nperiod: training\n").unwrap();
        assert_eq!(spec.period, "training");
        assert!(spec.validate_period().is_ok());
    }

    #[test]
    fn unknown_period_errors_with_valid_list() {
        let spec: LandscapeSpec = serde_yaml::from_str("gauges: {}\nperiod: bogus\n").unwrap();
        let err = spec.validate_period().unwrap_err();
        assert!(err.to_string().contains("bogus"));
        assert!(err.to_string().contains("testing"));
        assert!(err.to_string().contains("training"));
    }
}

//! CUDA arm of the fused stage op against its CPU arm.
//!
//! `StageBlock::step_fused` runs on NdArray (`use_cuda: false`) and on the
//! Cuda backend (`use_cuda: true`) with the same inputs. Bars (plan D7):
//! per hour and per field `max|dev - cpu| <= 1e-5 * max|cpu|` for the 3-hour
//! families and the 24-hour MC-inflow case, `1e-4` for the 48-hour Juniata and
//! Hudson runs; the five mass sums within `1e-5 * gross_flux_m3`; `steps`
//! equal. The fields are `q_next`, `stage()` and the ten `last_coefficients()`
//! fields. Each side feeds its own `q_next` forward. `run_pair_xb` returns the
//! first failing field as an `Err` and never widens a bar.
//!
//! The 48-hour tests skip themselves in debug builds (the `cfg!(debug_assertions)`
//! rule of `tests/stage_op_parity.rs`). Run them with
//! `cargo test --release --features cuda --test stage_op_cuda_parity`.
#![cfg(feature = "cuda")]

mod common;

use std::collections::BTreeMap;
use std::sync::Arc;

use burn::backend::Autodiff;
use burn::tensor::backend::{Backend, BackendTypes};
use burn::tensor::Tensor;

use ddrs::config::DownstreamBoundary;
use ddrs::data::store::ChannelGeometryRows;
use ddrs::data::ConusAdjacencyStore;
use ddrs::routing::stage::{StageBlock, StageNetwork, StageParams};
use ddrs::routing::{MuskingumCunge, RoutingInputs};
use ddrs::routing::stage_cuda::forward::{
    assembled_to_host, rhs_dev, stage_assemble_dev, stage_epilogue_dev, stage_readout_dev, readout_to_host, AssembledDev,
};
use ddrs::routing::stage_cuda::{HourDev, StageOpConstsDev};
use ddrs::routing::stage_cuda::forward::{hour_to_host, stage_hour_forward_dev, stage_prologue_dev};
use ddrs::routing::stage_op::{
    stage_assemble_host, stage_epilogue_host, stage_hour_forward, stage_prologue_host, stage_readout_host, Assembled,
    ControlOut, DepthOut, HourHost, NewtonOut, Readout, SectionHost, StageOpConsts,
};
use ddrs::sparse::tree_lu::{tree_lu_factor, tree_lu_solve_host};
use ddrs::sparse::cube_buf::{cube_from_host, cube_to_host, Cube};
use ddrs::sparse::tree_lu_cuda::{tree_factor_dev, tree_solve_dev, TreeFactorsDev};
use ddrs::sparse::host_reads_on_this_thread_for_test;
use ddrs::sparse::{CsrPattern, SparseAdjacency};

use common::{
    linear_chain_sparse, mock_config, mock_spatial_parameters, mock_streamflow, InnerBackend, TestBackend,
    TestDevice,
};

type Cpu = InnerBackend;
type Gpu = burn_cuda::Cuda<f32, i32>;

const DEPTH_LB: f32 = 0.01;

fn tv<B: Backend>(v: &[f32], device: &B::Device) -> Tensor<Autodiff<B>, 1> {
    Tensor::<Autodiff<B>, 1>::from_floats(v, device)
}

fn to_vec1<B: Backend>(t: Tensor<B, 1>) -> Vec<f32> {
    t.into_data().to_vec().unwrap()
}

// copied from tests/stage_op_parity.rs::chain
fn chain(len: &[f32], slope: &[f32]) -> (SparseAdjacency, Arc<CsrPattern>) {
    let n = len.len();
    let mut dense = vec![0.0_f32; n * n];
    for i in 0..n - 1 {
        dense[(i + 1) * n + i] = 1.0;
    }
    let adj = SparseAdjacency::from_dense(n, &dense, len.to_vec(), slope.to_vec());
    let pat = Arc::new(CsrPattern::from_sparse(&adj));
    (adj, pat)
}

// copied from tests/stage_op_parity.rs::table
fn table(rows: &[usize], tw: f32, bed: &[f32], n: f32, pool: &[Option<f32>]) -> ChannelGeometryRows {
    ChannelGeometryRows {
        rows: rows.to_vec(),
        top_width: vec![tw; rows.len()],
        bed_z: bed.to_vec(),
        n: vec![n; rows.len()],
        pool_z: pool.to_vec(),
    }
}

// copied from tests/stage_op_parity.rs::H0
enum H0 {
    /// Depth above `z_mid` of the block without a table.
    Depth(Vec<f32>),
    Abs(Vec<f32>),
}

// copied from tests/stage_op_parity.rs::Case
struct Case {
    label: String,
    adj: SparseAdjacency,
    pattern: Arc<CsrPattern>,
    seeds: Option<Vec<usize>>,
    dams: Vec<(usize, f32)>,
    table: Option<(ChannelGeometryRows, bool)>,
    prescribed: bool,
    series_base: f32,
    series_rate: f32,
    theta_c: f32,
    h0: H0,
    n: Vec<f32>,
    p: Vec<f32>,
    q: Vec<f32>,
    q_t0: Vec<f32>,
    /// Per hour; shorter lists repeat their last entry.
    qp: Vec<Vec<f32>>,
    mc: Vec<Vec<f32>>,
    hours: usize,
}

// copied from tests/stage_op_parity.rs::Case::new and Case::prescribed
impl Case {
    fn new(label: &str, adj: SparseAdjacency, pattern: Arc<CsrPattern>, h0: H0, q_t0: Vec<f32>) -> Self {
        let n = adj.n;
        Case {
            label: label.to_string(),
            adj,
            pattern,
            seeds: None,
            dams: vec![],
            table: None,
            prescribed: false,
            series_base: 0.0,
            series_rate: 0.0,
            theta_c: 0.5,
            h0,
            n: (0..n).map(|i| 0.03 + 0.005 * i as f32).collect(),
            q: (0..n).map(|i| 0.45 + 0.05 * (i % 3) as f32).collect(),
            p: (0..n).map(|i| 18.0 + 2.0 * (i % 4) as f32).collect(),
            q_t0,
            qp: vec![vec![1.0; n]],
            mc: vec![vec![0.0; n]],
            hours: 3,
        }
    }
    fn prescribed(mut self, base: f32, rate: f32) -> Self {
        self.prescribed = true;
        self.series_base = base;
        self.series_rate = rate;
        self
    }
}

// copied from tests/stage_op_parity.rs::params, with use_cuda a parameter
fn params(case: &Case, theta: f32, substeps: u32, use_cuda: bool) -> StageParams {
    StageParams {
        theta,
        theta_control: case.theta_c,
        substeps,
        boundary: if case.prescribed { DownstreamBoundary::PrescribedStage } else { DownstreamBoundary::NormalDepth },
        depth_lb: DEPTH_LB,
        bottom_width_lb: 0.1,
        dt: 3600.0,
        g: 9.81,
        use_cuda,
    }
}

// copied from tests/stage_op_parity.rs::make_block, generic in the inner backend
fn make_block<B: Backend + 'static>(
    case: &Case,
    theta: f32,
    substeps: u32,
    use_cuda: bool,
    device: &B::Device,
) -> StageBlock<B>
where
    B::FloatTensorPrimitive: 'static,
    B::Device: 'static,
{
    make_block_p::<B>(case, theta, substeps, use_cuda, false, device)
}

/// `make_block`; with `perturb` the initial stage moves by one ulp per entry.
fn make_block_p<B: Backend + 'static>(
    case: &Case,
    theta: f32,
    substeps: u32,
    use_cuda: bool,
    perturb: bool,
    device: &B::Device,
) -> StageBlock<B>
where
    B::FloatTensorPrimitive: 'static,
    B::Device: 'static,
{
    let net = StageNetwork::build(&case.adj, &case.pattern, case.seeds.as_deref(), 0.0, &case.dams).unwrap();
    let z0 = net.z_mid.clone();
    let mut block = StageBlock::<B>::new(net, params(case, theta, substeps, use_cuda), device);
    if case.prescribed {
        let series: Vec<f32> = (0..=case.hours).map(|k| case.series_base + case.series_rate * k as f32).collect();
        block.set_boundary_series(series).unwrap();
    }
    if let Some((rows, strict)) = &case.table {
        block.set_table_geometry(rows.clone(), *strict).unwrap();
    }
    let h0 = match &case.h0 {
        H0::Depth(d) => (0..case.adj.n).map(|i| z0[i] + d[i]).collect::<Vec<_>>(),
        H0::Abs(h) => h.clone(),
    };
    let h0 = if perturb { one_ulp(&h0, 1) } else { h0 };
    block.set_stage(tv::<B>(&h0, device));
    block
}

// copied from tests/stage_op_parity.rs::pick
fn pick<T: Clone>(v: &[Vec<T>], k: usize) -> Vec<T> {
    v[k.min(v.len() - 1)].clone()
}

// ------------------------------------------------------------ the pair run

// Forward noise gate (ruling B3), mirroring the B2 gradient gate of
// tests/stage_op_parity.rs. A miss of the strict bar on `q_next` or one of the
// ten `last_coefficients()` fields is accepted only when
// err <= FWD_NOISE_FACTOR * (noise_cpu + noise_cuda) and
// err <= FWD_NOISE_CAP_FRAC * max|cpu|; the accepted entries of a family are
// capped at a fraction of the compared entries.
const FWD_NOISE_FACTOR: f32 = 4.0;
const FWD_NOISE_CAP_FRAC: f32 = 5e-4;
const FWD_NOISE_BUDGET_FRAC: f64 = 0.01;
#[allow(dead_code)]
const FWD_NOISE_BUDGET_FRAC_TABLE: f64 = 0.02;

/// The 11 noise-eligible fields.
const ELIGIBLE: [&str; 11] = [
    "q_next", "alpha", "beta", "kappa", "a_up", "b_down", "area_edge", "radius_edge", "d_s", "dt_phi", "r_prime",
];

/// Move each finite non-zero entry by one ulp, the sign from a fixed-seed hash.
fn one_ulp(v: &[f32], seed: u32) -> Vec<f32> {
    v.iter()
        .enumerate()
        .map(|(i, &x)| {
            if x == 0.0 || !x.is_finite() {
                return x;
            }
            let h = (i as u32).wrapping_add(seed).wrapping_mul(2654435761);
            let bits = x.to_bits();
            f32::from_bits(if (h >> 16) & 1 == 0 { bits + 1 } else { bits - 1 })
        })
        .collect()
}

/// What one run records per hour.
struct HourRec {
    fields: Vec<Vec<f32>>,
    stage: Vec<f32>,
    mass: [f32; 5],
    steps: usize,
}

/// Run a case on one backend; with `perturb`, `h_t` and `q_t` of hour 0 move one ulp.
fn run_series<B: Backend + 'static>(
    case: &Case,
    theta: f32,
    substeps: u32,
    use_cuda: bool,
    perturb: bool,
    device: &B::Device,
) -> Vec<HourRec>
where
    B::FloatTensorPrimitive: 'static,
    B::Device: 'static,
{
    let mut block = make_block_p::<B>(case, theta, substeps, use_cuda, perturb, device);
    let (nn, pp, qq) = (tv::<B>(&case.n, device), tv::<B>(&case.p, device), tv::<B>(&case.q, device));
    let mut q = if perturb { one_ulp(&case.q_t0, 2) } else { case.q_t0.clone() };
    let mut out = vec![];
    for k in 0..case.hours {
        let (qp, mc) = (pick(&case.qp, k), pick(&case.mc, k));
        let o = to_vec1(block.step_fused(
            &case.pattern, nn.clone(), pp.clone(), qq.clone(), tv::<B>(&q, device), tv::<B>(&mc, device), tv::<B>(&qp, device),
        ));
        let c = block.last_coefficients().unwrap();
        let mut fields = vec![o.clone()];
        for t in [c.alpha, c.beta, c.kappa, c.a_up, c.b_down, c.area_edge, c.radius_edge, c.d_s, c.dt_phi, c.r_prime] {
            fields.push(to_vec1(t));
        }
        let mb = block.mass_balance();
        out.push(HourRec {
            fields,
            stage: to_vec1(block.stage().unwrap()),
            mass: [mb.storage_change_m3, mb.net_flux_m3, mb.floor_created_m3, mb.gross_flux_m3, mb.mc_inflow_m3],
            steps: mb.steps,
        });
        q = o;
    }
    out
}

/// One accepted entry.
struct Accepted {
    tag: String,
    field: &'static str,
    hour: usize,
    reach: usize,
    cpu: f32,
    dev: f32,
    err_scale: f32,
    err_noise: f32,
    noise_cpu: f32,
    noise_dev: f32,
}

/// How a trajectory comparison is gated (rulings B3 and B4).
#[derive(Clone, Copy, PartialEq)]
enum Gate {
    /// Every entry within the strict bar. No acceptance.
    Strict,
    /// Strict bar first; entries over it are printed; every field must stay within
    /// `STIFF_SANITY_FRAC * max|cpu|` and each mass term within `STIFF_SANITY_FRAC * gross`.
    Sanity,
    /// The B3 noise gate (Hudson).
    Noise,
}

/// Sanity bar of the stiff families (ruling B4).
const STIFF_SANITY_FRAC: f32 = 2e-3;

/// Counts of one family.
#[derive(Default)]
struct Tally {
    /// Entries over the strict bar (Sanity gate), with their lines.
    over_strict: usize,
    over_lines: Vec<String>,
    compared: usize,
    accepted: Vec<Accepted>,
    worst_scale: f32,
    worst_noise: f32,
}

/// The worst relative difference per field and hour, plus the gate tally.
struct Worst {
    per_field: BTreeMap<String, f32>,
    per_hour: Vec<f32>,
    mass_worst: f32,
    tally: Tally,
}

fn max_abs_diff(a: &[f32], b: &[f32]) -> f32 {
    a.iter().zip(b).fold(0.0f32, |m, (x, y)| m.max((x - y).abs()))
}

/// Run one case on both backends and apply the D7 forward bars with the B3
/// noise gate. `bar` is the strict relative bar. Returns the first entry that
/// fails the gate.
fn run_pair_xb(case: &Case, theta: f32, substeps: u32, bar: f32, gate: Gate) -> Result<Worst, String> {
    let (dc, dg): (<Cpu as BackendTypes>::Device, <Gpu as BackendTypes>::Device) = (Default::default(), Default::default());
    let cb = run_series::<Cpu>(case, theta, substeps, false, false, &dc);
    let noisy = gate == Gate::Noise;
    let cp = if noisy { run_series::<Cpu>(case, theta, substeps, false, true, &dc) } else { vec![] };
    let gb = run_series::<Gpu>(case, theta, substeps, true, false, &dg);
    let gp = if noisy { run_series::<Gpu>(case, theta, substeps, true, true, &dg) } else { vec![] };
    let mut w = Worst { per_field: BTreeMap::new(), per_hour: vec![], mass_worst: 0.0, tally: Tally::default() };
    for k in 0..case.hours {
        let tag = format!("{} theta {theta} substeps {substeps} hour {k}", case.label);
        let mut hour_worst = 0.0f32;
        // Stage: strict (Strict, Noise) or sanity-gated (Sanity).
        {
            let (d, c) = (&gb[k].stage, &cb[k].stage);
            let scale = c.iter().fold(0.0f32, |m, v| m.max(v.abs()));
            let worst = max_abs_diff(d, c);
            if gate == Gate::Sanity {
                for (i, (x, y)) in d.iter().zip(c).enumerate() {
                    let err = (x - y).abs();
                    w.tally.compared += 1;
                    if scale > 0.0 {
                        w.tally.worst_scale = w.tally.worst_scale.max(err / scale);
                    }
                    if err > bar * scale {
                        w.tally.over_strict += 1;
                        w.tally.over_lines.push(format!(
                            "{tag} field stage reach {i}: cpu {y} cuda {x} err/max|cpu| {:e}",
                            err / scale
                        ));
                    }
                    if !(err <= STIFF_SANITY_FRAC * scale) {
                        return Err(format!("{tag} stage reach {i}: dev {x} vs cpu {y} over the sanity bar"));
                    }
                }
            } else if !(worst <= bar * scale) {
                return Err(format!("{tag} stage: max diff {worst} vs max|cpu| {scale} (strict, no acceptance)"));
            }
            let rel = if scale > 0.0 { worst / scale } else { worst };
            hour_worst = hour_worst.max(rel);
            let e = w.per_field.entry("stage".to_string()).or_insert(0.0);
            *e = e.max(rel);
        }
        if gb[k].steps != cb[k].steps {
            return Err(format!("{tag} steps: dev {} vs cpu {}", gb[k].steps, cb[k].steps));
        }
        let tol = 1e-5 * cb[k].mass[3].abs();
        for (j, name) in ["storage", "net", "floor", "gross", "mc_in"].iter().enumerate() {
            let (a, r) = (gb[k].mass[j], cb[k].mass[j]);
            if gate == Gate::Sanity {
                w.tally.compared += 1;
                if !((a - r).abs() <= tol) {
                    w.tally.over_strict += 1;
                    w.tally.over_lines.push(format!(
                        "{tag} mass {name}: cpu {r} cuda {a} err/gross {:e}",
                        (a - r).abs() / cb[k].mass[3].abs()
                    ));
                }
                if !((a - r).abs() <= STIFF_SANITY_FRAC * cb[k].mass[3].abs()) {
                    return Err(format!("{tag} mass {name}: dev {a} vs cpu {r} over the sanity bar"));
                }
            } else if !((a - r).abs() <= tol) {
                return Err(format!("{tag} mass {name}: dev {a} vs cpu {r} (tol {tol:e})"));
            }
            w.mass_worst = w.mass_worst.max((a - r).abs() / cb[k].mass[3].abs().max(f32::MIN_POSITIVE));
        }
        // Gated: q_next and the ten coefficient fields.
        for (fi, name) in ELIGIBLE.iter().enumerate() {
            let (d, c) = (&gb[k].fields[fi], &cb[k].fields[fi]);
            if d.len() != c.len() {
                return Err(format!("{tag} {name}: length {} vs {}", d.len(), c.len()));
            }
            let scale = c.iter().fold(0.0f32, |m, v| m.max(v.abs()));
            let noise_cpu = if noisy { max_abs_diff(c, &cp[k].fields[fi]) } else { 0.0 };
            let noise_dev = if noisy { max_abs_diff(d, &gp[k].fields[fi]) } else { 0.0 };
            let limit = FWD_NOISE_FACTOR * (noise_cpu + noise_dev);
            let mut worst = 0.0f32;
            for (i, (x, y)) in d.iter().zip(c).enumerate() {
                if !(x.is_finite() && y.is_finite()) {
                    return Err(format!("{tag} {name} reach {i}: non-finite dev {x} cpu {y}"));
                }
                let err = (x - y).abs();
                w.tally.compared += 1;
                worst = worst.max(err);
                if scale > 0.0 {
                    w.tally.worst_scale = w.tally.worst_scale.max(err / scale);
                }
                if noise_cpu + noise_dev > 0.0 {
                    w.tally.worst_noise = w.tally.worst_noise.max(err / (noise_cpu + noise_dev));
                }
                if err > bar * scale && gate == Gate::Sanity {
                    w.tally.over_strict += 1;
                    w.tally.over_lines.push(format!(
                        "{tag} field {name} reach {i}: cpu {y} cuda {x} err/max|cpu| {:e}",
                        err / scale
                    ));
                    if !(err <= STIFF_SANITY_FRAC * scale) {
                        return Err(format!("{tag} {name} reach {i}: dev {x} vs cpu {y} over the sanity bar"));
                    }
                } else if err > bar * scale && gate == Gate::Strict {
                    return Err(format!(
                        "{tag} {name} reach {i}: dev {x} vs cpu {y} (err/max|cpu| {:e}, strict bar {bar:e}, no acceptance)",
                        err / scale
                    ));
                } else if err > bar * scale {
                    if err <= limit && err <= FWD_NOISE_CAP_FRAC * scale {
                        w.tally.accepted.push(Accepted {
                            tag: tag.clone(),
                            field: name,
                            hour: k,
                            reach: i,
                            cpu: *y,
                            dev: *x,
                            err_scale: err / scale,
                            err_noise: err / (noise_cpu + noise_dev),
                            noise_cpu,
                            noise_dev,
                        });
                    } else {
                        return Err(format!(
                            "{tag} {name} reach {i}: dev {x} vs cpu {y} (err/max|cpu| {:e}, strict bar {bar:e}, cap {:e}; err/noise-sum {:e}, limit {limit:e} = {FWD_NOISE_FACTOR} x (noise_cpu {noise_cpu:e} + noise_dev {noise_dev:e}))",
                            err / scale,
                            FWD_NOISE_CAP_FRAC,
                            err / (noise_cpu + noise_dev)
                        ));
                    }
                }
            }
            let rel = if scale > 0.0 { worst / scale } else { worst };
            hour_worst = hour_worst.max(rel);
            let e = w.per_field.entry(name.to_string()).or_insert(0.0);
            *e = e.max(rel);
        }
        w.per_hour.push(hour_worst);
    }
    Ok(w)
}

fn report(label: &str, theta: f32, substeps: u32, w: &Worst) {
    let overall = w.per_field.values().fold(0.0f32, |m, v| m.max(*v));
    let (field, _) = w.per_field.iter().fold(("", 0.0f32), |m, (k, v)| if *v >= m.1 { (k.as_str(), *v) } else { m });
    println!(
        "{label} theta {theta} substeps {substeps}: worst rel {overall:e} ({field}), mass diff/gross {:e}, accepted {}/{}",
        w.mass_worst,
        w.tally.accepted.len(),
        w.tally.compared
    );
}

impl Tally {
    fn add(&mut self, mut o: Tally) {
        self.compared += o.compared;
        self.over_strict += o.over_strict;
        self.over_lines.append(&mut o.over_lines);
        self.worst_scale = self.worst_scale.max(o.worst_scale);
        self.worst_noise = self.worst_noise.max(o.worst_noise);
        self.accepted.append(&mut o.accepted);
    }

    /// Print the family summary and every accepted entry; assert the budget.
    fn finish(&self, family: &str, budget: f64) {
        println!(
            "FAMILY {family}: compared {}, accepted {}, over strict bar {}, worst err/max|cpu| {:e}, worst err/noise {:e}, budget {}",
            self.compared,
            self.accepted.len(),
            self.over_strict,
            self.worst_scale,
            self.worst_noise,
            (budget * self.compared as f64).floor()
        );
        for l in &self.over_lines {
            println!("OVER {family}: {l}");
        }
        for a in &self.accepted {
            println!(
                "ACCEPTED {family}: {} field {} hour {} reach {}: cpu {} cuda {} err/max|cpu| {:e} err/noise {:e} (noise_cpu {:e}, noise_cuda {:e})",
                a.tag, a.field, a.hour, a.reach, a.cpu, a.dev, a.err_scale, a.err_noise, a.noise_cpu, a.noise_dev
            );
        }
        assert!(
            self.accepted.len() as f64 <= budget * self.compared as f64,
            "{family}: {} accepted entries exceed the budget {} of {} compared",
            self.accepted.len(),
            budget,
            self.compared
        );
    }
}

fn run_grid(case: &Case, bar: f32, gate: Gate, tally: &mut Tally) {
    for theta in [0.55f32, 0.6, 1.0] {
        for substeps in [1u32, 4] {
            match run_pair_xb(case, theta, substeps, bar, gate) {
                Ok(w) => {
                    report(&case.label, theta, substeps, &w);
                    tally.add(w.tally);
                }
                Err(e) => panic!("{e}"),
            }
        }
    }
}

fn family(name: &str, cases: Vec<Case>, budget: f64) {
    let mut t = Tally::default();
    for c in &cases {
        run_grid(c, 1e-5, Gate::Sanity, &mut t);
    }
    t.finish(name, budget);
}

// copied from tests/stage_op_parity.rs::chain_state_cases
fn chain_state_cases() -> Vec<Case> {
    let mut cases = vec![];
    // gradcheck_through_one_step
    {
        let adj = linear_chain_sparse(5);
        let pattern = Arc::new(CsrPattern::from_sparse(&adj));
        let net = StageNetwork::build(&adj, &pattern, None, 0.0, &[]).unwrap();
        let h0: Vec<f32> =
            (0..5).map(|i| net.z_mid[i] + 0.4 + 0.07 * (i as f32) + 0.03 * ((i * i) as f32 % 3.0)).collect();
        let mut c = Case::new("one_step", adj, pattern, H0::Abs(h0), vec![1.1, 2.3, 3.2, 4.6, 5.1]);
        c.n = vec![0.03, 0.035, 0.04, 0.045, 0.05];
        c.q = vec![0.45, 0.5, 0.55, 0.5, 0.6];
        c.p = vec![18.0, 21.0, 24.0, 20.0, 22.0];
        cases.push(c);
    }
    // reversed edge and negative flow
    {
        let (adj, pattern) = chain(&[1000.0; 5], &[2e-4; 5]);
        cases.push(Case::new(
            "reversed_edge",
            adj,
            pattern,
            H0::Depth(vec![0.3, 0.7, 0.6, 0.5, 0.5]),
            vec![0.8, -1.2, 1.5, 1.8, 2.0],
        ));
    }
    // dam control edge on row 2, T 1.23 d
    {
        let (adj, pattern) = chain(&[1000.0; 5], &[1e-3; 5]);
        let mut c = Case::new(
            "dam_edge",
            adj,
            pattern,
            H0::Depth(vec![0.45, 0.55, 0.8, 0.6, 0.7]),
            vec![1.1, 2.3, 3.2, 4.6, 5.1],
        );
        c.dams = vec![(2, 1.23)];
        cases.push(c);
    }
    // adverse flow: the b clamp binds
    {
        let (adj, pattern) = chain(&[1000.0; 3], &[1e-3; 3]);
        cases.push(Case::new(
            "adverse_flow",
            adj,
            pattern,
            H0::Depth(vec![0.3, 1.5, 1.0]),
            vec![10.0, 5.0, 6.0],
        ));
    }
    // Q = 0 exactly, flat bed, constant prescribed series
    {
        let (adj, pattern) = chain(&[1000.0; 3], &[0.0; 3]);
        let mut c = Case::new("zero_flow", adj, pattern, H0::Abs(vec![1.0; 3]), vec![0.0; 3]).prescribed(1.0, 0.0);
        c.qp = vec![vec![0.0; 3]];
        cases.push(c);
    }
    // the depth floor binds
    {
        let (adj, pattern) = chain(&[1000.0; 3], &[1e-3; 3]);
        let net = StageNetwork::build(&adj, &pattern, None, 0.0, &[]).unwrap();
        let h0: Vec<f32> = net.z_mid.iter().map(|z| z + 0.05).collect();
        let mut c = Case::new("depth_floor", adj, pattern, H0::Abs(h0), vec![0.0; 3]).prescribed(-5.0, 0.0);
        c.qp = vec![vec![0.0; 3]];
        c.hours = 6;
        cases.push(c);
    }
    cases
}

#[test]
fn forward_parity_chain_states() {
    family("chain_states", chain_state_cases(), 0.0);
}

// copied from tests/stage_op_parity.rs::table_and_mixed_cases
fn table_and_mixed_cases() -> Vec<Case> {
    let mut cases = vec![];
    // table_rows_use_the_rectangle_and_fixed_n
    {
        let (adj, pattern) = chain(&[1000.0; 3], &[1e-3; 3]);
        let net = StageNetwork::build(&adj, &pattern, None, 0.0, &[]).unwrap();
        let mut h0: Vec<f32> = net.z_mid.iter().map(|z| z + 1.0).collect();
        h0[2] = 1.5;
        let mut c = Case::new("table_rect", adj, pattern, H0::Abs(h0), vec![3.0, 5.0, 7.0]).prescribed(-5.0, 0.0);
        c.table = Some((table(&[2], 80.0, &[0.3], 0.02, &[None]), false));
        c.qp = vec![vec![0.5, 0.4, 0.3]];
        cases.push(c);
    }
    // pool_z_sets_the_release_reference: stage 1 m and 3 m above the bed
    for above in [1.0f32, 3.0] {
        let (adj, pattern) = chain(&[1000.0; 3], &[1e-3; 3]);
        let mut c = Case::new(
            &format!("pool_dam_{above}m"),
            adj,
            pattern,
            H0::Abs(vec![10.0 + above, 12.0, 12.0]),
            vec![0.0; 3],
        )
        .prescribed(-5.0, 0.0);
        c.dams = vec![(0, 1.5)];
        c.theta_c = 1.0;
        c.table = Some((table(&[0, 1, 2], 400.0, &[10.0, 10.0, 10.0], 0.03, &[Some(12.0), None, None]), true));
        c.qp = vec![vec![0.0; 3]];
        cases.push(c);
    }
    // mixed geometry: the table lists row 2 only, the head serves rows 0 and 1
    {
        let (adj, pattern) = chain(&[1000.0; 3], &[1e-3; 3]);
        let net = StageNetwork::build(&adj, &pattern, None, 0.0, &[]).unwrap();
        let z2 = net.z_mid[2];
        let mut c = Case::new("mixed", adj, pattern, H0::Depth(vec![1.0; 3]), vec![3.0, 5.0, 7.0]).prescribed(-5.0, 0.0);
        c.table = Some((table(&[2], 80.0, &[z2], 0.02, &[None]), false));
        c.qp = vec![vec![0.5, 0.4, 0.3]];
        cases.push(c);
    }
    cases
}

#[test]
fn forward_parity_table_and_mixed() {
    family("table_and_mixed", table_and_mixed_cases(), 0.0);
}

// copied from tests/stage_op_parity.rs::prescribed_boundary_cases
fn prescribed_boundary_cases() -> Vec<Case> {
    let mut cases = vec![];
    // flow_reversal_is_not_clamped: the outlet stage rises 1 m per hour
    {
        let (adj, pattern) = chain(&[1000.0; 2], &[1e-3; 2]);
        let net = StageNetwork::build(&adj, &pattern, None, 0.0, &[]).unwrap();
        let h: Vec<f32> = net.z_mid.iter().map(|z| z + 0.5).collect();
        let base = h[1];
        let mut c = Case::new("flow_reversal", adj, pattern, H0::Abs(h), vec![0.0; 2]).prescribed(base, 1.0);
        c.qp = vec![vec![0.0; 2]];
        cases.push(c);
    }
    // a 3-chain with a series rising 0.5 m per hour
    {
        let (adj, pattern) = chain(&[1000.0; 3], &[1e-3; 3]);
        let net = StageNetwork::build(&adj, &pattern, None, 0.0, &[]).unwrap();
        let h: Vec<f32> = net.z_mid.iter().map(|z| z + 0.8).collect();
        let base = h[2];
        let c = Case::new("rising_series", adj, pattern, H0::Abs(h), vec![2.0, 3.0, 4.0]).prescribed(base, 0.5);
        cases.push(c);
    }
    cases
}

#[test]
fn forward_parity_prescribed_boundary() {
    family("prescribed_boundary", prescribed_boundary_cases(), 0.0);
}

// copied from tests/stage_op_parity.rs::flat
fn flat(sf: &Tensor<TestBackend, 2>) -> Vec<f32> {
    sf.clone().into_data().to_vec().unwrap()
}

// copied from tests/stage_op_parity.rs::mc_inflow_case
fn mc_inflow_case(device: &TestDevice) -> Case {
    let n = 3;
    let hours = 24;
    let t = hours + 1;
    let adj = linear_chain_sparse(n);
    let pattern = Arc::new(CsrPattern::from_sparse(&adj));
    let sf = mock_streamflow(t, n, &device);
    let mut mcr = MuskingumCunge::<InnerBackend>::new(mock_config(), device.clone());
    mcr.setup_inputs(
        RoutingInputs { adjacency: adj.clone(), x_storage: Tensor::<TestBackend, 1>::ones([n], device) * 0.2 },
        sf.clone(),
        mock_spatial_parameters(n, &device),
        false,
        None,
    );
    let f: Vec<f32> = mcr.forward().into_data().to_vec().unwrap();
    let mc: Vec<Vec<f32>> = (0..n).map(|i| f[i * t..(i + 1) * t].to_vec()).collect();
    let sfl = flat(&sf);
    let q_t0: Vec<f32> = (0..n).map(|i| mc[i][0]).collect();
    let net = StageNetwork::build(&adj, &pattern, Some(&[1]), 0.0, &[]).unwrap();
    let h0: Vec<f32> = net.z_mid.iter().map(|z| z + 0.6).collect();
    let mut case = Case::new("mc_inflow", adj, pattern, H0::Abs(h0), q_t0);
    case.seeds = Some(vec![1]);
    case.hours = hours;
    case.qp = (0..hours).map(|k| sfl[k * n..(k + 1) * n].to_vec()).collect();
    case.mc = (0..hours).map(|k| (0..n).map(|i| mc[i][k + 1]).collect()).collect();
    case
}

#[test]
fn forward_parity_mc_inflow() {
    let device = TestDevice::default();
    let case = mc_inflow_case(&device);
    let mut t = Tally::default();
    run_grid(&case, 1e-5, Gate::Strict, &mut t);
    t.finish("mc_inflow", 0.0);
}

// copied from tests/stage_op_parity.rs::juniata_case, with the bundle path a parameter
fn bundle_case(label: &str, adjacency: &str, device: &TestDevice) -> Case {
    let store = ConusAdjacencyStore::open(adjacency).expect("open bundle conus adjacency");
    let adj = SparseAdjacency {
        n: store.n,
        rows: store.indices_0.clone(),
        cols: store.indices_1.clone(),
        values: vec![1.0; store.nnz],
        length_m: store.length_m.to_vec(),
        slope: store.slope.to_vec(),
        parent_offset: None,
    };
    let pattern = Arc::new(CsrPattern::from_sparse(&adj));
    let n = adj.n;
    // cumulative upstream sum of q' = 1
    let mut q_init = vec![1.0f32; n];
    let mut down = vec![usize::MAX; n];
    for (&r, &c) in adj.rows.iter().zip(&adj.cols) {
        down[c as usize] = r as usize;
    }
    for i in 0..n {
        if down[i] != usize::MAX {
            let v = q_init[i];
            q_init[down[i]] += v;
        }
    }
    // hotstart stage on a throwaway block
    let net = StageNetwork::build(&adj, &pattern, None, 0.0, &[]).unwrap();
    let mut hb = StageBlock::<InnerBackend>::new(
        net,
        StageParams {
            theta: 1.0,
            theta_control: 0.5,
            substeps: 1,
            boundary: DownstreamBoundary::NormalDepth,
            depth_lb: DEPTH_LB,
            bottom_width_lb: 0.1,
            dt: 3600.0,
            g: 9.81,
            use_cuda: false,
        },
        device,
    );
    let slope_f: Vec<f32> = adj.slope.iter().map(|s| s.max(1e-3)).collect();
    hb.hotstart(
        tv::<Cpu>(&q_init, device),
        tv::<Cpu>(&vec![0.035; n], device),
        tv::<Cpu>(&vec![21.0; n], device),
        tv::<Cpu>(&vec![0.5; n], device),
        tv::<Cpu>(&slope_f, device),
    );
    let h0 = to_vec1(hb.stage().unwrap());
    let mut case = Case::new(label, adj, pattern, H0::Abs(h0), q_init);
    case.n = vec![0.035; n];
    case.p = vec![21.0; n];
    case.q = vec![0.5; n];
    case.qp = vec![vec![1.0; n]];
    case.mc = vec![vec![0.0; n]];
    case.hours = 48;
    case
}

const JUNIATA: &str = "examples/juniata/data/juniata_conus_adjacency.zarr";
const HUDSON: &str = "examples/hudson/data/hudson_conus_adjacency.zarr";

fn juniata_case(device: &TestDevice) -> Case {
    bundle_case("juniata_48h", JUNIATA, device)
}

/// The Hudson bundle network, all stage, built as `juniata_case`.
fn hudson_net_case(device: &TestDevice) -> Case {
    bundle_case("hudson_net_48h", HUDSON, device)
}

fn long_run(case: &Case, grid: &[(f32, u32)], gate: Gate) {
    for &(theta, substeps) in grid {
        match run_pair_xb(case, theta, substeps, 1e-4, gate) {
            Ok(w) => {
                report(&case.label, theta, substeps, &w);
                let hours: Vec<String> = w.per_hour.iter().map(|v| format!("{v:.2e}")).collect();
                println!("{} theta {theta} substeps {substeps} worst per hour: [{}]", case.label, hours.join(", "));
                for (f, v) in &w.per_field {
                    println!("  field {f}: {v:.3e}");
                }
                w.tally.finish(&format!("{} th{theta} s{substeps}", case.label), if gate == Gate::Noise { FWD_NOISE_BUDGET_FRAC } else { 0.0 });
            }
            Err(e) => panic!("{e}"),
        }
    }
}

#[test]
fn forward_parity_juniata_48h() {
    if cfg!(debug_assertions) {
        eprintln!("skipped in debug builds: run with --release");
        return;
    }
    let device = TestDevice::default();
    long_run(&juniata_case(&device), &[(1.0, 1), (0.6, 4)], Gate::Strict);
}

#[test]
fn forward_parity_hudson_net_48h() {
    if cfg!(debug_assertions) {
        eprintln!("skipped in debug builds: run with --release");
        return;
    }
    let device = TestDevice::default();
    long_run(&hudson_net_case(&device), &[(0.6, 4)], Gate::Noise);
}

// ------------------------------------------------------- host-read gates

/// Three tracked hours on the Juniata case (record off). Outputs stay on the
/// device until the caller reads them. Returns the `q_next` tensors and the
/// counter value before and after the steps.
fn three_tracked_hours(
    case: &Case,
    use_cuda: bool,
) -> (Vec<Tensor<Autodiff<Gpu>, 1>>, u64, u64) {
    let dg: <Gpu as BackendTypes>::Device = Default::default();
    let mut block = make_block::<Gpu>(case, 0.6, 4, use_cuda, &dg);
    // Warm the per-block caches: the one-time read of the device constants is allowed.
    let _ = block.op_consts(&case.pattern);
    if use_cuda {
        let _ = block.op_consts_dev(&case.pattern);
    }
    let leaf = |v: &[f32]| tv::<Gpu>(v, &dg).require_grad();
    let (nn, pp, qq) = (leaf(&case.n), leaf(&case.p), leaf(&case.q));
    let (mc, qp) = (tv::<Gpu>(&pick(&case.mc, 0), &dg), tv::<Gpu>(&pick(&case.qp, 0), &dg));
    let mut q = leaf(&case.q_t0);
    let before = host_reads_on_this_thread_for_test();
    let mut outs = vec![];
    for _ in 0..3 {
        q = block.step_fused(&case.pattern, nn.clone(), pp.clone(), qq.clone(), q, mc.clone(), qp.clone());
        outs.push(q.clone());
    }
    let after = host_reads_on_this_thread_for_test();
    (outs, before, after)
}

#[test]
fn forward_makes_no_host_copy_on_cuda() {
    let device = TestDevice::default();
    let mut case = juniata_case(&device);
    case.hours = 3;
    // The NdArray values.
    let dc: <Cpu as BackendTypes>::Device = Default::default();
    let mut cpu = make_block::<Cpu>(&case, 0.6, 4, false, &dc);
    let (nn, pp, qq) = (tv::<Cpu>(&case.n, &dc), tv::<Cpu>(&case.p, &dc), tv::<Cpu>(&case.q, &dc));
    let (mc, qp) = (tv::<Cpu>(&pick(&case.mc, 0), &dc), tv::<Cpu>(&pick(&case.qp, 0), &dc));
    let mut q = tv::<Cpu>(&case.q_t0, &dc);
    let mut want = vec![];
    for _ in 0..3 {
        q = cpu.step_fused(&case.pattern, nn.clone(), pp.clone(), qq.clone(), q, mc.clone(), qp.clone());
        want.push(to_vec1(q.clone()));
    }
    // Control: Gpu with use_cuda false takes the host path on device tensors. It must read.
    let (outs, before, after) = three_tracked_hours(&case, false);
    println!("control host reads: {}", after - before);
    assert!(after > before, "the control arm must read the device ({before} -> {after})");
    for (k, (o, w)) in outs.into_iter().zip(&want).enumerate() {
        let got = to_vec1(o);
        for (i, (a, b)) in got.iter().zip(w).enumerate() {
            assert_eq!(a.to_bits(), b.to_bits(), "control hour {k} reach {i}: {a} vs {b}");
        }
    }
    // Gate: the device arm reads nothing in the step.
    let (outs, before, after) = three_tracked_hours(&case, true);
    println!("gate host reads: {}", after - before);
    assert_eq!(after, before, "the device step made {} host reads", after - before);
    for (k, (o, w)) in outs.into_iter().zip(&want).enumerate() {
        let got = to_vec1(o);
        let scale = w.iter().fold(0.0f32, |m, v| m.max(v.abs()));
        let worst = got.iter().zip(w).fold(0.0f32, |m, (a, b)| m.max((a - b).abs()));
        assert!(worst <= 1e-5 * scale, "gate hour {k}: worst {worst} vs max|cpu| {scale}");
    }
}

#[test]
fn untracked_forward_matches_tracked_on_cuda() {
    let device = TestDevice::default();
    let case = juniata_case(&device);
    let dg: <Gpu as BackendTypes>::Device = Default::default();
    let run = |tracked: bool| -> Vec<Vec<f32>> {
        let mut block = make_block::<Gpu>(&case, 0.6, 4, true, &dg);
        let leaf = |v: &[f32]| {
            let t = tv::<Gpu>(v, &dg);
            if tracked {
                t.require_grad()
            } else {
                t
            }
        };
        let (nn, pp, qq) = (leaf(&case.n), leaf(&case.p), leaf(&case.q));
        let (mc, qp) = (leaf(&pick(&case.mc, 0)), leaf(&pick(&case.qp, 0)));
        let mut q = leaf(&case.q_t0);
        let mut outs = vec![];
        for _ in 0..6 {
            q = block.step_fused(&case.pattern, nn.clone(), pp.clone(), qq.clone(), q, mc.clone(), qp.clone());
            outs.push(to_vec1(q.clone()));
        }
        outs
    };
    let (a, b) = (run(false), run(true));
    for (k, (x, y)) in a.iter().zip(&b).enumerate() {
        for (i, (p, q)) in x.iter().zip(y).enumerate() {
            assert_eq!(p.to_bits(), q.to_bits(), "hour {k} reach {i}: untracked {p} vs tracked {q}");
        }
    }
}

#[test]
#[should_panic(expected = "stage_routing CUDA path: Phase G")]
fn cuda_arm_is_the_phase_g_stub() {
    // The forward runs on the device; the backward is still the stub (Task 8).
    let device: <Gpu as BackendTypes>::Device = Default::default();
    let tv = |v: &[f32]| Tensor::<Autodiff<Gpu>, 1>::from_floats(v, &device).require_grad();
    let (adj, pattern) = chain(&[1000.0; 3], &[1e-3; 3]);
    let net = StageNetwork::build(&adj, &pattern, None, 0.0, &[]).unwrap();
    let z = net.z_mid.clone();
    let sp = StageParams {
        theta: 1.0,
        theta_control: 0.5,
        substeps: 1,
        boundary: DownstreamBoundary::NormalDepth,
        depth_lb: DEPTH_LB,
        bottom_width_lb: 0.1,
        dt: 3600.0,
        g: 9.81,
        use_cuda: true,
    };
    let mut block = StageBlock::<Gpu>::new(net, sp, &device);
    let h: Vec<f32> = z.iter().map(|z| z + 0.5).collect();
    block.set_stage(tv(&h));
    let out = block.step_fused(
        &pattern,
        tv(&[0.035; 3]),
        tv(&[21.0; 3]),
        tv(&[0.5; 3]),
        tv(&[1.0; 3]),
        tv(&[0.0; 3]),
        tv(&[1.0; 3]),
    );
    let _ = out.sum().backward();
}


// ------------------------------------------- synced-state substep checks
//
// The CPU host trajectory is run; at every substep the device gets the HOST
// start state, HourHost, a_values, r and dh, and each device block is compared
// with the host block. Strict everywhere: block bar 1e-5 x max|host|, 0/1
// indicators exact, same zero set on the clamped outputs. No noise gate.

/// Worst err/bar per field for one family; collects every miss.
/// Cancellation fraction of the synced-state bar for `r` and `q_new` (ruling B4).
const SYNC_CANCEL_FRAC: f32 = 1e-6;

struct Rec {
    worst: BTreeMap<String, f32>,
    /// Per field: (entries not bitwise equal, entries compared).
    bits: BTreeMap<String, (usize, usize)>,
    /// Every miss; the test fails at the end if this is not empty.
    misses: Vec<String>,
}

impl Rec {
    // copied from tests/stage_cuda_blocks.rs::close, recording the worst err/bar per field
    fn close(&mut self, ctx: &str, key: &str, dev: &[f32], host: &[f32]) {
        self.close_t(ctx, key, dev, host, None);
    }

    /// The block bar `1e-5 * max|host|`; with `terms`, the per-entry cancellation
    /// bar `max(1e-5 * max|host|, SYNC_CANCEL_FRAC * terms[i])` (ruling B4).
    fn close_t(&mut self, ctx: &str, key: &str, dev: &[f32], host: &[f32], terms: Option<&[f32]>) {
        assert_eq!(dev.len(), host.len(), "{ctx} {key}: length");
        let scale = host.iter().fold(0.0f32, |m, v| m.max(v.abs()));
        let mut worst_ratio = 0.0f32;
        {
            let e = self.bits.entry(key.to_string()).or_insert((0, 0));
            e.0 += dev.iter().zip(host).filter(|(d, h)| d.to_bits() != h.to_bits()).count();
            e.1 += dev.len();
        }
        for (i, (d, h)) in dev.iter().zip(host).enumerate() {
            if !(d.is_finite() && h.is_finite()) {
                self.misses.push(format!("{ctx} {key}[{i}]: non-finite dev {d} host {h}"));
                return;
            }
            let err = (d - h).abs();
            let mut bar = 1e-5 * scale;
            if let Some(t) = terms {
                bar = bar.max(SYNC_CANCEL_FRAC * t[i]);
            }
            let ratio = if bar > 0.0 { err / bar } else if err == 0.0 { 0.0 } else { f32::INFINITY };
            worst_ratio = worst_ratio.max(ratio);
            if !(err <= bar) {
                self.misses.push(format!(
                    "{ctx} {key} reach {i}: dev {d} vs host {h} (err/bar {ratio:e}, bar {bar:e}, max|host| {scale:e})"
                ));
            }
        }
        let e = self.worst.entry(key.to_string()).or_insert(0.0);
        *e = e.max(worst_ratio);
    }

    // copied from tests/stage_cuda_blocks.rs::exact
    fn exact(&mut self, ctx: &str, key: &str, dev: &[f32], host: &[f32]) {
        assert_eq!(dev.len(), host.len(), "{ctx} {key}: length");
        for (i, (d, h)) in dev.iter().zip(host).enumerate() {
            if d.to_bits() != h.to_bits() {
                self.misses.push(format!("{ctx} {key}[{i}]: dev {d} vs host {h}"));
            }
        }
        self.worst.entry(key.to_string()).or_insert(0.0);
    }

    // copied from tests/stage_cuda_blocks.rs::same_zero_set
    fn zero_set(&mut self, ctx: &str, key: &str, dev: &[f32], host: &[f32]) {
        for (i, (d, h)) in dev.iter().zip(host).enumerate() {
            if (*d == 0.0) != (*h == 0.0) {
                self.misses.push(format!("{ctx} {key} zero set, reach {i}: dev {d} host {h}"));
            }
        }
    }

    fn close_z(&mut self, ctx: &str, key: &str, dev: &[f32], host: &[f32]) {
        self.close(ctx, key, dev, host);
        self.zero_set(ctx, key, dev, host);
    }

    fn section(&mut self, ctx: &str, pre: &str, d: &SectionHost, h: &SectionHost) {
        macro_rules! f {
            ($($x:ident),*) => { $( self.close(ctx, &format!("{pre}.{}", stringify!($x)), &d.$x, &h.$x); )* };
        }
        f!(pw, tw, ss_raw, area, root, perim, radius, tw_d, ss_d, a_d, p_d, out_tw, out_area, out_perim, out_radius,
           out_a_d, out_p_d);
        self.close_z(ctx, &format!("{pre}.ss"), &d.ss, &h.ss);
        self.close(ctx, &format!("{pre}.bw_raw"), &d.bw_raw, &h.bw_raw);
        self.close_z(ctx, &format!("{pre}.bw"), &d.bw, &h.bw);
        self.close(ctx, &format!("{pre}.bw_d"), &d.bw_d, &h.bw_d);
    }

    fn depth(&mut self, ctx: &str, d: &DepthOut, h: &DepthOut) {
        for (k, a, b) in [("x", &d.x, &h.x), ("h_down", &d.h_down, &h.h_down), ("der", &d.der, &h.der)] {
            self.close(ctx, &format!("depth.{k}"), a, b);
        }
        self.close_z(ctx, "depth.depth", &d.depth, &h.depth);
        self.close_z(ctx, "depth.d_edge", &d.d_edge, &h.d_edge);
        self.exact(ctx, "depth.s_up", &d.s_up, &h.s_up);
        self.exact(ctx, "depth.s_dn", &d.s_dn, &h.s_dn);
    }

    fn newton(&mut self, ctx: &str, d: &NewtonOut, h: &NewtonOut) {
        macro_rules! f {
            ($($x:ident),*) => { $( self.close(ctx, &format!("newton.{}", stringify!($x)), &d.$x, &h.$x); )* };
        }
        f!(phi, dt_phi, den, alpha, beta, phi_d, dh_edge, k1, kappa, q_star, ra, rb);
        self.close_z(ctx, "newton.a_up", &d.a_up, &h.a_up);
        self.close_z(ctx, "newton.b_down", &d.b_down, &h.b_down);
    }

    fn control(&mut self, ctx: &str, d: &ControlOut, h: &ControlOut) {
        macro_rules! f {
            ($($x:ident),*) => { $( self.close(ctx, &format!("control.{}", stringify!($x)), &d.$x, &h.$x); )* };
        }
        f!(rd_head, rpd_head, rd, rpd, kk, w, dk_dd, r_t, rpr);
        self.close_z(ctx, "control.r_prime", &d.r_prime, &h.r_prime);
        if !h.ex.is_empty() {
            self.close(ctx, "control.ex", &d.ex, &h.ex);
            self.exact(ctx, "control.act", &d.act, &h.act);
        }
    }
}

/// Compare one substep. `asm`/`ro` are the reference blocks (computed by the host fns), `da`/`ddh`/`dr`
/// the blocks under test. `dh_ref` is the reference solve; `dh_used` the solve output both readouts got.
/// Same bars as ruling B4: block bar, cancellation bars for `r` and `q_new`, mass bars.
#[allow(clippy::too_many_arguments)]
fn compare_substep(
    rec: &mut Rec,
    ctx: &str,
    c: &StageOpConsts,
    hour: &HourHost,
    qc: &[f32],
    asm: &Assembled,
    ro: &Readout,
    dh_ref: &[f32],
    dh_used: &[f32],
    da: &Assembled,
    ddh: &[f32],
    dr: &Readout,
    r_iso: Option<&[f32]>,
) {
    rec.close(ctx, "i_mid", &da.i_mid, &asm.i_mid);
    assert_eq!((da.hb0, da.hb1, da.dt_s), (asm.hb0, asm.hb1, asm.dt_s), "{ctx} hb0, hb1, dt_s");
    rec.depth(ctx, &da.depth, &asm.depth);
    rec.section(ctx, "edge", &da.edge, &asm.edge);
    rec.section(ctx, "store", &da.store, &asm.store);
    rec.close(ctx, "d_s", &da.d_s, &asm.d_s);
    rec.newton(ctx, &da.newton, &asm.newton);
    rec.control(ctx, &da.control, &asm.control);
    rec.close(ctx, "a_values", &da.a_values, &asm.a_values);
    // r bar terms: the terms of the lines of `stage_op.rs::rhs_host` for row i:
    //   r_i = ((qp_i + i_mid_i) + ine_i) - eo_i  [+ outlet_i * b_down_i * (th * (hb1 - hb0))]
    //   ine_i = sum over the slots of row i of eo[col] * adj_stage
    //   eo_i  = fo_i + co_i, fo = friction*((q_star*th) + qc*(1-th)), co = w_ctrl*r_t + (ctrl-w_ctrl)*qc
    // so t_i = |qp_i| + |i_mid_i| + sum |eo[col]*adj_stage| + |eo_i| (+ the prescribed term).
    let th = c.theta;
    let eo: Vec<f32> = (0..c.n)
        .map(|i| {
            c.friction[i] * ((asm.newton.q_star[i] * th) + qc[i] * (1.0 - th))
                + (c.w_ctrl[i] * asm.control.r_t[i] + (c.ctrl[i] - c.w_ctrl[i]) * qc[i])
        })
        .collect();
    let mut r_terms: Vec<f32> = (0..c.n).map(|i| hour.qp_stage[i].abs() + asm.i_mid[i].abs() + eo[i].abs()).collect();
    for kk in 0..c.nnz {
        r_terms[c.row[kk]] += (eo[c.col[kk]] * c.adj_stage[kk]).abs();
    }
    if c.prescribed {
        for i in 0..c.n {
            r_terms[i] += ((c.outlet[i] * asm.newton.b_down[i]) * (th * (asm.hb1 - asm.hb0))).abs();
        }
    }
    rec.close_t(ctx, "r", &da.r, &asm.r, Some(&r_terms));
    if let Some(r_iso) = r_iso {
        rec.close_t(ctx, "r (rhs kernel on host inputs)", r_iso, &asm.r, Some(&r_terms));
    }
    rec.close(ctx, "dh", ddh, dh_ref);
    rec.close(ctx, "h_solved", &dr.h_solved, &ro.h_solved);
    rec.close_z(ctx, "h_new", &dr.h_new, &ro.h_new);
    rec.close(ctx, "dh_down", &dr.dh_down, &ro.dh_down);
    // q_new bar terms: the line `q_fr = (q_star + a_up*dh) - b_down*dhd` of
    // `stage_op.rs::stage_readout_host`: |q_star| + |a_up*dh| + |b_down*dhd|.
    let q_terms: Vec<f32> = (0..c.n)
        .map(|i| {
            asm.newton.q_star[i].abs() + (asm.newton.a_up[i] * dh_used[i]).abs() + (asm.newton.b_down[i] * ro.dh_down[i]).abs()
        })
        .collect();
    rec.close_t(ctx, "q_new", &dr.q_new, &ro.q_new, Some(&q_terms));
    // Mass bar: 1e-5 * gross + sum_j |coef_j| * |dq_new_j|. The coefficients are the
    // derivatives of the block-9 mass code of `stage_op.rs::stage_readout_host` with respect
    // to q_new_j (fe_j = (w_edge_j*q_new_j + keep*qc_j)*in_stage_j, so dfe_j/dq_new_j = w_edge_j*in_j):
    //   net   = sum_i ((into_i + i_mid_i) + qp_i - fe_i) * dt_s, into_i = sum_k fe[col k]*adj_stage_k
    //           coef_j = dt_s * w_edge_j*in_j * (sum_{k: col k = j} adj_stage_k - 1)
    //   gross = (sum_i |fe_i|*(in_i + has_down_i*in_i) + ...) * dt_s
    //           |coef_j| = dt_s * w_edge_j*in_j * (in_j + has_down_j*in_j)
    //   storage, floor, mc_in do not depend on q_new: coef = 0.
    let mut colsum = vec![0.0f32; c.n];
    for kk in 0..c.nnz {
        colsum[c.col[kk]] += c.adj_stage[kk];
    }
    let dt_s = asm.dt_s;
    let (mut dnet, mut dgross) = (0.0f32, 0.0f32);
    for j in 0..c.n {
        let dq = (dr.q_new[j] - ro.q_new[j]).abs();
        let wj = c.w_edge[j] * c.in_stage[j];
        dnet += (dt_s * wj * (colsum[j] - 1.0)).abs() * dq;
        dgross += (dt_s * wj * (c.in_stage[j] + c.has_down[j] * c.in_stage[j])).abs() * dq;
    }
    let extra = [0.0f32, dnet, 0.0, dgross, 0.0];
    for (j, name) in ["storage", "net", "floor", "gross", "mc_in"].iter().enumerate() {
        let err = (dr.mass[j] - ro.mass[j]).abs();
        let bar = 1e-5 * ro.mass[3].abs() + extra[j];
        let ratio = if bar > 0.0 { err / bar } else if err == 0.0 { 0.0 } else { f32::INFINITY };
        let e = rec.worst.entry(format!("mass.{name}")).or_insert(0.0);
        *e = e.max(ratio);
        if !(err <= bar) {
            rec.misses.push(format!("{ctx} mass {name}: dev {} vs host {} (err/bar {ratio:e}, bar {bar:e})", dr.mass[j], ro.mass[j]));
        }
    }
}

/// Run one case cell on the host and check every substep of every hour.
fn synced_cell(case: &Case, theta: f32, substeps: u32, rec: &mut Rec) {
    let dc: <Cpu as BackendTypes>::Device = Default::default();
    let mut block = make_block::<Cpu>(case, theta, substeps, false, &dc);
    let c = block.op_consts(&case.pattern);
    let mut h = to_vec1(block.stage().unwrap());
    let like = cube_from_host(&[0.0], &Default::default());
    let cd = Arc::new(StageOpConstsDev::upload(&c, &case.pattern, &like));
    let up = |v: &[f32]| -> Cube { cube_from_host(v, &Default::default()) };
    let mut qt = case.q_t0.clone();
    for k in 0..case.hours {
        let (qp, mc) = (pick(&case.qp, k), pick(&case.mc, k));
        let (b0, b1) = if case.prescribed {
            (case.series_base + case.series_rate * k as f32, case.series_base + case.series_rate * (k + 1) as f32)
        } else {
            (0.0, 0.0)
        };
        let hour = stage_prologue_host(&c, &case.n, &case.q, &case.p, &qt, &mc, &qp, b0, b1);
        let fwd = stage_hour_forward(&c, &case.pattern, &hour, &h, &qt, true);
        let hd = HourDev::upload(&hour, &like);
        let nsub = fwd.subs.len();
        for (s, sv) in fwd.subs.iter().enumerate() {
            let ctx = format!("{} theta {theta} substeps {substeps} hour {k} substep {s}:", case.label);
            let asm = stage_assemble_host(&c, &hour, s, &sv.h, &sv.qc);
            let ro = stage_readout_host(&c, &hour, &sv.h, &sv.qc, &asm, &sv.dh);
            // assemble from the host start state
            let da = assembled_to_host(&stage_assemble_dev(&cd, &hd, s, &up(&sv.h), &up(&sv.qc)));
            // Isolation: the rhs kernel alone on the host's own inputs.
            let r_iso = cube_to_host(&rhs_dev(
                &cd, &up(&sv.qc), &up(&asm.newton.q_star), &up(&asm.control.r_t), &up(&asm.newton.b_down),
                &up(&hour.qp_stage), &up(&asm.i_mid), asm.hb0, asm.hb1,
            ));
            // factor and solve on the host a_values and r
            let f = tree_factor_dev(&cd.tree, &up(&asm.a_values));
            let ddh = cube_to_host(&tree_solve_dev(&case.pattern, &f, &up(&asm.r)));
            // readout with the HOST dh and the host Assembled
            let asm_dev = AssembledDev::upload(&asm, &like, c.nnz);
            let dr = readout_to_host(&stage_readout_dev(&cd, &hd, &up(&sv.h), &up(&sv.qc), &asm_dev, &up(&sv.dh)));
            compare_substep(rec, &ctx, &c, &hour, &sv.qc, &asm, &ro, &sv.dh, &sv.dh, &da, &ddh, &dr, Some(&r_iso));
            if s + 1 == nsub {
                let out = cube_to_host(&stage_epilogue_dev(&cd, &hd, &up(&ro.h_new), &up(&ro.q_new)));
                let mut want = ro.h_new.clone();
                want.extend(stage_epilogue_host(&c, &hour, &ro.q_new));
                rec.close(&ctx, "epilogue out", &out, &want);
            }
        }
        h = fwd.h_end;
        qt = fwd.q_out;
    }
}

fn synced_family(name: &str, cases: &[Case]) -> usize {
    let mut rec = Rec { worst: BTreeMap::new(), bits: BTreeMap::new(), misses: vec![] };
    for c in cases {
        for theta in [0.55f32, 0.6, 1.0] {
            for substeps in [1u32, 4] {
                synced_cell(c, theta, substeps, &mut rec);
            }
        }
    }
    print_rec(name, &rec);
    finish_rec(name, &rec)
}

/// Print every miss of a family; returns their number.
fn finish_rec(name: &str, rec: &Rec) -> usize {
    for m in &rec.misses {
        println!("SYNCMISS {name}: {m}");
    }
    println!("SYNCED {name}: {} misses", rec.misses.len());
    rec.misses.len()
}

fn print_rec(name: &str, rec: &Rec) {
    let (f, v) = rec.worst.iter().fold(("", 0.0f32), |m, (k, v)| if *v >= m.1 { (k.as_str(), *v) } else { m });
    println!("SYNCED {name}: worst err/bar {v:e} ({f}) over {} fields", rec.worst.len());
    for (k, v) in &rec.worst {
        let (nb, tot) = rec.bits.get(k).copied().unwrap_or((0, 0));
        println!("SYNCED {name} field {k}: {v:.3e} bitwise {} ({nb} of {tot} entries differ)", if tot > 0 && nb == 0 { "yes" } else if tot == 0 { "n/a" } else { "no" });
    }
}

#[test]
fn forward_substeps_match_host_from_synced_state() {
    let device = TestDevice::default();
    let mut misses = 0usize;
    misses += synced_family("chain_states", &chain_state_cases());
    misses += synced_family("table_and_mixed", &table_and_mixed_cases());
    misses += synced_family("prescribed_boundary", &prescribed_boundary_cases());
    misses += synced_family("mc_inflow", &[mc_inflow_case(&device)]);
    // Juniata and Hudson: 6 hours, (1.0, 1) and (0.6, 4).
    let mut rec = Rec { worst: BTreeMap::new(), bits: BTreeMap::new(), misses: vec![] };
    let mut jc = juniata_case(&device);
    jc.hours = 6;
    for (th, sub) in [(1.0f32, 1u32), (0.6, 4)] {
        synced_cell(&jc, th, sub, &mut rec);
    }
    print_rec("juniata_6h", &rec);
    misses += finish_rec("juniata_6h", &rec);
    if cfg!(debug_assertions) {
        eprintln!("hudson part skipped in debug builds: run with --release");
        assert_eq!(misses, 0, "{misses} synced-state misses (see SYNCMISS lines)");
        return;
    }
    let mut rec = Rec { worst: BTreeMap::new(), bits: BTreeMap::new(), misses: vec![] };
    let mut hc = hudson_net_case(&device);
    hc.hours = 6;
    synced_cell(&hc, 0.6, 4, &mut rec);
    print_rec("hudson_net_6h", &rec);
    misses += finish_rec("hudson_net_6h", &rec);
    assert_eq!(misses, 0, "{misses} synced-state misses (see SYNCMISS lines)");
}


// ------------------------------------------ round 5: reverse check and branch diagnostic
//
// Diagnostic only; no bar is changed here. `gpu_traj` runs the device hour (prologue, hour forward)
// from the case's initial state, which is what `step_fused` runs on CUDA, and keeps every substep's
// start state and its own block outputs.

struct GpuSub {
    h: Vec<f32>,
    qc: Vec<f32>,
    asm: Assembled,
    dh: Vec<f32>,
    ro: Readout,
}

struct GpuHour {
    hour: HourHost,
    subs: Vec<GpuSub>,
    /// `[h_end, q_next]` as `step_fused` outputs them.
    out: Vec<f32>,
}

fn hour_b(case: &Case, k: usize) -> (f32, f32) {
    if case.prescribed {
        (case.series_base + case.series_rate * k as f32, case.series_base + case.series_rate * (k + 1) as f32)
    } else {
        (0.0, 0.0)
    }
}

fn gpu_traj(case: &Case, theta: f32, substeps: u32) -> (Arc<StageOpConsts>, Vec<GpuHour>) {
    let dc: <Cpu as BackendTypes>::Device = Default::default();
    let mut block = make_block::<Cpu>(case, theta, substeps, false, &dc);
    let c = block.op_consts(&case.pattern);
    let h0 = to_vec1(block.stage().unwrap());
    let like = cube_from_host(&[0.0], &Default::default());
    let cd = Arc::new(StageOpConstsDev::upload(&c, &case.pattern, &like));
    let up = |v: &[f32]| -> Cube { cube_from_host(v, &Default::default()) };
    let (nn, pp, qq) = (up(&case.n), up(&case.p), up(&case.q));
    let (mut h, mut qt) = (up(&h0), up(&case.q_t0));
    let n = c.n;
    let mut hours = vec![];
    for k in 0..case.hours {
        let (qp, mc) = (up(&pick(&case.qp, k)), up(&pick(&case.mc, k)));
        let (b0, b1) = hour_b(case, k);
        let hd = stage_prologue_dev(&cd, &nn, &qq, &pp, &qt, &mc, &qp, b0, b1);
        let fwd = stage_hour_forward_dev(&cd, &case.pattern, &hd, &h, &qt, true);
        let mut subs = vec![];
        for (s, sv) in fwd.subs.iter().enumerate() {
            let asm = stage_assemble_dev(&cd, &hd, s, &sv.h, &sv.qc);
            let ro = readout_to_host(&stage_readout_dev(&cd, &hd, &sv.h, &sv.qc, &asm, &sv.dh));
            subs.push(GpuSub {
                h: cube_to_host(&sv.h),
                qc: cube_to_host(&sv.qc),
                asm: assembled_to_host(&asm),
                dh: cube_to_host(&sv.dh),
                ro,
            });
        }
        let out = cube_to_host(&fwd.out);
        h = fwd.h_end.clone();
        qt = up(&out[n..]);
        hours.push(GpuHour { hour: hour_to_host(&hd), subs, out });
    }
    (c, hours)
}

/// Reverse check of one cell: along the GPU trajectory, run the host substep from the exact GPU
/// values and compare with the GPU substep's own outputs, under the same B4 bars.
fn reverse_cell(case: &Case, theta: f32, substeps: u32, rec: &mut Rec) {
    let (c, hours) = gpu_traj(case, theta, substeps);
    for (k, gh) in hours.iter().enumerate() {
        for (s, g) in gh.subs.iter().enumerate() {
            let ctx = format!("REV {} theta {theta} substeps {substeps} hour {k} substep {s}:", case.label);
            let asm_h = stage_assemble_host(&c, &gh.hour, s, &g.h, &g.qc);
            // mirror of the forward direction: the host solve gets the GPU's own a_values and r
            let f = tree_lu_factor(&case.pattern, &g.asm.a_values);
            let dh_ref = tree_lu_solve_host(&case.pattern, &f, &g.asm.r);
            // same inputs: the host solve on the GPU's own a_values and r, against the GPU dh
            let f2 = tree_lu_factor(&case.pattern, &g.asm.a_values);
            let dh_same = tree_lu_solve_host(&case.pattern, &f2, &g.asm.r);
            rec.close(&ctx, "dh (host solve on the GPU a_values, r)", &g.dh, &dh_same);
            // the GPU dh feeds the host readout
            let ro_h = stage_readout_host(&c, &gh.hour, &g.h, &g.qc, &asm_h, &g.dh);
            compare_substep(rec, &ctx, &c, &gh.hour, &g.qc, &asm_h, &ro_h, &dh_ref, &g.dh, &g.asm, &g.dh, &g.ro, None);
        }
    }
}

/// Per reach: s_up, s_dn, act (2 without a table), a_up == 0, b_down == 0, r_prime == 0, sign(qc) (0 neg, 1 zero, 2 pos).
type Pat = Vec<[u8; 7]>;
const PAT_NAMES: [&str; 7] = ["s_up", "s_dn", "act", "a_up==0", "b_down==0", "r_prime==0", "sign(qc)"];

fn pattern(asm: &Assembled, qc: &[f32]) -> Pat {
    (0..qc.len())
        .map(|i| {
            [
                (asm.depth.s_up[i] > 0.5) as u8,
                (asm.depth.s_dn[i] > 0.5) as u8,
                if asm.control.act.len() < qc.len() { 2 } else { (asm.control.act[i] > 0.5) as u8 },
                (asm.newton.a_up[i] == 0.0) as u8,
                (asm.newton.b_down[i] == 0.0) as u8,
                (asm.control.r_prime[i] == 0.0) as u8,
                if qc[i] < 0.0 { 0 } else if qc[i] == 0.0 { 1 } else { 2 },
            ]
        })
        .collect()
}

/// Host-trajectory patterns, `[hour][substep]`.
fn cpu_patterns(case: &Case, theta: f32, substeps: u32) -> Vec<Vec<Pat>> {
    let dc: <Cpu as BackendTypes>::Device = Default::default();
    let mut block = make_block::<Cpu>(case, theta, substeps, false, &dc);
    let c = block.op_consts(&case.pattern);
    let mut h = to_vec1(block.stage().unwrap());
    let mut qt = case.q_t0.clone();
    let mut out = vec![];
    for k in 0..case.hours {
        let (qp, mc) = (pick(&case.qp, k), pick(&case.mc, k));
        let (b0, b1) = hour_b(case, k);
        let hour = stage_prologue_host(&c, &case.n, &case.q, &case.p, &qt, &mc, &qp, b0, b1);
        let fwd = stage_hour_forward(&c, &case.pattern, &hour, &h, &qt, true);
        out.push(
            fwd.subs
                .iter()
                .enumerate()
                .map(|(s, sv)| pattern(&stage_assemble_host(&c, &hour, s, &sv.h, &sv.qc), &sv.qc))
                .collect(),
        );
        h = fwd.h_end;
        qt = fwd.q_out;
    }
    out
}

/// Hours at which any gated field, the stage or a mass term is over the sanity bar (2e-3 x max|cpu|, or x gross).
fn sanity_miss_hours(cb: &[HourRec], gb: &[HourRec]) -> Vec<usize> {
    let mut hours = vec![];
    for k in 0..cb.len() {
        let over = |d: &[f32], c: &[f32]| {
            let scale = c.iter().fold(0.0f32, |m, v| m.max(v.abs()));
            max_abs_diff(d, c) > STIFF_SANITY_FRAC * scale
        };
        let mut miss = over(&gb[k].stage, &cb[k].stage);
        for fi in 0..ELIGIBLE.len() {
            miss |= over(&gb[k].fields[fi], &cb[k].fields[fi]);
        }
        for j in 0..5 {
            miss |= (gb[k].mass[j] - cb[k].mass[j]).abs() > STIFF_SANITY_FRAC * cb[k].mass[3].abs();
        }
        if miss {
            hours.push(k);
        }
    }
    hours
}

#[test]
fn round5_reverse_and_branch_diagnostic() {
    let mut misses_total = 0usize;
    // 1. Reverse synced check, per family.
    let families: Vec<(&str, Vec<Case>)> = vec![
        ("chain_states", chain_state_cases()),
        ("table_and_mixed", table_and_mixed_cases()),
        ("prescribed_boundary", prescribed_boundary_cases()),
        ("mc_inflow", vec![mc_inflow_case(&TestDevice::default())]),
    ];
    for (name, cases) in &families {
        let mut rec = Rec { worst: BTreeMap::new(), bits: BTreeMap::new(), misses: vec![] };
        for c in cases {
            for theta in [0.55f32, 0.6, 1.0] {
                for substeps in [1u32, 4] {
                    reverse_cell(c, theta, substeps, &mut rec);
                }
            }
        }
        print_rec(&format!("REVERSE {name}"), &rec);
        misses_total += finish_rec(&format!("REVERSE {name}"), &rec);
    }
    {
        let df = chain_state_cases().into_iter().find(|c| c.label == "depth_floor").unwrap();
        for substeps in [1u32, 4] {
            let mut rec = Rec { worst: BTreeMap::new(), bits: BTreeMap::new(), misses: vec![] };
            reverse_cell(&df, 0.6, substeps, &mut rec);
            print_rec(&format!("REVERSE depth_floor th0.6 s{substeps}"), &rec);
            finish_rec(&format!("REVERSE depth_floor th0.6 s{substeps}"), &rec);
        }
    }
    println!("REVERSE total misses {misses_total}");

    // 2. Branch patterns, per cell of the three stiff families.
    let (dc, dg): (<Cpu as BackendTypes>::Device, <Gpu as BackendTypes>::Device) = (Default::default(), Default::default());
    let (mut cells, mut flipped, mut missed, mut before_flip) = (0, 0, 0, 0);
    let mut direct_mismatch = 0usize;
    for (name, cases) in families.iter().take(3) {
        for case in cases {
            for theta in [0.55f32, 0.6, 1.0] {
                for substeps in [1u32, 4] {
                    cells += 1;
                    let cp = cpu_patterns(case, theta, substeps);
                    let (_, gh) = gpu_traj(case, theta, substeps);
                    let cb = run_series::<Cpu>(case, theta, substeps, false, false, &dc);
                    let gb = run_series::<Gpu>(case, theta, substeps, true, false, &dg);
                    // the direct device run must equal step_fused
                    for k in 0..case.hours {
                        if gh[k].out[case.adj.n..] != gb[k].fields[0][..] {
                            direct_mismatch += 1;
                        }
                    }
                    let mut flip: Option<(usize, usize)> = None;
                    let mut flip_txt = String::from("none");
                    'outer: for k in 0..case.hours {
                        for s in 0..cp[k].len() {
                            let gp = pattern(&gh[k].subs[s].asm, &gh[k].subs[s].qc);
                            if gp != cp[k][s] {
                                let mut parts = vec![];
                                for (i, (a, b)) in cp[k][s].iter().zip(&gp).enumerate() {
                                    if a != b {
                                        let f: Vec<String> = (0..7)
                                            .filter(|&j| a[j] != b[j])
                                            .map(|j| format!("{}: cpu {} gpu {}", PAT_NAMES[j], a[j], b[j]))
                                            .collect();
                                        parts.push(format!("reach {i} [{}]", f.join(", ")));
                                    }
                                }
                                flip = Some((k, s));
                                flip_txt = format!("hour {k} substep {s}: {}", parts.join("; "));
                                break 'outer;
                            }
                        }
                    }
                    let miss = sanity_miss_hours(&cb, &gb);
                    if flip.is_some() {
                        flipped += 1;
                    }
                    if !miss.is_empty() {
                        missed += 1;
                    }
                    let early = match (&flip, miss.first()) {
                        (Some((fk, _)), Some(m)) => m < fk,
                        (None, Some(_)) => true,
                        _ => false,
                    };
                    if early {
                        before_flip += 1;
                    }
                    println!(
                        "BRANCH {name} {} theta {theta} substeps {substeps}: first flip {flip_txt}; sanity-miss hours {miss:?}{}",
                        case.label,
                        if early { "  <== MISS BEFORE FLIP" } else { "" }
                    );
                }
            }
        }
    }
    println!(
        "BRANCH summary: cells {cells}, cells with a flip {flipped}, cells with a sanity miss {missed}, cells with a miss before the first flip (or without a flip) {before_flip}, direct-vs-step_fused q_next mismatches {direct_mismatch}"
    );

    // 3. Onset table: depth_floor theta 0.6, per hour worst err/max|cpu| per field.
    let df = chain_state_cases().into_iter().find(|c| c.label == "depth_floor").unwrap();
    for substeps in [1u32, 4] {
        let cb = run_series::<Cpu>(&df, 0.6, substeps, false, false, &dc);
        let gb = run_series::<Gpu>(&df, 0.6, substeps, true, false, &dg);
        for k in 0..df.hours {
            let rel = |d: &[f32], c: &[f32]| {
                let scale = c.iter().fold(0.0f32, |m, v| m.max(v.abs()));
                if scale > 0.0 { max_abs_diff(d, c) / scale } else { max_abs_diff(d, c) }
            };
            let mut row = vec![format!("stage {:.2e}", rel(&gb[k].stage, &cb[k].stage))];
            for (fi, name) in ELIGIBLE.iter().enumerate() {
                row.push(format!("{name} {:.2e}", rel(&gb[k].fields[fi], &cb[k].fields[fi])));
            }
            let g = cb[k].mass[3].abs();
            row.push(format!(
                "mass(max over 5, /gross) {:.2e}",
                (0..5).map(|j| (gb[k].mass[j] - cb[k].mass[j]).abs() / g).fold(0.0f32, f32::max)
            ));
            println!("ONSET depth_floor th0.6 s{substeps} hour {k}: {}", row.join(", "));
        }
    }
}


// ----------------------------------------------- fma ruling F4.3: the cuSPARSE solve against two host solves
//
// (a) the shared host solve (`forward_sub_lower`, `back_sub_upper_transposed`: unfused) and (b) a fused
// host solve written here (rule F1 on the two triangular loops: `sum = fma(a, x, sum)`). The device solve
// runs on the host factors, so only the solve differs.

// the loops of src/sparse/mod.rs::forward_sub_lower with rule F1 applied
fn fused_forward_sub(p: &CsrPattern, l: &[f32], b: &[f32]) -> Vec<f32> {
    let mut x = vec![0.0f32; p.n];
    for i in 0..p.n {
        let (mut sum, mut diag) = (0.0f32, 0.0f32);
        for k in p.crow[i] as usize..p.crow[i + 1] as usize {
            let j = p.col[k] as usize;
            if j == i {
                diag = l[k];
            } else {
                sum = l[k].mul_add(x[j], sum);
            }
        }
        x[i] = (b[i] - sum) / diag;
    }
    x
}

// the loops of src/sparse/mod.rs::back_sub_upper_transposed with rule F1 applied
fn fused_back_sub(p: &CsrPattern, u: &[f32], b: &[f32]) -> Vec<f32> {
    let mut y = vec![0.0f32; p.n];
    for i in (0..p.n).rev() {
        let (mut sum, mut diag) = (0.0f32, 0.0f32);
        for k in p.trans_crow[i] as usize..p.trans_crow[i + 1] as usize {
            let j = p.trans_col[k] as usize;
            let val = u[p.trans_to_orig[k] as usize];
            if j == i {
                diag = val;
            } else {
                sum = val.mul_add(y[j], sum);
            }
        }
        y[i] = (b[i] - sum) / diag;
    }
    y
}

// variant (c): start from b and subtract with fused multiply-adds, `t = fma(-l, x, t)`, then divide
fn sub_from_b_forward(p: &CsrPattern, l: &[f32], b: &[f32], fused: bool, recip: bool) -> Vec<f32> {
    let mut x = vec![0.0f32; p.n];
    for i in 0..p.n {
        let (mut t, mut diag) = (b[i], 0.0f32);
        for k in p.crow[i] as usize..p.crow[i + 1] as usize {
            let j = p.col[k] as usize;
            if j == i {
                diag = l[k];
            } else {
                t = if fused { (-l[k]).mul_add(x[j], t) } else { t - l[k] * x[j] };
            }
        }
        x[i] = if recip { t * (1.0 / diag) } else { t / diag };
    }
    x
}

fn sub_from_b_back(p: &CsrPattern, u: &[f32], b: &[f32], fused: bool, recip: bool) -> Vec<f32> {
    let mut y = vec![0.0f32; p.n];
    for i in (0..p.n).rev() {
        let (mut t, mut diag) = (b[i], 0.0f32);
        for k in p.trans_crow[i] as usize..p.trans_crow[i + 1] as usize {
            let j = p.trans_col[k] as usize;
            let val = u[p.trans_to_orig[k] as usize];
            if j == i {
                diag = val;
            } else {
                t = if fused { (-val).mul_add(y[j], t) } else { t - val * y[j] };
            }
        }
        y[i] = if recip { t * (1.0 / diag) } else { t / diag };
    }
    y
}

/// Bitwise-equal counts: `[chain rows, confluence rows]` as (equal, total).
#[derive(Default, Clone, Copy)]
struct Cnt {
    dev_vs_a: [usize; 2],
    dev_vs_b: [usize; 2],
    dev_vs_c: [usize; 2],
    dev_vs_d: [usize; 2],
    dev_vs_e: [usize; 2],
    dev_vs_f: [usize; 2],
    a_vs_b: [usize; 2],
    total: [usize; 2],
    factor_eq: usize,
    factor_total: usize,
}

fn cusparse_cell(case: &Case, theta: f32, substeps: u32, cnt: &mut Cnt) {
    let dc: <Cpu as BackendTypes>::Device = Default::default();
    let mut block = make_block::<Cpu>(case, theta, substeps, false, &dc);
    let c = block.op_consts(&case.pattern);
    let mut h = to_vec1(block.stage().unwrap());
    let like = cube_from_host(&[0.0], &Default::default());
    let cd = Arc::new(StageOpConstsDev::upload(&c, &case.pattern, &like));
    let up = |v: &[f32]| -> Cube { cube_from_host(v, &Default::default()) };
    let p = &case.pattern;
    let class: Vec<usize> = (0..p.n).map(|i| ((p.crow[i + 1] - p.crow[i]) as usize >= 3) as usize).collect();
    let mut qt = case.q_t0.clone();
    for k in 0..case.hours {
        let (qp, mc) = (pick(&case.qp, k), pick(&case.mc, k));
        let (b0, b1) = hour_b(case, k);
        let hour = stage_prologue_host(&c, &case.n, &case.q, &case.p, &qt, &mc, &qp, b0, b1);
        let fwd = stage_hour_forward(&c, p, &hour, &h, &qt, true);
        for (s, sv) in fwd.subs.iter().enumerate() {
            let asm = stage_assemble_host(&c, &hour, s, &sv.h, &sv.qc);
            let f = tree_lu_factor(p, &asm.a_values);
            let fdev = tree_factor_dev(&cd.tree, &up(&asm.a_values));
            for (d, hh) in [(&fdev.l_values, &f.l_values), (&fdev.u_values, &f.u_values)] {
                let dv = cube_to_host(d);
                cnt.factor_total += dv.len();
                cnt.factor_eq += dv.iter().zip(hh).filter(|(a, b)| a.to_bits() == b.to_bits()).count();
            }
            let fd = TreeFactorsDev { l_values: up(&f.l_values), u_values: up(&f.u_values) };
            let dev = cube_to_host(&tree_solve_dev(p, &fd, &up(&asm.r)));
            let a = tree_lu_solve_host(p, &f, &asm.r);
            let b = fused_back_sub(p, &f.u_values, &fused_forward_sub(p, &f.l_values, &asm.r));
            let cc = sub_from_b_back(p, &f.u_values, &sub_from_b_forward(p, &f.l_values, &asm.r, true, false), true, false);
            let dd = sub_from_b_back(p, &f.u_values, &sub_from_b_forward(p, &f.l_values, &asm.r, false, true), false, true);
            let ee = sub_from_b_back(p, &f.u_values, &sub_from_b_forward(p, &f.l_values, &asm.r, true, true), true, true);
            let ff = sub_from_b_back(p, &f.u_values, &sub_from_b_forward(p, &f.l_values, &asm.r, false, false), true, false);
            for i in 0..p.n {
                let cl = class[i];
                cnt.total[cl] += 1;
                cnt.dev_vs_a[cl] += (dev[i].to_bits() == a[i].to_bits()) as usize;
                cnt.dev_vs_b[cl] += (dev[i].to_bits() == b[i].to_bits()) as usize;
                cnt.dev_vs_c[cl] += (dev[i].to_bits() == cc[i].to_bits()) as usize;
                cnt.dev_vs_d[cl] += (dev[i].to_bits() == dd[i].to_bits()) as usize;
                cnt.dev_vs_e[cl] += (dev[i].to_bits() == ee[i].to_bits()) as usize;
                cnt.dev_vs_f[cl] += (dev[i].to_bits() == ff[i].to_bits()) as usize;
                cnt.a_vs_b[cl] += (a[i].to_bits() == b[i].to_bits()) as usize;
            }
        }
        h = fwd.h_end;
        qt = fwd.q_out;
    }
}

fn print_cnt(name: &str, c: &Cnt) {
    println!(
        "CUSPARSE-VARIANTS {name}: chain d(unfused,recip) {}/{} e(fused,recip) {}/{} f(L unfused,U fused) {}/{}; confluence d {}/{} e {}/{} f {}/{}",
        c.dev_vs_d[0], c.total[0], c.dev_vs_e[0], c.total[0], c.dev_vs_f[0], c.total[0],
        c.dev_vs_d[1], c.total[1], c.dev_vs_e[1], c.total[1], c.dev_vs_f[1], c.total[1]
    );
    println!(
        "CUSPARSE {name}: factor bitwise {}/{}; chain rows: dev==a {}/{}, dev==b(fused) {}/{}, dev==c(sub-from-b fma) {}/{}, a==b {}/{}; confluence rows: dev==a {}/{}, dev==b {}/{}, dev==c {}/{}, a==b {}/{}",
        c.factor_eq, c.factor_total,
        c.dev_vs_a[0], c.total[0], c.dev_vs_b[0], c.total[0], c.dev_vs_c[0], c.total[0], c.a_vs_b[0], c.total[0],
        c.dev_vs_a[1], c.total[1], c.dev_vs_b[1], c.total[1], c.dev_vs_c[1], c.total[1], c.a_vs_b[1], c.total[1],
    );
}

#[test]
fn cusparse_solve_vs_host_solves() {
    let device = TestDevice::default();
    let fam: Vec<(&str, Vec<Case>)> = vec![
        ("chain_states", chain_state_cases()),
        ("table_and_mixed", table_and_mixed_cases()),
        ("prescribed_boundary", prescribed_boundary_cases()),
        ("mc_inflow", vec![mc_inflow_case(&device)]),
    ];
    for (name, cases) in &fam {
        let mut cnt = Cnt::default();
        for c in cases {
            for theta in [0.55f32, 0.6, 1.0] {
                for substeps in [1u32, 4] {
                    cusparse_cell(c, theta, substeps, &mut cnt);
                }
            }
        }
        print_cnt(name, &cnt);
    }
    let mut cnt = Cnt::default();
    let mut jc = juniata_case(&device);
    jc.hours = 6;
    for (th, sub) in [(1.0f32, 1u32), (0.6, 4)] {
        cusparse_cell(&jc, th, sub, &mut cnt);
    }
    print_cnt("juniata_6h", &cnt);
    if !cfg!(debug_assertions) {
        let mut cnt = Cnt::default();
        let mut hc = hudson_net_case(&device);
        hc.hours = 6;
        cusparse_cell(&hc, 0.6, 4, &mut cnt);
        print_cnt("hudson_net_6h", &cnt);
    }
}


// fma-ruling.md R1: the device solve (cuSPARSE) and `tree_lu_solve_host` must agree bitwise on the same
// factors, so a cuSPARSE change shows up here. Juniata (chains and confluences) and chain_states.
fn dh_bitwise_cell(case: &Case, theta: f32, substeps: u32, equal: &mut usize, total: &mut usize) {
    let dc: <Cpu as BackendTypes>::Device = Default::default();
    let mut block = make_block::<Cpu>(case, theta, substeps, false, &dc);
    let c = block.op_consts(&case.pattern);
    let mut h = to_vec1(block.stage().unwrap());
    let up = |v: &[f32]| -> Cube { cube_from_host(v, &Default::default()) };
    let mut qt = case.q_t0.clone();
    for k in 0..case.hours {
        let (qp, mc) = (pick(&case.qp, k), pick(&case.mc, k));
        let (b0, b1) = hour_b(case, k);
        let hour = stage_prologue_host(&c, &case.n, &case.q, &case.p, &qt, &mc, &qp, b0, b1);
        let fwd = stage_hour_forward(&c, &case.pattern, &hour, &h, &qt, true);
        for (s, sv) in fwd.subs.iter().enumerate() {
            let asm = stage_assemble_host(&c, &hour, s, &sv.h, &sv.qc);
            let f = tree_lu_factor(&case.pattern, &asm.a_values);
            let fd = TreeFactorsDev { l_values: up(&f.l_values), u_values: up(&f.u_values) };
            let dev = cube_to_host(&tree_solve_dev(&case.pattern, &fd, &up(&asm.r)));
            let host = tree_lu_solve_host(&case.pattern, &f, &asm.r);
            for (i, (a, b)) in dev.iter().zip(&host).enumerate() {
                *total += 1;
                if a.to_bits() == b.to_bits() {
                    *equal += 1;
                } else {
                    println!("DHDIFF {} th{theta} s{substeps} hour {k} substep {s} reach {i}: dev {a} host {b}", case.label);
                }
            }
        }
        h = fwd.h_end;
        qt = fwd.q_out;
    }
}

#[test]
fn dh_is_bitwise_equal_to_the_host_solve() {
    let device = TestDevice::default();
    let (mut eq, mut tot) = (0usize, 0usize);
    let mut jc = juniata_case(&device);
    jc.hours = 6;
    for (th, sub) in [(1.0f32, 1u32), (0.6, 4)] {
        dh_bitwise_cell(&jc, th, sub, &mut eq, &mut tot);
    }
    for c in chain_state_cases() {
        for theta in [0.55f32, 0.6, 1.0] {
            for substeps in [1u32, 4] {
                dh_bitwise_cell(&c, theta, substeps, &mut eq, &mut tot);
            }
        }
    }
    println!("DH bitwise: {eq} of {tot} entries equal");
    assert_eq!(eq, tot, "tree_solve_dev and tree_lu_solve_host differ in {} entries", tot - eq);
}

//! The whole engine (spatial parameters, Muskingum-Cunge rows, stage block, loss) on CUDA against the CPU.
//!
//! Each test routes a 5-day window of a committed example bundle twice. The CPU run uses NdArray with
//! `sparse_solver: cpu`. The CUDA run uses `Cuda<f32, i32>` with `sparse_solver: cuda`, which turns on the CUDA arms
//! of the stage op, the tree solve and the Muskingum-Cunge solve. The spatial parameters `n`, `q_spatial` and
//! `p_spatial` are `require_grad` leaves, so the gradients are those the KAN head would receive.
//!
//! Gates:
//! - Forward: routed q of every reach, per hour, within `1e-4 * max|cpu|` of that hour. The five mass sums within
//!   `1e-5 * gross` and the step counts equal. No acceptance budget.
//! - Gradients (48 h bar of the stage op parity tests): per entry `1e-3 * max(|g|, |c|)` or
//!   `1e-4 * max(1, max|c|)`. A miss is accepted only if `err <= 4 * (noise_cpu + noise_cuda)` and
//!   `err <= 5e-4 * max|g|` of its parameter, with at most 1 % of the compared entries accepted. Every accepted
//!   entry is printed. `noise_*` is the linearity noise of the loss on that backend, `max_i |gA + gB - gboth|`,
//!   where A weights the first half of the output hours and B the second half (the 5-day window gives 96 output hours,
//!   so 48 each).
//! - Structural zeros: the gradients of `n`, `q_spatial` and `p_spatial` on table rows are exactly 0 on both
//!   backends. Off that set, entries that are exactly 0 on one backend only are counted and printed.
//!
//! Run: `cargo test --features cuda --test stage_engine_cuda_parity -- --nocapture` (debug) and the `--release` form.
#![cfg(feature = "cuda")]

use std::time::{Duration, Instant};

use burn::backend::{Autodiff, NdArray};
use burn::tensor::backend::{Backend, BackendTypes};
use burn::tensor::Tensor;

use ddrs::config::{Config, SparseSolver};
use ddrs::data::{MeritGagesDataset, RhoWindow, Staid};
use ddrs::routing::stage::StageMassBalance;
use ddrs::routing::{MuskingumCunge, RoutingInputs, SpatialParameters};
use ddrs::training::forward::apply_stage_block;

type Cpu = NdArray<f32>;
type Gpu = burn_cuda::Cuda<f32, i32>;

const JUNIATA_CONFIG: &str = "examples/juniata/ddrs_stage.yaml";
const HUDSON_CONFIG: &str = "examples/hudson/ddrs.yaml";
const DAYS: usize = 5;
const NOISE_FACTOR: f32 = 4.0;
const NOISE_CAP_FRAC: f32 = 5e-4;
const BUDGET_FRAC: f64 = 0.01;
const NAMES: [&str; 3] = ["n", "q_spatial", "p_spatial"];

// copied from tests/stage_op_cuda_parity.rs::Rng
struct Rng(u64);
impl Rng {
    fn unit(&mut self) -> f64 {
        self.0 = self.0.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        ((z ^ (z >> 31)) >> 11) as f64 / (1u64 << 53) as f64
    }
    fn range(&mut self, lo: f64, hi: f64) -> f64 {
        lo + (hi - lo) * self.unit()
    }
}

// copied from tests/stage_op_cuda_parity.rs::vec_b
fn vec_b(n: usize, seed: u64, lo: f64, hi: f64) -> Vec<f32> {
    let mut rng = Rng(seed);
    (0..n).map(|_| rng.range(lo, hi) as f32).collect()
}

/// Which hours of the loss carry weight.
#[derive(Clone, Copy, PartialEq)]
enum Hours {
    All,
    /// The first half of the hours only.
    First,
    /// The second half of the hours only.
    Last,
}

/// What one engine run returns.
struct EngineRun {
    n: usize,
    t: usize,
    /// Routed q, `[reach][hour]` row-major.
    q: Vec<f32>,
    loss: f32,
    /// Gradients of `n`, `q_spatial`, `p_spatial`.
    grads: [Vec<f32>; 3],
    mass: Option<StageMassBalance>,
    /// Table geometry rows (empty without a table).
    table_rows: Vec<usize>,
    wall: Duration,
}

/// Route one window on backend `B` and take the gradient of `sum(w * q)` with respect to the three spatial
/// parameters.
// copied from tests/hudson_stage.rs::route (engine setup, stage block arming), generic in the inner backend,
// with the parameters as require_grad leaves and a weighted-sum loss
fn run<B: Backend + 'static>(
    config: &str,
    start: (i32, u32, u32),
    use_cuda: bool,
    hours: Hours,
    device: &B::Device,
) -> EngineRun
where
    B::FloatTensorPrimitive: 'static,
    B::Device: 'static,
{
    type Ad<B> = Autodiff<B>;
    let mut cfg = Config::from_yaml_file(config).expect("example config");
    assert!(!cfg.params.enforce_positivity, "the parity gate does not cover enforce_positivity");
    cfg.params.sparse_solver = if use_cuda { SparseSolver::Cuda } else { SparseSolver::Cpu };
    let ds = MeritGagesDataset::open(&cfg).expect("open dataset");
    let axis = ds.time_axis();
    let start = chrono::NaiveDate::from_ymd_opt(start.0, start.1, start.2).unwrap();
    let window = RhoWindow {
        start_day_idx: (start - axis.start).num_days() as usize,
        rho_days: DAYS,
        window_start: start,
    };
    let staids: Vec<Staid> = if config == HUDSON_CONFIG {
        // copied from tests/hudson_stage.rs::GAUGES
        ["01375000", "01358000", "01372058", "73005067"].iter().map(|s| Staid::from(*s)).collect()
    } else {
        vec![Staid::from("01567000")]
    };
    let batch = ds.collate(&staids, &window).expect("collate");
    let tensors = batch.to_tensors::<B>(device);
    let n = tensors.adjacency.n;
    let table_rows: Vec<usize> = tensors
        .stage
        .as_ref()
        .and_then(|s| s.geometry.as_ref())
        .map(|g| g.rows.clone())
        .unwrap_or_default();

    // Distinct values per reach, inside the range the head produces. Same seeds on both backends.
    let leaf = |seed: u64, lo: f64, hi: f64| {
        Tensor::<Ad<B>, 1>::from_floats(vec_b(n, seed, lo, hi).as_slice(), device).require_grad()
    };
    let (ln, lq, lp) = (leaf(11, 0.05, 0.2), leaf(12, 0.3, 0.7), leaf(13, 0.4, 0.7));
    let mut engine = MuskingumCunge::<B>::new(cfg.clone(), device.clone());
    engine.setup_inputs(
        RoutingInputs {
            adjacency: tensors.adjacency.clone(),
            x_storage: Tensor::<Ad<B>, 1>::full([n], 0.3, device),
        },
        Tensor::from_inner(tensors.q_prime.clone()),
        SpatialParameters {
            n: ln.clone(),
            q_spatial: lq.clone(),
            p_spatial: Some(lp.clone()),
            k_d: None,
            d_gw: None,
            leakance_factor: None,
            impervious_mask: None,
            gamma: None,
        },
        false,
        None,
    );
    apply_stage_block(
        &cfg,
        &mut engine,
        tensors.stage.as_ref(),
        tensors.reservoir_rows.as_ref(),
        tensors.window.window_start,
    );
    let t0 = Instant::now();
    let out = engine.forward();
    let [rows, t] = out.dims();
    assert_eq!(rows, n);
    // Weights in (0.5, 1.5) as in the stage op parity tests, zeroed outside the chosen hours.
    let mut w = vec_b(n * t, 7, 0.5, 1.5);
    for r in 0..n {
        for h in 0..t {
            let keep = match hours {
                Hours::All => true,
                Hours::First => h < t / 2,
                Hours::Last => h >= t / 2,
            };
            if !keep {
                w[r * t + h] = 0.0;
            }
        }
    }
    let wt = Tensor::<Ad<B>, 1>::from_floats(w.as_slice(), device).reshape([n, t]);
    let loss = (out.clone() * wt).sum();
    let g = loss.backward();
    let wall = t0.elapsed();
    let q: Vec<f32> = out.inner().into_data().to_vec().unwrap();
    let loss_v = loss.inner().into_data().to_vec::<f32>().unwrap()[0];
    let get = |x: &Tensor<Ad<B>, 1>, what: &str| -> Vec<f32> {
        x.grad(&g).unwrap_or_else(|| panic!("no gradient for {what}")).into_data().to_vec().unwrap()
    };
    EngineRun {
        n,
        t,
        q,
        loss: loss_v,
        grads: [get(&ln, "n"), get(&lq, "q_spatial"), get(&lp, "p_spatial")],
        mass: engine.stage_mass_balance(),
        table_rows,
        wall,
    }
}

/// Linearity noise of one backend: `max_i |gA + gB - gboth|` per parameter.
fn linearity_noise<B: Backend + 'static>(config: &str, start: (i32, u32, u32), use_cuda: bool, both: &[Vec<f32>; 3], device: &B::Device) -> [f32; 3]
where
    B::FloatTensorPrimitive: 'static,
    B::Device: 'static,
{
    let a = run::<B>(config, start, use_cuda, Hours::First, device);
    let b = run::<B>(config, start, use_cuda, Hours::Last, device);
    let mut out = [0.0f32; 3];
    for pi in 0..3 {
        for i in 0..both[pi].len() {
            out[pi] = out[pi].max((a.grads[pi][i] + b.grads[pi][i] - both[pi][i]).abs());
        }
    }
    out
}

fn mode() -> &'static str {
    if cfg!(debug_assertions) {
        "debug"
    } else {
        "release"
    }
}

fn engine_cpu_vs_cuda(label: &str, config: &str, start: (i32, u32, u32), expect_table_rows: Option<usize>) {
    let (dc, dg): (<Cpu as BackendTypes>::Device, <Gpu as BackendTypes>::Device) = (Default::default(), Default::default());
    let c = run::<Cpu>(config, start, false, Hours::All, &dc);
    let g = run::<Gpu>(config, start, true, Hours::All, &dg);
    println!("{label} ({}): wall forward+backward, cpu {:.2} s, cuda {:.2} s", mode(), c.wall.as_secs_f64(), g.wall.as_secs_f64());
    assert_eq!((c.n, c.t), (g.n, g.t));
    let (n, t) = (c.n, c.t);
    println!("{label}: {n} reaches, {t} hours, table rows {}", c.table_rows.len());
    if let Some(k) = expect_table_rows {
        assert_eq!(c.table_rows.len(), k, "table row count");
    }
    let mut failures: Vec<String> = Vec::new();

    // Forward: bar 1e-4 * max|cpu| of each hour.
    let mut worst = (0.0f32, 0usize, 0usize);
    let mut per_hour = vec![0.0f32; t];
    let mut first_miss: Option<usize> = None;
    let mut misses: Vec<(f32, usize, usize, f32, f32)> = Vec::new();
    for h in 0..t {
        let mx = (0..n).fold(0.0f32, |m, r| m.max(c.q[r * t + h].abs()));
        let bar = 1e-4 * mx;
        for r in 0..n {
            let (a, b) = (c.q[r * t + h], g.q[r * t + h]);
            if !a.is_finite() || !b.is_finite() {
                failures.push(format!("forward non-finite at reach {r} hour {h}: cpu {a} cuda {b}"));
                continue;
            }
            let ratio = (a - b).abs() / bar.max(f32::MIN_POSITIVE);
            per_hour[h] = per_hour[h].max(ratio);
            if ratio > worst.0 {
                worst = (ratio, r, h);
            }
            if ratio > 1.0 {
                first_miss.get_or_insert(h);
                misses.push((ratio, r, h, a, b));
            }
        }
    }
    println!(
        "{label}: forward worst err/bar {:.4e} at reach {} hour {} (cpu {}, cuda {})",
        worst.0,
        worst.1,
        worst.2,
        c.q[worst.1 * t + worst.2],
        g.q[worst.1 * t + worst.2]
    );
    let curve: Vec<String> = per_hour.iter().step_by(6).map(|v| format!("{v:.1e}")).collect();
    println!("{label}: forward err/bar every 6th hour: [{}]", curve.join(", "));
    if !misses.is_empty() {
        misses.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap());
        for (ratio, r, h, a, b) in misses.iter().take(10) {
            println!("FWDMISS {label}: reach {r} hour {h}: cpu {a} cuda {b} err/bar {ratio:.3}");
        }
        failures.push(format!(
            "forward: {} entries over the bar, first at hour {:?}, worst err/bar {:.3}",
            misses.len(),
            first_miss,
            worst.0
        ));
    }
    let dl = (c.loss - g.loss).abs();
    println!("{label}: loss cpu {} cuda {} diff {dl:e}", c.loss, g.loss);

    // Mass sums.
    match (c.mass, g.mass) {
        (Some(mc), Some(mg)) => {
            println!("{label}: mass cpu {mc:?}");
            println!("{label}: mass cuda {mg:?}");
            let tol = 1e-5 * mc.gross_flux_m3;
            let pairs = [
                ("storage_change", mc.storage_change_m3, mg.storage_change_m3),
                ("net_flux", mc.net_flux_m3, mg.net_flux_m3),
                ("floor_created", mc.floor_created_m3, mg.floor_created_m3),
                ("gross_flux", mc.gross_flux_m3, mg.gross_flux_m3),
                ("mc_inflow", mc.mc_inflow_m3, mg.mc_inflow_m3),
            ];
            for (name, a, b) in pairs {
                println!("{label}: mass {name}: diff {:e} (bar {tol:e})", (a - b).abs());
                if !((a - b).abs() <= tol) {
                    failures.push(format!("mass {name}: cpu {a} cuda {b} (bar {tol:e})"));
                }
            }
            if mc.steps != mg.steps {
                failures.push(format!("mass steps: cpu {} cuda {}", mc.steps, mg.steps));
            }
            println!("{label}: mass rel cpu {:.3e}, cuda {:.3e}", mc.rel(), mg.rel());
        }
        other => failures.push(format!("stage mass balance missing: {:?}", (other.0.is_some(), other.1.is_some()))),
    }

    // Structural zeros and the zero mismatch count.
    let mut is_table = vec![false; n];
    for &r in &c.table_rows {
        is_table[r] = true;
    }
    let mut zero_mismatch = [0usize; 3];
    for pi in 0..3 {
        for i in 0..n {
            if is_table[i] {
                for (who, run) in [("cpu", &c), ("cuda", &g)] {
                    if run.grads[pi][i] != 0.0 {
                        failures.push(format!("structural zero {}[{i}] on {who} is {}", NAMES[pi], run.grads[pi][i]));
                    }
                }
            } else if (c.grads[pi][i] == 0.0) != (g.grads[pi][i] == 0.0) {
                zero_mismatch[pi] += 1;
            }
        }
    }
    println!("{label}: structural-zero set size per parameter {} (parameters {NAMES:?})", c.table_rows.len());
    println!("{label}: zero mismatches off the structural set {zero_mismatch:?} (parameters {NAMES:?})");

    // Gradients.
    let mut compared = 0usize;
    let mut accepted = 0usize;
    let mut noise: Option<[f32; 3]> = None;
    let mut worst_err = [0.0f32; 3];
    let mut worst_ratio = [0.0f32; 3];
    let mut accepted_lines: Vec<String> = Vec::new();
    for pi in 0..3 {
        let (cc, gg) = (&c.grads[pi], &g.grads[pi]);
        let scale = cc.iter().fold(1.0f32, |m, v| m.max(v.abs()));
        let gmax = cc.iter().fold(0.0f32, |m, v| m.max(v.abs()));
        for i in 0..n {
            let (a, b) = (gg[i], cc[i]);
            if !a.is_finite() || !b.is_finite() {
                failures.push(format!("{}[{i}]: non-finite cuda {a} cpu {b}", NAMES[pi]));
                continue;
            }
            compared += 1;
            let err = (a - b).abs();
            let allowed = (1e-3 * a.abs().max(b.abs())).max(1e-4 * scale);
            worst_err[pi] = worst_err[pi].max(err);
            worst_ratio[pi] = worst_ratio[pi].max(err / allowed);
            if err <= allowed {
                continue;
            }
            let nz = *noise.get_or_insert_with(|| {
                let nc = linearity_noise::<Cpu>(config, start, false, &c.grads, &dc);
                let ng = linearity_noise::<Gpu>(config, start, true, &g.grads, &dg);
                println!("{label}: linearity noise cpu {nc:?} cuda {ng:?}");
                std::array::from_fn(|k| nc[k] + ng[k])
            });
            let ratio = err / nz[pi].max(f32::MIN_POSITIVE);
            let rel_max = err / gmax.max(f32::MIN_POSITIVE);
            if err > NOISE_FACTOR * nz[pi] {
                failures.push(format!(
                    "{}[{i}]: cuda {a} cpu {b} err {err:e} over {NOISE_FACTOR} x noise {:e} (err/noise {ratio:.2}, err/max|g| {rel_max:e}, bar {allowed:e})",
                    NAMES[pi], nz[pi]
                ));
            } else if err > NOISE_CAP_FRAC * gmax {
                failures.push(format!(
                    "{}[{i}]: cuda {a} cpu {b} err {err:e} over the cap {NOISE_CAP_FRAC} x max|g| {gmax:e} (err/noise {ratio:.2}, bar {allowed:e})",
                    NAMES[pi]
                ));
            } else {
                accepted += 1;
                accepted_lines.push(format!(
                    "ACCEPTED {label} {}[{i}]: cuda {a} cpu {b} err {err:e}, err/max|g| {rel_max:e}, err/noise {ratio:.2} (bar {allowed:e})",
                    NAMES[pi]
                ));
            }
        }
    }
    for l in &accepted_lines {
        println!("{l}");
    }
    println!("{label}: gradient worst err {worst_err:?}, worst err/bar {worst_ratio:?} (parameters {NAMES:?})");
    let budget = (BUDGET_FRAC * compared as f64).floor();
    println!("GRAD {label} ({}): compared {compared}, accepted {accepted} (budget {budget})", mode());
    if (accepted as f64) > BUDGET_FRAC * compared as f64 {
        failures.push(format!("{accepted} accepted misses exceed {BUDGET_FRAC} of {compared} entries"));
    }
    for f in &failures {
        println!("FAIL {label}: {f}");
    }
    assert!(failures.is_empty(), "{label}: {} failures (see FAIL lines)", failures.len());
}

#[test]
fn juniata_stage_engine_cpu_vs_cuda() {
    // 213 stage reaches and no table, so the structural-zero set is empty.
    engine_cpu_vs_cuda("juniata_stage_engine", JUNIATA_CONFIG, (1990, 3, 1), Some(0));
}

#[test]
fn hudson_stage_engine_cpu_vs_cuda() {
    // Table geometry on the 41 stage rows, a prescribed Battery boundary.
    engine_cpu_vs_cuda("hudson_stage_engine", HUDSON_CONFIG, (2010, 3, 1), Some(41));
}

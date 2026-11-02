//! Wall time of the fused stage op on CPU (NdArray) and CUDA across network sizes.
//!
//! This is a measurement, not a regression gate. It prints one row per network and substep
//! count: the reach count, the number of tree levels, CPU and CUDA milliseconds per hour
//! (forward plus one backward over 24 hours) and their ratio. It also prints the setup times
//! of `StageBlock::new`, `op_consts` and `op_consts_dev`.
//!
//! Networks: the Juniata and Hudson bundles, and, when `DDRS_CONUS_ADJ` names the CONUS
//! adjacency zarr, the upstream sub-basins closest to 1,000, 10,000 and 50,000 reaches plus the
//! whole network. Each sub-basin is re-indexed in the original order, so the pattern stays
//! lower triangular (`rows[k] >= cols[k]`), which the tree solve needs.
//!
//! The asserts are two. Every `q_next` is finite on both backends. The 24th timed hour's
//! `q_next` on CUDA is within `1e-4 * max|cpu|` of the CPU one. Both backends feed their own
//! `q_next` forward. The test runs in release builds only; a debug build skips it.
//!
//! `DDRS_SCALE_NET` (a label substring) and `DDRS_SCALE_SUBSTEPS` (1 or 4) limit the run to one row,
//! which is how a profiler is pointed at it.
//!
//! Run: `DDRS_CONUS_ADJ=<zarr> cargo test --release --features cuda --test stage_cuda_scale -- --nocapture`
#![cfg(feature = "cuda")]

mod common;

use std::sync::Arc;
use std::time::Instant;

use burn::backend::Autodiff;
use burn::tensor::backend::Backend;
use burn::tensor::Tensor;

use ddrs::config::DownstreamBoundary;
use ddrs::data::ConusAdjacencyStore;
use ddrs::routing::stage::{StageBlock, StageNetwork, StageParams};
use ddrs::sparse::tree_lu::tree_levels;
use ddrs::sparse::{CsrPattern, SparseAdjacency};

use common::{InnerBackend, TestDevice};

type Cpu = InnerBackend;
type Gpu = burn_cuda::Cuda<f32, i32>;

const DEPTH_LB: f32 = 0.01;
const WARMUP_HOURS: usize = 2;
const TIMED_HOURS: usize = 24;
/// Parity bar of the last timed hour, relative to `max|cpu|`.
const PARITY_BAR: f32 = 1e-4;

fn tv<B: Backend>(v: &[f32], device: &B::Device) -> Tensor<Autodiff<B>, 1> {
    Tensor::<Autodiff<B>, 1>::from_floats(v, device)
}

fn to_vec1<B: Backend>(t: Tensor<B, 1>) -> Vec<f32> {
    t.into_data().to_vec().unwrap()
}

/// One network with its start state.
struct Net {
    label: String,
    /// True for the CONUS rows, which run at theta 1.0 at every substep count.
    conus: bool,
    adj: SparseAdjacency,
    pattern: Arc<CsrPattern>,
    h0: Vec<f32>,
    q0: Vec<f32>,
}

// copied from tests/stage_op_cuda_parity.rs::params, with use_cuda a parameter
///
/// Theta: the bundle rows use the shipped values, 1.0 with one substep (as in
/// `examples/juniata/ddrs_stage.yaml`) and 0.6 with four (as in `examples/hudson/ddrs.yaml`).
/// The CONUS rows use 1.0 at both substep counts. With theta 0.6 the uniform input
/// (q' of 1.0 on every reach) blows up to NaN on the CPU path itself on the large networks,
/// so those rows would time a diverged state. The op does the same work at any theta.
fn params(substeps: u32, theta: f32, use_cuda: bool) -> StageParams {
    StageParams {
        theta,
        theta_control: 0.5,
        substeps,
        boundary: DownstreamBoundary::NormalDepth,
        depth_lb: DEPTH_LB,
        bottom_width_lb: 0.1,
        dt: 3600.0,
        g: 9.81,
        use_cuda,
    }
}

/// Build the start state of an adjacency: q0 is the cumulative upstream sum of q' = 1 and the
/// stage comes from an NdArray hotstart with n 0.035, p 21, q 0.5.
// copied from tests/stage_op_cuda_parity.rs::bundle_case (hotstart part)
fn net_from_adjacency(label: &str, conus: bool, adj: SparseAdjacency, device: &TestDevice) -> Net {
    let pattern = Arc::new(CsrPattern::from_sparse(&adj));
    let n = adj.n;
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
    let net = StageNetwork::build(&adj, &pattern, None, 0.0, &[]).unwrap();
    let mut hb = StageBlock::<Cpu>::new(net, params(1, 1.0, false), device);
    let slope_f: Vec<f32> = adj.slope.iter().map(|s| s.max(1e-3)).collect();
    hb.hotstart(
        tv::<Cpu>(&q_init, device),
        tv::<Cpu>(&vec![0.035; n], device),
        tv::<Cpu>(&vec![21.0; n], device),
        tv::<Cpu>(&vec![0.5; n], device),
        tv::<Cpu>(&slope_f, device),
    );
    let h0 = to_vec1(hb.stage().unwrap());
    Net { label: label.to_string(), conus, adj, pattern, h0, q0: q_init }
}

fn adjacency_of(store: &ConusAdjacencyStore) -> SparseAdjacency {
    SparseAdjacency {
        n: store.n,
        rows: store.indices_0.clone(),
        cols: store.indices_1.clone(),
        values: vec![1.0; store.nnz],
        length_m: store.length_m.to_vec(),
        slope: store.slope.to_vec(),
        parent_offset: None,
    }
}

/// The upstream sub-basin of reach `outlet`, re-indexed in the original order.
/// Rows are downstream of columns (`rows[k] > cols[k]`), so a descending sweep marks every
/// upstream reach, and keeping the original order keeps the pattern lower triangular.
fn sub_basin(adj: &SparseAdjacency, outlet: usize) -> SparseAdjacency {
    let n = adj.n;
    let mut down = vec![usize::MAX; n];
    for (&r, &c) in adj.rows.iter().zip(&adj.cols) {
        down[c as usize] = r as usize;
    }
    let mut keep = vec![false; n];
    keep[outlet] = true;
    for i in (0..n).rev() {
        if down[i] != usize::MAX && keep[down[i]] {
            keep[i] = true;
        }
    }
    let mut new_idx = vec![usize::MAX; n];
    let mut m = 0;
    for i in 0..n {
        if keep[i] {
            new_idx[i] = m;
            m += 1;
        }
    }
    let (mut rows, mut cols) = (vec![], vec![]);
    for (&r, &c) in adj.rows.iter().zip(&adj.cols) {
        // The outlet itself has its downstream reach outside the sub-basin.
        if keep[c as usize] && keep[r as usize] {
            rows.push(new_idx[r as usize] as i32);
            cols.push(new_idx[c as usize] as i32);
        }
    }
    assert!(rows.iter().zip(&cols).all(|(r, c)| r > c), "sub-basin pattern must stay lower triangular");
    let pick = |v: &[f32]| (0..n).filter(|&i| keep[i]).map(|i| v[i]).collect::<Vec<f32>>();
    SparseAdjacency {
        n: m,
        values: vec![1.0; rows.len()],
        rows,
        cols,
        length_m: pick(&adj.length_m),
        slope: pick(&adj.slope),
        parent_offset: None,
    }
}

/// The reach whose upstream set (itself included) has a size closest to `target`.
fn closest_basin(adj: &SparseAdjacency, target: usize) -> usize {
    let n = adj.n;
    let mut size = vec![1usize; n];
    let mut down = vec![usize::MAX; n];
    for (&r, &c) in adj.rows.iter().zip(&adj.cols) {
        down[c as usize] = r as usize;
    }
    for i in 0..n {
        if down[i] != usize::MAX {
            let s = size[i];
            size[down[i]] += s;
        }
    }
    (0..n).min_by_key(|&i| size[i].abs_diff(target)).unwrap()
}

/// What one backend run records.
struct Run {
    /// Milliseconds per hour over the timed hours (forward plus one backward).
    ms_per_hour: f64,
    /// `q_next` of every timed hour.
    hours: Vec<Vec<f32>>,
    setup_new_ms: f64,
    setup_consts_ms: f64,
    /// Zero on the CPU run.
    setup_consts_dev_ms: f64,
}

fn weights(n: usize) -> Vec<f32> {
    (0..n).map(|i| 0.5 + ((i * 7) % 13) as f32 / 13.0).collect()
}

/// Run a network on one backend. `sync` waits for the device queue.
fn run_backend<B: Backend + 'static>(
    net: &Net,
    substeps: u32,
    theta: f32,
    use_cuda: bool,
    device: &B::Device,
    sync: impl Fn(),
) -> Run
where
    B::FloatTensorPrimitive: 'static,
    B::Device: 'static,
{
    let n = net.adj.n;
    let stage_net = StageNetwork::build(&net.adj, &net.pattern, None, 0.0, &[]).unwrap();
    let t = Instant::now();
    let mut block = StageBlock::<B>::new(stage_net, params(substeps, theta, use_cuda), device);
    block.set_stage(tv::<B>(&net.h0, device));
    sync();
    let setup_new_ms = t.elapsed().as_secs_f64() * 1e3;
    let t = Instant::now();
    let _ = block.op_consts(&net.pattern);
    let setup_consts_ms = t.elapsed().as_secs_f64() * 1e3;
    let mut setup_consts_dev_ms = 0.0;
    if use_cuda {
        let t = Instant::now();
        let _ = block.op_consts_dev(&net.pattern);
        sync();
        setup_consts_dev_ms = t.elapsed().as_secs_f64() * 1e3;
    }

    let leaf = |v: &[f32]| tv::<B>(v, device).require_grad();
    let (nn, pp, qq) = (leaf(&vec![0.035; n]), leaf(&vec![21.0; n]), leaf(&vec![0.5; n]));
    let qp = leaf(&vec![1.0; n]);
    let mc = tv::<B>(&vec![0.0; n], device);
    let w = tv::<B>(&weights(n), device);
    let mut q = leaf(&net.q0);

    // One block of hours with one backward of sum(w * q_next) over them.
    let mut go = |hours: usize, q: &mut Tensor<Autodiff<B>, 1>| -> Vec<Tensor<Autodiff<B>, 1>> {
        let mut outs = vec![];
        let mut loss: Option<Tensor<Autodiff<B>, 1>> = None;
        for _ in 0..hours {
            let out = block.step_fused(&net.pattern, nn.clone(), pp.clone(), qq.clone(), q.clone(), mc.clone(), qp.clone());
            let term = (out.clone() * w.clone()).sum();
            loss = Some(match loss {
                None => term,
                Some(l) => l + term,
            });
            *q = out.clone();
            outs.push(out);
        }
        let grads = loss.unwrap().backward();
        // Keep the gradients alive until the queue is empty.
        drop(grads);
        outs
    };
    let _ = go(WARMUP_HOURS, &mut q);
    sync();
    let t = Instant::now();
    let outs = go(TIMED_HOURS, &mut q);
    sync();
    let ms_per_hour = t.elapsed().as_secs_f64() * 1e3 / TIMED_HOURS as f64;
    let hours = outs.into_iter().map(|o| to_vec1(o.inner())).collect();
    Run { ms_per_hour, hours, setup_new_ms, setup_consts_ms, setup_consts_dev_ms }
}

fn have_conus() -> Option<String> {
    std::env::var("DDRS_CONUS_ADJ").ok().filter(|s| !s.is_empty())
}

#[test]
fn stage_op_wall_time_cpu_vs_cuda() {
    if cfg!(debug_assertions) {
        eprintln!("skipped in debug builds: run with --release");
        return;
    }
    let device = TestDevice::default();
    let dg: <Gpu as burn::tensor::backend::BackendTypes>::Device = Default::default();

    let bundle = |label: &str, path: &str| {
        let store = ConusAdjacencyStore::open(path).expect("open bundle adjacency");
        net_from_adjacency(label, false, adjacency_of(&store), &device)
    };
    let mut nets = vec![
        bundle("juniata", "examples/juniata/data/juniata_conus_adjacency.zarr"),
        bundle("hudson", "examples/hudson/data/hudson_conus_adjacency.zarr"),
    ];
    match have_conus() {
        Some(path) => {
            let store = ConusAdjacencyStore::open(&path).expect("open CONUS adjacency");
            let full = adjacency_of(&store);
            for target in [1_000usize, 10_000, 50_000] {
                let outlet = closest_basin(&full, target);
                let sub = sub_basin(&full, outlet);
                let label = format!("conus_basin_{}", sub.n);
                nets.push(net_from_adjacency(&label, true, sub, &device));
            }
            nets.push(net_from_adjacency("conus_full", true, full, &device));
        }
        None => eprintln!("DDRS_CONUS_ADJ is unset: the CONUS rows are skipped"),
    }

    println!(
        "{:<20} {:>8} {:>7} {:>5} {:>5} {:>12} {:>12} {:>9}",
        "network", "n", "levels", "sub", "theta", "cpu ms/h", "cuda ms/h", "cuda/cpu"
    );
    let mut failures = vec![];
    // Optional filters for profiling one row: a label substring and a substep count.
    let only_net = std::env::var("DDRS_SCALE_NET").ok().filter(|s| !s.is_empty());
    let only_sub: Option<u32> = std::env::var("DDRS_SCALE_SUBSTEPS").ok().and_then(|s| s.parse().ok());
    for net in &nets {
        if only_net.as_ref().is_some_and(|f| !net.label.contains(f.as_str())) {
            continue;
        }
        let levels = tree_levels(&net.pattern).expect("tree pattern").n_levels;
        for substeps in [1u32, 4] {
            if only_sub.is_some_and(|k| k != substeps) {
                continue;
            }
            let theta = if net.conus || substeps == 1 { 1.0f32 } else { 0.6 };
            let cpu = run_backend::<Cpu>(net, substeps, theta, false, &device, || {});
            let gpu = run_backend::<Gpu>(net, substeps, theta, true, &dg, || {
                <Gpu as Backend>::sync(&dg).unwrap();
            });
            println!(
                "{:<20} {:>8} {:>7} {:>5} {:>5.1} {:>12.4} {:>12.4} {:>9.2}",
                net.label,
                net.adj.n,
                levels,
                substeps,
                theta,
                cpu.ms_per_hour,
                gpu.ms_per_hour,
                gpu.ms_per_hour / cpu.ms_per_hour
            );
            println!(
                "  setup ms (substeps {substeps}): cpu new {:.2} consts {:.2} | cuda new {:.2} consts {:.2} consts_dev {:.2}",
                cpu.setup_new_ms, cpu.setup_consts_ms, gpu.setup_new_ms, gpu.setup_consts_ms, gpu.setup_consts_dev_ms
            );
            // Finite everywhere.
            for (name, run) in [("cpu", &cpu), ("cuda", &gpu)] {
                for (k, h) in run.hours.iter().enumerate() {
                    if let Some(i) = h.iter().position(|v| !v.is_finite()) {
                        failures.push(format!("{} s{substeps} {name} hour {k}: q_next[{i}] = {}", net.label, h[i]));
                    }
                }
            }
            // Parity per hour, printed; the last hour is asserted.
            let mut per_hour = vec![];
            let mut worst_last = (0usize, 0.0f32, 0.0f32, 0.0f32);
            for (k, (c, g)) in cpu.hours.iter().zip(&gpu.hours).enumerate() {
                let scale = c.iter().fold(0.0f32, |m, v| m.max(v.abs()));
                let (mut wi, mut we) = (0usize, 0.0f32);
                for (i, (a, b)) in c.iter().zip(g).enumerate() {
                    let e = (a - b).abs();
                    if e > we || e.is_nan() {
                        (wi, we) = (i, e);
                    }
                }
                per_hour.push(we / (PARITY_BAR * scale));
                if k == cpu.hours.len() - 1 {
                    worst_last = (wi, we, c[wi], g[wi]);
                    if !(we <= PARITY_BAR * scale) {
                        failures.push(format!(
                            "{} s{substeps} hour {k}: reach {wi} cpu {} cuda {} err {we:e} vs bar {:e}",
                            net.label,
                            c[wi],
                            g[wi],
                            PARITY_BAR * scale
                        ));
                    }
                }
            }
            let hs: Vec<String> = per_hour.iter().map(|v| format!("{v:.2}")).collect();
            println!(
                "  parity err/bar per hour: [{}] (last: reach {} err {:.3e} cpu {} cuda {})",
                hs.join(", "),
                worst_last.0,
                worst_last.1,
                worst_last.2,
                worst_last.3
            );
        }
    }
    assert!(failures.is_empty(), "scale test failures:\n{}", failures.join("\n"));
}

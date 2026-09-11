//! Diagnostic: does a forward-only pass of `MuskingumCunge` on the Autodiff
//! backend retain memory across repeated construction/teardown when backward
//! is never called?
//!
//! Motivation: a landscape study that evaluates the loss ~3,000 times
//! (forward only, AD backend) grew from 31 GB to 77 GB RSS over the course
//! of the sweep. This probe isolates the minimal repro: build a fresh
//! `MuskingumCunge`, run one forward, read a scalar out, drop everything,
//! and watch RSS across iterations.
//!
//! Uses the same 5-reach sandbox fixture as `examples/compare_ddr_sandbox.rs`
//! (see that file + `src/routing/mmc.rs` for the setup this mirrors), but
//! tiles `q_prime` along the time axis so each forward runs >= 2000
//! timesteps -- the fixture itself is only 238 timesteps, too short to
//! amplify a per-timestep leak into something visible in 30-40 iterations.
//!
//! Modes (argv[1]):
//!   ad-notrack           Autodiff<NdArray<f32>>, `n`/`q_spatial` NOT
//!                        require_grad, no backward call.
//!   ad-track-backward    `n`/`q_spatial` require_grad, forward, then
//!                        loss.backward() on sum(output), then drop.
//!   ad-track-nobackward  `n`/`q_spatial` require_grad, forward only,
//!                        backward never called.
//!   inner                SKIPPED -- see the printed explanation below.
//!                        `MuskingumCunge<I>` cannot run without the
//!                        Autodiff wrapper even when `I` is already a
//!                        plain backend: every stateful field on the
//!                        struct (src/routing/mmc.rs -- `n`, `q_spatial`,
//!                        `length`, `slope`, `x_storage`, `discharge_t`,
//!                        `q_prime`, ...) is typed `Tensor<Autodiff<I>, _>`,
//!                        not `Tensor<I, _>`. Constructing
//!                        `MuskingumCunge::<NdArray<f32>>` therefore still
//!                        builds an `Autodiff<NdArray<f32>>` tape internally
//!                        regardless of what the caller passes for `I`.
//!                        There is no plain-inner-backend forward path here.
//!
//! argv[2] (optional): number of iterations N, default 30.
//!
//! RSS is read from `/proc/self/statm` (field 2, resident pages) * page
//! size, in MB. After the loop, prints the least-squares slope (MB per
//! iteration) over the last 20 samples (or fewer, if N < 20), and a final
//! `MODE <name> slope_mb_per_iter <x> total_growth_mb <y>` summary line.
//!
//! This file is measurement only -- nothing under `src/` is touched.

use std::path::Path;

use burn::backend::{Autodiff, NdArray};
use burn::tensor::Tensor;

use ddrs::config::Config;
use ddrs::routing::{MuskingumCunge, RoutingInputs, SpatialParameters};
use ddrs::sandbox::{self, N_REACHES};
use ddrs::sparse::SparseAdjacency;

// Fixed to the CPU inner backend -- mode D ("inner") is a no-op, so there's
// no need to make this generic over the inner backend.
type Inner = NdArray<f32>;
type AD = Autodiff<Inner>;
type Device = <Inner as burn::tensor::backend::BackendTypes>::Device;

/// Resident set size in MB, read from `/proc/self/statm`. Field 1 (0-indexed)
/// is resident pages; multiply by the page size to get bytes.
fn read_rss_mb() -> f64 {
    let statm = std::fs::read_to_string("/proc/self/statm").expect("read /proc/self/statm");
    let resident_pages: u64 = statm
        .split_whitespace()
        .nth(1)
        .expect("statm missing resident field")
        .parse()
        .expect("parse resident pages");
    let page_size = unsafe { libc::sysconf(libc::_SC_PAGESIZE) } as u64;
    (resident_pages * page_size) as f64 / (1024.0 * 1024.0)
}

/// Ordinary least-squares slope of `y` against `x` over the given samples.
fn slope_mb_per_iter(points: &[(usize, f64)]) -> f64 {
    let n = points.len() as f64;
    let sum_x: f64 = points.iter().map(|(x, _)| *x as f64).sum();
    let sum_y: f64 = points.iter().map(|(_, y)| *y).sum();
    let sum_xx: f64 = points.iter().map(|(x, _)| (*x as f64) * (*x as f64)).sum();
    let sum_xy: f64 = points.iter().map(|(x, y)| (*x as f64) * y).sum();
    let denom = n * sum_xx - sum_x * sum_x;
    if denom.abs() < 1e-12 {
        0.0
    } else {
        (n * sum_xy - sum_x * sum_y) / denom
    }
}

/// Build fresh tensors + a fresh `MuskingumCunge`, run one forward, read one
/// scalar out, optionally backward, then let everything drop at the end of
/// the function. Returns the scalar (kept only so the compiler can't dead-
/// code-eliminate the forward).
fn run_forward(
    device: &Device,
    tiled_qprime: &[f32],
    n_timesteps: usize,
    adjacency_flat: &[f32],
    cfg: &Config,
    require_grad: bool,
    do_backward: bool,
) -> f32 {
    let qprime: Tensor<AD, 2> = Tensor::<AD, 1>::from_floats(tiled_qprime, device)
        .reshape([n_timesteps, N_REACHES]);

    let adjacency = SparseAdjacency::from_dense(
        N_REACHES,
        adjacency_flat,
        vec![5000.0; N_REACHES],
        vec![0.001; N_REACHES],
    );
    let routing_inputs = RoutingInputs::<Inner> {
        adjacency,
        x_storage: Tensor::ones([N_REACHES], device) * 0.25,
    };

    let mut n_param: Tensor<AD, 1> = Tensor::ones([N_REACHES], device) * 0.5;
    let mut q_param: Tensor<AD, 1> = Tensor::ones([N_REACHES], device) * 0.5;
    if require_grad {
        n_param = n_param.require_grad();
        q_param = q_param.require_grad();
    }

    let params = SpatialParameters::<Inner> {
        n: n_param,
        q_spatial: q_param,
        p_spatial: None,
        k_d: None,
        d_gw: None,
        leakance_factor: None,
        impervious_mask: None,
    };

    let mut mc = MuskingumCunge::<Inner>::new(cfg.clone(), device.clone());
    mc.setup_inputs(routing_inputs, qprime, params, false, None);
    let out = mc.forward();

    if do_backward {
        let loss = out.clone().sum();
        let _grads = loss.backward();
    }

    out.inner().into_data().to_vec::<f32>().unwrap()[0]
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    if args.len() < 2 {
        eprintln!("usage: leak_probe <mode> [n_iters]");
        eprintln!("modes: ad-notrack | ad-track-backward | ad-track-nobackward | inner");
        std::process::exit(1);
    }
    let mode = args[1].as_str();
    let n_iters: usize = args
        .get(2)
        .map(|s| s.parse().expect("n_iters must be a usize"))
        .unwrap_or(30);

    if mode == "inner" {
        println!("MODE inner: SKIPPED.");
        println!(
            "MuskingumCunge<I> cannot run without the Autodiff wrapper even when I is \
             already a plain (non-autodiff) backend: every stateful field on the struct \
             (src/routing/mmc.rs -- `n`, `q_spatial`, `length`, `slope`, `x_storage`, \
             `discharge_t`, `q_prime`, ...) is typed `Tensor<Autodiff<I>, _>`, not \
             `Tensor<I, _>`. Constructing MuskingumCunge::<NdArray<f32>> therefore still \
             builds an Autodiff<NdArray<f32>> tape internally regardless of what the \
             caller passes for I. There is no plain-inner-backend forward path to probe."
        );
        return;
    }

    let (require_grad, do_backward) = match mode {
        "ad-notrack" => (false, false),
        "ad-track-backward" => (true, true),
        "ad-track-nobackward" => (true, false),
        other => {
            eprintln!(
                "unknown mode {other:?}; expected ad-notrack | ad-track-backward | \
                 ad-track-nobackward | inner"
            );
            std::process::exit(1);
        }
    };

    let device = Device::default();
    let fixtures = Path::new("fixtures/sandbox");
    let inputs = sandbox::load_from_dir(fixtures).expect("load sandbox fixtures");

    // Tile q_prime along the time axis: qprime_flat is row-major [T, N_REACHES],
    // so repeating the flat vec whole-hog repeats complete timestep rows in
    // order, giving [T * tile_count, N_REACHES].
    let base_t = inputs.n_timesteps;
    const MIN_TIMESTEPS: usize = 2000;
    let tile_count = MIN_TIMESTEPS.div_ceil(base_t).max(1);
    let n_timesteps = base_t * tile_count;
    let mut tiled_qprime = Vec::with_capacity(inputs.qprime_flat.len() * tile_count);
    for _ in 0..tile_count {
        tiled_qprime.extend_from_slice(&inputs.qprime_flat);
    }
    println!(
        "tiled q_prime along the time axis: fixture has {base_t} timesteps -> repeated \
         {tile_count}x -> {n_timesteps} timesteps (>= {MIN_TIMESTEPS})"
    );
    println!("MODE {mode}: N={n_iters} require_grad={require_grad} do_backward={do_backward}");

    let mut samples: Vec<(usize, f64)> = Vec::with_capacity(n_iters + 1);
    let rss0 = read_rss_mb();
    samples.push((0, rss0));
    println!("iter 0, rss_mb {rss0:.2}");

    for iter in 1..=n_iters {
        let scalar = run_forward(
            &device,
            &tiled_qprime,
            n_timesteps,
            &inputs.adjacency_flat,
            &inputs.config,
            require_grad,
            do_backward,
        );
        let _ = scalar; // read out to force the forward to actually run
        let rss = read_rss_mb();
        samples.push((iter, rss));
        println!("iter {iter}, rss_mb {rss:.2}");
    }

    let tail_n = 20.min(samples.len());
    let tail = &samples[samples.len() - tail_n..];
    let slope = slope_mb_per_iter(tail);
    let total_growth = samples.last().unwrap().1 - samples.first().unwrap().1;

    println!("slope over last {tail_n} iterations: {slope:.4} MB/iter");
    println!("MODE {mode} slope_mb_per_iter {slope:.4} total_growth_mb {total_growth:.2}");
}

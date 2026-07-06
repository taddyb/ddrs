# Zeta Gradient-Sensitivity Probe Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build the two no-training instruments that separate H4-starvation from H4-rejection: a per-reach adjoint reachability map (∂L1/∂leakance-params at trained + cold points) and a q′-perturbation detectability bound on GAGES-II Ref basins, with CONUS gradient maps and a findings report.

**Architecture:** Stage 1 replicates the training mini-batch loop (`src/training/driver.rs:91-181`) but swaps `forward` for a probe variant that detaches the three per-reach leakance vectors from the head graph and re-lifts them as `require_grad` leaves — the analytical `TimestepLeakanceOp` backward already supplies their exact grads, so no Backward impl changes. Stage 2 replicates the chunked eval loop (`src/training/eval.rs:94-111`) with additive `q_prime` perturbations injected at the tensor level. All Python (site selection, analysis, notebooks) runs against the exported netCDF/CSV artifacts.

**Tech Stack:** Rust (BURN 0.21, netcdf crate), Python under `~/projects/ddr` uv venv (numpy/xarray/scipy) and `./ddrs-py` venv (geopandas/matplotlib, per the `ddrs-eval-plots` skill).

**Spec:** `docs/superpowers/specs/2026-07-02-zeta-gradient-probe-design.md`
(Spec delta discovered while planning: params in `SpatialParameters` are
NORMALIZED [0,1]; denormalization happens inside `setup_inputs`
(`src/routing/mmc.rs:203-222`). The leaves are therefore lifted in normalized
space — grads are dimensionless. This is strictly better than the spec's
denormalized-units assumption; the within-parameter spatial analysis is
unchanged.)

**CPU-ONLY EXECUTION (2026-07-02 amendment):** the GPU is occupied by another
training task — ALL probe runs execute on the CPU (`NdArray<f32>` backend).
Consequences threaded through the tasks below:
- The probe binary takes `--backend {cpu,cuda}` (default `cpu`). On `cpu` it
  forces `cfg.params.sparse_solver = SparseSolver::Cpu` and ignores the
  config's CUDA `device:` ordinal (`burn::backend::ndarray::NdArrayDevice::Cpu`).
  `bin/eval.rs` stays untouched — only the new binary dispatches.
- CPU NdArray is DETERMINISTIC: the two stage-2 baselines must be
  byte-identical (noise floor ≡ 0). Detectability then reduces to the
  observational-uncertainty band alone; the second baseline becomes a
  determinism assertion, not a noise estimate.
- Runtimes are unknown a priori on CPU — Tasks 4 and 7 each start with a
  TIMING GATE (one unit of work, extrapolate, scale N accordingly). Stage 2
  gains an `--eval-days` flag (default 1095 = 3 years) — a constant planted
  delta does not need the 15-year window.
- Be a polite neighbor to the GPU job's data loaders: run probe commands
  under `nice -n 10`.

**Working directories:** Rust work in this worktree
(`/home/tbindas/projects/ddrs/.claude/worktrees/zeta-sensitivity`, branch
`worktree-zeta-sensitivity`). GPU runs execute with
`cwd = /home/tbindas/projects/ddrs` (the main tree — run dirs/data live there)
using the WORKTREE's binaries by ABSOLUTE path (see memory: relative
`target/release/...` there resolves to the main tree's stale binary).
`fixtures/{sandbox,gradcheck}` and `output/` are already set up in this
worktree.

**Key inputs (verified on disk):**
- Trained checkpoint: `/home/tbindas/projects/ddrs/.ddrs/runs/2026-07-01T13-43-32Z-train-and-test/checkpoints/epoch_5_mb_9` (dir with `head.mpk`)
- Config: `config/experiments/leakance_hourly_on.yaml` (loss block absent ⇒ L1; `batch_size: 64`, `rho: 90`, `warmup: 5`, seed 42)
- GAGES-II classes: `/mnt/ssd1/data/gage_shp_files/gagesII_9322_sept30_2011.dbf` (fields `STAID`, `CLASS`, `HCDN_2009`)
- Gauges: `/home/tbindas/projects/ddr/references/gage_info/gages_3000.csv` (STAID, DRAIN_SQKM, LAT_GAGE, LNG_GAGE, COMID)
- Attributes: `/home/tbindas/projects/ddr/data/merit_global_attributes_v2.nc` (`aridity`, `log10_uparea`, …)
- Hourly-ON `kan_parameters.nc` (for reachability-vs-zeta cross-refs): in the run dir above.

---

## File structure

| File | Change | Responsibility |
|---|---|---|
| `src/training/probe.rs` | create | `lift_leaf`, `ProbeLeaves`, `probe_forward` (head variant), `GradAccum` (COMID-keyed accumulation) |
| `src/training/mod.rs` | modify | `pub mod probe;` |
| `src/dump_parameters.rs` | modify | `write_grad_netcdf` (COMID_probe dim, same add-or-overwrite idiom as `write_zeta_netcdf`) |
| `src/bin/probe_zeta_gradient.rs` | create | CLI: stage-1 grad map (`--mode grad`), stage-2 perturbation runs (`--mode perturb`) |
| `tests/zeta_gradient_probe.rs` | create | leaf-grad + byte-parity + GradAccum tests on the mock network |
| `scripts/zeta_probe_sites.py` | create | GAGES-II Ref join, strata, round packing → `probe_plan.csv` |
| `scripts/zeta_gradient_analysis.py` | create | verdicts (H4-starvation / H4-rejection / detectability), tables |
| `<out>/plots/gradient_maps.ipynb` | create (Task 9, via ddrs-eval-plots) | CONUS per-gauge scatter + per-reach polygon map |
| `docs/<date>-zeta-gradient-probe-findings.md` | create (Task 10) | report |

Output convention: everything lands under the main tree's
`/home/tbindas/projects/ddrs/output/zeta_probe/` (`grad_trained.nc`,
`grad_cold.nc`, `probe_plan.csv`, `perturb/round_<k>.nc`,
`perturb/baseline_<1,2>.nc`, `plots/`).

---

### Task 1: Probe core — failing tests first

**Files:**
- Create: `tests/zeta_gradient_probe.rs`
- Test: same

The tests target `ddrs::training::probe::{lift_leaf, GradAccum}` plus
engine-level leaf gradients, using the existing mock network from
`tests/common.rs` (same helpers as `tests/zeta_accum.rs`).

- [ ] **Step 1: Write the failing tests**

```rust
//! Stage-1 probe core: lifting normalized leakance params to autograd leaves
//! must (a) leave the routed forward byte-identical, (b) yield finite nonzero
//! leaf grads on a losing chain, and (c) accumulate by COMID exactly.

mod common;

use burn::backend::Autodiff;
use burn::tensor::Tensor;
use common::{
    mock_config, mock_routing_inputs, mock_spatial_parameters, mock_streamflow, InnerBackend,
    TestDevice,
};
use ddrs::routing::{MuskingumCunge, SpatialParameters};
use ddrs::training::probe::{lift_leaf, GradAccum};

type AB = Autodiff<InnerBackend>;

/// Losing-regime leakance params with the three leakance vectors lifted as
/// require_grad leaves. Mirrors tests/zeta_accum.rs::leakance_params.
fn probed_params(
    n: usize,
    device: &TestDevice,
) -> (SpatialParameters<InnerBackend>, [Tensor<AB, 1>; 3]) {
    let k_d = lift_leaf::<InnerBackend>(Tensor::<AB, 1>::ones([n], device));
    let d_gw = lift_leaf::<InnerBackend>(Tensor::<AB, 1>::zeros([n], device));
    let factor = lift_leaf::<InnerBackend>(Tensor::<AB, 1>::ones([n], device) * 0.5);
    (
        SpatialParameters {
            n: Tensor::<AB, 1>::ones([n], device) * 0.5,
            q_spatial: Tensor::<AB, 1>::ones([n], device) * 0.5,
            p_spatial: None,
            k_d: Some(k_d.clone()),
            d_gw: Some(d_gw.clone()),
            leakance_factor: Some(factor.clone()),
        },
        [k_d, d_gw, factor],
    )
}

#[test]
fn lifted_leaves_do_not_perturb_forward() {
    let device = TestDevice::default();
    let (n, t) = (5usize, 24usize);
    let cfg = mock_config();

    // Plain (non-leaf) leakance run — same values as probed_params.
    let mut mc_plain = MuskingumCunge::<InnerBackend>::new(cfg.clone(), device.clone());
    mc_plain.setup_inputs(
        mock_routing_inputs(n, &device),
        mock_streamflow(t, n, &device),
        SpatialParameters {
            n: Tensor::<AB, 1>::ones([n], &device) * 0.5,
            q_spatial: Tensor::<AB, 1>::ones([n], &device) * 0.5,
            p_spatial: None,
            k_d: Some(Tensor::<AB, 1>::ones([n], &device)),
            d_gw: Some(Tensor::<AB, 1>::zeros([n], &device)),
            leakance_factor: Some(Tensor::<AB, 1>::ones([n], &device) * 0.5),
        },
        false,
    );
    let out_plain: Vec<f32> = mc_plain.forward().into_data().to_vec().unwrap();

    let (params, _leaves) = probed_params(n, &device);
    let mut mc_leaf = MuskingumCunge::<InnerBackend>::new(cfg, device.clone());
    mc_leaf.setup_inputs(
        mock_routing_inputs(n, &device),
        mock_streamflow(t, n, &device),
        params,
        false,
    );
    let out_leaf: Vec<f32> = mc_leaf.forward().into_data().to_vec().unwrap();

    assert_eq!(out_plain, out_leaf, "lifting leaves must not change routing");
}

#[test]
fn leaf_grads_are_finite_and_nonzero_on_losing_chain() {
    let device = TestDevice::default();
    let (n, t) = (5usize, 24usize);
    let cfg = mock_config();

    let (params, [k_d, d_gw, factor]) = probed_params(n, &device);
    let mut mc = MuskingumCunge::<InnerBackend>::new(cfg, device.clone());
    mc.setup_inputs(
        mock_routing_inputs(n, &device),
        mock_streamflow(t, n, &device),
        params,
        false,
    );
    let loss = mc.forward().sum(); // any scalar downstream of every q_next
    let grads = loss.backward();

    for (name, leaf) in [("k_d", &k_d), ("d_gw", &d_gw), ("factor", &factor)] {
        let g: Vec<f32> = leaf
            .grad(&grads)
            .unwrap_or_else(|| panic!("{name}: no grad on leaf"))
            .into_data()
            .to_vec()
            .unwrap();
        assert_eq!(g.len(), n);
        assert!(g.iter().all(|v| v.is_finite()), "{name}: non-finite grad {g:?}");
        assert!(
            g.iter().any(|v| v.abs() > 0.0),
            "{name}: all-zero grad on a losing chain {g:?}"
        );
    }
}

#[test]
fn grad_accum_by_comid_sums_across_batches() {
    let mut acc = GradAccum::new();
    // Batch 1: comids 10, 20.
    acc.add(&[10, 20], &[1.0, 2.0], &[1.0, -2.0]);
    // Batch 2: comids 20, 30 (overlap on 20).
    acc.add(&[20, 30], &[3.0, 4.0], &[3.0, 4.0]);

    let rows = acc.into_sorted_rows();
    // (comid, abs_sum, net_sum, count)
    assert_eq!(rows[0], (10, 1.0, 1.0, 1));
    assert_eq!(rows[1], (20, 5.0, 1.0, 2));
    assert_eq!(rows[2], (30, 4.0, 4.0, 1));
}
```

- [ ] **Step 2: Run to verify failure**

Run: `cargo test --test zeta_gradient_probe 2>&1 | tail -5`
Expected: compile error — `ddrs::training::probe` module does not exist.

- [ ] **Step 3: Commit**

```bash
git add tests/zeta_gradient_probe.rs
git commit -m "test(probe): leaf-lift parity + leaf grads + COMID accumulation (red)"
```

### Task 2: Implement `src/training/probe.rs`

**Files:**
- Create: `src/training/probe.rs`
- Modify: `src/training/mod.rs` (add `pub mod probe;`)

- [ ] **Step 1: Write the module**

```rust
//! Stage-1 adjoint reachability probe (spec:
//! docs/superpowers/specs/2026-07-02-zeta-gradient-probe-design.md).
//!
//! Gradients of the training objective w.r.t. the per-reach NORMALIZED
//! leakance parameters, read at a FIXED head (no optimizer step ever).
//! `lift_leaf` detaches a head output from its graph and re-registers it as
//! an autograd leaf; the analytical `TimestepLeakanceOp` backward already
//! provides exact grads for these parents (tests/leakance_gradcheck.rs), so
//! no Backward impl is touched.

use std::collections::HashMap;

use burn::backend::Autodiff;
use burn::prelude::Backend;
use burn::tensor::Tensor;

use crate::config::Config;
use crate::data::dataset::RoutingTensors;
use crate::nn::kan_head::KanHead;
use crate::routing::utils::denormalize;
use crate::routing::{MuskingumCunge, RoutingInputs, SpatialParameters};
use crate::training::forward::scatter_add_by_group;

/// Detach `t` from its autograd graph and re-lift it as a `require_grad`
/// leaf. Values are bit-identical; only the tape topology changes.
pub fn lift_leaf<I: Backend>(t: Tensor<Autodiff<I>, 1>) -> Tensor<Autodiff<I>, 1> {
    Tensor::<Autodiff<I>, 1>::from_inner(t.inner()).require_grad()
}

/// The three lifted per-reach leaves (normalized [0,1] space).
pub struct ProbeLeaves<I: Backend> {
    pub k_d: Tensor<Autodiff<I>, 1>,
    pub d_gw: Tensor<Autodiff<I>, 1>,
    pub factor: Tensor<Autodiff<I>, 1>,
}

/// Training-path forward (mirrors `forward`, src/training/forward.rs:169-252)
/// with the leakance vectors lifted as leaves. Returns gauge-hourly
/// predictions plus the leaves to read grads from after `loss.backward()`.
pub fn probe_forward<I: Backend>(
    cfg: &Config,
    tensors: &RoutingTensors<Autodiff<I>>,
    head: &KanHead<Autodiff<I>>,
    device: &I::Device,
) -> (Tensor<Autodiff<I>, 2>, ProbeLeaves<I>) {
    assert!(cfg.params.use_leakance, "probe requires params.use_leakance");
    let params_map = head.forward(tensors.spatial_attributes.clone());

    let n_param = params_map.get("n").expect("head missing n").clone();
    let q_param = params_map.get("q_spatial").expect("head missing q_spatial").clone();
    let p_param = params_map.get("p_spatial").cloned();

    let n_active = tensors.adjacency.n;
    let x_storage: Tensor<Autodiff<I>, 1> = match params_map.get("x_storage") {
        Some(x_norm) => denormalize(
            x_norm.clone(),
            cfg.params.parameter_ranges.x_storage,
            cfg.params.log_space_parameters.iter().any(|s| s == "x_storage"),
        ),
        None => Tensor::full([n_active], 0.3_f32, device),
    };

    let n_hourly = tensors.q_prime.dims()[0];
    let q_prime_hourly = match &head.disagg {
        Some(d) => d.forward(
            tensors.q_prime_daily.clone(),
            tensors.spatial_attributes.clone(),
            tensors.precip_hourly.clone(),
            tensors.temp_hourly.clone(),
            n_hourly,
        ),
        None => tensors.q_prime.clone(),
    };

    for key in &["K_D", "d_gw", "leakance_factor"] {
        assert!(
            params_map.contains_key(*key),
            "probe: head missing '{key}' — use a leakance experiment config"
        );
    }
    let leaves = ProbeLeaves {
        k_d: lift_leaf::<I>(params_map.get("K_D").unwrap().clone()),
        d_gw: lift_leaf::<I>(params_map.get("d_gw").unwrap().clone()),
        factor: lift_leaf::<I>(params_map.get("leakance_factor").unwrap().clone()),
    };

    let mut engine = MuskingumCunge::<I>::new(cfg.clone(), device.clone());
    engine.setup_inputs(
        RoutingInputs { adjacency: tensors.adjacency.clone(), x_storage },
        q_prime_hourly,
        SpatialParameters {
            n: n_param,
            q_spatial: q_param,
            p_spatial: p_param,
            k_d: Some(leaves.k_d.clone()),
            d_gw: Some(leaves.d_gw.clone()),
            leakance_factor: Some(leaves.factor.clone()),
        },
        false,
    );
    let runoff = engine.forward();

    (
        scatter_add_by_group(
            runoff,
            tensors.flat_indices.clone(),
            tensors.group_ids.clone(),
            tensors.num_gauges,
        ),
        leaves,
    )
}

/// COMID-keyed gradient accumulation across probe batches. Batches route
/// different subnetworks (unions of 64 gauge subgraphs), so per-batch grads
/// are folded into a CPU map keyed by COMID.
pub struct GradAccum {
    map: HashMap<i64, (f64, f64, u32)>, // (Σ|g|, Σg, n_windows)
}

impl GradAccum {
    pub fn new() -> Self {
        Self { map: HashMap::new() }
    }

    pub fn add(&mut self, comids: &[i64], grad: &[f32], grad_signed: &[f32]) {
        assert_eq!(comids.len(), grad.len());
        assert_eq!(comids.len(), grad_signed.len());
        for i in 0..comids.len() {
            let e = self.map.entry(comids[i]).or_insert((0.0, 0.0, 0));
            e.0 += grad[i].abs() as f64;
            e.1 += grad_signed[i] as f64;
            e.2 += 1;
        }
    }

    /// `(comid, abs_sum, net_sum, count)` sorted by COMID.
    pub fn into_sorted_rows(self) -> Vec<(i64, f64, f64, u32)> {
        let mut rows: Vec<_> = self.map.into_iter().map(|(c, (a, s, n))| (c, a, s, n)).collect();
        rows.sort_by_key(|r| r.0);
        rows
    }
}

impl Default for GradAccum {
    fn default() -> Self {
        Self::new()
    }
}
```

Note: `GradAccum::add` receives `grad` and `grad_signed` as separate slices so
the caller passes the SAME slice twice (abs applied inside via `.abs()`); the
test's asymmetric inputs pin the (Σ|g|, Σg) semantics.

In `src/training/mod.rs` add `pub mod probe;` next to the existing module
declarations (do not re-export contents; the binary uses the `probe::` path).
If `scatter_add_by_group` is not already `pub` at `crate::training::forward`,
adjust the import to the path `src/training/mod.rs` re-exports (it is listed
in the existing `pub use` at src/training/mod.rs:28).

- [ ] **Step 2: Run the tests**

Run: `cargo test --test zeta_gradient_probe 2>&1 | tail -5`
Expected: 3 passed. If `leaf.grad(&grads)` returns `None`, check that
`lift_leaf` calls `.require_grad()` AFTER `from_inner` (order matters).

- [ ] **Step 3: Run the guard suite**

Run: `cargo test --test leakance_gradcheck --test leakance_off_parity --test zeta_accum 2>&1 | grep "test result"`
Expected: 8, 3, 6 passed.

- [ ] **Step 4: Commit**

```bash
git add src/training/probe.rs src/training/mod.rs
git commit -m "feat(probe): leaf-lifted leakance gradients + COMID accumulation"
```

### Task 3: `write_grad_netcdf` + stage-1 binary

**Files:**
- Modify: `src/dump_parameters.rs` (new writer after `write_zeta_netcdf`)
- Create: `src/bin/probe_zeta_gradient.rs`

- [ ] **Step 1: Add the netCDF writer**

Append to `src/dump_parameters.rs` (same add-or-overwrite idiom as
`write_zeta_netcdf`, new dimension `COMID_probe`):

```rust
/// Write the stage-1 adjoint reachability map: per-reach mean |g| and mean
/// signed g of the training L1 loss w.r.t. the NORMALIZED leakance params,
/// plus the window-coverage count. Dimension `COMID_probe` = union of reaches
/// covered by the sampled batches.
#[allow(clippy::too_many_arguments)]
pub fn write_grad_netcdf(
    path: &Path,
    comids: &[i64],
    grad_factor_abs: &[f32],
    grad_factor_net: &[f32],
    grad_dgw_abs: &[f32],
    grad_dgw_net: &[f32],
    grad_kd_abs: &[f32],
    grad_kd_net: &[f32],
    n_windows: &[i32],
    checkpoint_label: &str,
    n_batches: usize,
    seed: u64,
) -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
    let mut file = if path.exists() {
        netcdf::append(path)?
    } else {
        netcdf::create(path)?
    };

    file.add_attribute("probe_checkpoint", checkpoint_label)?;
    file.add_attribute("probe_n_batches", n_batches as i64)?;
    file.add_attribute("probe_seed", seed as i64)?;
    file.add_attribute("probe_ddrs_version", env!("CARGO_PKG_VERSION"))?;
    file.add_attribute(
        "probe_note",
        "adjoint reachability: d(L1)/d(normalized leakance params), mean over covering windows",
    )?;

    match file.dimension("COMID_probe") {
        Some(d) if d.len() != comids.len() => {
            return Err(format!(
                "{}: existing COMID_probe has {} reaches, this probe covered {}",
                path.display(),
                d.len(),
                comids.len()
            )
            .into());
        }
        Some(_) => {}
        None => {
            file.add_dimension("COMID_probe", comids.len())?;
        }
    }

    if let Some(mut v) = file.variable_mut("COMID_probe") {
        v.put_values(comids, ..)?;
    } else {
        let mut v = file.add_variable::<i64>("COMID_probe", &["COMID_probe"])?;
        v.put_values(comids, ..)?;
        v.put_attribute("long_name", "MERIT reach identifier (probe-covered network)")?;
    }

    let mut put = |name: &str, vals: &[f32], long_name: &str| -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
        if let Some(mut v) = file.variable_mut(name) {
            v.put_values(vals, ..)?;
        } else {
            let mut v = file.add_variable::<f32>(name, &["COMID_probe"])?;
            v.put_values(vals, ..)?;
            v.put_attribute("long_name", long_name)?;
            v.put_attribute("units", "dimensionless (per normalized param)")?;
        }
        Ok(())
    };
    put("grad_factor_abs", grad_factor_abs, "mean |dL1/d leakance_factor| per covering window")?;
    put("grad_factor_net", grad_factor_net, "mean signed dL1/d leakance_factor")?;
    put("grad_dgw_abs", grad_dgw_abs, "mean |dL1/d d_gw|")?;
    put("grad_dgw_net", grad_dgw_net, "mean signed dL1/d d_gw")?;
    put("grad_kd_abs", grad_kd_abs, "mean |dL1/d K_D|")?;
    put("grad_kd_net", grad_kd_net, "mean signed dL1/d K_D")?;

    if let Some(mut v) = file.variable_mut("n_windows") {
        v.put_values(n_windows, ..)?;
    } else {
        let mut v = file.add_variable::<i32>("n_windows", &["COMID_probe"])?;
        v.put_values(n_windows, ..)?;
        v.put_attribute("long_name", "number of sampled windows covering this reach")?;
    }

    Ok(())
}
```

- [ ] **Step 2: Write the binary (stage-1 mode)**

`src/bin/probe_zeta_gradient.rs`. Model the scaffolding (CLI parse, config
load, dataset open, head init/load, device) on `src/bin/eval.rs:36-109`, and
the batch loop on `src/training/driver.rs:91-181` (loss block copied, the
optimizer step and checkpointing REMOVED). Structure:

```rust
//! Stage-1/2 driver for the zeta gradient-sensitivity probe.
//! Spec: docs/superpowers/specs/2026-07-02-zeta-gradient-probe-design.md
//!
//!   --mode grad     adjoint reachability map (default)
//!   --mode perturb  stage-2 q' perturbation runs (Task 6)

use clap::Parser;
use std::path::PathBuf;

#[derive(Parser)]
struct Cli {
    #[arg(long)]
    config: PathBuf,
    /// Checkpoint DIRECTORY (epoch_E_mb_M/). Omit for the cold (fresh-init) point.
    #[arg(long)]
    checkpoint: Option<PathBuf>,
    #[arg(long, default_value = "grad")]
    mode: String,
    /// Stage 1: number of training-style batches to sample.
    #[arg(long, default_value_t = 32)]
    windows: usize,
    /// Sampler seed (identical seed+windows ⇒ identical batch/window sample).
    #[arg(long, default_value_t = 42)]
    seed: u64,
    #[arg(long)]
    output: PathBuf,
    /// Stage 2 only: probe plan CSV (round,comid,delta).
    #[arg(long)]
    probe_plan: Option<PathBuf>,
    /// Backend: "cpu" (NdArray, deterministic; forces sparse_solver=cpu) or "cuda".
    #[arg(long, default_value = "cpu")]
    backend: String,
    /// Stage 2 only: route only the first D days of the eval period.
    #[arg(long, default_value_t = 1095)]
    eval_days: usize,
}
```

Backend dispatch in `main` (model on `src/cli/system.rs:166`'s NdArray usage):

```rust
match cli.backend.as_str() {
    "cpu" => {
        type I = burn::backend::NdArray<f32>;
        let device = burn::backend::ndarray::NdArrayDevice::Cpu;
        cfg.params.sparse_solver = ddrs::config::SparseSolver::Cpu;
        eprintln!("backend: cpu (NdArray, deterministic; sparse_solver forced to cpu)");
        <I as burn::tensor::backend::Backend>::seed(&device, cfg.seed);
        run::<I>(cfg, cli, device)
    }
    "cuda" => {
        type I = burn_cuda::Cuda<f32, i32>;
        let device = cubecl::cuda::CudaDevice::new(cfg.device);
        <I as burn::tensor::backend::Backend>::seed(&device, cfg.seed);
        run::<I>(cfg, cli, device)
    }
    other => return Err(format!("unknown --backend {other}").into()),
}
```

(`run<I>` is the shared generic body dispatching `--mode grad`/`perturb`. If
`cfg.params` fields are not `pub`-mutable, add a setter or construct the
override before `Config` freezes — check `src/config.rs` and follow its
idiom.)

Stage-1 body (inside a `fn run_grad<I: Backend>(...)` mirroring eval.rs's
backend dispatch):

1. Load config; assert `cfg.params.use_leakance`.
2. Open `MeritGagesDataset` exactly as `bin/eval.rs` does.
3. Head: `kan_config(head_section, cfg.seed).init::<Autodiff<I>>(&device)`;
   if `--checkpoint` given, `load_kan_head` from `head_base(dir)` (the
   TRAINED point), else keep the fresh init (the COLD point). NOTE: training
   loads the head on `Autodiff<I>`; follow `bootstrap.rs`'s pattern if
   `load_kan_head`'s backend parameter differs from eval's.
4. Sampler replica of driver.rs:70-93 with a local rng:
   `let mut rng = ChaCha12Rng::seed_from_u64(cli.seed);`
   `let mut sampler = BatchSource::Shuffle(RandomSampler::new(dataset.len(), exp.batch_size, true));`
   `sampler.reshuffle(&mut rng);`
   Loop `while processed < cli.windows`, calling `sampler.next_batch()`
   (reshuffle again when it returns None — more windows than one epoch's
   batches).
5. Per batch — copy driver.rs:92-181 verbatim EXCEPT:
   - `let (pred_hourly, leaves) = probe_forward::<I>(cfg, &tensors, &head, &device);`
   - after `let loss = crate::training::batch_loss(p_filt, o_filt, &exp.loss);`
     do `let grads = loss.backward();` then read the three leaf grads:
     `let g_f: Vec<f32> = leaves.factor.grad(&grads).expect("factor grad").into_data().into_vec().unwrap();`
     (same for `d_gw`, `k_d`), and
     `let comids: Vec<i64> = batch.divide_comids.iter().map(|c| c.0).collect();`
     (keep `batch` alive: clone `divide_comids` BEFORE `batch.to_tensors`
     consumes it — mirror how eval.rs captures `reach_comids` at
     src/training/eval.rs:60).
   - `accum_factor.add(&comids, &g_f, &g_f);` etc. (one `GradAccum` per param).
   - NO optimizer step, NO checkpoint write, NO grad clip.
   - print progress: `eprintln!("batch {processed}/{}: loss={loss_f32:.5} gauges={surviving_g}");`
6. After the loop: the three accumulators share the same COMID key set
   (identical batches); take `accum_factor.into_sorted_rows()` as the master
   order, divide sums by each reach's count for means, and call
   `write_grad_netcdf`.

Also register the binary in `Cargo.toml` ONLY if other `src/bin/*.rs` files
are explicitly registered there (check; auto-discovery may cover it).

- [ ] **Step 3: Build + smoke on CPU**

Run: `cargo build --release --bin probe_zeta_gradient 2>&1 | tail -3`
Expected: clean build.

- [ ] **Step 4: Commit**

```bash
git add src/dump_parameters.rs src/bin/probe_zeta_gradient.rs Cargo.toml
git commit -m "feat(probe): stage-1 adjoint reachability binary + grad netCDF writer"
```

### Task 4: Stage-1 CPU runs (trained + cold)

No code. CPU, sequential, main-tree cwd, worktree binary by ABSOLUTE path.

- [ ] **Step 0: TIMING GATE — one window on CPU**

```bash
cd /home/tbindas/projects/ddrs
WT=/home/tbindas/projects/ddrs/.claude/worktrees/zeta-sensitivity
mkdir -p output/zeta_probe
time nice -n 10 $WT/target/release/probe_zeta_gradient \
  --config config/experiments/leakance_hourly_on.yaml \
  --checkpoint .ddrs/runs/2026-07-01T13-43-32Z-train-and-test/checkpoints/epoch_5_mb_9 \
  --windows 1 --seed 42 \
  --output /tmp/grad_timing.nc
```

Decide N from the measured per-window time T_w: `N = min(32, floor(10h / (2·T_w)))`,
floored at 8 (below 8 windows, coverage is too thin — report to the user
instead of proceeding). Record T_w and the chosen N in the task report; use
the SAME N and seed for both runs.

- [ ] **Step 1: trained point (background; ~N × T_w)**

```bash
cd /home/tbindas/projects/ddrs
nice -n 10 $WT/target/release/probe_zeta_gradient \
  --config config/experiments/leakance_hourly_on.yaml \
  --checkpoint .ddrs/runs/2026-07-01T13-43-32Z-train-and-test/checkpoints/epoch_5_mb_9 \
  --windows <N> --seed 42 \
  --output output/zeta_probe/grad_trained.nc
```

Expected: N batch lines with finite losses (order 1–10 for L1 on m³/s), then
a netCDF write. Run in background; wait for completion before step 2.

- [ ] **Step 2: cold point (identical windows: same seed/windows flags)**

```bash
cd /home/tbindas/projects/ddrs
nice -n 10 $WT/target/release/probe_zeta_gradient \
  --config config/experiments/leakance_hourly_on.yaml \
  --windows <N> --seed 42 \
  --output output/zeta_probe/grad_cold.nc
```

- [ ] **Step 3: verify both**

```bash
cd ~/projects/ddr && uv run python - <<'EOF'
import numpy as np, xarray as xr
for f in ("grad_trained", "grad_cold"):
    ds = xr.open_dataset(f"/home/tbindas/projects/ddrs/output/zeta_probe/{f}.nc")
    g = ds.grad_factor_abs.values
    nw = ds.n_windows.values
    assert np.isfinite(g).all() and (g >= 0).all()
    print(f, "reaches:", len(g), "| median |g|:", np.median(g),
          "| frac zero:", (g == 0).mean(), "| median coverage:", np.median(nw))
EOF
```

Expected: tens of thousands of reaches, median coverage ≥ 2 (N is CPU-budget
bound), frac zero < 50%. The two files must have IDENTICAL COMID sets (same
seed ⇒ same batches). If median coverage < 2 at the budgeted N, report the
coverage histogram to the user before proceeding — don't silently burn
another day of CPU.

### Task 5: Site selection + round packing (Python)

**Files:**
- Create: `scripts/zeta_probe_sites.py`

Runs under `./ddrs-py` venv (geopandas/pyogrio available there — the
`ddrs-eval-plots` skill's environment). Writes
`output/zeta_probe/probe_plan.csv` (columns: `round,comid,delta,staid_nearest,
class,stratum_uparea,stratum_aridity,stratum_reach`).

- [ ] **Step 1: Write the script**

```python
#!/usr/bin/env python3
"""Stage-2 site selection: GAGES-II Ref basins, strata, round packing.

Run: cd <ddrs>/ddrs-py && uv run python ../scripts/zeta_probe_sites.py
(or .venv/bin/python on hosts where uv can't rebuild the maturin package)
"""

from __future__ import annotations

import argparse
from collections import defaultdict
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
import xarray as xr
import zarr

DDRS = Path("/home/tbindas/projects/ddrs")
GAGES2_DBF = Path("/mnt/ssd1/data/gage_shp_files/gagesII_9322_sept30_2011.shp")
GAGES_CSV = Path("/home/tbindas/projects/ddr/references/gage_info/gages_3000.csv")
ATTRS_NC = Path("/home/tbindas/projects/ddr/data/merit_global_attributes_v2.nc")
GAGES_ADJ = Path("/home/tbindas/projects/ddr/data/merit_gages_conus_adjacency.zarr")
GRAD_NC = DDRS / "output/zeta_probe/grad_trained.nc"
DELTAS = [0.01, 0.1]
N_PROBES = 250  # per delta (CPU budget); each extra ROUND costs a full eval


def tercile_labels(x: np.ndarray) -> np.ndarray:
    lo, hi = np.nanpercentile(x, 33), np.nanpercentile(x, 67)
    return np.where(x < lo, 0, np.where(x < hi, 1, 2))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=DDRS / "output/zeta_probe/probe_plan.csv")
    ap.add_argument("--gages-adj", type=Path, default=GAGES_ADJ)
    args = ap.parse_args()

    # 1. Gauge classification: STAID → CLASS (Ref / Non-ref).
    g2 = gpd.read_file(GAGES2_DBF)[["STAID", "CLASS"]]
    g2["STAID"] = g2["STAID"].astype(str).str.lstrip("0")
    gages = pd.read_csv(GAGES_CSV, dtype={"STAID": str})
    gages["STAID_KEY"] = gages["STAID"].str.lstrip("0")
    gages = gages.merge(g2, left_on="STAID_KEY", right_on="STAID", how="left",
                        suffixes=("", "_g2"))
    gages["CLASS"] = gages["CLASS"].fillna("Unknown")
    print(gages["CLASS"].value_counts())

    # 2. Reach → containing gauges, from the per-gauge subgraph zarr.
    #    Structure discovery first — print the top-level layout, then adapt
    #    the two ATTEMPT blocks below if group/array names differ.
    root = zarr.open_group(str(args.gages_adj), mode="r")
    print("gages_adjacency layout:", list(root.group_keys())[:5], list(root.array_keys())[:5])
    reach_gauges: dict[int, list[str]] = defaultdict(list)
    gauge_comids: dict[str, np.ndarray] = {}
    for gname in root.group_keys():  # ATTEMPT: one subgroup per gauge STAID
        sub = root[gname]
        arr_name = "order" if "order" in sub else list(sub.array_keys())[0]
        comids = np.asarray(sub[arr_name][:], dtype=np.int64)
        gauge_comids[gname] = comids
        for c in comids:
            reach_gauges[int(c)].append(gname)
    assert gauge_comids, (
        "no per-gauge groups found — inspect the printed layout and adapt "
        "(the engine's GagesAdjacencyStore builds one subgraph per gauge)"
    )

    # 3. Candidate reaches: probe-covered (stage-1) ∩ subgraphs of Ref gauges.
    grad = xr.open_dataset(GRAD_NC)
    probe_comids = grad["COMID_probe"].values.astype(np.int64)
    reach_abs = dict(zip(probe_comids, grad["grad_factor_abs"].values))

    staid_class = dict(zip(gages["STAID_KEY"], gages["CLASS"]))
    staid_drain = dict(zip(gages["STAID_KEY"], gages["DRAIN_SQKM"]))

    def norm(g: str) -> str:
        return g.lstrip("0")

    rows = []
    for c in probe_comids:
        containing = reach_gauges.get(int(c), [])
        if not containing:
            continue
        classes = {staid_class.get(norm(g), "Unknown") for g in containing}
        # Ref-only population: EVERY containing gauge must be Ref (a Non-ref
        # gauge downstream would receive the perturbation through regulation).
        cls = "Ref" if classes == {"Ref"} else ("Non-ref" if "Non-ref" in classes else "Mixed")
        nearest = min(containing, key=lambda g: staid_drain.get(norm(g), np.inf))
        rows.append((int(c), cls, nearest, len(containing)))
    cand = pd.DataFrame(rows, columns=["comid", "class", "staid_nearest", "n_gauges"])
    print("candidates:", cand["class"].value_counts().to_dict())

    # 4. Strata: uparea tercile × aridity tercile × stage-1 reachability tercile.
    attrs = xr.open_dataset(ATTRS_NC)
    acom = attrs["COMID"].values.astype(np.int64)
    order = np.argsort(acom)
    pos = order[np.clip(np.searchsorted(acom, cand["comid"].values, sorter=order), 0, len(acom) - 1)]
    ok = acom[pos] == cand["comid"].values
    for name in ("log10_uparea", "aridity"):
        v = attrs[name].values.astype(float)[pos]
        v[~ok] = np.nan
        cand[name] = v
    cand["reach_abs"] = cand["comid"].map(reach_abs)
    cand["s_up"] = tercile_labels(cand["log10_uparea"].values)
    cand["s_ar"] = tercile_labels(cand["aridity"].values)
    cand["s_re"] = tercile_labels(np.log10(np.maximum(cand["reach_abs"].values, 1e-30)))

    # 5. Sample: Ref primary (equal per stratum), Non-ref contrast at 20%.
    rng = np.random.default_rng(42)
    picked = []
    ref = cand[cand["class"] == "Ref"]
    per_stratum = max(1, N_PROBES // 27)
    for (a, b, c), grp in ref.groupby(["s_up", "s_ar", "s_re"]):
        take = grp.sample(min(per_stratum, len(grp)), random_state=42)
        picked.append(take)
    nonref = cand[cand["class"] == "Non-ref"].sample(
        min(N_PROBES // 5, (cand["class"] == "Non-ref").sum()), random_state=42)
    plan = pd.concat(picked + [nonref]).drop_duplicates("comid")
    print(f"picked {len(plan)} probe reaches ({(plan['class']=='Ref').sum()} Ref)")

    # 6. Round packing: no two probes in a round may share ANY containing gauge.
    plan_rows = []
    rounds: list[set[str]] = []
    for delta in DELTAS:
        for _, r in plan.iterrows():
            gset = set(reach_gauges[int(r["comid"])])
            for k, used in enumerate(rounds):
                if not (used & gset):
                    used |= gset
                    break
            else:
                k = len(rounds)
                rounds.append(set(gset))
            plan_rows.append((k, int(r["comid"]), delta, r["staid_nearest"], r["class"],
                              int(r["s_up"]), int(r["s_ar"]), int(r["s_re"])))
    out = pd.DataFrame(plan_rows, columns=["round", "comid", "delta", "staid_nearest",
                                           "class", "stratum_uparea", "stratum_aridity",
                                           "stratum_reach"])
    print(f"{len(out)} probes packed into {out['round'].nunique()} rounds")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(args.out, index=False)
    print("wrote", args.out)


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Run it**

Run: `cd /home/tbindas/projects/ddrs/ddrs-py && (uv run python ../scripts/zeta_probe_sites.py || .venv/bin/python ../scripts/zeta_probe_sites.py)`
Expected: class counts printed; ~500–600 probe reaches; packed into ≤ ~30
rounds; `probe_plan.csv` written. If the zarr layout ATTEMPT block fails, the
printed layout tells the implementer the actual group/array names — adapt the
loop, record the fix. If `merit_gages_conus_adjacency.zarr` isn't at the
default path, take it from `gages_adjacency:` in
`/home/tbindas/projects/ddrs/ddrs.yaml`.

- [ ] **Step 3: Commit (script only; the CSV is an artifact, not source)**

```bash
cd /home/tbindas/projects/ddrs/.claude/worktrees/zeta-sensitivity
git add scripts/zeta_probe_sites.py
git commit -m "feat(scripts): stage-2 probe site selection + round packing (GAGES-II Ref)"
```

### Task 6: Stage-2 perturbation mode in the binary

**Files:**
- Modify: `src/bin/probe_zeta_gradient.rs` (add `--mode perturb`)

- [ ] **Step 1: Implement the perturb mode**

Body of `fn run_perturb<I: Backend>(...)` — a copy of `evaluate`'s chunk loop
(src/training/eval.rs:56-111) with the head loaded like `bin/eval.rs` and a
per-round q′ perturbation, NO autograd anywhere, NO zeta sink:

1. Parse `--probe-plan` CSV into `Vec<(round: usize, comid: i64, delta: f32)>`
   with the `csv` crate (already a dependency of the gage reader; check
   `Cargo.toml`, add if absent).
2. `let rounds: BTreeMap<usize, Vec<(i64, f32)>>` grouping the plan.
3. Run two UNPERTURBED baselines (empty perturbation vec) first. On the CPU
   backend these MUST be byte-identical (NdArray is deterministic) — the
   binary asserts max|baseline_1 − baseline_2| == 0 and prints it; a nonzero
   value is a bug, not a noise floor.
3b. Respect `--eval-days`: route only the first `eval_days` days of the eval
   period (cap the chunk loop's `n_days_total`; default 1095). All rounds and
   baselines use the same cap.
4. Per round: chunked eval loop; per chunk, after
   `let tensors = batch.to_tensors::<I>(device);`:

```rust
// Map this round's COMIDs to batch reach columns and add the deltas to the
// lateral-inflow forcing. Perturb BOTH the daily (disagg input) and the
// hourly flat-repeat tensor so either forcing path carries it.
// `batch_divide_comids` is cloned from `batch.divide_comids` BEFORE
// `batch.to_tensors` consumes the batch (same capture pattern as
// src/training/eval.rs:60).
let comid_col: HashMap<i64, usize> = batch_divide_comids
    .iter()
    .enumerate()
    .map(|(i, c)| (c.0, i))
    .collect();
let n_reaches = tensors.q_prime.dims()[1];
let mut delta_row = vec![0.0f32; n_reaches];
for &(comid, delta) in &round_probes {
    if let Some(&col) = comid_col.get(&comid) {
        delta_row[col] = delta;
    }
}
let delta_t: Tensor<I, 1> =
    Tensor::from_data(TensorData::new(delta_row.clone(), [n_reaches]), device);
// Clone the two forcing fields first, then functional-record-update — moving
// a field inside the same FRU expression is E0382 (partially moved value).
let q_prime = tensors.q_prime.clone() + delta_t.clone().unsqueeze_dim::<2>(0);
let q_prime_daily = tensors.q_prime_daily.clone() + delta_t.unsqueeze_dim::<2>(0);
let tensors = RoutingTensors::<I> { q_prime, q_prime_daily, ..tensors };
```

   (`unsqueeze_dim::<2>(0)` broadcasts `[N]` → `[1, N]` over the time axis;
   if BURN's broadcasting rejects it, materialize with
   `.expand([t_rows, n_reaches])` where `t_rows` is the respective dim 0.)
5. Forward via `forward_eval::<I>(cfg, &tensors, &head, device, carry_state, None)`,
   assemble `(G, T_hours)` predictions across chunks, tau-trim + daily
   downsample exactly as `evaluate` does (src/training/eval.rs:113-145).
6. Write per-round daily gauge predictions as netCDF
   `output/zeta_probe/perturb/round_<k>.nc`: dims `(gauge, day)`, vars
   `predictions (f32)`, `gage_ids` written as a `string`-typed variable if the
   netcdf crate supports it, else as a sidecar `round_<k>.gauges.txt` (one
   STAID per line — the analysis script accepts either). Baselines →
   `baseline_1.nc`, `baseline_2.nc`.

Reuse, don't duplicate: if the chunk-assembly block is copy-heavy, factor a
`fn eval_daily_predictions<I>(cfg, dataset, head, device, batch_days, perturb: &[(i64, f32)]) -> (Array2<f32>, Vec<String>)`
INSIDE the binary (not in the library — YAGNI until a second consumer exists).

- [ ] **Step 2: Build**

Run: `cargo build --release --bin probe_zeta_gradient 2>&1 | tail -3`
Expected: clean.

- [ ] **Step 3: Guard suite + sandbox (src/ was touched in this arc)**

Run: `cargo test --test zeta_gradient_probe --test leakance_gradcheck --test leakance_off_parity --test zeta_accum 2>&1 | grep "test result"`
Expected: 3, 8, 3, 6 passed.
Run: `cargo run --release --example compare_ddr_sandbox 2>&1 | tail -3`
Expected: ABSOLUTE MATCH.

- [ ] **Step 4: Commit**

```bash
git add src/bin/probe_zeta_gradient.rs Cargo.toml
git commit -m "feat(probe): stage-2 q' perturbation rounds + noise-floor baselines"
```

### Task 7: Stage-2 CPU runs

No code. Sequential CPU evals over the `--eval-days` window. Use one
background command for the whole sweep (single completion notification).

- [ ] **Step 0: TIMING GATE — baselines only**

The binary runs the 2 baselines before any round; time the FIRST baseline
(watch the `round` progress lines). Per-round cost ≈ baseline cost. If
`(n_rounds + 2) × T_round` exceeds ~36 h, cut scope in this order and record
the decision: (1) drop the δ=0.1 rows from `probe_plan.csv` (halves rounds;
δ=0.01 is the decisive bound), (2) reduce `--eval-days` to 730, (3) report to
the user with the numbers. Determinism check: the binary's
baseline_1 == baseline_2 assertion must pass (CPU is deterministic).

- [ ] **Step 1: run the sweep**

```bash
cd /home/tbindas/projects/ddrs
WT=/home/tbindas/projects/ddrs/.claude/worktrees/zeta-sensitivity
nice -n 10 $WT/target/release/probe_zeta_gradient \
  --config config/experiments/leakance_hourly_on.yaml \
  --checkpoint .ddrs/runs/2026-07-01T13-43-32Z-train-and-test/checkpoints/epoch_5_mb_9 \
  --mode perturb \
  --probe-plan output/zeta_probe/probe_plan.csv \
  --eval-days 1095 \
  --output output/zeta_probe/perturb
```

(The binary iterates the 2 baselines + all rounds internally, printing
`round k/K` progress.)

- [ ] **Step 2: verify**

```bash
ls /home/tbindas/projects/ddrs/output/zeta_probe/perturb/ | head
cd ~/projects/ddr && uv run python - <<'EOF'
import xarray as xr, numpy as np
b1 = xr.open_dataset("/home/tbindas/projects/ddrs/output/zeta_probe/perturb/baseline_1.nc")
b2 = xr.open_dataset("/home/tbindas/projects/ddrs/output/zeta_probe/perturb/baseline_2.nc")
noise = np.abs(b1.predictions.values - b2.predictions.values)
print("determinism check (must be 0 on CPU): max", noise.max())
EOF
```

Expected: `round_*.nc` for every plan round + both baselines; on the CPU
backend the baseline diff must be exactly 0 (determinism). If it is nonzero,
STOP — that's a bug in the perturb loop (e.g. rng consumed unevenly), not a
noise floor.

### Task 8: Analysis script + verdicts

**Files:**
- Create: `scripts/zeta_gradient_analysis.py`

Runs under ddr's uv venv. Consumes `grad_trained.nc`, `grad_cold.nc`,
`probe_plan.csv`, `perturb/*.nc`, attributes, gages CSV. Implements the
spec's pre-registered criteria verbatim:

- [ ] **Step 1: Write the script**

```python
#!/usr/bin/env python3
"""Zeta gradient probe — verdicts.

H4-starvation: median |dL/dfactor| ungauged (and arid) >= 1-2 OOM below gauged,
               at BOTH parameter points.
H4-rejection:  magnitudes comparable off-gauge but signed grad pushes zeta down
               (dL/dfactor > 0 dominant) on arid/ungauged reaches.
Detectability NO-GO: <10% of Ref probes at delta=0.01 clear noise floor AND the
               5% obs band.
Cross-check:   stage-1 reachability rank-predicts stage-2 detectability.

Run: cd ~/projects/ddr && uv run python <ddrs>/scripts/zeta_gradient_analysis.py
"""

from __future__ import annotations

import argparse
import csv as csvmod
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr

OUT = Path("/home/tbindas/projects/ddrs/output/zeta_probe")
GAGES_CSV = Path("/home/tbindas/projects/ddr/references/gage_info/gages_3000.csv")
ATTRS_NC = Path("/home/tbindas/projects/ddr/data/merit_global_attributes_v2.nc")


def sec(t: str) -> None:
    print(f"\n{'=' * 72}\n{t}\n{'=' * 72}")


def attach(attrs_path: Path, comids: np.ndarray, names: list[str]) -> dict[str, np.ndarray]:
    ds = xr.open_dataset(attrs_path)
    acom = ds["COMID"].values.astype(np.int64)
    order = np.argsort(acom)
    pos = order[np.clip(np.searchsorted(acom, comids, sorter=order), 0, len(acom) - 1)]
    ok = acom[pos] == comids
    out = {}
    for n in names:
        v = ds[n].values.astype(float)[pos]
        v[~ok] = np.nan
        out[n] = v
    return out


def load_preds(path: Path) -> tuple[np.ndarray, list[str]]:
    ds = xr.open_dataset(path)
    preds = ds["predictions"].values
    if "gage_ids" in ds:
        gids = [str(x) for x in ds["gage_ids"].values]
    else:
        gids = (path.parent / (path.stem + ".gauges.txt")).read_text().split()
    return preds, gids


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--probe-dir", type=Path, default=OUT)
    args = ap.parse_args()
    verdicts = []

    # ---------- Stage 1: reachability ----------
    tr = xr.open_dataset(args.probe_dir / "grad_trained.nc")
    co = xr.open_dataset(args.probe_dir / "grad_cold.nc")
    comids = tr["COMID_probe"].values.astype(np.int64)
    assert (comids == co["COMID_probe"].values).all(), "trained/cold COMID sets differ"

    gages = pd.read_csv(GAGES_CSV)
    gauged = np.isin(comids, gages["COMID"].values.astype(np.int64))
    attrs = attach(ATTRS_NC, comids, ["aridity", "meanP", "log10_uparea"])
    dry_ix = attrs["aridity"] >= np.nanpercentile(attrs["aridity"], 67)
    # Orientation check as in leakance_diagnosis.py:
    from scipy.stats import spearmanr
    r = spearmanr(attrs["aridity"], attrs["meanP"], nan_policy="omit").statistic
    if r > 0:  # aridity is a wetness index here; flip
        dry_ix = attrs["aridity"] <= np.nanpercentile(attrs["aridity"], 33)
    print(f"aridity-vs-meanP spearman {r:+.2f}; dry tercile n={dry_ix.sum()}")

    sec("Stage 1 — |dL/dfactor| by stratum, trained vs cold")
    ratios = {}
    for label, ds in (("trained", tr), ("cold", co)):
        g = ds["grad_factor_abs"].values
        m_g, m_u = np.median(g[gauged]), np.median(g[~gauged])
        m_dry, m_wet = np.median(g[dry_ix]), np.median(g[~dry_ix & np.isfinite(attrs["aridity"])])
        ratios[label] = (m_g / max(m_u, 1e-300), m_wet / max(m_dry, 1e-300))
        print(f"{label:8s} gauged={m_g:.3e} ungauged={m_u:.3e} (ratio {ratios[label][0]:.1f})"
              f" | dry={m_dry:.3e} wet={m_wet:.3e}")
        net = ds["grad_factor_net"].values
        # dL/dfactor > 0 means the loss wants LESS leakance.
        for sl, m in (("ungauged", ~gauged), ("dry", dry_ix)):
            frac_down = (net[m] > 0).mean()
            print(f"  {label}/{sl}: frac pushing zeta DOWN = {frac_down * 100:.1f}%")

    starv = all(r[0] >= 10 for r in ratios.values())
    verdicts.append(("H4-starvation",
                     "SUPPORTED" if starv else "REFUTED",
                     f"gauged/ungauged |g| ratio trained={ratios['trained'][0]:.1f}, "
                     f"cold={ratios['cold'][0]:.1f} (bar: >=10 at both points)"))

    tr_net_dry = tr["grad_factor_net"].values[dry_ix]
    reject = (not starv) and (tr_net_dry > 0).mean() > 0.67
    verdicts.append(("H4-rejection",
                     "SUPPORTED" if reject else ("N/A (starvation holds)" if starv else "REFUTED"),
                     f"{(tr_net_dry > 0).mean() * 100:.1f}% of dry-tercile grads push zeta down"))

    # ---------- Stage 2: detectability ----------
    sec("Stage 2 — planted-delta detectability at nearest Ref gauges")
    plan = pd.read_csv(args.probe_dir / "probe_plan.csv", dtype={"staid_nearest": str})
    b1, gids = load_preds(args.probe_dir / "perturb/baseline_1.nc")
    b2, _ = load_preds(args.probe_dir / "perturb/baseline_2.nc")
    gid_ix = {g.lstrip("0"): i for i, g in enumerate(gids)}
    # CPU backend is deterministic: noise should be exactly 0 and detection
    # reduces to the obs-uncertainty band. The noise term stays in the
    # criterion so the same script scores CUDA-produced runs unchanged.
    noise = np.abs(b1 - b2)
    print("baseline determinism: max |b1-b2| =", float(noise.max()))

    rows = []
    for rnd, grp in plan.groupby("round"):
        pr, _ = load_preds(args.probe_dir / f"perturb/round_{rnd}.nc")
        dq = pr - b1
        for _, p in grp.iterrows():
            i = gid_ix.get(str(p["staid_nearest"]).lstrip("0"))
            if i is None:
                continue
            mean_dq = float(np.nanmean(dq[i]))
            peak_dq = float(np.nanmax(np.abs(dq[i])))
            nf = float(np.nanpercentile(noise[i], 99))
            band5 = 0.05 * float(np.nanmean(b1[i]))
            rows.append(dict(comid=p["comid"], delta=p["delta"], cls=p["class"],
                             mean_dq=mean_dq, peak_dq=peak_dq, noise=nf, band5=band5,
                             detect=(abs(mean_dq) > nf) and (abs(mean_dq) > band5),
                             s_re=p["stratum_reach"]))
    det = pd.DataFrame(rows)
    for (cls, delta), grp in det.groupby(["cls", "delta"]):
        print(f"{cls:8s} delta={delta}: detectable {grp['detect'].mean() * 100:.1f}% of {len(grp)}")
    ref001 = det[(det["cls"] == "Ref") & (det["delta"] == 0.01)]
    nogo = ref001["detect"].mean() < 0.10
    verdicts.append(("Detectability",
                     "NO-GO" if nogo else "GO",
                     f"{ref001['detect'].mean() * 100:.1f}% of Ref probes at delta=0.01 detectable "
                     "(NO-GO bar: <10%)"))

    # Cross-check: stage-1 reachability rank-predicts detectability.
    from scipy.stats import spearmanr as sp2
    grad_map = dict(zip(comids, tr["grad_factor_abs"].values))
    det["reach_abs"] = det["comid"].map(grad_map)
    rc = sp2(det["reach_abs"], det["detect"].astype(float), nan_policy="omit").statistic
    print(f"\ncross-check: spearman(reachability, detected) = {rc:+.2f}")
    verdicts.append(("Cross-check", "PASS" if rc > 0.3 else "SUSPECT",
                     f"rank corr {rc:+.2f} (bar: > 0.3)"))

    sec("VERDICTS")
    for name, v, detail in verdicts:
        print(f"  [{v}] {name}: {detail}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Run it, tee the output**

Run: `cd ~/projects/ddr && uv run python /home/tbindas/projects/ddrs/.claude/worktrees/zeta-sensitivity/scripts/zeta_gradient_analysis.py | tee /tmp/zeta_gradient_verdicts.txt`
Expected: all sections print; 4 verdict lines. Fix only concrete runtime
errors; thresholds and formulas stay as written; record fixes.

- [ ] **Step 3: Commit**

```bash
git add scripts/zeta_gradient_analysis.py
git commit -m "feat(scripts): gradient-probe verdict analysis (starvation/rejection/detectability)"
```

### Task 9: CONUS gradient maps (ddrs-eval-plots conventions)

**Files:**
- Create: `/home/tbindas/projects/ddrs/output/zeta_probe/plots/gradient_maps.ipynb` (artifact, not committed)
- The generating agent MUST read `.claude/skills/ddrs-eval-plots/SKILL.md` and
  `references/parameter_map.md` first and follow their conventions (venv at
  `./ddrs-py`, CONUS bounds `(-125, -66) × (24, 53)`, CartoDB.Positron alpha
  0.6, `dpi=300, bbox_inches="tight", facecolor="white"`).

- [ ] **Step 1: Notebook cell plan (per-gauge scatter — complete code)**

Markdown header cell (checkpoint, inputs, date), then:

```python
import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xarray as xr
import contextily as ctx

OUT = "/home/tbindas/projects/ddrs/output/zeta_probe"
tr = xr.open_dataset(f"{OUT}/grad_trained.nc")
co = xr.open_dataset(f"{OUT}/grad_cold.nc")
gages = pd.read_csv("/home/tbindas/projects/ddr/references/gage_info/gages_3000.csv")

# Per-gauge aggregate: median |g| over the gauge's subgraph. The subgraph
# membership map is rebuilt exactly as in scripts/zeta_probe_sites.py (zarr
# per-gauge groups); reuse that code block here.
# ... (membership: staid -> comids; grad lookup: comid -> value)

def gauge_agg(ds, staid_comids, var="grad_factor_abs"):
    lut = dict(zip(ds["COMID_probe"].values, ds[var].values))
    return {s: np.median([lut[c] for c in cs if c in lut] or [np.nan])
            for s, cs in staid_comids.items()}

fig, axes = plt.subplots(2, 2, figsize=(16, 10))
panels = [("trained |g|", tr, "grad_factor_abs", "viridis", True),
          ("trained sign", tr, "grad_factor_net", "coolwarm", False),
          ("cold |g|", co, "grad_factor_abs", "viridis", True),
          ("cold sign", co, "grad_factor_net", "coolwarm", False)]
for ax, (title, ds, var, cmap, logscale) in zip(axes.flat, panels):
    agg = gauge_agg(ds, staid_comids, var)
    df = gages.assign(val=gages["STAID"].astype(str).map(agg)).dropna(subset=["val"])
    c = np.log10(np.maximum(np.abs(df["val"]), 1e-30)) if logscale else np.sign(df["val"])
    gdf = gpd.GeoDataFrame(df, geometry=gpd.points_from_xy(df.LNG_GAGE, df.LAT_GAGE),
                           crs="EPSG:4326").to_crs(epsg=3857)
    gdf.plot(ax=ax, column=c.values if hasattr(c, "values") else c, cmap=cmap,
             markersize=8, legend=True)
    ctx.add_basemap(ax, source=ctx.providers.CartoDB.Positron, alpha=0.6, attribution="")
    ax.set_title(title)
    ax.set_axis_off()
fig.suptitle("dL1/d(leakance_factor): reachability across CONUS gauges")
fig.savefig(f"{OUT}/plots/gradient_gauge_map.png", dpi=300, bbox_inches="tight",
            facecolor="white")
```

Second notebook section: per-reach MERIT-polygon map of
`log10(grad_factor_abs)` following the `parameter_map` reference template
(that reference contains the complete polygon-join recipe; substitute the
variable and the `COMID_probe` dimension).

- [ ] **Step 2: Execute per the skill's workflow**

```bash
cd /home/tbindas/projects/ddrs/ddrs-py && (uv sync --extra plots || true)
cd /home/tbindas/projects/ddrs/ddrs-py && (uv run jupyter nbconvert --to notebook --execute \
  /home/tbindas/projects/ddrs/output/zeta_probe/plots/gradient_maps.ipynb \
  --output gradient_maps --output-dir /home/tbindas/projects/ddrs/output/zeta_probe/plots \
  || .venv/bin/jupyter nbconvert --to notebook --execute \
  /home/tbindas/projects/ddrs/output/zeta_probe/plots/gradient_maps.ipynb \
  --output gradient_maps --output-dir /home/tbindas/projects/ddrs/output/zeta_probe/plots)
ls /home/tbindas/projects/ddrs/output/zeta_probe/plots/*.png
```

Expected: `gradient_gauge_map.png` + the polygon map PNG. List every PNG path
in the report (skill requirement).

### Task 10: Findings report

**Files:**
- Create: `docs/2026-07-0X-zeta-gradient-probe-findings.md` (date = the day
  the battery completes; fill every number from `/tmp/zeta_gradient_verdicts.txt`
  and the run logs — no placeholders may survive)

- [ ] **Step 1: Write in the established report format**

Sections (mirroring `docs/2026-07-02-leakance-diagnosis-findings.md`):
1. Motivating question + the two rival mechanisms (starvation vs rejection)
2. Methods (stage-1 replica of the training loop, leaf-lifting, two parameter
   points; stage-2 site selection with the GAGES-II Ref filter, round
   packing, noise-floor baselines)
3. Results (verdict table + stratum tables + the two CONUS maps embedded by
   relative path)
4. Conclusions (which mechanism; what it means for the auxiliary-supervision
   remedy; whether gauge-only leakance training is viable at all)
5. Next steps
6. Raw verdict output (fenced)
7. Reproduce (exact commands from Tasks 4, 5, 7, 8, 9)

- [ ] **Step 2: Commit**

```bash
git add docs/2026-07-0*-zeta-gradient-probe-findings.md
git commit -m "docs: zeta gradient probe findings — <verdict summary>"
```

---

## Execution notes for sub-agents

- Rust: this worktree only. GPU: `cd /home/tbindas/projects/ddrs` + ABSOLUTE
  worktree binary paths (stale-binary memory). Python analysis: ddr venv;
  site-selection + notebooks: `./ddrs-py` venv.
- ALL runs are CPU (`--backend cpu`, the default) — the GPU belongs to another
  training job. Long jobs strictly sequential, `nice -n 10`, launched as ONE
  background command per sweep. Every heavy task starts with its TIMING GATE;
  never extrapolate past ~36 h without reporting to the user first.
- If any guard (gradcheck / off-parity / zeta_accum / sandbox) fails: STOP.
- The probe binary must never take an optimizer step — if you find yourself
  importing `optimizer.rs`, you've drifted from the spec.
- Numbers in the findings doc must be transcribed from the tee'd verdict file;
  a reviewer will diff them.

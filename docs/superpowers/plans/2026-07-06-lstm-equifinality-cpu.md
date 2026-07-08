# LSTM-Source Selective-Equifinality (CPU Arms 1–3) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Train the same MERIT CONUS routing model under 3 LSTM-derived Q′ arms on CPU, then measure parameter convergence (n vs geometry) at four levels including gradient alignment.

**Architecture:** Add a `--backend {cuda,cpu}` flag to `ddrs run` (workflow internals are already backend-generic); generalize the `probe_zeta_gradient` binary to lift arbitrary KAN-head outputs (`--params n,q_spatial,p_spatial`) instead of only the leakance trio; author three tracked experiment configs; run the arms sequentially via detached scripts; analyze cross-arm convergence with a new Python script.

**Tech Stack:** Rust (BURN 0.21, NdArray backend, clap, netcdf crate), YAML configs, Python via `uv` (numpy, scipy, xarray, zarr, icechunk, netCDF4).

**Spec:** `docs/superpowers/specs/2026-07-06-lstm-equifinality-cpu-design.md`
**Branch:** stay on `unit_catchments` (the runs depend on PR #24's LSTM-store support; do NOT branch off master).

**Standing rules for every task:**
- After ANY `src/` change: `cargo install --path .` (STALE-BINARY TRAP — `cargo build` does NOT update `~/.cargo/bin/ddrs`).
- Directory-style checkpoints (`epoch_E_mb_M/head.mpk`) = current binary; flat `.mpk` files = stale binary, invalid run.
- Never `TaskStop`/kill a shell whose command IS the compute. Verify a task is a `tail`/watch before stopping it.
- Multi-hour compute runs via detached `nohup` scripts, not foreground agent shells.

---

## File Structure

| File | Action | Responsibility |
|---|---|---|
| `src/bin/ddrs.rs` | Modify (~:73-90, ~:210-224) | `--backend` clap arg on `run`, plumb to `RunInput` |
| `src/cli/run.rs` | Modify | `RunInput.backend`, backend-generic dispatch, gated GPU pre-flight, backend-aware `--plot` post-step |
| `tests/cli_run_preflight.rs` | Modify | preflight gated on backend |
| `src/training/probe.rs` | Modify | named lifted leaves (`ProbeLeaves` generalization), conditional leakance lifting |
| `src/bin/probe_zeta_gradient.rs` | Modify | `--params` flag, generic grad extraction/accumulation |
| `src/dump_parameters.rs` | Modify | `write_param_grad_netcdf` (generic `grad_<name>_{abs,net}` writer; legacy leakance writer untouched) |
| `config/experiments/equif_daily_lstm_flat.yaml` | Create | arm R1 |
| `config/experiments/equif_daily_lstm_disagg.yaml` | Create | arm R2 |
| `config/experiments/equif_hourly_lstm.yaml` | Create | arm R3 |
| `scripts/run_equif_arms.sh` | Create | sequential detached run driver |
| `scripts/equif_convergence_analysis.py` | Create | 4-level cross-arm analysis (run from `~/projects/ddr` venv) |
| `docs/2026-07-XX-lstm-equifinality-findings.md` | Create (last task) | findings doc |

---

### Task 1: `--backend {cuda,cpu}` on `ddrs run`

**Files:**
- Modify: `src/bin/ddrs.rs:73-90` (Run subcommand args), `src/bin/ddrs.rs:210-224` (RunInput construction)
- Modify: `src/cli/run.rs:24-35` (RunInput), `:51-66` (pre-flight), `:103-124` (plot post-step), `:255-262` (dispatch)
- Test: `tests/cli_run_preflight.rs`

- [ ] **Step 1: Update the pre-flight test to be backend-aware (failing first)**

In `tests/cli_run_preflight.rs`, the existing test `run_train_requires_gpu_when_none_probed` constructs a `RunInput`. Add `backend: "cuda".into()` to that construction (it won't compile yet — that's the failing state), and add a second test asserting the CPU path does NOT hit the GPU pre-flight error:

```rust
#[test]
fn run_train_cpu_skips_gpu_preflight() {
    // Same setup as run_train_requires_gpu_when_none_probed, but
    // backend: "cpu". The run may fail later (missing data in the tmp
    // workspace) — assert only that the failure is NOT the GPU pre-flight
    // message.
    // ... same workspace/config scaffolding as the sibling test ...
    let err = ddrs::cli::run::run(ddrs::cli::run::RunInput {
        // ... same fields as sibling test ...
        backend: "cpu".into(),
    })
    .unwrap_err();
    let msg = err.to_string();
    assert!(
        !msg.contains("requires a CUDA GPU"),
        "cpu backend must skip GPU pre-flight, got: {msg}"
    );
}
```

- [ ] **Step 2: Run tests to verify they fail to compile**

Run: `cargo test --test cli_run_preflight 2>&1 | tail -20`
Expected: compile error — `RunInput` has no field `backend`.

- [ ] **Step 3: Add the flag and plumb it**

`src/bin/ddrs.rs` — in the `Run` subcommand struct add:

```rust
/// Backend for training/evaluation: "cuda" (default) or "cpu"
/// (NdArray, deterministic; sparse_solver forced to cpu).
#[arg(long, default_value = "cuda")]
backend: String,
```

and pass `backend` through in the `Cmd::Run { .. }` match arm into `RunInput`.

`src/cli/run.rs` — add `pub backend: String,` to `RunInput`. Validate early in `run()` (before workspace mutation):

```rust
if !matches!(input.backend.as_str(), "cpu" | "cuda") {
    return Err(CliError::Runtime(format!(
        "unknown --backend {} (expected \"cpu\" or \"cuda\")",
        input.backend
    )));
}
```

Gate the GPU pre-flight (run.rs:51-66) on `input.backend == "cuda"`, and extend its error text with `Use --backend cpu for CPU-only operation.`

- [ ] **Step 4: Make dispatch backend-generic**

Refactor `dispatch()` (run.rs:255): move its entire current body (from `let result = std::panic::catch_unwind(...)` to the end) into a new generic function, changing NOTHING inside except (a) delete the `type I = Cuda<f32, i32>;` line, (b) replace every `let device = cubecl::cuda::CudaDevice::new(pr.config.device);` with the `device` parameter (clone it — `I::Device: Clone`), (c) after each `Config::from_yaml_file_with_mode(...)` call add the CPU config mutation:

```rust
fn dispatch(
    input: &RunInput,
    pr: &PlanResult,
    run_dir: &Path,
) -> (RunStatus, Option<String>, serde_json::Value, RunOutputs) {
    match input.backend.as_str() {
        "cpu" => {
            type I = burn::backend::NdArray<f32>;
            let device = <I as burn::tensor::backend::BackendTypes>::Device::default();
            eprintln!("backend: cpu (NdArray, deterministic; sparse_solver forced to cpu)");
            dispatch_backend::<I>(input, pr, run_dir, device, true)
        }
        // "cuda" — validated in run()
        _ => {
            type I = Cuda<f32, i32>;
            let device = cubecl::cuda::CudaDevice::new(pr.config.device);
            dispatch_backend::<I>(input, pr, run_dir, device, false)
        }
    }
}

fn dispatch_backend<I>(
    input: &RunInput,
    pr: &PlanResult,
    run_dir: &Path,
    device: <I as burn::tensor::backend::BackendTypes>::Device,
    force_cpu: bool,
) -> (RunStatus, Option<String>, serde_json::Value, RunOutputs)
where
    I: burn::tensor::backend::Backend + burn::tensor::backend::BackendTypes,
    // copy any additional bounds the body requires from
    // bootstrap_head_and_state / training_train / evaluate signatures
    // (src/training/bootstrap.rs:43-54 needs Autodiff<I>: AutodiffBackend<InnerBackend = I>)
    burn::backend::Autodiff<I>:
        burn::tensor::backend::AutodiffBackend<InnerBackend = I>,
{
    // ... moved body ...
}
```

CPU config mutation, inserted after EACH `Config::from_yaml_file_with_mode` in the moved body (both the Train and TrainAndTest arms load `train_cfg`, TrainAndTest also loads an eval cfg):

```rust
if force_cpu {
    train_cfg.params.sparse_solver = crate::config::SparseSolver::Cpu;
    train_cfg.params.use_cuda_graphs = false;
}
```

(Exact import path for `SparseSolver`: `src/config.rs:384` — mirror `src/bin/train.rs:60-66`.)

- [ ] **Step 5: Make the `--plot` post-step backend-aware**

run.rs:103-124 — replace the hardcoded `type I = burn_cuda::Cuda<f32, i32>; let device = ...;` + `dump::<I>` call with a match on `input.backend` (cpu → NdArray + default device, cuda → current code), identical shape to Step 4's match.

- [ ] **Step 6: Compile and run the test file**

Run: `cargo test --test cli_run_preflight`
Expected: both tests PASS (existing test with `backend: "cuda"`, new test not hitting the GPU message).

- [ ] **Step 7: Full gates**

```bash
cargo test 2>&1 | tail -5                      # full suite green
cargo run --release --example compare_ddr_sandbox | grep -E "ABSOLUTE|max"
```
Expected: 0 failures; `ABSOLUTE MATCH`. (No routing-core files touched, but the gate is cheap.)

- [ ] **Step 8: Reinstall + commit**

```bash
cargo install --path . 2>&1 | tail -2
ddrs run --help | grep -A1 backend    # flag visible in installed binary
git add src/bin/ddrs.rs src/cli/run.rs tests/cli_run_preflight.rs
git commit -m "feat(cli): --backend {cuda,cpu} on ddrs run

CPU arm uses NdArray (deterministic), forces sparse_solver=cpu and
use_cuda_graphs=false, and skips the GPU pre-flight. Workflow internals
were already backend-generic; dispatch/plot-post-step now bind the
backend at runtime."
```

---

### Task 2: Generalize the gradient probe (`--params`)

**Files:**
- Modify: `src/training/probe.rs` (ProbeLeaves → named leaves; conditional lifting)
- Modify: `src/bin/probe_zeta_gradient.rs` (`--params` flag; generic extraction)
- Modify: `src/dump_parameters.rs` (new `write_param_grad_netcdf`)

Background (verified 2026-07-06): the probe currently asserts `use_leakance: true` (`probe_zeta_gradient.rs:225-228`, `probe.rs:46`) and lifts exactly `K_D`/`d_gw`/`leakance_factor` via `lift_leaf` (`probe.rs:81-85`: `Tensor::from_inner(t.inner()).require_grad()`). Gradients are w.r.t. NORMALIZED [0,1] params, extracted after `loss.backward()` (`probe_zeta_gradient.rs:355-361`), accumulated per COMID by `GradAccum` (`probe.rs:119-137`), written by `write_grad_netcdf` (`dump_parameters.rs:665-748`).

- [ ] **Step 1: Generalize `ProbeLeaves` to named leaves**

In `src/training/probe.rs`:

```rust
pub struct ProbeLeaves<I: Backend> {
    /// (head output name, lifted normalized tensor) — order matches the
    /// `lift` argument to probe_forward.
    pub leaves: Vec<(String, Tensor<Autodiff<I>, 1>)>,
}

impl<I: Backend> ProbeLeaves<I> {
    pub fn get(&self, name: &str) -> &Tensor<Autodiff<I>, 1> {
        &self.leaves
            .iter()
            .find(|(n, _)| n == name)
            .unwrap_or_else(|| panic!("probe leaf {name} not lifted"))
            .1
    }
}
```

Change `probe_forward`'s signature to take `lift: &[String]`, and build leaves from the head's `params_map` generically:

```rust
let leaves = ProbeLeaves {
    leaves: lift
        .iter()
        .map(|name| {
            let t = params_map
                .get(name.as_str())
                .unwrap_or_else(|| panic!("head does not emit `{name}` — is it in kan_head.learnable_parameters?"))
                .clone();
            (name.clone(), lift_leaf::<I>(t))
        })
        .collect(),
};
```

The lifted tensor must REPLACE the original in what flows onward (this is how the leakance trio works today — the lifted leaf is what `SpatialParameters` receives). Mirror the existing wiring: wherever `probe.rs` currently passes `leaves.k_d`/`leaves.d_gw`/`leaves.factor` into `SpatialParameters`, look the tensor up from the new `leaves` vec by name when present, else fall back to the un-lifted `params_map` entry (for `n`/`q_spatial`/`p_spatial`, assemble `SpatialParameters` exactly as `src/training/driver.rs` does, substituting lifted leaves for lifted names).

Relax the leakance assertion: `probe.rs:46` becomes conditional — assert `cfg.params.use_leakance` ONLY if `lift` contains any of `K_D`/`d_gw`/`leakance_factor`.

- [ ] **Step 2: Update the existing probe modes mechanically**

All existing call sites (`grad`, `perturb`, `teacher`, `floor`, `state-cache` modes in `probe_zeta_gradient.rs`) compile against the new API by passing `lift = ["K_D", "d_gw", "leakance_factor"]` and reading `leaves.get("K_D")` etc. where they previously read `.k_d`. No behavior change.

Run: `cargo build --release --bin probe_zeta_gradient`
Expected: compiles clean.

- [ ] **Step 3: Add the `--params` flag and generic grad-mode extraction**

In `probe_zeta_gradient.rs`'s Cli struct:

```rust
/// Comma-separated KAN-head outputs to probe (default: the leakance trio).
/// Example: --params n,q_spatial,p_spatial
#[arg(long)]
params: Option<String>,
```

At grad-mode entry, build the lift list and relax the top-level assert (`:225-228`):

```rust
let lift: Vec<String> = match &cli.params {
    Some(s) => s.split(',').map(|p| p.trim().to_string()).collect(),
    None => vec!["K_D".into(), "d_gw".into(), "leakance_factor".into()],
};
let wants_leakance = lift.iter().any(|p| matches!(p.as_str(), "K_D" | "d_gw" | "leakance_factor"));
if wants_leakance {
    assert!(cfg.params.use_leakance, "probing leakance params requires params.use_leakance: true");
}
```

Replace the three hardcoded grad extractions (`:355-361`) with a loop over `lift`, one `GradAccum` per name:

```rust
let grads = loss.backward();
for name in &lift {
    let g: Vec<f32> = leaves
        .get(name)
        .grad(&grads)
        .unwrap_or_else(|| panic!("no grad for {name}"))
        .into_data()
        .into_vec()
        .unwrap();
    accums.get_mut(name).unwrap().add(&comids, &g, &g);
}
```

(`accums: HashMap<String, GradAccum>` initialized before the window loop. `GradAccum::add` takes `(comids, abs-source, signed-source)` — pass `&g` for both; it applies `.abs()` internally to the first, per `probe.rs:119-137`.)

- [ ] **Step 4: Generic NetCDF writer**

In `src/dump_parameters.rs`, add (modeled directly on `write_grad_netcdf` at `:665-748` — same dims, attrs, and accumulator-to-mean logic; do NOT modify the legacy function, the leakance analysis scripts read its variable names):

```rust
/// Write per-reach mean |grad| / mean signed grad for an arbitrary set of
/// probed KAN-head outputs. Variables: grad_<name>_abs, grad_<name>_net
/// (f32, dimensionless — gradients are w.r.t. NORMALIZED params), plus
/// COMID_probe (i64) and n_windows (i32).
pub fn write_param_grad_netcdf(
    path: &Path,
    accums: &[(String, &GradAccum)],
    checkpoint_label: &str,
    n_batches: u32,
    seed: u64,
) -> Result<(), Box<dyn std::error::Error>> {
    // union of COMIDs across accums, sorted ascending (mirror :665-700)
    // one createVariable pair per accum name
    // identical global attrs to write_grad_netcdf, plus probe_params attr
    // listing the probed names.
    ...
}
```

Grad mode picks the writer: `--params` given → `write_param_grad_netcdf`; absent → legacy `write_grad_netcdf` (byte-identical legacy behavior).

- [ ] **Step 5: Compile + full suite + parity gates**

```bash
cargo build --release --bin probe_zeta_gradient
cargo test 2>&1 | tail -5
cargo test --test leakance_gradcheck --test leakance_off_parity --test zeta_accum 2>&1 | tail -3
cargo run --release --example compare_ddr_sandbox | grep ABSOLUTE
```
Expected: all green, `ABSOLUTE MATCH`. (probe.rs is not a gated file, but it assembles `SpatialParameters` — the off-parity and gradcheck tests guard the blast radius.)

- [ ] **Step 6: Manual smoke — one window, routing params, leakance-off config**

Uses R1's config from Task 3; if executing tasks in order, defer this step until Task 3 Step 3 is done, then:

```bash
mkdir -p output/equif_probe
target/release/probe_zeta_gradient \
  --config config/experiments/equif_daily_lstm_flat.yaml \
  --windows 1 --seed 42 --backend cpu \
  --params n,q_spatial,p_spatial \
  --output output/equif_probe/smoke.nc
python3 -c "
import netCDF4
ds = netCDF4.Dataset('output/equif_probe/smoke.nc')
for v in ['grad_n_abs','grad_q_spatial_abs','grad_p_spatial_abs']:
    assert v in ds.variables, v
print('vars ok, reaches:', ds.dimensions['COMID_probe'].size)"
```
Expected: `vars ok, reaches: <nonzero>` (cold-init head — no checkpoint needed for the smoke).

- [ ] **Step 7: Commit**

```bash
git add src/training/probe.rs src/bin/probe_zeta_gradient.rs src/dump_parameters.rs
git commit -m "feat(probe): --params flag lifts arbitrary KAN-head outputs for gradient probing

ProbeLeaves generalized to named leaves; leakance assert now conditional
on probing leakance params; generic grad_<name>_{abs,net} NetCDF writer.
Legacy leakance grad mode (no --params) byte-identical."
```

---

### Task 3: Three tracked experiment configs

**Files:**
- Create: `config/experiments/equif_daily_lstm_flat.yaml`
- Create: `config/experiments/equif_daily_lstm_disagg.yaml`
- Create: `config/experiments/equif_hourly_lstm.yaml`

Rules baked in: `data_sources` blocks are STRUCTURALLY identical to `config/sources/daily-lstm.yaml` / `hourly-lstm.yaml` so run IDs get the group tag; `sparse_solver: cpu` + `use_cuda_graphs: false` (CPU arms); no leakance; the only deltas between the three files are `streamflow:` path and the `disaggregation:` block.

- [ ] **Step 1: Write `config/experiments/equif_daily_lstm_flat.yaml`**

```yaml
# equif_daily_lstm_flat.yaml — selective-equifinality arm R1
# Q': daily CudaLSTM (NH), flat repeat-24 to hourly (NO disaggregation head).
# Spec: docs/superpowers/specs/2026-07-06-lstm-equifinality-cpu-design.md
# Constants across all equif_* arms: seed 42, 5 epochs, rho 90, warmup 5,
# L1 loss, train 1981/10/01-1995/09/30, eval 1995/10/01-2010/09/30, CPU.

mode: training
workflow: train-and-test
geodataset: merit
device: 0
seed: 42
np_seed: 42

data_sources:
  attributes: /home/tbindas/projects/ddr/data/merit_global_attributes_v2.nc
  conus_adjacency: /home/tbindas/projects/ddr/data/merit_conus_adjacency.zarr
  gages_adjacency: /home/tbindas/projects/ddr/data/merit_gages_conus_adjacency.zarr
  streamflow: /mnt/ssd1/data/icechunk/daily_lstm_merit_unit_catchments.ic
  observations: /mnt/ssd1/data/icechunk/usgs_daily_observations
  gages: /home/tbindas/projects/ddr/references/gage_info/gages_3000.csv
  aorc_precip: /mnt/ssd1/data/aorc/merit_unit_catchments.zarr

experiment:
  batch_size: 64
  start_time: 1981/10/01
  end_time: 1995/09/30
  epochs: 5
  rho: 90
  shuffle: true
  warmup: 5
  learning_rate:
    1: 0.001
    3: 0.0005
  grad_clip_max_norm: 1.0

kan_head:
  hidden_size: 21
  num_hidden_layers: 2
  grid: 50
  k: 2
  input_var_names:
    - SoilGrids1km_clay
    - aridity
    - meanelevation
    - meanP
    - NDVI
    - meanslope
    - log10_uparea
    - SoilGrids1km_sand
    - ETPOT_Hargr
    - Porosity
  learnable_parameters:
    - n
    - q_spatial
    - p_spatial

params:
  parameter_ranges:
    n: [0.015, 0.25]
    q_spatial: [0.0, 1.0]
    p_spatial: [1.0, 200.0]
  attribute_minimums:
    discharge: 1.0e-4
    slope: 1.0e-3
    velocity: 0.01
    depth: 0.01
    bottom_width: 0.01
  defaults:
    p_spatial: 21.0
  log_space_parameters:
    - p_spatial
  sparse_solver: cpu
  use_cuda_graphs: false

testing:
  start_time: 1995/10/01
  end_time: 2010/09/30
  batch_size: 15
  rho: null
```

- [ ] **Step 2: Write `config/experiments/equif_daily_lstm_disagg.yaml`**

Identical to Step 1 except the header comment (arm R2, precip-driven disaggregation) and ONE addition at the end of `kan_head:`:

```yaml
  # Precip-conditioned mass-preserving daily->hourly disaggregation head.
  disaggregation:
    hidden_size: 16
    use_attributes: true
    use_precip: true
```

- [ ] **Step 3: Write `config/experiments/equif_hourly_lstm.yaml`**

Identical to Step 1 except the header comment (arm R3, hourly-native MTS-LSTM; disagg omitted — it is a config ERROR with an hourly-native store, `src/data/dataset.rs:290-309`) and the streamflow line:

```yaml
  streamflow: /mnt/ssd1/data/icechunk/hourly_lstm_merit_unit_catchments.ic
```

The `aorc_precip` line STAYS (matches `config/sources/hourly-lstm.yaml` for group tagging; unused without a disagg block).

- [ ] **Step 4: Validate all three parse + group-tag + guard**

```bash
for c in equif_daily_lstm_flat equif_daily_lstm_disagg equif_hourly_lstm; do
  ddrs --config config/experiments/$c.yaml plan --workflow train-and-test 2>&1 | tail -3
done
```
Expected: each plan succeeds (first plan per source computes that arm's summed-Q′ baseline — minutes; subsequent are cache hits). Confirm in output: no drift errors, baseline metrics printed.

Negative check (guard works): temporarily add the disagg block to the hourly config and expect the "hourly-native; remove the disaggregation block" error, then revert. Run:
```bash
ddrs --config /tmp/equif_hourly_bad.yaml plan 2>&1 | grep -q "hourly-native" && echo GUARD-OK
```

- [ ] **Step 5: Commit**

```bash
git add config/experiments/equif_*.yaml
git commit -m "feat(config): three tracked equifinality arms (daily-lstm flat/disagg, hourly-lstm native)"
```

---

### Task 4: Smoke-time each arm on CPU (GO/NO-GO for budget)

**Files:** none created (measurement only; results recorded in the findings doc later).

- [ ] **Step 1: Verify installed binary is current**

```bash
ddrs --version
git rev-parse --short HEAD   # manifest stamps git SHA at runtime — the
                             # binary itself must be from Task 1-2's install
```
If Task 1/2 changed src/ since the last `cargo install --path .`, reinstall.

- [ ] **Step 2: Two-mini-batch smoke per arm, timed**

```bash
for c in equif_daily_lstm_flat equif_daily_lstm_disagg equif_hourly_lstm; do
  echo "=== $c ==="
  /usr/bin/time -v ddrs --config config/experiments/$c.yaml \
    run --backend cpu --workflow train --max-mini-batches 2 \
    2>&1 | tee /tmp/smoke_$c.log | grep -E "resolution|mini-batch|Elapsed"
done
```

Expected per arm:
- Log line `streamflow resolution: Daily` (R1, R2) / `Hourly` (R3) — the read-path self-check.
- Checkpoints under the new run dir are DIRECTORIES (`epoch_*_mb_*/head.mpk`).
- A measured seconds-per-mini-batch.

- [ ] **Step 3: Extrapolate and apply the pre-registered budget rule**

Full run ≈ `s_per_batch × batches_per_epoch × 5` (batches/epoch visible in the smoke log's epoch layout; ~37 for 2,365 gauges at batch_size 64). Record the three estimates.

**Gate (from the spec): if R3's projected single epoch exceeds ~4 h, STOP — do not cut epochs (unequal budgets confound convergence); report and wait for GPU.** R1/R2 reference point: ~85 min per 5-epoch daily CONUS CPU run (2026-07 leakance session).

- [ ] **Step 4: Clean up smoke runs**

```bash
ddrs gc --keep 5 --keep-successful   # or manually remove the smoke run dirs
```
(Do not let 2-mini-batch smoke manifests masquerade as real arms later.)

---

### Task 5: Launch the three full runs (detached, sequential)

**Files:**
- Create: `scripts/run_equif_arms.sh`

- [ ] **Step 1: Write the driver script**

```bash
#!/usr/bin/env bash
# Sequential CPU training of the three equifinality arms.
# Detach with: nohup scripts/run_equif_arms.sh > output/equif_runs.log 2>&1 &
# Survives agent/session death. ~85 min per daily arm, R3 longer (see smoke).
set -uo pipefail
cd "$(dirname "$0")/.."

ARMS=(equif_daily_lstm_flat equif_daily_lstm_disagg equif_hourly_lstm)
STATUS_FILE=output/equif_runs.status
mkdir -p output
: > "$STATUS_FILE"

for c in "${ARMS[@]}"; do
  echo "[$(date -u +%FT%TZ)] START $c" | tee -a "$STATUS_FILE"
  if ddrs --config "config/experiments/$c.yaml" \
       run --backend cpu --workflow train-and-test; then
    echo "[$(date -u +%FT%TZ)] OK    $c" | tee -a "$STATUS_FILE"
  else
    echo "[$(date -u +%FT%TZ)] FAIL  $c (exit $?)" | tee -a "$STATUS_FILE"
    # keep going — arms are independent; a failed arm is diagnosed from
    # its run.log, the others still produce science
  fi
done
echo "[$(date -u +%FT%TZ)] ALL DONE" | tee -a "$STATUS_FILE"
```

```bash
chmod +x scripts/run_equif_arms.sh
git add scripts/run_equif_arms.sh
git commit -m "feat(scripts): sequential detached driver for equifinality CPU arms"
```

- [ ] **Step 2: Launch detached**

```bash
nohup scripts/run_equif_arms.sh > output/equif_runs.log 2>&1 &
echo "driver pid: $!"
```

- [ ] **Step 3: Verify liveness, then wait**

```bash
sleep 120 && tail -5 output/equif_runs.log && tail -2 .ddrs/runs/$(ls -t .ddrs/runs | head -1)/run.log
```
Expected: run directory created with the `daily-lstm` group tag in its ID, loss lines advancing. Then wait for `ALL DONE` in `output/equif_runs.status` (check periodically; each arm's own `run.log` is the detailed view). NEVER kill the driver's process tree to "check on it".

- [ ] **Step 4: Post-run validation (all three arms)**

```bash
for d in $(ls -t .ddrs/runs | head -3); do
  echo "=== $d ==="
  python3 -c "import json; m=json.load(open('.ddrs/runs/$d/manifest.json')); print(m['status'], m['workflow'], m['git']['sha'][:8])"
  ls .ddrs/runs/$d/checkpoints/ | tail -1           # directory epoch_5_mb_*
  ls .ddrs/runs/$d/eval/predictions.zarr >/dev/null && echo eval-ok
  ls .ddrs/runs/$d/baseline/manifest.json >/dev/null && echo baseline-ok
  grep -m1 "streamflow resolution" .ddrs/runs/$d/run.log
done
```
Expected per arm: `Ok train-and-test <sha>`, a directory checkpoint at epoch 5, eval + baseline present, correct resolution line. Record the three run IDs — every later step keys off them.

---

### Task 6: Per-arm artifacts — parameter dumps and gradient probes

**Files:** outputs only (`.ddrs/runs/<id>/plot/kan_parameters.nc`, `output/equif_probe/grad_<arm>.nc`).

- [ ] **Step 1: Dump denormalized parameters per arm**

`ddrs run --plot` wasn't used during the runs; dump from checkpoints directly. CLI (verified `src/bin/dump_parameters.rs:16-38`): `--config --checkpoint --output --batch-size --backend`. **`--checkpoint` is a recorder BASE path (no `.mpk` suffix)** — for a directory checkpoint pass `<dir>/head`.

First set the three run IDs recorded in Task 5 Step 4:

```bash
RUN_R1=<run-id of equif_daily_lstm_flat>
RUN_R2=<run-id of equif_daily_lstm_disagg>
RUN_R3=<run-id of equif_hourly_lstm>

cargo build --release --bin dump_parameters
mkdir -p output/equif
for pair in "equif_daily_lstm_flat:R1:$RUN_R1" "equif_daily_lstm_disagg:R2:$RUN_R2" "equif_hourly_lstm:R3:$RUN_R3"; do
  IFS=: read -r c arm run_id <<< "$pair"
  ck=$(ls -d .ddrs/runs/$run_id/checkpoints/epoch_*_mb_* | sort -V | tail -1)
  target/release/dump_parameters --backend cpu \
    --config config/experiments/$c.yaml \
    --checkpoint "$ck/head" \
    --output output/equif/${arm}_kan_parameters.nc
done
```
Expected: three NetCDFs, each with `n`, `q_spatial`, `p_spatial`, `x_storage`, `slope` on the COMID dim (full CONUS), physical units (schema: `src/dump_parameters.rs:754-824`).

- [ ] **Step 2: Gradient probe per arm (96 windows, seed 42, CPU)**

```bash
# uses RUN_R1/RUN_R2/RUN_R3 from Step 1
for pair in "equif_daily_lstm_flat:R1:$RUN_R1" "equif_daily_lstm_disagg:R2:$RUN_R2" "equif_hourly_lstm:R3:$RUN_R3"; do
  IFS=: read -r c arm run_id <<< "$pair"
  ck=$(ls -d .ddrs/runs/$run_id/checkpoints/epoch_*_mb_* | sort -V | tail -1)
  # probe --checkpoint takes the checkpoint DIRECTORY (not the /head base)
  nohup nice -n 10 target/release/probe_zeta_gradient \
    --config config/experiments/$c.yaml \
    --checkpoint $ck \
    --windows 96 --seed 42 --backend cpu \
    --params n,q_spatial,p_spatial \
    --output output/equif_probe/grad_${arm}.nc \
    > output/equif_probe/probe_${arm}.log 2>&1
done
```
Run sequentially (each ~35 min on CPU per the leakance probe reference; hourly arm longer). Same `--seed 42` across arms ⇒ identical window samples ⇒ per-reach gradients are directly comparable.
Expected: three NetCDFs with `grad_{n,q_spatial,p_spatial}_{abs,net}` + `n_windows`.

---

### Task 7: Cross-arm convergence analysis script

**Files:**
- Create: `scripts/equif_convergence_analysis.py` (run from `~/projects/ddr` via `uv run`)

The script produces the H1–H4 evidence. Structure it as ONE file with clearly separated stages writing intermediate `.npz` caches to `output/equif/` (each stage is re-runnable). Full skeleton with the load-bearing code:

- [ ] **Step 1: Write the data-loading + network stage**

```python
#!/usr/bin/env python3
"""Cross-arm selective-equifinality analysis (spec 2026-07-06).

Usage (from ~/projects/ddr):
  uv run python ~/projects/ddrs/scripts/equif_convergence_analysis.py \
    --r1 <run-id> --r2 <run-id> --r3 <run-id> \
    --ddrs-root /home/tbindas/projects/ddrs
Arms: R1 daily-lstm flat, R2 daily-lstm disagg, R3 hourly-lstm native.
"""
import argparse, json
from pathlib import Path
import numpy as np
import netCDF4
import zarr
import xarray as xr

PARAM_RANGES = {"n": (0.015, 0.25), "q_spatial": (0.0, 1.0), "p_spatial": (1.0, 200.0)}
STORES = {
    "R1": "/mnt/ssd1/data/icechunk/daily_lstm_merit_unit_catchments.ic",
    "R2": "/mnt/ssd1/data/icechunk/daily_lstm_merit_unit_catchments.ic",
    "R3": "/mnt/ssd1/data/icechunk/hourly_lstm_merit_unit_catchments.ic",
    "COMMON": "/mnt/ssd1/data/icechunk/merit_dhbv2_UH_retrospective.ic",
}
EVAL_WINDOW = ("1995-10-01", "2010-09-30")

def load_params(nc_path: Path) -> dict[str, np.ndarray]:
    ds = netCDF4.Dataset(nc_path)
    out = {"COMID": ds["COMID"][:].astype(np.int64)}
    for v in ("n", "q_spatial", "p_spatial", "slope"):
        out[v] = np.asarray(ds[v][:], dtype=np.float64)
    return out

def eval_network(gages_adj: Path) -> tuple[np.ndarray, dict]:
    """Union of per-gauge upstream COMID sets ('order' arrays)."""
    root = zarr.open_group(str(gages_adj), mode="r")
    per_gauge = {g: np.asarray(root[g]["order"][:], dtype=np.int64) for g in root.group_keys()}
    comids = np.unique(np.concatenate(list(per_gauge.values())))
    return comids, per_gauge

def conus_edges(conus_adj: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Topologically ordered COMIDs + downstream edges (row drains-into col
    convention: rows[k] >= cols[k] positions in `order`)."""
    root = zarr.open_group(str(conus_adj), mode="r")
    order = np.asarray(root["order"][:], dtype=np.int64)
    i0 = np.asarray(root["indices_0"][:], dtype=np.int64)  # downstream position
    i1 = np.asarray(root["indices_1"][:], dtype=np.int64)  # upstream position
    return order, i0, i1
```

- [ ] **Step 2: Coverage intersection + summed upstream Q′ (topological accumulation)**

```python
def store_divides(path: str) -> np.ndarray:
    import icechunk as ic
    repo = ic.Repository.open(ic.local_filesystem_storage(path))
    ds = xr.open_zarr(repo.readonly_session("main").store, consolidated=False)
    return np.asarray(ds["divide_id"].values).astype(np.int64)
    # NOTE: verify variable/coord name against the store
    # (docs/nh-qprime-store-contract.md — Qr(divide_id, time)).

def upstream_fully_covered(order, i0, i1, covered_mask):
    """covered_closure[i] = covered[i] AND all upstream covered (topo pass)."""
    closure = covered_mask.copy()
    # order is topological (upstream before downstream); process edges so
    # downstream inherits ANDed upstream closure. One pass suffices because
    # edges reference already-finalized upstream positions.
    for k in np.argsort(i0, kind="stable"):
        closure[i0[k]] &= closure[i1[k]]
    return closure

def summed_upstream_qprime(order, i0, i1, qprime_daily):
    """Q_sum = (I - N)^-1 q'  by in-order accumulation.
    qprime_daily: [n_reaches, n_days] f32, 0.0 where uncovered."""
    q = qprime_daily.copy()
    for k in np.argsort(i0, kind="stable"):
        q[i0[k]] += q[i1[k]]
    return q
```

Compute per arm (and once for COMMON): read the store's daily Qr over the eval window restricted to eval-network COMIDs (hourly store: 24-h means — `ds.resample(time="1D").mean()` or slice+reshape), zero-fill uncovered, accumulate, take per-reach `median/p10/p90` over time, save to `output/equif/qref_<arm>.npz`. **Caution:** the edge-accumulation loop is Python-level over ~339k edges but each op is a length-5479 vector add — acceptable (~minutes); if too slow, replace with `scipy.sparse.linalg.spsolve_triangular` on the (I−N) CSR with `unit_diagonal=True`.

- [ ] **Step 3: Level 1 + Level 2 (parameters and realized geometry)**

Port of `~/projects/ddr/src/ddr/geometry/trapezoidal.py:14-108` to numpy (formulas verified against `src/geometry.rs:28-79`):

```python
def trapezoidal_geometry(n, p, q, Q, slope, depth_lb=0.01, bw_lb=0.01):
    q_eps = q + 1e-6
    depth = np.clip(((Q * n * (q_eps + 1)) / (p * np.sqrt(slope) + 1e-8))
                    ** (3.0 / (5.0 + 3.0 * q_eps)), depth_lb, None)
    top_width = p * depth ** q_eps
    side_slope = np.clip(top_width * q_eps / (2 * depth), 0.5, 50.0)
    bottom_width = np.clip(top_width - 2 * side_slope * depth, bw_lb, None)
    area = (top_width + bottom_width) * depth / 2
    wp = bottom_width + 2 * depth * np.sqrt(1 + side_slope**2)
    rh = area / wp
    return {"depth": depth, "top_width": top_width, "hydraulic_radius": rh}

def norm_spread(stack, lo, hi):
    """stack: [n_arms, n_reaches] physical values → per-reach (max-min)/(hi-lo)."""
    return (stack.max(0) - stack.min(0)) / (hi - lo)

def rel_spread(stack):
    """for realized geometry: per-reach (max-min)/mean."""
    m = stack.mean(0)
    return np.where(m > 0, (stack.max(0) - stack.min(0)) / m, np.nan)
```

Level 1: align the three param dumps on common COMIDs (restricted to the analysis set = eval network ∩ covered-closure of all arms), compute per-parameter cross-arm Spearman ρ (all 3 pairs) and `median(norm_spread)`.
Level 2 (H1, primary): per arm, `trapezoidal_geometry(n_a, p_a, q_a, Qref_OWN_a, slope)`; spread via `rel_spread`. Sensitivity: same with `Qref_COMMON`, and at p10/p90 flows. **H1 key comparison: `median rel_spread(geometry)` vs `median norm_spread(n)`.**

- [ ] **Step 4: Level 3 (skill) + Level 4 (gradients)**

Level 3 — reuse the loader pattern from `scripts/leakance_subset_analysis.py:52-117` (eval zarr: `predictions`/`observations`/`gage_ids`; baseline: raw f32 + manifest) and its NaN-safe `nse`/`kge` (`:199-217`). Report per arm: median NSE/KGE (trained) vs median NSE/KGE (that arm's own baseline), gauge count, window.

Level 4 — load `grad_<arm>.nc`, align on common `COMID_probe`:

```python
def cosine(a, b):
    m = np.isfinite(a) & np.isfinite(b)
    a, b = a[m], b[m]
    d = np.linalg.norm(a) * np.linalg.norm(b)
    return float(a @ b / d) if d > 0 else np.nan

# H3: for each param, cosine(g_net_armA, g_net_armB) for the 3 arm pairs
#     + per-reach sign-agreement fraction.
# H4: gauge COMIDs from gages_3000.csv (COMID column);
#     upstream BFS distance: dist[gauge reaches]=0, then repeatedly
#     dist[i1[k]] = min(dist[i1[k]], dist[i0[k]]+1) sweeping edges in
#     REVERSE topological order (downstream→upstream), until fixpoint
#     (2-3 sweeps). Report median |grad| per distance bin (0, 1-2, 3-5,
#     6-10, >10 hops) and the gauged/ungauged ratio, per param per arm.
```

- [ ] **Step 5: Verdict block + figures**

Print (and write `output/equif/verdicts.json`) a table applying the spec's falsification bars verbatim:

```
H1 SUPPORTED  iff median rel_spread(depth,width,Rh) < median norm_spread(n)   [primary Qref; report sensitivity]
H2 SUPPORTED  iff spearman(norm_spread_n, qprime_disagreement) > 0.2 AND norm_spread_n > spread(geometry)
H3 SUPPORTED  iff min over pairs cosine(g_p), cosine(g_q) > cosine(g_n)
H4 SUPPORTED  iff gauged/ungauged median |g| ratio > 1 with monotone decay over distance bins
```

(`qprime_disagreement` = per-reach relative range of eval-window mean summed Q′ across the arms' own stores.) Figures (matplotlib, save to `output/equif/figs/`): per-parameter cross-arm scatter matrices, spread CDFs (n vs geometry overlaid), gradient-alignment maps, |g| vs distance curves.

- [ ] **Step 6: Run and commit**

```bash
cd ~/projects/ddr
uv run python ~/projects/ddrs/scripts/equif_convergence_analysis.py \
  --r1 <run-id-R1> --r2 <run-id-R2> --r3 <run-id-R3> \
  --ddrs-root /home/tbindas/projects/ddrs
```
Expected: verdict table printed with all four H-verdicts and their key numbers; no NaN-only columns; analysis-reach count printed (log how many reaches the coverage intersection dropped — no silent truncation).

```bash
cd ~/projects/ddrs
git add scripts/equif_convergence_analysis.py
git commit -m "feat(analysis): cross-arm selective-equifinality convergence analysis (4 levels)"
```

---

### Task 8: Findings doc

**Files:**
- Create: `docs/2026-07-XX-lstm-equifinality-findings.md` (date = day the analysis completes)

- [ ] **Step 1: Write the findings doc** using the mandatory template from the `ddrs-docs-and-writing` skill: header block (spec/plan/script links), **one-line verdict first**, §1 pre-registered H1–H4 table (copied from the spec, NOT reverse-engineered), §2 methods (arms table with run IDs, git SHA, binary provenance note per the STALE-BINARY rule, `streamflow resolution` log lines), §3 results (SUPPORTED/REFUTED/INCONCLUSIVE + key number per hypothesis, every number with units + gauge count + window), §4 conclusions, §5 next steps (dHBV2 arms), §6 raw script output, §7 reproduce commands.

- [ ] **Step 2: Commit**

```bash
git add docs/2026-07-*-lstm-equifinality-findings.md
git commit -m "docs(findings): LSTM-source selective-equifinality results (H1-H4 verdicts)"
```

---

## Self-review notes (spec coverage)

- Spec Phase 1 (infra) → Tasks 1–3. Spec Phase 2 (runs) → Tasks 4–6. Spec Phase 3 (analysis) → Task 7. Deliverable 6 (findings) → Task 8.
- Spec's probe requirement had a hidden blocker (hard `use_leakance` assert) — Task 2 Steps 1/3 resolve it; this supersedes the spec's "~read-only adaptation" phrasing but changes no gated file.
- Budget gate (R3 > 4 h/epoch → stop) → Task 4 Step 3. Coverage intersection → Task 7 Steps 2/3. Arm-own vs common reference discharge → Task 7 Step 3. No-silent-caps → Task 7 Step 6 logs dropped reach counts.

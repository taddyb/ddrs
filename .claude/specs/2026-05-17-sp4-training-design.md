# SP-4 design: Training loop

**Status:** Draft, pending user review
**Parent:** [`2026-05-17-train_and_test-replication-design.md`](./2026-05-17-train_and_test-replication-design.md)
**Mirrors:** `scripts/train.py` and the gauge-extraction logic in
`ddr/routing/mmc.py` (~lines 344-441), plus `ddr/io/functions.py::downsample`.

## Why this sub-project

SP-4 is where the project's load-bearing verification finally binds.
Everything before it (data layer, batch construction, the MC engine) was
prerequisite plumbing. The verification bar from the master spec —
"given fixed MLP outputs + the same inputs, ddrs's per-batch L1 loss
equals DDR's within the f32 floor" — gets satisfied in SP-4's
integration test.

The non-verification work in SP-4: a real training loop that consumes
`RoutingBatch`es from SP-3, runs the MLP + MC engine, computes daily L1
loss, backpropagates, applies the Adam step, saves checkpoints, and
logs NSE/RMSE/KGE per batch. Same shape as DDR's `train.py:23-128`.

## Scope

In scope:

1. **`RoutingBatch::to_tensors<B>`** — materialize the `Array2<f32>`
   fields into BURN `Tensor<I, 2>` at the device boundary.
2. **`forward.rs`** — one training step's worth: MLP head, MC engine,
   gauge extraction (scatter_add over `outflow_idx`).
3. **`loss.rs`** — `tau`-trimmed daily downsample (`mean over 24`) +
   L1 loss + per-gauge NaN mask + warmup trim.
4. **`metrics.rs`** — NSE, RMSE, KGE per gauge over the post-warmup
   prediction/observation windows.
5. **`checkpoint.rs`** — BURN `CompactRecorder` save/load for the MLP
   + Adam optimizer state + `(epoch, mini_batch, rng_state)`.
6. **`loop.rs`** — `train(cfg, dataset, mlp, optimizer)` — outer epoch +
   inner mini-batch loop. Adam optimizer, lr schedule by epoch,
   gradient clipping, per-batch logging.
7. **Direct-param forward path** — a small variant of the forward pass
   that takes `(n, q_spatial, p_spatial)` as **direct** input vectors,
   bypassing the MLP. Used by the verification test (see V1 below).
8. **V1 verification test** — loss-equivalence vs DDR with fixed
   inputs + fixed parameters.

Out of scope (deferred to SP-5):

- Top-level `train_and_test` binary entrypoint.
- The Phase-2 (test) loop with `SequentialSampler` + zarr write of
  predictions/observations.
- NSE/RMSE/KGE distributional summary across all gauges (per-batch
  logging is enough for SP-4).
- Plotting (DDR's `plot_time_series`).
- Multi-GPU / distributed training.

## Verification protocol — three staged tests, each more stringent

The verification climbs a ladder, smallest → largest, training-free →
training-loop. Each stage is a separate test; later stages depend on
earlier ones passing.

### V1 — single small batch, frozen constant params

The minimum viable load-bearing test. Pin
`(n, q_spatial, p_spatial)` to scalar constants (the same scalar
applied uniformly across every reach), pick a small reproducible batch
(`batch_size=8`, `rho=90`), run forward + loss, assert agreement vs
DDR.

```rust
#[test]
fn v1_loss_matches_ddr_for_frozen_constant_params_small_batch() {
    let cfg = Config::from_yaml_file("config/merit_training.yaml")?;
    let ds = MeritGagesDataset::open(&cfg)?;
    let (staids, window) = pick_reproducible_batch(&ds, /*seed=*/42, /*batch_size=*/8, /*rho=*/90);
    let batch = ds.collate(&staids, &window)?;

    // Constant frozen params — scalar broadcast.
    let n_active = batch.adjacency.n;
    let frozen = FrozenParams {
        n:         vec![FROZEN_N;         n_active],  // FROZEN_N         = 0.05
        q_spatial: vec![FROZEN_Q_SPATIAL; n_active],  // FROZEN_Q_SPATIAL = 0.5
        p_spatial: vec![FROZEN_P_SPATIAL; n_active],  // FROZEN_P_SPATIAL = 21.0
    };

    let loss_ddrs = forward_with_frozen_params::<NdArray<f32>>(&cfg, &batch, &frozen);
    let ddr_loss = read_ddr_reference_loss("fixtures/v1_ddr_loss.json");
    let rel_diff = (loss_ddrs - ddr_loss).abs() / ddr_loss.abs();
    assert!(rel_diff < 1e-5, "V1: ddrs={loss_ddrs}, DDR={ddr_loss}, rel={rel_diff}");
}
```

`scripts/dump_ddr_loss.py` is a small Python script (run once under
DDR's `uv` venv) that mirrors the batch and frozen-params constants,
runs DDR's engine + loss, and writes the resulting loss to JSON. The
JSON is checked into `fixtures/` so the Rust test runs offline; user
regenerates when infrastructure changes.

**Why constants, not sigmoid-of-attrs.** The user chose this — and it's
the right call. The frozen-params recipe is the single highest-risk
source of cross-runtime divergence (a typo on either side breaks the
test for a non-bug reason). Scalar constants reduce that surface to
three numbers documented in both code paths.

### V2 — large CONUS-scale single batch, same frozen constants

Once V1 passes, V2 scales the batch up. Same frozen constants, but the
batch is constructed by concatenating ALL filtered gauges (~2365 from
SP-3's filter pipeline) into one network matrix. This stress-tests:

- The `compress` machinery on a real-CONUS-scale adjacency (~100K
  active segments).
- The scatter_add gauge extraction across thousands of gauges.
- Memory and compile times under autograd at scale.
- The forward solver's f32 accumulation over a long timestep stack.

```rust
#[test]
fn v2_loss_matches_ddr_for_frozen_constant_params_all_gauges() {
    let cfg = Config::from_yaml_file("config/merit_training.yaml")?;
    let ds = MeritGagesDataset::open(&cfg)?;
    // ALL filtered gauges in one batch.
    let staids = ds.staids().to_vec();
    let window = ds.time_axis().sample_rho_window(&mut StdRng::seed_from_u64(42), 90);
    let batch = ds.collate(&staids, &window)?;

    let n_active = batch.adjacency.n;
    let frozen = FrozenParams {
        n:         vec![FROZEN_N;         n_active],
        q_spatial: vec![FROZEN_Q_SPATIAL; n_active],
        p_spatial: vec![FROZEN_P_SPATIAL; n_active],
    };

    let loss_ddrs = forward_with_frozen_params::<NdArray<f32>>(&cfg, &batch, &frozen);
    let ddr_loss = read_ddr_reference_loss("fixtures/v2_ddr_loss.json");
    let rel_diff = (loss_ddrs - ddr_loss).abs() / ddr_loss.abs();
    // f32 accumulation tolerance is slightly looser at this scale —
    // we re-evaluate the bound based on actual divergence.
    assert!(rel_diff < 1e-4, "V2: ddrs={loss_ddrs}, DDR={ddr_loss}, rel={rel_diff}");
}
```

The slightly looser tolerance at V2 (`1e-4` vs V1's `1e-5`) is because
~100K reaches × 89 hourly steps accumulate enough f32 ULP-order noise
that 1e-5 may be unrealistic. We pick the actual threshold from the
first measured divergence — log it and adjust the constant once.

If V1 passes but V2 fails, the diagnostic surface is interesting: the
engine has scaled, not the math. Likely culprits: scatter_add for
thousands of gauges, autograd tape size, or the f32-accumulation drift
ceiling at CONUS scale.

V2 is a **single forward+loss pass**, no backward. We don't need
autograd for the verification; just `Tensor<I, 2>` (not
`Tensor<Autodiff<I>, 2>`). This keeps memory bounded.

### V3 — full training loop, multi-epoch

After V1+V2 pass, V3 exercises the real training loop end-to-end:
multiple epochs, RandomSampler, real MLP head, Adam optimizer,
gradient clipping, checkpoint save. This is NOT a per-batch loss-equal
test against DDR — once the MLP is trained, RNG, optimizer step
counter, and floating-point reduction order all diverge between Rust
and Python.

V3's assertions are coarser:

1. **It runs end-to-end without panicking or OOM** on the dev machine
   for `epochs=1`. (Multi-epoch is too slow for CI; one-epoch is the
   bar.)
2. **Loss decreases monotonically** for the first few mini-batches.
   With the MERIT defaults this should happen; if it doesn't, the
   training step isn't actually optimizing.
3. **Checkpoint round-trips** — save, load, forward returns the same
   loss within f32 floor.
4. **NSE/RMSE/KGE metrics are finite** for at least one gauge in the
   logged batches.

If V3 reveals a divergent training trajectory (e.g., loss explodes
after 50 batches), that's a real bug — but locating it from V3 alone
is hard. V1 and V2 isolate the deterministic pieces; V3 is the
shipping bar.

### Why this ladder

The user-requested staging maps directly to the failure modes:

- **V1 fail:** the engine / loss math is wrong on a single small
  graph. Diagnostic: inspect intermediates of one batch.
- **V2 fail (V1 passing):** scaling-only bug. Diagnostic: scatter_add
  for many gauges, or f32 accumulation drift.
- **V3 fail (V1+V2 passing):** training-loop bug. Diagnostic:
  optimizer state, grad-clip, lr schedule, or checkpoint round-trip.

Each test localizes the bug. Without the ladder, a V3-only test would
mix all three failure modes into one symptom.

Both V1 and V2 use **the same `scripts/dump_ddr_loss.py`** with a
different `--batch-spec` argument. The fixture JSON is two files
(`v1_ddr_loss.json` and `v2_ddr_loss.json`), each ~50 bytes, checked
into `fixtures/sp4/`.

## Architecture

```
                      RoutingBatch (Array2<f32> fields)
                                │
                                ▼
              ┌──────────────────────────────────────┐
              │  RoutingBatch::to_tensors::<B>       │
              │   → spatial_attributes: (N, F)       │
              │   → q_prime:            (T, N)       │
              │   → observations:       (rho_d, G)   │
              │   ↳ + flat_indices / group_ids       │
              │     concat-of-outflow_idx +          │
              │     per-element group tag (for       │
              │     scatter_add).                    │
              └──────────────────────────────────────┘
                                │
                                ▼
              ┌──────────────────────────────────────┐
              │  MLP::forward(attrs) → params dict   │
              │   {n, q_spatial, p_spatial}: (N,)    │
              │  OR frozen_params (verification)     │
              └──────────────────────────────────────┘
                                │
                                ▼
              ┌──────────────────────────────────────┐
              │  MuskingumCunge::setup_inputs +      │
              │  MuskingumCunge::forward             │
              │   → runoff: (N, T_hours)             │
              └──────────────────────────────────────┘
                                │
                                ▼
              ┌──────────────────────────────────────┐
              │  scatter_add by group_ids            │
              │   gathered = runoff[flat_indices, :] │
              │   per_gauge = scatter_add(gathered,  │
              │                  group_ids)          │
              │   per_gauge: (G, T_hours)            │
              └──────────────────────────────────────┘
                                │
                                ▼
              ┌──────────────────────────────────────┐
              │  tau-trim:                           │
              │    sliced = per_gauge[:, 13+tau :    │
              │                       -11+tau]       │
              │  daily downsample:                   │
              │    reshape (G, T_days, 24).mean(2)   │
              │    → daily: (G, T_days)              │
              └──────────────────────────────────────┘
                                │
                                ▼
              ┌──────────────────────────────────────┐
              │  NaN mask + warmup trim:             │
              │    pred = daily[:, warmup..]         │
              │           [valid_gauges_mask]        │
              │    target = obs[:, warmup..]         │
              │             [valid_gauges_mask]      │
              │  L1 loss = abs(pred - target).mean() │
              └──────────────────────────────────────┘
                                │
                                ▼
                        loss.backward()
                        grad_clip(mlp.params, 1.0)
                        optimizer.step()
```

## Components

### 1. `src/data/dataset.rs` — `RoutingBatch::to_tensors<B>`

Add an associated function (or method) that lifts the Array2 fields to
BURN tensors and pre-computes the `flat_indices` + `group_ids` arrays
needed for scatter_add. This is purely a materialization step — no
math.

```rust
pub struct RoutingTensors<B: Backend> {
    pub adjacency: SparseAdjacency,
    pub spatial_attributes: Tensor<B, 2>,    // (N, F)
    pub q_prime: Tensor<Autodiff<B>, 2>,     // (T_hours, N)
    pub observations: Array2<f32>,           // (T_days, G), stays on CPU
    pub flat_indices: Tensor<B, 1, Int>,     // concat(outflow_idx)
    pub group_ids: Tensor<B, 1, Int>,        // gauge group per flat index
    pub num_gauges: usize,
    pub gauge_staids: Vec<Staid>,
    pub window: RhoWindow,
}

impl RoutingBatch {
    pub fn to_tensors<B: Backend>(self, device: &B::Device) -> RoutingTensors<B>;
}
```

`flat_indices` and `group_ids` are computed from `outflow_idx` in pure
Rust before tensor lift. Pattern (mirrors DDR `mmc.py:347-358`):

```rust
let mut flat: Vec<i64> = vec![];
let mut group: Vec<i64> = vec![];
for (g_idx, segs) in self.outflow_idx.iter().enumerate() {
    flat.extend(segs.iter().map(|&s| s as i64));
    group.extend((0..segs.len()).map(|_| g_idx as i64));
}
```

`observations` stays on CPU (it's only used at loss time for masking +
comparison; no need to round-trip through GPU memory).

### 2. `src/training/forward.rs` — single forward pass

```rust
pub fn forward<B: Backend>(
    cfg: &Config,
    tensors: &RoutingTensors<B>,
    params: SpatialParameters<B>,
) -> Tensor<Autodiff<B>, 2>
```

Returns `(num_gauges, T_hours)` of per-gauge predictions. Steps:

1. Build `MuskingumCunge::new(cfg.params, device)`.
2. Build `RoutingInputs { adjacency: tensors.adjacency.clone(), x_storage: const 0.3 }`.
3. `engine.setup_inputs(inputs, tensors.q_prime.clone(), params, carry_state=false)`.
4. `let runoff = engine.forward();` — shape `(N, T_hours)`.
5. **Gauge scatter_add:** `let per_gauge = scatter_add_by_group(runoff, &tensors.flat_indices, &tensors.group_ids, tensors.num_gauges);`.

`scatter_add_by_group` is a thin helper. BURN's tensor API has
`scatter` / `select` primitives — pick whichever supports the
"gather + grouped sum" pattern. If BURN doesn't have a direct primitive,
the fallback is: for each gauge `g`, sum `runoff[outflow_idx[g], :]`.
Less elegant but correct.

`x_storage` is hardcoded to `0.3` per `_collate_gages`. Not a learnable
parameter. We materialize it once at forward time.

### 3. `src/training/loss.rs` — daily downsample + L1

```rust
pub fn daily_loss<B: Backend>(
    cfg: &Config,
    predictions_hourly: Tensor<Autodiff<B>, 2>,  // (G, T_hours)
    observations: &Array2<f32>,                  // (rho_days, G)
    warmup: usize,
    tau: usize,
) -> Tensor<Autodiff<B>, 0>
```

Steps:

1. **Tau-trim:** `sliced = predictions_hourly[:, 13+tau .. -(11+tau)]`.
   Result shape: `(G, T_hours - 24 - tau)`. `T_hours` is `(rho_days-1)*24`
   so the trimmed length is divisible by 24.
2. **Daily downsample:** reshape to `(G, T_days, 24)`, `.mean(dim=2)` →
   `(G, T_days)`.
3. **Warmup trim on observations + predictions.**
4. **NaN mask:** in DDR (`scripts/train.py:78-82`),
   `nan_mask = observations.isnull().any(dim="time")` — gauges with
   *any* NaN in the window are dropped entirely. Same here.
5. L1: `|pred[mask] - obs[mask]|.mean()`. Returns scalar `Tensor<_, 0>`.

`tau` defaults to `3` (DDR convention). The literal slicing constants
`[13, -11]` come from DDR — pre-routing spinup + post-routing trim.

### 4. `src/training/metrics.rs` — NSE, RMSE, KGE per gauge

```rust
pub struct Metrics {
    pub nse: Vec<f32>,    // length G
    pub rmse: Vec<f32>,
    pub kge: Vec<f32>,
}

impl Metrics {
    pub fn compute(pred: &Array2<f32>, target: &Array2<f32>) -> Self;
}
```

Per-gauge metrics over the post-warmup window. NaN-handling matches DDR
(`validation/metrics.py`). NSE: `1 - sum((p-t)^2) / sum((t-mean(t))^2)`.
RMSE: `sqrt(mean((p-t)^2))`. KGE: classic Gupta formulation.

Output is per-gauge `Vec<f32>` — same shape DDR's `Metrics.nse` returns.
SP-5 will aggregate distributionally; SP-4 just logs the values for
diagnostics.

### 5. `src/training/checkpoint.rs` — save/load via BURN

```rust
pub fn save_checkpoint<B: Backend>(
    path: &Path,
    mlp: &Mlp<Autodiff<B>>,
    optimizer_state: &OptimizerRecord<...>,
    epoch: usize,
    mini_batch: usize,
    rng_seed: u64,
) -> Result<()>;

pub fn load_checkpoint<B: Backend>(
    path: &Path,
    device: &B::Device,
) -> Result<Checkpoint<B>>;
```

BURN's `CompactRecorder` handles tensor serialization. The optimizer's
internal state (Adam's `m` and `v` moments per parameter) is recorded
alongside.

DDR saves with `.pt` extension; we use `.mpk` (msgpack) — the format
BURN ships. Cross-runtime checkpoint compatibility is **out of scope**.

### 6. `src/training/loop.rs` — `train(cfg, dataset, mlp, optimizer)`

```rust
pub fn train<B: Backend>(
    cfg: &Config,
    dataset: &MeritGagesDataset,
    mlp: Mlp<Autodiff<B>>,
    optimizer: impl Optimizer<Mlp<Autodiff<B>>, Autodiff<B>>,
    device: &B::Device,
) -> Result<Mlp<Autodiff<B>>>
```

Outer loop: `for epoch in 1..=cfg.experiment.epochs`. lr schedule
lookup from `cfg.experiment.learning_rate.get(&epoch)`.

Inner loop:

```rust
sampler.reshuffle(&mut rng);
while let Some(batch_idx) = sampler.next_batch() {
    let staids = batch_idx.iter().map(|&i| dataset.staids()[i].clone()).collect();
    let window = dataset.time_axis().sample_rho_window(&mut rng, cfg.experiment.rho.unwrap());
    let batch = dataset.collate(&staids, &window)?;
    let tensors = batch.to_tensors::<B>(device);

    let params = run_mlp(&mlp, &tensors.spatial_attributes);  // (n, q_spatial, p_spatial)
    let pred_hourly = forward(cfg, &tensors, params);
    let loss = daily_loss(cfg, pred_hourly, &tensors.observations, cfg.experiment.warmup, /*tau=*/3);

    let grads = loss.backward();
    let grads = clip_grad_norm(grads, max_norm = 1.0);
    let mlp = optimizer.step(lr, mlp, grads);

    // Logging + checkpoint per batch (DDR pattern).
    save_checkpoint(...)?;
    log_per_batch_metrics(...);
}
```

`clip_grad_norm` — BURN's `GradientsParams` has an API for this; if
not, implement via a global norm scan + scale. Documented in BURN docs.

### 7. Direct-param forward path

A small helper that swaps in `(n, q_spatial, p_spatial)` directly,
bypassing the MLP. Used only by the V1 verification test.

```rust
pub fn forward_with_frozen_params<B: Backend>(
    cfg: &Config,
    tensors: &RoutingTensors<B>,
    frozen: &FrozenParams,
) -> Tensor<Autodiff<B>, 0>  // returns the scalar loss
```

`FrozenParams { n: Vec<f32>, q_spatial: Vec<f32>, p_spatial: Vec<f32> }`
— all `length = N_active`. The forward path is identical to the
training one; only the `params` argument source differs.

A `frozen_params_for(batch)` helper computes the parameters
deterministically from the batch's normalized attributes:

```rust
// n in [0.015, 0.25]; q_spatial in [0.0, 1.0]; p_spatial in [1.0, 200.0].
// Use sigmoid of attribute means as the [0, 1] sampling point — same
// shape the MLP outputs.
let attrs_mean = batch.spatial_attributes_normalized.mean_axis(Axis(1)).unwrap();
let n_norm: Vec<f32> = attrs_mean.iter().map(|x| sigmoid(*x)).collect();
let q_norm: Vec<f32> = (0..N).map(|i| sigmoid(attrs_mean[i] * 0.5 + 0.1)).collect();
let p_norm: Vec<f32> = (0..N).map(|i| sigmoid(attrs_mean[i] * 0.3 + 0.2)).collect();
// Denormalize through the config ranges (same denormalize() the engine uses).
```

The exact recipe doesn't matter — what matters is that **DDR's Python
side uses the same formula**. `scripts/dump_ddr_loss.py` is the
canonical source for the recipe; the Rust side mirrors it.

## Concerns

1. **BURN's `scatter_add` may not exist** as a single primitive. If
   not, the fallback (per-gauge sum loop) is acceptable. Either way,
   the autodiff has to flow through.

2. **Gradient clipping in BURN.** Need to verify the exact API for
   global-norm clipping. May require a one-pass norm + scale.

3. **Adam parameter parity with PyTorch.** Defaults: PyTorch uses
   `betas=(0.9, 0.999)`, `eps=1e-8`. BURN's `AdamConfig` defaults match
   per the docs. Verify in the implementer's first compile cycle.

4. **The V1 test's frozen-params recipe.** Both ddrs and the Python
   `scripts/dump_ddr_loss.py` MUST use the same formula. If they
   diverge by one operation, the test fails for a non-bug reason.
   Document the recipe in both code locations.

5. **Memory at training scale.** Autograd tape over ~89 hourly
   timesteps × ~10K active reaches × ~5 intermediates per step ≈
   17 MB × 5 × 89 ≈ 7.5 GB peak. CPU `NdArray<f32>` should handle this
   on the dev machine; if not, batch_size knob.

6. **Checkpoint state for the RNG.** Saving the seed alone won't be
   enough if the sampler has consumed some draws. Need to save the
   *current* `StdRng` state (its internal seed/counter). BURN's
   recorder doesn't know about `rand::rngs::StdRng`; use
   `rand::rngs::StdRng::seed_from_u64(...)` plus an integer counter
   tracked manually.

7. **`tau=3` is a config-time constant.** It lives in
   `Params.tau` in DDR. Our `Config::params` doesn't carry it yet —
   add `tau: u32` (default 3) to `Params` as a one-line extension.

8. **Daily downsample uses `F.interpolate(mode="area")` in DDR.** For
   the integer 24:1 ratio, this is exactly `mean over 24-window`. For
   non-integer ratios it diverges — but our windows are always exact
   integer days, so the simple mean is faithful.

## Assumptions

1. BURN 0.21 supports the operations we need: scatter_add (or fallback
   gather + per-group loop), gradient clipping, Adam, scalar tensor
   types, `CompactRecorder`. Verified at SP-4 implementation time.
2. CPU is the target backend for SP-4. CUDA/GPU is SP-5 follow-up.
3. The user is on a machine with ~16+ GB RAM (memory budget per concern 5).
4. The frozen-params recipe doesn't need to mirror what the trained
   MLP would actually produce — it just needs to be in a physically
   sensible range so the engine doesn't NaN out. The recipe's job is
   reproducibility across Rust + Python, not realism.

## Module layout summary

```
src/training/                 (new directory)
├── mod.rs                    (new) re-exports public surface
├── forward.rs                (new) single forward pass
├── loss.rs                   (new) daily downsample + L1
├── metrics.rs                (new) NSE / RMSE / KGE
├── checkpoint.rs             (new) save/load via BURN recorder
└── loop.rs                   (new) epoch + mini-batch driver

src/data/dataset.rs           (extend) RoutingBatch::to_tensors + RoutingTensors
src/config.rs                 (extend) Params.tau: u32

src/lib.rs                    (+) pub mod training;

tests/training_loss.rs        (new) V1 loss-equivalence test
scripts/dump_ddr_loss.py      (new) one-shot Python reference loss dump
```

Approximate code size: forward.rs ~150 LOC, loss.rs ~100, metrics.rs
~150, checkpoint.rs ~100, loop.rs ~150, mod.rs ~30, dataset.rs +50,
V1 test ~200, Python dump script ~100. Total ~1000 LOC.

## Risks summary

| Risk | Likelihood | Mitigation |
|---|---|---|
| BURN scatter_add not available | Medium | Per-gauge sum loop fallback. |
| Adam param defaults differ from PyTorch | Low | Verify at compile time; document. |
| Grad-clip API surface uncertain | Medium | Implement global-norm via two-pass if needed. |
| Frozen-params recipe drift between Rust and Python | High (silent failure) | Document in both code paths; comment cross-reference. |
| Memory blowup at CONUS scale | Medium | batch_size knob; user already confirmed it's fine. |
| Daily downsample produces non-multiple-of-24 length after tau-trim | Low | Assert + clear error message. |
| RNG state not portable across checkpoints | Medium | Track manually via integer counter. |
| BURN's `CompactRecorder` serialization mismatch on optimizer state | Low | Use it as documented; check on first checkpoint. |

## What I'm NOT going to do

- No `trait Optimizer` wrapper. BURN's `OptimizerAdaptor<Adam>` used
  directly, same as `Mlp<Autodiff<B>>` is.
- No async training. CPU sync, like all SP-3 reads.
- No Tensorboard / wandb / external logging. `eprintln!` for now;
  SP-5 may add structured logging.
- No half-precision / mixed-precision. f32 throughout, per the
  routing-core invariant.
- No multi-GPU. CPU first; GPU when V1 passes.
- No DDR cross-checkpoint compatibility. Different recorder formats.
- No PyTorch-bit-identical RNG. The verification bar doesn't require
  it; SP-3's samplers are already best-effort.

## Open questions for review

None — resolved per user direction:

- Frozen params are scalar constants (uniform across reaches):
  `FROZEN_N = 0.05`, `FROZEN_Q_SPATIAL = 0.5`, `FROZEN_P_SPATIAL = 21.0`.
  Documented in both Rust and Python code paths.
- Verification ladders V1 (one small batch) → V2 (all gauges, one
  batch) → V3 (full training loop).

## Next step after approval

Invoke writing-plans → implementation plan → subagent-driven execution
(same workflow as SP-1/SP-2/SP-3).

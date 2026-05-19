# SP-5 Test Evaluation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build the test-phase `evaluate()` loop, DDR-compatible zarr output,
`bin/eval` entrypoint, and pass the V4 (single-batch frozen-params vs DDR)
+ V4b (multi-batch ddrs vs single-batch ddrs) verification ladder over the
full 1995-2010 test period.

**Architecture:** Reuses SP-1/2/3/4 primitives. The two new shape concerns
are (a) test mode iterates time chunks not gauges, and (b) chunks must be
contiguous in hourly time (unlike SP-3's `(rho-1)*24` training trim). A new
`TestWindow` solves (b); a lazy `StaticNetworkCache` on
`MeritGagesDataset` solves (a) without splitting the dataset type.

**Tech Stack:** Existing BURN 0.21 + zarrs 0.23 + ndarray + chrono + new
`clap = "4"` for the CLI binary.

**Spec:** `.claude/specs/2026-05-18-sp5-test-evaluation-design.md`
**Parent:** `.claude/specs/2026-05-17-train_and_test-replication-design.md`

**Verification ladder:**
- **V4:** single-batch full-test-period match against DDR reference, frozen
  scalar params, `1e-4` rel tolerance (may relax to `1e-3` if 5475-day f32
  accumulation demands).
- **V4b:** multi-batch ddrs (15-day chunks) reproduces single-batch ddrs
  result to f32 floor — validates `carry_state` plumbing without needing
  a second DDR dump.

**DDR reference (cite line numbers):**
- `~/projects/ddr/scripts/train_and_test.py:43-119` — `_test` loop
- `~/projects/ddr/src/ddr/scripts_utils.py::compute_daily_runoff` — tau slicing
- `~/projects/ddr/src/ddr/validation/metrics.py::Metrics` — already mirrored in SP-4

---

## File Structure

**Created:**

- `src/data/test_window.rs` — `TestWindow` type with contiguous hourly semantics
- `src/training/eval.rs` — `evaluate()`, `EvalParams`, `EvalOutput`
- `src/training/zarr_io.rs` — `write_predictions_zarr()` matching DDR
- `src/bin/eval.rs` — clap CLI binary
- `scripts/dump_ddr_test_predictions.py` — V4 reference dump
- `fixtures/sp5/v4_ddr_test.zarr/` — committed (force-add)

**Modified:**

- `src/data/dates.rs` — re-export `TestWindow`
- `src/data/dataset.rs` — add `static_network: OnceCell<StaticNetworkCache>` field
  + `MeritGagesDataset::collate_window` method
- `src/training/forward.rs` — add `carry_state: bool` to `forward` and
  `forward_with_frozen_params`; add new `forward_eval` for non-autodiff MLP
- `src/training/mod.rs` — re-export new public items
- `src/config.rs` — add `ConfigMode` enum + `from_yaml_file_with_mode`
  + `TestingOverrides` raw section
- `config/merit_training.yaml` — add `testing:` overlay section
- `Cargo.toml` — add `clap = { version = "4", features = ["derive"] }`
- `tests/training_verification.rs` — append V4 + V4b tests

---

## Conventions for this plan

- All forward code generic over `B: Backend`. Tests pin `NdArray<f32>`.
- Use existing `DataError` variants only (no new ones).
- Cite DDR line numbers in doc comments.
- Pre-existing clippy lints in routing-core code are out of scope —
  same precedent as SP-1/2/3/4.
- Do NOT amend commits — always create new ones.
- Each task ends with one commit naming only the files staged in this task.
- BURN 0.21 API: tensor construction follows the pattern in `src/sparse.rs`
  (`Tensor::from_floats(slice, device)` + `Tensor::from_data(TensorData::from(slice), device)`).

---

### Task 1: `TestWindow` for contiguous hourly chunks

**Files:**
- Create: `src/data/test_window.rs`
- Modify: `src/data/dates.rs` (re-export only)
- Modify: `src/data/mod.rs` (re-export only)

Per spec Concern #8: `RhoWindow::n_hourly = (rho_days - 1) * 24` drops the
last day to match DDR's pandas `inclusive='left'` semantic. That trim
creates a 24-hour gap between consecutive test-mode chunks. SP-5 introduces
a parallel `TestWindow` whose `n_hourly = n_days * 24` (no trim) so chunks
tile cleanly.

- [ ] **Step 1: Create `src/data/test_window.rs`**

```rust
//! Contiguous-hourly time window for test-mode chunking.
//!
//! Unlike `RhoWindow` (which drops the trailing day to mirror DDR pandas
//! `inclusive='left'` for training-mode random rho-window sampling), this
//! `TestWindow` exposes the full `n_days * 24` hours so that consecutive
//! chunks tile the hourly axis without gap or overlap.
//!
//! Used by `MeritGagesDataset::collate_window` and `evaluate()`.

use chrono::{Duration, NaiveDate};

use crate::data::dates::TimeAxis;

#[derive(Copy, Clone, Debug)]
pub struct TestWindow {
    /// 0-based index into the parent `TimeAxis` (daily resolution).
    pub start_day_idx: usize,
    /// Number of daily entries in this window.
    pub n_days: usize,
    /// Calendar date of the first day in the window.
    pub window_start: NaiveDate,
}

impl TestWindow {
    pub fn new(axis: &TimeAxis, start_day_idx: usize, n_days: usize) -> Self {
        assert!(
            start_day_idx + n_days <= axis.num_days,
            "TestWindow exceeds axis: start={start_day_idx} + n_days={n_days} > num_days={}",
            axis.num_days
        );
        Self {
            start_day_idx,
            n_days,
            window_start: axis.start + Duration::days(start_day_idx as i64),
        }
    }

    /// Half-open daily index range `[start, end)` into the parent axis.
    pub fn daily_range(&self) -> std::ops::Range<usize> {
        self.start_day_idx..self.start_day_idx + self.n_days
    }

    /// Contiguous hourly length: `n_days * 24`. No trailing-day trim.
    pub fn n_hourly(&self) -> usize {
        self.n_days * 24
    }

    /// Half-open hourly index range into the parent axis (assumes hourly-native
    /// store; daily-native stores use `daily_range()` + repeat-24).
    pub fn hourly_range(&self) -> std::ops::Range<usize> {
        let h0 = self.start_day_idx * 24;
        h0..h0 + self.n_hourly()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn fake_axis(num_days: usize) -> TimeAxis {
        TimeAxis::new(
            NaiveDate::from_ymd_opt(1995, 10, 1).unwrap(),
            NaiveDate::from_ymd_opt(1995, 10, 1).unwrap() + Duration::days(num_days as i64 - 1),
        )
    }

    #[test]
    fn test_window_is_contiguous_unlike_rho_window() {
        let axis = fake_axis(30);
        let w = TestWindow::new(&axis, 0, 15);
        // RhoWindow with rho_days=15 → n_hourly = 14*24 = 336.
        // TestWindow with n_days=15 → n_hourly = 15*24 = 360. No trim.
        assert_eq!(w.n_hourly(), 15 * 24);
        assert_eq!(w.daily_range(), 0..15);
        assert_eq!(w.hourly_range(), 0..360);
    }

    #[test]
    fn consecutive_test_windows_tile_with_no_gap() {
        let axis = fake_axis(45);
        let w0 = TestWindow::new(&axis, 0, 15);
        let w1 = TestWindow::new(&axis, 15, 15);
        assert_eq!(w0.hourly_range().end, w1.hourly_range().start);
        assert_eq!(w0.daily_range().end, w1.daily_range().start);
    }
}
```

- [ ] **Step 2: Re-export from `src/data/mod.rs`**

Add to the module's `pub mod` block and `pub use` block:

```rust
pub mod test_window;
pub use test_window::TestWindow;
```

- [ ] **Step 3: Build + test**

```
cargo test --lib data::test_window 2>&1 | tail -8
```

Expected: 2 tests pass.

- [ ] **Step 4: Commit**

```
git add src/data/test_window.rs src/data/mod.rs
git commit -m "$(cat <<'EOF'
Add TestWindow for contiguous-hourly test-mode chunking

Unlike RhoWindow's (rho_days - 1) * 24 trim, TestWindow exposes
n_days * 24 hours so consecutive chunks tile the hourly axis without
gap. Required by SP-5's evaluate() loop which iterates time chunks
with carry_state.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

### Task 2: `MeritGagesDataset::collate_window` + `StaticNetworkCache`

**Files:**
- Modify: `src/data/dataset.rs`

Add a lazy cache of the all-gauges static network. First call to
`collate_window` builds it; subsequent calls reuse it and only slice
`q_prime` + `observations` to the requested window.

- [ ] **Step 1: Add `StaticNetworkCache` and the dataset field**

In `src/data/dataset.rs`, near the top of the file (after imports), add:

```rust
use std::cell::OnceCell;
```

(If `OnceCell` from `std::cell` is not available in this MSRV, use
`once_cell::sync::OnceCell` and add `once_cell = "1"` to Cargo.toml. Verify
at compile time.)

Add the cache struct (after the existing `RoutingBatch` impl block):

```rust
/// Lazy cache of the all-gauges static network, computed on first call to
/// `MeritGagesDataset::collate_window`. Reused across every test-mode batch.
///
/// All fields are computed once at first call and never mutated thereafter.
struct StaticNetworkCache {
    adjacency: SparseAdjacency,
    outflow_idx: Vec<Vec<usize>>,
    flow_scale: Vec<f32>,
    /// (F, N_active) — pre-normalized attribute matrix
    spatial_attributes_normalized: ndarray::Array2<f32>,
    /// (n_days_full, n_all_gauges) — full-period observations
    full_observations: ndarray::Array2<f32>,
    /// All filtered gauge STAIDs (mirrors `MeritGagesDataset::staids()`).
    gauge_staids: Vec<crate::data::ids::Staid>,
}
```

Add the cache field to `MeritGagesDataset`:

```rust
pub struct MeritGagesDataset {
    // ... existing fields ...
    static_network: OnceCell<StaticNetworkCache>,
}
```

Update `MeritGagesDataset::open` to initialize `static_network: OnceCell::new()`.

- [ ] **Step 2: Implement `collate_window`**

Add a new method on `MeritGagesDataset`:

```rust
impl MeritGagesDataset {
    /// Test-mode collation. Lazily builds and caches the all-gauges static
    /// network on first call; subsequent calls slice q_prime + observations
    /// for the given day-window.
    ///
    /// Mirrors DDR's `_test` per-batch RoutingDataclass construction with
    /// the simplification that the network is the full filtered-gauge union.
    ///
    /// `window` defines a contiguous-hourly chunk; SP-5 uses TestWindow
    /// rather than RhoWindow because chunked time must tile without the
    /// trailing-day trim.
    pub fn collate_window(
        &self,
        window: &crate::data::TestWindow,
    ) -> crate::data::Result<RoutingBatch> {
        // 1. Build (or fetch cached) static network.
        let cache = self.get_or_build_static_network()?;

        // 2. Read q_prime slice for this window from the streamflow store.
        //    StreamflowStore::read_window expects an hourly half-open range;
        //    use TestWindow::hourly_range() for contiguous coverage.
        let q_prime = self.streamflow.read_window_for_active_indices(
            window.hourly_range(),
            cache.adjacency.active_indices(), // see Step 3 — adapt to existing API
        )?;
        // q_prime shape: (n_hourly, n_active). Pre-scale by flow_scale.
        let mut q_prime = q_prime;
        for col in 0..q_prime.ncols() {
            let s = cache.flow_scale[col];
            for t in 0..q_prime.nrows() {
                q_prime[(t, col)] *= s;
            }
        }

        // 3. Slice observations from the cached full-period array.
        let obs = cache
            .full_observations
            .slice(ndarray::s![window.daily_range(), ..])
            .to_owned();

        Ok(RoutingBatch {
            adjacency: cache.adjacency.clone(),
            spatial_attributes_normalized: cache.spatial_attributes_normalized.clone(),
            q_prime,
            observations: obs,
            outflow_idx: cache.outflow_idx.clone(),
            gauge_staids: cache.gauge_staids.clone(),
            // The window field on RoutingBatch is a RhoWindow today; either
            // (a) extend RoutingBatch with `window: WindowKind` enum, or
            // (b) keep RoutingBatch's window field as RhoWindow and construct a
            //     "fake" RhoWindow with rho_days = window.n_days, accepting
            //     that the engine code only reads start_day_idx + n_days.
            // Choose (b) for minimal churn — the RhoWindow on RoutingBatch is
            // diagnostic only (per its doc comment in SP-3).
            window: crate::data::RhoWindow {
                start_day_idx: window.start_day_idx,
                rho_days: window.n_days,
                window_start: window.window_start,
            },
        })
    }

    fn get_or_build_static_network(&self) -> crate::data::Result<&StaticNetworkCache> {
        // Use get_or_try_init when available; otherwise check then build.
        if let Some(c) = self.static_network.get() {
            return Ok(c);
        }
        let cache = self.build_static_network()?;
        // OnceCell::set returns Err if already set; ignore that race.
        let _ = self.static_network.set(cache);
        Ok(self.static_network.get().unwrap())
    }

    fn build_static_network(&self) -> crate::data::Result<StaticNetworkCache> {
        // Union all filtered gauges' subgraphs. Reuse the existing SP-3 helpers.
        // The exact function names live in src/data/collate.rs:
        //   subgraph_union(staids, gages_adj, conus_adj) -> UnionedCoo
        //   compress(union, conus_attrs) -> CompressedNetwork
        //   build_flow_scale(staids, ...) -> Vec<f32>
        // Mirror MeritGagesDataset::collate but pass ALL staids and skip the
        // per-batch attribute lookup since the active-reach set is bigger.
        //
        // Specifically:
        let all_staids = self.staids().to_vec();
        // (delegate to private helpers — see existing collate impl for the pattern)
        ...
    }
}
```

The `build_static_network` body should be a near-verbatim copy of the
relevant blocks from `MeritGagesDataset::collate`, with two changes:
- Operates on ALL filtered staids (not a per-batch subset)
- Does NOT slice q_prime/obs (those happen per call in `collate_window`)

If the existing `collate` implementation is straightforward to refactor —
extract the "build network + attributes + observations" prefix into a
private helper that both `collate` and `build_static_network` call — do
that. If it's tightly intertwined with the per-batch staid subset, just
duplicate the prefix and add a TODO comment for future de-duplication.

- [ ] **Step 3: Verify the streamflow read API**

The plan template calls `streamflow.read_window_for_active_indices(...)`.
Inspect `src/data/store/icechunk.rs` to confirm the actual method name. If
it's `read_window(window: &RhoWindow, active_indices: &[usize])` instead,
adapt by constructing a temporary RhoWindow with the same start/range.

If `read_window` expects a `RhoWindow` and the trailing-day semantic
matters there too, you may need to add a parallel
`read_window_test(test_window: &TestWindow, ...)` that uses
`test_window.hourly_range()` directly. Document this in the commit message
if you change the streamflow API.

- [ ] **Step 4: Add a unit test against live data**

Append to `src/data/dataset.rs`'s test module (or `tests/data_dataset.rs`
if integration-style):

```rust
#[test]
fn collate_window_static_network_reuses_across_calls() {
    // Requires live data. Skip if not present.
    let cfg_path = "config/merit_training.yaml";
    if !std::path::Path::new(cfg_path).exists() { return; }
    let cfg = crate::config::Config::from_yaml_file(cfg_path).expect("yaml");
    if !crate::data::store::all_paths_exist(&cfg) { return; } // or local check

    let ds = MeritGagesDataset::open(&cfg).expect("open");
    let axis = ds.time_axis();

    let w1 = crate::data::TestWindow::new(axis, 0, 15);
    let w2 = crate::data::TestWindow::new(axis, 15, 15);

    let b1 = ds.collate_window(&w1).expect("w1");
    let b2 = ds.collate_window(&w2).expect("w2");

    // Network identity — same adjacency rows/cols/length/slope.
    assert_eq!(b1.adjacency.n, b2.adjacency.n);
    assert_eq!(b1.adjacency.nnz(), b2.adjacency.nnz()); // verify nnz accessor exists
    assert_eq!(b1.gauge_staids, b2.gauge_staids);
    assert_eq!(b1.outflow_idx, b2.outflow_idx);

    // q_prime differs (different time window).
    assert_eq!(b1.q_prime.nrows(), 15 * 24);
    assert_eq!(b2.q_prime.nrows(), 15 * 24);
}
```

- [ ] **Step 5: Build + test**

```
cargo build --tests 2>&1 | tail -5
cargo test --lib data::dataset 2>&1 | tail -10
cargo test --test training_verification v1_loss_matches 2>&1 | tail -5
```

Expected: clean compile, all dataset tests pass, V1 still passes.

- [ ] **Step 6: Commit**

```
git add src/data/dataset.rs
git commit -m "$(cat <<'EOF'
Add MeritGagesDataset::collate_window + static-network cache

Test mode iterates time chunks over a fixed all-gauges network. The
cache is built lazily on first call to collate_window and reused for
every subsequent chunk. Avoids duplicating the dataset struct, so
future network types only need to add collate_window alongside
collate; no parallel TestDataset hierarchy.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

### Task 3: Verify `carry_state` in `MuskingumCunge`

**Files:**
- Modify: `tests/mmc.rs` (add a verification test only; no engine code changes if it passes)
- Optional: `src/routing/mmc.rs` (only if the verification test fails)

The engine signature already accepts `carry_state` (`src/routing/mmc.rs:115`)
and conditionally regenerates `discharge_t` (`mmc.rs:164`:
`if !carry_state || self.discharge_t.is_none()`). This task verifies the
behavior with a hand-rolled unit test before SP-5 depends on it.

- [ ] **Step 1: Add a focused unit test**

Append to `tests/mmc.rs`:

```rust
#[test]
fn carry_state_preserves_discharge_across_setup_inputs_calls() {
    use ddrs::routing::mmc::{MuskingumCunge, RoutingInputs, SpatialParameters};
    use ddrs::config::Config;
    use ddrs::sparse::SparseAdjacency;
    use burn::backend::{Autodiff, NdArray};
    use burn::tensor::backend::Backend;
    use burn::tensor::Tensor;

    type I = NdArray<f32>;
    type AD = Autodiff<I>;

    let device = <I as Backend>::Device::default();
    let cfg = Config::default();
    // Tiny linear-chain network (3 reaches) — reuse linear_chain_sparse from tests/common.rs.
    let adjacency = ddrs::sparse::SparseAdjacency::from_dense(
        3,
        &[0.0, 0.0, 0.0,
          1.0, 0.0, 0.0,
          0.0, 1.0, 0.0],
        vec![1000.0; 3],
        vec![0.001; 3],
    );
    let x_storage = Tensor::<AD, 1>::from_floats([0.3_f32, 0.3, 0.3].as_slice(), &device);

    // Two distinct q_prime windows — confirms the engine isn't accidentally
    // re-using q_prime from the previous setup.
    let q1 = Tensor::<AD, 2>::from_floats(vec![5.0_f32; 3 * 24].as_slice(), &device)
        .reshape([24, 3]);
    let q2 = Tensor::<AD, 2>::from_floats(vec![8.0_f32; 3 * 24].as_slice(), &device)
        .reshape([24, 3]);
    let n_param = Tensor::<AD, 1>::from_floats([0.4_f32; 3].as_slice(), &device);
    let q_param = Tensor::<AD, 1>::from_floats([0.5_f32; 3].as_slice(), &device);
    let p_param = Tensor::<AD, 1>::from_floats([0.5_f32; 3].as_slice(), &device);

    let mut engine = MuskingumCunge::<I>::new(cfg.clone(), device.clone());

    // First call — cold start.
    engine.setup_inputs(
        RoutingInputs { adjacency: adjacency.clone(), x_storage: x_storage.clone() },
        q1,
        SpatialParameters { n: n_param.clone(), q_spatial: q_param.clone(), p_spatial: Some(p_param.clone()) },
        false, // carry_state
    );
    let _ = engine.forward(); // advances discharge_t
    let state_after_first: Vec<f32> = engine
        .discharge_state()
        .expect("discharge_state populated after forward")
        .into_data().into_vec().unwrap();

    // Second call with carry_state=true — must keep the same discharge_t,
    // NOT cold-start from q2's first row.
    engine.setup_inputs(
        RoutingInputs { adjacency, x_storage },
        q2,
        SpatialParameters { n: n_param, q_spatial: q_param, p_spatial: Some(p_param) },
        true, // carry_state
    );
    let state_after_second_setup: Vec<f32> = engine
        .discharge_state()
        .expect("discharge_state should survive carry_state=true setup")
        .into_data().into_vec().unwrap();

    assert_eq!(
        state_after_first, state_after_second_setup,
        "carry_state=true reset discharge_t — engine extension needed for SP-5"
    );
}
```

- [ ] **Step 2: Run the test**

```
cargo test --test mmc carry_state_preserves 2>&1 | tail -10
```

Expected: PASS. Reading `src/routing/mmc.rs:164`, the conditional
`if !carry_state || self.discharge_t.is_none()` does NOT recompute when
`carry_state=true` and `discharge_t.is_some()`. So this test should pass
without engine changes.

- [ ] **Step 3: If the test FAILS, extend the engine**

If the test fails (the engine resets `discharge_t` when it shouldn't),
inspect `src/routing/mmc.rs::setup_inputs` and remove the offending reset
inside the `carry_state=true` branch. Re-run the test until it passes.

- [ ] **Step 4: Commit**

```
git add tests/mmc.rs [src/routing/mmc.rs if changed]
git commit -m "$(cat <<'EOF'
Verify MuskingumCunge::setup_inputs preserves discharge_t with carry_state

SP-5 evaluate() depends on the engine carrying _discharge_t across
chunked time-window calls. Test asserts that setup_inputs(..., true)
after a forward call leaves discharge_state unchanged. The condition
already in mmc.rs:164 covers this; the test pins it down.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

### Task 4: Thread `carry_state` through `forward` and `forward_with_frozen_params`

**Files:**
- Modify: `src/training/forward.rs`
- Modify: `src/training/driver.rs` (existing caller — pass `false` explicitly)

- [ ] **Step 1: Add `carry_state: bool` arg to both functions**

In `src/training/forward.rs`, change the signatures:

```rust
pub fn forward_with_frozen_params<I: Backend>(
    cfg: &Config,
    tensors: &RoutingTensors<I>,
    frozen: &FrozenParams,
    device: &I::Device,
    carry_state: bool,    // NEW — pass through to engine.setup_inputs
) -> Tensor<I, 2>;

pub fn forward<I: Backend>(
    cfg: &Config,
    tensors: &RoutingTensors<Autodiff<I>>,
    mlp: &Mlp<Autodiff<I>>,
    device: &I::Device,
    carry_state: bool,    // NEW
) -> Tensor<Autodiff<I>, 2>;
```

Inside each function, replace the existing `false` arg in
`engine.setup_inputs(..., false)` with `carry_state`.

- [ ] **Step 2: Update existing caller in driver.rs**

In `src/training/driver.rs`, the `train()` loop calls `forward(...)`. Pass
`false` explicitly:

```rust
let pred_hourly = forward::<I>(cfg, &tensors, &state.mlp, device, false);
```

Training never carries state between mini-batches (each mini-batch has a
different per-batch subgraph; the discharge_t shape wouldn't match anyway).

- [ ] **Step 3: Update existing tests**

In `tests/training_verification.rs`, V1 and V2 call
`forward_with_frozen_params`. Add `false` as the last arg:

```rust
let pred_hourly = forward_with_frozen_params::<NdArray<f32>>(
    &cfg, &tensors, &frozen, &device, false,
);
```

(Each test invokes the function exactly once; grep-and-edit.)

- [ ] **Step 4: Build + verify V1 still passes**

```
cargo build --tests 2>&1 | tail -5
cargo test --test training_verification v1_loss_matches 2>&1 | tail -5
```

Expected: clean compile, V1 passes.

- [ ] **Step 5: Commit**

```
git add src/training/forward.rs src/training/driver.rs tests/training_verification.rs
git commit -m "$(cat <<'EOF'
Thread carry_state through forward + forward_with_frozen_params

SP-5 evaluate() needs to pass carry_state=i>0 across chunked test-mode
batches. Existing training and V1/V2 callers explicitly pass false.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

### Task 5: Add `forward_eval` for non-autodiff MLP path

**Files:**
- Modify: `src/training/forward.rs`

`bin/eval` loads an `Mlp<I>` (non-autodiff) from a checkpoint and runs
inference. The existing `forward()` requires `Mlp<Autodiff<I>>` and
returns `Tensor<Autodiff<I>, 2>` with an active tape — wasteful for eval.
This task adds a third forward variant that operates entirely on the
inner backend.

- [ ] **Step 1: Append `forward_eval` to `src/training/forward.rs`**

```rust
/// MLP inference forward — no autograd anywhere. Used by `bin/eval` and
/// the Mlp arm of `EvalParams`.
///
/// Mirrors `forward` (production training path) but operates on the inner
/// backend `I` throughout. Caller passes an `Mlp<I>` loaded via
/// `checkpoint::load_mlp`.
pub fn forward_eval<I: Backend>(
    cfg: &Config,
    tensors: &RoutingTensors<I>,
    mlp: &Mlp<I>,
    device: &I::Device,
    carry_state: bool,
) -> Tensor<I, 2> {
    let params_map = mlp.forward(tensors.spatial_attributes.clone());

    let n_param = params_map.get("n").expect("MLP missing n").clone();
    let q_param = params_map.get("q_spatial").expect("MLP missing q_spatial").clone();
    let p_param = params_map.get("p_spatial").cloned();

    let n_active = tensors.adjacency.n;
    let x_storage: Tensor<I, 1> = Tensor::full([n_active], 0.3_f32, device);

    // Wrap to Autodiff at the engine boundary (engine requires Autodiff
    // even for forward-only). Drop the graph immediately after with .inner().
    let device_a: <burn::backend::Autodiff<I> as Backend>::Device = device.clone();
    let q_prime_ad: Tensor<burn::backend::Autodiff<I>, 2> =
        Tensor::from_inner(tensors.q_prime.clone());
    let n_ad = Tensor::from_inner(n_param);
    let q_ad = Tensor::from_inner(q_param);
    let p_ad = p_param.map(Tensor::from_inner);
    let x_ad = Tensor::from_inner(x_storage);

    let mut engine = MuskingumCunge::<I>::new(cfg.clone(), device.clone());
    engine.setup_inputs(
        RoutingInputs { adjacency: tensors.adjacency.clone(), x_storage: x_ad },
        q_prime_ad,
        SpatialParameters { n: n_ad, q_spatial: q_ad, p_spatial: p_ad },
        carry_state,
    );
    let runoff_ad = engine.forward();
    let runoff = runoff_ad.inner();

    scatter_add_by_group(
        runoff,
        tensors.flat_indices.clone(),
        tensors.group_ids.clone(),
        tensors.num_gauges,
    )
}
```

Note: the engine is built on `I`. `MuskingumCunge::<I>::new` always wraps
its internal tensors with `Autodiff<I>` (per the existing impl); the
caller-side `.inner()` drops the wrapper for the returned runoff. This
parallels `forward_with_frozen_params` exactly.

If `from_inner` for `p_param.map(...)` runs into a type-coercion issue
(specifically the `Option<Tensor<...>>` arm), fall back to:

```rust
let p_ad = match p_param {
    Some(t) => Some(Tensor::<burn::backend::Autodiff<I>, 1>::from_inner(t)),
    None => None,
};
```

- [ ] **Step 2: Re-export from `src/training/mod.rs`**

```rust
pub use forward::forward_eval;
```

- [ ] **Step 3: Build + verify no V1/V2 regressions**

```
cargo build --tests 2>&1 | tail -5
cargo test --test training_verification v1_loss_matches 2>&1 | tail -5
```

Expected: clean, V1 passes.

- [ ] **Step 4: Commit**

```
git add src/training/forward.rs src/training/mod.rs
git commit -m "$(cat <<'EOF'
Add forward_eval for non-autodiff MLP inference path

Parallels forward() but operates on the inner backend throughout. Used
by SP-5's evaluate() Mlp arm and the bin/eval CLI. Engine still requires
Autodiff internally; wrap at the boundary and drop the graph with
.inner() before returning.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

### Task 6: `evaluate()` driver + `EvalParams` + `EvalOutput`

**Files:**
- Create: `src/training/eval.rs`
- Modify: `src/training/mod.rs`

The main test-phase loop. Iterates contiguous time chunks, scatters per-batch
predictions into a full-period accumulator, applies a single end-of-pipeline
`tau_trim_and_downsample`, slices observations, and computes Metrics.

- [ ] **Step 1: Create `src/training/eval.rs`**

```rust
//! Test-phase evaluation loop. Mirrors
//! `~/projects/ddr/scripts/train_and_test.py::_test` (lines 43-119).
//!
//! Unlike the training loop, batches iterate TIME (not gauges) and the
//! network is the static all-gauges union. `carry_state=i>0` propagates
//! engine state across consecutive chunks.

use std::path::Path;

use burn::tensor::backend::Backend;
use chrono::NaiveDate;
use ndarray::{s, Array2};

use crate::config::Config;
use crate::data::dataset::MeritGagesDataset;
use crate::data::TestWindow;
use crate::data::error::Result;
use crate::nn::mlp::Mlp;
use crate::training::{
    forward_eval, forward_with_frozen_params, tau_trim_and_downsample,
    FrozenParams, Metrics,
};

/// Source of MC parameters at eval time.
pub enum EvalParams<'a, I: Backend> {
    /// V4 verification path — uniform scalar n/q/p across every reach.
    Frozen(&'a FrozenParams),
    /// Production path — pass through an already-trained MLP (non-autodiff).
    Mlp(&'a Mlp<I>),
}

pub struct EvalOutput {
    /// (n_all_gauges, n_days_trimmed) — daily-downsampled per-gauge predictions
    /// after tau-trim. n_days_trimmed = n_days_full - 1 (per SP-4 Task 4 math:
    /// (T_hours - 24) / 24).
    pub predictions_daily: Array2<f32>,
    /// (n_all_gauges, n_days_trimmed) — observations sliced [1..-1] along time
    /// to match DDR's compute_daily_runoff convention.
    pub observations_daily: Array2<f32>,
    pub gage_ids: Vec<String>,
    pub time_range_daily: Vec<NaiveDate>,
    pub metrics: Metrics,
}

/// Run the test-phase loop and return predictions + observations + metrics.
///
/// `batch_size_days` controls the chunk size. For V4 single-batch, pass
/// `dataset.time_axis().num_days`. For DDR-style multi-batch, pass 15.
pub fn evaluate<I: Backend>(
    cfg: &Config,
    dataset: &MeritGagesDataset,
    params: EvalParams<I>,
    device: &I::Device,
    batch_size_days: usize,
) -> Result<EvalOutput> {
    let axis = dataset.time_axis().clone();
    let n_days_total = axis.num_days;
    assert!(batch_size_days > 0, "batch_size_days must be positive");

    // Determine n_active by building one window (forces the static-network cache).
    let first_window = TestWindow::new(&axis, 0, batch_size_days.min(n_days_total));
    let first_batch = dataset.collate_window(&first_window)?;
    let n_active = first_batch.adjacency.n;
    let n_all_gauges = first_batch.gauge_staids.len();
    let n_hours_full = n_days_total * 24;

    // Accumulator: (n_all_gauges, n_hours_full) — written per chunk.
    let mut predictions_full = Array2::<f32>::zeros((n_all_gauges, n_hours_full));

    // Helper: lift a TestWindow chunk into a per-chunk prediction array.
    //          Returns (G, chunk_hours).
    let mut process_chunk = |window: &TestWindow,
                             carry_state: bool|
     -> Result<Array2<f32>> {
        let batch = dataset.collate_window(window)?;
        let tensors = batch.to_tensors::<I>(device);
        let pred = match params {
            EvalParams::Frozen(frozen) => {
                forward_with_frozen_params::<I>(cfg, &tensors, frozen, device, carry_state)
            }
            EvalParams::Mlp(mlp) => {
                forward_eval::<I>(cfg, &tensors, mlp, device, carry_state)
            }
        };
        let dims = pred.dims();
        debug_assert_eq!(dims[0], n_all_gauges);
        debug_assert_eq!(dims[1], window.n_hourly());
        let v: Vec<f32> = pred.into_data().into_vec().unwrap();
        Ok(Array2::from_shape_vec((dims[0], dims[1]), v).unwrap())
    };

    // Process first chunk (already built above for n_active probe).
    // To avoid double-engine-run, reconstruct tensors from first_batch:
    let tensors0 = first_batch.to_tensors::<I>(device);
    let pred0 = match params {
        EvalParams::Frozen(frozen) => {
            forward_with_frozen_params::<I>(cfg, &tensors0, frozen, device, false)
        }
        EvalParams::Mlp(mlp) => {
            forward_eval::<I>(cfg, &tensors0, mlp, device, false)
        }
    };
    let v0: Vec<f32> = pred0.into_data().into_vec().unwrap();
    let pred0_arr = Array2::from_shape_vec(
        (n_all_gauges, first_window.n_hourly()),
        v0,
    ).unwrap();
    let h0_end = first_window.n_hourly();
    predictions_full.slice_mut(s![.., 0..h0_end]).assign(&pred0_arr);

    // Remaining chunks.
    let mut day_offset = first_window.n_days;
    let mut chunk_idx = 1usize;
    while day_offset < n_days_total {
        let chunk_n = (n_days_total - day_offset).min(batch_size_days);
        let win = TestWindow::new(&axis, day_offset, chunk_n);
        let pred_arr = process_chunk(&win, true)?;
        let h_start = day_offset * 24;
        let h_end = h_start + win.n_hourly();
        predictions_full.slice_mut(s![.., h_start..h_end]).assign(&pred_arr);
        day_offset += chunk_n;
        chunk_idx += 1;
    }
    let _ = chunk_idx;

    // End-of-pipeline tau-trim + daily downsample. Lift to BURN tensor for
    // the existing tau_trim_and_downsample helper.
    let pred_full_vec: Vec<f32> = predictions_full.iter().copied().collect();
    let pred_full_t: burn::tensor::Tensor<I, 2> =
        burn::tensor::Tensor::<I, 1>::from_floats(pred_full_vec.as_slice(), device)
            .reshape([n_all_gauges, n_hours_full]);
    let daily_t = tau_trim_and_downsample(pred_full_t, cfg.params.tau);
    let daily_dims = daily_t.dims();
    let daily_vec: Vec<f32> = daily_t.into_data().into_vec().unwrap();
    let predictions_daily =
        Array2::from_shape_vec((daily_dims[0], daily_dims[1]), daily_vec).unwrap();

    // Observations: read full daily-period from the static cache via collate_window.
    // The cache holds (n_days_full, n_all_gauges); slice [1..-1] along time and
    // transpose to (n_all_gauges, n_days_full - 2) to match daily convention.
    //
    // n_days_trimmed from tau_trim = (n_hours_full - 24) / 24 = n_days_total - 1.
    // DDR's obs[:, 1:-1] gives n_days_total - 2 entries. MISMATCH BY ONE.
    //
    // Investigate: SP-4's tau_trim leaves (n_hours - 24)/24 daily entries when
    // input hours = n_days_total * 24. SP-3 RhoWindow training had (n_days-1)*24
    // hours so the trim left n_days-2 days. SP-5 uses n_days*24 hours, so the
    // trim leaves n_days-1 days. The trailing day in SP-5 is the OBSERVED last
    // day (no boundary trim before tau).
    //
    // DDR's obs[:, 1:-1] strips first AND last day. ddrs after-tau-trim
    // predictions correspond to days [1..n_days-1] (Python slice notation:
    // [1..-1]). So obs[1..-1] (n_days-2 entries) and predictions (n_days-1
    // entries) DIFFER BY ONE.
    //
    // The Python tau_trim slices `[13+tau : -11+tau]` = drop 13+tau hours from
    // the front and 11-tau from the back. For tau=3: front 16, back 8. With
    // n_hours = n_days * 24 (SP-5 convention), trimmed hours = n_days*24 - 24
    // = (n_days-1)*24, then /24 = n_days - 1 daily entries.
    //
    // CRITICAL: confirm DDR's obs[:, 1:-1] aligns with predictions[:, 0..-1]
    // (drop first day of obs, drop last day of predictions). Inspect
    // ~/projects/ddr/src/ddr/scripts_utils.py::compute_daily_runoff to verify
    // which days the trimmed window spans.
    //
    // SAFE CONSERVATIVE choice for this task: slice predictions[:, 0..-1] AND
    // obs[:, 1..-1] so both end up with (n_days - 2) days. If the reference
    // dump (Task 9) reveals the alignment differently, fix here.
    let obs_full = dataset.collate_window(&first_window)?.observations.clone();
    // ^ This re-builds first_batch — but the static_network cache means it's cheap.
    // (Better: extract a private dataset.cached_full_observations() helper.)
    // To get full observations across the whole window, we want a method that
    // returns the cache directly. As a pragmatic shortcut, collate one big
    // TestWindow covering the whole axis:
    let full_window = TestWindow::new(&axis, 0, n_days_total);
    let full_batch = dataset.collate_window(&full_window)?; // streams the q_prime — wasteful
    let obs_full: Array2<f32> = full_batch.observations; // (n_days_total, n_all_gauges)

    // Trim obs by [1..-1] along axis 0 → (n_days_total - 2, n_all_gauges)
    let obs_trimmed: Array2<f32> = obs_full
        .slice(s![1..-1, ..])
        .to_owned();
    // Transpose to (n_all_gauges, n_days_total - 2)
    let observations_daily: Array2<f32> = obs_trimmed.reversed_axes().as_standard_layout().to_owned();

    // Drop the trailing day of predictions to match (n_all_gauges, n_days_total - 2).
    let pd_dims = predictions_daily.dim();
    let predictions_daily = predictions_daily
        .slice(s![.., 0..pd_dims.1 - 1])
        .to_owned();

    debug_assert_eq!(predictions_daily.shape()[1], observations_daily.shape()[1]);

    // Daily time range = axis.start + 1 .. axis.end (exclusive), matching DDR's
    // daily_time_range[1:-1].
    let time_range_daily: Vec<NaiveDate> = (1..n_days_total - 1)
        .map(|i| axis.start + chrono::Duration::days(i as i64))
        .collect();

    let warmup = cfg.experiment.as_ref().expect("experiment").warmup;
    let metrics = Metrics::compute(
        &predictions_daily.slice(s![.., warmup..]).to_owned(),
        &observations_daily.slice(s![.., warmup..]).to_owned(),
    );

    let gage_ids: Vec<String> = first_batch
        .gauge_staids
        .iter()
        .map(|s| s.as_str().to_string())
        .collect();

    Ok(EvalOutput {
        predictions_daily,
        observations_daily,
        gage_ids,
        time_range_daily,
        metrics,
    })
}
```

**Note on observation re-fetch:** The implementation above calls
`collate_window` twice (once for the first chunk, once with the full
window) which re-reads q_prime needlessly. Acceptable for a first pass.
If profiling shows this hurts (it shouldn't materially — icechunk reads
are the bulk), add a `MeritGagesDataset::full_observations()` accessor
that returns the cached observation array without rebuilding the static
network. Defer.

**Note on `n_active`:** captured for diagnostic prints if needed; not used
in arithmetic. Drop the binding if clippy complains.

- [ ] **Step 2: Re-export from `src/training/mod.rs`**

```rust
pub mod eval;
pub use eval::{evaluate, EvalOutput, EvalParams};
```

- [ ] **Step 3: Build (no test yet — V4 covers it)**

```
cargo build --tests 2>&1 | tail -10
cargo test --test training_verification v1_loss_matches 2>&1 | tail -5
```

Expected: clean compile, V1 still passes.

- [ ] **Step 4: Commit**

```
git add src/training/eval.rs src/training/mod.rs
git commit -m "$(cat <<'EOF'
Add evaluate() driver with EvalParams + EvalOutput

Test-phase loop iterates contiguous time chunks via TestWindow,
scatters per-batch predictions into a full-period accumulator, runs
a single end-of-pipeline tau_trim_and_downsample, and returns daily
predictions + observations + post-warmup metrics. V4 (single-batch)
and V4b (multi-batch) tests in later tasks exercise it.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

### Task 7: Inspect DDR's zarr + implement `write_predictions_zarr`

**Files:**
- Create: `src/training/zarr_io.rs`
- Modify: `src/training/mod.rs`

- [ ] **Step 1: Inspect a reference DDR zarr to confirm codecs**

Run (under DDR's uv venv, since `zarr` package may not be available in ddrs):

```
cd ~/projects/ddr && uv run python -c "
import zarr
g = zarr.open('/mnt/ssd1/data/icechunk/usgs_daily_observations', mode='r')
print(g.info)
for k, v in g.items():
    print(k, v.shape, v.dtype, v.attrs.asdict(), v.metadata.codecs)
"
```

(Adjust the path to any existing DDR-produced zarr — the icechunk
observations store is the closest example. If that fails because it's an
icechunk store not a plain zarr, find a `model_test.zarr` from a prior
DDR run.)

Document the relevant findings in `src/training/zarr_io.rs`'s module doc
comment:
- zarr format version (v2 or v3)
- string codec for gage_ids (variable-length UTF-8 vs fixed-width ASCII)
- time encoding (int64 nanoseconds + units attr)
- chunking strategy DDR uses

If DDR uses zarr v2, ddrs writes v3 since `zarrs` 0.23 is v3-first. xarray
on the read side handles both — the V4 fixture path can use v3 and DDR's
notebooks just need a re-read.

- [ ] **Step 2: Create `src/training/zarr_io.rs`**

```rust
//! Write per-gauge daily predictions + observations to a zarr store with
//! a layout compatible with DDR's `_test` output.
//!
//! Verified DDR-side layout (see Task 7 Step 1):
//!   predictions  shape (n_gauges, n_days)  dtype f64  attrs {units, long_name}
//!   observations shape (n_gauges, n_days)  dtype f64  attrs {units, long_name}
//!   gage_ids     shape (n_gauges,)         dtype string
//!   time         shape (n_days,)           dtype i64 (ns since epoch)
//!
//! Group attrs: description, "start time", "end time", version,
//! "evaluation basins file", model.

use std::path::Path;
use std::sync::Arc;

use zarrs::array::Array as ZarrArray;
use zarrs::array::{ArrayBuilder, DataType, FillValue};
use zarrs::filesystem::FilesystemStore;
use zarrs::group::GroupBuilder;
use zarrs::storage::WritableStorage;

use crate::data::error::{DataError, Result};
use crate::training::eval::EvalOutput;

pub struct ZarrAttrs<'a> {
    pub start_time: &'a str,
    pub end_time: &'a str,
    pub version: &'a str,
    pub evaluation_basins_file: &'a Path,
    pub model_label: &'a str,
}

pub fn write_predictions_zarr(
    path: &Path,
    output: &EvalOutput,
    attrs: ZarrAttrs<'_>,
) -> Result<()> {
    let storage: WritableStorage =
        Arc::new(FilesystemStore::new(path).map_err(|e| zarr_err(path, e))?);

    // Root group with top-level attrs.
    let mut root_attrs = serde_json::Map::new();
    root_attrs.insert("description".into(),
        serde_json::Value::String("Predictions and obs for time period".into()));
    root_attrs.insert("start time".into(), attrs.start_time.into());
    root_attrs.insert("end time".into(), attrs.end_time.into());
    root_attrs.insert("version".into(), attrs.version.into());
    root_attrs.insert("evaluation basins file".into(),
        attrs.evaluation_basins_file.display().to_string().into());
    root_attrs.insert("model".into(), attrs.model_label.into());

    let _root = GroupBuilder::new()
        .attributes(root_attrs)
        .build(storage.clone(), "/")
        .map_err(|e| zarr_err(path, e))?
        .store_metadata()
        .map_err(|e| zarr_err(path, e))?;

    let (n_gauges, n_days) = output.predictions_daily.dim();

    // predictions array — f64 to match DDR.
    let predictions_f64: Vec<f64> = output
        .predictions_daily
        .iter()
        .map(|&v| v as f64)
        .collect();
    write_2d_f64_array(
        storage.clone(),
        path,
        "/predictions",
        &predictions_f64,
        (n_gauges, n_days),
        &[("units", "m3/s"), ("long_name", "Streamflow"),
          ("_ARRAY_DIMENSIONS", "gage_ids,time")],
    )?;

    let obs_f64: Vec<f64> = output
        .observations_daily
        .iter()
        .map(|&v| v as f64)
        .collect();
    write_2d_f64_array(
        storage.clone(),
        path,
        "/observations",
        &obs_f64,
        (n_gauges, n_days),
        &[("units", "m3/s"), ("long_name", "Observed Streamflow"),
          ("_ARRAY_DIMENSIONS", "gage_ids,time")],
    )?;

    // gage_ids — fixed-width ASCII fallback if VL string is fussy.
    // STAID is zero-padded 8 chars; use [u8; 8] / dtype "|S8".
    write_string_array(storage.clone(), path, "/gage_ids", &output.gage_ids)?;

    // time — int64 nanoseconds since epoch + units attr.
    let time_ns: Vec<i64> = output
        .time_range_daily
        .iter()
        .map(|d| {
            d.and_hms_opt(0, 0, 0).unwrap()
                .and_utc().timestamp_nanos_opt().unwrap()
        })
        .collect();
    write_1d_i64_array(
        storage,
        path,
        "/time",
        &time_ns,
        &[
            ("units", "nanoseconds since 1970-01-01"),
            ("calendar", "proleptic_gregorian"),
            ("_ARRAY_DIMENSIONS", "time"),
        ],
    )?;

    Ok(())
}

// ---------- private helpers ----------

fn write_2d_f64_array(
    storage: WritableStorage,
    path: &Path,
    array_path: &str,
    data: &[f64],
    shape: (usize, usize),
    attrs: &[(&str, &str)],
) -> Result<()> {
    let array = ArrayBuilder::new(
        vec![shape.0 as u64, shape.1 as u64],
        vec![shape.0 as u64, shape.1 as u64], // single chunk; tune later
        DataType::Float64,
        FillValue::from(0.0_f64),
    )
    .attributes(json_attrs(attrs))
    .build(storage, array_path)
    .map_err(|e| zarr_err(path, e))?;
    array.store_metadata().map_err(|e| zarr_err(path, e))?;
    let subset = array.subset_all();
    array
        .store_array_subset::<Vec<f64>>(&subset, data.to_vec())
        .map_err(|e| zarr_err(path, e))?;
    Ok(())
}

fn write_1d_i64_array(
    storage: WritableStorage,
    path: &Path,
    array_path: &str,
    data: &[i64],
    attrs: &[(&str, &str)],
) -> Result<()> {
    let array = ArrayBuilder::new(
        vec![data.len() as u64],
        vec![data.len() as u64],
        DataType::Int64,
        FillValue::from(0_i64),
    )
    .attributes(json_attrs(attrs))
    .build(storage, array_path)
    .map_err(|e| zarr_err(path, e))?;
    array.store_metadata().map_err(|e| zarr_err(path, e))?;
    let subset = array.subset_all();
    array
        .store_array_subset::<Vec<i64>>(&subset, data.to_vec())
        .map_err(|e| zarr_err(path, e))?;
    Ok(())
}

fn write_string_array(
    storage: WritableStorage,
    path: &Path,
    array_path: &str,
    strings: &[String],
) -> Result<()> {
    // Fixed-width ASCII 8 chars (STAID format). dtype |S8 in xarray.
    // If zarrs 0.23 doesn't expose fixed-width bytes natively, encode as
    // a [u8; 8 * n] flat array with dtype "|S8" attr.
    //
    // IMPLEMENTER: inspect zarrs' supported DataType variants. If VL string
    // is supported (DataType::String or similar), prefer it. If not, pad
    // each STAID to 8 bytes and store as raw bytes.
    //
    // Placeholder body — adapt at the keyboard:
    let n = strings.len();
    let mut buf = vec![0u8; n * 8];
    for (i, s) in strings.iter().enumerate() {
        let bytes = s.as_bytes();
        let len = bytes.len().min(8);
        buf[i * 8..i * 8 + len].copy_from_slice(&bytes[..len]);
    }
    // ... build a 2D u8 array (n, 8) OR a custom dtype array. Depends on zarrs.
    // For now, a workable fallback is:
    let array = ArrayBuilder::new(
        vec![n as u64, 8],
        vec![n as u64, 8],
        DataType::UInt8,
        FillValue::from(0_u8),
    )
    .attributes(serde_json::json!({
        "_ARRAY_DIMENSIONS": ["gage_ids", "char"],
        "_dtype_hint": "|S8",
    }).as_object().unwrap().clone())
    .build(storage, array_path)
    .map_err(|e| zarr_err(path, e))?;
    array.store_metadata().map_err(|e| zarr_err(path, e))?;
    let subset = array.subset_all();
    array
        .store_array_subset::<Vec<u8>>(&subset, buf)
        .map_err(|e| zarr_err(path, e))?;
    Ok(())
}

fn json_attrs(pairs: &[(&str, &str)]) -> serde_json::Map<String, serde_json::Value> {
    let mut m = serde_json::Map::new();
    for (k, v) in pairs {
        // _ARRAY_DIMENSIONS xarray convention is a list, not a string.
        if *k == "_ARRAY_DIMENSIONS" {
            m.insert(k.to_string(), serde_json::Value::Array(
                v.split(',').map(|s| serde_json::Value::String(s.to_string())).collect()
            ));
        } else {
            m.insert(k.to_string(), serde_json::Value::String(v.to_string()));
        }
    }
    m
}

fn zarr_err<E: std::error::Error + Send + Sync + 'static>(path: &Path, source: E) -> DataError {
    DataError::Zarr { path: path.to_path_buf(), source: Box::new(source) }
}
```

The `_ARRAY_DIMENSIONS` attr is xarray's convention for binding array
dimensions to coordinate names. Without it, xarray can still open the
zarr but loses dimension metadata.

The `write_string_array` fallback uses `dtype: |S8` semantically by
writing a `(n, 8)` UInt8 array. xarray reading this won't auto-decode it
as strings — the downstream notebooks may need a small adapter:
```python
ds["gage_ids"] = (("gage_ids",), [b"".join(row).decode().rstrip("\x00") for row in ds["gage_ids"].values])
```
Document this in the doc-comment.

If `zarrs` 0.23 supports zarr v3 variable-length strings (`DataType::String`
or similar), use that and skip the fallback.

- [ ] **Step 2: Re-export from `src/training/mod.rs`**

```rust
pub mod zarr_io;
pub use zarr_io::{write_predictions_zarr, ZarrAttrs};
```

- [ ] **Step 3: Add a round-trip unit test**

Append a test inside `src/training/zarr_io.rs`:

```rust
#[cfg(test)]
mod tests {
    use super::*;
    use crate::training::metrics::Metrics;
    use chrono::NaiveDate;
    use ndarray::array;

    #[test]
    fn write_then_read_round_trip() {
        // Build a tiny EvalOutput and write to /tmp.
        let pred = array![[1.0_f32, 2.0, 3.0], [4.0, 5.0, 6.0]]; // (G=2, T=3)
        let obs  = array![[1.1_f32, 2.1, 3.1], [4.1, 5.1, 6.1]];
        let out = EvalOutput {
            predictions_daily: pred,
            observations_daily: obs,
            gage_ids: vec!["00000001".into(), "00000002".into()],
            time_range_daily: vec![
                NaiveDate::from_ymd_opt(1995, 10, 2).unwrap(),
                NaiveDate::from_ymd_opt(1995, 10, 3).unwrap(),
                NaiveDate::from_ymd_opt(1995, 10, 4).unwrap(),
            ],
            metrics: Metrics { nse: vec![0.5, 0.6], rmse: vec![0.1, 0.1], kge: vec![0.4, 0.5] },
        };
        let dir = tempdir_for_zarr();
        let zpath = dir.join("test.zarr");
        let _ = std::fs::remove_dir_all(&zpath);
        std::fs::create_dir_all(&zpath).expect("mkdir");
        let attrs = ZarrAttrs {
            start_time: "1995-10-01",
            end_time: "1995-10-05",
            version: "test",
            evaluation_basins_file: std::path::Path::new("/tmp/fake_gages.csv"),
            model_label: "frozen",
        };
        write_predictions_zarr(&zpath, &out, attrs).expect("write");

        // Verify by reading the metadata back via zarrs.
        let storage: WritableStorage =
            std::sync::Arc::new(FilesystemStore::new(&zpath).expect("open"));
        let arr = ZarrArray::open(storage, "/predictions").expect("open predictions");
        let dims = arr.shape();
        assert_eq!(dims, &[2, 3]);
    }

    fn tempdir_for_zarr() -> std::path::PathBuf {
        let mut p = std::env::temp_dir();
        p.push(format!("ddrs_zarr_test_{}", std::process::id()));
        p
    }
}
```

The round-trip exercises the write path; full schema validation comes
in Task 11 (V4) when DDR's reference zarr is in hand.

- [ ] **Step 4: Build + run the test**

```
cargo test --lib training::zarr_io 2>&1 | tail -10
```

Expected: 1 test passes.

- [ ] **Step 5: Commit**

```
git add src/training/zarr_io.rs src/training/mod.rs
git commit -m "$(cat <<'EOF'
Add write_predictions_zarr with DDR-compatible layout

Two f64 arrays (predictions, observations), one i64 time array
(nanoseconds since epoch), and a fixed-width UInt8 gage_ids fallback
when zarrs lacks VL string support. _ARRAY_DIMENSIONS attrs preserve
xarray's coordinate binding. Round-trip test exercises the write path.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

### Task 8: Config `testing:` overlay + `ConfigMode`

**Files:**
- Modify: `src/config.rs`
- Modify: `config/merit_training.yaml`

- [ ] **Step 1: Extend `merit_training.yaml`**

Append a `testing:` section after `experiment:`:

```yaml
# Test-mode overlay. When the binary is run with --mode testing, these keys
# replace the corresponding ones in `experiment:`. Absent keys inherit.
#
# IMPORTANT: batch_size SEMANTIC SHIFTS between modes:
#   - experiment.batch_size (training) = number of GAUGES per mini-batch
#   - testing.batch_size             = number of DAYS per chunk
testing:
  start_time: 1995/10/01
  end_time: 2010/09/30
  batch_size: 15      # DAYS, not gauges
  rho: null           # disabled in test mode
```

- [ ] **Step 2: Add `ConfigMode` + raw section + overlay logic**

In `src/config.rs`:

```rust
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub enum ConfigMode {
    Training,
    Testing,
}

// Raw deserialization helper.
#[derive(Debug, Default, Deserialize)]
struct TestingOverridesRaw {
    pub start_time: Option<String>,
    pub end_time: Option<String>,
    pub batch_size: Option<usize>,
    pub rho: Option<Option<usize>>, // double Option: present-but-null vs absent
    pub warmup: Option<usize>,
    pub epochs: Option<usize>,
    pub grad_clip_max_norm: Option<f32>,
    pub checkpoint: Option<String>,
}
```

Add a sibling field on the existing top-level raw config:

```rust
#[derive(Debug, Default, Deserialize)]
struct ConfigRaw {
    // ... existing fields ...
    #[serde(default)]
    testing: TestingOverridesRaw,
}
```

Add the loader:

```rust
impl Config {
    /// Existing back-compat entrypoint = Training mode.
    pub fn from_yaml_file(path: impl AsRef<Path>) -> Result<Self> {
        Self::from_yaml_file_with_mode(path, ConfigMode::Training)
    }

    /// New entrypoint: load YAML, optionally overlay `testing:` keys.
    pub fn from_yaml_file_with_mode(
        path: impl AsRef<Path>,
        mode: ConfigMode,
    ) -> Result<Self> {
        let raw: ConfigRaw = read_yaml(path)?;
        let mut cfg: Self = raw.into();
        if mode == ConfigMode::Testing {
            let raw_test = /* re-read or re-borrow the testing section */;
            apply_testing_overlay(&mut cfg, raw_test);
        }
        Ok(cfg)
    }
}

fn apply_testing_overlay(cfg: &mut Config, overrides: TestingOverridesRaw) {
    let Some(exp) = cfg.experiment.as_mut() else { return; };
    if let Some(v) = overrides.start_time { exp.start_time = v; }
    if let Some(v) = overrides.end_time { exp.end_time = v; }
    if let Some(v) = overrides.batch_size { exp.batch_size = v; }
    if let Some(v) = overrides.rho { exp.rho = v; } // None-of-None = absent;
                                                    // Some(None) = explicit null override
    if let Some(v) = overrides.warmup { exp.warmup = v; }
    if let Some(v) = overrides.epochs { exp.epochs = v; }
    if let Some(v) = overrides.grad_clip_max_norm { exp.grad_clip_max_norm = Some(v); }
    if let Some(v) = overrides.checkpoint {
        exp.checkpoint = Some(std::path::PathBuf::from(v));
    }
}
```

The double-Option for `rho` (`Option<Option<usize>>`) lets the YAML
distinguish "key absent" from "key present with value null". serde-yaml
honors this convention.

If the existing `ConfigRaw → Config` `From` impl doesn't expose
`TestingOverridesRaw` to the loader, restructure to keep the raw struct
alive across the `From` conversion:

```rust
pub fn from_yaml_file_with_mode(...) -> Result<Self> {
    let raw: ConfigRaw = read_yaml(path)?;
    let testing_raw = raw.testing.clone();
    let mut cfg: Self = raw.into();
    if mode == ConfigMode::Testing {
        apply_testing_overlay(&mut cfg, testing_raw);
    }
    Ok(cfg)
}
```

- [ ] **Step 3: Add a test**

Append to `src/config.rs::tests`:

```rust
#[test]
fn testing_mode_overlays_apply_to_experiment() {
    let cfg = Config::from_yaml_file_with_mode(
        "config/merit_training.yaml",
        ConfigMode::Testing,
    ).expect("yaml");
    let exp = cfg.experiment.as_ref().unwrap();
    assert_eq!(exp.batch_size, 15);
    assert_eq!(exp.start_time, "1995/10/01");
    assert_eq!(exp.end_time, "2010/09/30");
    assert!(exp.rho.is_none());
}

#[test]
fn training_mode_does_not_apply_overlays() {
    let cfg = Config::from_yaml_file_with_mode(
        "config/merit_training.yaml",
        ConfigMode::Training,
    ).expect("yaml");
    let exp = cfg.experiment.as_ref().unwrap();
    assert_eq!(exp.batch_size, 64); // training default
    assert_eq!(exp.rho, Some(90));
}
```

- [ ] **Step 4: Build + test**

```
cargo test --lib config 2>&1 | tail -10
```

Expected: existing 2 + 2 new = 4 tests pass.

- [ ] **Step 5: Commit**

```
git add src/config.rs config/merit_training.yaml
git commit -m "$(cat <<'EOF'
Add ConfigMode + testing-overlay YAML section

config/merit_training.yaml gains a sibling testing: section. When the
binary is launched with --mode testing (Task 9), keys from that section
overlay onto experiment:. Absent keys inherit. batch_size semantically
shifts from GAUGES (training) to DAYS (testing) — flagged in YAML comment.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

### Task 9: `bin/eval.rs` CLI + clap dep

**Files:**
- Modify: `Cargo.toml`
- Create: `src/bin/eval.rs`

- [ ] **Step 1: Add clap to Cargo.toml**

In `[dependencies]`:

```toml
clap = { version = "4", features = ["derive"] }
```

- [ ] **Step 2: Create `src/bin/eval.rs`**

```rust
//! Test-phase entrypoint. Loads a config + MLP checkpoint (or runs with
//! frozen scalar params for dev), runs `evaluate()`, writes the
//! DDR-compatible predictions zarr, and logs a metrics summary.
//!
//! Usage:
//!   cargo run --release --bin eval -- \
//!       --config config/merit_training.yaml \
//!       --checkpoint output/saved_models/epoch_5 \
//!       --output output/model_test.zarr \
//!       --batch-size-days 15
//!
//! With --frozen, --checkpoint is optional (V4 dev path):
//!   cargo run --release --bin eval -- \
//!       --config config/merit_training.yaml \
//!       --frozen \
//!       --output output/v4_test.zarr
//!
//! NOT for distribution — the MLP architecture mirrors DDR's KAN at the I/O
//! contract level but the internal weights are not transferable from DDR
//! .pt files. Use a ddrs-trained .mpk checkpoint only.

use std::path::PathBuf;

use clap::Parser;
use burn::backend::NdArray;
use burn::tensor::backend::Backend;

use ddrs::config::{Config, ConfigMode};
use ddrs::data::dataset::MeritGagesDataset;
use ddrs::training::{
    evaluate, write_predictions_zarr, EvalParams, FrozenParams, ZarrAttrs,
};
use ddrs::training::checkpoint::load_mlp;
use ddrs::nn::mlp::{Mlp, MlpConfig};

#[derive(Parser, Debug)]
#[command(name = "eval", about = "ddrs test-phase evaluation")]
struct Cli {
    #[arg(long)]
    config: PathBuf,

    /// MLP checkpoint base path (no .mpk suffix). Required unless --frozen.
    #[arg(long)]
    checkpoint: Option<PathBuf>,

    /// Output zarr path.
    #[arg(long)]
    output: PathBuf,

    /// Days per chunk. Default 15 matches DDR's test config.
    #[arg(long, default_value_t = 15)]
    batch_size_days: usize,

    /// Use FROZEN_N/Q_SPATIAL/P_SPATIAL constants instead of an MLP.
    #[arg(long)]
    frozen: bool,
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let cli = Cli::parse();

    if !cli.frozen && cli.checkpoint.is_none() {
        eprintln!("--checkpoint is required unless --frozen is set");
        std::process::exit(2);
    }

    let cfg = Config::from_yaml_file_with_mode(&cli.config, ConfigMode::Testing)?;
    let dataset = MeritGagesDataset::open(&cfg)?;

    type I = NdArray<f32>;
    let device = <I as Backend>::Device::default();

    let output = if cli.frozen {
        // n_active is known only after first collate_window — accept the
        // (small) double-build cost by querying the dataset first.
        let axis = dataset.time_axis().clone();
        let probe_window = ddrs::data::TestWindow::new(&axis, 0, 1);
        let probe = dataset.collate_window(&probe_window)?;
        let frozen = FrozenParams::constant(probe.adjacency.n);
        evaluate::<I>(&cfg, &dataset, EvalParams::Frozen(&frozen),
                     &device, cli.batch_size_days)?
    } else {
        let mlp_section = cfg.mlp.as_ref().expect("mlp config required for MLP eval");
        let mlp_cfg = MlpConfig::new(
            mlp_section.input_var_names.clone(),
            mlp_section.learnable_parameters.clone(),
        )
            .with_hidden_size(mlp_section.hidden_size)
            .with_num_hidden_layers(mlp_section.num_hidden_layers);
        let mlp_template: Mlp<I> = mlp_cfg.init::<I>(&device);
        let mlp = load_mlp::<I>(cli.checkpoint.as_ref().unwrap(), mlp_template, &device)?;
        evaluate::<I>(&cfg, &dataset, EvalParams::Mlp(&mlp),
                     &device, cli.batch_size_days)?
    };

    // Write the zarr.
    let exp = cfg.experiment.as_ref().unwrap();
    let model_label = match &cli.checkpoint {
        Some(p) => p.display().to_string(),
        None => "frozen".to_string(),
    };
    let gages_csv_path = cfg.data_sources.as_ref().unwrap().gages.clone();
    write_predictions_zarr(
        &cli.output,
        &output,
        ZarrAttrs {
            start_time: &exp.start_time,
            end_time: &exp.end_time,
            version: env!("CARGO_PKG_VERSION"),
            evaluation_basins_file: &gages_csv_path,
            model_label: &model_label,
        },
    )?;

    // Metrics summary.
    let nse_clean: Vec<f32> = output.metrics.nse.iter()
        .copied().filter(|v| v.is_finite()).collect();
    let mean_nse = nse_clean.iter().sum::<f32>() / (nse_clean.len() as f32).max(1.0);
    println!("wrote {}", cli.output.display());
    println!("gauges with finite NSE: {} / {}", nse_clean.len(), output.metrics.nse.len());
    println!("mean NSE (finite only): {mean_nse:.4}");

    Ok(())
}
```

- [ ] **Step 3: Build the binary**

```
cargo build --release --bin eval 2>&1 | tail -10
```

Expected: clean compile.

- [ ] **Step 4: Smoke test (--frozen path, parse + first chunk only)**

This is a sanity check, not a full V4 run (which is expensive). Pass an
invalid output path on purpose so it fails fast after parse:

```
cargo run --release --bin eval -- \
    --config config/merit_training.yaml \
    --frozen \
    --output /tmp/ddrs_eval_smoke.zarr \
    --batch-size-days 1 2>&1 | head -30
```

Expected: starts running, may complete on the first 1-day chunk OR fail
later — whatever happens, the CLI parse succeeds and the loader works.
If the full run is fast enough to complete, even better.

If it OOMs at this stage (unlikely for 1-day), kill and report.

- [ ] **Step 5: Commit**

```
git add Cargo.toml src/bin/eval.rs
git commit -m "$(cat <<'EOF'
Add cargo run --bin eval CLI for test-phase evaluation

Loads merit_training.yaml in testing mode, opens MeritGagesDataset,
runs evaluate() with either FrozenParams (--frozen) or an MLP loaded
from a .mpk checkpoint, writes the DDR-compatible zarr, and logs a
metrics summary.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

### Task 10: `scripts/dump_ddr_test_predictions.py` + V4 reference fixture

**Files:**
- Create: `scripts/dump_ddr_test_predictions.py`
- Create: `fixtures/sp5/v4_ddr_test.zarr/` (force-add)

Mirror `scripts/dump_ddr_loss.py` (SP-4) for the test phase. Reuses the
same FROZEN constants and `physical_to_normalized` helper (including the
`+1e-6` log-space epsilon).

- [ ] **Step 1: Write the dump script**

```python
"""Compute DDR's reference test-period predictions for SP-5 V4 verification.

Usage (under DDR's uv venv):
    cd ~/projects/ddr && uv run python ~/projects/ddrs/scripts/dump_ddr_test_predictions.py

Writes ~/projects/ddrs/fixtures/sp5/v4_ddr_test.zarr/ with predictions +
observations + gage_ids + time, matching the layout that ddrs's
write_predictions_zarr produces.

Runs DDR's _test loop with batch_size = n_days_total (single batch, full
window) so the reference is unambiguous regardless of any DDR multi-batch
shape semantics.

The FROZEN_* constants MUST match
~/projects/ddrs/src/training/forward.rs::{FROZEN_N, FROZEN_Q_SPATIAL,
FROZEN_P_SPATIAL}.
"""

import json
from pathlib import Path

import numpy as np
import torch
import xarray as xr
import yaml
from omegaconf import OmegaConf

from ddr import dmc, streamflow
from ddr.io.functions import downsample
from ddr.validation import validate_config

FROZEN_N = 0.05
FROZEN_Q_SPATIAL = 0.5
FROZEN_P_SPATIAL = 21.0

OUTPUT_DIR = Path.home() / "projects/ddrs/fixtures/sp5"


def physical_to_normalized(physical, lo, hi, log_space):
    if log_space:
        log_lo = float(np.log(lo + 1e-6))
        log_hi = float(np.log(hi))
        return (float(np.log(physical)) - log_lo) / (log_hi - log_lo)
    return (physical - lo) / (hi - lo)


def load_cfg():
    # Same composition strategy as scripts/dump_ddr_loss.py: load DDR's own
    # merit_training_config.yaml, override the routing-physics keys from
    # ddrs's yaml, then apply test-mode overrides in-memory.
    ddr_yaml = Path.home() / "projects/ddr/config/merit_training_config.yaml"
    ddrs_yaml = Path.home() / "projects/ddrs/config/merit_training.yaml"
    with ddr_yaml.open() as f:
        ddr_raw = yaml.safe_load(f)
    with ddrs_yaml.open() as f:
        ddrs_raw = yaml.safe_load(f)
    cfg = OmegaConf.create(ddr_raw)
    # Override routing-physics + parameter ranges from ddrs config.
    cfg.params.parameter_ranges = ddrs_raw["params"]["parameter_ranges"]
    cfg.params.log_space_parameters = ddrs_raw["params"]["log_space_parameters"]
    cfg.params.attribute_minimums = ddrs_raw["params"]["attribute_minimums"]
    # Test-mode overrides — mirror ddrs's testing: section.
    test = ddrs_raw.get("testing", {})
    cfg.experiment.start_time = test.get("start_time", "1995/10/01")
    cfg.experiment.end_time = test.get("end_time", "2010/09/30")
    cfg.experiment.rho = None  # full window
    # batch_size handled below — we pass n_days_total directly.
    cfg.params.save_path = Path("/tmp/dump_ddr_test")
    cfg.device = "cpu"
    cfg.s3_region = "us-east-2"
    cfg.mode = "testing"
    return validate_config(cfg)


def main():
    cfg = load_cfg()
    dataset = cfg.geodataset.get_dataset_class(cfg=cfg)

    # Use the full window — DDR _test with batch_size = len(daily_time_range)
    # collapses to a single batch covering the whole test period.
    n_days_total = len(dataset.dates.daily_time_range)
    cfg.experiment.batch_size = n_days_total

    routing_dataclass = dataset.routing_dataclass
    n_active = routing_dataclass.spatial_attributes.shape[1]
    num_gauges = len(dataset.gage_ids)

    log_params = set(cfg.params.log_space_parameters)
    pr = cfg.params.parameter_ranges
    n_norm = physical_to_normalized(FROZEN_N, pr["n"][0], pr["n"][1], "n" in log_params)
    q_norm = physical_to_normalized(FROZEN_Q_SPATIAL, pr["q_spatial"][0], pr["q_spatial"][1], "q_spatial" in log_params)
    p_norm = physical_to_normalized(FROZEN_P_SPATIAL, pr["p_spatial"][0], pr["p_spatial"][1], "p_spatial" in log_params)

    device = torch.device(cfg.device)
    spatial_params = {
        "n":         torch.full((n_active,), float(n_norm), dtype=torch.float32),
        "q_spatial": torch.full((n_active,), float(q_norm), dtype=torch.float32),
        "p_spatial": torch.full((n_active,), float(p_norm), dtype=torch.float32),
    }

    flow = streamflow(cfg)
    routing_model = dmc(cfg=cfg, device="cpu")
    streamflow_predictions = flow(routing_dataclass=routing_dataclass, device="cpu", dtype=torch.float32)
    with torch.no_grad():
        dmc_output = routing_model(
            routing_dataclass=routing_dataclass,
            spatial_parameters=spatial_params,
            streamflow=streamflow_predictions,
            carry_state=False,
        )

    tau = cfg.params.tau
    daily_runoff = compute_daily_runoff(dmc_output["runoff"], tau).numpy()
    # daily_runoff shape: (n_gauges, n_days)
    # observations: (n_gauges, n_days_full), trim [1..-1]
    obs = dataset.routing_dataclass.observations.streamflow.values
    obs_trimmed = obs[:, 1:-1]
    assert obs_trimmed.shape[1] == daily_runoff.shape[1], \
        f"obs/pred shape mismatch: {obs_trimmed.shape} vs {daily_runoff.shape}"

    time_range = dataset.dates.daily_time_range[1:-1]
    gage_ids = np.array([str(g).zfill(8) for g in dataset.gage_ids])

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    out_zarr = OUTPUT_DIR / "v4_ddr_test.zarr"
    if out_zarr.exists():
        import shutil; shutil.rmtree(out_zarr)

    ds = xr.Dataset(
        data_vars={
            "predictions": (("gage_ids", "time"), daily_runoff.astype(np.float64),
                            {"units": "m3/s", "long_name": "Streamflow"}),
            "observations": (("gage_ids", "time"), obs_trimmed.astype(np.float64),
                             {"units": "m3/s", "long_name": "Observed Streamflow"}),
        },
        coords={
            "gage_ids": gage_ids,
            "time": time_range,
        },
        attrs={
            "description": "Predictions and obs for time period",
            "start time": "1995-10-01",
            "end time": "2010-09-30",
            "version": "sp5-v4-ref",
            "evaluation basins file": str(cfg.data_sources.gages),
            "model": "frozen",
        },
    )
    ds.to_zarr(out_zarr, mode="w")
    print(f"wrote {out_zarr}")
    print(f"predictions shape: {daily_runoff.shape}, mean={daily_runoff.mean():.4f}")


def compute_daily_runoff(hourly_predictions: torch.Tensor, tau: int) -> torch.Tensor:
    # Copy of ddr.scripts_utils.compute_daily_runoff for self-containment.
    sliced = hourly_predictions[:, (13 + tau) : (-11 + tau)]
    num_days = sliced.shape[1] // 24
    from ddr.io.functions import downsample
    return downsample(sliced, rho=num_days)


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Run the script under DDR's uv venv**

```
cd ~/projects/ddr && uv run python ~/projects/ddrs/scripts/dump_ddr_test_predictions.py 2>&1 | tail -10
```

Expected runtime: 10-30 minutes (full CONUS over 15 years, single batch).
If it OOMs (~65k active reaches × 131k hours × engine intermediates),
fall back to a 1-year reference window:
- Edit the script's date overrides to 1995-10-01 .. 1996-09-30
- Re-run
- Document in the V4 test (next task) that the comparison is over a 1-year
  window, not the full 15-year window.

If the run completes:

```
ls -la ~/projects/ddrs/fixtures/sp5/v4_ddr_test.zarr/
# Should see predictions/, observations/, gage_ids/, time/, .zattrs, .zmetadata, etc.
```

- [ ] **Step 3: Force-add fixture + commit**

`.gitignore` excludes `fixtures/`; use `git add -f`:

```
git add -f fixtures/sp5/v4_ddr_test.zarr
git add scripts/dump_ddr_test_predictions.py
git commit -m "$(cat <<'EOF'
Add DDR reference test-period predictions for SP-5 V4 verification

Runs DDR's _test pipeline with batch_size = n_days_total so the
reference is unambiguous (single batch, no multi-batch chunking
semantics). Frozen scalar n/q/p mirrored from
src/training/forward.rs. Output zarr matches the layout ddrs's
write_predictions_zarr produces; the V4 test (next task) compares
per-gauge daily predictions to f32 floor.

Force-added past the fixtures/ gitignore rule.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

### Task 11: V4 verification test (load-bearing)

**Files:**
- Modify: `tests/training_verification.rs`

- [ ] **Step 1: Append V4 test**

```rust
#[test]
fn v4_test_period_matches_ddr_for_frozen_constant_params() {
    use burn::backend::NdArray;
    use burn::tensor::backend::Backend;
    use std::path::Path;
    use ddrs::config::{Config, ConfigMode};
    use ddrs::data::dataset::MeritGagesDataset;
    use ddrs::training::{evaluate, EvalParams, FrozenParams};

    type I = NdArray<f32>;

    let fixture_path = "fixtures/sp5/v4_ddr_test.zarr";
    if !Path::new(fixture_path).exists() {
        eprintln!("skipping V4: {fixture_path} not present");
        return;
    }
    let cfg_path = "config/merit_training.yaml";
    if !Path::new(cfg_path).exists() { return; }
    let cfg = Config::from_yaml_file_with_mode(cfg_path, ConfigMode::Testing).expect("yaml");
    if !all_paths_exist(&cfg) { return; }

    let device = <I as Backend>::Device::default();
    let dataset = MeritGagesDataset::open(&cfg).expect("open dataset");

    // Probe to size FrozenParams.
    let axis = dataset.time_axis().clone();
    let n_days_total = axis.num_days;
    let probe = ddrs::data::TestWindow::new(&axis, 0, 1);
    let probe_batch = dataset.collate_window(&probe).expect("probe");
    let frozen = FrozenParams::constant(probe_batch.adjacency.n);

    // Single batch covering the whole window — mirrors the dump script.
    let output = evaluate::<I>(&cfg, &dataset, EvalParams::Frozen(&frozen),
                                &device, n_days_total).expect("evaluate");
    let pred_ddrs = &output.predictions_daily;

    // Read DDR reference. Use zarrs to load the predictions array.
    use zarrs::array::Array as ZarrArray;
    use zarrs::filesystem::FilesystemStore;
    use zarrs::storage::ReadableStorage;
    use std::sync::Arc;

    let storage: ReadableStorage =
        Arc::new(FilesystemStore::new(fixture_path).expect("open ref zarr"));
    let arr = ZarrArray::open(storage, "/predictions").expect("open /predictions");
    let dims = arr.shape();
    let subset = arr.subset_all();
    let pred_ddr_flat: Vec<f64> = arr.retrieve_array_subset::<Vec<f64>>(&subset).expect("read");
    let pred_ddr = ndarray::Array2::<f64>::from_shape_vec(
        (dims[0] as usize, dims[1] as usize), pred_ddr_flat
    ).expect("reshape");

    assert_eq!(
        pred_ddrs.shape(), pred_ddr.shape(),
        "V4 shape mismatch: ddrs={:?} ddr={:?}",
        pred_ddrs.shape(), pred_ddr.shape()
    );

    // Per-gauge max relative error.
    let mut worst_rel = 0.0_f32;
    let mut worst_g = 0usize;
    for g in 0..pred_ddrs.shape()[0] {
        for t in 0..pred_ddrs.shape()[1] {
            let p = pred_ddrs[(g, t)];
            let d = pred_ddr[(g, t)] as f32;
            let denom = d.abs().max(1e-6);
            let rel = (p - d).abs() / denom;
            if rel > worst_rel {
                worst_rel = rel;
                worst_g = g;
            }
        }
    }
    eprintln!("V4: worst rel error {worst_rel} at gauge index {worst_g}");

    // Tolerance 1e-4; relax to 1e-3 with a comment if reality demands.
    assert!(worst_rel < 1e-4,
        "V4 diverged: worst rel error {worst_rel} > 1e-4 at gauge index {worst_g}");
}
```

- [ ] **Step 2: Run V4**

```
cargo test --release --test training_verification v4_test_period 2>&1 | tail -40
```

Run in `--release` — full CONUS × 15 years is slow in debug.

Expected runtime: 10-30 min (single batch, large network).

- [ ] **Step 3: If V4 passes, commit**

```
git add tests/training_verification.rs
git commit -m "$(cat <<'EOF'
Add V4 SP-5 verification: full-test-period frozen-params loss equiv

V4 asserts ddrs's per-gauge daily test-period predictions agree with
DDR's reference to 1e-4 relative for the full 1995-2010 window with
frozen scalar n/q/p. Reference fixture dumped by
scripts/dump_ddr_test_predictions.py.

If V4 passes, the SP-1..SP-4 + SP-5 evaluate() pipeline is
integration-validated against DDR end-to-end.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

- [ ] **Step 4: If V4 fails**

Do NOT modify the engine, sparse solver, geometry, or earlier code to
chase the bug. Diagnose by:

1. Compare scalar summaries first:
   `pred_ddrs.sum()`, `pred_ddrs.mean()`, `pred_ddrs.max()` vs same on `pred_ddr`.
2. If summaries diverge by >1%, suspect a structural issue (network
   ordering, observation alignment, tau-trim semantics).
3. If summaries agree but per-gauge max rel is high, suspect f32
   accumulation drift — relax tolerance to 1e-3 with an inline comment.

Report the diagnostic numbers and stop. Do not proceed to V4b without
V4 green.

---

### Task 12: V4b (multi-batch self-consistency) + final regression sweep

**Files:**
- Modify: `tests/training_verification.rs`

- [ ] **Step 1: Append V4b test**

```rust
#[test]
fn v4b_multi_batch_matches_single_batch_with_carry_state() {
    use burn::backend::NdArray;
    use burn::tensor::backend::Backend;
    use std::path::Path;
    use ddrs::config::{Config, ConfigMode};
    use ddrs::data::dataset::MeritGagesDataset;
    use ddrs::training::{evaluate, EvalParams, FrozenParams};

    type I = NdArray<f32>;

    let cfg_path = "config/merit_training.yaml";
    if !Path::new(cfg_path).exists() { return; }
    let cfg = Config::from_yaml_file_with_mode(cfg_path, ConfigMode::Testing).expect("yaml");
    if !all_paths_exist(&cfg) { return; }

    let device = <I as Backend>::Device::default();
    let dataset = MeritGagesDataset::open(&cfg).expect("open");

    let axis = dataset.time_axis().clone();
    let n_days_total = axis.num_days;
    let probe = ddrs::data::TestWindow::new(&axis, 0, 1);
    let probe_batch = dataset.collate_window(&probe).expect("probe");
    let frozen = FrozenParams::constant(probe_batch.adjacency.n);

    // Single batch.
    let out_single = evaluate::<I>(&cfg, &dataset, EvalParams::Frozen(&frozen),
                                    &device, n_days_total).expect("single");
    // Multi-batch (15-day chunks).
    let out_multi = evaluate::<I>(&cfg, &dataset, EvalParams::Frozen(&frozen),
                                   &device, 15).expect("multi");

    assert_eq!(out_single.predictions_daily.shape(), out_multi.predictions_daily.shape());

    let mut worst_rel = 0.0_f32;
    let mut worst_at = (0usize, 0usize);
    for g in 0..out_single.predictions_daily.shape()[0] {
        for t in 0..out_single.predictions_daily.shape()[1] {
            let s = out_single.predictions_daily[(g, t)];
            let m = out_multi.predictions_daily[(g, t)];
            let denom = s.abs().max(1e-6);
            let rel = (s - m).abs() / denom;
            if rel > worst_rel {
                worst_rel = rel;
                worst_at = (g, t);
            }
        }
    }
    eprintln!("V4b: worst rel {worst_rel} at (g={}, t={})", worst_at.0, worst_at.1);
    assert!(worst_rel < 1e-4,
        "V4b diverged: ddrs single-batch != ddrs multi-batch at rel {worst_rel}");
}
```

- [ ] **Step 2: Run V4b**

```
cargo test --release --test training_verification v4b_multi_batch 2>&1 | tail -30
```

Expected runtime: ~2x V4 (runs the engine twice). If too slow, scope down
to a 1-year window inside the test (override cfg.experiment.start_time/
end_time before calling `MeritGagesDataset::open`).

- [ ] **Step 3: Final regression sweep**

```
cargo test --lib 2>&1 | grep "test result" | head -20
cargo test --test training_verification v1_loss_matches 2>&1 | tail -5
cargo test --test training_verification v3_train 2>&1 | tail -5
cargo run --release --example compare_ddr_sandbox 2>&1 | tail -5
cargo clippy --all-targets 2>&1 | grep -E "(eval|zarr_io|test_window|sp5)" | head -10
```

Expected:
- All lib tests pass.
- V1 still passes.
- V3 still passes.
- compare_ddr_sandbox reports ABSOLUTE MATCH.
- No new clippy warnings in SP-5 code.

- [ ] **Step 4: Commit (only if V4b + regression sweep pass)**

```
git add tests/training_verification.rs
git commit -m "$(cat <<'EOF'
Add V4b SP-5 verification: multi-batch matches single-batch with carry_state

V4b asserts ddrs's evaluate() produces identical per-gauge daily
predictions whether the full test period is run as one big batch or
as 15-day chunks with carry_state=i>0. Validates the chunked-time
plumbing in evaluate() and the engine's carry_state preservation in
setup_inputs. No second DDR dump required.

If V4b fails but V4 passes, the issue is in carry_state or chunked
scattering — not engine math.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Self-Review

### Spec coverage

| Spec section | Task |
|---|---|
| TestWindow for contiguous-hourly chunks (Concern #8) | 1 |
| MeritGagesDataset::collate_window + StaticNetworkCache | 2 |
| MuskingumCunge carry_state verification | 3 |
| carry_state plumbed through forward + forward_with_frozen_params | 4 |
| forward_eval for non-autodiff MLP inference | 5 |
| evaluate() + EvalParams + EvalOutput | 6 |
| write_predictions_zarr matching DDR layout | 7 |
| ConfigMode + testing-overlay loader | 8 |
| bin/eval.rs CLI + clap dep | 9 |
| scripts/dump_ddr_test_predictions.py + V4 reference fixture | 10 |
| V4 single-batch test | 11 |
| V4b multi-batch self-consistency | 12 |
| Final clippy + regression sweep | 12 |

All 9 spec phases + the TestWindow + carry_state verification additions
have corresponding tasks. Concern #8 (chunk-time semantics) addressed
explicitly in Task 1.

### Placeholder scan

- Task 2 Step 2's `build_static_network` body refers to "existing SP-3
  helpers" and ends with `...` for the verbatim block. The plan
  intentionally leaves the implementer to inspect `MeritGagesDataset::
  collate` and copy the prefix; alternative would be to inline the entire
  current collate body, which would balloon the plan. This is a known
  trade-off, flagged inline with the suggestion to extract a private
  helper if the existing code is clean.
- Task 7 Step 1's zarr inspection produces concrete findings that inform
  the codec choices in Step 2's `write_string_array` and `write_2d_f64_array`.
  If inspection reveals an unexpected zarr v2 vs v3 mismatch, the writer
  body adapts at the keyboard. Flagged in-place.
- Task 10's `compute_daily_runoff` helper inside the dump script duplicates
  DDR's own helper for self-containment — not a placeholder, a deliberate
  inline copy.

No "TBD" / "implement later" / "appropriate error handling" / "etc"
patterns found.

### Type/identifier consistency

- `TestWindow { start_day_idx, n_days, window_start }` — same shape
  everywhere it's used (Task 1, 2, 6, 9, 11, 12).
- `EvalParams<I> { Frozen(&FrozenParams), Mlp(&Mlp<I>) }` — same in
  Tasks 6, 9, 11, 12.
- `EvalOutput { predictions_daily, observations_daily, gage_ids,
  time_range_daily, metrics }` — same in Tasks 6, 7, 9, 11.
- `ZarrAttrs { start_time, end_time, version, evaluation_basins_file,
  model_label }` — same in Tasks 7 and 9.
- `forward_eval<I: Backend>(cfg, &RoutingTensors<I>, &Mlp<I>, device,
  carry_state)` — Tasks 5, 6, 9.
- `Config::from_yaml_file_with_mode(path, ConfigMode)` — Tasks 8, 9, 11, 12.
- `ConfigMode::{Training, Testing}` — Task 8 defines, Tasks 9/11/12 use.

### Internal consistency

- Task 6's observation-trim math (`predictions[.., 0..-1]` AND
  `obs[1..-1]`) is the SAFE CONSERVATIVE choice. Task 11 (V4 test) will
  surface any mismatch by comparing shapes; if DDR's actual semantics
  differ, fix in `evaluate()` then. Documented inline.
- Task 7's f64 output dtype matches DDR's xarray default (verified in
  Task 7 Step 1's inspection); Task 11 reads the f64 reference and casts
  to f32 for comparison.
- Task 9's `--frozen` mode lets the binary write a V4-style zarr without
  a checkpoint — useful for dev iteration, doesn't replace Task 11's
  cargo-test entry point.

---

## Execution choice

Plan complete and saved to `.claude/specs/2026-05-18-sp5-test-evaluation-plan.md`.

Two execution options:

1. **Subagent-Driven (recommended)** — fresh subagent per task with
   two-stage review. Same workflow as SP-1/2/3/4.
2. **Inline Execution** — batch with checkpoints.

Which approach?

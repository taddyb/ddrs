# DDR ↔ DDRS training-step parity — implementation plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Localize the residual `src/training/` divergence that causes DDRS's `n`
to saturate at median 0.030 vs DDR's 0.074 at identical config. Compare a single
mini-batch end-to-end at fixed init via fixture loading.

**Architecture:** Four layers (A audit → B forward pipeline → C loss + grads →
D Adam step), each producing a comparison artifact. Pause after Layer B sub-step 3
(MC forward) if it fails — that localizes the bug and obviates Layers C/D.

**Tech Stack:** Existing parity scaffold from PR #11/#12 (`KanHead::from_npz`,
`scripts/dump_kan_fixture.py`, `scripts/dump_ddr_trained_params.py`,
`fixtures` feature). New Python helpers under DDR's `uv` venv for the new
fixture-driver scripts. New Rust integration tests under `tests/`.

**Spec source of truth:**
`docs/superpowers/specs/2026-06-04-ddr-ddrs-training-step-parity-design.md`.

---

## Pre-flight verified

- Branch: `training-step-parity` already created from `master` (PR #12 merged).
- DDR's hot-start: `~/projects/ddr/src/ddr/routing/mmc.py:25` (`compute_hotstart_discharge`).
- DDRS's hot-start: `src/routing/utils.rs:88` (same name, ported).
- DDR's MC forward: `~/projects/ddr/src/ddr/routing/mmc.py:337` (`MuskingumCunge` class).
- DDRS's MC forward: `src/routing/mmc.rs::MuskingumCunge::forward`.
- DDR's training inner loop: `~/projects/ddr/scripts/train.py:50-130`.
- DDRS's training inner loop: `src/training/driver.rs:50-200`.
- Existing KAN fixture: `tests/fixtures/kan_head_init_seed42.npz` (PR #11).

---

## File structure

| Path | Status | Responsibility |
|------|--------|----------------|
| `docs/superpowers/specs/2026-06-04-ddr-ddrs-training-step-parity-design.md` | modify | Fill §4 Layer A audit table; append §5.1 verdict. |
| `scripts/dump_ddr_training_step.py` | create | Loads KAN fixture into DDR, runs one mini-batch through `ddr.routing.mmc.MuskingumCunge`, dumps every intermediate state to `tests/fixtures/training_step/`. |
| `tests/fixtures/training_step/manifest.json` | create | Records the fixture gauge (COMID + STAID), time window, expected file list, version. |
| `tests/fixtures/training_step/subgraph_<COMID>.npz` | create (committed) | DDR's loaded `(rows, cols, vals)` subgraph triplets for the fixture gauge. |
| `tests/fixtures/training_step/hotstart_<COMID>.npz` | create (committed) | DDR's hot-start discharge tensor `(n_reaches,)`. |
| `tests/fixtures/training_step/mc_forward_<COMID>.npz` | create (committed) | DDR's MC forward output `(n_reaches, rho_hours)`. |
| `tests/fixtures/training_step/daily_q_<COMID>.npz` | create (committed) | DDR's post-tau-trim daily Q `(n_reaches, n_days)`. |
| `tests/fixtures/training_step/loss_and_grads_<COMID>.npz` | create (committed) | DDR's L1 loss (scalar) + per-KAN-param gradients + post-clip grad norm. |
| `tests/fixtures/training_step/adam_step_<COMID>.npz` | create (committed) | DDR's post-Adam-step KAN params + per-param Adam moments. |
| `tests/training_step_layer_b.rs` | create | Layer B sub-steps 1-4 — load each fixture, run DDRS equivalent, assert per-sub-step tolerances. |
| `tests/training_step_layer_c.rs` | create | Layer C — loss + grads + post-clip norm. |
| `tests/training_step_layer_d.rs` | create | Layer D — Adam-step params + moment state. |

---

## Task 1: Layer A — inner-loop audit

**Spec ref:** §4 Layer A.

**Files:**
- Modify: `docs/superpowers/specs/2026-06-04-ddr-ddrs-training-step-parity-design.md` (the §4 Layer A table)

- [ ] **Step 1: Read DDR's training inner loop**

```bash
sed -n '50,140p' ~/projects/ddr/scripts/train.py
sed -n '20,90p'  ~/projects/ddr/src/ddr/routing/mmc.py
```

For each of the 5 (audit) rows in §4 Layer A, identify the DDR source lines.

- [ ] **Step 2: Read DDRS's training inner loop**

```bash
sed -n '50,200p' /home/tbindas/projects/ddrs/src/training/driver.rs
sed -n '1,140p' /home/tbindas/projects/ddrs/src/routing/mmc.rs
sed -n '80,140p' /home/tbindas/projects/ddrs/src/routing/utils.rs
```

For each (audit) row, identify the DDRS source lines.

- [ ] **Step 3: Fill in §4 Layer A table inline**

Replace each `(audit)` cell with one of: `✓`, `STAT only — <reason>`,
or `✗ (actual: X, want: Y)`. Cite source lines.

Specific things to check that are easy to miss:
- Subgraph adjacency: both load the same zarr; the question is whether the
  filter / sort / topological-order logic produces identical CSR triplets.
- Hot-start: DDR's `compute_hotstart_discharge` uses the streamflow window
  mean; verify DDRS's `compute_hotstart_discharge` (port at
  `src/routing/utils.rs:88`) uses the identical window + the same statistic.
- MC routing forward: per-timestep loop structure; check whether
  `_discharge_t` carry-over between timesteps is identical.
- grad_clip: DDR uses `torch.nn.utils.clip_grad_norm_(nn.parameters(),
  max_norm=1.0)`. DDRS uses `clip_grad_norm(grads, ..., 1.0)`. Verify the
  L2 norm is computed across the same set of parameters.
- Adam: burn 0.21 vs PyTorch 2.x differ on the order of operations in the
  bias-correction step. Read both bodies.

- [ ] **Step 4: Commit**

```bash
cd /home/tbindas/projects/ddrs
git add docs/superpowers/specs/2026-06-04-ddr-ddrs-training-step-parity-design.md
git commit -m "$(cat <<'EOF'
docs/spec: complete Layer A inner-loop audit

Audits the 5 untested fields in the training inner loop: subgraph
adjacency load, hot-start discharge, MC routing forward, grad_clip,
and Adam step. The KAN-head stages are inherited as ✓ from PR #11's
fixture-forward + fixture-backward tests; the loss / tau-trim stages
are inherited from the previous Layer 0 audit.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

- [ ] **Step 5:** If any ✗ surfaces — STOP and bring to user. As with
  the previous parity work, an audit ✗ is more likely the cause than
  anything Layers B-D would surface.

---

## Task 2: DDR-side fixture dumper

**Spec ref:** §4 Layer B prep + Layer C prep + Layer D prep.

**Files:**
- Create: `/home/tbindas/projects/ddrs/scripts/dump_ddr_training_step.py`
- Create: `/home/tbindas/projects/ddrs/tests/fixtures/training_step/manifest.json`
- Create: `/home/tbindas/projects/ddrs/tests/fixtures/training_step/*.npz` (6 files for one gauge)

- [ ] **Step 1: Pick the fixture gauge**

Inspect `~/projects/ddr/references/gage_info/gages_3000.csv` for the gauge
with the smallest upstream subgraph. Heuristic: smallest `DRAIN_SQKM`.

```bash
cd ~/projects/ddr && uv run python <<'PY'
import pandas as pd
df = pd.read_csv("references/gage_info/gages_3000.csv")
df = df.sort_values("DRAIN_SQKM").reset_index(drop=True)
print(df[["STAID", "STANAME", "DRAIN_SQKM"]].head(5).to_string(index=False))
PY
```

Pick the smallest. Record its STAID + COMID. (To get the COMID, you may
need to look in the gages_adjacency zarr — `xr.open_zarr(...)`.)

If the smallest is < 100 km² that's plenty.

- [ ] **Step 2: Write `scripts/dump_ddr_training_step.py`**

The script must:

1. Load DDR's KAN fixture via the existing `tests/fixtures/kan_head_init_seed42.npz`
   (created by PR #11 Task 7). Transpose Linear weights back from burn's
   `[in, out]` to PyTorch's `[out, in]`. Load into a `ddr.nn.kan.kan` instance.
2. Open `merit_gages_conus_adjacency.zarr`, select the fixture gauge's
   subgraph. Dump CSR triplets to `subgraph_<COMID>.npz`.
3. Open the streamflow icechunk store, slice the rho window starting at
   `1990/01/01`. Compute hot-start via DDR's `compute_hotstart_discharge`.
   Dump to `hotstart_<COMID>.npz`.
4. Run `dmc.routing.muskingum_cunge` for the full rho hourly window with the
   fixture KAN params. Dump the `(n_reaches, rho_hours)` Q tensor to
   `mc_forward_<COMID>.npz`.
5. Apply DDR's tau-trim `[13:-11+tau]` and daily downsample. Dump to
   `daily_q_<COMID>.npz`.
6. Slice obs from the icechunk store over the same window. Apply DDR's
   per-gauge NaN-mask. Compute L1 loss. Call `loss.backward()`. Dump the
   scalar loss + every KAN param's gradient + the post-`clip_grad_norm_`
   global norm to `loss_and_grads_<COMID>.npz`.
7. Initialize a fresh `torch.optim.Adam(nn.parameters(), lr=0.001,
   betas=(0.9, 0.999), eps=1e-8)`, call `.step()`. Dump the updated KAN
   params + the per-param `exp_avg` / `exp_avg_sq` Adam state to
   `adam_step_<COMID>.npz`.

8. Write `manifest.json` recording:
   - fixture KAN seed (= 42)
   - fixture gauge COMID + STAID + DRAIN_SQKM
   - n_reaches in subgraph
   - time window (start, end, rho_hours)
   - list of fixture files + their byte sizes
   - DDR git commit + version

The script will be ~150 lines. Use `~/projects/ddrs/scripts/dump_kan_fixture.py`
as a stylistic template — same argparse-free single-purpose pattern, same
`Path.expanduser()` style for repo paths.

- [ ] **Step 3: Run the script**

```bash
cd ~/projects/ddr && uv run python ~/projects/ddrs/scripts/dump_ddr_training_step.py
```

Expected: writes 6 `.npz` files + manifest.json under
`~/projects/ddrs/tests/fixtures/training_step/`. Prints fixture summary.

- [ ] **Step 4: Sanity-check the fixtures**

```bash
cd ~/projects/ddrs/ddrs-py && uv run --extra plots python <<'PY'
import json, numpy as np
from pathlib import Path
F = Path("/home/tbindas/projects/ddrs/tests/fixtures/training_step")
manifest = json.loads((F / "manifest.json").read_text())
print(json.dumps(manifest, indent=2))
for name in manifest["files"]:
    npz = np.load(F / name)
    print(f"\n{name}:")
    for k in npz.files:
        print(f"  {k}: shape={npz[k].shape} dtype={npz[k].dtype}")
PY
```

Confirm all shapes look reasonable (e.g. subgraph triplets have matching
`rows.shape == cols.shape == vals.shape`, hot-start is `(n_reaches,)`,
mc_forward is `(n_reaches, rho_hours)`).

- [ ] **Step 5: Commit**

```bash
cd /home/tbindas/projects/ddrs
git add scripts/dump_ddr_training_step.py tests/fixtures/training_step/
git commit -m "$(cat <<'EOF'
scripts: dump DDR's per-stage training-step state for parity tests

Loads the existing tests/fixtures/kan_head_init_seed42.npz into a
ddr.nn.kan.kan instance (transposing Linear weights back from burn's
[in, out] to PyTorch's [out, in]), runs one mini-batch through DDR's
full training pipeline on the smallest CONUS gauge, and dumps the
state at every stage to .npz files under tests/fixtures/training_step/.

Six artifacts per gauge: subgraph, hot-start, mc_forward, daily_q,
loss_and_grads, adam_step. Plus a manifest.json with provenance.

Consumed by training_step_layer_{b,c,d}.rs in the next commits.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 3: Layer B Rust test — forward-pipeline parity

**Spec ref:** §4 Layer B.

**Files:**
- Create: `/home/tbindas/projects/ddrs/tests/training_step_layer_b.rs`

The test file has 4 sub-tests, one per Layer B sub-step. Each loads its
DDR fixture, recomputes the DDRS equivalent, asserts a tolerance.

- [ ] **Step 1: Write the test scaffolding**

```rust
//! Layer B of the training-step parity plan: forward-pipeline parity at a
//! fixed mini-batch. Fixture artifacts are dumped by
//! scripts/dump_ddr_training_step.py.

#![cfg(feature = "fixtures")]

use std::path::Path;

use burn::backend::NdArray;
use ndarray::Array2;
use ndarray_npy::NpzReader;

type B = NdArray<f32>;

const FIXTURE_DIR: &str = "tests/fixtures/training_step";

fn fixture(name: &str) -> NpzReader<std::fs::File> {
    let path = Path::new(FIXTURE_DIR).join(name);
    NpzReader::new(std::fs::File::open(&path).unwrap_or_else(|e| {
        panic!("missing fixture {path:?}: {e}. Re-run scripts/dump_ddr_training_step.py")
    })).unwrap()
}

fn max_abs_diff(got: &[f32], want: &[f32]) -> f32 {
    assert_eq!(got.len(), want.len(), "shape mismatch");
    got.iter().zip(want).map(|(a, b)| (a - b).abs()).fold(0.0_f32, f32::max)
}
```

- [ ] **Step 2: Sub-step 1 — subgraph adjacency parity**

Add to the file:

```rust
/// Layer B sub-step 1: DDRS's loaded subgraph CSR triplets must byte-match
/// DDR's. The data source (merit_gages_conus_adjacency.zarr) is shared,
/// so this is a check on the filter / sort / topological-order code path.
#[test]
fn layer_b_step1_subgraph_adjacency_matches_ddr() {
    let mut npz = fixture(&format!("subgraph_<FIXTURE_COMID>.npz"));
    // ... where <FIXTURE_COMID> is read from manifest.json at test-build time
    //
    // (For now leave a placeholder; a build.rs or const can supply the COMID
    // after Task 2 lands. The test will fail loudly until then.)
    //
    // Once you have COMID:
    let ddr_rows: ndarray::Array1<i32> = npz.by_name("rows").unwrap()
        .into_dimensionality::<ndarray::Ix1>().unwrap();
    let ddr_cols: ndarray::Array1<i32> = npz.by_name("cols").unwrap()
        .into_dimensionality::<ndarray::Ix1>().unwrap();
    let ddr_vals: ndarray::Array1<f32> = npz.by_name("vals").unwrap()
        .into_dimensionality::<ndarray::Ix1>().unwrap();

    // Build the DDRS equivalent via the same loader path used in training:
    let store = ddrs::data::store::zarr::GagesAdjacencyStore::open(
        Path::new("/home/tbindas/projects/ddr/data/merit_gages_conus_adjacency.zarr"),
    ).unwrap();
    let subgraph = store.load_subgraph_for_gauge(/* COMID from manifest */).unwrap();
    let (ddrs_rows, ddrs_cols, ddrs_vals) = subgraph.to_triplets();

    assert_eq!(ddr_rows.as_slice().unwrap(), ddrs_rows.as_slice(), "rows differ");
    assert_eq!(ddr_cols.as_slice().unwrap(), ddrs_cols.as_slice(), "cols differ");
    let diff = max_abs_diff(ddrs_vals.as_slice(), ddr_vals.as_slice().unwrap());
    assert_eq!(diff, 0.0, "vals differ; max abs diff {diff}");
}
```

(Adjust `GagesAdjacencyStore::open` and `load_subgraph_for_gauge` to the
actual API names in `src/data/store/`. Look them up in
`src/data/store/zarr.rs`.)

If the `GagesAdjacencyStore` doesn't expose `load_subgraph_for_gauge`,
read the existing training driver code path that does this (probably
in `src/data/dataset.rs` collate path) and reproduce it inline in the
test.

- [ ] **Step 3: Sub-step 2 — hot-start discharge parity**

Add:

```rust
#[test]
fn layer_b_step2_hotstart_matches_ddr() {
    let mut npz = fixture("hotstart_<COMID>.npz");
    let ddr_hotstart: ndarray::Array1<f32> = npz.by_name("hotstart").unwrap()
        .into_dimensionality::<ndarray::Ix1>().unwrap();

    // Build the DDRS hot-start via the same path:
    let device = Default::default();
    let streamflow = /* load streamflow chunk for the fixture window — read
                        the same icechunk store DDR did. The fixture's
                        manifest.json should record the start/end timestamps */;
    let ddrs_hotstart = ddrs::routing::compute_hotstart_discharge::<B>(
        &streamflow, &device,
    );
    let ddrs_vec: Vec<f32> = ddrs_hotstart.into_data().to_vec().unwrap();
    let diff = max_abs_diff(&ddrs_vec, ddr_hotstart.as_slice().unwrap());
    assert!(diff <= 1e-6, "hotstart max abs diff {diff} > 1e-6");
}
```

The streamflow-loading is currently embedded in
`src/data/dataset.rs`; the test may need to factor that out or duplicate
the relevant lines. If duplicating, cite the source line: `// Mirrors
src/data/dataset.rs:XXX`.

- [ ] **Step 4: Sub-step 3 — MC forward parity** (the load-bearing one)

```rust
#[test]
fn layer_b_step3_mc_forward_matches_ddr() {
    let mut npz = fixture("mc_forward_<COMID>.npz");
    let ddr_q: ndarray::Array2<f32> = npz.by_name("Q").unwrap()
        .into_dimensionality::<ndarray::Ix2>().unwrap();
    let (n_reaches, rho_hours) = (ddr_q.shape()[0], ddr_q.shape()[1]);

    // Build the DDRS forward:
    // 1. Load the fixture KAN head via KanHead::from_npz (already in
    //    PR #11). Denormalize n/q_spatial/p_spatial using the same
    //    log_space + parameter_ranges as the fixture used.
    let cfg = /* the fixture-config — same hyperparams as
                 config/merit_training.yaml */;
    let head: ddrs::nn::KanHead<B> = ddrs::nn::KanHead::<B>::from_npz(
        Path::new("tests/fixtures/kan_head_init_seed42.npz"),
        &Default::default(),
        &cfg,
    ).unwrap();

    // 2. Load fixture gauge attributes, run head.forward, denormalize
    //    to per-reach (n, q_spatial, p_spatial).
    // 3. Set up MuskingumCunge, call forward.
    // 4. Compare to ddr_q.

    let ddrs_q: Vec<f32> = /* ... see src/training/driver.rs:70-90 for the
                              setup_inputs + forward call sequence */;
    let diff = max_abs_diff(&ddrs_q, ddr_q.as_slice().unwrap());
    assert!(diff <= 1e-5, "MC forward max abs diff {diff} > 1e-5 — \
        the bug is in the routing solver or geometry");
}
```

The actual call-sequence reproduction is the substantive work. Refer to
`src/training/driver.rs:50-110` for how training calls the head + MC pipeline.

- [ ] **Step 5: Sub-step 4 — tau-trim + daily downsample**

```rust
#[test]
fn layer_b_step4_daily_q_matches_ddr() {
    let mut npz = fixture("daily_q_<COMID>.npz");
    let ddr_daily: ndarray::Array2<f32> = npz.by_name("daily_q").unwrap()
        .into_dimensionality::<ndarray::Ix2>().unwrap();

    // The previous test already produced ddrs_q (hourly). Apply
    // ddrs::training::tau_trim_and_downsample to it.
    let mc_npz_q = /* re-derive from step 3 OR load mc_forward_<COMID>.npz */;
    let cfg_tau: u32 = 3;
    let daily = ddrs::training::tau_trim_and_downsample(mc_npz_q, cfg_tau);
    let daily_vec: Vec<f32> = daily.into_data().to_vec().unwrap();
    let diff = max_abs_diff(&daily_vec, ddr_daily.as_slice().unwrap());
    assert!(diff <= 1e-5, "daily Q max abs diff {diff} > 1e-5");
}
```

Note: the tau-slicing divergence (spec C7) is STAT-only — DDR uses
`[13 : -11+tau]`, DDRS uses `[13+tau : -11+tau]`. Both lose 3 hours but at
opposite ends. Per the spec, they both produce **89 days** of post-warmup
Q; the diff should be small but **not exactly zero** because the 3-hour
shift includes different forcing.

If this test fails, document the actual shift magnitude — it might be
larger than expected.

- [ ] **Step 6: Run the test crate**

```bash
cd /home/tbindas/projects/ddrs
cargo test --features fixtures --test training_step_layer_b -- --nocapture 2>&1 | tail -25
```

Expected: 4/4 pass.

If sub-step 3 fails — **STOP**. The bug is localized to the MC routing
forward; Layers C and D become irrelevant. Open a new spec to fix the
sparse solver / geometry / hot-start carry-over. Report the diff
magnitude in your report.

- [ ] **Step 7: Commit**

```bash
git add tests/training_step_layer_b.rs
git commit -m "$(cat <<'EOF'
test: Layer B — forward-pipeline parity per fixed mini-batch

Four sub-tests: subgraph adjacency triplets, hot-start discharge,
MC routing forward, post-tau-trim daily Q. Reads fixtures dumped by
scripts/dump_ddr_training_step.py.

Tolerances per spec A5: subgraph + hot-start byte-equal, MC forward
≤ 1e-5, daily Q ≤ 1e-5.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 4: Layer C Rust test — loss + gradient parity

**Spec ref:** §4 Layer C.

**Only run this task if Task 3 passes.** If sub-step 3 of Task 3 fails,
Layer C is irrelevant — the divergence is in the forward, not the loss.

**Files:**
- Create: `/home/tbindas/projects/ddrs/tests/training_step_layer_c.rs`

- [ ] **Step 1: Write the test**

```rust
//! Layer C of the training-step parity plan: loss + gradient parity at the
//! fixed mini-batch. Requires Layer B to have passed.

#![cfg(feature = "fixtures")]

use std::path::Path;

use burn::backend::{Autodiff, NdArray};
use burn::module::AutodiffModule;
use ndarray_npy::NpzReader;

type B = Autodiff<NdArray<f32>>;

const FIXTURE_DIR: &str = "tests/fixtures/training_step";

fn fixture(name: &str) -> NpzReader<std::fs::File> {
    let path = Path::new(FIXTURE_DIR).join(name);
    NpzReader::new(std::fs::File::open(&path).unwrap()).unwrap()
}

fn max_abs_diff(got: &[f32], want: &[f32]) -> f32 {
    got.iter().zip(want).map(|(a, b)| (a - b).abs()).fold(0.0_f32, f32::max)
}

#[test]
fn layer_c_loss_matches_ddr() {
    let mut npz = fixture("loss_and_grads_<COMID>.npz");
    let ddr_loss: f32 = npz.by_name::<f32, _>("loss").unwrap().into_scalar();

    // Replay the full Layer B pipeline under Autodiff<NdArray>, then
    // compute the same L1 loss DDRS uses (post-NaN-filter, post-warmup).
    let ddrs_loss: f32 = /* ... see src/training/driver.rs:115-160 */;

    assert!(
        (ddrs_loss - ddr_loss).abs() <= 1e-7,
        "loss diff {} > 1e-7 (got {ddrs_loss}, want {ddr_loss})",
        (ddrs_loss - ddr_loss).abs(),
    );
}

#[test]
fn layer_c_gradients_match_ddr() {
    let mut npz = fixture("loss_and_grads_<COMID>.npz");

    // ... full pipeline under Autodiff, then loss.backward()
    let head: ddrs::nn::KanHead<B> = /* from_npz */;
    let loss = /* compute */;
    let grads = loss.backward();

    // Compare every KAN param's grad to its DDR counterpart.
    // DDR uses PyTorch [out, in] for Linear weights; DDRS uses burn [in, out].
    // Transpose DDR's grad_input_weight + grad_output_weight before comparing.
    let pairs: Vec<(&str, Vec<f32>)> = vec![
        ("grad_input_weight",  /* transposed read from npz */),
        ("grad_input_bias",    /* read directly */),
        // ... etc
    ];
    for (key, want) in &pairs {
        let got = /* head.<param>.val().grad(&grads).unwrap().to_vec() */;
        let diff = max_abs_diff(&got, want);
        assert!(diff <= 1e-5, "{key}: max abs diff {diff}");
    }
}

#[test]
fn layer_c_post_clip_grad_norm_matches_ddr() {
    let mut npz = fixture("loss_and_grads_<COMID>.npz");
    let ddr_norm: f32 = npz.by_name::<f32, _>("post_clip_grad_norm").unwrap()
        .into_scalar();

    // Reproduce grads + clip + compute global L2 norm.
    let head = /* from_npz */;
    let loss = /* compute */;
    let grads = loss.backward();
    let clipped = ddrs::training::clip_grad_norm(grads, &head, 1.0);
    let norm = /* compute global L2 norm of clipped grads */;
    assert!((norm - ddr_norm).abs() <= 1e-6,
        "post-clip norm diff > 1e-6 (got {norm}, want {ddr_norm})");
}
```

- [ ] **Step 2: Run the test**

```bash
cargo test --features fixtures --test training_step_layer_c -- --nocapture 2>&1 | tail -15
```

Expected: 3/3 pass.

If any fails, report which one — that localizes the bug to the loss
computation, the autograd path, or the grad-clip implementation.

- [ ] **Step 3: Commit**

```bash
git add tests/training_step_layer_c.rs
git commit -m "$(cat <<'EOF'
test: Layer C — loss + gradient parity per fixed mini-batch

Three sub-tests: L1 loss scalar (≤ 1e-7), per-KAN-param gradients
(≤ 1e-5), post-clip_grad_norm global L2 norm (≤ 1e-6). Reads the
loss_and_grads fixture dumped by scripts/dump_ddr_training_step.py.

Linear-weight gradients are transposed from DDR's [out, in] to burn's
[in, out] before comparison (same fix as PR #11 Task 10).

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 5: Layer D Rust test — Adam-step parity

**Spec ref:** §4 Layer D.

**Only run if Task 4 passes.**

**Files:**
- Create: `/home/tbindas/projects/ddrs/tests/training_step_layer_d.rs`

- [ ] **Step 1: Write the test**

```rust
//! Layer D of the training-step parity plan: post-Adam-step parameter +
//! moment-state parity. Requires Layer C to have passed.

#![cfg(feature = "fixtures")]

use std::path::Path;

use burn::backend::{Autodiff, NdArray};
use burn::optim::{AdamConfig, Optimizer, GradientsParams};
use ndarray_npy::NpzReader;

type B = Autodiff<NdArray<f32>>;

const FIXTURE_DIR: &str = "tests/fixtures/training_step";

fn fixture(name: &str) -> NpzReader<std::fs::File> { /* same as B */ }
fn max_abs_diff(got: &[f32], want: &[f32]) -> f32 { /* same as B */ }

#[test]
fn layer_d_post_adam_step_params_match_ddr() {
    let mut npz = fixture("adam_step_<COMID>.npz");

    // Build head, compute loss + grads + clip (replays Layer C).
    let head = /* from_npz */;
    let loss = /* compute */;
    let grads = loss.backward();
    let clipped = ddrs::training::clip_grad_norm(grads, &head, 1.0);
    let gp = GradientsParams::from_grads(clipped, &head);

    // Initialize Adam at the same config DDR uses
    let mut optimizer = AdamConfig::new()
        .with_beta_1(0.9).with_beta_2(0.999).with_epsilon(1e-8)
        .init::<B, _>();
    let head_stepped = optimizer.step(0.001, head.clone(), gp);

    // Compare every param to DDR's stepped value.
    let pairs: Vec<(&str, Vec<f32>)> = vec![
        ("input_weight",  /* transposed read */),
        ("input_bias",    /* direct read */),
        ("output_weight", /* transposed read */),
        ("output_bias",   /* direct read */),
        // KanLayer trainables...
    ];
    for (key, want) in &pairs {
        let got = /* head_stepped.<param>.val().to_vec() */;
        let diff = max_abs_diff(&got, want);
        assert!(diff <= 1e-5, "{key}: max abs diff {diff}");
    }
}

#[test]
fn layer_d_adam_moments_match_ddr() {
    let mut npz = fixture("adam_step_<COMID>.npz");

    // Same setup as above; this time read the adam state out of the
    // optimizer after step(). Burn's AdamConfig exposes the state via
    // module storage — investigate the API.

    // For each param, compare:
    //   - exp_avg (first moment)
    //   - exp_avg_sq (second moment)
    // ≤ 1e-5

    // Burn 0.21's Adam state extraction may require a small helper or
    // module introspection. If burn doesn't expose this cleanly, fall
    // back to comparing only the stepped params (Step 1) and note the
    // limitation in the test docstring.
    todo!("burn 0.21 Adam state extraction — see optimizer.rs for the API")
}
```

The `layer_d_adam_moments_match_ddr` test may be hard to write if burn
doesn't expose Adam's internal state cleanly. If so, mark it `#[ignore]`
and add a comment pointing at the limitation. The stepped-params test is
the load-bearing one.

- [ ] **Step 2: Run the test**

```bash
cargo test --features fixtures --test training_step_layer_d -- --nocapture 2>&1 | tail -15
```

Expected: 1-2/2 pass. The moment-state test may be skipped if burn 0.21
doesn't expose Adam internals.

- [ ] **Step 3: Commit**

```bash
git add tests/training_step_layer_d.rs
git commit -m "$(cat <<'EOF'
test: Layer D — post-Adam-step parameter parity per fixed mini-batch

Compares DDRS's post-Adam-step KAN params to DDR's. Tolerances ≤ 1e-5
per spec A5. Adam moment-state test may be skipped (#[ignore]) if
burn 0.21 doesn't expose first/second moment estimates cleanly.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 6: Record §5.1 verdict

**Spec ref:** §5 outcome table → §5.1 (new).

**Files:**
- Modify: `docs/superpowers/specs/2026-06-04-ddr-ddrs-training-step-parity-design.md`

- [ ] **Step 1: Tally the layer results**

Pull the result counts from Tasks 3, 4, 5. Specifically:
- Layer B: 4 sub-tests, X passed.
- Layer C: 3 sub-tests, Y passed.
- Layer D: 1-2 sub-tests, Z passed.

The **first sub-test that failed** is the localization. If everything
passed, the verdict is §5 row 1 (all pass → batch-shuffle is the only
divergence).

- [ ] **Step 2: Append §5.1 to the spec**

Mirror the structure of the previous trained-parity spec's §5.1:

```markdown
---

## §5.1 Empirical verdict (Task 6 of the plan)

**Fixture gauge:** STAID=XXXXXXX, COMID=XXXXXXX, DRAIN_SQKM=XXX.
**Fixture KAN init:** `tests/fixtures/kan_head_init_seed42.npz` (PR #11).
**Time window:** 1990/01/01 + 90 days.

**Layer A audit result:** X rows ✓, Y rows STAT-only, Z rows ✗.

**Layer B-D results:**

| Layer | Sub-test | Tolerance | Actual max-abs-diff | Verdict |
|-------|----------|-----------|---------------------|---------|
| B1 | subgraph adjacency | byte-equal | 0 / divergent | ✓ / ✗ |
| B2 | hot-start | 1e-6 | … | … |
| B3 | MC forward | 1e-5 | … | … |
| B4 | daily Q | 1e-5 | … | … |
| C1 | L1 loss | 1e-7 | … | … |
| C2 | gradients | 1e-5 | … | … |
| C3 | post-clip norm | 1e-6 | … | … |
| D1 | Adam-stepped params | 1e-5 | … | … |
| D2 | Adam moments | 1e-5 | … / N/A | … |

**Outcome (from §5 table):** "<one of the three rows; quote it>"

**Next step:** <inherits from the §5 table's row for the matched outcome>
```

- [ ] **Step 3: Commit**

```bash
git add docs/superpowers/specs/2026-06-04-ddr-ddrs-training-step-parity-design.md
git commit -m "$(cat <<'EOF'
docs/spec: record Layer B-D empirical verdict

Tallies the per-sub-test results from Tasks 3-5 and names the
spec §5 outcome row + the next-step.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Spec coverage map

| Spec section | Plan task(s) |
|--------------|-------------|
| §2 C1-C7 (concerns) | Each is addressed by the relevant Task — C1 via Task 2's PyTorch hooks design, C2 via Task 3 sub-step 3's CPU-first approach, C3 via Task 2's documented inline replay, C4 via Task 3 sub-step 2, C5 via Task 3 sub-step 1, C6 via Task 5, C7 acknowledged in Task 3 sub-step 4. |
| §3 A1-A6 (assumptions) | A1 + A4 baked into the per-mini-batch design; A2-A3 already established; A5 is the per-test tolerance; A6 is the gauge-selection heuristic in Task 2 Step 1. |
| §4 Layer A | Task 1 |
| §4 Layer B (4 sub-steps) | Tasks 2 + 3 |
| §4 Layer C (3 sub-steps) | Tasks 2 + 4 |
| §4 Layer D (2 sub-steps) | Tasks 2 + 5 |
| §5 outcome | Task 6 |
| §6 implementation order | Plan task order, with Task 4 gated on Task 3 passing and Task 5 gated on Task 4 passing |

---

Plan complete and saved to
`docs/superpowers/plans/2026-06-04-ddr-ddrs-training-step-parity.md`.

**Two execution options:**

**1. Subagent-Driven (recommended)** — fresh subagent per task with spec + code-quality review between each.

**2. Inline Execution** — `superpowers:executing-plans` with batch checkpoints.

Which?

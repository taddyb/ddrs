# DDRS area-mode downsample fix — implementation plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace DDRS's strict reshape+mean daily downsample with
`F.interpolate(mode="area")` semantics so the per-step gradient noise that
drives the n-saturation goes away. After this lands, retrain + re-run PR #12's
trained-distribution parity notebook to formally close the saturation
investigation.

**Architecture:** Add a private `area_pool_weights` helper in
`src/training/loss.rs` that builds a constant `(M, L)` weight matrix
implementing area-mode adaptive average pooling. Rewrite
`tau_trim_and_downsample` to use it via `matmul`. burn-tensor's autograd
handles the backward pass for free — no custom `Backward` impl needed.

**Tech Stack:** Pure burn 0.21 tensor ops + 4 inline `#[cfg(test)]` tests.
Tightens existing PR #13 integration tests (Layer B4 / C2 / D1) at the new
1e-5 tolerance. Retrains DDRS once + re-runs PR #12's parity notebook.

**Spec source of truth:**
`docs/superpowers/specs/2026-06-04-ddrs-area-mode-downsample-fix-design.md`.
§4.2 contains the full Rust code; this plan is mostly about sequencing
+ verification.

**Branch:** Stay on `training-step-parity` (PR #13). Do NOT create a new
branch — the diagnosis + fix-design + implementation belong together.

---

## Pre-flight verified

- Current `src/training/loss.rs` is 120 lines.
- `tau_trim_and_downsample` at lines 16-36 currently does
  `slice → reshape(g, t_days, 24) → mean_dim(2) → squeeze::<2>()`.
- The asymmetric squeeze bug (PR #13 Layer B finding) collapses both size-1
  dims for `n_gauges=1`. The rewrite implicitly fixes this — matmul preserves
  the leading axis with no squeeze needed.
- `src/training/mod.rs:28` re-exports `tau_trim_and_downsample` — signature
  stays the same so this stays a drop-in.
- PR #13's integration tests use these old tolerances we will tighten:
  - `tests/training_step_layer_b.rs` Layer B4 — currently `0.1 m³/s`.
  - `tests/training_step_layer_c.rs` Layer C2 — currently `1e-3`.
  - `tests/training_step_layer_d.rs` Layer D1 — currently `2e-3`.
- DDR's reference: `~/projects/ddr/src/ddr/io/functions.py:22`
  `F.interpolate(data.unsqueeze(1), size=(rho,), mode="area").squeeze(1)`.

---

## File structure

| Path | Status | Responsibility |
|------|--------|----------------|
| `src/training/loss.rs` | modify | Replace `tau_trim_and_downsample` body; add private `area_pool_weights` helper; add 4 unit tests. |
| `tests/training_step_layer_b.rs` | modify | Tighten Layer B4 tolerance `0.1` → `1e-5`. |
| `tests/training_step_layer_c.rs` | modify | Tighten Layer C2 tolerance `1e-3` → `1e-5`. |
| `tests/training_step_layer_d.rs` | modify | Tighten Layer D1 tolerance `2e-3` → `1e-5`. |
| `docs/superpowers/specs/2026-06-04-ddrs-area-mode-downsample-fix-design.md` | modify | Append §5.1 (post-retrain verdict). |

---

## Task 1: Add `area_pool_weights` helper + unit tests

**Spec ref:** §4.2 + §4.3.

**Files:**
- Modify: `src/training/loss.rs`

- [ ] **Step 1: Read the current state of the file**

```bash
sed -n '1,40p' /home/tbindas/projects/ddrs/src/training/loss.rs
sed -n '94,120p' /home/tbindas/projects/ddrs/src/training/loss.rs
```

Confirm the current `tau_trim_and_downsample` body and the existing `#[cfg(test)] mod tests` block. The new helper + tests go in this file without disturbing the existing `filter_nan_gauges` / `l1_loss_post_warmup` / their tests.

- [ ] **Step 2: Append the helper and tests**

Add `area_pool_weights` immediately after `tau_trim_and_downsample` (after the closing brace of line 36, before the `pub struct FilteredPair` at line 38):

```rust
/// Construct the area-mode pooling weight matrix `W ∈ R^{M × L}` such that
/// `W[i, j] = overlap(input_cell_j, output_bin_i) / s` where `s = L / M`.
///
/// Each row sums to 1. Mirrors `torch.nn.functional.interpolate(mode="area")`
/// for the 1D case (DDR uses this at `ddr/io/functions.py:22`). The result
/// is a constant matrix that depends only on shape — compute once per
/// (L, M) pair and reuse.
///
/// Sparsity: each row has at most `ceil(L/M) + 1` nonzeros for `s > 1`.
fn area_pool_weights<B: Backend>(
    l: usize,
    m: usize,
    device: &B::Device,
) -> Tensor<B, 2> {
    assert!(l > 0 && m > 0 && l >= m, "need L >= M > 0; got L={l}, M={m}");
    let s = l as f32 / m as f32;
    let mut data: Vec<f32> = vec![0.0; m * l];

    for i in 0..m {
        let left = (i as f32) * s;
        let right = ((i + 1) as f32) * s;
        let j_lo = left.floor() as usize;
        let j_hi = (right.ceil() as usize).min(l);
        for j in j_lo..j_hi {
            let cell_left = (j as f32).max(left);
            let cell_right = ((j + 1) as f32).min(right);
            let weight = (cell_right - cell_left) / s;
            data[i * l + j] = weight;
        }
    }

    Tensor::<B, 1>::from_data(
        burn::tensor::TensorData::new(data, [m * l]),
        device,
    )
    .reshape([m, l])
}
```

Then add inside the existing `mod tests` block (after the `filter_nan_gauges_drops_columns` test at line 119, before the closing `}` at line 120):

```rust
    use burn::backend::NdArray;
    use burn::tensor::Tensor;
    type Bp = NdArray<f32>;

    #[test]
    fn area_pool_weights_rows_sum_to_one() {
        let device = Default::default();
        let w = area_pool_weights::<Bp>(2139, 89, &device);
        let row_sums: Tensor<Bp, 1> = w.sum_dim(1).squeeze(1);
        for v in row_sums.into_data().to_vec::<f32>().unwrap() {
            assert!((v - 1.0).abs() < 1e-5, "row sum {v} != 1");
        }
    }

    #[test]
    fn area_pool_matches_block_mean_when_divisible() {
        let device = Default::default();
        // input: 1..=48 over 48 hours, single gauge.
        let v: Vec<f32> = (1..=48).map(|x| x as f32).collect();
        let input: Tensor<Bp, 2> = Tensor::<Bp, 1>::from_data(
            burn::tensor::TensorData::new(v, [48]),
            &device,
        )
        .reshape([1, 48]);

        let w = area_pool_weights::<Bp>(48, 2, &device);
        let out: Tensor<Bp, 2> = input.matmul(w.transpose());
        let got: Vec<f32> = out.into_data().to_vec().unwrap();
        // Block 1 = mean(1..=24)  = 12.5
        // Block 2 = mean(25..=48) = 36.5
        assert!((got[0] - 12.5).abs() < 1e-5, "got {}", got[0]);
        assert!((got[1] - 36.5).abs() < 1e-5, "got {}", got[1]);
    }

    #[test]
    fn area_pool_handles_non_divisible_input() {
        let device = Default::default();
        let w = area_pool_weights::<Bp>(2139, 89, &device);
        let data: Vec<f32> = w.into_data().to_vec().unwrap();

        // Row 0 covers input range [0, 24.0337...). Cells 0-23 contribute
        // their full weight 1/s, cell 24 contributes the fractional piece.
        let s = 2139.0_f32 / 89.0;
        for j in 0..24 {
            let expected = 1.0 / s;
            assert!(
                (data[j] - expected).abs() < 1e-6,
                "row 0 col {j}: got {} want {expected}",
                data[j]
            );
        }
        let frac = (s - 24.0) / s;
        assert!(
            (data[24] - frac).abs() < 1e-4,
            "row 0 col 24: got {} want ~{frac}",
            data[24]
        );
        for j in 25..2139 {
            assert!(data[j].abs() < 1e-6, "row 0 col {j} should be 0; got {}", data[j]);
        }
    }
```

(The 4th test — `n_gauges_one_does_not_panic` — depends on the rewritten
`tau_trim_and_downsample` and is added in Task 2.)

- [ ] **Step 3: Build to confirm no syntax errors**

```bash
cd /home/tbindas/projects/ddrs && cargo build --lib 2>&1 | tail -10
```

Expected: clean build. The helper is unused at runtime — that triggers a
`dead_code` warning until Task 2 wires it in. That's fine; ignore for now
or add `#[allow(dead_code)]` temporarily.

- [ ] **Step 4: Run the new tests**

```bash
cargo test --lib training::loss::tests::area_pool 2>&1 | tail -15
```

Expected: 3 passes. If any fails, the helper math is wrong — diagnose
before continuing.

- [ ] **Step 5: Commit**

```bash
cd /home/tbindas/projects/ddrs
git add src/training/loss.rs
git commit -m "$(cat <<'EOF'
feat(loss): add area_pool_weights helper for area-mode downsample

Builds a constant (M, L) weight matrix implementing F.interpolate(
mode="area") semantics for 1D temporal data. Each row has at most
ceil(L/M)+1 nonzeros and sums to 1. Will replace the strict reshape+
mean in tau_trim_and_downsample (next commit) so DDRS matches DDR's
ddr/io/functions.py:22 exactly.

Three unit tests cover: divisible case (reduces to block-mean),
non-divisible 2139→89 weight shape (the production case), and the
rows-sum-to-1 invariant.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 2: Rewrite `tau_trim_and_downsample` body

**Spec ref:** §4.2.

**Files:**
- Modify: `src/training/loss.rs` (lines 16-36)

- [ ] **Step 1: Replace the function body**

Open `src/training/loss.rs`. Find the existing `tau_trim_and_downsample`
function (lines 16-36). Replace its body with the area-pool-matmul form.
The new body, in full:

```rust
/// Tau-trim then daily downsample via area-mode adaptive average pooling.
///
/// Mirrors DDR's `~/projects/ddr/src/ddr/io/functions.py:22`:
/// `F.interpolate(data.unsqueeze(1), size=(rho,), mode="area").squeeze(1)`.
///
/// Input shape `(G, T_hours)`. Slicing convention from DDR
/// `compute_daily_runoff`: `[13 + tau : -11 + tau]`. The trimmed length
/// does NOT need to be a multiple of 24 — fractional boundary hours are
/// handled by area-mode pooling.
///
/// Returns `(G, T_days)` where `T_days = T_hours_trimmed // 24` (matching
/// DDR's `num_days` computation at `scripts/train.py:78`).
pub fn tau_trim_and_downsample<B: Backend>(
    predictions_hourly: Tensor<B, 2>,
    tau: u32,
) -> Tensor<B, 2> {
    let dims = predictions_hourly.dims();
    let (g, t_hours) = (dims[0], dims[1]);
    let start = 13 + tau as usize;
    let end = t_hours - 11 + tau as usize;
    assert!(start < end, "tau-trim window degenerate: [{start}, {end})");
    let t_trimmed = end - start;
    let t_days = t_trimmed / 24;
    assert!(
        t_days > 0,
        "trimmed window too short: T_trimmed={t_trimmed}, T_days={t_days}"
    );

    let device = predictions_hourly.device();
    let sliced = predictions_hourly.slice([0..g, start..end]); // (G, L)
    let weights = area_pool_weights::<B>(t_trimmed, t_days, &device); // (M, L)
    // (G, L) @ (L, M) = (G, M)
    sliced.matmul(weights.transpose())
}
```

Concretely: lines 16-36 of the current file become this 26-line body.
Key removals:
- The `t_trimmed.is_multiple_of(24)` assertion (no longer required).
- The `reshape(g, t_days, 24)` and `mean_dim(2).squeeze::<2>()` chain.

Key additions:
- Slice → matmul with `area_pool_weights::<B>(...)`.

- [ ] **Step 2: Add the n_gauges=1 regression test**

Inside the same `mod tests` block, append (after the area-pool tests
landed in Task 1):

```rust
    #[test]
    fn n_gauges_one_does_not_panic() {
        let device = Default::default();
        // 2160 hourly input → 89 daily output for tau=3.
        let input: Tensor<Bp, 2> = Tensor::zeros([1, 2160], &device);
        let out = tau_trim_and_downsample(input, 3);
        assert_eq!(out.dims(), [1, 89]);
    }

    #[test]
    fn tau_trim_matches_old_block_mean_on_divisible_input() {
        // Verify the new area-pool body is a drop-in for the previous
        // reshape+mean impl whenever the trimmed window IS a multiple
        // of 24. tau=11 -> trimmed length = T - 11 + 11 - 13 = T - 13.
        // Pick T = 13 + 48 = 61 hours so trimmed = 48 = 2 days exactly.
        let device = Default::default();
        let v: Vec<f32> = (0..61).map(|x| x as f32).collect();
        let input: Tensor<Bp, 2> = Tensor::<Bp, 1>::from_data(
            burn::tensor::TensorData::new(v, [61]),
            &device,
        )
        .reshape([1, 61]);

        // tau = 11 -> start = 24, end = 61 - 11 + 11 = 61, trimmed = 37
        // Hmm, that's not multiple of 24. Adjust: pick T s.t. T-13-(11-tau) % 24 == 0.
        // With tau=11: end = T-11+11 = T, start = 13+11 = 24, trimmed = T-24.
        // Need T-24 multiple of 24 → T = 48, 72, 96, ...
        // Use T=72 hours, tau=11 → trimmed = 48 → 2 daily bins.

        let v: Vec<f32> = (0..72).map(|x| x as f32).collect();
        let input: Tensor<Bp, 2> = Tensor::<Bp, 1>::from_data(
            burn::tensor::TensorData::new(v, [72]),
            &device,
        )
        .reshape([1, 72]);
        let out = tau_trim_and_downsample(input, 11);
        let got: Vec<f32> = out.into_data().to_vec().unwrap();
        // Sliced = hours 24..72 (48 values: 24..=71).
        // Day 1 = mean(24..=47) = 35.5
        // Day 2 = mean(48..=71) = 59.5
        assert!((got[0] - 35.5).abs() < 1e-4, "got {}", got[0]);
        assert!((got[1] - 59.5).abs() < 1e-4, "got {}", got[1]);
    }
```

- [ ] **Step 3: Build + run all loss tests**

```bash
cd /home/tbindas/projects/ddrs && cargo test --lib training::loss 2>&1 | tail -20
```

Expected: all loss tests pass (existing `l1_loss_post_warmup_basic`,
`filter_nan_gauges_drops_columns`, plus the 3 area-pool tests from Task 1,
plus the 2 new tests added here = **7 passes**). If `dead_code` warning on
`area_pool_weights` is still present, remove the `#[allow(dead_code)]` if
you added one in Task 1.

- [ ] **Step 4: Commit**

```bash
cd /home/tbindas/projects/ddrs
git add src/training/loss.rs
git commit -m "$(cat <<'EOF'
fix(loss): use area-mode pooling in tau_trim_and_downsample

Replaces the strict reshape+mean (which required the trimmed window to
be a multiple of 24) with a matmul against area_pool_weights —
F.interpolate(mode="area") semantics, matching DDR's
ddr/io/functions.py:22.

Drop-in for the old behaviour when the trimmed window IS a multiple
of 24 (verified by tau_trim_matches_old_block_mean_on_divisible_input).
For non-divisible windows (the production case: T_trimmed=2139, T_days=89
when tau=3), each daily bin includes fractional contributions from
boundary hours instead of dropping them.

Also fixes the n_gauges=1 squeeze regression flagged in PR #13's Layer
B — matmul preserves the leading axis cleanly, so no squeeze is needed.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 3: Verify PR #13 parity tests at OLD tolerances

**Spec ref:** §4.4 — phase 1 (sanity).

**Files:** none modified.

The point of this task is a sanity gate before tightening anything: confirm
the rewrite doesn't regress any existing test. Tests should pass with
**much smaller** diff magnitudes than the old tolerances allowed.

- [ ] **Step 1: Run the full training-step parity suite**

```bash
cd /home/tbindas/projects/ddrs
cargo test --features fixtures \
    --test training_step_layer_b \
    --test training_step_layer_c \
    --test training_step_layer_d \
    -- --nocapture 2>&1 | tail -50
```

Expected: 10/10 pass (same as PR #13). The diff magnitudes printed via
`--nocapture` should be substantially smaller than before — capture them:

- Layer B4 daily Q max abs diff: was 5.01e-2; expect ~1e-6 to 1e-5.
- Layer C2 worst grad diff: was 1.76e-5; expect ~1e-6.
- Layer D1 worst stepped-param diff: was 1.49e-3; expect ~1e-5.

If ANY test fails, the fix has a bug somewhere — STOP and diagnose. Likely
causes: off-by-one in the weight matrix, wrong axis on `transpose()`, or
the `(G, L) @ (L, M) = (G, M)` shape contract isn't being respected by
burn's matmul.

- [ ] **Step 2: Record the new diff magnitudes**

Note them down — they'll inform Task 4's tolerance values + Task 6's spec
§5.1 entry. No commit yet.

---

## Task 4: Tighten PR #13 parity test tolerances

**Spec ref:** §4.4 — phase 2.

**Files:**
- Modify: `tests/training_step_layer_b.rs` (Layer B4 tolerance)
- Modify: `tests/training_step_layer_c.rs` (Layer C2 tolerance)
- Modify: `tests/training_step_layer_d.rs` (Layer D1 tolerance)

- [ ] **Step 1: Find the tolerance constants**

Each test file currently has the tolerance encoded inline. Locate:

```bash
grep -nE 'TOLERANCE|<= 0\.1|<= 1e-3|<= 2e-3|tol\s*=' \
    /home/tbindas/projects/ddrs/tests/training_step_layer_b.rs \
    /home/tbindas/projects/ddrs/tests/training_step_layer_c.rs \
    /home/tbindas/projects/ddrs/tests/training_step_layer_d.rs
```

- [ ] **Step 2: Tighten Layer B4 in `tests/training_step_layer_b.rs`**

Find the assert for sub-test 4 (daily Q parity). Change the tolerance
literal from `0.1_f32` (or whatever the current value is) to `1e-5_f32`.
Also update any inline comment that mentions "STAT-only" / "C7" to note
that the area-pool fix closed the gap:

```rust
    // Per the area-mode-downsample fix (commit <SHA from Task 2>):
    // the daily Q now matches DDR's F.interpolate(mode="area") result.
    // Residual diff is at f32 floor.
    assert!(diff <= 1e-5, "daily Q max abs diff {diff} > 1e-5");
```

- [ ] **Step 3: Tighten Layer C2 in `tests/training_step_layer_c.rs`**

Change the per-KAN-param gradient tolerance from `1e-3` to `1e-5`:

```rust
    let tol = 1e-5_f32; // tightened from 1e-3 post area-pool fix
```

Or wherever the constant lives. If the tolerance is hard-coded per-tensor,
update all instances.

- [ ] **Step 4: Tighten Layer D1 in `tests/training_step_layer_d.rs`**

Change the post-Adam-step params tolerance from `2e-3` to `1e-5`:

```rust
    assert!(diff <= 1e-5, "{label}: max abs diff {diff} > 1e-5");
```

- [ ] **Step 5: Re-run the suite at the new tolerances**

```bash
cd /home/tbindas/projects/ddrs
cargo test --features fixtures \
    --test training_step_layer_b \
    --test training_step_layer_c \
    --test training_step_layer_d \
    2>&1 | tail -15
```

Expected: 10/10 pass at the new tolerances.

If any fails, the residual diff is real — that's either:
- The C7 tau-asymmetry (intentional per the user) showing through; in
  which case widen the failing test's tolerance back up to the diff
  magnitude observed in Task 3 Step 2, with a comment citing C7.
- A second smaller divergence beyond C7; flag in the report.

- [ ] **Step 6: Commit**

```bash
cd /home/tbindas/projects/ddrs
git add tests/training_step_layer_b.rs \
        tests/training_step_layer_c.rs \
        tests/training_step_layer_d.rs
git commit -m "$(cat <<'EOF'
test: tighten Layer B4/C2/D1 tolerances post area-pool fix

After the tau_trim_and_downsample rewrite (commit <Task 2 SHA>), the
diff magnitudes across the training-step parity suite collapse to f32
floor. Tightens the tolerances so any regression will fail loudly:

  - Layer B4 (daily Q): 0.1     -> 1e-5
  - Layer C2 (gradients): 1e-3  -> 1e-5
  - Layer D1 (Adam-stepped):    -> 1e-5  (was 2e-3)

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 5: Retrain DDRS + dump kan_parameters.nc

**Spec ref:** §4.5.

**Files:** none committed (training outputs are gitignored).

- [ ] **Step 1: Patch the workspace ddrs.yaml**

The bootstrap-from-last-successful-run trap (documented in CLAUDE.md after
PR #12) means the gitignored `ddrs.yaml` is stale relative to
`config/merit_training.yaml`. Use the explicit `--config` flag to bypass
it:

```bash
cd /home/tbindas/projects/ddrs
grep -A2 log_space config/merit_training.yaml
```

Confirm `log_space_parameters: [p_spatial]` is present (it should be from
PR #12).

- [ ] **Step 2: Launch the retrain**

```bash
cd /home/tbindas/projects/ddrs
cargo run --release --bin ddrs -- \
    --config config/merit_training.yaml \
    run --workflow train-and-test \
    > /tmp/ddrs_train_areapool_fix.log 2>&1 &
echo "PID: $!"
```

Capture the PID. Monitor:

```bash
tail -f /tmp/ddrs_train_areapool_fix.log | grep --line-buffered -E 'mb=|chunk|median NSE|run complete|nan|NaN|error'
```

Watch for:
- Loss decreasing smoothly (no NaN, no divergence).
- Run completes (~30-45 min on a 4090).
- Final line `run complete → ./.ddrs/runs/<timestamp>-train-and-test`.

Capture the new run-id `<timestamp>-train-and-test`.

- [ ] **Step 3: Dump kan_parameters.nc**

```bash
RUN=.ddrs/runs/<timestamp>-train-and-test
CKPT=$RUN/checkpoints/$(ls $RUN/checkpoints/ | grep '\.mpk$' | sort -V | tail -1 | sed 's/\.mpk$//')
echo "CKPT=$CKPT"
cargo run --release --bin dump_parameters -- \
    --config $RUN/config.yaml \
    --checkpoint $CKPT \
    --output $RUN/kan_parameters.nc 2>&1 | tail -3
```

Expected: `wrote 346321 reaches → ...`.

- [ ] **Step 4: Quick distribution check**

```bash
cd /home/tbindas/projects/ddrs/ddrs-py && uv run --extra plots python <<PY
import xarray as xr
import numpy as np
RUN = "/home/tbindas/projects/ddrs/.ddrs/runs/<timestamp>-train-and-test"
new = xr.open_dataset(f"{RUN}/kan_parameters.nc").n.values
print(f"DDR reference (target):    median 0.0744  mean 0.0735  p5 0.0387  p95 0.1047  frac<.035 0.031")
print(f"DDRS post-area-pool fix:   median {np.median(new):.4f}  mean {np.mean(new):.4f}  "
      f"p5 {np.percentile(new,5):.4f}  p95 {np.percentile(new,95):.4f}  "
      f"frac<.035 {(new<0.035).mean():.3f}")
PY
```

Expected: median `n` moves into the 0.05-0.10 range (close to DDR's
0.074), `frac<0.035` drops to ~10% or below. **Capture the actual
numbers** — they feed §5.1.

If the median is still saturated (< 0.04), the fix didn't close the gap
fully. That's not a `Task 5` failure — proceed to Task 6 and record the
result honestly. The next investigation would be a fresh spec.

---

## Task 6: Re-run PR #12's Layer 2 notebook + record §5.1

**Spec ref:** §4.5 + §5.1 entry.

**Files:**
- Modify: `docs/superpowers/specs/2026-06-04-ddrs-area-mode-downsample-fix-design.md`
  (append §5.1)

- [ ] **Step 1: Generate DDR reference NetCDF if needed**

```bash
ls -la /tmp/kan_params_trained_ddr.nc 2>/dev/null
```

If missing (it's `/tmp`-based, so may have been cleared):

```bash
DDR_CKPT=~/projects/ddr/output/ddr-v0.5.2.dev2+g21a3a96b5-merit-training/2026-03-14_06-03-23/saved_models/_ddr-v0.5.2.dev2+g21a3a96b5-merit-training_epoch_5_mb_35.pt
cd ~/projects/ddr && uv run python /home/tbindas/projects/ddrs/scripts/dump_ddr_trained_params.py \
    --checkpoint "$DDR_CKPT" \
    --conus-comids /home/tbindas/projects/ddrs/.ddrs/runs/<TASK 5 RUN-ID>/kan_parameters.nc \
    --out /tmp/kan_params_trained_ddr.nc 2>&1 | tail -3
```

Expected: `wrote 346321 CONUS reaches → /tmp/kan_params_trained_ddr.nc`.

- [ ] **Step 2: Generate + execute the parity_trained notebook**

```bash
cd /home/tbindas/projects/ddrs/ddrs-py
RUN_DIR=/home/tbindas/projects/ddrs/.ddrs/runs/<TASK 5 RUN-ID>
mkdir -p "$RUN_DIR/plots"

RUN_DIR="$RUN_DIR" uv run --extra plots python <<'PY'
import nbformat as nbf
import os, re
skill_md = open("/home/tbindas/projects/ddrs/.claude/skills/ddrs-eval-plots/references/parity_trained.md").read()
section = skill_md.split("## Notebook cells", 1)[1].split("## Pass criterion", 1)[0]
code_blocks = re.findall(r"```python\n(.*?)```", section, re.DOTALL)
assert len(code_blocks) == 5
nb = nbf.v4.new_notebook()
nb.cells.append(nbf.v4.new_markdown_cell("# DDR ↔ DDRS trained-`n` parity (post area-pool fix)"))
for src in code_blocks:
    nb.cells.append(nbf.v4.new_code_cell(src.strip()))
out = os.path.join(os.environ["RUN_DIR"], "plots", "parity_trained.ipynb")
nbf.write(nb, out)
print(f"wrote {out}")
PY

uv run --extra plots jupyter nbconvert --to notebook --execute \
    "$RUN_DIR/plots/parity_trained.ipynb" \
    --output parity_trained.ipynb \
    --output-dir "$RUN_DIR/plots" 2>&1 | tail -10
```

- [ ] **Step 3: Extract Cell 5's verdict**

```bash
cd /home/tbindas/projects/ddrs/ddrs-py
RUN_DIR=/home/tbindas/projects/ddrs/.ddrs/runs/<TASK 5 RUN-ID>
uv run --extra plots python <<PY
import json, os
nb = json.load(open(f"{os.environ['RUN_DIR']}/plots/parity_trained.ipynb"))
# Cell 5 is the verdict cell (markdown header at 0, code 1-5)
for out in nb["cells"][5].get("outputs", []):
    t = out.get("text") or (out.get("data", {}) or {}).get("text/plain")
    if t:
        print("".join(t) if isinstance(t, list) else t)
PY
```

Capture verbatim — these three lines (n / q_spatial / p_spatial verdicts)
go into §5.1.

- [ ] **Step 4: Append §5.1 to the fix spec**

Open `docs/superpowers/specs/2026-06-04-ddrs-area-mode-downsample-fix-design.md`.
Append at EOF:

```markdown
---

## §5.1 Empirical verdict (Task 6 of the plan)

**Pre-fix baseline (PR #12's `2026-06-04T01-57-45Z-train-and-test`):**
- DDRS median n = 0.0296, KS vs DDR = 0.6916, Spearman = +0.1790 → ✗ real divergence.

**Post-fix retrain (`<TASK 5 RUN-ID>`):**
- DDRS median n = <captured from Task 5 Step 4>
- DDR reference median n = 0.0744

**Parity notebook Cell 5 verdict:**

```
<paste the 3 lines from Task 6 Step 3 verbatim>
```

**Layer B4 / C2 / D1 tightened tolerances** all pass at 1e-5 after the
area-pool fix (commit <Task 4 SHA>):

| Test | Old tol | New tol | New diff magnitude |
|---|---|---|---|
| Layer B4 daily Q | 0.1 | 1e-5 | <captured from Task 3> |
| Layer C2 gradients | 1e-3 | 1e-5 | <captured from Task 3> |
| Layer D1 post-Adam params | 2e-3 | 1e-5 | <captured from Task 3> |

**Outcome (from §5 of this spec):**

<one of the four §5 rows; quote it verbatim>

**Next step:** <inherited from the §5 row that matched, OR "investigation
closed" if the n distribution lands at KS ≤ 0.10 and Spearman ≥ 0.70.>
```

- [ ] **Step 5: Commit**

```bash
cd /home/tbindas/projects/ddrs
git add docs/superpowers/specs/2026-06-04-ddrs-area-mode-downsample-fix-design.md
git commit -m "$(cat <<'EOF'
docs/spec: record area-pool fix verdict

After the area-pool downsample fix landed (commits <Task 2 SHA>,
<Task 4 SHA>), tightened Layer B4/C2/D1 tolerances all pass at 1e-5,
and a fresh DDRS retrain produced the trained-n distribution recorded
in §5.1.

The trained-distribution parity notebook (PR #12's parity_trained.md)
returned <verdict from Task 6 Step 3>. <closes the n-saturation
investigation / next investigation needed>.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Spec coverage map

| Spec section | Plan task(s) |
|--------------|-------------|
| §2 C1 (no built-in interpolate) | Task 1 — hand-coded `area_pool_weights` |
| §2 C2 (n_gauges=1 squeeze bug) | Task 2 — matmul preserves leading axis; `n_gauges_one_does_not_panic` test |
| §2 C3 (autograd correctness) | Tasks 3-4 — Layer C2 + D1 cover backward via existing PR #13 tests |
| §2 C4 (old checkpoints invalid) | Task 5 — fresh retrain |
| §2 C5 (unsqueeze/squeeze shape adapter) | Task 1 — confirmed no semantic effect; per-batch math is the same |
| §3 A1 (no custom Backward needed) | Task 1's matmul-based implementation, validated by Task 3 |
| §3 A2 (tightened tolerances are the verification) | Task 4 |
| §3 A3 (squeeze bug bundled) | Task 2 |
| §3 A4 (signature preserved) | Task 2 — kept the `pub fn tau_trim_and_downsample` signature |
| §3 A5 (tau-slicing unchanged) | Tasks 1-2 — only the downsample changed, not the slice |
| §4.1-4.3 (algorithm + implementation + unit tests) | Tasks 1-2 |
| §4.4 (tighten PR #13 tolerances) | Task 4 |
| §4.5 (retrain + verify) | Tasks 5-6 |
| §5 (outcome) | Task 6 |
| §6 (implementation order) | Plan task order |

---

Plan complete and saved to
`docs/superpowers/plans/2026-06-04-ddrs-area-mode-downsample-fix.md`.

**Two execution options:**

**1. Subagent-Driven (recommended)** — fresh subagent per task with spec + code-quality review between each.

**2. Inline Execution** — `superpowers:executing-plans` with batch checkpoints.

Which?

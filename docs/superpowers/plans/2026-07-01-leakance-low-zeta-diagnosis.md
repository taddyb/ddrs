# Leakance Low-Zeta Diagnosis Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Explain why the learned leakance flux is tiny (median |zeta| 6.4e-4 m³/s, K_D 100% ceiling-pinned) via a 7-hypothesis falsification battery, then run one gated K_D-widening retrain if warranted.

**Architecture:** Phase 1 extends the existing eval-time zeta accumulator (`MuskingumCunge` → `ZetaSums` → `EvalOutput` → `write_zeta_netcdf`) to also export per-reach eval-window mean depth, plan-view area (`area_z`), and discharge. Phase 2 is a standalone Python script over the netCDF exports + attributes. Phase 3 is one config clone + retrain, gated on Phase-2 verdicts.

**Tech Stack:** Rust (BURN 0.21, netcdf crate), Python under `~/projects/ddr`'s uv venv (numpy/xarray/netCDF4/scipy).

**Spec:** `docs/superpowers/specs/2026-07-01-leakance-low-zeta-diagnosis-design.md`
(Spec delta: we additionally export `area_z_mean` — it's free from `LeakanceSaved` and makes H1 exact instead of re-deriving geometry in Python.)

**Working directories — read carefully.** Code work happens in this worktree
(`~/projects/ddrs/.claude/worktrees/leakance-diagnosis`, branch
`worktree-leakance-diagnosis`). GPU runs and run-dir artifacts live in the MAIN
tree's workspace (`/home/tbindas/projects/ddrs/.ddrs/...`) — evals are invoked
with `cd /home/tbindas/projects/ddrs` using the WORKTREE's freshly built binary
(absolute path). Never install over `~/.cargo/bin/ddrs` from this worktree.

**Key run dirs (all under `/home/tbindas/projects/ddrs/.ddrs/runs/`):**

| arm | run id | has kan_parameters.nc? |
|---|---|---|
| hourly-ON | `2026-07-01T13-43-32Z-train-and-test` | yes (full dump + zeta) |
| daily-ON | `2026-07-01T21-20-27Z-train-and-test` | yes (full dump + zeta) |
| hourly-OFF | `2026-06-23T02-49-12Z-conus-hourly-train-and-test` | **no** (Task 5 creates it) |
| daily-OFF | `2026-06-05T01-41-16Z-train-and-test` | yes |

---

## File structure

| File | Change | Responsibility |
|---|---|---|
| `src/routing/mmc_op.rs` | modify | `ZetaStepDiag<I>` struct; sink now carries zeta+depth+area_z |
| `src/routing/mmc.rs` | modify | accumulate depth/area_z/q sums; `zeta_sums()` returns `ZetaSumTensors<I>` |
| `src/routing/mod.rs` | modify | re-export `ZetaSumTensors` |
| `src/training/forward.rs` | modify | `ZetaSums` gains depth/area_z/q sums; `merge(ZetaSumTensors)` |
| `src/training/eval.rs` | modify | `EvalOutput` gains 3 mean vectors |
| `src/training/zarr_io.rs` | modify | test literal gains 3 `None` fields |
| `src/dump_parameters.rs` | modify | `write_zeta_netcdf` writes `depth_mean`/`area_z_mean`/`q_mean` |
| `src/cli/run.rs`, `src/bin/eval.rs` | modify | updated call sites |
| `tests/zeta_accum.rs` | modify | struct destructuring + 2 new tests |
| `scripts/leakance_diagnosis.py` | create | Phase-2 hypothesis battery |
| `docs/2026-07-02-leakance-diagnosis-findings.md` | create | ranked verdicts |
| `config/experiments/leakance_hourly_on_kd4.yaml` | create (Task 9, gated) | widened K_D arm |

---

## Phase 1 — Rust instrumentation

### Task 1: Extend `tests/zeta_accum.rs` (write the failing tests)

**Files:**
- Modify: `tests/zeta_accum.rs`

The accumulator API changes from tuple-returning `zeta_sums() -> Option<(Tensor, Tensor, usize)>` to a struct `ZetaSumTensors { abs, net, depth, area_z, q, steps }`. Update the three existing destructurings and add two new tests.

- [ ] **Step 1: Update existing destructurings to the struct API**

In `tests/zeta_accum.rs`, replace (in `accumulation_does_not_perturb_discharge`, currently line 99):

```rust
    let (abs, net, steps) = mc_accum.zeta_sums().expect("zeta sums present");
    assert_eq!(steps, t - 1, "one accumulated step per routed timestep");
    let abs_v: Vec<f32> = abs.into_data().to_vec().unwrap();
    let net_v: Vec<f32> = net.into_data().to_vec().unwrap();
```

with:

```rust
    let sums = mc_accum.zeta_sums().expect("zeta sums present");
    assert_eq!(sums.steps, t - 1, "one accumulated step per routed timestep");
    let abs_v: Vec<f32> = sums.abs.into_data().to_vec().unwrap();
    let net_v: Vec<f32> = sums.net.into_data().to_vec().unwrap();
```

In `accumulated_zeta_equals_headwater_qnext_difference` (currently line 141), replace:

```rust
    let (abs, _, steps) = mc_on.zeta_sums().expect("zeta sums present");
    assert_eq!(steps, 1);
    let zeta: Vec<f32> = abs.into_data().to_vec().unwrap();
```

with:

```rust
    let sums = mc_on.zeta_sums().expect("zeta sums present");
    assert_eq!(sums.steps, 1);
    let zeta: Vec<f32> = sums.abs.into_data().to_vec().unwrap();
```

In `zeta_is_linear_in_leakance_factor_on_single_step` (currently line 170), replace:

```rust
        let (abs, _, _) = mc.zeta_sums().expect("zeta sums present");
        abs.into_data().to_vec().unwrap()
```

with:

```rust
        let sums = mc.zeta_sums().expect("zeta sums present");
        sums.abs.into_data().to_vec().unwrap()
```

- [ ] **Step 2: Append the two new tests at the end of the file**

```rust
#[test]
fn q_mean_matches_routed_discharge() {
    let device = TestDevice::default();
    let (n, t) = (5usize, 24usize);
    let cfg = mock_config();

    let mut mc = MuskingumCunge::<InnerBackend>::new(cfg, device.clone());
    mc.enable_zeta_accumulation();
    mc.setup_inputs(
        mock_routing_inputs(n, &device),
        mock_streamflow(t, n, &device),
        leakance_params(n, 1.0, &device),
        false,
    );
    let out = forward_vec(&mut mc); // [n, t] row-major

    // q_sum accumulates the SAME q_next tensors that become output columns
    // 1..t, in the same order, so the sums match to f32 addition noise.
    let sums = mc.zeta_sums().expect("zeta sums present");
    assert_eq!(sums.steps, t - 1);
    let q_sum: Vec<f32> = sums.q.into_data().to_vec().unwrap();
    assert_eq!(q_sum.len(), n);
    for i in 0..n {
        let expected: f32 = (1..t).map(|j| out[i * t + j]).sum();
        assert!(
            (q_sum[i] - expected).abs() <= 1e-5 * expected.abs().max(1.0),
            "reach {i}: q_sum ({}) must equal summed routed discharge ({expected})",
            q_sum[i]
        );
    }
}

#[test]
fn depth_and_area_z_are_leakance_independent_primitives() {
    let device = TestDevice::default();
    let (n, t) = (5usize, 2usize); // single routed timestep

    // Depth at t=1 is a function of the hotstart Q0 only, so depth and area_z
    // must be identical across leakance_factor values while zeta scales.
    let cfg = mock_config();
    let run = |factor_norm: f32| {
        let mut mc = MuskingumCunge::<InnerBackend>::new(cfg.clone(), device.clone());
        mc.enable_zeta_accumulation();
        mc.setup_inputs(
            mock_routing_inputs(n, &device),
            mock_streamflow(t, n, &device),
            leakance_params(n, factor_norm, &device),
            false,
        );
        let _ = mc.forward();
        mc.zeta_sums().expect("zeta sums present")
    };

    let full = run(1.0);
    let half = run(0.5);

    let depth_f: Vec<f32> = full.depth.into_data().to_vec().unwrap();
    let depth_h: Vec<f32> = half.depth.into_data().to_vec().unwrap();
    let area_f: Vec<f32> = full.area_z.into_data().to_vec().unwrap();
    let area_h: Vec<f32> = half.area_z.into_data().to_vec().unwrap();
    assert_eq!(depth_f, depth_h, "depth must not depend on leakance_factor");
    assert_eq!(area_f, area_h, "area_z must not depend on leakance_factor");

    // Structural identity: zeta = factor·area_z·K_D·(depth − d_gw). With
    // uniform factor/K_D and d_gw = −2 m (leakance_params denormalizes to the
    // bottom of [-2, 2]), zeta/(area_z·(depth+2)) is the SAME for every reach.
    let abs_f: Vec<f32> = full.abs.into_data().to_vec().unwrap();
    let ratios: Vec<f32> = (0..n)
        .map(|i| abs_f[i] / (area_f[i] * (depth_f[i] + 2.0)))
        .collect();
    for r in &ratios {
        assert!(r.is_finite() && *r > 0.0, "ratio must be positive finite, got {r}");
        assert!(
            (r - ratios[0]).abs() <= 1e-5 * ratios[0],
            "zeta/(area_z·(depth−d_gw)) must be uniform across reaches: {ratios:?}"
        );
    }
}
```

- [ ] **Step 3: Run to verify the tests fail to compile**

Run: `cargo test --test zeta_accum 2>&1 | tail -20`
Expected: compile errors — `zeta_sums()` returns a tuple (no `.abs`/`.steps` fields), no `depth`/`area_z`/`q` fields exist yet.

- [ ] **Step 4: Commit the failing tests**

```bash
git add tests/zeta_accum.rs
git commit -m "test(zeta_accum): depth/area_z/q accumulation contract (red)"
```

### Task 2: Implement the diag sink + accumulation

**Files:**
- Modify: `src/routing/mmc_op.rs` (sink struct + population, ~line 35 and ~line 1431–1539)
- Modify: `src/routing/mmc.rs` (fields ~90–131, leakance branch ~326–348, accessors ~436–451)
- Modify: `src/routing/mod.rs` (re-export)

- [ ] **Step 1: Add `ZetaStepDiag` in `src/routing/mmc_op.rs`**

Directly above `pub(crate) struct LeakanceSaved<I: Backend>` (line 35), add:

```rust
/// Per-step eval-time leakance diagnostics captured by the zeta sink: this
/// step's zeta plus the primitives needed to interpret it (routed depth and
/// plan-view wetted area). Inner backend, no tape.
pub struct ZetaStepDiag<I: Backend> {
    pub zeta: Tensor<I, 1>,
    pub depth: Tensor<I, 1>,
    pub area_z: Tensor<I, 1>,
}
```

- [ ] **Step 2: Change the sink type and population in `timestep_forward_leakance`**

Change the parameter (line 1451) from

```rust
    zeta_out: Option<&mut Option<Tensor<I, 1>>>,
```

to

```rust
    zeta_out: Option<&mut Option<ZetaStepDiag<I>>>,
```

and update the doc comment above the function: `receives this step's zeta` → `receives this step's zeta, routed depth, and area_z`. Replace the population block (lines 1531–1539):

```rust
    // Eval-time zeta diagnostic: zeta = factor · area_z · K_D · (depth − d_gw),
    // recomputed from the saved primitives (cheap: 3 elementwise kernels,
    // only when a sink is supplied). Depth and area_z ride along for the
    // low-zeta diagnosis (driving head + structural-ceiling analyses).
    if let Some(out) = zeta_out {
        let depth = wrap(depth_p.clone());
        let area_z = wrap(leak.area_z.clone());
        let m = depth.clone() - wrap(leak.d_gw.clone());
        *out = Some(ZetaStepDiag {
            zeta: wrap(leak.leakance_factor.clone()) * area_z.clone() * wrap(leak.k_d.clone()) * m,
            depth,
            area_z,
        });
    }
```

(`tests/leakance_gradcheck.rs:151` passes a bare `None` for this parameter — it still compiles unchanged.)

- [ ] **Step 3: Extend `MuskingumCunge` accumulation in `src/routing/mmc.rs`**

Fields (after `zeta_net_sum`, line 95):

```rust
    zeta_abs_sum: Option<Tensor<I, 1>>,
    zeta_net_sum: Option<Tensor<I, 1>>,
    depth_sum: Option<Tensor<I, 1>>,
    area_z_sum: Option<Tensor<I, 1>>,
    q_sum: Option<Tensor<I, 1>>,
    zeta_steps: usize,
```

Constructor init (line 128 area): add `depth_sum: None, area_z_sum: None, q_sum: None,`.

Replace the leakance-branch accumulation (lines 326–347):

```rust
            let mut zeta_step: Option<crate::routing::mmc_op::ZetaStepDiag<I>> = None;
            let q_next = crate::routing::mmc_op::timestep_forward_leakance::<I>(
                &self.cfg, pattern, assembler,
                n, q_spatial, p_spatial,
                q_t, q_prime_clamp,
                length, slope, x_storage,
                k_d, d_gw, leakance_factor,
                if self.collect_zeta { Some(&mut zeta_step) } else { None },
            );
            if let Some(diag) = zeta_step {
                fn add<I: Backend>(slot: &mut Option<Tensor<I, 1>>, v: Tensor<I, 1>) {
                    *slot = Some(match slot.take() {
                        Some(s) => s + v,
                        None => v,
                    });
                }
                add(&mut self.zeta_abs_sum, diag.zeta.clone().abs());
                add(&mut self.zeta_net_sum, diag.zeta);
                add(&mut self.depth_sum, diag.depth);
                add(&mut self.area_z_sum, diag.area_z);
                add(&mut self.q_sum, q_next.clone().inner());
                self.zeta_steps += 1;
            }
            return q_next;
```

- [ ] **Step 4: Replace `zeta_sums()` with the struct-returning version**

Above `impl MuskingumCunge` (or directly above the method), add the struct; replace the method (lines 443–451):

```rust
/// Accumulated eval-time leakance diagnostics (inner backend, no tape).
/// All fields are per-reach sums over the accumulated timesteps; divide by
/// `steps` for eval-window means.
pub struct ZetaSumTensors<I: Backend> {
    /// Σ|zeta| (m³/s · steps).
    pub abs: Tensor<I, 1>,
    /// Σ zeta, signed (positive = losing reach).
    pub net: Tensor<I, 1>,
    /// Σ routed depth (m · steps).
    pub depth: Tensor<I, 1>,
    /// Σ plan-view wetted area `area_z` (m² · steps).
    pub area_z: Tensor<I, 1>,
    /// Σ routed discharge `q_next` (m³/s · steps).
    pub q: Tensor<I, 1>,
    /// Number of accumulated timesteps.
    pub steps: usize,
}
```

```rust
    /// Eval-time leakance diagnostic sums accumulated across `route_timestep`
    /// calls since construction. `None` until the first accumulated step.
    pub fn zeta_sums(&self) -> Option<ZetaSumTensors<I>> {
        match (
            &self.zeta_abs_sum,
            &self.zeta_net_sum,
            &self.depth_sum,
            &self.area_z_sum,
            &self.q_sum,
        ) {
            (Some(a), Some(n), Some(d), Some(az), Some(q)) => Some(ZetaSumTensors {
                abs: a.clone(),
                net: n.clone(),
                depth: d.clone(),
                area_z: az.clone(),
                q: q.clone(),
                steps: self.zeta_steps,
            }),
            _ => None,
        }
    }
```

Place `ZetaSumTensors` at module level in `mmc.rs` (not inside the impl block).

- [ ] **Step 5: Re-export from `src/routing/mod.rs`**

```rust
pub use mmc::{MuskingumCunge, RoutingInputs, SpatialParameters, ZetaSumTensors};
```

- [ ] **Step 6: Run the accumulator tests — expect the downstream plumbing to still break the build**

Run: `cargo test --test zeta_accum 2>&1 | tail -20`
Expected: compile errors now come from `src/training/forward.rs:385` (destructuring the old tuple). That's Task 3. If the errors are in `mmc.rs`/`mmc_op.rs`, fix them here first.

### Task 3: Plumb through training/eval/netcdf/CLI

**Files:**
- Modify: `src/training/forward.rs:254–285, 379–388`
- Modify: `src/training/eval.rs:28–41, 171–192`
- Modify: `src/training/zarr_io.rs:296` (test literal)
- Modify: `src/dump_parameters.rs:499–576`
- Modify: `src/cli/run.rs:404–416`
- Modify: `src/bin/eval.rs:130–152`

- [ ] **Step 1: `ZetaSums` in `src/training/forward.rs`**

Replace the struct + impl (lines 257–279):

```rust
pub struct ZetaSums<I: Backend> {
    pub abs_sum: Option<Tensor<I, 1>>,
    pub net_sum: Option<Tensor<I, 1>>,
    pub depth_sum: Option<Tensor<I, 1>>,
    pub area_z_sum: Option<Tensor<I, 1>>,
    pub q_sum: Option<Tensor<I, 1>>,
    pub steps: usize,
}

impl<I: Backend> ZetaSums<I> {
    pub fn new() -> Self {
        Self {
            abs_sum: None,
            net_sum: None,
            depth_sum: None,
            area_z_sum: None,
            q_sum: None,
            steps: 0,
        }
    }

    fn merge(&mut self, sums: crate::routing::ZetaSumTensors<I>) {
        fn add<I: Backend>(slot: &mut Option<Tensor<I, 1>>, v: Tensor<I, 1>) {
            *slot = Some(match slot.take() {
                Some(s) => s + v,
                None => v,
            });
        }
        add(&mut self.abs_sum, sums.abs);
        add(&mut self.net_sum, sums.net);
        add(&mut self.depth_sum, sums.depth);
        add(&mut self.area_z_sum, sums.area_z);
        add(&mut self.q_sum, sums.q);
        self.steps += sums.steps;
    }
}
```

And the merge call in `forward_eval` (lines 384–388):

```rust
    if let Some(sink) = zeta {
        if let Some(sums) = engine.zeta_sums() {
            sink.merge(sums);
        }
    }
```

- [ ] **Step 2: `EvalOutput` + means in `src/training/eval.rs`**

Add after `zeta_net_mean` (line 38):

```rust
    /// Eval-window mean routed depth per reach (m). Same gating as zeta.
    pub zeta_depth_mean: Option<Vec<f32>>,
    /// Eval-window mean plan-view wetted area `area_z` per reach (m²).
    pub zeta_area_z_mean: Option<Vec<f32>>,
    /// Eval-window mean routed discharge per reach (m³/s).
    pub zeta_q_mean: Option<Vec<f32>>,
```

Replace the means block (lines 171–181):

```rust
    // Leakance diagnostic: sums → per-reach means over the routed timesteps.
    let (zeta_abs_mean, zeta_net_mean, zeta_depth_mean, zeta_area_z_mean, zeta_q_mean, zeta_comids) =
        match (
            zeta_sums.abs_sum,
            zeta_sums.net_sum,
            zeta_sums.depth_sum,
            zeta_sums.area_z_sum,
            zeta_sums.q_sum,
            zeta_sums.steps,
        ) {
            (Some(abs), Some(net), Some(depth), Some(area_z), Some(q), steps) if steps > 0 => {
                let scale = 1.0_f32 / steps as f32;
                let mean = |t: burn::tensor::Tensor<I, 1>| -> Vec<f32> {
                    (t * scale).into_data().into_vec().unwrap()
                };
                (
                    Some(mean(abs)),
                    Some(mean(net)),
                    Some(mean(depth)),
                    Some(mean(area_z)),
                    Some(mean(q)),
                    Some(reach_comids),
                )
            }
            _ => (None, None, None, None, None, None),
        };
```

And extend the `Ok(EvalOutput { ... })` literal with `zeta_depth_mean, zeta_area_z_mean, zeta_q_mean,`.

- [ ] **Step 3: Fix the `zarr_io.rs` test literal**

In the `EvalOutput` literal at `src/training/zarr_io.rs:296`, after the existing `zeta_abs_mean: None, zeta_net_mean: None,` fields add:

```rust
            zeta_depth_mean: None,
            zeta_area_z_mean: None,
            zeta_q_mean: None,
```

- [ ] **Step 4: Extend `write_zeta_netcdf` in `src/dump_parameters.rs`**

Change the signature (line 511):

```rust
pub fn write_zeta_netcdf(
    path: &Path,
    comids: &[i64],
    zeta_abs_mean: &[f32],
    zeta_net_mean: &[f32],
    depth_mean: &[f32],
    area_z_mean: &[f32],
    q_mean: &[f32],
    model_label: &str,
) -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
```

Update its doc comment to mention the three new variables. After the `zeta_net` block (line 573), add three blocks in the same add-or-overwrite style:

```rust
    if let Some(mut v) = file.variable_mut("depth_mean") {
        v.put_values(depth_mean, ..)?;
    } else {
        let mut v = file.add_variable::<f32>("depth_mean", &["COMID_eval"])?;
        v.put_values(depth_mean, ..)?;
        v.put_attribute("long_name", "eval-window mean routed flow depth")?;
        v.put_attribute("units", "m")?;
    }

    if let Some(mut v) = file.variable_mut("area_z_mean") {
        v.put_values(area_z_mean, ..)?;
    } else {
        let mut v = file.add_variable::<f32>("area_z_mean", &["COMID_eval"])?;
        v.put_values(area_z_mean, ..)?;
        v.put_attribute("long_name", "eval-window mean plan-view wetted area (leakance area_z)")?;
        v.put_attribute("units", "m^2")?;
    }

    if let Some(mut v) = file.variable_mut("q_mean") {
        v.put_values(q_mean, ..)?;
    } else {
        let mut v = file.add_variable::<f32>("q_mean", &["COMID_eval"])?;
        v.put_values(q_mean, ..)?;
        v.put_attribute("long_name", "eval-window mean routed discharge")?;
        v.put_attribute("units", "m^3/s")?;
    }
```

- [ ] **Step 5: Update the two call sites**

`src/cli/run.rs` (lines 404–416):

```rust
                if let (Some(za), Some(zn), Some(zd), Some(zaz), Some(zq), Some(zc)) = (
                    &output.zeta_abs_mean,
                    &output.zeta_net_mean,
                    &output.zeta_depth_mean,
                    &output.zeta_area_z_mean,
                    &output.zeta_q_mean,
                    &output.zeta_comids,
                ) {
                    let nc = run_dir.join("kan_parameters.nc");
                    match crate::dump_parameters::write_zeta_netcdf(
                        &nc, zc, za, zn, zd, zaz, zq, &latest.display().to_string(),
                    ) {
                        Ok(()) => eprintln!("zeta diagnostic → {}", nc.display()),
                        Err(e) => eprintln!("warning: zeta netcdf write failed: {e}"),
                    }
                }
```

`src/bin/eval.rs` (lines 131–152) — same shape:

```rust
    if let Some(zpath) = &cli.zeta_output {
        match (
            &output.zeta_abs_mean,
            &output.zeta_net_mean,
            &output.zeta_depth_mean,
            &output.zeta_area_z_mean,
            &output.zeta_q_mean,
            &output.zeta_comids,
        ) {
            (Some(za), Some(zn), Some(zd), Some(zaz), Some(zq), Some(zc)) => {
                ddrs::dump_parameters::write_zeta_netcdf(zpath, zc, za, zn, zd, zaz, zq, &model_label)
                    .map_err(|e| -> Box<dyn std::error::Error> { e })?;
                let frac_above = za.iter().filter(|&&z| z > 0.01).count() as f64 / za.len() as f64;
                let mut sorted = za.clone();
                sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());
                println!(
                    "zeta → {} ({} reaches; median |zeta|={:.4e} m³/s; |zeta|>0.01 on {:.1}%)",
                    zpath.display(),
                    zc.len(),
                    sorted[sorted.len() / 2],
                    frac_above * 100.0,
                );
            }
            _ => eprintln!(
                "warning: --zeta-output requested but no zeta was accumulated \
                 (params.use_leakance off, or --frozen)"
            ),
        }
    }
```

- [ ] **Step 6: Build + tests**

Run: `cargo test --test zeta_accum`
Expected: PASS — all 6 tests (4 updated + 2 new).

Run: `cargo test --lib 2>&1 | tail -3`
Expected: all lib tests pass (172 at last count).

- [ ] **Step 7: Commit**

```bash
git add src/routing/mmc_op.rs src/routing/mmc.rs src/routing/mod.rs \
        src/training/forward.rs src/training/eval.rs src/training/zarr_io.rs \
        src/dump_parameters.rs src/cli/run.rs src/bin/eval.rs tests/zeta_accum.rs
git commit -m "feat(zeta): export per-reach eval-window depth_mean/area_z_mean/q_mean

Extends the leakance zeta accumulator so the low-zeta diagnosis (H1
structural ceiling, H2 driving head, H6 fractional loss) runs on exact
routed quantities instead of geometry proxies. Eval-only; training path
untouched."
```

### Task 4: Gradient/parity guard suite

No code changes — verification only. All four must pass before any GPU run.

- [ ] **Step 1: Leakance guards**

Run: `cargo test --test zeta_accum --test leakance_gradcheck --test leakance_off_parity 2>&1 | grep -E "test result|running"`
Expected: zeta_accum 6 passed; leakance_gradcheck 8 passed; leakance_off_parity 3 passed.

- [ ] **Step 2: DDR parity (the invariant that must never break)**

Run: `cargo run --release --example compare_ddr_sandbox 2>&1 | tail -5`
Expected: `ABSOLUTE MATCH` (max abs diff < 1e-3 m³/s).

### Task 5: Re-run the two ON evals + hourly-OFF dump (GPU, main tree)

**Files:** none (artifacts only). Sequential GPU jobs — do not run two evals at once.

- [ ] **Step 1: Build the release binaries in the worktree**

```bash
cd /home/tbindas/projects/ddrs/.claude/worktrees/leakance-diagnosis
cargo build --release --bin eval --bin dump_parameters
```

Expected: clean build. Binary paths used below:
`WT=/home/tbindas/projects/ddrs/.claude/worktrees/leakance-diagnosis/target/release`

- [ ] **Step 2: Back up both existing kan_parameters.nc files**

```bash
cd /home/tbindas/projects/ddrs
cp .ddrs/runs/2026-07-01T13-43-32Z-train-and-test/kan_parameters.nc /tmp/kanparams_hourly_on.nc.bak
cp .ddrs/runs/2026-07-01T21-20-27Z-train-and-test/kan_parameters.nc /tmp/kanparams_daily_on.nc.bak
```

- [ ] **Step 3: Re-eval hourly-ON (~10 min)**

```bash
cd /home/tbindas/projects/ddrs
$WT/eval --config config/experiments/leakance_hourly_on.yaml \
  --checkpoint .ddrs/runs/2026-07-01T13-43-32Z-train-and-test/checkpoints/epoch_5_mb_9 \
  --output /tmp/diag_eval_hourly_on.zarr \
  --zeta-output .ddrs/runs/2026-07-01T13-43-32Z-train-and-test/kan_parameters.nc
```

Expected stdout: `zeta → ... (64892 reaches; median |zeta|=6.4e-4 ...; |zeta|>0.01 on 10.4%)` — the zeta numbers must reproduce the prior export (same checkpoint, same window); small CUDA-nondeterminism wiggle is fine.

- [ ] **Step 4: Re-eval daily-ON (~10 min)**

```bash
cd /home/tbindas/projects/ddrs
$WT/eval --config config/experiments/leakance_daily_on.yaml \
  --checkpoint .ddrs/runs/2026-07-01T21-20-27Z-train-and-test/checkpoints/epoch_5_mb_9 \
  --output /tmp/diag_eval_daily_on.zarr \
  --zeta-output .ddrs/runs/2026-07-01T21-20-27Z-train-and-test/kan_parameters.nc
```

Expected: `|zeta|>0.01 on ~14.2%`.

- [ ] **Step 5: Dump hourly-OFF KAN params (needed by H5; no eval, CPU-cheap)**

```bash
cd /home/tbindas/projects/ddrs
LATEST=$(ls -v .ddrs/runs/2026-06-23T02-49-12Z-conus-hourly-train-and-test/checkpoints | tail -1)
$WT/dump_parameters \
  --config .ddrs/runs/2026-06-23T02-49-12Z-conus-hourly-train-and-test/config.yaml \
  --checkpoint .ddrs/runs/2026-06-23T02-49-12Z-conus-hourly-train-and-test/checkpoints/$LATEST \
  --output .ddrs/runs/2026-06-23T02-49-12Z-conus-hourly-train-and-test/kan_parameters.nc
```

Expected: netCDF with `n`, `q_spatial`, `p_spatial`, `x_storage`, `slope` on full-CONUS `COMID` (no leakance vars — this run had none).

- [ ] **Step 6: Verify the exports**

```bash
cd ~/projects/ddr && uv run python - <<'EOF'
import xarray as xr
for arm, rid in [("hourly-ON", "2026-07-01T13-43-32Z-train-and-test"),
                 ("daily-ON",  "2026-07-01T21-20-27Z-train-and-test")]:
    ds = xr.open_dataset(f"/home/tbindas/projects/ddrs/.ddrs/runs/{rid}/kan_parameters.nc")
    missing = [v for v in ("zeta", "zeta_net", "depth_mean", "area_z_mean", "q_mean", "K_D") if v not in ds]
    assert not missing, f"{arm}: missing {missing}"
    print(arm, "ok:", float(ds.depth_mean.median()), "m median depth,",
          float(ds.q_mean.median()), "m3/s median q")
EOF
```

Expected: both lines print with positive finite medians and no assertion.

---

## Phase 2 — hypothesis battery

### Task 6: Write `scripts/leakance_diagnosis.py`

**Files:**
- Create: `scripts/leakance_diagnosis.py`

Runs under ddr's venv (`cd ~/projects/ddr && uv run python ...`). Consumes the four run dirs, the attributes netCDF, and the gages CSV; prints one section per hypothesis with a suggested verdict. Complete file:

```python
#!/usr/bin/env python3
"""Leakance low-zeta diagnosis — 7-hypothesis falsification battery.

Spec: ddrs docs/superpowers/specs/2026-07-01-leakance-low-zeta-diagnosis-design.md
Run:  cd ~/projects/ddr && uv run python ~/projects/ddrs/scripts/leakance_diagnosis.py

H1 structural ceiling   H2 driving head      H3 KAN variance collapse
H4 gauge bias           H5 equifinality      H6 fractional loss
H7 model form (connected-only law)
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import xarray as xr

RUNS = Path("/home/tbindas/projects/ddrs/.ddrs/runs")
ARM_IDS = {
    "hourly_on": "2026-07-01T13-43-32Z-train-and-test",
    "daily_on": "2026-07-01T21-20-27Z-train-and-test",
    "hourly_off": "2026-06-23T02-49-12Z-conus-hourly-train-and-test",
    "daily_off": "2026-06-05T01-41-16Z-train-and-test",
}
K_D_CEIL = 1e-6          # current range top (1/s)
K_D_WIDE = 1e-4          # literature sand-bed leakance (litreview §A1)
D_GW_FLOOR = -2.0        # current d_gw range bottom (m)
ZETA_BAR = 0.01          # GO/NO-GO magnitude bar (m3/s)
ATTRS = ["aridity", "permeability", "Porosity", "log10_uparea", "meanP", "meanslope"]


def sec(title: str) -> None:
    print(f"\n{'=' * 72}\n{title}\n{'=' * 72}")


def q(x: np.ndarray, ps=(5, 25, 50, 75, 95)) -> str:
    return " ".join(f"p{p}={np.nanpercentile(x, p):.4g}" for p in ps)


def spearman(a: np.ndarray, b: np.ndarray) -> float:
    m = np.isfinite(a) & np.isfinite(b)
    if m.sum() < 10:
        return float("nan")
    from scipy.stats import spearmanr

    return float(spearmanr(a[m], b[m]).statistic)


class Arm:
    """One run's kan_parameters.nc: full-CONUS params + eval-network diagnostics."""

    def __init__(self, run_dir: Path):
        self.ds = xr.open_dataset(run_dir / "kan_parameters.nc")
        self.comid = self.ds["COMID"].values.astype(np.int64)
        # Map eval-network reaches into the full-CONUS param vectors.
        if "COMID_eval" in self.ds:
            self.comid_eval = self.ds["COMID_eval"].values.astype(np.int64)
            order = np.argsort(self.comid)
            pos = np.searchsorted(self.comid, self.comid_eval, sorter=order)
            self.eval_ix = order[pos]
            assert (self.comid[self.eval_ix] == self.comid_eval).all(), "COMID_eval not a subset of COMID"

    def on_eval(self, var: str) -> np.ndarray:
        """A full-CONUS variable subset to the eval network, or an eval-native one."""
        v = self.ds[var]
        return v.values[self.eval_ix] if v.dims == ("COMID",) else v.values


def attach_attributes(attrs_path: Path, comids: np.ndarray) -> dict[str, np.ndarray]:
    ds = xr.open_dataset(attrs_path)
    acom = ds["COMID"].values.astype(np.int64)
    order = np.argsort(acom)
    pos = np.searchsorted(acom, comids, sorter=order)
    ix = order[np.clip(pos, 0, len(acom) - 1)]
    ok = acom[ix] == comids
    out = {}
    for name in ATTRS:
        v = ds[name].values.astype(np.float64)[ix]
        v[~ok] = np.nan
        out[name] = v
    print(f"attributes matched for {ok.mean() * 100:.1f}% of {len(comids)} reaches")
    # Orient aridity: if it anti-correlates with meanP it is a dryness index.
    r = spearman(out["aridity"], out["meanP"])
    print(f"aridity vs meanP spearman = {r:.2f} → aridity is a "
          f"{'DRYNESS' if r < 0 else 'WETNESS'} index")
    out["_aridity_is_dryness"] = np.array([r < 0])
    return out


def verdict(name: str, supported: bool | None, detail: str) -> str:
    tag = "INCONCLUSIVE" if supported is None else ("SUPPORTED" if supported else "REFUTED")
    line = f"[{tag}] {name}: {detail}"
    print(f"\n  → {line}")
    return line


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs-dir", type=Path, default=RUNS)
    ap.add_argument("--attributes", type=Path,
                    default=Path("/home/tbindas/projects/ddr/data/merit_global_attributes_v2.nc"))
    ap.add_argument("--gages-csv", type=Path,
                    default=Path("/home/tbindas/projects/ddr/references/gage_info/gages_3000.csv"))
    args = ap.parse_args()

    arms = {k: Arm(args.runs_dir / rid) for k, rid in ARM_IDS.items()}
    hon = arms["hourly_on"]
    verdicts: list[str] = []

    zeta = hon.on_eval("zeta")
    depth = hon.on_eval("depth_mean")
    area_z = hon.on_eval("area_z_mean")
    q_mean = hon.on_eval("q_mean")
    k_d = hon.on_eval("K_D")
    d_gw = hon.on_eval("d_gw")
    factor = hon.on_eval("leakance_factor")
    attrs = attach_attributes(args.attributes, hon.comid_eval)

    # ---------------- H1: structural ceiling ----------------
    sec("H1 — structural ceiling: can zeta exceed the bar inside the current box?")
    zmax_now = 1.0 * area_z * K_D_CEIL * (depth - D_GW_FLOOR)
    zmax_wide = 1.0 * area_z * K_D_WIDE * (depth - D_GW_FLOOR)
    frac_now = float((zmax_now > ZETA_BAR).mean())
    frac_wide = float((zmax_wide > ZETA_BAR).mean())
    util = zeta / np.maximum(zmax_now, 1e-30)
    print(f"zeta_max within CURRENT box:  {q(zmax_now)} | frac > {ZETA_BAR}: {frac_now * 100:.1f}%")
    print(f"zeta_max with K_D={K_D_WIDE}: {q(zmax_wide)} | frac > {ZETA_BAR}: {frac_wide * 100:.1f}%")
    print(f"utilization zeta/zeta_max:    {q(util)}")
    verdicts.append(verdict(
        "H1 structural ceiling",
        frac_now < 0.5,
        f"only {frac_now * 100:.1f}% of reaches CAN exceed {ZETA_BAR} m³/s inside the current box "
        f"(vs {frac_wide * 100:.1f}% at K_D={K_D_WIDE}); median utilization {np.median(util):.2f}",
    ))

    # ---------------- H2: driving-head starvation ----------------
    sec("H2 — driving head (depth_mean − d_gw)")
    head = depth - d_gw
    print(f"depth_mean: {q(depth)}  |  d_gw: {q(d_gw)}  |  head: {q(head)}")
    f_neg, f_small = float((head <= 0).mean()), float((head < 0.1).mean())
    print(f"head ≤ 0 (gaining/neutral at the mean): {f_neg * 100:.1f}%   head < 0.1 m: {f_small * 100:.1f}%")
    verdicts.append(verdict(
        "H2 driving-head starvation",
        f_small > 0.5,
        f"{f_small * 100:.1f}% of reaches have <0.1 m mean driving head ({f_neg * 100:.1f}% ≤ 0)",
    ))

    # ---------------- H3: KAN variance collapse ----------------
    sec("H3 — KAN variance collapse (leakance vs routing params, full CONUS)")
    rows = {}
    for name in ["K_D", "d_gw", "leakance_factor", "n", "q_spatial", "x_storage"]:
        v = hon.ds[name].values.astype(np.float64)
        v = np.log10(v) if name == "K_D" else v
        iqr = np.nanpercentile(v, 75) - np.nanpercentile(v, 25)
        rng = np.nanpercentile(v, 95) - np.nanpercentile(v, 5)
        rows[name] = (np.nanmedian(v), iqr, iqr / max(rng, 1e-30))
        print(f"{name:16s} median={rows[name][0]:9.4g}  IQR={iqr:9.4g}  IQR/p5-p95={rows[name][2]:.3f}")
    print("\nspearman |corr| of leakance params vs attributes (eval network):")
    max_r = 0.0
    for pname, pv in [("K_D", np.log10(k_d)), ("d_gw", d_gw), ("factor", factor), ("zeta", np.log10(np.maximum(zeta, 1e-12)))]:
        rs = {a: spearman(pv, attrs[a]) for a in ATTRS}
        max_r = max(max_r, max(abs(r) for r in rs.values() if np.isfinite(r)))
        print(f"  {pname:8s} " + "  ".join(f"{a}={r:+.2f}" for a, r in rs.items()))
    verdicts.append(verdict(
        "H3 KAN variance collapse",
        max_r < 0.2,
        f"max |spearman| of any leakance param vs any attribute = {max_r:.2f} "
        "(<0.2 ⇒ no learned spatial structure; ≥0.4 ⇒ clearly attribute-driven)",
    ))

    # ---------------- H4: gauge bias / gradient starvation ----------------
    sec("H4 — stratification by gauged-ness, upstream area, aridity")
    import csv

    with open(args.gages_csv) as f:
        gauged_comids = {int(row["COMID"]) for row in csv.DictReader(f)}
    gmask = np.isin(hon.comid_eval, list(gauged_comids))
    print(f"gauged reaches on eval network: {gmask.sum()} / {len(gmask)}")
    for label, m in [("gauged", gmask), ("ungauged", ~gmask)]:
        print(f"  {label:9s} median|zeta|={np.median(zeta[m]):.3e}  frac>{ZETA_BAR}: "
              f"{(zeta[m] > ZETA_BAR).mean() * 100:.1f}%  median q={np.median(q_mean[m]):.3g}")
    arid = attrs["aridity"]
    dry = arid >= np.nanpercentile(arid, 67) if attrs["_aridity_is_dryness"][0] else arid <= np.nanpercentile(arid, 33)
    wet = ~dry & np.isfinite(arid)
    print(f"  dry tercile  median|zeta|={np.median(zeta[dry]):.3e}  frac>{ZETA_BAR}: {(zeta[dry] > ZETA_BAR).mean() * 100:.1f}%")
    print(f"  wet tercile  median|zeta|={np.median(zeta[wet]):.3e}  frac>{ZETA_BAR}: {(zeta[wet] > ZETA_BAR).mean() * 100:.1f}%")
    r_up = spearman(np.log10(np.maximum(zeta, 1e-12)), attrs["log10_uparea"])
    print(f"  spearman log|zeta| vs log10_uparea = {r_up:+.2f}")
    # If zeta is big only where q is big (uparea-driven) and shows no dry/wet
    # contrast, the term follows the discharge signal, not losing-ness.
    dry_wet_ratio = np.median(zeta[dry]) / max(np.median(zeta[wet]), 1e-30)
    verdicts.append(verdict(
        "H4 gauge bias / gradient starvation",
        dry_wet_ratio < 2.0 and r_up > 0.5,
        f"dry/wet median-zeta ratio = {dry_wet_ratio:.2f} (physics says dry ≫ wet), "
        f"zeta–uparea corr {r_up:+.2f} (zeta tracks river size, not aridity)",
    ))

    # ---------------- H5: equifinality with routing params ----------------
    sec("H5 — did n / x_storage shift between paired ON/OFF runs?")
    any_shift = False
    for pair, on_key, off_key in [("hourly", "hourly_on", "hourly_off"), ("daily", "daily_on", "daily_off")]:
        on, off = arms[on_key], arms[off_key]
        assert (on.comid == off.comid).all(), f"{pair}: COMID order mismatch"
        for pname in ["n", "x_storage"]:
            a, b = on.ds[pname].values, off.ds[pname].values
            d = a - b
            shift = abs(np.median(d)) / max(np.nanpercentile(np.abs(b - np.median(b)), 75), 1e-30)
            any_shift |= shift > 0.5
            print(f"  {pair:6s} Δ{pname:9s} median={np.median(d):+.4g}  IQR={np.percentile(d, 75) - np.percentile(d, 25):.4g}  "
                  f"median-shift/param-IQR={shift:.2f}")
    verdicts.append(verdict(
        "H5 equifinality",
        any_shift,
        "routing params shifted materially between ON/OFF (shift > 0.5 IQR) — "
        "n/storage absorb what leakance would explain" if any_shift else
        "routing params essentially unchanged between ON/OFF pairs",
    ))

    # ---------------- H6: fractional loss ----------------
    sec("H6 — |zeta| / q_mean (is the loss non-trivial RELATIVE to local flow?)")
    frac_loss = zeta / np.maximum(q_mean, 1e-4)
    print(f"|zeta|/q: {q(frac_loss)}")
    f1, f5 = float((frac_loss > 0.01).mean()), float((frac_loss > 0.05).mean())
    print(f"frac loss > 1% of local flow: {f1 * 100:.1f}%   > 5% (gauge-detectability band): {f5 * 100:.1f}%")
    verdicts.append(verdict(
        "H6 wrong yardstick",
        f1 > 0.3,
        f"{f1 * 100:.1f}% of reaches lose >1% of local flow ({f5 * 100:.1f}% >5%) — "
        "the absolute 0.01 m³/s bar under/over-states the term's activity",
    ))

    # ---------------- H7: model form (connected-only law) ----------------
    sec("H7 — d_gw boundary-pinning where disconnection is plausible (dry reaches)")
    lo, hi = D_GW_FLOOR, 2.0
    pin_hi = (d_gw > hi - 0.05 * (hi - lo))
    pin_lo = (d_gw < lo + 0.05 * (hi - lo))
    print(f"d_gw within 5% of bounds: floor {pin_lo.mean() * 100:.1f}%  ceiling {pin_hi.mean() * 100:.1f}% (overall)")
    print(f"  dry tercile: floor {pin_lo[dry].mean() * 100:.1f}%  ceiling {pin_hi[dry].mean() * 100:.1f}%")
    print(f"  wet tercile: floor {pin_lo[wet].mean() * 100:.1f}%  ceiling {pin_hi[wet].mean() * 100:.1f}%")
    dry_pin = float((pin_lo | pin_hi)[dry].mean())
    verdicts.append(verdict(
        "H7 model-form error",
        dry_pin > 0.3,
        f"{dry_pin * 100:.1f}% of dry-tercile reaches pin d_gw at a bound — the linear "
        "connected-regime law is straining toward the saturating (disconnected) regime",
    ))

    # ---------------- summary ----------------
    sec("SUMMARY (suggested verdicts — final judgment in the findings doc)")
    for v in verdicts:
        print(f"  {v}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 1: Write the file exactly as above**

- [ ] **Step 2: Smoke-run it**

Run: `cd ~/projects/ddr && uv run python ~/projects/ddrs/.claude/worktrees/leakance-diagnosis/scripts/leakance_diagnosis.py 2>&1 | tail -40`
Expected: all 7 sections print with finite numbers; SUMMARY lists 7 verdict lines. If an attribute/variable name KeyErrors, print `list(ds.variables)` for the offending file, fix the name in the script, and note the correction.

- [ ] **Step 3: Commit**

```bash
git add scripts/leakance_diagnosis.py
git commit -m "feat(scripts): leakance low-zeta 7-hypothesis diagnosis battery"
```

### Task 7: Run the battery and capture output

- [ ] **Step 1: Full run, captured**

```bash
cd ~/projects/ddr && uv run python \
  ~/projects/ddrs/.claude/worktrees/leakance-diagnosis/scripts/leakance_diagnosis.py \
  | tee /tmp/leakance_diagnosis_output.txt
```

Expected: exit 0; `/tmp/leakance_diagnosis_output.txt` holds the full report.

### Task 8: Findings document

**Files:**
- Create: `docs/2026-07-02-leakance-diagnosis-findings.md`

- [ ] **Step 1: Write the findings doc from the captured output**

Structure (fill every `<...>` from `/tmp/leakance_diagnosis_output.txt` — no placeholders may survive):

```markdown
# Leakance low-zeta diagnosis — findings (2026-07-02)

Spec: docs/superpowers/specs/2026-07-01-leakance-low-zeta-diagnosis-design.md
Lit review: docs/2026-07-01-leakance-litreview.md
Script: scripts/leakance_diagnosis.py (output archived below)

**One-line answer:** <why is zeta small — the top-ranked supported hypothesis(es)>

## Ranked verdicts

| # | Hypothesis | Verdict | Key number |
|---|---|---|---|
| H1 | structural ceiling | <SUPPORTED/REFUTED> | <frac of reaches that CAN exceed 0.01 in-box> |
| H2 | driving-head starvation | <> | <frac head < 0.1 m> |
| H3 | KAN variance collapse | <> | <max attr |spearman|> |
| H4 | gauge bias / gradient starvation | <> | <dry/wet zeta ratio; uparea corr> |
| H5 | equifinality | <> | <max median-shift/IQR> |
| H6 | wrong yardstick | <> | <frac losing >1% of flow> |
| H7 | model-form error | <> | <dry-tercile d_gw pinning> |

## Interpretation
<3–6 paragraphs tying the verdicts to the lit review's ranked explanations
(gauge bias #1, equifinality #2, detection limit #3, clipping #5) and to the
user's original hypothesis (H3). State explicitly which literature prediction
held and which didn't.>

## Phase-3 gate decision
Gate: H1 SUPPORTED and gradient alive (H3 REFUTED or H4 shows stratification).
Decision: <GO for widened-K_D retrain / NO-GO with recommended alternative>.

## Raw script output
<paste /tmp/leakance_diagnosis_output.txt in a fenced block>
```

- [ ] **Step 2: Commit**

```bash
git add docs/2026-07-02-leakance-diagnosis-findings.md
git commit -m "docs: leakance low-zeta diagnosis findings + phase-3 gate decision"
```

---

## Phase 3 — gated fix (run ONLY if the Task-8 gate says GO)

### Task 9: Widened-K_D retrain

**Files:**
- Create: `config/experiments/leakance_hourly_on_kd4.yaml`

- [ ] **Step 1: Clone the config with the widened range**

```bash
cd /home/tbindas/projects/ddrs/.claude/worktrees/leakance-diagnosis
sed -e 's|K_D: \[1.0e-8, 1.0e-6\]|K_D: [1.0e-8, 1.0e-4]|' \
    config/experiments/leakance_hourly_on.yaml > config/experiments/leakance_hourly_on_kd4.yaml
```

Then edit the header comment block of the new file to say: widened `K_D` upper
bound `1e-6 → 1e-4` per the literature (Calver 2001 pooled streambed K; ddrs
litreview §A1 — sand-bed leakance regime), testing whether the ceiling clipped
zeta magnitude and the losing-subset skill delta. Verify the sed took:

```bash
grep -n "K_D:" config/experiments/leakance_hourly_on_kd4.yaml
```

Expected: `K_D: [1.0e-8, 1.0e-4]` in `parameter_ranges` (the `log_space_parameters` entry stays).

- [ ] **Step 2: Commit the config**

```bash
git add config/experiments/leakance_hourly_on_kd4.yaml
git commit -m "experiment: leakance hourly arm with literature-widened K_D ceiling (1e-4)"
```

- [ ] **Step 3: Launch the retrain (hours; background; main tree workspace, worktree binary)**

```bash
cd /home/tbindas/projects/ddrs
WT=/home/tbindas/projects/ddrs/.claude/worktrees/leakance-diagnosis
(cd $WT && cargo build --release --bin ddrs)
$WT/target/release/ddrs --config $WT/config/experiments/leakance_hourly_on_kd4.yaml \
  plan --workflow train-and-test
$WT/target/release/ddrs --config $WT/config/experiments/leakance_hourly_on_kd4.yaml \
  run --workflow train-and-test
```

NEVER use the installed `~/.cargo/bin/ddrs` here (stale-binary trap — see CLAUDE.md). Note the new run id printed at start.

- [ ] **Step 4: Verify identifiability + magnitude on the new run**

```bash
cd ~/projects/ddr && uv run python - <<'EOF'
import sys, numpy as np, xarray as xr
rid = sys.argv[1] if len(sys.argv) > 1 else input("run id: ")
ds = xr.open_dataset(f"/home/tbindas/projects/ddrs/.ddrs/runs/{rid}/kan_parameters.nc")
kd = ds.K_D.values
print("K_D median", np.median(kd), "| frac@new-ceiling(1e-4):", (kd > 0.95e-4).mean(),
      "| frac@floor:", (kd < 1.2e-8).mean())
z = np.abs(ds.zeta.values)
print("median |zeta|", np.median(z), "| frac>0.01:", (z > 0.01).mean())
EOF
```

Expected: K_D interior (not re-pinned at 1e-4) ⇒ the true optimum was inside the widened range; frac>0.01 materially above 10.4% if H1 was the binding limiter.

- [ ] **Step 5: Subset analysis (the 2×2 delta with the new arm)**

```bash
cd ~/projects/ddr && uv run python ~/projects/ddrs/scripts/leakance_subset_analysis.py \
  --hourly-on  <NEW-RUN-ID> \
  --daily-on   2026-07-01T21-20-27Z-train-and-test \
  --hourly-off 2026-06-23T02-49-12Z-conus-hourly-train-and-test \
  --daily-off  2026-06-05T01-41-16Z-train-and-test \
  --ddrs-runs-dir /home/tbindas/projects/ddrs/.ddrs/runs
```

- [ ] **Step 6: Update the findings doc + commit**

Append a "Phase 3 — widened K_D result" section to
`docs/2026-07-02-leakance-diagnosis-findings.md` with: new K_D distribution,
new zeta stats, subset ΔNSE/ΔKGE vs the 1e-6 arm, and the refreshed GO/NO-GO.

```bash
git add docs/2026-07-02-leakance-diagnosis-findings.md
git commit -m "docs(findings): widened-K_D arm result + refreshed GO/NO-GO"
```

---

## Execution notes for sub-agents

- Rust work: this worktree only. GPU/eval work: `cd /home/tbindas/projects/ddrs` with the worktree's absolute binary path. Python: always `cd ~/projects/ddr && uv run python ...`.
- GPU jobs are serialized — one eval or train at a time.
- If any guard in Task 4 fails, STOP and report; do not proceed to Task 5.
- The netCDF appends in Task 5 target files that already contain full-CONUS dump variables; the length-mismatch guard in `write_zeta_netcdf` will error rather than corrupt — if it errors, restore from the `/tmp/*.bak` backups and investigate.
```

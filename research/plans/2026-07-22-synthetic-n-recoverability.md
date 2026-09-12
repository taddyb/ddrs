# Synthetic n-Recoverability Across Real Q' Sources Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a ground-truth-anchored synthetic-twin control that tests whether learned Manning's n absorbs Q'-source bias while channel geometry (q_spatial/p_spatial) does not, using this campaign's 4 real Q' stores as the forcing.

**Architecture:** Extend the existing `probe_zeta_gradient --mode teacher` driver (already builds synthetic gauge observations from a chunked continuous forward pass) to also accept an n/q_spatial/p_spatial donor override — reusing the exact `load_comid_field`/`gather_by_comid`/`physical_to_normalized`/`RoutingParamOverride` machinery `--mode eval-loss`'s "full-swap" composition already uses. Generate a truth donor NetCDF (prescribed n + consensus geometry from this campaign's 4 already-converged checkpoints), route the standard benchmark Q' store through it to get noise-free synthetic gauge observations, then train 4 fresh students — one per real campaign Q' store — against those synthetic observations. Compare each student's recovered n/q_spatial/p_spatial (via the existing `dump_parameters` binary) against the known truth.

**Tech Stack:** Rust (BURN 0.21), Python (`ddrs-py` venv: xarray, netCDF4, numpy, pandas), existing `ddrs` CLI binaries (`probe_zeta_gradient`, `dump_parameters`, `train`).

Spec: `docs/superpowers/specs/2026-07-22-synthetic-n-recoverability-design.md`

---

## File Structure

| File | Responsibility |
|---|---|
| `src/bin/probe_zeta_gradient.rs` (modify `run_teacher`, `Cli`, module doc, the eval-loss-only CLI guard) | Teacher mode gains an optional `--donor-params-nc` n/q/p override; `--plant-file`/`--zeta-output`/`use_leakance` become independent of it, not required |
| `tests/teacher_donor_override_parity.rs` (new) | Parity gate: injecting a checkpoint's own dump as a full n/q/p donor must reproduce that checkpoint's un-overridden synthetic obs |
| `scripts/synthetic_n_consensus_geometry.py` (new) | Runs `dump_parameters` against the 4 real campaign checkpoints, computes per-COMID median q_spatial/p_spatial |
| `scripts/synthetic_n_truth_fields.py` (new) | Computes the Leopold-Maddock and Gaussian-noise truth-n fields from `log10_uparea`; combines with consensus geometry into two donor NetCDFs |
| `config/experiments/synthetic_n_teacher.yaml` (new) | Teacher config: standard benchmark Q' store, full 1981-2010 window |
| `config/experiments/synthetic_n_student_{distributed,lumped,daily_lstm,hourly_lstm}.yaml` (new, 4 files) | Student configs: exact copies of this campaign's own 4 arms, observations repointed to the synthetic obs store |
| `scripts/synthetic_n_recoverability_analysis.py` (new) | Computes S1-S5 pre-registered verdicts from the 4 students' `dump_parameters` output vs the truth NetCDF |

No changes to `src/routing/`, `src/geometry.rs`, `src/sparse.rs`, or any `Backward` impl.

---

### Task 1: Extend `probe_zeta_gradient --mode teacher` with an optional n/q/p donor override

**Files:**
- Modify: `src/bin/probe_zeta_gradient.rs:35-42` (module doc, Stage 3 section)
- Modify: `src/bin/probe_zeta_gradient.rs:204,208,212,228-234` (Cli struct doc comments)
- Modify: `src/bin/probe_zeta_gradient.rs:337-351` (donor-flag mode guard)
- Modify: `src/bin/probe_zeta_gradient.rs:1058-1150` (`run_teacher` body)
- Test: `tests/teacher_donor_override_parity.rs`

- [ ] **Step 1: Update the Stage 3 module doc comment**

Replace lines 35-42:

```rust
//! Stage 3 (`--mode teacher`): planted-leakance world — overrides the KAN
//! head's normalized leakance outputs at specified reaches with values from a
//! CSV, then runs the chunked eval loop and writes (a) synthetic daily gauge
//! observations as a zarr-v2 store and (b) a per-reach zeta answer key netCDF.
//! `--output` is not used in teacher mode.
//!   cargo run --release --bin probe_zeta_gradient -- \
//!       --mode teacher \
//!       --config config/experiments/leakance_hourly_on.yaml \
//!       --checkpoint .ddrs/runs/<id>/checkpoints/epoch_5_mb_9 \
//!       --eval-days 1095 \
//!       --plant-file output/plant_sites.csv \
//!       --obs-output output/teacher_obs/ \
//!       --zeta-output output/teacher_zeta.nc
```

with:

```rust
//! Stage 3 (`--mode teacher`): synthetic-twin ground-truth generator — runs
//! the chunked eval loop and writes synthetic daily gauge observations as a
//! zarr-v2 store. Two INDEPENDENT, orthogonal overrides may be combined or
//! used alone:
//!   (a) planted-leakance world (`--plant-file` + `--zeta-output`) — overrides
//!       the KAN head's normalized leakance outputs at specified reaches with
//!       values from a CSV; also writes a per-reach zeta answer-key netCDF.
//!       Requires `params.use_leakance: true`.
//!   (b) routing-parameter donor world (`--donor-params-nc`, docs:
//!       docs/superpowers/specs/2026-07-22-synthetic-n-recoverability-design.md)
//!       — overrides ALL THREE of n/q_spatial/p_spatial from a
//!       `dump_parameters::write_netcdf`-schema donor NetCDF (the same
//!       mechanism `--mode eval-loss`'s "full-swap" composition uses).
//! `--output` is not used in teacher mode.
//!
//! Leakance-only (original usage, unchanged):
//!   cargo run --release --bin probe_zeta_gradient -- \
//!       --mode teacher \
//!       --config config/experiments/leakance_hourly_on.yaml \
//!       --checkpoint .ddrs/runs/<id>/checkpoints/epoch_5_mb_9 \
//!       --eval-days 1095 \
//!       --plant-file output/plant_sites.csv \
//!       --obs-output output/teacher_obs/ \
//!       --zeta-output output/teacher_zeta.nc
//!
//! Routing-parameter donor only (no leakance, no --plant-file/--zeta-output):
//!   cargo run --release --bin probe_zeta_gradient -- \
//!       --mode teacher --backend cpu \
//!       --config config/experiments/synthetic_n_teacher.yaml \
//!       --checkpoint .ddrs/runs/2026-07-16T02-22-14Z-train-and-test/checkpoints/epoch_5_mb_35 \
//!       --eval-days 999999 \
//!       --donor-params-nc output/synthetic_n/truth_leopold_maddock.nc \
//!       --obs-output output/synthetic_n/synthetic_obs/
```

- [ ] **Step 2: Broaden the `--plant-file`/`--obs-output`/`--zeta-output`/`--donor-params-nc` doc comments**

Replace (around line 202-212):

```rust
    /// teacher mode: plant CSV (comid,k_d_norm,d_gw_norm,factor_norm,...).
    #[arg(long)]
    plant_file: Option<PathBuf>,

    /// teacher mode: directory for the synthetic-obs zarr-v2 store.
    #[arg(long)]
    obs_output: Option<PathBuf>,

    /// teacher mode: answer-key netCDF (zeta accumulation over the window).
    #[arg(long)]
    zeta_output: Option<PathBuf>,
```

with:

```rust
    /// teacher mode: plant CSV (comid,k_d_norm,d_gw_norm,factor_norm,...).
    /// Optional — omit together with --zeta-output when only overriding
    /// n/q_spatial/p_spatial via --donor-params-nc (no leakance planting).
    #[arg(long)]
    plant_file: Option<PathBuf>,

    /// teacher mode: directory for the synthetic-obs zarr-v2 store. Always
    /// required in teacher mode regardless of which override(s) are active.
    #[arg(long)]
    obs_output: Option<PathBuf>,

    /// teacher mode: answer-key netCDF (zeta accumulation over the window).
    /// Required IFF --plant-file is given (leakance-planting world only);
    /// omit both together for a routing-parameter-donor-only teacher run.
    #[arg(long)]
    zeta_output: Option<PathBuf>,
```

Replace the `donor_params_nc` doc comment (around line 228-234):

```rust
    /// eval-loss mode: donor NetCDF (dump_parameters::write_netcdf schema,
    /// COMID-keyed, physical units) supplying n/q_spatial/p_spatial for any
    /// composition other than "own". Required unless --compositions is "own"
    /// only.
    #[arg(long)]
    donor_params_nc: Option<PathBuf>,
```

with:

```rust
    /// eval-loss mode: donor NetCDF (dump_parameters::write_netcdf schema,
    /// COMID-keyed, physical units) supplying n/q_spatial/p_spatial for any
    /// composition other than "own". Required unless --compositions is "own"
    /// only.
    ///
    /// teacher mode: same donor-NetCDF schema, but ALL THREE of
    /// n/q_spatial/p_spatial are always overridden together (no partial
    /// swap) — the synthetic-twin ground-truth generator (see
    /// docs/superpowers/specs/2026-07-22-synthetic-n-recoverability-design.md).
    /// Optional; independent of --plant-file/--zeta-output.
    #[arg(long)]
    donor_params_nc: Option<PathBuf>,
```

- [ ] **Step 3: Allow `--donor-params-nc` in teacher mode**

Modify the guard at lines 337-351 from:

```rust
    // --donor-params-nc/--compositions/--loss-output/--per-gauge-output are
    // only valid in eval-loss mode.
    if (cli.donor_params_nc.is_some()
        || cli.compositions.is_some()
        || cli.loss_output.is_some()
        || cli.per_gauge_output.is_some())
        && mode != Mode::EvalLoss
    {
        return Err(format!(
            "--donor-params-nc/--compositions/--loss-output/--per-gauge-output are only \
             valid in --mode eval-loss (got --mode {})",
            cli.mode
        )
        .into());
    }
```

to:

```rust
    // --compositions/--loss-output/--per-gauge-output are only valid in
    // eval-loss mode. --donor-params-nc is ALSO valid in teacher mode (the
    // synthetic-n routing-parameter donor override).
    if (cli.compositions.is_some() || cli.loss_output.is_some() || cli.per_gauge_output.is_some())
        && mode != Mode::EvalLoss
    {
        return Err(format!(
            "--compositions/--loss-output/--per-gauge-output are only \
             valid in --mode eval-loss (got --mode {})",
            cli.mode
        )
        .into());
    }
    if cli.donor_params_nc.is_some() && mode != Mode::EvalLoss && mode != Mode::Teacher {
        return Err(format!(
            "--donor-params-nc is only valid in --mode eval-loss or --mode teacher (got --mode {})",
            cli.mode
        )
        .into());
    }
```

- [ ] **Step 4: Rewrite `run_teacher`'s setup section to make leakance optional**

Replace lines 1058-1090 (function signature through `checkpoint` binding) from:

```rust
fn run_teacher<I: Backend>(
    cfg: Config,
    cli: Cli,
    device: I::Device,
) -> Result<(), Box<dyn std::error::Error>> {
    // Large chunk size reduces disagg boundary-artifact density (left-clamp at chunk
    // day 0 and precip right-clamp at chunk day C-1). With C=365 each artifact appears
    // only ~14 times over 5115 teacher days (0.82%), vs 341 times with C=15 (20%).
    // 70 GB RAM easily holds a 365-day AORC precip chunk (~2.3 GB).
    const BATCH_SIZE_DAYS: usize = 365;
    assert!(cfg.params.use_leakance, "teacher requires params.use_leakance: true");

    let plants = parse_plant_file(
        cli.plant_file.as_ref().ok_or("--plant-file is required in teacher mode")?,
    )?;
    let obs_dir = cli.obs_output.as_ref().ok_or("--obs-output is required in teacher mode")?;
    if obs_dir.exists() && obs_dir.read_dir()?.next().is_some() {
        return Err(format!(
            "--obs-output {} already exists and is non-empty; remove it before \
             re-running teacher mode to prevent stale gauge data",
            obs_dir.display()
        )
        .into());
    }
    let zeta_path = cli.zeta_output.as_ref().ok_or("--zeta-output is required in teacher mode")?;
    if let Some(p) = zeta_path.parent() {
        if !p.as_os_str().is_empty() && !p.exists() {
            return Err(
                format!("--zeta-output parent dir does not exist: {}", p.display()).into(),
            );
        }
    }
    let checkpoint = cli.checkpoint.as_ref().ok_or("--checkpoint is required in teacher mode")?;
```

with:

```rust
fn run_teacher<I: Backend>(
    cfg: Config,
    cli: Cli,
    device: I::Device,
) -> Result<(), Box<dyn std::error::Error>> {
    // Large chunk size reduces disagg boundary-artifact density (left-clamp at chunk
    // day 0 and precip right-clamp at chunk day C-1). With C=365 each artifact appears
    // only ~14 times over 5115 teacher days (0.82%), vs 341 times with C=15 (20%).
    // 70 GB RAM easily holds a 365-day AORC precip chunk (~2.3 GB).
    const BATCH_SIZE_DAYS: usize = 365;

    let plants = match &cli.plant_file {
        Some(p) => parse_plant_file(p)?,
        None => Vec::new(),
    };
    let leakance_active = !plants.is_empty();
    if leakance_active {
        assert!(
            cfg.params.use_leakance,
            "teacher requires params.use_leakance: true when --plant-file is given"
        );
    }

    let obs_dir = cli.obs_output.as_ref().ok_or("--obs-output is required in teacher mode")?;
    if obs_dir.exists() && obs_dir.read_dir()?.next().is_some() {
        return Err(format!(
            "--obs-output {} already exists and is non-empty; remove it before \
             re-running teacher mode to prevent stale gauge data",
            obs_dir.display()
        )
        .into());
    }
    let zeta_path: Option<PathBuf> = if leakance_active {
        let p = cli
            .zeta_output
            .as_ref()
            .ok_or("--zeta-output is required in teacher mode when --plant-file is given")?;
        if let Some(parent) = p.parent() {
            if !parent.as_os_str().is_empty() && !parent.exists() {
                return Err(
                    format!("--zeta-output parent dir does not exist: {}", parent.display())
                        .into(),
                );
            }
        }
        Some(p.clone())
    } else {
        None
    };
    let checkpoint = cli.checkpoint.as_ref().ok_or("--checkpoint is required in teacher mode")?;
```

- [ ] **Step 5: Build the leakance override and the new param override conditionally**

Find this block (a few lines after the network/plant-coverage checks, right before the chunked-forward loop):

```rust
    // Dense override vectors over the network's reach columns.
    let comid_col: HashMap<i64, usize> =
        network_comids.iter().enumerate().map(|(i, &c)| (c, i)).collect();
    let n_reaches = network_comids.len();
    let mut ov = LeakanceOverride {
        mask: vec![0.0; n_reaches],
        k_d: vec![0.0; n_reaches],
        d_gw: vec![0.0; n_reaches],
        factor: vec![0.0; n_reaches],
    };
    for &(comid, k, d, f) in &plants {
        let col = comid_col[&comid];
        ov.mask[col] = 1.0;
        ov.k_d[col] = k;
        ov.d_gw[col] = d;
        ov.factor[col] = f;
    }
```

Replace with:

```rust
    // Dense override vectors over the network's reach columns.
    let comid_col: HashMap<i64, usize> =
        network_comids.iter().enumerate().map(|(i, &c)| (c, i)).collect();
    let n_reaches = network_comids.len();

    let leakance_ov: Option<LeakanceOverride> = if leakance_active {
        let mut ov = LeakanceOverride {
            mask: vec![0.0; n_reaches],
            k_d: vec![0.0; n_reaches],
            d_gw: vec![0.0; n_reaches],
            factor: vec![0.0; n_reaches],
        };
        for &(comid, k, d, f) in &plants {
            let col = comid_col[&comid];
            ov.mask[col] = 1.0;
            ov.k_d[col] = k;
            ov.d_gw[col] = d;
            ov.factor[col] = f;
        }
        Some(ov)
    } else {
        None
    };

    // Optional n/q_spatial/p_spatial donor override (the synthetic-n
    // routing-parameter twin — docs/superpowers/specs/2026-07-22-synthetic-n-
    // recoverability-design.md). Reuses the same --donor-params-nc /
    // load_comid_field / gather_by_comid / physical_to_normalized machinery
    // as --mode eval-loss's full-swap composition.
    let param_ov: Option<RoutingParamOverride> = match &cli.donor_params_nc {
        Some(donor_path) => {
            let log_space =
                |name: &str| cfg.params.log_space_parameters.iter().any(|s| s == name);
            let n_map = ddrs::data::load_comid_field(donor_path, "n")?;
            let q_map = ddrs::data::load_comid_field(donor_path, "q_spatial")?;
            let p_map = ddrs::data::load_comid_field(donor_path, "p_spatial")?;
            let n_vals = gather_by_comid(&n_map, &network_comids)?;
            let q_vals = gather_by_comid(&q_map, &network_comids)?;
            let p_vals = gather_by_comid(&p_map, &network_comids)?;
            Some(RoutingParamOverride {
                n: Some(physical_to_normalized(
                    &n_vals,
                    cfg.params.parameter_ranges.n,
                    log_space("n"),
                )),
                q_spatial: Some(physical_to_normalized(
                    &q_vals,
                    cfg.params.parameter_ranges.q_spatial,
                    log_space("q_spatial"),
                )),
                p_spatial: Some(physical_to_normalized(
                    &p_vals,
                    cfg.params.parameter_ranges.p_spatial,
                    log_space("p_spatial"),
                )),
            })
        }
        None => None,
    };
```

- [ ] **Step 6: Make zeta accumulation conditional and pass both overrides through**

Find:

```rust
    let mut zeta_sink = ZetaSums::<I>::new();
    let mut predictions_full = Array2::<f32>::zeros((n_all_gauges, n_hours));
```

Replace with:

```rust
    let mut zeta_sink: Option<ZetaSums<I>> = leakance_active.then(ZetaSums::<I>::new);
    let mut predictions_full = Array2::<f32>::zeros((n_all_gauges, n_hours));
```

Find the forward call inside the `while` loop:

```rust
        let runoff = forward_eval_reaches::<I>(
            &cfg,
            &tensors,
            &head,
            &device,
            false,
            Some(&mut zeta_sink),
            Some(&ov),
            None,
        );
```

Replace with:

```rust
        let runoff = forward_eval_reaches::<I>(
            &cfg,
            &tensors,
            &head,
            &device,
            false,
            zeta_sink.as_mut(),
            leakance_ov.as_ref(),
            param_ov.as_ref(),
        );
```

- [ ] **Step 7: Make the zeta answer-key write conditional**

Find the block after the `while` loop (tau-trim/obs-write is unchanged above this):

```rust
    // Answer key: zeta means over the routed window.
    let scale = 1.0_f32 / zeta_sink.steps as f32;
    assert!(scale.is_finite() && scale > 0.0, "zeta accumulation empty — leakance inactive?");
    let mean_vec = |t: Option<Tensor<I, 1>>| -> Vec<f32> {
        (t.expect("zeta sums present") * scale).into_data().into_vec().unwrap()
    };
    write_zeta_netcdf(
        zeta_path,
        &network_comids,
        &mean_vec(zeta_sink.abs_sum),
        &mean_vec(zeta_sink.net_sum),
        &mean_vec(zeta_sink.depth_sum),
        &mean_vec(zeta_sink.area_z_sum),
        &mean_vec(zeta_sink.q_sum),
        &format!("teacher:{}", checkpoint.display()),
    )
    .map_err(|e| -> Box<dyn std::error::Error> { e })?;
    println!("answer key → {} ({} reaches)", zeta_path.display(), network_comids.len());
    Ok(())
}
```

Replace with:

```rust
    // Answer key: zeta means over the routed window (only when leakance was
    // planted — a routing-parameter-donor-only teacher run has no zeta term
    // to report).
    if let Some(sink) = zeta_sink {
        let zeta_path = zeta_path.expect("zeta_path is set whenever leakance_active");
        let scale = 1.0_f32 / sink.steps as f32;
        assert!(scale.is_finite() && scale > 0.0, "zeta accumulation empty — leakance inactive?");
        let mean_vec = |t: Option<Tensor<I, 1>>| -> Vec<f32> {
            (t.expect("zeta sums present") * scale).into_data().into_vec().unwrap()
        };
        write_zeta_netcdf(
            &zeta_path,
            &network_comids,
            &mean_vec(sink.abs_sum),
            &mean_vec(sink.net_sum),
            &mean_vec(sink.depth_sum),
            &mean_vec(sink.area_z_sum),
            &mean_vec(sink.q_sum),
            &format!("teacher:{}", checkpoint.display()),
        )
        .map_err(|e| -> Box<dyn std::error::Error> { e })?;
        println!("answer key → {} ({} reaches)", zeta_path.display(), network_comids.len());
    } else {
        println!("no --plant-file given — skipping zeta answer-key write");
    }
    Ok(())
}
```

- [ ] **Step 8: Build and fix any compile errors**

Run: `cargo build --release --bin probe_zeta_gradient`
Expected: clean build. If `ddrs::data::load_comid_field` isn't re-exported from the crate root used by this binary, check the existing `use` block at the top of the file (it already imports `ddrs::data::{load_comid_field, ...}` — line ~166) and call it as bare `load_comid_field(...)` instead of `ddrs::data::load_comid_field(...)` in Step 5.

- [ ] **Step 9: Write the failing parity test**

Create `tests/teacher_donor_override_parity.rs`:

```rust
//! Parity gate for teacher mode's new n/q_spatial/p_spatial donor override
//! (docs/superpowers/specs/2026-07-22-synthetic-n-recoverability-design.md).
//!
//! Injecting a checkpoint's OWN dump_parameters output as the teacher's
//! donor field must reproduce the SAME synthetic gauge observations as
//! running teacher mode with no donor override at all — this exercises the
//! real donor-file I/O path (COMID keying, f32 round-trip, gather-by-comid)
//! that `RoutingParamOverride`'s unit tests in `src/training/forward.rs`
//! don't cover, mirroring `tests/eval_loss_own_parity.rs`'s pattern applied
//! to teacher mode.
//!
//! Skips gracefully if the real checkpoint/dump aren't present (machine-local,
//! gitignored) so CI on a clean checkout doesn't break.

use std::path::{Path, PathBuf};
use std::process::Command;

const CONFIG: &str = "/home/tbindas/projects/ddrs/config/experiments/synthetic_n_teacher.yaml";
const CHECKPOINT: &str =
    "/home/tbindas/projects/ddrs/.ddrs/runs/2026-07-16T02-22-14Z-train-and-test/checkpoints/epoch_5_mb_35";
const OWN_DUMP: &str = "/home/tbindas/projects/ddrs/output/synthetic_n/own_dump_for_parity_test.nc";

fn skip_if_missing(path: &str) -> Option<PathBuf> {
    let p = PathBuf::from(path);
    p.exists().then_some(p)
}

fn run_teacher(donor: Option<&Path>, obs_output: &Path) {
    let mut cmd = Command::new(env!("CARGO_BIN_EXE_probe_zeta_gradient"));
    cmd.args(["--mode", "teacher", "--backend", "cpu", "--config", CONFIG])
        .arg("--checkpoint")
        .arg(CHECKPOINT)
        .args(["--eval-days", "10"])
        .arg("--obs-output")
        .arg(obs_output);
    if let Some(d) = donor {
        cmd.arg("--donor-params-nc").arg(d);
    }
    let status = cmd.status().expect("run probe_zeta_gradient");
    assert!(status.success(), "teacher mode run failed (donor={donor:?})");
}

#[test]
fn own_donor_reproduces_no_donor_synthetic_obs() {
    let Some(config) = skip_if_missing(CONFIG) else {
        eprintln!("skipping: {CONFIG} not present on this machine");
        return;
    };
    let Some(checkpoint_head) = skip_if_missing(&format!("{CHECKPOINT}/head.mpk")) else {
        eprintln!("skipping: {CHECKPOINT}/head.mpk not present on this machine");
        return;
    };
    let Some(own_dump) = skip_if_missing(OWN_DUMP) else {
        eprintln!(
            "skipping: {OWN_DUMP} not present — generate via \
             `cargo run --release --bin dump_parameters -- --backend cpu \
             --config {CONFIG} --checkpoint {CHECKPOINT}/head --output {OWN_DUMP}`"
        );
        return;
    };
    drop((config, checkpoint_head));

    let tmp = std::env::temp_dir().join(format!(
        "teacher_donor_parity_{}",
        std::process::id()
    ));
    let no_donor = tmp.join("no_donor");
    let with_donor = tmp.join("with_donor");
    std::fs::create_dir_all(&tmp).unwrap();

    run_teacher(None, &no_donor);
    run_teacher(Some(&own_dump), &with_donor);

    // Compare the raw zarr-v2 chunk bytes for a sample of gauges — the
    // writer's chunk layout is deterministic given identical inputs, so
    // byte-identical chunks prove the donor path reproduced the exact same
    // routed discharge as the no-donor path.
    let mut compared = 0;
    for entry in std::fs::read_dir(&no_donor).unwrap() {
        let entry = entry.unwrap();
        if !entry.path().is_dir() {
            continue;
        }
        let gauge = entry.file_name();
        let a = std::fs::read(entry.path().join("0")).unwrap();
        let b = std::fs::read(with_donor.join(&gauge).join("0")).unwrap();
        assert_eq!(a, b, "gauge {gauge:?}: own-donor chunk diverged from no-donor chunk");
        compared += 1;
    }
    assert!(compared > 0, "no gauge chunks found to compare — obs writer produced nothing");
    eprintln!("compared {compared} gauges — own-donor teacher run byte-identical to no-donor");

    std::fs::remove_dir_all(&tmp).ok();
}
```

- [ ] **Step 10: Run the test to verify it currently skips (no fixtures yet) or fails**

Run: `cargo test --test teacher_donor_override_parity -- --nocapture`
Expected: test passes trivially by skipping (prints "skipping: ... not present"), since `config/experiments/synthetic_n_teacher.yaml` doesn't exist yet (Task 4) and `OWN_DUMP` doesn't exist yet. This confirms the test compiles and the skip path works; the real assertion is exercised once Task 4's config and a checkpoint dump exist (re-run at the end of Task 4).

- [ ] **Step 11: Commit**

```bash
git add src/bin/probe_zeta_gradient.rs tests/teacher_donor_override_parity.rs
git commit -m "feat(probe): teacher mode gains an optional n/q/p donor override

Reuses eval-loss mode's load_comid_field/gather_by_comid/RoutingParamOverride
machinery so teacher mode can generate synthetic-twin ground truth for the
routing parameters, not just leakance. --plant-file/--zeta-output/use_leakance
are now independent of the new --donor-params-nc path."
```

---

### Task 2: Consensus-geometry generation script

**Files:**
- Create: `scripts/synthetic_n_consensus_geometry.py`

- [ ] **Step 1: Write the script**

```python
"""Consensus geometry for the synthetic-n recoverability experiment
(docs/superpowers/specs/2026-07-22-synthetic-n-recoverability-design.md).

Runs `dump_parameters` against the 4 already-converged real-Q'-source
checkpoints from this campaign, then computes the per-COMID MEDIAN
q_spatial/p_spatial across them — the "most common trained value trend",
used as the FIXED geometry truth for every synthetic-n student.

Run from ddrs-py's venv:
    cd ddrs-py && uv run python ../scripts/synthetic_n_consensus_geometry.py
"""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import numpy as np
import xarray as xr

REPO = Path(__file__).resolve().parent.parent
OUT_DIR = REPO / "output/synthetic_n"
OUT_DIR.mkdir(parents=True, exist_ok=True)

CHECKPOINTS = [
    {
        "label": "aorc2f_distributed",
        "config": REPO / ".ddrs/runs/2026-07-16T02-22-14Z-train-and-test/config.yaml",
        "checkpoint": REPO / ".ddrs/runs/2026-07-16T02-22-14Z-train-and-test/checkpoints/epoch_5_mb_35/head",
    },
    {
        "label": "aorc2f_lumped",
        "config": REPO / ".ddrs/runs/2026-07-16T02-23-20Z-train-and-test/config.yaml",
        "checkpoint": REPO / ".ddrs/runs/2026-07-16T02-23-20Z-train-and-test/checkpoints/epoch_5_mb_35/head",
    },
    {
        "label": "daily_lstm",
        "config": REPO / ".ddrs/runs/2026-07-16T11-31-50Z-train-and-test/config.yaml",
        "checkpoint": REPO / ".ddrs/runs/2026-07-16T11-31-50Z-train-and-test/checkpoints/epoch_5_mb_35/head",
    },
    {
        "label": "hourly_lstm",
        "config": REPO / ".ddrs/runs/2026-07-16T11-31-52Z-train-and-test/config.yaml",
        "checkpoint": REPO / ".ddrs/runs/2026-07-16T11-31-52Z-train-and-test/checkpoints/epoch_5_mb_35/head",
    },
]


def dump_one(ckpt: dict) -> Path:
    out = OUT_DIR / f"{ckpt['label']}_kan_parameters.nc"
    if out.exists():
        print(f"{out} already exists, skipping dump_parameters re-run")
        return out
    cmd = [
        "cargo", "run", "--release", "--bin", "dump_parameters", "--",
        "--backend", "cpu",
        "--config", str(ckpt["config"]),
        "--checkpoint", str(ckpt["checkpoint"]),
        "--output", str(out),
    ]
    print("running:", " ".join(cmd))
    subprocess.run(cmd, cwd=REPO, check=True)
    return out


def main() -> None:
    dumps = [dump_one(c) for c in CHECKPOINTS]

    datasets = [xr.open_dataset(d) for d in dumps]
    comids_0 = datasets[0]["COMID"].values
    for d, ds in zip(dumps, datasets):
        if not np.array_equal(np.sort(ds["COMID"].values), np.sort(comids_0)):
            raise SystemExit(
                f"{d}: COMID set differs from {dumps[0]} — cannot take a per-COMID "
                "median across checkpoints with different networks"
            )

    # Re-index every dump to dumps[0]'s COMID order before stacking, since
    # dump_parameters row order isn't guaranteed identical across runs.
    order = comids_0
    q_stack = np.stack([ds.set_index(COMID="COMID").sel(COMID=order)["q_spatial"].values for ds in datasets])
    p_stack = np.stack([ds.set_index(COMID="COMID").sel(COMID=order)["p_spatial"].values for ds in datasets])

    q_median = np.median(q_stack, axis=0).astype(np.float32)
    p_median = np.median(p_stack, axis=0).astype(np.float32)

    out = OUT_DIR / "consensus_geometry.nc"
    xr.Dataset(
        {
            "q_spatial": ("COMID", q_median),
            "p_spatial": ("COMID", p_median),
        },
        coords={"COMID": order},
    ).to_netcdf(out)
    print(f"consensus geometry ({len(order)} reaches) -> {out}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Run it**

Run: `cd ddrs-py && uv run python ../scripts/synthetic_n_consensus_geometry.py`
Expected: prints 4 `cargo run --bin dump_parameters` invocations (each a single CPU forward pass, minutes not hours), then `consensus geometry (<N> reaches) -> .../output/synthetic_n/consensus_geometry.nc`.

- [ ] **Step 3: Commit**

```bash
git add scripts/synthetic_n_consensus_geometry.py
git commit -m "feat(scripts): consensus-geometry generator for synthetic-n experiment"
```

---

### Task 3: Truth-n field generation script

**Files:**
- Create: `scripts/synthetic_n_truth_fields.py`

- [ ] **Step 1: Write the script**

```python
"""Prescribed truth-n fields for the synthetic-n recoverability experiment
(docs/superpowers/specs/2026-07-22-synthetic-n-recoverability-design.md §1).

Combines each prescribed n field with the fixed consensus geometry
(scripts/synthetic_n_consensus_geometry.py) into a single donor NetCDF per
variant, in the dump_parameters::write_netcdf schema (COMID dim, f32 vars)
that probe_zeta_gradient's --mode teacher --donor-params-nc reads.

Run from ddrs-py's venv, AFTER synthetic_n_consensus_geometry.py:
    cd ddrs-py && uv run python ../scripts/synthetic_n_truth_fields.py
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import xarray as xr

REPO = Path(__file__).resolve().parent.parent
OUT_DIR = REPO / "output/synthetic_n"
ATTRS = REPO.parent / "ddr/data/merit_global_attributes_v2.nc"

N_LO, N_HI = 0.015, 0.15
N_CENTER = 0.08
SEED = 42


def leopold_maddock_n(log10_uparea: np.ndarray) -> np.ndarray:
    """Decreasing power law: n = clip(N_CENTER * (uparea/uparea_median)^-b, N_LO, N_HI).

    b is calibrated so the field spans roughly [N_LO, N_HI] across the real
    CONUS log10_uparea distribution (see design spec §1 footnote — a tuning
    detail, not a design fork).
    """
    median = np.median(log10_uparea)
    b = 0.15
    n = N_CENTER * 10.0 ** (-b * (log10_uparea - median))
    return np.clip(n, N_LO, N_HI).astype(np.float32)


def gaussian_noise_n(n_reaches: int) -> np.ndarray:
    """IID Gaussian field, no spatial structure — the null control."""
    rng = np.random.default_rng(SEED)
    spread = (N_HI - N_LO) / 4.0  # ~2 std devs to each bound from N_CENTER
    n = rng.normal(loc=N_CENTER, scale=spread, size=n_reaches)
    return np.clip(n, N_LO, N_HI).astype(np.float32)


def main() -> None:
    geom = xr.open_dataset(OUT_DIR / "consensus_geometry.nc")
    attrs = xr.open_dataset(ATTRS)
    attrs_by_comid = attrs.set_index(COMID="COMID").sel(COMID=geom["COMID"].values)
    log10_uparea = attrs_by_comid["log10_uparea"].values.astype(np.float64)

    variants = {
        "truth_leopold_maddock.nc": leopold_maddock_n(log10_uparea),
        "truth_gaussian.nc": gaussian_noise_n(len(geom["COMID"])),
    }

    for filename, n_vals in variants.items():
        out = OUT_DIR / filename
        xr.Dataset(
            {
                "n": ("COMID", n_vals),
                "q_spatial": ("COMID", geom["q_spatial"].values),
                "p_spatial": ("COMID", geom["p_spatial"].values),
            },
            coords={"COMID": geom["COMID"].values},
        ).to_netcdf(out)
        print(
            f"{out}: n range [{n_vals.min():.4f}, {n_vals.max():.4f}], "
            f"median {np.median(n_vals):.4f} ({len(n_vals)} reaches)"
        )


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Run it**

Run: `cd ddrs-py && uv run python ../scripts/synthetic_n_truth_fields.py`
Expected: two lines reporting `output/synthetic_n/truth_leopold_maddock.nc` and `output/synthetic_n/truth_gaussian.nc`, each with an n range inside `[0.015, 0.15]`.

- [ ] **Step 3: Sanity-check the Leopold-Maddock slope sign**

Run:
```bash
cd ddrs-py && uv run python -c "
import numpy as np, xarray as xr
from pathlib import Path
geom = xr.open_dataset('../output/synthetic_n/consensus_geometry.nc')
truth = xr.open_dataset('../output/synthetic_n/truth_leopold_maddock.nc')
attrs = xr.open_dataset('/home/tbindas/projects/ddr/data/merit_global_attributes_v2.nc')
a = attrs.set_index(COMID='COMID').sel(COMID=geom['COMID'].values)
slope = np.polyfit(a['log10_uparea'].values, truth['n'].values, 1)[0]
print('fitted slope (should be NEGATIVE — n decreases downstream):', slope)
assert slope < 0, 'Leopold-Maddock truth field has the wrong sign!'
"
```
Expected: prints a negative slope value, assertion passes.

- [ ] **Step 4: Commit**

```bash
git add scripts/synthetic_n_truth_fields.py
git commit -m "feat(scripts): prescribed truth-n fields (Leopold-Maddock + Gaussian null)"
```

---

### Task 4: Teacher config and teacher run

**Files:**
- Create: `config/experiments/synthetic_n_teacher.yaml`

- [ ] **Step 1: Write the config**

Base it on `config/experiments/aorc2f_distributed_frozen_chunk1.yaml` (read it first to copy `kan_head`/`params`/`experiment` blocks verbatim), changing only `data_sources.streamflow` to the standard benchmark store and widening `testing:` to span the full simulation window with 1-day padding on each side (matching the `recoverability_teacher.yaml` padding convention so tau-trim + last-day-drop produce exactly the 1981/10/01-2010/09/30 axis the students need):

```yaml
# synthetic_n_teacher.yaml — synthetic-n recoverability ground-truth generator.
# GENERATED from aorc2f_distributed_frozen_chunk1.yaml — teacher world: full
# simulation window, standard benchmark Q' store, no leakance.
# Spec: docs/superpowers/specs/2026-07-22-synthetic-n-recoverability-design.md

mode: training
workflow: train-and-test
geodataset: merit
seed: 42
np_seed: 42

data_sources:
  attributes: /home/tbindas/projects/ddr/data/merit_global_attributes_v2.nc
  conus_adjacency: /home/tbindas/projects/ddr/data/merit_conus_adjacency.zarr
  gages_adjacency: /home/tbindas/projects/ddr/data/merit_gages_conus_adjacency.zarr
  streamflow: /mnt/ssd1/data/icechunk/merit_dhbv2_UH_retrospective.ic
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
  disaggregation:
    hidden_size: 16
    num_hidden_layers: 2
    grid: 20
    k: 3
    chunk_days: 1
    freeze: true
    pretrained_checkpoint: /home/tbindas/projects/ddrs/output/disagg_pretrain/capacity_chunk1.mpk

params:
  use_leakance: false
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

# 1-day padding each side of the union of both student axes (1981/10/01-
# 1995/09/30 train + 1995/10/01-2010/09/30 test), so tau-trim + the
# teacher's last-day drop produce synthetic obs covering EXACTLY
# 1981/10/01-2010/09/30 — matching recoverability_teacher.yaml's padding
# convention.
testing:
  start_time: 1981/09/30
  end_time: 2010/10/01
  batch_size: 15
  rho: null
```

Note: check the real `aorc2f_distributed_frozen_chunk1.yaml`'s `kan_head.disaggregation` block for the exact `num_hidden_layers`/`grid`/`k`/`chunk_days`/`pretrained_checkpoint` values before finalizing — copy them verbatim rather than retyping from memory, since a mismatch will fail config load against the frozen checkpoint's architecture.

- [ ] **Step 2: Validate the config parses**

Run: `cargo run --release --bin ddrs -- --config config/experiments/synthetic_n_teacher.yaml --workspace /tmp/synthetic_n_teacher_validate plan --workflow train-and-test`
Expected: exits 0 (or fails only on the GPU probe, which is irrelevant here — this step just validates YAML/schema, not that training will run). If it fails on schema, fix the config and retry.

- [ ] **Step 3: Re-run the Task 1 parity test now that the config exists**

Run:
```bash
cargo run --release --bin dump_parameters -- --backend cpu \
  --config config/experiments/synthetic_n_teacher.yaml \
  --checkpoint .ddrs/runs/2026-07-16T02-22-14Z-train-and-test/checkpoints/epoch_5_mb_35/head \
  --output output/synthetic_n/own_dump_for_parity_test.nc
cargo test --release --test teacher_donor_override_parity -- --nocapture
```
Expected: `own_donor_reproduces_no_donor_synthetic_obs` runs for real this time (not skipped) and PASSES — "compared N gauges — own-donor teacher run byte-identical to no-donor".

- [ ] **Step 4: Commit**

```bash
git add config/experiments/synthetic_n_teacher.yaml
git commit -m "feat(config): synthetic-n teacher config (full-window, standard Q' store)"
```

- [ ] **Step 5: Launch the Phase-1 teacher run (Leopold-Maddock truth n)**

This is a long-running full-CONUS 29-year continuous forward pass on CPU — launch in the background and monitor.

Run:
```bash
mkdir -p output/synthetic_n/logs
nohup cargo run --release --bin probe_zeta_gradient -- \
  --mode teacher --backend cpu \
  --config config/experiments/synthetic_n_teacher.yaml \
  --checkpoint .ddrs/runs/2026-07-16T02-22-14Z-train-and-test/checkpoints/epoch_5_mb_35 \
  --eval-days 999999 \
  --donor-params-nc output/synthetic_n/truth_leopold_maddock.nc \
  --obs-output output/synthetic_n/synthetic_obs_lm \
  > output/synthetic_n/logs/teacher_lm.log 2>&1 &
```
Expected: log shows `teacher: 0 plants, 2365 gauges, <N> reaches, <days> days` then periodic `chunk k/n_chunks_total` lines, ending with `synthetic obs → output/synthetic_n/synthetic_obs_lm (2365 gauges, <days> days from 1981-10-01)` and `no --plant-file given — skipping zeta answer-key write`. Wall-time: expect several hours (the leakance recoverability teacher's 14-year window took ~2.6h CPU; this is a ~29-year window over the same chunking, so budget roughly double).

- [ ] **Step 6: Verify the synthetic obs store**

Run:
```bash
cd ddrs-py && uv run python -c "
import xarray as xr
import zarr
g = zarr.open_group('../output/synthetic_n/synthetic_obs_lm', mode='r')
print('gauges:', len(list(g.group_keys())))
import numpy as np
sample = list(g.group_keys())[0]
arr = g[sample]['0'][:]
print(sample, 'len', len(arr), 'finite fraction', np.isfinite(arr).mean())
"
```
Expected: `gauges: 2365`, and the sample gauge array's finite fraction should be less than 1.0 (NaN-padded before 1981-10-01) but the tail should be finite daily values.

---

### Task 5: Student configs

**Files:**
- Create: `config/experiments/synthetic_n_student_distributed.yaml`
- Create: `config/experiments/synthetic_n_student_lumped.yaml`
- Create: `config/experiments/synthetic_n_student_daily_lstm.yaml`
- Create: `config/experiments/synthetic_n_student_hourly_lstm.yaml`

- [ ] **Step 1: Create each student config as an exact copy of its real campaign counterpart**

```bash
cp config/experiments/aorc2f_distributed_frozen_chunk1.yaml config/experiments/synthetic_n_student_distributed.yaml
cp config/experiments/aorc2f_lumped_frozen_chunk1.yaml config/experiments/synthetic_n_student_lumped.yaml
cp config/experiments/lstm_daily_frozen_chunk1.yaml config/experiments/synthetic_n_student_daily_lstm.yaml
cp config/experiments/lstm_hourly_native.yaml config/experiments/synthetic_n_student_hourly_lstm.yaml
```

- [ ] **Step 2: Repoint `data_sources.observations` in all 4 to the synthetic obs store**

For each of the 4 new files, change the `data_sources.observations:` line (originally `/mnt/ssd1/data/icechunk/usgs_daily_observations`) to:

```yaml
  observations: /home/tbindas/projects/ddrs/output/synthetic_n/synthetic_obs_lm
```

Use `Edit` on each file (do not touch any other line — `streamflow`, `kan_head`, `params`, `experiment` stay exactly as the real campaign's own arms, since the whole point is that everything except the forcing and the observation target is identical across students).

- [ ] **Step 3: Validate each config parses**

```bash
for f in synthetic_n_student_distributed synthetic_n_student_lumped synthetic_n_student_daily_lstm synthetic_n_student_hourly_lstm; do
  echo "=== $f ==="
  cargo run --release --bin ddrs -- --config config/experiments/$f.yaml \
    --workspace /tmp/${f}_validate plan --workflow train-and-test || true
done
```
Expected: each either exits 0 or fails only on the GPU-probe / real-data-source-availability step (irrelevant to config schema validity) — no YAML/schema errors.

- [ ] **Step 4: Commit**

```bash
git add config/experiments/synthetic_n_student_*.yaml
git commit -m "feat(config): 4 synthetic-n student configs (real Q' sources, synthetic obs)"
```

---

### Task 6: Launch the 4 students and dump their recovered parameters

**Files:** none (execution only)

- [ ] **Step 1: Launch all 4 students in parallel, isolated workspaces, CPU backend**

```bash
mkdir -p output/synthetic_n/logs
for arm in distributed lumped daily_lstm hourly_lstm; do
  mkdir -p .ddrs-synthetic-n-$arm
  nohup cargo run --release --bin ddrs -- \
    --config config/experiments/synthetic_n_student_$arm.yaml \
    --workspace /home/tbindas/projects/ddrs/.ddrs-synthetic-n-$arm \
    run --workflow train-and-test --backend cpu \
    > output/synthetic_n/logs/student_$arm.log 2>&1 &
done
wait
```
Expected: each log ends with a completed `train-and-test` workflow and a manifest reporting `epochs_completed: 5`. Wall-time comparable to this campaign's own CPU arms (hours; the lumped arm took ~8h26m end-to-end) — monitor via `tail -f output/synthetic_n/logs/student_*.log`.

- [ ] **Step 2: Locate each student's final checkpoint and dump parameters**

```bash
mkdir -p output/synthetic_n
for arm in distributed lumped daily_lstm hourly_lstm; do
  RUN_ID=$(ls -t .ddrs-synthetic-n-$arm/runs/ | head -1)
  CKPT=$(ls -d .ddrs-synthetic-n-$arm/runs/$RUN_ID/checkpoints/epoch_5_mb_* | sort -V | tail -1)
  echo "$arm -> $CKPT"
  cargo run --release --bin dump_parameters -- --backend cpu \
    --config config/experiments/synthetic_n_student_$arm.yaml \
    --checkpoint "$CKPT/head" \
    --output output/synthetic_n/recovered_$arm.nc
done
```
Expected: 4 NetCDF files `output/synthetic_n/recovered_{distributed,lumped,daily_lstm,hourly_lstm}.nc`, each with `n`/`q_spatial`/`p_spatial` per COMID.

---

### Task 7: Analysis script and findings

**Files:**
- Create: `scripts/synthetic_n_recoverability_analysis.py`

- [ ] **Step 1: Write the script**

```python
"""Pre-registered verdicts S1-S5 for the synthetic-n recoverability
experiment (docs/superpowers/specs/2026-07-22-synthetic-n-recoverability-design.md §3).

Run from ddrs-py's venv, after all 4 students' dump_parameters outputs exist:
    cd ddrs-py && uv run python ../scripts/synthetic_n_recoverability_analysis.py
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr
from scipy.stats import pearsonr

REPO = Path(__file__).resolve().parent.parent
OUT_DIR = REPO / "output/synthetic_n"
ATTRS = REPO.parent / "ddr/data/merit_global_attributes_v2.nc"

ARMS = ["distributed", "lumped", "daily_lstm", "hourly_lstm"]

# Mean daily Q' volume ratio of each real store vs the standard benchmark
# store, for S5 — filled in manually from `ddrs import --dry-run` /
# icechunk inspection once available; None disables S5 for that arm. S5 is
# WIRED IN below (pearsonr against n_errors) but only computed when at
# least 3 arms have a non-None ratio — with only 4 arms total, fewer points
# than that isn't a meaningful correlation, so it's reported as
# "not computed", never silently skipped without saying why.
VOLUME_RATIO_VS_TRUTH: dict[str, float | None] = {
    "distributed": None,
    "lumped": None,
    "daily_lstm": None,
    "hourly_lstm": None,
}


def median_abs_error(recovered: np.ndarray, truth: np.ndarray) -> float:
    return float(np.median(np.abs(recovered - truth)))


def fitted_slope(log10_uparea: np.ndarray, n: np.ndarray) -> float:
    return float(np.polyfit(log10_uparea, n, 1)[0])


def main() -> None:
    truth = xr.open_dataset(OUT_DIR / "truth_leopold_maddock.nc")
    truth_comids = truth["COMID"].values
    truth_n = truth["n"].values
    truth_q = truth["q_spatial"].values
    truth_p = truth["p_spatial"].values

    attrs = xr.open_dataset(ATTRS).set_index(COMID="COMID").sel(COMID=truth_comids)
    log10_uparea = attrs["log10_uparea"].values.astype(np.float64)
    true_slope = fitted_slope(log10_uparea, truth_n)

    rows = []
    n_errors, geom_errors = {}, {}
    for arm in ARMS:
        rec = xr.open_dataset(OUT_DIR / f"recovered_{arm}.nc").set_index(COMID="COMID").sel(
            COMID=truth_comids
        )
        n_err = median_abs_error(rec["n"].values, truth_n)
        q_err = median_abs_error(rec["q_spatial"].values, truth_q)
        p_err = median_abs_error(rec["p_spatial"].values, truth_p)
        slope = fitted_slope(log10_uparea, rec["n"].values)
        n_errors[arm] = n_err
        geom_errors[arm] = (q_err + p_err) / 2.0
        rows.append(
            {
                "arm": arm,
                "n_median_abs_error": n_err,
                "q_median_abs_error": q_err,
                "p_median_abs_error": p_err,
                "recovered_n_slope": slope,
                "true_n_slope": true_slope,
                "slope_sign_flipped": bool(slope > 0 and true_slope < 0),
            }
        )

    df = pd.DataFrame(rows)
    csv_path = OUT_DIR / "recoverability_rows.csv"
    df.to_csv(csv_path, index=False)

    n_spread = max(n_errors.values()) - min(n_errors.values())
    geom_spread = max(geom_errors.values()) - min(geom_errors.values())
    s4_ratio = n_spread / geom_spread if geom_spread > 0 else float("inf")

    any_flip = df["slope_sign_flipped"].any()

    # S5: pearsonr of n_errors against VOLUME_RATIO_VS_TRUTH, restricted to
    # arms with a filled-in ratio. Needs >=3 points to be worth reporting at
    # all (spec §3: "only 4 data points" — 2 points is a line, not a
    # correlation). Prints an explicit reason when it can't run, rather than
    # silently doing nothing.
    filled = {a: r for a, r in VOLUME_RATIO_VS_TRUTH.items() if r is not None}
    if len(filled) >= 3:
        arms_with_ratio = list(filled.keys())
        s5_r, s5_p = pearsonr(
            [n_errors[a] for a in arms_with_ratio],
            [filled[a] for a in arms_with_ratio],
        )
        s5_line = f"  [S5] pearson r={s5_r:.3f} (p={s5_p:.3f}) over {len(filled)} arms: {filled}"
    else:
        s5_r = s5_p = None
        s5_line = (
            f"  [S5] not computed — only {len(filled)}/{len(ARMS)} arms have a filled-in "
            "VOLUME_RATIO_VS_TRUTH (need >=3). Fill in the dict from `ddrs import --dry-run` "
            "/ icechunk mean-daily-volume inspection per arm to enable this."
        )

    print(df.to_string(index=False))
    print()
    print("========================================================================")
    print("VERDICTS (bars pre-registered in the design spec)")
    print("========================================================================")
    print(f"  [S1] n median-abs-error per arm: {n_errors}")
    print(f"  [S2] true slope={true_slope:.5f}; any arm sign-flipped positive: {any_flip}")
    print(f"  [S3] geometry median-abs-error per arm: {geom_errors}")
    print(f"  [S4 {'PASS' if s4_ratio >= 3 else 'FAIL'}] n-spread/geom-spread = {s4_ratio:.2f} (bar: >=3)")
    print(s5_line)
    print(f"  HEADLINE: {'PASS' if (s4_ratio >= 3 and any_flip) else 'FAIL'} "
          "(requires S4>=3x AND at least one slope sign flip; S5 is supporting evidence only)")
    print()
    print(f"per-arm rows -> {csv_path}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Confirm `scipy` is available in `ddrs-py`, add it if not**

Run: `cd ddrs-py && uv run python -c "import scipy; print(scipy.__version__)"`
Expected: prints a version. If it errors with `ModuleNotFoundError`, run `cd ddrs-py && uv add scipy` first, then retry.

- [ ] **Step 3: Run it**

Run: `cd ddrs-py && uv run python ../scripts/synthetic_n_recoverability_analysis.py`
Expected: a printed per-arm table, then the VERDICTS block (including the `[S5]` line — either a pearson r/p or the explicit "not computed" message), then `per-arm rows -> .../output/synthetic_n/recoverability_rows.csv`.

- [ ] **Step 4: Write the findings doc**

Create `docs/2026-07-22-synthetic-n-recoverability-findings.md` following this campaign's established findings-doc structure (see `docs/2026-07-16-aorc2f-wave1-findings.md` for the template: what this tests, execution notes, results table, interpretation). Populate it with the actual VERDICTS block output and the per-arm CSV, and state explicitly whether the headline S4/slope-flip bar passed.

The findings doc MUST include these two notes explicitly (design spec §6 concern 1 and §1's naming note — do not silently drop either):

1. **Disagg-head confound caveat on S3.** State plainly that geometry
   (q_spatial/p_spatial) recovery error being small/consistent across arms
   is NOT independent proof that geometry is physically identifiable
   regardless of Q'-source bias — every arm (teacher and all 4 students)
   shares the SAME frozen capacity-boosted chunk1 disagg head, so a latent
   disagg-head effect on q/p would look identical to genuine geometry
   robustness. This experiment's S3 result should be reported as
   "consistent under a shared, frozen disagg head," not as "geometry is
   identifiable in general."
2. **Naming/campaign-provenance note.** This experiment's 4 arms
   (`aorc2f_distributed`/`aorc2f_lumped`/`daily_lstm`/`hourly_lstm`, from
   the 2026-07-16 AORC2F/LSTM wave campaign) are a DIFFERENT arm set from
   the pre-registered LSTM-equifinality campaign's R1/R2/R3 naming (the
   paper's `tab:arms` table). State this explicitly and do not let the two
   numbering schemes bleed together in any table or cross-reference.

- [ ] **Step 5: Decide on Phase 2 (Gaussian-noise confirmatory run)**

If the headline verdict PASSED (S4 ≥ 3× and at least one slope flip): identify the 2 arms with the largest `n_median_abs_error` from `recoverability_rows.csv`, then repeat Task 4 Step 5 - Task 6 for just those 2 arms using `--donor-params-nc output/synthetic_n/truth_gaussian.nc` and a fresh `--obs-output output/synthetic_n/synthetic_obs_gaussian`, producing `config/experiments/synthetic_n_student_{arm}_gaussian.yaml` variants (copy + repoint observations). Extend the analysis script with a `--truth gaussian` mode, or a second invocation pointed at the Gaussian truth/recovered files, before re-running Step 2 for those 2 arms only.

If the headline verdict FAILED: do not run Phase 2 — document in the findings doc why (e.g. n_spread not distinguishable from geometry spread, or the sign never flips), and note this as informative for, but not conclusive against, the standing equifinality campaign's own registered verdict (per design spec §8 — this experiment does not amend that campaign's results).

- [ ] **Step 6: Commit**

```bash
git add scripts/synthetic_n_recoverability_analysis.py docs/2026-07-22-synthetic-n-recoverability-findings.md output/synthetic_n/recoverability_rows.csv
git commit -m "docs: synthetic-n recoverability findings across the 4 real Q' sources"
```

---

## Self-Review Notes

**Spec coverage:** §2 architecture (teacher override, students, measurement) → Tasks 1, 4, 5, 6. §3 verdicts S1-S5 → Task 7. §4 components 1-7 → Tasks 1-7 map 1:1 except component 7 (findings report) folded into Task 7 Step 3. §5 execution sequence/staging → Tasks 4-7, with Phase 2 gated in Task 7 Step 4. §6 concerns (disagg-head mismatch, cold-start noise floor, 8-run cost, consensus-geometry provenance) → addressed structurally (Task 2 uses the 4 real checkpoints' own disagg-head architecture; Task 6 launches cold; Task 7 Step 4 stages Phase 2 conditionally) and should be called out explicitly in the Task 7 findings doc. §7 testing → Task 1's parity test; no `Backward`/`src/routing`/`src/sparse.rs` changes anywhere in this plan, so `compare_ddr_sandbox` is not re-run as a gate (nothing in its blast radius changed).

**Placeholder scan:** no TBD/TODO. `VOLUME_RATIO_VS_TRUTH` in Task 7's script starts `None`-filled (the real per-arm volume ratios aren't known until someone inspects the icechunk stores), but S5 IS wired in — it computes `pearsonr` over whichever arms have a filled-in ratio once `>= 3` are present, and otherwise prints an explicit "not computed, need >=3, here's how to fill it in" line rather than silently doing nothing. (An earlier draft of this plan left the dict unread by `main()` — a real dead-code bug flagged in review — fixed here.)

**Type consistency:** `RoutingParamOverride { n, q_spatial, p_spatial }` field names used identically in Task 1's Rust code and match the struct defined at `src/training/forward.rs:356-360`. `load_comid_field`/`gather_by_comid`/`physical_to_normalized` signatures used in Task 1 match their existing definitions read from `src/data/store/param_dump.rs:18` and `src/bin/probe_zeta_gradient.rs:1613`/`src/training/forward.rs:72`. Python scripts consistently use `COMID` as the NetCDF dim/coord name and `n`/`q_spatial`/`p_spatial` as var names across Tasks 2, 3, 6, 7.

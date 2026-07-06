# Synthetic Losing-Reach Recoverability (Positive Control) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Plant detectable-scale leakance losses in a synthetic teacher world, train three student runs on the teacher's gauge discharge, and measure whether the optimizer attributes the missing water to the leakance term (recovery) or to routing parameters (H5 absorption).

**Architecture:** A `--mode teacher` in the existing probe binary runs the trained checkpoint with per-COMID leakance-parameter overrides, writing synthetic observations (zarr-v2, `ObservationsStore`-compatible) plus an exact answer key (zeta accumulator). Students A (warm/ON), B (warm/OFF), C (cold/ON) train in parallel on CPU via the legacy `train` binary with the obs source swapped; recovery is judged per-reach against the answer key via `bin/eval --zeta-output`.

**Tech Stack:** Rust (BURN 0.21, NdArray CPU backend, zarrs, netcdf, clap), Python under `ddrs-py` uv venv (xarray, netCDF4, pandas, geopandas) for sites/analysis/maps.

**Spec:** `docs/superpowers/specs/2026-07-03-synthetic-recoverability-design.md`

**Plan-time deviations from the spec (all simplifications, discovered by reading current code):**
1. Spec §4.2 prescribed a new `experiment.init_head` config key. NOT needed: `bootstrap_head_and_state` (src/training/bootstrap.rs:74-105) already implements weights-only warm-start when `experiment.checkpoint` points at a directory containing only `head.mpk` — it loads the head, logs "Adam starts cold" and "restarting at epoch 1 with a fresh shuffle". We create such a directory by copying `head.mpk`.
2. Spec §4.3's Run-B config relaxation is NOT needed: the only leakance config validation is the `use_cuda_graphs` conflict (src/config.rs:626); both training `forward` (src/training/forward.rs:210) and `forward_eval` (:349) simply skip the three params when `use_leakance: false`, and the head emits whatever `learnable_parameters` lists. `use_leakance: false` with the ON head config Just Works.
3. Spec §4.1 put the obs writer inside the teacher mode; it lives in its own module `src/data/store/obs_writer.rs` so it gets a roundtrip test against the real reader.
4. NEW work the spec didn't list: `bin/train.rs`, `bin/eval.rs`, `bin/dump_parameters.rs` are hardcoded to `Cuda<f32,i32>`; each needs the probe binary's `--backend {cpu,cuda}` dispatch (src/bin/probe_zeta_gradient.rs:123-146) since ALL runs are CPU-only.

**Hard rules for every task:**
- Work in `/home/tbindas/projects/ddrs/.claude/worktrees/zeta-sensitivity` (branch `worktree-zeta-sensitivity`). NEVER `cargo install`. Heavy runs use the ABSOLUTE worktree binary path `/home/tbindas/projects/ddrs/.claude/worktrees/zeta-sensitivity/target/release/<bin>` with cwd `/home/tbindas/projects/ddrs` (main tree — data + `.ddrs/` live there). A relative `target/release/...` from the main tree runs a STALE binary (see `.claude` memory `ddrs-worktree-gotchas`).
- Heavy runs: `nice -n 10`, tee output to a log file under `/home/tbindas/projects/ddrs/output/recoverability/logs/`.
- Guard suites must stay green after every code task: `cargo test --test leakance_gradcheck --test leakance_off_parity --test zeta_accum` and `cargo run --release --example compare_ddr_sandbox` (must print ABSOLUTE MATCH). No autograd `Backward` impl may be touched.

**Runtime artifact layout (all under the MAIN tree):**

```
/home/tbindas/projects/ddrs/output/recoverability/
├── init_head/head.mpk            # weights-only warm-start dir (Task 5)
├── plants.csv                    # plant plan (Task 6)
├── sites_report.txt              # kept/dropped sites (Task 6)
├── synthetic_obs/                # zarr-v2 obs store (Task 7)
├── answer_key.nc                 # teacher zeta accumulation (Task 7)
├── baseline_zeta.nc              # unmodified-checkpoint zeta field (Task 7)
├── logs/                         # tee'd run logs
├── students/{a,b,c}/             # per-student checkpoint dirs
├── eval_{a,c}.zarr, zeta_{a,c}.nc  # measurement passes (Task 9)
├── params_orig.nc, params_b.nc   # dump_parameters for R4 (Task 9)
├── recovery_rows.csv             # per-reach analysis rows (Task 9)
└── plots/                        # notebook PNGs (Task 10)
```

**Key existing code to read before starting any task:** `src/bin/probe_zeta_gradient.rs` (backend dispatch :123-146, perturb chunk loop :523-608), `src/training/forward.rs` (`forward_eval` :309-409, `ZetaSums` :257-296), `src/data/store/zarr_obs.rs` (reader + test fixture :214-236), `src/bin/eval.rs` (zeta finalize :131-151), `src/training/bootstrap.rs`.

---

### Task 1: `LeakanceOverride` + `forward_eval` override seam

**Files:**
- Modify: `src/training/forward.rs` (struct + apply inside `forward_eval` :349-365 block; unit test at bottom)
- Modify: `src/training/mod.rs` (re-export)
- Modify: `src/bin/eval.rs` (pass `None` at the two `evaluate` internals — see step 4)
- Modify: `src/training/eval.rs` (thread the param through `evaluate`'s `forward_eval` call as `None`)
- Modify: `src/bin/probe_zeta_gradient.rs:569` (pass `None`)

- [ ] **Step 1: Write the failing unit test** at the bottom of `src/training/forward.rs`:

```rust
#[cfg(test)]
mod tests {
    use super::*;
    use burn::backend::NdArray;

    type B = NdArray<f32>;

    #[test]
    fn leakance_override_apply_replaces_masked_entries_only() {
        let device = <B as burn::tensor::backend::BackendTypes>::Device::default();
        let ov = LeakanceOverride {
            mask: vec![0.0, 1.0, 0.0, 1.0],
            k_d: vec![0.0, 1.0, 0.0, 0.5],
            d_gw: vec![0.0, 0.0, 0.0, 0.25],
            factor: vec![0.0, 0.9, 0.0, 0.1],
        };
        let param: Tensor<B, 1> =
            Tensor::from_floats([0.7, 0.7, 0.7, 0.7], &device);
        let out: Vec<f32> = ov
            .apply(param.clone(), &ov.k_d, &device)
            .into_data()
            .into_vec()
            .unwrap();
        assert_eq!(out, vec![0.7, 1.0, 0.7, 0.5]);
        let out_f: Vec<f32> = ov
            .apply(param, &ov.factor, &device)
            .into_data()
            .into_vec()
            .unwrap();
        assert_eq!(out_f, vec![0.7, 0.9, 0.7, 0.1]);
    }
}
```

- [ ] **Step 2: Run to verify it fails**

Run: `cargo test --lib leakance_override_apply -- --nocapture`
Expected: FAIL — `LeakanceOverride` not found.

- [ ] **Step 3: Implement.** In `src/training/forward.rs`, above `forward_eval`:

```rust
/// Per-reach override of the NORMALIZED leakance head outputs, applied inside
/// `forward_eval` between `head.forward` and denormalization (which happens
/// in `setup_inputs`). Vectors are dense over the network's reach columns
/// (same order as `divide_comids`): `mask[i] == 1.0` replaces reach i's
/// normalized K_D/d_gw/factor with the corresponding value; `mask[i] == 0.0`
/// leaves the head's output untouched. Eval-path only — the training
/// `forward` never sees this type.
pub struct LeakanceOverride {
    pub mask: Vec<f32>,
    pub k_d: Vec<f32>,
    pub d_gw: Vec<f32>,
    pub factor: Vec<f32>,
}

impl LeakanceOverride {
    fn apply<I: Backend>(
        &self,
        param: Tensor<I, 1>,
        vals: &[f32],
        device: &I::Device,
    ) -> Tensor<I, 1> {
        let n = self.mask.len();
        let mask_t: Tensor<I, 1> =
            Tensor::from_data(TensorData::new(self.mask.clone(), [n]), device);
        let vals_t: Tensor<I, 1> =
            Tensor::from_data(TensorData::new(vals.to_vec(), [n]), device);
        param * (mask_t.ones_like() - mask_t.clone()) + vals_t * mask_t
    }
}
```

(Add `use burn::tensor::TensorData;` if not already imported.) Then change the `forward_eval` signature and the `use_leakance` arm:

```rust
pub fn forward_eval<I: Backend>(
    cfg: &Config,
    tensors: &RoutingTensors<I>,
    head: &KanHead<I>,
    device: &I::Device,
    carry_state: bool,
    zeta: Option<&mut ZetaSums<I>>,
    overrides: Option<&LeakanceOverride>,
) -> Tensor<I, 2> {
```

and inside the `if cfg.params.use_leakance` arm (:349-365), after fetching the three tensors:

```rust
        let (mut k_d_t, mut d_gw_t, mut factor_t) = (
            params_map.get("K_D").expect("checked above").clone(),
            params_map.get("d_gw").expect("checked above").clone(),
            params_map.get("leakance_factor").expect("checked above").clone(),
        );
        if let Some(ov) = overrides {
            assert_eq!(
                ov.mask.len(),
                k_d_t.dims()[0],
                "LeakanceOverride length {} != network reaches {}",
                ov.mask.len(),
                k_d_t.dims()[0]
            );
            k_d_t = ov.apply::<I>(k_d_t, &ov.k_d, device);
            d_gw_t = ov.apply::<I>(d_gw_t, &ov.d_gw, device);
            factor_t = ov.apply::<I>(factor_t, &ov.factor, device);
        }
        (Some(k_d_t), Some(d_gw_t), Some(factor_t))
```

(Keep the existing missing-key panic loop before this.) Update ALL call sites to pass the extra arg: `src/training/eval.rs`'s `evaluate` calls `forward_eval(..., zeta_sink_arg, None)`; `src/bin/probe_zeta_gradient.rs:569` becomes `forward_eval::<I>(&cfg, &tensors, &head, &device, chunk_idx > 0, None, None)`. Re-export in `src/training/mod.rs` next to `forward_eval`: `pub use forward::LeakanceOverride;`.

- [ ] **Step 4: Compile everything and run the test + guards**

Run: `cargo test --lib leakance_override_apply && cargo build --release --bins && cargo test --test leakance_off_parity`
Expected: PASS / clean build / 3 passed.

- [ ] **Step 5: Commit**

```bash
git add src/training/forward.rs src/training/mod.rs src/training/eval.rs src/bin/eval.rs src/bin/probe_zeta_gradient.rs
git commit -m "feat(eval): LeakanceOverride — per-reach normalized leakance param override in forward_eval"
```

---

### Task 2: Synthetic-observations zarr-v2 writer

**Files:**
- Create: `src/data/store/obs_writer.rs`
- Modify: `src/data/store/mod.rs` (add `pub mod obs_writer;`)

- [ ] **Step 1: Write the failing roundtrip test** inside `src/data/store/obs_writer.rs` (write module skeleton + test; test reads back through the REAL dispatching reader):

```rust
//! Writer for a minimal zarr-v2 observations store readable by
//! `ObservationsStore` (format template: the hand-written fixture in
//! `zarr_obs.rs` tests, :214-236). One f64 array per STAID, single chunk,
//! uncompressed, little-endian, C order. Index 0 = `epoch` (the reader's
//! implicit 1980-01-01); rows before `day0` and after the data are NaN.

use std::path::Path;

use chrono::NaiveDate;

use crate::data::error::{DataError, Result};

pub fn write_obs_zarr_v2(
    dir: &Path,
    staids: &[String],
    epoch: NaiveDate,
    day0: NaiveDate,
    daily: &ndarray::Array2<f32>, // (G, D) m³/s, row g = staids[g]
) -> Result<()> {
    todo!()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::data::ids::Staid;
    use crate::data::store::ObservationsStore;

    #[test]
    fn roundtrips_through_observations_store() {
        let tmp = tempfile::tempdir().unwrap();
        let epoch = NaiveDate::from_ymd_opt(1980, 1, 1).unwrap();
        let day0 = NaiveDate::from_ymd_opt(1980, 1, 4).unwrap(); // 3 NaN pad rows
        let staids = vec!["01010000".to_string(), "02020000".to_string()];
        let daily = ndarray::Array2::from_shape_vec(
            (2, 5),
            vec![1.0, 2.0, 3.0, 4.0, 5.0, 10.0, 20.0, 30.0, 40.0, 50.0],
        )
        .unwrap();
        write_obs_zarr_v2(tmp.path(), &staids, epoch, day0, &daily).unwrap();

        // The DISPATCHING open (sniff must pick zarr-v2, not icechunk).
        let store = ObservationsStore::open(tmp.path()).unwrap();
        assert!(store.contains(&Staid::new("01010000")));
        let ids = [Staid::new("01010000"), Staid::new("02020000")];
        // Read across the pad boundary: days 2..7 (0-based from epoch).
        let win = store
            .read_window_daily(NaiveDate::from_ymd_opt(1980, 1, 3).unwrap(), 6, &ids)
            .unwrap();
        assert_eq!(win.shape(), &[6, 2]);
        assert!(win[(0, 0)].is_nan()); // 1980-01-03 = pad
        assert_eq!(win[(1, 0)], 1.0); // day0
        assert_eq!(win[(1, 1)], 10.0);
        assert_eq!(win[(5, 0)], 5.0); // last data day
    }
}
```

NOTE: check `ObservationsStore`'s actual method names in `src/data/store/mod.rs` before finalizing the test — the dispatching enum may name the window reader differently than `GlobalObservationsStore::read_window_daily` (the dataset calls `observations.read_window(window, &gauge_staids)` and `observations.contains(s)`, dataset.rs:366/:520). Use whichever methods the dataset uses, with a `TimeWindow`-style argument if that's the real signature — the test must exercise the exact call path `MeritGagesDataset` uses.

- [ ] **Step 2: Run to verify it fails**

Run: `cargo test --lib roundtrips_through_observations_store`
Expected: FAIL — `todo!()` panic (or compile error until signatures align).

- [ ] **Step 3: Implement the writer**

```rust
pub fn write_obs_zarr_v2(
    dir: &Path,
    staids: &[String],
    epoch: NaiveDate,
    day0: NaiveDate,
    daily: &ndarray::Array2<f32>,
) -> Result<()> {
    let io = |e: std::io::Error| DataError::Io { path: dir.to_path_buf(), source: e };
    let (g, d) = daily.dim();
    assert_eq!(g, staids.len(), "daily rows != staids");
    let pad = (day0 - epoch).num_days();
    assert!(pad >= 0, "day0 before epoch");
    let n_time = pad as usize + d;

    std::fs::create_dir_all(dir).map_err(io)?;
    std::fs::write(dir.join(".zgroup"), r#"{"zarr_format": 2}"#).map_err(io)?;
    let zarray = format!(
        r#"{{"chunks": [{n_time}], "compressor": null, "dtype": "<f8",
"fill_value": "NaN", "filters": null, "order": "C",
"shape": [{n_time}], "zarr_format": 2}}"#
    );
    for (gi, staid) in staids.iter().enumerate() {
        let adir = dir.join(staid);
        std::fs::create_dir_all(&adir).map_err(io)?;
        std::fs::write(adir.join(".zarray"), &zarray).map_err(io)?;
        let mut bytes = Vec::with_capacity(n_time * 8);
        for _ in 0..pad {
            bytes.extend_from_slice(&f64::NAN.to_le_bytes());
        }
        for di in 0..d {
            bytes.extend_from_slice(&(daily[(gi, di)] as f64).to_le_bytes());
        }
        std::fs::write(adir.join("0"), bytes).map_err(io)?;
    }
    Ok(())
}
```

If the reader rejects `"fill_value": "NaN"`, use `"fill_value": null` (zarr-v2 spec allows both for float dtypes; the fixture used `0.0` but we must NOT fill missing with 0 — verify against the reader and pick the accepted spelling that decodes reads correctly; the chunk is fully written so fill_value is never materialized either way).

- [ ] **Step 4: Run the test + the existing obs-store tests**

Run: `cargo test --lib roundtrips_through_observations_store && cargo test --lib zarr_obs`
Expected: PASS both.

- [ ] **Step 5: Commit**

```bash
git add src/data/store/obs_writer.rs src/data/store/mod.rs
git commit -m "feat(data): zarr-v2 observations writer (ObservationsStore-compatible, NaN-padded from epoch)"
```

---

### Task 3: `--backend cpu` dispatch for `train`, `eval`, `dump_parameters`

**Files:**
- Modify: `src/bin/train.rs`, `src/bin/eval.rs`, `src/bin/dump_parameters.rs`

Copy the probe binary's dispatch pattern (src/bin/probe_zeta_gradient.rs:123-146) into each: add the CLI arg, move the existing body into a generic `fn run<I: Backend>(...)` (for train: `where Autodiff<I>: AutodiffBackend<InnerBackend = I>`, matching `bootstrap_head_and_state`'s bounds), and dispatch.

- [ ] **Step 1: Add to each binary's `Cli` struct**

```rust
    /// Backend: "cpu" (NdArray, deterministic; forces sparse_solver=cpu) or "cuda".
    #[arg(long, default_value = "cuda")]
    backend: String,
```

Default `"cuda"` — these binaries' existing users (and the 2×2 reproduce commands) must be unaffected.

- [ ] **Step 2: Genericize each `main`.** Pattern for `train.rs` (eval.rs and dump_parameters.rs are the same shape with their own bodies; each `run` takes the already-loaded `Config` so the cpu arm can mutate it):

```rust
fn main() -> Result<(), Box<dyn std::error::Error>> {
    // ...deprecation eprintln + Cli::parse() + create_dir_all unchanged...
    let mut cfg = Config::from_yaml_file_with_mode(&cli.config, ConfigMode::Training)?;
    match cli.backend.as_str() {
        "cpu" => {
            type I = burn::backend::NdArray<f32>;
            let device = <I as burn::tensor::backend::BackendTypes>::Device::default();
            cfg.params.sparse_solver = ddrs::config::SparseSolver::Cpu;
            eprintln!("backend: cpu (NdArray; sparse_solver forced to cpu)");
            run::<I>(cfg, cli, device)
        }
        "cuda" => {
            type I = burn_cuda::Cuda<f32, i32>;
            let device = cubecl::cuda::CudaDevice::new(cfg.device);
            run::<I>(cfg, cli, device)
        }
        other => Err(format!("unknown --backend {other}").into()),
    }
}
```

`run<I>` contains the previous `main` body verbatim from the `dataset` open onward (train also keeps its `<I as Backend>::seed` equivalent via `bootstrap_head_and_state`; eval keeps its explicit `seed` call, generic over `I`). `use_cuda_graphs` guard: the config already has it `false` for leakance configs; the cpu arm additionally sets `cfg.params.use_cuda_graphs = false;` defensively (CUDA graphs are meaningless on NdArray).

- [ ] **Step 3: Verify compile + behavior unchanged on default path**

Run: `cargo build --release --bin train --bin eval --bin dump_parameters && cargo test --lib`
Expected: clean build; lib tests pass.

- [ ] **Step 4: Guard sweep**

Run: `cargo test --test leakance_gradcheck --test leakance_off_parity --test zeta_accum && cargo run --release --example compare_ddr_sandbox`
Expected: all pass; ABSOLUTE MATCH.

- [ ] **Step 5: Commit**

```bash
git add src/bin/train.rs src/bin/eval.rs src/bin/dump_parameters.rs
git commit -m "feat(bins): --backend cpu dispatch for train/eval/dump_parameters (probe-binary pattern)"
```

---

### Task 4: Teacher mode in the probe binary

**Files:**
- Modify: `src/bin/probe_zeta_gradient.rs`

- [ ] **Step 1: Add CLI args + mode variant.** In `Cli`:

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

Add `Teacher` to `enum Mode`, parse `"teacher"`, map it to `ConfigMode::Testing` in the `cfg_mode` match, and route it to `run_teacher::<I>` in both backend arms.

- [ ] **Step 2: Implement `run_teacher`** (clone of `run_perturb`'s structure — same chunk loop, no perturbation, plus override/zeta/obs plumbing):

```rust
/// Plant-file row: normalized override values for one reach.
fn parse_plant_file(
    path: &Path,
) -> Result<Vec<(i64, f32, f32, f32)>, Box<dyn std::error::Error>> {
    let mut rdr = csv::Reader::from_path(path)?;
    let headers = rdr.headers()?.clone();
    let col = |name: &str| -> Result<usize, Box<dyn std::error::Error>> {
        headers
            .iter()
            .position(|h| h == name)
            .ok_or_else(|| format!("{}: missing column '{name}'", path.display()).into())
    };
    let (ci_c, ci_k, ci_d, ci_f) =
        (col("comid")?, col("k_d_norm")?, col("d_gw_norm")?, col("factor_norm")?);
    let mut rows = Vec::new();
    let mut seen = HashSet::new();
    for rec in rdr.records() {
        let rec = rec?;
        let comid: i64 = rec[ci_c].parse()?;
        if !seen.insert(comid) {
            return Err(format!("duplicate plant comid {comid}").into());
        }
        let norm = |s: &str, name: &str| -> Result<f32, Box<dyn std::error::Error>> {
            let v: f32 = s.parse()?;
            if !(0.0..=1.0).contains(&v) {
                return Err(format!("{name} {v} outside [0,1] for comid {comid}").into());
            }
            Ok(v)
        };
        rows.push((
            comid,
            norm(&rec[ci_k], "k_d_norm")?,
            norm(&rec[ci_d], "d_gw_norm")?,
            norm(&rec[ci_f], "factor_norm")?,
        ));
    }
    if rows.is_empty() {
        return Err(format!("{}: no plant rows", path.display()).into());
    }
    Ok(rows)
}

fn run_teacher<I: Backend>(
    cfg: Config,
    cli: Cli,
    device: I::Device,
) -> Result<(), Box<dyn std::error::Error>> {
    const BATCH_SIZE_DAYS: usize = 15;
    assert!(cfg.params.use_leakance, "teacher requires params.use_leakance: true");

    let plants = parse_plant_file(
        cli.plant_file.as_ref().ok_or("--plant-file is required in teacher mode")?,
    )?;
    let obs_dir = cli.obs_output.as_ref().ok_or("--obs-output required in teacher mode")?;
    let zeta_path = cli.zeta_output.as_ref().ok_or("--zeta-output required in teacher mode")?;
    let checkpoint = cli.checkpoint.as_ref().ok_or("--checkpoint required in teacher mode")?;

    let head_section = cfg.kan_head.as_ref().expect("kan_head config required");
    let head_cfg = kan_config(head_section, cfg.seed);
    let head_template: KanHead<I> = head_cfg.init::<I>(&device);
    let head = load_kan_head::<I>(&head_base(checkpoint), head_template, &device)?;
    eprintln!("loaded checkpoint: {}", head_base(checkpoint).display());

    let dataset = MeritGagesDataset::open(&cfg)?;
    let axis = dataset.time_axis().clone();
    let n_days = axis.num_days.min(cli.eval_days);
    if n_days < axis.num_days {
        eprintln!(
            "WARNING: truncated teacher window ({n_days}/{} days via --eval-days) — \
             synthetic obs will NOT cover the student training axis; timing-gate use only",
            axis.num_days
        );
    }
    let n_hours = n_days * 24;

    let probe = TestWindow::new(&axis, 0, 1);
    let probe_batch = dataset.collate_window(&probe)?;
    let n_all_gauges = probe_batch.gauge_staids.len();
    let gauge_staids: Vec<String> =
        probe_batch.gauge_staids.iter().map(|s| s.as_str().to_string()).collect();
    let network_comids: Vec<i64> = probe_batch.divide_comids.iter().map(|c| c.0).collect();

    // Fail fast: every plant COMID must be in the eval network.
    {
        let set: HashSet<i64> = network_comids.iter().copied().collect();
        let missing: Vec<i64> =
            plants.iter().map(|p| p.0).filter(|c| !set.contains(c)).collect();
        if !missing.is_empty() {
            return Err(format!("plant COMIDs not in network: {missing:?}").into());
        }
    }
    eprintln!(
        "teacher: {} plants, {n_all_gauges} gauges, {} reaches, {n_days} days",
        plants.len(),
        network_comids.len()
    );

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

    // Chunked forward with overrides + zeta accumulation (mirrors run_perturb's
    // run_one; column order is asserted stable across chunks).
    let mut zeta_sink = ZetaSums::<I>::new();
    let mut predictions_full = Array2::<f32>::zeros((n_all_gauges, n_hours));
    let n_chunks_total = n_days.div_ceil(BATCH_SIZE_DAYS);
    let (mut day_offset, mut chunk_idx) = (0usize, 0usize);
    while day_offset < n_days {
        let chunk_n = (n_days - day_offset).min(BATCH_SIZE_DAYS);
        let win = TestWindow::new(&axis, day_offset, chunk_n);
        let batch = dataset.collate_window(&win)?;
        assert!(
            batch.divide_comids.iter().map(|c| c.0).eq(network_comids.iter().copied()),
            "reach column order changed between chunks — override vectors invalid"
        );
        let tensors = batch.to_tensors::<I>(&device);
        let pred = forward_eval::<I>(
            &cfg, &tensors, &head, &device, chunk_idx > 0, Some(&mut zeta_sink), Some(&ov),
        );
        let dims = pred.dims();
        let v: Vec<f32> = pred.into_data().into_vec().unwrap();
        let pred_arr = Array2::from_shape_vec((dims[0], dims[1]), v).unwrap();
        let h_start = day_offset * 24;
        predictions_full
            .slice_mut(ndarray::s![.., h_start..h_start + win.n_hourly()])
            .assign(&pred_arr);
        eprintln!("  chunk {}/{n_chunks_total}", chunk_idx + 1);
        day_offset += chunk_n;
        chunk_idx += 1;
    }

    // tau-trim + daily downsample + drop last day (same as run_perturb).
    let pred_full_vec: Vec<f32> = predictions_full.iter().copied().collect();
    let pred_full_t: Tensor<I, 2> =
        Tensor::<I, 1>::from_floats(pred_full_vec.as_slice(), &device)
            .reshape([n_all_gauges, n_hours]);
    let daily_t = tau_trim_and_downsample(pred_full_t, cfg.params.tau);
    let dd = daily_t.dims();
    let daily_vec: Vec<f32> = daily_t.into_data().into_vec().unwrap();
    let daily_all = Array2::from_shape_vec((dd[0], dd[1]), daily_vec).unwrap();
    let daily = daily_all.slice(ndarray::s![.., 0..dd[1] - 1]).to_owned();

    // Synthetic obs: day0 = axis.start + 1 (tau-trim drops day 0).
    let epoch = chrono::NaiveDate::from_ymd_opt(1980, 1, 1).unwrap();
    let day0 = axis.start + chrono::Duration::days(1);
    ddrs::data::store::obs_writer::write_obs_zarr_v2(obs_dir, &gauge_staids, epoch, day0, &daily)?;
    eprintln!(
        "synthetic obs → {} ({n_all_gauges} gauges, {} days from {day0})",
        obs_dir.display(),
        daily.dim().1
    );

    // Answer key: zeta means over the routed window.
    let steps = zeta_sink.steps as f32;
    assert!(steps > 0.0, "zeta accumulation empty — leakance inactive?");
    let mean_vec = |t: Option<Tensor<I, 1>>| -> Vec<f32> {
        (t.expect("zeta sums present") / steps).into_data().into_vec().unwrap()
    };
    ddrs::dump_parameters::write_zeta_netcdf(
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

Add the needed imports (`ddrs::training::{forward_eval, ZetaSums}`, `ddrs::training::LeakanceOverride` — match the actual re-export paths from Task 1). `axis.start`/`axis.num_days` field names: copy from `run_perturb` (:468, :515) — they are already used there.

- [ ] **Step 3: Compile + tiny smoke via `--eval-days`**

Run: `cargo build --release --bin probe_zeta_gradient`
Expected: clean. (Functional smoke happens in Task 7's timing gate — teacher mode needs the real dataset.)

- [ ] **Step 4: Commit**

```bash
git add src/bin/probe_zeta_gradient.rs
git commit -m "feat(probe): --mode teacher — planted-leakance world, synthetic obs + zeta answer key"
```

---

### Task 5: Experiment configs + warm-start dir + guard sweep

**Files:**
- Create: `config/experiments/recoverability_teacher.yaml`, `recoverability_student_a.yaml`, `recoverability_student_b.yaml`, `recoverability_student_c.yaml`, `recoverability_measure.yaml`
- Create (runtime): `/home/tbindas/projects/ddrs/output/recoverability/init_head/head.mpk`

All five configs derive from `config/experiments/leakance_hourly_on.yaml` by a generation script so the base stays single-source. The diffs:

| Config | window | observations | checkpoint | use_leakance | sparse_solver |
|---|---|---|---|---|---|
| teacher | `testing:` 1981/09/30 → 1995/10/01 | real USGS (unchanged) | — (CLI) | true | cpu |
| student_a | `experiment:` unchanged (1981/10/01 → 1995/09/30) | synthetic | `output/recoverability/init_head` | true | cpu |
| student_b | unchanged | synthetic | `output/recoverability/init_head` | **false** | cpu |
| student_c | unchanged | synthetic | **none** | true | cpu |
| measure | `testing:` 1981/09/30 → 1995/10/01 | synthetic | — (CLI) | true | cpu |

NOTE (2026-07-03 amendment): all five configs also widen params.parameter_ranges.K_D to [1e-8, 1e-5] — see spec §Plant sites.

The teacher `testing:` window is 1 day wider on each side than the student training axis because tau-trim drops the first prediction day and the writer drops the last — the synthetic obs then cover exactly 1981-10-01..1995-09-30. All paths absolute.

- [ ] **Step 1: Write the generator** and run it from the worktree root:

```bash
python3 - <<'EOF'
import re, pathlib
base = pathlib.Path("config/experiments/leakance_hourly_on.yaml").read_text()
outdir = pathlib.Path("config/experiments")
ROOT = "/home/tbindas/projects/ddrs"
SYN = f"{ROOT}/output/recoverability/synthetic_obs"

def patch(text, subs):
    for pat, rep in subs:
        text, n = re.subn(pat, rep, text, count=1, flags=re.M)
        assert n == 1, f"pattern not found: {pat}"
    return text

cpu = [(r"^  sparse_solver: cuda$", "  sparse_solver: cpu")]
syn_obs = [(r"^  observations: .*$", f"  observations: {SYN}")]
test_win = [(r"^  start_time: 1995/10/01$", "  start_time: 1981/09/30"),
            (r"^  end_time: 2010/09/30$", "  end_time: 1995/10/01")]
warm = [(r"^experiment:$",
         f"experiment:\n  checkpoint: {ROOT}/output/recoverability/init_head")]
leak_off = [(r"^  use_leakance: true$", "  use_leakance: false")]

hdr = lambda name, note: (
    f"# {name} — synthetic losing-reach recoverability (positive control).\n"
    f"# GENERATED from leakance_hourly_on.yaml — {note}\n"
    f"# Spec: docs/superpowers/specs/2026-07-03-synthetic-recoverability-design.md\n")

variants = {
    "recoverability_teacher.yaml":
        (cpu + test_win, "teacher world: testing window = training axis ±1 day, real USGS obs"),
    "recoverability_student_a.yaml":
        (cpu + syn_obs + warm, "student A: warm-start (weights only), leakance ON, synthetic obs"),
    "recoverability_student_b.yaml":
        (cpu + syn_obs + warm + leak_off, "student B: warm-start, leakance OFF (H5 absorption control)"),
    "recoverability_student_c.yaml":
        (cpu + syn_obs, "student C: cold init, leakance ON, synthetic obs"),
    "recoverability_measure.yaml":
        (cpu + syn_obs + test_win, "measurement pass: eval-with-zeta over the training axis"),
}
for name, (subs, note) in variants.items():
    (outdir / name).write_text(hdr(name, note) + patch(base, subs))
    print("wrote", name)
EOF
```

- [ ] **Step 2: Validate all five parse and carry the right knobs**

```bash
for f in config/experiments/recoverability_*.yaml; do
  echo "== $f"
  grep -E "start_time|end_time|observations:|checkpoint:|use_leakance|sparse_solver" "$f"
done
```

Expected: teacher/measure show `1981/09/30`→`1995/10/01` under testing (experiment section keeps 1981/10/01→1995/09/30); a/b show the `checkpoint:` line; b shows `use_leakance: false`; a/b/c/measure show the synthetic obs path; all show `sparse_solver: cpu`.

- [ ] **Step 3: Create the weights-only warm-start dir** (head.mpk ONLY — no optim.mpk/state.json, so `bootstrap_head_and_state` starts Adam cold at epoch 1):

```bash
mkdir -p /home/tbindas/projects/ddrs/output/recoverability/{init_head,logs}
cp /home/tbindas/projects/ddrs/.ddrs/runs/2026-07-01T13-43-32Z-train-and-test/checkpoints/epoch_5_mb_9/head.mpk \
   /home/tbindas/projects/ddrs/output/recoverability/init_head/head.mpk
ls /home/tbindas/projects/ddrs/output/recoverability/init_head/
```

Expected: exactly `head.mpk`.

- [ ] **Step 4: Full guard sweep** (last code task before runs)

Run: `cargo test && cargo run --release --example compare_ddr_sandbox`
Expected: all tests pass (pre-existing cuda_graph doctest failures are known-identical on master and excluded from judgment); ABSOLUTE MATCH.

- [ ] **Step 5: Commit**

```bash
git add config/experiments/recoverability_*.yaml
git commit -m "config: recoverability teacher/student/measure configs (generated from leakance_hourly_on)"
```

---

### Task 6: Plant-site selection script

**Files:**
- Create: `scripts/recoverability_sites.py`

Inputs (all exist): `output/zeta_probe/probe_plan.csv` (columns `round,comid,delta,staid_nearest,class,stratum_*`), `output/zeta_probe/detectability_rows.csv` (columns `comid,delta,cls,mean_dq,peak_dq,noise,band5,detect,s_re,reach_abs`), and the hourly-ON eval diagnostic `/home/tbindas/projects/ddrs/.ddrs/runs/2026-07-01T13-43-32Z-train-and-test/kan_parameters.nc` (vars `depth_mean`, `area_z_mean` on dim `COMID_eval`).

- [ ] **Step 1: Write the script**

```python
#!/usr/bin/env python3
"""Plant plan for the synthetic recoverability positive control.

Per Ref probe reach: target = min(2*band5, 0.5*ceiling) with
ceiling = area_z_mean * K_D_MAX * (depth_mean + 2). Overrides are expressed in
NORMALIZED space: K_D at its log-range ceiling -> 1.0, d_gw at floor -> 0.0,
factor = target/ceiling (physical == normalized for [0,1]).
Drop rule (spec section 2): ceiling < 0.25*band5 -> reach excluded, logged.
"""
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr

ROOT = Path("/home/tbindas/projects/ddrs")
PROBE = ROOT / "output/zeta_probe"
OUT = ROOT / "output/recoverability"
ZETA_NC = ROOT / ".ddrs/runs/2026-07-01T13-43-32Z-train-and-test/kan_parameters.nc"
K_D_MAX = 1e-5  # widened per user decision 2026-07-03
D_GW_MIN = -2.0

plan = pd.read_csv(PROBE / "probe_plan.csv", dtype={"staid_nearest": str})
ref = plan[(plan["class"] == "Ref") & (plan["delta"] == 0.01)]
ref = ref.drop_duplicates("comid")[["comid", "staid_nearest"]]

rows = pd.read_csv(PROBE / "detectability_rows.csv")
band = rows[(rows["cls"] == "Ref") & (rows["delta"] == 0.01)][["comid", "band5"]]
band = band.drop_duplicates("comid")
sites = ref.merge(band, on="comid", how="inner")
print(f"Ref probe reaches: {len(ref)}; with band5: {len(sites)}")

ds = xr.open_dataset(ZETA_NC)
diag = pd.DataFrame({
    "comid": ds["COMID_eval"].values,
    "depth_mean": ds["depth_mean"].values,
    "area_z_mean": ds["area_z_mean"].values,
})
sites = sites.merge(diag, on="comid", how="left")
missing = sites["depth_mean"].isna()
assert not missing.any(), f"reaches missing diagnostics: {sites[missing]['comid'].tolist()}"

sites["ceiling_flux"] = sites["area_z_mean"] * K_D_MAX * (sites["depth_mean"] - D_GW_MIN)
sites["target_flux"] = np.minimum(2.0 * sites["band5"], 0.5 * sites["ceiling_flux"])
dropped = sites[sites["ceiling_flux"] < 0.25 * sites["band5"]].copy()
kept = sites[sites["ceiling_flux"] >= 0.25 * sites["band5"]].copy()

kept["k_d_norm"] = 1.0
kept["d_gw_norm"] = 0.0
kept["factor_norm"] = (kept["target_flux"] / kept["ceiling_flux"]).clip(0.0, 1.0)
assert (kept["factor_norm"] <= 0.5 + 1e-9).all(), "factor should be <=0.5 by the target rule"
assert (kept["target_flux"] > 0).all()

OUT.mkdir(parents=True, exist_ok=True)
cols = ["comid", "k_d_norm", "d_gw_norm", "factor_norm",
        "staid_nearest", "band5", "target_flux", "ceiling_flux"]
kept[cols].to_csv(OUT / "plants.csv", index=False)

def q(s):
    return " ".join(f"p{p}={np.percentile(s, p):.3e}" for p in (10, 50, 90))

report = [
    f"plant sites kept: {len(kept)}  dropped (ceiling < 0.25*band5): {len(dropped)}",
    f"band5      [m3/s]: {q(kept['band5'])}",
    f"ceiling    [m3/s]: {q(kept['ceiling_flux'])}",
    f"target     [m3/s]: {q(kept['target_flux'])}",
    f"factor_norm      : {q(kept['factor_norm'])}",
    f"targets band-limited (2*band5 < 0.5*ceiling): "
    f"{int((2 * kept['band5'] < 0.5 * kept['ceiling_flux']).sum())}/{len(kept)}",
    "dropped comids: " + (", ".join(map(str, dropped["comid"])) or "none"),
]
(OUT / "sites_report.txt").write_text("\n".join(report) + "\n")
print("\n".join(report))
print(f"\nwrote {OUT / 'plants.csv'}")
```

- [ ] **Step 2: Run it**

```bash
cd /home/tbindas/projects/ddrs/ddrs-py && uv run python \
  /home/tbindas/projects/ddrs/.claude/worktrees/zeta-sensitivity/scripts/recoverability_sites.py
```

Expected: `plants.csv` + `sites_report.txt` under `output/recoverability/`; kept count printed. **Judgment checkpoint:** if kept < 30 (most sites drop on the expressibility rule), STOP and surface to the human — the positive control would be under-powered and the P3-linked magnitude rule needs revisiting.

- [ ] **Step 3: Commit**

```bash
git add scripts/recoverability_sites.py
git commit -m "feat(scripts): recoverability plant-site selection (band5 x expressibility ceiling)"
```

---

### Task 7: Timing gates, teacher run, baseline zeta run

No new code — execution + verification. All commands run with cwd `/home/tbindas/projects/ddrs`; `WT=/home/tbindas/projects/ddrs/.claude/worktrees/zeta-sensitivity`.

- [ ] **Step 1: Teacher timing gate** (30-day truncation; expect the WARNING line; ~2 min):

```bash
nice -n 10 $WT/target/release/probe_zeta_gradient \
  --mode teacher --backend cpu \
  --config $WT/config/experiments/recoverability_teacher.yaml \
  --checkpoint .ddrs/runs/2026-07-01T13-43-32Z-train-and-test/checkpoints/epoch_5_mb_9 \
  --plant-file output/recoverability/plants.csv \
  --eval-days 30 \
  --obs-output /tmp/recov_gate_obs --zeta-output /tmp/recov_gate_zeta.nc \
  2>&1 | tee output/recoverability/logs/gate_teacher.log
```

Expected: WARNING about truncation; completes; per-chunk time × (5115/30 ≈ 171× the gate's chunks) projects the full run. If the projection exceeds 18 h, STOP and surface. Verify the gate's answer key: planted reaches have `zeta_net` >> non-planted (quick xarray check against `plants.csv` comids).

- [ ] **Step 2: Student timing gate** (2 mini-batches; ~5 min):

```bash
nice -n 10 $WT/target/release/train --backend cpu \
  --config $WT/config/experiments/recoverability_student_c.yaml \
  --checkpoint-dir /tmp/recov_gate_train --max-mini-batches 2 \
  2>&1 | tee output/recoverability/logs/gate_train.log
```

NOTE: student configs point `observations:` at the not-yet-written synthetic store — for THIS gate only, the dataset open will fail. Run the gate with the teacher config's obs (real USGS) instead: generate a throwaway gate config by `sed 's|output/recoverability/synthetic_obs|/mnt/ssd1/data/icechunk/usgs_daily_observations|'` of student_c into `/tmp/gate_student.yaml` and pass that. Expected: two `mb=... loss=...` lines; per-batch time × 480 (5 epochs × 96) projects the student run. If projection > 12 h, STOP and surface.

- [ ] **Step 3: Full teacher run** (~5 h; background) and, in parallel, the **baseline zeta run** (unmodified checkpoint over the same window, for R2):

```bash
nice -n 10 $WT/target/release/probe_zeta_gradient \
  --mode teacher --backend cpu \
  --config $WT/config/experiments/recoverability_teacher.yaml \
  --checkpoint .ddrs/runs/2026-07-01T13-43-32Z-train-and-test/checkpoints/epoch_5_mb_9 \
  --plant-file output/recoverability/plants.csv \
  --eval-days 999999 \
  --obs-output output/recoverability/synthetic_obs \
  --zeta-output output/recoverability/answer_key.nc \
  2>&1 | tee output/recoverability/logs/teacher.log

nice -n 10 $WT/target/release/eval --backend cpu \
  --config $WT/config/experiments/recoverability_teacher.yaml \
  --checkpoint .ddrs/runs/2026-07-01T13-43-32Z-train-and-test/checkpoints/epoch_5_mb_9 \
  --output /tmp/recov_baseline_eval.zarr \
  --zeta-output output/recoverability/baseline_zeta.nc \
  2>&1 | tee output/recoverability/logs/baseline_zeta.log
```

(The baseline uses the TEACHER config — real USGS obs, same window — because the synthetic store may not exist yet when it starts; eval reads obs only for the output zarr, not the routing.)

- [ ] **Step 4: Verify teacher outputs.**
  1. No truncation WARNING in `teacher.log`.
  2. Synthetic obs sanity (from `ddrs-py`): open a few gauges' arrays, assert day count == 5478 + NaN pad == (1981-10-01 − 1980-01-01), values positive and finite in-window.
  3. **Answer-key vs target:** per planted comid, `zeta_net(answer_key)` vs `target_flux` from plants.csv — expect same order of magnitude (ratio p10–p90 within [0.3, 3]; the ceiling used mean depth so scatter is fine). Record the distribution.
  4. **Gauge-resolution check (spec concern 5):** run the student gate command from Step 2 again, now with the REAL student_c config (synthetic obs). The dataset open must log the observations filter keeping the same gauge count as a real training run (compare against `gate_train.log`'s count). If gauges drop, the STAID keys mismatch — STOP and fix the writer's naming before any student run.

- [ ] **Step 5: Commit** (logs/artifacts are runtime outputs in the main tree — nothing to commit unless a fix was needed; commit any fix with its own message).

---

### Task 8: Student training runs A, B, C (parallel)

- [ ] **Step 1: Launch all three in parallel** (~3–4 h each; same machine, `nice`d):

```bash
cd /home/tbindas/projects/ddrs
for s in a b c; do
  nice -n 10 $WT/target/release/train --backend cpu \
    --config $WT/config/experiments/recoverability_student_$s.yaml \
    --checkpoint-dir output/recoverability/students/$s \
    > output/recoverability/logs/student_$s.log 2>&1 &
done
wait
```

- [ ] **Step 2: Verify boot lines.** `student_a.log` and `student_b.log` must contain `warm start: loaded KAN head from .../init_head/head` AND `no .../optim.mpk — Adam starts cold` AND `restarting at epoch 1 with a fresh shuffle`; `student_c.log` must contain NONE of them. All three logs show the same gauge count as the real run and `epoch 1 lr=0.001`.

- [ ] **Step 3: Verify completion.** Each log ends with `Training complete`; each `students/<s>/` contains checkpoint DIRECTORIES (e.g. `epoch_5_mb_96/` with `head.mpk` inside — flat `.mpk` files would mean a stale binary ran). Record wall-clock per student.

- [ ] **Step 4: Sanity — A's first-epoch loss should start SMALL** (warm start on self-generated obs; the residual is only the planted signal) while C's starts large. Grep the first `mb=1 loss=` of each and record. If A's initial loss is NOT far below C's, the synthetic world is inconsistent (obs/forcing mismatch) — STOP and investigate before burning the measurement passes.

---

### Task 9: Measurement passes + analysis script + verdicts

**Files:**
- Create: `scripts/recoverability_analysis.py`

- [ ] **Step 1: Measurement passes** (A and C zeta fields; B has no zeta by construction). Find each student's FINAL checkpoint dir (`ls -d output/recoverability/students/<s>/epoch_5_mb_* | sort -V | tail -1`):

```bash
for s in a c; do
  CKPT=$(ls -d output/recoverability/students/$s/epoch_5_mb_* | sort -V | tail -1)
  nice -n 10 $WT/target/release/eval --backend cpu \
    --config $WT/config/experiments/recoverability_measure.yaml \
    --checkpoint "$CKPT" \
    --output output/recoverability/eval_$s.zarr \
    --zeta-output output/recoverability/zeta_$s.nc \
    2>&1 | tee output/recoverability/logs/measure_$s.log &
done
wait
```

- [ ] **Step 2: R4 inputs — dump_parameters for B and the original head** (Manning's n per COMID):

```bash
CKPT_B=$(ls -d output/recoverability/students/b/epoch_5_mb_* | sort -V | tail -1)
nice -n 10 $WT/target/release/dump_parameters --backend cpu \
  --config $WT/config/experiments/recoverability_student_b.yaml \
  --checkpoint "$CKPT_B/head" \
  --output output/recoverability/params_b.nc
nice -n 10 $WT/target/release/dump_parameters --backend cpu \
  --config $WT/config/experiments/recoverability_student_b.yaml \
  --checkpoint output/recoverability/init_head/head \
  --output output/recoverability/params_orig.nc
```

(Check `dump_parameters --help` for the exact `--checkpoint` form — it wants the recorder BASE without `.mpk`, src/bin/dump_parameters.rs:21-23.)

- [ ] **Step 3: Write the analysis script**

```python
#!/usr/bin/env python3
"""Pre-registered verdicts R1-R5 for the synthetic recoverability control.

Spec section 3 of 2026-07-03-synthetic-recoverability-design.md. Bars are
FIXED there; this script only reports.
"""
import re
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr

ROOT = Path("/home/tbindas/projects/ddrs")
OUT = ROOT / "output/recoverability"

plants = pd.read_csv(OUT / "plants.csv")
planted = set(plants["comid"])

def zeta_frame(path, prefix):
    ds = xr.open_dataset(path)
    return pd.DataFrame({
        "comid": ds["COMID_eval"].values,
        f"{prefix}_net": ds["zeta_net"].values,
        f"{prefix}_abs": ds["zeta"].values,
    })

key = zeta_frame(OUT / "answer_key.nc", "key")
base = zeta_frame(OUT / "baseline_zeta.nc", "base")
za = zeta_frame(OUT / "zeta_a.nc", "a")
zc = zeta_frame(OUT / "zeta_c.nc", "c")
df = key.merge(base, on="comid").merge(za, on="comid").merge(zc, on="comid")
df["planted"] = df["comid"].isin(planted)
n_p = int(df["planted"].sum())
assert n_p == len(planted), f"planted rows {n_p} != plan {len(planted)}"

# R1 recovery ratio (planted reaches, run A vs answer key)
p = df[df["planted"]]
ratio = (p["a_net"] / p["key_net"]).values
r1 = float(np.median(ratio))
r1_verdict = "RECOVERED" if r1 >= 0.5 else ("FAILED" if r1 <= 0.1 else "PARTIAL")

# R2 spatial precision (non-planted |zeta_net|, A vs baseline field)
np_a = float(np.median(np.abs(df.loc[~df["planted"], "a_net"])))
np_b = float(np.median(np.abs(df.loc[~df["planted"], "base_net"])))
r2_ratio = np_a / np_b if np_b > 0 else float("inf")
r2_verdict = "PRECISE" if r2_ratio < 2.0 else "SMEARED"

# R3 absorption gap: mean final-epoch loss from student logs
def final_epoch_mean_loss(log):
    text = Path(log).read_text()
    epochs = re.findall(r"^epoch (\d+) ", text, re.M)
    last = epochs[-1]
    seg = text.split(f"epoch {last} ")[-1]
    losses = [float(m) for m in re.findall(r"mb=\d+ loss=([0-9.eE+-]+)", seg)]
    assert losses, f"no losses parsed from {log}"
    return float(np.mean(losses)), len(losses)

loss_a, na = final_epoch_mean_loss(OUT / "logs/student_a.log")
loss_b, nb = final_epoch_mean_loss(OUT / "logs/student_b.log")
rel_gap = (loss_b - loss_a) / loss_b if loss_b else 0.0
r3_verdict = "A<B (leakance needed)" if rel_gap > 0.05 else (
    "A~B (H5 absorption confirmed)" if abs(rel_gap) <= 0.05 else "B<A (INVESTIGATE)")

# R4 absorption map (descriptive): where did B move Manning's n?
po = xr.open_dataset(OUT / "params_orig.nc")
pb = xr.open_dataset(OUT / "params_b.nc")
dn = pd.DataFrame({"comid": po["COMID"].values,
                   "dn": pb["n"].values - po["n"].values})
dn_planted = dn[dn["comid"].isin(planted)]["dn"]
r4 = (f"median dn planted={np.median(dn_planted):.3e} "
      f"all={np.median(dn['dn']):.3e} "
      f"p90|dn| planted={np.percentile(np.abs(dn_planted), 90):.3e}")

# R5 cold emergence
c_p = float(np.median(np.abs(p["c_net"])))
c_np = float(np.median(np.abs(df.loc[~df["planted"], "c_net"])))
r5_ratio = c_p / c_np if c_np > 0 else float("inf")
r5_verdict = "EMERGES" if r5_ratio > 3.0 else "SUPPRESSED"

df.to_csv(OUT / "recovery_rows.csv", index=False)

print("=" * 72)
print("VERDICTS (bars pre-registered in the spec)")
print("=" * 72)
print(f"  [R1 {r1_verdict}] recovery ratio median={r1:.3f} "
      f"(p10={np.percentile(ratio,10):.3f} p90={np.percentile(ratio,90):.3f}, "
      f"n={n_p}; bar: >=0.5 recovered, <=0.1 failed)")
print(f"  [R2 {r2_verdict}] non-planted |zeta_net| A/baseline = {r2_ratio:.2f} "
      f"(A={np_a:.3e}, base={np_b:.3e}; bar: <2)")
print(f"  [R3 {r3_verdict}] final-epoch mean loss A={loss_a:.5f} (n={na}) "
      f"B={loss_b:.5f} (n={nb}) rel gap={rel_gap:+.1%} (bar: 5%)")
print(f"  [R4] {r4}")
print(f"  [R5 {r5_verdict}] cold planted/non-planted |zeta_net| = {r5_ratio:.2f} (bar: >3)")
headline = "PASS" if (r1 >= 0.5 and rel_gap > 0.05) else "FAIL"
print(f"\n  HEADLINE: positive control {headline} "
      f"(requires R1>=0.5 AND A beats B)")
print(f"\nper-reach rows -> {OUT / 'recovery_rows.csv'}")
```

- [ ] **Step 4: Run it**

```bash
cd /home/tbindas/projects/ddrs/ddrs-py && uv run python \
  $WT/scripts/recoverability_analysis.py 2>&1 | tee \
  /home/tbindas/projects/ddrs/output/recoverability/logs/verdicts.log
```

Expected: the VERDICTS block prints with all five rows and the HEADLINE. If any input is missing or a parse assert fires, fix the input path/regex — do NOT soften asserts.

- [ ] **Step 5: Commit**

```bash
git add scripts/recoverability_analysis.py
git commit -m "feat(scripts): recoverability verdicts R1-R5 (pre-registered bars)"
```

---

### Task 10: Recovery maps notebook

**Files:**
- Create: `scripts/notebooks/recovery_maps.ipynb`

Follow the `ddrs-eval-plots` skill conventions (same as the committed `scripts/notebooks/gradient_maps.ipynb` — read it first; reuse its MERIT-shapefile join and CONUS basemap cells). Cells:

1. Markdown header: checkpoints, inputs (`recovery_rows.csv`, `plants.csv`), date, spec link.
2. Load `recovery_rows.csv` + `plants.csv`; join gauge lat/lon the same way `gradient_maps.ipynb` does for its gauge scatter (via the probe plan's `staid_nearest` and the gage CSV).
3. **Panel 1 — recovery-ratio gauge scatter (CONUS):** planted sites colored by `a_net/key_net`, diverging colormap (`RdBu_r`, vcenter=0.5, vmin=0, vmax=1.5), CONUS bounds `xlim=(-125,-66)`, `ylim=(24,53)`, CartoDB.Positron basemap alpha 0.6.
4. **Panel 2 — Δn absorption map:** per-reach `dn` (from `params_b.nc` − `params_orig.nc`) on the MERIT reach map around planted basins, colormap `plasma_r`.
5. Save both to `/home/tbindas/projects/ddrs/output/recoverability/plots/recovery_ratio_map.png` and `absorption_dn_map.png` (`dpi=300, bbox_inches="tight", facecolor="white"`).

- [ ] **Step 1: Write the notebook** (mirror `gradient_maps.ipynb`'s structure/imports).
- [ ] **Step 2: Execute it**

```bash
cd /home/tbindas/projects/ddrs/ddrs-py && uv run jupyter nbconvert --to notebook --execute \
  $WT/scripts/notebooks/recovery_maps.ipynb --output recovery_maps \
  --output-dir /home/tbindas/projects/ddrs/output/recoverability/plots
```

Expected: both PNGs exist; list their absolute paths.

- [ ] **Step 3: Commit**

```bash
git add scripts/notebooks/recovery_maps.ipynb
git commit -m "feat(notebooks): CONUS recovery-ratio + absorption dn maps"
```

---

### Task 11: Findings report + final guard sweep

**Files:**
- Create: `docs/2026-07-XX-synthetic-recoverability-findings.md` (XX = actual run date)

- [ ] **Step 1: Write the findings doc** in the established structure (mirror `docs/2026-07-03-zeta-gradient-probe-findings.md`): §1 Hypothesis (the positive-control question + R1–R5 table with bars), §2 What was changed to test it (LeakanceOverride, obs writer, teacher mode, backend dispatch, configs, scripts — cite commits), §3 The experiment (teacher world numbers: plants kept/dropped, target distribution, window; student roster + wall-clocks), §4 Did the test pass or fail (verdicts table verbatim from `verdicts.log`, per-verdict interpretation), §5 Conclusions, §6 Next steps (what the verdict means for the auxiliary-constraint experiment), §7 Raw verdict output (byte-copied from `verdicts.log`), §8 Reproduce (the exact commands from Tasks 6–10). Every number must come from an artifact, not memory.

- [ ] **Step 2: Final guard sweep**

Run: `cargo test && cargo run --release --example compare_ddr_sandbox`
Expected: green + ABSOLUTE MATCH.

- [ ] **Step 3: Commit**

```bash
git add docs/2026-07-*-synthetic-recoverability-findings.md
git commit -m "docs(findings): synthetic recoverability positive control — results"
```

---

## Self-review (done at write time)

- **Spec coverage:** §2 teacher/plants/roster → Tasks 4/5/6/7/8; §3 verdicts → Task 9 (bars match: 0.5/0.1, 2×, 5%, 3×); §4.1 → Tasks 1/2/4; §4.2 → obsoleted by plan deviation 1; §4.3 → obsoleted by deviation 2; §4.4 → Task 6; §4.5 → Task 9; §4.6 → Task 10; §4.7 → Task 11; §5 gates/parallelism → Tasks 7/8; §6 concern 5 → Task 7 Step 4.4; §7 tests → Tasks 1/2 + guard sweeps in 3/5/11.
- **Known unknowns flagged inline** (not placeholders — each has a resolution path): `ObservationsStore` dispatching method names (Task 2 Step 1 note), zarr fill_value spelling (Task 2 Step 3), `dump_parameters --checkpoint` base form (Task 9 Step 2), exact `ZetaSums` re-export path (Task 4 Step 2).
- **Type consistency:** `LeakanceOverride` fields (`mask/k_d/d_gw/factor`, `Vec<f32>`) match between Tasks 1 and 4; `write_obs_zarr_v2(dir, staids, epoch, day0, daily)` matches between Tasks 2 and 4; config filenames match between Tasks 5 and 7–9.

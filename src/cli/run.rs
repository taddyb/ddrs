use std::fs;
use std::panic::AssertUnwindSafe;
use std::path::{Path, PathBuf};
use std::process::Command;
use std::time::Instant;

use burn::backend::Autodiff;
use burn::tensor::backend::BackendTypes;
use burn_cuda::Cuda;

use crate::cli::{
    manifest::{GitInfo, Manifest, RunOutputs, SourceLockRef},
    plan::{plan, PlanResult},
    types::{RunStatus, Workflow},
    workspace::Workspace,
};
use crate::config::{Config, ConfigMode};
use crate::data::dataset::MeritGagesDataset;
use crate::error::CliError;
use crate::nn::kan_head::{KanHead, KanHeadConfig};
use crate::training::bootstrap::bootstrap_head_and_state;
use crate::training::checkpoint::load_kan_head;
use crate::training::driver::train as training_train;
use crate::training::optimizer::build_adam;
use crate::training::{evaluate, write_predictions_zarr, EvalParams, ZarrAttrs};

pub struct RunInput {
    pub workspace: Workspace,
    pub config_path: PathBuf,
    pub workflow: Option<Workflow>,
    pub plot: bool,
    pub strict: bool,
    pub max_mini_batches: Option<usize>,
}

pub fn run(input: RunInput) -> Result<PathBuf, CliError> {
    // 1. Plan as a library call (reused — not re-parsed in run).
    let pr: PlanResult = plan(&input.config_path, input.workflow, &input.workspace)?;

    // 1b. GPU pre-flight for workflows that need training kernels.
    if matches!(pr.workflow, Workflow::Train | Workflow::TrainAndTest) {
        let has_gpu = crate::cli::system::probe()
            .ok()
            .flatten()
            .map(|p| !p.gpu.is_empty())
            .unwrap_or(false);
        if !has_gpu {
            return Err(CliError::Runtime(format!(
                "run: workflow `{}` requires a CUDA GPU; system probe found none. \
                 Smoke verified the routing core works on CPU, but production \
                 training does not.",
                workflow_slug(pr.workflow)
            )));
        }
    }

    // 2. Drift policy.
    if !pr.drift.is_empty() {
        if input.strict {
            return Err(CliError::LockDrift { fields: pr.drift });
        }
        eprintln!("warning: data source drift: {:?}", pr.drift);
    }

    // 3. Create run directory.
    let started_at = chrono::Utc::now().to_rfc3339_opts(chrono::SecondsFormat::Secs, true);
    let run_id = format!("{}-{}", started_at.replace(':', "-"), workflow_slug(pr.workflow));
    let run_dir = input.workspace.runs_dir().join(&run_id);
    fs::create_dir_all(run_dir.join("checkpoints"))?;
    fs::copy(&input.config_path, run_dir.join("config.yaml"))?;
    let _ = copy_cargo_lock_if_reachable(&run_dir);
    eprintln!("run output → {}", run_dir.display());

    // 4. Dispatch to the workflow (in-process, v1 — no stdout/stderr tee
    // beyond what training::driver already prints to terminal). Tee'd
    // log capture is deferred to v1.1 alongside a run-as-subprocess
    // refactor; the manifest schema already supports stdout.log/stderr.log
    // paths but we don't populate them yet.
    let (status, exit_reason, metrics, mut outputs) = dispatch(&input, &pr, &run_dir);

    // 5. --plot post-step: call dump_parameters::dump if a checkpoint exists.
    //    Output format: kan_parameters.nc (NetCDF — the dump_parameters
    //    body writes NetCDF, not CSV, despite the spec/plan's older "CSV"
    //    framing). See dump_parameters.rs for details.
    if input.plot && matches!(status, RunStatus::Ok) {
        if let Some(ck_base) = latest_checkpoint_base(&run_dir.join("checkpoints")) {
            let plot_dir = run_dir.join("plot");
            fs::create_dir_all(&plot_dir).ok();
            let nc = plot_dir.join("kan_parameters.nc");
            type I = burn_cuda::Cuda<f32, i32>;
            let device = <I as burn::tensor::backend::BackendTypes>::Device::default();
            let res = crate::dump_parameters::dump::<I>(&pr.config, &ck_base, &nc, 50_000, &device);
            if let Err(e) = res {
                eprintln!("warning: --plot post-step failed: {e}");
            } else {
                outputs.plot = Some(PathBuf::from("plot/kan_parameters.nc"));
                eprintln!(
                    "plot NetCDF written to {}. To visualize:\n  \
                     jupyter run ~/projects/ddr/examples/merit/plot_parameter_map.ipynb \
                     --nc {}",
                    nc.display(),
                    nc.display(),
                );
            }
        }
    }

    // 6. Finalize manifest.json.
    let manifest = Manifest {
        run_id: run_id.clone(),
        ddrs_version: env!("CARGO_PKG_VERSION").into(),
        git: capture_git(),
        workflow: pr.workflow,
        config_path: run_dir.join("config.yaml"),
        started_at,
        finished_at: Some(chrono::Utc::now().to_rfc3339_opts(chrono::SecondsFormat::Secs, true)),
        status,
        exit_reason,
        system: Default::default(),
        sources: pr.sources.clone(),
        source_lock: SourceLockRef {
            lockfile: input.workspace.lockfile(),
            matched: pr.drift.is_empty(),
            drift: pr.drift.clone(),
        },
        outputs,
        metrics,
        max_mini_batches: input.max_mini_batches,
    };
    manifest.write_atomic(&run_dir.join("manifest.json"))?;
    Ok(run_dir)
}

fn workflow_slug(w: Workflow) -> &'static str {
    match w {
        Workflow::Train => "train",
        Workflow::Eval => "eval",
        Workflow::TrainAndTest => "train-and-test",
    }
}

fn latest_checkpoint_base(dir: &Path) -> Option<PathBuf> {
    // Returns the path WITHOUT the `.mpk` suffix (CompactRecorder appends it).
    let mut entries: Vec<_> = fs::read_dir(dir).ok()?
        .filter_map(Result::ok)
        .map(|e| e.path())
        .filter(|p| p.extension().map(|e| e == "mpk").unwrap_or(false))
        .collect();
    entries.sort();
    let latest = entries.pop()?;
    Some(latest.with_extension(""))
}

fn copy_cargo_lock_if_reachable(run_dir: &Path) -> std::io::Result<()> {
    for p in [Path::new("Cargo.lock"), Path::new("../Cargo.lock")] {
        if p.is_file() {
            return fs::copy(p, run_dir.join("Cargo.lock")).map(|_| ());
        }
    }
    Ok(())
}

fn capture_git() -> GitInfo {
    fn out(args: &[&str]) -> String {
        Command::new("git").args(args).output().ok()
            .and_then(|o| String::from_utf8(o.stdout).ok())
            .map(|s| s.trim().to_string())
            .unwrap_or_default()
    }
    let dirty = !out(&["status", "--porcelain"]).is_empty();
    GitInfo {
        sha: out(&["rev-parse", "HEAD"]),
        dirty,
        branch: out(&["rev-parse", "--abbrev-ref", "HEAD"]),
    }
}

/// Execute the requested workflow in-process.
///
/// Wraps the body in `catch_unwind` so a thread panic (e.g. CUDA init
/// failure) produces `RunStatus::Failed` with a populated `exit_reason`
/// rather than crashing before the manifest is written.
fn dispatch(
    input: &RunInput,
    pr: &PlanResult,
    run_dir: &Path,
) -> (RunStatus, Option<String>, serde_json::Value, RunOutputs) {
    type I = Cuda<f32, i32>;
    type AB = Autodiff<I>;

    let result = std::panic::catch_unwind(AssertUnwindSafe(|| -> Result<(serde_json::Value, RunOutputs), CliError> {
        match pr.workflow {
            Workflow::Eval => {
                return Err(CliError::Runtime(
                    "standalone --workflow eval needs a --from-run <run-id> flag; \
                     use --workflow train-and-test for now"
                        .into(),
                ));
            }
            Workflow::Train => {
                let device = <I as BackendTypes>::Device::default();
                let phase1_start = Instant::now();

                let train_cfg = Config::from_yaml_file_with_mode(&input.config_path, ConfigMode::Training)
                    .map_err(|e| CliError::Other(Box::new(e)))?;
                let train_dataset = MeritGagesDataset::open(&train_cfg)
                    .map_err(|e| CliError::Other(Box::new(e)))?;

                let head_section = train_cfg.kan_head.as_ref().expect("kan_head config required");
                let _head_cfg = KanHeadConfig::new(
                    head_section.input_var_names.clone(),
                    head_section.learnable_parameters.clone(),
                    train_cfg.seed,
                )
                .with_hidden_size(head_section.hidden_size)
                .with_num_hidden_layers(head_section.num_hidden_layers)
                .with_grid(head_section.grid)
                .with_k(head_section.k);

                let (_, mut state) = bootstrap_head_and_state::<I>(&train_cfg, &device);
                let mut optimizer = build_adam::<KanHead<AB>, AB>();

                let ckpt_dir = run_dir.join("checkpoints");
                training_train::<I>(
                    &train_cfg,
                    &train_dataset,
                    &mut state,
                    &mut optimizer,
                    &device,
                    &ckpt_dir,
                    input.max_mini_batches,
                )
                .map_err(|e| CliError::Other(Box::new(e)))?;

                let phase1_elapsed = phase1_start.elapsed();
                let epochs_completed = state.epoch.saturating_sub(1);
                let final_mini_batch = state.mini_batch;

                drop(optimizer);
                drop(state);
                drop(train_dataset);

                let metrics = serde_json::json!({
                    "epochs_completed": epochs_completed,
                    "final_mini_batch": final_mini_batch,
                    "phase1_seconds": phase1_elapsed.as_secs_f32(),
                });
                let outputs = RunOutputs {
                    checkpoints: list_mpk_files(&ckpt_dir),
                    plot: None,
                    eval_zarr: None,
                };
                Ok((metrics, outputs))
            }
            Workflow::TrainAndTest => {
                let device = <I as BackendTypes>::Device::default();

                // --- Phase 1: training ---
                let phase1_start = Instant::now();
                let train_cfg = Config::from_yaml_file_with_mode(&input.config_path, ConfigMode::Training)
                    .map_err(|e| CliError::Other(Box::new(e)))?;
                let train_dataset = MeritGagesDataset::open(&train_cfg)
                    .map_err(|e| CliError::Other(Box::new(e)))?;

                let head_section = train_cfg.kan_head.as_ref().expect("kan_head config required");
                let head_cfg = KanHeadConfig::new(
                    head_section.input_var_names.clone(),
                    head_section.learnable_parameters.clone(),
                    train_cfg.seed,
                )
                .with_hidden_size(head_section.hidden_size)
                .with_num_hidden_layers(head_section.num_hidden_layers)
                .with_grid(head_section.grid)
                .with_k(head_section.k);

                let (_, mut state) = bootstrap_head_and_state::<I>(&train_cfg, &device);
                let mut optimizer = build_adam::<KanHead<AB>, AB>();

                let ckpt_dir = run_dir.join("checkpoints");
                training_train::<I>(
                    &train_cfg,
                    &train_dataset,
                    &mut state,
                    &mut optimizer,
                    &device,
                    &ckpt_dir,
                    input.max_mini_batches,
                )
                .map_err(|e| CliError::Other(Box::new(e)))?;

                let phase1_elapsed = phase1_start.elapsed();
                let epochs_completed = state.epoch.saturating_sub(1);
                let final_mini_batch = state.mini_batch;

                drop(optimizer);
                drop(state);
                drop(train_dataset);

                // --- Phase 2: testing ---
                let phase2_start = Instant::now();
                let test_cfg = Config::from_yaml_file_with_mode(&input.config_path, ConfigMode::Testing)
                    .map_err(|e| CliError::Other(Box::new(e)))?;
                let test_dataset = MeritGagesDataset::open(&test_cfg)
                    .map_err(|e| CliError::Other(Box::new(e)))?;

                let latest = latest_checkpoint_base(&ckpt_dir)
                    .ok_or_else(|| CliError::Runtime("no .mpk checkpoints found after Phase 1".into()))?;

                let head_template: KanHead<I> = head_cfg.init::<I>(&device);
                let head = load_kan_head::<I>(&latest, head_template, &device)
                    .map_err(|e| CliError::Other(Box::new(e)))?;

                // In Testing mode, experiment.batch_size carries DAYS (not gauges)
                // because the testing: overlay sets `batch_size: 15`.
                let batch_size_days = test_cfg
                    .experiment
                    .as_ref()
                    .map(|e| e.batch_size)
                    .unwrap_or(15);

                let output = evaluate::<I>(
                    &test_cfg,
                    &test_dataset,
                    EvalParams::KanHead(&head),
                    &device,
                    batch_size_days,
                )
                .map_err(|e| CliError::Other(Box::new(e)))?;

                let phase2_elapsed = phase2_start.elapsed();

                let eval_dir = run_dir.join("eval");
                fs::create_dir_all(&eval_dir)?;
                let zarr_path = eval_dir.join("predictions.zarr");

                let exp = test_cfg.experiment.as_ref().unwrap();
                let gages_csv_path = test_cfg.data_sources.as_ref().unwrap().gages.clone();
                write_predictions_zarr(
                    &zarr_path,
                    &output,
                    ZarrAttrs {
                        start_time: &exp.start_time,
                        end_time: &exp.end_time,
                        version: env!("CARGO_PKG_VERSION"),
                        evaluation_basins_file: &gages_csv_path,
                        model_label: &latest.display().to_string(),
                    },
                )
                .map_err(|e| CliError::Other(Box::new(e)))?;

                let nse_clean: Vec<f32> = output
                    .metrics
                    .nse
                    .iter()
                    .copied()
                    .filter(|v| v.is_finite())
                    .collect();
                let mean_nse = nse_clean.iter().sum::<f32>() / (nse_clean.len() as f32).max(1.0);
                let median_nse = {
                    let mut sorted = nse_clean.clone();
                    sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());
                    if sorted.is_empty() { f32::NAN } else { sorted[sorted.len() / 2] }
                };

                let metrics = serde_json::json!({
                    "epochs_completed": epochs_completed,
                    "final_mini_batch": final_mini_batch,
                    "phase1_seconds": phase1_elapsed.as_secs_f32(),
                    "phase2_seconds": phase2_elapsed.as_secs_f32(),
                    "n_gauges_finite_nse": nse_clean.len(),
                    "n_gauges_total": output.metrics.nse.len(),
                    "mean_nse_finite": mean_nse,
                    "median_nse_finite": median_nse,
                });
                let outputs = RunOutputs {
                    checkpoints: list_mpk_files(&ckpt_dir),
                    plot: None,
                    eval_zarr: Some(PathBuf::from("eval/predictions.zarr")),
                };
                Ok((metrics, outputs))
            }
        }
    }));

    match result {
        Ok(Ok((metrics, outputs))) => (RunStatus::Ok, None, metrics, outputs),
        Ok(Err(e)) => (
            RunStatus::Failed,
            Some(e.to_string()),
            serde_json::json!({}),
            RunOutputs::default(),
        ),
        Err(_) => (
            RunStatus::Failed,
            Some("workflow panicked".into()),
            serde_json::json!({}),
            RunOutputs::default(),
        ),
    }
}

/// Returns paths to all `.mpk` files in `dir`, relative to `dir`'s parent
/// (i.e. `checkpoints/epoch_5_mb_0.mpk`).
fn list_mpk_files(dir: &Path) -> Vec<PathBuf> {
    let Ok(entries) = fs::read_dir(dir) else { return vec![] };
    let dir_name = dir.file_name().unwrap_or_default();
    let mut paths: Vec<PathBuf> = entries
        .filter_map(Result::ok)
        .map(|e| e.path())
        .filter(|p| p.extension().map(|e| e == "mpk").unwrap_or(false))
        .map(|p| {
            let fname = p.file_name().unwrap_or_default();
            PathBuf::from(dir_name).join(fname)
        })
        .collect();
    paths.sort();
    paths
}

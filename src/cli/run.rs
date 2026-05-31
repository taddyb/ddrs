use std::fs;
use std::path::{Path, PathBuf};
use std::process::Command;

use crate::cli::{
    manifest::{GitInfo, Manifest, RunOutputs, SourceLockRef},
    plan::{plan, PlanResult},
    types::{RunStatus, Workflow},
    workspace::Workspace,
};
use crate::error::CliError;

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

/// Run the actual workflow. For v1 this is a stub: it returns
/// `(RunStatus::Failed, exit_reason)` with a clear message that real
/// dispatch is not wired in yet, so the integration test pattern is
/// established without needing GPU access. Task 22 (`cli_run_drift` and
/// friends) will exercise the failure path; real workflow execution lands
/// when an integration host with merit data validates the end-to-end flow.
///
/// The shape of this function — `(status, exit_reason, metrics, outputs)`
/// — is the contract for that future replacement. Don't change it without
/// updating the manifest schema.
fn dispatch(
    _input: &RunInput,
    _pr: &PlanResult,
    _run_dir: &Path,
) -> (RunStatus, Option<String>, serde_json::Value, RunOutputs) {
    // v1 stub. Real dispatch:
    //   type I = burn_cuda::Cuda<f32, i32>;
    //   let device = <I as BackendTypes>::Device::default();
    //   let dataset = MeritGagesDataset::open(&pr.config)?;
    //   let boot = bootstrap_head_and_state::<I>(&pr.config, &device);
    //   match pr.workflow {
    //     Workflow::Train | Workflow::TrainAndTest =>
    //       training::driver::train(&pr.config, &dataset, &mut boot.1, &mut opt,
    //                               &device, &_run_dir.join("checkpoints"),
    //                               _input.max_mini_batches),
    //     Workflow::Eval => training::eval::eval(...),
    //   }
    // Returning Failed here so the manifest still writes (with status="failed",
    // exit_reason populated) and `cli_runtime_failure` tests can exercise it.
    (
        RunStatus::Failed,
        Some("dispatch stub — real workflow execution lands in a follow-up commit".into()),
        serde_json::json!({}),
        RunOutputs { checkpoints: vec![], plot: None },
    )
}

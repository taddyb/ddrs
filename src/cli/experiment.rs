//! `ddrs experiment <name>` — run a paper study from a checked-in bundle over
//! already-trained runs. See `src/experiment/mod.rs`.

use std::path::{Path, PathBuf};

use crate::cli::workspace::Workspace;
use crate::cli::{tee, CliError};
use crate::experiment::{adjoint, resolve_arm, ExperimentManifest, ExperimentRun, ExperimentSpec, ResolvedArm};

pub struct ExperimentInput {
    pub workspace: Workspace,
    pub name: String,
    /// Bundle directory; defaults to `experiments/<name>` relative to cwd.
    pub bundle: Option<PathBuf>,
    pub backend: String,
    pub arms: Option<Vec<String>>,
    pub max_gauges: Option<usize>,
    pub skip_validate: bool,
    /// Arms run concurrently, one thread each (default: all selected arms).
    pub jobs: Option<usize>,
    /// Select gauges, write gauges.csv, and stop.
    pub dry_run: bool,
}

pub fn run_experiment(input: ExperimentInput) -> Result<PathBuf, CliError> {
    let bundle = input
        .bundle
        .clone()
        .unwrap_or_else(|| PathBuf::from("experiments").join(&input.name));
    let spec_path = bundle.join("experiment.yaml");
    let spec = ExperimentSpec::load(&bundle)?;

    if let Some(filter) = &input.arms {
        for name in filter {
            if !spec.arms.iter().any(|a| &a.name == name) {
                return Err(CliError::ConfigInvalid {
                    path: spec_path.clone(),
                    source: format!("--arms names unknown arm `{name}`").into(),
                });
            }
        }
    }
    let mut arms: Vec<ResolvedArm> = Vec::new();
    for a in spec
        .arms
        .iter()
        .filter(|a| input.arms.as_ref().map_or(true, |f| f.contains(&a.name)))
    {
        arms.push(resolve_arm(&input.workspace, a)?);
    }

    let run = ExperimentRun::create(&input.workspace, &input.name)?;
    std::fs::copy(&spec_path, run.dir.join("experiment.yaml"))?;
    let mut manifest = ExperimentManifest {
        study: spec.study.clone(),
        started_utc: run.started_utc.clone(),
        finished_utc: None,
        status: "running".into(),
        bundle_dir: bundle.clone(),
        spec: spec.clone(),
        arms: arms.clone(),
        git: crate::cli::run::capture_git(),
        backend: input.backend.clone(),
        ddrs_version: env!("CARGO_PKG_VERSION").into(),
        validation: None,
        notes: vec![],
    };
    let manifest_path = run.dir.join("manifest.json");
    manifest.write(&manifest_path)?;
    eprintln!("experiment `{}` → {}", input.name, run.dir.display());

    let log_path = run.dir.join("run.log");
    let outcome = tee::tee_to(&log_path, || dispatch(&spec, &spec_path, &arms, &run.dir, &input, &mut manifest));

    manifest.finished_utc = Some(chrono::Utc::now().format("%Y-%m-%dT%H-%M-%SZ").to_string());
    manifest.status = if outcome.is_ok() { "success".into() } else { "failed".into() };
    if let Err(e) = &outcome {
        manifest.notes.push(e.to_string());
    }
    manifest.write(&manifest_path)?;
    outcome?;
    Ok(run.dir)
}

fn dispatch(
    spec: &ExperimentSpec,
    spec_path: &Path,
    arms: &[ResolvedArm],
    out_dir: &Path,
    input: &ExperimentInput,
    manifest: &mut ExperimentManifest,
) -> Result<(), CliError> {
    match spec.study.as_str() {
        "adjoint" => {
            let adj = spec.adjoint.as_ref().ok_or_else(|| CliError::ConfigInvalid {
                path: spec_path.to_path_buf(),
                source: "study: adjoint requires an `adjoint:` block".into(),
            })?;
            let opts = adjoint::AdjointOptions {
                max_gauges: input.max_gauges,
                skip_validate: input.skip_validate,
                force_cpu: input.backend == "cpu",
                jobs: input.jobs.unwrap_or(arms.len().max(1)),
                dry_run: input.dry_run,
            };
            match input.backend.as_str() {
                "cpu" => {
                    type I = burn::backend::NdArray<f32>;
                    let device = <I as burn::tensor::backend::BackendTypes>::Device::default();
                    println!("backend: cpu (NdArray, deterministic; sparse_solver forced to cpu)");
                    adjoint::run_adjoint::<I>(adj, arms, out_dir, &opts, &device, manifest)?;
                }
                "cuda" => {
                    type I = burn_cuda::Cuda<f32, i32>;
                    let device = cubecl::cuda::CudaDevice::new(0);
                    println!("backend: cuda (device 0)");
                    adjoint::run_adjoint::<I>(adj, arms, out_dir, &opts, &device, manifest)?;
                }
                other => {
                    return Err(CliError::ConfigInvalid {
                        path: PathBuf::from("--backend"),
                        source: format!("unknown backend `{other}` (expected cpu or cuda)").into(),
                    })
                }
            }
            Ok(())
        }
        other => Err(CliError::ConfigInvalid {
            path: spec_path.to_path_buf(),
            source: format!("unknown study `{other}` (known: adjoint)").into(),
        }),
    }
}

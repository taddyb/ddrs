//! `ddrs` CLI entrypoint. Dispatches to subcommands defined in
//! `ddrs::cli::*`. See spec at
//! `docs/superpowers/specs/2026-05-30-ddrs-cli-lifecycle-design.md`.

use std::path::PathBuf;

use clap::{Parser, Subcommand};

use ddrs::cli::workspace::{discover_config, Workspace};
use ddrs::cli::{CliError, ExitCode, Workflow};

#[derive(Parser)]
#[command(name = "ddrs", about = "Differentiable Distributed Routing")]
struct Cli {
    #[arg(long, global = true)]
    config: Option<PathBuf>,
    #[arg(long, global = true)]
    workspace: Option<PathBuf>,
    #[command(subcommand)]
    cmd: Cmd,
}

#[derive(Subcommand)]
enum Cmd {
    Init {
        #[arg(long)] force: bool,
        #[arg(long, default_value_t = 8.0)] min_free_gpu_gb: f32,
    },
    Plan {
        #[arg(long, value_enum)] workflow: Option<Workflow>,
        #[arg(long)] json: bool,
    },
    Run {
        #[arg(long, value_enum)] workflow: Option<Workflow>,
        #[arg(long)] plot: bool,
        #[arg(long)] strict: bool,
        #[arg(long)] max_mini_batches: Option<usize>,
        #[arg(long)] json: bool,
    },
    Show { run_id: String, #[arg(long)] json: bool },
    Status { #[arg(long)] json: bool },
    Gc {
        #[arg(long)] keep: Option<usize>,
        #[arg(long)] keep_successful: bool,
        #[arg(long)] older_than: Option<String>,
        #[arg(long)] dry_run: bool,
    },
}

fn main() {
    let cli = Cli::parse();
    if let Err(e) = dispatch(cli) {
        eprintln!("error: {e}");
        ExitCode::from(&e).exit();
    }
}

fn dispatch(cli: Cli) -> Result<(), CliError> {
    let cfg_path = cli.config.clone()
        .or_else(|| discover_config(std::path::Path::new(".")));
    let ws_root = cli.workspace.unwrap_or_else(|| {
        cfg_path.as_ref()
            .and_then(|p| p.parent())
            .map(|d| d.join(".ddrs"))
            .unwrap_or_else(|| std::path::PathBuf::from(".ddrs"))
    });
    let ws = Workspace::with_root(&ws_root);

    match cli.cmd {
        Cmd::Init { force, min_free_gpu_gb } => {
            ddrs::cli::init::run_init(ddrs::cli::init::InitInput {
                workspace: ws_root,
                config_path: cfg_path,
                min_free_gpu_gb,
                force,
                skip_smoke: false,
            }).map(|_| ())
        }
        Cmd::Plan { workflow, json } => {
            // Task 19 wires the bootstrap-on-missing-config path; for now,
            // require an existing config.
            let cfg = cfg_path.ok_or_else(|| CliError::ConfigInvalid {
                path: ".".into(),
                source: "no ddrs.yaml found (bootstrap wired in Task 19)".into(),
            })?;
            let pr = ddrs::cli::plan::plan(&cfg, workflow, &ws)?;
            if json {
                println!("{}", serde_json::to_string_pretty(&pr)
                    .map_err(|e| CliError::Other(Box::new(e)))?);
            } else {
                println!("workflow {:?}", pr.workflow);
                println!("drift    {:?}", pr.drift);
            }
            Ok(())
        }
        Cmd::Run { workflow, plot, strict, max_mini_batches, json: _ } => {
            let cfg = cfg_path.ok_or_else(|| CliError::ConfigInvalid {
                path: ".".into(),
                source: "no ddrs.yaml found".into(),
            })?;
            ddrs::cli::run::run(ddrs::cli::run::RunInput {
                workspace: Workspace::with_root(ws.root()),
                config_path: cfg,
                workflow,
                plot,
                strict,
                max_mini_batches,
            }).map(|_| ())
        }
        Cmd::Show { run_id, json } => ddrs::cli::show::run_show(&ws, &run_id, json),
        Cmd::Status { json } => ddrs::cli::status::run_status(&ws, json),
        Cmd::Gc { keep, keep_successful, older_than, dry_run } => {
            let dur = older_than.as_deref()
                .map(humantime::parse_duration)
                .transpose()
                .map_err(|e| CliError::Other(Box::new(e)))?;
            let deleted = ddrs::cli::gc::run_gc(&ws, ddrs::cli::gc::GcInput {
                keep,
                keep_successful,
                older_than: dur,
                dry_run,
            })?;
            for p in &deleted {
                println!(
                    "{} {}",
                    if dry_run { "would delete" } else { "deleted" },
                    p.display(),
                );
            }
            Ok(())
        }
    }
}

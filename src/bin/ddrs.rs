//! `ddrs` CLI entrypoint. Dispatches to subcommands defined in
//! `ddrs::cli::*`. See spec at
//! `docs/superpowers/specs/2026-05-30-ddrs-cli-lifecycle-design.md`.

use std::path::PathBuf;

use clap::{Parser, Subcommand};

use ddrs::cli::workspace::{discover_config, Workspace};
use ddrs::cli::{CliError, ExitCode, Workflow};

#[derive(Parser)]
#[command(
    name = "ddrs",
    about = "Differentiable Distributed Routing — train and evaluate a \
             Muskingum-Cunge routing model with a KAN parameter head",
    after_help = "\
LIFECYCLE:
    ddrs plan    Prepare + preview: probes the GPU (first run only), bootstraps
                 ddrs.yaml if missing, locks data sources, validates the config,
                 and builds adjacency/baseline caches. Idempotent — run anytime.
    ddrs run     Execute a workflow. Re-plans internally, then trains/evaluates.
    ddrs show    Inspect a past run's manifest.
    ddrs status  Workspace summary + disk usage.
    ddrs gc      Prune old runs from .ddrs/runs/.

WORKFLOWS (--workflow flag, or the `workflow:` key in ddrs.yaml):
    train           Train the KAN head            (needs `mode: training`)
    eval            Evaluate a checkpoint         (needs `mode: testing`)
    train-and-test  Train, evaluate, and compare vs. the summed-Q' baseline

STARTING FRESH:
    rm ddrs.yaml && ddrs plan — you'll be asked whether to start from your
    last successful run's config or the clean bundled template."
)]
struct Cli {
    /// Path to the experiment config (default: discover ddrs.yaml upward
    /// from the current directory, stopping at the first .git ancestor).
    #[arg(long, global = true)]
    config: Option<PathBuf>,
    /// Workspace directory (default: .ddrs/ beside the config).
    #[arg(long, global = true)]
    workspace: Option<PathBuf>,
    #[command(subcommand)]
    cmd: Cmd,
}

#[derive(Subcommand)]
enum Cmd {
    /// Deprecated: merged into `ddrs plan`. Stub removed in 0.4.
    // disable_help_flag so `ddrs init --help` is treated as an unknown arg and
    // exits non-zero instead of silently succeeding with exit 0.
    #[command(hide = true, disable_help_flag = true)]
    Init {
        #[arg(long, hide = true)] force: bool,
        #[arg(long, default_value_t = 8.0, hide = true)] min_free_gpu_gb: f32,
    },
    /// Prepare the workspace and preview the workflow: GPU probe + cached
    /// smoke test, ddrs.yaml bootstrap, data-source locking, config
    /// validation, adjacency/baseline cache builds. Idempotent.
    Plan {
        /// Override the `workflow:` key in ddrs.yaml for this invocation.
        #[arg(long, value_enum)] workflow: Option<Workflow>,
        /// Print the plan result as JSON instead of human-readable text.
        #[arg(long)] json: bool,
        /// Re-run the GPU smoke test even if a cached verdict exists.
        #[arg(long)] force: bool,
        /// Warn when free GPU memory at probe time is below this many GB.
        #[arg(long, default_value_t = 8.0)] min_free_gpu_gb: f32,
    },
    /// Execute a workflow: re-plans, then trains and/or evaluates, writing
    /// checkpoints + manifest to .ddrs/runs/<id>/.
    Run {
        /// Override the `workflow:` key in ddrs.yaml for this invocation.
        #[arg(long, value_enum)] workflow: Option<Workflow>,
        /// After a successful run, dump per-COMID KAN parameters to
        /// plot/kan_parameters.nc (NetCDF).
        #[arg(long)] plot: bool,
        /// Exit with code 4 if data sources changed since the last plan,
        /// instead of warning and relocking.
        #[arg(long)] strict: bool,
        /// Stop each training epoch after this many mini-batches (debugging).
        #[arg(long)] max_mini_batches: Option<usize>,
        /// Replay a captured mini-batch order from JSON (matched-batch parity
        /// experiment). When set, overrides the default per-epoch shuffle.
        /// JSON schema: array of {"epoch": int, "mb": int, "staids": [str, ...]}.
        #[arg(long, value_name = "PATH")] batch_order_from: Option<PathBuf>,
        /// Print the run result as JSON instead of human-readable text.
        #[arg(long)] json: bool,
    },
    /// Inspect a past run's manifest.
    Show {
        /// Run ID under .ddrs/runs/ (list them with `ddrs status`).
        run_id: String,
        /// Print the manifest as JSON.
        #[arg(long)] json: bool,
    },
    /// Named data-source groups ("save files") under config/sources/.
    Sources {
        #[command(subcommand)]
        cmd: SourcesCmd,
    },
    /// Summarize the workspace: runs, lockfile state, disk usage.
    Status {
        /// Print the summary as JSON.
        #[arg(long)] json: bool,
    },
    /// Delete old run directories from .ddrs/runs/.
    Gc {
        /// Keep the N most recent runs.
        #[arg(long)] keep: Option<usize>,
        /// Never delete successful runs.
        #[arg(long)] keep_successful: bool,
        /// Only delete runs older than this duration (e.g. "30d", "12h").
        #[arg(long)] older_than: Option<String>,
        /// List what would be deleted without deleting anything.
        #[arg(long)] dry_run: bool,
    },
}

#[derive(Subcommand)]
enum SourcesCmd {
    /// Snapshot the current config's data_sources block as a named group
    Save { name: String, #[arg(long)] force: bool },
    /// Switch the config's data_sources to a named group and re-lock
    Use { name: String },
    /// List saved groups ('*' marks the one currently in the config)
    List,
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
        Cmd::Init { .. } => {
            eprintln!("ddrs init has been merged into ddrs plan — run `ddrs plan`");
            ExitCode::ConfigInvalid.exit();
        }
        Cmd::Plan { workflow, json, force, min_free_gpu_gb } => {
            let pr = ddrs::cli::plan::plan(
                ddrs::cli::plan::PlanInput {
                    config_path: cfg_path,
                    workflow,
                    force,
                    min_free_gpu_gb,
                    skip_smoke: false,
                    strict: false,
                },
                &ws,
            )?;
            if json {
                println!("{}", serde_json::to_string_pretty(&pr)
                    .map_err(|e| CliError::Other(Box::new(e)))?);
            } else {
                println!("workflow {:?}", pr.workflow);
                println!("drift    {:?}", pr.drift);
                let ra = &pr.resolved_adjacency;
                println!("adjacency");
                println!("  network {}", ra.conus.display());
                println!("  gauges  {}", ra.gages.display());
                if let Some(ref key) = ra.cache_key {
                    println!(
                        "  cache  {} ({})",
                        key,
                        if ra.cache_hit == Some(true) { "hit" } else { "built" },
                    );
                } else {
                    println!("  cache  (explicit paths — no managed build)");
                }
                if let Some(ref b) = pr.baseline {
                    ddrs::baseline::print_metrics_summary(&b.metrics, b.n_gauges);
                    eprintln!(
                        "baseline {} → {}",
                        if b.cache_hit { "cached" } else { "computed" },
                        b.cache_dir.display(),
                    );
                }
            }
            Ok(())
        }
        Cmd::Run { workflow, plot, strict, max_mini_batches, batch_order_from, json: _ } => {
            let cfg = cfg_path.ok_or_else(|| CliError::ConfigInvalid {
                path: ".".into(),
                source: "no ddrs.yaml found in current directory. \
                         Run `ddrs plan` first.".into(),
            })?;
            let run_dir = ddrs::cli::run::run(ddrs::cli::run::RunInput {
                workspace: Workspace::with_root(ws.root()),
                config_path: cfg,
                workflow,
                plot,
                strict,
                max_mini_batches,
                batch_order_from,
            })?;
            eprintln!("run complete → {}", run_dir.display());
            Ok(())
        }
        Cmd::Show { run_id, json } => ddrs::cli::show::run_show(&ws, &run_id, json),
        Cmd::Sources { cmd } => {
            let cfg = cfg_path.ok_or_else(|| CliError::ConfigInvalid {
                path: ".".into(),
                source: "no ddrs.yaml found in current directory.".into(),
            })?;
            match cmd {
                SourcesCmd::Save { name, force } => {
                    let dest = ddrs::cli::sources::run_save(&cfg, &name, force)?;
                    println!("saved data_sources -> {}", dest.display());
                }
                SourcesCmd::Use { name } => {
                    let relocked = ddrs::cli::sources::run_use(&cfg, &name, &ws)?;
                    println!("switched {} to group {name:?}", cfg.display());
                    if relocked {
                        println!("sources.lock refreshed");
                    } else {
                        println!("no workspace yet -- `ddrs plan` will lock these sources");
                    }
                }
                SourcesCmd::List => {
                    let entries = ddrs::cli::sources::run_list(&cfg)?;
                    if entries.is_empty() {
                        println!(
                            "no groups in {} -- create one with `ddrs sources save <name>`",
                            ddrs::cli::sources::groups_dir(&cfg).display(),
                        );
                    }
                    for e in entries {
                        println!("{} {}", if e.active { "*" } else { " " }, e.name);
                    }
                }
            }
            Ok(())
        }
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
            // Adjacency caches are content-addressed and expensive to rebuild;
            // v1 gc never touches them (key-based GC is a follow-up).
            if ws.root().join("adjacency").is_dir() {
                println!("note: .ddrs/adjacency/ caches are kept (not pruned by gc in v1)");
            }
            Ok(())
        }
    }
}

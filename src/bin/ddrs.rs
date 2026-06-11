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
        /// Replay a captured mini-batch order from JSON (matched-batch parity
        /// experiment). When set, overrides the default per-epoch shuffle.
        /// JSON schema: array of {"epoch": int, "mb": int, "staids": [str, ...]}.
        #[arg(long, value_name = "PATH")] batch_order_from: Option<PathBuf>,
        #[arg(long)] json: bool,
    },
    Show { run_id: String, #[arg(long)] json: bool },
    /// Named data-source groups ("save files") under config/sources/
    Sources {
        #[command(subcommand)]
        cmd: SourcesCmd,
    },
    Status { #[arg(long)] json: bool },
    Gc {
        #[arg(long)] keep: Option<usize>,
        #[arg(long)] keep_successful: bool,
        #[arg(long)] older_than: Option<String>,
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
            let cfg_path = cfg_path.ok_or_else(|| CliError::ConfigInvalid {
                path: ".".into(),
                source: "no ddrs.yaml found in current directory. \
                         Run `ddrs init` first.".into(),
            })?;
            let pr = ddrs::cli::plan::plan(&cfg_path, workflow, &ws)?;
            if json {
                println!("{}", serde_json::to_string_pretty(&pr)
                    .map_err(|e| CliError::Other(Box::new(e)))?);
            } else {
                println!("workflow {:?}", pr.workflow);
                println!("drift    {:?}", pr.drift);
                let ra = &pr.resolved_adjacency;
                println!("adjacency");
                println!("  conus  {}", ra.conus.display());
                println!("  gages  {}", ra.gages.display());
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
                         Run `ddrs init` first.".into(),
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
                        println!("no workspace yet -- `ddrs init` will lock these sources");
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

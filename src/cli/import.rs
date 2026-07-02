//! `ddrs import` — validate a Q' store against the DDR store contract and
//! register it as a named data-source group.
//!
//! One command turns a conforming store (see docs/nh-qprime-store-contract.md)
//! into a routable dataset:
//!
//! ```text
//! ddrs import /mnt/ssd1/data/icechunk/hourly_lstm_merit_unit_catchments.ic \
//!     --name hourly-lstm
//! ddrs sources use hourly-lstm && ddrs plan && ddrs run --workflow train
//! ```
//!
//! Validation opens the store through the same `StreamflowSource::open` the
//! training loop uses, so "import succeeded" means "training will read it".
//! The coverage report is best-effort: it needs a resolvable adjacency
//! (explicit paths or a warm `.ddrs/adjacency` cache) and degrades to a
//! warning without one.

use std::fs;
use std::path::{Path, PathBuf};

use crate::cli::sources;
use crate::cli::workspace::Workspace;
use crate::config::{Config, ConfigMode};
use crate::data::dates::Frequency;
use crate::data::store::{ConusAdjacencyStore, StreamflowSource};
use crate::error::CliError;

pub struct ImportInput {
    pub store_path: PathBuf,
    /// Group name to register under `config/sources/`. `None` is only valid
    /// with `dry_run`.
    pub name: Option<String>,
    /// Validate + report only; write nothing.
    pub dry_run: bool,
    /// Overwrite an existing group of the same name.
    pub force: bool,
}

pub fn run_import(
    cfg_path: Option<&Path>,
    ws: &Workspace,
    input: ImportInput,
) -> Result<(), CliError> {
    if input.name.is_none() && !input.dry_run {
        return Err(CliError::Runtime(
            "pass --name <group> to register the store, or --dry-run to \
             validate only"
                .into(),
        ));
    }
    if let Some(name) = &input.name {
        // Fail on a bad name BEFORE the (possibly slow) store open.
        sources::validate_name(name)?;
    }
    if !input.store_path.exists() {
        return Err(CliError::DataSourceMissing {
            path: input.store_path.clone(),
        });
    }

    // ---- 1. Open & detect (same code path the training loop uses) ----
    let source = StreamflowSource::open(&input.store_path)
        .map_err(|e| CliError::Runtime(format!("store failed to open: {e}")))?;

    println!("store       {}", input.store_path.display());
    match &source {
        StreamflowSource::Icechunk(s) => {
            let (res_str, n_days) = match s.resolution {
                Frequency::Daily => ("daily", s.n_time),
                Frequency::Hourly => ("hourly", s.n_time / 24),
            };
            let time_end = s.time_start + chrono::Duration::days(n_days as i64 - 1);
            println!("format      icechunk");
            println!("resolution  {res_str}");
            println!(
                "time        {} .. {} ({} native steps)",
                s.time_start, time_end, s.n_time
            );
            println!("divides     {}", s.index.len());

            // ---- 2. Contract checks ----
            match s.qr_units() {
                Some(u) if u == "m^3/s" => println!("Qr units    m^3/s"),
                Some(u) => println!(
                    "Qr units    WARNING: {u:?} (contract expects \"m^3/s\"; \
                     the solver will treat values as m³/s regardless)"
                ),
                None => println!(
                    "Qr units    WARNING: no units attribute (contract expects \
                     \"m^3/s\")"
                ),
            }
            sample_read(s)?;

            // ---- 3. Coverage report (best-effort) ----
            coverage_report(cfg_path, ws, s);
        }
        StreamflowSource::GlobalZarr(_) => {
            println!("format      global zarr v2 (daily)");
            println!(
                "note        detailed contract validation and coverage are \
                 icechunk-only; open succeeded, which exercises the same \
                 reader the training loop uses"
            );
        }
    }

    // ---- 4. Register ----
    if input.dry_run {
        println!("dry-run     no group written");
        return Ok(());
    }
    let name = input.name.expect("checked at entry");
    let cfg = cfg_path.ok_or_else(|| CliError::ConfigInvalid {
        path: ".".into(),
        source: "no ddrs.yaml found — registration copies its data_sources \
                 block. Run inside a ddrs workspace or pass --config."
            .into(),
    })?;
    let cfg_text = fs::read_to_string(cfg)?;
    let block = sources::extract_block(&cfg_text, cfg)?;
    let swapped = swap_streamflow_line(&block, &input.store_path)?;
    let dest = sources::save_block(cfg, &name, &swapped, input.force)?;
    println!("registered  {}", dest.display());
    println!("activate    ddrs sources use {name}");
    Ok(())
}

/// Read a tiny sample (first 5 divides × up to 3 days) and require finite,
/// positive values — catches unit disasters and all-NaN stores.
fn sample_read(s: &crate::data::store::StreamflowStore) -> Result<(), CliError> {
    let comids: Vec<_> = s.index.ids().iter().take(5).copied().collect();
    let n_days_native = match s.resolution {
        Frequency::Daily => s.n_time,
        Frequency::Hourly => s.n_time / 24,
    };
    let n_days = n_days_native.min(3);
    let q = s
        .read_window_daily(s.time_start, n_days, &comids)
        .map_err(|e| CliError::Runtime(format!("sample read failed: {e}")))?;
    for &v in q.iter() {
        if !v.is_finite() || v <= 0.0 {
            return Err(CliError::Runtime(format!(
                "sample read violates the contract: value {v} (must be \
                 finite and > 0; producers floor to 1e-6)"
            )));
        }
    }
    println!(
        "sample      {} COMIDs × {} days: finite, positive ✓",
        comids.len(),
        n_days
    );
    Ok(())
}

/// Intersect the store's divide_ids with the resolved CONUS adjacency and
/// report coverage. Best-effort: any failure (no config, unreadable
/// adjacency) prints a warning instead of failing the import. NOTE: with a
/// fabric-only config and a cold cache this triggers the managed adjacency
/// build (~10 s CONUS), same as `ddrs plan`.
fn coverage_report(
    cfg_path: Option<&Path>,
    ws: &Workspace,
    s: &crate::data::store::StreamflowStore,
) {
    let Some(cfg_path) = cfg_path else {
        println!("coverage    skipped (no ddrs.yaml — run inside a workspace for a report)");
        return;
    };
    let resolved = Config::from_yaml_file_with_mode(cfg_path, ConfigMode::Training)
        .map_err(|e| e.to_string())
        .and_then(|config| {
            crate::cli::plan::resolve_adjacency(&config, cfg_path, ws)
                .map_err(|e| e.to_string())
        })
        .and_then(|resolved| {
            ConusAdjacencyStore::open(&resolved.conus).map_err(|e| e.to_string())
        });
    match resolved {
        Ok(conus) => {
            let total = conus.order.len();
            let covered = conus.order.iter().filter(|c| s.index.contains(c)).count();
            let pct = 100.0 * covered as f64 / total.max(1) as f64;
            println!(
                "coverage    {covered}/{total} fabric COMIDs ({pct:.1}%); \
                 the rest read as 0.001 m³/s fill"
            );
        }
        Err(e) => println!("coverage    skipped ({e})"),
    }
}

/// Replace the value of the `streamflow:` key inside a `data_sources:` block,
/// preserving indentation and every other line (comments included).
fn swap_streamflow_line(block: &str, store_path: &Path) -> Result<String, CliError> {
    let mut out = String::new();
    let mut swapped = false;
    for line in block.lines() {
        let trimmed = line.trim_start();
        if !swapped && trimmed.starts_with("streamflow:") {
            let indent = &line[..line.len() - trimmed.len()];
            out.push_str(&format!("{indent}streamflow: {}\n", store_path.display()));
            swapped = true;
        } else {
            out.push_str(line);
            out.push('\n');
        }
    }
    if !swapped {
        return Err(CliError::Runtime(
            "config's data_sources block has no `streamflow:` key".into(),
        ));
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn swap_streamflow_preserves_everything_else() {
        let block = "\
data_sources:
  attributes: /a.nc
  # comment stays
  streamflow: /old.ic
  observations: /obs
";
        let out = swap_streamflow_line(block, Path::new("/new/store.ic")).unwrap();
        assert!(out.contains("streamflow: /new/store.ic"));
        assert!(!out.contains("/old.ic"));
        assert!(out.contains("# comment stays"));
        assert!(out.contains("attributes: /a.nc"));
        assert!(out.contains("observations: /obs"));
    }

    #[test]
    fn swap_errors_without_streamflow_key() {
        let err = swap_streamflow_line("data_sources:\n  gages: /g.csv\n", Path::new("/x"))
            .unwrap_err();
        assert!(err.to_string().contains("streamflow"));
    }
}

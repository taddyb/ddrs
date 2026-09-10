use std::path::Path;

use crate::cli::{manifest::Manifest, workspace::Workspace};
use crate::error::CliError;

pub fn run_show(ws: &Workspace, run_id: &str, as_json: bool) -> Result<(), CliError> {
    let run_dir = ws.runs_dir().join(run_id);
    let path = run_dir.join("manifest.json");
    let m = Manifest::read(&path)?;
    if as_json {
        println!("{}", serde_json::to_string_pretty(&m)
            .map_err(|e| CliError::Other(Box::new(e)))?);
    } else {
        println!("run      {}", m.run_id);
        println!("status   {:?}", m.status);
        println!("workflow {:?}", m.workflow);
        println!("started  {}", m.started_at);
        if let Some(f) = &m.finished_at { println!("finished {}", f); }
        println!("git      {} ({})", m.git.sha, if m.git.dirty { "dirty" } else { "clean" });
        println!("drift    {:?}", m.source_lock.drift);
        if let Some(ra) = &m.resolved_adjacency {
            println!("adjacency");
            println!("  conus  {}", ra.conus.display());
            println!("  gages  {}", ra.gages.display());
            if let Some(key) = &ra.cache_key {
                println!(
                    "  cache  {} ({})",
                    key,
                    if ra.cache_hit == Some(true) { "hit" } else { "built" },
                );
            }
        }
        if let Some(p) = &m.outputs.plot { println!("plot     {}", p.display()); }
        print_metrics(&m.metrics, &run_dir);
    }
    Ok(())
}

/// Render the manifest's `metrics` map, then the summed-Q' baseline comparison
/// when the run carries one.
///
/// Text mode used to print nothing from `metrics`, so `ddrs show <id> | grep nse`
/// came back empty on a perfectly good `train-and-test` run and read like a
/// failure. The numbers were only ever in `--json`.
fn print_metrics(metrics: &serde_json::Value, run_dir: &Path) {
    let Some(map) = metrics.as_object() else { return };
    if map.is_empty() {
        return;
    }
    println!("metrics");
    // Sorted for a stable, greppable order (serde_json preserves insertion
    // order only with the preserve_order feature).
    let mut keys: Vec<&String> = map.keys().collect();
    keys.sort();
    for k in keys {
        println!("  {:<22} {}", k, render_scalar(&map[k]));
    }
    print_baseline_comparison(map, run_dir);
}

fn render_scalar(v: &serde_json::Value) -> String {
    match v {
        serde_json::Value::Number(n) => match n.as_f64() {
            // Integers (epoch counts, gauge counts) shouldn't gain decimals.
            Some(f) if f.fract() == 0.0 && f.abs() < 1e15 => format!("{}", f as i64),
            Some(f) => format!("{f:.4}"),
            None => n.to_string(),
        },
        serde_json::Value::String(s) => s.clone(),
        other => other.to_string(),
    }
}

/// `train-and-test` copies the summed-Q' baseline to
/// `<run_dir>/baseline/manifest.json`, but its metrics are **per-gauge arrays**,
/// not reduced medians, so a reader comparing them to the run's
/// `median_nse_finite` was comparing a scalar against a list. Reduce here.
fn print_baseline_comparison(
    run_metrics: &serde_json::Map<String, serde_json::Value>,
    run_dir: &Path,
) {
    let text = match std::fs::read_to_string(run_dir.join("baseline/manifest.json")) {
        Ok(t) => t,
        Err(_) => return, // train-only run, or baseline copy unavailable
    };
    let Ok(b): std::result::Result<serde_json::Value, _> = serde_json::from_str(&text) else {
        return;
    };
    let bm = &b["metrics"];
    println!("baseline (summed Q', median over gauges)");
    for (label, key, run_key) in [
        ("NSE", "nse", "median_nse_finite"),
        ("KGE", "kge", "median_kge_finite"),
    ] {
        let Some(base) = median_finite(&bm[key]) else { continue };
        match run_metrics.get(run_key).and_then(|v| v.as_f64()) {
            Some(routed) => println!(
                "  {:<22} {:.4}   routed {:.4}   Δ {:+.4}{}",
                label,
                base,
                routed,
                routed - base,
                if routed > base { "" } else { "  ← routed does NOT beat baseline" },
            ),
            None => println!("  {:<22} {:.4}", label, base),
        }
    }
}

/// Median of the finite entries of a JSON array, or `None` if there are none.
fn median_finite(v: &serde_json::Value) -> Option<f64> {
    let mut xs: Vec<f64> = v
        .as_array()?
        .iter()
        .filter_map(|x| x.as_f64())
        .filter(|x| x.is_finite())
        .collect();
    if xs.is_empty() {
        return None;
    }
    xs.sort_by(|a, b| a.partial_cmp(b).expect("finite values compare"));
    let n = xs.len();
    Some(if n % 2 == 1 {
        xs[n / 2]
    } else {
        (xs[n / 2 - 1] + xs[n / 2]) / 2.0
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn median_finite_handles_odd_even_and_nonfinite() {
        assert_eq!(median_finite(&json!([3.0, 1.0, 2.0])), Some(2.0));
        assert_eq!(median_finite(&json!([1.0, 2.0, 3.0, 4.0])), Some(2.5));
        // nulls (serde's NaN encoding here) and missing entries are skipped.
        assert_eq!(median_finite(&json!([1.0, null, 3.0])), Some(2.0));
        assert_eq!(median_finite(&json!([])), None);
        assert_eq!(median_finite(&json!([null])), None);
        assert_eq!(median_finite(&json!("not an array")), None);
    }

    #[test]
    fn render_scalar_keeps_counts_integral() {
        assert_eq!(render_scalar(&json!(30)), "30");
        assert_eq!(render_scalar(&json!(620.0)), "620");
        assert_eq!(render_scalar(&json!(0.7510778)), "0.7511");
        assert_eq!(render_scalar(&json!("ok")), "ok");
    }
}

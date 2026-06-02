//! Console-friendly metrics summary table for the summed Q' baseline.
//! Mirrors `~/projects/ddr/scripts/summed_q_prime.py:85-110` so the
//! output is recognizable to anyone familiar with the DDR script.

use std::io::Write;

use crate::training::metrics::Metrics;

/// Per-metric distribution stats over the finite (non-NaN) subset.
struct Stats {
    median: f32,
    mean: f32,
    q25: f32,
    q75: f32,
    valid: usize,
}

fn stats(xs: &[f32]) -> Stats {
    let mut finite: Vec<f32> = xs.iter().copied().filter(|v| v.is_finite()).collect();
    finite.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let valid = finite.len();
    if valid == 0 {
        return Stats {
            median: f32::NAN,
            mean: f32::NAN,
            q25: f32::NAN,
            q75: f32::NAN,
            valid: 0,
        };
    }
    let mean = finite.iter().sum::<f32>() / (valid as f32);
    // Mid-rank percentile (mirrors numpy's default linear interpolation
    // closely enough for a summary table; exact agreement isn't the goal).
    let pct = |p: f32| -> f32 {
        let idx = ((valid - 1) as f32 * p).round() as usize;
        finite[idx]
    };
    Stats {
        median: pct(0.5),
        mean,
        q25: pct(0.25),
        q75: pct(0.75),
        valid,
    }
}

/// Print a DDR-parity metrics table for the baseline. Writes to `out`
/// so tests can capture the output.
pub fn write_metrics_summary<W: Write>(
    out: &mut W,
    m: &Metrics,
    total_gauges: usize,
) -> std::io::Result<()> {
    writeln!(out)?;
    writeln!(out, "{}", "=".repeat(80))?;
    writeln!(out, "{:^80}", "SUMMED Q' METRICS SUMMARY")?;
    writeln!(out, "{}", "=".repeat(80))?;
    writeln!(out, "Total Gauges Evaluated: {total_gauges}")?;
    writeln!(out, "{}", "-".repeat(80))?;
    writeln!(
        out,
        "{:<12} {:>9} {:>9} {:>9} {:>9} {:>7}",
        "METRIC", "MEDIAN", "MEAN", "Q25", "Q75", "VALID"
    )?;
    writeln!(out, "{}", "-".repeat(80))?;
    for (name, xs, decimals) in [
        ("Bias", &m.bias, 3usize),
        ("FLV (%)", &m.flv, 2),
        ("FHV (%)", &m.fhv, 2),
        ("KGE", &m.kge, 3),
        ("NSE", &m.nse, 3),
    ] {
        let s = stats(xs);
        writeln!(
            out,
            "{:<12} {:>9.*} {:>9.*} {:>9.*} {:>9.*} {:>7}",
            name, decimals, s.median, decimals, s.mean, decimals, s.q25, decimals, s.q75, s.valid,
        )?;
    }
    writeln!(out, "{}", "=".repeat(80))?;
    Ok(())
}

/// Convenience wrapper: print to stdout.
pub fn print_metrics_summary(m: &Metrics, total_gauges: usize) {
    let stdout = std::io::stdout();
    let mut handle = stdout.lock();
    let _ = write_metrics_summary(&mut handle, m, total_gauges);
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn renders_table_headers_and_rows() {
        let m = Metrics {
            nse: vec![0.5, 0.6, 0.7, 0.8, 0.9],
            rmse: vec![0.1; 5],
            kge: vec![0.4, 0.5, 0.6, 0.7, 0.8],
            bias: vec![0.0; 5],
            fhv: vec![-2.0, -1.5, -1.0, -0.5, 0.0],
            flv: vec![1.0, 2.0, 3.0, 4.0, 5.0],
        };
        let mut buf = Vec::new();
        write_metrics_summary(&mut buf, &m, 5).unwrap();
        let out = String::from_utf8(buf).unwrap();
        assert!(out.contains("SUMMED Q' METRICS SUMMARY"));
        assert!(out.contains("Total Gauges Evaluated: 5"));
        assert!(out.contains("METRIC"));
        assert!(out.contains("Bias"));
        assert!(out.contains("FLV (%)"));
        assert!(out.contains("FHV (%)"));
        assert!(out.contains("KGE"));
        assert!(out.contains("NSE"));
        // Each metric row reports VALID = 5 in the last column.
        let nse_line = out
            .lines()
            .find(|l| l.starts_with("NSE"))
            .expect("NSE row");
        assert!(nse_line.ends_with("5"));
    }

    #[test]
    fn nan_only_metric_shows_zero_valid() {
        let m = Metrics {
            nse: vec![f32::NAN; 3],
            rmse: vec![f32::NAN; 3],
            kge: vec![f32::NAN; 3],
            bias: vec![f32::NAN; 3],
            fhv: vec![f32::NAN; 3],
            flv: vec![f32::NAN; 3],
        };
        let mut buf = Vec::new();
        write_metrics_summary(&mut buf, &m, 3).unwrap();
        let out = String::from_utf8(buf).unwrap();
        for needle in ["Bias", "FLV", "FHV", "KGE", "NSE"] {
            let line = out.lines().find(|l| l.starts_with(needle)).unwrap();
            assert!(
                line.ends_with("0"),
                "{needle} row should end with valid=0, got {line:?}"
            );
        }
    }
}

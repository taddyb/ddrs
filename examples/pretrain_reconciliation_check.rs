//! Phase 2 verification: reconcile camels_hourly's daily-mean-of-real-hourly
//! (unit-converted to m³/s) against production's OWN `usgs_daily_observations`
//! store, for every gauge overlapping gages_3000.csv. Per the pretraining
//! plan, gauges outside [ratio 0.9-1.1, corr>=0.98] are excluded from
//! pretraining -- and if dozens fail, that's evidence of a bug in THIS
//! pipeline, not data badness (stop-and-debug, not exclude-and-proceed).
//!
//!   cargo run --release --example pretrain_reconciliation_check

use chrono::NaiveDate;
use ndarray::Array1;

use ddrs::data::ids::Staid;
use ddrs::data::store::gage_csv::GageMetadata;
use ddrs::data::store::icechunk::UsgsObservationsStore;
use ddrs::data::store::CamelsHourlyStore;
use ddrs::pretrain::{extract_complete_days, qobs_mm_hr_to_m3s, reconcile_gauge};

const GAGES_CSV: &str = "/home/tbindas/projects/ddr/references/gage_info/gages_3000.csv";
const CAMELS_NC: &str = "/mnt/ssd1/data/camels_hourly/usgs-streamflow-nldas_hourly.nc";
const USGS_DAILY: &str = "/mnt/ssd1/data/icechunk/usgs_daily_observations";

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let gages = GageMetadata::open(GAGES_CSV)?;
    let camels = CamelsHourlyStore::open(CAMELS_NC)?;
    let obs = UsgsObservationsStore::open(USGS_DAILY)?;

    let gauge_set: Vec<&ddrs::data::store::gage_csv::GageRow> = gages
        .rows
        .iter()
        .filter(|r| camels.index.position(&r.staid).is_some())
        .collect();
    println!("overlap gauges: {}/{}", gauge_set.len(), gages.rows.len());

    // Window matching the plan's chosen pretrain-train period: 1998-2013.
    let window_start_date = NaiveDate::from_ymd_opt(1998, 1, 1).unwrap();
    let window_start_dt = window_start_date.and_hms_opt(0, 0, 0).unwrap();
    let n_days = (NaiveDate::from_ymd_opt(2013, 12, 31).unwrap() - window_start_date).num_days() as usize + 1;
    let n_hours = n_days * 24;

    let mut kept = 0usize;
    let mut excluded = 0usize;
    let mut ratios = Vec::new();
    let mut corrs = Vec::new();
    let mut flagged: Vec<(Staid, f32, f32, usize)> = Vec::new();

    for row in &gauge_set {
        let (qobs, _precip) = match camels.read_window(window_start_dt, n_hours, std::slice::from_ref(&row.staid)) {
            Ok(v) => v,
            Err(e) => {
                eprintln!("{}: read failed: {e}", row.staid);
                continue;
            }
        };
        let qobs_col: Array1<f32> = qobs.column(0).to_owned();
        let complete_days = extract_complete_days(&qobs_col);
        if complete_days.len() < 30 {
            continue;
        }
        let camels_daily_m3s: Vec<(NaiveDate, f32)> = complete_days
            .iter()
            .map(|r| {
                let date = window_start_date + chrono::Duration::days(r.day_index as i64);
                (date, qobs_mm_hr_to_m3s(r.daily_q_m3s, row.drain_sqkm))
            })
            .collect();

        match reconcile_gauge(&row.staid, &camels_daily_m3s, &obs) {
            Ok(result) => {
                if result.keep {
                    kept += 1;
                    ratios.push(result.median_ratio);
                    corrs.push(result.correlation);
                } else {
                    excluded += 1;
                    flagged.push((row.staid.clone(), result.median_ratio, result.correlation, result.n_overlap_days));
                }
            }
            Err(e) => {
                eprintln!("{}: reconcile failed: {e}", row.staid);
                excluded += 1;
            }
        }
    }

    println!("\n=== Reconciliation summary (1998-2013) ===");
    println!("kept:     {kept}");
    println!("excluded: {excluded}");
    if !ratios.is_empty() {
        let mut sorted_r = ratios.clone();
        sorted_r.sort_by(|a, b| a.partial_cmp(b).unwrap());
        let mut sorted_c = corrs.clone();
        sorted_c.sort_by(|a, b| a.partial_cmp(b).unwrap());
        println!("median_ratio (of kept):       {:.4}", sorted_r[sorted_r.len() / 2]);
        println!("median_correlation (of kept):  {:.4}", sorted_c[sorted_c.len() / 2]);
    }
    if !flagged.is_empty() {
        println!("\n--- flagged (excluded) gauges ---");
        for (staid, ratio, corr, n) in &flagged {
            println!("{staid}: ratio={ratio:.4} corr={corr:.4} n_overlap_days={n}");
        }
    }
    Ok(())
}

//! Mass-balance verification against a TRAINED checkpoint's disaggregation
//! head, using REAL CONUS test-window data (not synthetic inputs) — the
//! strongest available check that the KAN disagg head preserves mass on
//! actual gauges. Loads a contiguous test-window `RoutingBatch` (real daily
//! Q' + real basin-normalized AORC precip), runs the trained disagg head on
//! a handful of gauge outlet reaches, and dumps a tidy CSV so the Python
//! side can (a) overlay the KAN-disaggregated hourly curve against the flat
//! repeat-24 baseline and the daily input step function, and (b) verify the
//! daily mean of the disaggregated hourly series equals the daily input
//! exactly (mass conservation, see `src/nn/disagg_head.rs` docs).
//!
//!   cargo run --release --example kan_disagg_mass_balance_real -- \
//!       --config <run_dir>/config.yaml \
//!       --checkpoint <run_dir>/checkpoints/epoch_5_mb_9/head \
//!       --output <run_dir>/plots/disagg_mass_balance_real.csv \
//!       --n-days 10 --n-gauges 3

use std::path::PathBuf;

use burn::backend::NdArray;
use burn::tensor::backend::{Backend, BackendTypes};
use burn::tensor::{Tensor, TensorData};
use clap::Parser;
use ndarray::Array2;

use ddrs::config::{Config, ConfigMode};
use ddrs::data::store::AorcPrecipStore;
use ddrs::data::{Comid, MeritGagesDataset, TestWindow};
use ddrs::nn::kan_head::KanHead;
use ddrs::training::load_kan_head;

type I = NdArray<f32>;

#[derive(Parser, Debug)]
struct Cli {
    #[arg(long)]
    config: PathBuf,
    #[arg(long)]
    checkpoint: PathBuf,
    #[arg(long)]
    output: PathBuf,
    #[arg(long, default_value_t = 10)]
    n_days: usize,
    #[arg(long, default_value_t = 3)]
    n_gauges: usize,
}

fn select_columns(arr: &Array2<f32>, cols: &[usize]) -> Array2<f32> {
    let (t, _) = arr.dim();
    Array2::from_shape_fn((t, cols.len()), |(row, j)| arr[(row, cols[j])])
}

fn to_tensor(arr: &Array2<f32>, device: &<I as BackendTypes>::Device) -> Tensor<I, 2> {
    let (t, n) = arr.dim();
    let data: Vec<f32> = arr.iter().copied().collect();
    Tensor::<I, 2>::from_data(TensorData::new(data, [t, n]), device)
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let cli = Cli::parse();
    if let Some(parent) = cli.output.parent() {
        std::fs::create_dir_all(parent)?;
    }

    let cfg = Config::from_yaml_file_with_mode(&cli.config, ConfigMode::Testing)?;
    let head_cfg = cfg
        .kan_head
        .as_ref()
        .expect("kan_head section required for trained-KAN inference");

    let dataset = MeritGagesDataset::open(&cfg)?;
    let axis = dataset.time_axis().clone();
    let window = TestWindow::new(&axis, 0, cli.n_days);
    let batch = dataset.collate_window(&window)?;

    let n_gauges = cli.n_gauges.min(batch.gauge_staids.len());
    let mut cols: Vec<usize> = Vec::with_capacity(n_gauges);
    let mut staid_labels: Vec<String> = Vec::with_capacity(n_gauges);
    let mut comid_labels: Vec<i64> = Vec::with_capacity(n_gauges);
    for g in 0..n_gauges {
        let Some(&col) = batch.outflow_idx[g].first() else {
            continue;
        };
        cols.push(col);
        staid_labels.push(batch.gauge_staids[g].to_string());
        comid_labels.push(batch.divide_comids[col].0);
    }

    let daily_sel = select_columns(&batch.q_prime_daily, &cols); // (D, n_gauges)
    let precip_sel = select_columns(&batch.precip_hourly, &cols); // (T_hours, n_gauges), KAN INPUT (normalized)
    let flat_sel = select_columns(&batch.q_prime, &cols); // (T_hours, n_gauges) — repeat-24 baseline

    // Raw AORC precip (mm/hr) for the SAME reaches/window, straight from the
    // store — a physical value for plotting, independent of the KAN's
    // basin-normalized input feature (`precip_sel` above).
    let ds = cfg.data_sources.as_ref().expect("data_sources section required");
    let aorc_path = ds.aorc_precip.as_ref().expect("aorc_precip source required for this example");
    let aorc = AorcPrecipStore::open(aorc_path)?;
    let selected_comids: Vec<Comid> = comid_labels.iter().map(|&c| Comid(c)).collect();
    let precip_raw_sel = aorc.read_test_window(&window, &selected_comids)?; // (T_hours, n_gauges), mm/hr

    let device = Default::default();
    <I as Backend>::seed(&device, cfg.seed);
    let head_template: KanHead<I> = ddrs::config::kan_config(head_cfg, cfg.seed).init::<I>(&device);
    let head = load_kan_head::<I>(&cli.checkpoint, head_template, &device)?;
    let disagg = head.disagg.expect("checkpoint's kan_head has no disaggregation block");

    let daily_tensor = to_tensor(&daily_sel, &device);
    let precip_tensor = to_tensor(&precip_sel, &device);
    let n_hourly = window.n_hourly();
    let hourly = disagg.forward(daily_tensor, precip_tensor, n_hourly); // (T_hours, n_gauges)
    let hourly_vec: Vec<f32> = hourly.into_data().to_vec().unwrap();

    let mut f = std::fs::File::create(&cli.output)?;
    use std::io::Write;
    writeln!(
        f,
        "staid,comid,hour,day,daily_input,disagg_hourly,flat_repeat_hourly,precip_input,precip_raw_mm_hr"
    )?;
    for (j, (staid, comid)) in staid_labels.iter().zip(comid_labels.iter()).enumerate() {
        for h in 0..n_hourly {
            let d = h / 24;
            let daily_input = daily_sel[(d, j)];
            let disagg_val = hourly_vec[h * cols.len() + j];
            let flat_val = flat_sel[(h, j)];
            let precip_val = precip_sel[(h, j)];
            let precip_raw = precip_raw_sel[(h, j)];
            writeln!(
                f,
                "{staid},{comid},{h},{d},{daily_input},{disagg_val},{flat_val},{precip_val},{precip_raw}"
            )?;
        }
    }
    println!(
        "wrote {} ({} gauges x {} hours)",
        cli.output.display(),
        cols.len(),
        n_hourly
    );
    Ok(())
}

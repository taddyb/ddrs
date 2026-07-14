//! Direct visual comparison for the 72-hour-joint-mass-balance experimental
//! head: real hourly USGS discharge vs. the Window72Head prediction vs. a
//! per-gauge climatology template, over a real 3-day storm window. Mirrors
//! `examples/pretrain_disagg_storm_compare.rs` but for the wider window and
//! the relaxed (72h-aggregate-only) mass constraint.
//!
//!   cargo run --release --example pretrain_disagg_window72_storm_compare -- \
//!       --checkpoint output/disagg_pretrain/best_disagg_window72 \
//!       --staid 02198100 --storm-date 2016-09-02 \
//!       --output output/disagg_pretrain/window72_storm_02198100.csv

use burn::backend::NdArray;
use burn::config::Config;
use burn::module::Module;
use burn::nn::Linear;
use burn::record::{CompactRecorder, Recorder};
use burn::tensor::activation::softmax;
use burn::tensor::backend::{Backend, BackendTypes};
use burn::tensor::{Tensor, TensorData};
use chrono::NaiveDate;
use clap::Parser;
use rand::SeedableRng;
use rskan::{KanLayer, KanLayerConfig};

use ddrs::data::ids::Staid;
use ddrs::data::store::gage_csv::GageMetadata;
use ddrs::data::store::CamelsHourlyStore;
use ddrs::pretrain::{extract_complete_day_windows, normalize_gauge_precip, qobs_mm_hr_to_m3s};

type I = NdArray<f32>;

const GAGES_CSV: &str = "/home/tbindas/projects/ddr/references/gage_info/gages_3000.csv";
const CAMELS_NC: &str = "/mnt/ssd1/data/camels_hourly/usgs-streamflow-nldas_hourly.nc";
const SEED: u64 = 42;
const CLIMATOLOGY_START: (i32, u32, u32) = (1998, 1, 1);
const CLIMATOLOGY_END: (i32, u32, u32) = (2013, 12, 31);
const WINDOW_DAYS: usize = 3;
const WINDOW_HOURS: usize = WINDOW_DAYS * 24;
const NUM_FEATURES: usize = WINDOW_DAYS + WINDOW_HOURS;
const LOG_EPS: f32 = 1.0e-3;
const HIDDEN: usize = 16;

#[derive(Module, Debug)]
struct Window72Head<B: Backend> {
    input: Linear<B>,
    hidden: Vec<KanLayer<B>>,
    output: Linear<B>,
}

#[derive(Config, Debug)]
struct Window72HeadConfig {
    seed: u64,
    #[config(default = 16)]
    hidden_size: usize,
    #[config(default = 1)]
    num_hidden_layers: usize,
    #[config(default = 3)]
    grid: usize,
    #[config(default = 3)]
    k: usize,
}

impl Window72HeadConfig {
    fn init<B: Backend>(&self, device: &B::Device) -> Window72Head<B> {
        let h = self.hidden_size;
        let mut rng = rand::rngs::StdRng::seed_from_u64(self.seed);
        let input_weight = ddrs::nn::init::sample_kaiming_normal_relu(&mut rng, NUM_FEATURES, h);
        let output_weight = ddrs::nn::init::sample_xavier_normal(&mut rng, h, WINDOW_HOURS, 1.0);
        let input = Linear {
            weight: ddrs::nn::init::to_param_weight::<B>(input_weight, device),
            bias: Some(ddrs::nn::init::zero_bias_tensor::<B>(h, device)),
        };
        let output = Linear {
            weight: ddrs::nn::init::to_param_weight::<B>(output_weight, device),
            bias: Some(ddrs::nn::init::zero_bias_tensor::<B>(WINDOW_HOURS, device)),
        };
        let hidden: Vec<KanLayer<B>> = (0..self.num_hidden_layers)
            .map(|_| KanLayerConfig::new(h, h, self.seed).with_num(self.grid).with_k(self.k).init(device))
            .collect();
        Window72Head { input, hidden, output }
    }
}

impl<B: Backend> Window72Head<B> {
    fn forward(&self, daily_q_3: Tensor<B, 2>, precip_72: Tensor<B, 2>) -> Tensor<B, 2> {
        let [_, n] = daily_q_3.dims();
        let logq = daily_q_3.clone().add_scalar(LOG_EPS).log().transpose();
        let precip_t = precip_72.transpose();
        let feats = Tensor::cat(vec![logq, precip_t], 1);
        let mut x = self.input.forward(feats);
        for layer in &self.hidden {
            x = layer.forward(x);
        }
        let logits = self.output.forward(x);
        let shape = softmax(logits, 1);
        let three_day_mean = daily_q_3.mean_dim(0).reshape([n, 1]);
        (shape * three_day_mean * (WINDOW_HOURS as f32)).transpose()
    }
}

#[derive(Parser, Debug)]
struct Cli {
    #[arg(long, default_value = "output/disagg_pretrain/best_disagg_window72")]
    checkpoint: String,
    #[arg(long)]
    staid: String,
    #[arg(long)]
    storm_date: String,
    #[arg(long)]
    output: String,
}

fn ymd(t: (i32, u32, u32)) -> NaiveDate {
    NaiveDate::from_ymd_opt(t.0, t.1, t.2).unwrap()
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let cli = Cli::parse();
    if let Some(parent) = std::path::Path::new(&cli.output).parent() {
        std::fs::create_dir_all(parent)?;
    }

    let gages = GageMetadata::open(GAGES_CSV)?;
    let camels = CamelsHourlyStore::open(CAMELS_NC)?;
    let staid = Staid::new(&cli.staid);
    let gauge = gages.rows.iter().find(|r| r.staid == staid).expect("staid not in gages_3000.csv");

    // Climatology: 24h template from the gauge's own 1998-2013 days.
    let clim_start_dt = ymd(CLIMATOLOGY_START).and_hms_opt(0, 0, 0).unwrap();
    let clim_n_days = (ymd(CLIMATOLOGY_END) - ymd(CLIMATOLOGY_START)).num_days() as usize + 1;
    let (clim_qobs, _) = camels.read_window(clim_start_dt, clim_n_days * 24, std::slice::from_ref(&staid))?;
    let clim_qobs_m3s: ndarray::Array1<f32> = clim_qobs.column(0).mapv(|v| if v.is_finite() { qobs_mm_hr_to_m3s(v, gauge.drain_sqkm) } else { f32::NAN });
    let clim_windows = extract_complete_day_windows(&clim_qobs_m3s, 1);
    let mut clim_shape_24 = [0f32; 24];
    for w in &clim_windows {
        let r = &w[0];
        for h in 0..24 {
            clim_shape_24[h] += r.target_hourly_m3s[h] / (24.0 * r.daily_q_m3s);
        }
    }
    for v in &mut clim_shape_24 { *v /= clim_windows.len() as f32; }
    println!("climatology template from {} training-period days", clim_windows.len());

    let device: <I as BackendTypes>::Device = Default::default();
    let template: Window72Head<I> = Window72HeadConfig::new(SEED).with_hidden_size(HIDDEN).init(&device);
    let record = CompactRecorder::new().load(cli.checkpoint.clone().into(), &device)?;
    let head: Window72Head<I> = template.load_record(record);

    let storm_date = NaiveDate::parse_from_str(&cli.storm_date, "%Y-%m-%d")?;
    // Center the 3-day window on the storm date.
    let win_start = storm_date - chrono::Duration::days(1);
    let start_dt = win_start.and_hms_opt(0, 0, 0).unwrap();
    let (qobs, precip_raw) = camels.read_window(start_dt, WINDOW_HOURS, std::slice::from_ref(&staid))?;
    let qobs_m3s: ndarray::Array1<f32> = qobs.column(0).mapv(|v| if v.is_finite() { qobs_mm_hr_to_m3s(v, gauge.drain_sqkm) } else { f32::NAN });
    let windows = extract_complete_day_windows(&qobs_m3s, WINDOW_DAYS);
    assert!(!windows.is_empty(), "no complete 3-day window around the storm date -- pick a different one");
    let w = &windows[0];

    let precip_col: ndarray::Array1<f32> = precip_raw.column(0).to_owned();
    let precip_norm = normalize_gauge_precip(precip_col.clone());
    let precip_feat: Vec<f32> = (0..WINDOW_HOURS).map(|h| precip_norm[(h, 0)]).collect();

    let mut daily_q_3 = [0f32; WINDOW_DAYS];
    let mut target_72 = [0f32; WINDOW_HOURS];
    for (i, r) in w.iter().enumerate() {
        daily_q_3[i] = r.daily_q_m3s;
        target_72[i * 24..i * 24 + 24].copy_from_slice(&r.target_hourly_m3s);
    }

    let daily_q_t = Tensor::<I, 2>::from_data(TensorData::new(daily_q_3.to_vec(), [WINDOW_DAYS, 1]), &device);
    let precip_t = Tensor::<I, 2>::from_data(TensorData::new(precip_feat, [WINDOW_HOURS, 1]), &device);
    let pred = head.forward(daily_q_t, precip_t);
    let pred_vec: Vec<f32> = pred.into_data().to_vec().unwrap();

    let mut f = std::fs::File::create(&cli.output)?;
    use std::io::Write;
    writeln!(f, "staid,hour,daily_input,real_hourly,head_pred,climatology_pred,precip_raw_mm_hr")?;
    for h in 0..WINDOW_HOURS {
        let d = h / 24;
        let hh = h % 24;
        let clim_pred = clim_shape_24[hh] * 24.0 * daily_q_3[d];
        writeln!(f, "{staid},{h},{},{},{},{},{}", daily_q_3[d], target_72[h], pred_vec[h], clim_pred, precip_col[h])?;
    }
    println!("wrote {}", cli.output);
    Ok(())
}

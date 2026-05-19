//! Test-phase evaluation loop. Mirrors
//! `~/projects/ddr/scripts/train_and_test.py::_test` (lines 43-119).
//!
//! Unlike the training loop, batches iterate TIME (not gauges) and the
//! network is the static all-gauges union. `carry_state=i>0` propagates
//! engine state across consecutive chunks.

use burn::tensor::backend::Backend;
use chrono::NaiveDate;
use ndarray::{s, Array2};

use crate::config::Config;
use crate::data::dataset::MeritGagesDataset;
use crate::data::error::Result;
use crate::data::TestWindow;
use crate::nn::mlp::Mlp;
use crate::training::{
    forward_eval, forward_with_frozen_params, tau_trim_and_downsample, FrozenParams, Metrics,
};

/// Source of MC parameters at eval time.
pub enum EvalParams<'a, I: Backend> {
    Frozen(&'a FrozenParams),
    Mlp(&'a Mlp<I>),
}

pub struct EvalOutput {
    pub predictions_daily: Array2<f32>,  // (n_all_gauges, n_days_trimmed)
    pub observations_daily: Array2<f32>, // (n_all_gauges, n_days_trimmed)
    pub gage_ids: Vec<String>,
    pub time_range_daily: Vec<NaiveDate>,
    pub metrics: Metrics,
}

pub fn evaluate<I: Backend>(
    cfg: &Config,
    dataset: &MeritGagesDataset,
    params: EvalParams<I>,
    device: &I::Device,
    batch_size_days: usize,
) -> Result<EvalOutput> {
    let axis = dataset.time_axis().clone();
    let n_days_total = axis.num_days;
    assert!(batch_size_days > 0, "batch_size_days must be positive");

    // Probe with a 1-day chunk to size gauges (cheapest path that forces the
    // static-network cache to build).
    let probe = TestWindow::new(&axis, 0, 1);
    let probe_batch = dataset.collate_window(&probe)?;
    let n_all_gauges = probe_batch.gauge_staids.len();
    let gauge_staids = probe_batch.gauge_staids.clone();
    let n_hours_full = n_days_total * 24;

    // Accumulator: (n_all_gauges, n_hours_full) — written per chunk.
    let mut predictions_full = Array2::<f32>::zeros((n_all_gauges, n_hours_full));

    // Helper: dispatch the forward based on EvalParams. Returns (n_all_gauges, chunk_hours).
    let run_chunk = |window: &TestWindow, carry_state: bool| -> Result<Array2<f32>> {
        let batch = dataset.collate_window(window)?;
        let tensors = batch.to_tensors::<I>(device);
        let pred = match &params {
            EvalParams::Frozen(frozen) => {
                forward_with_frozen_params::<I>(cfg, &tensors, frozen, device, carry_state)
            }
            EvalParams::Mlp(mlp) => {
                forward_eval::<I>(cfg, &tensors, mlp, device, carry_state)
            }
        };
        let dims = pred.dims();
        debug_assert_eq!(dims[0], n_all_gauges);
        debug_assert_eq!(dims[1], window.n_hourly());
        let v: Vec<f32> = pred.into_data().into_vec().unwrap();
        Ok(Array2::from_shape_vec((dims[0], dims[1]), v).unwrap())
    };

    // Iterate chunks. First chunk is cold-start (carry_state=false); all
    // subsequent chunks carry the engine state.
    let mut day_offset = 0usize;
    let mut chunk_idx = 0usize;
    while day_offset < n_days_total {
        let chunk_n = (n_days_total - day_offset).min(batch_size_days);
        let win = TestWindow::new(&axis, day_offset, chunk_n);
        let pred_arr = run_chunk(&win, chunk_idx > 0)?;
        let h_start = day_offset * 24;
        let h_end = h_start + win.n_hourly();
        predictions_full.slice_mut(s![.., h_start..h_end]).assign(&pred_arr);
        day_offset += chunk_n;
        chunk_idx += 1;
    }

    // End-of-pipeline tau-trim + daily downsample. Lift the f32 accumulator
    // into a BURN tensor for the existing tau_trim_and_downsample helper.
    let pred_full_vec: Vec<f32> = predictions_full.iter().copied().collect();
    let pred_full_t: burn::tensor::Tensor<I, 2> =
        burn::tensor::Tensor::<I, 1>::from_floats(pred_full_vec.as_slice(), device)
            .reshape([n_all_gauges, n_hours_full]);
    let daily_t = tau_trim_and_downsample(pred_full_t, cfg.params.tau);
    let daily_dims = daily_t.dims();
    let daily_vec: Vec<f32> = daily_t.into_data().into_vec().unwrap();
    let predictions_daily =
        Array2::from_shape_vec((daily_dims[0], daily_dims[1]), daily_vec).unwrap();

    // Observations: use the cached full-period array (does NOT trigger a
    // streamflow read). Slice [1..-1] along axis 0 and transpose to
    // (G, n_days_full - 2) to match DDR's compute_daily_runoff convention.
    let obs_full = dataset.full_observations()?; // borrow of (n_days_full, G)
    let obs_trimmed: Array2<f32> = obs_full.slice(s![1..-1, ..]).to_owned();
    // Transpose (T, G) -> (G, T) and ensure contiguous storage.
    let observations_daily: Array2<f32> = obs_trimmed
        .reversed_axes()
        .as_standard_layout()
        .to_owned();

    // Predictions after tau_trim_and_downsample: shape (G, n_days_full - 1).
    // (Math: T_hours = n_days_full * 24; trim drops 24 hours total; /24 = n_days_full - 1.)
    // To match observations_daily's (G, n_days_full - 2), drop the LAST day
    // of predictions. (This SAFE CONSERVATIVE alignment is documented in the
    // SP-5 plan Task 6 design note; Task 11 V4 will surface any drift.)
    let pd_dims = predictions_daily.dim();
    let predictions_daily = predictions_daily
        .slice(s![.., 0..pd_dims.1 - 1])
        .to_owned();

    debug_assert_eq!(
        predictions_daily.shape()[1],
        observations_daily.shape()[1],
        "predictions/observations time-axis mismatch after [1..-1] alignment",
    );

    // Daily time range = axis.start + 1 .. axis.start + (n_days_full - 1).
    // Length n_days_full - 2 — matches DDR's daily_time_range[1:-1].
    let time_range_daily: Vec<NaiveDate> = (1..n_days_total - 1)
        .map(|i| axis.start + chrono::Duration::days(i as i64))
        .collect();
    debug_assert_eq!(time_range_daily.len(), predictions_daily.shape()[1]);

    let warmup = cfg.experiment.as_ref().expect("experiment").warmup;
    let metrics = Metrics::compute(
        &predictions_daily.slice(s![.., warmup..]).to_owned(),
        &observations_daily.slice(s![.., warmup..]).to_owned(),
    );

    let gage_ids: Vec<String> = gauge_staids
        .iter()
        .map(|s| s.as_str().to_string())
        .collect();

    Ok(EvalOutput {
        predictions_daily,
        observations_daily,
        gage_ids,
        time_range_daily,
        metrics,
    })
}

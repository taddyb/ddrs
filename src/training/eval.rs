//! Test-phase evaluation loop. Mirrors
//! `~/projects/ddr/scripts/train_and_test.py::_test` (lines 43-119).
//!
//! Unlike the training loop, batches iterate TIME (not gauges) and the
//! network is the static all-gauges union. Cross-chunk routing state is
//! threaded via `tensors.initial_state` (the final per-reach discharge column
//! of each chunk is injected as the initial discharge of the next chunk).

use std::path::Path;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Once;

use burn::tensor::backend::Backend;
use chrono::NaiveDate;
use ndarray::{s, Array2};

use crate::config::Config;
use crate::data::dataset::MeritGagesDataset;
use crate::data::error::{DataError, Result};
use crate::data::TestWindow;
use crate::nn::kan_head::KanHead;
use crate::training::{
    forward_eval_reaches, forward_with_frozen_params, scatter_add_by_group,
    tau_trim_and_downsample, FrozenParams, Metrics, ZetaSums,
};

/// Source of MC parameters at eval time.
pub enum EvalParams<'a, I: Backend> {
    Frozen(&'a FrozenParams),
    KanHead(&'a KanHead<I>),
}

pub struct EvalOutput {
    pub predictions_daily: Array2<f32>,  // (n_all_gauges, n_days_trimmed)
    pub observations_daily: Array2<f32>, // (n_all_gauges, n_days_trimmed)
    pub gage_ids: Vec<String>,
    pub time_range_daily: Vec<NaiveDate>,
    pub metrics: Metrics,
    /// Eval-window mean |zeta| per eval-network reach (m³/s). `Some` only when
    /// leakance was active (KanHead params + `use_leakance`).
    pub zeta_abs_mean: Option<Vec<f32>>,
    /// Eval-window mean signed zeta per reach (m³/s; positive = losing).
    pub zeta_net_mean: Option<Vec<f32>>,
    /// Eval-window mean routed depth per reach (m). Same gating as zeta.
    pub zeta_depth_mean: Option<Vec<f32>>,
    /// Eval-window mean plan-view wetted area `area_z` per reach (m²).
    pub zeta_area_z_mean: Option<Vec<f32>>,
    /// Eval-window mean routed discharge per reach (m³/s).
    pub zeta_q_mean: Option<Vec<f32>>,
    /// COMIDs aligned to the zeta vectors (eval-network topological order).
    pub zeta_comids: Option<Vec<i64>>,
}

/// Returns a diagnostic reason when a chunk's predictions are provably wrong
/// rather than merely low-skill: all exactly zero, or containing a
/// non-finite value. This is ONE fingerprint of the silent-corruption
/// failure mode below — kept as defense-in-depth alongside
/// [`worker_panic_detected`], since it doesn't depend on the panic hook
/// having fired.
fn corrupted_chunk_reason(pred: &Array2<f32>) -> Option<String> {
    if pred.iter().any(|v| !v.is_finite()) {
        return Some("non-finite value(s) in predictions".to_string());
    }
    if pred.iter().all(|&v| v == 0.0) {
        return Some("all-zero predictions".to_string());
    }
    None
}

/// Set by a global panic hook whenever ANY thread panics during an
/// `evaluate()` call. Primary detector for the silent-corruption failure
/// mode: a cubecl-cuda background worker thread (`DSD-0-0`) panics on a CUDA
/// OOM, but the panic never propagates to the caller and the main thread
/// keeps looping with whatever half-written buffer cubecl left behind.
/// `corrupted_chunk_reason`'s all-zero/non-finite check is NOT sufficient on
/// its own — confirmed 2026-07-16: a repeat of this exact incident produced
/// a chunk with plausible-looking finite, non-zero values (stale GPU memory
/// from a previous op) that slipped past it while the worker thread
/// panicked dozens of times underneath.
static WORKER_PANICKED: AtomicBool = AtomicBool::new(false);
static INSTALL_PANIC_HOOK: Once = Once::new();

/// Idempotent; wraps (does not replace) the default hook so panic messages
/// still print. Must be called before the eval chunk loop starts.
fn ensure_panic_hook_installed() {
    INSTALL_PANIC_HOOK.call_once(|| {
        let default_hook = std::panic::take_hook();
        std::panic::set_hook(Box::new(move |info| {
            WORKER_PANICKED.store(true, Ordering::SeqCst);
            default_hook(info);
        }));
    });
}

/// Clears any panic recorded before this call (e.g. from an unrelated prior
/// `evaluate()` invocation in the same process) and reports whether a panic
/// happened since.
fn take_worker_panicked() -> bool {
    WORKER_PANICKED.swap(false, Ordering::SeqCst)
}

pub fn evaluate<I: Backend>(
    cfg: &Config,
    dataset: &MeritGagesDataset,
    params: EvalParams<I>,
    device: &I::Device,
    batch_size_days: usize,
    checkpoint_path: &Path,
) -> Result<EvalOutput> {
    ensure_panic_hook_installed();
    take_worker_panicked(); // clear any stale flag from an unrelated prior call

    let axis = dataset.time_axis().clone();
    let n_days_total = axis.num_days;
    assert!(batch_size_days > 0, "batch_size_days must be positive");

    // Probe with a 1-day chunk to size gauges (cheapest path that forces the
    // static-network cache to build).
    let probe = TestWindow::new(&axis, 0, 1);
    let probe_batch = dataset.collate_window(&probe)?;
    let n_all_gauges = probe_batch.gauge_staids.len();
    let gauge_staids = probe_batch.gauge_staids.clone();
    let reach_comids: Vec<i64> = probe_batch.divide_comids.iter().map(|c| c.0).collect();
    let n_hours_full = n_days_total * 24;

    // Accumulator: (n_all_gauges, n_hours_full) — written per chunk.
    let mut predictions_full = Array2::<f32>::zeros((n_all_gauges, n_hours_full));

    // Leakance diagnostic: accumulate per-reach zeta sums across chunks.
    // Stays empty (steps == 0) when leakance is off or params are Frozen.
    let mut zeta_sums: ZetaSums<I> = ZetaSums::new();

    // Helper: dispatch the forward based on EvalParams.
    //
    // Returns `(gauge_predictions, Option<per_reach_final_col>)`:
    //   - gauge_predictions: (n_all_gauges, chunk_hours) — written into predictions_full.
    //   - per_reach_final_col: the last discharge column over all n_reaches in the
    //     eval network (not gauge-aggregated). `Some` for the KanHead path, `None`
    //     for FrozenParams (legacy baseline path that does not support state injection).
    //
    // `initial_state`: when `Some`, sets `tensors.initial_state` before the forward so
    // the engine starts from the previous chunk's final discharge instead of hotstarting.
    let mut run_chunk = |window: &TestWindow, initial_state: Option<Vec<f32>>| -> Result<(Array2<f32>, Option<Vec<f32>>)> {
        let batch = dataset.collate_window(window)?;
        let mut tensors = batch.to_tensors::<I>(device);
        // Inject cross-chunk state into tensors before the forward pass. This
        // takes priority over both the state cache and the hotstart heuristic.
        if let Some(ref q0) = initial_state {
            tensors.initial_state =
                Some(burn::tensor::Tensor::<I, 1>::from_floats(q0.as_slice(), device));
        }
        match &params {
            EvalParams::Frozen(frozen) => {
                // FrozenParams produces gauge-aggregated (G, T) directly with no
                // per-reach output to extract. State injection is NOT supported on
                // this path — it cold-restarts from hotstart on every chunk.
                // This is an acceptable limitation: FrozenParams is a legacy
                // verification baseline, never used in production eval.
                let pred = forward_with_frozen_params::<I>(cfg, &tensors, frozen, device, false);
                let dims = pred.dims();
                debug_assert_eq!(dims[0], n_all_gauges);
                debug_assert_eq!(dims[1], window.n_hourly());
                let v: Vec<f32> = pred.into_data().into_vec().unwrap();
                Ok((Array2::from_shape_vec((dims[0], dims[1]), v).unwrap(), None))
            }
            EvalParams::KanHead(head) => {
                // forward_eval_reaches returns per-reach (n_reaches, chunk_hours).
                // Capture the final column for cross-chunk state injection before
                // scatter-adding to gauge predictions.
                let runoff_reaches = forward_eval_reaches::<I>(
                    cfg,
                    &tensors,
                    head,
                    device,
                    false,
                    Some(&mut zeta_sums),
                    None,
                    None,
                );
                let [n_reaches, chunk_hours] = runoff_reaches.dims();
                let final_col: Vec<f32> = runoff_reaches
                    .clone()
                    .slice([0..n_reaches, chunk_hours - 1..chunk_hours])
                    .reshape([n_reaches])
                    .into_data()
                    .into_vec()
                    .unwrap();
                let pred = scatter_add_by_group(
                    runoff_reaches,
                    tensors.flat_indices.clone(),
                    tensors.group_ids.clone(),
                    tensors.num_gauges,
                );
                let dims = pred.dims();
                debug_assert_eq!(dims[0], n_all_gauges);
                debug_assert_eq!(dims[1], window.n_hourly());
                let v: Vec<f32> = pred.into_data().into_vec().unwrap();
                Ok((Array2::from_shape_vec((dims[0], dims[1]), v).unwrap(), Some(final_col)))
            }
        }
    };

    // Iterate chunks. The first chunk cold-starts (initial_state=None); each
    // subsequent chunk receives the previous chunk's final per-reach discharge
    // column as its initial state, eliminating the 35-56 m³/s restarts that
    // occurred when every chunk cold-hostarted from its own q'[0].
    //
    // FrozenParams note: the Frozen arm returns None for final_col, so
    // prev_final_state stays None and every chunk cold-restarts. This is
    // intentional — see the FrozenParams arm comment above.
    let n_chunks_total = n_days_total.div_ceil(batch_size_days);
    let mut prev_final_state: Option<Vec<f32>> = None;
    let mut day_offset = 0usize;
    let mut chunk_idx = 0usize;
    while day_offset < n_days_total {
        let chunk_n = (n_days_total - day_offset).min(batch_size_days);
        let win = TestWindow::new(&axis, day_offset, chunk_n);
        if chunk_idx > 0 {
            if let EvalParams::Frozen(_) = &params {
                eprintln!(
                    "WARN(eval): FrozenParams does not support cross-chunk state injection; \
                     chunk {}/{} cold-restarts from hotstart (legacy baseline path).",
                    chunk_idx + 1,
                    n_chunks_total,
                );
            }
        }
        let (pred_arr, final_col) = run_chunk(&win, prev_final_state.take())?;
        let panic_message = take_worker_panicked()
            .then(|| "a background thread panicked during this chunk".to_string());
        if let Some(message) = panic_message.or_else(|| corrupted_chunk_reason(&pred_arr)) {
            return Err(DataError::CorruptedEvalChunk {
                path: checkpoint_path.to_path_buf(),
                chunk: chunk_idx + 1,
                total: n_chunks_total,
                message,
            });
        }
        prev_final_state = final_col;
        let h_start = day_offset * 24;
        let h_end = h_start + win.n_hourly();
        predictions_full.slice_mut(s![.., h_start..h_end]).assign(&pred_arr);
        eprintln!(
            "  chunk {}/{}: days {}..{} ({} days)",
            chunk_idx + 1,
            n_chunks_total,
            day_offset,
            day_offset + chunk_n,
            chunk_n,
        );
        day_offset += chunk_n;
        chunk_idx += 1;
    }

    // DIAGNOSTIC (opt-in, off by default): dump the PRE-TRIM hourly series so
    // `params.tau` can be swept EXACTLY offline instead of re-running eval once
    // per tau. Raw row-major f32 (n_gauges, n_hours) + a `.json` dims sidecar.
    // An env var rather than a new parameter because `evaluate` has several
    // call sites and this is a throwaway probe.
    if let Ok(dump_path) = std::env::var("DDRS_HOURLY_DUMP") {
        use std::io::Write;
        let raw: Vec<f32> = predictions_full.iter().copied().collect();
        let bytes: &[u8] = unsafe {
            std::slice::from_raw_parts(raw.as_ptr() as *const u8, std::mem::size_of_val(&raw[..]))
        };
        std::fs::File::create(&dump_path)
            .and_then(|mut f| f.write_all(bytes))
            .unwrap_or_else(|e| panic!("DDRS_HOURLY_DUMP write to {dump_path} failed: {e}"));
        let meta = format!(
            r#"{{"n_gauges":{n_all_gauges},"n_hours":{n_hours_full},"dtype":"f32","order":"C","tau_shipped":{}}}"#,
            cfg.params.tau
        );
        std::fs::write(format!("{dump_path}.json"), meta).ok();
        eprintln!(
            "  hourly dump -> {dump_path} ({n_all_gauges} x {n_hours_full} f32, {:.2} GB)",
            (n_all_gauges * n_hours_full * 4) as f64 / 1e9
        );
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
    let n_days_full = obs_full.nrows();
    let obs_trimmed: Array2<f32> = obs_full.slice(s![1..n_days_full - 1, ..]).to_owned();
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

    // Leakance diagnostic: sums → per-reach means over the routed timesteps.
    let (zeta_abs_mean, zeta_net_mean, zeta_depth_mean, zeta_area_z_mean, zeta_q_mean, zeta_comids) =
        match (
            zeta_sums.abs_sum,
            zeta_sums.net_sum,
            zeta_sums.depth_sum,
            zeta_sums.area_z_sum,
            zeta_sums.q_sum,
            zeta_sums.steps,
        ) {
            (Some(abs), Some(net), Some(depth), Some(area_z), Some(q), steps) if steps > 0 => {
                let scale = 1.0_f32 / steps as f32;
                let mean = |t: burn::tensor::Tensor<I, 1>| -> Vec<f32> {
                    (t * scale).into_data().into_vec().unwrap()
                };
                (
                    Some(mean(abs)),
                    Some(mean(net)),
                    Some(mean(depth)),
                    Some(mean(area_z)),
                    Some(mean(q)),
                    Some(reach_comids),
                )
            }
            _ => (None, None, None, None, None, None),
        };

    // Final gate: the tau-trim/downsample and zeta-mean readbacks above also
    // run on the device, after the last per-chunk check — a worker panic
    // there would otherwise slip through as a clean `Ok`.
    if take_worker_panicked() {
        return Err(DataError::CorruptedEvalChunk {
            path: checkpoint_path.to_path_buf(),
            chunk: n_chunks_total,
            total: n_chunks_total,
            message: "a background thread panicked during post-loop tensor readback".to_string(),
        });
    }

    Ok(EvalOutput {
        predictions_daily,
        observations_daily,
        gage_ids,
        time_range_daily,
        metrics,
        zeta_abs_mean,
        zeta_net_mean,
        zeta_depth_mean,
        zeta_area_z_mean,
        zeta_q_mean,
        zeta_comids,
    })
}

#[cfg(test)]
mod tests {
    use super::{corrupted_chunk_reason, ensure_panic_hook_installed, take_worker_panicked};
    use ndarray::Array2;
    use std::sync::Mutex;

    // WORKER_PANICKED is a process-global static; serialize the two tests
    // that touch it so they can't race against each other under the default
    // parallel test harness.
    static PANIC_FLAG_TEST_LOCK: Mutex<()> = Mutex::new(());

    #[test]
    fn worker_thread_panic_sets_flag() {
        let _guard = PANIC_FLAG_TEST_LOCK.lock().unwrap();
        ensure_panic_hook_installed();
        take_worker_panicked(); // clear any residual state before asserting

        let result = std::thread::spawn(|| panic!("simulated cubecl worker panic")).join();
        assert!(result.is_err());
        assert!(take_worker_panicked(), "flag should be set after a background thread panic");
        assert!(!take_worker_panicked(), "flag should be cleared by the previous take");
    }

    #[test]
    fn no_panic_leaves_flag_clear() {
        let _guard = PANIC_FLAG_TEST_LOCK.lock().unwrap();
        ensure_panic_hook_installed();
        take_worker_panicked(); // clear any residual state
        assert!(!take_worker_panicked());
    }

    #[test]
    fn healthy_chunk_is_not_corrupted() {
        let pred = Array2::from_shape_vec((2, 3), vec![0.1, 1.0, 2.5, 0.0, 3.0, 4.0]).unwrap();
        assert!(corrupted_chunk_reason(&pred).is_none());
    }

    #[test]
    fn all_zero_chunk_is_corrupted() {
        let pred = Array2::<f32>::zeros((2, 3));
        assert_eq!(
            corrupted_chunk_reason(&pred),
            Some("all-zero predictions".to_string())
        );
    }

    #[test]
    fn nan_chunk_is_corrupted() {
        let mut pred = Array2::from_shape_vec((2, 2), vec![1.0, 2.0, 3.0, 4.0]).unwrap();
        pred[[1, 1]] = f32::NAN;
        assert_eq!(
            corrupted_chunk_reason(&pred),
            Some("non-finite value(s) in predictions".to_string())
        );
    }

    #[test]
    fn inf_chunk_is_corrupted() {
        let mut pred = Array2::from_shape_vec((1, 2), vec![1.0, 2.0]).unwrap();
        pred[[0, 1]] = f32::INFINITY;
        assert_eq!(
            corrupted_chunk_reason(&pred),
            Some("non-finite value(s) in predictions".to_string())
        );
    }
}

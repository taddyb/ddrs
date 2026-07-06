//! Cross-chunk routing continuity — guards the evaluate() fix.
//!
//! ## What the fix provides
//! `MuskingumCunge::forward()` returns shape `(n, T)` where column 0 is the
//! initial discharge Q_0 (hotstart or injected) and columns 1..T are the T-1
//! routed values. When evaluate() concatenates chunks, column 0 of each chunk
//! lands at `h_start = day_offset * 24` in `predictions_full`.
//!
//! With the OLD code (carry_state=chunk_idx>0 but fresh engine each chunk):
//!   - col 0 of chunk k+1 = a fresh hotstart derived from chunk k+1's own q'[0]
//!   - This does NOT equal chunk k's last routed value → discharge jump of 35-56 m³/s
//!
//! With the FIX (initial_state injection):
//!   - col 0 of chunk k+1 = chunk k's last routed column (Q_{T_k-1})
//!   - No jump; the boundary transitions smoothly
//!
//! ## On byte-identical concatenation
//! Equal-sized chunks cannot produce a byte-identical single-shot result because
//! `forward()` does not consume q'[T-1] (the last forcing row). When chunk2's q'
//! starts at sf_full[T_1], it uses sf_full[T_1] for the first routing step, whereas
//! the single-shot uses sf_full[T_1-1] — a one-step forcing offset. The second test
//! below uses an asymmetric split (chunk1 has T_1+1 rows, chunk2 starts 1 row back)
//! to achieve exact byte-identical concatenation.
//!
//! Both tests use the `MuskingumCunge::initial_state` injection mechanism that is
//! the exact mechanism `evaluate()` uses via `tensors.initial_state`.

mod common;

use burn::backend::Autodiff;
use burn::tensor::Tensor;
use common::{
    mock_config, mock_routing_inputs, mock_spatial_parameters, mock_streamflow, InnerBackend,
    TestDevice,
};
use ddrs::routing::MuskingumCunge;

type AB = Autodiff<InnerBackend>;

/// Route with the given streamflow tensor and optional initial discharge state.
/// Returns the full output as a row-major Vec<f32> of shape (n_reaches, t).
fn route_segment(
    sf: Tensor<AB, 2>,
    n: usize,
    device: &TestDevice,
    initial_state: Option<Vec<f32>>,
) -> Vec<f32> {
    let cfg = mock_config();
    let mut mc = MuskingumCunge::<InnerBackend>::new(cfg, device.clone());
    let initial_state_t =
        initial_state.map(|v| Tensor::<AB, 1>::from_floats(v.as_slice(), device));
    mc.setup_inputs(
        mock_routing_inputs(n, device),
        sf,
        mock_spatial_parameters(n, device),
        false,
        initial_state_t,
    );
    mc.forward().into_data().to_vec::<f32>().unwrap()
}

/// Extract the last timestep column from a (n, t) row-major Vec<f32>.
fn last_col(data: &[f32], n: usize, t: usize) -> Vec<f32> {
    (0..n).map(|r| data[r * t + (t - 1)]).collect()
}

/// Extract column `col_idx` from a (n, t) row-major Vec<f32>.
fn col(data: &[f32], n: usize, t: usize, col_idx: usize) -> Vec<f32> {
    (0..n).map(|r| data[r * t + col_idx]).collect()
}

// ---------------------------------------------------------------------------
// Test 1: boundary continuity — the primary defect guard
// ---------------------------------------------------------------------------

/// With state injection, col 0 of chunk2 equals chunk1's last col (no jump).
/// Without injection (cold restart, old behavior), col 0 differs — proving
/// the test detects the continuity defect.
#[test]
fn state_injection_gives_continuous_boundary() {
    let device = TestDevice::default();
    let n = 5usize;
    let half = 15usize; // steps per chunk
    let full = half * 2;

    let sf_full = mock_streamflow(full, n, &device);
    let sf1 = sf_full.clone().slice([0..half, 0..n]);
    let sf2 = sf_full.clone().slice([half..full, 0..n]);

    // Chunk 1: no injection.
    let out1 = route_segment(sf1, n, &device, None);
    let q_final_chunk1 = last_col(&out1, n, half); // Q_{T_1-1}

    // NEW behavior: inject chunk1's final discharge as initial_state for chunk2.
    let out2_injected = route_segment(sf2.clone(), n, &device, Some(q_final_chunk1.clone()));

    // Key property: col 0 of injected chunk2 must exactly equal the injected state.
    let injected_col0 = col(&out2_injected, n, half, 0);
    assert_eq!(
        injected_col0, q_final_chunk1,
        "with injection, chunk2 col 0 must equal chunk1's final discharge (no boundary jump)"
    );

    // OLD behavior: cold-restart chunk2 from its own hotstart.
    let out2_cold = route_segment(sf2, n, &device, None);
    let cold_col0 = col(&out2_cold, n, half, 0);

    // Cold-restart col 0 must DIFFER from chunk1's final discharge —
    // this proves the test detects the defect that evaluate() had.
    assert_ne!(
        cold_col0, q_final_chunk1,
        "without injection (old cold-restart), chunk2 col 0 must differ from chunk1's \
         final discharge — proves the test detects the continuity defect"
    );
}

// ---------------------------------------------------------------------------
// Test 2: byte-identical with asymmetric split
// ---------------------------------------------------------------------------

/// Demonstrates that with an asymmetric chunk split (chunk1 has T_1+1 rows,
/// chunk2's q' starts one row back from where chunk1's unused final row sits),
/// the routing is byte-identical to the single-shot run on the CPU backend.
///
/// Specifically: chunk1(T=16, sf[0:16]) + chunk2(T=16, sf[15:31], initial=chunk1[-1])
/// → skip col 0 of chunk2 (the injected state) → concatenated cols match full[0:31].
#[test]
fn asymmetric_split_is_byte_identical_to_single_shot() {
    let device = TestDevice::default();
    let n = 5usize;
    // full window: T_full = T_1+1 + T_2 = 16 + 15 = 31 steps.
    let t1_plus1 = 16usize; // chunk1 steps (includes the overlap row)
    let t2 = 16usize;        // chunk2 steps
    let t_full = 31usize;    // single-shot steps

    let sf_full = mock_streamflow(t_full, n, &device);
    // chunk1 uses rows 0..16; chunk2 uses rows 15..31 (overlaps by 1 row —
    // that overlap is the q'[T_1] that chunk1 left unused).
    let sf1 = sf_full.clone().slice([0..t1_plus1, 0..n]);
    let sf2 = sf_full.clone().slice([t1_plus1 - 1..t_full, 0..n]);

    // Single-shot ground truth: T=31 → output (n, 31).
    let out_full = route_segment(sf_full, n, &device, None);

    // Chunk 1 (T=16): output (n, 16) = [Q_0..Q_15].
    let out1 = route_segment(sf1, n, &device, None);

    // Chunk 1 must match full[0..16] exactly.
    for r in 0..n {
        for t in 0..t1_plus1 {
            let got = out1[r * t1_plus1 + t];
            let want = out_full[r * t_full + t];
            assert_eq!(
                got, want,
                "chunk1[reach={r}, t={t}] = {got} != full[reach={r}, t={t}] = {want}"
            );
        }
    }

    // Inject chunk1's last column as initial_state for chunk2.
    let q0 = last_col(&out1, n, t1_plus1);
    let out2 = route_segment(sf2, n, &device, Some(q0.clone()));

    // col 0 of chunk2 = injected Q_15.
    let chunk2_col0 = col(&out2, n, t2, 0);
    assert_eq!(chunk2_col0, q0, "chunk2 col 0 must equal the injected state Q_15");

    // chunk2 cols 1..t2 must match full[t1_plus1..t_full] exactly.
    // (chunk2 uses sf_full[15..30] → q'[0]=sf_full[15] routes Q_15→Q_16, same as full.)
    for r in 0..n {
        for t in 1..t2 {
            let got = out2[r * t2 + t];
            let want = out_full[r * t_full + (t1_plus1 - 1 + t)];
            assert_eq!(
                got, want,
                "chunk2[reach={r}, t={t}] = {got} != full[reach={r}, t={}] = {want}",
                t1_plus1 - 1 + t
            );
        }
    }

    // Old behavior: chunk2 cold-restart must differ from the injected version.
    let sf2_cold = mock_streamflow(t_full, n, &device)
        .slice([t1_plus1 - 1..t_full, 0..n]);
    let out2_cold = route_segment(sf2_cold, n, &device, None);
    assert_ne!(
        out2_cold, out2,
        "cold-restart chunk2 must differ from the state-injected version"
    );
}

//! The S28 clamp silently turns negative solves into +1e-4. This counter is
//! the only way to see how often Muskingum's non-negative-coefficient
//! condition (2X <= Cr <= 2(1-X)) is violated in a real run.
//!
//! Both test cases run inside ONE `#[test]` function to avoid races on the
//! process-global `NEG_SOLVES`/`TOTAL_SOLVES` atomics. Running this file
//! with `--test-threads=1` would also be safe but is not required here.
use std::sync::Arc;

use burn::backend::NdArray;
use burn::tensor::Tensor;

use ddrs::config::Config;
use ddrs::routing::mmc_op::{negative_solve_stats, reset_negative_solve_stats, timestep_forward};
use ddrs::sparse::{AValuesAssembler, CsrPattern, SparseAdjacency};

type I = NdArray<f32>;

/// 4-reach linear chain (reach k feeds reach k+1).
fn linear_chain(n: usize, length_m: Vec<f32>, slope: Vec<f32>) -> SparseAdjacency {
    let mut dense = vec![0.0_f32; n * n];
    for i in 0..n - 1 {
        dense[(i + 1) * n + i] = 1.0;
    }
    SparseAdjacency::from_dense(n, &dense, length_m, slope)
}

fn base_cfg() -> Config {
    let mut cfg = Config::default();
    cfg.params.parameter_ranges.n = [0.01, 0.3];
    cfg.params.parameter_ranges.q_spatial = [0.1, 0.9];
    cfg.params.parameter_ranges.p_spatial = [1.0, 200.0];
    cfg.params.attribute_minimums.velocity = 0.01;
    cfg.params.attribute_minimums.depth = 0.001;
    cfg.params.attribute_minimums.discharge = 1e-4;
    cfg.params.attribute_minimums.bottom_width = 0.01;
    cfg.params.attribute_minimums.slope = 0.0001;
    cfg.params.defaults.insert("p_spatial".to_string(), 1.0);
    cfg.params.log_space_parameters = vec![];
    cfg
}

#[test]
fn counter_starts_at_zero_and_positive_control() {
    let device = <I as burn::tensor::backend::BackendTypes>::Device::default();

    // -----------------------------------------------------------------------
    // Case A: zero counter after reset (tautology guard)
    // -----------------------------------------------------------------------
    reset_negative_solve_stats();
    let (neg, total) = negative_solve_stats();
    assert_eq!(neg, 0, "neg count must be zero after reset");
    assert_eq!(total, 0, "total count must be zero after reset");

    // -----------------------------------------------------------------------
    // Case B: positive control — inputs that drive x_sol[headwater] < 0
    //
    // Design target: c3 = (2K(1-X) - dt) / denom < 0 at every reach.
    //   K = L / celerity,  celerity = v_clamped * 5/3
    // With L=100 m, slope=0.05, n=0.01, q_t=1e5 m³/s (large) the Manning
    // formula gives a large velocity → large celerity → small K.
    //   At these numbers: depth ≈ large, v_Manning well above clamp →
    //   celerity ≈ 15 * 5/3 = 25 m/s → K ≈ 100/25 = 4 s.
    //   dt = 3600 s, X = 0.3 → 2K(1-X) = 5.6 s << 3600 → c3 < 0.
    // The headwater reach has no upstream (i_t[0]=0), so
    //   b_rhs[0] = c2*0 + c3*q_t[0] + c4*q_prime_t[0]
    // With q_t[0]=1e5 and q_prime_t[0]=1e-3, the large negative c3*q_t
    // dominates → b_rhs[0] < 0 → x_sol[0] < 0 before the S28 clamp.
    // -----------------------------------------------------------------------
    reset_negative_solve_stats();

    const N: usize = 4;
    let adj = linear_chain(
        N,
        vec![100.0_f32; N], // very short reaches (100 m)
        vec![0.05_f32; N],  // steep slope
    );
    let pattern = Arc::new(CsrPattern::from_sparse(&adj));
    let assembler = AValuesAssembler::<I>::new(&pattern, &device);
    let cfg = base_cfg();

    let mk = |v: f32| Tensor::<burn::backend::Autodiff<I>, 1>::from_floats([v; N].as_ref(), &device);
    // n=0.01 → low Manning → high velocity; q_spatial=0.5; p_spatial=20; x_storage=0.3
    let n_t = mk(0.01);
    let qsp_t = mk(0.5);
    let psp_t = mk(20.0);
    let x_t = mk(0.3);
    // Very large current discharge to maximise |c3 * q_t|
    let qt_t: burn::tensor::Tensor<burn::backend::Autodiff<I>, 1> =
        Tensor::from_floats([1e5_f32; N].as_ref(), &device);
    // Tiny inflow so c4*q_prime_t is negligible
    let qpt_t: burn::tensor::Tensor<burn::backend::Autodiff<I>, 1> =
        Tensor::from_floats([1e-3_f32; N].as_ref(), &device);

    let _q_next = timestep_forward::<I>(
        &cfg,
        &pattern,
        &assembler,
        n_t,
        qsp_t,
        psp_t,
        qt_t,
        qpt_t,
        Tensor::from_floats(adj.length_m.as_slice(), &device),
        Tensor::from_floats(adj.slope.as_slice(), &device),
        x_t,
        true, // track_neg = ON
    );

    let (neg_b, total_b) = negative_solve_stats();
    assert!(
        neg_b > 0,
        "positive control: expected at least one negative solve before the S28 clamp, \
         got 0 out of {total_b}. Check that c3 < 0 with the chosen L/slope/n/q_t."
    );

    // -----------------------------------------------------------------------
    // Case C: negative control — inputs that keep c3 >= 0
    //
    // Design target: c3 >= 0  ⟺  2K(1-X) >= dt = 3600 s
    //   With L=10000 m, low celerity (large n, low q): K = L/c ≈ 10000/0.1 = 1e5 s
    //   → 2K(1-X) = 1.4e5 >> 3600 → c3 > 0.
    // With smooth inflow q_t ≈ q_prime_t all coefficients positive →
    // x_sol > 0 at every reach, no negatives.
    // -----------------------------------------------------------------------
    reset_negative_solve_stats();

    let adj2 = linear_chain(
        N,
        vec![10_000.0_f32; N], // long reaches (10 km)
        vec![0.001_f32; N],    // gentle slope
    );
    let pattern2 = Arc::new(CsrPattern::from_sparse(&adj2));
    let assembler2 = AValuesAssembler::<I>::new(&pattern2, &device);
    let cfg2 = base_cfg();

    let mk2 = |v: f32| Tensor::<burn::backend::Autodiff<I>, 1>::from_floats([v; N].as_ref(), &device);
    let n_t2 = mk2(0.04);   // typical Manning's n
    let qsp_t2 = mk2(0.4);
    let psp_t2 = mk2(20.0);
    let x_t2 = mk2(0.3);
    let qt_t2 = mk2(100.0); // moderate discharge
    let qpt_t2 = mk2(10.0); // smooth inflow

    let _q_next2 = timestep_forward::<I>(
        &cfg2,
        &pattern2,
        &assembler2,
        n_t2,
        qsp_t2,
        psp_t2,
        qt_t2,
        qpt_t2,
        Tensor::from_floats(adj2.length_m.as_slice(), &device),
        Tensor::from_floats(adj2.slope.as_slice(), &device),
        x_t2,
        true, // track_neg = ON
    );

    let (neg_c, total_c) = negative_solve_stats();
    assert_eq!(
        neg_c, 0,
        "negative control: expected zero negative solves with long reaches / gentle slope / \
         typical flow, got {neg_c} out of {total_c}"
    );
}

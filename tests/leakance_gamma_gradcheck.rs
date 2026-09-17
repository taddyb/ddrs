//! Finite-difference autograd gradcheck for the fused `TimestepLeakanceGammaOp`
//! (`Backward<I, 9>`): losing-stream leakance and a LEARNED stage-dependent
//! roughness `n(d) = n_0·(d/d_ref)^(−gamma)` active at the same time.
//!
//! Why this file exists separately from `leakance_gradcheck.rs`: before
//! 2026-09-17 the two features could not coexist. `TimestepLeakanceOp` has
//! eight parents and none of them is gamma, so a learned gamma would have
//! silently received no gradient, and `validate_learned_gamma` rejected the
//! combination at config load rather than let that happen. The nine-parent
//! sibling op lifts that restriction; `leakance_gradcheck.rs` still pins the
//! eight-parent path unchanged.
//!
//! What is actually at risk here, and therefore what this test is for: gamma
//! and zeta both reach `b_rhs`, but by different routes. Gamma re-weights the
//! whole geometry chain below depth (the depth exponent B5, the velocity's
//! explicit `(d/d_ref)^gamma` B15, and the celerity's `gamma·A/(T·d)` B17),
//! while zeta reads `depth`, `p_spatial`, and `q_eps` and is subtracted from
//! `b_rhs` (B27). The two therefore couple through the shared geometry
//! accumulators in `timestep_backward_core`. Analytically that coupling needs
//! no new cross-term, because the hook injects zeta's geometry grads into the
//! same accumulators gamma already reads. This test is what makes that a
//! measured claim rather than an assumed one: every one of the nine parents is
//! swept against central differences with BOTH features on.
//!
//! Base point: leakance params interior (K_D=5e-7, d_gw=0.0, factor=0.5) on a
//! losing chain (depth > d_gw), gamma=0.3 interior to the `[0, 0.5]` box.
//! Tolerance matches `sp8_gradcheck` and `leakance_gradcheck`.

use std::sync::Arc;

use burn::backend::{Autodiff, NdArray};
use burn::tensor::Tensor;

use ddrs::config::Config;
use ddrs::routing::mmc_op::timestep_forward_leakance;
use ddrs::sparse::{AValuesAssembler, CsrPattern, SparseAdjacency};

type I = NdArray<f32>;
type AB = Autodiff<I>;

const N: usize = 4;
const EPS: f32 = 1e-3;
const REL_TOL: f32 = 5e-3;
const ABS_TOL: f32 = 1e-4;

const K_D: f32 = 5e-7;
const D_GW: f32 = 0.0;
const LEAK_FAC: f32 = 0.5;
/// Interior of `parameter_ranges.gamma` (`[0, 0.5]`), and the same value
/// `sp8_gradcheck` uses for its learned-gamma sweep.
const GAMMA: f32 = 0.3;

#[derive(Copy, Clone, Debug)]
enum Parent {
    N,
    QSpatial,
    PSpatial,
    QT,
    QPrimeT,
    KD,
    DGW,
    LeakFactor,
    Gamma,
}

fn linear_chain_sparse() -> SparseAdjacency {
    let mut dense = vec![0.0_f32; N * N];
    for i in 0..N - 1 {
        dense[(i + 1) * N + i] = 1.0;
    }
    SparseAdjacency::from_dense(N, &dense, vec![1000.0; N], vec![0.001; N])
}

/// `d_ref` must be set even though gamma is learned: it is read from config in
/// both cases (only the exponent is per-reach), so a config with no
/// `stage_roughness` block would supply `d_ref = 1.0` by default. Set it
/// explicitly so the test does not depend on that default.
fn mock_cfg() -> Config {
    let mut cfg = Config::default();
    cfg.params.stage_roughness = Some(ddrs::config::StageRoughnessSection {
        gamma: 0.0,
        d_ref: 1.0,
    });
    cfg.params.parameter_ranges.n = [0.01, 0.1];
    cfg.params.parameter_ranges.q_spatial = [0.1, 0.9];
    cfg.params.parameter_ranges.p_spatial = [1.0, 200.0];
    cfg.params.attribute_minimums.velocity = 0.1;
    cfg.params.attribute_minimums.depth = 0.01;
    cfg.params.attribute_minimums.discharge = 0.001;
    cfg.params.attribute_minimums.bottom_width = 0.1;
    cfg.params.attribute_minimums.slope = 0.001;
    cfg.params.defaults.insert("p_spatial".to_string(), 1.0);
    cfg.params.log_space_parameters = vec![];
    cfg
}

fn default_inputs() -> (Vec<f32>, Vec<f32>, Vec<f32>, Vec<f32>, Vec<f32>) {
    let n_vec = vec![0.035f32; N];
    let qsp_vec = vec![0.4f32; N];
    let psp_vec = vec![20.0f32; N];
    let qt_vec = vec![100.0f32, 120.0, 140.0, 160.0];
    let qpt_vec = vec![10.0f32, 12.0, 14.0, 16.0];
    (n_vec, qsp_vec, psp_vec, qt_vec, qpt_vec)
}

fn x_storage_vec() -> Vec<f32> { vec![0.3f32; N] }
fn kd_vec() -> Vec<f32> { vec![K_D; N] }
fn dgw_vec() -> Vec<f32> { vec![D_GW; N] }
fn fac_vec() -> Vec<f32> { vec![LEAK_FAC; N] }
fn gamma_vec() -> Vec<f32> { vec![GAMMA; N] }

struct GradTensors {
    n: Tensor<AB, 1>,
    qsp: Tensor<AB, 1>,
    psp: Tensor<AB, 1>,
    qt: Tensor<AB, 1>,
    qpt: Tensor<AB, 1>,
    kd: Tensor<AB, 1>,
    dgw: Tensor<AB, 1>,
    fac: Tensor<AB, 1>,
    gamma: Option<Tensor<AB, 1>>,
}

struct Inputs<'a> {
    n: &'a [f32],
    qsp: &'a [f32],
    psp: &'a [f32],
    qt: &'a [f32],
    qpt: &'a [f32],
    kd: &'a [f32],
    dgw: &'a [f32],
    fac: &'a [f32],
    /// `None` routes through the eight-parent `TimestepLeakanceOp`, which is
    /// what the identity tests at the bottom of this file compare against.
    gamma: Option<&'a [f32]>,
}

fn run_forward(
    cfg: &Config,
    pattern: &Arc<CsrPattern>,
    assembler: &AValuesAssembler<I>,
    device: &<I as burn::tensor::backend::BackendTypes>::Device,
    length_vec: &[f32],
    slope_vec: &[f32],
    inp: &Inputs,
    req: Option<Parent>,
) -> (Tensor<AB, 1>, GradTensors) {
    let mk = |data: &[f32], on: bool| -> Tensor<AB, 1> {
        let t: Tensor<AB, 1> = Tensor::from_floats(data, device);
        if on { t.require_grad() } else { t }
    };
    let n_t = mk(inp.n, matches!(req, Some(Parent::N)));
    let qsp_t = mk(inp.qsp, matches!(req, Some(Parent::QSpatial)));
    let psp_t = mk(inp.psp, matches!(req, Some(Parent::PSpatial)));
    let qt_t = mk(inp.qt, matches!(req, Some(Parent::QT)));
    let qpt_t = mk(inp.qpt, matches!(req, Some(Parent::QPrimeT)));
    let kd_t = mk(inp.kd, matches!(req, Some(Parent::KD)));
    let dgw_t = mk(inp.dgw, matches!(req, Some(Parent::DGW)));
    let fac_t = mk(inp.fac, matches!(req, Some(Parent::LeakFactor)));
    let gamma_t = inp.gamma.map(|g| mk(g, matches!(req, Some(Parent::Gamma))));
    let xst_t = mk(&x_storage_vec(), false);
    let length_t = mk(length_vec, false);
    let slope_t = mk(slope_vec, false);

    let q_next = timestep_forward_leakance::<I>(
        cfg,
        pattern,
        assembler,
        n_t.clone(),
        qsp_t.clone(),
        psp_t.clone(),
        qt_t.clone(),
        qpt_t.clone(),
        length_t,
        slope_t,
        xst_t,
        kd_t.clone(),
        dgw_t.clone(),
        fac_t.clone(),
        None,
        None,
        false,
        gamma_t.clone(),
    );

    (
        q_next,
        GradTensors {
            n: n_t, qsp: qsp_t, psp: psp_t, qt: qt_t, qpt: qpt_t,
            kd: kd_t, dgw: dgw_t, fac: fac_t, gamma: gamma_t,
        },
    )
}

struct Harness {
    cfg: Config,
    pattern: Arc<CsrPattern>,
    assembler: AValuesAssembler<I>,
    device: <I as burn::tensor::backend::BackendTypes>::Device,
    length: Vec<f32>,
    slope: Vec<f32>,
}

fn harness() -> Harness {
    let cfg = mock_cfg();
    let adj = linear_chain_sparse();
    let device = <I as burn::tensor::backend::BackendTypes>::Device::default();
    let pattern = Arc::new(CsrPattern::from_sparse(&adj));
    let assembler = AValuesAssembler::<I>::new(&pattern, &device);
    Harness {
        cfg,
        pattern,
        assembler,
        device,
        length: adj.length_m.clone(),
        slope: adj.slope.clone(),
    }
}

fn compute_analytical_grad(parent: Parent) -> Vec<f32> {
    let h = harness();
    let (n_vec, qsp_vec, psp_vec, qt_vec, qpt_vec) = default_inputs();
    let (kd, dgw, fac, gm) = (kd_vec(), dgw_vec(), fac_vec(), gamma_vec());
    let inp = Inputs {
        n: &n_vec, qsp: &qsp_vec, psp: &psp_vec, qt: &qt_vec, qpt: &qpt_vec,
        kd: &kd, dgw: &dgw, fac: &fac, gamma: Some(&gm),
    };
    let (q_next, parents) = run_forward(
        &h.cfg, &h.pattern, &h.assembler, &h.device, &h.length, &h.slope, &inp, Some(parent),
    );

    let grads = q_next.sum().backward();
    let g = match parent {
        Parent::N => parents.n.grad(&grads).expect("grad on n"),
        Parent::QSpatial => parents.qsp.grad(&grads).expect("grad on q_spatial"),
        Parent::PSpatial => parents.psp.grad(&grads).expect("grad on p_spatial"),
        Parent::QT => parents.qt.grad(&grads).expect("grad on q_t"),
        Parent::QPrimeT => parents.qpt.grad(&grads).expect("grad on q_prime_t"),
        Parent::KD => parents.kd.grad(&grads).expect("grad on K_D"),
        Parent::DGW => parents.dgw.grad(&grads).expect("grad on d_gw"),
        Parent::LeakFactor => parents.fac.grad(&grads).expect("grad on leakance_factor"),
        Parent::Gamma => parents
            .gamma
            .as_ref()
            .expect("gamma tensor present")
            .grad(&grads)
            .expect("grad on gamma — the ninth parent is the whole point of this op"),
    };
    g.into_data().to_vec::<f32>().unwrap()
}

fn compute_fd_grad(parent: Parent) -> Vec<f32> {
    let h = harness();
    let (n_vec, qsp_vec, psp_vec, qt_vec, qpt_vec) = default_inputs();
    let (base_kd, base_dgw, base_fac, base_gm) = (kd_vec(), dgw_vec(), fac_vec(), gamma_vec());

    let eval_loss = |n: &[f32], qsp: &[f32], psp: &[f32], qt: &[f32], qpt: &[f32],
                     kd: &[f32], dgw: &[f32], fac: &[f32], gm: &[f32]| -> f32 {
        let inp = Inputs {
            n, qsp, psp, qt, qpt, kd, dgw, fac, gamma: Some(gm),
        };
        let (q_next, _) = run_forward(
            &h.cfg, &h.pattern, &h.assembler, &h.device, &h.length, &h.slope, &inp, None,
        );
        q_next.sum().into_data().to_vec::<f32>().unwrap()[0]
    };

    let mut grad = vec![0.0f32; N];
    for i in 0..N {
        let mut p = (n_vec.clone(), qsp_vec.clone(), psp_vec.clone(), qt_vec.clone(),
                     qpt_vec.clone(), base_kd.clone(), base_dgw.clone(), base_fac.clone(),
                     base_gm.clone());
        let mut m = p.clone();

        let (plus, minus, base): (&mut Vec<f32>, &mut Vec<f32>, &Vec<f32>) = match parent {
            Parent::N => (&mut p.0, &mut m.0, &n_vec),
            Parent::QSpatial => (&mut p.1, &mut m.1, &qsp_vec),
            Parent::PSpatial => (&mut p.2, &mut m.2, &psp_vec),
            Parent::QT => (&mut p.3, &mut m.3, &qt_vec),
            Parent::QPrimeT => (&mut p.4, &mut m.4, &qpt_vec),
            Parent::KD => (&mut p.5, &mut m.5, &base_kd),
            Parent::DGW => (&mut p.6, &mut m.6, &base_dgw),
            Parent::LeakFactor => (&mut p.7, &mut m.7, &base_fac),
            Parent::Gamma => (&mut p.8, &mut m.8, &base_gm),
        };
        // zeta is EXACTLY LINEAR in K_D, d_gw and leakance_factor, so central
        // differences carry zero truncation error for them at any step size.
        // A step scaled to K_D (≈5e-10) would vanish into the f32 round-off
        // floor of `q_next.sum()` (O(500)), so those three take a large step to
        // lift the signal above round-off. Same reasoning as
        // `leakance_gradcheck.rs`; gamma is NOT linear and keeps sp8's scheme.
        let eps = match parent {
            Parent::KD => 4e-7,
            Parent::DGW => 1.5,
            Parent::LeakFactor => 0.4,
            _ => (EPS * base[i].abs()).max(EPS),
        };
        plus[i] = base[i] + eps;
        minus[i] = base[i] - eps;

        let l_plus = eval_loss(&p.0, &p.1, &p.2, &p.3, &p.4, &p.5, &p.6, &p.7, &p.8);
        let l_minus = eval_loss(&m.0, &m.1, &m.2, &m.3, &m.4, &m.5, &m.6, &m.7, &m.8);
        grad[i] = (l_plus - l_minus) / (2.0 * eps);
    }
    grad
}

fn compare_grads(name: &str, analytical: &[f32], fd: &[f32]) {
    assert_eq!(analytical.len(), fd.len());
    println!("--- {name} ---");
    let mut worst_rel = 0.0f32;
    let mut worst_abs = 0.0f32;
    for i in 0..analytical.len() {
        let (a, f) = (analytical[i], fd[i]);
        let abs_diff = (a - f).abs();
        let rel_diff = abs_diff / a.abs().max(f.abs()).max(1e-12);
        worst_abs = worst_abs.max(abs_diff);
        worst_rel = worst_rel.max(rel_diff);
        println!("  [{i}] analytical={a:.6e}  fd={f:.6e}  abs={abs_diff:.3e}  rel={rel_diff:.3e}");
    }
    println!("  worst abs={worst_abs:.3e}  worst rel={worst_rel:.3e}");
    assert!(
        worst_rel < REL_TOL || worst_abs < ABS_TOL,
        "{name}: gradcheck failed (worst rel={worst_rel:.3e}, abs={worst_abs:.3e})"
    );
}

fn run(name: &str, parent: Parent) {
    compare_grads(name, &compute_analytical_grad(parent), &compute_fd_grad(parent));
}

// --- the nine parents, all swept with BOTH leakance and n(d) active ---------

#[test] fn gradcheck_n() { run("n", Parent::N); }
#[test] fn gradcheck_q_spatial() { run("q_spatial", Parent::QSpatial); }
#[test] fn gradcheck_p_spatial() { run("p_spatial", Parent::PSpatial); }
#[test] fn gradcheck_q_t() { run("q_t", Parent::QT); }
#[test] fn gradcheck_q_prime_t() { run("q_prime_t", Parent::QPrimeT); }
#[test] fn gradcheck_k_d() { run("K_D", Parent::KD); }
#[test] fn gradcheck_d_gw() { run("d_gw", Parent::DGW); }
#[test] fn gradcheck_leakance_factor() { run("leakance_factor", Parent::LeakFactor); }

/// The one that did not exist before the nine-parent op. A regression here
/// means gamma is being trained against a wrong gradient while leakance is on.
#[test] fn gradcheck_gamma() { run("gamma", Parent::Gamma); }

// --- identities: the new path must not disturb the old one -----------------

fn forward_only(cfg: &Config, gamma: Option<&[f32]>) -> Vec<f32> {
    let h = Harness { cfg: cfg.clone(), ..harness() };
    let (n_vec, qsp_vec, psp_vec, qt_vec, qpt_vec) = default_inputs();
    let (kd, dgw, fac) = (kd_vec(), dgw_vec(), fac_vec());
    let inp = Inputs {
        n: &n_vec, qsp: &qsp_vec, psp: &psp_vec, qt: &qt_vec, qpt: &qpt_vec,
        kd: &kd, dgw: &dgw, fac: &fac, gamma,
    };
    let (q, _) = run_forward(
        &h.cfg, &h.pattern, &h.assembler, &h.device, &h.length, &h.slope, &inp, None,
    );
    q.into_data().to_vec::<f32>().unwrap()
}

/// A learned gamma of all zeros must reproduce plain leakance BIT-FOR-BIT.
///
/// This is the guard that adding the ninth parent did not perturb any existing
/// leakance run: `gamma = 0` makes `(d/d_ref)^(−gamma) ≡ 1`, and multiplying
/// by exactly 1.0 is exact in f32, so "close enough" is not the right bar here.
#[test]
fn learned_gamma_zero_is_bit_identical_to_plain_leakance() {
    let cfg = mock_cfg();
    let zeros = vec![0.0f32; N];
    let with_gamma = forward_only(&cfg, Some(&zeros));
    let without = forward_only(&cfg, None);
    assert_eq!(
        with_gamma, without,
        "a learned gamma of 0 changed the leakance forward; the nine-parent op \
         is not a superset of the eight-parent one"
    );
}

/// A learned gamma held constant at `c` must agree with the global
/// `stage_roughness.gamma = c` path, which routes through the eight-parent op.
///
/// Not bit-exact by construction: the constant path broadcasts a scalar while
/// the learned path carries a per-reach tensor, so the two differ in operation
/// order. Agreement to the f32 floor is the claim.
#[test]
fn learned_constant_gamma_matches_the_global_stage_roughness_path() {
    let learned = forward_only(&mock_cfg(), Some(&gamma_vec()));

    let mut cfg_const = mock_cfg();
    cfg_const.params.stage_roughness = Some(ddrs::config::StageRoughnessSection {
        gamma: GAMMA,
        d_ref: 1.0,
    });
    let global = forward_only(&cfg_const, None);

    let worst = learned
        .iter()
        .zip(&global)
        .map(|(a, b)| (a - b).abs() / a.abs().max(b.abs()).max(1e-12))
        .fold(0.0f32, f32::max);
    println!("learned-vs-global constant gamma: worst rel = {worst:.3e}");
    assert!(
        worst < 1e-5,
        "learned constant gamma disagrees with the global path (worst rel {worst:.3e})"
    );
}

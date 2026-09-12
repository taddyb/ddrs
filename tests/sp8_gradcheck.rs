//! SP-8 Task 3: numerical-vs-analytical gradient check for the fused
//! TimestepOp. Builds a small synthetic NdArray network, runs one
//! `timestep_forward`, and compares the analytical gradient against central
//! finite differences for each of the 5 tracked parents.
//!
//! Tolerance: 1e-3 relative — matches the existing sparse_gradcheck pattern.
//! For slots that are saturated by clamps (analytical grad == 0), we accept
//! ABS_TOL on the difference, since the finite-difference estimate may pick
//! up tiny one-sided shifts.

use std::sync::Arc;

use burn::backend::{Autodiff, NdArray};
use burn::tensor::Tensor;

use ddrs::config::Config;
use ddrs::routing::mmc_op::timestep_forward;
use ddrs::sparse::{AValuesAssembler, CsrPattern, SparseAdjacency};

type I = NdArray<f32>;
type AB = Autodiff<I>;

const N: usize = 4;
const EPS: f32 = 1e-3;
const REL_TOL: f32 = 5e-3;
const ABS_TOL: f32 = 1e-4;

#[derive(Copy, Clone, Debug)]
enum Parent {
    N,
    QSpatial,
    PSpatial,
    QT,
    QPrimeT,
    /// The learned stage-roughness exponent, i.e. the sixth parent that only
    /// `TimestepGammaOp` carries.
    Gamma,
}

fn linear_chain_sparse() -> SparseAdjacency {
    let mut dense = vec![0.0_f32; N * N];
    for i in 0..N - 1 {
        // adj[i+1, i] = 1: each reach inherits flow from upstream.
        dense[(i + 1) * N + i] = 1.0;
    }
    SparseAdjacency::from_dense(N, &dense, vec![1000.0; N], vec![0.001; N])
}

fn mock_cfg() -> Config {
    mock_cfg_gamma(0.0)
}

/// Same fixture with stage-dependent roughness switched on.
///
/// `gamma > 0` opens three gradient paths that do not exist at `gamma = 0`:
/// the depth exponent's denominator (B5), the velocity's explicit
/// `(d/d_ref)^gamma` (B15), and the celerity's `gamma·A/(T·d)` (B17). Two of
/// those feed NEW contributions into the depth accumulator, so the whole
/// geometry chain below depth is re-weighted. None of it is exercised by the
/// forward-only checks in `tests/stage_roughness.rs`.
fn mock_cfg_gamma(gamma: f32) -> Config {
    let mut cfg = Config::default();
    if gamma != 0.0 {
        cfg.params.stage_roughness = Some(ddrs::config::StageRoughnessSection {
            gamma,
            d_ref: 1.0,
        });
    }
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

/// Build inputs that put values in well-conditioned (non-saturated) regimes:
///   - n        ≈ 0.035 (Manning's)
///   - q_spatial ≈ 0.4
///   - p_spatial ≈ 20.0
///   - q_t      ≈ 100 m³/s
///   - q_prime  ≈ 10 m³/s
/// On a 4-reach 1km/0.1% chain, this produces velocity ≈ 1.5 m/s, no clamp saturation.
fn default_inputs() -> (Vec<f32>, Vec<f32>, Vec<f32>, Vec<f32>, Vec<f32>) {
    let n_vec = vec![0.035f32; N];
    let qsp_vec = vec![0.4f32; N];
    let psp_vec = vec![20.0f32; N];
    let qt_vec = vec![100.0f32, 120.0, 140.0, 160.0];
    let qpt_vec = vec![10.0f32, 12.0, 14.0, 16.0];
    (n_vec, qsp_vec, psp_vec, qt_vec, qpt_vec)
}

#[allow(clippy::too_many_arguments)]
fn run_forward_loss(
    cfg: &Config,
    pattern: &Arc<CsrPattern>,
    assembler: &AValuesAssembler<I>,
    device: &<I as burn::tensor::backend::BackendTypes>::Device,
    n_vec: &[f32],
    qsp_vec: &[f32],
    psp_vec: &[f32],
    qt_vec: &[f32],
    qpt_vec: &[f32],
    length_vec: &[f32],
    slope_vec: &[f32],
    x_storage_vec: &[f32],
    require_grad_parent: Option<Parent>,
    // Per-reach stage-roughness exponent. `Some` routes through the six-parent
    // `TimestepGammaOp`, which is the only way gamma receives a gradient.
    gamma_vec: Option<&[f32]>,
) -> (Tensor<AB, 1>, GradTensors) {
    // Construct each parent tensor; mark `.require_grad()` on the one we want gradient for.
    let mk = |data: &[f32], req: bool| -> Tensor<AB, 1> {
        let t: Tensor<AB, 1> = Tensor::from_floats(data, device);
        if req { t.require_grad() } else { t }
    };
    let n_t = mk(n_vec, matches!(require_grad_parent, Some(Parent::N)));
    let qsp_t = mk(qsp_vec, matches!(require_grad_parent, Some(Parent::QSpatial)));
    let psp_t = mk(psp_vec, matches!(require_grad_parent, Some(Parent::PSpatial)));
    let qt_t = mk(qt_vec, matches!(require_grad_parent, Some(Parent::QT)));
    let qpt_t = mk(qpt_vec, matches!(require_grad_parent, Some(Parent::QPrimeT)));
    let length_t = mk(length_vec, false);
    let slope_t = mk(slope_vec, false);
    let xst_t = mk(x_storage_vec, false);
    let gamma_t =
        gamma_vec.map(|g| mk(g, matches!(require_grad_parent, Some(Parent::Gamma))));

    let q_next = timestep_forward::<I>(
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
        false,
        gamma_t.clone(),
    );

    (
        q_next,
        GradTensors {
            gamma: gamma_t,
            n: n_t,
            qsp: qsp_t,
            psp: psp_t,
            qt: qt_t,
            qpt: qpt_t,
        },
    )
}

struct GradTensors {
    n: Tensor<AB, 1>,
    qsp: Tensor<AB, 1>,
    psp: Tensor<AB, 1>,
    qt: Tensor<AB, 1>,
    qpt: Tensor<AB, 1>,
    gamma: Option<Tensor<AB, 1>>,
}

fn compute_analytical_grad(parent: Parent) -> Vec<f32> {
    compute_analytical_grad_gamma(parent, 0.0)
}

fn compute_analytical_grad_gamma(parent: Parent, gamma: f32) -> Vec<f32> {
    compute_analytical_grad_inner(parent, gamma, false)
}

/// `learned` routes gamma through the six-parent op as a per-reach tensor
/// instead of the global config scalar.
fn compute_analytical_grad_inner(parent: Parent, gamma: f32, learned: bool) -> Vec<f32> {
    let cfg = if learned { mock_cfg() } else { mock_cfg_gamma(gamma) };
    let learned_gamma: Option<Vec<f32>> = learned.then(|| vec![gamma; N]);
    let adj = linear_chain_sparse();
    let device = <I as burn::tensor::backend::BackendTypes>::Device::default();
    let pattern = Arc::new(CsrPattern::from_sparse(&adj));
    let assembler = AValuesAssembler::<I>::new(&pattern, &device);

    let (n_vec, qsp_vec, psp_vec, qt_vec, qpt_vec) = default_inputs();
    let length_vec = adj.length_m.clone();
    let slope_vec = adj.slope.clone();
    let x_storage_vec = vec![0.3f32; N];

    let (q_next, parents) = run_forward_loss(
        &cfg,
        &pattern,
        &assembler,
        &device,
        &n_vec,
        &qsp_vec,
        &psp_vec,
        &qt_vec,
        &qpt_vec,
        &length_vec,
        &slope_vec,
        &x_storage_vec,
        Some(parent),
        learned_gamma.as_deref(),
    );

    let loss = q_next.sum();
    let grads = loss.backward();

    let g = match parent {
        Parent::N => parents.n.grad(&grads).expect("grad on n"),
        Parent::QSpatial => parents.qsp.grad(&grads).expect("grad on q_spatial"),
        Parent::PSpatial => parents.psp.grad(&grads).expect("grad on p_spatial"),
        Parent::QT => parents.qt.grad(&grads).expect("grad on q_t"),
        Parent::QPrimeT => parents.qpt.grad(&grads).expect("grad on q_prime_t"),
        Parent::Gamma => parents
            .gamma
            .as_ref()
            .expect("gamma tensor was not built; pass learned = true")
            .grad(&grads)
            .expect("grad on gamma"),
    };
    g.into_data().to_vec::<f32>().unwrap()
}

fn compute_fd_grad(parent: Parent) -> Vec<f32> {
    compute_fd_grad_gamma(parent, 0.0)
}

fn compute_fd_grad_gamma(parent: Parent, gamma: f32) -> Vec<f32> {
    compute_fd_grad_inner(parent, gamma, false)
}

fn compute_fd_grad_inner(parent: Parent, gamma: f32, learned: bool) -> Vec<f32> {
    let cfg = if learned { mock_cfg() } else { mock_cfg_gamma(gamma) };
    let learned_gamma: Option<Vec<f32>> = learned.then(|| vec![gamma; N]);
    let adj = linear_chain_sparse();
    let device = <I as burn::tensor::backend::BackendTypes>::Device::default();
    let pattern = Arc::new(CsrPattern::from_sparse(&adj));
    let assembler = AValuesAssembler::<I>::new(&pattern, &device);

    let (n_vec, qsp_vec, psp_vec, qt_vec, qpt_vec) = default_inputs();
    let length_vec = adj.length_m.clone();
    let slope_vec = adj.slope.clone();
    let x_storage_vec = vec![0.3f32; N];

    let eval_loss = |n: &[f32],
                     qsp: &[f32],
                     psp: &[f32],
                     qt: &[f32],
                     qpt: &[f32],
                     gm: Option<&[f32]>|
     -> f32 {
        let (q_next, _) = run_forward_loss(
            &cfg,
            &pattern,
            &assembler,
            &device,
            n,
            qsp,
            psp,
            qt,
            qpt,
            &length_vec,
            &slope_vec,
            &x_storage_vec,
            None,
            gm,
        );
        let v: Vec<f32> = q_next.sum().into_data().to_vec::<f32>().unwrap();
        v[0]
    };

    let mut grad = vec![0.0f32; N];
    for i in 0..N {
        let mut plus_n = n_vec.clone();
        let mut plus_qsp = qsp_vec.clone();
        let mut plus_psp = psp_vec.clone();
        let mut plus_qt = qt_vec.clone();
        let mut plus_qpt = qpt_vec.clone();
        let mut minus_n = n_vec.clone();
        let mut minus_qsp = qsp_vec.clone();
        let mut minus_psp = psp_vec.clone();
        let mut minus_qt = qt_vec.clone();
        let mut minus_qpt = qpt_vec.clone();
        let gamma_vec = learned_gamma.clone().unwrap_or_default();
        let mut plus_gm = gamma_vec.clone();
        let mut minus_gm = gamma_vec.clone();

        let (plus, minus, base) = match parent {
            Parent::N => (&mut plus_n, &mut minus_n, &n_vec),
            Parent::QSpatial => (&mut plus_qsp, &mut minus_qsp, &qsp_vec),
            Parent::PSpatial => (&mut plus_psp, &mut minus_psp, &psp_vec),
            Parent::QT => (&mut plus_qt, &mut minus_qt, &qt_vec),
            Parent::QPrimeT => (&mut plus_qpt, &mut minus_qpt, &qpt_vec),
            Parent::Gamma => (&mut plus_gm, &mut minus_gm, &gamma_vec),
        };
        let eps = (EPS * base[i].abs()).max(EPS);
        plus[i] = base[i] + eps;
        minus[i] = base[i] - eps;

        // Only the gamma case perturbs the gamma slice; every other parent
        // passes the unperturbed one so gamma is held fixed.
        let (gm_plus, gm_minus): (Option<&[f32]>, Option<&[f32]>) = if learned_gamma.is_some() {
            (Some(&plus_gm), Some(&minus_gm))
        } else {
            (None, None)
        };
        let l_plus = eval_loss(&plus_n, &plus_qsp, &plus_psp, &plus_qt, &plus_qpt, gm_plus);
        let l_minus = eval_loss(&minus_n, &minus_qsp, &minus_psp, &minus_qt, &minus_qpt, gm_minus);
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
        let a = analytical[i];
        let f = fd[i];
        let abs_diff = (a - f).abs();
        let denom = a.abs().max(f.abs()).max(1e-12);
        let rel_diff = abs_diff / denom;
        worst_abs = worst_abs.max(abs_diff);
        worst_rel = worst_rel.max(rel_diff);
        println!("  [{i}] analytical={a:.6e}  fd={f:.6e}  abs={abs_diff:.3e}  rel={rel_diff:.3e}");
    }
    println!("  worst abs={worst_abs:.3e}  worst rel={worst_rel:.3e}");
    // Pass if either rel < REL_TOL OR abs < ABS_TOL (handle saturated/zero slots).
    let pass = analytical
        .iter()
        .zip(fd)
        .all(|(&a, &f)| {
            let abs_diff = (a - f).abs();
            let denom = a.abs().max(f.abs()).max(1e-12);
            let rel_diff = abs_diff / denom;
            rel_diff < REL_TOL || abs_diff < ABS_TOL
        });
    assert!(pass, "{name}: gradcheck failed (worst rel={worst_rel:.3e}, abs={worst_abs:.3e})");
}

#[test]
fn gradcheck_n() {
    let a = compute_analytical_grad(Parent::N);
    let fd = compute_fd_grad(Parent::N);
    compare_grads("n", &a, &fd);
}

#[test]
fn gradcheck_q_spatial() {
    let a = compute_analytical_grad(Parent::QSpatial);
    let fd = compute_fd_grad(Parent::QSpatial);
    compare_grads("q_spatial", &a, &fd);
}

#[test]
fn gradcheck_p_spatial() {
    let a = compute_analytical_grad(Parent::PSpatial);
    let fd = compute_fd_grad(Parent::PSpatial);
    compare_grads("p_spatial", &a, &fd);
}

#[test]
fn gradcheck_q_t() {
    let a = compute_analytical_grad(Parent::QT);
    let fd = compute_fd_grad(Parent::QT);
    compare_grads("q_t", &a, &fd);
}

#[test]
fn gradcheck_q_prime_t() {
    let a = compute_analytical_grad(Parent::QPrimeT);
    let fd = compute_fd_grad(Parent::QPrimeT);
    compare_grads("q_prime_t", &a, &fd);
}

// ===========================================================================
// Stage-dependent roughness: the same five gradients with `gamma > 0`.
//
// These are the gate for the hand-written backward terms added with
// `params.stage_roughness`. `tests/stage_roughness.rs` checks the FORWARD
// (exponents, celerity identity, bit-identity at gamma = 0) and says nothing
// about gradients; this says nothing about the forward. Both are needed.
// ===========================================================================

const GAMMA: f32 = 0.35;

fn gradcheck_with_gamma(name: &str, parent: Parent) {
    let analytical = compute_analytical_grad_gamma(parent, GAMMA);
    let fd = compute_fd_grad_gamma(parent, GAMMA);
    compare_grads(&format!("{name} (gamma={GAMMA})"), &analytical, &fd);
}

#[test]
fn gradcheck_n_stage_roughness() {
    gradcheck_with_gamma("n", Parent::N);
}

#[test]
fn gradcheck_q_spatial_stage_roughness() {
    gradcheck_with_gamma("q_spatial", Parent::QSpatial);
}

#[test]
fn gradcheck_p_spatial_stage_roughness() {
    gradcheck_with_gamma("p_spatial", Parent::PSpatial);
}

#[test]
fn gradcheck_q_t_stage_roughness() {
    gradcheck_with_gamma("q_t", Parent::QT);
}

#[test]
fn gradcheck_q_prime_t_stage_roughness() {
    gradcheck_with_gamma("q_prime_t", Parent::QPrimeT);
}

/// The gamma terms must actually change the gradient, or the tests above would
/// pass on a backward that silently ignored `gamma` entirely.
#[test]
fn gamma_changes_the_gradients_it_is_supposed_to() {
    let g0 = compute_analytical_grad_gamma(Parent::N, 0.0);
    let g1 = compute_analytical_grad_gamma(Parent::N, GAMMA);
    let rel: f32 = g0
        .iter()
        .zip(&g1)
        .map(|(a, b)| (a - b).abs() / a.abs().max(1e-12))
        .fold(0.0, f32::max);
    assert!(
        rel > 1e-3,
        "gamma={GAMMA} left dL/dn essentially unchanged (max rel {rel:.2e}); \
         the backward is probably ignoring it"
    );
}

// ===========================================================================
// A LEARNED per-reach gamma. This is the gate for `TimestepGammaOp`, the
// six-parent sibling that exists solely so gamma receives a gradient: a tensor
// that is not a parent is not differentiated, so without this the KAN could
// emit gamma and it would silently never move.
//
// ∂L/∂gamma accumulates from three places gamma enters the forward:
//   B5   the depth exponent 3/(5 + 3q + 3·gamma)
//   B15  the velocity's explicit (d/d_ref)^gamma
//   B17  the celerity's gamma·A/(T·d)
// ===========================================================================

const LEARNED_GAMMA: f32 = 0.3;

fn gradcheck_learned(name: &str, parent: Parent) {
    let analytical = compute_analytical_grad_inner(parent, LEARNED_GAMMA, true);
    let fd = compute_fd_grad_inner(parent, LEARNED_GAMMA, true);
    compare_grads(&format!("{name} (learned gamma={LEARNED_GAMMA})"), &analytical, &fd);
}

#[test]
fn gradcheck_gamma_learned() {
    gradcheck_learned("gamma", Parent::Gamma);
}

#[test]
fn gradcheck_n_with_learned_gamma() {
    gradcheck_learned("n", Parent::N);
}

#[test]
fn gradcheck_q_spatial_with_learned_gamma() {
    gradcheck_learned("q_spatial", Parent::QSpatial);
}

#[test]
fn gradcheck_p_spatial_with_learned_gamma() {
    gradcheck_learned("p_spatial", Parent::PSpatial);
}

#[test]
fn gradcheck_q_t_with_learned_gamma() {
    gradcheck_learned("q_t", Parent::QT);
}

#[test]
fn gradcheck_q_prime_t_with_learned_gamma() {
    gradcheck_learned("q_prime_t", Parent::QPrimeT);
}

/// The gradient into gamma must be non-trivial. A backward that returned zeros
/// would pass a finite-difference check against a forward that ignored gamma,
/// so assert the gradient actually has magnitude.
#[test]
fn gamma_gradient_is_not_zero() {
    let g = compute_analytical_grad_inner(Parent::Gamma, LEARNED_GAMMA, true);
    let worst = g.iter().fold(0.0f32, |a, b| a.max(b.abs()));
    assert!(
        worst > 1e-6,
        "dL/dgamma is everywhere ~zero (max |g| = {worst:.3e}); gamma would \
         never train even though it is registered as a parent"
    );
}

/// A learned gamma field equal to a constant must reproduce the global-scalar
/// path. Same physics, two different code paths and two different ops.
#[test]
fn learned_constant_gamma_matches_the_global_scalar() {
    let learned = compute_analytical_grad_inner(Parent::N, LEARNED_GAMMA, true);
    let scalar = compute_analytical_grad_inner(Parent::N, LEARNED_GAMMA, false);
    for (i, (a, b)) in learned.iter().zip(&scalar).enumerate() {
        let rel = (a - b).abs() / a.abs().max(1e-12);
        assert!(
            rel < 1e-4,
            "reach {i}: dL/dn differs between the learned-tensor and \
             global-scalar gamma paths ({a:.6e} vs {b:.6e}, rel {rel:.2e})"
        );
    }
}

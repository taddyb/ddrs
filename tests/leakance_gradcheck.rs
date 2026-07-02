//! Finite-difference autograd gradcheck for the fused `TimestepLeakanceOp`
//! (`Backward<I, 8>`). Mirrors `tests/sp8_gradcheck.rs` but exercises
//! `timestep_forward_leakance` end-to-end (the leakance op subtracts `zeta`
//! from `b_rhs`) and adds the three leakance parents to the swept set:
//! {n, q_spatial, p_spatial, x_storage, K_D, d_gw, leakance_factor}.
//!
//! Leakance params sit in the INTERIOR of their ranges (K_D=5e-7, d_gw=0.0,
//! leakance_factor=0.5) on a losing config (depth > d_gw) so no clamp
//! saturates. Tolerance matches sp8_gradcheck (≈5e-3 rel, 1e-4 abs for the
//! saturated/zero slots).
//!
//! NOTE: Task 8 (`route_timestep` dispatch) is not done yet, so this test calls
//! `timestep_forward_leakance` directly with `require_grad` params — this is the
//! entry point that exercises `TimestepLeakanceOp` end-to-end.

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

// Interior leakance values (losing config: depth > d_gw, factor > 0, K_D > 0).
const K_D: f32 = 5e-7;
const D_GW: f32 = 0.0;
const LEAK_FAC: f32 = 0.5;

// x_storage is NOT swept: it is a non-differentiated constant of the op (not a
// registered parent of `TimestepLeakanceOp`), so `.grad()` on it is None. The
// gradient-bearing parents are the 5 base parents + 3 leakance parents; this
// test sweeps all 8.
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
}

fn linear_chain_sparse() -> SparseAdjacency {
    let mut dense = vec![0.0_f32; N * N];
    for i in 0..N - 1 {
        dense[(i + 1) * N + i] = 1.0;
    }
    SparseAdjacency::from_dense(N, &dense, vec![1000.0; N], vec![0.001; N])
}

fn mock_cfg() -> Config {
    let mut cfg = Config::default();
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

/// Same well-conditioned base point as sp8_gradcheck. depth ≈ a few meters on
/// this chain, so depth > d_gw (=0.0): a losing stream, zeta > 0.
fn default_inputs() -> (Vec<f32>, Vec<f32>, Vec<f32>, Vec<f32>, Vec<f32>) {
    let n_vec = vec![0.035f32; N];
    let qsp_vec = vec![0.4f32; N];
    let psp_vec = vec![20.0f32; N];
    let qt_vec = vec![100.0f32, 120.0, 140.0, 160.0];
    let qpt_vec = vec![10.0f32, 12.0, 14.0, 16.0];
    (n_vec, qsp_vec, psp_vec, qt_vec, qpt_vec)
}

fn x_storage_vec() -> Vec<f32> {
    vec![0.3f32; N]
}
fn kd_vec() -> Vec<f32> {
    vec![K_D; N]
}
fn dgw_vec() -> Vec<f32> {
    vec![D_GW; N]
}
fn fac_vec() -> Vec<f32> {
    vec![LEAK_FAC; N]
}

struct GradTensors {
    n: Tensor<AB, 1>,
    qsp: Tensor<AB, 1>,
    psp: Tensor<AB, 1>,
    qt: Tensor<AB, 1>,
    qpt: Tensor<AB, 1>,
    kd: Tensor<AB, 1>,
    dgw: Tensor<AB, 1>,
    fac: Tensor<AB, 1>,
}

#[allow(clippy::too_many_arguments)]
fn run_forward(
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
    xst_vec: &[f32],
    kd: &[f32],
    dgw: &[f32],
    fac: &[f32],
    req: Option<Parent>,
) -> (Tensor<AB, 1>, GradTensors) {
    let mk = |data: &[f32], on: bool| -> Tensor<AB, 1> {
        let t: Tensor<AB, 1> = Tensor::from_floats(data, device);
        if on {
            t.require_grad()
        } else {
            t
        }
    };
    let n_t = mk(n_vec, matches!(req, Some(Parent::N)));
    let qsp_t = mk(qsp_vec, matches!(req, Some(Parent::QSpatial)));
    let psp_t = mk(psp_vec, matches!(req, Some(Parent::PSpatial)));
    let qt_t = mk(qt_vec, matches!(req, Some(Parent::QT)));
    let qpt_t = mk(qpt_vec, matches!(req, Some(Parent::QPrimeT)));
    let kd_t = mk(kd, matches!(req, Some(Parent::KD)));
    let dgw_t = mk(dgw, matches!(req, Some(Parent::DGW)));
    let fac_t = mk(fac, matches!(req, Some(Parent::LeakFactor)));
    let xst_t = mk(xst_vec, false);
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
        xst_t.clone(),
        kd_t.clone(),
        dgw_t.clone(),
        fac_t.clone(),
        None,
    );

    (
        q_next,
        GradTensors {
            n: n_t,
            qsp: qsp_t,
            psp: psp_t,
            qt: qt_t,
            qpt: qpt_t,
            kd: kd_t,
            dgw: dgw_t,
            fac: fac_t,
        },
    )
}

fn compute_analytical_grad(parent: Parent) -> Vec<f32> {
    let cfg = mock_cfg();
    let adj = linear_chain_sparse();
    let device = <I as burn::tensor::backend::BackendTypes>::Device::default();
    let pattern = Arc::new(CsrPattern::from_sparse(&adj));
    let assembler = AValuesAssembler::<I>::new(&pattern, &device);

    let (n_vec, qsp_vec, psp_vec, qt_vec, qpt_vec) = default_inputs();
    let length_vec = adj.length_m.clone();
    let slope_vec = adj.slope.clone();

    let (q_next, parents) = run_forward(
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
        &x_storage_vec(),
        &kd_vec(),
        &dgw_vec(),
        &fac_vec(),
        Some(parent),
    );

    let loss = q_next.sum();
    let grads = loss.backward();

    let g = match parent {
        Parent::N => parents.n.grad(&grads).expect("grad on n"),
        Parent::QSpatial => parents.qsp.grad(&grads).expect("grad on q_spatial"),
        Parent::PSpatial => parents.psp.grad(&grads).expect("grad on p_spatial"),
        Parent::QT => parents.qt.grad(&grads).expect("grad on q_t"),
        Parent::QPrimeT => parents.qpt.grad(&grads).expect("grad on q_prime_t"),
        Parent::KD => parents.kd.grad(&grads).expect("grad on K_D"),
        Parent::DGW => parents.dgw.grad(&grads).expect("grad on d_gw"),
        Parent::LeakFactor => parents.fac.grad(&grads).expect("grad on leakance_factor"),
    };
    g.into_data().to_vec::<f32>().unwrap()
}

fn compute_fd_grad(parent: Parent) -> Vec<f32> {
    let cfg = mock_cfg();
    let adj = linear_chain_sparse();
    let device = <I as burn::tensor::backend::BackendTypes>::Device::default();
    let pattern = Arc::new(CsrPattern::from_sparse(&adj));
    let assembler = AValuesAssembler::<I>::new(&pattern, &device);

    let (n_vec, qsp_vec, psp_vec, qt_vec, qpt_vec) = default_inputs();
    let length_vec = adj.length_m.clone();
    let slope_vec = adj.slope.clone();

    let xst_const = x_storage_vec();

    #[allow(clippy::too_many_arguments)]
    let eval_loss = |n: &[f32],
                     qsp: &[f32],
                     psp: &[f32],
                     qt: &[f32],
                     qpt: &[f32],
                     kd: &[f32],
                     dgw: &[f32],
                     fac: &[f32]|
     -> f32 {
        let (q_next, _) = run_forward(
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
            &xst_const,
            kd,
            dgw,
            fac,
            None,
        );
        let v: Vec<f32> = q_next.sum().into_data().to_vec::<f32>().unwrap();
        v[0]
    };

    let base_kd = kd_vec();
    let base_dgw = dgw_vec();
    let base_fac = fac_vec();

    let mut grad = vec![0.0f32; N];
    for i in 0..N {
        let mut p_n = n_vec.clone();
        let mut p_qsp = qsp_vec.clone();
        let mut p_psp = psp_vec.clone();
        let mut p_qt = qt_vec.clone();
        let mut p_qpt = qpt_vec.clone();
        let mut p_kd = base_kd.clone();
        let mut p_dgw = base_dgw.clone();
        let mut p_fac = base_fac.clone();
        let mut m_n = n_vec.clone();
        let mut m_qsp = qsp_vec.clone();
        let mut m_psp = psp_vec.clone();
        let mut m_qt = qt_vec.clone();
        let mut m_qpt = qpt_vec.clone();
        let mut m_kd = base_kd.clone();
        let mut m_dgw = base_dgw.clone();
        let mut m_fac = base_fac.clone();

        let (plus, minus, base): (&mut Vec<f32>, &mut Vec<f32>, &Vec<f32>) = match parent {
            Parent::N => (&mut p_n, &mut m_n, &n_vec),
            Parent::QSpatial => (&mut p_qsp, &mut m_qsp, &qsp_vec),
            Parent::PSpatial => (&mut p_psp, &mut m_psp, &psp_vec),
            Parent::QT => (&mut p_qt, &mut m_qt, &qt_vec),
            Parent::QPrimeT => (&mut p_qpt, &mut m_qpt, &qpt_vec),
            Parent::KD => (&mut p_kd, &mut m_kd, &base_kd),
            Parent::DGW => (&mut p_dgw, &mut m_dgw, &base_dgw),
            Parent::LeakFactor => (&mut p_fac, &mut m_fac, &base_fac),
        };
        // Per-parent FD step. zeta is EXACTLY LINEAR in K_D, d_gw, and
        // leakance_factor, so central differences have zero truncation error
        // for them regardless of step size. A tiny step (e.g. 1e-3·K_D ≈ 5e-10)
        // makes Δloss vanish into the f32 round-off floor of `q_next.sum()`
        // (O(500)), so we deliberately use a LARGE step for these three to lift
        // the signal above round-off — exactness in K_D/d_gw/factor means this
        // does not introduce bias. The geometry params keep sp8's scheme.
        let eps = match parent {
            Parent::KD => 4e-7,         // base 5e-7 → sweeps 1e-7..9e-7 (>0)
            Parent::DGW => 1.5,         // base 0.0  → sweeps ±1.5 (in [-2,2])
            Parent::LeakFactor => 0.4,  // base 0.5  → sweeps 0.1..0.9 (in [0,1])
            _ => (EPS * base[i].abs()).max(EPS),
        };
        plus[i] = base[i] + eps;
        minus[i] = base[i] - eps;

        let l_plus = eval_loss(&p_n, &p_qsp, &p_psp, &p_qt, &p_qpt, &p_kd, &p_dgw, &p_fac);
        let l_minus = eval_loss(&m_n, &m_qsp, &m_psp, &m_qt, &m_qpt, &m_kd, &m_dgw, &m_fac);
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
    let pass = analytical.iter().zip(fd).all(|(&a, &f)| {
        let abs_diff = (a - f).abs();
        let denom = a.abs().max(f.abs()).max(1e-12);
        let rel_diff = abs_diff / denom;
        rel_diff < REL_TOL || abs_diff < ABS_TOL
    });
    assert!(
        pass,
        "{name}: gradcheck failed (worst rel={worst_rel:.3e}, abs={worst_abs:.3e})"
    );
}

fn run(name: &str, parent: Parent) {
    let a = compute_analytical_grad(parent);
    let fd = compute_fd_grad(parent);
    compare_grads(name, &a, &fd);
}

#[test]
fn gradcheck_n() {
    run("n", Parent::N);
}

#[test]
fn gradcheck_q_spatial() {
    run("q_spatial", Parent::QSpatial);
}

#[test]
fn gradcheck_p_spatial() {
    run("p_spatial", Parent::PSpatial);
}

#[test]
fn gradcheck_q_t() {
    run("q_t", Parent::QT);
}

#[test]
fn gradcheck_q_prime_t() {
    run("q_prime_t", Parent::QPrimeT);
}

#[test]
fn gradcheck_k_d() {
    run("K_D", Parent::KD);
}

#[test]
fn gradcheck_d_gw() {
    run("d_gw", Parent::DGW);
}

#[test]
fn gradcheck_leakance_factor() {
    run("leakance_factor", Parent::LeakFactor);
}

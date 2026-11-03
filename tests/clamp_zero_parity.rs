//! Zero-pattern parity: the hand-written backward of the fused MC timestep
//! against plain Burn autodiff of the same timestep
//! (`src/routing/reference_step.rs::reference_timestep`).
//!
//! Fixture: a 6-reach chain with per-reach inputs chosen so that the clamp
//! sites are active on some reach. The loss is `sum(q_next * [1..6])`. Every
//! parent is tracked on both sides.
//!
//! Checks per case and parent:
//! 1. Every gradient entry is finite.
//! 2. The set of entries equal to 0.0 is the same for both gradients.
//! 3. The nonzero entries agree at rel < 1e-3 or abs < 1e-6.
//!
//! The test collects every mismatch over all cases and asserts at the end, so
//! one failing case does not hide the others.

use std::sync::Arc;

use burn::backend::{Autodiff, NdArray};
use burn::tensor::Tensor;

use ddrs::config::{Config, GeometryForm, StageRoughnessSection};
use ddrs::routing::clamp_ledger::{ClampLedger, ClampSite};
use ddrs::routing::mmc_op::timestep_forward_diag;
use ddrs::routing::reference_step::reference_timestep;
use ddrs::sparse::{AValuesAssembler, CsrPattern, SparseAdjacency};

type I = NdArray<f32>;
type AB = Autodiff<I>;

const N: usize = 6;
const WEIGHTS: [f32; N] = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
const REL_TOL: f32 = 1e-3;
const ABS_TOL: f32 = 1e-6;

#[derive(Clone)]
struct Z6 {
    n: [f32; N],
    qsp: [f32; N],
    psp: [f32; N],
    qt: [f32; N],
    qpt: [f32; N],
    length: [f32; N],
    slope: [f32; N],
}

/// Reach 0: DischargeFloor and DepthFloor (`Q_t` tiny, `q'` zero, no upstream).
/// Reach 1: SideSlopeLow. Reach 2: BottomWidthFloor. Reach 3: VelocityHigh.
/// Reach 4: XBandLow (short reach). Reach 5: DepthFloor alone (`q'` keeps the
/// solve above the discharge floor).
fn z6() -> Z6 {
    let mut z = Z6 {
        n: [0.035; N],
        qsp: [0.4; N],
        psp: [20.0; N],
        qt: [100.0, 120.0, 140.0, 160.0, 180.0, 200.0],
        qpt: [10.0, 12.0, 14.0, 16.0, 18.0, 20.0],
        length: [10_000.0; N],
        slope: [1e-3; N],
    };
    z.qt[0] = 1e-5;
    z.qpt[0] = 0.0;
    z.qsp[1] = 0.084;
    z.qt[1] = 1000.0;
    z.qsp[2] = 0.999;
    z.n[3] = 0.01;
    z.slope[3] = 0.5;
    z.length[4] = 50.0;
    z.qt[5] = 1e-5;
    z
}

fn cfg(form: GeometryForm, enforce_positivity: bool) -> Config {
    cfg_sr(form, enforce_positivity, false)
}

/// `stage_roughness`: add a stage-roughness block with `d_ref` 2.0 (not 1), so
/// the `d_ref^gamma` factor on S2 runs. The learned gamma overrides the scalar
/// gamma elsewhere.
fn cfg_sr(form: GeometryForm, enforce_positivity: bool, stage_roughness: bool) -> Config {
    let mut cfg = Config::default();
    cfg.params.ddr_match = false;
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
    cfg.params.geometry_form = form;
    cfg.params.enforce_positivity = enforce_positivity;
    if stage_roughness {
        cfg.params.stage_roughness =
            Some(StageRoughnessSection { gamma: 0.2, d_ref: 2.0 });
    }
    cfg
}

fn chain(z: &Z6) -> SparseAdjacency {
    let mut dense = vec![0.0_f32; N * N];
    for i in 0..N - 1 {
        dense[(i + 1) * N + i] = 1.0;
    }
    SparseAdjacency::from_dense(N, &dense, z.length.to_vec(), z.slope.to_vec())
}

fn device() -> <I as burn::tensor::backend::BackendTypes>::Device {
    Default::default()
}

struct Grads {
    /// Parent order: n, q_spatial, p_spatial, q_t, q_prime_t, then gamma if learned.
    parents: Vec<Vec<f32>>,
    ledger: Option<ClampLedger>,
}

const PARENT_NAMES: [&str; 6] = ["n", "q_spatial", "p_spatial", "q_t", "q_prime_t", "gamma"];

/// Gradients of the weighted loss. `fused` picks the fused op (with a ledger)
/// or the autodiff reference.
fn grads(cfg: &Config, z: &Z6, gamma: Option<f32>, fused: bool) -> Grads {
    let adj = chain(z);
    let dev = device();
    let pattern = Arc::new(CsrPattern::from_sparse(&adj));
    let assembler = AValuesAssembler::<I>::new(&pattern, &dev);
    let tracked = |v: &[f32]| -> Tensor<AB, 1> { Tensor::<AB, 1>::from_floats(v, &dev).require_grad() };
    let constant = |v: &[f32]| -> Tensor<AB, 1> { Tensor::<AB, 1>::from_floats(v, &dev) };
    let n = tracked(&z.n);
    let qsp = tracked(&z.qsp);
    let psp = tracked(&z.psp);
    let qt = tracked(&z.qt);
    let qpt = tracked(&z.qpt);
    let gam = gamma.map(|g| tracked(&[g; N]));
    let length = constant(&z.length);
    let slope = constant(&z.slope);
    let xs = constant(&[0.3; N]);
    let mut ledger = ClampLedger::default();
    let q_next = if fused {
        timestep_forward_diag::<I>(
            cfg,
            &pattern,
            &assembler,
            n.clone(),
            qsp.clone(),
            psp.clone(),
            qt.clone(),
            qpt.clone(),
            length,
            slope,
            xs,
            gam.clone(),
            Some(&mut ledger),
            None,
        )
    } else {
        reference_timestep::<I>(
            cfg,
            &pattern,
            &assembler,
            n.clone(),
            qsp.clone(),
            psp.clone(),
            qt.clone(),
            qpt.clone(),
            length,
            slope,
            xs,
            gam.clone(),
        )
    };
    let loss = (q_next * constant(&WEIGHTS)).sum();
    let g = loss.backward();
    let mut parents = Vec::new();
    for t in [Some(&n), Some(&qsp), Some(&psp), Some(&qt), Some(&qpt), gam.as_ref()].into_iter().flatten() {
        parents.push(t.grad(&g).expect("tracked parent has a gradient").into_data().to_vec::<f32>().unwrap());
    }
    Grads { parents, ledger: fused.then_some(ledger) }
}

/// One line per site: the ledger state of reach `i`.
fn ledger_row(l: &ClampLedger, i: usize) -> String {
    let st = &l.steps[0];
    ClampSite::ALL
        .iter()
        .filter_map(|s| st.active.get(s).map(|v| format!("{}={}", s.name(), v[i])))
        .collect::<Vec<_>>()
        .join(" ")
}

fn zeros_of(v: &[f32]) -> Vec<usize> {
    v.iter().enumerate().filter(|(_, x)| **x == 0.0).map(|(i, _)| i).collect()
}

/// Compare the fused and reference gradients. Returns the mismatches.
fn compare(case: &str, fused: &Grads, reference: &Grads) -> Vec<String> {
    let mut bad = Vec::new();
    let led = fused.ledger.as_ref().expect("fused run has a ledger");
    assert_eq!(fused.parents.len(), reference.parents.len());
    for (p, (gf, gr)) in fused.parents.iter().zip(&reference.parents).enumerate() {
        let name = PARENT_NAMES[p];
        let finite_f = gf.iter().all(|x| x.is_finite());
        let finite_r = gr.iter().all(|x| x.is_finite());
        if !(finite_f && finite_r) {
            bad.push(format!("{case} {name}: non-finite gradient. fused {gf:?} reference {gr:?}"));
            continue;
        }
        let (zf, zr) = (zeros_of(gf), zeros_of(gr));
        println!("  {name}: zero entries fused {zf:?} reference {zr:?}");
        if zf != zr {
            println!("  {case} {name}: zero set fused {zf:?} reference {zr:?}");
            for i in 0..N {
                if (gf[i] == 0.0) != (gr[i] == 0.0) {
                    bad.push(format!(
                        "{case} {name}[{i}]: zero-set mismatch, fused {:e}, reference {:e}; ledger: {}",
                        gf[i],
                        gr[i],
                        ledger_row(led, i)
                    ));
                }
            }
        }
        for i in 0..N {
            let (a, b) = (gf[i], gr[i]);
            if a == 0.0 || b == 0.0 {
                continue;
            }
            let abs = (a - b).abs();
            let rel = abs / a.abs().max(b.abs());
            if !(rel < REL_TOL || abs < ABS_TOL) {
                bad.push(format!(
                    "{case} {name}[{i}]: value mismatch, fused {a:e}, reference {b:e}, rel {rel:e}; ledger: {}",
                    ledger_row(led, i)
                ));
            }
        }
    }
    bad
}

const TRAP_SITES: [ClampSite; 6] = [
    ClampSite::DepthFloor,
    ClampSite::SideSlopeLow,
    ClampSite::BottomWidthFloor,
    ClampSite::VelocityHigh,
    ClampSite::XBandLow,
    ClampSite::DischargeFloor,
];
const POWER_SITES: [ClampSite; 4] = [
    ClampSite::DepthFloor,
    ClampSite::VelocityHigh,
    ClampSite::XBandLow,
    ClampSite::DischargeFloor,
];

fn run_case(
    case: &str,
    form: GeometryForm,
    pos: bool,
    sr: bool,
    gamma: Option<f32>,
    targets: &[ClampSite],
    bad: &mut Vec<String>,
) {
    let c = cfg_sr(form, pos, sr);
    let z = z6();
    let fused = grads(&c, &z, gamma, true);
    let led = fused.ledger.as_ref().unwrap();
    println!("== {case} ==");
    for s in ClampSite::ALL {
        let per_reach = led.steps[0]
            .active
            .get(&s)
            .map(|v| format!("{} (reaches {:?})", v.iter().filter(|a| **a).count(), zeros_of(&v.iter().map(|a| if *a { 0.0 } else { 1.0 }).collect::<Vec<f32>>())))
            .unwrap_or_else(|| "n/a".to_string());
        println!("  {:<20} active: {per_reach}", s.name());
    }
    for s in targets {
        if led.count(*s) == 0 {
            bad.push(format!("{case}: target site {} is not active on any reach", s.name()));
        }
    }
    let reference = grads(&c, &z, gamma, false);
    bad.extend(compare(case, &fused, &reference));
}

#[test]
fn fused_gradient_matches_autodiff_zero_pattern() {
    let mut bad = Vec::new();
    run_case("trapezoid", GeometryForm::Trapezoid, false, false, None, &TRAP_SITES, &mut bad);
    run_case("trapezoid+positivity", GeometryForm::Trapezoid, true, false, None, &TRAP_SITES, &mut bad);
    run_case("power_law", GeometryForm::PowerLaw, false, false, None, &POWER_SITES, &mut bad);
    run_case("power_law+positivity", GeometryForm::PowerLaw, true, false, None, &POWER_SITES, &mut bad);
    run_case("power_law+gamma0.2", GeometryForm::PowerLaw, false, false, Some(0.2), &POWER_SITES, &mut bad);
    run_case("trapezoid+gamma0.2+d_ref2", GeometryForm::Trapezoid, false, true, Some(0.2), &TRAP_SITES, &mut bad);
    for b in &bad {
        println!("MISMATCH: {b}");
    }
    assert!(bad.is_empty(), "{} mismatches:\n{}", bad.len(), bad.join("\n"));
}

/// Pins Burn's `min_pair` tie: at `a == b` the gradient goes to the first operand.
#[test]
fn burn_min_pair_tie_goes_to_first_operand() {
    let dev = device();
    let a = Tensor::<AB, 1>::from_floats([0.5f32, 1.0, 2.0], &dev).require_grad();
    let b = Tensor::<AB, 1>::from_floats([0.5f32, 2.0, 1.0], &dev).require_grad();
    let g = a.clone().min_pair(b.clone()).sum().backward();
    let ga = a.grad(&g).unwrap().into_data().to_vec::<f32>().unwrap();
    let gb = b.grad(&g).unwrap().into_data().to_vec::<f32>().unwrap();
    assert_eq!(ga, vec![1.0, 1.0, 0.0], "gradient of the first operand");
    assert_eq!(gb, vec![0.0, 0.0, 1.0], "gradient of the second operand");
}

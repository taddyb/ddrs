//! The per-dam flood pool (law FA of `experiments/reservoir/laws_v6`;
//! `release_head.flood_pool` / `params.reservoir_flood_pool`,
//! `ddrs::routing::mmc::FloodPool`). Before each solve, from the dam's inflow
//! at the step start `I = (N·Q_t)_d + q'_d` and its pool `F`:
//! `Vc = min(phi·max(I − Qc, 0)·dt, Fmax − F)`, `p = I − Vc/dt`,
//! `Ve = min(F, max(Qc − p, 0)·dt)`, the dam row's `q'` gains `(Ve − Vc)/dt`,
//! and `F ← F + Vc − Ve` after the solve (`Qc = kc·Ibar`, `Fmax = z·Ibar`).
//!
//! 1. `z = 0` routes bitwise like no pool (option C additive rows, and the
//!    release path with a rule curve); `z > 0` changes the routing.
//! 2. One year of forced floods on an interior additive dam row: the dam
//!    row's volume balance closes with the pool, inflow = outflow + ΔS +
//!    F_end to f32 tolerance, the discharge floor never binds, and the pool
//!    account closes (`captured − evacuated = F_end`).
//! 3. Finite-difference gradcheck of `kc`, `phi`, `z`, `T0` and an upstream
//!    reach's Manning `n` (the inflow `I` is not detached) in three regimes,
//!    each verified to occur: the capture binds (the pool never fills), the
//!    pool's cap binds, and the evacuation at the release target binds.
//! 4. A 15-day-chunked test phase with the pool carried
//!    (`training::forward::carry_dam_state`, `DamClampSums::merge_pool`)
//!    routes like one engine over the whole period; dropping the pool at each
//!    boundary does not.
//! 5. `set_flood_pool` / `set_flood_pool_state` refuse bad input.
//! 6. The training path (per-dam raw parameters through the transform) and
//!    the resolved test-phase table route bitwise identically, for
//!    `flood_pool: all` and `flood_control` (which pools only the
//!    `purpose_flood` dams); `release_params.csv` carries `kc, phi, z`.
//! 7. Dataset open refuses a pool without `inflow_mean_m3s`, `flood_control`
//!    without `purpose_flood`, and `params.reservoir_flood_pool: true` on a
//!    fixed table without the `kc, phi, z` columns.
//! 8. Frozen-routing training on the Juniata bundle (real driver) moves
//!    Raystown's `kc`, `phi`, `z` while the routing head stays bitwise at its
//!    checkpoint, and a resume restores them bitwise.

use std::path::{Path, PathBuf};

use burn::backend::{Autodiff, NdArray};
use burn::module::{Module, ModuleVisitor, Param};
use burn::tensor::{backend::Backend, Tensor};
use chrono::{Duration, NaiveDate};

use ddrs::config::{kan_config, Config, DamRow, FloodPoolMode, ReleaseHeadSection, ReservoirRelease};
use ddrs::data::ids::Comid;
use ddrs::data::store::{map_reservoir_rows, DamFeatures, ReservoirTable};
use ddrs::nn::dam_params::{flood_pool_transform, DamParams};
use ddrs::nn::release_head::init_release_head;
use ddrs::nn::KanHead;
use ddrs::routing::mmc::{DamRelease, FloodPool, RuleCurve, DT_SECONDS};
use ddrs::routing::release::{rule_curve_phase_start, seasonal_phase};
use ddrs::routing::{MuskingumCunge, RoutingInputs, SpatialParameters};
use ddrs::sparse::SparseAdjacency;
use ddrs::training::forward::{apply_reservoir_rows_with, carry_dam_state};
use ddrs::training::release_eval::{resolve_release_table_with, write_release_params_csv, DamClampSums};
use ddrs::training::{bootstrap_head_and_state, head_base, load_kan_head, save_kan_head, train};

type I = NdArray<f32>;
type AB = Autodiff<I>;
type Device = <I as burn::tensor::backend::BackendTypes>::Device;

const REL_TOL: f64 = 5e-3;
const ABS_TOL: f64 = 1e-4;
const DT: f64 = DT_SECONDS as f64;

fn network(n: usize, edges: &[(usize, usize)], length: f32) -> SparseAdjacency {
    let mut dense = vec![0.0_f32; n * n];
    for &(up, down) in edges {
        dense[down * n + up] = 1.0;
    }
    SparseAdjacency::from_dense(n, &dense, vec![length; n], vec![0.001; n])
}

/// `1 → 3`, `2 → 3`, `0 → 4`, `3 → 4`.
fn sandbox5() -> SparseAdjacency {
    network(5, &[(1, 3), (2, 3), (0, 4), (3, 4)], 5000.0)
}

/// `0 → 1 → 2`: the dam is reach 1.
fn chain3(length: f32) -> SparseAdjacency {
    network(3, &[(0, 1), (1, 2)], length)
}

/// An engine on `adjacency` with `q_prime` (`[rows, n]` row-major), starting
/// from `initial` or the cold start; Manning's `n` (normalised) from `n_norm`.
fn engine_with(
    cfg: &Config,
    adjacency: SparseAdjacency,
    q_prime: &[f32],
    initial: Option<&[f32]>,
    n_norm: Tensor<AB, 1>,
) -> MuskingumCunge<I> {
    let device = Device::default();
    let n = adjacency.n;
    let rows = q_prime.len() / n;
    let leaf = |v: f32| Tensor::<AB, 1>::from_floats(vec![v; n].as_slice(), &device);
    let q = Tensor::<AB, 1>::from_floats(q_prime, &device).reshape([rows, n]);
    let mut mc = MuskingumCunge::<I>::new(cfg.clone(), device);
    mc.setup_inputs(
        RoutingInputs { adjacency, x_storage: Tensor::ones([n], &device) * 0.3 },
        q,
        SpatialParameters {
            n: n_norm,
            q_spatial: leaf(0.5),
            p_spatial: Some(leaf(0.575)),
            k_d: None,
            d_gw: None,
            leakance_factor: None,
            impervious_mask: None,
            gamma: None,
        },
        false,
        initial.map(|v| Tensor::<AB, 1>::from_floats(v, &device)),
    );
    mc
}

fn engine(cfg: &Config, adjacency: SparseAdjacency, q_prime: &[f32], initial: Option<&[f32]>) -> MuskingumCunge<I> {
    let n = adjacency.n;
    let n_norm = Tensor::<AB, 1>::from_floats(vec![0.5_f32; n].as_slice(), &Device::default());
    engine_with(cfg, adjacency, q_prime, initial, n_norm)
}

fn t1(v: f32) -> Tensor<AB, 1> {
    Tensor::from_floats([v], &Device::default())
}

fn host2(q: Tensor<AB, 2>) -> Vec<f32> {
    q.into_data().to_vec().unwrap()
}

fn assert_bitwise(a: &[f32], b: &[f32], what: &str) {
    assert_eq!(a.len(), b.len(), "{what}: length");
    for (i, (x, y)) in a.iter().zip(b).enumerate() {
        assert_eq!(x.to_bits(), y.to_bits(), "{what}: idx {i}: {x} vs {y}");
    }
}

/// `row` of a routed output `[n, cols]`, f64.
fn series(out: &[f32], cols: usize, row: usize) -> Vec<f64> {
    out[row * cols..(row + 1) * cols].iter().map(|&v| v as f64).collect()
}

/// `Σ dt·(v_t + v_{t+1})/2`, m³.
fn trapezoid(v: &[f64]) -> f64 {
    v.windows(2).map(|w| 0.5 * (w[0] + w[1]) * DT).sum()
}

/// Flood pulses on a base flow: every `period` hours a `dur`-hour
/// `sin²` pulse from `base` to `peak`, the first starting at hour `first`;
/// rows `0..=steps`.
fn floods(steps: usize, base: f32, peak: f32, first: usize, period: usize, dur: usize) -> Vec<f32> {
    (0..=steps)
        .map(|s| {
            if s < first {
                return base;
            }
            let k = (s - first) % period;
            if k < dur {
                let x = (std::f32::consts::PI * k as f32 / dur as f32).sin();
                base + (peak - base) * x * x
            } else {
                base
            }
        })
        .collect()
}

/// Lateral inflow of the chain `[steps + 1, 3]`: `q_up` on reach 0, a
/// constant `q_dam` on the dam, nothing downstream.
fn chain_q(q_up: &[f32], q_dam: f32) -> Vec<f32> {
    q_up.iter().flat_map(|&v| [v, q_dam, 0.0]).collect()
}

fn start() -> NaiveDate {
    NaiveDate::from_ymd_opt(2001, 3, 1).unwrap()
}

/// The dam (reach 1 of the chain) as a constant-`T` release on `row`, plus a
/// pool `(kc, phi, z)` with `Ibar`.
fn arm_chain_dam(
    mc: &mut MuskingumCunge<I>,
    window_start: NaiveDate,
    rows: usize,
    row: DamRow,
    t0: Tensor<AB, 1>,
    pool: Option<(Tensor<AB, 1>, Tensor<AB, 1>, Tensor<AB, 1>, f32)>,
) {
    mc.set_dam_release(DamRelease {
        rows: vec![1],
        t0_days: t0,
        seasonal: None,
        phase: seasonal_phase(window_start, rows),
        dam_row: row,
        rule_curve: None,
    })
    .unwrap();
    if let Some((kc, phi, z, ibar)) = pool {
        mc.set_flood_pool(FloodPool { rows: vec![1], kc, phi, z_days: z, inflow_mean: vec![ibar] }).unwrap();
    }
}

// ---------------------------------------------------------------------------
// 1. z = 0 is no pool, bit for bit.
// ---------------------------------------------------------------------------

fn sandbox_q_prime(steps: usize) -> Vec<f32> {
    let base = [22.0_f32, 11.0, 11.0, 11.0, 22.0];
    (0..=steps)
        .flat_map(|t| {
            let s = (t as f32 / 24.0 * std::f32::consts::PI).sin();
            base.map(|b| b * (1.0 + 1.5 * s * s))
        })
        .collect()
}

#[test]
fn zero_pool_size_routes_bitwise_like_no_pool() {
    let device = Device::default();
    let steps = 120;
    let q = sandbox_q_prime(steps);
    let pools = |z: f32| FloodPool {
        rows: vec![3, 4],
        kc: Tensor::from_floats([1.2_f32, 0.8], &device),
        phi: Tensor::from_floats([0.7_f32, 0.9], &device),
        z_days: Tensor::from_floats([z, z], &device),
        inflow_mean: vec![30.0, 60.0],
    };

    // Option C rows on the additive row.
    let route_c = |z: Option<f32>| {
        let mut mc = engine(&Config::default(), sandbox5(), &q, None);
        mc.set_reservoir_rows_as(&[3, 4], &[0.8, 2.5], DamRow::Additive).unwrap();
        if let Some(z) = z {
            mc.set_flood_pool(pools(z)).unwrap();
        }
        let out = host2(mc.forward());
        (out, mc.dam_account().unwrap(), mc.flood_pool_account())
    };
    let (none, acc_none, _) = route_c(None);
    let (zero, acc_zero, pool_zero) = route_c(Some(0.0));
    assert_bitwise(&none, &zero, "option C additive: z = 0 vs no pool");
    assert_eq!(acc_none, acc_zero, "the clamp account is unchanged");
    let p = pool_zero.expect("a z = 0 pool is armed");
    assert!(p.captured_m3.iter().chain(&p.evacuated_m3).chain(&p.f_end_m3).all(|&v| v == 0.0));
    let (some, _, pool_some) = route_c(Some(1.5));
    assert!(none.iter().zip(&some).any(|(a, b)| a != b), "z > 0 must change the routing");
    assert!(pool_some.unwrap().captured_m3.iter().all(|&v| v > 0.0), "the case must capture");

    // The release path with a rule curve and a tracked T0: outputs bitwise.
    let route_r = |z: Option<f32>| {
        let mut mc = engine(&Config::default(), sandbox5(), &q, None);
        mc.set_dam_release(DamRelease {
            rows: vec![3, 4],
            t0_days: Tensor::<AB, 1>::from_floats([0.4_f32, 1.1], &device).require_grad(),
            seasonal: None,
            phase: seasonal_phase(start(), steps + 1),
            dam_row: DamRow::Additive,
            rule_curve: Some(RuleCurve {
                coeffs: Tensor::<AB, 1>::from_floats([0.05_f32, 0.03, -0.02, 0.01, 0.0, 0.1, 0.0, -0.1], &device)
                    .reshape([2, 4]),
                inflow_mean: Tensor::from_floats([30.0_f32, 60.0], &device),
                phase0: rule_curve_phase_start(start()),
            }),
        })
        .unwrap();
        if let Some(z) = z {
            mc.set_flood_pool(pools(z)).unwrap();
        }
        host2(mc.forward())
    };
    assert_bitwise(&route_r(None), &route_r(Some(0.0)), "release + rule curve: z = 0 vs no pool");
}

// ---------------------------------------------------------------------------
// 2. A year of floods: the dam row with its pool conserves mass.
// ---------------------------------------------------------------------------

/// Additive interior dam (reach 1 of `0 → 1 → 2`, 10 km reaches) with `K`, `X`
/// constant: `ddr_match` takes `X` from `x_storage` (0.3) and a 2 m/s
/// velocity floor, above every velocity the case reaches, fixes the celerity
/// at `2·5/3` m/s (the dam-floor tests' case).
fn constant_kx_cfg() -> (Config, f64, f64) {
    let mut cfg = Config::default();
    cfg.params.ddr_match = true;
    cfg.params.attribute_minimums.velocity = 2.0;
    let k = (10_000.0_f32 / (2.0_f32 * (5.0_f32 / 3.0_f32))) as f64;
    let x = 0.3_f32 as f64;
    (cfg, k * x, k * (1.0 - x))
}

const YEAR_STEPS: usize = 8766;

#[test]
fn a_year_of_floods_conserves_the_dam_rows_volume_with_the_pool() {
    let (cfg, kx, k1x) = constant_kx_cfg();
    let steps = YEAR_STEPS;
    // Base 4 m3/s upstream with a 3-day flood to 40 every 20 days; the dam's
    // own q' 1 m3/s. Ibar 5 -> Qc = 10 m3/s, Fmax = 2 d of Ibar = 864,000 m3,
    // less than a flood's captured excess, so the cap binds too.
    let q_up = floods(steps, 4.0, 40.0, 48, 480, 72);
    let q_dam = 1.0_f32;
    let q = chain_q(&q_up, q_dam);
    let t_days = 0.5_f32;
    let (kc, phi, z, ibar) = (2.0_f32, 0.8_f32, 2.0_f32, 5.0_f32);

    let route = |pool: bool| {
        let mut mc = engine(&cfg, chain3(10_000.0), &q, None);
        arm_chain_dam(
            &mut mc,
            start(),
            steps + 1,
            DamRow::Additive,
            t1(t_days),
            pool.then(|| (t1(kc), t1(phi), t1(z), ibar)),
        );
        let out = host2(mc.forward());
        (out, mc.dam_account().unwrap(), mc.flood_pool_account())
    };
    let (out, acc, pool) = route(true);
    let (plain, _, _) = route(false);
    let pool = pool.expect("pool armed");
    let cols = steps + 1;
    let up = series(&out, cols, 0);
    let dam = series(&out, cols, 1);
    let inflow = trapezoid(&up) + steps as f64 * q_dam as f64 * DT;
    let outflow = trapezoid(&dam);
    let t_s = t_days as f64 * 86_400.0;
    let storage = |i: usize| kx * up[i] + (k1x + t_s) * dam[i];
    let ds = storage(steps) - storage(0);
    let (captured, evacuated, f_end) = (pool.captured_m3[0], pool.evacuated_m3[0], pool.f_end_m3[0]);
    let resid = inflow - outflow - ds - f_end;
    println!(
        "inflow {inflow:.6e} m3, outflow {outflow:.6e}, dS {ds:.3e}, F_end {f_end:.4e}, residual {resid:.3e} \
         ({:.2e} of inflow); captured {captured:.4e}, evacuated {evacuated:.4e}, peak F {:.4e} of Fmax {:.4e}; \
         clamp steps {}, created {:.3e}",
        resid / inflow,
        pool.f_peak_m3[0],
        pool.fmax_m3[0],
        acc.clamp_steps[0],
        acc.created_m3[0]
    );
    // The pool never relies on the discharge floor.
    assert_eq!((acc.clamp_steps[0], acc.created_m3[0]), (0, 0.0), "the floor must never bind");
    // It works: captures a meaningful share of the floods, fills, evacuates.
    assert!(captured > 0.05 * inflow, "captured {captured:.3e} of inflow {inflow:.3e}");
    assert!((pool.f_peak_m3[0] - pool.fmax_m3[0]).abs() <= 1e-4 * pool.fmax_m3[0], "the cap binds");
    assert!(evacuated > 0.9 * captured);
    // The pool's account closes...
    assert!(
        (captured - evacuated - f_end).abs() <= 1e-4 * captured,
        "captured {captured:.6e} − evacuated {evacuated:.6e} ≠ F_end {f_end:.6e}"
    );
    // ...and so does the dam row's: inflow = outflow + ΔS + F_end.
    assert!(resid.abs() <= 1e-5 * inflow, "dam row balance residual {resid:.3e} of {inflow:.3e}");
    // The pool reshapes the flood hydrograph (lower peaks) but not the volume.
    let (peak_pool, peak_plain) =
        (dam.iter().cloned().fold(0.0, f64::max), series(&plain, cols, 1).iter().cloned().fold(0.0, f64::max));
    assert!(peak_pool < 0.95 * peak_plain, "the pool attenuates the flood peak ({peak_pool:.2} vs {peak_plain:.2})");
}

// ---------------------------------------------------------------------------
// 3. Gradcheck in three regimes.
// ---------------------------------------------------------------------------

const GC_STEPS: usize = 200;
const GC_IBAR: f32 = 5.0;
const GC_T0: f32 = 0.3;

/// Upstream flood 4 -> 40 m3/s over hours 24..96, the dam's own q' 1 m3/s.
fn gc_q() -> Vec<f32> {
    chain_q(&floods(GC_STEPS, 4.0, 40.0, 24, 1000, 72), 1.0)
}

/// The parameters a regime sweeps: `[kc, phi, z, T0, n_up]`.
type GcParams = [f32; 5];

fn gc_route(p: GcParams, leaves: Option<&[Tensor<AB, 1>; 5]>) -> Tensor<AB, 2> {
    let device = Device::default();
    let [kc, phi, z, t0, n_up] = match leaves {
        Some(l) => l.clone(),
        None => p.map(|v| Tensor::<AB, 1>::from_floats([v], &device)),
    };
    // Manning n of the upstream reach from the leaf; the others fixed.
    let n_norm = Tensor::cat(vec![n_up, Tensor::from_floats([0.5_f32, 0.5], &device)], 0);
    let mut mc = engine_with(&Config::default(), chain3(5000.0), &gc_q(), None, n_norm);
    arm_chain_dam(&mut mc, start(), GC_STEPS + 1, DamRow::Additive, t0, Some((kc, phi, z, GC_IBAR)));
    mc.forward()
}

/// Loss weights that vary strongly in time: a pool that returns its volume
/// within the window changes a near-uniform weighted sum (a volume) hardly at
/// all, and the finite differences then sink into f32 routing noise.
fn gc_weights() -> Vec<f32> {
    (0..3 * (GC_STEPS + 1))
        .map(|i| {
            let (r, t) = (i / (GC_STEPS + 1), i % (GC_STEPS + 1));
            let wave = (2.0 * std::f32::consts::PI * t as f32 / 37.0).sin();
            if t == 0 { 0.0 } else { 1.0 + 0.3 * r as f32 + 0.9 * wave }
        })
        .collect()
}

fn gc_loss(p: GcParams) -> f64 {
    host2(gc_route(p, None)).iter().zip(gc_weights()).map(|(&v, w)| v as f64 * w as f64).sum()
}

/// Steps in each regime of the law, replayed on the host (f64) from the
/// routed upstream series: `(capture binds, cap binds, evacuation at the
/// target binds, evacuation empties the pool)`.
fn regimes(p: GcParams) -> (usize, usize, usize, usize) {
    let [kc, phi, z, _, _] = p.map(|v| v as f64);
    let out = host2(gc_route(p, None));
    let up = series(&out, GC_STEPS + 1, 0);
    let q = gc_q();
    let ibar = GC_IBAR as f64;
    let (qc, fmax) = (kc * ibar, z * ibar * 86_400.0);
    let mut f = 0.0_f64;
    let mut n = (0, 0, 0, 0);
    for s in 0..GC_STEPS {
        let inflow = up[s] + q[s * 3 + 1] as f64;
        let want = phi * (inflow - qc).max(0.0) * DT;
        let room = (fmax - f).max(0.0);
        let vc = want.min(room);
        if want > 0.0 {
            if want < room { n.0 += 1 } else { n.1 += 1 }
        }
        let head = (qc - (inflow - vc / DT)).max(0.0) * DT;
        let ve = f.min(head);
        if f > 0.0 && head > 0.0 {
            if head < f { n.2 += 1 } else { n.3 += 1 }
        }
        f += vc - ve;
    }
    n
}

#[test]
fn flood_pool_gradcheck_in_each_regime() {
    let device = Device::default();
    let names = ["kc", "phi", "z", "T0", "n_up"];
    // (label, [kc, phi, z, T0, n_up], which regimes must occur)
    let cases: [(&str, GcParams); 3] = [
        // Pool of 30 d never fills: the capture binds; after the flood the
        // pool drains at the target, past the window's end.
        ("capture", [2.0, 0.6, 30.0, GC_T0, 0.5]),
        // Pool of 0.3 d fills within hours: the cap binds; it then drains at
        // the target and empties.
        ("cap", [2.0, 0.6, 0.3, GC_T0, 0.5]),
        // Pool of 3 d fills, then ~48 h of evacuation at the target.
        ("evacuation", [2.5, 0.9, 3.0, GC_T0, 0.5]),
    ];
    let mut failures = Vec::new();
    for (label, p) in cases {
        let (n_capture, n_cap, n_target, n_empty) = regimes(p);
        println!("{label}: steps capture-limited {n_capture}, cap {n_cap}, evacuation at target {n_target}, emptying {n_empty}");
        match label {
            "capture" => assert!(n_capture > 20 && n_cap == 0 && n_target > 20, "{label}: regimes"),
            "cap" => assert!(n_cap > 20 && n_target > 0 && n_empty > 0, "{label}: regimes"),
            _ => assert!(n_cap > 0 && n_target > 40, "{label}: regimes"),
        }
        let leaves: [Tensor<AB, 1>; 5] = p.map(|v| Tensor::<AB, 1>::from_floats([v], &device).require_grad());
        let w = Tensor::<AB, 1>::from_floats(gc_weights().as_slice(), &device).reshape([3, GC_STEPS + 1]);
        let grads = (gc_route(p, Some(&leaves)) * w).sum().backward();
        for (k, name) in names.iter().enumerate() {
            let a = leaves[k].grad(&grads).map_or(0.0, |g| g.into_scalar() as f64);
            // Swept 1e-4..1e-1 (relative): below ~1e-3 the differences sink
            // into f32 routing noise, above ~1e-2 they straddle the pool's
            // fill / empty / onset kinks; 3e-3 sits between, where every
            // regime and parameter agrees to within 0.3 %.
            let eps = 3e-3 * p[k].abs().max(1.0);
            let bump = |d: f32| {
                let mut q = p;
                q[k] += d;
                gc_loss(q)
            };
            let fd = (bump(eps) - bump(-eps)) / (2.0 * eps as f64);
            let abs = (a - fd).abs();
            let rel = abs / a.abs().max(fd.abs()).max(1e-12);
            println!("  {label} {name}: analytical={a:.6e} fd={fd:.6e} rel={rel:.3e}");
            // z has no gradient while the cap never binds; everything else must.
            if label == "capture" && *name == "z" {
                assert_eq!(a, 0.0, "{label}: z gets no gradient while the pool never fills");
                assert!(fd.abs() < 1e-3, "{label}: z's finite difference {fd:.3e} must vanish too");
                continue;
            }
            assert!(a != 0.0 && a.is_finite(), "{label} {name}: vacuous gradient");
            if !(rel < REL_TOL || abs < ABS_TOL) {
                failures.push(format!("{label} {name}: analytical {a:.6e} vs fd {fd:.6e} (rel {rel:.3e})"));
            }
        }
    }
    assert!(failures.is_empty(), "gradcheck failed: {failures:?}");
}

// ---------------------------------------------------------------------------
// 4. The pool runs across test-phase chunks.
// ---------------------------------------------------------------------------

const CHUNK_TOTAL: usize = 120 * 24;

fn chunk_case() -> (Config, Vec<f32>) {
    let q_up = floods(CHUNK_TOTAL, 4.0, 40.0, 30, 17 * 24 + 5, 60);
    (Config::default(), chain_q(&q_up, 1.0))
}

fn chunk_pool() -> (Tensor<AB, 1>, Tensor<AB, 1>, Tensor<AB, 1>, f32) {
    (t1(2.0), t1(0.8), t1(4.0), 5.0)
}

/// The dam's routed series over the whole period as 15-day chunks threaded
/// like `training::eval::evaluate` (final discharge column, the dam state via
/// `carry_dam_state`, `DamClampSums::merge_pool`), with the pool carried or
/// dropped at every boundary.
fn route_chunked(carry_pool: bool) -> (Vec<f64>, DamClampSums) {
    let (cfg, q) = chunk_case();
    let mut sums = DamClampSums::default();
    let mut dam = Vec::with_capacity(CHUNK_TOTAL + 1);
    let mut state: Option<Vec<f32>> = None;
    let mut h0 = 0;
    while h0 < CHUNK_TOTAL {
        let h1 = (h0 + 15 * 24).min(CHUNK_TOTAL);
        let chunk_start = start() + Duration::days((h0 / 24) as i64);
        let mut mc = engine(&cfg, chain3(5000.0), &q[h0 * 3..(h1 + 1) * 3], state.as_deref());
        arm_chain_dam(&mut mc, chunk_start, h1 - h0 + 1, DamRow::Additive, t1(0.4), Some(chunk_pool()));
        if carry_pool {
            carry_dam_state(&mut mc, &sums);
        }
        let out = host2(mc.forward());
        sums.merge(&mc.dam_account().unwrap());
        sums.merge_pool(&mc.flood_pool_account().unwrap());
        let cols = h1 - h0 + 1;
        let row = series(&out, cols, 1);
        if h0 == 0 {
            dam.push(row[0]);
        }
        dam.extend_from_slice(&row[1..]);
        state = Some((0..3).map(|r| out[r * cols + cols - 1]).collect());
        h0 = h1;
    }
    (dam, sums)
}

#[test]
fn the_pool_runs_across_test_phase_chunks() {
    let (cfg, q) = chunk_case();
    let mut mc = engine(&cfg, chain3(5000.0), &q, None);
    arm_chain_dam(&mut mc, start(), CHUNK_TOTAL + 1, DamRow::Additive, t1(0.4), Some(chunk_pool()));
    let single = series(&host2(mc.forward()), CHUNK_TOTAL + 1, 1);
    let single_pool = mc.flood_pool_account().unwrap();
    let (chunked, sums) = route_chunked(true);
    let (dropped, _) = route_chunked(false);
    assert_eq!(chunked.len(), single.len());
    let max_diff = |a: &[f64], b: &[f64]| a.iter().zip(b).map(|(x, y)| (x - y).abs()).fold(0.0_f64, f64::max);
    let scale = single.iter().fold(0.0_f64, |m, &v| m.max(v.abs()));
    let (d_carry, d_drop) = (max_diff(&chunked, &single), max_diff(&dropped, &single));
    let rec = sums.pool_by_row[&1];
    println!(
        "one engine vs 15-day chunks: max |dQ| {d_carry:.3e} m3/s with the pool carried, {d_drop:.3e} dropped \
         (scale {scale:.3e}); captured {:.6e} / {:.6e}, evacuated {:.6e} / {:.6e}, F_end {:.4e} / {:.4e}",
        single_pool.captured_m3[0], rec.captured_m3, single_pool.evacuated_m3[0], rec.evacuated_m3,
        single_pool.f_end_m3[0], rec.f_end_m3
    );
    assert!(d_carry <= 1e-6 * scale, "chunked with the pool carried differs by {d_carry:.3e}");
    assert!(d_drop > 1e-2 * scale, "dropping the pool must change the routing ({d_drop:.3e})");
    let close = |a: f64, b: f64| (a - b).abs() <= 1e-5 * a.abs().max(b.abs()).max(1.0);
    assert!(close(rec.captured_m3, single_pool.captured_m3[0]), "captured");
    assert!(close(rec.evacuated_m3, single_pool.evacuated_m3[0]), "evacuated");
    assert!(close(rec.f_end_m3, single_pool.f_end_m3[0]), "F_end");
    assert_eq!(rec.steps, CHUNK_TOTAL as u64);
    assert!(rec.f_peak_m3 > 0.0 && rec.captured_m3 > 0.0);
}

// ---------------------------------------------------------------------------
// 5. Input checks.
// ---------------------------------------------------------------------------

#[test]
fn set_flood_pool_refuses_bad_input() {
    let q = sandbox_q_prime(4);
    let pool = |rows: Vec<usize>, ibar: Vec<f32>| FloodPool {
        kc: Tensor::from_floats(vec![1.0_f32; rows.len()].as_slice(), &Device::default()),
        phi: Tensor::from_floats(vec![0.5_f32; rows.len()].as_slice(), &Device::default()),
        z_days: Tensor::from_floats(vec![1.0_f32; rows.len()].as_slice(), &Device::default()),
        rows,
        inflow_mean: ibar,
    };
    let mut bare = engine(&Config::default(), sandbox5(), &q, None);
    assert!(bare.set_flood_pool(pool(vec![3], vec![1.0])).unwrap_err().contains("no dam rows"));
    assert!(bare.set_flood_pool_state(&[0.0]).unwrap_err().contains("no flood pool"));
    let mut mc = engine(&Config::default(), sandbox5(), &q, None);
    mc.set_reservoir_rows_as(&[3, 4], &[0.8, 2.5], DamRow::Additive).unwrap();
    assert!(mc.set_flood_pool(pool(vec![2], vec![1.0])).unwrap_err().contains("not an armed dam row"));
    assert!(mc.set_flood_pool(pool(vec![3, 3], vec![1.0, 1.0])).unwrap_err().contains("twice"));
    assert!(mc.set_flood_pool(pool(vec![3], vec![1.0, 2.0])).unwrap_err().contains("Ibar has 2"));
    assert!(mc.set_flood_pool(pool(vec![3], vec![-1.0])).unwrap_err().contains(">= 0"));
    assert!(mc.flood_pool_rows().is_none(), "a refused pool leaves the engine without one");
    mc.set_flood_pool(pool(vec![4, 3], vec![5.0, 7.0])).unwrap();
    assert_eq!(mc.flood_pool_rows(), Some(&[4_usize, 3][..]));
    assert!(mc.set_flood_pool_state(&[1.0]).unwrap_err().contains("2 pooled dams"));
    assert!(mc.set_flood_pool_state(&[1.0, -1.0]).unwrap_err().contains(">= 0"));
    mc.set_flood_pool_state(&[100.0, 0.0]).unwrap();
    // Re-arming the dams drops the pool.
    mc.set_reservoir_rows_as(&[3, 4], &[0.8, 2.5], DamRow::Additive).unwrap();
    assert!(mc.flood_pool_rows().is_none());
}

// ---------------------------------------------------------------------------
// 6. Training path = resolved test-phase table.
// ---------------------------------------------------------------------------

fn section(mode: FloodPoolMode) -> ReleaseHeadSection {
    ReleaseHeadSection {
        hidden_size: 6,
        num_hidden_layers: 1,
        grid: 5,
        k: 3,
        input_var_names: vec!["f1".into(), "f2".into()],
        seasonal: false,
        dam_row: DamRow::Additive,
        dam_floor: ddrs::config::DamFloor::Forgive,
        dam_row_positivity: false,
        routing_checkpoint: None,
        freeze_routing: false,
        rule_curve: true,
        rule_curve_max: 0.5,
        per_dam_t0: true,
        per_dam_lr: 0.05,
        per_dam_l2: 0.0,
        rule_curve_penalty: 0.0,
        rule_curve_alpha: 0.9,
        flood_pool: mode,
    }
}

fn learned_cfg(mode: FloodPoolMode) -> Config {
    let mut cfg = Config::default();
    cfg.params.use_reservoirs = true;
    cfg.params.reservoir_release = ReservoirRelease::Learned;
    cfg.release_head = Some(section(mode));
    cfg
}

/// Dams on reaches 3 and 4 (COMIDs 103, 104) plus one outside the network;
/// only 104 is a flood-control dam.
fn features() -> DamFeatures {
    DamFeatures {
        comids: vec![Comid(104), Comid(999), Comid(103)],
        names: vec!["f1".into(), "f2".into()],
        values: ndarray::array![[0.8_f32, -1.2], [0.0, 0.0], [-0.5, 1.7]],
        years: vec![None, None, None],
        inflow_mean: Some(vec![20.0, 5.0, 8.0]),
        purpose_flood: Some(vec![true, false, false]),
    }
}

/// Per-dam parameters with distinct pools (table order: 104, 999, 103).
fn dam_params() -> DamParams<AB> {
    let device = Device::default();
    let mut d = DamParams::<AB>::zeros_with_pool(3, true, true, true, &device);
    d.theta = Some(Param::from_tensor(Tensor::from_floats(
        [[0.2, -0.1, 0.1, 0.0], [0.0; 4], [-0.3, 0.1, 0.0, -0.2]],
        &device,
    )));
    d.delta = Some(Param::from_tensor(Tensor::from_floats([0.5, 0.0, -0.4], &device)));
    d.pool = Some(Param::from_tensor(Tensor::from_floats(
        [[-0.4, 0.8, 1.3], [0.0; 3], [-0.9, -0.5, 1.1]],
        &device,
    )));
    d
}

#[test]
fn training_and_resolved_flood_pool_route_identically() {
    use burn::module::AutodiffModule;
    let steps = 120;
    let start = NaiveDate::from_ymd_opt(1990, 4, 20).unwrap();
    let network: Vec<Comid> = (0..5).map(|i| Comid(100 + i)).collect();
    let q = sandbox_q_prime(steps);
    let device = Device::default();
    for (mode, pooled_rows) in [(FloodPoolMode::All, vec![3, 4]), (FloodPoolMode::FloodControl, vec![4])] {
        let cfg = learned_cfg(mode);
        let head = init_release_head::<AB>(&section(mode), &cfg.params.parameter_ranges, 42, &device);
        let dams = dam_params();
        let rows = map_reservoir_rows(&ReservoirTable::Learned(features()), &network);
        let mut a = engine(&cfg, sandbox5(), &q, None);
        apply_reservoir_rows_with(&cfg, &mut a, Some(&rows), start, steps + 1, Some(&head), Some(&dams));
        let mut armed: Vec<usize> = a.flood_pool_rows().expect("pool armed").to_vec();
        armed.sort_unstable();
        assert_eq!(armed, pooled_rows, "{mode:?}: pooled rows");
        let train = host2(a.forward());
        let train_pool = a.flood_pool_account().unwrap();
        assert!(train_pool.captured_m3.iter().any(|&v| v > 0.0), "{mode:?}: the case must capture");

        let table = resolve_release_table_with::<I>(&head.valid(), Some(&dams.valid()), &features(), &cfg);
        let pool = table.flood_pool.as_ref().expect("the resolved table carries the pool");
        // Row 0 (COMID 104): the transform of its raw parameters.
        let r = Tensor::<I, 1>::from_floats([-0.4_f32, 0.8, 1.3], &Default::default()).reshape([1, 3]);
        let t = flood_pool_transform(r);
        let host = |t: Tensor<I, 1>| -> f32 { t.into_data().to_vec::<f32>().unwrap()[0] };
        assert_eq!(pool[0], [host(t.kc), host(t.phi), host(t.z_days)]);
        match mode {
            FloodPoolMode::FloodControl => assert_eq!((pool[1][2], pool[2][2]), (0.0, 0.0), "not flood-control: z = 0"),
            _ => assert!(pool[2][2] > 0.0),
        }

        let rows_b = map_reservoir_rows(&ReservoirTable::Fixed(table.clone()), &network);
        let mut b = engine(&cfg, sandbox5(), &q, None);
        apply_reservoir_rows_with(&cfg, &mut b, Some(&rows_b), start, steps + 1, None, None);
        let resolved = host2(b.forward());
        assert_bitwise(&train, &resolved, &format!("{mode:?}: training vs resolved flood pool"));
        assert_eq!(b.flood_pool_account().unwrap().captured_m3, train_pool.captured_m3);

        let dir = tempfile::tempdir().unwrap();
        let csv = dir.path().join("release_params.csv");
        write_release_params_csv(&csv, &table, DamRow::Additive).unwrap();
        let text = std::fs::read_to_string(&csv).unwrap();
        assert!(
            text.starts_with("COMID,T0_days,a,b,T_min_days,T_max_days,c1s,c1c,c2s,c2c,kc,phi,z,inflow_mean_m3s\n"),
            "{text}"
        );
        // The pool columns hold the resolved values (T0_days is not the fixed
        // reader's T_days, so the file is not itself a fixed table).
        let row1: Vec<&str> = text.lines().nth(1).unwrap().split(',').collect();
        let kc_phi_z: Vec<f32> = row1[10..13].iter().map(|v| v.parse().unwrap()).collect();
        assert_eq!(kc_phi_z, pool[0].to_vec());
    }
}

// ---------------------------------------------------------------------------
// 7. Dataset open.
// ---------------------------------------------------------------------------

const JUNIATA_CONFIG: &str = "examples/juniata/ddrs.yaml";
const DAM_TABLE: &str = "examples/juniata/data/juniata_dam_features.csv";

fn juniata_yaml() -> serde_yaml::Value {
    let mut cfg: serde_yaml::Value =
        serde_yaml::from_str(&std::fs::read_to_string(JUNIATA_CONFIG).unwrap()).unwrap();
    cfg["experiment"]["epochs"] = 3.into();
    cfg["experiment"]["rho"] = 20.into();
    cfg["experiment"]["warmup"] = 2.into();
    cfg
}

fn write_cfg(dir: &Path, name: &str, cfg: &serde_yaml::Value) -> Config {
    let path = dir.join(name);
    std::fs::write(&path, serde_yaml::to_string(cfg).unwrap()).unwrap();
    Config::from_yaml_file(&path).expect("config loads")
}

/// The Juniata dam table with extra columns appended to every row.
fn dam_table_with(dir: &Path, name: &str, header_extra: &str, row_extra: &str, drop_flood: bool) -> PathBuf {
    let table = std::fs::read_to_string(DAM_TABLE).unwrap();
    let mut lines = table.lines();
    let mut header: Vec<String> = lines.next().unwrap().split(',').map(String::from).collect();
    let flood_col = header.iter().position(|h| h == "purpose_flood").unwrap();
    let mut body: Vec<Vec<String>> = lines.map(|l| l.split(',').map(String::from).collect()).collect();
    if drop_flood {
        header.remove(flood_col);
        for r in &mut body {
            r.remove(flood_col);
        }
    }
    let mut text = header.join(",") + header_extra + "\n";
    for r in body {
        text.push_str(&(r.join(",") + row_extra + "\n"));
    }
    let path = dir.join(name);
    std::fs::write(&path, text).unwrap();
    path
}

fn learned_juniata(dir: &Path, name: &str, table: &Path, features: &[&str], extra: &[(&str, serde_yaml::Value)]) -> Config {
    let mut cfg = juniata_yaml();
    cfg["params"]["use_reservoirs"] = true.into();
    cfg["params"]["reservoir_release"] = "learned".into();
    cfg["data_sources"]["reservoirs"] = table.display().to_string().into();
    let mut rh = serde_yaml::Mapping::new();
    rh.insert(
        "input_var_names".into(),
        serde_yaml::Value::Sequence(features.iter().map(|s| (*s).into()).collect()),
    );
    for (k, v) in extra {
        rh.insert((*k).into(), v.clone());
    }
    cfg["release_head"] = serde_yaml::Value::Mapping(rh);
    write_cfg(dir, name, &cfg)
}

fn open_err(cfg: &Config) -> String {
    match ddrs::data::MeritGagesDataset::open(cfg) {
        Ok(_) => panic!("the dataset must not open"),
        Err(e) => e.to_string(),
    }
}

#[test]
fn dataset_open_refuses_a_pool_without_its_columns() {
    let dir = tempfile::tempdir().unwrap();
    // No inflow_mean_m3s.
    let cfg = learned_juniata(dir.path(), "a.yaml", Path::new(DAM_TABLE), &["log10_storage"], &[("flood_pool", "all".into())]);
    let err = open_err(&cfg);
    assert!(err.contains("inflow_mean_m3s") && err.contains("flood_pool"), "{err}");
    // flood_control without purpose_flood (the head does not read it either).
    let no_flood = dam_table_with(dir.path(), "no_flood.csv", ",inflow_mean_m3s", ",33.006", true);
    let cfg = learned_juniata(dir.path(), "b.yaml", &no_flood, &["log10_storage"], &[("flood_pool", "flood_control".into())]);
    let err = open_err(&cfg);
    assert!(err.contains("purpose_flood") && err.contains("flood_control"), "{err}");
    // The same table opens with `all`.
    let cfg = learned_juniata(dir.path(), "c.yaml", &no_flood, &["log10_storage"], &[("flood_pool", "all".into())]);
    assert!(ddrs::data::MeritGagesDataset::open(&cfg).is_ok());
    // A fixed table with `reservoir_flood_pool: true` but no kc, phi, z.
    let mut y = juniata_yaml();
    y["params"]["use_reservoirs"] = true.into();
    y["params"]["reservoir_dam_row"] = "additive".into();
    y["params"]["reservoir_flood_pool"] = true.into();
    y["data_sources"]["reservoirs"] = "examples/juniata/data/juniata_reservoirs.csv".into();
    let cfg = write_cfg(dir.path(), "d.yaml", &y);
    let err = open_err(&cfg);
    assert!(err.contains("reservoir_flood_pool") && err.contains("kc"), "{err}");
}

// ---------------------------------------------------------------------------
// 8. Frozen-routing training moves the pool.
// ---------------------------------------------------------------------------

/// Every float parameter of a module, in visit order.
struct Collect(Vec<f32>);

impl<B: Backend> ModuleVisitor<B> for Collect {
    fn visit_float<const D: usize>(&mut self, param: &Param<Tensor<B, D>>) {
        self.0.extend(param.val().into_data().convert::<f32>().to_vec::<f32>().unwrap());
    }
}

fn params<B: Backend, M: Module<B>>(m: &M) -> Vec<f32> {
    let mut c = Collect(Vec::new());
    m.visit(&mut c);
    c.0
}

const DAM_FEATURES: [&str; 19] = [
    "log10_storage",
    "log10_storage_max",
    "log10_surface",
    "log10_drainage",
    "log10_storage_per_area",
    "log10_max_discharge",
    "height",
    "year",
    "log10_storage_max_missing",
    "log10_surface_missing",
    "log10_max_discharge_missing",
    "year_missing",
    "purpose_flood",
    "purpose_hydro",
    "purpose_supply",
    "purpose_irrigation",
    "purpose_recreation",
    "purpose_navigation",
    "purpose_other",
];

#[test]
fn frozen_routing_training_moves_the_pool_and_resumes_it() {
    let device = Device::default();
    let tmp = tempfile::tempdir().unwrap();
    // A routing checkpoint that is not the run's own seed-42 init.
    let plain = write_cfg(tmp.path(), "plain.yaml", &juniata_yaml());
    let head = kan_config(plain.kan_head.as_ref().unwrap(), 7).init::<I>(&device);
    let ckpt = tmp.path().join("routing_ckpt").join("epoch_9_mb_0");
    std::fs::create_dir_all(&ckpt).unwrap();
    save_kan_head(&head_base(&ckpt), &head).unwrap();
    // Raystown with a small Ibar (10 m3/s) so Qc = 3·Ibar sits inside its
    // flow range and the pool captures in 20-day windows.
    let table = dam_table_with(tmp.path(), "dams.csv", ",inflow_mean_m3s", ",10.0", false);
    let mut y = juniata_yaml();
    y["params"]["use_reservoirs"] = true.into();
    y["params"]["reservoir_release"] = "learned".into();
    y["data_sources"]["reservoirs"] = table.display().to_string().into();
    let mut rh = serde_yaml::Mapping::new();
    rh.insert(
        "input_var_names".into(),
        serde_yaml::Value::Sequence(DAM_FEATURES.iter().map(|s| (*s).into()).collect()),
    );
    for (k, v) in [
        ("routing_checkpoint", ckpt.display().to_string().into()),
        ("freeze_routing", true.into()),
        ("dam_row", "additive".into()),
        ("dam_row_positivity", true.into()),
        ("seasonal", false.into()),
        ("per_dam_t0", true.into()),
        ("flood_pool", "flood_control".into()),
        ("per_dam_l2", serde_yaml::Value::from(1e-3)),
    ] {
        rh.insert(k.into(), v);
    }
    y["release_head"] = serde_yaml::Value::Mapping(rh);
    let cfg = write_cfg(tmp.path(), "pool.yaml", &y);
    assert!(cfg.flood_pool_on() && cfg.routing_frozen());

    let template = kan_config(cfg.kan_head.as_ref().unwrap(), cfg.seed).init::<AB>(&device);
    let reference = params(&load_kan_head::<AB>(&head_base(&ckpt), template, &device).unwrap());
    let dataset = ddrs::data::MeritGagesDataset::open(&cfg).unwrap();
    let (_, mut state, mut optimizer) = bootstrap_head_and_state::<I>(&cfg, &device).unwrap();
    let pool0 = state.release.as_ref().unwrap().dams.as_ref().unwrap().params.pool.as_ref().unwrap().val();
    assert!(pool0.into_data().to_vec::<f32>().unwrap().iter().all(|&v| v == 0.0), "pool raw parameters start at 0");

    let run_ckpts = tmp.path().join("checkpoints");
    train::<I>(&cfg, &dataset, &mut state, &mut optimizer, &device, &run_ckpts, None, None).unwrap();

    let head_now: &KanHead<AB> = &state.head;
    let now = params(head_now);
    assert_eq!(now.len(), reference.len());
    assert!(now.iter().zip(&reference).all(|(a, b)| a.to_bits() == b.to_bits()), "routing head moved");
    let d = &state.release.as_ref().unwrap().dams.as_ref().unwrap().params;
    let pool: Vec<f32> = d.pool.as_ref().unwrap().val().into_data().to_vec().unwrap();
    println!("Raystown pool raw (r_kc, r_phi, r_z) after 3 steps: {pool:?}");
    assert!(pool.iter().all(|&v| v != 0.0), "kc, phi and z all train: {pool:?}");

    // Resume from the last checkpoint: the pool comes back bitwise.
    let mut saved: Vec<PathBuf> = std::fs::read_dir(&run_ckpts).unwrap().map(|e| e.unwrap().path()).collect();
    saved.sort();
    let last = saved.last().unwrap();
    let mut resume = y.clone();
    resume["experiment"]["checkpoint"] = last.display().to_string().into();
    let cfg2 = write_cfg(tmp.path(), "pool_resume.yaml", &resume);
    let (_, state2, _) = bootstrap_head_and_state::<I>(&cfg2, &device).unwrap();
    let d2 = &state2.release.as_ref().unwrap().dams.as_ref().unwrap();
    let pool2: Vec<f32> = d2.params.pool.as_ref().unwrap().val().into_data().to_vec().unwrap();
    assert_bitwise(&pool2, &pool, "pool after resume");
    assert!(d2.optimizer.pool.as_ref().is_some_and(|s| s.t.iter().any(|&t| t > 0)), "pool Adam state restored");
}

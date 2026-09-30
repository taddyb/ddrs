//! Differentiable Muskingum-Cunge routing core.
//!
//! Port of `~/projects/ddr/src/ddr/routing/mmc.py` (`MuskingumCunge` class).
//! See `~/projects/ddr/CLAUDE.md` for the algorithm overview.
//!
//! ```text
//! per timestep t:
//!   1. trapezoidal geometry from Q_t  →  velocity v
//!   2. celerity  c = clamp(v, v_lb, 15) · 5/3
//!   3. k = L/c;  Muskingum c1..c4  (dt = 3600 s, hardcoded)
//!   4. solve (I − c1·N) · Q_{t+1} = c2·(N·Q_t) + c3·Q_t + c4·q'      (CSR + analytical backward)
//!   5. Q_{t+1} := clamp(Q_{t+1}, discharge_lb)
//! ```
//!
//! Type story: the engine is generic over an *inner* backend `I: Backend`,
//! and stores all autograd-participating tensors as `Tensor<Autodiff<I>, ...>`.
//! Pure forward callers can still construct the engine on `I = NdArray<f32>` and
//! simply never `.require_grad()` anything — Autodiff's overhead in that mode
//! is negligible.

use std::sync::Arc;

use burn::backend::Autodiff;
use burn::tensor::{backend::Backend, Bool, IndexingUpdateOp, Int, Tensor, TensorData};

use burn::tensor::TensorPrimitive;

use crate::config::{Config, DamFloor, DamRow, SparseSolver};
use crate::routing::mmc_op::{ReleaseParent, ReservoirTensors};
use crate::routing::release::{MIN_T_DAYS, SECONDS_PER_DAY};
use crate::routing::utils::denormalize;
use crate::sparse::{triangular_csr_solve, AValuesAssembler, CsrPattern, SparseAdjacency};

/// Hardcoded routing timestep in seconds. Matches `self.t` in `mmc.py:192`.
pub const DT_SECONDS: f32 = 3600.0;

/// Per-row lateral-inflow divisor from a `parent_offset` map: every row owned
/// by parent `p` gets `m_p = parent_offset[p + 1] - parent_offset[p]`, the
/// number of sub-reach pieces the parent was split into.
///
/// Returns `None` when no parent owns more than one row — the network is not
/// subdivided, every divisor would be `1.0`, and the caller can skip the
/// division entirely instead of emitting a no-op tensor op.
fn pieces_per_row_divisor(parent_offset: &[i32], n: usize) -> Option<Vec<f32>> {
    let mut divisor = Vec::with_capacity(n);
    let mut subdivided = false;
    for w in parent_offset.windows(2) {
        let m = w[1] - w[0];
        assert!(
            m >= 1,
            "parent_offset must be strictly increasing; parent owns {m} rows"
        );
        subdivided |= m > 1;
        divisor.extend(std::iter::repeat_n(m as f32, m as usize));
    }
    assert_eq!(
        divisor.len(),
        n,
        "parent_offset covers {} sub-reach rows but the network has {n}",
        divisor.len()
    );
    subdivided.then_some(divisor)
}

/// Static channel attributes and topology for a network.
///
/// Adjacency, channel length, and slope come bundled inside `SparseAdjacency`
/// — they share the same topological order and are loaded together from the
/// underlying zarr/COO source. `x_storage` (Muskingum storage weight) is a
/// numerical-scheme parameter, kept separate so it can be supplied as a
/// learnable or per-batch tensor.
pub struct RoutingInputs<I: Backend> {
    pub adjacency: SparseAdjacency,
    pub x_storage: Tensor<Autodiff<I>, 1>,
}

/// NN-derived parameters in `[0, 1]`; denormalized inside `setup_inputs`.
pub struct SpatialParameters<I: Backend> {
    pub n: Tensor<Autodiff<I>, 1>,
    pub q_spatial: Tensor<Autodiff<I>, 1>,
    pub p_spatial: Option<Tensor<Autodiff<I>, 1>>,
    /// Leakance params (all-or-nothing). Present ⇒ route via `TimestepLeakanceOp`.
    /// All three must be `Some` or all `None`; partial presence routes the
    /// non-leakance path (any `None` ⇒ leakance disabled).
    pub k_d: Option<Tensor<Autodiff<I>, 1>>,
    pub d_gw: Option<Tensor<Autodiff<I>, 1>>,
    pub leakance_factor: Option<Tensor<Autodiff<I>, 1>>,
    /// Optional per-reach impervious hard-zero mask (0.0 = impervious, 1.0 = normal).
    /// Precomputed from `corridor_impervious` attribute by the caller using
    /// `cfg.params.leakance_impervious_threshold`. Constant, not autograd-tracked
    /// (inner backend `I`, no gradient). `None` ⇒ all-ones (no-op, back-compat).
    pub impervious_mask: Option<Tensor<I, 1>>,
    /// Learned per-reach stage-roughness exponent for
    /// `n(d) = n_0·(d/d_ref)^(−gamma)`. Present ⇒ gamma is the timestep op's
    /// sixth parent and receives a gradient. Absent ⇒ the global
    /// `params.stage_roughness.gamma` (or no stage roughness at all).
    pub gamma: Option<Tensor<Autodiff<I>, 1>>,
}

/// A dam release for [`MuskingumCunge::set_dam_release`]: each dam row is a
/// linear reservoir `S = T_d(t)·Q` with
/// `T_d(t) = max(T0_d·exp(a_d·sin ω_t + b_d·cos ω_t), 1/24 d)` (the replace
/// row), or adds `T_d(t)·Q`, unfloored, to the reach's channel storage (the
/// additive row, `dam_row: Additive`). See `crate::routing::release` for the
/// phase convention, the two rows and the clamp.
///
/// The tensors may be autodiff-tracked (the learned release: `T0`, `a`, `b`
/// come from the release head) or constants (a seasonal `fixed` table). Either
/// way `T` reaches the timestep op as a parent, so a tracked `T0` gets its
/// gradient through the hand-written backward (B19''' / B19'''') and ordinary
/// autodiff.
pub struct DamRelease<I: Backend> {
    /// Dam row positions in the network, unique.
    pub rows: Vec<usize>,
    /// `[rows.len()]` residence time scale `T0`, days.
    pub t0_days: Tensor<Autodiff<I>, 1>,
    /// `(a, b)`, each `[rows.len()]`. `None` is a constant `T = T0`.
    pub seasonal: Option<(Tensor<Autodiff<I>, 1>, Tensor<Autodiff<I>, 1>)>,
    /// `(sin ω, cos ω)` per lateral-inflow row (hour of the window); must
    /// cover every routed step. `crate::routing::release::seasonal_phase`.
    pub phase: Vec<[f32; 2]>,
    /// `Replace` (the dam row is the reservoir alone, `T` floored at 1 h) or
    /// `Additive` (the reservoir's storage added to the reach's channel
    /// storage, `T >= 0`, no floor). See `crate::routing::release`.
    pub dam_row: DamRow,
    /// The harmonic rule curve, `S = T·Q + S0_d(t)`; `None` is none.
    pub rule_curve: Option<RuleCurve<I>>,
}

/// Per-dam harmonic rule curve for [`DamRelease`] (`crate::routing::release`):
/// the flux `r_d(t) = Ibar_d·Σ_k(c_{k,s} sin kω_t + c_{k,c} cos kω_t)`, whose
/// step increment `(S0_{t+1} − S0_t)/dt` is taken off the dam row's lateral
/// inflow. The coefficients may be autodiff-tracked (learned per-dam free
/// parameters); their gradient arrives through the timestep op's `q'` parent.
pub struct RuleCurve<I: Backend> {
    /// `[n_dams, 4]`, `(c1s, c1c, c2s, c2c)`, dimensionless.
    pub coeffs: Tensor<Autodiff<I>, 2>,
    /// `[n_dams]` `Ibar`, m³/s (the table's `inflow_mean_m3s`).
    pub inflow_mean: Tensor<Autodiff<I>, 1>,
    /// Continuous seasonal phase `ω` (rad) at lateral-inflow row 0, from
    /// `crate::routing::release::rule_curve_phase_start(window_start)`; every
    /// row advances it by `Ω·dt`. Not the `T` law's day-of-year `phase`.
    pub phase0: f64,
}

/// Per-dam flood pool for [`MuskingumCunge::set_flood_pool`]: law FA of
/// `experiments/reservoir/laws_v6` (report §2, §5), on top of whatever the dam
/// row already is (option C, the seasonal or learned bucket, a rule curve).
///
/// Each pooled dam `d` carries a pool `F_d` (m³, `>= 0`), empty at a window
/// start (or the previous test-phase chunk's closing value, set with
/// [`MuskingumCunge::set_flood_pool_state`]). Before each step's solve, from
/// the dam's inflow at the step start
/// `I_d = (N·Q_t)_d + q'_d` (routed upstream inflow plus the reach's own
/// lateral inflow after the floor and the sub-reach divisor, before the rule
/// curve's flux: the `Qin` of the feasibility penalty and of the clamp
/// account), with `Qc = kc·Ibar`, `Fmax = z·Ibar·86400`:
///
/// ```text
/// Vc = min(phi·max(I − Qc, 0)·dt, max(Fmax − F, 0))      captured, m³
/// p  = I − Vc/dt                                          passed through, m³/s
/// Ve = min(F, max(Qc − p, 0)·dt)                          evacuated at the target, m³
/// q'_d ← q'_d + (Ve − Vc)/dt                              the dam row's lateral inflow
/// F  ← (F + Vc) − Ve                                      after the solve
/// ```
///
/// Mass is conserved by construction: the pool captures only what enters
/// (`Vc <= phi·I·dt <= I·dt`) and evacuates only what it holds (`Ve <= F`),
/// and `F + Vc − Ve >= 0` exactly in f32 (`fl(F + Vc) >= F >= Ve`). It never
/// relies on the discharge floor. The system matrix, the CSR pattern and the
/// hand-written backward are unchanged: the fluxes act on the dam row's `q'`
/// like the rule curve's, and reach `kc`, `phi`, `z` (and, through `I`, the
/// routed upstream discharge and `q'`) by ordinary autodiff through the
/// timestep op's `q'` parent. `F` stays on the autodiff tape across the
/// window's steps (`O(n_pool)` nodes per step); `I` is NOT detached, so a
/// routing parameter upstream of a pooled dam sees the pool's response to its
/// inflow (e.g. `1 − phi` of a change above `Qc` while the pool captures).
/// `min`/`max` take their subgradients (Burn's `mask_where`: the selected
/// operand, the left one at a tie).
///
/// `z = 0` is no pool, bit for bit: `Fmax = F = 0`, so `Vc = Ve = 0` and the
/// dam row's `q'` gains an exact `+0`.
pub struct FloodPool<I: Backend> {
    /// Network rows of the pooled dams: armed dam rows, unique.
    pub rows: Vec<usize>,
    /// `[rows.len()]` release target multiplier, `Qc = kc·Ibar` (m³/s).
    pub kc: Tensor<Autodiff<I>, 1>,
    /// `[rows.len()]` capture share of the inflow above `Qc`, in `[0, 1]`.
    pub phi: Tensor<Autodiff<I>, 1>,
    /// `[rows.len()]` pool size in days of mean inflow,
    /// `Fmax = z·Ibar·86400` m³; `>= 0`.
    pub z_days: Tensor<Autodiff<I>, 1>,
    /// `[rows.len()]` `Ibar`, m³/s: the dam's training-period mean inflow
    /// (the table's `inflow_mean_m3s`), a constant.
    pub inflow_mean: Vec<f32>,
}

/// [`FloodPool`] armed on an engine: device tensors, the pool state and the
/// pool's account (inner backend, detached).
struct ArmedFloodPool<I: Backend> {
    rows: Vec<usize>,
    /// The pooled rows, for the gather of `q'` and the scatter onto it.
    rows_ad: Tensor<Autodiff<I>, 1, Int>,
    /// Upstream edges of the pooled rows: source row, pool slot, and the
    /// adjacency weight (`None` when every weight is 1). `None` when no
    /// pooled dam has an upstream reach.
    upstream: Option<(Tensor<Autodiff<I>, 1, Int>, Tensor<Autodiff<I>, 1, Int>, Option<Tensor<Autodiff<I>, 1>>)>,
    /// `Qc = kc·Ibar`, m³/s.
    qc: Tensor<Autodiff<I>, 1>,
    phi: Tensor<Autodiff<I>, 1>,
    /// `Fmax = z·Ibar·86400`, m³.
    fmax: Tensor<Autodiff<I>, 1>,
    /// The pool `F`, m³, on the tape.
    f: Tensor<Autodiff<I>, 1>,
    inflow_mean: Vec<f32>,
    /// Account: `Σ Vc`, `Σ Ve`, `max F` (including the carried-in state),
    /// `Σ I·dt`, m³, and the steps routed.
    captured: Tensor<I, 1>,
    evacuated: Tensor<I, 1>,
    f_peak: Tensor<I, 1>,
    inflow: Tensor<I, 1>,
    steps: usize,
}

impl<I: Backend> ArmedFloodPool<I> {
    /// One step's pool fluxes (see [`FloodPool`]): `(Ve − Vc)/dt` per pooled
    /// dam, m³/s, to add to its `q'`; updates `F` and the account. `q_t` is the
    /// routed discharge at the step start (`[n]`), `q_prime` the lateral
    /// inflow after the floor and the divisor, before the rule-curve flux.
    fn step(&mut self, q_t: Tensor<Autodiff<I>, 1>, q_prime: Tensor<Autodiff<I>, 1>, dt: f32) -> Tensor<Autodiff<I>, 1> {
        let own = q_prime.select(0, self.rows_ad.clone());
        let inflow = match &self.upstream {
            Some((src, dst, w)) => {
                let up = q_t.select(0, src.clone());
                let up = match w {
                    Some(w) => up * w.clone(),
                    None => up,
                };
                let n_pool = self.rows.len();
                Tensor::<Autodiff<I>, 1>::zeros([n_pool], &own.device()).select_assign(
                    0,
                    dst.clone(),
                    up,
                    IndexingUpdateOp::Add,
                ) + own
            }
            None => own,
        };
        let want = (inflow.clone() - self.qc.clone()).clamp_min(0.0) * self.phi.clone() * dt;
        let room = (self.fmax.clone() - self.f.clone()).clamp_min(0.0);
        let vc = want.min_pair(room);
        let pass = inflow.clone() - vc.clone() / dt;
        let head = (self.qc.clone() - pass).clamp_min(0.0) * dt;
        let ve = self.f.clone().min_pair(head);
        self.f = self.f.clone() + vc.clone() - ve.clone();
        self.captured = self.captured.clone() + vc.clone().inner();
        self.evacuated = self.evacuated.clone() + ve.clone().inner();
        self.f_peak = self.f_peak.clone().max_pair(self.f.clone().inner());
        self.inflow = self.inflow.clone() + inflow.inner() * dt;
        self.steps += 1;
        (ve - vc) / dt
    }
}

/// Host copy of the engine's flood pool account
/// ([`MuskingumCunge::flood_pool_account`]) over the routed steps; one entry
/// per pooled dam, in the armed order.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct FloodPoolAccount {
    /// Network rows of the pooled dams.
    pub rows: Vec<usize>,
    /// `Σ Vc`: volume the pool captured, m³.
    pub captured_m3: Vec<f64>,
    /// `Σ Ve`: volume it evacuated, m³.
    pub evacuated_m3: Vec<f64>,
    /// `F` after the last routed step, m³ (the next test-phase chunk starts
    /// from it).
    pub f_end_m3: Vec<f64>,
    /// Largest `F` held, m³ (the carried-in state counts).
    pub f_peak_m3: Vec<f64>,
    /// Capacity `Fmax = z·Ibar·86400`, m³.
    pub fmax_m3: Vec<f64>,
    /// `Ibar`, m³/s.
    pub inflow_mean_m3s: Vec<f64>,
    /// `Σ I·dt`: the dam's inflow (routed upstream plus its own `q'`) at the
    /// step starts, m³.
    pub inflow_m3: Vec<f64>,
    /// Routed steps.
    pub steps: u64,
}

/// [`DamRelease`] armed on an engine: the dam rows as device tensors, plus the
/// constant `T` (seconds) when the release is not seasonal, and the step `T`
/// `forward` hands to `route_timestep`.
struct ArmedRelease<I: Backend> {
    rows: Tensor<I, 1, Int>,
    mask: Tensor<I, 1, Bool>,
    t0_days: Tensor<Autodiff<I>, 1>,
    seasonal: Option<(Tensor<Autodiff<I>, 1>, Tensor<Autodiff<I>, 1>)>,
    phase: Vec<[f32; 2]>,
    dam_row: DamRow,
    /// `max(T0, 1/24)·86400` (replace) or `T0·86400` (additive),
    /// precomputed once for a non-seasonal release.
    t_constant: Option<Tensor<Autodiff<I>, 1>>,
    /// This step's `(T_{t+1}, T_t)` (seconds), set by `forward` before
    /// `route_timestep`: the residence time at the step's end and start.
    t_step: Option<(Tensor<Autodiff<I>, 1>, Tensor<Autodiff<I>, 1>)>,
    /// Rule curve, armed: the per-step flux `(S0_{t+1} − S0_t)/dt` is built
    /// one step at a time ([`ArmedRuleCurve::flux_at`]).
    rule_curve: Option<ArmedRuleCurve<I>>,
}

/// The harmonic rule curve armed on an engine (`crate::routing::release`).
///
/// The step producing routed column `t` takes
/// `(S0_{t+1} − S0_t)/dt = (Ibar/dt) ⊙ (c · ΔH_{t−1})` off the dam rows' `q'`,
/// with `ΔH_s` a constant `[4]` vector of phase increments (seconds) and `c`
/// the `[n_dams, 4]` coefficients. Built per step from `c` directly, so each
/// step's autodiff nodes are `[n_dams]`-sized and the backward is
/// `O(n_steps · n_dams)`. (The first build formed the whole
/// `[n_steps, n_dams]` flux once and sliced a row per step; every slice's
/// backward materialised a full-size gradient, `O(n_steps² · n_dams)`.)
struct ArmedRuleCurve<I: Backend> {
    /// `[n_dams, 4]` coefficients `(c1s, c1c, c2s, c2c)`, autodiff.
    coeffs: Tensor<Autodiff<I>, 2>,
    /// `[n_dams]` `Ibar / dt`, 1/s·m³/s.
    scale: Tensor<Autodiff<I>, 1>,
    /// `ΔH_s` per step `s` (row `s` → `s + 1`), seconds.
    dh: Vec<[f32; 4]>,
    /// The dam rows as an autodiff index, for the scatter onto `q'`.
    rows: Tensor<Autodiff<I>, 1, Int>,
}

impl<I: Backend> ArmedRuleCurve<I> {
    /// `(S0_{s+1} − S0_s)/dt` per dam, m³/s, `[n_dams]`: the flux of step `s`
    /// (the step producing routed column `s + 1`). Positive is storing.
    fn flux_at(&self, s: usize) -> Tensor<Autodiff<I>, 1> {
        let n_dams = self.scale.dims()[0];
        let dh = Tensor::<Autodiff<I>, 1>::from_floats(self.dh[s], &self.scale.device()).reshape([4, 1]);
        self.coeffs.clone().matmul(dh).reshape([n_dams]) * self.scale.clone()
    }
}

impl<I: Backend> ArmedRelease<I> {
    /// `T` (seconds) at phase row `row` (hour `row` of the window), in autodiff.
    fn t_seconds_at(&self, row: usize) -> Tensor<Autodiff<I>, 1> {
        match (&self.seasonal, &self.t_constant) {
            (None, Some(c)) => c.clone(),
            (Some((a, b)), _) => {
                let [sin_w, cos_w] = self.phase[row];
                let expo = a.clone() * sin_w + b.clone() * cos_w;
                let t_days = self.t0_days.clone() * expo.exp();
                match self.dam_row {
                    DamRow::Replace => t_days.clamp_min(MIN_T_DAYS) * SECONDS_PER_DAY,
                    // No floor: T >= 0 whenever T0 >= 0, and T = 0 is the
                    // channel row. No clamp op either, so no zeroed gradient.
                    DamRow::Additive => t_days * SECONDS_PER_DAY,
                }
            }
            (None, None) => unreachable!("a non-seasonal release precomputes t_constant"),
        }
    }

    /// `(T_{t+1}, T_t)` for the step producing routed column `t`: the
    /// residence time at the step's end (phase row `t`) and start (row
    /// `t − 1`), both from the closed form, so a window or test-phase chunk
    /// start needs no carried state. A constant release hands the same tensor
    /// for both.
    fn t_step_at(&self, t: usize) -> (Tensor<Autodiff<I>, 1>, Tensor<Autodiff<I>, 1>) {
        match &self.t_constant {
            Some(c) => (c.clone(), c.clone()),
            None => (self.t_seconds_at(t), self.t_seconds_at(t - 1)),
        }
    }
}

/// Per-dam accounting of the S28 discharge clamp on the dam rows, kept on the
/// inner backend (no tape, no host sync per step) over every step `forward`
/// routes while dam rows are armed (option C rows or a release).
///
/// # The water the clamp creates
///
/// A dam row is the trapezoid on its storage
/// `S = K·X·I + (K·(1 − X) + T)·Q (+ S0)`: on the additive row `K`, `X` are the
/// reach's own at this step and `T` the reservoir's; on the replace row `K = T`,
/// `X = 0`; with a rule curve `S0` moves by the flux, which is already off
/// `q'_eff`. For the pre-clamp solve `x` the step is exact:
///
/// ```text
/// S(I*_{t+1}, x) − S(I_t, Q_t) = dt·[(I_t + I*_{t+1})/2 + q'_eff − (Q_t + x)/2]
/// D = 2K(1 − X) + 2T_{t+1} + dt,   c4 = 2·dt/D
/// ```
///
/// (`I*_{t+1} = (N·x)_d`, the upstream rows' pre-clamp solve.) S28 then sets
/// `Q_{t+1} = max(x, lb)`. Where `x < lb`, with `δ = lb − x` and the dam row's
/// routed series `Q` as its outflow, the step's balance gains two terms:
///
/// ```text
/// storage  (K(1 − X) + T_{t+1})·δ = (dt/c4 − dt/2)·δ   S(·, lb) − S(·, x): forgiven storage
/// outflow  (dt/2)·δ                                    the routed series reports lb at the
///                                                      step end; the solve passed x < lb to
///                                                      the rows below (a negative outflow
///                                                      when x < 0)
/// created  = δ·dt/c4 = δ·D/2
/// ```
///
/// `δ·dt/c4` is also the lateral volume that would have held the row at `lb`:
/// `A = I − c1·N` is unit lower triangular and `q'_d` enters only row `d`, so
/// `∂x_d/∂q'_d = c4_d`, and `δ/c4` m³/s over `dt`. Summed over a window, the
/// dam row's volume balance with its routed series as the outflow closes
/// exactly (`tests/reservoir_rule_curve.rs`, the mass-balance tests):
///
/// ```text
/// Σ dt·(I_t + I_{t+1})/2 + Σ q'·dt − Σ r·dt − Σ dt·(Q_t + Q_{t+1})/2 − ΔS + created = 0
/// ```
///
/// with `r` the rule-curve flux and `ΔS` the change in `S` without `S0`.
/// `storage` alone closes it against the outflow the rows below received
/// (`(Q_t + x)/2` in the clamp step). Not counted here: a row below a clamped
/// dam re-reads its inflow as `lb` at the next step start, so its own
/// `K·X·I` wedge jumps by `K·X·δ`, and it may clamp itself (review v3,
/// finding 4); both show only in the network-wide negative-solve count.
///
/// Per step and dam row `d`, with `I_t = (N·Q_t)_d` the routed inflow at the
/// step start and `q'_d` the dam reach's lateral inflow after the floor (and
/// the sub-reach divisor) but BEFORE the rule-curve flux:
///
/// ```text
/// created_d     += max(lb − x_d, 0)·dt/c4_d                  m³
/// storage_d     += max(lb − x_d, 0)·(dt/c4_d − dt/2)         m³
/// inflow_d      += (I_t,d + q'_d)·dt                          m³  (the dam's inflow)
/// clamp_steps_d += 1[x_d < lb]
/// ```
///
/// With a rule curve armed it also keeps each step's `Qin_d = I_t,d + q'_d`
/// (m³/s), the detached inflow the rule-curve feasibility penalty compares
/// the flux against.
///
/// # The carried floor (`DamFloor::Carry`)
///
/// With `dam_floor: carry` each dam also carries an owed volume `owed_d`
/// (m³, >= 0), which makes the floor mass-conserving: the clamp still holds
/// the step at `lb`, but the water it created is paid back out of the dam's
/// own later inflow instead of staying in the river.
///
/// ```text
/// before the solve:  paid_d = min(owed_d, max(I_t,d + q'_d − lb, 0)·dt)     m³
///                    q'_eff,d −= paid_d/dt;  owed_d −= paid_d;  repaid_d += paid_d
/// after the solve:   owed_d += max(lb − x_d, 0)·dt/c4_d                     (= created this step)
/// ```
///
/// `owed_0 + created − repaid − owed = 0` at every step (`owed_0` the volume
/// carried in with `set_dam_owed`, 0 at a window start), so the dam row's
/// balance with its routed series as the outflow reads
/// `inflow + lateral − flux-stored − outflow − ΔS = owed_0 − owed`: the dam passes
/// on its inflow less its storage change, and only the still-unpaid volume
/// is extra. A repayment can itself take the next solve below the floor
/// (where `c3 < 0`, or when the inflow falls within the step); that clamp is
/// owed like any other. While a dam is in debt it passes all its inflow to
/// the repayment, so its law's flux drives the solve well below `lb` every
/// step, and `created` counts each of those steps.
///
/// What carry conserves is the dam row's ROUTED series. The rows below read
/// the pre-clamp `x` in a clamped step (their solve uses `N·x`), i.e. the
/// series less `(dt/2)·δ`; that below-floor part, `created − storage`, is
/// owed and repaid as well, so over a period the rows below receive the
/// dam's inflow less `ΔS` less `created − storage` (about `dt/D` of the
/// in-debt flux per step, `D/2 = K(1 − X) + T + dt/2`). Owing only the
/// storage part would conserve what the rows below receive instead, and
/// leave the dam's own series `created − storage` high; holding the dam's
/// in-step outflow at `lb` (a second solve with the lateral increment
/// `δ/c4`) would conserve both.
///
/// KNOWN FAILURE on the additive row (v4 re-evaluations, 2026-09-28): a debt
/// larger than the dam's storage pins its outflow at `lb`, where the
/// channel's Cunge `K`, `X` (evaluated at the dam's own `Q_t`) are days and
/// 0.5, so `c1` is about −0.5; every rising step of the upstream inflow then
/// clamps and owes about `K·X·ΔI`, while the wedge's release on falling
/// steps (`x > lb`) passes downstream instead of repaying. The debt pumps
/// itself up (replay L2: a dam owing 3.7x its 15-year inflow).
/// `.claude/skills/ddrs-dev/references/config.md` §Reservoirs, `dam_floor`.
/// `dam_row_positivity` (S19p in `mmc_op`) caps the dam row's wedge so
/// `c1 > 0`: in debt at the floor the solve then sits at `lb + c1·ΔI` with a
/// `c1` of order `δ·dt/D`, and the synthetic pump's new debt goes from
/// 1.95e7 m³ to 0 (`tests/reservoir_dam_positivity.rs`).
///
/// The owed state is on the inner backend, outside the autodiff tape: the
/// repayment is a constant cut to `q'`. Training therefore sees neither the
/// debt a flux incurs nor the later repayment as a consequence of `θ`, `T0`
/// or the routing parameters; the gradient through a clamped step stays
/// zero, as with `forgive`, and the repayment acts like a change to the
/// forcing. The feasibility penalty remains the only restoring gradient
/// against storing more than the dam receives.
struct DamAccount<I: Backend> {
    /// Dam rows, in the order the dams were armed.
    rows: Vec<usize>,
    rows_t: Tensor<I, 1, Int>,
    created: Tensor<I, 1>,
    storage: Tensor<I, 1>,
    inflow: Tensor<I, 1>,
    clamp_steps: Tensor<I, 1>,
    steps: usize,
    /// `Some` with a rule curve: `Qin` per routed step, `[n_dams]` each.
    qin: Option<Vec<Tensor<I, 1>>>,
    /// Carried floor: the volume each dam still owes, m³ (stays 0 with
    /// `forgive`).
    owed: Tensor<I, 1>,
    /// Carried floor: the volume each dam has paid back, m³.
    repaid: Tensor<I, 1>,
    /// The smallest `c1` the dam row routed with (`+∞` before any step).
    c1_min: Tensor<I, 1>,
    /// Steps at which the dam row's `c1` was negative.
    neg_c1_steps: Tensor<I, 1>,
}

impl<I: Backend> DamAccount<I> {
    fn new(rows: &[usize], record_qin: bool, device: &I::Device) -> Self {
        let n = rows.len();
        let rows_i: Vec<i64> = rows.iter().map(|&r| r as i64).collect();
        Self {
            rows: rows.to_vec(),
            rows_t: Tensor::from_data(TensorData::new(rows_i, [n]), device),
            created: Tensor::zeros([n], device),
            storage: Tensor::zeros([n], device),
            inflow: Tensor::zeros([n], device),
            clamp_steps: Tensor::zeros([n], device),
            steps: 0,
            qin: record_qin.then(Vec::new),
            owed: Tensor::zeros([n], device),
            repaid: Tensor::zeros([n], device),
            c1_min: Tensor::full([n], f32::INFINITY, device),
            neg_c1_steps: Tensor::zeros([n], device),
        }
    }

    /// Carried floor, before a solve: the repayment rate `paid/dt` per dam
    /// (m³/s, `[n_dams]`), taken out of the owed volume. `i_t` is the full
    /// `N·Q_t` (`[n]`), `q_prime` the lateral inflow after the floor and
    /// before the rule-curve flux, as [`Self::record`] reads them.
    fn repay(&mut self, i_t: Tensor<I, 1>, q_prime: Tensor<I, 1>, lb: f32, dt: f32) -> Tensor<I, 1> {
        let qin = i_t.select(0, self.rows_t.clone()) + q_prime.select(0, self.rows_t.clone());
        // In m³: `owed − paid` is exactly 0 where the whole debt is paid.
        let paid = self.owed.clone().min_pair((qin - lb).clamp_min(0.0) * dt);
        self.owed = self.owed.clone() - paid.clone();
        self.repaid = self.repaid.clone() + paid.clone();
        paid / dt
    }

    /// Add one routed step (see the struct docs). `carry`: the created volume
    /// is also owed.
    fn record(
        &mut self,
        diag: crate::routing::mmc_op::DamStepDiag<I>,
        q_prime: Tensor<I, 1>,
        lb: f32,
        dt: f32,
        carry: bool,
    ) {
        let x = diag.x_sol.select(0, self.rows_t.clone());
        let c4 = diag.c4.select(0, self.rows_t.clone());
        let c1 = diag.c1.select(0, self.rows_t.clone());
        self.c1_min = self.c1_min.clone().min_pair(c1.clone());
        self.neg_c1_steps = self.neg_c1_steps.clone() + c1.lower_elem(0.0).float();
        let qin = diag.i_t.select(0, self.rows_t.clone()) + q_prime.select(0, self.rows_t.clone());
        // δ = max(lb − x, 0) (m³/s) and dt/c4 = D/2 (s).
        let deficit = (-x.clone() + lb).clamp_min(0.0);
        let held = c4.recip() * dt;
        let created = deficit.clone() * held.clone();
        if carry {
            self.owed = self.owed.clone() + created.clone();
        }
        self.created = self.created.clone() + created;
        self.storage = self.storage.clone() + deficit * (held - 0.5 * dt);
        self.clamp_steps = self.clamp_steps.clone() + x.lower_elem(lb).float();
        self.inflow = self.inflow.clone() + qin.clone() * dt;
        self.steps += 1;
        if let Some(q) = self.qin.as_mut() {
            q.push(qin);
        }
    }
}

/// Host copy of the engine's per-dam clamp account
/// ([`MuskingumCunge::dam_account`]), summed over the routed steps.
#[derive(Clone, Debug, PartialEq)]
pub struct DamClampAccount {
    /// Dam rows of the network, in the order they were armed.
    pub rows: Vec<usize>,
    /// Water the S28 clamp created on each dam row, `Σ max(lb − x, 0)·dt/c4`,
    /// m³: the storage it forgave plus the below-floor outflow the pre-clamp
    /// solve passed down (see `DamAccount`).
    pub created_m3: Vec<f64>,
    /// The storage part of `created_m3`, `Σ max(lb − x, 0)·(dt/c4 − dt/2)`, m³.
    pub storage_m3: Vec<f64>,
    /// The dam's inflow, `Σ (I_t + q')·dt` (routed upstream inflow plus the
    /// reach's own lateral inflow, before any rule-curve flux), m³. Per dam:
    /// a dam below another counts the upper dam's outflow in its own inflow,
    /// so these do not sum to a network inflow.
    pub inflow_m3: Vec<f64>,
    /// Steps at which the dam row's solve fell below the floor.
    pub clamp_steps: Vec<u64>,
    /// Routed steps (the same for every dam).
    pub steps: u64,
    /// Carried floor (`DamFloor::Carry`): the volume each dam paid back out
    /// of its inflow, m³. 0 with `forgive`.
    pub repaid_m3: Vec<f64>,
    /// Carried floor: the volume each dam still owes after the last routed
    /// step, m³ (including any owed carried in with
    /// [`MuskingumCunge::set_dam_owed`]). 0 with `forgive`.
    pub owed_m3: Vec<f64>,
    /// The smallest `c1` each dam row routed with (`+∞` before any step).
    /// Negative on the additive row wherever the channel wedge
    /// `K_r·X_r > dt/2`; `dam_row_positivity` keeps it `> 0`.
    pub c1_min: Vec<f64>,
    /// Steps at which each dam row's `c1` was negative.
    pub neg_c1_steps: Vec<u64>,
}

impl DamClampAccount {
    /// `(Σ clamp steps, dam-row steps)` over the dams.
    pub fn clamp_share(&self) -> (u64, u64) {
        (self.clamp_steps.iter().sum(), self.steps * self.rows.len() as u64)
    }
}

/// Differentiable Muskingum-Cunge routing engine.
pub struct MuskingumCunge<I: Backend> {
    cfg: Config,

    n: Option<Tensor<Autodiff<I>, 1>>,
    q_spatial: Option<Tensor<Autodiff<I>, 1>>,
    p_spatial: Tensor<Autodiff<I>, 1>,
    /// Learned per-reach stage-roughness exponent, when the KAN emits it.
    gamma: Option<Tensor<Autodiff<I>, 1>>,
    length: Option<Tensor<Autodiff<I>, 1>>,
    slope: Option<Tensor<Autodiff<I>, 1>>,
    x_storage: Option<Tensor<Autodiff<I>, 1>>,
    /// Denormalized leakance params. All-or-nothing: when all three are `Some`,
    /// `route_timestep` dispatches to `timestep_forward_leakance`.
    k_d: Option<Tensor<Autodiff<I>, 1>>,
    d_gw: Option<Tensor<Autodiff<I>, 1>>,
    leakance_factor: Option<Tensor<Autodiff<I>, 1>>,
    /// Per-reach impervious hard-zero mask (inner backend, constant).
    /// Stored from `SpatialParameters::impervious_mask` at `setup_inputs`.
    impervious_mask: Option<Tensor<I, 1>>,
    /// Linear-reservoir rows (option C in `.claude/RESERVOIRS.md`): per-row
    /// dam mask and residence time `T` in seconds (inner backend, constant).
    /// Set by `set_reservoir_rows`; `None` keeps the timestep op
    /// byte-identical to the no-reservoir path.
    reservoir: Option<ReservoirTensors<I>>,
    /// Seasonal or learned dam release (`set_dam_release`). Exclusive with
    /// `reservoir`. `None` keeps the timestep op byte-identical.
    release: Option<ArmedRelease<I>>,
    /// Per-dam clamp account, `Some` whenever dam rows are armed
    /// (`set_reservoir_rows_as` / `set_dam_release`). Reads only.
    dam_account: Option<DamAccount<I>>,
    /// The last routed step's pre-clamp solve and routed inflow, handed from
    /// `route_timestep` to `forward`'s accounting.
    last_dam_diag: Option<crate::routing::mmc_op::DamStepDiag<I>>,
    /// Per-dam flood pool on armed dam rows ([`FloodPool`],
    /// `set_flood_pool`). `None` routes bitwise as before the pool existed.
    flood_pool: Option<ArmedFloodPool<I>>,
    /// What the S28 clamp does with the water it creates on a dam row
    /// ([`DamFloor`]; `Config::dam_floor` at construction). `Forgive` routes
    /// bitwise as before the option existed.
    dam_floor: DamFloor,
    /// Dam-row positivity on the additive row (S19p in `mmc_op`;
    /// `Config::dam_row_positivity` at construction). `false` routes bitwise
    /// as before the option existed.
    dam_row_positivity: bool,
    /// Network size cached for output shape / hot-start sizing. The dense
    /// `N` tensor is gone — all network use goes through `pattern`/`assembler`.
    n_segments: Option<usize>,

    /// CSR non-zero structure of `A = I − c·N`. Built once at `setup_inputs`,
    /// reused across timesteps. Index arrays only — no float values.
    /// `Arc` so the per-timestep autograd state is a refcount bump.
    pattern: Option<Arc<CsrPattern>>,
    /// Pre-uploaded constants for differentiable `A_values` assembly. Cached
    /// once per network so the per-timestep cost is gather + mul + add only.
    assembler: Option<AValuesAssembler<I>>,

    q_prime: Option<Tensor<Autodiff<I>, 2>>,
    /// Per-row lateral-inflow divisor `[n]`: the piece count of each row's
    /// parent reach, from `SparseAdjacency::parent_offset`. Built once at
    /// `setup_inputs`; `None` when the network is not subdivided (every
    /// divisor would be `1.0`), so the un-subdivided path skips the op and
    /// stays byte-identical.
    pieces_per_row: Option<Tensor<Autodiff<I>, 1>>,
    /// Whether the cold-start solve `(I − N)·Q_0 = q'_0` sees the SAME divided
    /// forcing `forward` routes. **Default `true`.** On an un-subdivided network
    /// there is no divisor, so this is an exact no-op there; under subdivision
    /// the undivided `q'_0` makes parent `p`'s outlet start at `m_p ×` its true
    /// steady state.
    ///
    /// Measured (2026-08-05, 1,841 CONUS gauges / 184,676 sub-reach rows,
    /// `max_pieces: 8`): the undivided cold start put 2.94× the correct total
    /// discharge into the network, and the A/B difference took **221 hourly
    /// steps to fall below 10 %** and 282 to fall below 5 % — against a
    /// configured `warmup` of 5 days = 120 steps, at which point it was still
    /// 41.7 %. The inflated state therefore leaks into the scored window, so it
    /// is divided by default. `false` reproduces the un-divided behaviour for
    /// `probe_courant --divide-hotstart` A/B runs.
    pub divide_hotstart_by_pieces: bool,
    discharge_t: Option<Tensor<Autodiff<I>, 1>>,

    /// Eval-time zeta accumulation (leakance diagnostics). Off by default;
    /// `enable_zeta_accumulation` turns it on. Sums live on the inner backend
    /// (no autograd tape) and grow by one elementwise add per timestep.
    collect_zeta: bool,

    /// Per-timestep negative-discharge tracking. Off by default;
    /// `enable_negative_discharge_tracking` turns it on. When off, the host
    /// sync in `forward_chain_inner` is skipped entirely — zero added cost.
    /// Enable in the training forward path; leave off in eval (the diagnostic
    /// is only meaningful during training where we want to observe the rate).
    track_negative_discharge: bool,
    zeta_abs_sum: Option<Tensor<I, 1>>,
    zeta_net_sum: Option<Tensor<I, 1>>,
    depth_sum: Option<Tensor<I, 1>>,
    area_z_sum: Option<Tensor<I, 1>>,
    q_sum: Option<Tensor<I, 1>>,
    zeta_steps: usize,

    dt: f32,
    device: I::Device,
    sparse_solver: SparseSolver,
}

/// Accumulated eval-time leakance diagnostics (inner backend, no tape).
/// All fields are per-reach sums over the accumulated timesteps; divide by
/// `steps` for eval-window means.
pub struct ZetaSumTensors<I: Backend> {
    /// Σ|zeta| (m³/s · steps).
    pub abs: Tensor<I, 1>,
    /// Σ zeta, signed (positive = losing reach).
    pub net: Tensor<I, 1>,
    /// Σ routed depth (m · steps).
    pub depth: Tensor<I, 1>,
    /// Σ plan-view wetted area `area_z` (m² · steps).
    pub area_z: Tensor<I, 1>,
    /// Σ routed discharge `q_next` (m³/s · steps).
    pub q: Tensor<I, 1>,
    /// Number of accumulated timesteps.
    pub steps: usize,
}

impl<I: Backend> MuskingumCunge<I> {
    pub fn new(cfg: Config, device: I::Device) -> Self {
        let sparse_solver = cfg.params.sparse_solver;
        let p_default = *cfg
            .params
            .defaults
            .get("p_spatial")
            .expect("cfg.params.defaults must contain p_spatial");
        let p_spatial = Tensor::<Autodiff<I>, 1>::from_floats([p_default], &device);
        let dam_floor = cfg.dam_floor();
        let dam_row_positivity = cfg.dam_row_positivity();
        Self {
            cfg,
            n: None,
            q_spatial: None,
            gamma: None,
            p_spatial,
            length: None,
            slope: None,
            x_storage: None,
            k_d: None,
            d_gw: None,
            leakance_factor: None,
            impervious_mask: None,
            reservoir: None,
            release: None,
            dam_account: None,
            last_dam_diag: None,
            flood_pool: None,
            dam_floor,
            dam_row_positivity,
            n_segments: None,
            pattern: None,
            assembler: None,
            q_prime: None,
            pieces_per_row: None,
            divide_hotstart_by_pieces: true,
            discharge_t: None,
            collect_zeta: false,
            track_negative_discharge: false,
            zeta_abs_sum: None,
            zeta_net_sum: None,
            depth_sum: None,
            area_z_sum: None,
            q_sum: None,
            zeta_steps: 0,
            dt: DT_SECONDS,
            device,
            sparse_solver,
        }
    }

    /// Bind static channel attributes, lateral inflows, and learned [0,1]
    /// parameters; build CSR pattern; denormalize; cold-start discharge.
    ///
    /// `initial_state`: optional window-start discharge `Q_0` (m³/s, per-reach,
    /// same order as the network). When `Some`, it replaces the hotstart
    /// heuristic — the injected value is used as-is (clamped to `discharge_lb`).
    /// When `None`, falls back to the existing `carry_state` / hotstart logic
    /// unchanged, so all existing callers that pass `None` are byte-identical to
    /// the pre-Task-3 code path.
    pub fn setup_inputs(
        &mut self,
        inputs: RoutingInputs<I>,
        streamflow: Tensor<Autodiff<I>, 2>,
        params: SpatialParameters<I>,
        carry_state: bool,
        initial_state: Option<Tensor<Autodiff<I>, 1>>,
    ) where
        I::FloatTensorPrimitive: 'static,
        I::Device: 'static,
    {
        let n = inputs.adjacency.n;

        // Upload per-reach channel attributes from the bundled SparseAdjacency.
        // length_m and slope live as plain Vec<f32> on disk and only need to
        // become Autodiff tensors at the solver boundary.
        let length = Tensor::<Autodiff<I>, 1>::from_floats(
            inputs.adjacency.length_m.as_slice(),
            &self.device,
        );
        let slope_min = self.cfg.params.attribute_minimums.slope;
        let slope = Tensor::<Autodiff<I>, 1>::from_floats(
            inputs.adjacency.slope.as_slice(),
            &self.device,
        )
        .clamp_min(slope_min);

        // Per-row lateral-inflow divisor, built once here rather than per
        // timestep. A reach split into `m` pieces of length `L/m` gives each
        // piece `q'/m` (see `forward`).
        self.pieces_per_row = inputs
            .adjacency
            .parent_offset
            .as_deref()
            .and_then(|off| pieces_per_row_divisor(off, n))
            .map(|d| Tensor::<Autodiff<I>, 1>::from_floats(d.as_slice(), &self.device));

        // Build CSR pattern + assembler constants directly from COO (O(nnz)).
        let pattern = Arc::new(CsrPattern::from_sparse(&inputs.adjacency));
        self.assembler = Some(AValuesAssembler::<I>::new(&pattern, &self.device));
        self.pattern = Some(pattern);

        self.n_segments = Some(n);
        self.length = Some(length);
        self.slope = Some(slope);
        self.x_storage = Some(inputs.x_storage);
        self.q_prime = Some(streamflow);

        let ranges = &self.cfg.params.parameter_ranges;
        let log_space = &self.cfg.params.log_space_parameters;
        self.n = Some(denormalize(
            params.n,
            ranges.n,
            log_space.iter().any(|s| s == "n"),
        ));
        self.q_spatial = Some(denormalize(
            params.q_spatial,
            ranges.q_spatial,
            log_space.iter().any(|s| s == "q_spatial"),
        ));
        // Stage-roughness exponent, when learned. Denormalised the same way as
        // every other KAN output; `parameter_ranges.gamma` sets the box.
        if let Some(g) = params.gamma {
            self.gamma = Some(denormalize(
                g,
                ranges.gamma,
                log_space.iter().any(|s| s == "gamma"),
            ));
        }
        if let Some(p) = params.p_spatial {
            self.p_spatial = denormalize(
                p,
                ranges.p_spatial,
                log_space.iter().any(|s| s == "p_spatial"),
            );
        }

        // Leakance params: denormalize when present, clear otherwise.
        self.k_d = params.k_d.map(|t| denormalize(t, ranges.k_d, log_space.iter().any(|s| s == "K_D")));
        self.d_gw = params.d_gw.map(|t| denormalize(t, ranges.d_gw, log_space.iter().any(|s| s == "d_gw")));
        self.leakance_factor = params.leakance_factor
            .map(|t| denormalize(t, ranges.leakance_factor, log_space.iter().any(|s| s == "leakance_factor")));
        // Impervious mask: constant, no denormalization — stored as-is.
        self.impervious_mask = params.impervious_mask;
        // Reservoir rows index the previous network, so they do not survive
        // a new one: re-arm with `set_reservoir_rows` after this call.
        self.reservoir = None;
        self.release = None;
        self.dam_account = None;
        self.flood_pool = None;

        match initial_state {
            Some(q0_ext) => {
                // Use the externally provided window-start state (state-cache path).
                // Clamp for numerical safety — same floor as the hotstart heuristic.
                self.discharge_t = Some(
                    q0_ext.clamp_min(self.cfg.params.attribute_minimums.discharge),
                );
            }
            None => {
                // No cache → existing carry_state / hotstart logic, byte-identical
                // to the pre-Task-3 code path.
                if !carry_state || self.discharge_t.is_none() {
                    let q_prime_0 = self
                        .q_prime
                        .as_ref()
                        .unwrap()
                        .clone()
                        .slice([0..1, 0..n])
                        .reshape([n]);
                    // Give the cold start the same `q'/m` lateral inflow
                    // `forward` routes. Without it a subdivided network starts
                    // ~m× too wet at every parent outlet and drains that
                    // surplus for ~220 hourly steps — far past `warmup`.
                    // `None` divisor (un-subdivided) ⇒ exact no-op.
                    let q_prime_0 = match (
                        self.divide_hotstart_by_pieces,
                        self.pieces_per_row.as_ref(),
                    ) {
                        (true, Some(d)) => q_prime_0 / d.clone(),
                        _ => q_prime_0,
                    };
                    // Hotstart: solve (I − N) · Q_0 = q'_0 via the same CSR solver
                    // with c = 1 (all-ones vector), then clamp.
                    let device = self.device.clone();
                    let ones: Tensor<Autodiff<I>, 1> = Tensor::ones([n], &device);
                    let pattern = self.pattern.as_ref().unwrap();
                    let assembler = self.assembler.as_ref().unwrap();
                    let a_values = assembler.assemble(ones);
                    let q0 = triangular_csr_solve::<I>(
                        pattern,
                        a_values,
                        q_prime_0,
                        self.sparse_solver == SparseSolver::Cuda,
                    )
                    .clamp_min(self.cfg.params.attribute_minimums.discharge);
                    self.discharge_t = Some(q0);
                }
            }
        }

        // SP-10: eagerly capture the per-timestep CUDA graph once, here at
        // setup time, so the per-step path can just replay it. Bypassed if
        // graphs aren't requested, on the CPU sparse path, or if the inner
        // backend isn't `Cuda<f32, i32>` (the CPU-fallback case: the user
        // requested cuda but is on NdArray — capture would TypeId-panic).
        if self.cfg.params.use_cuda_graphs
            && self.sparse_solver == SparseSolver::Cuda
            && crate::sparse::dispatch::backend_is_cuda::<I>()
        {
            self.try_capture_forward_graph();
        }
    }

    /// SP-10: extract inner-backend primitives from the autograd-tracked
    /// inputs and call into `cusparse::try_capture_forward`. Lives behind a
    /// `use_cuda_graphs` gate; default V1 config skips it entirely.
    fn try_capture_forward_graph(&self)
    where
        I::FloatTensorPrimitive: 'static,
        I::Device: 'static,
    {
        let into_inner = |t: Tensor<Autodiff<I>, 1>| -> I::FloatTensorPrimitive {
            match t.into_primitive() {
                TensorPrimitive::Float(p) => p.primitive,
                _ => unreachable!(),
            }
        };
        let pattern = self.pattern.as_ref().unwrap();
        // SAFETY: setup_inputs is the training thread's entry point; no
        // other thread has access to this pattern's cuda cache. The
        // returned &mut is valid for the duration of try_capture_forward.
        let cache =
            unsafe { crate::sparse::cusparse::ensure_cuda_cache_mut::<I>(pattern, &self.device) };
        let n_seg = self.n_segments.expect("n_segments set");
        crate::sparse::cusparse::try_capture_forward::<I>(
            cache,
            &self.cfg,
            pattern,
            into_inner(self.n.as_ref().unwrap().clone()),
            into_inner(self.q_spatial.as_ref().unwrap().clone()),
            into_inner(self.p_spatial_broadcast(n_seg)),
            into_inner(self.length.as_ref().unwrap().clone()),
            into_inner(self.slope.as_ref().unwrap().clone()),
            into_inner(self.x_storage.as_ref().unwrap().clone()),
            &self.device,
        );
    }

    /// Route `rows` as linear reservoirs with residence times `t_days` (days).
    ///
    /// Muskingum storage `S = K·[X·I + (1−X)·Q]` at `X = 0`, `K = T` is the
    /// linear reservoir `S = T·Q`, so on those rows only the timestep op
    /// replaces the K it computed with `T = t_days·86400 s` and the X with 0
    /// (S19'' in `mmc_op::forward_chain_inner`). Every other row is bitwise
    /// unchanged. `T` is prescribed data, not learned, so a dam row's `n`,
    /// `q_spatial` and `p_spatial` get exactly zero gradient. See
    /// `.claude/RESERVOIRS.md`, option C.
    ///
    /// Call AFTER [`Self::setup_inputs`]: `rows` index its network. Empty
    /// `rows` leaves the override off. Returns `Err`, changing nothing, on a
    /// length mismatch, a row `>= n_segments`, a duplicate row, or a `T` that
    /// is not finite or is below one hour (at `X = 0`, `c3 >= 0` needs
    /// `T >= dt/2`; one hour leaves margin).
    ///
    /// Only the plain timestep op carries the override: `route_timestep`
    /// panics if rows are set while leakance or the CUDA-graph path is active.
    pub fn set_reservoir_rows(&mut self, rows: &[usize], t_days: &[f32]) -> Result<(), String> {
        self.set_reservoir_rows_as(rows, t_days, DamRow::Replace)
    }

    /// [`Self::set_reservoir_rows`] with the dam-row form chosen. `Replace`
    /// is option C exactly (T >= 1 h). `Additive` keeps each dam reach's own
    /// Muskingum K and X and adds the reservoir storage `T·Q` to it
    /// (S19'''' in `mmc_op`; `crate::routing::release`), so `T >= 0` and
    /// `T = 0` routes the reach as the channel it is; the dam row's `n`,
    /// `q_spatial`, `p_spatial` keep their gradient through K and X.
    pub fn set_reservoir_rows_as(
        &mut self,
        rows: &[usize],
        t_days: &[f32],
        dam_row: DamRow,
    ) -> Result<(), String> {
        let n = self
            .n_segments
            .ok_or("set_reservoir_rows: call setup_inputs first")?;
        if self.release.is_some() {
            return Err(
                "set_reservoir_rows: a dam release is already set with set_dam_release; \
                 the two are exclusive"
                    .into(),
            );
        }
        if rows.len() != t_days.len() {
            return Err(format!(
                "set_reservoir_rows: {} rows but {} residence times",
                rows.len(),
                t_days.len()
            ));
        }
        let mut mask = vec![false; n];
        let mut t_seconds = vec![0.0_f32; n];
        for (&row, &t) in rows.iter().zip(t_days) {
            if row >= n {
                return Err(format!(
                    "set_reservoir_rows: row {row} is outside the {n}-reach network"
                ));
            }
            if mask[row] {
                return Err(format!("set_reservoir_rows: row {row} is listed twice"));
            }
            match dam_row {
                DamRow::Replace if !t.is_finite() || t < MIN_T_DAYS => {
                    return Err(format!(
                        "set_reservoir_rows: row {row} has T = {t} d; T must be finite and >= 1/24 d"
                    ));
                }
                DamRow::Additive if !t.is_finite() || t < 0.0 => {
                    return Err(format!(
                        "set_reservoir_rows: row {row} has T = {t} d; the additive dam row needs \
                         a finite T >= 0"
                    ));
                }
                _ => {}
            }
            mask[row] = true;
            t_seconds[row] = t * SECONDS_PER_DAY;
        }

        let additive = dam_row == DamRow::Additive;
        self.reservoir = (!rows.is_empty()).then(|| {
            let t_seconds: Tensor<I, 1> = Tensor::from_floats(t_seconds.as_slice(), &self.device);
            ReservoirTensors {
                mask: Tensor::from_data(TensorData::from(mask.as_slice()), &self.device),
                // Constant T: the Muskingum row at K = T, X = 0 already
                // conserves S = T·Q, so c3 stays as computed (bit-identical
                // option C). The additive row reads T at both ends (S19'''').
                t_prev_seconds: additive.then(|| t_seconds.clone()),
                t_seconds,
                additive,
                positivity: self.dam_row_positivity,
            }
        });
        self.dam_account = (!rows.is_empty()).then(|| DamAccount::new(rows, false, &self.device));
        // A pool indexes the previous dam rows: re-arm it after this call.
        self.flood_pool = None;
        Ok(())
    }

    /// Route `release.rows` as seasonal linear reservoirs (see [`DamRelease`]).
    ///
    /// Call AFTER [`Self::setup_inputs`]. Empty `rows` leaves the engine as
    /// with no release. Returns `Err`, changing nothing, when option C rows
    /// are already set (the two are exclusive), on a row `>= n_segments` or
    /// listed twice, a `T0`/`a`/`b` length that is not `rows.len()`, or a
    /// `phase` shorter than the lateral-inflow window. `T0`'s own values are
    /// not checked: on the replace row the clamp at one hour holds `T >= dt`
    /// whatever they are; the additive row takes `T` as it comes (the release
    /// head's `T0 > 0`, and `T = 0` is the channel row).
    ///
    /// Only the plain timestep op carries the release: `route_timestep`
    /// panics if it is set while leakance or the CUDA-graph path is active.
    pub fn set_dam_release(&mut self, release: DamRelease<I>) -> Result<(), String> {
        let n = self
            .n_segments
            .ok_or("set_dam_release: call setup_inputs first")?;
        if self.reservoir.is_some() {
            return Err(
                "set_dam_release: option C rows are already set with set_reservoir_rows; \
                 the two are exclusive"
                    .into(),
            );
        }
        let n_dams = release.rows.len();
        let mut mask = vec![false; n];
        for &row in &release.rows {
            if row >= n {
                return Err(format!("set_dam_release: row {row} is outside the {n}-reach network"));
            }
            if mask[row] {
                return Err(format!("set_dam_release: row {row} is listed twice"));
            }
            mask[row] = true;
        }
        if release.t0_days.dims()[0] != n_dams {
            return Err(format!(
                "set_dam_release: T0 has {} entries but there are {n_dams} dam rows",
                release.t0_days.dims()[0]
            ));
        }
        if let Some((a, b)) = &release.seasonal {
            if a.dims()[0] != n_dams || b.dims()[0] != n_dams {
                return Err(format!(
                    "set_dam_release: a has {} and b has {} entries but there are {n_dams} dam rows",
                    a.dims()[0],
                    b.dims()[0]
                ));
            }
        }
        let n_rows = self
            .q_prime
            .as_ref()
            .map(|q| q.dims()[0])
            .ok_or("set_dam_release: call setup_inputs first")?;
        if release.phase.len() < n_rows {
            return Err(format!(
                "set_dam_release: phase has {} rows but the window routes {n_rows}",
                release.phase.len()
            ));
        }
        if let Some(rc) = &release.rule_curve {
            if rc.coeffs.dims() != [n_dams, 4] || rc.inflow_mean.dims()[0] != n_dams {
                return Err(format!(
                    "set_dam_release: rule curve has coefficients {:?} and Ibar {:?}; \
                     want [{n_dams}, 4] and [{n_dams}]",
                    rc.coeffs.dims(),
                    rc.inflow_mean.dims()
                ));
            }
        }
        // A pool indexes the previous dam rows: re-arm it after this call.
        self.flood_pool = None;
        if n_dams == 0 {
            self.release = None;
            self.dam_account = None;
            return Ok(());
        }

        let rows_i32: Vec<i32> = release.rows.iter().map(|&r| r as i32).collect();
        let dam_row = release.dam_row;
        let t_constant = release.seasonal.is_none().then(|| match dam_row {
            DamRow::Replace => release.t0_days.clone().clamp_min(MIN_T_DAYS) * SECONDS_PER_DAY,
            DamRow::Additive => release.t0_days.clone() * SECONDS_PER_DAY,
        });
        // Rule curve: the per-step phase increments ΔH [n_rows − 1][4] (host
        // f64, `rule_curve_increments`) and Ibar/dt, once; `forward` builds
        // each step's (S0_{t+1} − S0_t)/dt = (Ibar/dt)·(c·ΔH) and takes it off
        // the dam rows' q', after the discharge floor on q' (a negative
        // effective lateral inflow is the reservoir storing more than its
        // reach adds).
        let rule_curve = release.rule_curve.as_ref().map(|rc| {
            let dh = crate::routing::release::rule_curve_increments(rc.phase0, n_rows);
            let rows_i64: Vec<i64> = release.rows.iter().map(|&r| r as i64).collect();
            ArmedRuleCurve {
                coeffs: rc.coeffs.clone(),
                scale: rc.inflow_mean.clone() / self.dt,
                dh: dh.chunks_exact(4).map(|c| [c[0], c[1], c[2], c[3]]).collect(),
                rows: Tensor::<Autodiff<I>, 1, Int>::from_data(
                    TensorData::new(rows_i64, [n_dams]),
                    &self.device,
                ),
            }
        });
        self.release = Some(ArmedRelease {
            rows: Tensor::from_data(TensorData::new(rows_i32, [n_dams]), &self.device),
            mask: Tensor::from_data(TensorData::from(mask.as_slice()), &self.device),
            t0_days: release.t0_days,
            seasonal: release.seasonal,
            phase: release.phase,
            dam_row,
            t_constant,
            t_step: None,
            rule_curve,
        });
        let record_qin = self.release.as_ref().is_some_and(|r| r.rule_curve.is_some());
        self.dam_account = Some(DamAccount::new(&release.rows, record_qin, &self.device));
        Ok(())
    }

    /// Arm the per-dam flood pool ([`FloodPool`]) on dam rows already armed
    /// with `set_reservoir_rows_as` or `set_dam_release` (call after them;
    /// re-arming the dams drops the pool). Every pool starts empty; see
    /// [`Self::set_flood_pool_state`] to carry one in. Empty `rows` leaves
    /// the engine without a pool.
    ///
    /// `Err`, changing nothing, without armed dam rows, on a row that is not
    /// an armed dam row or is listed twice, a parameter length that is not
    /// `rows.len()`, or an `Ibar` that is negative or not finite.
    pub fn set_flood_pool(&mut self, pool: FloodPool<I>) -> Result<(), String> {
        let dams = self
            .dam_account
            .as_ref()
            .ok_or("set_flood_pool: no dam rows are armed; call set_reservoir_rows_as / set_dam_release first")?;
        let n_pool = pool.rows.len();
        let mut seen = std::collections::HashSet::new();
        for &row in &pool.rows {
            if !dams.rows.contains(&row) {
                return Err(format!("set_flood_pool: row {row} is not an armed dam row"));
            }
            if !seen.insert(row) {
                return Err(format!("set_flood_pool: row {row} is listed twice"));
            }
        }
        for (name, len) in [
            ("kc", pool.kc.dims()[0]),
            ("phi", pool.phi.dims()[0]),
            ("z", pool.z_days.dims()[0]),
            ("Ibar", pool.inflow_mean.len()),
        ] {
            if len != n_pool {
                return Err(format!("set_flood_pool: {name} has {len} entries but there are {n_pool} pooled dams"));
            }
        }
        if let Some(v) = pool.inflow_mean.iter().find(|v| !(v.is_finite() && **v >= 0.0)) {
            return Err(format!("set_flood_pool: Ibar {v} must be finite and >= 0"));
        }
        if n_pool == 0 {
            self.flood_pool = None;
            return Ok(());
        }
        let device = self.device.clone();
        let pattern = self.pattern.as_ref().ok_or("set_flood_pool: call setup_inputs first")?;
        // Upstream edges of each pooled row, from the CSR pattern (diagonal
        // slots carry weight 0 and are skipped).
        let (mut src, mut dst, mut w) = (Vec::new(), Vec::new(), Vec::new());
        for (slot, &row) in pool.rows.iter().enumerate() {
            for k in pattern.crow[row] as usize..pattern.crow[row + 1] as usize {
                let col = pattern.col[k] as usize;
                if col != row && pattern.adj_values[k] != 0.0 {
                    src.push(col as i64);
                    dst.push(slot as i64);
                    w.push(pattern.adj_values[k]);
                }
            }
        }
        let idx = |v: Vec<i64>| {
            let n = v.len();
            Tensor::<Autodiff<I>, 1, Int>::from_data(TensorData::new(v, [n]), &device)
        };
        let upstream = (!src.is_empty()).then(|| {
            let weights = w
                .iter()
                .any(|&x| x != 1.0)
                .then(|| Tensor::<Autodiff<I>, 1>::from_floats(w.as_slice(), &device));
            (idx(src), idx(dst), weights)
        });
        let ibar = Tensor::<Autodiff<I>, 1>::from_floats(pool.inflow_mean.as_slice(), &device);
        let rows_i: Vec<i64> = pool.rows.iter().map(|&r| r as i64).collect();
        self.flood_pool = Some(ArmedFloodPool {
            rows: pool.rows.clone(),
            rows_ad: idx(rows_i),
            upstream,
            qc: pool.kc * ibar.clone(),
            phi: pool.phi,
            fmax: pool.z_days * (ibar * SECONDS_PER_DAY),
            f: Tensor::zeros([n_pool], &device),
            inflow_mean: pool.inflow_mean,
            captured: Tensor::zeros([n_pool], &device),
            evacuated: Tensor::zeros([n_pool], &device),
            f_peak: Tensor::zeros([n_pool], &device),
            inflow: Tensor::zeros([n_pool], &device),
            steps: 0,
        });
        Ok(())
    }

    /// Start the armed pools holding `f_m3` (m³, one per pooled dam in the
    /// armed order, [`FloodPoolAccount::rows`]) instead of empty, e.g. the
    /// pools at the end of the previous test-phase chunk. A constant (no
    /// gradient). Call after [`Self::set_flood_pool`], before `forward`.
    /// `Err`, changing nothing, without a pool, on a length mismatch, or on a
    /// value that is negative or not finite.
    pub fn set_flood_pool_state(&mut self, f_m3: &[f64]) -> Result<(), String> {
        let p = self.flood_pool.as_mut().ok_or("set_flood_pool_state: no flood pool is armed")?;
        if f_m3.len() != p.rows.len() {
            return Err(format!(
                "set_flood_pool_state: {} pool volumes for {} pooled dams",
                f_m3.len(),
                p.rows.len()
            ));
        }
        if let Some(v) = f_m3.iter().find(|v| !(v.is_finite() && **v >= 0.0)) {
            return Err(format!("set_flood_pool_state: pool volume {v} must be finite and >= 0"));
        }
        let v: Vec<f32> = f_m3.iter().map(|&v| v as f32).collect();
        p.f = Tensor::from_floats(v.as_slice(), &self.device);
        p.f_peak = Tensor::from_floats(v.as_slice(), &self.device);
        Ok(())
    }

    /// The pooled dams' rows, in the armed order; `None` without a pool.
    pub fn flood_pool_rows(&self) -> Option<&[usize]> {
        self.flood_pool.as_ref().map(|p| p.rows.as_slice())
    }

    /// The flood pools' volumes now, m³, on the autodiff tape (the pooled
    /// dams in the armed order); `None` without a pool. After `forward`, the
    /// window's closing pools.
    pub fn flood_pool_state(&self) -> Option<Tensor<Autodiff<I>, 1>> {
        self.flood_pool.as_ref().map(|p| p.f.clone())
    }

    /// The flood pool account ([`FloodPoolAccount`]) over every step
    /// `forward` has routed since the pool was armed; `None` without a pool.
    pub fn flood_pool_account(&self) -> Option<FloodPoolAccount> {
        let p = self.flood_pool.as_ref()?;
        let host = |t: Tensor<I, 1>| -> Vec<f64> {
            t.into_data().convert::<f32>().to_vec::<f32>().expect("f32").into_iter().map(f64::from).collect()
        };
        Some(FloodPoolAccount {
            rows: p.rows.clone(),
            captured_m3: host(p.captured.clone()),
            evacuated_m3: host(p.evacuated.clone()),
            f_end_m3: host(p.f.clone().inner()),
            f_peak_m3: host(p.f_peak.clone()),
            fmax_m3: host(p.fmax.clone().inner()),
            inflow_mean_m3s: p.inflow_mean.iter().map(|&v| v as f64).collect(),
            inflow_m3: host(p.inflow.clone()),
            steps: p.steps as u64,
        })
    }

    /// Muskingum-Cunge coefficients `(c1, c2, c3, c4)`. Direct port of
    /// `calculate_muskingum_coefficients`.
    pub fn calculate_muskingum_coefficients(
        &self,
        length: Tensor<Autodiff<I>, 1>,
        velocity: Tensor<Autodiff<I>, 1>,
        x_storage: Tensor<Autodiff<I>, 1>,
    ) -> (
        Tensor<Autodiff<I>, 1>,
        Tensor<Autodiff<I>, 1>,
        Tensor<Autodiff<I>, 1>,
        Tensor<Autodiff<I>, 1>,
    ) {
        let k = length / velocity;
        let one_minus_x = -x_storage.clone() + 1.0;
        let two_k = k.clone() * 2.0;
        let two_kx = two_k.clone() * x_storage;
        let two_k_1mx = two_k * one_minus_x;
        let denom = two_k_1mx.clone() + self.dt;

        let c1 = (-two_kx.clone() + self.dt) / denom.clone();
        let c2 = (two_kx + self.dt) / denom.clone();
        let c3 = (two_k_1mx - self.dt) / denom.clone();
        let c4 = denom.recip() * (2.0 * self.dt);
        (c1, c2, c3, c4)
    }

    /// Advance one timestep. Returns next-step discharge `Q_{t+1}` (shape `[n]`).
    pub fn route_timestep(&mut self, q_prime_clamp: Tensor<Autodiff<I>, 1>) -> Tensor<Autodiff<I>, 1>
    where
        I::FloatTensorPrimitive: 'static,
        I::Device: 'static,
    {
        let n = self.n.as_ref().unwrap().clone();
        let q_spatial = self.q_spatial.as_ref().unwrap().clone();
        let p_spatial = self.p_spatial_broadcast(self.n_segments.expect("setup_inputs not called"));
        let length = self.length.as_ref().unwrap().clone();
        let slope = self.slope.as_ref().unwrap().clone();
        let x_storage = self.x_storage.as_ref().unwrap().clone();
        let q_t = self.discharge_t.as_ref().unwrap().clone();
        let pattern = self.pattern.as_ref().unwrap();
        let assembler = self.assembler.as_ref().unwrap();

        // Leakance dispatch: when all three leakance params are present, use the
        // leakance op (never via CUDA graphs — leakance forces use_cuda_graphs=false).
        if let (Some(k_d), Some(d_gw), Some(leakance_factor)) = (
            self.k_d.as_ref().cloned(),
            self.d_gw.as_ref().cloned(),
            self.leakance_factor.as_ref().cloned(),
        ) {
            assert!(
                self.reservoir.is_none() && self.release.is_none(),
                "set_reservoir_rows / set_dam_release are not supported with leakance \
                 (params.use_leakance): the leakance op has no linear-reservoir K/X override"
            );
            let mut zeta_step: Option<crate::routing::mmc_op::ZetaStepDiag<I>> = None;
            let q_next = crate::routing::mmc_op::timestep_forward_leakance::<I>(
                &self.cfg, pattern, assembler,
                n, q_spatial, p_spatial,
                q_t, q_prime_clamp,
                length, slope, x_storage,
                k_d, d_gw, leakance_factor,
                self.impervious_mask.as_ref().cloned(),
                if self.collect_zeta { Some(&mut zeta_step) } else { None },
                self.track_negative_discharge,
                self.gamma.as_ref().cloned(),
            );
            if let Some(diag) = zeta_step {
                fn add<I: Backend>(slot: &mut Option<Tensor<I, 1>>, v: Tensor<I, 1>) {
                    *slot = Some(match slot.take() {
                        Some(s) => s + v,
                        None => v,
                    });
                }
                add(&mut self.zeta_abs_sum, diag.zeta.clone().abs());
                add(&mut self.zeta_net_sum, diag.zeta);
                add(&mut self.depth_sum, diag.depth);
                add(&mut self.area_z_sum, diag.area_z);
                add(&mut self.q_sum, q_next.clone().inner());
                self.zeta_steps += 1;
            }
            return q_next;
        }

        // SP-10: dispatch to the graph-replay path when graphs are on, we
        // are running on the CUDA sparse solver, AND the inner backend is
        // `Cuda<f32, i32>` (TypeId-gated to avoid the cuSPARSE/cubecl call
        // path on NdArray, which would panic in `compute_client`). The
        // replay function falls back to `timestep_forward` if capture
        // failed and no graph is installed on the cache.
        if self.cfg.params.use_cuda_graphs
            && self.sparse_solver == SparseSolver::Cuda
            && crate::sparse::dispatch::backend_is_cuda::<I>()
        {
            assert!(
                self.reservoir.is_none() && self.release.is_none(),
                "set_reservoir_rows / set_dam_release are not supported with use_cuda_graphs: \
                 the captured graph has no linear-reservoir K/X override"
            );
            crate::routing::mmc_op::timestep_forward_via_graph::<I>(
                &self.cfg, pattern, assembler,
                n, q_spatial, p_spatial,
                q_t, q_prime_clamp,
                length, slope, x_storage,
            )
        } else {
            crate::routing::mmc_op::timestep_forward_with_reservoirs::<I>(
                &self.cfg, pattern, assembler,
                n, q_spatial, p_spatial,
                q_t, q_prime_clamp,
                length, slope, x_storage,
                self.track_negative_discharge,
                self.gamma.as_ref().cloned(),
                self.reservoir.as_ref(),
                self.release.as_mut().map(|r| {
                    let (t_next, t_prev) = r.t_step.take().expect(
                        "a dam release is set but no step T was prepared; route through forward()",
                    );
                    ReleaseParent {
                        t_dams: t_next,
                        t_prev_dams: t_prev,
                        rows: r.rows.clone(),
                        mask: r.mask.clone(),
                        additive: r.dam_row == DamRow::Additive,
                        positivity: self.dam_row_positivity,
                    }
                }),
                self.dam_account.is_some().then_some(&mut self.last_dam_diag),
            )
        }
    }

    /// Forward over the full window. Output shape `[n, T]` (segment × time).
    pub fn forward(&mut self) -> Tensor<Autodiff<I>, 2> {
        let q_prime = self.q_prime.as_ref().unwrap().clone();
        let dims = q_prime.dims();
        let (num_timesteps, num_segments) = (dims[0], dims[1]);

        let discharge_lb = self.cfg.params.attribute_minimums.discharge;
        // Clamp once (single op + single tape node) instead of T times in-loop.
        let q_prime_clamped = q_prime.clamp_min(discharge_lb);
        // Split lateral inflow evenly along a subdivided reach: a piece of
        // length `L/m` receives `q'/m`. This is HEC-HMS's own treatment — its
        // lateral term is `C4·(q_L·Δx)` with `q_L` an inflow per unit length —
        // and it conserves each parent reach's total `q'` exactly, because the
        // pieces chain in series so the outlet piece still carries the whole
        // reach's runoff.
        //
        // MUST stay AFTER the clamp above. Clamping first applies the
        // `discharge_lb` floor once, to the parent's inflow; dividing first
        // would floor each of the `m` pieces independently, so a dry reach
        // would inject `m · discharge_lb` instead of `discharge_lb` — mass
        // created in proportion to the piece count.
        let q_prime_clamped = match self.pieces_per_row.as_ref() {
            Some(d) => q_prime_clamped / d.clone().unsqueeze_dim::<2>(0),
            None => q_prime_clamped,
        };
        let initial = self
            .discharge_t
            .as_ref()
            .unwrap()
            .clone()
            .clamp_min(discharge_lb);

        // SP-10 Phase 3: when CUDA graphs are active, the captured-graph
        // replay path in `route_timestep` causes cubecl's `Auto`-mode
        // memory pool to reclaim the slot underlying `initial` (the hot-
        // start Q0 tensor) before `Tensor::cat` reads column 0 at the end
        // of this function. Empirically the slot gets overwritten with
        // unrelated f32 bytes between the first route_timestep call and
        // the final cat. Force a host roundtrip on `initial` to detach
        // its data from any cubecl-pool slice that downstream replays
        // might recycle.
        //
        // This affects only the CUDA-graphs path; the default V1 path
        // does not exhibit the corruption (the BURN-chain forward holds
        // its own refs on every intermediate, so cubecl can't recycle
        // `initial`'s slot until `forward()` returns).
        //
        // Cost: a single host roundtrip on a [n] f32 tensor at setup.
        // Negligible compared to the per-timestep overhead the graphs path
        // saves.
        let initial = if self.cfg.params.use_cuda_graphs
            && self.sparse_solver == SparseSolver::Cuda
            && crate::sparse::dispatch::backend_is_cuda::<I>()
        {
            let data = initial.into_data();
            Tensor::<Autodiff<I>, 1>::from_data(data, &self.device)
        } else {
            initial
        };

        let mut columns: Vec<Tensor<Autodiff<I>, 2>> = Vec::with_capacity(num_timesteps);
        columns.push(initial.unsqueeze_dim::<2>(1));

        crate::routing::mmc_op::reset_negative_solve_stats();
        // The carried dam floor needs dam rows; without them it is a no-op.
        let carry = self.dam_floor == DamFloor::Carry && self.dam_account.is_some();

        for t in 1..num_timesteps {
            let q_prime_t: Tensor<Autodiff<I>, 1> = q_prime_clamped
                .clone()
                .slice([(t - 1)..t, 0..num_segments])
                .reshape([num_segments]);
            // The dam's own lateral inflow for the clamp account, before the
            // rule-curve flux below.
            let q_prime_pre = self.dam_account.is_some().then(|| q_prime_t.clone().inner());
            // The same, on the tape, for the flood pool's inflow.
            let q_prime_pool = self.flood_pool.is_some().then(|| q_prime_t.clone());
            // Dam release: T at this step's end (hour t) and start (hour
            // t − 1), for the storage-conserving dam row
            // (`crate::routing::release`, S19''' in `mmc_op`).
            if let Some(r) = self.release.as_mut() {
                r.t_step = Some(r.t_step_at(t));
            }
            // Rule curve: q'_eff = q' − (S0_{t+1} − S0_t)/dt on the dam rows
            // (`crate::routing::release`). A scatter-add of −flux onto the
            // dam rows only; every other entry is copied bit for bit.
            let q_prime_t = match self.release.as_ref().and_then(|r| r.rule_curve.as_ref()) {
                Some(rc) => q_prime_t.select_assign(0, rc.rows.clone(), -rc.flux_at(t - 1), IndexingUpdateOp::Add),
                None => q_prime_t,
            };
            // Carried floor (`DamFloor::Carry`): the dams repay what they owe
            // out of their inflow at the step start, `I_t + q'` (the `Qin` the
            // feasibility penalty reads; `I_t = N·Q_t` by the op's own SpMV),
            // before the solve. A detached constant on the dam rows' `q'`
            // (`DamAccount`); `forgive` skips this block.
            let q_prime_t = match (carry, self.dam_account.as_mut(), q_prime_pre.as_ref()) {
                (true, Some(acc), Some(q_pre)) => {
                    let q_t = match self.discharge_t.as_ref().expect("setup_inputs ran").clone().inner().into_primitive() {
                        TensorPrimitive::Float(p) => p,
                        _ => unreachable!("discharge is a float tensor"),
                    };
                    let pattern = self.pattern.as_ref().expect("setup_inputs ran");
                    let i_t = Tensor::<I, 1>::from_primitive(TensorPrimitive::Float(crate::sparse::spmv_primitive::<I>(
                        pattern,
                        q_t,
                        &self.device,
                        self.sparse_solver == SparseSolver::Cuda,
                        None,
                    )));
                    let rate = acc.repay(i_t, q_pre.clone(), discharge_lb, self.dt);
                    let rows = Tensor::<Autodiff<I>, 1, Int>::from_inner(acc.rows_t.clone());
                    q_prime_t.select_assign(0, rows, -Tensor::<Autodiff<I>, 1>::from_inner(rate), IndexingUpdateOp::Add)
                }
                _ => q_prime_t,
            };
            // Flood pool (law FA, `FloodPool`): capture above the release
            // target from the dam's step-start inflow `(N·Q_t)_d + q'_d`,
            // evacuate at the target, both on the dam row's q'. On the tape.
            let q_prime_t = match (self.flood_pool.as_mut(), q_prime_pool) {
                (Some(pool), Some(q_own)) => {
                    let q_t = self.discharge_t.as_ref().expect("setup_inputs ran").clone();
                    let rate = pool.step(q_t, q_own, self.dt);
                    q_prime_t.select_assign(0, pool.rows_ad.clone(), rate, IndexingUpdateOp::Add)
                }
                _ => q_prime_t,
            };
            let q_next = self.route_timestep(q_prime_t);
            if let (Some(acc), Some(q_pre)) = (self.dam_account.as_mut(), q_prime_pre) {
                let diag = self
                    .last_dam_diag
                    .take()
                    .expect("dam rows are armed, so the plain timestep op returned its diagnostics");
                acc.record(diag, q_pre, discharge_lb, self.dt, carry);
            }
            columns.push(q_next.clone().unsqueeze_dim::<2>(1));
            self.discharge_t = Some(q_next);
        }

        // Fix 1: distinguish three cases after the timestep loop.
        //
        // When `use_cuda_graphs` is true, `route_timestep` dispatches to
        // `timestep_forward_via_graph`, which replays an on-device CUDA graph
        // and never enters `forward_chain_inner`. Both counters stay at zero,
        // which is *indistinguishable* from "measured, found zero negatives" —
        // the silence would be a lie. Print an UNAVAILABLE notice so the
        // output can never be misread as a zero-negative measurement.
        //
        // The fallback inside `timestep_forward_via_graph` (capture failed →
        // direct launch) also carries this notice, because the user requested
        // graphs and we cannot distinguish "all replays succeeded" from "some
        // fell back silently". A config-based check is acceptable here because
        // the leakance guard already rejects `use_cuda_graphs + leakance` at
        // load time, so reaching this point with graphs on means the non-
        // leakance graph path was requested.
        let graphs_requested = self.cfg.params.use_cuda_graphs
            && self.sparse_solver == SparseSolver::Cuda
            && crate::sparse::dispatch::backend_is_cuda::<I>();

        if graphs_requested {
            if self.track_negative_discharge {
                eprintln!(
                    "  negative solves: UNAVAILABLE — use_cuda_graphs is true; \
                     the CUDA-graph path does not enter forward_chain_inner, so \
                     no count was taken. Disable use_cuda_graphs to measure."
                );
            }
        } else {
            let (neg, total) = crate::routing::mmc_op::negative_solve_stats();
            if total > 0 && neg > 0 {
                eprintln!(
                    "  reaches with negative discharges pre clamp: {neg}/{total} ({:.3}%)",
                    100.0 * neg as f64 / total as f64
                );
            }
            // The dam rows' clamp account (`dam_account`) is not logged here:
            // training logs it per mini-batch (`training::driver`), the test
            // phase per dam in `release_clamp.csv`.
        }

        Tensor::cat(columns, 1)
    }

    /// Turn on per-timestep zeta accumulation (leakance diagnostics). Only
    /// meaningful when leakance params are bound; otherwise `zeta_sums`
    /// stays `None`. Eval-time use — the training path never enables this.
    pub fn enable_zeta_accumulation(&mut self) {
        self.collect_zeta = true;
    }

    /// Turn on per-timestep negative-discharge tracking. When off (the
    /// default), the host sync in `forward_chain_inner` is skipped entirely
    /// — zero added cost on the forward. Enable in the training driver path
    /// (per-micro-batch) so the negative-solve rate is visible. The CUDA-graph
    /// path (`use_cuda_graphs: true`) never enters `forward_chain_inner`, so
    /// enabling this flag there has no effect; `forward` will print an
    /// UNAVAILABLE notice instead of a count. Training use only — the eval
    /// path never enables this.
    pub fn enable_negative_discharge_tracking(&mut self) {
        self.track_negative_discharge = true;
    }

    /// The per-dam clamp account ([`DamClampAccount`]) accumulated over every
    /// step `forward` has routed since the dam rows were armed; `None` when
    /// no dam rows are armed. Reads the device sums to the host (a few
    /// floats per dam).
    pub fn dam_account(&self) -> Option<DamClampAccount> {
        let a = self.dam_account.as_ref()?;
        let host = |t: &Tensor<I, 1>| -> Vec<f32> { t.clone().into_data().convert::<f32>().to_vec().expect("f32") };
        Some(DamClampAccount {
            rows: a.rows.clone(),
            created_m3: host(&a.created).into_iter().map(f64::from).collect(),
            storage_m3: host(&a.storage).into_iter().map(f64::from).collect(),
            inflow_m3: host(&a.inflow).into_iter().map(f64::from).collect(),
            clamp_steps: host(&a.clamp_steps).into_iter().map(|v| v.round() as u64).collect(),
            steps: a.steps as u64,
            repaid_m3: host(&a.repaid).into_iter().map(f64::from).collect(),
            owed_m3: host(&a.owed).into_iter().map(f64::from).collect(),
            c1_min: host(&a.c1_min).into_iter().map(f64::from).collect(),
            neg_c1_steps: host(&a.neg_c1_steps).into_iter().map(|v| v.round() as u64).collect(),
        })
    }

    /// The armed dam rows, in the order they were armed (the order of
    /// [`DamClampAccount`]'s vectors and of [`Self::set_dam_owed`]); `None`
    /// when no dam rows are armed.
    pub fn dam_rows(&self) -> Option<&[usize]> {
        self.dam_account.as_ref().map(|a| a.rows.as_slice())
    }

    /// The dam floor this engine routes with ([`DamFloor`]); from the config
    /// ([`Config::dam_floor`](crate::config::Config::dam_floor)) unless
    /// [`Self::set_dam_floor`] changed it.
    pub fn dam_floor(&self) -> DamFloor {
        self.dam_floor
    }

    /// Route with `floor` instead of the config's dam floor. Takes effect from
    /// the next [`Self::forward`]; any owed volume already carried is kept.
    pub fn set_dam_floor(&mut self, floor: DamFloor) {
        self.dam_floor = floor;
    }

    /// Whether the additive dam rows keep `c1 >= 0` (S19p in `mmc_op`); from
    /// the config ([`Config::dam_row_positivity`](crate::config::Config::dam_row_positivity))
    /// unless [`Self::set_dam_row_positivity`] changed it. Acts on the
    /// additive row only.
    pub fn dam_row_positivity(&self) -> bool {
        self.dam_row_positivity
    }

    /// Route the additive dam rows with (`true`) or without the positivity
    /// cap, instead of the config's setting. Takes effect from the next step,
    /// including on rows already armed.
    pub fn set_dam_row_positivity(&mut self, on: bool) {
        self.dam_row_positivity = on;
        if let Some(r) = self.reservoir.as_mut() {
            r.positivity = on;
        }
    }

    /// Carried floor: start the armed dams with `owed_m3` (m³, one per dam in
    /// the armed order, [`DamClampAccount::rows`]) instead of 0, e.g. the owed
    /// volume at the end of the previous test-phase chunk. Call after arming
    /// the dams (`set_reservoir_rows_as` / `set_dam_release`) and before
    /// `forward`. `Err`, changing nothing, without armed dams, on a length
    /// mismatch, or on a value that is negative or not finite.
    pub fn set_dam_owed(&mut self, owed_m3: &[f64]) -> Result<(), String> {
        let a = self
            .dam_account
            .as_mut()
            .ok_or("set_dam_owed: no dam rows are armed")?;
        if owed_m3.len() != a.rows.len() {
            return Err(format!(
                "set_dam_owed: {} owed volumes for {} armed dams",
                owed_m3.len(),
                a.rows.len()
            ));
        }
        if let Some(v) = owed_m3.iter().find(|v| !(v.is_finite() && **v >= 0.0)) {
            return Err(format!("set_dam_owed: owed volume {v} must be finite and >= 0"));
        }
        let v: Vec<f32> = owed_m3.iter().map(|&v| v as f32).collect();
        a.owed = Tensor::from_floats(v.as_slice(), &self.device);
        Ok(())
    }

    /// The dams' per-step inflow `Qin = I_t + q'` (m³/s, routed upstream
    /// inflow plus the reach's own lateral inflow, before the flux),
    /// `[n_steps, n_dams]` on the inner backend, detached: the rule-curve
    /// feasibility penalty's reference. `None` without a rule curve or before
    /// `forward`.
    pub fn dam_inflow_record(&self) -> Option<Tensor<I, 2>> {
        let q = self.dam_account.as_ref()?.qin.as_ref()?;
        if q.is_empty() {
            return None;
        }
        Some(Tensor::stack(q.clone(), 0))
    }

    /// The armed rule curve's per-step phase increments `ΔH_s` (seconds),
    /// row `s` for the step producing routed column `s + 1`; `None` without
    /// a rule curve. With `Ibar` and `c` they give the flux
    /// `(Ibar/dt)·(c·ΔH_s)` the engine took off the dam rows.
    pub fn rule_curve_increments(&self) -> Option<&[[f32; 4]]> {
        self.release.as_ref()?.rule_curve.as_ref().map(|rc| rc.dh.as_slice())
    }

    /// Eval-time leakance diagnostic sums accumulated across `route_timestep`
    /// calls since construction. `None` until the first accumulated step.
    pub fn zeta_sums(&self) -> Option<ZetaSumTensors<I>> {
        match (
            &self.zeta_abs_sum,
            &self.zeta_net_sum,
            &self.depth_sum,
            &self.area_z_sum,
            &self.q_sum,
        ) {
            (Some(a), Some(n), Some(d), Some(az), Some(q)) => Some(ZetaSumTensors {
                abs: a.clone(),
                net: n.clone(),
                depth: d.clone(),
                area_z: az.clone(),
                q: q.clone(),
                steps: self.zeta_steps,
            }),
            _ => None,
        }
    }

    pub fn discharge_state(&self) -> Option<Tensor<Autodiff<I>, 1>> {
        self.discharge_t.clone()
    }
    pub fn n(&self) -> Option<Tensor<Autodiff<I>, 1>> {
        self.n.clone()
    }
    pub fn q_spatial(&self) -> Option<Tensor<Autodiff<I>, 1>> {
        self.q_spatial.clone()
    }
    pub fn p_spatial(&self) -> Tensor<Autodiff<I>, 1> {
        self.p_spatial.clone()
    }
    pub fn pattern(&self) -> Option<&Arc<CsrPattern>> {
        self.pattern.as_ref()
    }

    /// Inner-backend snapshot of every input `forward_chain_inner` consumes,
    /// for the S18'/S19' Courant diagnostic in
    /// [`crate::routing::courant_probe`]. Diagnostic only: strips the tape,
    /// changes nothing. Call after `setup_inputs`.
    pub fn probe_inputs(&self) -> crate::routing::courant_probe::ProbeInputs<I> {
        let n_seg = self.n_segments.expect("setup_inputs not called");
        crate::routing::courant_probe::ProbeInputs {
            pattern: self.pattern.as_ref().expect("pattern").clone(),
            n: self.n.as_ref().expect("n").clone().inner(),
            q_spatial: self.q_spatial.as_ref().expect("q_spatial").clone().inner(),
            p_spatial: self.p_spatial_broadcast(n_seg).inner(),
            length: self.length.as_ref().expect("length").clone().inner(),
            slope: self.slope.as_ref().expect("slope").clone().inner(),
            x_storage: self.x_storage.as_ref().expect("x_storage").clone().inner(),
            q_prime: self.q_prime.as_ref().expect("q_prime").clone().inner(),
            pieces_per_row: self.pieces_per_row.as_ref().map(|d| d.clone().inner()),
            q0: self.discharge_t.as_ref().expect("discharge_t").clone().inner(),
            n_segments: n_seg,
        }
    }

    fn p_spatial_broadcast(&self, n: usize) -> Tensor<Autodiff<I>, 1> {
        let dims = self.p_spatial.dims();
        if dims[0] == n {
            self.p_spatial.clone()
        } else if dims[0] == 1 {
            let ones: Tensor<Autodiff<I>, 1> = Tensor::ones([n], &self.device);
            ones * self.p_spatial.clone().reshape([1]).slice([0..1])
        } else {
            panic!(
                "p_spatial length {} cannot broadcast to {} reaches",
                dims[0], n
            );
        }
    }
}

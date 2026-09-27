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

use crate::config::{Config, DamRow, SparseSolver};
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
            }
        });
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
        if n_dams == 0 {
            self.release = None;
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
            let dh = crate::routing::release::rule_curve_increments(&release.phase, n_rows);
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
        Ok(())
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
                    }
                }),
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

        for t in 1..num_timesteps {
            let q_prime_t: Tensor<Autodiff<I>, 1> = q_prime_clamped
                .clone()
                .slice([(t - 1)..t, 0..num_segments])
                .reshape([num_segments]);
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
            let q_next = self.route_timestep(q_prime_t);
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
            // Dam rows at the S28 discharge floor, whenever dams are armed
            // (the rule curve can ask a dam to store more than it holds).
            let (dam_hit, dam_total) = crate::routing::mmc_op::dam_clamp_stats();
            if dam_total > 0 {
                eprintln!(
                    "  dam-row steps at the discharge clamp: {dam_hit}/{dam_total} ({:.3}%)",
                    100.0 * dam_hit as f64 / dam_total as f64
                );
            }
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

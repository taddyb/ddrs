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
use burn::tensor::{backend::Backend, Tensor};

use burn::tensor::TensorPrimitive;

use crate::config::{Config, SparseSolver};
use crate::routing::utils::denormalize;
use crate::sparse::{triangular_csr_solve, AValuesAssembler, CsrPattern, SparseAdjacency};

/// Hardcoded routing timestep in seconds. Matches `self.t` in `mmc.py:192`.
pub const DT_SECONDS: f32 = 3600.0;

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
}

/// Differentiable Muskingum-Cunge routing engine.
pub struct MuskingumCunge<I: Backend> {
    cfg: Config,

    n: Option<Tensor<Autodiff<I>, 1>>,
    q_spatial: Option<Tensor<Autodiff<I>, 1>>,
    p_spatial: Tensor<Autodiff<I>, 1>,
    length: Option<Tensor<Autodiff<I>, 1>>,
    slope: Option<Tensor<Autodiff<I>, 1>>,
    x_storage: Option<Tensor<Autodiff<I>, 1>>,
    /// Denormalized leakance params. All-or-nothing: when all three are `Some`,
    /// `route_timestep` dispatches to `timestep_forward_leakance`.
    k_d: Option<Tensor<Autodiff<I>, 1>>,
    d_gw: Option<Tensor<Autodiff<I>, 1>>,
    leakance_factor: Option<Tensor<Autodiff<I>, 1>>,
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
    discharge_t: Option<Tensor<Autodiff<I>, 1>>,

    dt: f32,
    device: I::Device,
    sparse_solver: SparseSolver,
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
            p_spatial,
            length: None,
            slope: None,
            x_storage: None,
            k_d: None,
            d_gw: None,
            leakance_factor: None,
            n_segments: None,
            pattern: None,
            assembler: None,
            q_prime: None,
            discharge_t: None,
            dt: DT_SECONDS,
            device,
            sparse_solver,
        }
    }

    /// Bind static channel attributes, lateral inflows, and learned [0,1]
    /// parameters; build CSR pattern; denormalize; cold-start discharge.
    pub fn setup_inputs(
        &mut self,
        inputs: RoutingInputs<I>,
        streamflow: Tensor<Autodiff<I>, 2>,
        params: SpatialParameters<I>,
        carry_state: bool,
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

        if !carry_state || self.discharge_t.is_none() {
            let q_prime_0 = self
                .q_prime
                .as_ref()
                .unwrap()
                .clone()
                .slice([0..1, 0..n])
                .reshape([n]);
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
    pub fn route_timestep(&self, q_prime_clamp: Tensor<Autodiff<I>, 1>) -> Tensor<Autodiff<I>, 1>
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
            return crate::routing::mmc_op::timestep_forward_leakance::<I>(
                &self.cfg, pattern, assembler,
                n, q_spatial, p_spatial,
                q_t, q_prime_clamp,
                length, slope, x_storage,
                k_d, d_gw, leakance_factor,
            );
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
            crate::routing::mmc_op::timestep_forward_via_graph::<I>(
                &self.cfg, pattern, assembler,
                n, q_spatial, p_spatial,
                q_t, q_prime_clamp,
                length, slope, x_storage,
            )
        } else {
            crate::routing::mmc_op::timestep_forward::<I>(
                &self.cfg, pattern, assembler,
                n, q_spatial, p_spatial,
                q_t, q_prime_clamp,
                length, slope, x_storage,
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

        for t in 1..num_timesteps {
            let q_prime_t: Tensor<Autodiff<I>, 1> = q_prime_clamped
                .clone()
                .slice([(t - 1)..t, 0..num_segments])
                .reshape([num_segments]);
            let q_next = self.route_timestep(q_prime_t);
            columns.push(q_next.clone().unsqueeze_dim::<2>(1));
            self.discharge_t = Some(q_next);
        }

        Tensor::cat(columns, 1)
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

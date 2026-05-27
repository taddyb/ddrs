//! SP-10: persistent per-MC-instance scratch buffers.
//!
//! Allocated once during `setup_inputs`, dropped when `CudaPatternCache`
//! drops. All buffers are `[n × f32]`. Total: ~33n × 4 bytes (~540 KB for
//! n=5K gauge subgraph; ~45 MB for full CONUS n=346,321).

use burn::tensor::backend::Backend;
use burn_cubecl::cubecl::server::Handle;

/// Pre-allocated GPU buffers reused across every graph replay. Pointers are
/// stable for the lifetime of the cache.
pub struct PersistentScratch {
    pub n_segments: usize,

    // ── forward inputs / outputs ───────────────────────────────────────
    pub in_q: Handle,
    pub in_qp: Handle,
    pub out_q: Handle,

    // 24 saved-state outputs (forward) / inputs (backward).
    pub state_depth: Handle,
    pub state_top_width: Handle,
    pub state_side_slope: Handle,
    pub state_bottom_width: Handle,
    pub state_hydraulic_radius: Handle,
    pub state_velocity_unclamped: Handle,
    pub state_velocity_clamped: Handle,
    pub state_celerity: Handle,
    pub state_k_muskingum: Handle,
    pub state_denom: Handle,
    pub state_c1: Handle,
    pub state_c2: Handle,
    pub state_c3: Handle,
    pub state_c4: Handle,
    pub state_a_values: Handle,
    pub state_b_rhs: Handle,
    pub state_i_t: Handle,
    pub state_x_sol: Handle,
    pub state_ratio: Handle,
    pub state_denominator: Handle,
    pub state_q_eps: Handle,
    pub state_side_slope_raw: Handle,
    pub state_bw_raw: Handle,

    // ── backward inputs / outputs ──────────────────────────────────────
    pub in_grad_q_next: Handle,
    pub out_grad_n: Handle,
    pub out_grad_q_spatial: Handle,
    pub out_grad_p_spatial: Handle,
    pub out_grad_q_t: Handle,
    pub out_grad_q_prime_t: Handle,
}

impl PersistentScratch {
    pub fn allocate<B: Backend + 'static>(
        n_segments: usize,
        device: &B::Device,
    ) -> Self {
        let client = crate::sparse::cusparse::compute_client::<B>(device);
        let bytes = (n_segments * std::mem::size_of::<f32>()) as u64;
        let mk = || client.empty(bytes as usize);

        Self {
            n_segments,
            in_q: mk(), in_qp: mk(), out_q: mk(),
            state_depth: mk(), state_top_width: mk(), state_side_slope: mk(),
            state_bottom_width: mk(), state_hydraulic_radius: mk(),
            state_velocity_unclamped: mk(), state_velocity_clamped: mk(),
            state_celerity: mk(), state_k_muskingum: mk(), state_denom: mk(),
            state_c1: mk(), state_c2: mk(), state_c3: mk(), state_c4: mk(),
            state_a_values: mk(), state_b_rhs: mk(), state_i_t: mk(),
            state_x_sol: mk(), state_ratio: mk(), state_denominator: mk(),
            state_q_eps: mk(), state_side_slope_raw: mk(), state_bw_raw: mk(),
            in_grad_q_next: mk(),
            out_grad_n: mk(), out_grad_q_spatial: mk(), out_grad_p_spatial: mk(),
            out_grad_q_t: mk(), out_grad_q_prime_t: mk(),
        }
    }
}

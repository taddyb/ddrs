//! Bitwise pin of the CPU fused stage step, captured before Phase G.
//!
//! Phase G adds a CUDA path next to the host path. The host path must keep its
//! bits. This test hashes (FNV-1a 64 over the f32 bits) the outputs and the
//! gradients of `StageBlock::step` (the fused op) on `NdArray` with
//! `use_cuda: false`, for five (case, theta, substeps) rows. The hashes in
//! `GOLDEN` were captured at the Phase G base commit and must never be
//! recaptured on CI. If CI differs from a local run, gate the test on an env
//! var for local runs and record why.
//!
//! The case builders are copies of `tests/stage_op_parity.rs`.

mod common;

use std::sync::Arc;

use burn::tensor::Tensor;

use ddrs::config::DownstreamBoundary;
use ddrs::data::store::ChannelGeometryRows;
use ddrs::data::ConusAdjacencyStore;
use ddrs::routing::stage::{StageBlock, StageNetwork, StageParams};
use ddrs::sparse::{CsrPattern, SparseAdjacency};

use common::{linear_chain_sparse, InnerBackend, TestBackend, TestDevice};

// copied from tests/stage_op_parity.rs::{DEPTH_LB, tv, to_vec1, chain, table, H0, Case, Case::new, Case::prescribed, params, make_block, pick}
const DEPTH_LB: f32 = 0.01;

fn tv(v: &[f32], device: &TestDevice) -> Tensor<TestBackend, 1> {
    Tensor::<TestBackend, 1>::from_floats(v, device)
}

fn to_vec1<B: burn::tensor::backend::Backend>(t: Tensor<B, 1>) -> Vec<f32> {
    t.into_data().to_vec().unwrap()
}

fn chain(len: &[f32], slope: &[f32]) -> (SparseAdjacency, Arc<CsrPattern>) {
    let n = len.len();
    let mut dense = vec![0.0_f32; n * n];
    for i in 0..n - 1 {
        dense[(i + 1) * n + i] = 1.0;
    }
    let adj = SparseAdjacency::from_dense(n, &dense, len.to_vec(), slope.to_vec());
    let pat = Arc::new(CsrPattern::from_sparse(&adj));
    (adj, pat)
}

fn table(rows: &[usize], tw: f32, bed: &[f32], n: f32, pool: &[Option<f32>]) -> ChannelGeometryRows {
    ChannelGeometryRows {
        rows: rows.to_vec(),
        top_width: vec![tw; rows.len()],
        bed_z: bed.to_vec(),
        n: vec![n; rows.len()],
        pool_z: pool.to_vec(),
    }
}

enum H0 {
    /// Depth above `z_mid` of the block without a table.
    Depth(Vec<f32>),
    Abs(Vec<f32>),
}

struct Case {
    label: String,
    adj: SparseAdjacency,
    pattern: Arc<CsrPattern>,
    seeds: Option<Vec<usize>>,
    dams: Vec<(usize, f32)>,
    table: Option<(ChannelGeometryRows, bool)>,
    prescribed: bool,
    series_base: f32,
    series_rate: f32,
    theta_c: f32,
    h0: H0,
    n: Vec<f32>,
    p: Vec<f32>,
    q: Vec<f32>,
    q_t0: Vec<f32>,
    /// Per hour; shorter lists repeat their last entry.
    qp: Vec<Vec<f32>>,
    mc: Vec<Vec<f32>>,
    hours: usize,
}

impl Case {
    fn new(label: &str, adj: SparseAdjacency, pattern: Arc<CsrPattern>, h0: H0, q_t0: Vec<f32>) -> Self {
        let n = adj.n;
        Case {
            label: label.to_string(),
            adj,
            pattern,
            seeds: None,
            dams: vec![],
            table: None,
            prescribed: false,
            series_base: 0.0,
            series_rate: 0.0,
            theta_c: 0.5,
            h0,
            n: (0..n).map(|i| 0.03 + 0.005 * i as f32).collect(),
            q: (0..n).map(|i| 0.45 + 0.05 * (i % 3) as f32).collect(),
            p: (0..n).map(|i| 18.0 + 2.0 * (i % 4) as f32).collect(),
            q_t0,
            qp: vec![vec![1.0; n]],
            mc: vec![vec![0.0; n]],
            hours: 3,
        }
    }
    fn prescribed(mut self, base: f32, rate: f32) -> Self {
        self.prescribed = true;
        self.series_base = base;
        self.series_rate = rate;
        self
    }
}

fn params(case: &Case, theta: f32, substeps: u32) -> StageParams {
    StageParams {
        theta,
        theta_control: case.theta_c,
        substeps,
        boundary: if case.prescribed { DownstreamBoundary::PrescribedStage } else { DownstreamBoundary::NormalDepth },
        depth_lb: DEPTH_LB,
        bottom_width_lb: 0.1,
        dt: 3600.0,
        g: 9.81,
        use_cuda: false,
    }
}

fn make_block(case: &Case, theta: f32, substeps: u32, device: &TestDevice) -> StageBlock<InnerBackend> {
    let net = StageNetwork::build(&case.adj, &case.pattern, case.seeds.as_deref(), 0.0, &case.dams).unwrap();
    let z0 = net.z_mid.clone();
    let mut block = StageBlock::<InnerBackend>::new(net, params(case, theta, substeps), device);
    if case.prescribed {
        let series: Vec<f32> = (0..=case.hours).map(|k| case.series_base + case.series_rate * k as f32).collect();
        block.set_boundary_series(series).unwrap();
    }
    if let Some((rows, strict)) = &case.table {
        block.set_table_geometry(rows.clone(), *strict).unwrap();
    }
    let h0 = match &case.h0 {
        H0::Depth(d) => (0..case.adj.n).map(|i| z0[i] + d[i]).collect::<Vec<_>>(),
        H0::Abs(h) => h.clone(),
    };
    block.set_stage(tv(&h0, device));
    block
}

fn pick<T: Clone>(v: &[Vec<T>], k: usize) -> Vec<T> {
    v[k.min(v.len() - 1)].clone()
}

// copied from tests/stage_op_parity.rs::chain_state_cases
fn chain_state_cases() -> Vec<Case> {
    let mut cases = vec![];
    // gradcheck_through_one_step
    {
        let adj = linear_chain_sparse(5);
        let pattern = Arc::new(CsrPattern::from_sparse(&adj));
        let net = StageNetwork::build(&adj, &pattern, None, 0.0, &[]).unwrap();
        let h0: Vec<f32> =
            (0..5).map(|i| net.z_mid[i] + 0.4 + 0.07 * (i as f32) + 0.03 * ((i * i) as f32 % 3.0)).collect();
        let mut c = Case::new("one_step", adj, pattern, H0::Abs(h0), vec![1.1, 2.3, 3.2, 4.6, 5.1]);
        c.n = vec![0.03, 0.035, 0.04, 0.045, 0.05];
        c.q = vec![0.45, 0.5, 0.55, 0.5, 0.6];
        c.p = vec![18.0, 21.0, 24.0, 20.0, 22.0];
        cases.push(c);
    }
    // reversed edge and negative flow
    {
        let (adj, pattern) = chain(&[1000.0; 5], &[2e-4; 5]);
        cases.push(Case::new(
            "reversed_edge",
            adj,
            pattern,
            H0::Depth(vec![0.3, 0.7, 0.6, 0.5, 0.5]),
            vec![0.8, -1.2, 1.5, 1.8, 2.0],
        ));
    }
    // dam control edge on row 2, T 1.23 d
    {
        let (adj, pattern) = chain(&[1000.0; 5], &[1e-3; 5]);
        let mut c = Case::new(
            "dam_edge",
            adj,
            pattern,
            H0::Depth(vec![0.45, 0.55, 0.8, 0.6, 0.7]),
            vec![1.1, 2.3, 3.2, 4.6, 5.1],
        );
        c.dams = vec![(2, 1.23)];
        cases.push(c);
    }
    // adverse flow: the b clamp binds
    {
        let (adj, pattern) = chain(&[1000.0; 3], &[1e-3; 3]);
        cases.push(Case::new(
            "adverse_flow",
            adj,
            pattern,
            H0::Depth(vec![0.3, 1.5, 1.0]),
            vec![10.0, 5.0, 6.0],
        ));
    }
    // Q = 0 exactly, flat bed, constant prescribed series
    {
        let (adj, pattern) = chain(&[1000.0; 3], &[0.0; 3]);
        let mut c = Case::new("zero_flow", adj, pattern, H0::Abs(vec![1.0; 3]), vec![0.0; 3]).prescribed(1.0, 0.0);
        c.qp = vec![vec![0.0; 3]];
        cases.push(c);
    }
    // the depth floor binds
    {
        let (adj, pattern) = chain(&[1000.0; 3], &[1e-3; 3]);
        let net = StageNetwork::build(&adj, &pattern, None, 0.0, &[]).unwrap();
        let h0: Vec<f32> = net.z_mid.iter().map(|z| z + 0.05).collect();
        let mut c = Case::new("depth_floor", adj, pattern, H0::Abs(h0), vec![0.0; 3]).prescribed(-5.0, 0.0);
        c.qp = vec![vec![0.0; 3]];
        c.hours = 6;
        cases.push(c);
    }
    cases
}

// copied from tests/stage_op_parity.rs::table_and_mixed_cases
fn table_and_mixed_cases() -> Vec<Case> {
    let mut cases = vec![];
    // table_rows_use_the_rectangle_and_fixed_n
    {
        let (adj, pattern) = chain(&[1000.0; 3], &[1e-3; 3]);
        let net = StageNetwork::build(&adj, &pattern, None, 0.0, &[]).unwrap();
        let mut h0: Vec<f32> = net.z_mid.iter().map(|z| z + 1.0).collect();
        h0[2] = 1.5;
        let mut c = Case::new("table_rect", adj, pattern, H0::Abs(h0), vec![3.0, 5.0, 7.0]).prescribed(-5.0, 0.0);
        c.table = Some((table(&[2], 80.0, &[0.3], 0.02, &[None]), false));
        c.qp = vec![vec![0.5, 0.4, 0.3]];
        cases.push(c);
    }
    // pool_z_sets_the_release_reference: stage 1 m and 3 m above the bed
    for above in [1.0f32, 3.0] {
        let (adj, pattern) = chain(&[1000.0; 3], &[1e-3; 3]);
        let mut c = Case::new(
            &format!("pool_dam_{above}m"),
            adj,
            pattern,
            H0::Abs(vec![10.0 + above, 12.0, 12.0]),
            vec![0.0; 3],
        )
        .prescribed(-5.0, 0.0);
        c.dams = vec![(0, 1.5)];
        c.theta_c = 1.0;
        c.table = Some((table(&[0, 1, 2], 400.0, &[10.0, 10.0, 10.0], 0.03, &[Some(12.0), None, None]), true));
        c.qp = vec![vec![0.0; 3]];
        cases.push(c);
    }
    // mixed geometry: the table lists row 2 only, the head serves rows 0 and 1
    {
        let (adj, pattern) = chain(&[1000.0; 3], &[1e-3; 3]);
        let net = StageNetwork::build(&adj, &pattern, None, 0.0, &[]).unwrap();
        let z2 = net.z_mid[2];
        let mut c = Case::new("mixed", adj, pattern, H0::Depth(vec![1.0; 3]), vec![3.0, 5.0, 7.0]).prescribed(-5.0, 0.0);
        c.table = Some((table(&[2], 80.0, &[z2], 0.02, &[None]), false));
        c.qp = vec![vec![0.5, 0.4, 0.3]];
        cases.push(c);
    }
    cases
}

// copied from tests/stage_op_parity.rs::prescribed_boundary_cases
fn prescribed_boundary_cases() -> Vec<Case> {
    let mut cases = vec![];
    // flow_reversal_is_not_clamped: the outlet stage rises 1 m per hour
    {
        let (adj, pattern) = chain(&[1000.0; 2], &[1e-3; 2]);
        let net = StageNetwork::build(&adj, &pattern, None, 0.0, &[]).unwrap();
        let h: Vec<f32> = net.z_mid.iter().map(|z| z + 0.5).collect();
        let base = h[1];
        let mut c = Case::new("flow_reversal", adj, pattern, H0::Abs(h), vec![0.0; 2]).prescribed(base, 1.0);
        c.qp = vec![vec![0.0; 2]];
        cases.push(c);
    }
    // a 3-chain with a series rising 0.5 m per hour
    {
        let (adj, pattern) = chain(&[1000.0; 3], &[1e-3; 3]);
        let net = StageNetwork::build(&adj, &pattern, None, 0.0, &[]).unwrap();
        let h: Vec<f32> = net.z_mid.iter().map(|z| z + 0.8).collect();
        let base = h[2];
        let c = Case::new("rising_series", adj, pattern, H0::Abs(h), vec![2.0, 3.0, 4.0]).prescribed(base, 0.5);
        cases.push(c);
    }
    cases
}


// copied from tests/stage_op_parity.rs::juniata_case, with a closed-form start stage instead of hotstart
// (hotstart calls src/geometry.rs, which another branch may change).
fn juniata_case(_device: &TestDevice) -> Case {
    let store = ConusAdjacencyStore::open("examples/juniata/data/juniata_conus_adjacency.zarr")
        .expect("open bundle conus adjacency");
    let adj = SparseAdjacency {
        n: store.n,
        rows: store.indices_0.clone(),
        cols: store.indices_1.clone(),
        values: vec![1.0; store.nnz],
        length_m: store.length_m.to_vec(),
        slope: store.slope.to_vec(),
        parent_offset: None,
    };
    let pattern = Arc::new(CsrPattern::from_sparse(&adj));
    let n = adj.n;
    // cumulative upstream sum of q' = 1
    let mut q_init = vec![1.0f32; n];
    let mut down = vec![usize::MAX; n];
    for (&r, &c) in adj.rows.iter().zip(&adj.cols) {
        down[c as usize] = r as usize;
    }
    for i in 0..n {
        if down[i] != usize::MAX {
            let v = q_init[i];
            q_init[down[i]] += v;
        }
    }
    // Closed-form start stage: no hotstart, so no call into src/geometry.rs.
    let net = StageNetwork::build(&adj, &pattern, None, 0.0, &[]).unwrap();
    let h0: Vec<f32> = (0..n).map(|i| net.z_mid[i] + 0.3 + 0.002 * (i % 50) as f32).collect();
    let mut case = Case::new("juniata", adj, pattern, H0::Abs(h0), q_init);
    case.n = vec![0.035; n];
    case.p = vec![21.0; n];
    case.q = vec![0.5; n];
    case.qp = vec![vec![1.0; n]];
    case.mc = vec![vec![0.0; n]];
    case.hours = 6;
    case
}

// copied from tests/stage_op_parity.rs::{Rng, vec_b, weights}
/// Copy of the generator of `tests/tree_lu_gradcheck.rs` (`Rng`, `vec_b`); kept as a copy on purpose.
struct Rng(u64);
impl Rng {
    fn unit(&mut self) -> f64 {
        self.0 = self.0.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        ((z ^ (z >> 31)) >> 11) as f64 / (1u64 << 53) as f64
    }
    fn range(&mut self, lo: f64, hi: f64) -> f64 {
        lo + (hi - lo) * self.unit()
    }
}

fn vec_b(n: usize, seed: u64, lo: f64, hi: f64) -> Vec<f32> {
    let mut rng = Rng(seed);
    (0..n).map(|_| rng.range(lo, hi) as f32).collect()
}

/// Loss weights, uniform in [0.5, 1.5].
fn weights(seed: u64, len: usize) -> Vec<f32> {
    vec_b(len, seed, 0.5, 1.5)
}

/// FNV-1a 64 over the bits of every f32, in order.
fn fnv(bits: impl Iterator<Item = u32>) -> u64 {
    let mut h: u64 = 0xcbf29ce484222325;
    for b in bits {
        for byte in b.to_le_bytes() {
            h ^= byte as u64;
            h = h.wrapping_mul(0x100000001b3);
        }
    }
    h
}

/// (case label, theta, substeps, hash). First captured at the worktree's base commit before any Phase G
/// change. Re-pinned 2026-11-01 (user ruling): the CPU forward now fuses every `a*b + c` site
/// like the CUDA kernels do (`f32::mul_add`), so the forward bits and the gradients computed from them change.
/// Old values: juniata 1/1 0x2ae92b67fad3d5e1, juniata 0.6/4 0x22e3e3b39497095d, table_rect 0.6/1
/// 0x3eb5586bb6085ac2, flow_reversal 0.55/4 0x041ea849e871639d, dam_edge 1/1 0x030531fdfcc73026.
/// Re-pinned again when the forward solve moved to the cuSPARSE reciprocal form (`tree_lu.rs`).
/// Values before that change: juniata 1/1 0x299ce4709ea656c3, juniata 0.6/4 0xf59c6bf6f60e8ed7, table_rect 0.6/1
/// 0x36234345f6cd664b, flow_reversal 0.55/4 0x33bd0272895f5336, dam_edge 1/1 0x088f4ef3393a5f3b.
/// Re-pinned 2026-11-02: the CPU adjoint of the readout, RHS and matrix blocks now fuses the same sites as
/// the CUDA backward kernels. Values before that change: juniata 0.6/4 0xc92e200f90dc88e7, table_rect 0.6/1
/// 0x0422c60a9e076e5d, flow_reversal 0.55/4 0x61f847bd464e6d11 (juniata 1/1 and dam_edge 1/1 did not change).
/// Re-pinned again 2026-11-02: the CPU adjoint of the section, Newton, control, depth, assemble-glue
/// and prologue blocks now fuses the same sites as the CUDA backward kernels. Values before that change:
/// juniata 1/1 0x17261ad23acc08a9, juniata 0.6/4 0x1fde63b5606e2832, table_rect 0.6/1 0x431b2de1e30bc9e0,
/// flow_reversal 0.55/4 0xe78b0648c482edb3, dam_edge 1/1 0xd7c87cb2dfed1762.
const GOLDEN: &[(&str, f32, u32, u64)] = &[
    ("juniata", 1.0, 1, 0x11f8076cfa233298),
    ("juniata", 0.6, 4, 0xfa72cab1155ba8d6),
    ("table_rect", 0.6, 1, 0xccb63562455dd578),
    ("flow_reversal", 0.55, 4, 0xa82c2cc3d74bc201),
    ("dam_edge", 1.0, 1, 0xaae366862eb36ca5),
];

fn case_for(label: &str, device: &TestDevice) -> Case {
    let mut c = match label {
        "juniata" => juniata_case(device),
        "table_rect" => table_and_mixed_cases().remove(0),
        "flow_reversal" => prescribed_boundary_cases().remove(0),
        "dam_edge" => chain_state_cases().remove(2),
        other => panic!("unknown golden case {other}"),
    };
    assert_eq!(c.label, label);
    c.hours = if label == "juniata" { 6 } else { 3 };
    c
}

/// Hash of one run: every hour's q_next, the final stage, the five mass sums,
/// the step count, then the gradients of the seven parents.
fn run_hash(case: &Case, theta: f32, substeps: u32, device: &TestDevice) -> u64 {
    let n = case.adj.n;
    let mut block = make_block(case, theta, substeps, device);
    let z_mid = block.op_consts(&case.pattern).z_mid.clone();
    // copied from tests/stage_op_parity.rs::run_grads_part (every parent a leaf)
    let lf = |v: &[f32]| tv(v, device).require_grad();
    let h0 = to_vec1(block.stage().unwrap());
    let h_leaf = lf(&h0);
    block.set_stage(h_leaf.clone());
    let (nn, pp, qq) = (lf(&case.n), lf(&case.p), lf(&case.q));
    let qt_leaf = lf(&case.q_t0);
    let mut q_t = qt_leaf.clone();
    let (mut qp_l, mut mc_l) = (vec![], vec![]);
    let mut loss: Option<Tensor<TestBackend, 1>> = None;
    let mut bits: Vec<u32> = vec![];
    for k in 0..case.hours {
        let qpl = lf(&pick(&case.qp, k));
        let mcl = lf(&pick(&case.mc, k));
        qp_l.push(qpl.clone());
        mc_l.push(mcl.clone());
        let out = block.step(&case.pattern, nn.clone(), pp.clone(), qq.clone(), q_t.clone(), mcl, qpl);
        bits.extend(to_vec1(out.clone()).iter().map(|v| v.to_bits()));
        let wq = weights(1 + k as u64, n);
        let term = (out.clone() * tv(&wq, device)).sum();
        loss = Some(match loss {
            None => term,
            Some(l) => l + term,
        });
        q_t = out;
    }
    let h_end = block.stage().unwrap();
    bits.extend(to_vec1(h_end.clone()).iter().map(|v| v.to_bits()));
    let m = block.mass_balance();
    bits.extend(
        [m.storage_change_m3, m.net_flux_m3, m.floor_created_m3, m.gross_flux_m3, m.mc_inflow_m3]
            .iter()
            .map(|v| v.to_bits()),
    );
    bits.push(m.steps as u32);
    let off = h_end - tv(&z_mid, device);
    let wv = weights(9999, n);
    let loss = loss.unwrap() + (off * tv(&wv, device)).sum();
    let g = loss.backward();
    let get = |t: &Tensor<TestBackend, 1>, what: &str| -> Vec<f32> {
        to_vec1(t.grad(&g).unwrap_or_else(|| panic!("no gradient for {what}")))
    };
    for (t, what) in [(&nn, "n"), (&qq, "q"), (&pp, "p"), (&h_leaf, "h_t"), (&qt_leaf, "q_t")] {
        bits.extend(get(t, what).iter().map(|v| v.to_bits()));
    }
    for t in &qp_l {
        bits.extend(get(t, "q_prime").iter().map(|v| v.to_bits()));
    }
    for t in &mc_l {
        bits.extend(get(t, "q_next_mc").iter().map(|v| v.to_bits()));
    }
    fnv(bits.into_iter())
}

#[test]
fn cpu_fused_step_is_bitwise_pinned() {
    let device = TestDevice::default();
    let mut bad = vec![];
    for &(label, theta, substeps, want) in GOLDEN {
        let case = case_for(label, &device);
        let got = run_hash(&case, theta, substeps, &device);
        println!("{label} {theta} {substeps} {got:#018x}");
        if got != want {
            bad.push(format!("{label} {theta} {substeps}: got {got:#018x}, want {want:#018x}"));
        }
    }
    assert!(bad.is_empty(), "CPU fused step bits changed:\n{}", bad.join("\n"));
}

//! Parity of the fused stage op against `StageBlock::step_reference`, the
//! definition, on the states of the stage tests (`stage_step.rs`).
//!
//! Forward: `StageBlock::step_fused` hour by hour. Bars (overview
//! 16.6): per hour and per field `max|f - r| <= 1e-6 max|r|`; the five mass
//! sums within `1e-5 * gross_flux_m3`; `steps` equal. Each family prints
//! `bitwise: true|false`.
//!
//! Gradients: all seven parents tracked, against Burn autodiff of
//! the reference. `gradient_parity_hour1_strict` holds one-hour runs to the
//! strict bar with no allowance. The multi-hour `gradient_parity_*` tests use
//! the strict bar plus a bounded noise allowance (see `grad_pair`).
//! `fused_op_gradcheck_by_central_differences` checks the op by central
//! differences; `fused_op_untracked_matches_tracked` checks the untracked branch.
//!
//! Timing: every test here runs in a debug build in under 60 s, except
//! `forward_parity_juniata_48h` and `gradient_parity_juniata_48h`, which skip
//! themselves in debug builds (the `cfg!(debug_assertions)` rule of
//! `tests/juniata_acceptance.rs`). `gradient_parity_hour1_strict` includes the
//! Juniata case only in release builds. Run them with
//! `cargo test --release --test stage_op_parity`.

mod common;

use std::sync::Arc;

use burn::tensor::Tensor;

use ddrs::config::DownstreamBoundary;
use ddrs::data::store::ChannelGeometryRows;
use ddrs::data::ConusAdjacencyStore;
use ddrs::routing::stage::{StageBlock, StageNetwork, StageParams};
use ddrs::routing::{MuskingumCunge, RoutingInputs};
use ddrs::sparse::{CsrPattern, SparseAdjacency};

use common::{
    linear_chain_sparse, mock_config, mock_spatial_parameters, mock_streamflow, InnerBackend, TestBackend,
    TestDevice,
};

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

struct Cmp {
    bitwise: bool,
    worst: f32,
    /// Largest mass-sum difference as a fraction of `gross_flux_m3`.
    mass_worst: f32,
}

/// `max|a - r| <= 1e-6 max|r|`; records whether every value is equal.
fn check(cmp: &mut Cmp, what: &str, a: &[f32], r: &[f32]) {
    assert_eq!(a.len(), r.len(), "{what}: length");
    let scale = r.iter().fold(0.0f32, |m, v| m.max(v.abs()));
    let mut diff = 0.0f32;
    for (x, y) in a.iter().zip(r) {
        assert!(x.is_finite() == y.is_finite(), "{what}: finiteness differs ({x} vs {y})");
        if x != y {
            cmp.bitwise = false;
            if x.is_finite() {
                diff = diff.max((x - y).abs());
            }
        }
    }
    let rel = if scale > 0.0 { diff / scale } else { diff };
    cmp.worst = cmp.worst.max(rel);
    assert!(diff <= 1e-6 * scale, "{what}: max abs diff {diff} vs max|ref| {scale} (rel {rel:e})");
    // Every case is bitwise: hold the fields to equality.
    assert!(a.iter().zip(r).all(|(x, y)| x == y), "{what}: not bitwise (rel {rel:e})");
}

fn run_pair(case: &Case, theta: f32, substeps: u32, device: &TestDevice) -> Cmp {
    let n = case.adj.n;
    let mut reference = make_block(case, theta, substeps, device);
    let mut fused = make_block(case, theta, substeps, device);
    let (nn, pp, qq) = (tv(&case.n, device), tv(&case.p, device), tv(&case.q, device));
    let (mut qr, mut qf) = (case.q_t0.clone(), case.q_t0.clone());
    let mut cmp = Cmp { bitwise: true, worst: 0.0, mass_worst: 0.0 };
    for k in 0..case.hours {
        let qp = pick(&case.qp, k);
        let mc = pick(&case.mc, k);
        let out_r = to_vec1(reference.step_reference(
            &case.pattern, nn.clone(), pp.clone(), qq.clone(), tv(&qr, device), tv(&mc, device), tv(&qp, device),
        ));
        let out_f = to_vec1(fused.step_fused(
            &case.pattern, nn.clone(), pp.clone(), qq.clone(), tv(&qf, device), tv(&mc, device), tv(&qp, device),
        ));
        let tag = format!("{} theta {theta} substeps {substeps} hour {k}", case.label);
        check(&mut cmp, &format!("{tag} q_next"), &out_f, &out_r);
        check(
            &mut cmp,
            &format!("{tag} stage"),
            &to_vec1(fused.stage().unwrap()),
            &to_vec1(reference.stage().unwrap()),
        );
        let (cf, cr) = (fused.last_coefficients().unwrap(), reference.last_coefficients().unwrap());
        for (name, a, r) in [
            ("alpha", cf.alpha, cr.alpha),
            ("beta", cf.beta, cr.beta),
            ("kappa", cf.kappa, cr.kappa),
            ("a_up", cf.a_up, cr.a_up),
            ("b_down", cf.b_down, cr.b_down),
            ("area_edge", cf.area_edge, cr.area_edge),
            ("radius_edge", cf.radius_edge, cr.radius_edge),
            ("d_s", cf.d_s, cr.d_s),
            ("dt_phi", cf.dt_phi, cr.dt_phi),
            ("r_prime", cf.r_prime, cr.r_prime),
        ] {
            check(&mut cmp, &format!("{tag} {name}"), &to_vec1(a), &to_vec1(r));
        }
        let (mf, mr) = (fused.mass_balance(), reference.mass_balance());
        assert_eq!(mf.steps, mr.steps, "{tag} steps");
        let tol = 1e-5 * mr.gross_flux_m3.abs();
        for (name, a, r) in [
            ("storage", mf.storage_change_m3, mr.storage_change_m3),
            ("net", mf.net_flux_m3, mr.net_flux_m3),
            ("floor", mf.floor_created_m3, mr.floor_created_m3),
            ("gross", mf.gross_flux_m3, mr.gross_flux_m3),
            ("mc_in", mf.mc_inflow_m3, mr.mc_inflow_m3),
        ] {
            assert!((a - r).abs() <= tol, "{tag} mass {name}: {a} vs {r} (tol {tol})");
            cmp.mass_worst = cmp.mass_worst.max((a - r).abs() / mr.gross_flux_m3.abs().max(f32::MIN_POSITIVE));
        }
        qr = out_r;
        qf = out_f;
        assert_eq!(qr.len(), n);
    }
    cmp
}

/// Run a case on the parity grid and report.
fn run_grid(case: &Case, device: &TestDevice) -> bool {
    let mut all = true;
    for theta in [0.55f32, 0.6, 1.0] {
        for substeps in [1u32, 4] {
            let c = run_pair(case, theta, substeps, device);
            println!("{} theta {theta} substeps {substeps}: bitwise: {} (worst rel {:e}, mass diff/gross {:e})", case.label, c.bitwise, c.worst, c.mass_worst);
            all &= c.bitwise;
        }
    }
    all
}

fn family(cases: Vec<Case>) {
    let device = TestDevice::default();
    let mut all = true;
    for c in &cases {
        all &= run_grid(c, &device);
    }
    println!("bitwise: {all}");
}

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

#[test]
fn forward_parity_chain_states() {
    family(chain_state_cases());
}

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

#[test]
fn forward_parity_table_and_mixed() {
    family(table_and_mixed_cases());
}

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

#[test]
fn forward_parity_prescribed_boundary() {
    family(prescribed_boundary_cases());
}

fn flat(sf: &Tensor<TestBackend, 2>) -> Vec<f32> {
    sf.clone().into_data().to_vec().unwrap()
}

fn mc_inflow_case(device: &TestDevice) -> Case {
    let n = 3;
    let hours = 24;
    let t = hours + 1;
    let adj = linear_chain_sparse(n);
    let pattern = Arc::new(CsrPattern::from_sparse(&adj));
    let sf = mock_streamflow(t, n, &device);
    let mut mcr = MuskingumCunge::<InnerBackend>::new(mock_config(), device.clone());
    mcr.setup_inputs(
        RoutingInputs { adjacency: adj.clone(), x_storage: Tensor::<TestBackend, 1>::ones([n], device) * 0.2 },
        sf.clone(),
        mock_spatial_parameters(n, &device),
        false,
        None,
    );
    let f: Vec<f32> = mcr.forward().into_data().to_vec().unwrap();
    let mc: Vec<Vec<f32>> = (0..n).map(|i| f[i * t..(i + 1) * t].to_vec()).collect();
    let sfl = flat(&sf);
    let q_t0: Vec<f32> = (0..n).map(|i| mc[i][0]).collect();
    let net = StageNetwork::build(&adj, &pattern, Some(&[1]), 0.0, &[]).unwrap();
    let h0: Vec<f32> = net.z_mid.iter().map(|z| z + 0.6).collect();
    let mut case = Case::new("mc_inflow", adj, pattern, H0::Abs(h0), q_t0);
    case.seeds = Some(vec![1]);
    case.hours = hours;
    case.qp = (0..hours).map(|k| sfl[k * n..(k + 1) * n].to_vec()).collect();
    case.mc = (0..hours).map(|k| (0..n).map(|i| mc[i][k + 1]).collect()).collect();
    case
}

#[test]
fn forward_parity_mc_inflow() {
    let device = TestDevice::default();
    let case = mc_inflow_case(&device);
    let mut all = true;
    for theta in [0.55f32, 0.6, 1.0] {
        for substeps in [1u32, 4] {
            let c = run_pair(&case, theta, substeps, &device);
            println!("mc_inflow theta {theta} substeps {substeps}: bitwise: {} (worst rel {:e}, mass diff/gross {:e})", c.bitwise, c.worst, c.mass_worst);
            all &= c.bitwise;
        }
    }
    println!("bitwise: {all}");
}

fn juniata_case(device: &TestDevice) -> Case {
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
    // hotstart stage on a throwaway block
    let net = StageNetwork::build(&adj, &pattern, None, 0.0, &[]).unwrap();
    let mut hb = StageBlock::<InnerBackend>::new(
        net,
        StageParams {
            theta: 1.0,
            theta_control: 0.5,
            substeps: 1,
            boundary: DownstreamBoundary::NormalDepth,
            depth_lb: DEPTH_LB,
            bottom_width_lb: 0.1,
            dt: 3600.0,
            g: 9.81,
            use_cuda: false,
        },
        &device,
    );
    let slope_f: Vec<f32> = adj.slope.iter().map(|s| s.max(1e-3)).collect();
    hb.hotstart(
        tv(&q_init, &device),
        tv(&vec![0.035; n], &device),
        tv(&vec![21.0; n], &device),
        tv(&vec![0.5; n], &device),
        tv(&slope_f, &device),
    );
    let h0 = to_vec1(hb.stage().unwrap());
    let mut case = Case::new("juniata_48h", adj, pattern, H0::Abs(h0), q_init);
    case.n = vec![0.035; n];
    case.p = vec![21.0; n];
    case.q = vec![0.5; n];
    case.qp = vec![vec![1.0; n]];
    case.mc = vec![vec![0.0; n]];
    case.hours = 48;
    case
}

#[test]
fn forward_parity_juniata_48h() {
    if cfg!(debug_assertions) {
        eprintln!("skipped in debug builds: run with --release");
        return;
    }
    let device = TestDevice::default();
    let case = juniata_case(&device);
    let mut all = true;
    for (theta, substeps) in [(1.0f32, 1u32), (0.6, 4)] {
        let c = run_pair(&case, theta, substeps, &device);
        println!("juniata_48h theta {theta} substeps {substeps}: bitwise: {} (worst rel {:e}, mass diff/gross {:e})", c.bitwise, c.worst, c.mass_worst);
        all &= c.bitwise;
    }
    println!("bitwise: {all}");
}

#[test]
fn op_consts_match_the_device_constants() {
    let device = TestDevice::default();
    let (adj, pattern) = chain(&[1000.0, 1200.0, 900.0], &[1e-3, 2e-3, 0.0]);
    let net = StageNetwork::build(&adj, &pattern, None, 0.0, &[(1, 1.23)]).unwrap();
    let net_copy = net.clone();
    let params = StageParams {
        theta: 0.7,
        theta_control: 0.4,
        substeps: 2,
        boundary: DownstreamBoundary::NormalDepth,
        depth_lb: DEPTH_LB,
        bottom_width_lb: 0.1,
        dt: 3600.0,
        g: 9.81,
        use_cuda: false,
    };
    let mut block = StageBlock::<InnerBackend>::new(net, params, &device);
    let c = block.op_consts(&pattern);
    let n = 3;
    assert_eq!((c.n, c.nnz, c.substeps), (n, pattern.nnz(), 2));
    assert_eq!((c.theta, c.theta_c, c.g, c.depth_lb, c.bw_lb, c.dt), (0.7, 0.4, 9.81, DEPTH_LB, 0.1, 3600.0));
    assert!(!c.prescribed && !c.has_table);
    assert_eq!(c.down, vec![1, 2, 2]);
    assert_eq!(c.in_stage, vec![1.0; 3]);
    // dam at row 1; the normal-depth outlet is row 2
    assert_eq!(c.dam, vec![0.0, 1.0, 0.0]);
    assert_eq!(c.nd, vec![0.0, 0.0, 1.0]);
    assert_eq!(c.friction, vec![1.0, 0.0, 0.0]);
    assert_eq!(c.ctrl, vec![0.0, 1.0, 1.0]);
    assert_eq!(c.w_ctrl, vec![0.0, 0.4, 0.7]);
    assert_eq!(c.w_edge, vec![0.7, 0.4, 0.7]);
    assert_eq!(c.has_down, vec![1.0, 1.0, 0.0]);
    assert_eq!(c.outlet, vec![0.0, 0.0, 1.0]);
    assert_eq!(c.z_mid, net_copy.z_mid);
    assert_eq!(c.length, net_copy.length_m);
    assert_eq!(c.edge_length, net_copy.edge_length);
    assert_eq!(c.dam_t, vec![1.0, 1.23 * 86_400.0, 1.0]);
    let s0: Vec<f32> = net_copy.slope.iter().map(|s| s.max(1e-5).sqrt()).collect();
    assert_eq!(c.s0_sqrt, s0);
    assert!(c.t_mask.is_empty() && c.t_tw.is_empty() && c.t_n.is_empty() && c.t_pool.is_empty());
    for k in 0..pattern.nnz() {
        assert_eq!(c.row[k], pattern.row_for_nnz[k] as usize);
        assert_eq!(c.col[k], pattern.col[k] as usize);
    }
    assert_eq!(c.adj_stage, net_copy.adj_stage_to_stage);
    assert_eq!(c.adj_mc, net_copy.adj_mc_to_stage);
    assert_eq!(c.diag_slot.iter().sum::<f32>(), n as f32);
    // the dam row 1 is a column of the slot (2, 1): a dam slot, not a friction slot
    let dam_slots: Vec<usize> = (0..c.nnz).filter(|&k| c.dam_slot[k] == 1.0).collect();
    assert_eq!(dam_slots.len(), 1);
    assert_eq!((c.row[dam_slots[0]], c.col[dam_slots[0]]), (2, 1));
    // a second call returns the same Arc
    let c2 = block.op_consts(&pattern);
    assert!(Arc::ptr_eq(&c, &c2));
    // a table row replaces the Arc
    block.set_table_geometry(table(&[2], 80.0, &[0.3], 0.02, &[None]), false).unwrap();
    let c3 = block.op_consts(&pattern);
    assert!(!Arc::ptr_eq(&c, &c3));
    assert!(c3.has_table);
    assert_eq!(c3.z_mid[2], 0.3);
    assert_eq!(c3.z_mid[0], net_copy.z_mid[0]);
    assert_eq!(c3.t_mask, vec![0.0, 0.0, 1.0]);
    assert_eq!(c3.t_tw[2], 80.0);
    assert_eq!(c3.t_n[2], 0.02);
    assert_eq!(c3.t_pool[2], 0.3);
}

/// `use_cuda: true` on a CPU backend falls back to the host path, bit for bit.
#[test]
fn use_cuda_on_a_cpu_backend_falls_back_bitwise() {
    let device = TestDevice::default();
    let (adj, pattern) = chain(&[1000.0; 3], &[1e-3; 3]);
    let run = |use_cuda: bool| {
        let net = StageNetwork::build(&adj, &pattern, None, 0.0, &[]).unwrap();
        let z = net.z_mid.clone();
        let sp = StageParams {
            theta: 1.0,
            theta_control: 0.5,
            substeps: 1,
            boundary: DownstreamBoundary::NormalDepth,
            depth_lb: DEPTH_LB,
            bottom_width_lb: 0.1,
            dt: 3600.0,
            g: 9.81,
            use_cuda,
        };
        let mut block = StageBlock::<InnerBackend>::new(net, sp, &device);
        let h: Vec<f32> = z.iter().map(|z| z + 0.5).collect();
        block.set_stage(tv(&h, &device));
        let mut q = vec![1.0f32; 3];
        let mut outs = Vec::new();
        for _ in 0..3 {
            q = to_vec1(block.step_fused(
                &pattern,
                tv(&[0.035; 3], &device),
                tv(&[21.0; 3], &device),
                tv(&[0.5; 3], &device),
                tv(&q, &device),
                tv(&[0.0; 3], &device),
                tv(&[1.0; 3], &device),
            ));
            let mb = block.mass_balance();
            let mut row = q.clone();
            row.extend(to_vec1(block.stage().unwrap()));
            row.extend([
                mb.storage_change_m3,
                mb.net_flux_m3,
                mb.floor_created_m3,
                mb.gross_flux_m3,
                mb.mc_inflow_m3,
            ]);
            outs.push(row);
        }
        outs
    };
    let (on, off) = (run(true), run(false));
    for (hour, (a, b)) in on.iter().zip(&off).enumerate() {
        assert_eq!(a.len(), b.len());
        for (i, (x, y)) in a.iter().zip(b).enumerate() {
            assert_eq!(x.to_bits(), y.to_bits(), "hour {hour} entry {i}");
        }
    }
}

// ================================================================ gradients
//
// Gradient parity of the fused op against Burn autodiff of
// `step_reference`, and a central-difference gradcheck of the op.
//
// Gradient bar: per entry `|g_f - g_r| <= 1e-4 max(|g_f|, |g_r|)`
// or `<= 1e-6 max(1, max_j |g_r,j|)` per parent. Loss
// `L = sum_t sum_i w[t][i] q_next[t][i] + sum_i v_i (h_end_i - z_mid_i)`, with
// `w` and `v` seeded uniform in [0.5, 1.5].

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

const PARENTS: [&str; 7] = ["n", "q", "p", "h_t", "q_t", "q_next_mc", "q_prime"];

/// The loss and the gradient of each parent (hours concatenated for `q_next_mc`
/// and `q_prime`). A fused gradient that is `None` is a test failure.
#[derive(Clone, Copy, PartialEq)]
enum LossPart {
    Both,
    /// Only the `q_next` terms.
    Q,
    /// Only the `h_end` term.
    H,
}

fn run_grads(case: &Case, theta: f32, substeps: u32, fused: bool, device: &TestDevice) -> (f32, Vec<Vec<f32>>) {
    run_grads_part(case, theta, substeps, fused, LossPart::Both, device)
}

fn run_grads_part(
    case: &Case,
    theta: f32,
    substeps: u32,
    fused: bool,
    part: LossPart,
    device: &TestDevice,
) -> (f32, Vec<Vec<f32>>) {
    let n = case.adj.n;
    let mut block = make_block(case, theta, substeps, device);
    let z_mid = block.op_consts(&case.pattern).z_mid.clone();
    let lf = |v: &[f32]| tv(v, device).require_grad();
    let h0 = to_vec1(block.stage().unwrap());
    let h_leaf = lf(&h0);
    block.set_stage(h_leaf.clone());
    let (nn, pp, qq) = (lf(&case.n), lf(&case.p), lf(&case.q));
    let qt_leaf = lf(&case.q_t0);
    let mut q_t = qt_leaf.clone();
    let (mut qp_l, mut mc_l) = (vec![], vec![]);
    let mut loss: Option<Tensor<TestBackend, 1>> = None;
    for k in 0..case.hours {
        let qpl = lf(&pick(&case.qp, k));
        let mcl = lf(&pick(&case.mc, k));
        qp_l.push(qpl.clone());
        mc_l.push(mcl.clone());
        let out = if fused {
            block.step_fused(&case.pattern, nn.clone(), pp.clone(), qq.clone(), q_t.clone(), mcl, qpl)
        } else {
            block.step_reference(&case.pattern, nn.clone(), pp.clone(), qq.clone(), q_t.clone(), mcl, qpl)
        };
        let wq = if part == LossPart::H { vec![0.0; n] } else { weights(1 + k as u64, n) };
        let term = (out.clone() * tv(&wq, device)).sum();
        loss = Some(match loss {
            None => term,
            Some(l) => l + term,
        });
        q_t = out;
    }
    let h_end = block.stage().unwrap();
    let off = h_end - tv(&z_mid, device);
    let wv = if part == LossPart::Q { vec![0.0; n] } else { weights(9999, n) };
    let loss = loss.unwrap() + (off * tv(&wv, device)).sum();
    let value = loss.clone().into_scalar();
    let g = loss.backward();
    let get = |t: &Tensor<TestBackend, 1>, what: &str| -> Vec<f32> {
        match t.grad(&g) {
            Some(v) => to_vec1(v),
            None => {
                assert!(!fused, "fused: no gradient registered for tracked parent {what}");
                vec![0.0; n]
            }
        }
    };
    let cat = |ts: &[Tensor<TestBackend, 1>], what: &str| -> Vec<f32> {
        ts.iter().flat_map(|t| get(t, what)).collect()
    };
    let grads = vec![
        get(&nn, "n"),
        get(&qq, "q"),
        get(&pp, "p"),
        get(&h_leaf, "h_t"),
        get(&qt_leaf, "q_t"),
        cat(&mc_l, "q_next_mc"),
        cat(&qp_l, "q_prime"),
    ];
    (value, grads)
}

/// Rounding noise of a gradient run: per parent, `max_i |g_q[i] + g_h[i] - g_both[i]|`,
/// the failure of linearity in the loss. The three runs differ only in the
/// order and size of the f32 sums, so this measures the rounding noise of the
/// path itself.
fn linearity_noise(case: &Case, theta: f32, substeps: u32, fused: bool, both: &[Vec<f32>], device: &TestDevice) -> [f32; 7] {
    let (_, gq) = run_grads_part(case, theta, substeps, fused, LossPart::Q, device);
    let (_, gh) = run_grads_part(case, theta, substeps, fused, LossPart::H, device);
    let mut out = [0.0f32; 7];
    for pi in 0..7 {
        for i in 0..both[pi].len() {
            out[pi] = out[pi].max((gq[pi][i] + gh[pi][i] - both[pi][i]).abs());
        }
    }
    out
}

/// Why a bounded allowance exists. Both paths are f32. The Burn autodiff
/// reference is itself only accurate to f32 rounding, and on entries made by
/// cancelling large terms (a badly conditioned PrescribedStage outlet row,
/// sums over 2 or more hours) its error exceeds the strict bar: an f64 ground
/// truth shows the reference misses the strict bar at the
/// same rate as the fused path. One-hour runs are noise-free and are held to
/// the strict bar by `gradient_parity_hour1_strict`. For the multi-hour runs a
/// strict-bar miss is accepted only when all three hold:
/// 1. `err <= NOISE_FACTOR` times the measured linearity noise of both paths;
/// 2. `err <= NOISE_CAP_FRAC * max|g|` of its parent;
/// 3. the accepted entries of the family are at most `NOISE_BUDGET_FRAC` of
///    the compared entries.
const NOISE_FACTOR: f32 = 4.0;
const NOISE_CAP_FRAC: f32 = 5e-4;
const NOISE_BUDGET_FRAC: f64 = 0.01;

#[derive(Default)]
struct Tally {
    compared: usize,
    accepted: usize,
}

/// One state: returns the worst `err / allowed` per parent. With `strict_only`
/// every miss fails. Otherwise a miss must meet the bounded acceptance above;
/// accepted entries are printed and counted in `tally`.
fn grad_pair(
    family: &str,
    case: &Case,
    theta: f32,
    substeps: u32,
    device: &TestDevice,
    strict_only: bool,
    tally: &mut Tally,
) -> [f32; 7] {
    let (lr, gr) = run_grads(case, theta, substeps, false, device);
    let (lf, gf) = run_grads(case, theta, substeps, true, device);
    let tag = format!("{} theta {theta} substeps {substeps}", case.label);
    assert_eq!(lr, lf, "{tag}: loss differs (forward is bitwise)");
    let mut worst = [0.0f32; 7];
    let mut noise: Option<[f32; 7]> = None;
    for pi in 0..7 {
        let (g, r) = (&gf[pi], &gr[pi]);
        assert_eq!(g.len(), r.len(), "{tag} {}: length", PARENTS[pi]);
        let scale = r.iter().fold(1.0f32, |m, v| m.max(v.abs()));
        let gmax = r.iter().fold(0.0f32, |m, v| m.max(v.abs()));
        for (i, (a, b)) in g.iter().zip(r).enumerate() {
            assert!(a.is_finite(), "{tag} {}[{i}]: non-finite fused gradient {a}", PARENTS[pi]);
            assert!(b.is_finite(), "{tag} {}[{i}]: non-finite reference gradient {b}", PARENTS[pi]);
            tally.compared += 1;
            let err = (a - b).abs();
            let allowed = (1e-4 * a.abs().max(b.abs())).max(1e-6 * scale);
            worst[pi] = worst[pi].max(err / allowed);
            if err <= allowed {
                continue;
            }
            assert!(
                !strict_only,
                "{tag} {}[{i}]: fused {a} vs autodiff {b} (err {err:e}, allowed {allowed:e})",
                PARENTS[pi]
            );
            let nz = *noise.get_or_insert_with(|| {
                let nr = linearity_noise(case, theta, substeps, false, &gr, device);
                let nf = linearity_noise(case, theta, substeps, true, &gf, device);
                std::array::from_fn(|k| nr[k] + nf[k])
            });
            let ratio = err / nz[pi].max(f32::MIN_POSITIVE);
            println!(
                "accepted: {family} {tag} {}[{i}]: fused {a} vs autodiff {b} err {err:e}, err/max|g| {:e}, err/noise {ratio:.2}",
                PARENTS[pi],
                err / gmax.max(f32::MIN_POSITIVE)
            );
            assert!(
                err <= NOISE_FACTOR * nz[pi],
                "{tag} {}[{i}]: err {err:e} exceeds {NOISE_FACTOR} x noise {:e}",
                PARENTS[pi], nz[pi]
            );
            assert!(
                err <= NOISE_CAP_FRAC * gmax,
                "{tag} {}[{i}]: err {err:e} exceeds the cap {NOISE_CAP_FRAC} x max|g| {gmax:e}",
                PARENTS[pi]
            );
            tally.accepted += 1;
        }
    }
    worst
}

fn grad_family_with(family: &str, cases: Vec<Case>, grid: &[(f32, u32)], strict_only: bool) {
    grad_family_budget(family, cases, grid, strict_only, NOISE_BUDGET_FRAC);
}

fn grad_family_budget(family: &str, cases: Vec<Case>, grid: &[(f32, u32)], strict_only: bool, budget: f64) {
    let device = TestDevice::default();
    let mut tally = Tally::default();
    for case in &cases {
        for &(theta, substeps) in grid {
            let w = grad_pair(family, case, theta, substeps, &device, strict_only, &mut tally);
            println!("{family}: {} theta {theta} substeps {substeps}: worst err/allowed {w:?} (parents {PARENTS:?})", case.label);
        }
    }
    println!("{family}: compared {} entries, accepted {} strict-bar misses", tally.compared, tally.accepted);
    assert!(
        (tally.accepted as f64) <= budget * tally.compared as f64,
        "{family}: {} accepted misses exceed {budget} of {} compared entries",
        tally.accepted,
        tally.compared
    );
}

fn grad_family(family: &str, cases: Vec<Case>, grid: &[(f32, u32)]) {
    grad_family_with(family, cases, grid, false);
}

fn full_grid() -> Vec<(f32, u32)> {
    let mut g = vec![];
    for theta in [0.55f32, 0.6, 1.0] {
        for substeps in [1u32, 4] {
            g.push((theta, substeps));
        }
    }
    g
}

#[test]
fn gradient_parity_chain_states() {
    grad_family("chain_states", chain_state_cases(), &full_grid());
}

#[test]
fn gradient_parity_table_and_mixed() {
    grad_family("table_and_mixed", table_and_mixed_cases(), &full_grid());
}

#[test]
fn gradient_parity_prescribed_boundary() {
    grad_family("prescribed_boundary", prescribed_boundary_cases(), &full_grid());
}

#[test]
fn gradient_parity_mc_inflow() {
    let device = TestDevice::default();
    grad_family("mc_inflow", vec![mc_inflow_case(&device)], &full_grid());
}

#[test]
fn gradient_parity_juniata_48h() {
    if cfg!(debug_assertions) {
        eprintln!("skipped in debug builds: run with --release");
        return;
    }
    let device = TestDevice::default();
    grad_family("juniata_48h", vec![juniata_case(&device)], &[(1.0, 1), (0.6, 4)]);
}

/// The strict bar with no noise allowance, on one-hour runs of the
/// families whose one-hour gradients are noise-free: chain, prescribed
/// boundary, MC inflow and (release only) Juniata. This pins the formulas.
#[test]
fn gradient_parity_hour1_strict() {
    let device = TestDevice::default();
    let mut families: Vec<(&str, Vec<Case>, Vec<(f32, u32)>)> = vec![
        ("chain_states", chain_state_cases(), full_grid()),
        ("prescribed_boundary", prescribed_boundary_cases(), full_grid()),
        ("mc_inflow", vec![mc_inflow_case(&device)], full_grid()),
    ];
    if !cfg!(debug_assertions) {
        families.push(("juniata_48h", vec![juniata_case(&device)], vec![(1.0, 1), (0.6, 4)]));
    }
    for (name, mut cases, grid) in families {
        for c in cases.iter_mut() {
            c.hours = 1;
        }
        grad_family_with(&format!("{name} (hour 1)"), cases, &grid, true);
    }
}

/// One-hour table_and_mixed runs are NOT noise-free: their PrescribedStage
/// outlet row is a table row (80 m wide, badly conditioned), and with the
/// loss weights 6 of 504 entries miss the strict bar (worst 1.97 x the bar,
/// err/noise <= 1.0, err/max|g| <= 1.6e-4). The count changes with the weight
/// seed (6, 5, 2, 0 misses for four seeds). They get the noise and cap gates
/// with the looser budget `NOISE_BUDGET_FRAC_TABLE_HOUR1` (2 %). The controller ruled that
/// table_and_mixed hour 1 stays under the noise and cap gate with this budget,
/// because its misses sit on the badly conditioned table PrescribedStage outlet row.
const NOISE_BUDGET_FRAC_TABLE_HOUR1: f64 = 0.02;

#[test]
fn gradient_parity_hour1_table_and_mixed() {
    let mut cases = table_and_mixed_cases();
    for c in cases.iter_mut() {
        c.hours = 1;
    }
    grad_family_budget("table_and_mixed (hour 1)", cases, &full_grid(), false, NOISE_BUDGET_FRAC_TABLE_HOUR1);
}

#[test]
fn fused_op_gradcheck_by_central_differences() {
    // 5-chain, stage set closure([1]): row 0 is MC, so q_t and q_next_mc enter
    // through N_mc. Dam on row 2 (T 1.23 d), NormalDepth, theta 0.8.
    let device = TestDevice::default();
    let (adj, pattern) = chain(&[1000.0; 5], &[1e-3; 5]);
    let n = 5;
    let net0 = StageNetwork::build(&adj, &pattern, Some(&[1]), 0.0, &[(2, 1.23)]).unwrap();
    let depth0 = [0.45f32, 0.55, 0.8, 0.6, 0.7];
    let h0: Vec<f32> = (0..n).map(|i| net0.z_mid[i] + depth0[i]).collect();
    let z_mid = net0.z_mid.clone();
    let base: [Vec<f32>; 7] = [
        (0..n).map(|i| 0.03 + 0.005 * i as f32).collect(),
        (0..n).map(|i| 0.45 + 0.05 * (i % 3) as f32).collect(),
        (0..n).map(|i| 18.0 + 2.0 * (i % 4) as f32).collect(),
        h0,
        vec![1.1, 2.3, 3.2, 4.6, 5.1],
        vec![1.4, 0.0, 0.0, 0.0, 0.0],
        vec![1.0; n],
    ];
    // n, q, p, h_t, q_t, q_next_mc, q'
    const EPS: [f32; 7] = [1e-3, 1e-2, 0.2, 1e-2, 1e-2, 1e-2, 1e-2];
    let w = weights(5, n);
    let v = weights(6, n);
    for substeps in [1u32, 4] {
        let mut case = Case::new("gradcheck", adj.clone(), pattern.clone(), H0::Abs(base[3].clone()), base[4].clone());
        case.seeds = Some(vec![1]);
        case.dams = vec![(2, 1.23)];
        case.hours = 1;
        let eval = |x: &[Vec<f32>; 7], with_grad: bool| -> (f32, Vec<Vec<f32>>) {
            let mut block = make_block(&case, 0.8, substeps, &device);
            let ts: Vec<Tensor<TestBackend, 1>> = x.iter().map(|a| tv(a, &device).require_grad()).collect();
            block.set_stage(ts[3].clone());
            let out = block.step_fused(
                &pattern, ts[0].clone(), ts[2].clone(), ts[1].clone(), ts[4].clone(), ts[5].clone(), ts[6].clone(),
            );
            let h_end = block.stage().unwrap();
            let loss = (out * tv(&w, &device)).sum() + ((h_end - tv(&z_mid, &device)) * tv(&v, &device)).sum();
            let l = loss.clone().into_scalar();
            if !with_grad {
                return (l, vec![]);
            }
            let g = loss.backward();
            (l, ts.iter().map(|t| to_vec1(t.grad(&g).expect("grad"))).collect())
        };
        let (l0, grads) = eval(&base, true);
        println!("gradcheck substeps {substeps}: loss {l0}");
        let mut worst = [0.0f32; 7];
        for pi in 0..7 {
            for i in 0..n {
                let eps = EPS[pi];
                let mut up = base.clone();
                let mut dn = base.clone();
                up[pi][i] += eps;
                dn[pi][i] -= eps;
                let fd = (eval(&up, false).0 - eval(&dn, false).0) / (2.0 * eps);
                let an = grads[pi][i];
                let err = (fd - an).abs();
                let allowed = (5e-3 * fd.abs().max(an.abs())).max(1e-4);
                assert!(
                    an.is_finite() && err <= allowed,
                    "substeps {substeps}: d loss/d {}[{i}]: fd {fd} analytic {an}",
                    PARENTS[pi]
                );
                worst[pi] = worst[pi].max(err / allowed);
            }
        }
        println!("gradcheck substeps {substeps}: worst err/allowed {worst:?} (parents {PARENTS:?})");
    }
}

/// The autodiff requirement of the output node (`is_require_grad` is false for
/// every derived node, so it cannot tell tracked from untracked).
fn requires_grad(t: &Tensor<TestBackend, 1>) -> bool {
    match t.clone().into_primitive() {
        burn::tensor::TensorPrimitive::Float(a) => !format!("{:?}", a.node.requirement).contains("None"),
        _ => panic!("expected float tensor"),
    }
}

#[test]
fn fused_op_untracked_matches_tracked() {
    let device = TestDevice::default();
    let case = chain_state_cases().remove(0);
    let (theta, substeps) = (0.6f32, 4u32);
    let mut tracked = make_block(&case, theta, substeps, &device);
    let mut plain = make_block(&case, theta, substeps, &device);
    let h0 = to_vec1(tracked.stage().unwrap());
    tracked.set_stage(tv(&h0, &device).require_grad());
    let mut q_tr = tv(&case.q_t0, &device).require_grad();
    let mut q_pl = tv(&case.q_t0, &device);
    let (nl, ql, pl) = (tv(&case.n, &device), tv(&case.q, &device), tv(&case.p, &device));
    let n_leaf = nl.clone().require_grad();
    for k in 0..case.hours {
        let qp = tv(&pick(&case.qp, k), &device);
        let mc = tv(&pick(&case.mc, k), &device);
        let out_t = tracked.step_fused(
            &case.pattern, n_leaf.clone(), pl.clone(), ql.clone(), q_tr.clone(), mc.clone(), qp.clone(),
        );
        let out_p = plain.step_fused(&case.pattern, nl.clone(), pl.clone(), ql.clone(), q_pl.clone(), mc, qp);
        assert!(!requires_grad(&out_p), "hour {k}: untracked output must not be tracked");
        assert!(requires_grad(&out_t), "hour {k}: tracked output must be tracked");
        assert_eq!(to_vec1(out_t.clone()), to_vec1(out_p.clone()), "hour {k}: q_next bits");
        assert_eq!(to_vec1(tracked.stage().unwrap()), to_vec1(plain.stage().unwrap()), "hour {k}: stage bits");
        q_tr = out_t;
        q_pl = out_p;
    }
    // The tracked run also carries a gradient to its leaf.
    let g = q_tr.sum().backward();
    assert!(n_leaf.grad(&g).is_some(), "tracked run: no gradient reached the n leaf");
}

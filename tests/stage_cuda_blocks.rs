//! The elementwise forward blocks of the stage op on CUDA, against the host
//! block fns of `src/routing/stage_op.rs`, on the same inputs.
//!
//! Bar (plan D7, block row): `max|dev - host| <= 1e-5 * max|host|` per field,
//! 0/1 indicator fields bitwise equal, `same_zero_set` on the clamped
//! outputs. Device inputs come from the `upload` fns, never from another
//! device fn. Every test prints the worst relative difference per field.
//! Gather kernels run on the Juniata network (it has confluences and
//! outlets). Exact IEEE ties are inputs and are checked on the host first.
#![cfg(feature = "cuda")]

use std::sync::Arc;

use burn::backend::NdArray;
use burn::backend::Autodiff;
use burn::tensor::Tensor;

use ddrs::config::DownstreamBoundary;
use ddrs::data::store::ChannelGeometryRows;
use ddrs::data::ConusAdjacencyStore;
use ddrs::routing::stage::{StageBlock, StageNetwork, StageParams};
use ddrs::routing::stage_cuda::forward::{
    assembled_to_host, control_dev, control_to_host, depth_dev, depth_to_host, hour_to_host, i_mid_dev, matrix_dev,
    newton_dev, newton_to_host, readout_to_host, rhs_dev, section_dev, section_to_host, stage_assemble_dev,
    stage_prologue_dev, stage_readout_dev, storage_rate_dev, AssembledDev,
};
use ddrs::routing::stage_cuda::backward::{
    assemble_backward_dev, control_backward_dev, control_grads_to_host, depth_backward_dev, hour_acc_to_host,
    hour_grads_to_host, matrix_backward_dev, matrix_grads_to_host, newton_backward_dev, newton_grads_to_host,
    output_backward_dev, prologue_backward_dev, readout_backward_dev, readout_grads_to_host, rhs_backward_dev,
    rhs_grads_to_host, section_backward_dev, section_grads_to_host, ControlBarDev, HourAccDev, NewtonBarDev,
    ReadoutGradsDev, SectionBarDev,
};
use ddrs::routing::stage_cuda::{ControlDev, DepthDev, HourDev, NewtonDev, SectionDev, StageOpConstsDev};
use ddrs::routing::stage_op::{
    assemble_backward_host, control_backward_host, depth_backward_host, matrix_backward_host, newton_backward_host,
    output_backward_host, prologue_backward_host, readout_backward_host, rhs_backward_host, section_backward_host,
    stage_hour_forward, control_host, depth_host, matrix_host, newton_host, rhs_host, section_host,
    stage_assemble_host, stage_prologue_host, stage_readout_host, Assembled, ControlBar, ControlOut, DepthOut,
    HourAcc, HourHost, NewtonBar, NewtonOut, SectionBar, SectionHost, StageOpConsts,
};
use ddrs::sparse::cube_buf::{cube_from_host, cube_to_host, cube_to_host_i32, Cube, Gpu};
use ddrs::sparse::tree_lu::{
    tree_levels, tree_lu_factor, tree_lu_grad_a_host, tree_lu_solve_host, tree_lu_solve_transposed_host,
};
use ddrs::sparse::{CsrPattern, SparseAdjacency};

type Cpu = NdArray<f32>;

const DEPTH_LB: f32 = 0.01;
const BW_LB: f32 = 0.1;

// ----------------------------------------------------------------- bars

/// `max|d - h| <= 1e-5 * max|h|` (both all zero passes); every entry finite;
/// prints the worst relative difference.
fn close(what: &str, dev: &[f32], host: &[f32]) {
    assert_eq!(dev.len(), host.len(), "{what}: length");
    let scale = host.iter().fold(0.0f32, |m, v| m.max(v.abs()));
    let mut worst = 0.0f32;
    let mut at = 0usize;
    for (i, (d, h)) in dev.iter().zip(host).enumerate() {
        assert!(d.is_finite() && h.is_finite(), "{what}[{i}]: non-finite dev {d} host {h}");
        let diff = (d - h).abs();
        if diff > worst {
            worst = diff;
            at = i;
        }
    }
    let rel = if scale > 0.0 { worst / scale } else { worst };
    println!("{what}: worst |d-h|/max|h| {rel:.3e} (reach {at}, max|h| {scale:.3e})");
    assert!(
        worst <= 1e-5 * scale,
        "{what}: reach {at}: dev {} vs host {} (max|h| {scale:e})",
        dev[at],
        host[at]
    );
}

/// 0/1 indicator fields: bitwise equal.
fn exact(what: &str, dev: &[f32], host: &[f32]) {
    assert_eq!(dev.len(), host.len(), "{what}: length");
    for (i, (d, h)) in dev.iter().zip(host).enumerate() {
        assert!(d.to_bits() == h.to_bits(), "{what}[{i}]: dev {d} vs host {h}");
    }
}

/// Per entry `(d == 0.0) == (h == 0.0)`, with -0.0 == 0.0.
fn same_zero_set(what: &str, dev: &[f32], host: &[f32]) {
    assert_eq!(dev.len(), host.len(), "{what}: length");
    let bad: Vec<String> = dev
        .iter()
        .zip(host)
        .enumerate()
        .filter(|(_, (d, h))| (**d == 0.0) != (**h == 0.0))
        .take(5)
        .map(|(i, (d, h))| format!("reach {i}: dev {d} host {h}"))
        .collect();
    assert!(bad.is_empty(), "{what}: zero sets differ: {bad:?}");
}

// ------------------------------------------------------------- fixtures

fn gpu() -> <Gpu as burn::tensor::backend::BackendTypes>::Device {
    Default::default()
}

fn like() -> Cube {
    cube_from_host(&[0.0], &gpu())
}

fn up(v: &[f32]) -> Cube {
    cube_from_host(v, &gpu())
}

// copied from tests/stage_op_adjoint.rs::bar
fn bar(seed: u32, len: usize, signed: bool) -> Vec<f32> {
    let mut s = 0x9E3779B9u32.wrapping_mul(seed.wrapping_add(1));
    (0..len)
        .map(|i| {
            s = s.wrapping_mul(1664525).wrapping_add(1013904223);
            let m = 0.5 + (s >> 8) as f32 / (1u32 << 24) as f32;
            if signed && (s >> 7) & 1 == 1 && i % 2 == 0 {
                -m
            } else {
                m
            }
        })
        .collect()
}

/// `lo + (hi - lo) * u` with `u` in [0, 1).
fn gen(seed: u32, len: usize, lo: f32, hi: f32) -> Vec<f32> {
    bar(seed, len, false).iter().map(|b| lo + (hi - lo) * (b - 0.5)).collect()
}

// copied from tests/stage_op_adjoint.rs::chain
fn chain(n: usize) -> (SparseAdjacency, Arc<CsrPattern>) {
    let mut dense = vec![0.0_f32; n * n];
    for i in 0..n - 1 {
        dense[(i + 1) * n + i] = 1.0;
    }
    let adj = SparseAdjacency::from_dense(n, &dense, vec![1000.0; n], vec![1e-3; n]);
    let pat = Arc::new(CsrPattern::from_sparse(&adj));
    (adj, pat)
}

// copied from tests/stage_op_parity.rs::juniata_case (network part)
fn juniata() -> (SparseAdjacency, Arc<CsrPattern>) {
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
    (adj, pattern)
}

fn params(theta: f32, substeps: u32, prescribed: bool) -> StageParams {
    params_tc(theta, substeps, prescribed, 0.5)
}

fn params_tc(theta: f32, substeps: u32, prescribed: bool, theta_control: f32) -> StageParams {
    StageParams {
        theta,
        theta_control,
        substeps,
        boundary: if prescribed { DownstreamBoundary::PrescribedStage } else { DownstreamBoundary::NormalDepth },
        depth_lb: DEPTH_LB,
        bottom_width_lb: BW_LB,
        dt: 3600.0,
        g: 9.81,
        use_cuda: false,
    }
}

struct Consts {
    host: Arc<StageOpConsts>,
    dev: Arc<StageOpConstsDev>,
    pattern: Arc<CsrPattern>,
}

/// The consts of a `StageBlock::<Gpu>` on the network, as host and device.
fn consts_of(
    adj: &SparseAdjacency,
    pattern: &Arc<CsrPattern>,
    seeds: Option<&[usize]>,
    table: Option<ChannelGeometryRows>,
    p: StageParams,
) -> Consts {
    consts_with(adj, pattern, seeds, &[], table, p)
}

/// `consts_of` with dam rows `(row, residence days)`.
fn consts_with(
    adj: &SparseAdjacency,
    pattern: &Arc<CsrPattern>,
    seeds: Option<&[usize]>,
    dams: &[(usize, f32)],
    table: Option<ChannelGeometryRows>,
    p: StageParams,
) -> Consts {
    let net = StageNetwork::build(adj, pattern, seeds, 0.0, dams).unwrap();
    let mut block = StageBlock::<Gpu>::new(net, p, &gpu());
    if let Some(rows) = table {
        block.set_table_geometry(rows, false).unwrap();
    }
    let host = block.op_consts(pattern);
    let dev = block.op_consts_dev(pattern);
    Consts { host, dev, pattern: pattern.clone() }
}

/// Edit a host copy of the consts and upload it directly.
fn edited(c: &Consts, edit: impl FnOnce(&mut StageOpConsts)) -> Consts {
    let mut h = (*c.host).clone();
    edit(&mut h);
    let host = Arc::new(h);
    let dev = Arc::new(StageOpConstsDev::upload(&host, &c.pattern, &like()));
    Consts { host, dev, pattern: c.pattern.clone() }
}

/// Hotstart stage of the network on a throwaway NdArray block.
// copied from tests/stage_op_parity.rs::juniata_case (hotstart part)
fn hotstart_stage(adj: &SparseAdjacency, pattern: &Arc<CsrPattern>) -> (Vec<f32>, Vec<f32>) {
    let n = adj.n;
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
    let dev = Default::default();
    let tv = |v: &[f32]| Tensor::<Autodiff<Cpu>, 1>::from_floats(v, &dev);
    let net = StageNetwork::build(adj, pattern, None, 0.0, &[]).unwrap();
    let mut hb = StageBlock::<Cpu>::new(net, params(1.0, 1, false), &dev);
    let slope_f: Vec<f32> = adj.slope.iter().map(|s| s.max(1e-3)).collect();
    hb.hotstart(tv(&q_init), tv(&vec![0.035; n]), tv(&vec![21.0; n]), tv(&vec![0.5; n]), tv(&slope_f));
    let h0: Vec<f32> = hb.stage().unwrap().into_data().to_vec().unwrap();
    (h0, q_init)
}

// copied from tests/stage_op_parity.rs::table_and_mixed_cases (table_rect: table(&[2], 80.0, &[0.3], 0.02, &[None]))
fn table_rows() -> ChannelGeometryRows {
    ChannelGeometryRows {
        rows: vec![2],
        top_width: vec![80.0],
        bed_z: vec![0.3],
        n: vec![0.02],
        pool_z: vec![None],
    }
}

fn non_outlets(c: &StageOpConsts) -> Vec<usize> {
    (0..c.n).filter(|&i| c.down[i] != i).collect()
}

// ------------------------------------------------------------------ tests

fn assert_bits(what: &str, dev: &Cube, host: &[f32], len_if_empty: usize) {
    let got = cube_to_host(dev);
    if host.is_empty() {
        assert_eq!(got, vec![0.0; len_if_empty], "{what}: empty host Vec must upload as zeros");
    } else {
        assert_eq!(got.len(), host.len(), "{what}: length");
        for (i, (d, h)) in got.iter().zip(host).enumerate() {
            assert!(d.to_bits() == h.to_bits(), "{what}[{i}]: dev {d} vs host {h}");
        }
    }
}

#[test]
fn consts_dev_match_host() {
    let (adj, pattern) = juniata();
    let lower: Vec<usize> = (adj.n / 2..adj.n).collect();
    let cj = consts_of(&adj, &pattern, Some(&lower), None, params(0.6, 4, false));
    let (adj3, pat3) = chain(3);
    let rows = table_rows();
    let ct = consts_of(&adj3, &pat3, None, Some(rows), params(1.0, 1, true));
    for (label, c) in [("juniata", &cj), ("table chain", &ct)] {
        let (h, d) = (c.host.as_ref(), c.dev.as_ref());
        let (n, nnz) = (h.n, h.nnz);
        let down: Vec<i32> = h.down.iter().map(|&x| x as i32).collect();
        assert_eq!(cube_to_host_i32(&d.down), down, "{label}: down");
        let t = tree_levels(&c.pattern).unwrap();
        assert_eq!(cube_to_host_i32(&d.tree.crow), t.crow, "{label}: tree.crow");
        assert_eq!(cube_to_host_i32(&d.tree.col), t.col, "{label}: tree.col");
        assert_eq!(cube_to_host_i32(&d.tree.down), t.down, "{label}: tree.down");
        for (name, dv, hv) in [
            ("in_stage", &d.in_stage, &h.in_stage),
            ("friction", &d.friction, &h.friction),
            ("dam", &d.dam, &h.dam),
            ("ctrl", &d.ctrl, &h.ctrl),
            ("nd", &d.nd, &h.nd),
            ("w_ctrl", &d.w_ctrl, &h.w_ctrl),
            ("w_edge", &d.w_edge, &h.w_edge),
            ("has_down", &d.has_down, &h.has_down),
            ("outlet", &d.outlet, &h.outlet),
            ("z_mid", &d.z_mid, &h.z_mid),
            ("z_down", &d.z_down, &h.z_down),
            ("length", &d.length, &h.length),
            ("edge_length", &d.edge_length, &h.edge_length),
            ("dam_t", &d.dam_t, &h.dam_t),
            ("s0_sqrt", &d.s0_sqrt, &h.s0_sqrt),
        ] {
            assert_bits(&format!("{label}: {name}"), dv, hv, n);
        }
        for (name, dv, hv) in [
            ("t_mask", &d.t_mask, &h.t_mask),
            ("t_tw", &d.t_tw, &h.t_tw),
            ("t_n", &d.t_n, &h.t_n),
            ("t_pool", &d.t_pool, &h.t_pool),
        ] {
            assert_bits(&format!("{label}: {name}"), dv, hv, 1);
        }
        for (name, dv, hv) in [
            ("diag_slot", &d.diag_slot, &h.diag_slot),
            ("adj_stage", &d.adj_stage, &h.adj_stage),
            ("adj_mc", &d.adj_mc, &h.adj_mc),
            ("adj_friction", &d.adj_friction, &h.adj_friction),
            ("dam_slot", &d.dam_slot, &h.dam_slot),
        ] {
            assert_bits(&format!("{label}: {name}"), dv, hv, nnz);
        }
        println!("{label}: n {n}, nnz {nnz}, every array bitwise equal");
    }
    assert!(ct.host.has_table && ct.dev.t_mask.meta.shape()[0] == ct.host.n);
    assert_eq!(cj.dev.t_mask.meta.shape()[0], 1, "no table: length-1 dummy");
}

fn compare_hour(what: &str, dev: &HourDev, host: &HourHost) {
    let d = hour_to_host(dev);
    close(&format!("{what} p"), &d.p, &host.p);
    close(&format!("{what} q_eps"), &d.q_eps, &host.q_eps);
    close(&format!("{what} n_eff"), &d.n_eff, &host.n_eff);
    close(&format!("{what} n2"), &d.n2, &host.n2);
    close(&format!("{what} qp_stage"), &d.qp_stage, &host.qp_stage);
    close(&format!("{what} i_t"), &d.i_t, &host.i_t);
    close(&format!("{what} i_delta"), &d.i_delta, &host.i_delta);
    close(&format!("{what} q_next_mc"), &d.q_next_mc, &host.q_next_mc);
    assert_eq!((d.b0, d.b1), (host.b0, host.b1), "{what}: b0, b1");
}

fn prologue_case(what: &str, c: &Consts) {
    let n = c.host.n;
    let nm = gen(1, n, 0.02, 0.08);
    let q = gen(2, n, 0.3, 0.7);
    let p = gen(3, n, 5.0, 40.0);
    let q_t = gen(4, n, 0.5, 30.0);
    let q_next_mc = gen(5, n, 0.5, 30.0);
    let qp = gen(6, n, 0.0, 3.0);
    let (b0, b1) = (0.25f32, 0.75f32);
    let host = stage_prologue_host(&c.host, &nm, &q, &p, &q_t, &q_next_mc, &qp, b0, b1);
    let dev = stage_prologue_dev(&c.dev, &up(&nm), &up(&q), &up(&p), &up(&q_t), &up(&q_next_mc), &up(&qp), b0, b1);
    compare_hour(what, &dev, &host);
    // i_mid: the host oracle is stage_assemble_host on the same consts, hour and a state.
    let subs = c.host.substeps.max(1) as usize;
    let hd = HourDev::upload(&host, &like());
    let h: Vec<f32> = c.host.z_mid.iter().map(|z| z + 0.5).collect();
    for s in 0..subs {
        let asm = stage_assemble_host(&c.host, &host, s, &h, &q_t);
        close(&format!("{what} i_mid s={s}"), &cube_to_host(&i_mid_dev(&c.dev, &hd, s)), &asm.i_mid);
    }
}

#[test]
fn prologue_matches_host() {
    let (adj, pattern) = juniata();
    let lower: Vec<usize> = (adj.n / 2..adj.n).collect();
    let c = consts_of(&adj, &pattern, Some(&lower), None, params(0.6, 4, false));
    assert!(c.host.adj_mc.iter().any(|&w| w != 0.0), "MC-to-stage slots must exist");
    // A confluence: a row with two upstream reaches.
    assert!((0..adj.n).any(|i| (pattern.crow[i + 1] - pattern.crow[i]) as usize >= 3));
    prologue_case("juniata", &c);

    // Table chain of table_and_mixed_cases()[0] (copied from tests/stage_op_parity.rs).
    let (adj3, pat3) = chain(3);
    let rows = table_rows();
    let ct = consts_of(&adj3, &pat3, None, Some(rows), params(1.0, 1, true));
    assert!(ct.host.has_table);
    prologue_case("table chain", &ct);
}

fn compare_depth(what: &str, dev: &DepthDev, host: &DepthOut) {
    let d = depth_to_host(dev);
    close(&format!("{what} x"), &d.x, &host.x);
    close(&format!("{what} depth"), &d.depth, &host.depth);
    close(&format!("{what} h_down"), &d.h_down, &host.h_down);
    close(&format!("{what} der"), &d.der, &host.der);
    close(&format!("{what} d_edge"), &d.d_edge, &host.d_edge);
    exact(&format!("{what} s_up"), &d.s_up, &host.s_up);
    exact(&format!("{what} s_dn"), &d.s_dn, &host.s_dn);
    same_zero_set(&format!("{what} depth"), &d.depth, &host.depth);
    same_zero_set(&format!("{what} d_edge"), &d.d_edge, &host.d_edge);
}

#[test]
fn depth_matches_host() {
    let (adj, pattern) = juniata();
    let n = adj.n;
    let c = consts_of(&adj, &pattern, None, None, params(1.0, 1, false));
    let (h0, _) = hotstart_stage(&adj, &pattern);
    let pert = bar(7, n, false);
    let no = non_outlets(&c.host);
    let (f1, f2, a, b, cc) = (no[60], no[80], no[5], no[20], no[40]);
    // At the bed elevations of the bundle (about 143 m) the f32 grid has no h with
    // h - z == 0.01, so the tie rows get z_mid = z_down = 0 in the host copy of the
    // consts (uploaded directly). Then h = lb gives x == lb and der == lb exactly.
    let c = edited(&c, |h| {
        h.z_mid[a] = 0.0;
        h.z_mid[b] = 0.0;
        h.z_down[b] = 0.0;
    });
    let mut h: Vec<f32> = (0..n).map(|i| h0[i] + 0.3 * (pert[i] - 1.0)).collect();
    let (zm, lb) = (&c.host.z_mid, DEPTH_LB);
    let (bd, cd) = (c.host.down[b], c.host.down[cc]);
    let rows = [f1, f2, a, b, bd, cc, cd];
    for (i, x) in rows.iter().enumerate() {
        assert!(rows.iter().skip(i + 1).all(|y| y != x), "tie rows must be distinct: {rows:?}");
    }
    // The floor binds on two rows.
    h[f1] = zm[f1] - 0.5;
    h[f2] = zm[f2] - 0.2;
    // x == lb exactly.
    h[a] = lb;
    // der == lb exactly, with h[b] >= h[down b] so that mh = h[b].
    h[b] = lb;
    h[bd] = h[b] - 0.5;
    // h == hd on an interior row.
    h[cc] = h[cd];
    let host = depth_host(&c.host, &h, 0.0);
    assert!(host.x[f1] < lb && host.x[f2] < lb, "floor rows must bind");
    assert_eq!(host.x[a], lb, "x tie");
    assert_eq!(host.der[b], lb, "der tie");
    assert_eq!(host.h_down[cc], h[cc], "h == hd tie");
    assert!(non_outlets(&c.host).len() < n, "an outlet row exists");
    let dev = depth_dev(&c.dev, &up(&h), 0.0);
    compare_depth("juniata", &dev, &host);

    // PrescribedStage 3-chain with hb0 = 0.4: the outlet reads hb0.
    let (adj3, pat3) = chain(3);
    let c3 = consts_of(&adj3, &pat3, None, None, params(1.0, 1, true));
    assert!(c3.host.prescribed);
    let h3: Vec<f32> = c3.host.z_mid.iter().zip([0.8f32, 0.5, 0.3]).map(|(z, d)| z + d).collect();
    let host3 = depth_host(&c3.host, &h3, 0.4);
    assert_eq!(host3.h_down[2], 0.4);
    compare_depth("prescribed 3-chain", &depth_dev(&c3.dev, &up(&h3), 0.4), &host3);
}

// copied from tests/stage_op_adjoint.rs::section_cases
fn section_cases() -> (Vec<f32>, Vec<f32>, Vec<f32>) {
    let pq = [(21.0f32, 0.5f32), (1000.0, 0.0), (200.0, 0.9), (5.0, 0.9)];
    let depths = [0.02f32, 0.3, 1.0, 5.0];
    let (mut p, mut q, mut d) = (vec![], vec![], vec![]);
    for (pp, qq) in pq {
        for dd in depths {
            p.push(pp);
            q.push(qq);
            d.push(dd);
        }
    }
    (p, q, d)
}

fn section_run(label: &str, with_table: bool) {
    let (p, q, d) = section_cases();
    let n = p.len();
    let (adj, pat) = chain(n);
    let base = consts_of(&adj, &pat, None, None, params(1.0, 1, false));
    // copied from tests/stage_op_adjoint.rs::section_run: every other reach is a table row.
    let c = if with_table {
        edited(&base, |h| {
            h.has_table = true;
            h.t_mask = (0..n).map(|i| (i % 2) as f32).collect();
            h.t_tw = vec![500.0; n];
            h.t_n = vec![0.03; n];
            h.t_pool = vec![0.0; n];
        })
    } else {
        base
    };
    let q_eps: Vec<f32> = q.iter().map(|v| v + 1e-6f32).collect();
    let host = section_host(&c.host, &p, &q_eps, &d);
    let sd = section_dev(&c.dev, &up(&p), &up(&q_eps), &up(&d));
    let s = section_to_host(&sd);
    let t = |f: &str| format!("{label} {f}");
    close(&t("pw"), &s.pw, &host.pw);
    close(&t("tw"), &s.tw, &host.tw);
    close(&t("ss_raw"), &s.ss_raw, &host.ss_raw);
    close(&t("ss"), &s.ss, &host.ss);
    close(&t("bw_raw"), &s.bw_raw, &host.bw_raw);
    close(&t("bw"), &s.bw, &host.bw);
    close(&t("area"), &s.area, &host.area);
    close(&t("root"), &s.root, &host.root);
    close(&t("perim"), &s.perim, &host.perim);
    close(&t("radius"), &s.radius, &host.radius);
    close(&t("tw_d"), &s.tw_d, &host.tw_d);
    close(&t("ss_d"), &s.ss_d, &host.ss_d);
    close(&t("bw_d"), &s.bw_d, &host.bw_d);
    close(&t("a_d"), &s.a_d, &host.a_d);
    close(&t("p_d"), &s.p_d, &host.p_d);
    close(&t("out_tw"), &s.out_tw, &host.out_tw);
    close(&t("out_area"), &s.out_area, &host.out_area);
    close(&t("out_perim"), &s.out_perim, &host.out_perim);
    close(&t("out_radius"), &s.out_radius, &host.out_radius);
    close(&t("out_a_d"), &s.out_a_d, &host.out_a_d);
    close(&t("out_p_d"), &s.out_p_d, &host.out_p_d);
    same_zero_set(&t("bw"), &s.bw, &host.bw);
    same_zero_set(&t("ss"), &s.ss, &host.ss);
    // Both clamps and the interior are exercised (same grid as the adjoint test).
    assert!(host.ss_raw.iter().any(|&v| v < 0.5) && host.ss_raw.iter().any(|&v| v > 50.0));
    assert!(host.bw_raw.iter().any(|&v| v < BW_LB));
    // 2h: the host oracle is stage_assemble_host. The state h = z_mid + d gives depth = d
    // (every d of the grid is above the depth floor), so asm.store is this section.
    let hour = HourHost {
        p: p.clone(),
        q_eps: q_eps.clone(),
        n_eff: vec![0.03; n],
        n2: vec![0.03 * 0.03; n],
        qp_stage: vec![0.0; n],
        i_t: vec![0.0; n],
        i_delta: vec![0.0; n],
        q_next_mc: vec![0.0; n],
        b0: 0.0,
        b1: 0.0,
    };
    let h: Vec<f32> = (0..n).map(|i| c.host.z_mid[i] + d[i]).collect();
    let asm = stage_assemble_host(&c.host, &hour, 0, &h, &vec![1.0; n]);
    close(&t("asm.store.out_tw"), &asm.store.out_tw, &host.out_tw);
    let store = SectionDev::upload(&asm.store, &like());
    close(&t("storage_rate"), &cube_to_host(&storage_rate_dev(&c.dev, &store, asm.dt_s)), &asm.d_s);
}

#[test]
fn section_matches_host() {
    section_run("section", false);
}

#[test]
fn section_with_a_table_matches_host() {
    section_run("section+table", true);
}

fn compare_newton(what: &str, dev: &NewtonDev, host: &ddrs::routing::stage_op::NewtonOut) {
    let d = newton_to_host(dev);
    close(&format!("{what} phi"), &d.phi, &host.phi);
    close(&format!("{what} dt_phi"), &d.dt_phi, &host.dt_phi);
    close(&format!("{what} den"), &d.den, &host.den);
    close(&format!("{what} alpha"), &d.alpha, &host.alpha);
    close(&format!("{what} beta"), &d.beta, &host.beta);
    close(&format!("{what} phi_d"), &d.phi_d, &host.phi_d);
    close(&format!("{what} dh_edge"), &d.dh_edge, &host.dh_edge);
    close(&format!("{what} k1"), &d.k1, &host.k1);
    close(&format!("{what} kappa"), &d.kappa, &host.kappa);
    close(&format!("{what} q_star"), &d.q_star, &host.q_star);
    close(&format!("{what} ra"), &d.ra, &host.ra);
    close(&format!("{what} rb"), &d.rb, &host.rb);
    close(&format!("{what} a_up"), &d.a_up, &host.a_up);
    close(&format!("{what} b_down"), &d.b_down, &host.b_down);
    same_zero_set(&format!("{what} a_up"), &d.a_up, &host.a_up);
    same_zero_set(&format!("{what} b_down"), &d.b_down, &host.b_down);
}

#[test]
fn newton_matches_host() {
    let (adj, pattern) = juniata();
    let n = adj.n;
    let (h0, q0) = hotstart_stage(&adj, &pattern);
    let pert = bar(9, n, false);
    let h: Vec<f32> = (0..n).map(|i| h0[i] + 0.2 * (pert[i] - 1.0)).collect();
    let dt_s = 3600.0f32;
    for theta in [0.55f32, 1.0] {
        let c = consts_of(&adj, &pattern, None, None, params(theta, 1, false));
        let no = non_outlets(&c.host);
        let mut qc = q0.clone();
        for &i in &[no[3], no[30], no[90]] {
            qc[i] = 0.0;
        }
        for &i in &[no[7], no[50], no[120]] {
            qc[i] = -qc[i];
        }
        let n_eff = gen(11, n, 0.02, 0.06);
        let n2: Vec<f32> = n_eff.iter().map(|v| v * v).collect();
        let p = vec![21.0f32; n];
        let q_eps: Vec<f32> = vec![0.5f32 + 1e-6; n];
        let dep = depth_host(&c.host, &h, 0.0);
        let edge = section_host(&c.host, &p, &q_eps, &dep.d_edge);
        let host = newton_host(&c.host, dt_s, &h, &qc, &n2, &dep, &edge);
        // The |Q| tie rows, and ra, rb off zero by more than 1e-6 relative.
        for &i in &[no[3], no[30], no[90]] {
            assert_eq!(qc[i], 0.0);
        }
        for i in 0..n {
            let scale = host.beta[i].abs() * theta + host.kappa[i].abs();
            // Rows with friction 0 (the outlet) give ra = rb = 0 exactly on both backends: no tie.
            if c.host.friction[i] == 0.0 {
                assert_eq!((host.ra[i], host.rb[i]), (0.0, 0.0), "row {i}");
                continue;
            }
            assert!(
                host.ra[i].abs() > 1e-6 * scale && host.rb[i].abs() > 1e-6 * scale,
                "row {i}: ra {} rb {} within 1e-6 of 0 (scale {scale})",
                host.ra[i],
                host.rb[i]
            );
        }
        // The 3f clamps bind on friction rows (not only the outlet's structural zero).
        let live: Vec<usize> = (0..n).filter(|&i| c.host.friction[i] != 0.0).collect();
        let (neg_a, neg_b) = (
            live.iter().filter(|&&i| host.ra[i] < 0.0).count(),
            live.iter().filter(|&&i| host.rb[i] < 0.0).count(),
        );
        println!("newton theta {theta}: friction rows with ra < 0: {neg_a}, with rb < 0: {neg_b}");
        assert!(neg_a > 0 && neg_b > 0, "3f clamps must bind: ra<0 rows {neg_a}, rb<0 rows {neg_b}");
        let dev = newton_dev(
            &c.dev,
            dt_s,
            &up(&h),
            &up(&qc),
            &up(&n2),
            &DepthDev::upload(&dep, &like()),
            &SectionDev::upload(&edge, &like()),
        );
        compare_newton(&format!("newton theta {theta}"), &dev, &host);
    }
}

/// Inputs of one control block case.
struct ControlCase {
    c: Consts,
    dt_s: f32,
    h: Vec<f32>,
    p: Vec<f32>,
    q_eps: Vec<f32>,
    n_eff: Vec<f32>,
    depth: Vec<f32>,
    d_s: Vec<f32>,
    store: SectionHost,
}

// copied from tests/stage_op_adjoint.rs::control_run (inputs and edits)
fn control_case(with_table: bool) -> ControlCase {
    let n = 8;
    let (adj, pat) = chain(n);
    let base = consts_of(&adj, &pat, None, None, params(0.6, 1, false));
    let c = edited(&base, |c| {
        c.length = vec![2000.0; n];
        c.dam_t = vec![106272.0; n];
        c.s0_sqrt = vec![0.03; n];
        c.dam = vec![1.0, 0.0, 0.0, 1.0, 1.0, 1.0, 0.0, 1.0];
        c.nd = vec![0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0];
        c.has_table = with_table;
        if with_table {
            c.t_mask = vec![0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 0.0, 0.0];
            c.t_tw = vec![60.0; n];
            c.t_n = vec![0.03; n];
            c.t_pool = vec![0.0, 0.0, 0.0, 2.5, 2.5, 2.5, 0.0, 0.0];
        } else {
            c.t_mask = vec![];
            c.t_tw = vec![];
            c.t_n = vec![];
            c.t_pool = vec![];
        }
    });
    let h = vec![3.0f32, 2.0, 1.5, 3.5, 1.5, 2.5, 2.2, 3.1];
    let p = vec![21.0f32, 25.0, 18.0, 30.0, 22.0, 19.0, 24.0, 26.0];
    let q = [0.5f32, 0.4, 0.6, 0.5, 0.45, 0.55, 0.35, 0.65];
    let n_eff = vec![0.03f32, 0.035, 0.04, 0.03, 0.05, 0.045, 0.03, 0.04];
    let depth = vec![1.2f32, 0.8, 0.5, 1.5, 0.4, 1.0, 0.7, 1.3];
    let d_s = vec![50.0f32, 40.0, 30.0, 60.0, 25.0, 45.0, 35.0, -20.0];
    let a_s = [40.0f32, 30.0, 20.0, 50.0, 15.0, 35.0, 10.0, 28.0];
    let p_s = [30.0f32, 25.0, 20.0, 35.0, 18.0, 28.0, 20.0, 26.0];
    let r_s: Vec<f32> = (0..n).map(|i| a_s[i] / p_s[i]).collect();
    let ad_s = [20.0f32, 18.0, 15.0, 25.0, 12.0, 22.0, 4.0, 17.0];
    let pd_s = [2.1f32, 2.0, 2.0, 2.2, 2.0, 2.1, 40.0, 2.0];
    let q_eps: Vec<f32> = q.iter().map(|v| v + 1e-6f32).collect();
    let store = ddrs::routing::stage_op::SectionHost {
        out_area: a_s.to_vec(),
        out_perim: p_s.to_vec(),
        out_radius: r_s,
        out_a_d: ad_s.to_vec(),
        out_p_d: pd_s.to_vec(),
        ..Default::default()
    };
    ControlCase { c, dt_s: 3600.0, h, p, q_eps, n_eff, depth, d_s, store }
}

fn control_run(with_table: bool) {
    let ControlCase { c, dt_s, h, p, q_eps, n_eff, depth, d_s, store } = control_case(with_table);
    let host = control_host(&c.host, dt_s, &h, &p, &q_eps, &n_eff, &depth, &store, &d_s);
    assert_eq!(host.r_prime[6], 0.0, "nd clamp row must bind");
    if with_table {
        assert_eq!(host.ex[5], 0.0, "tie row: h == pool exactly");
        assert!(host.ex[3] > 0.0 && host.ex[4] < 0.0);
    }
    let dev = control_dev(
        &c.dev,
        dt_s,
        &up(&h),
        &up(&p),
        &up(&q_eps),
        &up(&n_eff),
        &up(&depth),
        &SectionDev::upload(&store, &like()),
        &up(&d_s),
    );
    compare_control(if with_table { "control+table" } else { "control" }, &dev, &host, with_table);
}

fn compare_control(what: &str, dev: &ControlDev, host: &ControlOut, with_table: bool) {
    let d = control_to_host(dev);
    close(&format!("{what} rd_head"), &d.rd_head, &host.rd_head);
    close(&format!("{what} rpd_head"), &d.rpd_head, &host.rpd_head);
    close(&format!("{what} rd"), &d.rd, &host.rd);
    close(&format!("{what} rpd"), &d.rpd, &host.rpd);
    close(&format!("{what} kk"), &d.kk, &host.kk);
    close(&format!("{what} w"), &d.w, &host.w);
    close(&format!("{what} dk_dd"), &d.dk_dd, &host.dk_dd);
    close(&format!("{what} r_t"), &d.r_t, &host.r_t);
    close(&format!("{what} rpr"), &d.rpr, &host.rpr);
    close(&format!("{what} r_prime"), &d.r_prime, &host.r_prime);
    same_zero_set(&format!("{what} r_prime"), &d.r_prime, &host.r_prime);
    if with_table {
        close(&format!("{what} ex"), &d.ex, &host.ex);
        exact(&format!("{what} act"), &d.act, &host.act);
    } else {
        assert_eq!(d.ex.len(), 1, "no table: ex is a length-1 dummy");
        assert_eq!(d.act.len(), 1, "no table: act is a length-1 dummy");
    }
}

#[test]
fn control_matches_host() {
    control_run(false);
    control_run(true);
}

#[test]
fn op_consts_dev_is_cached_and_cleared() {
    let (adj, pat) = chain(3);
    let net = StageNetwork::build(&adj, &pat, None, 0.0, &[]).unwrap();
    let mut block = StageBlock::<Gpu>::new(net, params(1.0, 1, false), &gpu());
    let (a, b) = (block.op_consts_dev(&pat), block.op_consts_dev(&pat));
    assert!(Arc::ptr_eq(&a, &b), "a second call returns the cached handle");
    block.set_table_geometry(table_rows(), false).unwrap();
    let c = block.op_consts_dev(&pat);
    assert!(!Arc::ptr_eq(&a, &c), "set_table_geometry clears the cache");
    assert!(c.host.has_table);
    assert!(Arc::ptr_eq(&c, &block.op_consts_dev(&pat)));
}

// ------------------------------------------------- matrix, rhs, readout

/// The Juniata network with MC rows in its upper half, a perturbed hotstart
/// stage and the hour prologue (copied from `prologue_case` and `depth_matches_host`).
struct JuniataState {
    c: Consts,
    hour: HourHost,
    h: Vec<f32>,
    qc: Vec<f32>,
}

fn juniata_state(theta: f32, substeps: u32) -> JuniataState {
    let (adj, pattern) = juniata();
    let n = adj.n;
    let lower: Vec<usize> = (n / 2..n).collect();
    let c = consts_of(&adj, &pattern, Some(&lower), None, params(theta, substeps, false));
    let (h0, q_init) = hotstart_stage(&adj, &pattern);
    let pert = bar(7, n, false);
    let h: Vec<f32> = (0..n).map(|i| h0[i] + 0.3 * (pert[i] - 1.0)).collect();
    let nm = gen(1, n, 0.02, 0.08);
    let q = gen(2, n, 0.3, 0.7);
    let p = gen(3, n, 5.0, 40.0);
    let q_next_mc = gen(5, n, 0.5, 30.0);
    let qp = gen(6, n, 0.0, 3.0);
    let hour = stage_prologue_host(&c.host, &nm, &q, &p, &q_init, &q_next_mc, &qp, 0.0, 0.0);
    JuniataState { c, hour, h, qc: q_init }
}

#[test]
fn matrix_matches_host() {
    let case = |what: &str, c: &Consts, d_s: &[f32], a_up: &[f32], b_down: &[f32], r_prime: &[f32]| {
        let host = matrix_host(&c.host, d_s, a_up, b_down, r_prime);
        let dev = cube_to_host(&matrix_dev(&c.dev, &up(d_s), &up(a_up), &up(b_down), &up(r_prime)));
        assert_eq!(dev.len(), 2 * c.host.nnz);
        close(what, &dev, &host);
        same_zero_set(what, &dev, &host);
    };
    // Juniata state: a confluence row gives a sibling sum over two slots.
    let st = juniata_state(0.6, 4);
    let asm = stage_assemble_host(&st.c.host, &st.hour, 0, &st.h, &st.qc);
    let pat = &st.c.pattern;
    assert!((0..pat.n).any(|i| (pat.crow[i + 1] - pat.crow[i]) as usize >= 3), "a confluence row exists");
    assert!(asm.newton.a_up.iter().any(|&v| v == 0.0), "the a clamp binds somewhere");
    case("matrix juniata", &st.c, &asm.d_s, &asm.newton.a_up, &asm.newton.b_down, &asm.control.r_prime);
    close("matrix juniata vs asm.a_values", &asm.a_values, &matrix_host(&st.c.host, &asm.d_s, &asm.newton.a_up, &asm.newton.b_down, &asm.control.r_prime));

    // The dam 5-chain of readout_solve_run: MC headwater (row 0), a dam on row 2.
    let (adj, pat5) = chain(5);
    let c5 = consts_with(&adj, &pat5, Some(&[1]), &[(2, 1.23)], None, params(0.7, 1, true));
    assert!(c5.host.dam_slot.iter().any(|&v| v != 0.0), "a dam slot exists");
    let pos = |seed, lo: f32, hi: f32| gen(seed, 5, lo, hi);
    case("matrix dam chain", &c5, &pos(1, 1.0, 2.0), &pos(2, 0.0, 0.3), &pos(3, 0.0, 0.3), &pos(4, 0.0, 0.3));
}

#[test]
fn rhs_matches_host() {
    let case = |what: &str, c: &Consts, seed: u32, hb0: f32, hb1: f32| {
        let n = c.host.n;
        let qc = bar(seed, n, true);
        let q_star = bar(seed + 1, n, true);
        let r_t = bar(seed + 2, n, true);
        let b_down = gen(seed + 3, n, 0.0, 0.3);
        let qp = gen(seed + 4, n, 0.0, 3.0);
        let i_mid = bar(seed + 5, n, false);
        let host = rhs_host(&c.host, &qc, &q_star, &r_t, &b_down, &qp, &i_mid, hb0, hb1);
        let dev = rhs_dev(&c.dev, &up(&qc), &up(&q_star), &up(&r_t), &up(&b_down), &up(&qp), &up(&i_mid), hb0, hb1);
        close(what, &cube_to_host(&dev), &host);
    };
    // Juniata: its own state.
    let st = juniata_state(0.6, 4);
    let asm = stage_assemble_host(&st.c.host, &st.hour, 1, &st.h, &st.qc);
    let host = rhs_host(
        &st.c.host, &st.qc, &asm.newton.q_star, &asm.control.r_t, &asm.newton.b_down, &st.hour.qp_stage, &asm.i_mid,
        asm.hb0, asm.hb1,
    );
    let dev = rhs_dev(
        &st.c.dev, &up(&st.qc), &up(&asm.newton.q_star), &up(&asm.control.r_t), &up(&asm.newton.b_down),
        &up(&st.hour.qp_stage), &up(&asm.i_mid), asm.hb0, asm.hb1,
    );
    close("rhs juniata", &cube_to_host(&dev), &host);
    close("rhs juniata vs asm.r", &host, &asm.r);
    case("rhs juniata random", &st.c, 11, 0.0, 0.0);
    // PrescribedStage 3-chain: the outlet term (hb1 - hb0) is live.
    let (adj3, pat3) = chain(3);
    let cp = consts_of(&adj3, &pat3, None, None, params(0.6, 1, true));
    assert!(cp.host.prescribed);
    case("rhs prescribed 3-chain", &cp, 21, 0.4, 0.9);
    // MC-inflow 3-chain: row 0 is an MC reach.
    let cm = consts_of(&adj3, &pat3, Some(&[1]), None, params(0.55, 1, false));
    assert!(cm.host.in_stage[0] == 0.0 && cm.host.in_stage[1] == 1.0);
    case("rhs mc-inflow 3-chain", &cm, 31, 0.0, 0.0);
}

#[allow(clippy::too_many_arguments)]
fn compare_readout(
    what: &str,
    c: &Consts,
    asm_dev: &AssembledDev,
    hour: &HourHost,
    h: &[f32],
    qc: &[f32],
    dh: &[f32],
    asm: &Assembled,
) {
    let host = stage_readout_host(&c.host, hour, h, qc, asm, dh);
    let hd = HourDev::upload(hour, &like());
    let ro = stage_readout_dev(&c.dev, &hd, &up(h), &up(qc), asm_dev, &up(dh));
    let d = readout_to_host(&ro);
    close(&format!("{what} h_solved"), &d.h_solved, &host.h_solved);
    close(&format!("{what} h_new"), &d.h_new, &host.h_new);
    close(&format!("{what} dh_down"), &d.dh_down, &host.dh_down);
    close(&format!("{what} q_new"), &d.q_new, &host.q_new);
    same_zero_set(&format!("{what} h_new"), &d.h_new, &host.h_new);
    let tol = 1e-5 * host.mass[3].abs();
    for (k, name) in ["storage", "net", "floor", "gross", "mc_in"].iter().enumerate() {
        println!("{what} mass {name}: dev {} host {} (tol {tol:e})", d.mass[k], host.mass[k]);
        assert!(
            (d.mass[k] - host.mass[k]).abs() <= tol,
            "{what} mass {name}: dev {} vs host {} (tol {tol:e})",
            d.mass[k],
            host.mass[k]
        );
    }
}

// copied from tests/stage_op_adjoint.rs::readout_solve_run (inputs; the host oracle is stage_readout_host)
fn readout_chain_run(floored: bool) {
    let n = 5;
    let (adj, pat) = chain(n);
    let c = consts_with(&adj, &pat, Some(&[1]), &[(2, 1.23)], None, params(0.7, 1, true));
    assert!(c.host.prescribed && c.host.dam.iter().any(|&v| v == 1.0));
    let pos = |seed, lo: f32, hi: f32| gen(seed, n, lo, hi);
    let d_s = pos(1, 1.0, 2.0);
    let a_up = pos(2, 0.0, 0.3);
    let b_down = pos(3, 0.0, 0.3);
    let r_prime = pos(4, 0.0, 0.3);
    let r_t = bar(5, n, true);
    let q_star = bar(6, n, true);
    let qc = bar(7, n, true);
    let qp: Vec<f32> = bar(8, n, false).iter().zip(&c.host.in_stage).map(|(v, m)| v * m).collect();
    let i_mid = bar(9, n, false);
    let h: Vec<f32> = if floored {
        c.host.z_mid.iter().map(|z| z - 3.0).collect()
    } else {
        c.host.z_mid.iter().enumerate().map(|(i, z)| z + 0.5 + 0.1 * i as f32).collect()
    };
    let (hb0, hb1) = (0.4f32, 0.9f32);
    let a_values = matrix_host(&c.host, &d_s, &a_up, &b_down, &r_prime);
    let r = rhs_host(&c.host, &qc, &q_star, &r_t, &b_down, &qp, &i_mid, hb0, hb1);
    let f = tree_lu_factor(&c.pattern, &a_values);
    let dh = tree_lu_solve_host(&c.pattern, &f, &r);
    let hour = HourHost { qp_stage: qp.clone(), ..Default::default() };
    let asm = Assembled {
        i_mid: i_mid.clone(),
        hb0,
        hb1,
        dt_s: 3600.0,
        d_s: d_s.clone(),
        newton: NewtonOut { q_star, a_up, b_down, ..Default::default() },
        control: ControlOut { r_t, r_prime, ..Default::default() },
        ..Default::default()
    };
    let asm_dev = AssembledDev::upload(&asm, &like(), c.host.nnz);
    let label = if floored { "readout chain floored" } else { "readout chain" };
    compare_readout(label, &c, &asm_dev, &hour, &h, &qc, &dh, &asm);
    let host = stage_readout_host(&c.host, &hour, &h, &qc, &asm, &dh);
    let hf: Vec<f32> = c.host.z_mid.iter().map(|z| z + c.host.depth_lb).collect();
    if floored {
        assert_eq!(host.h_new, hf, "every row is on the floor");
    } else {
        assert!(host.h_new != hf, "not floored");
    }
    // The outlet row reads hb1 - hb0 (PrescribedStage).
    assert_eq!(host.dh_down[n - 1], hb1 - hb0);
}

#[test]
fn readout_and_mass_match_host() {
    readout_chain_run(false);
    readout_chain_run(true);

    // (b) the Juniata state, with ties and floor rows.
    let st = juniata_state(0.6, 4);
    let c = &st.c;
    let asm = stage_assemble_host(&c.host, &st.hour, 0, &st.h, &st.qc);
    let f = tree_lu_factor(&c.pattern, &asm.a_values);
    let mut dh = tree_lu_solve_host(&c.pattern, &f, &asm.r);
    let mut h = st.h.clone();
    let (zm, lb) = (&c.host.z_mid, c.host.depth_lb);
    let no = non_outlets(&c.host);
    // (c) two tie rows hs == hf (dh = 0, h = z_mid + lb) and two floor rows (hs < hf).
    for &t in &[no[10], no[30]] {
        h[t] = zm[t] + lb;
        dh[t] = 0.0;
    }
    for &t in &[no[50], no[70]] {
        h[t] = zm[t] - 0.5;
        dh[t] = 0.0;
    }
    let host = stage_readout_host(&c.host, &st.hour, &h, &st.qc, &asm, &dh);
    for &t in &[no[10], no[30]] {
        assert_eq!(h[t] + 0.0, zm[t] + lb, "tie: h + 0 == z_mid + lb on the host");
        assert_eq!(host.h_new[t], host.h_solved[t], "tie row keeps h_solved");
    }
    for &t in &[no[50], no[70]] {
        assert_eq!(host.h_new[t], zm[t] + lb, "floor row binds");
    }
    // A confluence row (two upstream reaches) feeds the `into` gather.
    let pat = &c.pattern;
    assert!((0..pat.n).any(|i| (pat.crow[i + 1] - pat.crow[i]) as usize >= 3));
    let asm_dev = AssembledDev::upload(&asm, &like(), c.host.nnz);
    compare_readout("readout juniata", c, &asm_dev, &st.hour, &h, &st.qc, &dh, &asm);
}

fn compare_section_host(what: &str, d: &SectionHost, h: &SectionHost) {
    macro_rules! f {
        ($($x:ident),*) => { $( close(&format!("{what} {}", stringify!($x)), &d.$x, &h.$x); )* };
    }
    f!(pw, tw, ss_raw, ss, bw_raw, bw, area, root, perim, radius, tw_d, ss_d, bw_d, a_d, p_d, out_tw, out_area,
       out_perim, out_radius, out_a_d, out_p_d);
}

#[test]
fn assemble_matches_host() {
    let st = juniata_state(0.6, 4);
    let c = &st.c;
    let hd = HourDev::upload(&st.hour, &like());
    for s in [0usize, 3] {
        let host = stage_assemble_host(&c.host, &st.hour, s, &st.h, &st.qc);
        let dev = stage_assemble_dev(&c.dev, &hd, s, &up(&st.h), &up(&st.qc));
        let d = assembled_to_host(&dev);
        let w = format!("assemble s={s}");
        close(&format!("{w} i_mid"), &d.i_mid, &host.i_mid);
        assert_eq!((d.hb0, d.hb1, d.dt_s), (host.hb0, host.hb1, host.dt_s), "{w}: hb0, hb1, dt_s");
        compare_depth(&format!("{w} depth"), &dev.depth, &host.depth);
        compare_section_host(&format!("{w} edge"), &d.edge, &host.edge);
        compare_section_host(&format!("{w} store"), &d.store, &host.store);
        close(&format!("{w} d_s"), &d.d_s, &host.d_s);
        compare_newton(&format!("{w} newton"), &dev.newton, &host.newton);
        compare_control(&format!("{w} control"), &dev.control, &host.control, false);
        close(&format!("{w} a_values"), &d.a_values, &host.a_values);
        close(&format!("{w} r"), &d.r, &host.r);
    }
}

// ------------------------------------------------- backward blocks (B8 to B5)

/// The backward kernels fuse the same sites as the host adjoint fns, so
/// every output field must match the host bit for bit.
fn bitwise(what: &str, dev: &[f32], host: &[f32]) {
    assert_eq!(dev.len(), host.len(), "{what}: length");
    let bad: Vec<String> = dev
        .iter()
        .zip(host)
        .enumerate()
        .filter(|(_, (d, h))| d.to_bits() != h.to_bits())
        .take(5)
        .map(|(i, (d, h))| format!("[{i}]: dev {d} host {h}"))
        .collect();
    println!("{what}: bitwise {}", if bad.is_empty() { "yes" } else { "NO" });
    assert!(bad.is_empty(), "{what}: not bitwise: {bad:?}");
}

thread_local! {
    /// Per field name (the last word of the label): did any case give a non-zero host value?
    static SEEN: std::cell::RefCell<std::collections::BTreeMap<String, bool>> = Default::default();
}

/// Record whether the host field has a non-zero value in this case.
fn note_nonzero(what: &str, host: &[f32]) {
    let key = what.rsplit(' ').next().unwrap_or(what).to_string();
    let nz = host.iter().any(|&v| v != 0.0);
    SEEN.with(|m| *m.borrow_mut().entry(key).or_insert(false) |= nz);
}

/// End of a test: every compared field was non-zero in at least one case, so a
/// fixture change cannot silently turn a field into an all-zero field. `zero_by_design`
/// names fields that are all zero in every case on purpose.
fn assert_fields_nonzero(zero_by_design: &[&str]) {
    SEEN.with(|m| {
        for (k, nz) in m.borrow().iter() {
            if zero_by_design.contains(&k.as_str()) {
                assert!(!*nz, "field {k} is listed as all zero but has a non-zero value");
            } else {
                assert!(*nz, "field {k} is zero in every case of this test");
            }
        }
        // Clear the map, so a field seen in one test cannot hide an all-zero field in another test on this thread.
        m.borrow_mut().clear();
    });
}

/// D7 bar, same zero set and bitwise, for one field.
fn all_bars(what: &str, dev: &[f32], host: &[f32]) {
    note_nonzero(what, host);
    close(what, dev, host);
    same_zero_set(what, dev, host);
    bitwise(what, dev, host);
}

/// D7 bar and same zero set for a field whose chain has a `powf` or `ln`. The CUDA
/// and glibc versions differ by a few ulp, so bitwise equality is reported, not asserted.
fn lib_bars(what: &str, dev: &[f32], host: &[f32]) {
    note_nonzero(what, host);
    close(what, dev, host);
    same_zero_set(what, dev, host);
    let nbad = dev.iter().zip(host).filter(|(d, h)| d.to_bits() != h.to_bits()).count();
    println!("{what}: bitwise {} ({nbad} of {} entries differ; libdevice powf/ln in the chain)", if nbad == 0 { "yes" } else { "NO" }, dev.len());
}

#[test]
fn output_backward_matches_host() {
    let st = juniata_state(0.6, 4);
    let c = &st.c;
    let n = c.host.n;
    assert!(c.host.in_stage.iter().any(|&v| v == 0.0) && c.host.in_stage.iter().any(|&v| v == 1.0), "MC rows present");
    let g_out = bar(31, 2 * n, true);
    let (hg, hq, hm) = output_backward_host(&c.host, &g_out);
    let (dg, dq, dm) = output_backward_dev(&c.dev, &up(&g_out));
    all_bars("output g_h", &cube_to_host(&dg), &hg);
    all_bars("output g_q", &cube_to_host(&dq), &hq);
    all_bars("output g_mc", &cube_to_host(&dm), &hm);
    assert_fields_nonzero(&[]);
}

/// Compare every field of `readout_backward_dev` with `readout_backward_host`.
#[allow(clippy::too_many_arguments)]
fn readout_bwd_case(
    what: &str,
    c: &Consts,
    asm: &Assembled,
    h: &[f32],
    dh: &[f32],
    g_h_new: &[f32],
    g_q_new: &[f32],
) -> ddrs::routing::stage_op::ReadoutGrads {
    let host = readout_backward_host(&c.host, h, asm, dh, g_h_new, g_q_new);
    let asm_dev = AssembledDev::upload(asm, &like(), c.host.nnz);
    let dev = readout_backward_dev(&c.dev, &up(h), &asm_dev, &up(dh), &up(g_h_new), &up(g_q_new));
    let d = readout_grads_to_host(&dev);
    all_bars(&format!("{what} dh"), &d.dh, &host.dh);
    all_bars(&format!("{what} h"), &d.h, &host.h);
    all_bars(&format!("{what} q_star"), &d.q_star, &host.q_star);
    all_bars(&format!("{what} a_up"), &d.a_up, &host.a_up);
    all_bars(&format!("{what} b_down"), &d.b_down, &host.b_down);
    all_bars(&format!("{what} r_t"), &d.r_t, &host.r_t);
    all_bars(&format!("{what} r_prime"), &d.r_prime, &host.r_prime);
    host
}

// copied from tests/stage_op_adjoint.rs::readout_solve_run (inputs; the host oracle is readout_backward_host)
fn readout_bwd_chain_run(floored: bool) {
    let n = 5;
    let (adj, pat) = chain(n);
    let c = consts_with(&adj, &pat, Some(&[1]), &[(2, 1.23)], None, params(0.7, 1, true));
    assert!(c.host.prescribed && c.host.dam.iter().any(|&v| v == 1.0));
    let pos = |seed, lo: f32, hi: f32| gen(seed, n, lo, hi);
    let d_s = pos(1, 1.0, 2.0);
    let a_up = pos(2, 0.0, 0.3);
    let b_down = pos(3, 0.0, 0.3);
    let r_prime = pos(4, 0.0, 0.3);
    let r_t = bar(5, n, true);
    let q_star = bar(6, n, true);
    let qc = bar(7, n, true);
    let qp: Vec<f32> = bar(8, n, false).iter().zip(&c.host.in_stage).map(|(v, m)| v * m).collect();
    let i_mid = bar(9, n, false);
    let h: Vec<f32> = if floored {
        c.host.z_mid.iter().map(|z| z - 3.0).collect()
    } else {
        c.host.z_mid.iter().enumerate().map(|(i, z)| z + 0.5 + 0.1 * i as f32).collect()
    };
    let (hb0, hb1) = (0.4f32, 0.9f32);
    let a_values = matrix_host(&c.host, &d_s, &a_up, &b_down, &r_prime);
    let r = rhs_host(&c.host, &qc, &q_star, &r_t, &b_down, &qp, &i_mid, hb0, hb1);
    let f = tree_lu_factor(&c.pattern, &a_values);
    let dh = tree_lu_solve_host(&c.pattern, &f, &r);
    let asm = Assembled {
        i_mid,
        hb0,
        hb1,
        dt_s: 3600.0,
        d_s,
        newton: NewtonOut { q_star, a_up, b_down, ..Default::default() },
        control: ControlOut { r_t, r_prime, ..Default::default() },
        ..Default::default()
    };
    let label = if floored { "readout bwd chain floored" } else { "readout bwd chain" };
    let host = readout_bwd_case(label, &c, &asm, &h, &dh, &bar(20, n, true), &bar(21, n, true));
    if floored {
        assert!(host.h.iter().all(|&v| v == 0.0), "every row is on the floor, so the floor passes nothing");
    } else {
        assert!(host.h.iter().all(|&v| v != 0.0), "no row is floored");
    }
}

#[test]
fn readout_backward_matches_host() {
    readout_bwd_chain_run(false);
    readout_bwd_chain_run(true);

    // NormalDepth 3-chain: the outlet self-adds in the gather's transpose.
    let n = 3;
    let (adj3, pat3) = chain(n);
    let c3 = consts_of(&adj3, &pat3, None, None, params(0.6, 1, false));
    assert!(!c3.host.prescribed && c3.host.down[n - 1] == n - 1);
    let asm3 = Assembled {
        hb0: 0.0,
        hb1: 0.0,
        newton: NewtonOut {
            q_star: bar(41, n, true),
            a_up: gen(42, n, 0.0, 0.3),
            b_down: gen(43, n, 0.0, 0.3),
            ..Default::default()
        },
        control: ControlOut { r_t: bar(44, n, true), r_prime: gen(45, n, 0.0, 0.3), ..Default::default() },
        ..Default::default()
    };
    let h3: Vec<f32> = c3.host.z_mid.iter().enumerate().map(|(i, z)| z + 0.3 + 0.2 * i as f32).collect();
    readout_bwd_case("readout bwd normal-depth 3-chain", &c3, &asm3, &h3, &bar(46, n, true), &bar(20, n, true), &bar(21, n, true));

    // Juniata: confluence rows, tie rows (hs == hf) and floor rows (hs < hf).
    let st = juniata_state(0.6, 4);
    let c = &st.c;
    let nn = c.host.n;
    let asm = stage_assemble_host(&c.host, &st.hour, 0, &st.h, &st.qc);
    let f = tree_lu_factor(&c.pattern, &asm.a_values);
    let mut dh = tree_lu_solve_host(&c.pattern, &f, &asm.r);
    let mut h = st.h.clone();
    let (zm, lb) = (&c.host.z_mid, c.host.depth_lb);
    let no = non_outlets(&c.host);
    for &t in &[no[10], no[30]] {
        h[t] = zm[t] + lb;
        dh[t] = 0.0;
    }
    for &t in &[no[50], no[70]] {
        h[t] = zm[t] - 0.5;
        dh[t] = 0.0;
    }
    let pat = &c.pattern;
    assert!((0..pat.n).any(|i| (pat.crow[i + 1] - pat.crow[i]) as usize >= 3), "a confluence row exists");
    let g_h_new = bar(20, nn, true);
    let g_q_new = bar(21, nn, true);
    let host = readout_bwd_case("readout bwd juniata", c, &asm, &h, &dh, &g_h_new, &g_q_new);
    for &t in &[no[10], no[30]] {
        assert_eq!(h[t] + dh[t], zm[t] + lb, "tie: hs == hf on the host");
        assert_eq!(host.h[t], g_h_new[t], "tie row passes the cotangent");
    }
    for &t in &[no[50], no[70]] {
        assert_eq!(host.h[t], 0.0, "floor row blocks the cotangent");
    }
    assert_fields_nonzero(&[]);
}

#[test]
fn rhs_backward_matches_host() {
    let case = |what: &str, c: &Consts, hb0: f32, hb1: f32| {
        let n = c.host.n;
        let asm = Assembled { hb0, hb1, ..Default::default() };
        let g_r = bar(22, n, true);
        let host = rhs_backward_host(&c.host, &asm, &g_r);
        let asm_dev = AssembledDev::upload(&asm, &like(), c.host.nnz);
        let d = rhs_grads_to_host(&rhs_backward_dev(&c.dev, &asm_dev, &up(&g_r)));
        all_bars(&format!("{what} qp"), &d.qp, &host.qp);
        all_bars(&format!("{what} i_mid"), &d.i_mid, &host.i_mid);
        all_bars(&format!("{what} q_star"), &d.q_star, &host.q_star);
        all_bars(&format!("{what} qc"), &d.qc, &host.qc);
        all_bars(&format!("{what} r_t"), &d.r_t, &host.r_t);
        all_bars(&format!("{what} b_down"), &d.b_down, &host.b_down);
        host
    };
    let st = juniata_state(0.6, 4);
    let pat = &st.c.pattern;
    assert!((0..pat.n).any(|i| (pat.crow[i + 1] - pat.crow[i]) as usize >= 3), "a confluence row exists");
    case("rhs bwd juniata", &st.c, 0.0, 0.0);
    // PrescribedStage 3-chain: the outlet term (hb1 - hb0) is live.
    let (adj3, pat3) = chain(3);
    let cp = consts_of(&adj3, &pat3, None, None, params(0.6, 1, true));
    assert!(cp.host.prescribed);
    let host = case("rhs bwd prescribed 3-chain", &cp, 0.4, 0.9);
    assert!(host.b_down[2] != 0.0, "the outlet b_down term is live");
    // The dam 5-chain of readout_solve_run: w_ctrl is non-zero on the dam row, so r_t is live.
    let (adj5, pat5) = chain(5);
    let c5 = consts_with(&adj5, &pat5, Some(&[1]), &[(2, 1.23)], None, params(0.7, 1, true));
    assert!(c5.host.w_ctrl.iter().any(|&v| v != 0.0), "a dam row has w_ctrl != 0");
    let host5 = case("rhs bwd dam 5-chain", &c5, 0.4, 0.9);
    assert!(host5.r_t.iter().any(|&v| v != 0.0));
    assert_fields_nonzero(&[]);
}

#[test]
fn matrix_backward_matches_host() {
    let case = |what: &str, c: &Consts| {
        let g_a = bar(23, 2 * c.host.nnz, true);
        let host = matrix_backward_host(&c.host, &g_a);
        let d = matrix_grads_to_host(&matrix_backward_dev(&c.dev, &up(&g_a)));
        all_bars(&format!("{what} d_s"), &d.d_s, &host.d_s);
        all_bars(&format!("{what} a_up"), &d.a_up, &host.a_up);
        all_bars(&format!("{what} b_down"), &d.b_down, &host.b_down);
        all_bars(&format!("{what} r_prime"), &d.r_prime, &host.r_prime);
    };
    let st = juniata_state(0.6, 4);
    let pat = &st.c.pattern;
    assert!((0..pat.n).any(|i| (pat.crow[i + 1] - pat.crow[i]) as usize >= 3), "a confluence row exists");
    case("matrix bwd juniata", &st.c);
    // The dam 5-chain of readout_solve_run: a dam on row 2 makes r_prime live.
    let (adj, pat5) = chain(5);
    let c5 = consts_with(&adj, &pat5, Some(&[1]), &[(2, 1.23)], None, params(0.7, 1, true));
    assert!(c5.host.dam_slot.iter().any(|&v| v != 0.0), "a dam slot exists");
    case("matrix bwd dam chain", &c5);
    assert_fields_nonzero(&[]);
}

// ------------------------------- backward blocks (B4 to B1), glue and prologue
//
// Bitwise expectation per field: a field whose chain has no `powf` or `ln` call must be
// bitwise equal to the host (`all_bars`). A field with one is held to the D7 bar and the
// same zero set, and its bitwise state is printed (`lib_bars`). The saved values the
// kernels read are uploaded from the host structs, so both sides start from the same bits.

fn section_bwd_run(label: &str, with_table: bool) {
    let (p, q, d) = section_cases();
    let n = p.len();
    let (adj, pat) = chain(n);
    let base = consts_of(&adj, &pat, None, None, params(1.0, 1, false));
    // copied from section_run: every other reach is a table row.
    let c = if with_table {
        edited(&base, |h| {
            h.has_table = true;
            h.t_mask = (0..n).map(|i| (i % 2) as f32).collect();
            h.t_tw = vec![500.0; n];
            h.t_n = vec![0.03; n];
            h.t_pool = vec![0.0; n];
        })
    } else {
        base
    };
    let q_eps: Vec<f32> = q.iter().map(|v| v + 1e-6f32).collect();
    let mut s = section_host(&c.host, &p, &q_eps, &d);
    // Ties come in through the saved values (powf results cannot be steered on both backends):
    // ss_raw exactly 0.5 and 50, bw_raw exactly the bottom-width floor. Rows 2, 4, 6 are head rows.
    s.ss_raw[2] = 0.5;
    s.ss_raw[4] = 50.0;
    s.bw_raw[6] = c.host.bw_lb;
    let sbar = SectionBar {
        top_width: bar(40, n, true),
        area: bar(41, n, true),
        perim: bar(42, n, true),
        radius: bar(43, n, true),
        a_d: bar(44, n, true),
        p_d: bar(45, n, true),
    };
    let host = section_backward_host(&c.host, &p, &q_eps, &d, &s, &sbar);
    let dev = section_backward_dev(
        &c.dev,
        &up(&p),
        &up(&q_eps),
        &up(&d),
        &SectionDev::upload(&s, &like()),
        &SectionBarDev::upload(&sbar, &like()),
    );
    let g = section_grads_to_host(&dev);
    // p has no powf or ln in its chain (the saved pw is read); q_eps and d do.
    all_bars(&format!("{label} p"), &g.p, &host.p);
    lib_bars(&format!("{label} q_eps"), &g.q_eps, &host.q_eps);
    lib_bars(&format!("{label} d"), &g.d, &host.d);
    if with_table {
        for i in (1..n).step_by(2) {
            assert_eq!((host.p[i], host.q_eps[i]), (0.0, 0.0), "host table row {i}: p and q gradients exactly 0");
            assert_eq!((g.p[i], g.q_eps[i]), (0.0, 0.0), "device table row {i}: p and q gradients exactly 0");
        }
    }
}

#[test]
fn section_backward_matches_host() {
    section_bwd_run("section bwd", false);
    section_bwd_run("section bwd+table", true);
    assert_fields_nonzero(&[]);
}

#[test]
fn newton_backward_matches_host() {
    let (adj, pattern) = juniata();
    let n = adj.n;
    let (h0, q0) = hotstart_stage(&adj, &pattern);
    let pert = bar(9, n, false);
    let h: Vec<f32> = (0..n).map(|i| h0[i] + 0.2 * (pert[i] - 1.0)).collect();
    let dt_s = 3600.0f32;
    for theta in [0.55f32, 1.0] {
        let c = consts_of(&adj, &pattern, None, None, params(theta, 1, false));
        let no = non_outlets(&c.host);
        let mut qc = q0.clone();
        for &i in &[no[3], no[30], no[90]] {
            qc[i] = 0.0;
        }
        for &i in &[no[7], no[50], no[120]] {
            qc[i] = -qc[i];
        }
        let n_eff = gen(11, n, 0.02, 0.06);
        let n2: Vec<f32> = n_eff.iter().map(|v| v * v).collect();
        let p = vec![21.0f32; n];
        let q_eps: Vec<f32> = vec![0.5f32 + 1e-6; n];
        let dep = depth_host(&c.host, &h, 0.0);
        let edge = section_host(&c.host, &p, &q_eps, &dep.d_edge);
        let mut o = newton_host(&c.host, dt_s, &h, &qc, &n2, &dep, &edge);
        // Saved ra = rb = 0 exactly on two rows: the tie of the 3f clamps.
        for &i in &[no[5], no[40]] {
            o.ra[i] = 0.0;
            o.rb[i] = 0.0;
        }
        let nbar = NewtonBar { q_star: bar(47, n, true), a_up: bar(48, n, true), b_down: bar(49, n, true) };
        let host = newton_backward_host(&c.host, dt_s, &h, &qc, &n2, &dep, &edge, &o, &nbar);
        let dev = newton_backward_dev(
            &c.dev,
            dt_s,
            &up(&h),
            &up(&qc),
            &up(&n2),
            &DepthDev::upload(&dep, &like()),
            &SectionDev::upload(&edge, &like()),
            &NewtonDev::upload(&o, &like()),
            &NewtonBarDev::upload(&nbar, &like()),
        );
        let g = newton_grads_to_host(&dev);
        let w = format!("newton bwd theta {theta}");
        // qc and n2 carry the powf of R^(4/3); the other seven fields do not.
        lib_bars(&format!("{w} qc"), &g.qc, &host.qc);
        lib_bars(&format!("{w} n2"), &g.n2, &host.n2);
        all_bars(&format!("{w} h"), &g.h, &host.h);
        all_bars(&format!("{w} h_down"), &g.h_down, &host.h_down);
        all_bars(&format!("{w} edge_area"), &g.edge.area, &host.edge.area);
        all_bars(&format!("{w} edge_perim"), &g.edge.perim, &host.edge.perim);
        all_bars(&format!("{w} edge_radius"), &g.edge.radius, &host.edge.radius);
        all_bars(&format!("{w} edge_a_d"), &g.edge.a_d, &host.edge.a_d);
        all_bars(&format!("{w} edge_p_d"), &g.edge.p_d, &host.edge.p_d);
        assert!(g.edge.top_width.is_empty() && host.edge.top_width.is_empty());
    }
    assert_fields_nonzero(&[]);
}

fn control_bwd_run(with_table: bool) {
    let ControlCase { c, dt_s, h, p, q_eps, n_eff, depth, d_s, store } = control_case(with_table);
    let mut o = control_host(&c.host, dt_s, &h, &p, &q_eps, &n_eff, &depth, &store, &d_s);
    // Saved rpr exactly 0 on row 1: the tie of the 4d clamp. Row 6's clamp binds on its own,
    // and with a table, row 5 keeps the ex == 0 tie of control_run.
    o.rpr[1] = 0.0;
    if with_table {
        assert_eq!(o.ex[5], 0.0, "tie row: h == pool exactly");
    }
    let n = h.len();
    let cbar = ControlBar { r_t: bar(50, n, true), r_prime: bar(51, n, true) };
    let host = control_backward_host(&c.host, dt_s, &h, &p, &q_eps, &n_eff, &depth, &store, &d_s, &o, &cbar);
    let dev = control_backward_dev(
        &c.dev,
        dt_s,
        &up(&h),
        &up(&p),
        &up(&q_eps),
        &up(&n_eff),
        &up(&depth),
        &SectionDev::upload(&store, &like()),
        &up(&d_s),
        &ControlDev::upload(&o, &like()),
        &ControlBarDev::upload(&cbar, &like()),
    );
    let g = control_grads_to_host(&dev);
    let w = if with_table { "control bwd+table" } else { "control bwd" };
    // p, depth and q_eps carry the powf of the 4a power; store.area and store.radius carry the
    // powf of R^(2/3). h, n_eff, d_s and the other three store fields do not.
    lib_bars(&format!("{w} p"), &g.p, &host.p);
    lib_bars(&format!("{w} depth"), &g.depth, &host.depth);
    lib_bars(&format!("{w} q_eps"), &g.q_eps, &host.q_eps);
    lib_bars(&format!("{w} store_area"), &g.store.area, &host.store.area);
    lib_bars(&format!("{w} store_radius"), &g.store.radius, &host.store.radius);
    all_bars(&format!("{w} h"), &g.h, &host.h);
    all_bars(&format!("{w} n_eff"), &g.n_eff, &host.n_eff);
    all_bars(&format!("{w} d_s"), &g.d_s, &host.d_s);
    all_bars(&format!("{w} store_perim"), &g.store.perim, &host.store.perim);
    all_bars(&format!("{w} store_a_d"), &g.store.a_d, &host.store.a_d);
    all_bars(&format!("{w} store_p_d"), &g.store.p_d, &host.store.p_d);
    assert!(g.store.top_width.is_empty() && host.store.top_width.is_empty());
}

#[test]
fn control_backward_matches_host() {
    control_bwd_run(false);
    control_bwd_run(true);
    assert_fields_nonzero(&[]);
}

#[test]
fn depth_backward_matches_host() {
    let case = |what: &str, c: &Consts, h: &[f32], hb0: f32| {
        let n = c.host.n;
        let dep = depth_host(&c.host, h, hb0);
        let (g_depth, g_d_edge, g_h_down) = (bar(52, n, true), bar(53, n, true), bar(54, n, true));
        let host = depth_backward_host(&c.host, h, &dep, &g_depth, &g_d_edge, &g_h_down);
        let dev = depth_backward_dev(
            &c.dev,
            &up(h),
            &DepthDev::upload(&dep, &like()),
            &up(&g_depth),
            &up(&g_d_edge),
            &up(&g_h_down),
        );
        all_bars(&format!("{what} gh"), &cube_to_host(&dev), &host);
        dep
    };
    // Juniata: the tie rows of depth_matches_host (x == lb, der == lb, h == hd), floor rows, an outlet.
    let (adj, pattern) = juniata();
    let n = adj.n;
    let c = consts_of(&adj, &pattern, None, None, params(1.0, 1, false));
    let (h0, _) = hotstart_stage(&adj, &pattern);
    let pert = bar(7, n, false);
    let no = non_outlets(&c.host);
    let (f1, f2, a, b, cc) = (no[60], no[80], no[5], no[20], no[40]);
    // copied from depth_matches_host: the tie rows get z_mid = z_down = 0 in the host copy of the consts.
    let c = edited(&c, |h| {
        h.z_mid[a] = 0.0;
        h.z_mid[b] = 0.0;
        h.z_down[b] = 0.0;
    });
    let mut h: Vec<f32> = (0..n).map(|i| h0[i] + 0.3 * (pert[i] - 1.0)).collect();
    let (zm, lb) = (&c.host.z_mid, DEPTH_LB);
    let (bd, cd) = (c.host.down[b], c.host.down[cc]);
    h[f1] = zm[f1] - 0.5;
    h[f2] = zm[f2] - 0.2;
    h[a] = lb;
    h[b] = lb;
    h[bd] = h[b] - 0.5;
    h[cc] = h[cd];
    let dep = case("depth bwd juniata", &c, &h, 0.0);
    assert_eq!(dep.x[a], lb, "x tie");
    assert_eq!(dep.der[b], lb, "der tie");
    assert_eq!(dep.h_down[cc], h[cc], "h == hd tie");
    assert!(dep.x[f1] < lb && dep.x[f2] < lb, "floor rows bind");
    let pat = &c.pattern;
    assert!((0..pat.n).any(|i| (pat.crow[i + 1] - pat.crow[i]) as usize >= 3), "a confluence row exists");

    // NormalDepth 3-chain: the outlet self-adds in the gather's transpose.
    let (adj3, pat3) = chain(3);
    let cn = consts_of(&adj3, &pat3, None, None, params(1.0, 1, false));
    assert!(!cn.host.prescribed && cn.host.down[2] == 2);
    let hn: Vec<f32> = cn.host.z_mid.iter().zip([0.8f32, 0.5, 0.3]).map(|(z, d)| z + d).collect();
    case("depth bwd normal-depth 3-chain", &cn, &hn, 0.0);

    // PrescribedStage 3-chain with hb0 = 0.4: the outlet reads hb0 and its h_down cotangent is masked.
    let cp = consts_of(&adj3, &pat3, None, None, params(1.0, 1, true));
    assert!(cp.host.prescribed);
    let dep3 = case("depth bwd prescribed 3-chain", &cp, &hn, 0.4);
    assert_eq!(dep3.h_down[2], 0.4);

    // A floored state: every depth is below the floor, so the floor mask blocks g_depth.
    let hf: Vec<f32> = cp.host.z_mid.iter().map(|z| z - 0.5).collect();
    let depf = case("depth bwd prescribed 3-chain floored", &cp, &hf, 0.4);
    assert!(depf.x.iter().all(|&x| x < DEPTH_LB), "every row is floored");
    assert_fields_nonzero(&[]);
}

/// One whole substep adjoint (readout adjoint, transposed solve, grad A, B6 to B1) on the
/// host and on the device. The device sees uploaded copies of the host inputs only.
fn assemble_bwd_case(what: &str, c: &Consts, hour: &HourHost, h0: &[f32], q0: &[f32], substeps: &[usize]) {
    let n = c.host.n;
    let fwd = stage_hour_forward(&c.host, &c.pattern, hour, h0, q0, true);
    for &s in substeps {
        let sv = &fwd.subs[s];
        let asm = stage_assemble_host(&c.host, hour, s, &sv.h, &sv.qc);
        let g_h_new = bar(70, n, true);
        let g_q_new = bar(71, n, true);
        let rg = readout_backward_host(&c.host, &sv.h, &asm, &sv.dh, &g_h_new, &g_q_new);
        let g_r = tree_lu_solve_transposed_host(&c.pattern, &sv.factors, &rg.dh);
        let g_a = tree_lu_grad_a_host(&c.pattern, &g_r, &sv.dh);
        // A non-zero start, so the order of every accumulation is exercised.
        let acc0 = HourAcc {
            i_t: bar(72, n, true),
            i_delta: bar(73, n, true),
            n2: bar(74, n, true),
            n_eff: bar(75, n, true),
            p: bar(76, n, true),
            q_eps: bar(77, n, true),
            qp: bar(78, n, true),
        };
        let mut acc_h = acc0.clone();
        let (gh, gq) = assemble_backward_host(&c.host, hour, s, &sv.h, &sv.qc, &asm, &rg, &g_r, &g_a, &mut acc_h);

        let hd = HourDev::upload(hour, &like());
        let asm_dev = AssembledDev::upload(&asm, &like(), c.host.nnz);
        let rg_dev = ReadoutGradsDev::upload(&rg, &like());
        let mut acc_d = HourAccDev::upload(&acc0, &like());
        let (dgh, dgq) = assemble_backward_dev(
            &c.dev,
            &hd,
            s,
            &up(&sv.h),
            &up(&sv.qc),
            &asm_dev,
            &rg_dev,
            &up(&g_r),
            &up(&g_a),
            &mut acc_d,
        );
        let da = hour_acc_to_host(&acc_d);
        let w = format!("{what} s={s}");
        // Every chain below runs through section, Newton or control, which call powf or ln:
        // g_h (depth adjoint of the section depths), g_q (Newton cc), n2, p, q_eps.
        // These four accumulators have none: n_eff (control, saved kk and w), qp, i_t and i_delta.
        lib_bars(&format!("{w} g_h"), &cube_to_host(&dgh), &gh);
        lib_bars(&format!("{w} g_q"), &cube_to_host(&dgq), &gq);
        lib_bars(&format!("{w} acc_n2"), &da.n2, &acc_h.n2);
        lib_bars(&format!("{w} acc_p"), &da.p, &acc_h.p);
        lib_bars(&format!("{w} acc_q_eps"), &da.q_eps, &acc_h.q_eps);
        all_bars(&format!("{w} acc_n_eff"), &da.n_eff, &acc_h.n_eff);
        all_bars(&format!("{w} acc_qp"), &da.qp, &acc_h.qp);
        all_bars(&format!("{w} acc_i_t"), &da.i_t, &acc_h.i_t);
        all_bars(&format!("{w} acc_i_delta"), &da.i_delta, &acc_h.i_delta);
    }
}

/// Hour prologue of a small chain case (copied from `Case::new` of tests/stage_op_parity.rs:
/// n, q, p, and the MC inflow of zeros).
fn chain_hour(c: &Consts, q_t0: &[f32], qp: &[f32], b: f32) -> HourHost {
    let n = c.host.n;
    let nm: Vec<f32> = (0..n).map(|i| 0.03 + 0.005 * i as f32).collect();
    let q: Vec<f32> = (0..n).map(|i| 0.45 + 0.05 * (i % 3) as f32).collect();
    let p: Vec<f32> = (0..n).map(|i| 18.0 + 2.0 * (i % 4) as f32).collect();
    stage_prologue_host(&c.host, &nm, &q, &p, q_t0, &vec![0.0; n], qp, b, b)
}

#[test]
fn assemble_backward_matches_host() {
    // Juniata: confluences, outlets and MC rows.
    let st = juniata_state(0.6, 4);
    assemble_bwd_case("assemble bwd juniata", &st.c, &st.hour, &st.h, &st.qc, &[0, 3]);

    // copied from tests/stage_op_parity.rs::table_and_mixed_cases()[0] (table_rect).
    let (adj3, pat3) = chain(3);
    let ct = consts_of(&adj3, &pat3, None, Some(table_rows()), params(0.6, 4, true));
    let mut h0: Vec<f32> = ct.host.z_mid.iter().map(|z| z + 1.0).collect();
    h0[2] = 1.5;
    let q0 = [3.0f32, 5.0, 7.0];
    let hour = chain_hour(&ct, &q0, &[0.5, 0.4, 0.3], -5.0);
    assemble_bwd_case("assemble bwd table_rect", &ct, &hour, &h0, &q0, &[0, 3]);

    // copied from tests/stage_op_parity.rs::table_and_mixed_cases (entries [1] (pool_dam_1m) and [2] (pool_dam_3m)): a dam on a table row.
    // At 1 m the stage is below the pool (ex < 0), so the release is clamped; the 3 m case has ex > 0,
    // so the 4b release adjoint carries cotangent into h.
    for above in [1.0f32, 3.0] {
        let rows = ChannelGeometryRows {
            rows: vec![0, 1, 2],
            top_width: vec![400.0; 3],
            bed_z: vec![10.0; 3],
            n: vec![0.03; 3],
            pool_z: vec![Some(12.0), None, None],
        };
        let cd = consts_with(&adj3, &pat3, None, &[(0, 1.5)], Some(rows), params_tc(0.6, 4, true, 1.0));
        let hd0 = vec![10.0 + above, 12.0, 12.0];
        let qd0 = [0.0f32; 3];
        let hour = chain_hour(&cd, &qd0, &[0.0; 3], -5.0);
        let label = format!("assemble bwd pool_dam_{above}m");
        assemble_bwd_case(&label, &cd, &hour, &hd0, &qd0, &[0, 3]);
        let fwd = stage_hour_forward(&cd.host, &cd.pattern, &hour, &hd0, &qd0, true);
        let asm = stage_assemble_host(&cd.host, &hour, 0, &fwd.subs[0].h, &fwd.subs[0].qc);
        if above > 2.0 {
            assert!(asm.control.ex[0] > 0.0, "the release is active at 3 m");
        } else {
            assert!(asm.control.ex[0] < 0.0, "the release is clamped at 1 m");
        }
    }
    assert_fields_nonzero(&[]);
}

fn prologue_bwd_case(what: &str, c: &Consts, hour: &HourHost) {
    let n = c.host.n;
    let acc = HourAcc {
        i_t: bar(60, n, true),
        i_delta: bar(61, n, true),
        n2: bar(62, n, true),
        n_eff: bar(63, n, true),
        p: bar(64, n, true),
        q_eps: bar(65, n, true),
        qp: bar(66, n, true),
    };
    let (g_h0, g_q0, g_mc) = (bar(67, n, true), bar(68, n, true), bar(69, n, true));
    let host = prologue_backward_host(&c.host, hour, &acc, &g_h0, &g_q0, &g_mc);
    let dev = prologue_backward_dev(
        &c.dev,
        &HourDev::upload(hour, &like()),
        &HourAccDev::upload(&acc, &like()),
        &up(&g_h0),
        &up(&g_q0),
        &up(&g_mc),
    );
    let g = hour_grads_to_host(&dev);
    all_bars(&format!("{what} n"), &g.n, &host.n);
    all_bars(&format!("{what} q"), &g.q, &host.q);
    all_bars(&format!("{what} p"), &g.p, &host.p);
    all_bars(&format!("{what} h_t"), &g.h_t, &host.h_t);
    all_bars(&format!("{what} q_t"), &g.q_t, &host.q_t);
    all_bars(&format!("{what} q_next_mc"), &g.q_next_mc, &host.q_next_mc);
    all_bars(&format!("{what} q_prime"), &g.q_prime, &host.q_prime);
}

#[test]
fn prologue_backward_matches_host() {
    // Juniata with seeds: MC rows, so adj_mc slots are live and in_stage has zeros.
    let st = juniata_state(0.6, 4);
    assert!(st.c.host.adj_mc.iter().any(|&w| w != 0.0), "MC-to-stage slots must exist");
    let pat = &st.c.pattern;
    assert!((0..pat.n).any(|i| (pat.crow[i + 1] - pat.crow[i]) as usize >= 3), "a confluence row exists");
    prologue_bwd_case("prologue bwd juniata", &st.c, &st.hour);

    // Table chain: the n_eff gradient is masked by keep on the table row.
    let (adj3, pat3) = chain(3);
    let ct = consts_of(&adj3, &pat3, None, Some(table_rows()), params(1.0, 1, true));
    let hour = chain_hour(&ct, &[3.0, 5.0, 7.0], &[0.5, 0.4, 0.3], -5.0);
    prologue_bwd_case("prologue bwd table chain", &ct, &hour);
    assert_fields_nonzero(&[]);
}

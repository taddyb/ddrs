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
    control_dev, control_to_host, depth_dev, depth_to_host, hour_to_host, i_mid_dev, newton_dev,
    newton_to_host, section_dev, section_to_host, stage_prologue_dev, storage_rate_dev,
};
use ddrs::routing::stage_cuda::{ControlDev, DepthDev, HourDev, NewtonDev, SectionDev, StageOpConstsDev};
use ddrs::routing::stage_op::{
    control_host, depth_host, newton_host, section_host, stage_assemble_host, stage_prologue_host, ControlOut, DepthOut,
    HourHost, StageOpConsts,
};
use ddrs::sparse::cube_buf::{cube_from_host, cube_to_host, cube_to_host_i32, Cube, Gpu};
use ddrs::sparse::tree_lu::tree_levels;
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
    StageParams {
        theta,
        theta_control: 0.5,
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
    let net = StageNetwork::build(adj, pattern, seeds, 0.0, &[]).unwrap();
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

// copied from tests/stage_op_adjoint.rs::control_run (inputs and edits)
fn control_run(with_table: bool) {
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
    let dt_s = 3600.0f32;
    let h = [3.0f32, 2.0, 1.5, 3.5, 1.5, 2.5, 2.2, 3.1];
    let p = [21.0f32, 25.0, 18.0, 30.0, 22.0, 19.0, 24.0, 26.0];
    let q = [0.5f32, 0.4, 0.6, 0.5, 0.45, 0.55, 0.35, 0.65];
    let n_eff = [0.03f32, 0.035, 0.04, 0.03, 0.05, 0.045, 0.03, 0.04];
    let depth = [1.2f32, 0.8, 0.5, 1.5, 0.4, 1.0, 0.7, 1.3];
    let d_s = [50.0f32, 40.0, 30.0, 60.0, 25.0, 45.0, 35.0, -20.0];
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

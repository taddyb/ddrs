//! The clamp ledger (forward) and the clamp probe (backward) are explicit
//! sinks. They record where each clamp is active and what gradient each
//! backward mask lets through. They must not change any value.
//!
//! Fixture: a 4-reach chain of 10 km reaches (a copy of the one in
//! `tests/cuda_backward_parity.rs`), so the Cunge X is interior.

mod common;

use std::sync::{Arc, Mutex};

use burn::backend::{Autodiff, NdArray};
use burn::tensor::Tensor;

use common::{mock_config, mock_routing_inputs, mock_streamflow, InnerBackend, TestBackend, TestDevice};
use ddrs::config::{Config, GeometryForm};
use ddrs::routing::clamp_ledger::{ClampLedger, ClampProbe, ClampSite, ProbeSite};
use ddrs::routing::mmc_op::{timestep_forward, timestep_forward_diag};
use ddrs::routing::{MuskingumCunge, SpatialParameters};
use ddrs::sparse::{AValuesAssembler, CsrPattern, SparseAdjacency};

type I = NdArray<f32>;
type AB = Autodiff<I>;

const N: usize = 4;

struct Fixture {
    length: Vec<f32>,
    slope: Vec<f32>,
    n: Vec<f32>,
    qsp: Vec<f32>,
    psp: Vec<f32>,
    qt: Vec<f32>,
    qpt: Vec<f32>,
}

fn f4_10k() -> Fixture {
    Fixture {
        length: vec![10_000.0; 4],
        slope: vec![1e-3; 4],
        n: vec![0.035; 4],
        qsp: vec![0.4; 4],
        psp: vec![20.0; 4],
        qt: vec![100.0, 120.0, 140.0, 160.0],
        qpt: vec![10.0, 12.0, 14.0, 16.0],
    }
}

fn chain(f: &Fixture) -> SparseAdjacency {
    let mut dense = vec![0.0_f32; N * N];
    for i in 0..N - 1 {
        dense[(i + 1) * N + i] = 1.0;
    }
    SparseAdjacency::from_dense(N, &dense, f.length.clone(), f.slope.clone())
}

fn cfg(form: GeometryForm, enforce_positivity: bool) -> Config {
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
    cfg
}

struct Run {
    q_nexts: Vec<Vec<f32>>,
    grads: Vec<Vec<f32>>,
    ledger: Option<ClampLedger>,
    probe: Option<ClampProbe>,
}

fn host(t: Tensor<I, 1>) -> Vec<f32> {
    t.into_data().convert::<f32>().into_vec::<f32>().unwrap()
}

/// Route `steps` chained steps. Every parent is tracked. With `sinks`, the
/// diag entry runs with a ledger and a probe; otherwise `timestep_forward`.
fn run(cfg: &Config, f: &Fixture, steps: usize, sinks: bool) -> Run {
    let adj = chain(f);
    let dev: <I as burn::tensor::backend::BackendTypes>::Device = Default::default();
    let pattern = Arc::new(CsrPattern::from_sparse(&adj));
    let assembler = AValuesAssembler::<I>::new(&pattern, &dev);
    let leaf = |d: &[f32]| -> Tensor<AB, 1> { Tensor::<AB, 1>::from_floats(d, &dev).require_grad() };
    let n = leaf(&f.n);
    let qsp = leaf(&f.qsp);
    let psp = leaf(&f.psp);
    let qt0 = leaf(&f.qt);
    let qpt = leaf(&f.qpt);
    let length: Tensor<AB, 1> = Tensor::from_floats(adj.length_m.as_slice(), &dev);
    let slope: Tensor<AB, 1> = Tensor::from_floats(adj.slope.as_slice(), &dev);
    let xs: Tensor<AB, 1> = Tensor::from_floats(vec![0.3f32; N].as_slice(), &dev);

    let mut ledger = sinks.then(ClampLedger::default);
    let probe = sinks.then(|| Arc::new(Mutex::new(ClampProbe::default())));
    let mut q = qt0.clone();
    let mut q_nexts = Vec::new();
    for _ in 0..steps {
        q = if sinks {
            timestep_forward_diag::<I>(
                cfg, &pattern, &assembler,
                n.clone(), qsp.clone(), psp.clone(), q.clone(), qpt.clone(),
                length.clone(), slope.clone(), xs.clone(), None,
                ledger.as_mut(), probe.clone(),
            )
        } else {
            timestep_forward::<I>(
                cfg, &pattern, &assembler,
                n.clone(), qsp.clone(), psp.clone(), q.clone(), qpt.clone(),
                length.clone(), slope.clone(), xs.clone(), false, None, None,
            )
        };
        q_nexts.push(host(q.clone().inner()));
    }
    let g = q.clone().sum().backward();
    let grads = [&n, &qsp, &psp, &qt0, &qpt]
        .iter()
        .map(|t| host(t.grad(&g).expect("tracked parent has a gradient")))
        .collect();
    let probe = probe.map(|p| p.lock().unwrap().clone());
    Run { q_nexts, grads, ledger, probe }
}

fn bits(v: &[Vec<f32>]) -> Vec<Vec<u32>> {
    v.iter().map(|r| r.iter().map(|x| x.to_bits()).collect()).collect()
}

fn assert_sinks_change_nothing(form: GeometryForm) {
    for enforce in [false, true] {
        let c = cfg(form, enforce);
        let f = f4_10k();
        let a = run(&c, &f, 3, false);
        let b = run(&c, &f, 3, true);
        assert_eq!(bits(&a.q_nexts), bits(&b.q_nexts), "{form:?} enforce={enforce}: q_next");
        assert_eq!(bits(&a.grads), bits(&b.grads), "{form:?} enforce={enforce}: gradients");
        assert_eq!(b.ledger.as_ref().unwrap().steps.len(), 3);
        assert_eq!(b.probe.as_ref().unwrap().steps.len(), 3);
    }
}

#[test]
fn sinks_change_no_values_trapezoid() {
    assert_sinks_change_nothing(GeometryForm::Trapezoid);
}

#[test]
fn sinks_change_no_values_power_law() {
    assert_sinks_change_nothing(GeometryForm::PowerLaw);
}

#[test]
fn engine_ledger_changes_no_values() {
    const T: usize = 24;
    const R: usize = 5;
    let device = TestDevice::default();
    let route = |with_ledger: bool| -> (Vec<f32>, Option<ClampLedger>) {
        let mut c = mock_config();
        c.params.ddr_match = false;
        let mut mc = MuskingumCunge::<InnerBackend>::new(c, device.clone());
        let params = SpatialParameters {
            n: Tensor::<TestBackend, 1>::ones([R], &device) * 0.5,
            q_spatial: Tensor::<TestBackend, 1>::ones([R], &device) * 0.5,
            p_spatial: Some(Tensor::<TestBackend, 1>::ones([R], &device) * 0.5),
            k_d: None,
            d_gw: None,
            leakance_factor: None,
            impervious_mask: None,
            gamma: None,
        };
        mc.setup_inputs(
            mock_routing_inputs(R, &device),
            mock_streamflow(T, R, &device),
            params,
            false,
            None,
        );
        if with_ledger {
            mc.enable_clamp_ledger();
        }
        let out = mc.forward();
        let v: Vec<f32> = out.inner().into_data().to_vec().unwrap();
        (v, mc.take_clamp_ledger())
    };
    let (plain, none) = route(false);
    assert!(none.is_none(), "no ledger unless enabled");
    let (with, ledger) = route(true);
    let ledger = ledger.expect("ledger was enabled");
    let a: Vec<u32> = plain.iter().map(|x| x.to_bits()).collect();
    let b: Vec<u32> = with.iter().map(|x| x.to_bits()).collect();
    assert_eq!(a, b, "the engine ledger changed a value");
    // `MuskingumCunge::forward` loops `for t in 1..num_timesteps`, one
    // `route_timestep` call per iteration.
    let route_timestep_calls = T - 1;
    println!("route_timestep calls = {route_timestep_calls}, ledger steps = {}", ledger.steps.len());
    assert_eq!(ledger.steps.len(), route_timestep_calls);
    assert!(ledger.steps.iter().all(|s| s.n == R));
}

fn depth_fixture() -> Fixture {
    let mut f = f4_10k();
    f.qt = vec![1e-6, 120.0, 140.0, 160.0];
    f
}

#[test]
fn ledger_marks_the_depth_floor() {
    for form in [GeometryForm::Trapezoid, GeometryForm::PowerLaw] {
        let r = run(&cfg(form, false), &depth_fixture(), 1, true);
        let ledger = r.ledger.unwrap();
        let depth = &ledger.steps[0].active[&ClampSite::DepthFloor];
        assert_eq!(depth, &vec![true, false, false, false], "{form:?}");
        if form == GeometryForm::PowerLaw {
            for s in [ClampSite::SideSlopeLow, ClampSite::SideSlopeHigh, ClampSite::BottomWidthFloor] {
                assert!(!ledger.steps[0].active.contains_key(&s), "power law has no {}", s.name());
            }
        } else {
            assert!(ledger.steps[0].active.contains_key(&ClampSite::BottomWidthFloor));
        }
    }
}

fn short_fixture() -> Fixture {
    let mut f = f4_10k();
    f.length = vec![1_000.0; 4];
    f
}

fn high_w_fixture() -> Fixture {
    let mut f = f4_10k();
    f.length = vec![100.0; 4];
    f.slope = vec![1e-5; 4];
    f
}

#[test]
fn probe_cuts_exactly_where_the_ledger_is_active() {
    // Active entries with a nonzero incoming gradient, per probe site. A cut
    // of a zero gradient proves nothing, so only these entries count.
    let mut live_cuts: std::collections::BTreeMap<ProbeSite, usize> = Default::default();
    for form in [GeometryForm::Trapezoid, GeometryForm::PowerLaw] {
        for enforce in [false, true] {
            for f in [f4_10k(), depth_fixture(), short_fixture(), high_w_fixture()] {
                let steps = 3;
                let r = run(&cfg(form, enforce), &f, steps, true);
                let ledger = r.ledger.unwrap();
                let probe = r.probe.unwrap();
                assert_eq!(probe.steps.len(), steps);
                for (k, sp) in probe.steps.iter().enumerate() {
                    // Backward call order is the reverse of the forward order.
                    let st = &ledger.steps[steps - 1 - k];
                    for (site, sg) in &sp.sites {
                        assert_eq!(sg.g_in.len(), N);
                        assert_eq!(sg.g_out.len(), N);
                        for i in 0..N {
                            let active = ClampSite::ALL
                                .iter()
                                .filter(|s| s.probe_site() == Some(*site))
                                .any(|s| st.active.get(s).is_some_and(|v| v[i]));
                            if active {
                                if sg.g_in[i] != 0.0 {
                                    *live_cuts.entry(*site).or_default() += 1;
                                }
                                assert_eq!(
                                    sg.g_out[i], 0.0,
                                    "{form:?} enforce={enforce} step {k} {site:?} reach {i}: active but gradient passes"
                                );
                            } else {
                                assert_eq!(
                                    sg.g_out[i].to_bits(),
                                    sg.g_in[i].to_bits(),
                                    "{form:?} enforce={enforce} step {k} {site:?} reach {i}: inactive but gradient changed"
                                );
                            }
                        }
                    }
                }
            }
        }
    }
    println!("live cuts per probe site = {live_cuts:?}");
    for s in [ProbeSite::Depth, ProbeSite::Velocity, ProbeSite::XCap, ProbeSite::KFloor, ProbeSite::XBand] {
        assert!(live_cuts.get(&s).copied().unwrap_or(0) > 0, "no live cut at {s:?}");
    }
}

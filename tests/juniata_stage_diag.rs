//! A dam inside the stage set is stage-routed, so the MC-side diagnostics
//! must not count it (Task 13 carry-over from Task 11's review).
//!
//! Own file: the negative-solve counters are process-global, and every
//! `forward()` resets them. One test, no parallel neighbours.

use burn::backend::{Autodiff, NdArray};
use burn::tensor::Tensor;
use ddrs::config::Config;
use ddrs::data::{MeritGagesDataset, RhoWindow, Staid};
use ddrs::routing::mmc_op::negative_solve_stats;
use ddrs::routing::{MuskingumCunge, RoutingInputs, SpatialParameters};
use ddrs::training::forward::{apply_reservoir_rows, apply_stage_block};

type I = NdArray<f32>;
type Ad = Autodiff<I>;

const DAM_ROW: usize = 177;

struct Out {
    account_rows: Option<Vec<usize>>,
    neg_total: u64,
    steps: usize,
    n: usize,
    stage_rows: usize,
}

fn run(cfg: &Config) -> Out {
    let device = Default::default();
    let ds = MeritGagesDataset::open(cfg).expect("open");
    let start = chrono::NaiveDate::from_ymd_opt(1990, 1, 1).unwrap();
    let window = RhoWindow {
        start_day_idx: (start - ds.time_axis().start).num_days() as usize,
        rho_days: 10,
        window_start: start,
    };
    let batch = ds.collate(&[Staid::from("01567000")], &window).expect("collate");
    let tensors = batch.to_tensors::<I>(&device);
    let n = tensors.adjacency.n;
    let col = |v: f32| Tensor::<Ad, 1>::full([n], v, &device);
    let mut engine = MuskingumCunge::<I>::new(cfg.clone(), device);
    engine.setup_inputs(
        RoutingInputs { adjacency: tensors.adjacency.clone(), x_storage: col(0.3) },
        Tensor::from_inner(tensors.q_prime.clone()),
        SpatialParameters {
            n: col(0.0851),
            q_spatial: col(0.5),
            p_spatial: Some(col(0.575)),
            k_d: None,
            d_gw: None,
            leakance_factor: None,
            impervious_mask: None,
            gamma: None,
        },
        false,
        None,
    );
    apply_reservoir_rows(
        cfg,
        &mut engine,
        tensors.reservoir_rows.as_ref(),
        tensors.window.window_start,
        tensors.q_prime.dims()[0],
        None,
    );
    apply_stage_block(
        cfg,
        &mut engine,
        tensors.stage.as_ref(),
        tensors.reservoir_rows.as_ref(),
        tensors.window.window_start,
    );
    engine.enable_negative_discharge_tracking();
    let out = engine.forward();
    let steps = out.dims()[1] - 1;
    let stage_rows = engine
        .stage_block_mut()
        .map(|b| b.stage_rows().iter().filter(|&&s| s).count())
        .unwrap_or(0);
    Out {
        account_rows: engine.dam_account().map(|a| a.rows),
        neg_total: negative_solve_stats().1,
        steps,
        n,
        stage_rows,
    }
}

#[test]
fn a_stage_set_dam_is_not_double_counted() {
    let mut mc = Config::from_yaml_file("examples/juniata/ddrs_stage_raystown.yaml").unwrap();
    mc.params.stage_routing = None;
    let a = run(&mc);
    assert_eq!(a.account_rows, Some(vec![DAM_ROW]), "plain MC accounts the dam row");
    assert_eq!(a.neg_total, (a.n * a.steps) as u64);

    let b = run(&Config::from_yaml_file("examples/juniata/ddrs_stage_raystown.yaml").unwrap());
    println!("stage set rows {}, MC solves counted {} of {}", b.stage_rows, b.neg_total, b.n * b.steps);
    assert!(b.stage_rows > 1);
    assert_eq!(b.account_rows, None, "the stage-routed dam is not in the MC clamp account");
    assert_eq!(
        b.neg_total,
        ((b.n - b.stage_rows) * b.steps) as u64,
        "stage rows are left out of the negative-discharge count"
    );
}

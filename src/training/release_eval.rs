//! Test-phase resolution of the learned dam release.
//!
//! The release head is a pure per-dam function of the dam feature table, so
//! after training it is evaluated ONCE over every table dam and the result is
//! installed in the test dataset as a fixed seasonal table
//! (`MeritGagesDataset::resolve_learned_release`). The test phase then routes
//! through the ordinary `fixed` path, with no head threaded through
//! `training::eval`. `tests/reservoir_release_training.rs` pins that the
//! training path and this resolved path route bitwise identically.
//!
//! The resolved per-dam parameters are also the run's learned output:
//! [`write_release_params_csv`] writes `COMID, T0_days, a, b, T_min_days,
//! T_max_days` (the clamped extremes of the seasonal law over a year).

use std::path::Path;

use burn::tensor::{backend::Backend, Tensor};

use crate::config::{Config, DamRow, ReservoirRelease};
use crate::data::dataset::MeritGagesDataset;
use crate::data::error::{DataError, Result};
use crate::data::store::{DamFeatures, FixedDam, FixedTable, ReservoirTable};
use crate::nn::kan_head::KanHead;
use crate::nn::release_head::{init_release_head, release_params};
use crate::routing::release::{seasonal_t_range, t_floor_days};
use crate::nn::dam_params::DamParams;
use crate::training::checkpoint::{load_dam_params, load_kan_head, release_dams_base, release_head_base};

/// Evaluate `head` on every dam of `features` and return the fixed table it
/// resolves to, dams in the feature table's order. A non-seasonal head gives
/// a non-seasonal table with `T_days = max(T0, floor)`, the floor the training
/// path applies for `cfg.dam_row()` (1/24 d on the replace row, none on the
/// additive row), so the resolved table routes through the same row.
pub fn resolve_release_table<B: Backend>(
    head: &KanHead<B>,
    features: &DamFeatures,
    cfg: &Config,
) -> FixedTable {
    resolve_release_table_with(head, None, features, cfg)
}

/// [`resolve_release_table`] with the per-dam parameters
/// (`release_head.rule_curve` / `per_dam_t0`): `T_days` is the effective
/// `T0_head·exp(δ)`, and the table carries each dam's rule-curve
/// coefficients `rule_curve_max·tanh(θ)` and its `Ibar`, computed with the
/// same tensor ops training applies to the batch's rows, so the test phase
/// routes the same rule curve bit for bit.
pub fn resolve_release_table_with<B: Backend>(
    head: &KanHead<B>,
    dams: Option<&DamParams<B>>,
    features: &DamFeatures,
    cfg: &Config,
) -> FixedTable {
    let (n, f) = features.values.dim();
    let device = head.output.weight.val().device();
    let x = Tensor::<B, 1>::from_floats(
        features.values.as_standard_layout().as_slice().expect("standard layout"),
        &device,
    )
    .reshape([n, f]);
    let p = release_params(head, x, &cfg.params.parameter_ranges);
    let host = |t: Tensor<B, 1>| -> Vec<f32> { t.into_data().to_vec::<f32>().unwrap() };
    let section = cfg.release_head.as_ref();
    let all: Vec<usize> = (0..n).collect();
    let idx = || crate::nn::dam_params::table_index::<B>(&all, &device);
    let t0_days = match (dams, section.is_some_and(|s| s.per_dam_t0)) {
        (Some(d), true) => p.t0_days * d.t0_factor(idx()).expect("per_dam_t0 carries delta"),
        _ => p.t0_days,
    };
    let t0 = host(t0_days);
    let rule_curve = match (dams, section.filter(|s| s.rule_curve)) {
        (Some(d), Some(s)) => {
            let c: Vec<f32> = d
                .coefficients(idx(), s.rule_curve_max)
                .expect("rule_curve carries theta")
                .into_data()
                .to_vec()
                .unwrap();
            Some(c.chunks(4).map(|r| [r[0], r[1], r[2], r[3]]).collect::<Vec<[f32; 4]>>())
        }
        _ => None,
    };
    // Ibar travels with the rule curve (and whenever the table has it).
    let inflow_mean = features.inflow_mean.clone();
    let seasonal = p.seasonal.is_some();
    let floor = t_floor_days(cfg.dam_row());
    let (a, b) = match p.seasonal {
        Some((a, b)) => (host(a), host(b)),
        None => (vec![0.0; n], vec![0.0; n]),
    };
    let dams = (0..n)
        .map(|i| FixedDam {
            comid: features.comids[i],
            // Non-seasonal: option C reads T_days directly, so apply the clamp
            // here (a no-op unless the sigmoid saturated at the box floor;
            // always a no-op on the additive row, whose floor is 0 < T0).
            t_days: if seasonal { t0[i] } else { t0[i].max(floor) },
            a: a[i],
            b: b[i],
            // The activation year travels with the dam, so the test phase
            // switches it on in the same chunk training would have.
            year_completed: features.years.get(i).copied().flatten(),
        })
        .collect();
    assert!(
        rule_curve.is_none() || inflow_mean.is_some(),
        "a rule curve resolves only with the table's inflow_mean_m3s (checked at dataset open)"
    );
    FixedTable { dams, seasonal, rule_curve, inflow_mean }
}

/// For a `reservoir_release: learned` config: load the release head from
/// `ckpt_dir/release_head.mpk`, resolve it over `dataset`'s feature table and
/// install the result (see the module docs). Returns the resolved table, or
/// `None` when the config is not learned (then nothing is touched).
pub fn resolve_learned_release<B: Backend>(
    cfg: &Config,
    dataset: &mut MeritGagesDataset,
    ckpt_dir: &Path,
    device: &B::Device,
) -> Result<Option<FixedTable>> {
    if !(cfg.params.use_reservoirs && cfg.params.reservoir_release == ReservoirRelease::Learned) {
        return Ok(None);
    }
    let section = cfg.release_head.as_ref().ok_or_else(|| DataError::Malformed {
        path: ckpt_dir.to_path_buf(),
        message: "reservoir_release: learned needs a release_head block".into(),
    })?;
    let template = init_release_head::<B>(section, &cfg.params.parameter_ranges, cfg.seed, device);
    let base = release_head_base(ckpt_dir);
    let head = load_kan_head::<B>(&base, template, device)?;
    let features = match dataset.reservoir_table() {
        Some(ReservoirTable::Learned(f)) => f.clone(),
        _ => {
            return Err(DataError::Malformed {
                path: ckpt_dir.to_path_buf(),
                message: "the test dataset carries no learned dam feature table".into(),
            })
        }
    };
    // The per-dam parameters (rule curve, per-dam T0), saved at full
    // precision next to the release head. A config that trains them cannot
    // be resolved without them.
    let dams = if section.has_per_dam() {
        let dbase = release_dams_base(ckpt_dir);
        let template = DamParams::<B>::zeros(
            features.comids.len(),
            section.rule_curve,
            section.per_dam_t0,
            device,
        );
        Some(load_dam_params::<B>(&dbase, template, device)?)
    } else {
        None
    };
    let table = resolve_release_table_with(&head, dams.as_ref(), &features, cfg);
    dataset.resolve_learned_release(table.clone())?;
    use std::io::Write;
    let _ = writeln!(
        std::io::stderr(),
        "release head: resolved {} dams from {}.mpk (seasonal: {}, rule curve: {}, per-dam T0: {})",
        table.dams.len(),
        base.display(),
        table.seasonal,
        table.rule_curve.is_some(),
        section.per_dam_t0
    );
    Ok(Some(table))
}

/// Write the resolved per-dam release parameters as CSV:
/// `COMID,T0_days,a,b,T_min_days,T_max_days`, the extremes after `dam_row`'s
/// floor.
pub fn write_release_params_csv(path: &Path, table: &FixedTable, dam_row: DamRow) -> Result<()> {
    // With a rule curve, the coefficients and Ibar follow (the columns the
    // fixed-table reader accepts; T0_days is the effective T0).
    let rule = table.rule_curve.as_ref().zip(table.inflow_mean.as_ref());
    let mut out = String::from("COMID,T0_days,a,b,T_min_days,T_max_days");
    if rule.is_some() {
        out.push_str(",c1s,c1c,c2s,c2c,inflow_mean_m3s");
    }
    out.push('\n');
    for (i, d) in table.dams.iter().enumerate() {
        let (lo, hi) = if table.seasonal {
            seasonal_t_range(d.t_days, d.a, d.b, dam_row)
        } else {
            (d.t_days, d.t_days)
        };
        out.push_str(&format!("{},{},{},{},{},{}", d.comid.0, d.t_days, d.a, d.b, lo, hi));
        if let Some((c, q)) = rule {
            let [c1s, c1c, c2s, c2c] = c[i];
            out.push_str(&format!(",{c1s},{c1c},{c2s},{c2c},{}", q[i]));
        }
        out.push('\n');
    }
    std::fs::write(path, out).map_err(|source| DataError::Io { path: path.to_path_buf(), source })
}

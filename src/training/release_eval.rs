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

use crate::config::{Config, ReservoirRelease};
use crate::data::dataset::MeritGagesDataset;
use crate::data::error::{DataError, Result};
use crate::data::store::{DamFeatures, FixedDam, FixedTable, ReservoirTable};
use crate::nn::kan_head::KanHead;
use crate::nn::release_head::{init_release_head, release_params};
use crate::routing::release::{seasonal_t_range, MIN_T_DAYS};
use crate::training::checkpoint::{load_kan_head, release_head_base};

/// Evaluate `head` on every dam of `features` and return the fixed table it
/// resolves to, dams in the feature table's order. A non-seasonal head gives
/// a non-seasonal table with `T_days = max(T0, 1/24)`, the same clamp the
/// training path applies.
pub fn resolve_release_table<B: Backend>(
    head: &KanHead<B>,
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
    let t0 = host(p.t0_days);
    let seasonal = p.seasonal.is_some();
    let (a, b) = match p.seasonal {
        Some((a, b)) => (host(a), host(b)),
        None => (vec![0.0; n], vec![0.0; n]),
    };
    let dams = (0..n)
        .map(|i| FixedDam {
            comid: features.comids[i],
            // Non-seasonal: option C reads T_days directly, so apply the clamp
            // here (a no-op unless the sigmoid saturated at the box floor).
            t_days: if seasonal { t0[i] } else { t0[i].max(MIN_T_DAYS) },
            a: a[i],
            b: b[i],
            // The activation year travels with the dam, so the test phase
            // switches it on in the same chunk training would have.
            year_completed: features.years.get(i).copied().flatten(),
        })
        .collect();
    FixedTable { dams, seasonal }
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
    let table = resolve_release_table(&head, &features, cfg);
    dataset.resolve_learned_release(table.clone())?;
    use std::io::Write;
    let _ = writeln!(
        std::io::stderr(),
        "release head: resolved {} dams from {}.mpk (seasonal: {})",
        table.dams.len(),
        base.display(),
        table.seasonal
    );
    Ok(Some(table))
}

/// Write the resolved per-dam release parameters as CSV:
/// `COMID,T0_days,a,b,T_min_days,T_max_days`.
pub fn write_release_params_csv(path: &Path, table: &FixedTable) -> Result<()> {
    let mut out = String::from("COMID,T0_days,a,b,T_min_days,T_max_days\n");
    for d in &table.dams {
        let (lo, hi) = if table.seasonal {
            seasonal_t_range(d.t_days, d.a, d.b)
        } else {
            (d.t_days, d.t_days)
        };
        out.push_str(&format!("{},{},{},{},{},{}\n", d.comid.0, d.t_days, d.a, d.b, lo, hi));
    }
    std::fs::write(path, out).map_err(|source| DataError::Io { path: path.to_path_buf(), source })
}

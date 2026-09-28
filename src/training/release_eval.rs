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

use crate::config::{Config, DamFloor, DamRow, ReservoirRelease};
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

/// The S28 clamp account of the test phase's dam rows, summed over every
/// chunk (`crate::routing::mmc::DamClampAccount`, one engine per chunk),
/// keyed by network row. Built by `training::eval::evaluate`; written as
/// `<run>/release_clamp.csv` by [`write_release_clamp_csv`].
#[derive(Clone, Debug, Default, PartialEq)]
pub struct DamClampSums {
    pub by_row: std::collections::BTreeMap<usize, DamClampRecord>,
}

/// One dam's clamp account over the routed test period (or, in training, one
/// dam over one window: [`account_records`]).
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct DamClampRecord {
    /// MERIT COMID of the dam reach (0 until [`DamClampSums::records`] names it).
    pub comid: i64,
    /// Water the S28 clamp created, `Σ max(lb − x, 0)·dt/c4`, m³: the storage
    /// it forgave plus the below-floor outflow the pre-clamp solve passed down
    /// (`crate::routing::mmc` `DamAccount` has the derivation).
    pub created_m3: f64,
    /// The storage part of `created_m3`, `Σ max(lb − x, 0)·(dt/c4 − dt/2)`, m³.
    pub storage_m3: f64,
    /// Carried floor (`dam_floor: carry`): the volume the dam paid back out of
    /// its inflow, m³. 0 with `forgive`.
    pub repaid_m3: f64,
    /// Carried floor: the volume the dam still owes at the end of the period
    /// (the last chunk that armed it), m³. `created − repaid − owed = 0`
    /// (to f32 rounding) over the period. 0 with `forgive`.
    pub owed_m3: f64,
    /// The dam's inflow, `Σ (I_t + q')·dt`, m³ (before any rule-curve flux).
    pub inflow_m3: f64,
    /// Steps at which the dam row's pre-clamp solve fell below the floor.
    pub clamp_steps: u64,
    /// Steps the dam was armed (a dam completed inside the period counts
    /// only its active chunks).
    pub steps: u64,
}

impl DamClampRecord {
    /// `created / inflow`: 0 when the clamp created nothing, `+inf` when it
    /// created water at a dam with no inflow.
    pub fn created_share(&self) -> f64 {
        if self.created_m3 == 0.0 {
            0.0
        } else {
            self.created_m3 / self.inflow_m3.max(0.0)
        }
    }
}

impl DamClampSums {
    /// Add one engine's account (one chunk). The owed volume is a state, not
    /// a sum: each dam's `owed_m3` becomes the chunk's closing value, which
    /// the next chunk starts from ([`Self::owed_for`]).
    pub fn merge(&mut self, a: &crate::routing::mmc::DamClampAccount) {
        for (i, &row) in a.rows.iter().enumerate() {
            let r = self.by_row.entry(row).or_default();
            r.created_m3 += a.created_m3[i];
            r.storage_m3 += a.storage_m3[i];
            r.repaid_m3 += a.repaid_m3[i];
            r.owed_m3 = a.owed_m3[i];
            r.inflow_m3 += a.inflow_m3[i];
            r.clamp_steps += a.clamp_steps[i];
            r.steps += a.steps;
        }
    }

    /// Carried floor across test-phase chunks: the volume each of `rows`
    /// (network rows, in the engine's armed order) still owed at the end of
    /// the last chunk that armed it, m³; 0 for a dam not armed before (the
    /// first chunk, or a dam completed since: an inactive dam has no owed
    /// state).
    pub fn owed_for(&self, rows: &[usize]) -> Vec<f64> {
        rows.iter().map(|r| self.by_row.get(r).map_or(0.0, |rec| rec.owed_m3)).collect()
    }

    /// The records in row order, each named by `comids[row]` (the network's
    /// COMID order).
    pub fn records(&self, comids: &[i64]) -> Vec<DamClampRecord> {
        self.by_row
            .iter()
            .map(|(&row, r)| DamClampRecord { comid: comids[row], ..*r })
            .collect()
    }
}

/// One record per dam of one engine's account (`comid` 0): one dam over one
/// training window, the unit training logs its clamp summary over.
pub fn account_records(a: &crate::routing::mmc::DamClampAccount) -> Vec<DamClampRecord> {
    (0..a.rows.len())
        .map(|i| DamClampRecord {
            comid: 0,
            created_m3: a.created_m3[i],
            storage_m3: a.storage_m3[i],
            repaid_m3: a.repaid_m3[i],
            owed_m3: a.owed_m3[i],
            inflow_m3: a.inflow_m3[i],
            clamp_steps: a.clamp_steps[i],
            steps: a.steps,
        })
        .collect()
}

/// The clamp account of a set of records, as the training log line, the test
/// phase's log line and the manifest's `metrics.release_clamp` read it.
///
/// No pooled created share: a dam below another counts the upper dam's
/// outflow in its own inflow, so `Σ created / Σ inflow` would divide by the
/// cascaded water twice (review v3, finding 6). The shares are per record
/// (per dam, or per dam and window in training), `created_m3 / inflow_m3`.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct ClampSummary {
    /// The dam floor the records were routed with.
    pub floor: DamFloor,
    /// Records (dams, or dam-windows).
    pub n: usize,
    pub clamp_steps: u64,
    /// Dam-row steps, `Σ steps`.
    pub steps: u64,
    pub created_m3: f64,
    pub storage_m3: f64,
    /// Carried floor: `Σ repaid_m3` and `Σ owed_m3` (the owed volume at the
    /// end of each record's window or period). With `carry`, `created`
    /// counts every step a dam sits at the floor, including the steps it
    /// spends repaying (its law still asks for more than it has), so it runs
    /// well above the `forgive` figure; what the floor finally adds to the
    /// river is the owed volume.
    pub repaid_m3: f64,
    pub owed_m3: f64,
    /// Carried floor: the largest per-record `owed_m3 / inflow_m3`.
    pub owed_share_max: f64,
    /// `Σ inflow_m3`: counts a cascaded dam's inflow at every dam it passes.
    pub inflow_m3: f64,
    /// Per-record created share: median, 90th percentile, maximum (nearest
    /// rank on the sorted shares).
    pub share_median: f64,
    pub share_p90: f64,
    pub share_max: f64,
    /// Records whose created share is at least 0.5 % / 5 %.
    pub n_over_half_pct: usize,
    pub n_over_5_pct: usize,
}

/// [`ClampSummary`] of `records`, routed with `floor`.
pub fn clamp_summary(records: &[DamClampRecord], floor: DamFloor) -> ClampSummary {
    let mut shares: Vec<f64> = records.iter().map(DamClampRecord::created_share).collect();
    shares.sort_by(f64::total_cmp);
    let rank = |q: f64| -> f64 {
        if shares.is_empty() {
            return 0.0;
        }
        let i = ((q * shares.len() as f64).ceil() as usize).clamp(1, shares.len()) - 1;
        shares[i]
    };
    ClampSummary {
        floor,
        n: records.len(),
        clamp_steps: records.iter().map(|r| r.clamp_steps).sum(),
        steps: records.iter().map(|r| r.steps).sum(),
        created_m3: records.iter().map(|r| r.created_m3).sum(),
        storage_m3: records.iter().map(|r| r.storage_m3).sum(),
        repaid_m3: records.iter().map(|r| r.repaid_m3).sum(),
        owed_m3: records.iter().map(|r| r.owed_m3).sum(),
        owed_share_max: records
            .iter()
            .map(|r| if r.owed_m3 == 0.0 { 0.0 } else { r.owed_m3 / r.inflow_m3.max(0.0) })
            .fold(0.0, f64::max),
        inflow_m3: records.iter().map(|r| r.inflow_m3).sum(),
        share_median: rank(0.5),
        share_p90: rank(0.9),
        share_max: shares.last().copied().unwrap_or(0.0),
        n_over_half_pct: shares.iter().filter(|&&s| s >= 0.005).count(),
        n_over_5_pct: shares.iter().filter(|&&s| s >= 0.05).count(),
    }
}

impl ClampSummary {
    /// The log fragment: clamped steps, created volume (and its storage
    /// part), the summed inflow, and the per-record created shares; with the
    /// carried floor also the repaid and still-owed volumes. `unit` names a
    /// record ("dam", "dam-window").
    pub fn describe(&self, unit: &str) -> String {
        let carry = match self.floor {
            DamFloor::Forgive => String::new(),
            DamFloor::Carry => format!(
                "; dam_floor carry: repaid {:.4e} m3, owed at end {:.4e} m3 (max {:.4}% of a {unit}'s inflow)",
                self.repaid_m3,
                self.owed_m3,
                100.0 * self.owed_share_max
            ),
        };
        format!(
            "dam-row steps at the discharge clamp {}/{} ({:.3}%); clamp-created volume {:.4e} m3 \
             (storage {:.4e}); dam inflow {:.4e} m3 (summed per dam); created share per {unit}: \
             median {:.4}%, p90 {:.4}%, max {:.4}%, {} of {} >= 0.5%, {} >= 5%{carry}",
            self.clamp_steps,
            self.steps,
            100.0 * self.clamp_steps as f64 / (self.steps as f64).max(1.0),
            self.created_m3,
            self.storage_m3,
            self.inflow_m3,
            100.0 * self.share_median,
            100.0 * self.share_p90,
            100.0 * self.share_max,
            self.n_over_half_pct,
            self.n,
            self.n_over_5_pct,
        )
    }

    /// `metrics.release_clamp` in the run manifest.
    pub fn to_json(&self) -> serde_json::Value {
        // A share of +inf (water created at a dam with no inflow) is not JSON.
        let finite = |v: f64| if v.is_finite() { serde_json::json!(v) } else { serde_json::Value::Null };
        serde_json::json!({
            "dam_floor": format!("{:?}", self.floor).to_lowercase(),
            "n_dams": self.n,
            "clamp_steps": self.clamp_steps,
            "steps": self.steps,
            "created_m3": self.created_m3,
            "storage_m3": self.storage_m3,
            "repaid_m3": self.repaid_m3,
            "owed_m3": self.owed_m3,
            "owed_share_max": finite(self.owed_share_max),
            "created_share_median": finite(self.share_median),
            "created_share_p90": finite(self.share_p90),
            "created_share_max": finite(self.share_max),
            "n_dams_created_share_ge_0p5pct": self.n_over_half_pct,
            "n_dams_created_share_ge_5pct": self.n_over_5_pct,
        })
    }
}

/// Write the test phase's per-dam clamp account:
/// `COMID,created_m3,storage_m3,repaid_m3,owed_m3,inflow_m3,created_share,clamp_steps,steps`,
/// one row per dam armed at least once, in network row order (`repaid_m3`,
/// `owed_m3`: the carried floor; 0 with `forgive`).
pub fn write_release_clamp_csv(path: &Path, records: &[DamClampRecord]) -> Result<()> {
    let mut out =
        String::from("COMID,created_m3,storage_m3,repaid_m3,owed_m3,inflow_m3,created_share,clamp_steps,steps\n");
    for r in records {
        out.push_str(&format!(
            "{},{:.6e},{:.6e},{:.6e},{:.6e},{:.6e},{:.6e},{},{}\n",
            r.comid,
            r.created_m3,
            r.storage_m3,
            r.repaid_m3,
            r.owed_m3,
            r.inflow_m3,
            r.created_share(),
            r.clamp_steps,
            r.steps
        ));
    }
    std::fs::write(path, out).map_err(|source| DataError::Io { path: path.to_path_buf(), source })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::routing::mmc::DamClampAccount;

    #[test]
    fn clamp_sums_merge_chunks_by_row_and_name_them_by_comid() {
        let mut sums = DamClampSums::default();
        // Chunk 1: dams on rows 4 and 1; chunk 2: row 4 only (row 1 not yet built
        // is the other way round, but the merge is order-free).
        sums.merge(&DamClampAccount {
            rows: vec![4, 1],
            created_m3: vec![10.0, 0.0],
            storage_m3: vec![8.0, 0.0],
            inflow_m3: vec![1000.0, 50.0],
            clamp_steps: vec![2, 0],
            steps: 360,
            repaid_m3: vec![6.0, 0.0],
            owed_m3: vec![4.0, 0.0],
        });
        // The next chunk starts row 4 from the 4 m3 it still owes; row 7 (a
        // dam completed since) and row 1 (not armed in chunk 2) from their own.
        assert_eq!(sums.owed_for(&[4, 7, 1]), vec![4.0, 0.0, 0.0]);
        sums.merge(&DamClampAccount {
            rows: vec![4],
            created_m3: vec![5.0],
            storage_m3: vec![4.0],
            inflow_m3: vec![500.0],
            clamp_steps: vec![1],
            steps: 96,
            repaid_m3: vec![9.0],
            owed_m3: vec![0.0],
        });
        let comids = [100, 101, 102, 103, 104];
        let recs = sums.records(&comids);
        assert_eq!(recs.len(), 2);
        assert_eq!((recs[0].comid, recs[0].steps, recs[0].created_m3), (101, 360, 0.0));
        assert_eq!(
            (recs[1].comid, recs[1].created_m3, recs[1].storage_m3, recs[1].inflow_m3, recs[1].clamp_steps, recs[1].steps),
            (104, 15.0, 12.0, 1500.0, 3, 456)
        );
        // Repaid sums; owed is the last chunk's closing state (created 15 =
        // repaid 15 + owed 0 over the period).
        assert_eq!((recs[1].repaid_m3, recs[1].owed_m3), (15.0, 0.0));
        assert_eq!(recs[1].created_share(), 0.01);

        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("release_clamp.csv");
        write_release_clamp_csv(&path, &recs).unwrap();
        let text = std::fs::read_to_string(&path).unwrap();
        assert_eq!(
            text,
            "COMID,created_m3,storage_m3,repaid_m3,owed_m3,inflow_m3,created_share,clamp_steps,steps\n\
             101,0.000000e0,0.000000e0,0.000000e0,0.000000e0,5.000000e1,0.000000e0,0,360\n\
             104,1.500000e1,1.200000e1,1.500000e1,0.000000e0,1.500000e3,1.000000e-2,3,456\n"
        );
    }

    #[test]
    fn clamp_summary_reports_per_dam_shares_not_a_pooled_ratio() {
        // A cascade: dam B sits below dam A, so B's inflow counts A's outflow.
        // A pooled Σcreated/Σinflow would be 30/3000 = 1 %; per dam the shares
        // are 2 % and 0.5 %.
        let rec = |created: f64, inflow: f64, clamp: u64| DamClampRecord {
            comid: 0,
            created_m3: created,
            storage_m3: 0.5 * created,
            repaid_m3: 0.75 * created,
            owed_m3: 0.25 * created,
            inflow_m3: inflow,
            clamp_steps: clamp,
            steps: 100,
        };
        let recs = [rec(20.0, 1000.0, 3), rec(10.0, 2000.0, 1), rec(0.0, 0.0, 0)];
        let s = clamp_summary(&recs, DamFloor::Carry);
        assert_eq!((s.n, s.clamp_steps, s.steps), (3, 4, 300));
        assert_eq!((s.created_m3, s.storage_m3, s.inflow_m3), (30.0, 15.0, 3000.0));
        assert_eq!((s.repaid_m3, s.owed_m3), (22.5, 7.5));
        // Sorted shares [0, 0.005, 0.02]: nearest rank.
        assert_eq!((s.share_median, s.share_p90, s.share_max), (0.005, 0.02, 0.02));
        assert_eq!((s.n_over_half_pct, s.n_over_5_pct), (2, 0));
        let j = s.to_json();
        assert_eq!(j["n_dams_created_share_ge_0p5pct"], 2);
        assert_eq!(j["dam_floor"], "carry");
        assert!(j.get("created_share").is_none(), "no pooled share in the manifest");
        assert!(s.describe("dam").contains("dam_floor carry: repaid 2.2500e1 m3, owed at end 7.5000e0 m3 (max 0.5000% of a dam's inflow)"));
        assert!(!clamp_summary(&recs, DamFloor::Forgive).describe("dam").contains("carry"));
        // Water created at a dam with no inflow: an infinite share, null in JSON.
        let s = clamp_summary(&[rec(1.0, 0.0, 1)], DamFloor::Forgive);
        assert!(s.share_max.is_infinite());
        assert!(s.to_json()["created_share_max"].is_null());
        assert_eq!(clamp_summary(&[], DamFloor::Forgive), ClampSummary::default());
    }
}

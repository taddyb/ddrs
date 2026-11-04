//! Per-reach table geometry for the stage block (`data_sources.
//! channel_geometry`, `params.stage_routing.geometry: table | mixed`).
//!
//! CSV header `COMID,top_width_m,bed_z_m,n` with an optional `pool_z_m`.
//! Other columns (e.g. `source`) are ignored. Lines starting with `#` are
//! comments. A table row replaces the head's trapezoid on its reach by a
//! rectangle of width `top_width_m` with a fixed Manning `n` and a fixed bed
//! `bed_z_m` (it may be negative, e.g. -10 m NAVD88). `pool_z_m` is the dam
//! release reference stage (same datum as `bed_z_m`).

use std::collections::{HashMap, HashSet};
use std::path::{Path, PathBuf};

use crate::data::error::{DataError, Result};
use crate::data::ids::Comid;

#[derive(Debug, Clone)]
struct GeometryRow {
    comid: Comid,
    top_width: f32,
    bed_z: f32,
    n: f32,
    pool_z: Option<f32>,
}

#[derive(Debug, Clone)]
pub struct ChannelGeometryStore {
    pub path: PathBuf,
    rows: Vec<GeometryRow>,
}

/// Table rows in one network's reach order. All vectors have the same length.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct ChannelGeometryRows {
    /// Reach row index in the network.
    pub rows: Vec<usize>,
    pub top_width: Vec<f32>,
    pub bed_z: Vec<f32>,
    pub n: Vec<f32>,
    pub pool_z: Vec<Option<f32>>,
}

impl ChannelGeometryStore {
    pub fn open(path: impl AsRef<Path>) -> Result<Self> {
        let path = path.as_ref().to_path_buf();
        let malformed = |message: String| DataError::Malformed {
            path: path.clone(),
            message,
        };
        let mut rdr = csv::ReaderBuilder::new()
            .trim(csv::Trim::All)
            .comment(Some(b'#'))
            .from_path(&path)
            .map_err(|source| DataError::Csv {
                path: path.clone(),
                source,
            })?;
        let headers = rdr
            .headers()
            .map_err(|source| DataError::Csv {
                path: path.clone(),
                source,
            })?
            .clone();
        let find = |name: &str| headers.iter().position(|h| h == name);
        let col = |name: &str| {
            find(name).ok_or_else(|| malformed(format!("missing required column `{name}`")))
        };
        let (ci, wi, bi, ni) = (col("COMID")?, col("top_width_m")?, col("bed_z_m")?, col("n")?);
        let pi = find("pool_z_m");

        let mut rows = Vec::new();
        let mut seen = HashSet::new();
        for (i, rec) in rdr.records().enumerate() {
            let row = i + 1; // 1-based data row
            let rec = rec.map_err(|source| DataError::Csv {
                path: path.clone(),
                source,
            })?;
            let num = |idx: usize, name: &str| -> Result<f32> {
                let s = rec.get(idx).unwrap_or("");
                let v: f32 = s
                    .parse()
                    .map_err(|_| malformed(format!("row {row}: {name} {s:?} is not a number")))?;
                if !v.is_finite() {
                    return Err(malformed(format!("row {row}: {name} {s:?} is not finite")));
                }
                Ok(v)
            };
            let cs = rec.get(ci).unwrap_or("");
            let comid = Comid(
                cs.parse::<i64>()
                    .map_err(|_| malformed(format!("row {row}: COMID {cs:?} is not an integer")))?,
            );
            let top_width = num(wi, "top_width_m")?;
            if top_width <= 0.0 {
                return Err(malformed(format!(
                    "row {row}: top_width_m {top_width} must be > 0"
                )));
            }
            let bed_z = num(bi, "bed_z_m")?;
            let n = num(ni, "n")?;
            if n <= 0.0 {
                return Err(malformed(format!("row {row}: n {n} must be > 0")));
            }
            let pool_z = match pi {
                Some(p) if !rec.get(p).unwrap_or("").is_empty() => Some(num(p, "pool_z_m")?),
                _ => None,
            };
            if !seen.insert(comid) {
                return Err(malformed(format!("row {row}: duplicate COMID {}", comid.0)));
            }
            rows.push(GeometryRow {
                comid,
                top_width,
                bed_z,
                n,
                pool_z,
            });
        }
        if rows.is_empty() {
            return Err(malformed("no data rows".into()));
        }
        Ok(Self { path, rows })
    }

    /// Map the table onto `network`'s order. Table COMIDs outside the network
    /// are skipped.
    pub fn rows_for(&self, network: &[Comid]) -> ChannelGeometryRows {
        let by_comid: HashMap<Comid, usize> = self
            .rows
            .iter()
            .enumerate()
            .map(|(i, r)| (r.comid, i))
            .collect();
        let mut out = ChannelGeometryRows::default();
        for (row, comid) in network.iter().enumerate() {
            if let Some(&i) = by_comid.get(comid) {
                let r = &self.rows[i];
                out.rows.push(row);
                out.top_width.push(r.top_width);
                out.bed_z.push(r.bed_z);
                out.n.push(r.n);
                out.pool_z.push(r.pool_z);
            }
        }
        out
    }

    /// COMIDs of the stage rows (`in_stage[i]`) that have no table row.
    pub fn uncovered_stage_comids(&self, network: &[Comid], in_stage: &[bool]) -> Vec<Comid> {
        let have: HashSet<Comid> = self.rows.iter().map(|r| r.comid).collect();
        network
            .iter()
            .zip(in_stage)
            .filter(|(c, &s)| s && !have.contains(c))
            .map(|(c, _)| *c)
            .collect()
    }
}

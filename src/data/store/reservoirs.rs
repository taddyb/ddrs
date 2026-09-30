//! Reservoir table reader for `data_sources.reservoirs` and the COMID → row
//! mapping that feeds `MuskingumCunge::set_reservoir_rows` (option C in
//! `.claude/RESERVOIRS.md`: a dam reach is routed as a linear reservoir
//! `S = T·Q`).
//!
//! CSV contract: a header row naming `COMID` (MERIT reach id, integer) and
//! `T_days` (residence time `T`, days). Any other columns (name, GRanD id,
//! fit provenance) are allowed and ignored. Every row must parse, `T_days`
//! must be finite and at least one hour (the floor `set_reservoir_rows`
//! enforces), COMIDs must be unique, and the table must not be empty.
//!
//! Two extensions (the learned dam release, 2026-09-26):
//! - `reservoir_release: fixed` may add both `a` and `b` columns, making each
//!   dam a seasonal bucket `T(t) = T_days·exp(a·sin ω_t + b·cos ω_t)`
//!   ([`read_fixed_release_table`]). Without them the table is option C.
//! - `reservoir_release: learned` reads a dam feature table
//!   ([`read_dam_features`]), the release head's inputs, built by
//!   `experiments/reservoir/release_head/build_dam_features.py`.
//!
//! Both kinds may carry an optional `year_completed` column (NID "Year
//! Completed", empty when unknown). A dam is routed as a reservoir only in a
//! window (training) or chunk (test phase) that starts on or after 1 January
//! of that year; before, its reach is an ordinary channel
//! ([`ReservoirRows::active_on`], applied when the rows are armed). An empty
//! year, or a table without the column, is always active.
//!
//! The harmonic rule curve (`release_head.rule_curve`, 2026-09-27): a raw
//! `inflow_mean_m3s` column (`Ibar_d`, the training-period mean of the
//! upstream-summed Q', `build_dam_inflow_clim.py`) scales each dam's flux.
//! Both kinds may carry it; a `fixed` table may also carry all four
//! coefficient columns `c1s, c1c, c2s, c2c` (a resolved learned rule curve),
//! which then require `inflow_mean_m3s`.

use std::collections::HashMap;
use std::path::Path;

use crate::data::error::{DataError, Result};
use crate::data::ids::Comid;

/// Smallest residence time accepted, in days: one hour, the routing `dt`.
/// Same bound and same f32 comparison as `MuskingumCunge::set_reservoir_rows`,
/// so every table this reader accepts is one the engine accepts.
const MIN_T_DAYS: f32 = 1.0 / 24.0;

/// Read the reservoir table at `path` into `(COMID, T_days)` pairs, in file
/// order.
pub fn read_reservoir_table(path: impl AsRef<Path>) -> Result<Vec<(Comid, f32)>> {
    let path = path.as_ref().to_path_buf();
    let malformed = |message: String| DataError::Malformed { path: path.clone(), message };
    let csv_err = |source: csv::Error| DataError::Csv { path: path.clone(), source };

    let file = std::fs::File::open(&path).map_err(|source| DataError::Io {
        path: path.clone(),
        source,
    })?;
    let mut rdr = csv::ReaderBuilder::new()
        .has_headers(true)
        .trim(csv::Trim::All)
        .from_reader(file);
    let headers = rdr.headers().map_err(csv_err)?.clone();
    let column = |name: &str| {
        headers.iter().position(|h| h == name).ok_or_else(|| {
            malformed(format!(
                "reservoir table has no `{name}` column (header: {:?}); \
                 expected `COMID,T_days`, extra columns allowed",
                headers.iter().collect::<Vec<_>>()
            ))
        })
    };
    let (comid_col, t_col) = (column("COMID")?, column("T_days")?);

    let mut table: Vec<(Comid, f32)> = Vec::new();
    let mut first_row: HashMap<Comid, usize> = HashMap::new();
    for (i, record) in rdr.records().enumerate() {
        let row = i + 1; // 1-based data row, header excluded
        let record = record.map_err(csv_err)?;
        let field = |col: usize| record.get(col).unwrap_or("");
        let comid = field(comid_col).parse::<i64>().map(Comid).map_err(|e| {
            malformed(format!("row {row}: COMID {:?} is not an integer ({e})", field(comid_col)))
        })?;
        let t_days = field(t_col).parse::<f32>().map_err(|e| {
            malformed(format!("row {row}: T_days {:?} is not a number ({e})", field(t_col)))
        })?;
        if !t_days.is_finite() || t_days < MIN_T_DAYS {
            return Err(malformed(format!(
                "row {row}: COMID {} has T_days = {t_days}; T_days must be finite and \
                 >= 1/24 (one hour, the routing time step)",
                comid.0
            )));
        }
        if let Some(first) = first_row.insert(comid, row) {
            return Err(malformed(format!(
                "row {row}: COMID {} is listed twice (first at row {first})",
                comid.0
            )));
        }
        table.push((comid, t_days));
    }
    if table.is_empty() {
        return Err(malformed("reservoir table has no rows".into()));
    }
    Ok(table)
}

/// One dam of a `reservoir_release: fixed` table: residence time `T0` in days
/// and the seasonal coefficients of `T(t) = T0·exp(a·sin ω_t + b·cos ω_t)`.
/// `a = b = 0` when the table has no `a`/`b` columns.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct FixedDam {
    pub comid: Comid,
    pub t_days: f32,
    pub a: f32,
    pub b: f32,
    /// NID completion year; `None` (unknown, or no `year_completed` column)
    /// is always active. See [`dam_is_active`].
    pub year_completed: Option<i32>,
}

/// A `reservoir_release: fixed` table. `seasonal` is true exactly when the CSV
/// carries both `a` and `b` columns; a table without them is option C as it
/// always was (constant `T`, the engine's `set_reservoir_rows` path).
#[derive(Clone, Debug, PartialEq)]
pub struct FixedTable {
    pub dams: Vec<FixedDam>,
    pub seasonal: bool,
    /// Per-dam rule-curve coefficients `(c1s, c1c, c2s, c2c)`, aligned with
    /// `dams`, when the table carries them (a resolved learned rule curve).
    pub rule_curve: Option<Vec<[f32; 4]>>,
    /// Per-dam `Ibar` (m³/s, `inflow_mean_m3s`), aligned with `dams`, when
    /// the table carries it. Required with `rule_curve`.
    pub inflow_mean: Option<Vec<f32>>,
    /// Per-dam flood pool `(kc, phi, z_days)` (`crate::routing::mmc::FloodPool`),
    /// aligned with `dams`, when the table carries the `kc`, `phi`, `z`
    /// columns (a resolved learned pool, or an offline fit to replay).
    /// Requires `inflow_mean`. Routed only with `Config::flood_pool_on`; a
    /// dam with `z = 0` has no pool.
    pub flood_pool: Option<Vec<[f32; 3]>>,
}

/// A `reservoir_release: learned` table: the release head's per-dam inputs,
/// columns in `release_head.input_var_names` order.
#[derive(Clone, Debug, PartialEq)]
pub struct DamFeatures {
    pub comids: Vec<Comid>,
    pub names: Vec<String>,
    /// `[n_dams, n_features]`, row `i` belongs to `comids[i]`.
    pub values: ndarray::Array2<f32>,
    /// NID completion year per row (`year_completed`); `None` is always active.
    pub years: Vec<Option<i32>>,
    /// `Ibar` per row (m³/s, the raw `inflow_mean_m3s` column) when the table
    /// carries it; the rule curve's flux scale.
    pub inflow_mean: Option<Vec<f32>>,
    /// The raw `purpose_flood` column (0/1; nonzero is a flood-control dam)
    /// when the table carries it, whether or not the head reads it:
    /// `release_head.flood_pool: flood_control` arms a pool at these dams.
    pub purpose_flood: Option<Vec<bool>>,
}

/// The reservoir table a dataset carries, by `params.reservoir_release`.
#[derive(Clone, Debug, PartialEq)]
pub enum ReservoirTable {
    Fixed(FixedTable),
    Learned(DamFeatures),
}

impl ReservoirTable {
    /// Number of COMIDs in the table.
    pub fn len(&self) -> usize {
        match self {
            Self::Fixed(t) => t.dams.len(),
            Self::Learned(f) => f.comids.len(),
        }
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Completion year per table dam (`None` when unknown or no column).
    pub fn years(&self) -> Vec<Option<i32>> {
        match self {
            Self::Fixed(t) => t.dams.iter().map(|d| d.year_completed).collect(),
            Self::Learned(f) => f.years.clone(),
        }
    }

    /// `(active on start, completed after start and on or before end, completed
    /// after end, no year)`: how the table's dams split over a time axis, for
    /// the dataset-open log line.
    pub fn activation_counts(
        &self,
        start: chrono::NaiveDate,
        end: chrono::NaiveDate,
    ) -> (usize, usize, usize, usize) {
        let years = self.years();
        let active = years.iter().filter(|&&y| dam_is_active(y, start)).count();
        let later = years.iter().filter(|&&y| !dam_is_active(y, end)).count();
        let no_year = years.iter().filter(|y| y.is_none()).count();
        (active, years.len() - active - later, later, no_year)
    }
}

/// Whether a dam completed in `year_completed` is routed as a reservoir in a
/// window or chunk starting on `window_start`: active from 1 January of the
/// completion year (NID gives a year, not a date), always active when the year
/// is unknown. The decision is per window because windows (90 d) and test
/// chunks (15 d) are far shorter than the year the NID resolves.
pub fn dam_is_active(year_completed: Option<i32>, window_start: chrono::NaiveDate) -> bool {
    use chrono::Datelike;
    year_completed.is_none_or(|y| window_start.year() >= y)
}

/// Parse an optional `year_completed` field: empty ⇒ `None`; an integer or an
/// integral float (`1960.0`) ⇒ the year.
fn parse_year(field: &str) -> std::result::Result<Option<i32>, String> {
    if field.is_empty() {
        return Ok(None);
    }
    match field.parse::<f64>() {
        Ok(v) if v.is_finite() && v.fract() == 0.0 && v.abs() < 1e5 => Ok(Some(v as i32)),
        _ => Err(format!("year_completed {field:?} is not a year (integer, or empty for unknown)")),
    }
}

/// Open `path` as a headered, trimmed CSV and return the reader plus its
/// header row. Errors name the file.
fn open_csv(path: &Path) -> Result<(csv::Reader<std::fs::File>, csv::StringRecord)> {
    let file = std::fs::File::open(path).map_err(|source| DataError::Io {
        path: path.to_path_buf(),
        source,
    })?;
    let mut rdr = csv::ReaderBuilder::new()
        .has_headers(true)
        .trim(csv::Trim::All)
        .from_reader(file);
    let headers = rdr
        .headers()
        .map_err(|source| DataError::Csv { path: path.to_path_buf(), source })?
        .clone();
    Ok((rdr, headers))
}

/// Read a `reservoir_release: fixed` table: `COMID`, `T_days`, and optionally
/// both `a` and `b` (the seasonal coefficients). `COMID`/`T_days` follow
/// [`read_reservoir_table`]'s contract exactly (same parse, same one-hour
/// floor), so a table without `a`/`b` yields the same `(COMID, T_days)` bits.
/// `a` and `b` must both be present or both absent, and finite.
pub fn read_fixed_release_table(path: impl AsRef<Path>) -> Result<FixedTable> {
    let path = path.as_ref();
    let mut table = read_fixed_release_table_core(path)?;
    let (rule_curve, inflow_mean) = read_rule_curve_columns(path, table.dams.len())?;
    table.rule_curve = rule_curve;
    table.flood_pool = read_flood_pool_columns(path, inflow_mean.is_some())?;
    table.inflow_mean = inflow_mean;
    Ok(table)
}

/// The optional flood pool columns of a fixed table: all three of `kc`,
/// `phi`, `z` (days) or none; `kc` finite and `>= 0`, `phi` in `[0, 1]`, `z`
/// finite and `>= 0` (`z = 0`: no pool at that dam). They require the
/// `inflow_mean_m3s` column (`has_inflow`), which scales `Qc` and `Fmax`.
fn read_flood_pool_columns(path: &Path, has_inflow: bool) -> Result<Option<Vec<[f32; 3]>>> {
    let malformed = |message: String| DataError::Malformed { path: path.to_path_buf(), message };
    let (mut rdr, headers) = open_csv(path)?;
    let names = ["kc", "phi", "z"];
    let cols: Vec<Option<usize>> = names.iter().map(|n| headers.iter().position(|h| h == *n)).collect();
    let cols: Vec<usize> = match cols.iter().filter(|c| c.is_some()).count() {
        0 => return Ok(None),
        3 => cols.into_iter().flatten().collect(),
        _ => {
            return Err(malformed(
                "a flood pool table needs all three of `kc`, `phi`, `z` (or none)".into(),
            ))
        }
    };
    if !has_inflow {
        return Err(malformed(format!(
            "flood pool columns need the `{INFLOW_MEAN_COLUMN}` column (Ibar, m3/s), which scales \
             Qc = kc·Ibar and Fmax = z·Ibar"
        )));
    }
    let mut out = Vec::new();
    for (i, record) in rdr.records().enumerate() {
        let row = i + 1;
        let record = record.map_err(|source| DataError::Csv { path: path.to_path_buf(), source })?;
        let mut v = [0.0_f32; 3];
        for (k, (&c, name)) in cols.iter().zip(names).enumerate() {
            v[k] = parse_finite(path, row, name, record.get(c).unwrap_or(""))?;
        }
        let [kc, phi, z] = v;
        if kc < 0.0 {
            return Err(malformed(format!("row {row}: flood pool kc = {kc}; want kc >= 0")));
        }
        if !(0.0..=1.0).contains(&phi) {
            return Err(malformed(format!("row {row}: flood pool phi = {phi}; want phi in [0, 1]")));
        }
        if z < 0.0 {
            return Err(malformed(format!("row {row}: flood pool z = {z} days; want z >= 0")));
        }
        out.push(v);
    }
    Ok(Some(out))
}

/// Parse a finite `f32` field for [`read_rule_curve_columns`] /
/// [`read_dam_features`], naming the row, column and file on failure.
fn parse_finite(path: &Path, row: usize, name: &str, s: &str) -> Result<f32> {
    let malformed = |message: String| DataError::Malformed { path: path.to_path_buf(), message };
    let v = s
        .parse::<f32>()
        .map_err(|e| malformed(format!("row {row}: {name} {s:?} is not a number ({e})")))?;
    if !v.is_finite() {
        return Err(malformed(format!("row {row}: {name} = {v} is not finite")));
    }
    Ok(v)
}

/// The optional rule-curve columns of a fixed table: all four of
/// `c1s, c1c, c2s, c2c` or none, and `inflow_mean_m3s` (finite, `>= 0`),
/// which the coefficients require. `n` is the table's row count.
fn read_rule_curve_columns(
    path: &Path,
    n: usize,
) -> Result<(Option<Vec<[f32; 4]>>, Option<Vec<f32>>)> {
    let malformed = |message: String| DataError::Malformed { path: path.to_path_buf(), message };
    let (mut rdr, headers) = open_csv(path)?;
    let col = |name: &str| headers.iter().position(|h| h == name);
    let names = ["c1s", "c1c", "c2s", "c2c"];
    let coef_cols: Vec<Option<usize>> = names.iter().map(|n| col(n)).collect();
    let inflow_col = col(INFLOW_MEAN_COLUMN);
    let coefs = match coef_cols.iter().filter(|c| c.is_some()).count() {
        0 => None,
        4 => Some(coef_cols.iter().map(|c| c.unwrap()).collect::<Vec<_>>()),
        _ => {
            return Err(malformed(
                "a rule-curve table needs all four of `c1s`, `c1c`, `c2s`, `c2c` (or none)".into(),
            ))
        }
    };
    if coefs.is_some() && inflow_col.is_none() {
        return Err(malformed(format!(
            "rule-curve coefficients need the `{INFLOW_MEAN_COLUMN}` column (Ibar, m3/s)"
        )));
    }
    if coefs.is_none() && inflow_col.is_none() {
        return Ok((None, None));
    }
    let mut rule = Vec::with_capacity(n);
    let mut inflow = Vec::with_capacity(n);
    for (i, record) in rdr.records().enumerate() {
        let row = i + 1;
        let record = record.map_err(|source| DataError::Csv { path: path.to_path_buf(), source })?;
        if let Some(cols) = coefs.as_ref() {
            let mut c = [0.0_f32; 4];
            for (k, (&ci, name)) in cols.iter().zip(names).enumerate() {
                c[k] = parse_finite(path, row, name, record.get(ci).unwrap_or(""))?;
            }
            rule.push(c);
        }
        if let Some(ci) = inflow_col {
            let v = parse_finite(path, row, INFLOW_MEAN_COLUMN, record.get(ci).unwrap_or(""))?;
            if v < 0.0 {
                return Err(malformed(format!("row {row}: {INFLOW_MEAN_COLUMN} = {v} is negative")));
            }
            inflow.push(v);
        }
    }
    Ok((coefs.map(|_| rule), inflow_col.map(|_| inflow)))
}

/// The raw column of the dam tables carrying `Ibar_d` (m³/s), the rule
/// curve's flux scale (`experiments/reservoir/release_head/build_dam_inflow_clim.py`).
pub const INFLOW_MEAN_COLUMN: &str = "inflow_mean_m3s";

/// [`read_fixed_release_table`] without the rule-curve columns.
fn read_fixed_release_table_core(path: &Path) -> Result<FixedTable> {
    let base = read_reservoir_table(path)?;
    let malformed = |message: String| DataError::Malformed { path: path.to_path_buf(), message };

    let (mut rdr, headers) = open_csv(path)?;
    let col = |name: &str| headers.iter().position(|h| h == name);
    let years: Vec<Option<i32>> = match col("year_completed") {
        None => vec![None; base.len()],
        Some(c) => {
            let (mut yr, _) = open_csv(path)?;
            yr.records()
                .enumerate()
                .map(|(i, rec)| {
                    let rec = rec.map_err(|source| DataError::Csv { path: path.to_path_buf(), source })?;
                    parse_year(rec.get(c).unwrap_or("")).map_err(|e| malformed(format!("row {}: {e}", i + 1)))
                })
                .collect::<Result<_>>()?
        }
    };
    let (a_col, b_col) = match (col("a"), col("b")) {
        (Some(a), Some(b)) => (a, b),
        (None, None) => {
            return Ok(FixedTable {
                dams: base
                    .into_iter()
                    .zip(years)
                    .map(|((comid, t_days), year_completed)| FixedDam {
                        comid,
                        t_days,
                        a: 0.0,
                        b: 0.0,
                        year_completed,
                    })
                    .collect(),
                seasonal: false,
                rule_curve: None,
                inflow_mean: None,
                flood_pool: None,
            })
        }
        _ => {
            return Err(malformed(
                "a seasonal reservoir table needs both `a` and `b` columns (or neither, for a \
                 constant T)"
                    .into(),
            ))
        }
    };

    let mut dams = Vec::with_capacity(base.len());
    for (i, ((record, (comid, t_days)), year_completed)) in
        rdr.records().zip(base).zip(years).enumerate()
    {
        let row = i + 1;
        let record = record.map_err(|source| DataError::Csv { path: path.to_path_buf(), source })?;
        let coef = |c: usize, name: &str| -> Result<f32> {
            let s = record.get(c).unwrap_or("");
            let v = s.parse::<f32>().map_err(|e| {
                malformed(format!("row {row}: {name} {s:?} is not a number ({e})"))
            })?;
            if !v.is_finite() {
                return Err(malformed(format!("row {row}: COMID {} has {name} = {v}", comid.0)));
            }
            Ok(v)
        };
        dams.push(FixedDam {
            comid,
            t_days,
            a: coef(a_col, "a")?,
            b: coef(b_col, "b")?,
            year_completed,
        });
    }
    Ok(FixedTable { dams, seasonal: true, rule_curve: None, inflow_mean: None, flood_pool: None })
}

/// Read a `reservoir_release: learned` feature table: a `COMID` column plus
/// every column in `names` (the release head's `input_var_names`), returned in
/// `names` order. Values must be finite (the table is pre-filled and
/// normalised by `experiments/reservoir/release_head/build_dam_features.py`),
/// COMIDs unique, at least one row. Other columns are ignored.
pub fn read_dam_features(path: impl AsRef<Path>, names: &[String]) -> Result<DamFeatures> {
    let path = path.as_ref();
    let malformed = |message: String| DataError::Malformed { path: path.to_path_buf(), message };
    if names.is_empty() {
        return Err(malformed(
            "no feature columns requested: `release_head.input_var_names` is empty".into(),
        ));
    }
    let (mut rdr, headers) = open_csv(path)?;
    let column = |name: &str| {
        headers.iter().position(|h| h == name).ok_or_else(|| {
            malformed(format!(
                "dam feature table has no `{name}` column (header: {:?})",
                headers.iter().collect::<Vec<_>>()
            ))
        })
    };
    let comid_col = column("COMID")?;
    let cols: Vec<usize> = names.iter().map(|n| column(n)).collect::<Result<_>>()?;
    let year_col = headers.iter().position(|h| h == "year_completed");
    let inflow_col = headers.iter().position(|h| h == INFLOW_MEAN_COLUMN);
    let flood_col = headers.iter().position(|h| h == PURPOSE_FLOOD_COLUMN);

    let mut comids = Vec::new();
    let mut years = Vec::new();
    let mut inflow = Vec::new();
    let mut flood = Vec::new();
    let mut flat: Vec<f32> = Vec::new();
    let mut first_row: HashMap<Comid, usize> = HashMap::new();
    for (i, record) in rdr.records().enumerate() {
        let row = i + 1;
        let record = record.map_err(|source| DataError::Csv { path: path.to_path_buf(), source })?;
        let field = |c: usize| record.get(c).unwrap_or("");
        let comid = field(comid_col).parse::<i64>().map(Comid).map_err(|e| {
            malformed(format!("row {row}: COMID {:?} is not an integer ({e})", field(comid_col)))
        })?;
        if let Some(first) = first_row.insert(comid, row) {
            return Err(malformed(format!(
                "row {row}: COMID {} is listed twice (first at row {first})",
                comid.0
            )));
        }
        for (&c, name) in cols.iter().zip(names) {
            let v = field(c).parse::<f32>().map_err(|e| {
                malformed(format!("row {row}: {name} {:?} is not a number ({e})", field(c)))
            })?;
            if !v.is_finite() {
                return Err(malformed(format!(
                    "row {row}: COMID {} has {name} = {v}; features must be finite",
                    comid.0
                )));
            }
            flat.push(v);
        }
        let year = match year_col {
            Some(c) => parse_year(field(c)).map_err(|e| malformed(format!("row {row}: {e}")))?,
            None => None,
        };
        years.push(year);
        if let Some(c) = inflow_col {
            let v = parse_finite(path, row, INFLOW_MEAN_COLUMN, field(c))?;
            if v < 0.0 {
                return Err(malformed(format!(
                    "row {row}: COMID {} has {INFLOW_MEAN_COLUMN} = {v}; it must be >= 0",
                    comid.0
                )));
            }
            inflow.push(v);
        }
        if let Some(c) = flood_col {
            flood.push(parse_finite(path, row, PURPOSE_FLOOD_COLUMN, field(c))? != 0.0);
        }
        comids.push(comid);
    }
    if comids.is_empty() {
        return Err(malformed("dam feature table has no rows".into()));
    }
    let values = ndarray::Array2::from_shape_vec((comids.len(), names.len()), flat)
        .expect("one value per (row, feature)");
    Ok(DamFeatures {
        comids,
        names: names.to_vec(),
        values,
        years,
        inflow_mean: inflow_col.map(|_| inflow),
        purpose_flood: flood_col.map(|_| flood),
    })
}

/// The raw 0/1 column of the dam feature table marking a flood-control dam
/// (`release_head.flood_pool: flood_control`).
pub const PURPOSE_FLOOD_COLUMN: &str = "purpose_flood";

/// The dam rows of one routed network, ready for
/// `MuskingumCunge::set_reservoir_rows(&rows, &t_days)` (option C) or
/// `MuskingumCunge::set_dam_release` (seasonal or learned).
#[derive(Clone, Debug, Default, PartialEq)]
pub struct ReservoirRows {
    /// Positions in the network's COMID order, ascending.
    pub rows: Vec<usize>,
    /// Residence time in days, aligned with `rows`. Empty for a learned table.
    pub t_days: Vec<f32>,
    /// `(a, b)` aligned with `rows`, for a seasonal fixed table. `None` for a
    /// plain option C table (constant `T`) and for a learned table.
    pub seasonal: Option<(Vec<f32>, Vec<f32>)>,
    /// Release-head inputs `[rows.len(), n_features]`, for a learned table.
    pub features: Option<ndarray::Array2<f32>>,
    /// Completion year per row, aligned with `rows`, when the table carries
    /// any; EMPTY when it carries none (every dam always active).
    pub years: Vec<Option<i32>>,
    /// Row of each dam in the LEARNED feature table, aligned with `rows`: the
    /// index into the per-dam parameters (`crate::nn::dam_params`). Empty for
    /// a fixed table.
    pub table_index: Vec<usize>,
    /// `Ibar` (m³/s) per row, aligned with `rows`, when the table carries
    /// `inflow_mean_m3s`; empty otherwise.
    pub inflow_mean: Vec<f32>,
    /// Rule-curve coefficients `(c1s, c1c, c2s, c2c)` per row, for a fixed
    /// table that carries them (a resolved learned rule curve).
    pub rule_curve: Option<Vec<[f32; 4]>>,
    /// Flood pool `(kc, phi, z_days)` per row, for a fixed table that carries
    /// the `kc`, `phi`, `z` columns.
    pub flood_pool: Option<Vec<[f32; 3]>>,
    /// `purpose_flood` per row, aligned with `rows`, for a learned table that
    /// carries the column; EMPTY otherwise.
    pub purpose_flood: Vec<bool>,
}

impl ReservoirRows {
    /// The dams built by `window_start` ([`dam_is_active`]), every field
    /// subset consistently. A dam left out is not a dam row for that window:
    /// its reach routes as an ordinary channel. When every dam is active the
    /// result equals `self`, so arming it is exactly the pre-year behaviour.
    pub fn active_on(&self, window_start: chrono::NaiveDate) -> ReservoirRows {
        if self.years.is_empty() || self.years.iter().all(|&y| dam_is_active(y, window_start)) {
            return self.clone();
        }
        let keep: Vec<usize> = (0..self.rows.len())
            .filter(|&i| dam_is_active(self.years[i], window_start))
            .collect();
        let pick = |v: &Vec<f32>| -> Vec<f32> {
            if v.is_empty() { Vec::new() } else { keep.iter().map(|&i| v[i]).collect() }
        };
        ReservoirRows {
            rows: keep.iter().map(|&i| self.rows[i]).collect(),
            t_days: pick(&self.t_days),
            seasonal: self.seasonal.as_ref().map(|(a, b)| (pick(a), pick(b))),
            features: self.features.as_ref().map(|f| f.select(ndarray::Axis(0), &keep)),
            years: keep.iter().map(|&i| self.years[i]).collect(),
            table_index: if self.table_index.is_empty() {
                Vec::new()
            } else {
                keep.iter().map(|&i| self.table_index[i]).collect()
            },
            inflow_mean: pick(&self.inflow_mean),
            rule_curve: self.rule_curve.as_ref().map(|c| keep.iter().map(|&i| c[i]).collect()),
            flood_pool: self.flood_pool.as_ref().map(|c| keep.iter().map(|&i| c[i]).collect()),
            purpose_flood: if self.purpose_flood.is_empty() {
                Vec::new()
            } else {
                keep.iter().map(|&i| self.purpose_flood[i]).collect()
            },
        }
    }

    /// How many of the rows are active on `window_start`.
    pub fn n_active_on(&self, window_start: chrono::NaiveDate) -> usize {
        if self.years.is_empty() {
            return self.rows.len();
        }
        self.years.iter().filter(|&&y| dam_is_active(y, window_start)).count()
    }
}

/// Map a [`ReservoirTable`] onto a network's COMID order. Table COMIDs absent
/// from the network are skipped. A plain fixed table gives exactly
/// [`reservoir_rows`]'s result.
pub fn map_reservoir_rows(table: &ReservoirTable, network: &[Comid]) -> ReservoirRows {
    match table {
        ReservoirTable::Fixed(t) => {
            let by_comid: HashMap<Comid, usize> =
                t.dams.iter().enumerate().map(|(i, d)| (d.comid, i)).collect();
            let mut out = ReservoirRows::default();
            let (mut a, mut b, mut years) = (Vec::new(), Vec::new(), Vec::new());
            let (mut rule, mut inflow, mut pool) = (Vec::new(), Vec::new(), Vec::new());
            for (row, comid) in network.iter().enumerate() {
                if let Some(&i) = by_comid.get(comid) {
                    let d = &t.dams[i];
                    out.rows.push(row);
                    out.t_days.push(d.t_days);
                    a.push(d.a);
                    b.push(d.b);
                    years.push(d.year_completed);
                    if let Some(c) = t.rule_curve.as_ref() {
                        rule.push(c[i]);
                    }
                    if let Some(q) = t.inflow_mean.as_ref() {
                        inflow.push(q[i]);
                    }
                    if let Some(p) = t.flood_pool.as_ref() {
                        pool.push(p[i]);
                    }
                }
            }
            out.flood_pool = t.flood_pool.as_ref().map(|_| pool);
            if t.seasonal {
                out.seasonal = Some((a, b));
            }
            if t.dams.iter().any(|d| d.year_completed.is_some()) {
                out.years = years;
            }
            out.rule_curve = t.rule_curve.as_ref().map(|_| rule);
            out.inflow_mean = inflow;
            out
        }
        ReservoirTable::Learned(f) => {
            let by_comid: HashMap<Comid, usize> =
                f.comids.iter().enumerate().map(|(i, c)| (*c, i)).collect();
            let mut rows = Vec::new();
            let mut src = Vec::new();
            for (row, comid) in network.iter().enumerate() {
                if let Some(&i) = by_comid.get(comid) {
                    rows.push(row);
                    src.push(i);
                }
            }
            let features = f.values.select(ndarray::Axis(0), &src);
            let years = if f.years.iter().any(Option::is_some) {
                src.iter().map(|&i| f.years[i]).collect()
            } else {
                Vec::new()
            };
            let inflow_mean = match f.inflow_mean.as_ref() {
                Some(q) => src.iter().map(|&i| q[i]).collect(),
                None => Vec::new(),
            };
            let purpose_flood = match f.purpose_flood.as_ref() {
                Some(p) => src.iter().map(|&i| p[i]).collect(),
                None => Vec::new(),
            };
            ReservoirRows {
                rows,
                t_days: Vec::new(),
                seasonal: None,
                features: Some(features),
                years,
                table_index: src,
                inflow_mean,
                rule_curve: None,
                flood_pool: None,
                purpose_flood,
            }
        }
    }
}

/// Map a reservoir table onto a network's COMID order (the order the batch
/// gathers Q' and attributes in, e.g. `RoutingBatch::divide_comids`). Table
/// COMIDs absent from the network are skipped; no overlap gives empty rows,
/// which `set_reservoir_rows` treats as "no reservoirs".
pub fn reservoir_rows(table: &[(Comid, f32)], network: &[Comid]) -> ReservoirRows {
    let t_by_comid: HashMap<Comid, f32> = table.iter().copied().collect();
    let mut out = ReservoirRows::default();
    for (row, comid) in network.iter().enumerate() {
        if let Some(&t) = t_by_comid.get(comid) {
            out.rows.push(row);
            out.t_days.push(t);
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::PathBuf;

    fn write_csv(dir: &tempfile::TempDir, text: &str) -> PathBuf {
        let path = dir.path().join("reservoirs.csv");
        std::fs::write(&path, text).unwrap();
        path
    }

    /// Asserts `text` is rejected with a `DataError` whose message names the
    /// file and contains `needle`.
    fn rejected(text: &str, needle: &str) {
        let dir = tempfile::tempdir().unwrap();
        let path = write_csv(&dir, text);
        let msg = read_reservoir_table(&path)
            .expect_err("table must be rejected")
            .to_string();
        assert!(
            msg.contains(&path.display().to_string()) && msg.contains(needle),
            "error should name {} and contain {needle:?}, got: {msg}",
            path.display()
        );
    }

    #[test]
    fn good_table_with_extra_columns_loads() {
        let dir = tempfile::tempdir().unwrap();
        let path = write_csv(
            &dir,
            "name,COMID,grand_id,T_days\nRaystown Lake,73005301,1613,1.23\nOther,71000001,7,0.5\n",
        );
        let table = read_reservoir_table(&path).expect("good table");
        assert_eq!(table, vec![(Comid(73005301), 1.23), (Comid(71000001), 0.5)]);
    }

    #[test]
    fn committed_juniata_fixture_loads() {
        let table = read_reservoir_table("examples/juniata/data/juniata_reservoirs.csv")
            .expect("fixture");
        assert_eq!(table, vec![(Comid(73005301), 1.23)]);
    }

    #[test]
    fn missing_file_is_an_error_naming_the_path() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("absent.csv");
        let msg = read_reservoir_table(&path).unwrap_err().to_string();
        assert!(msg.contains(&path.display().to_string()), "{msg}");
    }

    #[test]
    fn missing_comid_column_rejected() {
        rejected("reach,T_days\n73005301,1.23\n", "COMID");
    }

    #[test]
    fn missing_t_days_column_rejected() {
        rejected("COMID,T\n73005301,1.23\n", "T_days");
    }

    #[test]
    fn unparseable_row_rejected() {
        rejected("COMID,T_days\n73005301,1.23\nnot_a_comid,2.0\n", "row 2");
        rejected("COMID,T_days\n73005301,soon\n", "row 1");
    }

    #[test]
    fn non_finite_t_rejected() {
        rejected("COMID,T_days\n73005301,NaN\n", "T_days");
        rejected("COMID,T_days\n73005301,inf\n", "T_days");
    }

    #[test]
    fn t_below_one_hour_rejected() {
        rejected("COMID,T_days\n73005301,0.04\n", "T_days");
        rejected("COMID,T_days\n73005301,0\n", "T_days");
    }

    #[test]
    fn one_hour_is_accepted() {
        let dir = tempfile::tempdir().unwrap();
        let path = write_csv(&dir, &format!("COMID,T_days\n73005301,{}\n", 1.0_f32 / 24.0));
        assert_eq!(read_reservoir_table(&path).unwrap(), vec![(Comid(73005301), 1.0 / 24.0)]);
    }

    #[test]
    fn duplicate_comid_rejected() {
        rejected("COMID,T_days\n73005301,1.23\n73005301,2.0\n", "73005301");
    }

    #[test]
    fn empty_table_rejected() {
        rejected("COMID,T_days\n", "no rows");
    }

    #[test]
    fn rows_map_to_network_positions_with_their_t() {
        let table = [(Comid(30), 2.0), (Comid(10), 1.5), (Comid(99), 3.0)];
        let network = [Comid(10), Comid(20), Comid(30), Comid(40)];
        assert_eq!(
            reservoir_rows(&table, &network),
            ReservoirRows { rows: vec![0, 2], t_days: vec![1.5, 2.0], ..Default::default() },
            "COMID 99 is not in the network and must be skipped"
        );
    }

    #[test]
    fn no_overlap_gives_empty_rows() {
        let table = [(Comid(99), 3.0)];
        let network = [Comid(10), Comid(20)];
        assert_eq!(reservoir_rows(&table, &network), ReservoirRows::default());
    }

    // ---- fixed seasonal table (`reservoir_release: fixed` with a, b) ----

    fn fixed_rejected(text: &str, needle: &str) {
        let dir = tempfile::tempdir().unwrap();
        let path = write_csv(&dir, text);
        let msg = read_fixed_release_table(&path)
            .expect_err("table must be rejected")
            .to_string();
        assert!(
            msg.contains(&path.display().to_string()) && msg.contains(needle),
            "error should name {} and contain {needle:?}, got: {msg}",
            path.display()
        );
    }

    #[test]
    fn fixed_table_without_a_b_is_not_seasonal_and_matches_the_plain_reader() {
        let dir = tempfile::tempdir().unwrap();
        let path = write_csv(&dir, "COMID,T_days\n73005301,1.23\n71000001,0.5\n");
        let t = read_fixed_release_table(&path).expect("plain table");
        assert!(!t.seasonal);
        let plain = read_reservoir_table(&path).unwrap();
        let got: Vec<(Comid, f32)> = t.dams.iter().map(|d| (d.comid, d.t_days)).collect();
        assert_eq!(got, plain, "same T_days bits as the option C reader");
        assert!(t.dams.iter().all(|d| d.a == 0.0 && d.b == 0.0));
    }

    #[test]
    fn fixed_table_with_a_b_is_seasonal() {
        let dir = tempfile::tempdir().unwrap();
        let path = write_csv(&dir, "COMID,T_days,a,b,name\n73005301,1.23,0.5,-1.25,x\n");
        let t = read_fixed_release_table(&path).expect("seasonal table");
        assert!(t.seasonal);
        assert_eq!(
            t.dams,
            vec![FixedDam { comid: Comid(73005301), t_days: 1.23, a: 0.5, b: -1.25, year_completed: None }]
        );
    }

    #[test]
    fn fixed_table_with_only_one_of_a_b_rejected() {
        fixed_rejected("COMID,T_days,a\n73005301,1.23,0.5\n", "`a` and `b`");
        fixed_rejected("COMID,T_days,b\n73005301,1.23,0.5\n", "`a` and `b`");
    }

    #[test]
    fn fixed_table_non_finite_a_rejected() {
        fixed_rejected("COMID,T_days,a,b\n73005301,1.23,NaN,0\n", "row 1");
        fixed_rejected("COMID,T_days,a,b\n73005301,1.23,0,inf\n", "row 1");
    }

    #[test]
    fn fixed_table_keeps_the_plain_readers_t_floor() {
        fixed_rejected("COMID,T_days,a,b\n73005301,0.01,0,0\n", "T_days");
    }

    // ---- dam feature table (`reservoir_release: learned`) ----

    fn features_rejected(text: &str, names: &[&str], needle: &str) {
        let dir = tempfile::tempdir().unwrap();
        let path = write_csv(&dir, text);
        let names: Vec<String> = names.iter().map(|s| s.to_string()).collect();
        let msg = read_dam_features(&path, &names)
            .expect_err("table must be rejected")
            .to_string();
        assert!(
            msg.contains(&path.display().to_string()) && msg.contains(needle),
            "error should name {} and contain {needle:?}, got: {msg}",
            path.display()
        );
    }

    #[test]
    fn feature_table_reads_the_listed_columns_in_config_order() {
        let dir = tempfile::tempdir().unwrap();
        let path = write_csv(
            &dir,
            "COMID,f1,f2,f3,largest_name\n10,1.0,2.0,3.0,A\n30,4.0,5.0,6.0,B\n",
        );
        let names = vec!["f3".to_string(), "f1".to_string()];
        let f = read_dam_features(&path, &names).expect("feature table");
        assert_eq!(f.comids, vec![Comid(10), Comid(30)]);
        assert_eq!(f.names, names);
        assert_eq!(f.values, ndarray::array![[3.0_f32, 1.0], [6.0, 4.0]]);
    }

    #[test]
    fn feature_table_missing_listed_column_rejected() {
        features_rejected("COMID,f1\n10,1.0\n", &["f1", "f9"], "f9");
    }

    #[test]
    fn feature_table_non_finite_value_rejected() {
        features_rejected("COMID,f1\n10,NaN\n", &["f1"], "row 1");
    }

    #[test]
    fn feature_table_duplicate_comid_rejected() {
        features_rejected("COMID,f1\n10,1\n10,2\n", &["f1"], "listed twice");
    }

    #[test]
    fn feature_table_needs_rows_and_names() {
        features_rejected("COMID,f1\n", &["f1"], "no rows");
        features_rejected("COMID,f1\n10,1\n", &[], "input_var_names");
    }

    #[test]
    fn committed_juniata_feature_fixture_loads() {
        let names = vec!["log10_storage".to_string(), "purpose_flood".to_string()];
        let f = read_dam_features("examples/juniata/data/juniata_dam_features.csv", &names)
            .expect("fixture");
        assert_eq!(f.comids, vec![Comid(73005301)]);
        assert_eq!(f.values[[0, 1]], 1.0, "Raystown's primary purpose is flood risk reduction");
    }

    #[test]
    fn map_rows_carries_seasonal_a_b_in_network_order() {
        let table = ReservoirTable::Fixed(FixedTable {
            dams: vec![
                FixedDam { comid: Comid(30), t_days: 2.0, a: 0.3, b: -0.3, year_completed: None },
                FixedDam { comid: Comid(10), t_days: 1.5, a: 0.1, b: 0.2, year_completed: None },
                FixedDam { comid: Comid(99), t_days: 3.0, a: 0.0, b: 0.0, year_completed: None },
            ],
            seasonal: true,
            rule_curve: None,
            inflow_mean: None,
            flood_pool: None,
        });
        let network = [Comid(10), Comid(20), Comid(30)];
        let rows = map_reservoir_rows(&table, &network);
        assert_eq!(rows.rows, vec![0, 2]);
        assert_eq!(rows.t_days, vec![1.5, 2.0]);
        assert_eq!(rows.seasonal, Some((vec![0.1, 0.3], vec![0.2, -0.3])));
        assert!(rows.features.is_none());
    }

    #[test]
    fn map_rows_of_a_plain_fixed_table_equals_the_option_c_mapping() {
        let plain = [(Comid(30), 2.0), (Comid(10), 1.5)];
        let table = ReservoirTable::Fixed(FixedTable {
            dams: plain
                .iter()
                .map(|&(comid, t_days)| FixedDam { comid, t_days, a: 0.0, b: 0.0, year_completed: None })
                .collect(),
            seasonal: false,
            rule_curve: None,
            inflow_mean: None,
            flood_pool: None,
        });
        let network = [Comid(10), Comid(20), Comid(30)];
        assert_eq!(map_reservoir_rows(&table, &network), reservoir_rows(&plain, &network));
    }

    #[test]
    fn map_rows_carries_feature_rows_in_network_order() {
        let table = ReservoirTable::Learned(DamFeatures {
            comids: vec![Comid(30), Comid(10)],
            names: vec!["f".into(), "g".into()],
            values: ndarray::array![[3.0_f32, 30.0], [1.0, 10.0]],
            years: vec![None, None],
            inflow_mean: None,
            purpose_flood: None,
        });
        let network = [Comid(10), Comid(20), Comid(30)];
        let rows = map_reservoir_rows(&table, &network);
        assert_eq!(rows.rows, vec![0, 2]);
        assert!(rows.t_days.is_empty() && rows.seasonal.is_none());
        assert_eq!(rows.features, Some(ndarray::array![[1.0_f32, 10.0], [3.0, 30.0]]));
    }

    // ---- activation year (`year_completed`) ----

    fn day(y: i32, m: u32, d: u32) -> chrono::NaiveDate {
        chrono::NaiveDate::from_ymd_opt(y, m, d).unwrap()
    }

    #[test]
    fn feature_table_reads_optional_year_completed() {
        let dir = tempfile::tempdir().unwrap();
        let path = write_csv(&dir, "COMID,f1,year_completed\n10,1.0,1987\n30,2.0,\n40,3.0,1960.0\n");
        let f = read_dam_features(&path, &["f1".to_string()]).expect("table with years");
        assert_eq!(f.years, vec![Some(1987), None, Some(1960)]);
        let no_col = write_csv(&dir, "COMID,f1\n10,1.0\n");
        assert_eq!(read_dam_features(&no_col, &["f1".to_string()]).unwrap().years, vec![None]);
        features_rejected("COMID,f1,year_completed\n10,1.0,soon\n", &["f1"], "year_completed");
    }

    #[test]
    fn fixed_table_reads_optional_year_completed() {
        let dir = tempfile::tempdir().unwrap();
        let path = write_csv(&dir, "COMID,T_days,a,b,year_completed\n10,1.0,0,0,2001\n20,2.0,0.5,0,\n");
        let t = read_fixed_release_table(&path).expect("fixed table with years");
        assert_eq!(t.dams.iter().map(|d| d.year_completed).collect::<Vec<_>>(), vec![Some(2001), None]);
        let plain = write_csv(&dir, "COMID,T_days,year_completed\n10,1.0,1999\n");
        let t = read_fixed_release_table(&plain).expect("plain table with years");
        assert!(!t.seasonal);
        assert_eq!(t.dams[0].year_completed, Some(1999));
    }

    #[test]
    fn a_dam_is_active_from_january_first_of_its_completion_year() {
        assert!(!dam_is_active(Some(1990), day(1989, 12, 31)));
        assert!(dam_is_active(Some(1990), day(1990, 1, 1)));
        assert!(dam_is_active(Some(1990), day(2005, 6, 1)));
        assert!(dam_is_active(None, day(1900, 1, 1)), "a dam without a year is always active");
    }

    #[test]
    fn active_on_keeps_only_built_dams_in_every_field() {
        let table = ReservoirTable::Learned(DamFeatures {
            comids: vec![Comid(10), Comid(20), Comid(30)],
            names: vec!["f".into()],
            values: ndarray::array![[1.0_f32], [2.0], [3.0]],
            years: vec![Some(1990), None, Some(1985)],
            inflow_mean: None,
            purpose_flood: None,
        });
        let network = [Comid(10), Comid(15), Comid(20), Comid(30)];
        let rows = map_reservoir_rows(&table, &network);
        assert_eq!(rows.years, vec![Some(1990), None, Some(1985)]);

        let before = rows.active_on(day(1989, 12, 31));
        assert_eq!(before.rows, vec![2, 3], "the 1990 dam (row 0) is not built yet");
        assert_eq!(before.features, Some(ndarray::array![[2.0_f32], [3.0]]));
        assert_eq!(before.years, vec![None, Some(1985)]);
        assert_eq!(rows.n_active_on(day(1989, 12, 31)), 2);
        assert_eq!(rows.active_on(day(1990, 1, 1)), rows, "every dam built: the rows unchanged");
        assert_eq!(rows.active_on(day(1984, 1, 1)).rows, vec![2], "only the undated dam");

        let fixed = ReservoirTable::Fixed(FixedTable {
            dams: vec![
                FixedDam { comid: Comid(10), t_days: 1.0, a: 0.1, b: 0.2, year_completed: Some(2000) },
                FixedDam { comid: Comid(30), t_days: 3.0, a: 0.3, b: 0.4, year_completed: Some(1970) },
            ],
            seasonal: true,
            rule_curve: None,
            inflow_mean: None,
            flood_pool: None,
        });
        let f = map_reservoir_rows(&fixed, &network).active_on(day(1995, 3, 1));
        assert_eq!((f.rows, f.t_days), (vec![3], vec![3.0]));
        assert_eq!(f.seasonal, Some((vec![0.3], vec![0.4])));
    }

    #[test]
    fn a_table_without_years_is_always_fully_active() {
        let table = [(Comid(30), 2.0), (Comid(10), 1.5)];
        let rows = reservoir_rows(&table, &[Comid(10), Comid(30)]);
        assert!(rows.years.is_empty());
        assert_eq!(rows.active_on(day(1800, 1, 1)), rows);
        assert_eq!(rows.n_active_on(day(1800, 1, 1)), 2);
    }

    // ---- rule curve (`inflow_mean_m3s`, `c1s, c1c, c2s, c2c`) ----

    #[test]
    fn fixed_table_reads_optional_rule_curve_columns() {
        let dir = tempfile::tempdir().unwrap();
        let path = write_csv(
            &dir,
            "COMID,T_days,a,b,c1s,c1c,c2s,c2c,inflow_mean_m3s\n10,1.0,0,0,0.1,-0.2,0.3,-0.4,12.5\n\
             20,2.0,0.5,0,0,0,0,0,0.0\n",
        );
        let t = read_fixed_release_table(&path).expect("rule-curve table");
        assert_eq!(t.rule_curve, Some(vec![[0.1, -0.2, 0.3, -0.4], [0.0; 4]]));
        assert_eq!(t.inflow_mean, Some(vec![12.5, 0.0]));
        // Ibar alone is allowed; no coefficient columns is no rule curve.
        let path = write_csv(&dir, "COMID,T_days,inflow_mean_m3s\n10,1.0,3.0\n");
        let t = read_fixed_release_table(&path).unwrap();
        assert_eq!((t.rule_curve, t.inflow_mean), (None, Some(vec![3.0])));
        let path = write_csv(&dir, "COMID,T_days\n10,1.0\n");
        let t = read_fixed_release_table(&path).unwrap();
        assert_eq!((t.rule_curve, t.inflow_mean), (None, None));
    }

    #[test]
    fn fixed_table_rule_curve_rejections() {
        fixed_rejected("COMID,T_days,c1s,c1c,inflow_mean_m3s\n10,1.0,0.1,0.2,3.0\n", "all four");
        fixed_rejected("COMID,T_days,c1s,c1c,c2s,c2c\n10,1.0,0.1,0.2,0,0\n", "inflow_mean_m3s");
        fixed_rejected("COMID,T_days,inflow_mean_m3s\n10,1.0,-1.0\n", "negative");
        fixed_rejected("COMID,T_days,c1s,c1c,c2s,c2c,inflow_mean_m3s\n10,1.0,x,0,0,0,1\n", "c1s");
    }

    #[test]
    fn feature_table_reads_inflow_mean_and_rows_carry_it_with_the_table_index() {
        let dir = tempfile::tempdir().unwrap();
        let path = write_csv(&dir, "COMID,f1,inflow_mean_m3s\n30,1.0,7.5\n10,2.0,0.25\n");
        let f = read_dam_features(&path, &["f1".to_string()]).unwrap();
        assert_eq!(f.inflow_mean, Some(vec![7.5, 0.25]));
        let rows = map_reservoir_rows(&ReservoirTable::Learned(f), &[Comid(10), Comid(20), Comid(30)]);
        assert_eq!(rows.rows, vec![0, 2]);
        assert_eq!(rows.table_index, vec![1, 0], "row of each dam in the feature table");
        assert_eq!(rows.inflow_mean, vec![0.25, 7.5]);
        let no_col = write_csv(&dir, "COMID,f1\n10,1.0\n");
        assert_eq!(read_dam_features(&no_col, &["f1".to_string()]).unwrap().inflow_mean, None);
        features_rejected("COMID,f1,inflow_mean_m3s\n10,1.0,-2\n", &["f1"], "inflow_mean_m3s");
    }

    // ---- flood pool (`kc`, `phi`, `z`; `purpose_flood`) ----

    #[test]
    fn fixed_table_reads_optional_flood_pool_columns() {
        let dir = tempfile::tempdir().unwrap();
        let path = write_csv(
            &dir,
            "COMID,T_days,inflow_mean_m3s,kc,phi,z,year_completed\n10,1.0,5.0,3.5,0.75,20,1990\n\
             20,2.0,1.0,1,0,0,\n",
        );
        let t = read_fixed_release_table(&path).expect("pool table");
        assert_eq!(t.flood_pool, Some(vec![[3.5, 0.75, 20.0], [1.0, 0.0, 0.0]]));
        assert_eq!(t.inflow_mean, Some(vec![5.0, 1.0]));
        let rows = map_reservoir_rows(&ReservoirTable::Fixed(t), &[Comid(20), Comid(15), Comid(10)]);
        assert_eq!(rows.rows, vec![0, 2]);
        assert_eq!(rows.flood_pool, Some(vec![[1.0, 0.0, 0.0], [3.5, 0.75, 20.0]]));
        assert!(rows.purpose_flood.is_empty());
        let later = rows.active_on(day(1985, 1, 1));
        assert_eq!((later.rows, later.flood_pool), (vec![0], Some(vec![[1.0, 0.0, 0.0]])));
        let path = write_csv(&dir, "COMID,T_days\n10,1.0\n");
        assert_eq!(read_fixed_release_table(&path).unwrap().flood_pool, None);
    }

    #[test]
    fn fixed_table_flood_pool_rejections() {
        fixed_rejected("COMID,T_days,inflow_mean_m3s,kc,phi\n10,1.0,1,3,0.5\n", "all three");
        fixed_rejected("COMID,T_days,kc,phi,z\n10,1.0,3,0.5,2\n", "inflow_mean_m3s");
        fixed_rejected("COMID,T_days,inflow_mean_m3s,kc,phi,z\n10,1.0,1,3,1.5,2\n", "phi in [0, 1]");
        fixed_rejected("COMID,T_days,inflow_mean_m3s,kc,phi,z\n10,1.0,1,3,0.5,-2\n", "z >= 0");
        fixed_rejected("COMID,T_days,inflow_mean_m3s,kc,phi,z\n10,1.0,1,-3,0.5,2\n", "kc >= 0");
        fixed_rejected("COMID,T_days,inflow_mean_m3s,kc,phi,z\n10,1.0,1,NaN,0.5,2\n", "kc");
    }

    #[test]
    fn feature_table_reads_purpose_flood_whether_or_not_the_head_does() {
        let dir = tempfile::tempdir().unwrap();
        let path = write_csv(&dir, "COMID,f1,purpose_flood\n30,1.0,1\n10,2.0,0\n40,3.0,1.0\n");
        let f = read_dam_features(&path, &["f1".to_string()]).unwrap();
        assert_eq!(f.purpose_flood, Some(vec![true, false, true]));
        let rows = map_reservoir_rows(&ReservoirTable::Learned(f), &[Comid(10), Comid(30)]);
        assert_eq!(rows.purpose_flood, vec![false, true]);
        let no_col = write_csv(&dir, "COMID,f1\n10,1.0\n");
        assert_eq!(read_dam_features(&no_col, &["f1".to_string()]).unwrap().purpose_flood, None);
        features_rejected("COMID,f1,purpose_flood\n10,1.0,yes\n", &["f1"], "purpose_flood");
    }

    #[test]
    fn active_on_subsets_the_rule_curve_fields() {
        let table = ReservoirTable::Fixed(FixedTable {
            dams: vec![
                FixedDam { comid: Comid(10), t_days: 1.0, a: 0.0, b: 0.0, year_completed: Some(2000) },
                FixedDam { comid: Comid(30), t_days: 3.0, a: 0.0, b: 0.0, year_completed: Some(1970) },
            ],
            seasonal: false,
            rule_curve: Some(vec![[0.1, 0.2, 0.3, 0.4], [0.5, 0.6, 0.7, 0.8]]),
            inflow_mean: Some(vec![1.0, 3.0]),
            flood_pool: None,
        });
        let rows = map_reservoir_rows(&table, &[Comid(10), Comid(30)]);
        assert_eq!(rows.rule_curve, Some(vec![[0.1, 0.2, 0.3, 0.4], [0.5, 0.6, 0.7, 0.8]]));
        let f = rows.active_on(day(1995, 3, 1));
        assert_eq!(f.rows, vec![1]);
        assert_eq!(f.rule_curve, Some(vec![[0.5, 0.6, 0.7, 0.8]]));
        assert_eq!(f.inflow_mean, vec![3.0]);
    }
}

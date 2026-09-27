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
}

/// A `reservoir_release: fixed` table. `seasonal` is true exactly when the CSV
/// carries both `a` and `b` columns; a table without them is option C as it
/// always was (constant `T`, the engine's `set_reservoir_rows` path).
#[derive(Clone, Debug, PartialEq)]
pub struct FixedTable {
    pub dams: Vec<FixedDam>,
    pub seasonal: bool,
}

/// A `reservoir_release: learned` table: the release head's per-dam inputs,
/// columns in `release_head.input_var_names` order.
#[derive(Clone, Debug, PartialEq)]
pub struct DamFeatures {
    pub comids: Vec<Comid>,
    pub names: Vec<String>,
    /// `[n_dams, n_features]`, row `i` belongs to `comids[i]`.
    pub values: ndarray::Array2<f32>,
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
    let base = read_reservoir_table(path)?;
    let malformed = |message: String| DataError::Malformed { path: path.to_path_buf(), message };

    let (mut rdr, headers) = open_csv(path)?;
    let col = |name: &str| headers.iter().position(|h| h == name);
    let (a_col, b_col) = match (col("a"), col("b")) {
        (Some(a), Some(b)) => (a, b),
        (None, None) => {
            return Ok(FixedTable {
                dams: base
                    .into_iter()
                    .map(|(comid, t_days)| FixedDam { comid, t_days, a: 0.0, b: 0.0 })
                    .collect(),
                seasonal: false,
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
    for (i, (record, (comid, t_days))) in rdr.records().zip(base).enumerate() {
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
        dams.push(FixedDam { comid, t_days, a: coef(a_col, "a")?, b: coef(b_col, "b")? });
    }
    Ok(FixedTable { dams, seasonal: true })
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

    let mut comids = Vec::new();
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
        comids.push(comid);
    }
    if comids.is_empty() {
        return Err(malformed("dam feature table has no rows".into()));
    }
    let values = ndarray::Array2::from_shape_vec((comids.len(), names.len()), flat)
        .expect("one value per (row, feature)");
    Ok(DamFeatures { comids, names: names.to_vec(), values })
}

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
}

/// Map a [`ReservoirTable`] onto a network's COMID order. Table COMIDs absent
/// from the network are skipped. A plain fixed table gives exactly
/// [`reservoir_rows`]'s result.
pub fn map_reservoir_rows(table: &ReservoirTable, network: &[Comid]) -> ReservoirRows {
    match table {
        ReservoirTable::Fixed(t) => {
            let by_comid: HashMap<Comid, FixedDam> = t.dams.iter().map(|d| (d.comid, *d)).collect();
            let mut out = ReservoirRows::default();
            let (mut a, mut b) = (Vec::new(), Vec::new());
            for (row, comid) in network.iter().enumerate() {
                if let Some(d) = by_comid.get(comid) {
                    out.rows.push(row);
                    out.t_days.push(d.t_days);
                    a.push(d.a);
                    b.push(d.b);
                }
            }
            if t.seasonal {
                out.seasonal = Some((a, b));
            }
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
            ReservoirRows { rows, t_days: Vec::new(), seasonal: None, features: Some(features) }
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
            vec![FixedDam { comid: Comid(73005301), t_days: 1.23, a: 0.5, b: -1.25 }]
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
                FixedDam { comid: Comid(30), t_days: 2.0, a: 0.3, b: -0.3 },
                FixedDam { comid: Comid(10), t_days: 1.5, a: 0.1, b: 0.2 },
                FixedDam { comid: Comid(99), t_days: 3.0, a: 0.0, b: 0.0 },
            ],
            seasonal: true,
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
                .map(|&(comid, t_days)| FixedDam { comid, t_days, a: 0.0, b: 0.0 })
                .collect(),
            seasonal: false,
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
        });
        let network = [Comid(10), Comid(20), Comid(30)];
        let rows = map_reservoir_rows(&table, &network);
        assert_eq!(rows.rows, vec![0, 2]);
        assert!(rows.t_days.is_empty() && rows.seasonal.is_none());
        assert_eq!(rows.features, Some(ndarray::array![[1.0_f32, 10.0], [3.0, 30.0]]));
    }
}

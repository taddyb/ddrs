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

/// The dam rows of one routed network, ready for
/// `MuskingumCunge::set_reservoir_rows(&rows, &t_days)`.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct ReservoirRows {
    /// Positions in the network's COMID order, ascending.
    pub rows: Vec<usize>,
    /// Residence time in days, aligned with `rows`.
    pub t_days: Vec<f32>,
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
            ReservoirRows { rows: vec![0, 2], t_days: vec![1.5, 2.0] },
            "COMID 99 is not in the network and must be skipped"
        );
    }

    #[test]
    fn no_overlap_gives_empty_rows() {
        let table = [(Comid(99), 3.0)];
        let network = [Comid(10), Comid(20)];
        assert_eq!(reservoir_rows(&table, &network), ReservoirRows::default());
    }
}

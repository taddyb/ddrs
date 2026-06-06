//! Read flowpath attributes from a MERIT Hydro `.dbf` file.
//!
//! Only the attribute table is consumed — the `.shp` geometry is never opened.
//! If the caller supplies a `.shp` path the reader resolves the sibling `.dbf`
//! automatically (matches geopandas behaviour in `ddr/src/ddr/geodatazoo/merit.py:72-78`).
//!
//! ## dBASE null semantics
//!
//! dBASE `Numeric` fields that contain only spaces are returned by the `dbase`
//! crate as `FieldValue::Numeric(None)`.  For float-valued columns (`lengthkm`,
//! `slope`) we map `None → f64::NAN`, matching geopandas's default fill so that
//! the downstream mean-fill step (see DDR `readers.py::fill_nans`) produces the
//! same values as the Python pipeline.  For integer-valued columns (`COMID`,
//! `NextDownID`, `up1..up4`) a null is malformed data and we return an error.

use std::path::{Path, PathBuf};

use crate::data::error::DataError;

/// Flowpath record extracted from a MERIT Hydro `.dbf` attribute table.
///
/// Field names match MERIT's dBASE columns exactly:
///   `COMID`, `lengthkm`, `slope`, `NextDownID`, `up1`, `up2`, `up3`, `up4`.
///
/// `NextDownID == 0` → terminal / outlet reach.
/// `up[i] == 0` → no i-th upstream reach.
/// Interpretation of sentinel zeros is left to the builder (Task 3).
#[derive(Debug, Clone, PartialEq)]
pub struct FlowpathRecord {
    pub comid: i64,
    pub lengthkm: f64,
    pub slope: f64,
    pub next_down_id: i64,
    pub up: [i64; 4],
}

/// Read all flowpath records from a MERIT Hydro `.dbf` (or `.shp`) file.
///
/// If `path` ends with `.shp` the sibling `.dbf` (same stem, same directory)
/// is opened instead — `.shp` geometry is never read.
///
/// Returns a `Vec<FlowpathRecord>` in file order.
pub fn read_flowpath_records(path: &Path) -> crate::data::error::Result<Vec<FlowpathRecord>> {
    let dbf_path = resolve_dbf_path(path);

    let mut reader =
        dbase::Reader::from_path(&dbf_path).map_err(|e| DataError::Malformed {
            path: dbf_path.clone(),
            message: format!("failed to open dbf: {e}"),
        })?;

    let mut records = Vec::with_capacity(reader.header().num_records as usize);
    for (row, rec) in reader.iter_records().enumerate() {
        let rec = rec.map_err(|e| DataError::Malformed {
            path: dbf_path.clone(),
            message: format!("failed to read dbf record at row {row}: {e}"),
        })?;
        records.push(parse_record(&dbf_path, row, rec)?);
    }
    Ok(records)
}

// ---------------------------------------------------------------------------
// Internal helpers
// ---------------------------------------------------------------------------

/// If `path` has a `.shp` extension, swap it for `.dbf`; otherwise return as-is.
fn resolve_dbf_path(path: &Path) -> PathBuf {
    match path.extension().and_then(|e| e.to_str()) {
        Some(ext) if ext.eq_ignore_ascii_case("shp") => path.with_extension("dbf"),
        _ => path.to_path_buf(),
    }
}

/// Extract one `FlowpathRecord` from a raw dbase `Record`.
fn parse_record(
    path: &PathBuf,
    row: usize,
    rec: dbase::Record,
) -> crate::data::error::Result<FlowpathRecord> {
    let comid = extract_int(path, row, &rec, "COMID")?;
    let lengthkm = extract_float_nullable(path, row, &rec, "lengthkm")?;
    let slope = extract_float_nullable(path, row, &rec, "slope")?;
    let next_down_id = extract_int(path, row, &rec, "NextDownID")?;
    let up1 = extract_int(path, row, &rec, "up1")?;
    let up2 = extract_int(path, row, &rec, "up2")?;
    let up3 = extract_int(path, row, &rec, "up3")?;
    let up4 = extract_int(path, row, &rec, "up4")?;

    Ok(FlowpathRecord {
        comid,
        lengthkm,
        slope,
        next_down_id,
        up: [up1, up2, up3, up4],
    })
}

/// Extract an integer-valued dBASE `Numeric` field.
///
/// dBASE stores integers as `Numeric` (decimal text).  `None` (all-spaces
/// field) is a malformed record — a flowline without a COMID / topology ID
/// is unusable, so we error with full path + row context.
fn extract_int(
    path: &PathBuf,
    row: usize,
    rec: &dbase::Record,
    col: &str,
) -> crate::data::error::Result<i64> {
    match rec.get(col) {
        Some(dbase::FieldValue::Numeric(Some(v))) => Ok(*v as i64),
        Some(dbase::FieldValue::Numeric(None)) => Err(DataError::Malformed {
            path: path.clone(),
            message: format!("row {row}: column '{col}' is null — expected non-null integer"),
        }),
        Some(other) => Err(DataError::Malformed {
            path: path.clone(),
            message: format!("row {row}: column '{col}' has unexpected type {other:?}"),
        }),
        None => Err(DataError::Malformed {
            path: path.clone(),
            message: format!("row {row}: column '{col}' not found in record"),
        }),
    }
}

/// Extract a float-valued dBASE `Numeric` field, mapping `None → NaN`.
///
/// Mirrors geopandas null semantics (`merit.py:72-78`) so the downstream
/// `naninfmean` fill produces identical values to the Python pipeline.
fn extract_float_nullable(
    path: &PathBuf,
    row: usize,
    rec: &dbase::Record,
    col: &str,
) -> crate::data::error::Result<f64> {
    match rec.get(col) {
        Some(dbase::FieldValue::Numeric(Some(v))) => Ok(*v),
        Some(dbase::FieldValue::Numeric(None)) => Ok(f64::NAN),
        Some(other) => Err(DataError::Malformed {
            path: path.clone(),
            message: format!("row {row}: column '{col}' has unexpected type {other:?}"),
        }),
        None => Err(DataError::Malformed {
            path: path.clone(),
            message: format!("row {row}: column '{col}' not found in record"),
        }),
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use dbase::{FieldName, TableWriterBuilder};
    use std::convert::TryFrom;
    use tempfile::NamedTempFile;

    /// Write a synthetic `.dbf` to a temp file via `dbase::TableWriterBuilder`,
    /// read it back through `read_flowpath_records`, and assert correctness.
    ///
    /// Columns: COMID(N,10,0), lengthkm(N,12,6), slope(N,12,6),
    ///          NextDownID(N,10,0), up1..up4(N,10,0).
    /// Row 0: fully populated.
    /// Row 1: lengthkm and slope are null → should become NaN.
    fn write_synthetic_dbf() -> NamedTempFile {
        let tmp = NamedTempFile::with_suffix(".dbf").unwrap();

        // Build schema — 8 Numeric columns matching MERIT's layout.
        let builder = TableWriterBuilder::new()
            .add_numeric_field(FieldName::try_from("COMID").unwrap(), 10, 0)
            .add_numeric_field(FieldName::try_from("lengthkm").unwrap(), 12, 6)
            .add_numeric_field(FieldName::try_from("slope").unwrap(), 12, 6)
            .add_numeric_field(FieldName::try_from("NextDownID").unwrap(), 10, 0)
            .add_numeric_field(FieldName::try_from("up1").unwrap(), 10, 0)
            .add_numeric_field(FieldName::try_from("up2").unwrap(), 10, 0)
            .add_numeric_field(FieldName::try_from("up3").unwrap(), 10, 0)
            .add_numeric_field(FieldName::try_from("up4").unwrap(), 10, 0);

        let mut writer = builder
            .build_with_file_dest(tmp.path())
            .expect("create dbf writer");

        // Row 0: fully populated, two upstream reaches, two zeros.
        let mut r0 = dbase::Record::default();
        r0.insert("COMID".to_owned(), dbase::FieldValue::Numeric(Some(123456.0)));
        r0.insert("lengthkm".to_owned(), dbase::FieldValue::Numeric(Some(2.345)));
        r0.insert("slope".to_owned(), dbase::FieldValue::Numeric(Some(0.001)));
        r0.insert("NextDownID".to_owned(), dbase::FieldValue::Numeric(Some(654321.0)));
        r0.insert("up1".to_owned(), dbase::FieldValue::Numeric(Some(111111.0)));
        r0.insert("up2".to_owned(), dbase::FieldValue::Numeric(Some(222222.0)));
        r0.insert("up3".to_owned(), dbase::FieldValue::Numeric(Some(0.0)));
        r0.insert("up4".to_owned(), dbase::FieldValue::Numeric(Some(0.0)));
        writer.write_record(&r0).expect("write row 0");

        // Row 1: lengthkm and slope null → NaN downstream.
        let mut r1 = dbase::Record::default();
        r1.insert("COMID".to_owned(), dbase::FieldValue::Numeric(Some(999999.0)));
        r1.insert("lengthkm".to_owned(), dbase::FieldValue::Numeric(None));
        r1.insert("slope".to_owned(), dbase::FieldValue::Numeric(None));
        r1.insert("NextDownID".to_owned(), dbase::FieldValue::Numeric(Some(0.0)));
        r1.insert("up1".to_owned(), dbase::FieldValue::Numeric(Some(0.0)));
        r1.insert("up2".to_owned(), dbase::FieldValue::Numeric(Some(0.0)));
        r1.insert("up3".to_owned(), dbase::FieldValue::Numeric(Some(0.0)));
        r1.insert("up4".to_owned(), dbase::FieldValue::Numeric(Some(0.0)));
        writer.write_record(&r1).expect("write row 1");

        writer.finalize().expect("close dbf writer");
        tmp
    }

    #[test]
    fn reads_basic_fields() {
        let tmp = write_synthetic_dbf();
        let records = read_flowpath_records(tmp.path()).expect("read dbf");
        assert_eq!(records.len(), 2);

        let r0 = &records[0];
        assert_eq!(r0.comid, 123456);
        assert!((r0.lengthkm - 2.345).abs() < 1e-9);
        assert!((r0.slope - 0.001).abs() < 1e-9);
        assert_eq!(r0.next_down_id, 654321);
        assert_eq!(r0.up, [111111, 222222, 0, 0]);
    }

    #[test]
    fn null_floats_become_nan() {
        let tmp = write_synthetic_dbf();
        let records = read_flowpath_records(tmp.path()).expect("read dbf");

        let r1 = &records[1];
        assert_eq!(r1.comid, 999999);
        assert!(r1.lengthkm.is_nan(), "lengthkm null should be NaN");
        assert!(r1.slope.is_nan(), "slope null should be NaN");
        assert_eq!(r1.next_down_id, 0); // terminal reach
        assert_eq!(r1.up, [0, 0, 0, 0]);
    }

    #[test]
    fn shp_path_resolved_to_dbf() {
        let tmp = write_synthetic_dbf();
        // Rename to .shp so the path ends in .shp — reader should find sibling .dbf.
        // Easier: just pass a path ending in .shp that points to the .dbf on disk.
        let shp_path = tmp.path().with_extension("shp");
        // The .shp file doesn't exist, but resolve_dbf_path returns the .dbf path.
        // We can test resolve_dbf_path directly without needing an actual .shp file.
        let resolved = resolve_dbf_path(&shp_path);
        assert_eq!(resolved.extension().and_then(|e| e.to_str()), Some("dbf"));
        assert_eq!(resolved.file_stem(), tmp.path().file_stem());

        // Also verify that passing the .dbf path directly works end-to-end.
        let records = read_flowpath_records(tmp.path()).expect("read via .dbf path");
        assert_eq!(records.len(), 2);
    }

    #[test]
    fn shp_path_reads_sibling_dbf() {
        let tmp = write_synthetic_dbf();
        // Create a .shp path that is a sibling of the .dbf (same stem, same dir).
        // We need a path ending in .shp where replacing .shp→.dbf gives tmp.path().
        // Construct that path and call read_flowpath_records with it.
        let dbf_path = tmp.path();
        let shp_path = dbf_path.with_extension("shp");
        // resolve_dbf_path will map shp_path → dbf_path, which exists.
        let records = read_flowpath_records(&shp_path).expect("read via .shp sibling path");
        assert_eq!(records.len(), 2);
        assert_eq!(records[0].comid, 123456);
    }

    #[test]
    fn null_comid_is_an_error() {
        // Write a dbf with a null COMID.
        let tmp = NamedTempFile::with_suffix(".dbf").unwrap();
        let builder = TableWriterBuilder::new()
            .add_numeric_field(FieldName::try_from("COMID").unwrap(), 10, 0)
            .add_numeric_field(FieldName::try_from("lengthkm").unwrap(), 12, 6)
            .add_numeric_field(FieldName::try_from("slope").unwrap(), 12, 6)
            .add_numeric_field(FieldName::try_from("NextDownID").unwrap(), 10, 0)
            .add_numeric_field(FieldName::try_from("up1").unwrap(), 10, 0)
            .add_numeric_field(FieldName::try_from("up2").unwrap(), 10, 0)
            .add_numeric_field(FieldName::try_from("up3").unwrap(), 10, 0)
            .add_numeric_field(FieldName::try_from("up4").unwrap(), 10, 0);

        let mut writer = builder.build_with_file_dest(tmp.path()).unwrap();
        let mut bad = dbase::Record::default();
        bad.insert("COMID".to_owned(), dbase::FieldValue::Numeric(None));
        bad.insert("lengthkm".to_owned(), dbase::FieldValue::Numeric(Some(1.0)));
        bad.insert("slope".to_owned(), dbase::FieldValue::Numeric(Some(0.01)));
        bad.insert("NextDownID".to_owned(), dbase::FieldValue::Numeric(Some(0.0)));
        bad.insert("up1".to_owned(), dbase::FieldValue::Numeric(Some(0.0)));
        bad.insert("up2".to_owned(), dbase::FieldValue::Numeric(Some(0.0)));
        bad.insert("up3".to_owned(), dbase::FieldValue::Numeric(Some(0.0)));
        bad.insert("up4".to_owned(), dbase::FieldValue::Numeric(Some(0.0)));
        writer.write_record(&bad).unwrap();
        writer.finalize().unwrap();

        let result = read_flowpath_records(tmp.path());
        assert!(result.is_err(), "null COMID should produce an error");
        let msg = result.unwrap_err().to_string();
        assert!(msg.contains("COMID"), "error message should mention COMID: {msg}");
    }

    #[test]
    fn resolve_dbf_path_unchanged_for_dbf() {
        let p = PathBuf::from("/some/dir/rivers.dbf");
        assert_eq!(resolve_dbf_path(&p), p);
    }

    #[test]
    fn resolve_dbf_path_swaps_shp_extension() {
        let shp = PathBuf::from("/some/dir/rivers.shp");
        let expected = PathBuf::from("/some/dir/rivers.dbf");
        assert_eq!(resolve_dbf_path(&shp), expected);
    }
}

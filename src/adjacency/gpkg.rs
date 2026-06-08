//! Read flowpath attributes from a MERIT Hydro GeoPackage (`.gpkg`) file.
//!
//! A GeoPackage is a SQLite database; the attribute columns are plain SQL
//! columns and the geometry is a BLOB column we never touch — only
//! `SELECT`ing the eight topology/attribute columns keeps reads fast even on
//! a multi-GB global fabric (geometry blobs are skipped by column selection).
//!
//! Rows are read in `ROWID` order. **This ordering is load-bearing**: record
//! order feeds the graph-insertion order in `build.rs`, which feeds the
//! deterministic DFS toposort, and the content-addressed adjacency cache
//! requires byte-identical rebuilds from identical inputs.
//!
//! ## Null semantics (mirrors `dbf.rs` — its module docs are the contract)
//!
//! Float columns (`lengthkm`, `slope`): SQL `NULL` → `f64::NAN`, matching the
//! dBASE reader and geopandas fill so the downstream mean-fill step produces
//! identical values. Integer columns (`COMID`, `NextDownID`, `up1..up4`):
//! SQL `NULL` is malformed data and errors with row + column context.
//!
//! SQLite typing is dynamic: integer columns may come back as `REAL` (e.g.
//! when written by tooling that promotes to double). `REAL` is accepted for
//! integer columns via `as i64`, mirroring `dbf.rs`'s `*v as i64` on dBASE
//! `Numeric` fields. `TEXT`/`BLOB` storage is an error.

use std::path::Path;

use rusqlite::types::ValueRef;
use rusqlite::{Connection, OpenFlags};

use crate::adjacency::dbf::FlowpathRecord;
use crate::data::error::DataError;

/// Read all flowpath records from a GeoPackage feature layer, in ROWID order.
///
/// `layer` selects the feature table. When `None`, the gpkg must contain
/// exactly one feature layer (per `gpkg_contents`); zero or multiple layers
/// produce an error naming the candidates and the `geospatial_fabric_layer`
/// config key.
pub fn read_flowpath_records_gpkg(
    path: &Path,
    layer: Option<&str>,
) -> crate::data::error::Result<Vec<FlowpathRecord>> {
    let conn = Connection::open_with_flags(path, OpenFlags::SQLITE_OPEN_READ_ONLY)
        .map_err(|e| malformed(path, format!("failed to open gpkg: {e}")))?;

    let table = resolve_layer(path, &conn, layer)?;

    // `ORDER BY ROWID` — see module docs; determinism is required by the
    // content-addressed cache.
    let sql = format!(
        "SELECT COMID, lengthkm, slope, NextDownID, up1, up2, up3, up4 \
         FROM \"{table}\" ORDER BY ROWID"
    );
    let mut stmt = conn.prepare(&sql).map_err(|e| {
        malformed(
            path,
            format!("layer '{table}': failed to prepare attribute query (missing column?): {e}"),
        )
    })?;

    let mut records = Vec::new();
    let mut rows = stmt
        .query([])
        .map_err(|e| malformed(path, format!("layer '{table}': query failed: {e}")))?;
    let mut row_idx = 0usize;
    while let Some(row) = rows
        .next()
        .map_err(|e| malformed(path, format!("layer '{table}': row {row_idx}: read failed: {e}")))?
    {
        let comid = extract_int(path, row_idx, row, 0, "COMID")?;
        let lengthkm = extract_float_nullable(path, row_idx, row, 1, "lengthkm")?;
        let slope = extract_float_nullable(path, row_idx, row, 2, "slope")?;
        let next_down_id = extract_int(path, row_idx, row, 3, "NextDownID")?;
        let up1 = extract_int(path, row_idx, row, 4, "up1")?;
        let up2 = extract_int(path, row_idx, row, 5, "up2")?;
        let up3 = extract_int(path, row_idx, row, 6, "up3")?;
        let up4 = extract_int(path, row_idx, row, 7, "up4")?;
        records.push(FlowpathRecord {
            comid,
            lengthkm,
            slope,
            next_down_id,
            up: [up1, up2, up3, up4],
        });
        row_idx += 1;
    }
    Ok(records)
}

// ---------------------------------------------------------------------------
// Internal helpers
// ---------------------------------------------------------------------------

fn malformed(path: &Path, message: String) -> DataError {
    DataError::Malformed {
        path: path.to_path_buf(),
        message,
    }
}

/// Resolve the feature table name from `gpkg_contents`.
fn resolve_layer(
    path: &Path,
    conn: &Connection,
    layer: Option<&str>,
) -> crate::data::error::Result<String> {
    let mut stmt = conn
        .prepare("SELECT table_name FROM gpkg_contents WHERE data_type = 'features'")
        .map_err(|e| {
            malformed(
                path,
                format!("not a GeoPackage (no readable gpkg_contents table): {e}"),
            )
        })?;
    let layers: Vec<String> = stmt
        .query_map([], |row| row.get::<_, String>(0))
        .and_then(|rows| rows.collect())
        .map_err(|e| malformed(path, format!("failed to list gpkg feature layers: {e}")))?;

    match layer {
        Some(want) => {
            if layers.iter().any(|l| l == want) {
                Ok(want.to_string())
            } else {
                Err(malformed(
                    path,
                    format!(
                        "feature layer '{want}' not found; available layers: [{}]",
                        layers.join(", ")
                    ),
                ))
            }
        }
        None => match layers.as_slice() {
            [only] => Ok(only.clone()),
            [] => Err(malformed(
                path,
                "no feature layers in gpkg_contents".to_string(),
            )),
            many => Err(malformed(
                path,
                format!(
                    "multiple feature layers: [{}] — set data_sources.geospatial_fabric_layer \
                     to choose one",
                    many.join(", ")
                ),
            )),
        },
    }
}

/// Extract an integer-valued column. SQL `NULL` is malformed (a flowline
/// without a topology ID is unusable). `REAL` storage is accepted via
/// `as i64` (mirrors `dbf.rs::extract_int`).
fn extract_int(
    path: &Path,
    row_idx: usize,
    row: &rusqlite::Row<'_>,
    col_idx: usize,
    col: &str,
) -> crate::data::error::Result<i64> {
    match row.get_ref(col_idx) {
        Ok(ValueRef::Integer(v)) => Ok(v),
        Ok(ValueRef::Real(v)) => Ok(v as i64),
        Ok(ValueRef::Null) => Err(malformed(
            path,
            format!("row {row_idx}: column '{col}' is null — expected non-null integer"),
        )),
        Ok(other) => Err(malformed(
            path,
            format!(
                "row {row_idx}: column '{col}' has unexpected SQLite type {:?}",
                other.data_type()
            ),
        )),
        Err(e) => Err(malformed(
            path,
            format!("row {row_idx}: column '{col}': {e}"),
        )),
    }
}

/// Extract a float-valued column, mapping SQL `NULL → NaN`
/// (mirrors `dbf.rs::extract_float_nullable`).
fn extract_float_nullable(
    path: &Path,
    row_idx: usize,
    row: &rusqlite::Row<'_>,
    col_idx: usize,
    col: &str,
) -> crate::data::error::Result<f64> {
    match row.get_ref(col_idx) {
        Ok(ValueRef::Real(v)) => Ok(v),
        Ok(ValueRef::Integer(v)) => Ok(v as f64),
        Ok(ValueRef::Null) => Ok(f64::NAN),
        Ok(other) => Err(malformed(
            path,
            format!(
                "row {row_idx}: column '{col}' has unexpected SQLite type {:?}",
                other.data_type()
            ),
        )),
        Err(e) => Err(malformed(
            path,
            format!("row {row_idx}: column '{col}': {e}"),
        )),
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::NamedTempFile;

    /// Build a minimal-but-valid GeoPackage: `gpkg_contents` plus one feature
    /// table with MERIT's attribute columns (geometry column present but
    /// unused, as in real gpkg files). Raw SQL — no GDAL involved.
    pub(crate) fn write_synthetic_gpkg(rows: &[(i64, Option<f64>, Option<f64>, i64, [i64; 4])]) -> NamedTempFile {
        let tmp = NamedTempFile::with_suffix(".gpkg").unwrap();
        let conn = Connection::open(tmp.path()).unwrap();
        conn.execute_batch(
            "CREATE TABLE gpkg_contents (
                 table_name TEXT NOT NULL PRIMARY KEY,
                 data_type TEXT NOT NULL,
                 identifier TEXT,
                 srs_id INTEGER
             );
             CREATE TABLE flowlines (
                 fid INTEGER PRIMARY KEY AUTOINCREMENT,
                 geom BLOB,
                 COMID INTEGER,
                 lengthkm DOUBLE,
                 slope DOUBLE,
                 NextDownID INTEGER,
                 up1 INTEGER, up2 INTEGER, up3 INTEGER, up4 INTEGER
             );
             INSERT INTO gpkg_contents (table_name, data_type, identifier, srs_id)
             VALUES ('flowlines', 'features', 'flowlines', 4326);",
        )
        .unwrap();
        let mut stmt = conn
            .prepare(
                "INSERT INTO flowlines
                 (geom, COMID, lengthkm, slope, NextDownID, up1, up2, up3, up4)
                 VALUES (NULL, ?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8)",
            )
            .unwrap();
        for (comid, lengthkm, slope, next_down, up) in rows {
            stmt.execute(rusqlite::params![
                comid, lengthkm, slope, next_down, up[0], up[1], up[2], up[3]
            ])
            .unwrap();
        }
        drop(stmt);
        drop(conn);
        tmp
    }

    #[test]
    fn reads_basic_fields() {
        let tmp = write_synthetic_gpkg(&[
            (123456, Some(2.345), Some(0.001), 654321, [111111, 222222, 0, 0]),
            (999999, None, None, 0, [0, 0, 0, 0]),
        ]);
        let records = read_flowpath_records_gpkg(tmp.path(), None).expect("read gpkg");
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
        let tmp = write_synthetic_gpkg(&[(999999, None, None, 0, [0, 0, 0, 0])]);
        let records = read_flowpath_records_gpkg(tmp.path(), None).expect("read gpkg");
        let r = &records[0];
        assert!(r.lengthkm.is_nan(), "lengthkm null should be NaN");
        assert!(r.slope.is_nan(), "slope null should be NaN");
        assert_eq!(r.next_down_id, 0);
    }

    #[test]
    fn null_comid_is_an_error() {
        let tmp = write_synthetic_gpkg(&[]);
        let conn = Connection::open(tmp.path()).unwrap();
        conn.execute(
            "INSERT INTO flowlines (geom, COMID, lengthkm, slope, NextDownID, up1, up2, up3, up4)
             VALUES (NULL, NULL, 1.0, 0.01, 0, 0, 0, 0, 0)",
            [],
        )
        .unwrap();
        drop(conn);
        let result = read_flowpath_records_gpkg(tmp.path(), None);
        assert!(result.is_err(), "null COMID should produce an error");
        let msg = result.unwrap_err().to_string();
        assert!(msg.contains("COMID"), "error must mention COMID: {msg}");
    }

    #[test]
    fn explicit_layer_selection_works() {
        let tmp = write_synthetic_gpkg(&[(1, Some(1.0), Some(0.01), 0, [0, 0, 0, 0])]);
        let records =
            read_flowpath_records_gpkg(tmp.path(), Some("flowlines")).expect("explicit layer");
        assert_eq!(records.len(), 1);
    }

    #[test]
    fn missing_explicit_layer_names_available() {
        let tmp = write_synthetic_gpkg(&[]);
        let err = read_flowpath_records_gpkg(tmp.path(), Some("nope")).unwrap_err();
        let msg = err.to_string();
        assert!(msg.contains("nope") && msg.contains("flowlines"), "{msg}");
    }

    #[test]
    fn multiple_layers_without_selection_errors() {
        let tmp = write_synthetic_gpkg(&[(1, Some(1.0), Some(0.01), 0, [0, 0, 0, 0])]);
        let conn = Connection::open(tmp.path()).unwrap();
        conn.execute_batch(
            "CREATE TABLE catchments (fid INTEGER PRIMARY KEY, geom BLOB, COMID INTEGER);
             INSERT INTO gpkg_contents (table_name, data_type, identifier, srs_id)
             VALUES ('catchments', 'features', 'catchments', 4326);",
        )
        .unwrap();
        drop(conn);
        let err = read_flowpath_records_gpkg(tmp.path(), None).unwrap_err();
        let msg = err.to_string();
        assert!(
            msg.contains("geospatial_fabric_layer") && msg.contains("catchments"),
            "must list layers and name the config key: {msg}"
        );
    }

    #[test]
    fn real_typed_integer_columns_accepted() {
        // SQLite dynamic typing: write COMID as REAL; reader must coerce.
        let tmp = write_synthetic_gpkg(&[]);
        let conn = Connection::open(tmp.path()).unwrap();
        conn.execute(
            "INSERT INTO flowlines (geom, COMID, lengthkm, slope, NextDownID, up1, up2, up3, up4)
             VALUES (NULL, 42.0, 1.0, 0.01, 0.0, 0, 0, 0, 0)",
            [],
        )
        .unwrap();
        drop(conn);
        let records = read_flowpath_records_gpkg(tmp.path(), None).expect("read gpkg");
        assert_eq!(records[0].comid, 42);
        assert_eq!(records[0].next_down_id, 0);
    }

    #[test]
    fn rows_come_back_in_rowid_order() {
        let tmp = write_synthetic_gpkg(&[
            (30, Some(1.0), Some(0.01), 0, [0, 0, 0, 0]),
            (10, Some(1.0), Some(0.01), 0, [0, 0, 0, 0]),
            (20, Some(1.0), Some(0.01), 0, [0, 0, 0, 0]),
        ]);
        let records = read_flowpath_records_gpkg(tmp.path(), None).expect("read gpkg");
        let comids: Vec<i64> = records.iter().map(|r| r.comid).collect();
        assert_eq!(comids, vec![30, 10, 20], "must preserve insertion (ROWID) order");
    }
}

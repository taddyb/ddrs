//! GeoPackage fabric integration tests.
//!
//! The keystone is the dbf↔gpkg parity test: the same flowpath records
//! written as a dBASE table and as a GeoPackage feature layer must produce
//! (a) element-identical `Vec<FlowpathRecord>` through `read_fabric_records`,
//! and (b) byte-identical adjacency (`order`/`indices_0`/`indices_1`) through
//! `build_conus_adjacency` — the same bar as `tests/adjacency_parity.rs`
//! holds engine-vs-builder stores to.
//!
//! The `#[ignore]` smoke test exercises the real merged global fabric at
//! `/projects/mhpi/data/MERIT/raw/global_merit_riv.gpkg` (2,939,408 reaches);
//! run with `cargo test --release --test gpkg_fabric -- --ignored`.

use std::convert::TryFrom;
use std::path::Path;

use ddrs::adjacency::build::build_conus_adjacency;
use ddrs::adjacency::fabric::read_fabric_records;

/// (COMID, lengthkm, slope, NextDownID, [up1..up4])
type Row = (i64, f64, f64, i64, [i64; 4]);

/// A small dendritic network with a confluence and two terminal outlets:
///
/// ```text
///   101 ─┐
///        ├──► 103 ──► 104 (outlet)
///   102 ─┘
///   201 ──► 202 (outlet)
/// ```
fn synthetic_rows() -> Vec<Row> {
    vec![
        (101, 1.5, 0.010, 103, [0, 0, 0, 0]),
        (102, 2.0, 0.020, 103, [0, 0, 0, 0]),
        (103, 3.0, 0.005, 104, [101, 102, 0, 0]),
        (104, 4.0, 0.002, 0, [103, 0, 0, 0]),
        (201, 1.0, 0.030, 202, [0, 0, 0, 0]),
        (202, 2.5, 0.008, 0, [201, 0, 0, 0]),
    ]
}

fn write_dbf(path: &Path, rows: &[Row]) {
    use dbase::{FieldName, FieldValue, Record, TableWriterBuilder};
    let builder = TableWriterBuilder::new()
        .add_numeric_field(FieldName::try_from("COMID").unwrap(), 10, 0)
        .add_numeric_field(FieldName::try_from("lengthkm").unwrap(), 12, 6)
        .add_numeric_field(FieldName::try_from("slope").unwrap(), 12, 6)
        .add_numeric_field(FieldName::try_from("NextDownID").unwrap(), 10, 0)
        .add_numeric_field(FieldName::try_from("up1").unwrap(), 10, 0)
        .add_numeric_field(FieldName::try_from("up2").unwrap(), 10, 0)
        .add_numeric_field(FieldName::try_from("up3").unwrap(), 10, 0)
        .add_numeric_field(FieldName::try_from("up4").unwrap(), 10, 0);
    let mut writer = builder.build_with_file_dest(path).expect("dbf writer");
    for (comid, lengthkm, slope, next_down, up) in rows {
        let mut r = Record::default();
        r.insert("COMID".into(), FieldValue::Numeric(Some(*comid as f64)));
        r.insert("lengthkm".into(), FieldValue::Numeric(Some(*lengthkm)));
        r.insert("slope".into(), FieldValue::Numeric(Some(*slope)));
        r.insert("NextDownID".into(), FieldValue::Numeric(Some(*next_down as f64)));
        for (i, u) in up.iter().enumerate() {
            r.insert(format!("up{}", i + 1), FieldValue::Numeric(Some(*u as f64)));
        }
        writer.write_record(&r).expect("write dbf row");
    }
    writer.finalize().expect("finalize dbf");
}

fn write_gpkg(path: &Path, rows: &[Row]) {
    let conn = rusqlite::Connection::open(path).expect("create gpkg");
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
    .expect("gpkg schema");
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
        .expect("insert gpkg row");
    }
}

#[test]
fn dbf_and_gpkg_yield_identical_records_and_adjacency() {
    let dir = tempfile::tempdir().unwrap();
    let dbf_path = dir.path().join("fabric.dbf");
    let gpkg_path = dir.path().join("fabric.gpkg");
    let rows = synthetic_rows();
    write_dbf(&dbf_path, &rows);
    write_gpkg(&gpkg_path, &rows);

    let rec_dbf = read_fabric_records(&dbf_path, None).expect("read dbf");
    let rec_gpkg = read_fabric_records(&gpkg_path, None).expect("read gpkg");
    assert_eq!(rec_dbf.len(), rows.len());
    assert_eq!(
        rec_dbf, rec_gpkg,
        "FlowpathRecord streams must be element-identical across formats"
    );

    // Same bar as tests/adjacency_parity.rs: order/indices byte-identical.
    let adj_dbf = build_conus_adjacency(&rec_dbf).expect("build from dbf");
    let adj_gpkg = build_conus_adjacency(&rec_gpkg).expect("build from gpkg");
    assert_eq!(adj_dbf.order, adj_gpkg.order, "`order` must be identical");
    assert_eq!(adj_dbf.rows, adj_gpkg.rows, "`indices_0` must be identical");
    assert_eq!(adj_dbf.cols, adj_gpkg.cols, "`indices_1` must be identical");
    assert_eq!(adj_dbf.length_m, adj_gpkg.length_m);
    assert_eq!(adj_dbf.slope, adj_gpkg.slope);
    assert!(adj_dbf.dropped_comids.is_empty());
}

#[test]
fn gpkg_read_is_deterministic() {
    // Two reads of the same gpkg must agree element-for-element — guards the
    // ORDER BY ROWID contract the content-addressed cache depends on.
    let dir = tempfile::tempdir().unwrap();
    let gpkg_path = dir.path().join("fabric.gpkg");
    write_gpkg(&gpkg_path, &synthetic_rows());

    let a = read_fabric_records(&gpkg_path, None).expect("first read");
    let b = read_fabric_records(&gpkg_path, None).expect("second read");
    assert_eq!(a, b);
}

#[test]
fn explicit_layer_matches_implicit_single_layer() {
    let dir = tempfile::tempdir().unwrap();
    let gpkg_path = dir.path().join("fabric.gpkg");
    write_gpkg(&gpkg_path, &synthetic_rows());

    let implicit = read_fabric_records(&gpkg_path, None).expect("implicit layer");
    let explicit = read_fabric_records(&gpkg_path, Some("flowlines")).expect("explicit layer");
    assert_eq!(implicit, explicit);
}

/// Real-data smoke on the merged global MERIT flowlines. Ignored by default
/// (host-specific path, multi-GB input); run with `-- --ignored` on wukong.
#[test]
#[ignore = "requires /projects/mhpi/data/MERIT/raw/global_merit_riv.gpkg (wukong)"]
fn global_merit_riv_gpkg_builds() {
    let path = Path::new("/projects/mhpi/data/MERIT/raw/global_merit_riv.gpkg");
    assert!(path.exists(), "global fabric not found at {}", path.display());

    let t0 = std::time::Instant::now();
    let records = read_fabric_records(path, Some("flowlines")).expect("read global gpkg");
    eprintln!("read {} records in {:.1}s", records.len(), t0.elapsed().as_secs_f64());
    assert_eq!(records.len(), 2_939_408, "global fabric reach count");

    // COMIDs must be unique (the merge already verified; re-check at the
    // FlowpathRecord level so a reader bug can't silently duplicate).
    let mut comids: Vec<i64> = records.iter().map(|r| r.comid).collect();
    comids.sort_unstable();
    comids.dedup();
    assert_eq!(comids.len(), 2_939_408, "COMIDs must be unique");

    let t1 = std::time::Instant::now();
    let adj = build_conus_adjacency(&records).expect("build global adjacency");
    eprintln!(
        "built adjacency: n={} nnz={} dropped={} in {:.1}s",
        adj.order.len(),
        adj.rows.len(),
        adj.dropped_comids.len(),
        t1.elapsed().as_secs_f64()
    );
    assert_eq!(
        adj.order.len() + adj.dropped_comids.len(),
        2_939_408,
        "every reach is either ordered or dropped-on-cycle"
    );
    // Lower-triangular invariant the forward-sub solver assumes.
    assert!(
        adj.rows.iter().zip(&adj.cols).all(|(r, c)| r >= c),
        "adjacency must be lower-triangular (rows[k] >= cols[k])"
    );
}

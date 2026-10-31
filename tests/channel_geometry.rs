//! Table geometry reader (`data_sources.channel_geometry`).

use std::path::PathBuf;

use ddrs::data::error::DataError;
use ddrs::data::ids::Comid;
use ddrs::data::store::ChannelGeometryStore;

fn write_csv(dir: &tempfile::TempDir, body: &str) -> PathBuf {
    let p = dir.path().join("geometry.csv");
    std::fs::write(&p, body).unwrap();
    p
}

#[test]
fn channel_geometry_reads_and_maps_rows() {
    let dir = tempfile::tempdir().unwrap();
    // A negative bed (NAVD88), an extra `source` column, a comment line, and a
    // row with and a row without pool_z_m (C25).
    let p = write_csv(
        &dir,
        "# Hudson estuary, hand table\n\
         COMID,top_width_m,bed_z_m,n,pool_z_m,source\n\
         101,1000.0,-10.0,0.022,,usgs\n\
         103,250.5,115.2,0.035,118.0,dem\n\
         999,10.0,0.0,0.03,,none\n",
    );
    let store = ChannelGeometryStore::open(&p).unwrap();
    assert_eq!(store.path, p);
    let network = [Comid(103), Comid(7), Comid(101)];
    let rows = store.rows_for(&network);
    assert_eq!(rows.rows, vec![0, 2]);
    assert_eq!(rows.top_width, vec![250.5, 1000.0]);
    assert_eq!(rows.bed_z, vec![115.2, -10.0]);
    assert_eq!(rows.n, vec![0.035, 0.022]);
    assert_eq!(rows.pool_z, vec![Some(118.0), None]);
}

#[test]
fn channel_geometry_rejects_bad_rows() {
    let dir = tempfile::tempdir().unwrap();
    let cases: &[(&str, &str)] = &[
        ("COMID,top_width_m,bed_z_m\n1,10,0\n", "n"),
        ("COMID,top_width_m,bed_z_m,n\n1,0,0,0.03\n", "top_width_m"),
        ("COMID,top_width_m,bed_z_m,n\n1,10,0,0\n", "n"),
        ("COMID,top_width_m,bed_z_m,n\n1,10,0,-0.03\n", "n"),
        ("COMID,top_width_m,bed_z_m,n\n1,10,0,0.03\n1,20,0,0.03\n", "duplicate"),
        ("COMID,top_width_m,bed_z_m,n\n1,NaN,0,0.03\n", "top_width_m"),
        ("COMID,top_width_m,bed_z_m,n\n1,10,NaN,0.03\n", "bed_z_m"),
        ("COMID,top_width_m,bed_z_m,n\n1,10,0,NaN\n", "n"),
        ("COMID,top_width_m,bed_z_m,n,pool_z_m\n1,10,0,0.03,inf\n", "pool_z_m"),
        ("COMID,top_width_m,bed_z_m,n\n", "no data rows"),
    ];
    for (body, needle) in cases {
        let p = write_csv(&dir, body);
        match ChannelGeometryStore::open(&p) {
            Err(DataError::Malformed { path, message }) => {
                assert_eq!(path, p);
                assert!(message.contains(needle), "{body:?}: message {message:?} lacks {needle:?}");
            }
            other => panic!("{body:?}: expected Malformed, got {other:?}"),
        }
    }
    // The row number is named.
    let p = write_csv(&dir, "COMID,top_width_m,bed_z_m,n\n1,10,0,0.03\n2,-5,0,0.03\n");
    let msg = match ChannelGeometryStore::open(&p) {
        Err(DataError::Malformed { message, .. }) => message,
        other => panic!("{other:?}"),
    };
    assert!(msg.contains("row 2"), "{msg}");
}

#[test]
fn channel_geometry_must_cover_the_stage_set() {
    let dir = tempfile::tempdir().unwrap();
    let p = write_csv(
        &dir,
        "COMID,top_width_m,bed_z_m,n\n10,100,0,0.03\n30,100,0,0.03\n",
    );
    let store = ChannelGeometryStore::open(&p).unwrap();
    let network = [Comid(10), Comid(20), Comid(30)];
    // Stage set {20, 30}: COMID 20 has no row. 10 is not stage, so it is not asked for.
    let missing = store.uncovered_stage_comids(&network, &[false, true, true]);
    assert_eq!(missing, vec![Comid(20)]);
    assert!(store.uncovered_stage_comids(&network, &[true, false, true]).is_empty());
}

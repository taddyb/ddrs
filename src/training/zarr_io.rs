//! Write per-gauge daily predictions + observations to a zarr store with
//! a layout compatible with DDR's `_test` output.
//!
//! Format: zarr v3 (zarrs 0.23 default).
//!
//! Inspection of `ddr/data/merit_conus_adjacency.zarr`:
//!   zarr_format = 3
//!   Arrays use float32/int32/uint8; no `_ARRAY_DIMENSIONS` attrs (COO store
//!   has no xarray convention). DDR's xarray-based prediction output will use
//!   float64 for predictions/observations and int64 (ns since epoch) for time.
//!
//! Layout:
//!   /predictions   (n_gauges, n_days)  f64  attrs {units, long_name, _ARRAY_DIMENSIONS}
//!   /observations  (n_gauges, n_days)  f64  attrs {units, long_name, _ARRAY_DIMENSIONS}
//!   /gage_ids      (n_gauges, 8)       u8   attrs {_ARRAY_DIMENSIONS, _dtype_hint=|S8}
//!   /time          (n_days,)           i64  attrs {units, calendar, _ARRAY_DIMENSIONS}
//!
//! Group attrs: description, "start time", "end time", version,
//! "evaluation basins file", model.
//!
//! Implementation note: each array is written as a single chunk (full array =
//! one chunk) via `store_chunk`, which only requires `WritableStorageTraits`.
//! This avoids the `ReadableWritableStorageTraits` bound that `store_array_subset`
//! would impose.

use std::path::Path;
use std::sync::Arc;

use zarrs::array::{ArrayBuilder, data_type};
use zarrs::filesystem::FilesystemStore;
use zarrs::group::GroupBuilder;

use crate::data::error::{DataError, Result};
use crate::training::eval::EvalOutput;

pub struct ZarrAttrs<'a> {
    pub start_time: &'a str,
    pub end_time: &'a str,
    pub version: &'a str,
    pub evaluation_basins_file: &'a Path,
    pub model_label: &'a str,
}

pub fn write_predictions_zarr(
    path: &Path,
    output: &EvalOutput,
    attrs: ZarrAttrs<'_>,
) -> Result<()> {
    let storage = Arc::new(FilesystemStore::new(path).map_err(|e| zarr_err(path, e))?);

    // Root group attrs.
    let mut root_attrs = serde_json::Map::new();
    root_attrs.insert(
        "description".into(),
        serde_json::Value::String("Predictions and obs for time period".into()),
    );
    root_attrs.insert(
        "start time".into(),
        serde_json::Value::String(attrs.start_time.into()),
    );
    root_attrs.insert(
        "end time".into(),
        serde_json::Value::String(attrs.end_time.into()),
    );
    root_attrs.insert(
        "version".into(),
        serde_json::Value::String(attrs.version.into()),
    );
    root_attrs.insert(
        "evaluation basins file".into(),
        serde_json::Value::String(attrs.evaluation_basins_file.display().to_string()),
    );
    root_attrs.insert(
        "model".into(),
        serde_json::Value::String(attrs.model_label.into()),
    );

    let root = GroupBuilder::new()
        .attributes(root_attrs)
        .build(storage.clone(), "/")
        .map_err(|e| zarr_err(path, e))?;
    root.store_metadata().map_err(|e| zarr_err(path, e))?;

    let (n_gauges, n_days) = output.predictions_daily.dim();

    // /predictions — f64, (n_gauges, n_days), single chunk.
    let predictions_f64: Vec<f64> = output.predictions_daily.iter().map(|&v| v as f64).collect();
    write_2d_f64(
        &storage,
        path,
        "/predictions",
        &predictions_f64,
        (n_gauges, n_days),
        &[("units", "m3/s"), ("long_name", "Streamflow")],
        &["gage_ids", "time"],
    )?;

    // /observations — f64, (n_gauges, n_days), single chunk.
    let obs_f64: Vec<f64> = output.observations_daily.iter().map(|&v| v as f64).collect();
    write_2d_f64(
        &storage,
        path,
        "/observations",
        &obs_f64,
        (n_gauges, n_days),
        &[("units", "m3/s"), ("long_name", "Observed Streamflow")],
        &["gage_ids", "time"],
    )?;

    // /gage_ids — u8, (n_gauges, 8), fixed-width ASCII.
    // zarr v3 has no native fixed-length string dtype; we encode as UInt8
    // with a `_dtype_hint: "|S8"` attr for downstream readers.
    write_gage_ids_u8(&storage, path, &output.gage_ids)?;

    // /time — i64, (n_days,), nanoseconds since epoch.
    let time_ns: Vec<i64> = output
        .time_range_daily
        .iter()
        .map(|d| {
            d.and_hms_opt(0, 0, 0)
                .unwrap()
                .and_utc()
                .timestamp_nanos_opt()
                .unwrap()
        })
        .collect();
    write_1d_i64(
        &storage,
        path,
        "/time",
        &time_ns,
        &[
            ("units", "nanoseconds since 1970-01-01"),
            ("calendar", "proleptic_gregorian"),
        ],
        &["time"],
    )?;

    Ok(())
}

// ---------- private helpers ----------

fn write_2d_f64(
    storage: &Arc<FilesystemStore>,
    store_path: &Path,
    array_path: &str,
    data: &[f64],
    shape: (usize, usize),
    kv_attrs: &[(&str, &str)],
    array_dimensions: &[&str],
) -> Result<()> {
    let mut attr_map = str_attrs(kv_attrs);
    attr_map.insert(
        "_ARRAY_DIMENSIONS".into(),
        serde_json::Value::Array(
            array_dimensions
                .iter()
                .map(|s| serde_json::Value::String(s.to_string()))
                .collect(),
        ),
    );

    let array = ArrayBuilder::new(
        vec![shape.0 as u64, shape.1 as u64],
        vec![shape.0 as u64, shape.1 as u64], // one chunk covers the whole array
        data_type::float64(),
        0.0_f64,
    )
    .attributes(attr_map)
    .build(storage.clone(), array_path)
    .map_err(|e| zarr_err(store_path, e))?;

    array.store_metadata().map_err(|e| zarr_err(store_path, e))?;
    // Single chunk at index [0, 0].
    array
        .store_chunk(&[0, 0], data)
        .map_err(|e| zarr_err(store_path, e))?;
    Ok(())
}

fn write_1d_i64(
    storage: &Arc<FilesystemStore>,
    store_path: &Path,
    array_path: &str,
    data: &[i64],
    kv_attrs: &[(&str, &str)],
    array_dimensions: &[&str],
) -> Result<()> {
    let mut attr_map = str_attrs(kv_attrs);
    attr_map.insert(
        "_ARRAY_DIMENSIONS".into(),
        serde_json::Value::Array(
            array_dimensions
                .iter()
                .map(|s| serde_json::Value::String(s.to_string()))
                .collect(),
        ),
    );

    let array = ArrayBuilder::new(
        vec![data.len() as u64],
        vec![data.len() as u64],
        data_type::int64(),
        0_i64,
    )
    .attributes(attr_map)
    .build(storage.clone(), array_path)
    .map_err(|e| zarr_err(store_path, e))?;

    array.store_metadata().map_err(|e| zarr_err(store_path, e))?;
    // Single chunk at index [0].
    array
        .store_chunk(&[0], data)
        .map_err(|e| zarr_err(store_path, e))?;
    Ok(())
}

fn write_gage_ids_u8(
    storage: &Arc<FilesystemStore>,
    store_path: &Path,
    strings: &[String],
) -> Result<()> {
    let n = strings.len();
    // Zero-padded fixed-width ASCII, width = longest ID (min 8 to keep the
    // historical |S8 layout for USGS STAIDs). A hardcoded 8 silently
    // truncated global `Provider__GageId` names — e.g. `GRDC__1286661` →
    // `GRDC__12` — collapsing 5,224 gauges onto 93 distinct prefixes.
    let width = strings.iter().map(|s| s.len()).max().unwrap_or(0).max(8);
    let mut buf = vec![0u8; n * width];
    for (i, s) in strings.iter().enumerate() {
        let bytes = s.as_bytes();
        buf[i * width..i * width + bytes.len()].copy_from_slice(bytes);
    }

    let mut attr_map = serde_json::Map::new();
    attr_map.insert(
        "_ARRAY_DIMENSIONS".into(),
        serde_json::Value::Array(vec![
            serde_json::Value::String("gage_ids".into()),
            serde_json::Value::String("char".into()),
        ]),
    );
    attr_map.insert(
        "_dtype_hint".into(),
        serde_json::Value::String(format!("|S{width}")),
    );

    let array = ArrayBuilder::new(
        vec![n as u64, width as u64],
        vec![n as u64, width as u64],
        data_type::uint8(),
        0_u8,
    )
    .attributes(attr_map)
    .build(storage.clone(), "/gage_ids")
    .map_err(|e| zarr_err(store_path, e))?;

    array.store_metadata().map_err(|e| zarr_err(store_path, e))?;
    array
        .store_chunk(&[0, 0], buf.as_slice())
        .map_err(|e| zarr_err(store_path, e))?;
    Ok(())
}

fn str_attrs(pairs: &[(&str, &str)]) -> serde_json::Map<String, serde_json::Value> {
    let mut m = serde_json::Map::new();
    for (k, v) in pairs {
        m.insert(k.to_string(), serde_json::Value::String(v.to_string()));
    }
    m
}

fn zarr_err<E: std::error::Error + Send + Sync + 'static>(path: &Path, source: E) -> DataError {
    DataError::Zarr {
        path: path.to_path_buf(),
        source: Box::new(source),
    }
}

// ---------- tests ----------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::training::metrics::Metrics;
    use chrono::NaiveDate;
    use ndarray::array;
    use zarrs::array::Array as ZarrArray;
    use zarrs::storage::ReadableStorage;

    #[test]
    fn write_then_read_round_trip() {
        let pred = array![[1.0_f32, 2.0, 3.0], [4.0, 5.0, 6.0]]; // (G=2, T=3)
        let obs = array![[1.1_f32, 2.1, 3.1], [4.1, 5.1, 6.1]];
        let out = EvalOutput {
            predictions_daily: pred,
            observations_daily: obs,
            gage_ids: vec!["00000001".into(), "00000002".into()],
            time_range_daily: vec![
                NaiveDate::from_ymd_opt(1995, 10, 2).unwrap(),
                NaiveDate::from_ymd_opt(1995, 10, 3).unwrap(),
                NaiveDate::from_ymd_opt(1995, 10, 4).unwrap(),
            ],
            metrics: Metrics {
                nse: vec![0.5, 0.6],
                rmse: vec![0.1, 0.1],
                kge: vec![0.4, 0.5],
                bias: vec![0.0, 0.0],
                fhv: vec![0.0, 0.0],
                flv: vec![0.0, 0.0],
            },
            zeta_abs_mean: None,
            zeta_net_mean: None,
            zeta_comids: None,
        };

        let mut zpath = std::env::temp_dir();
        zpath.push(format!("ddrs_zarr_test_{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&zpath);
        std::fs::create_dir_all(&zpath).expect("mkdir");

        let attrs = ZarrAttrs {
            start_time: "1995-10-01",
            end_time: "1995-10-05",
            version: "test",
            evaluation_basins_file: std::path::Path::new("/tmp/fake_gages.csv"),
            model_label: "frozen",
        };
        write_predictions_zarr(&zpath, &out, attrs).expect("write");

        // Read back predictions and verify shape.
        let read_storage: ReadableStorage =
            Arc::new(FilesystemStore::new(&zpath).expect("open store"));
        let arr = ZarrArray::open(read_storage, "/predictions").expect("open predictions");
        assert_eq!(arr.shape(), &[2, 3]);
    }

    #[test]
    fn gage_ids_wider_than_8_bytes_round_trip_unclipped() {
        // Global Provider__GageId names exceed the historical 8-byte STAID
        // width; the writer must widen, not truncate (the old behavior
        // collapsed 5,224 global gauges onto 93 distinct 8-byte prefixes).
        let ids = vec![
            "GRDC__1286661".to_string(),       // 13 bytes
            "BOMAustralia__403213".to_string(), // 20 bytes
            "01013500".to_string(),             // legacy USGS, 8 bytes
        ];

        let mut zpath = std::env::temp_dir();
        zpath.push(format!("ddrs_zarr_gageids_test_{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&zpath);
        std::fs::create_dir_all(&zpath).expect("mkdir");
        let storage = Arc::new(FilesystemStore::new(&zpath).expect("store"));

        write_gage_ids_u8(&storage, &zpath, &ids).expect("write");

        let read_storage: ReadableStorage =
            Arc::new(FilesystemStore::new(&zpath).expect("open store"));
        let arr = ZarrArray::open(read_storage, "/gage_ids").expect("open gage_ids");
        let width = ids.iter().map(|s| s.len()).max().unwrap();
        assert_eq!(arr.shape(), &[3, width as u64]);
        assert_eq!(
            arr.attributes()["_dtype_hint"],
            serde_json::json!(format!("|S{width}")),
        );

        let bytes: Vec<u8> = arr
            .retrieve_array_subset::<Vec<u8>>(&arr.subset_all())
            .expect("read");
        for (i, id) in ids.iter().enumerate() {
            let row = &bytes[i * width..(i + 1) * width];
            let decoded: Vec<u8> =
                row.iter().copied().take_while(|&b| b != 0).collect();
            assert_eq!(std::str::from_utf8(&decoded).unwrap(), id);
        }
        let _ = std::fs::remove_dir_all(&zpath);
    }
}

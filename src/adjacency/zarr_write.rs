//! Write the managed-adjacency outputs as zarr-v3 stores that are
//! byte-compatible with `ddr_engine/core/zarr_io.py`.
//!
//! Two writers mirror the two readers in `src/data/store/zarr.rs` (the
//! compatibility oracle):
//!
//!   - [`write_conus_store`] writes a single root group holding the CONUS COO
//!     (`indices_0`/`indices_1` int32, `values` uint8 ones, `order` int32)
//!     **plus** `length_m`/`slope` (float32). The float columns are this port's
//!     addition (the engine's `coo_to_zarr` doesn't write them; see
//!     `build.rs` module docs for why they live in the builder now).
//!
//!   - [`write_gauges_store`] writes an empty root group with one subgroup per
//!     STAID, each carrying the same array set as the engine's
//!     `coo_to_zarr_group_generic` (`indices_0`/`indices_1`/`values`/`order`)
//!     plus the `gage_catchment`/`gage_idx` attrs the reader requires.
//!
//! ## Layout decisions vs the engine (`zarr_io.py`)
//!
//! - **Root attrs**: `format: "COO"`, `shape: [n, n]`, `geodataset: "merit"`,
//!   `data_types: {indices_0, indices_1, values}` — matches the engine and the
//!   real on-disk store's `zarr.json` exactly. The gauges store's root group
//!   carries no attrs (matches the real store); the per-STAID subgroups carry
//!   the full attr set plus `gage_catchment`/`gage_idx`.
//!
//! - **Codecs**: `bytes` (little-endian) + `zstd(level 0, checksum false)`.
//!   This is what zarr-python wrote into the real store's per-array `zarr.json`
//!   (level 0 == zstd's "use default" sentinel). The uint8 `values` array gets
//!   the same codec chain.
//!
//! - **Chunking**: each array is written as a **single chunk** covering the
//!   whole array. The engine/zarr-python split the big CONUS arrays into ~4
//!   chunks (e.g. `order` 346321 → chunk 86581); we use one chunk because the
//!   reader (`retrieve_array_subset` over `subset_all`) is chunk-count
//!   agnostic, and a single chunk keeps the writer trivial. This is the only
//!   structural divergence from the engine and it is invisible to the reader.
//!   The per-STAID gauge subgroups in the real store *already* use a single
//!   full-size chunk, so for those we match exactly.

use std::path::Path;
use std::sync::Arc;

use zarrs::array::codec::ZstdCodec;
use zarrs::array::{ArrayBuilder, BytesToBytesCodecTraits, data_type};
use zarrs::filesystem::FilesystemStore;
use zarrs::group::GroupBuilder;

use crate::adjacency::build::ConusAdjacency;
use crate::adjacency::gauges::GaugeSubgraph;
use crate::data::error::{DataError, Result};

/// Write the CONUS adjacency store at `dest` (a directory path).
///
/// Round-trips through [`crate::data::store::zarr::ConusAdjacencyStore::open`].
pub fn write_conus_store(adj: &ConusAdjacency, dest: &Path) -> Result<()> {
    let storage = Arc::new(FilesystemStore::new(dest).map_err(|e| zarr_err(dest, e))?);

    let n = adj.order.len();
    let nnz = adj.rows.len();

    // Root group attrs — mirrors coo_to_zarr (zarr_io.py:131-138) and the real
    // store's root zarr.json.
    let root_attrs = coo_root_attrs(n);
    let root = GroupBuilder::new()
        .attributes(root_attrs)
        .build(storage.clone(), "/")
        .map_err(|e| zarr_err(dest, e))?;
    root.store_metadata().map_err(|e| zarr_err(dest, e))?;

    write_i32(&storage, dest, "/indices_0", &adj.rows)?;
    write_i32(&storage, dest, "/indices_1", &adj.cols)?;
    write_u8_ones(&storage, dest, "/values", nnz)?;
    write_i32(&storage, dest, "/order", &adj.order)?;
    // Port addition: per-reach length_m / slope (float32), aligned to `order`.
    write_f32(&storage, dest, "/length_m", &adj.length_m)?;
    write_f32(&storage, dest, "/slope", &adj.slope)?;

    Ok(())
}

/// Write the per-gauge adjacency store at `dest` (a directory path): an empty
/// root group with one subgroup per STAID.
///
/// Round-trips through [`crate::data::store::zarr::GagesAdjacencyStore::open`].
pub fn write_gauges_store(subgraphs: &[GaugeSubgraph], dest: &Path) -> Result<()> {
    let storage = Arc::new(FilesystemStore::new(dest).map_err(|e| zarr_err(dest, e))?);

    // Empty root group (matches the real merit_gages store's root zarr.json).
    let root = GroupBuilder::new()
        .build(storage.clone(), "/")
        .map_err(|e| zarr_err(dest, e))?;
    root.store_metadata().map_err(|e| zarr_err(dest, e))?;

    for sg in subgraphs {
        let group_path = format!("/{}", sg.staid.as_str());
        // CONUS dimension is implied by the gauge's position space; the engine
        // stores the full CONUS [n, n] shape in `shape`. We don't know n here
        // (subgraphs carry only their own nodes), so derive the COO shape from
        // the engine's contract: shape is [n, n] but the reader never reads it.
        // We store the subset's own extent (max position + 1) which keeps the
        // attr self-consistent without the CONUS count; the reader ignores it.
        let dim = subgraph_dim(sg);

        let mut attrs = coo_root_attrs(dim);
        attrs.insert(
            "gage_catchment".into(),
            serde_json::Value::Number(sg.gage_catchment.into()),
        );
        attrs.insert(
            "gage_idx".into(),
            serde_json::Value::Number((sg.gage_idx as i64).into()),
        );

        let group = GroupBuilder::new()
            .attributes(attrs)
            .build(storage.clone(), &group_path)
            .map_err(|e| zarr_err(dest, e))?;
        group.store_metadata().map_err(|e| zarr_err(dest, e))?;

        write_i32(&storage, dest, &format!("{group_path}/indices_0"), &sg.rows)?;
        write_i32(&storage, dest, &format!("{group_path}/indices_1"), &sg.cols)?;
        write_u8_ones(&storage, dest, &format!("{group_path}/values"), sg.rows.len())?;
        write_i32(&storage, dest, &format!("{group_path}/order"), &sg.order)?;
    }

    Ok(())
}

// ---------- private helpers ----------

/// The COO `format`/`shape`/`geodataset`/`data_types` attr block shared by the
/// CONUS root and every gauge subgroup (zarr_io.py:131-138, 382-392).
fn coo_root_attrs(dim: usize) -> serde_json::Map<String, serde_json::Value> {
    let mut m = serde_json::Map::new();
    m.insert("format".into(), serde_json::Value::String("COO".into()));
    m.insert(
        "shape".into(),
        serde_json::Value::Array(vec![
            serde_json::Value::Number((dim as i64).into()),
            serde_json::Value::Number((dim as i64).into()),
        ]),
    );
    m.insert(
        "geodataset".into(),
        serde_json::Value::String("merit".into()),
    );
    let mut data_types = serde_json::Map::new();
    data_types.insert("indices_0".into(), serde_json::Value::String("int32".into()));
    data_types.insert("indices_1".into(), serde_json::Value::String("int32".into()));
    data_types.insert("values".into(), serde_json::Value::String("uint8".into()));
    m.insert("data_types".into(), serde_json::Value::Object(data_types));
    m
}

/// The CONUS extent for a gauge subgraph's `shape` attr. The reader ignores it;
/// we report `max position + 1` over the COO so the value is self-consistent.
fn subgraph_dim(sg: &GaugeSubgraph) -> usize {
    let max_pos = sg
        .rows
        .iter()
        .chain(sg.cols.iter())
        .chain(std::iter::once(&(sg.gage_idx as i32)))
        .copied()
        .max()
        .unwrap_or(0);
    (max_pos as usize) + 1
}

/// `bytes` (little-endian) + `zstd(0, false)` — the engine's codec chain.
fn zstd_codecs() -> Vec<Arc<dyn BytesToBytesCodecTraits>> {
    vec![Arc::new(ZstdCodec::new(0, false))]
}

fn write_i32(
    storage: &Arc<FilesystemStore>,
    store_path: &Path,
    array_path: &str,
    data: &[i32],
) -> Result<()> {
    let array = ArrayBuilder::new(
        vec![data.len() as u64],
        vec![data.len().max(1) as u64], // single chunk; min 1 for empty arrays
        data_type::int32(),
        0_i32,
    )
    .bytes_to_bytes_codecs(zstd_codecs())
    .build(storage.clone(), array_path)
    .map_err(|e| zarr_err(store_path, e))?;
    array.store_metadata().map_err(|e| zarr_err(store_path, e))?;
    if !data.is_empty() {
        array
            .store_chunk(&[0], data)
            .map_err(|e| zarr_err(store_path, e))?;
    }
    Ok(())
}

fn write_f32(
    storage: &Arc<FilesystemStore>,
    store_path: &Path,
    array_path: &str,
    data: &[f32],
) -> Result<()> {
    let array = ArrayBuilder::new(
        vec![data.len() as u64],
        vec![data.len().max(1) as u64],
        data_type::float32(),
        0.0_f32,
    )
    .bytes_to_bytes_codecs(zstd_codecs())
    .build(storage.clone(), array_path)
    .map_err(|e| zarr_err(store_path, e))?;
    array.store_metadata().map_err(|e| zarr_err(store_path, e))?;
    if !data.is_empty() {
        array
            .store_chunk(&[0], data)
            .map_err(|e| zarr_err(store_path, e))?;
    }
    Ok(())
}

/// `values` array: `nnz` ones (uint8). Mirrors `coo.data` for an adjacency COO.
fn write_u8_ones(
    storage: &Arc<FilesystemStore>,
    store_path: &Path,
    array_path: &str,
    nnz: usize,
) -> Result<()> {
    let data = vec![1_u8; nnz];
    let array = ArrayBuilder::new(
        vec![nnz as u64],
        vec![nnz.max(1) as u64],
        data_type::uint8(),
        0_u8,
    )
    .bytes_to_bytes_codecs(zstd_codecs())
    .build(storage.clone(), array_path)
    .map_err(|e| zarr_err(store_path, e))?;
    array.store_metadata().map_err(|e| zarr_err(store_path, e))?;
    if nnz > 0 {
        array
            .store_chunk(&[0], data.as_slice())
            .map_err(|e| zarr_err(store_path, e))?;
    }
    Ok(())
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
    use crate::data::ids::{Comid, Staid};
    use crate::data::store::zarr::{ConusAdjacencyStore, GagesAdjacencyStore};

    /// Synthetic 6-reach dendritic CONUS (same shape as gauges.rs tests):
    /// edges (downstream pos, upstream pos): (2,0) (2,1) (4,2) (4,3).
    fn synthetic_conus() -> ConusAdjacency {
        ConusAdjacency {
            order: vec![10, 20, 30, 40, 50, 60],
            rows: vec![2, 2, 4, 4],
            cols: vec![0, 1, 2, 3],
            length_m: vec![100.0, 200.0, 300.0, 400.0, 500.0, 600.0],
            slope: vec![0.001, 0.002, 0.003, 0.004, 0.005, 0.006],
            dropped_comids: vec![],
        }
    }

    fn tmp_dir(tag: &str) -> std::path::PathBuf {
        let mut p = std::env::temp_dir();
        p.push(format!("ddrs_zarr_write_{tag}_{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&p);
        std::fs::create_dir_all(&p).expect("mkdir");
        p
    }

    #[test]
    fn conus_round_trips_through_reader() {
        let adj = synthetic_conus();
        let dir = tmp_dir("conus");
        write_conus_store(&adj, &dir).expect("write conus");

        let store = ConusAdjacencyStore::open(&dir).expect("open conus");
        assert_eq!(store.n, 6);
        assert_eq!(store.nnz, 4);
        assert_eq!(
            store.order,
            vec![
                Comid(10),
                Comid(20),
                Comid(30),
                Comid(40),
                Comid(50),
                Comid(60)
            ]
        );
        assert_eq!(store.indices_0, adj.rows);
        assert_eq!(store.indices_1, adj.cols);
        assert_eq!(store.length_m.to_vec(), adj.length_m);
        assert_eq!(store.slope.to_vec(), adj.slope);

        // Cheap structural check on the root group attrs.
        let root_json =
            std::fs::read_to_string(dir.join("zarr.json")).expect("read root zarr.json");
        let v: serde_json::Value = serde_json::from_str(&root_json).expect("parse");
        assert_eq!(v["attributes"]["format"], "COO");
        assert_eq!(v["attributes"]["geodataset"], "merit");
        assert_eq!(v["attributes"]["shape"][0], 6);
        assert_eq!(v["attributes"]["data_types"]["indices_0"], "int32");
        assert_eq!(v["node_type"], "group");
        assert_eq!(v["zarr_format"], 3);

        // Codec chain on an array matches the engine (bytes + zstd).
        let arr_json = std::fs::read_to_string(dir.join("order").join("zarr.json"))
            .expect("read order zarr.json");
        let av: serde_json::Value = serde_json::from_str(&arr_json).expect("parse");
        let codec_names: Vec<&str> = av["codecs"]
            .as_array()
            .unwrap()
            .iter()
            .map(|c| c["name"].as_str().unwrap())
            .collect();
        assert_eq!(codec_names, vec!["bytes", "zstd"]);
        assert_eq!(av["data_type"], "int32");
    }

    #[test]
    fn gauges_round_trip_through_reader() {
        // Two gauges in CONUS position space.
        let g50 = GaugeSubgraph {
            staid: Staid::from("00000050"),
            gage_catchment: 50,
            gage_idx: 4,
            order: vec![10, 20, 30, 40, 50],
            rows: vec![2, 2, 4, 4],
            cols: vec![0, 1, 2, 3],
        };
        let g30 = GaugeSubgraph {
            staid: Staid::from("00000030"),
            gage_catchment: 30,
            gage_idx: 2,
            order: vec![10, 20, 30],
            rows: vec![2, 2],
            cols: vec![0, 1],
        };
        let subs = vec![g50.clone(), g30.clone()];

        let dir = tmp_dir("gauges");
        write_gauges_store(&subs, &dir).expect("write gauges");

        let staids = vec![Staid::from("00000050"), Staid::from("00000030")];
        let store = GagesAdjacencyStore::open(&dir, &staids).expect("open gauges");
        assert_eq!(store.len(), 2);

        let r50 = store.get(&Staid::from("00000050")).expect("g50");
        assert_eq!(r50.gage_idx, 4);
        // gage_catchment is written as a JSON int; the reader stringifies it
        // (matches the real store, where gage_catchment is a bare int).
        assert_eq!(r50.gage_catchment, "50");
        assert_eq!(r50.indices_0, g50.rows);
        assert_eq!(r50.indices_1, g50.cols);

        let r30 = store.get(&Staid::from("00000030")).expect("g30");
        assert_eq!(r30.gage_idx, 2);
        assert_eq!(r30.gage_catchment, "30");
        assert_eq!(r30.indices_0, g30.rows);
        assert_eq!(r30.indices_1, g30.cols);

        // Empty root group, per-STAID subgroup attrs present.
        let sub_json = std::fs::read_to_string(dir.join("00000050").join("zarr.json"))
            .expect("read subgroup zarr.json");
        let sv: serde_json::Value = serde_json::from_str(&sub_json).expect("parse");
        assert_eq!(sv["attributes"]["format"], "COO");
        assert_eq!(sv["attributes"]["gage_idx"], 4);
        assert_eq!(sv["attributes"]["gage_catchment"], 50);
        assert_eq!(sv["node_type"], "group");
    }

    #[test]
    fn headwater_gauge_with_no_edges_round_trips() {
        // A gauge whose subgraph is a single headwater: empty COO arrays.
        let hw = GaugeSubgraph {
            staid: Staid::from("00000010"),
            gage_catchment: 10,
            gage_idx: 0,
            order: vec![10],
            rows: vec![],
            cols: vec![],
        };
        let dir = tmp_dir("headwater");
        write_gauges_store(&[hw], &dir).expect("write");

        let store =
            GagesAdjacencyStore::open(&dir, &[Staid::from("00000010")]).expect("open");
        let r = store.get(&Staid::from("00000010")).expect("hw");
        assert_eq!(r.gage_idx, 0);
        assert!(r.indices_0.is_empty());
        assert!(r.indices_1.is_empty());
    }
}

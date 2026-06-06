//! Up-front existence checks for explicitly-configured adjacency zarr stores.
//!
//! The zarrs readers (`ConusAdjacencyStore::open` / `GagesAdjacencyStore::open`)
//! surface a missing array as the bare zarrs error "array metadata is missing",
//! which doesn't name which array or which store. These checks run BEFORE the
//! readers open the stores so a missing array produces an actionable message
//! that (a) names the array and the store path and (b) suggests the two
//! remedies: remove the adjacency keys (managed build) or repair the store.
//!
//! These are cheap filesystem existence checks on `<array>/zarr.json` — they do
//! NOT read or open the 346K-element arrays. Deeper corruption (truncated
//! chunks, dtype mismatch) still surfaces at open time as it does today.

use std::path::Path;

/// Required root-level arrays in a CONUS adjacency store, mirroring the reads in
/// `ConusAdjacencyStore::open` (`src/data/store/zarr.rs:56-75`).
const CONUS_REQUIRED_ARRAYS: &[&str] = &["order", "length_m", "slope", "indices_0", "indices_1"];

/// Error naming the missing array, the store path, and the two remedies.
#[derive(Debug, thiserror::Error)]
#[error(
    "adjacency store {store} is missing required array `{array}` \
     (expected {store}/{array}/zarr.json). Either remove the \
     `conus_adjacency`/`gages_adjacency` keys from data_sources so ddrs builds \
     adjacency from `geospatial_fabric` (managed build), or repair/regenerate \
     the store."
)]
pub struct StoreLayoutError {
    pub store: String,
    pub array: String,
}

/// Validate that a CONUS adjacency store has every required root-level array.
///
/// Checks `<store>/<array>/zarr.json` exists for each of [`CONUS_REQUIRED_ARRAYS`].
pub fn validate_conus_store_layout(store: &Path) -> Result<(), StoreLayoutError> {
    for array in CONUS_REQUIRED_ARRAYS {
        let zarr_json = store.join(array).join("zarr.json");
        if !zarr_json.is_file() {
            return Err(StoreLayoutError {
                store: store.display().to_string(),
                array: (*array).to_string(),
            });
        }
    }
    Ok(())
}

/// Validate that a gages adjacency store has a root group (`<store>/zarr.json`).
///
/// Per-gauge subgroups are keyed by STAID and are opened lazily by
/// `GagesAdjacencyStore::open` (missing subgroups are silently skipped, mirroring
/// DDR's `valid_gauges_mask`), so the only cheap up-front check is the root
/// group's presence. Subgroup-array/attr problems surface at open time.
pub fn validate_gages_store_layout(store: &Path) -> Result<(), StoreLayoutError> {
    let zarr_json = store.join("zarr.json");
    if !zarr_json.is_file() {
        return Err(StoreLayoutError {
            store: store.display().to_string(),
            array: "<root group>".to_string(),
        });
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::fs;

    fn tmp_dir(tag: &str) -> std::path::PathBuf {
        let p = std::env::temp_dir().join(format!(
            "ddrs_adj_validate_{}_{}",
            tag,
            std::process::id()
        ));
        let _ = fs::remove_dir_all(&p);
        fs::create_dir_all(&p).unwrap();
        p
    }

    /// Write a minimal store skeleton: a `zarr.json` under each named array dir.
    fn write_conus_skeleton(store: &Path) {
        fs::create_dir_all(store).unwrap();
        fs::write(store.join("zarr.json"), "{}").unwrap();
        for array in CONUS_REQUIRED_ARRAYS {
            let dir = store.join(array);
            fs::create_dir_all(&dir).unwrap();
            fs::write(dir.join("zarr.json"), "{}").unwrap();
        }
    }

    #[test]
    fn complete_conus_store_is_ok() {
        let store = tmp_dir("conus_ok").join("merit_conus_adjacency.zarr");
        write_conus_skeleton(&store);
        assert!(validate_conus_store_layout(&store).is_ok());
        let _ = fs::remove_dir_all(store.parent().unwrap());
    }

    #[test]
    fn missing_length_m_names_array_and_path() {
        let store = tmp_dir("conus_missing").join("merit_conus_adjacency.zarr");
        write_conus_skeleton(&store);
        // Delete length_m/zarr.json.
        fs::remove_file(store.join("length_m").join("zarr.json")).unwrap();

        let err = validate_conus_store_layout(&store).unwrap_err();
        assert_eq!(err.array, "length_m");
        let msg = err.to_string();
        assert!(msg.contains("length_m"), "must name the array: {msg}");
        assert!(
            msg.contains(&store.display().to_string()),
            "must name the store path: {msg}"
        );
        assert!(
            msg.contains("managed build"),
            "must suggest the managed-build remedy: {msg}"
        );
        assert!(
            msg.contains("repair") || msg.contains("regenerate"),
            "must suggest the repair remedy: {msg}"
        );
        let _ = fs::remove_dir_all(store.parent().unwrap());
    }

    #[test]
    fn missing_gages_root_group_errors() {
        let store = tmp_dir("gages_missing").join("merit_gages_conus_adjacency.zarr");
        fs::create_dir_all(&store).unwrap();
        // No zarr.json at the root.
        let err = validate_gages_store_layout(&store).unwrap_err();
        assert_eq!(err.array, "<root group>");
        let _ = fs::remove_dir_all(store.parent().unwrap());
    }

    #[test]
    fn present_gages_root_group_is_ok() {
        let store = tmp_dir("gages_ok").join("merit_gages_conus_adjacency.zarr");
        fs::create_dir_all(&store).unwrap();
        fs::write(store.join("zarr.json"), "{}").unwrap();
        assert!(validate_gages_store_layout(&store).is_ok());
        let _ = fs::remove_dir_all(store.parent().unwrap());
    }
}

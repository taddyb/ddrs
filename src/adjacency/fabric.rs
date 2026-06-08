//! Fabric-format dispatch: route a `geospatial_fabric` path to the right
//! attribute reader by extension.
//!
//! Accepted forms:
//!   - `.shp`  → sibling `.dbf` is read (geometry never opened) — `dbf.rs`
//!   - `.dbf`  → read directly — `dbf.rs`
//!   - `.gpkg` → attribute columns read via SQL (geometry blobs never
//!               deserialized) — `gpkg.rs`
//!
//! Both readers produce the same `Vec<FlowpathRecord>`, so everything
//! downstream (`build`, `gauges`, `zarr_write`, `cache`) is format-agnostic.

use std::path::{Path, PathBuf};

use crate::adjacency::dbf::{self, FlowpathRecord};
use crate::adjacency::gpkg;
use crate::data::error::DataError;

/// A fabric input resolved to the concrete file that is read and hashed.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum FabricKind {
    /// dBASE attribute table (resolved from `.shp` sibling or given directly).
    Dbf(PathBuf),
    /// GeoPackage (SQLite) — the whole file is the hashable artifact.
    Gpkg(PathBuf),
}

impl FabricKind {
    /// The file whose bytes are content-fingerprinted by the adjacency cache.
    pub fn resolved_path(&self) -> &Path {
        match self {
            FabricKind::Dbf(p) | FabricKind::Gpkg(p) => p,
        }
    }
}

/// Classify a fabric path by extension, resolving `.shp` → sibling `.dbf`.
///
/// Unknown extensions error, naming the three accepted forms.
pub fn resolve_fabric(path: &Path) -> crate::data::error::Result<FabricKind> {
    match path.extension().and_then(|e| e.to_str()) {
        Some(ext) if ext.eq_ignore_ascii_case("gpkg") => Ok(FabricKind::Gpkg(path.to_path_buf())),
        Some(ext) if ext.eq_ignore_ascii_case("shp") || ext.eq_ignore_ascii_case("dbf") => {
            Ok(FabricKind::Dbf(dbf::resolve_dbf_path(path)))
        }
        _ => Err(DataError::Malformed {
            path: path.to_path_buf(),
            message: "unsupported geospatial_fabric extension — expected .shp \
                      (sibling .dbf read), .dbf, or .gpkg"
                .to_string(),
        }),
    }
}

/// Read flowpath records from any supported fabric format.
///
/// `layer` is only meaningful for `.gpkg` inputs (selects the feature table);
/// it is ignored for dBASE inputs (config validation rejects setting it
/// alongside a non-gpkg fabric before we get here).
pub fn read_fabric_records(
    path: &Path,
    layer: Option<&str>,
) -> crate::data::error::Result<Vec<FlowpathRecord>> {
    match resolve_fabric(path)? {
        FabricKind::Dbf(dbf_path) => dbf::read_flowpath_records(&dbf_path),
        FabricKind::Gpkg(gpkg_path) => gpkg::read_flowpath_records_gpkg(&gpkg_path, layer),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn shp_resolves_to_sibling_dbf() {
        let kind = resolve_fabric(Path::new("/d/rivers.shp")).unwrap();
        assert_eq!(kind, FabricKind::Dbf(PathBuf::from("/d/rivers.dbf")));
    }

    #[test]
    fn dbf_passes_through() {
        let kind = resolve_fabric(Path::new("/d/rivers.dbf")).unwrap();
        assert_eq!(kind, FabricKind::Dbf(PathBuf::from("/d/rivers.dbf")));
    }

    #[test]
    fn gpkg_recognized() {
        let kind = resolve_fabric(Path::new("/d/global_merit_riv.gpkg")).unwrap();
        assert_eq!(
            kind,
            FabricKind::Gpkg(PathBuf::from("/d/global_merit_riv.gpkg"))
        );
    }

    #[test]
    fn unknown_extension_errors_naming_accepted_forms() {
        let err = resolve_fabric(Path::new("/d/rivers.geojson")).unwrap_err();
        let msg = err.to_string();
        assert!(msg.contains(".shp") && msg.contains(".dbf") && msg.contains(".gpkg"), "{msg}");
    }
}

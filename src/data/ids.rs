//! Domain identifier types and the shared "domain ID → array position" map.
//!
//! DDR's Python uses raw `int` for COMIDs and raw `str` for STAIDs, which has
//! been a recurring bug surface (forgot-to-zfill mistakes, COMID-vs-divide_id
//! mixups). Newtypes here let the compiler catch those mismatches.
//!
//! `IdIndex<T>` is the one piece of cross-store boilerplate worth pulling out:
//! every store builds one at open time, every read consumes one to map domain
//! IDs (`Comid`, `Staid`) to integer array positions inside the zarr/netcdf/
//! icechunk arrays.
//!
//! See `src/data/mod.rs` for the design rationale (anti-Box<dyn>, per-store
//! types, single sync facade owning a tokio runtime).
use std::collections::HashMap;
use std::hash::Hash;

/// MERIT catchment identifier — used by `merit_conus_adjacency.zarr`'s `order`
/// array and as the spatial dimension in attributes/streamflow stores.
#[derive(Copy, Clone, Eq, PartialEq, Hash, Ord, PartialOrd, Debug)]
pub struct Comid(pub i64);

/// USGS gauge identifier — zero-padded to 8 characters at construction.
#[derive(Clone, Eq, PartialEq, Hash, Ord, PartialOrd, Debug)]
pub struct Staid(String);

impl Staid {
    /// Zero-pad to 8 characters to match DDR's canonical form
    /// (`base_geodataset.py:35`, `readers.py:131`).
    pub fn new(s: &str) -> Self {
        let mut padded = s.to_string();
        while padded.len() < 8 {
            padded.insert(0, '0');
        }
        Self(padded)
    }

    pub fn as_str(&self) -> &str {
        &self.0
    }
}

impl From<&str> for Staid {
    fn from(s: &str) -> Self {
        Self::new(s)
    }
}

impl std::fmt::Display for Staid {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(&self.0)
    }
}

/// Maps domain identifiers to their integer position inside an array.
///
/// Every store builds one of these at open time. `positions_of` returns both
/// the resolved positions *and* the indices of the requested IDs that were
/// missing — callers decide whether to warn, error, or fill with sentinels.
pub struct IdIndex<Id: Eq + Hash + Clone> {
    lookup: HashMap<Id, usize>,
    ids_in_order: Vec<Id>,
}

impl<Id: Eq + Hash + Clone> IdIndex<Id> {
    pub fn new(ids: Vec<Id>) -> Self {
        let lookup = ids
            .iter()
            .enumerate()
            .map(|(i, id)| (id.clone(), i))
            .collect();
        Self {
            lookup,
            ids_in_order: ids,
        }
    }

    pub fn position(&self, id: &Id) -> Option<usize> {
        self.lookup.get(id).copied()
    }

    pub fn contains(&self, id: &Id) -> bool {
        self.lookup.contains_key(id)
    }

    /// Resolve a slice of IDs to their array positions.
    ///
    /// Returns `(positions, missing_indices_into_input)` — `positions[i]` is
    /// the array position of `ids[i]`, **only for** `i ∉ missing_indices`.
    /// `positions.len() + missing_indices.len() == ids.len()`.
    pub fn positions_of(&self, ids: &[Id]) -> (Vec<usize>, Vec<usize>) {
        let mut positions = Vec::with_capacity(ids.len());
        let mut missing = Vec::new();
        for (i, id) in ids.iter().enumerate() {
            match self.lookup.get(id) {
                Some(&pos) => positions.push(pos),
                None => missing.push(i),
            }
        }
        (positions, missing)
    }

    pub fn len(&self) -> usize {
        self.ids_in_order.len()
    }

    pub fn is_empty(&self) -> bool {
        self.ids_in_order.is_empty()
    }

    pub fn id_at(&self, pos: usize) -> Option<&Id> {
        self.ids_in_order.get(pos)
    }

    pub fn ids(&self) -> &[Id] {
        &self.ids_in_order
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn staid_zfill_8() {
        assert_eq!(Staid::new("1563500").as_str(), "01563500");
        assert_eq!(Staid::new("01563500").as_str(), "01563500");
        assert_eq!(Staid::new("123456789").as_str(), "123456789"); // longer untouched
    }

    #[test]
    fn id_index_roundtrip() {
        let idx = IdIndex::new(vec![Comid(10), Comid(20), Comid(30)]);
        assert_eq!(idx.position(&Comid(20)), Some(1));
        assert_eq!(idx.position(&Comid(99)), None);
        let (positions, missing) =
            idx.positions_of(&[Comid(30), Comid(99), Comid(10), Comid(42)]);
        assert_eq!(positions, vec![2, 0]);
        assert_eq!(missing, vec![1, 3]);
    }
}

//! NetCDF attribute reader.
//!
//! Mirrors `~/projects/ddr/src/ddr/io/readers.py::AttributesReader` and
//! `~/projects/ddr/src/ddr/geodatazoo/merit.py::_get_attributes` for the
//! MERIT branch (single `merit_global_attributes_v2.nc` file, 1D vars on a
//! `COMID` dim).
//!
//! Strategy: at `open` we materialize a `(F, N)` f32 matrix where `N` is
//! the number of requested COMIDs that were present in the file. The full
//! NetCDF column is read once per attribute (`~24 MB` at 2.94M f64),
//! cast to f32, then sliced — fancy indexing is unnecessary and the
//! peak transient is bounded by `F * 24 MB`.

use std::collections::HashMap;
use std::path::PathBuf;

use ndarray::{Array1, Array2};

use crate::data::error::{DataError, Result};
use crate::data::ids::{Comid, IdIndex};
use crate::data::statistics::naninfmean;

#[derive(Debug)]
pub struct AttributesStore {
    pub path: PathBuf,
    pub attr_names: Vec<String>,
    pub attrs: Array2<f32>,
    pub index: IdIndex<Comid>,
    pub row_means: Array1<f32>,
}

impl AttributesStore {
    pub fn open(
        path: impl Into<PathBuf>,
        attr_names: &[String],
        comids: &[Comid],
    ) -> Result<Self> {
        let path = path.into();
        let file = netcdf::open(&path).map_err(|e| DataError::NetCdf {
            path: path.clone(),
            source: e,
        })?;

        // ----- COMID coord → HashMap<i64, file_pos> -----
        let comid_var = file
            .variable("COMID")
            .ok_or_else(|| DataError::Malformed {
                path: path.clone(),
                message: "missing 'COMID' coord variable".to_string(),
            })?;
        // FULL READ: read the entire COMID coord (length ~2.94M) into a Vec<i64>.
        let comid_flat: Vec<i64> = comid_var
            .get_values::<i64, _>(..)
            .map_err(|e| DataError::NetCdf {
                path: path.clone(),
                source: e,
            })?;
        let comid_to_pos: HashMap<i64, usize> = comid_flat
            .iter()
            .enumerate()
            .map(|(i, &c)| (c, i))
            .collect();

        // Resolve requested COMIDs → file positions; track present subset.
        let mut requested_positions: Vec<usize> = Vec::with_capacity(comids.len());
        let mut present_comids: Vec<Comid> = Vec::with_capacity(comids.len());
        for c in comids {
            if let Some(&p) = comid_to_pos.get(&c.0) {
                requested_positions.push(p);
                present_comids.push(*c);
            }
        }
        let n_present = present_comids.len();

        let f = attr_names.len();
        let mut attrs = Array2::<f32>::zeros((f, n_present));
        let mut row_means = Array1::<f32>::zeros(f);

        for (fi, name) in attr_names.iter().enumerate() {
            let var = file.variable(name).ok_or_else(|| DataError::Malformed {
                path: path.clone(),
                message: format!("missing attribute variable '{name}'"),
            })?;
            let col_f64: Vec<f64> = var
                .get_values::<f64, _>(..)
                .map_err(|e| DataError::NetCdf {
                    path: path.clone(),
                    source: e,
                })?;
            let col_f32: Vec<f32> = col_f64.iter().map(|&x| x as f32).collect();
            row_means[fi] = naninfmean(&col_f32);
            for (out_col, &src_pos) in requested_positions.iter().enumerate() {
                attrs[(fi, out_col)] = col_f32[src_pos];
            }
        }

        let index = IdIndex::new(present_comids);
        Ok(Self {
            path,
            attr_names: attr_names.to_vec(),
            attrs,
            index,
            row_means,
        })
    }

    /// Like [`open`] but keeps **all** requested `comids` in the index, in
    /// order. COMIDs absent from the file are NaN-filled rather than dropped.
    ///
    /// Used by [`open_multi`] so every store shares the same COMID basis and
    /// the merged `(F, N)` matrix columns are aligned.
    pub fn open_aligned(
        path: impl Into<PathBuf>,
        attr_names: &[String],
        comids: &[Comid],
    ) -> Result<Self> {
        let path = path.into();
        let file = netcdf::open(&path).map_err(|e| DataError::NetCdf {
            path: path.clone(),
            source: e,
        })?;

        // ----- COMID coord → HashMap<i64, file_pos> -----
        let comid_var = file
            .variable("COMID")
            .ok_or_else(|| DataError::Malformed {
                path: path.clone(),
                message: "missing 'COMID' coord variable".to_string(),
            })?;
        let comid_flat: Vec<i64> = comid_var
            .get_values::<i64, _>(..)
            .map_err(|e| DataError::NetCdf {
                path: path.clone(),
                source: e,
            })?;
        let comid_to_pos: HashMap<i64, usize> = comid_flat
            .iter()
            .enumerate()
            .map(|(i, &c)| (c, i))
            .collect();

        let n = comids.len();
        let f = attr_names.len();
        // NaN-init: absent COMIDs stay NaN (fill_nans later replaces via row_means).
        let mut attrs = Array2::<f32>::from_elem((f, n), f32::NAN);
        let mut row_means = Array1::<f32>::zeros(f);

        for (fi, name) in attr_names.iter().enumerate() {
            let var = file.variable(name).ok_or_else(|| DataError::Malformed {
                path: path.clone(),
                message: format!("missing attribute variable '{name}'"),
            })?;
            let col_f64: Vec<f64> = var
                .get_values::<f64, _>(..)
                .map_err(|e| DataError::NetCdf {
                    path: path.clone(),
                    source: e,
                })?;
            let col_f32: Vec<f32> = col_f64.iter().map(|&x| x as f32).collect();
            row_means[fi] = naninfmean(&col_f32);
            for (out_col, c) in comids.iter().enumerate() {
                if let Some(&src_pos) = comid_to_pos.get(&c.0) {
                    attrs[(fi, out_col)] = col_f32[src_pos];
                }
                // else: NaN already from init
            }
        }

        let index = IdIndex::new(comids.to_vec());
        Ok(Self {
            path,
            attr_names: attr_names.to_vec(),
            attrs,
            index,
            row_means,
        })
    }

    /// Open multiple attribute files and assemble a merged `AttributesStore`.
    ///
    /// Each name in `attr_names` must be present in **exactly one** of the
    /// given files:
    /// - Ambiguous (present in > 1 file) → hard error naming the variable and
    ///   both file paths.
    /// - Missing (in zero files) → hard error naming the variable.
    ///
    /// All stores are aligned to `comids` (via [`open_aligned`]), so COMIDs
    /// absent from a particular store get NaN values for that store's variables.
    /// The merged `attrs` is `(F, N)` with variables in `attr_names` order and
    /// COMIDs in `comids` order.
    ///
    /// Use this for `data_sources.attributes` lists with length > 1. The
    /// single-path case routes through the existing [`open`] (byte-identical).
    pub fn open_multi(
        paths: &[std::path::PathBuf],
        attr_names: &[String],
        comids: &[Comid],
    ) -> Result<Self> {
        debug_assert!(paths.len() >= 2, "open_multi requires at least two paths");

        // --- Pass 1: probe each file to build ownership (var_name → store idx) ---
        // var_owner[fi] = Some(store_idx) once the owning file is found.
        let mut var_owner: Vec<Option<usize>> = vec![None; attr_names.len()];

        for (store_idx, path) in paths.iter().enumerate() {
            let file = netcdf::open(path).map_err(|e| DataError::NetCdf {
                path: path.clone(),
                source: e,
            })?;
            for (fi, name) in attr_names.iter().enumerate() {
                if file.variable(name.as_str()).is_some() {
                    match var_owner[fi] {
                        Some(prev_idx) => {
                            return Err(DataError::Malformed {
                                path: path.clone(),
                                message: format!(
                                    "attribute '{name}' found in both '{}' and '{}' — \
                                     each variable must belong to exactly one store",
                                    paths[prev_idx].display(),
                                    path.display()
                                ),
                            });
                        }
                        None => var_owner[fi] = Some(store_idx),
                    }
                }
            }
        }

        // Hard-error on any variable present in no store.
        for (fi, name) in attr_names.iter().enumerate() {
            if var_owner[fi].is_none() {
                return Err(DataError::Malformed {
                    path: paths[0].clone(),
                    message: format!(
                        "attribute '{name}' not found in any of the {} attribute stores",
                        paths.len()
                    ),
                });
            }
        }

        // --- Pass 2: group variable indices by their owning store ---
        let mut store_fi_lists: Vec<Vec<usize>> = vec![vec![]; paths.len()];
        for (fi, &owner) in var_owner.iter().enumerate() {
            store_fi_lists[owner.unwrap()].push(fi);
        }

        // --- Pass 3: open_aligned per store; copy into merged matrix ---
        let n = comids.len();
        let f = attr_names.len();
        let mut merged_attrs = Array2::<f32>::from_elem((f, n), f32::NAN);
        let mut merged_row_means = Array1::<f32>::zeros(f);

        for (store_idx, fi_list) in store_fi_lists.iter().enumerate() {
            if fi_list.is_empty() {
                continue;
            }
            let store_names: Vec<String> =
                fi_list.iter().map(|&fi| attr_names[fi].clone()).collect();
            let store = Self::open_aligned(&paths[store_idx], &store_names, comids)?;
            for (local_fi, &global_fi) in fi_list.iter().enumerate() {
                for col in 0..n {
                    merged_attrs[(global_fi, col)] = store.attrs[(local_fi, col)];
                }
                merged_row_means[global_fi] = store.row_means[local_fi];
            }
        }

        let index = IdIndex::new(comids.to_vec());
        Ok(Self {
            path: paths[0].clone(), // representative path for diagnostics
            attr_names: attr_names.to_vec(),
            attrs: merged_attrs,
            index,
            row_means: merged_row_means,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Write a minimal two-variable netCDF4 file for use in unit tests.
    /// `comids`: i64 COMID array; `var_name`: variable name; `values`: f64 data.
    fn write_test_nc(path: &std::path::Path, comids: &[i64], var_name: &str, values: &[f64]) {
        let mut f = netcdf::create(path).expect("create netcdf");
        f.add_dimension("COMID", comids.len()).unwrap();
        let mut cv = f.add_variable::<i64>("COMID", &["COMID"]).unwrap();
        cv.put_values(comids, ..).unwrap();
        let mut v = f.add_variable::<f64>(var_name, &["COMID"]).unwrap();
        v.put_values(values, ..).unwrap();
    }

    #[test]
    fn open_multi_two_store_alignment() {
        // Store A: COMIDs {10,20,30}, var 'a' = [1.0, 2.0, 3.0]
        // Store B: COMIDs {20,30,40}, var 'b' = [4.0, 5.0, 6.0]
        // Request comids=[10,20,30,40], attr_names=[a,b]
        // Expected: shape (2,4); a[10]=1.0, a[40]=NaN, b[10]=NaN, b[40]=6.0
        let dir = tempfile::tempdir().unwrap();
        let path_a = dir.path().join("a.nc");
        let path_b = dir.path().join("b.nc");
        write_test_nc(&path_a, &[10, 20, 30], "a", &[1.0, 2.0, 3.0]);
        write_test_nc(&path_b, &[20, 30, 40], "b", &[4.0, 5.0, 6.0]);

        let comids = vec![Comid(10), Comid(20), Comid(30), Comid(40)];
        let attr_names = vec!["a".to_string(), "b".to_string()];
        let paths = vec![path_a, path_b];

        let store = AttributesStore::open_multi(&paths, &attr_names, &comids).unwrap();

        // Shape (F=2, N=4)
        assert_eq!(store.attrs.shape(), &[2, 4]);
        // All 4 comids in index
        assert_eq!(store.index.len(), 4);
        // a[comid=10] = 1.0, a[comid=40] = NaN (absent from store A)
        assert!((store.attrs[(0, 0)] - 1.0).abs() < 1e-6, "a[10] = 1.0");
        assert!(store.attrs[(0, 3)].is_nan(), "a[40] must be NaN");
        // b[comid=10] = NaN (absent from store B), b[comid=40] = 6.0
        assert!(store.attrs[(1, 0)].is_nan(), "b[10] must be NaN");
        assert!((store.attrs[(1, 3)] - 6.0).abs() < 1e-6, "b[40] = 6.0");
        // column order matches requested comids
        assert_eq!(store.index.position(&Comid(10)), Some(0));
        assert_eq!(store.index.position(&Comid(40)), Some(3));
        // attr row order matches [a, b]
        assert_eq!(store.attr_names, &["a".to_string(), "b".to_string()]);
    }

    #[test]
    fn open_multi_ambiguity_error() {
        let dir = tempfile::tempdir().unwrap();
        let path_a = dir.path().join("a.nc");
        let path_b = dir.path().join("b.nc");
        // Both stores have var 'a' → ambiguity error
        write_test_nc(&path_a, &[10, 20], "a", &[1.0, 2.0]);
        write_test_nc(&path_b, &[20, 30], "a", &[3.0, 4.0]);

        let comids = vec![Comid(10), Comid(20), Comid(30)];
        let attr_names = vec!["a".to_string()];
        let paths = vec![path_a, path_b];

        let err = AttributesStore::open_multi(&paths, &attr_names, &comids).unwrap_err();
        let msg = format!("{err}");
        assert!(
            msg.contains("'a'") && msg.contains("found in both"),
            "expected ambiguity error, got: {msg}"
        );
    }

    #[test]
    fn open_multi_missing_var_error() {
        let dir = tempfile::tempdir().unwrap();
        let path_a = dir.path().join("a.nc");
        let path_b = dir.path().join("b.nc");
        write_test_nc(&path_a, &[10, 20], "a", &[1.0, 2.0]);
        write_test_nc(&path_b, &[20, 30], "b", &[3.0, 4.0]);

        let comids = vec![Comid(10), Comid(20), Comid(30)];
        // 'missing' is not in either store
        let attr_names = vec!["a".to_string(), "missing".to_string()];
        let paths = vec![path_a, path_b];

        let err = AttributesStore::open_multi(&paths, &attr_names, &comids).unwrap_err();
        let msg = format!("{err}");
        assert!(
            msg.contains("'missing'") && msg.contains("not found"),
            "expected missing-var error, got: {msg}"
        );
    }

    #[test]
    fn open_aligned_full_coverage_matches_open() {
        // When all requested comids are present, open_aligned gives the same
        // attrs/row_means as open (index len matches since no drops).
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("attrs.nc");
        write_test_nc(&path, &[10, 20, 30], "a", &[1.0, 2.0, 3.0]);

        let comids = vec![Comid(10), Comid(20), Comid(30)];
        let attr_names = vec!["a".to_string()];

        let s1 = AttributesStore::open(&path, &attr_names, &comids).unwrap();
        let s2 = AttributesStore::open_aligned(&path, &attr_names, &comids).unwrap();

        assert_eq!(s1.attrs, s2.attrs, "attrs must be identical with full coverage");
        assert_eq!(s1.row_means, s2.row_means, "row_means must match");
        assert_eq!(s1.index.len(), s2.index.len(), "index length must match");
    }
}

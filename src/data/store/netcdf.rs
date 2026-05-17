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
}

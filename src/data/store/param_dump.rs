//! Reader for per-reach routing-parameter NetCDF dumps, as written by
//! `dump_parameters::write_netcdf` (`COMID`-keyed, full-CONUS, physical
//! units — e.g. `n`, `q_spatial`, `p_spatial`, `x_storage`, `slope`). Used by
//! the H5/H6 selective-equifinality parameter-swap tests (`--donor-params-nc`
//! in `probe_zeta_gradient`'s eval-loss mode) to load a donor arm's fields
//! for injection into another arm's checkpoint.

use std::collections::HashMap;
use std::path::PathBuf;

use crate::data::error::{DataError, Result};

/// Read one f32 variable from a COMID-keyed NetCDF dump into a `HashMap`
/// keyed by COMID — NOT positional order. Donor and target checkpoints both
/// cover full CONUS but are not guaranteed to share a row ordering, so any
/// caller gathering these values into a batch's reach order must look them
/// up by COMID, never by position.
pub fn load_comid_field(path: impl Into<PathBuf>, var: &str) -> Result<HashMap<i64, f32>> {
    let path = path.into();
    let file = netcdf::open(&path).map_err(|e| DataError::NetCdf {
        path: path.clone(),
        source: e,
    })?;

    let comid_var = file.variable("COMID").ok_or_else(|| DataError::Malformed {
        path: path.clone(),
        message: "missing 'COMID' coord variable".to_string(),
    })?;
    let comids: Vec<i64> = comid_var
        .get_values::<i64, _>(..)
        .map_err(|e| DataError::NetCdf {
            path: path.clone(),
            source: e,
        })?;

    let val_var = file.variable(var).ok_or_else(|| DataError::Malformed {
        path: path.clone(),
        message: format!("missing variable '{var}'"),
    })?;
    let vals: Vec<f32> = val_var
        .get_values::<f32, _>(..)
        .map_err(|e| DataError::NetCdf {
            path: path.clone(),
            source: e,
        })?;

    if vals.len() != comids.len() {
        return Err(DataError::Malformed {
            path,
            message: format!(
                "'{var}' length {} != 'COMID' length {}",
                vals.len(),
                comids.len()
            ),
        });
    }

    Ok(comids.into_iter().zip(vals).collect())
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Write a minimal COMID-keyed f32 netCDF4 file, mirroring
    /// `dump_parameters::write_netcdf`'s schema (COMID dim + i64 coord +
    /// named f32 variables), for use in unit tests.
    fn write_test_dump(path: &std::path::Path, comids: &[i64], vars: &[(&str, &[f32])]) {
        let mut f = netcdf::create(path).expect("create netcdf");
        f.add_dimension("COMID", comids.len()).unwrap();
        let mut cv = f.add_variable::<i64>("COMID", &["COMID"]).unwrap();
        cv.put_values(comids, ..).unwrap();
        for (name, values) in vars {
            let mut v = f.add_variable::<f32>(name, &["COMID"]).unwrap();
            v.put_values(values, ..).unwrap();
        }
    }

    #[test]
    fn load_comid_field_reads_by_comid_not_position() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("dump.nc");
        write_test_dump(
            &path,
            &[30, 10, 20],
            &[("n", &[0.3, 0.1, 0.2]), ("q_spatial", &[0.53, 0.51, 0.52])],
        );

        let n = load_comid_field(&path, "n").unwrap();
        assert_eq!(n.len(), 3);
        assert!((n[&10] - 0.1).abs() < 1e-6);
        assert!((n[&20] - 0.2).abs() < 1e-6);
        assert!((n[&30] - 0.3).abs() < 1e-6);

        let q = load_comid_field(&path, "q_spatial").unwrap();
        assert!((q[&10] - 0.51).abs() < 1e-6);
    }

    #[test]
    fn load_comid_field_missing_variable_errors() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("dump.nc");
        write_test_dump(&path, &[10, 20], &[("n", &[0.1, 0.2])]);

        let err = load_comid_field(&path, "p_spatial").unwrap_err();
        let msg = format!("{err}");
        assert!(
            msg.contains("p_spatial") && msg.contains("missing"),
            "expected missing-variable error, got: {msg}"
        );
    }

    #[test]
    fn load_comid_field_missing_comid_dim_errors() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("no_comid.nc");
        let mut f = netcdf::create(&path).expect("create netcdf");
        f.add_dimension("ROW", 2).unwrap();
        let mut v = f.add_variable::<f32>("n", &["ROW"]).unwrap();
        v.put_values(&[0.1_f32, 0.2], ..).unwrap();
        drop(f);

        let err = load_comid_field(&path, "n").unwrap_err();
        let msg = format!("{err}");
        assert!(
            msg.contains("COMID"),
            "expected missing-COMID-coord error, got: {msg}"
        );
    }
}

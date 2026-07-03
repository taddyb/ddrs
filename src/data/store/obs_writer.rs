//! Writer for a minimal zarr-v2 observations store readable by
//! `ObservationsStore` (format template: the hand-written fixture in
//! `zarr_obs.rs` tests). One f64 array per STAID, single chunk,
//! uncompressed, little-endian, C order. Index 0 = `epoch` (the reader's
//! implicit 1980-01-01); rows before `day0` are NaN padding.

use std::path::Path;

use chrono::NaiveDate;

use crate::data::error::{DataError, Result};

pub fn write_obs_zarr_v2(
    dir: &Path,
    staids: &[String],
    epoch: NaiveDate,
    day0: NaiveDate,
    daily: &ndarray::Array2<f32>, // (G, D) m³/s, row g = staids[g]
) -> Result<()> {
    let io = |e: std::io::Error| DataError::Io {
        path: dir.to_path_buf(),
        source: e,
    };
    let (g, d) = daily.dim();
    assert_eq!(g, staids.len(), "daily rows != staids");
    let pad = (day0 - epoch).num_days();
    assert!(pad >= 0, "day0 before epoch");
    let n_time = pad as usize + d;

    std::fs::create_dir_all(dir).map_err(io)?;
    std::fs::write(dir.join(".zgroup"), r#"{"zarr_format": 2}"#).map_err(io)?;
    let zarray = format!(
        r#"{{"chunks": [{n_time}], "compressor": null, "dtype": "<f8",
"fill_value": "NaN", "filters": null, "order": "C",
"shape": [{n_time}], "zarr_format": 2}}"#
    );
    for (gi, staid) in staids.iter().enumerate() {
        let adir = dir.join(staid);
        std::fs::create_dir_all(&adir).map_err(io)?;
        std::fs::write(adir.join(".zarray"), &zarray).map_err(io)?;
        let mut bytes = Vec::with_capacity(n_time * 8);
        for _ in 0..pad {
            bytes.extend_from_slice(&f64::NAN.to_le_bytes());
        }
        for di in 0..d {
            bytes.extend_from_slice(&(daily[(gi, di)] as f64).to_le_bytes());
        }
        std::fs::write(adir.join("0"), bytes).map_err(io)?;
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use chrono::NaiveDate;
    use ndarray::array;

    use crate::data::dates::RhoWindow;
    use crate::data::ids::Staid;
    use crate::data::store::ObservationsStore;

    #[test]
    fn roundtrip_through_dispatching_store() {
        let tmp = tempfile::tempdir().unwrap();
        let dir = tmp.path();

        // epoch 1980-01-01, day0 1980-01-04 → 3 NaN pad rows (store indices 0,1,2)
        let epoch = NaiveDate::from_ymd_opt(1980, 1, 1).unwrap();
        let day0 = NaiveDate::from_ymd_opt(1980, 1, 4).unwrap();
        let staids = vec!["01010000".to_string(), "02020000".to_string()];
        // (G=2, D=5): gauge 0 = 1..5, gauge 1 = 10..50
        let daily = array![[1.0f32, 2.0, 3.0, 4.0, 5.0], [10.0, 20.0, 30.0, 40.0, 50.0]];

        write_obs_zarr_v2(dir, &staids, epoch, day0, &daily).unwrap();

        // Open through the DISPATCHING store — sniff must fire
        let store = ObservationsStore::open(dir).unwrap();

        // contains check
        assert!(store.contains(&Staid::new("01010000")));
        assert!(store.contains(&Staid::new("02020000")));
        assert!(!store.contains(&Staid::new("99999999")));

        // Window read: start 1980-01-03 (store day 2, last pad day), 6 days
        // covers store days 2,3,4,5,6,7  →  NaN, data[0..4]
        let window = RhoWindow {
            start_day_idx: 2, // relative to whatever TimeAxis; unused by obs read
            rho_days: 6,
            window_start: NaiveDate::from_ymd_opt(1980, 1, 3).unwrap(),
        };
        let gauge_staids = [Staid::new("01010000"), Staid::new("02020000")];
        let obs = store.read_window(&window, &gauge_staids).unwrap();

        assert_eq!(obs.shape(), &[6, 2]);
        // row 0 (store day 2): NaN pad
        assert!(obs[(0, 0)].is_nan(), "row0 g0 should be NaN");
        assert!(obs[(0, 1)].is_nan(), "row0 g1 should be NaN");
        // row 1 (store day 3 = day0): first real data
        assert_eq!(obs[(1, 0)], 1.0, "row1 g0");
        assert_eq!(obs[(1, 1)], 10.0, "row1 g1");
        // row 5 (store day 7 = data[4]): last data day
        assert_eq!(obs[(5, 0)], 5.0, "row5 g0");
        assert_eq!(obs[(5, 1)], 50.0, "row5 g1");
    }
}

//! Gage metadata CSV reader.
//!
//! Mirrors `~/projects/ddr/src/ddr/io/readers.py::read_gage_info` (lines
//! ~100-160). Required columns: `STAID, STANAME, DRAIN_SQKM, LAT_GAGE,
//! LNG_GAGE`. Optional columns: `COMID, COMID_DRAIN_SQKM,
//! COMID_UNITAREA_SQKM, ABS_DIFF, DA_VALID, FLOW_SCALE`.
//!
//! STAID values are zero-padded to 8 characters at construction (matches
//! DDR's canonical form).

use std::collections::HashMap;
use std::io::Read;
use std::path::PathBuf;

use serde::Deserialize;

use crate::data::error::{DataError, Result};
use crate::data::ids::Staid;

#[derive(Clone, Debug)]
pub struct GageRow {
    pub staid: Staid,
    pub staname: String,
    pub drain_sqkm: f64,
    pub lat_gage: f64,
    pub lng_gage: f64,
    pub comid: Option<i64>,
    pub comid_drain_sqkm: Option<f64>,
    pub comid_unitarea_sqkm: Option<f64>,
    pub abs_diff: Option<f64>,
    pub da_valid: Option<bool>,
    pub flow_scale: Option<f32>,
}

#[derive(Debug)]
pub struct GageMetadata {
    pub path: PathBuf,
    pub rows: Vec<GageRow>,
    pub by_staid: HashMap<Staid, usize>,
}

#[derive(Debug, Deserialize)]
struct RawRow {
    #[serde(rename = "STAID")]
    staid: String,
    #[serde(rename = "STANAME")]
    staname: Option<String>,
    #[serde(rename = "DRAIN_SQKM")]
    drain_sqkm: f64,
    #[serde(rename = "LAT_GAGE")]
    lat_gage: f64,
    #[serde(rename = "LNG_GAGE")]
    lng_gage: f64,
    #[serde(rename = "COMID", default)]
    comid: Option<i64>,
    #[serde(rename = "COMID_DRAIN_SQKM", default)]
    comid_drain_sqkm: Option<f64>,
    #[serde(rename = "COMID_UNITAREA_SQKM", default)]
    comid_unitarea_sqkm: Option<f64>,
    #[serde(rename = "ABS_DIFF", default)]
    abs_diff: Option<f64>,
    #[serde(rename = "DA_VALID", default)]
    da_valid: Option<DaValid>,
    #[serde(rename = "FLOW_SCALE", default)]
    flow_scale: Option<f32>,
}

#[derive(Debug, Copy, Clone)]
struct DaValid(bool);

impl<'de> serde::Deserialize<'de> for DaValid {
    fn deserialize<D: serde::Deserializer<'de>>(d: D) -> std::result::Result<Self, D::Error> {
        let s: String = String::deserialize(d)?;
        match s.as_str() {
            "True" | "true" | "1" => Ok(DaValid(true)),
            "False" | "false" | "0" => Ok(DaValid(false)),
            other => Err(serde::de::Error::custom(format!(
                "DA_VALID expected True/False/1/0, got {other}"
            ))),
        }
    }
}

impl GageMetadata {
    pub fn open(path: impl Into<PathBuf>) -> Result<Self> {
        let path = path.into();
        let file = std::fs::File::open(&path).map_err(|e| DataError::Io {
            path: path.clone(),
            source: e,
        })?;
        Self::from_reader(file, path)
    }

    fn from_reader<R: Read>(rdr: R, path: PathBuf) -> Result<Self> {
        let mut csv_rdr = csv::ReaderBuilder::new().has_headers(true).from_reader(rdr);
        let mut rows: Vec<GageRow> = Vec::new();
        for rec in csv_rdr.deserialize::<RawRow>() {
            let raw = rec.map_err(|e| DataError::Csv {
                path: path.clone(),
                source: e,
            })?;
            rows.push(GageRow {
                staid: Staid::new(&raw.staid),
                staname: raw.staname.unwrap_or_else(|| raw.staid.clone()),
                drain_sqkm: raw.drain_sqkm,
                lat_gage: raw.lat_gage,
                lng_gage: raw.lng_gage,
                comid: raw.comid,
                comid_drain_sqkm: raw.comid_drain_sqkm,
                comid_unitarea_sqkm: raw.comid_unitarea_sqkm,
                abs_diff: raw.abs_diff,
                da_valid: raw.da_valid.map(|v| v.0),
                flow_scale: raw.flow_scale,
            });
        }
        let by_staid: HashMap<Staid, usize> = rows
            .iter()
            .enumerate()
            .map(|(i, r)| (r.staid.clone(), i))
            .collect();
        Ok(Self {
            path,
            rows,
            by_staid,
        })
    }

    pub fn staids(&self) -> Vec<Staid> {
        self.rows.iter().map(|r| r.staid.clone()).collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const SAMPLE_CSV: &str = "\
STAID,STANAME,DRAIN_SQKM,LAT_GAGE,LNG_GAGE,COMID,COMID_DRAIN_SQKM,COMID_UNITAREA_SQKM,ABS_DIFF,DA_VALID,FLOW_SCALE
14190500,\"LUCKIAMUTE RIVER NEAR SUVER, OR\",603.4942,44.783175,-123.234543,78022248,659.0194570316622,57.94836462236595,55.52525703166225,True,0.04181494346724791
1563500,FAKE,1.0,2.0,3.0,99,2.0,2.0,1.0,False,0.5
";

    #[test]
    fn parses_required_fields_and_zfills_staid() {
        let m = GageMetadata::from_reader(
            SAMPLE_CSV.as_bytes(),
            std::path::PathBuf::from("<inline>"),
        )
        .expect("parse");
        assert_eq!(m.rows.len(), 2);
        assert_eq!(m.rows[0].staid.as_str(), "14190500");
        assert_eq!(m.rows[1].staid.as_str(), "01563500");
        assert!((m.rows[0].drain_sqkm - 603.4942).abs() < 1e-6);
    }

    #[test]
    fn parses_optional_fields() {
        let m = GageMetadata::from_reader(
            SAMPLE_CSV.as_bytes(),
            std::path::PathBuf::from("<inline>"),
        )
        .expect("parse");
        let r0 = &m.rows[0];
        assert_eq!(r0.comid, Some(78022248));
        assert_eq!(r0.da_valid, Some(true));
        assert!(r0.flow_scale.is_some());
        let r1 = &m.rows[1];
        assert_eq!(r1.da_valid, Some(false));
    }

    #[test]
    fn lookup_by_staid_uses_padded_form() {
        let m = GageMetadata::from_reader(
            SAMPLE_CSV.as_bytes(),
            std::path::PathBuf::from("<inline>"),
        )
        .expect("parse");
        use crate::data::ids::Staid;
        assert!(m.by_staid.contains_key(&Staid::new("1563500")));
        assert!(m.by_staid.contains_key(&Staid::new("01563500")));
    }
}

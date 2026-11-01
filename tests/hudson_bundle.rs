//! Contract test for the committed Hudson tidal example bundle
//! (`examples/hudson/data/`).
//!
//! Asserts the bundle through the real readers, the same code paths
//! `ddrs plan` and `ddrs run` use. Unlike the live-store tests, this never
//! skips: the bundle is committed in the repo.

use std::collections::{HashMap, HashSet};

use chrono::{Duration, NaiveDate, NaiveDateTime};
use ddrs::config::Config;
use ddrs::data::store::stage_boundary::TIME_FORMAT;
use ddrs::data::store::{ChannelGeometryStore, StageBoundaryStore};
use ddrs::data::{
    Comid, ConusAdjacencyStore, GagesAdjacencyStore, MeritGagesDataset, ObservationsStore,
    RhoWindow, Staid, StreamflowSource,
};

const BUNDLE: &str = "examples/hudson/data";
const HARBOR: i64 = 73005067;
const SEED: i64 = 73001655;
const POUGHKEEPSIE: &str = "01372058";
const GAUGES: [&str; 4] = ["01372058", "01358000", "01375000", "73005067"];
const BATTERY_CSV: &str = "examples/hudson/data/hudson_battery_stage_navd88.csv";
const GEOMETRY_CSV: &str = "examples/hudson/data/hudson_channel_geometry.csv";
/// 2001-05-01 to 2019-12-31 inclusive, in days. The Battery has no data before 2001-05-01.
const RECORD_DAYS: usize = 6_819;

fn conus() -> ConusAdjacencyStore {
    ConusAdjacencyStore::open(format!("{BUNDLE}/hudson_conus_adjacency.zarr"))
        .expect("open bundle conus adjacency")
}

fn start_of_record() -> NaiveDateTime {
    NaiveDate::from_ymd_opt(2001, 5, 1).unwrap().and_hms_opt(0, 0, 0).unwrap()
}

#[test]
fn conus_adjacency_is_756_reaches_lower_triangular() {
    let store = conus();
    assert_eq!(store.n, 756);
    assert_eq!(store.nnz, 755);
    assert_eq!(store.order.len(), store.n);
    let violations = store
        .indices_0
        .iter()
        .zip(&store.indices_1)
        .filter(|(r, c)| r < c)
        .count();
    assert_eq!(violations, 0, "{violations} edges violate row >= col");

    // Exactly one outlet: a reach that is nobody's upstream. It is the harbor reach.
    let uppers: HashSet<i32> = store.indices_1.iter().copied().collect();
    let outlet_rows: Vec<usize> = (0..store.n).filter(|i| !uppers.contains(&(*i as i32))).collect();
    assert_eq!(outlet_rows.len(), 1, "outlets: {outlet_rows:?}");
    assert_eq!(store.order[outlet_rows[0]], Comid(HARBOR));
}

#[test]
fn gauge_subgraphs_have_the_measured_sizes() {
    let store = conus();
    let staids: Vec<Staid> = GAUGES.iter().map(|s| Staid::from(*s)).collect();
    let gages = GagesAdjacencyStore::open(format!("{BUNDLE}/hudson_gages_adjacency.zarr"), &staids)
        .expect("open bundle gages adjacency");
    let want = [("01372058", 661), ("01358000", 447), ("01375000", 27), ("73005067", 756)];
    for (g, n) in want {
        let sub = gages.subgraphs.get(&Staid::from(g)).unwrap_or_else(|| panic!("{g} missing"));
        assert!(!sub.is_headwater(), "{g} must not be a headwater");
        assert_eq!(sub.upstream_comids(&store).len(), n, "upstream size of {g}");
    }
}

fn example_cfg(name: &str) -> Config {
    Config::from_yaml_file(format!("examples/hudson/{name}")).expect("load example config")
}

#[test]
fn harbor_pseudo_gauge_survives_the_gauge_filter() {
    let cfg = example_cfg("ddrs.yaml");
    let ds = MeritGagesDataset::open(&cfg).expect("open Hudson dataset");
    assert_eq!(ds.len(), 4, "gauges: {:?}", ds.staids());
    let harbor = Staid::from("73005067");
    assert!(ds.staids().contains(&harbor), "pseudo-gauge was filtered out");

    // One collated training batch holding every gauge.
    let start = NaiveDate::from_ymd_opt(2002, 10, 1).unwrap();
    let axis = ddrs::data::TimeAxis::new(start, NaiveDate::from_ymd_opt(2012, 9, 30).unwrap());
    let window = RhoWindow { start_day_idx: 0, rho_days: 90, window_start: start };
    assert_eq!(axis.day_index(start), Some(0));
    let all: Vec<Staid> = ds.staids().to_vec();
    let batch = ds.collate(&all, &window).expect("collate the whole domain");
    assert_eq!(batch.adjacency.n, 756);
    let ups: HashSet<i32> = batch.adjacency.cols.iter().copied().collect();
    let outlets: Vec<usize> = (0..batch.adjacency.n).filter(|i| !ups.contains(&(*i as i32))).collect();
    assert_eq!(outlets.len(), 1, "outlets: {outlets:?}");
    assert_eq!(batch.divide_comids[outlets[0]], Comid(HARBOR));

    // The pseudo-gauge's observation column is all NaN; the other three are not.
    let col = batch.gauge_staids.iter().position(|s| *s == harbor).expect("harbor column");
    let obs = &batch.observations;
    assert!(obs.column(col).iter().all(|v| v.is_nan()), "harbor observations must be NaN");
    assert_eq!(batch.gauge_staids.len(), 4);
}

#[test]
fn stores_start_2001_05_01_and_readers_accept_it() {
    let day = NaiveDate::from_ymd_opt(2001, 5, 1).unwrap();
    let q = StreamflowSource::open(format!("{BUNDLE}/hudson_qprime.ic")).expect("open Q'");
    match &q {
        StreamflowSource::Icechunk(s) => {
            assert_eq!(s.time_start, day);
            assert_eq!(s.n_time, RECORD_DAYS);
        }
        _ => panic!("expected an icechunk Q' store"),
    }
    let o = ObservationsStore::open(format!("{BUNDLE}/hudson_obs.ic")).expect("open obs");
    match &o {
        ObservationsStore::Usgs(s) => {
            assert_eq!(s.time_start, day);
            assert_eq!(s.n_time, RECORD_DAYS);
        }
        _ => panic!("expected an icechunk obs store"),
    }
    // The MC-only config opens the dataset with these stores.
    let cfg = example_cfg("ddrs_mc_only.yaml");
    let ds = MeritGagesDataset::open(&cfg).expect("open MC-only dataset");
    assert_eq!(ds.len(), 4);
}

#[test]
fn poughkeepsie_obs_has_6159_valid_days() {
    let o = ObservationsStore::open(format!("{BUNDLE}/hudson_obs.ic")).expect("open obs");
    let day = NaiveDate::from_ymd_opt(2001, 5, 1).unwrap();
    let ids: Vec<Staid> = GAUGES.iter().map(|s| Staid::from(*s)).collect();
    let m = o.read_window_daily(day, RECORD_DAYS, &ids).expect("read obs");
    let valid = |g: &str| {
        let c = ids.iter().position(|s| s.as_str() == g).unwrap();
        m.column(c).iter().filter(|v| v.is_finite()).count()
    };
    assert_eq!(valid(POUGHKEEPSIE), 6159);
    assert_eq!(valid("73005067"), 0, "the pseudo-gauge has no observations");
    assert!(valid("01358000") > 6000, "Green Island is a real series");
    assert!(valid("01375000") > 6000, "Croton is a real series");
    // The merged series is NaN before the 72137 record starts (2002-03-15).
    let c = ids.iter().position(|s| s.as_str() == POUGHKEEPSIE).unwrap();
    let first = (NaiveDate::from_ymd_opt(2002, 3, 15).unwrap() - day).num_days() as usize;
    assert!(m.column(c).iter().take(first).all(|v| v.is_nan()));
}

fn parse_time(s: &str) -> NaiveDateTime {
    NaiveDateTime::parse_from_str(s, TIME_FORMAT).unwrap_or_else(|e| panic!("time {s:?}: {e}"))
}

#[test]
fn battery_series_is_contiguous_and_gap_log_matches() {
    let store = StageBoundaryStore::open(BATTERY_CSV).expect("open Battery series");
    let n_hours = RECORD_DAYS * 24;
    assert_eq!(n_hours, 163_656);
    let t0 = start_of_record();
    assert_eq!(store.slice(t0, n_hours).expect("whole record").len(), n_hours);
    assert!(store.slice(t0, n_hours + 1).is_err());
    assert!(store.slice(t0 - Duration::hours(1), 1).is_err());

    // The CSV text: first row, row count, and the exact time format on every row.
    let text = std::fs::read_to_string(BATTERY_CSV).unwrap();
    let mut lines = text.lines();
    assert_eq!(lines.next(), Some("time,stage_m"));
    let mut rows = 0usize;
    let mut times: HashSet<String> = HashSet::new();
    for (i, line) in lines.enumerate() {
        let t = line.split(',').next().unwrap();
        let parsed = parse_time(t);
        assert_eq!(parsed.format(TIME_FORMAT).to_string(), t, "row {}", i + 1);
        if i == 0 {
            assert_eq!(t, "2001-05-01T00:00:00Z");
        }
        times.insert(t.to_string());
        rows += 1;
    }
    assert_eq!(rows, n_hours);

    // The gap log: every logged hour is a row of the record, once, with a known method.
    let gaps = std::fs::read_to_string(format!("{BATTERY_CSV}.gaps.txt")).expect("gap log");
    let mut seen: HashSet<&str> = HashSet::new();
    let mut by_method: HashMap<String, usize> = HashMap::new();
    for line in gaps.lines().filter(|l| !l.starts_with('#') && !l.trim().is_empty()) {
        let mut parts = line.split(',');
        let t = parts.next().unwrap().trim();
        let method = parts.next().unwrap_or("").trim();
        assert!(
            ["6min_mean", "predictions_plus_residual", "linear"].contains(&method),
            "unknown fill method in: {line}"
        );
        assert!(times.contains(t), "gap hour {t} is not in the record");
        assert!(seen.insert(t), "gap hour {t} is logged twice");
        *by_method.entry(method.to_string()).or_default() += 1;
    }
    // The controller ruling replaces the 0.1 % target: the share stays under 3 %.
    let filled = seen.len();
    assert!(filled > 0 && (filled as f64) < 0.03 * n_hours as f64, "{filled} filled hours");
    assert_eq!(by_method.values().sum::<usize>(), filled);
}

/// COMIDs from the seed reach down to the outlet, walking the bundle's adjacency.
fn stage_closure(store: &ConusAdjacencyStore) -> Vec<Comid> {
    let down: HashMap<i32, i32> = store
        .indices_1
        .iter()
        .copied()
        .zip(store.indices_0.iter().copied())
        .collect();
    let mut cur = store.order.iter().position(|c| c.0 == SEED).expect("seed in bundle") as i32;
    let mut out = vec![store.order[cur as usize]];
    while let Some(&d) = down.get(&cur) {
        out.push(store.order[d as usize]);
        cur = d;
    }
    out
}

#[test]
fn channel_geometry_covers_the_41_stage_reaches() {
    let store = conus();
    let closure = stage_closure(&store);
    assert_eq!(closure.len(), 41);
    assert_eq!(closure.last(), Some(&Comid(HARBOR)));

    let geom = ChannelGeometryStore::open(GEOMETRY_CSV).expect("open geometry table");
    let rows = geom.rows_for(&closure);
    assert_eq!(rows.rows.len(), 41, "every stage reach needs a row");
    for i in 0..41 {
        assert!(rows.top_width[i] > 0.0, "row {i}: top_width_m");
        assert!(rows.n[i] > 0.0, "row {i}: n");
        assert!(rows.bed_z_m_is_finite(i), "row {i}: bed_z_m");
    }

    // The CSV text: exact header, 5 fields on every data row.
    let text = std::fs::read_to_string(GEOMETRY_CSV).unwrap();
    let mut data = text.lines().filter(|l| !l.starts_with('#') && !l.trim().is_empty());
    assert_eq!(data.next(), Some("COMID,top_width_m,bed_z_m,n,source"));
    let mut count = 0;
    for line in data {
        assert_eq!(line.split(',').count(), 5, "row needs 5 fields: {line}");
        count += 1;
    }
    assert_eq!(count, 41);

    // The anchor in ddrs.yaml equals the outlet row's bed (L6).
    let cfg = example_cfg("ddrs.yaml");
    let anchor = cfg.params.stage_routing.as_ref().expect("stage_routing").anchor_elevation_m;
    let outlet = closure.iter().position(|c| c.0 == HARBOR).unwrap();
    assert_eq!(anchor, rows.bed_z[outlet], "anchor_elevation_m must equal the outlet bed");
}

trait BedFinite {
    fn bed_z_m_is_finite(&self, i: usize) -> bool;
}
impl BedFinite for ddrs::data::store::ChannelGeometryRows {
    fn bed_z_m_is_finite(&self, i: usize) -> bool {
        self.bed_z[i].is_finite()
    }
}

#[test]
fn example_configs_parse() {
    let full = example_cfg("ddrs.yaml");
    let sr = full.params.stage_routing.as_ref().expect("stage_routing in ddrs.yaml");
    assert!(sr.enabled);
    assert_eq!(sr.boundary_comid, Some(HARBOR));
    assert_eq!(full.experiment.as_ref().unwrap().batch_size, 4);
    let mc = example_cfg("ddrs_mc_only.yaml");
    assert!(mc.params.stage_routing.is_none());
    assert_eq!(mc.experiment.as_ref().unwrap().batch_size, 4);
}

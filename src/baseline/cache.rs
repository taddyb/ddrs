//! Content-addressed cache for the summed Q' baseline.
//!
//! Layout under `<workspace_root>/baselines/<key>/`:
//!
//! ```text
//! predictions.f32   — raw f32 LE, row-major (n_gauges × n_days)
//! observations.f32  — raw f32 LE, row-major (n_gauges × n_days)
//! manifest.json     — gage_ids, time_range, metrics, dims, source-path provenance
//! ```
//!
//! The key is blake3 of the canonicalized data-source paths and the
//! testing eval window. Training-only fields (seed, KAN config, lr) do
//! NOT participate, so re-running `ddrs plan` after tweaking training
//! knobs is a cache hit.

use std::fs;
use std::io::{Read, Write};
use std::path::{Path, PathBuf};

use chrono::NaiveDate;
use ndarray::Array2;
use serde::{Deserialize, Serialize};

use crate::config::Config;
use crate::data::ids::Staid;
use crate::training::metrics::Metrics;

use super::summed_q_prime::{compute, BaselineError, SummedQPrime};

/// 16-hex-char (64-bit) prefix of blake3 over the data-source paths +
/// eval window. Safe for filesystem use; collision-free at our scale.
pub fn cache_key(test_cfg: &Config) -> Result<String, BaselineError> {
    let ds = test_cfg
        .data_sources
        .as_ref()
        .ok_or(BaselineError::ConfigMissing("data_sources"))?;
    let exp = test_cfg
        .experiment
        .as_ref()
        .ok_or(BaselineError::ConfigMissing("experiment"))?;

    // `ddrs plan` resolves adjacency (explicit or managed build) and materializes
    // the resolved paths into the config before this runs, so these are defensive
    // — they only fire if a caller bypasses plan with a fabric-only config.
    let conus_adj = ds
        .conus_adjacency
        .as_ref()
        .ok_or(BaselineError::ConfigMissing(
            "conus_adjacency — adjacency not resolved; run via `ddrs plan`/`ddrs run`, \
             or set conus_adjacency/gages_adjacency explicitly",
        ))?;
    let gages_adj = ds
        .gages_adjacency
        .as_ref()
        .ok_or(BaselineError::ConfigMissing(
            "gages_adjacency — adjacency not resolved; run via `ddrs plan`/`ddrs run`, \
             or set conus_adjacency/gages_adjacency explicitly",
        ))?;
    let mut h = blake3::Hasher::new();
    for p in [
        &ds.streamflow,
        &ds.observations,
        &ds.gages,
        gages_adj,
        conus_adj,
    ] {
        h.update(canonicalize_or_raw(p).as_bytes());
        h.update(b"\n");
    }
    h.update(exp.start_time.as_bytes());
    h.update(b"\n");
    h.update(exp.end_time.as_bytes());

    let hex = h.finalize().to_hex();
    Ok(hex.as_str()[..16].to_string())
}

fn canonicalize_or_raw(p: &Path) -> String {
    p.canonicalize()
        .map(|c| c.display().to_string())
        .unwrap_or_else(|_| p.display().to_string())
}

#[derive(Debug, Serialize, Deserialize)]
struct CacheManifest {
    key: String,
    n_gauges: usize,
    n_days: usize,
    gage_ids: Vec<String>,
    time_range_daily: Vec<String>, // ISO-8601 dates
    // NaN-safe — serde_json writes NaN as `null` but won't deserialize
    // `null` into f32, so wrap each vector in Option per element.
    metrics: MetricsJson,
    sources: SourceProvenance,
}

/// JSON-safe variant of `Metrics` where NaN is encoded as `null`.
#[derive(Debug, Serialize, Deserialize)]
struct MetricsJson {
    nse: Vec<Option<f32>>,
    rmse: Vec<Option<f32>>,
    kge: Vec<Option<f32>>,
    bias: Vec<Option<f32>>,
    fhv: Vec<Option<f32>>,
    flv: Vec<Option<f32>>,
}

fn finite_to_option(v: f32) -> Option<f32> {
    if v.is_finite() { Some(v) } else { None }
}

fn option_to_nan(v: Option<f32>) -> f32 {
    v.unwrap_or(f32::NAN)
}

impl From<&Metrics> for MetricsJson {
    fn from(m: &Metrics) -> Self {
        MetricsJson {
            nse: m.nse.iter().copied().map(finite_to_option).collect(),
            rmse: m.rmse.iter().copied().map(finite_to_option).collect(),
            kge: m.kge.iter().copied().map(finite_to_option).collect(),
            bias: m.bias.iter().copied().map(finite_to_option).collect(),
            fhv: m.fhv.iter().copied().map(finite_to_option).collect(),
            flv: m.flv.iter().copied().map(finite_to_option).collect(),
        }
    }
}

impl From<MetricsJson> for Metrics {
    fn from(m: MetricsJson) -> Self {
        Metrics {
            nse: m.nse.into_iter().map(option_to_nan).collect(),
            rmse: m.rmse.into_iter().map(option_to_nan).collect(),
            kge: m.kge.into_iter().map(option_to_nan).collect(),
            bias: m.bias.into_iter().map(option_to_nan).collect(),
            fhv: m.fhv.into_iter().map(option_to_nan).collect(),
            flv: m.flv.into_iter().map(option_to_nan).collect(),
        }
    }
}

#[derive(Debug, Serialize, Deserialize)]
struct SourceProvenance {
    streamflow: PathBuf,
    observations: PathBuf,
    gages: PathBuf,
    gages_adjacency: PathBuf,
    conus_adjacency: PathBuf,
    start_time: String,
    end_time: String,
}

/// Resolve `<workspace_root>/baselines/<key>/` for the given config.
pub fn cache_dir(workspace_root: &Path, key: &str) -> PathBuf {
    workspace_root.join("baselines").join(key)
}

/// Try to load a cached baseline. Returns `Ok(None)` if the directory is
/// missing or any file fails to parse — the caller should fall back to
/// recomputation.
pub fn load_cached(workspace_root: &Path, key: &str) -> Option<SummedQPrime> {
    let dir = cache_dir(workspace_root, key);
    let manifest_json = fs::read_to_string(dir.join("manifest.json")).ok()?;
    let manifest: CacheManifest = serde_json::from_str(&manifest_json).ok()?;
    let predictions = read_f32_matrix(&dir.join("predictions.f32"), manifest.n_gauges, manifest.n_days)
        .ok()?;
    let observations = read_f32_matrix(
        &dir.join("observations.f32"),
        manifest.n_gauges,
        manifest.n_days,
    )
    .ok()?;
    let gage_ids: Vec<Staid> = manifest
        .gage_ids
        .iter()
        .map(|s| Staid::from(s.as_str()))
        .collect();
    let time_range_daily: Vec<NaiveDate> = manifest
        .time_range_daily
        .into_iter()
        .map(|s| NaiveDate::parse_from_str(&s, "%Y-%m-%d").ok())
        .collect::<Option<Vec<_>>>()?;
    Some(SummedQPrime {
        predictions,
        observations,
        gage_ids,
        time_range_daily,
        metrics: manifest.metrics.into(),
    })
}

/// Write a baseline to the cache. Overwrites any prior content under
/// `<workspace_root>/baselines/<key>/`.
pub fn save_cached(
    workspace_root: &Path,
    key: &str,
    test_cfg: &Config,
    q: &SummedQPrime,
) -> Result<(), BaselineError> {
    let dir = cache_dir(workspace_root, key);
    fs::create_dir_all(&dir).map_err(io_err)?;
    let (n_gauges, n_days) = q.predictions.dim();

    write_f32_matrix(&dir.join("predictions.f32"), &q.predictions)?;
    write_f32_matrix(&dir.join("observations.f32"), &q.observations)?;

    let ds = test_cfg
        .data_sources
        .as_ref()
        .ok_or(BaselineError::ConfigMissing("data_sources"))?;
    let exp = test_cfg
        .experiment
        .as_ref()
        .ok_or(BaselineError::ConfigMissing("experiment"))?;
    // Defensive: `ddrs plan` materializes resolved paths before save_cached runs.
    let conus_adj = ds
        .conus_adjacency
        .clone()
        .ok_or(BaselineError::ConfigMissing(
            "conus_adjacency — adjacency not resolved; run via `ddrs plan`/`ddrs run`, \
             or set conus_adjacency/gages_adjacency explicitly",
        ))?;
    let gages_adj = ds
        .gages_adjacency
        .clone()
        .ok_or(BaselineError::ConfigMissing(
            "gages_adjacency — adjacency not resolved; run via `ddrs plan`/`ddrs run`, \
             or set conus_adjacency/gages_adjacency explicitly",
        ))?;
    let manifest = CacheManifest {
        key: key.to_string(),
        n_gauges,
        n_days,
        gage_ids: q.gage_ids.iter().map(|s| s.as_str().to_string()).collect(),
        time_range_daily: q
            .time_range_daily
            .iter()
            .map(|d| d.format("%Y-%m-%d").to_string())
            .collect(),
        metrics: (&q.metrics).into(),
        sources: SourceProvenance {
            streamflow: ds.streamflow.clone(),
            observations: ds.observations.clone(),
            gages: ds.gages.clone(),
            gages_adjacency: gages_adj,
            conus_adjacency: conus_adj,
            start_time: exp.start_time.clone(),
            end_time: exp.end_time.clone(),
        },
    };
    let json = serde_json::to_string_pretty(&manifest)
        .expect("serializing CacheManifest cannot fail");
    fs::write(dir.join("manifest.json"), json).map_err(io_err)?;
    Ok(())
}

/// Load the cached baseline if present and valid; otherwise compute and
/// persist. The hot path for both `ddrs plan` and `ddrs run`.
pub fn compute_or_load_cached(
    test_cfg: &Config,
    workspace_root: &Path,
) -> Result<(SummedQPrime, String, bool), BaselineError> {
    let key = cache_key(test_cfg)?;
    if let Some(cached) = load_cached(workspace_root, &key) {
        return Ok((cached, key, true));
    }
    let q = compute(test_cfg)?;
    save_cached(workspace_root, &key, test_cfg, &q)?;
    Ok((q, key, false))
}

// ---------------------------- raw I/O helpers ----------------------------

fn write_f32_matrix(path: &Path, m: &Array2<f32>) -> Result<(), BaselineError> {
    // Ensure row-major layout. `as_standard_layout()` is a no-op when already
    // contiguous, which is the case for `Array2::zeros((g, t))`.
    let owned = m.as_standard_layout();
    let bytes: &[u8] = bytemuck_cast(owned.as_slice().expect("standard layout has slice"));
    let mut f = fs::File::create(path).map_err(io_err)?;
    f.write_all(bytes).map_err(io_err)?;
    Ok(())
}

fn read_f32_matrix(path: &Path, n_rows: usize, n_cols: usize) -> Result<Array2<f32>, BaselineError> {
    let mut buf = Vec::with_capacity(n_rows * n_cols * 4);
    fs::File::open(path).map_err(io_err)?.read_to_end(&mut buf).map_err(io_err)?;
    let expected = n_rows * n_cols * 4;
    if buf.len() != expected {
        return Err(BaselineError::Data(crate::data::error::DataError::Malformed {
            path: path.to_path_buf(),
            message: format!(
                "expected {expected} bytes for ({n_rows}, {n_cols}) f32 matrix, got {}",
                buf.len()
            ),
        }));
    }
    let floats: Vec<f32> = buf
        .chunks_exact(4)
        .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect();
    Array2::from_shape_vec((n_rows, n_cols), floats).map_err(|e| {
        BaselineError::Data(crate::data::error::DataError::Malformed {
            path: path.to_path_buf(),
            message: format!("from_shape_vec failed: {e}"),
        })
    })
}

/// Reinterpret `&[f32]` as `&[u8]` (LE on x86_64). Avoids a bytemuck dep.
fn bytemuck_cast(slice: &[f32]) -> &[u8] {
    // SAFETY: f32 is Copy + has no padding; len in bytes is len * 4.
    unsafe { std::slice::from_raw_parts(slice.as_ptr() as *const u8, std::mem::size_of_val(slice)) }
}

fn io_err(e: std::io::Error) -> BaselineError {
    BaselineError::Data(crate::data::error::DataError::Io {
        path: PathBuf::from("<baseline cache>"),
        source: e,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::array;
    use std::collections::BTreeMap;

    use crate::baseline::summed_q_prime::assemble_from_arrays;
    use crate::data::ids::Comid;

    fn fake_summed_q_prime() -> SummedQPrime {
        let all_needed = vec![Comid(100), Comid(200)];
        let qr = array![[1.0_f32, 2.0], [10.0, 20.0], [100.0, 200.0]];
        let obs = array![[3.0_f32], [30.0], [300.0]];
        let mut basins = BTreeMap::new();
        basins.insert(Staid::from("A"), vec![Comid(100), Comid(200)]);
        assemble_from_arrays(
            &qr,
            &obs,
            &all_needed,
            &basins,
            vec![Staid::from("A")],
            NaiveDate::from_ymd_opt(2000, 1, 1).unwrap(),
            3,
        )
    }

    fn fake_config() -> Config {
        // Construct a Config with the minimum data_sources + experiment
        // fields the cache_key + provenance need. The data-source paths
        // don't need to exist for the key hash.
        use crate::config::{DataSources, Experiment};
        let mut cfg: Config = Config::default();
        cfg.mode = "testing".into();
        cfg.seed = 0;
        cfg.data_sources = Some(DataSources {
            attributes: PathBuf::from("/dev/null/attrs.nc"),
            conus_adjacency: Some(PathBuf::from("/dev/null/conus.zarr")),
            gages_adjacency: Some(PathBuf::from("/dev/null/gages_adj.zarr")),
            geospatial_fabric: None,
            geospatial_fabric_layer: None,
            streamflow: PathBuf::from("/dev/null/sf.ic"),
            observations: PathBuf::from("/dev/null/obs.ic"),
            gages: PathBuf::from("/dev/null/gages.csv"),
            aorc_precip: None,
        });
        cfg.experiment = Some(Experiment {
            batch_size: 1,
            start_time: "2000/01/01".into(),
            end_time: "2000/01/03".into(),
            epochs: 1,
            rho: None,
            shuffle: false,
            warmup: 0,
            learning_rate: Default::default(),
            grad_clip_max_norm: None,
            checkpoint: None,
            loss: Default::default(),
        });
        cfg
    }

    #[test]
    fn cache_key_is_stable_and_short() {
        let cfg = fake_config();
        let k1 = cache_key(&cfg).unwrap();
        let k2 = cache_key(&cfg).unwrap();
        assert_eq!(k1, k2);
        assert_eq!(k1.len(), 16);
    }

    #[test]
    fn cache_key_invalidates_on_window_change() {
        let mut a = fake_config();
        let mut b = fake_config();
        b.experiment.as_mut().unwrap().end_time = "2001/01/01".into();
        assert_ne!(cache_key(&a).unwrap(), cache_key(&b).unwrap());

        // Sanity: changing a non-key field (seed) does NOT invalidate.
        a.seed = 999;
        let mut a2 = fake_config();
        a2.seed = 0;
        assert_eq!(cache_key(&a).unwrap(), cache_key(&a2).unwrap());
    }

    #[test]
    fn round_trip_save_load_preserves_values() {
        let cfg = fake_config();
        let key = cache_key(&cfg).unwrap();
        let q = fake_summed_q_prime();

        let tmp = std::env::temp_dir()
            .join(format!("ddrs_baseline_cache_{}", std::process::id()));
        let _ = fs::remove_dir_all(&tmp);
        fs::create_dir_all(&tmp).unwrap();

        save_cached(&tmp, &key, &cfg, &q).unwrap();
        let loaded = load_cached(&tmp, &key).expect("load should succeed");

        assert_eq!(loaded.predictions, q.predictions);
        assert_eq!(loaded.observations, q.observations);
        assert_eq!(loaded.gage_ids, q.gage_ids);
        assert_eq!(loaded.time_range_daily, q.time_range_daily);
        assert_eq!(loaded.metrics.nse.len(), q.metrics.nse.len());
        for (a, b) in loaded.metrics.nse.iter().zip(&q.metrics.nse) {
            // NaN-aware equality
            if a.is_nan() && b.is_nan() {
                continue;
            }
            assert!((a - b).abs() < 1e-6);
        }

        let _ = fs::remove_dir_all(&tmp);
    }

    #[test]
    fn load_missing_returns_none() {
        let tmp = std::env::temp_dir()
            .join(format!("ddrs_baseline_missing_{}", std::process::id()));
        let _ = fs::remove_dir_all(&tmp);
        assert!(load_cached(&tmp, "deadbeefdeadbeef").is_none());
    }

    /// When adjacency paths are absent (fabric-only config, pre-Task 7),
    /// `cache_key` must return `Err` rather than panicking.
    #[test]
    fn cache_key_returns_err_when_adjacency_absent() {
        use crate::config::{DataSources, Experiment};
        let mut cfg = Config::default();
        cfg.mode = "testing".into();
        cfg.seed = 0;
        cfg.data_sources = Some(DataSources {
            attributes: PathBuf::from("/dev/null/attrs.nc"),
            conus_adjacency: None,      // ← fabric-only config
            gages_adjacency: None,
            geospatial_fabric: Some(PathBuf::from("/dev/null/fabric.shp")),
            geospatial_fabric_layer: None,
            streamflow: PathBuf::from("/dev/null/sf.ic"),
            observations: PathBuf::from("/dev/null/obs.ic"),
            gages: PathBuf::from("/dev/null/gages.csv"),
            aorc_precip: None,
        });
        cfg.experiment = Some(Experiment {
            batch_size: 1,
            start_time: "2000/01/01".into(),
            end_time: "2000/01/03".into(),
            epochs: 1,
            rho: None,
            shuffle: false,
            warmup: 0,
            learning_rate: Default::default(),
            grad_clip_max_norm: None,
            checkpoint: None,
            loss: Default::default(),
        });
        let result = cache_key(&cfg);
        assert!(
            result.is_err(),
            "cache_key must return Err when adjacency paths are absent"
        );
        let msg = result.unwrap_err().to_string();
        assert!(
            msg.contains("conus_adjacency"),
            "error message should mention conus_adjacency, got: {msg}"
        );
    }
}

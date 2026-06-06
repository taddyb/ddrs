//! KAN parameter dump as a library function.
//!
//! Same logic that `src/bin/dump_parameters.rs` runs; lifted out so
//! `cli::run --plot` and the standalone binary share one path.

use std::path::Path;

use burn::tensor::backend::{Backend, BackendTypes};
use burn::tensor::Tensor;
use ndarray::{s, Array2};

use crate::config::Config;
use crate::data::{fill_nans, AttrStats, AttributesStore, ConusAdjacencyStore};
use crate::error::CliError;
use crate::nn::{KanHead, KanHeadConfig};
use crate::routing::denormalize;
use crate::training::load_kan_head;

/// Run a trained KAN head over the full CONUS attribute matrix, denormalize
/// into physical units, and write `(COMID, n, q_spatial, p_spatial, slope)`
/// NetCDF4 to `output_path`. Returns the number of reaches written.
///
/// `checkpoint` is the base path (no `.mpk` suffix).
pub fn dump<I>(
    cfg: &Config,
    checkpoint: &Path,
    output_path: &Path,
    batch_size: usize,
    device: &<I as BackendTypes>::Device,
) -> Result<usize, CliError>
where
    I: Backend + BackendTypes,
{
    let ds = cfg.data_sources.as_ref().expect("data_sources section required");
    let head_cfg = cfg
        .kan_head
        .as_ref()
        .expect("kan_head section required for trained-KAN inference");

    // ---------- 1. CONUS topology: COMID order + per-reach slope ----------
    // Defensive: callers route through `ddrs run`/`plan`, which materialize the
    // resolved adjacency paths into the config before dump runs.
    let conus_path = ds.conus_adjacency.as_ref().ok_or_else(|| CliError::ConfigInvalid {
        path: "<config>".into(),
        source: "conus_adjacency not resolved — invoke via `ddrs run --plot` \
                 (which resolves adjacency), or set conus_adjacency/gages_adjacency \
                 explicitly".into(),
    })?;
    eprintln!("opening CONUS adjacency: {}", conus_path.display());
    let conus = ConusAdjacencyStore::open(conus_path)
        .map_err(|e| CliError::Other(Box::new(e)))?;
    let n_reaches = conus.order.len();
    eprintln!("CONUS reaches: {n_reaches}");

    // ---------- 2. Attributes + z-score stats ----------
    eprintln!("opening attributes: {}", ds.attributes.display());
    let attrs = AttributesStore::open(&ds.attributes, &head_cfg.input_var_names, &conus.order)
        .map_err(|e| CliError::Other(Box::new(e)))?;

    // Replicates the private `stats_path_from_attrs` helper in
    // `src/data/dataset.rs:570`. Kept inline to avoid widening that fn's
    // visibility just for one module.
    let stats_path = {
        let dir = ds.attributes.parent().unwrap_or_else(|| std::path::Path::new("."));
        let fname = ds
            .attributes
            .file_name()
            .map(|f| f.to_string_lossy().to_string())
            .unwrap_or_default();
        dir.join("statistics")
            .join(format!("merit_attribute_statistics_{fname}.json"))
    };
    let stats = AttrStats::open(&stats_path).map_err(|e| CliError::Other(Box::new(e)))?;
    let means = stats.means_f32(&head_cfg.input_var_names);
    let stds = stats.stds_f32(&head_cfg.input_var_names);

    // ---------- 3. Build normalized (N, F) attribute tensor ----------
    // Mirrors `MeritGagesDataset::finalize_attrs` (`src/data/dataset.rs:403`).
    let f = head_cfg.input_var_names.len();
    let mut a: Array2<f32> = Array2::zeros((f, n_reaches));
    for (out_col, comid) in conus.order.iter().enumerate() {
        if let Some(src_col) = attrs.index.position(comid) {
            for fi in 0..f {
                a[(fi, out_col)] = attrs.attrs[(fi, src_col)];
            }
        } else {
            for fi in 0..f {
                a[(fi, out_col)] = f32::NAN;
            }
        }
    }
    fill_nans(a.view_mut(), &attrs.row_means);
    for fi in 0..f {
        let m = means[fi];
        let s = stds[fi];
        for col in 0..n_reaches {
            a[(fi, col)] = (a[(fi, col)] - m) / s;
        }
    }
    let attrs_nf: Array2<f32> = a.reversed_axes().into_owned(); // (N, F)

    // ---------- 4. Load trained KAN head ----------
    <I as Backend>::seed(device, cfg.seed);

    let head_template: KanHead<I> = KanHeadConfig::new(
        head_cfg.input_var_names.clone(),
        head_cfg.learnable_parameters.clone(),
        cfg.seed,
    )
    .with_hidden_size(head_cfg.hidden_size)
    .with_num_hidden_layers(head_cfg.num_hidden_layers)
    .with_grid(head_cfg.grid)
    .with_k(head_cfg.k)
    .init::<I>(device);
    eprintln!("loading checkpoint: {}.mpk", checkpoint.display());
    let head = load_kan_head::<I>(checkpoint, head_template, device)
        .map_err(|e| CliError::Other(Box::new(e)))?;

    // ---------- 5. Forward in batches, denormalize ----------
    let log_space = &cfg.params.log_space_parameters;
    let learnable = &head_cfg.learnable_parameters;
    let is_log = |k: &str| log_space.iter().any(|s| s == k);
    let learn_has = |k: &str| learnable.iter().any(|s| s == k);
    let p_default = *cfg.params.defaults.get("p_spatial").unwrap_or(&21.0);

    let mut n_phys: Vec<f32> = Vec::with_capacity(n_reaches);
    let mut q_phys: Vec<f32> = Vec::with_capacity(n_reaches);
    let mut p_phys: Vec<f32> = Vec::with_capacity(n_reaches);

    for start in (0..n_reaches).step_by(batch_size) {
        let end = (start + batch_size).min(n_reaches);
        let rows = end - start;

        let chunk: Vec<f32> = attrs_nf.slice(s![start..end, ..]).iter().copied().collect();
        let input: Tensor<I, 2> =
            Tensor::<I, 1>::from_floats(chunk.as_slice(), device).reshape([rows, f]);

        let raw = head.forward(input);

        let n_d = denormalize(raw["n"].clone(), cfg.params.parameter_ranges.n, is_log("n"));
        let q_d = denormalize(
            raw["q_spatial"].clone(),
            cfg.params.parameter_ranges.q_spatial,
            is_log("q_spatial"),
        );
        n_phys.extend(n_d.into_data().to_vec::<f32>().unwrap());
        q_phys.extend(q_d.into_data().to_vec::<f32>().unwrap());

        if learn_has("p_spatial") {
            let p_d = denormalize(
                raw["p_spatial"].clone(),
                cfg.params.parameter_ranges.p_spatial,
                is_log("p_spatial"),
            );
            p_phys.extend(p_d.into_data().to_vec::<f32>().unwrap());
        } else {
            p_phys.extend(std::iter::repeat(p_default).take(rows));
        }

        eprintln!("  batch {start:>7}..{end:<7}  ({} reaches)", end - start);
    }

    // ---------- 6. Write NetCDF4 ----------
    let slope_lb = cfg.params.attribute_minimums.slope;
    let comids_i64: Vec<i64> = conus.order.iter().map(|c| c.0).collect();
    let slope_clamped: Vec<f32> = conus.slope.iter().map(|&s| s.max(slope_lb)).collect();

    write_netcdf(
        output_path,
        &comids_i64,
        &n_phys,
        &q_phys,
        &p_phys,
        &slope_clamped,
        &checkpoint.display().to_string(),
    )
    .map_err(CliError::Other)?;

    Ok(n_reaches)
}

/// Same as [`dump`], but uses a freshly-initialized `KanHead` (no checkpoint).
/// Used by `examples/dump_init_params.rs` to write the t=0 init distribution
/// over all CONUS reaches for DDR↔DDRS parity comparison (Task 11).
///
/// Body duplicates `dump` modulo the `load_kan_head` call — refactor later if
/// a third caller appears.
pub fn dump_init<I>(
    cfg: &Config,
    output_path: &Path,
    batch_size: usize,
    device: &<I as BackendTypes>::Device,
) -> Result<usize, CliError>
where
    I: Backend + BackendTypes,
{
    let ds = cfg.data_sources.as_ref().expect("data_sources section required");
    let head_cfg = cfg
        .kan_head
        .as_ref()
        .expect("kan_head section required for init dump");

    // ---------- 1. CONUS topology: COMID order + per-reach slope ----------
    // Defensive: callers route through `ddrs run`/`plan`, which materialize the
    // resolved adjacency paths into the config before dump_init runs.
    let conus_path = ds.conus_adjacency.as_ref().ok_or_else(|| CliError::ConfigInvalid {
        path: "<config>".into(),
        source: "conus_adjacency not resolved — invoke via `ddrs run --plot` \
                 (which resolves adjacency), or set conus_adjacency/gages_adjacency \
                 explicitly".into(),
    })?;
    eprintln!("opening CONUS adjacency: {}", conus_path.display());
    let conus = ConusAdjacencyStore::open(conus_path)
        .map_err(|e| CliError::Other(Box::new(e)))?;
    let n_reaches = conus.order.len();
    eprintln!("CONUS reaches: {n_reaches}");

    // ---------- 2. Attributes + z-score stats ----------
    eprintln!("opening attributes: {}", ds.attributes.display());
    let attrs = AttributesStore::open(&ds.attributes, &head_cfg.input_var_names, &conus.order)
        .map_err(|e| CliError::Other(Box::new(e)))?;

    let stats_path = {
        let dir = ds.attributes.parent().unwrap_or_else(|| std::path::Path::new("."));
        let fname = ds
            .attributes
            .file_name()
            .map(|f| f.to_string_lossy().to_string())
            .unwrap_or_default();
        dir.join("statistics")
            .join(format!("merit_attribute_statistics_{fname}.json"))
    };
    let stats = AttrStats::open(&stats_path).map_err(|e| CliError::Other(Box::new(e)))?;
    let means = stats.means_f32(&head_cfg.input_var_names);
    let stds = stats.stds_f32(&head_cfg.input_var_names);

    // ---------- 3. Build normalized (N, F) attribute tensor ----------
    let f = head_cfg.input_var_names.len();
    let mut a: Array2<f32> = Array2::zeros((f, n_reaches));
    for (out_col, comid) in conus.order.iter().enumerate() {
        if let Some(src_col) = attrs.index.position(comid) {
            for fi in 0..f {
                a[(fi, out_col)] = attrs.attrs[(fi, src_col)];
            }
        } else {
            for fi in 0..f {
                a[(fi, out_col)] = f32::NAN;
            }
        }
    }
    fill_nans(a.view_mut(), &attrs.row_means);
    for fi in 0..f {
        let m = means[fi];
        let s = stds[fi];
        for col in 0..n_reaches {
            a[(fi, col)] = (a[(fi, col)] - m) / s;
        }
    }
    let attrs_nf: Array2<f32> = a.reversed_axes().into_owned(); // (N, F)

    // ---------- 4. Build fresh KAN head (no checkpoint load) ----------
    <I as Backend>::seed(device, cfg.seed);

    let head: KanHead<I> = KanHeadConfig::new(
        head_cfg.input_var_names.clone(),
        head_cfg.learnable_parameters.clone(),
        cfg.seed,
    )
    .with_hidden_size(head_cfg.hidden_size)
    .with_num_hidden_layers(head_cfg.num_hidden_layers)
    .with_grid(head_cfg.grid)
    .with_k(head_cfg.k)
    .init::<I>(device);

    // ---------- 5. Forward in batches, denormalize ----------
    let log_space = &cfg.params.log_space_parameters;
    let learnable = &head_cfg.learnable_parameters;
    let is_log = |k: &str| log_space.iter().any(|s| s == k);
    let learn_has = |k: &str| learnable.iter().any(|s| s == k);
    let p_default = *cfg.params.defaults.get("p_spatial").unwrap_or(&21.0);

    let mut n_phys: Vec<f32> = Vec::with_capacity(n_reaches);
    let mut q_phys: Vec<f32> = Vec::with_capacity(n_reaches);
    let mut p_phys: Vec<f32> = Vec::with_capacity(n_reaches);

    for start in (0..n_reaches).step_by(batch_size) {
        let end = (start + batch_size).min(n_reaches);
        let rows = end - start;

        let chunk: Vec<f32> = attrs_nf.slice(s![start..end, ..]).iter().copied().collect();
        let input: Tensor<I, 2> =
            Tensor::<I, 1>::from_floats(chunk.as_slice(), device).reshape([rows, f]);

        let raw = head.forward(input);

        let n_d = denormalize(raw["n"].clone(), cfg.params.parameter_ranges.n, is_log("n"));
        let q_d = denormalize(
            raw["q_spatial"].clone(),
            cfg.params.parameter_ranges.q_spatial,
            is_log("q_spatial"),
        );
        n_phys.extend(n_d.into_data().to_vec::<f32>().unwrap());
        q_phys.extend(q_d.into_data().to_vec::<f32>().unwrap());

        if learn_has("p_spatial") {
            let p_d = denormalize(
                raw["p_spatial"].clone(),
                cfg.params.parameter_ranges.p_spatial,
                is_log("p_spatial"),
            );
            p_phys.extend(p_d.into_data().to_vec::<f32>().unwrap());
        } else {
            p_phys.extend(std::iter::repeat(p_default).take(rows));
        }

        eprintln!("  batch {start:>7}..{end:<7}  ({} reaches)", end - start);
    }

    // ---------- 6. Write NetCDF4 ----------
    let slope_lb = cfg.params.attribute_minimums.slope;
    let comids_i64: Vec<i64> = conus.order.iter().map(|c| c.0).collect();
    let slope_clamped: Vec<f32> = conus.slope.iter().map(|&s| s.max(slope_lb)).collect();

    write_netcdf(
        output_path,
        &comids_i64,
        &n_phys,
        &q_phys,
        &p_phys,
        &slope_clamped,
        "init-only (no checkpoint)",
    )
    .map_err(CliError::Other)?;

    Ok(n_reaches)
}

/// Write a NetCDF4 file with the COMID-keyed KAN parameter schema. Each var
/// carries a `long_name` + `units` attribute so xarray-based plotting code
/// (e.g. DDR's `plot_parameter_map.ipynb`) gets self-describing axes.
fn write_netcdf(
    path: &Path,
    comids: &[i64],
    n_vals: &[f32],
    q_vals: &[f32],
    p_vals: &[f32],
    slope: &[f32],
    checkpoint: &str,
) -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
    let mut file = netcdf::create(path)?;

    file.add_attribute("checkpoint", checkpoint)?;
    file.add_attribute("ddrs_version", env!("CARGO_PKG_VERSION"))?;
    file.add_attribute("n_reaches", comids.len() as i64)?;
    file.add_attribute(
        "note",
        "KAN parameters only; geometry statistics omitted (non-distributable streamflow inputs)",
    )?;

    file.add_dimension("COMID", comids.len())?;

    let mut v = file.add_variable::<i64>("COMID", &["COMID"])?;
    v.put_values(comids, ..)?;
    v.put_attribute("long_name", "MERIT reach identifier")?;

    let mut v = file.add_variable::<f32>("n", &["COMID"])?;
    v.put_values(n_vals, ..)?;
    v.put_attribute("long_name", "Manning's roughness coefficient")?;
    v.put_attribute("units", "s/m^(1/3)")?;

    let mut v = file.add_variable::<f32>("q_spatial", &["COMID"])?;
    v.put_values(q_vals, ..)?;
    v.put_attribute("long_name", "spatial discharge scaling exponent")?;
    v.put_attribute("units", "dimensionless")?;

    let mut v = file.add_variable::<f32>("p_spatial", &["COMID"])?;
    v.put_values(p_vals, ..)?;
    v.put_attribute("long_name", "spatial width-to-depth ratio")?;
    v.put_attribute("units", "dimensionless")?;

    let mut v = file.add_variable::<f32>("slope", &["COMID"])?;
    v.put_values(slope, ..)?;
    v.put_attribute("long_name", "channel slope (clamped to attribute_minimums.slope)")?;
    v.put_attribute("units", "m/m")?;

    Ok(())
}

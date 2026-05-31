//! Dump per-reach KAN parameter predictions for a trained ddrs checkpoint.
//!
//! Mirrors the inference path of `~/projects/ddr/scripts/geometry_predictor.py`
//! restricted to KAN-head parameters (no geometry-statistics water-year loop).
//! Loads a trained `KanHead` `.mpk` checkpoint, runs the head on the CONUS
//! MERIT attribute matrix, denormalizes the outputs into physical units, and
//! writes a NetCDF4 file keyed by COMID — schema-compatible with the KAN
//! subset of DDR's `merit_geometry_predictions.nc`. The
//! `{depth,top_width,discharge}_{min,max,median,mean}` geometry-statistics
//! variables are intentionally **omitted** (they require a water-year forward
//! routing loop over distributable streamflow, which the project cannot ship).
//!
//! Architecture parity: the head template is constructed from the YAML's
//! `kan_head` section using the **same** `KanHeadConfig::init` path that
//! training uses (`src/bin/train.rs` and `src/bin/eval.rs`). The layering
//! order — `Linear(F, H) → KanLayer(H, H) × num_hidden_layers → Linear(H, P)
//! → Sigmoid` with the DDR-Python same-seed quirk for every inner KanLayer —
//! is enforced by `KanHeadConfig::init`. If the checkpoint was trained with a
//! different hidden_size/num_hidden_layers/grid/k, `load_kan_head` will fail
//! at `load_record` time.
//!
//! Usage:
//!   cargo run --release --bin dump_parameters -- \
//!       --config config/merit_training.yaml \
//!       --checkpoint output/saved_models/epoch_5_mb_0 \
//!       --output output/kan_parameters.nc
//!
//! The checkpoint path is the base path (no `.mpk` suffix — `CompactRecorder`
//! appends it). NetCDF schema: dim `COMID`; vars `COMID (i64)`, `n (f32)`,
//! `q_spatial (f32)`, `p_spatial (f32)`, `slope (f32)`, all on `[COMID]`.
//! DDR's `examples/merit/plot_parameter_map.ipynb` works as-is:
//!   `ds = xr.open_dataset("output/kan_parameters.nc")`

use std::path::PathBuf;

use burn::tensor::backend::{Backend, BackendTypes};
use burn::tensor::Tensor;
use burn_cuda::Cuda;
use clap::Parser;
use ndarray::{s, Array2};

use ddrs::config::{Config, ConfigMode};
use ddrs::data::{
    fill_nans, AttrStats, AttributesStore, ConusAdjacencyStore,
};
use ddrs::nn::{KanHead, KanHeadConfig};
use ddrs::routing::denormalize;
use ddrs::training::load_kan_head;

#[derive(Parser, Debug)]
#[command(name = "dump_parameters", about = "Dump trained KAN parameter predictions per CONUS reach")]
struct Cli {
    /// Training YAML (same one used at train time — supplies kan_head section,
    /// parameter_ranges, log_space_parameters, and data_sources.attributes).
    #[arg(long)]
    config: PathBuf,

    /// Trained KAN checkpoint base path (no `.mpk` suffix).
    #[arg(long)]
    checkpoint: PathBuf,

    /// Output CSV path.
    #[arg(long)]
    output: PathBuf,

    /// Reaches per KAN forward call. 50_000 matches DDR's
    /// `geometry_predictor.py` and fits comfortably on a 24 GB GPU.
    #[arg(long, default_value_t = 50_000)]
    batch_size: usize,
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let cli = Cli::parse();

    // Testing mode mirrors eval.rs — applies any test overlays from the YAML
    // (the head architecture and data paths are mode-invariant, so this is a
    // no-op for our purposes, but keeps behavior aligned with eval/test runs).
    let cfg = Config::from_yaml_file_with_mode(&cli.config, ConfigMode::Testing)?;

    let ds = cfg.data_sources.as_ref().expect("data_sources section required");
    let head_cfg = cfg
        .kan_head
        .as_ref()
        .expect("kan_head section required for trained-KAN inference");

    // ---------- 1. CONUS topology: COMID order + per-reach slope ----------
    eprintln!("opening CONUS adjacency: {}", ds.conus_adjacency.display());
    let conus = ConusAdjacencyStore::open(&ds.conus_adjacency)?;
    let n_reaches = conus.order.len();
    eprintln!("CONUS reaches: {n_reaches}");

    // ---------- 2. Attributes + z-score stats ----------
    eprintln!("opening attributes: {}", ds.attributes.display());
    let attrs = AttributesStore::open(&ds.attributes, &head_cfg.input_var_names, &conus.order)?;

    // Replicates the private `stats_path_from_attrs` helper in
    // `src/data/dataset.rs:570`. Kept inline to avoid widening that fn's
    // visibility just for one bin.
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
    let stats = AttrStats::open(&stats_path)?;
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
    type I = Cuda<f32, i32>;
    let device = <I as BackendTypes>::Device::default();
    <I as Backend>::seed(&device, cfg.seed);

    let head_template: KanHead<I> = KanHeadConfig::new(
        head_cfg.input_var_names.clone(),
        head_cfg.learnable_parameters.clone(),
        cfg.seed,
    )
    .with_hidden_size(head_cfg.hidden_size)
    .with_num_hidden_layers(head_cfg.num_hidden_layers)
    .with_grid(head_cfg.grid)
    .with_k(head_cfg.k)
    .init::<I>(&device);
    eprintln!("loading checkpoint: {}.mpk", cli.checkpoint.display());
    let head = load_kan_head::<I>(&cli.checkpoint, head_template, &device)?;

    // ---------- 5. Forward in batches, denormalize ----------
    let log_space = &cfg.params.log_space_parameters;
    let learnable = &head_cfg.learnable_parameters;
    let is_log = |k: &str| log_space.iter().any(|s| s == k);
    let learn_has = |k: &str| learnable.iter().any(|s| s == k);
    let p_default = *cfg.params.defaults.get("p_spatial").unwrap_or(&21.0);

    let mut n_phys: Vec<f32> = Vec::with_capacity(n_reaches);
    let mut q_phys: Vec<f32> = Vec::with_capacity(n_reaches);
    let mut p_phys: Vec<f32> = Vec::with_capacity(n_reaches);

    for start in (0..n_reaches).step_by(cli.batch_size) {
        let end = (start + cli.batch_size).min(n_reaches);
        let rows = end - start;

        let chunk: Vec<f32> = attrs_nf.slice(s![start..end, ..]).iter().copied().collect();
        let input: Tensor<I, 2> =
            Tensor::<I, 1>::from_floats(chunk.as_slice(), &device).reshape([rows, f]);

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

    // ---------- 6. Write NetCDF4 (dim: COMID; vars: COMID, n, q_spatial,
    //              p_spatial, slope — schema-compatible with DDR's
    //              merit_geometry_predictions.nc subset).
    let slope_lb = cfg.params.attribute_minimums.slope;
    let comids_i64: Vec<i64> = conus.order.iter().map(|c| c.0).collect();
    let slope_clamped: Vec<f32> = conus.slope.iter().map(|&s| s.max(slope_lb)).collect();

    write_netcdf(
        &cli.output,
        &comids_i64,
        &n_phys,
        &q_phys,
        &p_phys,
        &slope_clamped,
        &cli.checkpoint.display().to_string(),
        &cli.config.display().to_string(),
    )?;

    println!("wrote {n_reaches} reaches → {}", cli.output.display());
    println!(
        "  n         min={:.4}  max={:.4}",
        min(&n_phys),
        max(&n_phys)
    );
    println!(
        "  q_spatial min={:.4}  max={:.4}",
        min(&q_phys),
        max(&q_phys)
    );
    println!(
        "  p_spatial min={:.4}  max={:.4}",
        min(&p_phys),
        max(&p_phys)
    );
    Ok(())
}

fn min(v: &[f32]) -> f32 {
    v.iter().copied().fold(f32::INFINITY, f32::min)
}

fn max(v: &[f32]) -> f32 {
    v.iter().copied().fold(f32::NEG_INFINITY, f32::max)
}

/// Write a NetCDF4 file with the COMID-keyed KAN parameter schema. Each var
/// carries a `long_name` + `units` attribute so xarray-based plotting code
/// (e.g. DDR's `plot_parameter_map.ipynb`) gets self-describing axes.
#[allow(clippy::too_many_arguments)]
fn write_netcdf(
    path: &std::path::Path,
    comids: &[i64],
    n_vals: &[f32],
    q_vals: &[f32],
    p_vals: &[f32],
    slope: &[f32],
    checkpoint: &str,
    config: &str,
) -> Result<(), Box<dyn std::error::Error>> {
    let mut file = netcdf::create(path)?;

    file.add_attribute("checkpoint", checkpoint)?;
    file.add_attribute("config", config)?;
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

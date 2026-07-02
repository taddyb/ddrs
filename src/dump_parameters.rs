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

    let head_template: KanHead<I> = crate::config::kan_config(head_cfg, cfg.seed).init::<I>(device);
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
    let mut x_phys: Vec<f32> = Vec::with_capacity(n_reaches);
    // Leakance params (populated only with use_leakance + the keys learnable).
    let dump_leakance = cfg.params.use_leakance
        && learn_has("K_D")
        && learn_has("d_gw")
        && learn_has("leakance_factor");
    let mut kd_phys: Vec<f32> = Vec::new();
    let mut dgw_phys: Vec<f32> = Vec::new();
    let mut lfac_phys: Vec<f32> = Vec::new();

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

        // Muskingum X: denormalize when learnable, else the routing constant 0.3.
        if learn_has("x_storage") {
            let x_d = denormalize(
                raw["x_storage"].clone(),
                cfg.params.parameter_ranges.x_storage,
                is_log("x_storage"),
            );
            x_phys.extend(x_d.into_data().to_vec::<f32>().unwrap());
        } else {
            x_phys.extend(std::iter::repeat(0.3_f32).take(rows));
        }

        // Leakance — the identifiability evidence. K_D/leakance_factor pinned at
        // their lower bound ⇒ the sub-0.01-m³/s collapse that caused DDR's
        // revert; non-trivial values ⇒ the term is identifiable.
        if dump_leakance {
            let kd_d = denormalize(raw["K_D"].clone(), cfg.params.parameter_ranges.k_d, is_log("K_D"));
            let dgw_d =
                denormalize(raw["d_gw"].clone(), cfg.params.parameter_ranges.d_gw, is_log("d_gw"));
            let lfac_d = denormalize(
                raw["leakance_factor"].clone(),
                cfg.params.parameter_ranges.leakance_factor,
                is_log("leakance_factor"),
            );
            kd_phys.extend(kd_d.into_data().to_vec::<f32>().unwrap());
            dgw_phys.extend(dgw_d.into_data().to_vec::<f32>().unwrap());
            lfac_phys.extend(lfac_d.into_data().to_vec::<f32>().unwrap());
        }

        eprintln!("  batch {start:>7}..{end:<7}  ({} reaches)", end - start);
    }

    // Summarize the learned leakance distribution — the decisive identifiability
    // check (the revert criterion: values must clear ~0.01, not collapse to ~0).
    if dump_leakance {
        let summarize = |label: &str, v: &[f32], lo: f32, hi: f32| {
            let mut s = v.to_vec();
            s.sort_by(|a, b| a.partial_cmp(b).unwrap());
            let pct = |p: f64| s[((p * (s.len() - 1) as f64).round() as usize).min(s.len() - 1)];
            let at_floor = v.iter().filter(|&&x| x <= lo * 1.001).count() as f64 / v.len() as f64;
            let at_ceil = v.iter().filter(|&&x| x >= hi * 0.999).count() as f64 / v.len() as f64;
            eprintln!(
                "learned {label}: min={:.4e} p10={:.4e} median={:.4e} p90={:.4e} max={:.4e}  \
                 frac@floor={:.1}%  frac@ceil={:.1}%",
                s[0], pct(0.10), pct(0.50), pct(0.90), s[s.len() - 1],
                at_floor * 100.0, at_ceil * 100.0
            );
        };
        let r = &cfg.params.parameter_ranges;
        summarize("K_D (1/s)", &kd_phys, r.k_d[0], r.k_d[1]);
        summarize("d_gw (m)", &dgw_phys, r.d_gw[0], r.d_gw[1]);
        summarize("leakance_factor", &lfac_phys, r.leakance_factor[0], r.leakance_factor[1]);
    }

    // Summarize the learned X distribution — the decisive check on whether the
    // routing actually moved its attenuation knob off the sigmoid-init (~0.25).
    if learn_has("x_storage") {
        let mut xs = x_phys.clone();
        xs.sort_by(|a, b| a.partial_cmp(b).unwrap());
        let pct = |p: f64| xs[((p * (xs.len() - 1) as f64).round() as usize).min(xs.len() - 1)];
        let mean = x_phys.iter().sum::<f32>() / x_phys.len() as f32;
        let near_lo = x_phys.iter().filter(|&&x| x < 0.05).count() as f64 / x_phys.len() as f64;
        let near_hi = x_phys.iter().filter(|&&x| x > 0.45).count() as f64 / x_phys.len() as f64;
        eprintln!(
            "learned X over {} reaches: min={:.4} p10={:.4} median={:.4} mean={:.4} p90={:.4} max={:.4}",
            xs.len(), xs[0], pct(0.10), pct(0.50), mean, pct(0.90), xs[xs.len() - 1]
        );
        eprintln!(
            "  range [0,0.5]; sigmoid-init ≈ 0.25.  fraction X<0.05 (max attenuation)={:.1}%  X>0.45 (pure lag)={:.1}%",
            near_lo * 100.0, near_hi * 100.0
        );
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
        &x_phys,
        &slope_clamped,
        &checkpoint.display().to_string(),
        if dump_leakance {
            Some((kd_phys.as_slice(), dgw_phys.as_slice(), lfac_phys.as_slice()))
        } else {
            None
        },
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
    let mut x_phys: Vec<f32> = Vec::with_capacity(n_reaches);
    // Leakance params (populated only with use_leakance + the keys learnable).
    let dump_leakance = cfg.params.use_leakance
        && learn_has("K_D")
        && learn_has("d_gw")
        && learn_has("leakance_factor");
    let mut kd_phys: Vec<f32> = Vec::new();
    let mut dgw_phys: Vec<f32> = Vec::new();
    let mut lfac_phys: Vec<f32> = Vec::new();

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

        // Muskingum X: denormalize when learnable, else the routing constant 0.3.
        if learn_has("x_storage") {
            let x_d = denormalize(
                raw["x_storage"].clone(),
                cfg.params.parameter_ranges.x_storage,
                is_log("x_storage"),
            );
            x_phys.extend(x_d.into_data().to_vec::<f32>().unwrap());
        } else {
            x_phys.extend(std::iter::repeat(0.3_f32).take(rows));
        }

        // Leakance — the identifiability evidence. K_D/leakance_factor pinned at
        // their lower bound ⇒ the sub-0.01-m³/s collapse that caused DDR's
        // revert; non-trivial values ⇒ the term is identifiable.
        if dump_leakance {
            let kd_d = denormalize(raw["K_D"].clone(), cfg.params.parameter_ranges.k_d, is_log("K_D"));
            let dgw_d =
                denormalize(raw["d_gw"].clone(), cfg.params.parameter_ranges.d_gw, is_log("d_gw"));
            let lfac_d = denormalize(
                raw["leakance_factor"].clone(),
                cfg.params.parameter_ranges.leakance_factor,
                is_log("leakance_factor"),
            );
            kd_phys.extend(kd_d.into_data().to_vec::<f32>().unwrap());
            dgw_phys.extend(dgw_d.into_data().to_vec::<f32>().unwrap());
            lfac_phys.extend(lfac_d.into_data().to_vec::<f32>().unwrap());
        }

        eprintln!("  batch {start:>7}..{end:<7}  ({} reaches)", end - start);
    }

    // Summarize the learned leakance distribution — the decisive identifiability
    // check (the revert criterion: values must clear ~0.01, not collapse to ~0).
    if dump_leakance {
        let summarize = |label: &str, v: &[f32], lo: f32, hi: f32| {
            let mut s = v.to_vec();
            s.sort_by(|a, b| a.partial_cmp(b).unwrap());
            let pct = |p: f64| s[((p * (s.len() - 1) as f64).round() as usize).min(s.len() - 1)];
            let at_floor = v.iter().filter(|&&x| x <= lo * 1.001).count() as f64 / v.len() as f64;
            let at_ceil = v.iter().filter(|&&x| x >= hi * 0.999).count() as f64 / v.len() as f64;
            eprintln!(
                "learned {label}: min={:.4e} p10={:.4e} median={:.4e} p90={:.4e} max={:.4e}  \
                 frac@floor={:.1}%  frac@ceil={:.1}%",
                s[0], pct(0.10), pct(0.50), pct(0.90), s[s.len() - 1],
                at_floor * 100.0, at_ceil * 100.0
            );
        };
        let r = &cfg.params.parameter_ranges;
        summarize("K_D (1/s)", &kd_phys, r.k_d[0], r.k_d[1]);
        summarize("d_gw (m)", &dgw_phys, r.d_gw[0], r.d_gw[1]);
        summarize("leakance_factor", &lfac_phys, r.leakance_factor[0], r.leakance_factor[1]);
    }

    // Summarize the learned X distribution — the decisive check on whether the
    // routing actually moved its attenuation knob off the sigmoid-init (~0.25).
    if learn_has("x_storage") {
        let mut xs = x_phys.clone();
        xs.sort_by(|a, b| a.partial_cmp(b).unwrap());
        let pct = |p: f64| xs[((p * (xs.len() - 1) as f64).round() as usize).min(xs.len() - 1)];
        let mean = x_phys.iter().sum::<f32>() / x_phys.len() as f32;
        let near_lo = x_phys.iter().filter(|&&x| x < 0.05).count() as f64 / x_phys.len() as f64;
        let near_hi = x_phys.iter().filter(|&&x| x > 0.45).count() as f64 / x_phys.len() as f64;
        eprintln!(
            "learned X over {} reaches: min={:.4} p10={:.4} median={:.4} mean={:.4} p90={:.4} max={:.4}",
            xs.len(), xs[0], pct(0.10), pct(0.50), mean, pct(0.90), xs[xs.len() - 1]
        );
        eprintln!(
            "  range [0,0.5]; sigmoid-init ≈ 0.25.  fraction X<0.05 (max attenuation)={:.1}%  X>0.45 (pure lag)={:.1}%",
            near_lo * 100.0, near_hi * 100.0
        );
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
        &x_phys,
        &slope_clamped,
        "init-only (no checkpoint)",
        if dump_leakance {
            Some((kd_phys.as_slice(), dgw_phys.as_slice(), lfac_phys.as_slice()))
        } else {
            None
        },
    )
    .map_err(CliError::Other)?;

    Ok(n_reaches)
}

/// Write the eval-time per-reach zeta diagnostic — the `|zeta| > 0.01 m³/s`
/// GO/NO-GO magnitude bar for the leakance experiment
/// (`scripts/leakance_subset_analysis.py::maybe_load_zeta` reads the `zeta`
/// variable from `<run_dir>/kan_parameters.nc`).
///
/// `zeta` = eval-window mean |zeta| per reach; `zeta_net` = mean signed zeta
/// (positive = losing reach); `depth_mean`, `area_z_mean`, and `q_mean` are
/// eval-window means of routed flow depth (m), plan-view wetted area (m²),
/// and routed discharge (m³/s). All live on the `COMID_eval` dimension — the
/// EVAL network (gauge-subgraph union), NOT full CONUS — so they can coexist
/// with `dump_parameters`' full-CONUS `COMID` variables in the same file:
/// when `path` exists (e.g. a prior dump into the run dir), the zeta
/// variables are APPENDED, preserving everything already there; existing
/// zeta variables of matching length are overwritten in place.
#[allow(clippy::too_many_arguments)]
pub fn write_zeta_netcdf(
    path: &Path,
    comids: &[i64],
    zeta_abs_mean: &[f32],
    zeta_net_mean: &[f32],
    depth_mean: &[f32],
    area_z_mean: &[f32],
    q_mean: &[f32],
    model_label: &str,
) -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
    let mut file = if path.exists() {
        netcdf::append(path)?
    } else {
        netcdf::create(path)?
    };

    file.add_attribute("zeta_checkpoint", model_label)?;
    file.add_attribute("zeta_ddrs_version", env!("CARGO_PKG_VERSION"))?;
    file.add_attribute(
        "zeta_note",
        "eval-time leakance zeta diagnostic; COMID_eval is the eval \
         (gauge-subgraph union) network, not full CONUS",
    )?;

    match file.dimension("COMID_eval") {
        Some(d) if d.len() != comids.len() => {
            return Err(format!(
                "{}: existing COMID_eval dimension has {} reaches but this eval \
                 routed {}; delete the file (or the stale zeta vars) and re-run",
                path.display(),
                d.len(),
                comids.len()
            )
            .into());
        }
        Some(_) => {}
        None => {
            file.add_dimension("COMID_eval", comids.len())?;
        }
    }

    if let Some(mut v) = file.variable_mut("COMID_eval") {
        v.put_values(comids, ..)?;
    } else {
        let mut v = file.add_variable::<i64>("COMID_eval", &["COMID_eval"])?;
        v.put_values(comids, ..)?;
        v.put_attribute("long_name", "MERIT reach identifier (eval network)")?;
    }

    if let Some(mut v) = file.variable_mut("zeta") {
        v.put_values(zeta_abs_mean, ..)?;
    } else {
        let mut v = file.add_variable::<f32>("zeta", &["COMID_eval"])?;
        v.put_values(zeta_abs_mean, ..)?;
        v.put_attribute("long_name", "eval-window mean |zeta| (leakance GW-SW exchange magnitude)")?;
        v.put_attribute("units", "m^3/s")?;
    }

    if let Some(mut v) = file.variable_mut("zeta_net") {
        v.put_values(zeta_net_mean, ..)?;
    } else {
        let mut v = file.add_variable::<f32>("zeta_net", &["COMID_eval"])?;
        v.put_values(zeta_net_mean, ..)?;
        v.put_attribute("long_name", "eval-window mean signed zeta (positive = losing reach)")?;
        v.put_attribute("units", "m^3/s")?;
    }

    if let Some(mut v) = file.variable_mut("depth_mean") {
        v.put_values(depth_mean, ..)?;
    } else {
        let mut v = file.add_variable::<f32>("depth_mean", &["COMID_eval"])?;
        v.put_values(depth_mean, ..)?;
        v.put_attribute("long_name", "eval-window mean routed flow depth")?;
        v.put_attribute("units", "m")?;
    }

    if let Some(mut v) = file.variable_mut("area_z_mean") {
        v.put_values(area_z_mean, ..)?;
    } else {
        let mut v = file.add_variable::<f32>("area_z_mean", &["COMID_eval"])?;
        v.put_values(area_z_mean, ..)?;
        v.put_attribute("long_name", "eval-window mean plan-view wetted area (leakance area_z)")?;
        v.put_attribute("units", "m^2")?;
    }

    if let Some(mut v) = file.variable_mut("q_mean") {
        v.put_values(q_mean, ..)?;
    } else {
        let mut v = file.add_variable::<f32>("q_mean", &["COMID_eval"])?;
        v.put_values(q_mean, ..)?;
        v.put_attribute("long_name", "eval-window mean routed discharge")?;
        v.put_attribute("units", "m^3/s")?;
    }

    Ok(())
}

/// Write a NetCDF4 file with the COMID-keyed KAN parameter schema. Each var
/// carries a `long_name` + `units` attribute so xarray-based plotting code
/// (e.g. DDR's `plot_parameter_map.ipynb`) gets self-describing axes.
#[allow(clippy::too_many_arguments)]
fn write_netcdf(
    path: &Path,
    comids: &[i64],
    n_vals: &[f32],
    q_vals: &[f32],
    p_vals: &[f32],
    x_vals: &[f32],
    slope: &[f32],
    checkpoint: &str,
    leakance: Option<(&[f32], &[f32], &[f32])>,
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

    let mut v = file.add_variable::<f32>("x_storage", &["COMID"])?;
    v.put_values(x_vals, ..)?;
    v.put_attribute("long_name", "Muskingum X storage weight (0=attenuation, 0.5=pure lag)")?;
    v.put_attribute("units", "dimensionless")?;

    let mut v = file.add_variable::<f32>("slope", &["COMID"])?;
    v.put_values(slope, ..)?;
    v.put_attribute("long_name", "channel slope (clamped to attribute_minimums.slope)")?;
    v.put_attribute("units", "m/m")?;

    // Leakance (GW–SW water-loss) learned params — only when use_leakance.
    if let Some((k_d, d_gw, leakance_factor)) = leakance {
        let mut v = file.add_variable::<f32>("K_D", &["COMID"])?;
        v.put_values(k_d, ..)?;
        v.put_attribute("long_name", "leakance hydraulic exchange rate")?;
        v.put_attribute("units", "1/s")?;

        let mut v = file.add_variable::<f32>("d_gw", &["COMID"])?;
        v.put_values(d_gw, ..)?;
        v.put_attribute("long_name", "leakance groundwater depth threshold")?;
        v.put_attribute("units", "m")?;

        let mut v = file.add_variable::<f32>("leakance_factor", &["COMID"])?;
        v.put_values(leakance_factor, ..)?;
        v.put_attribute("long_name", "leakance gating/scaling factor")?;
        v.put_attribute("units", "dimensionless")?;
    }

    Ok(())
}

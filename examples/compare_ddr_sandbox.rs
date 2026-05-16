//! Bit-comparable replay of DDR's `tests/benchmarks/test_ddr.py` sandbox routing.
//!
//! Reads fixtures dumped by `scripts/export_ddr_sandbox.py`:
//!   * `qprime_topo.csv`         — lateral inflow in topological order
//!   * `adjacency_topo.csv`      — dense adjacency, lower-triangular
//!   * `topo_order.csv`          — reach IDs in topological order
//!   * `rapid2_order.csv`        — reach IDs in RAPID2 order [10,20,30,40,50]
//!   * `ddr_discharge_rapid2.csv` — DDR's routed output, in RAPID2 order
//!
//! Then runs `MuskingumCunge` on identical inputs, reorders the output to
//! RAPID2 order, and emits:
//!   * `output/ddrs_vs_ddr.csv`  — per-reach max/mean abs diff, correlation
//!   * `output/ddrs_vs_ddr.png`  — both hydrographs overlaid

use std::fs::File;
use std::io::{BufWriter, Write};
use std::path::Path;

use burn::backend::{Autodiff, NdArray};
use burn::tensor::Tensor;
use plotters::prelude::*;

use ddrs::config::Config;
use ddrs::routing::{MuskingumCunge, RoutingInputs, SpatialParameters};
use ddrs::sparse::SparseAdjacency;

type Inner = NdArray<f32>;
type B = Autodiff<Inner>;
type D = <Inner as burn::tensor::backend::BackendTypes>::Device;

const N_REACHES: usize = 5;

fn read_matrix_csv(path: &Path, expect_rows: usize, expect_cols: usize) -> Vec<f32> {
    let s = std::fs::read_to_string(path).expect("read csv");
    let mut data = Vec::with_capacity(expect_rows * expect_cols);
    let mut rows = 0;
    for line in s.lines() {
        let line = line.trim();
        if line.is_empty() || line.starts_with('#') {
            continue;
        }
        let cols: Vec<f32> = line.split(',').map(|x| x.trim().parse().unwrap()).collect();
        assert_eq!(cols.len(), expect_cols, "wrong col count in {:?}", path);
        data.extend(cols);
        rows += 1;
    }
    assert_eq!(rows, expect_rows, "wrong row count in {:?}", path);
    data
}

fn read_int_csv(path: &Path) -> Vec<i32> {
    std::fs::read_to_string(path)
        .expect("read int csv")
        .lines()
        .map(str::trim)
        .filter(|l| !l.is_empty() && !l.starts_with('#'))
        .map(|l| l.parse().unwrap())
        .collect()
}

fn ddr_config() -> Config {
    let mut cfg = Config::default();
    cfg.params.parameter_ranges.n = [0.015, 0.25];
    cfg.params.parameter_ranges.q_spatial = [0.0, 1.0];
    cfg.params.parameter_ranges.p_spatial = [1.0, 200.0];
    cfg.params.attribute_minimums.discharge = 1e-4;
    cfg.params.attribute_minimums.slope = 1e-3;
    cfg.params.attribute_minimums.velocity = 0.01;
    cfg.params.attribute_minimums.depth = 0.01;
    cfg.params.attribute_minimums.bottom_width = 0.1; // matches DDR sandbox config
    cfg.params.log_space_parameters = vec!["n".to_string()]; // DDR sandbox: n is log-space
    cfg.params.defaults.insert("p_spatial".to_string(), 21.0);
    cfg
}

fn main() -> std::io::Result<()> {
    let fixtures = Path::new("fixtures/sandbox");

    // ---- load fixtures ----
    let topo_order = read_int_csv(&fixtures.join("topo_order.csv"));
    let rapid2_order = read_int_csv(&fixtures.join("rapid2_order.csv"));
    assert_eq!(topo_order.len(), N_REACHES);
    assert_eq!(rapid2_order.len(), N_REACHES);

    // qprime_topo.csv has shape (T, N) — must read in row order.
    let qprime_raw = std::fs::read_to_string(fixtures.join("qprime_topo.csv")).expect("read q'");
    let qprime_rows: Vec<Vec<f32>> = qprime_raw
        .lines()
        .map(str::trim)
        .filter(|l| !l.is_empty())
        .map(|l| l.split(',').map(|x| x.parse().unwrap()).collect())
        .collect();
    let n_timesteps = qprime_rows.len();
    assert!(qprime_rows.iter().all(|r| r.len() == N_REACHES));
    let qprime_flat: Vec<f32> = qprime_rows.into_iter().flatten().collect();

    let adjacency_flat = read_matrix_csv(&fixtures.join("adjacency_topo.csv"), N_REACHES, N_REACHES);

    // ddr_discharge_rapid2.csv has shape (N, T)
    let ddr_rapid2_flat = read_matrix_csv(&fixtures.join("ddr_discharge_rapid2.csv"), N_REACHES, n_timesteps);

    // ---- set up BURN tensors ----
    let device = D::default();
    let qprime: Tensor<B, 2> =
        Tensor::<B, 1>::from_floats(qprime_flat.as_slice(), &device).reshape([n_timesteps, N_REACHES]);

    let adjacency = SparseAdjacency::from_dense(
        N_REACHES,
        &adjacency_flat,
        vec![5000.0; N_REACHES],
        vec![0.001; N_REACHES],
    );
    let inputs = RoutingInputs::<Inner> {
        adjacency,
        x_storage: Tensor::ones([N_REACHES], &device) * 0.25,
    };
    let params = SpatialParameters::<Inner> {
        n: Tensor::ones([N_REACHES], &device) * 0.5,
        q_spatial: Tensor::ones([N_REACHES], &device) * 0.5,
        p_spatial: None,
    };

    // ---- run MuskingumCunge ----
    let mut mc = MuskingumCunge::<Inner>::new(ddr_config(), device.clone());
    mc.setup_inputs(inputs, qprime, params, false);
    let out = mc.forward(); // shape [N, T] in topological order
    let topo_data: Vec<f32> = out.into_data().to_vec().unwrap();

    // ---- reorder topo -> RAPID2 ----
    let rapid2_idx_in_topo: Vec<usize> = rapid2_order
        .iter()
        .map(|rid| {
            topo_order
                .iter()
                .position(|t| t == rid)
                .expect("RAPID2 id missing from topo")
        })
        .collect();
    let mut ddrs_rapid2 = vec![0.0_f32; N_REACHES * n_timesteps];
    for (r2_pos, &topo_pos) in rapid2_idx_in_topo.iter().enumerate() {
        for t in 0..n_timesteps {
            ddrs_rapid2[r2_pos * n_timesteps + t] = topo_data[topo_pos * n_timesteps + t];
        }
    }

    // ---- numerical comparison ----
    println!("Sandbox benchmark comparison: ddrs (Rust/BURN) vs DDR (PyTorch)");
    println!("  reaches:    {} (RAPID2 order: {:?})", N_REACHES, rapid2_order);
    println!("  timesteps:  {}", n_timesteps);
    println!();
    println!(
        "  {:>6}  {:>15}  {:>15}  {:>15}  {:>15}  {:>10}",
        "reach", "max_abs_diff", "mean_abs_diff", "max_rel_diff", "ddr_mean", "corr"
    );
    let mut overall_max_abs: f32 = 0.0;
    let mut overall_max_rel: f32 = 0.0;
    let mut detail = BufWriter::new(File::create("output/ddrs_vs_ddr.csv")?);
    writeln!(detail, "reach_id,max_abs_diff,mean_abs_diff,max_rel_diff,ddr_mean,ddrs_mean,corr")?;
    for (r2_pos, &rid) in rapid2_order.iter().enumerate() {
        let row_ddr = &ddr_rapid2_flat[r2_pos * n_timesteps..(r2_pos + 1) * n_timesteps];
        let row_rs = &ddrs_rapid2[r2_pos * n_timesteps..(r2_pos + 1) * n_timesteps];

        let mut max_abs = 0.0_f32;
        let mut sum_abs = 0.0_f32;
        let mut max_rel = 0.0_f32;
        for (a, b) in row_ddr.iter().zip(row_rs.iter()) {
            let d = (a - b).abs();
            max_abs = max_abs.max(d);
            sum_abs += d;
            if a.abs() > 1e-6 {
                max_rel = max_rel.max(d / a.abs());
            }
        }
        let mean_abs = sum_abs / n_timesteps as f32;
        let ddr_mean = row_ddr.iter().sum::<f32>() / n_timesteps as f32;
        let rs_mean = row_rs.iter().sum::<f32>() / n_timesteps as f32;

        // Pearson correlation
        let mut sxy = 0.0_f32;
        let mut sxx = 0.0_f32;
        let mut syy = 0.0_f32;
        for (a, b) in row_ddr.iter().zip(row_rs.iter()) {
            let da = *a - ddr_mean;
            let db = *b - rs_mean;
            sxy += da * db;
            sxx += da * da;
            syy += db * db;
        }
        let corr = if sxx > 0.0 && syy > 0.0 {
            sxy / (sxx.sqrt() * syy.sqrt())
        } else {
            1.0
        };

        println!(
            "  {:>6}  {:>15.6e}  {:>15.6e}  {:>15.6e}  {:>15.4}  {:>10.7}",
            rid, max_abs, mean_abs, max_rel, ddr_mean, corr
        );
        writeln!(
            detail,
            "{},{:.6e},{:.6e},{:.6e},{:.6},{:.6},{:.7}",
            rid, max_abs, mean_abs, max_rel, ddr_mean, rs_mean, corr
        )?;
        overall_max_abs = overall_max_abs.max(max_abs);
        overall_max_rel = overall_max_rel.max(max_rel);
    }
    detail.flush()?;
    println!();
    println!("  overall max abs diff: {:.6e} m³/s", overall_max_abs);
    println!("  overall max rel diff: {:.6e}", overall_max_rel);

    let absolute_match = overall_max_abs < 1e-3;
    println!(
        "  verdict: {}",
        if absolute_match {
            "ABSOLUTE MATCH (max abs < 1e-3 m³/s)"
        } else if overall_max_rel < 1e-2 {
            "close match (max rel < 1%) — see plot for visual confirmation"
        } else {
            "DIVERGENCE — investigate"
        }
    );

    // ---- side-by-side PNG ----
    let png_path = Path::new("output/ddrs_vs_ddr.png");
    draw_comparison_png(png_path, &ddr_rapid2_flat, &ddrs_rapid2, n_timesteps, &rapid2_order)?;
    println!();
    println!("  csv diff:  output/ddrs_vs_ddr.csv");
    println!("  png:       {}", png_path.display());

    Ok(())
}

/// Draw DDR (solid) and ddrs (dashed) on the same axes, one panel per reach.
fn draw_comparison_png(
    path: &Path,
    ddr: &[f32],
    ddrs: &[f32],
    n_timesteps: usize,
    rapid2_order: &[i32],
) -> std::io::Result<()> {
    let n = rapid2_order.len();
    let w = 1500u32;
    let h_per = 200u32;
    let h = h_per * n as u32 + 60;
    let root = BitMapBackend::new(path, (w, h)).into_drawing_area();
    root.fill(&WHITE)
        .map_err(|e| std::io::Error::new(std::io::ErrorKind::Other, e.to_string()))?;
    let title_area = root.titled(
        "DDR (solid) vs ddrs (dashed) — Sandbox Hydrograph",
        ("sans-serif", 22).into_font(),
    ).map_err(|e| std::io::Error::new(std::io::ErrorKind::Other, e.to_string()))?;

    let panels = title_area.split_evenly((n, 1));
    let palette: [RGBColor; 5] = [
        RGBColor(0x1F, 0x77, 0xB4),
        RGBColor(0xFF, 0x7F, 0x0E),
        RGBColor(0x2C, 0xA0, 0x2C),
        RGBColor(0xD6, 0x27, 0x28),
        RGBColor(0x94, 0x67, 0xBD),
    ];
    for (i, area) in panels.into_iter().enumerate() {
        let row_ddr = &ddr[i * n_timesteps..(i + 1) * n_timesteps];
        let row_rs = &ddrs[i * n_timesteps..(i + 1) * n_timesteps];
        let y_max = row_ddr.iter().chain(row_rs.iter()).cloned().fold(f32::NEG_INFINITY, f32::max);
        let y_min = row_ddr.iter().chain(row_rs.iter()).cloned().fold(f32::INFINITY, f32::min).min(0.0);
        let y_hi = y_max + 0.05 * (y_max - y_min).abs().max(1.0);

        let color = palette[i % palette.len()];
        let mut chart = ChartBuilder::on(&area)
            .caption(
                format!("Reach {}", rapid2_order[i]),
                ("sans-serif", 16).into_font(),
            )
            .margin(8)
            .x_label_area_size(28)
            .y_label_area_size(60)
            .build_cartesian_2d(0f32..(n_timesteps as f32 - 1.0), y_min..y_hi)
            .map_err(|e| std::io::Error::new(std::io::ErrorKind::Other, e.to_string()))?;
        chart
            .configure_mesh()
            .light_line_style(RGBColor(0xEE, 0xEE, 0xEE))
            .bold_line_style(RGBColor(0xCC, 0xCC, 0xCC))
            .x_desc("Hour")
            .y_desc("m³/s")
            .axis_desc_style(("sans-serif", 12))
            .label_style(("sans-serif", 11))
            .draw()
            .map_err(|e| std::io::Error::new(std::io::ErrorKind::Other, e.to_string()))?;

        let ddr_series: Vec<(f32, f32)> = row_ddr.iter().enumerate().map(|(t, v)| (t as f32, *v)).collect();
        let rs_series: Vec<(f32, f32)> = row_rs.iter().enumerate().map(|(t, v)| (t as f32, *v)).collect();

        chart
            .draw_series(LineSeries::new(ddr_series, color.stroke_width(2)))
            .map_err(|e| std::io::Error::new(std::io::ErrorKind::Other, e.to_string()))?
            .label("DDR (PyTorch)")
            .legend(move |(x, y)| PathElement::new(vec![(x, y), (x + 18, y)], color.stroke_width(2)));

        // Dashed: render as short line segments.
        let dashed_color = BLACK.mix(0.65);
        let dash_step = 4;
        let mut dash_chunks: Vec<Vec<(f32, f32)>> = Vec::new();
        let mut cur: Vec<(f32, f32)> = Vec::new();
        for (idx, pt) in rs_series.iter().enumerate() {
            if (idx / dash_step) % 2 == 0 {
                cur.push(*pt);
            } else if !cur.is_empty() {
                dash_chunks.push(std::mem::take(&mut cur));
            }
        }
        if !cur.is_empty() {
            dash_chunks.push(cur);
        }
        for chunk in dash_chunks {
            chart
                .draw_series(LineSeries::new(chunk, dashed_color.stroke_width(1)))
                .map_err(|e| std::io::Error::new(std::io::ErrorKind::Other, e.to_string()))?;
        }
        // One synthetic legend entry for the dashed line.
        chart
            .draw_series(std::iter::empty::<Circle<(f32, f32), i32>>())
            .map_err(|e| std::io::Error::new(std::io::ErrorKind::Other, e.to_string()))?
            .label("ddrs (Rust/BURN)")
            .legend(move |(x, y)| PathElement::new(vec![(x, y), (x + 18, y)], dashed_color.stroke_width(1)));

        chart
            .configure_series_labels()
            .position(SeriesLabelPosition::UpperLeft)
            .background_style(WHITE.mix(0.85))
            .border_style(RGBColor(0xDD, 0xDD, 0xDD))
            .label_font(("sans-serif", 11))
            .draw()
            .map_err(|e| std::io::Error::new(std::io::ErrorKind::Other, e.to_string()))?;
    }
    root.present()
        .map_err(|e| std::io::Error::new(std::io::ErrorKind::Other, e.to_string()))?;
    Ok(())
}

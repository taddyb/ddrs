//! Benchmark: route a diurnal-style lateral-inflow signal through a 10-reach
//! linear chain and dump the resulting hydrograph to CSV.
//!
//! Setup mirrors the unit tests' `mock_*` builders:
//!   - 10 reaches, each 1000 m long, slope 0.001, x = 0.2
//!   - 72 hourly timesteps (3 days)
//!   - lateral inflow: 5 m³/s baseline + 2·sin sweep across 4π
//!   - learned params at [0,1] midpoint (n=0.5 → 0.055 m⁻¹/³s, q_spatial=0.5)
//!
//! Output: `output/hydrograph.csv` with columns
//!   `t_hours, reach_0, reach_1, ..., reach_9`.

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

fn linear_chain_sparse(n: usize) -> SparseAdjacency {
    let mut dense = vec![0.0_f32; n * n];
    for i in 0..n - 1 {
        dense[(i + 1) * n + i] = 1.0;
    }
    SparseAdjacency::from_dense(n, &dense, vec![1000.0; n], vec![0.001; n])
}

fn lateral_inflow(t: usize, n: usize, device: &D) -> Tensor<B, 2> {
    let mut data = vec![0.0_f32; t * n];
    for ti in 0..t {
        let phase = (ti as f32) / (t.max(2) - 1) as f32 * 4.0 * std::f32::consts::PI;
        let v = (5.0 + phase.sin() * 2.0).max(0.1);
        for ri in 0..n {
            data[ti * n + ri] = v;
        }
    }
    Tensor::<B, 1>::from_floats(data.as_slice(), device).reshape([t, n])
}

fn benchmark_config() -> Config {
    let mut cfg = Config::default();
    // DDR-style defaults; same as the routing tests.
    cfg.params.parameter_ranges.n = [0.015, 0.25];
    cfg.params.parameter_ranges.q_spatial = [0.0, 1.0];
    cfg.params.parameter_ranges.p_spatial = [1.0, 200.0];
    cfg.params.attribute_minimums.discharge = 1e-4;
    cfg.params.attribute_minimums.slope = 1e-3;
    cfg.params.attribute_minimums.velocity = 0.01;
    cfg.params.attribute_minimums.depth = 0.01;
    cfg.params.attribute_minimums.bottom_width = 0.01;
    cfg.params.defaults.insert("p_spatial".to_string(), 21.0);
    cfg
}

/// Render the hydrograph to a PNG in the style of DDR's `plot_routing_hydrograph`.
///
/// `data` is the row-major `[n_reaches, n_timesteps]` discharge array.
fn draw_hydrograph_png(
    path: &Path,
    data: &[f32],
    n_reaches: usize,
    n_timesteps: usize,
) -> std::io::Result<()> {
    // 10 × 4.5 in at 150 dpi → 1500 × 675 px (matches DDR's figsize/dpi).
    let root = BitMapBackend::new(path, (1500, 675)).into_drawing_area();
    root.fill(&WHITE)
        .map_err(|e| std::io::Error::new(std::io::ErrorKind::Other, e.to_string()))?;

    let y_max = data.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    let y_min = data.iter().cloned().fold(f32::INFINITY, f32::min).min(0.0);
    // 5% headroom above the peak so the legend doesn't collide with the line.
    let y_hi = y_max + 0.05 * (y_max - y_min).abs().max(1.0);

    let mut chart = ChartBuilder::on(&root)
        .caption(
            "DDR Routed Discharge",
            ("sans-serif", 26).into_font().color(&BLACK),
        )
        .margin(20)
        .x_label_area_size(50)
        .y_label_area_size(70)
        .build_cartesian_2d(0f32..(n_timesteps as f32 - 1.0), y_min..y_hi)
        .map_err(|e| std::io::Error::new(std::io::ErrorKind::Other, e.to_string()))?;

    // Hide top/right spines by drawing only the bottom/left axes via mesh config.
    chart
        .configure_mesh()
        .light_line_style(RGBColor(0xE5, 0xE5, 0xE5))
        .bold_line_style(RGBColor(0xCC, 0xCC, 0xCC))
        .x_desc("Time (hours)")
        .y_desc("Discharge (m³/s)")
        .axis_desc_style(("sans-serif", 18))
        .label_style(("sans-serif", 14))
        .draw()
        .map_err(|e| std::io::Error::new(std::io::ErrorKind::Other, e.to_string()))?;

    // One line per reach, color-cycle through a tab10-like palette.
    let palette: [RGBColor; 10] = [
        RGBColor(0x1F, 0x77, 0xB4),
        RGBColor(0xFF, 0x7F, 0x0E),
        RGBColor(0x2C, 0xA0, 0x2C),
        RGBColor(0xD6, 0x27, 0x28),
        RGBColor(0x94, 0x67, 0xBD),
        RGBColor(0x8C, 0x56, 0x4B),
        RGBColor(0xE3, 0x77, 0xC2),
        RGBColor(0x7F, 0x7F, 0x7F),
        RGBColor(0xBC, 0xBD, 0x22),
        RGBColor(0x17, 0xBE, 0xCF),
    ];

    for r in 0..n_reaches {
        let color = palette[r % palette.len()];
        let series: Vec<(f32, f32)> = (0..n_timesteps)
            .map(|t| (t as f32, data[r * n_timesteps + t]))
            .collect();
        chart
            .draw_series(LineSeries::new(series, color.stroke_width(2)))
            .map_err(|e| std::io::Error::new(std::io::ErrorKind::Other, e.to_string()))?
            .label(format!("Reach {}", r))
            .legend(move |(x, y)| {
                PathElement::new(vec![(x, y), (x + 18, y)], color.stroke_width(2))
            });
    }

    chart
        .configure_series_labels()
        .position(SeriesLabelPosition::UpperLeft)
        .background_style(WHITE.mix(0.85))
        .border_style(RGBColor(0xDD, 0xDD, 0xDD))
        .label_font(("sans-serif", 13))
        .draw()
        .map_err(|e| std::io::Error::new(std::io::ErrorKind::Other, e.to_string()))?;

    root.present()
        .map_err(|e| std::io::Error::new(std::io::ErrorKind::Other, e.to_string()))?;
    Ok(())
}

fn main() -> std::io::Result<()> {
    let n_reaches = 10usize;
    let n_timesteps = 72usize;
    let device = D::default();

    let cfg = benchmark_config();
    let mut mc = MuskingumCunge::<Inner>::new(cfg, device.clone());

    let inputs = RoutingInputs::<Inner> {
        adjacency: linear_chain_sparse(n_reaches),
        x_storage: Tensor::<B, 1>::ones([n_reaches], &device) * 0.2,
    };
    let q_prime = lateral_inflow(n_timesteps, n_reaches, &device);
    let params = SpatialParameters::<Inner> {
        n: Tensor::<B, 1>::ones([n_reaches], &device) * 0.5,
        q_spatial: Tensor::<B, 1>::ones([n_reaches], &device) * 0.5,
        p_spatial: None,
    };

    let t_start = std::time::Instant::now();
    mc.setup_inputs(inputs, q_prime, params, false);
    let setup_ms = t_start.elapsed().as_secs_f64() * 1000.0;

    let t_forward = std::time::Instant::now();
    let hydrograph = mc.forward();
    let forward_ms = t_forward.elapsed().as_secs_f64() * 1000.0;

    let dims = hydrograph.dims();
    assert_eq!(dims, [n_reaches, n_timesteps]);
    let data: Vec<f32> = hydrograph.into_data().to_vec().unwrap();

    let out_path = Path::new("output/hydrograph.csv");
    let mut w = BufWriter::new(File::create(out_path)?);
    write!(w, "t_hours")?;
    for r in 0..n_reaches {
        write!(w, ",reach_{}", r)?;
    }
    writeln!(w)?;
    for t in 0..n_timesteps {
        write!(w, "{}", t)?;
        for r in 0..n_reaches {
            // data laid out [n_reaches, n_timesteps], row-major: idx = r * T + t
            write!(w, ",{:.6}", data[r * n_timesteps + t])?;
        }
        writeln!(w)?;
    }
    w.flush()?;

    // PNG hydrograph — styled to match `plot_routing_hydrograph` in
    // ~/projects/ddr/src/ddr/validation/plots.py: 10×4.5 in figure at 150 dpi
    // (→ 1500×675 px), white background, one line per reach, top/right spines
    // hidden, legend, "DDR Routed Discharge" title, m³/s y-label.
    let png_path = Path::new("output/hydrograph.png");
    draw_hydrograph_png(png_path, &data, n_reaches, n_timesteps)?;

    // Quick stats per reach for terminal feedback.
    println!("Muskingum-Cunge benchmark");
    println!("  reaches:    {}", n_reaches);
    println!("  timesteps:  {} ({}h)", n_timesteps, n_timesteps);
    println!("  setup:      {:.2} ms", setup_ms);
    println!("  forward:    {:.2} ms ({:.1} steps/s)",
        forward_ms, (n_timesteps as f64 - 1.0) / (forward_ms / 1000.0));
    println!("  csv:        {}", out_path.display());
    println!("  png:        {}", png_path.display());
    println!();
    println!("  per-reach Q (min / mean / max, m³/s):");
    for r in 0..n_reaches {
        let row = &data[r * n_timesteps..(r + 1) * n_timesteps];
        let min = row.iter().cloned().fold(f32::INFINITY, f32::min);
        let max = row.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
        let mean = row.iter().sum::<f32>() / row.len() as f32;
        println!("    reach {:>2}:  {:>8.3}  {:>8.3}  {:>8.3}", r, min, mean, max);
    }
    Ok(())
}

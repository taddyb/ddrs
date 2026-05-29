//! SP-10 V7a: assert CUDA-with-graphs wall-time ≤ 0.7× CPU wall-time on the
//! smoke train.
//!
//! Mirrors SP-8/SP-9's `sp8_v7_perf.rs` but injects `use_cuda_graphs: true`
//! into the CUDA YAML so the forward-graph capture path fires. Backward still
//! uses the SP-9 direct-launch path.
//!
//! Run manually:
//!   cargo test --release --test sp10_v7a_perf -- --ignored --nocapture
//!
//! Median of three runs each (first run discarded — JIT warmup).

use std::path::{Path, PathBuf};
use std::process::Command;

const MAX_MINI_BATCHES: &str = "3";
const RUNS_PER_VARIANT: usize = 4; // first is warmup; median of last 3
const RATIO_THRESHOLD: f32 = 0.7;

fn data_files_present() -> bool {
    Path::new("/home/tbindas/projects/ddr/data/merit_conus_adjacency.zarr").exists()
        && Path::new("/home/tbindas/projects/ddr/data/merit_global_attributes_v2.nc").exists()
}

fn cuda_available() -> bool {
    std::panic::catch_unwind(|| {
        type CudaInner = burn_cuda::Cuda<f32, i32>;
        type Dev = <CudaInner as burn::tensor::backend::BackendTypes>::Device;
        let _d: Dev = Default::default();
    })
    .is_ok()
}

/// Build a temp YAML with `sparse_solver` set, and optionally
/// `use_cuda_graphs: true` for the CUDA-with-graphs variant.
fn write_override_yaml(label: &str, sparse_solver: &str, use_graphs: bool) -> PathBuf {
    let base = std::fs::read_to_string("config/merit_training.yaml")
        .expect("read merit_training.yaml");
    let mut lines: Vec<String> = base
        .lines()
        .filter(|l| {
            let t = l.trim_start();
            !t.starts_with("sparse_solver:")
                && !t.starts_with("# sparse_solver:")
                && !t.starts_with("use_cuda_graphs:")
                && !t.starts_with("# use_cuda_graphs:")
        })
        .map(String::from)
        .collect();
    let params_idx = lines
        .iter()
        .position(|l| l.trim_start().starts_with("params:"))
        .expect("params: not found");
    // Insert in reverse so they appear in the natural order after params:.
    if use_graphs {
        lines.insert(params_idx + 1, "  use_cuda_graphs: true".to_string());
    }
    lines.insert(params_idx + 1, format!("  sparse_solver: {sparse_solver}"));
    let path = PathBuf::from(format!("/tmp/sp10_v7a_{label}.yaml"));
    std::fs::write(&path, lines.join("\n") + "\n").expect("write override yaml");
    path
}

fn run_train_minutes(config_path: &Path) -> f32 {
    let stem = config_path.file_stem().unwrap().to_string_lossy().into_owned();
    let ckpt_dir = format!("/tmp/sp10_v7a_ckpt_{stem}");
    let _ = std::fs::remove_dir_all(&ckpt_dir);
    let output = Command::new("cargo")
        .args([
            "run", "--release", "--bin", "train", "--",
            "--config", config_path.to_str().unwrap(),
            "--checkpoint-dir", &ckpt_dir,
            "--max-mini-batches", MAX_MINI_BATCHES,
        ])
        .output()
        .expect("spawn cargo run");
    let stdout = String::from_utf8_lossy(&output.stdout);
    let stderr = String::from_utf8_lossy(&output.stderr);
    assert!(output.status.success(),
        "train failed ({}): stdout=\n{}\nstderr=\n{}", stem, stdout, stderr);
    // Surface the "graph captured" signal so the operator can confirm the
    // capture path actually fired during this run.
    let combined = format!("{stdout}\n{stderr}");
    if combined.contains("SP-10 forward graph captured") {
        eprintln!("  [{stem}] forward graph capture confirmed");
    } else if stem.contains("cuda_graphs") {
        eprintln!("  [{stem}] WARNING: 'SP-10 forward graph captured' not seen");
    }
    for line in combined.lines() {
        if let Some(idx) = line.find("Training complete in ") {
            let tail = &line[idx + "Training complete in ".len()..];
            if let Some(min_idx) = tail.find(" min") {
                if let Ok(m) = tail[..min_idx].trim().parse::<f32>() {
                    return m;
                }
            }
        }
    }
    panic!("could not parse training minutes from output");
}

fn median(values: &mut [f32]) -> f32 {
    values.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let mid = values.len() / 2;
    if values.len() % 2 == 0 {
        (values[mid - 1] + values[mid]) / 2.0
    } else {
        values[mid]
    }
}

#[test]
#[ignore]
fn v7a_cuda_graphs_at_least_30_percent_faster_than_cpu() {
    if !data_files_present() {
        eprintln!("V7a skip: data files not present");
        return;
    }
    if !cuda_available() {
        eprintln!("V7a skip: no CUDA device");
        return;
    }

    let cpu_yaml = write_override_yaml("cpu", "cpu", false);
    let cuda_yaml = write_override_yaml("cuda_graphs", "cuda", true);

    let mut cpu_times = Vec::with_capacity(RUNS_PER_VARIANT - 1);
    let mut cuda_times = Vec::with_capacity(RUNS_PER_VARIANT - 1);

    for i in 0..RUNS_PER_VARIANT {
        let cpu_min = run_train_minutes(&cpu_yaml);
        let cuda_min = run_train_minutes(&cuda_yaml);
        eprintln!("V7a run {i}: cpu={cpu_min:.3} min, cuda+graphs={cuda_min:.3} min");
        if i > 0 {
            cpu_times.push(cpu_min);
            cuda_times.push(cuda_min);
        }
    }
    let cpu_med = median(&mut cpu_times);
    let cuda_med = median(&mut cuda_times);
    let ratio = cuda_med / cpu_med;
    eprintln!(
        "V7a: cpu_median={cpu_med:.3} min, cuda_graphs_median={cuda_med:.3} min, ratio={ratio:.3}"
    );
    assert!(
        ratio <= RATIO_THRESHOLD,
        "V7a FAILED: cuda_graphs/cpu ratio = {ratio:.3} > {RATIO_THRESHOLD}",
    );
}

//! SP-8 Task 1: gather evidence that the `scatter_kernel_t_f32_i_i32`
//! hotspot is BURN's gradient accumulation across per-timestep autograd ops
//! in `route_timestep`. Read-only investigation: runs a short training
//! invocation under nsys, parses the kernel stats, prints attribution.
//!
//! Run manually (requires CUDA + nsys + the merit data files):
//!   cargo test --release --test sp8_diagnosis -- --ignored --nocapture
//!
//! This test does NOT assert anything. It collects evidence the human reviews
//! and commits to `.claude/specs/2026-05-22-sp8-diagnosis-findings.md`.

use std::path::Path;
use std::process::Command;

const MAX_MINI_BATCHES: &str = "3";

fn data_files_present() -> bool {
    Path::new("/home/tbindas/projects/ddr/data/merit_conus_adjacency.zarr").exists()
        && Path::new("/home/tbindas/projects/ddr/data/merit_global_attributes_v2.nc").exists()
}

#[test]
#[ignore]
fn sp8_diagnosis_run() {
    if !data_files_present() {
        eprintln!("sp8_diagnosis: skip — data files not present");
        return;
    }
    if which::which("nsys").is_err() {
        eprintln!("sp8_diagnosis: skip — nsys not on PATH");
        return;
    }

    // 1. Run nsys on bin/train --max-mini-batches 3.
    let nsys_dir = std::env::var("NSYS_DIR")
        .unwrap_or_else(|_| format!("{}/nsys_out", std::env::var("HOME").unwrap()));
    std::fs::create_dir_all(&nsys_dir).expect("mkdir nsys_dir");
    let report_path = format!("{nsys_dir}/sp8_diagnosis");
    let ckpt_dir = "/tmp/sp8_diagnosis_ckpt";
    let _ = std::fs::remove_dir_all(ckpt_dir);
    let nsys_status = Command::new("nsys")
        .args([
            "profile",
            "--trace=cuda",
            "--sample=none",
            "--cpuctxsw=none",
            "--output", &report_path,
            "--force-overwrite=true",
            "target/release/train",
            "--config", "config/merit_training.yaml",
            "--checkpoint-dir", ckpt_dir,
            "--max-mini-batches", MAX_MINI_BATCHES,
        ])
        .status()
        .expect("spawn nsys");
    assert!(nsys_status.success(), "nsys profile failed");

    // 2. Run nsys stats and write to a file.
    let stats_path = format!("{nsys_dir}/sp8_diagnosis_stats.txt");
    let out = Command::new("nsys")
        .args([
            "stats",
            &format!("{report_path}.nsys-rep"),
            "--report",
            "cuda_api_sum,cuda_kern_exec_sum,cuda_gpu_mem_time_sum,cuda_gpu_kern_sum",
        ])
        .output()
        .expect("spawn nsys stats");
    std::fs::write(&stats_path, &out.stdout).expect("write stats");
    eprintln!("sp8_diagnosis: wrote {stats_path}");

    // 3. Echo the scatter rows so a human can eyeball the dominance.
    let stats = String::from_utf8_lossy(&out.stdout);
    for line in stats.lines() {
        if line.contains("scatter_kernel") {
            eprintln!("sp8_diagnosis SCATTER: {line}");
        }
    }
}

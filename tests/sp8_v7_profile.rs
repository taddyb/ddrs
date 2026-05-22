//! SP-8 V7b: assert scatter_kernel_t_f32_i_i32 below 30% of GPU compute time
//! via nsys profile.
//!
//! Run manually:
//!   cargo test --release --test sp8_v7_profile -- --ignored --nocapture

use std::path::Path;
use std::process::Command;

#[test]
#[ignore]
fn v7b_scatter_below_30_percent() {
    if which::which("nsys").is_err() {
        eprintln!("V7b skip: nsys not on PATH");
        return;
    }
    if !Path::new("/home/tbindas/projects/ddr/data/merit_conus_adjacency.zarr").exists() {
        eprintln!("V7b skip: data files not present");
        return;
    }
    let status = Command::new("bash")
        .arg("scripts/sp8_check_scatter.sh")
        .status()
        .expect("spawn sp8_check_scatter.sh");
    assert!(status.success(),
        "V7b FAILED: scatter_kernel_t_f32_i_i32 ≥ 30% of GPU time. See $NSYS_DIR/sp8_v7b_stats.txt.");
}

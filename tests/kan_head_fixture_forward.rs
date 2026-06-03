//! Layer 2 of the DDR↔DDRS KAN parity plan: assert bit-identical forward
//! pass on both backends given a fixture-loaded head.
//!
//! Build with: `cargo test --features fixtures --test kan_head_fixture_forward`

#![cfg(feature = "fixtures")]

use std::path::Path;

use burn::backend::NdArray;
use burn::tensor::{Tensor, TensorData};
use ndarray::Array2;
use ndarray_npy::NpzReader;

use ddrs::nn::{KanHead, KanHeadConfig};

type B = NdArray<f32>;

const FIXTURE: &str = "tests/fixtures/kan_head_init_seed42.npz";

fn parity_cfg() -> KanHeadConfig {
    KanHeadConfig::new(
        vec![
            "SoilGrids1km_clay", "aridity", "meanelevation", "meanP", "NDVI",
            "meanslope", "log10_uparea", "SoilGrids1km_sand", "ETPOT_Hargr", "Porosity",
        ].into_iter().map(String::from).collect(),
        vec!["n".into(), "q_spatial".into(), "p_spatial".into()],
        42,
    )
    .with_hidden_size(21)
    .with_num_hidden_layers(2)
    .with_grid(50)
    .with_k(2)
}

fn read_array2(key: &str) -> Array2<f32> {
    let mut npz = NpzReader::new(std::fs::File::open(FIXTURE).unwrap()).unwrap();
    let a: ndarray::ArrayD<f32> = npz.by_name(key).unwrap();
    a.into_dimensionality::<ndarray::Ix2>().unwrap()
}

fn read_vec(key: &str) -> Vec<f32> {
    let mut npz = NpzReader::new(std::fs::File::open(FIXTURE).unwrap()).unwrap();
    let a: ndarray::ArrayD<f32> = npz.by_name(key).unwrap();
    a.into_raw_vec_and_offset().0
}

fn max_abs_diff(got: &[f32], want: &[f32]) -> f32 {
    assert_eq!(got.len(), want.len(), "shape mismatch");
    got.iter().zip(want).map(|(a, b)| (a - b).abs()).fold(0.0_f32, f32::max)
}

#[test]
fn forward_matches_ddr_fixture_ndarray() {
    let device = Default::default();
    let cfg = parity_cfg();
    let head: KanHead<B> = KanHead::<B>::from_npz(Path::new(FIXTURE), &device, &cfg).unwrap();

    let inputs_arr = read_array2("inputs");
    let (n, f) = (inputs_arr.shape()[0], inputs_arr.shape()[1]);
    let inputs: Tensor<B, 2> = Tensor::from_data(
        TensorData::new(inputs_arr.as_slice().unwrap().to_vec(), [n, f]),
        &device,
    );

    let out = head.forward(inputs);
    for key in ["n", "q_spatial", "p_spatial"] {
        let got: Vec<f32> = out[key].clone().into_data().to_vec().unwrap();
        let want = read_vec(&format!("expected_{key}"));
        let diff = max_abs_diff(&got, &want);
        println!("{key}: max abs diff = {diff:.2e}");
        assert!(
            diff <= 1e-6,
            "{key}: max abs diff {diff} > 1e-6 on NdArray backend"
        );
    }
}

#[cfg(feature = "cuda")]
#[test]
fn forward_matches_ddr_fixture_cuda() {
    // burn::backend::Cuda needs the umbrella crate's "cuda" feature — use the
    // direct crate instead, matching the convention in tests/cusparse_ptr_spike.rs.
    type Bc = burn_cuda::Cuda<f32, i32>;

    let device = Default::default();
    let cfg = parity_cfg();
    let head: KanHead<Bc> = KanHead::<Bc>::from_npz(Path::new(FIXTURE), &device, &cfg).unwrap();

    let inputs_arr = read_array2("inputs");
    let (n, f) = (inputs_arr.shape()[0], inputs_arr.shape()[1]);
    let inputs: Tensor<Bc, 2> = Tensor::from_data(
        TensorData::new(inputs_arr.as_slice().unwrap().to_vec(), [n, f]),
        &device,
    );

    let out = head.forward(inputs);
    for key in ["n", "q_spatial", "p_spatial"] {
        let got: Vec<f32> = out[key].clone().into_data().to_vec().unwrap();
        let want = read_vec(&format!("expected_{key}"));
        let diff = max_abs_diff(&got, &want);
        println!("{key}: max abs diff = {diff:.2e}");
        assert!(
            diff <= 1e-4,
            "{key}: max abs diff {diff} > 1e-4 on CUDA backend"
        );
    }
}

//! Layer 3 of the DDR↔DDRS KAN parity plan: assert gradient parity on
//! both backends given a fixture-loaded head. Wraps everything in
//! `Autodiff` and reproduces the loss DDR's `dump_kan_fixture.py` uses:
//!     loss = out["n"].sum() + out["q_spatial"].sum() + out["p_spatial"].sum()
//!
//! Linear weight gradients (grad_input_weight, grad_output_weight) are
//! transposed from DDR's [out, in] layout to burn's [in, out] layout before
//! comparison. All other gradient tensors compare directly.

#![cfg(feature = "fixtures")]

use std::path::Path;

use burn::backend::{Autodiff, NdArray};
use burn::module::AutodiffModule;
use burn::tensor::{Tensor, TensorData};
use ndarray::Array2;
use ndarray_npy::NpzReader;

use ddrs::nn::{KanHead, KanHeadConfig};

type B = Autodiff<NdArray<f32>>;

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

/// DDR stores Linear weight + gradient in [out, in] layout.
/// burn stores Linear weight + gradient in [in, out] layout.
/// Read as a 2D array, transpose, return as a row-major Vec.
fn read_grad_weight_transposed(key: &str) -> Vec<f32> {
    let mut npz = NpzReader::new(std::fs::File::open(FIXTURE).unwrap()).unwrap();
    let arr: ndarray::ArrayD<f32> = npz.by_name(key).unwrap();
    let arr2 = arr.into_dimensionality::<ndarray::Ix2>().unwrap();
    let transposed = arr2.reversed_axes();
    transposed.as_standard_layout().iter().copied().collect()
}

fn max_abs_diff(got: &[f32], want: &[f32]) -> f32 {
    assert_eq!(got.len(), want.len(), "shape mismatch (got {}, want {})", got.len(), want.len());
    got.iter().zip(want).map(|(a, b)| (a - b).abs()).fold(0.0_f32, f32::max)
}

#[test]
fn backward_matches_ddr_fixture_ndarray() {
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
    let loss = out["n"].clone().sum()
        + out["q_spatial"].clone().sum()
        + out["p_spatial"].clone().sum();
    let grads = loss.backward();

    let tol = 1e-5_f32;

    // Linear weight gradients: transpose DDR's [out, in] to burn's [in, out].
    let linear_pairs: Vec<(&str, Vec<f32>, Vec<f32>)> = vec![
        (
            "grad_input_weight",
            head.input.weight.val().grad(&grads).unwrap().into_data().to_vec().unwrap(),
            read_grad_weight_transposed("grad_input_weight"),
        ),
        (
            "grad_input_bias",
            head.input.bias.as_ref().unwrap().val().grad(&grads).unwrap().into_data().to_vec().unwrap(),
            read_vec("grad_input_bias"),
        ),
        (
            "grad_output_weight",
            head.output.weight.val().grad(&grads).unwrap().into_data().to_vec().unwrap(),
            read_grad_weight_transposed("grad_output_weight"),
        ),
        (
            "grad_output_bias",
            head.output.bias.as_ref().unwrap().val().grad(&grads).unwrap().into_data().to_vec().unwrap(),
            read_vec("grad_output_bias"),
        ),
    ];
    for (key, got, want) in &linear_pairs {
        let diff = max_abs_diff(got, want);
        println!("{key}: max abs diff = {diff:.2e}");
        assert!(
            diff <= tol,
            "{key}: max abs grad diff {diff} > {tol}"
        );
    }

    // Inner KanLayer trainables (coef, scale_base, scale_sp — no transpose needed).
    for (b, layer) in head.hidden.iter().enumerate() {
        for (field, grad_vec) in [
            ("coef",       layer.coef.val().grad(&grads).unwrap().into_data().to_vec::<f32>().unwrap()),
            ("scale_base", layer.scale_base.val().grad(&grads).unwrap().into_data().to_vec::<f32>().unwrap()),
            ("scale_sp",   layer.scale_sp.val().grad(&grads).unwrap().into_data().to_vec::<f32>().unwrap()),
        ] {
            let key = format!("grad_block_{b}_{field}");
            let want = read_vec(&key);
            let diff = max_abs_diff(&grad_vec, &want);
            println!("{key}: max abs diff = {diff:.2e}");
            assert!(
                diff <= tol,
                "{key}: max abs grad diff {diff} > {tol}"
            );
        }
    }

    // Sanity — confirm valid module conversion works (consistency with
    // the existing kan_head.rs `_ = model_ad.valid()` check).
    let _ = head.valid();
}

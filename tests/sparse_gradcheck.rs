//! Gradcheck: our analytical CSR backward must match DDR's `TriangularSparseSolver`
//! (`~/projects/ddr/src/ddr/routing/utils.py:515`) at f32 precision.
//!
//! Fixtures are produced by `scripts/dump_solver_gradcheck.py` (must be run once
//! against DDR before this test). The test loads the same `(A, b, grad_output)`,
//! runs our solver with `Autodiff<NdArray<f32>>`, backprops `(grad_output · x).sum()`,
//! and asserts the parameter gradients match the ones DDR computed.

use std::path::{Path, PathBuf};

use burn::backend::{Autodiff, NdArray};
use burn::tensor::Tensor;

use ddrs::sparse::{triangular_csr_solve, CsrPattern};

type Inner = NdArray<f32>;
type B = Autodiff<Inner>;
type D = <Inner as burn::tensor::backend::BackendTypes>::Device;

fn fixtures_dir() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("fixtures/gradcheck")
}

fn read_f32_csv(path: &Path) -> Vec<f32> {
    std::fs::read_to_string(path)
        .unwrap_or_else(|_| panic!("cannot read {path:?} — did you run scripts/dump_solver_gradcheck.py?"))
        .lines()
        .map(|l| l.trim().parse::<f32>().expect("parse f32"))
        .collect()
}

fn read_i32_csv(path: &Path) -> Vec<i32> {
    std::fs::read_to_string(path)
        .unwrap_or_else(|_| panic!("cannot read {path:?}"))
        .lines()
        .map(|l| l.trim().parse::<i32>().expect("parse i32"))
        .collect()
}

#[test]
fn gradcheck_against_ddr_triangular_sparse_solver() {
    let dir = fixtures_dir();
    let a_values_data = read_f32_csv(&dir.join("a_values.csv"));
    let crow = read_i32_csv(&dir.join("crow.csv"));
    let col = read_i32_csv(&dir.join("col.csv"));
    let b_data = read_f32_csv(&dir.join("b.csv"));
    let x_ddr = read_f32_csv(&dir.join("x.csv"));
    let grad_out_data = read_f32_csv(&dir.join("grad_output.csv"));
    let grad_a_ddr = read_f32_csv(&dir.join("grad_a_values.csv"));
    let grad_b_ddr = read_f32_csv(&dir.join("grad_b.csv"));

    let n = b_data.len();
    let nnz = a_values_data.len();
    assert_eq!(crow.len(), n + 1, "crow shape");
    assert_eq!(col.len(), nnz, "col shape");
    assert_eq!(grad_a_ddr.len(), nnz, "grad_a shape");
    assert_eq!(grad_b_ddr.len(), n, "grad_b shape");

    let device = D::default();
    let pattern = std::sync::Arc::new(CsrPattern::from_csr_structure(n, crow, col));

    let a_values: Tensor<B, 1> =
        Tensor::<B, 1>::from_floats(a_values_data.as_slice(), &device).require_grad();
    let b: Tensor<B, 1> =
        Tensor::<B, 1>::from_floats(b_data.as_slice(), &device).require_grad();

    let x = triangular_csr_solve::<Inner>(&pattern, a_values.clone(), b.clone());

    // Forward: max abs error vs DDR's x.
    let x_ddrs: Vec<f32> = x.clone().into_data().to_vec().unwrap();
    let fwd_err = max_abs_diff(&x_ddrs, &x_ddr);
    println!("forward max abs diff (ddrs vs DDR):   {fwd_err:.3e}");
    assert!(
        fwd_err < 1e-5,
        "forward x mismatch: max abs diff {fwd_err:.3e} >= 1e-5"
    );

    // Backward: loss = (grad_output · x).sum()  →  ∂loss/∂x = grad_output (vector).
    let grad_out: Tensor<B, 1> =
        Tensor::<B, 1>::from_floats(grad_out_data.as_slice(), &device);
    let loss = (x * grad_out).sum();
    let grads = loss.backward();

    let grad_a_ddrs: Vec<f32> = a_values
        .grad(&grads)
        .expect("∂L/∂A_values present")
        .into_data()
        .to_vec()
        .unwrap();
    let grad_b_ddrs: Vec<f32> =
        b.grad(&grads).expect("∂L/∂b present").into_data().to_vec().unwrap();

    let ga_err = max_abs_diff(&grad_a_ddrs, &grad_a_ddr);
    let gb_err = max_abs_diff(&grad_b_ddrs, &grad_b_ddr);

    println!("∂L/∂A_values max abs diff (ddrs vs DDR): {ga_err:.3e}");
    println!("∂L/∂b      max abs diff (ddrs vs DDR):   {gb_err:.3e}");

    // f32 precision floor: DDR upcasts to f64 inside SciPy, then casts back; we
    // stay in f32 throughout. A few ulps of slack is expected.
    assert!(
        ga_err < 1e-5,
        "∂L/∂A_values mismatch: max abs diff {ga_err:.3e} >= 1e-5\n  ddrs[0..]: {:?}\n  ddr[0..]:  {:?}",
        &grad_a_ddrs[..grad_a_ddrs.len().min(8)],
        &grad_a_ddr[..grad_a_ddr.len().min(8)],
    );
    assert!(
        gb_err < 1e-5,
        "∂L/∂b mismatch: max abs diff {gb_err:.3e} >= 1e-5\n  ddrs: {grad_b_ddrs:?}\n  ddr:  {grad_b_ddr:?}",
    );
}

fn max_abs_diff(a: &[f32], b: &[f32]) -> f32 {
    assert_eq!(a.len(), b.len());
    a.iter()
        .zip(b)
        .map(|(x, y)| (x - y).abs())
        .fold(0.0f32, f32::max)
}

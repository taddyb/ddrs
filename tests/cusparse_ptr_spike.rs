//! Spike: verify we can borrow a Cuda tensor as a raw device pointer.
//! Skips cleanly on CPU-only hosts.

use burn::tensor::Tensor;
use burn::tensor::TensorPrimitive;
use burn::tensor::backend::BackendTypes;

#[test]
fn round_trip_via_pointer() {
    // Use burn_cuda::Cuda directly because burn's burn::backend::Cuda re-export
    // requires the "cuda" feature on the burn umbrella crate.
    type B = burn_cuda::Cuda<f32, i32>;
    type Dev = <B as BackendTypes>::Device;

    // CudaDevice::default() panics on hosts without CUDA; gate the test on a probe.
    let cuda_available = std::panic::catch_unwind(|| {
        let _device: Dev = Default::default();
    })
    .is_ok();
    if !cuda_available {
        eprintln!("skipping: no CUDA device");
        return;
    }
    let device: Dev = Default::default();

    let t = Tensor::<B, 1>::from_floats([1.0_f32, 2.0, 3.0, 4.0], &device);
    let prim = match t.clone().into_primitive() {
        TensorPrimitive::Float(p) => p,
        TensorPrimitive::QFloat(_) => panic!("expected float tensor"),
    };

    let len = ddrs::sparse::cusparse::__spike_extract_len::<B>(&prim);
    assert_eq!(len, 4, "extracted len does not match tensor length");
}

/// SP-7 Task 4b spike: verify that cubecl_stream_active returns a real non-null
/// CUstream pointer and is not a sentinel value (usize::MAX-1).
///
/// `cubecl_stream_active` calls `ComputeClient::exclusive_with_server`
/// (added in the SP-7 cubecl-runtime fork) which passes `&mut CudaServer` to
/// the closure, allowing `server.stream(StreamId::current())` to be called
/// directly on the server-bound thread.
#[test]
fn cubecl_active_stream_is_non_null() {
    type B = burn_cuda::Cuda<f32, i32>;
    type Dev = <B as BackendTypes>::Device;
    let cuda_available = std::panic::catch_unwind(|| {
        let _d: Dev = Default::default();
    })
    .is_ok();
    if !cuda_available {
        eprintln!("skipping: no CUDA device");
        return;
    }
    let device: Dev = Default::default();
    let _t = burn::tensor::Tensor::<B, 1>::from_floats([0.0_f32], &device);
    let active = ddrs::sparse::cusparse::__spike_active_stream::<B>(&device);
    eprintln!("cubecl active stream pointer: {:#x}", active as usize);
    assert!(!active.is_null(), "cubecl active stream is null");
    assert!(
        (active as usize) < usize::MAX - 16,
        "active stream looks like a sentinel: {:#x}",
        active as usize,
    );
}

/// SP-7 Task 4 spike: verify that a CubeTensor created via from_handle round-trips
/// to the correct f32 values when read back through BURN's tensor API.
#[test]
fn cube_tensor_round_trip_to_primitive() {
    type B = burn_cuda::Cuda<f32, i32>;
    type Dev = <B as BackendTypes>::Device;
    let cuda_available = std::panic::catch_unwind(|| {
        let _d: Dev = Default::default();
    })
    .is_ok();
    if !cuda_available {
        eprintln!("skipping: no CUDA device");
        return;
    }
    let device: Dev = Default::default();
    let recovered: Vec<f32> =
        ddrs::sparse::cusparse::__spike_cube_round_trip::<B>(&device, vec![10.0, 20.0, 30.0]);
    assert_eq!(recovered, vec![10.0_f32, 20.0, 30.0]);
}

/// Verify that the TypeId gate correctly returns None for a non-CUDA backend.
#[test]
fn non_cuda_backend_returns_none() {
    use burn::backend::NdArray;

    type B = NdArray<f32>;
    type Dev = <B as BackendTypes>::Device;
    let device: Dev = Default::default();

    let t = Tensor::<B, 1>::from_floats([1.0_f32, 2.0], &device);
    let prim = match t.into_primitive() {
        TensorPrimitive::Float(p) => p,
        TensorPrimitive::QFloat(_) => panic!("expected float tensor"),
    };

    let result = ddrs::sparse::cusparse::primitive_as_cuda_view::<B>(&prim);
    assert!(
        result.is_none(),
        "primitive_as_cuda_view must return None for NdArray backend"
    );
}

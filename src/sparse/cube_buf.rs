//! Device buffer helpers for the CUDA kernels.
//!
//! Device code is not generic. It works on the concrete `Cube` and converts
//! at the op boundary with `into_cube` and `from_cube`. The `#[doc(hidden)]`
//! hooks at the end are for tests. They copy to the host and count the copy.

use std::any::TypeId;

use burn::tensor::backend::Backend;
use burn::tensor::{DType, Shape, TensorData};
use burn_cubecl::ops::numeric::{empty_device_contiguous_dtype, zeros_client};
use burn_cubecl::tensor::CubeTensor;
use cubecl::cuda::CudaRuntime;

use super::cusparse::cube_tensor_to_primitive;
use super::{count_host_read, primitive_to_vec};

/// The one backend the device code runs on.
pub type Gpu = burn_cuda::Cuda<f32, i32>;
/// A device tensor on that backend.
pub type Cube = CubeTensor<CudaRuntime>;

// Compile-time proof that the backend primitive is the cube type. The unsafe
// reads in `into_cube` rely on it.
const _: fn(<Gpu as burn::tensor::backend::BackendTypes>::FloatTensorPrimitive) -> Cube = |p| p;

/// View a backend float primitive as a `Cube`. Panics unless `B` is `Cuda<f32, i32>`.
pub(crate) fn into_cube<B: Backend + 'static>(p: B::FloatTensorPrimitive) -> Cube {
    assert_eq!(
        TypeId::of::<B>(),
        TypeId::of::<Gpu>(),
        "cube_buf: backend is not Cuda<f32, i32>"
    );
    let p = std::mem::ManuallyDrop::new(p);
    // SAFETY: the TypeId check makes `B::FloatTensorPrimitive` the same type as
    // `Cube`. ManuallyDrop plus ptr::read moves ownership without a double drop.
    unsafe { std::ptr::read(&*p as *const B::FloatTensorPrimitive as *const Cube) }
}

/// Wrap a `Cube` as the backend float primitive. Panics unless `B` is `Cuda<f32, i32>`.
pub(crate) fn from_cube<B: Backend + 'static>(c: Cube) -> B::FloatTensorPrimitive {
    cube_tensor_to_primitive::<B>(c)
}

/// An uninitialised contiguous f32 buffer of `len` on the device of `like`.
// The CUDA kernels will call this. Remove the allow when they do.
#[allow(dead_code)]
pub(crate) fn empty_f32(like: &Cube, len: usize) -> Cube {
    empty_device_contiguous_dtype(
        like.client.clone(),
        like.device.clone(),
        Shape::from(vec![len]),
        DType::F32,
    )
}

/// A zero-filled f32 buffer of `len` on the device of `like`.
// The CUDA kernels will call this. Remove the allow when they do.
#[allow(dead_code)]
pub(crate) fn zeros_f32(like: &Cube, len: usize) -> Cube {
    zeros_client(
        like.client.clone(),
        like.device.clone(),
        Shape::from(vec![len]),
        DType::F32,
    )
}

/// View a slice of plain numbers as bytes.
fn as_bytes<T: Copy>(v: &[T]) -> &[u8] {
    // SAFETY: `T` is `f32` or `i32` here, with no padding. The length in bytes
    // is exact and `u8` has alignment 1.
    unsafe { std::slice::from_raw_parts(v.as_ptr() as *const u8, std::mem::size_of_val(v)) }
}

fn upload_bytes(like: &Cube, bytes: &[u8], len: usize, dtype: DType) -> Cube {
    let handle = like.client.create_from_slice(bytes);
    CubeTensor::new_contiguous(
        like.client.clone(),
        like.device.clone(),
        Shape::from(vec![len]),
        handle,
        dtype,
    )
}

/// Copy `v` to the device of `like` as f32.
// The CUDA kernels will call this. Remove the allow when they do.
#[allow(dead_code)]
pub(crate) fn upload_f32(like: &Cube, v: &[f32]) -> Cube {
    upload_bytes(like, as_bytes(v), v.len(), DType::F32)
}

/// Copy `v` to the device of `like` as i32.
pub(crate) fn upload_i32(like: &Cube, v: &[i32]) -> Cube {
    upload_bytes(like, as_bytes(v), v.len(), DType::I32)
}

/// Launch shape for one thread per element: 256 threads per cube, at least one cube.
// The CUDA kernels will call this. Remove the allow when they do.
#[allow(dead_code)]
pub(crate) fn grid(len: usize) -> (cubecl::CubeCount, cubecl::CubeDim) {
    const THREADS: u32 = 256;
    let cubes = (len as u32).div_ceil(THREADS).max(1);
    (cubecl::CubeCount::Static(cubes, 1, 1), cubecl::CubeDim::new_1d(THREADS))
}

/// Test hook: copy a host slice to the device of `device`.
#[doc(hidden)]
pub fn cube_from_host(
    v: &[f32],
    device: &<Gpu as burn::tensor::backend::BackendTypes>::Device,
) -> Cube {
    let client = <CudaRuntime as cubecl::Runtime>::client(device);
    let handle = client.create_from_slice(as_bytes(v));
    CubeTensor::new_contiguous(client, device.clone(), Shape::from(vec![v.len()]), handle, DType::F32)
}

/// Test hook: copy an f32 buffer to the host. Counts as one host read.
#[doc(hidden)]
pub fn cube_to_host(c: &Cube) -> Vec<f32> {
    primitive_to_vec::<Gpu>(from_cube::<Gpu>(c.clone()))
}

/// Test hook: copy an i32 buffer to the host. Counts as one host read.
#[doc(hidden)]
pub fn cube_to_host_i32(c: &Cube) -> Vec<i32> {
    let data: TensorData = burn_cubecl::ops::into_data_sync::<CudaRuntime>(c.clone());
    count_host_read();
    data.to_vec::<i32>().expect("expected i32 buffer")
}

/// Test wrapper of `into_cube`.
#[doc(hidden)]
pub fn into_cube_for_test<B: Backend + 'static>(p: B::FloatTensorPrimitive) -> Cube {
    into_cube::<B>(p)
}

/// Test hook: `upload_i32` for integration tests.
#[doc(hidden)]
pub fn upload_i32_for_test(like: &Cube, v: &[i32]) -> Cube {
    upload_i32(like, v)
}

#[cfg(all(test, feature = "cuda"))]
mod tests {
    use super::*;

    #[test]
    fn helpers_round_trip_on_the_gpu() {
        let like = cube_from_host(&[0.0], &Default::default());
        let up = upload_f32(&like, &[1.5, -2.0, 4.0]);
        assert_eq!(cube_to_host(&up), vec![1.5, -2.0, 4.0]);
        assert_eq!(cube_to_host(&zeros_f32(&like, 5)), vec![0.0; 5]);
        assert_eq!(empty_f32(&like, 7).meta.shape()[0], 7);
        // into_cube and from_cube are inverses on the Cuda backend.
        let back = into_cube::<Gpu>(from_cube::<Gpu>(up));
        assert_eq!(cube_to_host(&back), vec![1.5, -2.0, 4.0]);
        let cubes = |len| match grid(len).0 {
            cubecl::CubeCount::Static(x, 1, 1) => x,
            _ => panic!("grid is not 1-d static"),
        };
        assert_eq!((cubes(0), cubes(256), cubes(257)), (1, 1, 2));
    }
}

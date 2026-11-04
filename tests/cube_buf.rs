//! Device buffer helpers: round trips, the backend guard, and the tree upload.
#![cfg(feature = "cuda")]

use burn::backend::NdArray;
use burn::tensor::{Tensor, TensorPrimitive};
use ddrs::data::ConusAdjacencyStore;
use ddrs::sparse::cube_buf::{
    cube_from_host, cube_to_host, cube_to_host_i32, into_cube_for_test, Gpu,
};
use ddrs::sparse::host_reads_on_this_thread_for_test as reads;
use ddrs::sparse::tree_lu::tree_levels;
use ddrs::sparse::tree_lu_cuda::TreeDev;
use ddrs::sparse::{CsrPattern, SparseAdjacency};

fn device() -> <Gpu as burn::tensor::backend::BackendTypes>::Device {
    Default::default()
}

#[test]
fn round_trip_counts_one_host_read() {
    let v = [1.0f32, -2.5, 3.0];
    let c = cube_from_host(&v, &device());
    let r0 = reads();
    let back = cube_to_host(&c);
    assert_eq!(reads(), r0 + 1);
    assert_eq!(back.iter().map(|x| x.to_bits()).collect::<Vec<_>>(),
               v.iter().map(|x| x.to_bits()).collect::<Vec<_>>());

    let iv = [3i32, -1, 7, 0];
    let ic = ddrs::sparse::cube_buf::upload_i32_for_test(&c, &iv);
    let r1 = reads();
    let iback = cube_to_host_i32(&ic);
    assert_eq!(reads(), r1 + 1);
    assert_eq!(iback, iv);
}

#[test]
#[should_panic(expected = "backend is not Cuda<f32, i32>")]
fn into_cube_rejects_a_cpu_primitive() {
    type B = NdArray<f32>;
    let t: Tensor<B, 1> = Tensor::from_floats([1.0, 2.0], &Default::default());
    let TensorPrimitive::Float(p) = t.into_primitive() else { unreachable!() };
    let _ = into_cube_for_test::<B>(p);
}

#[test]
fn tree_dev_uploads_every_array() {
    // copied from tests/tree_lu.rs::juniata_pattern_is_a_tree
    let store = ConusAdjacencyStore::open("examples/juniata/data/juniata_conus_adjacency.zarr")
        .expect("open juniata adjacency");
    let adj = SparseAdjacency {
        n: store.n,
        rows: store.indices_0.clone(),
        cols: store.indices_1.clone(),
        values: vec![1.0; store.nnz],
        length_m: store.length_m.to_vec(),
        slope: store.slope.to_vec(),
        parent_offset: None,
    };
    let t = tree_levels(&CsrPattern::from_sparse(&adj)).unwrap();
    let like = cube_from_host(&[0.0], &device());
    let d = TreeDev::upload(&t, &like);
    assert_eq!((d.n, d.nnz, d.n_levels), (t.n, t.nnz, t.n_levels));
    assert_eq!(cube_to_host_i32(&d.crow), t.crow);
    assert_eq!(cube_to_host_i32(&d.col), t.col);
    assert_eq!(cube_to_host_i32(&d.row), t.row);
    assert_eq!(cube_to_host_i32(&d.col_diag), t.col_diag);
    assert_eq!(cube_to_host_i32(&d.down), t.down);
    assert_eq!(cube_to_host_i32(&d.down_slot), t.down_slot);
    assert_eq!(cube_to_host_i32(&d.lvl_ptr), t.lvl_ptr);
    assert_eq!(cube_to_host_i32(&d.lvl_idx), t.lvl_idx);
}

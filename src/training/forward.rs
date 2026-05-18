//! One forward pass — direct-param path (verification) and MLP path
//! (training). Plus the scatter_add-by-group helper that turns the MC
//! engine's `(N, T)` output into per-gauge `(G, T)` via the
//! `outflow_idx`-derived `flat_indices` + `group_ids`.

use burn::tensor::{backend::Backend, IndexingUpdateOp, Int, Tensor};

/// Gather + grouped sum: `output[g, t] = sum_{k : group_ids[k] == g} runoff[flat_indices[k], t]`.
///
/// Mirrors DDR `~/projects/ddr/src/ddr/routing/mmc.py:401-410`. Used to
/// extract per-gauge predictions from the engine's all-segments output.
pub fn scatter_add_by_group<B: Backend>(
    runoff: Tensor<B, 2>,             // (N, T)
    flat_indices: Tensor<B, 1, Int>,  // (K,)
    group_ids: Tensor<B, 1, Int>,     // (K,)
    num_gauges: usize,
) -> Tensor<B, 2> {
    // 1. Gather rows: (K, T).
    let gathered = runoff.select(0, flat_indices);
    let [k, t] = gathered.dims();

    // 2. Expand group_ids from (K,) to (K, T) so scatter indices match values shape.
    let group_2d = group_ids.unsqueeze_dim::<2>(1).expand([k, t]);

    // 3. Scatter-add into (G, T) output.
    let zeros = Tensor::<B, 2>::zeros([num_gauges, t], &gathered.device());
    zeros.scatter(0, group_2d, gathered, IndexingUpdateOp::Add)
}

#[cfg(test)]
mod tests {
    use super::*;
    use burn::backend::NdArray;
    use burn::tensor::TensorData;

    type B = NdArray<f32>;
    type Dev = <B as burn::tensor::backend::BackendTypes>::Device;

    #[test]
    fn scatter_add_three_gauges_two_groups() {
        // 4 segments × 2 timesteps.
        //   runoff = [[1, 10], [2, 20], [3, 30], [4, 40]]
        // outflow_idx = [[0, 1], [2], [3]]
        //   → flat_indices = [0, 1, 2, 3], group_ids = [0, 0, 1, 2]
        // expected (G=3, T=2): [[1+2=3, 10+20=30], [3, 30], [4, 40]]
        let device = Dev::default();
        let runoff = Tensor::<B, 2>::from_data(
            TensorData::new(vec![1.0f32, 10.0, 2.0, 20.0, 3.0, 30.0, 4.0, 40.0], [4, 2]),
            &device,
        );
        let flat = Tensor::<B, 1, Int>::from_data(
            TensorData::from([0i32, 1, 2, 3].as_slice()),
            &device,
        );
        let group = Tensor::<B, 1, Int>::from_data(
            TensorData::from([0i32, 0, 1, 2].as_slice()),
            &device,
        );
        let out = scatter_add_by_group(runoff, flat, group, 3);
        let v: Vec<f32> = out.into_data().into_vec().unwrap();
        assert_eq!(v, vec![3.0, 30.0, 3.0, 30.0, 4.0, 40.0]);
    }
}

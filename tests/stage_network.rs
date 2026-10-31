//! Stage network constants: closure, bed elevation, edge lengths, masks, dams.

use ddrs::data::ConusAdjacencyStore;
use ddrs::routing::stage::{bed_elevation, downstream_closure, StageNetwork};
use ddrs::sparse::{CsrPattern, SparseAdjacency};

/// Chains laid out upstream to downstream; each chain `k..k+len` flows to `k+len-1`.
fn chains(lens: &[usize], length: Vec<f32>, slope: Vec<f32>) -> SparseAdjacency {
    let n: usize = lens.iter().sum();
    let (mut rows, mut cols) = (vec![], vec![]);
    let mut start = 0;
    for &l in lens {
        for i in start..start + l - 1 {
            rows.push((i + 1) as i32);
            cols.push(i as i32);
        }
        start += l;
    }
    SparseAdjacency {
        n,
        values: vec![1.0; rows.len()],
        rows,
        cols,
        length_m: length,
        slope,
        parent_offset: None,
    }
}

fn chain3() -> SparseAdjacency {
    chains(&[3], vec![1000.0, 2000.0, 3000.0], vec![1e-3, 2e-3, 3e-3])
}

fn two_chains() -> SparseAdjacency {
    chains(&[3, 3], vec![1000.0; 6], vec![1e-3; 6])
}

#[test]
fn downstream_closure_adds_the_path_to_the_outlet() {
    let adj = chains(&[5], vec![100.0; 5], vec![1e-3; 5]);
    let (in_stage, added) = downstream_closure(&adj, &[1]);
    assert_eq!(in_stage, vec![false, true, true, true, true]);
    assert_eq!(added, 3);
    let pat = CsrPattern::from_sparse(&adj);
    let net = StageNetwork::build(&adj, &pat, Some(&[1]), 0.0, &[]).unwrap();
    assert_eq!(net.n_stage, 4);
    assert_eq!(net.n_added, 3);
}

#[test]
fn bed_elevation_integrates_slope_times_length() {
    let (z_out, z_mid) = bed_elevation(&chain3(), 0.0);
    assert_eq!(z_out, vec![13.0, 9.0, 0.0]);
    assert_eq!(z_mid, vec![13.5, 11.0, 4.5]);
}

#[test]
fn edge_length_ends_at_the_outlet_downstream_end() {
    let adj = chain3();
    let pat = CsrPattern::from_sparse(&adj);
    let net = StageNetwork::build(&adj, &pat, None, 0.0, &[]).unwrap();
    assert_eq!(net.edge_length, vec![1500.0, 2500.0, 1500.0]);
    assert_eq!(net.z_out[net.outlet], 0.0);
    let net = StageNetwork::build(&adj, &pat, None, -10.0, &[]).unwrap();
    assert_eq!(net.z_out[2], -10.0);
}

#[test]
fn juniata_network_facts() {
    let store = ConusAdjacencyStore::open("examples/juniata/data/juniata_conus_adjacency.zarr")
        .expect("open bundle");
    let adj = SparseAdjacency {
        n: store.n,
        rows: store.indices_0.clone(),
        cols: store.indices_1.clone(),
        values: vec![1.0; store.nnz],
        length_m: store.length_m.to_vec(),
        slope: store.slope.to_vec(),
        parent_offset: None,
    };
    let pat = CsrPattern::from_sparse(&adj);
    let net = StageNetwork::build(&adj, &pat, None, 0.0, &[]).unwrap();
    assert_eq!(net.outlet, 212);
    assert!(net.in_stage.iter().all(|&b| b));
    assert_eq!(net.n_added, 0);
    assert!((net.z_out[177] - 145.256).abs() < 0.01, "z_out[177] = {}", net.z_out[177]);
    let (closure, _) = downstream_closure(&adj, &[177]);
    assert_eq!(closure.iter().filter(|&&b| b).count(), 19);
    assert!((net.edge_length[212] - 1033.5).abs() < 0.1, "{}", net.edge_length[212]);
}

#[test]
fn stage_set_with_two_outlets_is_rejected() {
    let adj = two_chains();
    let pat = CsrPattern::from_sparse(&adj);
    let err = StageNetwork::build(&adj, &pat, None, 0.0, &[]).unwrap_err();
    assert!(err.contains('2') && err.contains("outlet"), "{err}");
}

#[test]
fn stage_set_on_one_of_two_trees_is_accepted() {
    let adj = two_chains();
    let pat = CsrPattern::from_sparse(&adj);
    let net = StageNetwork::build(&adj, &pat, Some(&[3]), 0.0, &[]).unwrap();
    assert_eq!(net.outlet, 5);
    assert_eq!(&net.in_stage[..3], &[false, false, false]);
    assert!(net.in_stage[3..].iter().all(|&b| b));
}

#[test]
fn dam_days_become_seconds() {
    let adj = chain3();
    let pat = CsrPattern::from_sparse(&adj);
    let net = StageNetwork::build(&adj, &pat, None, 0.0, &[(1, 1.23)]).unwrap();
    assert_eq!(net.dam_rows, vec![1]);
    let want = 1.23f32 * 86_400.0;
    assert!((net.dam_t_seconds[0] - want).abs() / want < 1e-6);
}

fn ones(pat: &CsrPattern, mask: &[f32]) -> Vec<(i32, i32)> {
    let mut v: Vec<(i32, i32)> = (0..pat.nnz())
        .filter(|&s| mask[s] == 1.0)
        .map(|s| (pat.row_for_nnz[s], pat.col[s]))
        .collect();
    v.sort();
    v
}

#[test]
fn edge_masks_split_mc_to_stage_and_stage_to_stage() {
    // Y junction: headwaters 0 and 1 join at 2, then 2 -> 3 -> 4 (outlet).
    let adj = SparseAdjacency {
        n: 5,
        rows: vec![2, 2, 3, 4],
        cols: vec![0, 1, 2, 3],
        values: vec![1.0; 4],
        length_m: vec![1000.0; 5],
        slope: vec![1e-3; 5],
        parent_offset: None,
    };
    let pat = CsrPattern::from_sparse(&adj);
    let net = StageNetwork::build(&adj, &pat, Some(&[2]), 0.0, &[]).unwrap();
    assert_eq!(net.in_stage, vec![false, false, true, true, true]);
    // Stage-to-stage: (3,2) and (4,3). MC-to-stage: (2,0) and (2,1).
    assert_eq!(ones(&pat, &net.adj_stage_to_stage), vec![(3, 2), (4, 3)]);
    assert_eq!(ones(&pat, &net.adj_mc_to_stage), vec![(2, 0), (2, 1)]);
    // Diagonal slots stay 0 and no slot is set in both masks.
    for s in 0..pat.nnz() {
        if pat.row_for_nnz[s] == pat.col[s] {
            assert_eq!(net.adj_mc_to_stage[s], 0.0);
            assert_eq!(net.adj_stage_to_stage[s], 0.0);
        }
        assert!(net.adj_mc_to_stage[s] * net.adj_stage_to_stage[s] == 0.0);
    }
    // MC-to-MC would need a row in {0,1}: those rows have no off-diagonal entries here,
    // so total ones equal the four stage-row edges.
    let total = ones(&pat, &net.adj_mc_to_stage).len() + ones(&pat, &net.adj_stage_to_stage).len();
    assert_eq!(total, 4);
}

#[test]
fn dam_outside_the_stage_set_is_skipped() {
    let adj = two_chains();
    let pat = CsrPattern::from_sparse(&adj);
    // Stage set = tree 2 (rows 3..6). Dam at row 4 is inside, dam at row 1 is on tree 1.
    let net = StageNetwork::build(&adj, &pat, Some(&[3]), 0.0, &[(1, 2.0), (4, 1.23)]).unwrap();
    assert_eq!(net.dam_rows, vec![4]);
    let want = 1.23f32 * 86_400.0;
    assert_eq!(net.dam_t_seconds.len(), 1);
    assert!((net.dam_t_seconds[0] - want).abs() / want < 1e-6);
}

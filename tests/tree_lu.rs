//! Zero-fill tree LU on the `CsrPattern` layout: host factor, solves, gradient.
//!
//! The reference is a test-local f64 dense solver (the f32-only rule covers
//! `src/`, not tests).

use ddrs::data::ConusAdjacencyStore;
use ddrs::sparse::tree_lu::{
    assert_tree, diag_slot, tree_lu_factor, tree_lu_grad_a_host, tree_lu_solve_host,
    tree_lu_solve_transposed_host,
};
use ddrs::sparse::{CsrPattern, SparseAdjacency};

/// Small deterministic generator (splitmix64), so the tests need no rand crate.
struct Rng(u64);
impl Rng {
    fn next_u64(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }
    fn unit(&mut self) -> f64 {
        (self.next_u64() >> 11) as f64 / (1u64 << 53) as f64
    }
    fn range(&mut self, lo: f64, hi: f64) -> f64 {
        lo + (hi - lo) * self.unit()
    }
    fn below(&mut self, n: usize) -> usize {
        (self.next_u64() % n as u64) as usize
    }
}

/// Gaussian elimination with partial pivoting, f64, dense row-major.
fn dense_solve_f64(n: usize, a: &[f64], b: &[f64]) -> Vec<f64> {
    let mut m = a.to_vec();
    let mut x = b.to_vec();
    for c in 0..n {
        let mut p = c;
        for r in c + 1..n {
            if m[r * n + c].abs() > m[p * n + c].abs() {
                p = r;
            }
        }
        if p != c {
            for k in 0..n {
                m.swap(c * n + k, p * n + k);
            }
            x.swap(c, p);
        }
        for r in c + 1..n {
            let f = m[r * n + c] / m[c * n + c];
            if f != 0.0 {
                for k in c..n {
                    m[r * n + k] -= f * m[c * n + k];
                }
                x[r] -= f * x[c];
            }
        }
    }
    for c in (0..n).rev() {
        let mut s = x[c];
        for k in c + 1..n {
            s -= m[c * n + k] * x[k];
        }
        x[c] = s / m[c * n + c];
    }
    x
}

fn transpose(n: usize, a: &[f64]) -> Vec<f64> {
    let mut t = vec![0.0; n * n];
    for i in 0..n {
        for j in 0..n {
            t[j * n + i] = a[i * n + j];
        }
    }
    t
}

/// Each reach `i < n-1` is an outlet with probability 0.05, else drains to a
/// random downstream in `(i, n)`. Reach `n-1` is always an outlet.
fn random_tree(n: usize, seed: u64) -> SparseAdjacency {
    let mut rng = Rng(seed.wrapping_mul(7919).wrapping_add(n as u64));
    let (mut rows, mut cols) = (Vec::new(), Vec::new());
    for i in 0..n.saturating_sub(1) {
        if rng.unit() < 0.05 {
            continue;
        }
        let d = i + 1 + rng.below(n - 1 - i);
        rows.push(d as i32);
        cols.push(i as i32);
    }
    let values = vec![1.0; rows.len()];
    SparseAdjacency {
        n,
        rows,
        cols,
        values,
        length_m: vec![1000.0; n],
        slope: vec![1e-3; n],
        parent_offset: None,
    }
}

/// Slot of the lower-half entry `(row, col)`.
fn slot(p: &CsrPattern, row: usize, col: usize) -> usize {
    (p.crow[row] as usize..p.crow[row + 1] as usize)
        .find(|&s| p.col[s] as usize == col)
        .expect("slot in pattern")
}

/// Assemble the stage-style matrix on both layouts. Friction edge `k -> d`:
/// `A[k,k] += G`, `A[d,d] += G`, `A[d,k] = -G`, `A[k,d] = -G`. Control edge:
/// `A[j,j] += R'`, `A[d,j] = -R'`, `A[j,d] = 0`.
fn random_tree_matrix(
    adj: &SparseAdjacency,
    seed: u64,
    n_one_way: usize,
) -> (Vec<f32>, Vec<f64>) {
    let p = CsrPattern::from_sparse(adj);
    let n = adj.n;
    let nnz = p.nnz();
    let mut rng = Rng(seed ^ 0xABCD_1234);
    let mut a = vec![0.0f32; 2 * nnz];
    let mut dense = vec![0.0f64; n * n];
    let add_diag = |a: &mut Vec<f32>, dense: &mut Vec<f64>, i: usize, v: f32| {
        a[diag_slot(&p, i)] += v;
        dense[i * n + i] += v as f64;
    };
    for i in 0..n {
        let ds = rng.range(1.0, 1000.0) as f32;
        add_diag(&mut a, &mut dense, i, ds);
    }
    let ne = adj.rows.len();
    let mut control = vec![false; ne];
    let mut left = n_one_way.min(ne);
    while left > 0 {
        let e = rng.below(ne);
        if !control[e] {
            control[e] = true;
            left -= 1;
        }
    }
    for e in 0..ne {
        let d = adj.rows[e] as usize;
        let k = adj.cols[e] as usize;
        let s = slot(&p, d, k);
        if control[e] {
            let r = rng.range(0.0, 50.0) as f32;
            add_diag(&mut a, &mut dense, k, r);
            a[s] = -r;
            dense[d * n + k] = -(r as f64);
            // upper half and dense A[k,d] stay 0
        } else {
            let g = rng.range(0.1, 100.0) as f32;
            add_diag(&mut a, &mut dense, k, g);
            add_diag(&mut a, &mut dense, d, g);
            a[s] = -g;
            a[nnz + s] = -g;
            dense[d * n + k] = -(g as f64);
            dense[k * n + d] = -(g as f64);
        }
    }
    (a, dense)
}

fn rhs(n: usize, seed: u64) -> Vec<f32> {
    let mut rng = Rng(seed ^ 0x55AA);
    (0..n).map(|_| rng.range(-10.0, 10.0) as f32).collect()
}

fn max_rel_err(got: &[f32], want: &[f64]) -> f64 {
    let scale = want.iter().fold(0.0f64, |m, v| m.max(v.abs())).max(1e-30);
    got.iter()
        .zip(want)
        .fold(0.0f64, |m, (g, w)| m.max((*g as f64 - w).abs() / scale))
}

#[test]
fn factor_matches_dense_on_random_trees() {
    for &n in &[5usize, 17, 64, 211, 500] {
        for seed in 0..3u64 {
            let adj = random_tree(n, seed);
            let p = CsrPattern::from_sparse(&adj);
            assert_tree(&p).expect("tree");
            let (a, dense) = random_tree_matrix(&adj, seed, n / 10);
            let f = tree_lu_factor(&p, &a);
            let b = rhs(n, seed);
            let x = tree_lu_solve_host(&p, &f, &b);
            let b64: Vec<f64> = b.iter().map(|v| *v as f64).collect();
            let want = dense_solve_f64(n, &dense, &b64);
            let err = max_rel_err(&x, &want);
            assert!(err < 1e-5, "n={n} seed={seed}: rel err {err}");
        }
    }
}

#[test]
fn transposed_solve_matches_dense() {
    for &n in &[5usize, 17, 64, 211, 500] {
        for seed in 0..3u64 {
            let adj = random_tree(n, seed);
            let p = CsrPattern::from_sparse(&adj);
            let (a, dense) = random_tree_matrix(&adj, seed, n / 10);
            let f = tree_lu_factor(&p, &a);
            let g = rhs(n, seed + 100);
            let y = tree_lu_solve_transposed_host(&p, &f, &g);
            let g64: Vec<f64> = g.iter().map(|v| *v as f64).collect();
            let want = dense_solve_f64(n, &transpose(n, &dense), &g64);
            let err = max_rel_err(&y, &want);
            assert!(err < 1e-5, "n={n} seed={seed}: rel err {err}");
        }
    }
}

#[test]
fn grad_a_matches_csr_solve_rule() {
    let n = 64;
    let adj = random_tree(n, 3);
    let p = CsrPattern::from_sparse(&adj);
    let nnz = p.nnz();
    let g = rhs(n, 1);
    let x = rhs(n, 2);
    let ga = tree_lu_grad_a_host(&p, &g, &x);
    assert_eq!(ga.len(), 2 * nnz);
    for s in 0..nnz {
        let r = p.row_for_nnz[s] as usize;
        let c = p.col[s] as usize;
        assert_eq!(ga[s].to_bits(), (-g[r] * x[c]).to_bits(), "lower slot {s}");
        let want_up = if r == c { 0.0 } else { -g[c] * x[r] };
        assert_eq!(ga[nnz + s].to_bits(), want_up.to_bits(), "upper slot {s}");
    }
}

#[test]
fn one_way_edge_with_zero_slope_decouples() {
    // chain 0 -> 1 -> 2; the edge 1 -> 2 is a control edge with R' = 0.
    let adj = SparseAdjacency {
        n: 3,
        rows: vec![1, 2],
        cols: vec![0, 1],
        values: vec![1.0, 1.0],
        length_m: vec![1000.0; 3],
        slope: vec![1e-3; 3],
        parent_offset: None,
    };
    let p = CsrPattern::from_sparse(&adj);
    let nnz = p.nnz();
    let d_s = [3.0f32, 5.0, 7.0];
    let th2_beta = 2.5f32; // theta^2 * beta on the friction edge 0 -> 1
    let mut a = vec![0.0f32; 2 * nnz];
    for i in 0..3 {
        a[diag_slot(&p, i)] = d_s[i];
    }
    a[diag_slot(&p, 0)] += th2_beta;
    a[diag_slot(&p, 1)] += th2_beta;
    let s10 = slot(&p, 1, 0);
    a[s10] = -th2_beta;
    a[nnz + s10] = -th2_beta;
    // control edge 1 -> 2 with R' = 0: slots stay 0, A[1,1] gains 0.
    let f = tree_lu_factor(&p, &a);
    assert_eq!(f.u_values[diag_slot(&p, 2)], d_s[2]);
    let x1 = tree_lu_solve_host(&p, &f, &[1.0, 2.0, 3.0]);
    let x2 = tree_lu_solve_host(&p, &f, &[-40.0, 90.0, 3.0]);
    assert_eq!(x1[2], x2[2]);
    assert_eq!(x1[2], 3.0 / d_s[2]);
}

#[test]
fn column_dominance_holds_with_one_way_edges() {
    for seed in 0..10u64 {
        let n = 80;
        let adj = random_tree(n, seed);
        let (_, dense) = random_tree_matrix(&adj, seed, 3);
        for c in 0..n {
            let off: f64 = (0..n).filter(|&r| r != c).map(|r| dense[r * n + c].abs()).sum();
            assert!(dense[c * n + c].abs() > off, "seed {seed} column {c}");
        }
    }
}

#[test]
fn assert_tree_rejects_two_downstreams() {
    // reach 0 drains to both 1 and 2
    let adj = SparseAdjacency {
        n: 3,
        rows: vec![1, 2],
        cols: vec![0, 0],
        values: vec![1.0, 1.0],
        length_m: vec![1000.0; 3],
        slope: vec![1e-3; 3],
        parent_offset: None,
    };
    let p = CsrPattern::from_sparse(&adj);
    let err = assert_tree(&p).unwrap_err();
    assert!(err.contains("two downstream"), "{err}");
}

#[test]
fn assert_tree_rejects_a_row_without_its_diagonal() {
    // Row 1 holds only column 0: no diagonal slot.
    let p = CsrPattern::from_csr_structure(2, vec![0, 1, 2], vec![0, 0]);
    let err = assert_tree(&p).unwrap_err();
    assert!(err.contains("diagonal"), "{err}");
}

#[test]
fn juniata_pattern_is_a_tree() {
    let path = std::path::PathBuf::from("examples/juniata/data/juniata_conus_adjacency.zarr");
    let store = ConusAdjacencyStore::open(&path).expect("open juniata adjacency");
    let n = store.n;
    let adj = SparseAdjacency {
        n,
        rows: store.indices_0.clone(),
        cols: store.indices_1.clone(),
        values: vec![1.0; store.nnz],
        length_m: store.length_m.to_vec(),
        slope: store.slope.to_vec(),
        parent_offset: None,
    };
    let p = CsrPattern::from_sparse(&adj);
    assert_eq!(n, 213);
    assert_tree(&p).expect("juniata is a tree");
}

//! Gradient checks for `TreeSolveOp` (`tree_lu_solve`) on `Autodiff<NdArray<f32>>`.

use std::sync::Arc;

use burn::backend::{Autodiff, NdArray};
use burn::tensor::{Tensor, TensorData};
use ddrs::sparse::tree_lu::{
    diag_slot, tree_lu_factor, tree_lu_solve, tree_lu_solve_host, tree_lu_solve_transposed_host,
};
use ddrs::sparse::{CsrPattern, SparseAdjacency};

type Inner = NdArray<f32>;
type B = Autodiff<Inner>;

struct Rng(u64);
impl Rng {
    fn unit(&mut self) -> f64 {
        self.0 = self.0.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        ((z ^ (z >> 31)) >> 11) as f64 / (1u64 << 53) as f64
    }
    fn range(&mut self, lo: f64, hi: f64) -> f64 {
        lo + (hi - lo) * self.unit()
    }
}

/// 7-node tree. Edges (k -> d): 0->2, 1->2, 2->4, 3->4, 4->6, 5->6.
const EDGES: [(usize, usize); 6] = [(0, 2), (1, 2), (2, 4), (3, 4), (4, 6), (5, 6)];
const N: usize = 7;

fn tree() -> (SparseAdjacency, Arc<CsrPattern>) {
    let adj = SparseAdjacency {
        n: N,
        rows: EDGES.iter().map(|e| e.1 as i32).collect(),
        cols: EDGES.iter().map(|e| e.0 as i32).collect(),
        values: vec![1.0; EDGES.len()],
        length_m: vec![1000.0; N],
        slope: vec![1e-3; N],
        parent_offset: None,
    };
    let p = Arc::new(CsrPattern::from_sparse(&adj));
    (adj, p)
}

fn slot(p: &CsrPattern, row: usize, col: usize) -> usize {
    (p.crow[row] as usize..p.crow[row + 1] as usize)
        .find(|&s| p.col[s] as usize == col)
        .expect("slot")
}

/// Diagonally dominant matrix; edge 1 and 4 are one-way (upper half 0).
fn matrix(p: &CsrPattern, seed: u64) -> Vec<f32> {
    let nnz = p.nnz();
    let mut rng = Rng(seed);
    let mut a = vec![0.0f32; 2 * nnz];
    for i in 0..N {
        a[diag_slot(p, i)] = rng.range(4.0, 8.0) as f32;
    }
    for (e, &(k, d)) in EDGES.iter().enumerate() {
        let s = slot(p, d, k);
        a[s] = -(rng.range(0.3, 1.0) as f32);
        if e != 1 && e != 4 {
            a[nnz + s] = -(rng.range(0.3, 1.0) as f32);
        }
    }
    a
}

fn vec_b(n: usize, seed: u64, lo: f64, hi: f64) -> Vec<f32> {
    let mut rng = Rng(seed);
    (0..n).map(|_| rng.range(lo, hi) as f32).collect()
}

fn t1(v: &[f32], grad: bool) -> Tensor<B, 1> {
    let t = Tensor::<B, 1>::from_data(TensorData::new(v.to_vec(), [v.len()]), &Default::default());
    if grad {
        t.require_grad()
    } else {
        t
    }
}

fn host_vec(t: Tensor<B, 1>) -> Vec<f32> {
    t.into_data().to_vec::<f32>().unwrap()
}

#[test]
fn forward_matches_host_solve() {
    let (_, p) = tree();
    let a = matrix(&p, 1);
    let b = vec_b(N, 2, -5.0, 5.0);
    let x = host_vec(tree_lu_solve(&p, t1(&a, true), t1(&b, true), false));
    let want = tree_lu_solve_host(&p, &tree_lu_factor(&p, &a), &b);
    for (g, w) in x.iter().zip(&want) {
        assert_eq!(g.to_bits(), w.to_bits());
    }
}

/// `use_cuda: true` on a CPU backend falls back to the host path, bit for bit.
#[test]
fn use_cuda_on_a_cpu_backend_falls_back_bitwise() {
    let (_, p) = tree();
    let a = matrix(&p, 1);
    let b = vec_b(N, 2, -5.0, 5.0);
    let w = vec_b(N, 5, -1.0, 1.0);
    let run = |use_cuda: bool| {
        let (at, bt) = (t1(&a, true), t1(&b, true));
        let x = tree_lu_solve(&p, at.clone(), bt.clone(), use_cuda);
        let loss = (x.clone() * t1(&w, false)).sum();
        let grads = loss.backward();
        (
            host_vec(x),
            at.grad(&grads).unwrap().into_data().to_vec::<f32>().unwrap(),
            bt.grad(&grads).unwrap().into_data().to_vec::<f32>().unwrap(),
        )
    };
    let (x_on, ga_on, gb_on) = run(true);
    let (x_off, ga_off, gb_off) = run(false);
    for (name, on, off) in [("x", &x_on, &x_off), ("grad a", &ga_on, &ga_off), ("grad b", &gb_on, &gb_off)] {
        assert_eq!(on.len(), off.len(), "{name} length");
        for (i, (g, w)) in on.iter().zip(off.iter()).enumerate() {
            assert_eq!(g.to_bits(), w.to_bits(), "{name}[{i}]");
        }
    }
}

fn loss_of(p: &Arc<CsrPattern>, a: &[f32], b: &[f32], w: &[f32]) -> f32 {
    let x = tree_lu_solve_host(p, &tree_lu_factor(p, a), b);
    x.iter().zip(w).map(|(x, w)| x * w).sum()
}

#[test]
fn gradcheck_a_and_b_by_central_differences() {
    let (_, p) = tree();
    let nnz = p.nnz();
    let a = matrix(&p, 3);
    let b = vec_b(N, 4, -5.0, 5.0);
    let w = vec_b(N, 5, -1.0, 1.0);

    let at = t1(&a, true);
    let bt = t1(&b, true);
    let wt = t1(&w, false);
    let x = tree_lu_solve(&p, at.clone(), bt.clone(), false);
    let loss = (x * wt).sum();
    let grads = loss.backward();
    let ga = at.grad(&grads).unwrap().into_data().to_vec::<f32>().unwrap();
    let gb = bt.grad(&grads).unwrap().into_data().to_vec::<f32>().unwrap();
    assert_eq!(ga.len(), 2 * nnz);

    const REL_TOL: f32 = 5e-3;
    const ABS_TOL: f32 = 1e-4;
    let check = |name: &str, i: usize, analytic: f32, fd: f32| {
        let err = (analytic - fd).abs();
        assert!(
            err <= ABS_TOL || err <= REL_TOL * fd.abs().max(analytic.abs()),
            "{name}[{i}]: analytic {analytic} vs fd {fd}"
        );
    };
    for i in 0..2 * nnz {
        // Structural zeros (upper half of the diagonal and of one-way edges)
        // are still inputs; sweep them too.
        let eps = (1e-3 * a[i].abs()).max(1e-3);
        let (mut hi, mut lo) = (a.clone(), a.clone());
        hi[i] += eps;
        lo[i] -= eps;
        let fd = (loss_of(&p, &hi, &b, &w) - loss_of(&p, &lo, &b, &w)) / (2.0 * eps);
        // The diagonal slot of the upper half is ignored by the op: its
        // gradient is 0 by construction and so is the finite difference.
        check("a", i, ga[i], fd);
    }
    for i in 0..N {
        let eps = (1e-3 * b[i].abs()).max(1e-3);
        let (mut hi, mut lo) = (b.clone(), b.clone());
        hi[i] += eps;
        lo[i] -= eps;
        let fd = (loss_of(&p, &a, &hi, &w) - loss_of(&p, &a, &lo, &w)) / (2.0 * eps);
        check("b", i, gb[i], fd);
    }
}

#[test]
fn friction_edge_gradient_identity() {
    let (_, p) = tree();
    let nnz = p.nnz();
    let theta2 = 0.55f32 * 0.55f32;
    let ne = EDGES.len();
    let beta = vec_b(ne, 6, 0.3, 1.0);
    let ds = vec_b(N, 7, 4.0, 8.0);
    let w = vec_b(N, 8, -1.0, 1.0);
    let b = vec_b(N, 9, -5.0, 5.0);

    // A = P_beta * beta + P_d * D_s, built with Burn ops (differentiable).
    let mut pb = vec![0.0f32; 2 * nnz * ne];
    let mut pd = vec![0.0f32; 2 * nnz * N];
    for (e, &(k, d)) in EDGES.iter().enumerate() {
        let s = slot(&p, d, k);
        pb[s * ne + e] = -theta2;
        pb[(nnz + s) * ne + e] = -theta2;
        pb[diag_slot(&p, k) * ne + e] += theta2;
        pb[diag_slot(&p, d) * ne + e] += theta2;
    }
    for i in 0..N {
        pd[diag_slot(&p, i) * N + i] = 1.0;
    }
    let dev = Default::default();
    let pb_t = Tensor::<B, 2>::from_data(TensorData::new(pb, [2 * nnz, ne]), &dev);
    let pd_t = Tensor::<B, 2>::from_data(TensorData::new(pd, [2 * nnz, N]), &dev);
    let beta_t = t1(&beta, true);
    let ds_t = t1(&ds, true);
    let a_t = (pb_t.matmul(beta_t.clone().reshape([ne, 1])) + pd_t.matmul(ds_t.clone().reshape([N, 1])))
        .reshape([2 * nnz]);
    let a_host = a_t.clone().into_data().to_vec::<f32>().unwrap();

    let x = tree_lu_solve(&p, a_t, t1(&b, false), false);
    let x_host = x.clone().into_data().to_vec::<f32>().unwrap();
    let loss = (x * t1(&w, false)).sum();
    let grads = loss.backward();
    let g_beta = beta_t.grad(&grads).unwrap().into_data().to_vec::<f32>().unwrap();
    let g_ds = ds_t.grad(&grads).unwrap().into_data().to_vec::<f32>().unwrap();

    let f = tree_lu_factor(&p, &a_host);
    let g = tree_lu_solve_transposed_host(&p, &f, &w);
    let close = |got: f32, want: f32, what: &str| {
        let scale = want.abs().max(1e-6);
        assert!((got - want).abs() / scale < 1e-4, "{what}: {got} vs {want}");
    };
    for (e, &(k, d)) in EDGES.iter().enumerate() {
        let want = -theta2 * (g[k] - g[d]) * (x_host[k] - x_host[d]);
        close(g_beta[e], want, &format!("dL/dbeta[{e}]"));
    }
    for i in 0..N {
        close(g_ds[i], -g[i] * x_host[i], &format!("dL/dD_s[{i}]"));
    }
}

#[test]
fn untracked_inputs_take_the_untracked_branch() {
    let (_, p) = tree();
    let a = matrix(&p, 1);
    let b = vec_b(N, 2, -5.0, 5.0);
    let x = host_vec(tree_lu_solve(&p, t1(&a, false), t1(&b, false), false));
    let want = tree_lu_solve_host(&p, &tree_lu_factor(&p, &a), &b);
    assert_eq!(x, want);
}

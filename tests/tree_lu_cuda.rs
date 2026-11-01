//! CUDA arm of the tree LU: factor, solves, grad-A and the autodiff op,
//! against the host path.
#![cfg(feature = "cuda")]

use std::sync::Arc;

use burn::backend::{Autodiff, NdArray};
use burn::tensor::backend::{Backend, BackendTypes};
use burn::tensor::{Tensor, TensorData};
use ddrs::data::ConusAdjacencyStore;
use ddrs::sparse::cube_buf::{cube_from_host, cube_to_host, Cube, Gpu};
use ddrs::sparse::tree_lu::{
    diag_slot, tree_levels, tree_lu_factor, tree_lu_grad_a_host, tree_lu_solve,
    tree_lu_solve_host, tree_lu_solve_transposed_host,
};
use ddrs::sparse::tree_lu_cuda::{
    tree_factor_dev, tree_grad_a_dev, tree_solve_dev, tree_solve_transposed_dev, TreeDev,
};
use ddrs::sparse::{host_reads_on_this_thread_for_test, CsrPattern, SparseAdjacency};

type Cpu = NdArray<f32>;
const XBACKEND_REL_TOL: f32 = 1e-5;

// copied from tests/tree_lu.rs::Rng
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

// copied from tests/tree_lu.rs::random_tree
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

// copied from tests/tree_lu.rs::slot
fn slot(p: &CsrPattern, row: usize, col: usize) -> usize {
    (p.crow[row] as usize..p.crow[row + 1] as usize)
        .find(|&s| p.col[s] as usize == col)
        .expect("slot in pattern")
}

/// Stage-style matrix on the `2 * nnz` layout.
/// Adapted from tests/tree_lu.rs::random_tree_matrix: same values, no dense copy,
/// so it also runs at CONUS size.
fn random_tree_matrix(adj: &SparseAdjacency, p: &CsrPattern, seed: u64, n_one_way: usize) -> Vec<f32> {
    let (n, nnz) = (adj.n, p.nnz());
    let mut rng = Rng(seed ^ 0xABCD_1234);
    let mut a = vec![0.0f32; 2 * nnz];
    for i in 0..n {
        a[diag_slot(p, i)] += rng.range(1.0, 1000.0) as f32;
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
        let s = slot(p, d, k);
        if control[e] {
            let r = rng.range(0.0, 50.0) as f32;
            a[diag_slot(p, k)] += r;
            a[s] = -r;
        } else {
            let g = rng.range(0.1, 100.0) as f32;
            a[diag_slot(p, k)] += g;
            a[diag_slot(p, d)] += g;
            a[s] = -g;
            a[nnz + s] = -g;
        }
    }
    a
}

// copied from tests/stage_op_parity.rs::juniata_case (adjacency part)
fn bundle(path: &str) -> (SparseAdjacency, Arc<CsrPattern>) {
    let store = ConusAdjacencyStore::open(path).expect("open bundle adjacency");
    let adj = SparseAdjacency {
        n: store.n,
        rows: store.indices_0.clone(),
        cols: store.indices_1.clone(),
        values: vec![1.0; store.nnz],
        length_m: store.length_m.to_vec(),
        slope: store.slope.to_vec(),
        parent_offset: None,
    };
    let p = Arc::new(CsrPattern::from_sparse(&adj));
    (adj, p)
}

const JUNIATA: &str = "examples/juniata/data/juniata_conus_adjacency.zarr";
const HUDSON: &str = "examples/hudson/data/hudson_conus_adjacency.zarr";

fn device() -> <Gpu as BackendTypes>::Device {
    Default::default()
}

// copied from tests/tree_lu_gradcheck.rs::vec_b (renamed)
fn vec_rand(n: usize, seed: u64, lo: f64, hi: f64) -> Vec<f32> {
    let mut rng = Rng(seed);
    (0..n).map(|_| rng.range(lo, hi) as f32).collect()
}

fn max_abs(v: &[f32]) -> f32 {
    v.iter().fold(0.0f32, |m, x| m.max(x.abs()))
}

/// Largest difference relative to `max|want|`.
fn rel_worst(got: &[f32], want: &[f32]) -> f32 {
    assert_eq!(got.len(), want.len());
    let scale = max_abs(want).max(1e-30);
    got.iter().zip(want).fold(0.0f32, |m, (g, w)| m.max((g - w).abs() / scale))
}

fn same_zero_set(a: &[f32], b: &[f32], what: &str) {
    for (i, (x, y)) in a.iter().zip(b).enumerate() {
        assert_eq!(*x == 0.0, *y == 0.0, "{what}[{i}]: {x} vs {y}");
    }
}

fn dev_tree(p: &CsrPattern, like: &Cube) -> TreeDev {
    TreeDev::upload(&tree_levels(p).expect("tree"), like)
}

fn check_factor(name: &str, adj: &SparseAdjacency, p: &CsrPattern, seed: u64) -> f32 {
    let a = random_tree_matrix(adj, p, seed, adj.n / 7);
    let host = tree_lu_factor(p, &a);
    let a_cube = cube_from_host(&a, &device());
    let t = dev_tree(p, &a_cube);
    let f = tree_factor_dev(&t, &a_cube);
    let (dl, du) = (cube_to_host(&f.l_values), cube_to_host(&f.u_values));
    let (el, eu) = (rel_worst(&dl, &host.l_values), rel_worst(&du, &host.u_values));
    println!("{name}: worst factor diff, l {el:e}, u {eu:e}");
    assert!(el <= 1e-6 && eu <= 1e-6, "{name}: l {el:e} u {eu:e}");
    el.max(eu)
}

#[test]
fn factor_matches_host_on_random_trees_and_bundles() {
    let mut worst = 0.0f32;
    // n = 5000 puts well over 1024 reaches in level 0, so the strided loop of
    // the factor kernel takes several uneven passes.
    for seed in 0..3u64 {
        let adj = random_tree(5000, seed);
        let p = CsrPattern::from_sparse(&adj);
        let levels = tree_levels(&p).unwrap();
        let widest = (0..levels.n_levels)
            .map(|l| levels.lvl_ptr[l + 1] - levels.lvl_ptr[l])
            .max()
            .unwrap();
        assert!(widest > 2048, "widest level {widest} is too narrow");
        worst = worst.max(check_factor(&format!("random_tree(5000) seed {seed}"), &adj, &p, seed));
    }
    for seed in 0..3u64 {
        let adj = random_tree(300, seed);
        let p = CsrPattern::from_sparse(&adj);
        worst = worst.max(check_factor(&format!("random_tree(300) seed {seed}"), &adj, &p, seed));
    }
    for (name, path) in [("Juniata", JUNIATA), ("Hudson", HUDSON)] {
        let (adj, p) = bundle(path);
        worst = worst.max(check_factor(name, &adj, &p, 11));
    }
    println!("worst factor diff over all cases: {worst:e}");
}

#[test]
fn factor_matches_host_on_conus() {
    if cfg!(debug_assertions) {
        eprintln!("skipped: release only");
        return;
    }
    let Ok(path) = std::env::var("DDRS_CONUS_ADJ") else {
        eprintln!("skipped: DDRS_CONUS_ADJ is not set");
        return;
    };
    let (adj, p) = bundle(&path);
    let a = random_tree_matrix(&adj, &p, 5, adj.n / 7);
    let host = tree_lu_factor(&p, &a);
    let a_cube = cube_from_host(&a, &device());
    let t = dev_tree(&p, &a_cube);
    let levels = t.n_levels;
    let f = tree_factor_dev(&t, &a_cube);
    let (dl, du) = (cube_to_host(&f.l_values), cube_to_host(&f.u_values));
    let (el, eu) = (rel_worst(&dl, &host.l_values), rel_worst(&du, &host.u_values));
    println!("CONUS n {} levels {levels}: worst factor diff, l {el:e}, u {eu:e}", adj.n);
    assert!(el <= 1e-6 && eu <= 1e-6, "CONUS: l {el:e} u {eu:e}");

    // Warm up, then time a batch of factors with a sync at the end.
    let dev = device();
    for _ in 0..5 {
        let _ = tree_factor_dev(&t, &a_cube);
    }
    <Gpu as Backend>::sync(&dev).unwrap();
    let reps = 20;
    let t0 = std::time::Instant::now();
    for _ in 0..reps {
        let _ = tree_factor_dev(&t, &a_cube);
    }
    <Gpu as Backend>::sync(&dev).unwrap();
    let per = t0.elapsed().as_secs_f64() * 1e3 / reps as f64;
    let t1 = std::time::Instant::now();
    let _ = tree_factor_dev(&t, &a_cube);
    <Gpu as Backend>::sync(&dev).unwrap();
    let single = t1.elapsed().as_secs_f64() * 1e3;
    let t2 = std::time::Instant::now();
    let _ = tree_lu_factor(&p, &a);
    let cpu = t2.elapsed().as_secs_f64() * 1e3;
    println!("CONUS factor: {per:.3} ms mean of {reps}, {single:.3} ms single synced; host factor {cpu:.3} ms");
}

#[test]
fn solves_match_host() {
    for (name, path) in [("Juniata", JUNIATA), ("Hudson", HUDSON)] {
        let (adj, p) = bundle(path);
        let (n, nnz) = (p.n, p.nnz());
        let a = random_tree_matrix(&adj, &p, 21, n / 7);
        let host = tree_lu_factor(&p, &a);
        let a_cube = cube_from_host(&a, &device());
        let t = dev_tree(&p, &a_cube);
        let f = tree_factor_dev(&t, &a_cube);

        let b = vec_rand(n, 22, -10.0, 10.0);
        let b_cube = cube_from_host(&b, &device());
        let x_dev = cube_to_host(&tree_solve_dev(&p, &f, &b_cube));
        let x_host = tree_lu_solve_host(&p, &host, &b);
        let ex = rel_worst(&x_dev, &x_host);

        let g_dev = cube_to_host(&tree_solve_transposed_dev(&p, &f, &b_cube));
        let g_host = tree_lu_solve_transposed_host(&p, &host, &b);
        let eg = rel_worst(&g_dev, &g_host);

        let x_cube = cube_from_host(&x_host, &device());
        let ga_dev = cube_to_host(&tree_grad_a_dev(&t, &b_cube, &x_cube));
        let ga_host = tree_lu_grad_a_host(&p, &b, &x_host);
        let ea = rel_worst(&ga_dev, &ga_host);
        println!("{name}: solve {ex:e}, transposed solve {eg:e}, grad a {ea:e}");
        assert!(ex <= 1e-6, "{name}: solve {ex:e}");
        assert!(eg <= 1e-6, "{name}: transposed solve {eg:e}");
        assert!(ea <= 1e-6, "{name}: grad a {ea:e}");

        assert_eq!(ga_dev.len(), 2 * nnz);
        same_zero_set(&ga_dev, &ga_host, "grad a");
        for i in 0..n {
            let s = nnz + diag_slot(&p, i);
            assert_eq!(ga_dev[s], 0.0, "{name}: dev diagonal slot {i}");
            assert_eq!(ga_host[s], 0.0, "{name}: host diagonal slot {i}");
        }
    }
}

// copied from tests/tree_lu_gradcheck.rs::{EDGES, N, tree, matrix}
const EDGES: [(usize, usize); 6] = [(0, 2), (1, 2), (2, 4), (3, 4), (4, 6), (5, 6)];
const N: usize = 7;

fn small_tree() -> (SparseAdjacency, Arc<CsrPattern>) {
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

/// Diagonally dominant matrix; edge 1 and 4 are one-way (upper half 0).
fn small_matrix(p: &CsrPattern, seed: u64) -> Vec<f32> {
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

// copied from tests/tree_lu_gradcheck.rs::t1 (generic in the backend)
fn t1<I: Backend>(v: &[f32], grad: bool) -> Tensor<Autodiff<I>, 1> {
    let t = Tensor::<Autodiff<I>, 1>::from_data(TensorData::new(v.to_vec(), [v.len()]), &Default::default());
    if grad {
        t.require_grad()
    } else {
        t
    }
}

struct Run {
    x: Vec<f32>,
    ga: Vec<f32>,
    gb: Vec<f32>,
}

fn run_op<I: Backend + 'static>(p: &Arc<CsrPattern>, a: &[f32], b: &[f32], w: &[f32], use_cuda: bool) -> Run
where
    I::FloatTensorPrimitive: 'static,
    I::Device: 'static,
{
    let (at, bt) = (t1::<I>(a, true), t1::<I>(b, true));
    let x = tree_lu_solve(p, at.clone(), bt.clone(), use_cuda);
    let loss = (x.clone() * t1::<I>(w, false)).sum();
    let grads = loss.backward();
    Run {
        x: x.into_data().to_vec::<f32>().unwrap(),
        ga: at.grad(&grads).unwrap().into_data().to_vec::<f32>().unwrap(),
        gb: bt.grad(&grads).unwrap().into_data().to_vec::<f32>().unwrap(),
    }
}

#[test]
fn tree_lu_solve_matches_cpu_forward_and_gradients() {
    let (_, small) = small_tree();
    let (jadj, jp) = bundle(JUNIATA);
    let cases: Vec<(&str, Arc<CsrPattern>, Vec<f32>)> = vec![
        ("gradcheck tree", small.clone(), small_matrix(&small, 1)),
        ("Juniata", jp.clone(), random_tree_matrix(&jadj, &jp, 31, jadj.n / 7)),
    ];
    for (name, p, a) in cases {
        let b = vec_rand(p.n, 2, -5.0, 5.0);
        let w = vec_rand(p.n, 5, -1.0, 1.0);
        let cpu = run_op::<Cpu>(&p, &a, &b, &w, false);
        let gpu = run_op::<Gpu>(&p, &a, &b, &w, true);
        let (ex, ea, eb) = (rel_worst(&gpu.x, &cpu.x), rel_worst(&gpu.ga, &cpu.ga), rel_worst(&gpu.gb, &cpu.gb));
        println!("{name}: x {ex:e}, grad a {ea:e}, grad b {eb:e}");
        assert!(ex <= XBACKEND_REL_TOL, "{name}: x {ex:e}");
        assert!(ea <= XBACKEND_REL_TOL, "{name}: grad a {ea:e}");
        assert!(eb <= XBACKEND_REL_TOL, "{name}: grad b {eb:e}");
        same_zero_set(&gpu.ga, &cpu.ga, &format!("{name} grad a"));
    }
}

fn reads_of(p: &Arc<CsrPattern>, a: &[f32], b: &[f32], w: &[f32], use_cuda: bool) -> (u64, u64) {
    // Inputs and weights are built before the first counter read.
    let (at, bt, wt) = (t1::<Gpu>(a, true), t1::<Gpu>(b, true), t1::<Gpu>(w, false));
    let start = host_reads_on_this_thread_for_test();
    let x = tree_lu_solve(p, at.clone(), bt.clone(), use_cuda);
    let after_forward = host_reads_on_this_thread_for_test();
    let loss = (x * wt).sum();
    let grads = loss.backward();
    let after_backward = host_reads_on_this_thread_for_test();
    assert!(at.grad(&grads).is_some() && bt.grad(&grads).is_some());
    (after_forward - start, after_backward - after_forward)
}

#[test]
fn tree_lu_solve_makes_no_host_copy_on_cuda() {
    let (jadj, p) = bundle(JUNIATA);
    let a = random_tree_matrix(&jadj, &p, 41, jadj.n / 7);
    let b = vec_rand(p.n, 42, -5.0, 5.0);
    let w = vec_rand(p.n, 43, -1.0, 1.0);
    // Control: the host path on a CUDA backend copies, and the counter sees it.
    let (cf, cb) = reads_of(&p, &a, &b, &w, false);
    println!("host path on Gpu: forward reads {cf}, backward reads {cb}");
    assert!(cf > 0 && cb > 0, "control saw no host reads: the counter is blind");
    let (f, bw) = reads_of(&p, &a, &b, &w, true);
    println!("device path: forward reads {f}, backward reads {bw}");
    assert_eq!(f, 0, "forward made host copies");
    assert_eq!(bw, 0, "backward made host copies");
}

// copied from tests/tree_lu_gradcheck.rs::{loss_of, gradcheck_a_and_b_by_central_differences}
fn loss_of(p: &Arc<CsrPattern>, a: &[f32], b: &[f32], w: &[f32]) -> f32 {
    let x = tree_lu_solve_host(p, &tree_lu_factor(p, a), b);
    x.iter().zip(w).map(|(x, w)| x * w).sum()
}

#[test]
fn gradcheck_a_and_b_on_cuda() {
    let (_, p) = small_tree();
    let nnz = p.nnz();
    let a = small_matrix(&p, 3);
    let b = vec_rand(N, 4, -5.0, 5.0);
    let w = vec_rand(N, 5, -1.0, 1.0);
    let run = run_op::<Gpu>(&p, &a, &b, &w, true);
    let (ga, gb) = (run.ga, run.gb);
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
        let eps = (1e-3 * a[i].abs()).max(1e-3);
        let (mut hi, mut lo) = (a.clone(), a.clone());
        hi[i] += eps;
        lo[i] -= eps;
        let fd = (loss_of(&p, &hi, &b, &w) - loss_of(&p, &lo, &b, &w)) / (2.0 * eps);
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

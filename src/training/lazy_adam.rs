//! Row-sparse ("lazy") Adam for the per-dam free parameters
//! (`crate::nn::dam_params::DamParams`: rule-curve logits `θ [n_dams, 4]`,
//! `T0` multipliers `δ [n_dams]`).
//!
//! A dam's parameters get a gradient only in a step whose batch has a gauge
//! below that dam, so the gradient is row-sparse: `select`'s backward hands
//! every other row an exact zero. Burn's dense Adam keeps ONE step counter
//! per tensor and decays every row's moments each step, so a row that had a
//! gradient once keeps moving on its stale first moment for ~10 more steps
//! with no gradient at all (review v2, finding 2). This optimizer updates a
//! row only in a step where that row's gradient is nonzero (any of its
//! entries), with a per-row step counter for the bias correction; a row that
//! never gets a gradient stays exactly at its init, moments and all.
//!
//! The per-row arithmetic is Burn's `AdaptiveMomentum` (burn-optim
//! `adam.rs`) op for op, in f32:
//!
//! ```text
//! m ← β1·m + (1 − β1)·g        v ← β2·v + (1 − β2)·g²       t ← t + 1
//! s = sqrt(1 − β2^t)           p ← p − lr · (m · s/(1 − β1^t)) / (sqrt(v) + ε·s)
//! ```
//!
//! with β1 = 0.9, β2 = 0.999, ε = 1e-8 (`build_head_optimizer`'s Adam), so a
//! row that gets a gradient every step follows dense Adam bit for bit
//! (`tests` below). The parameters are a few thousand floats, so the update
//! runs on the host.
//!
//! The state is saved as JSON with every f32 as its bit pattern
//! (`release_dams_optim.json` in a checkpoint), so a resume restores it
//! bitwise.

use std::path::Path;

use burn::module::Param;
use burn::optim::GradientsParams;
use burn::tensor::{backend::AutodiffBackend, Tensor, TensorData};
use serde::{Deserialize, Serialize};

use crate::data::error::{DataError, Result};
use crate::nn::dam_params::DamParams;

/// Adam's β1, β2, ε, as `training::optimizer::build_head_optimizer` builds it.
pub const BETA_1: f32 = 0.9;
pub const BETA_2: f32 = 0.999;
pub const EPSILON: f32 = 1e-8;

/// One parameter's per-row Adam state: rows of `width` entries.
#[derive(Clone, Debug, PartialEq)]
pub struct RowAdamState {
    /// Entries per row (4 for `θ`, 1 for `δ`).
    pub width: usize,
    /// First moment, row-major `[n_rows · width]`.
    pub m: Vec<f32>,
    /// Second moment, row-major `[n_rows · width]`.
    pub v: Vec<f32>,
    /// Updates each row has taken: its bias-correction step.
    pub t: Vec<u32>,
}

impl RowAdamState {
    fn zeros(n_rows: usize, width: usize) -> Self {
        Self { width, m: vec![0.0; n_rows * width], v: vec![0.0; n_rows * width], t: vec![0; n_rows] }
    }

    /// Update every row of `p` whose gradient `g` has a nonzero entry;
    /// return how many rows that was.
    fn step(&mut self, lr: f32, p: &mut [f32], g: &[f32]) -> usize {
        let w = self.width;
        assert_eq!(p.len(), self.m.len(), "parameter size changed under the optimizer");
        assert_eq!(g.len(), p.len(), "gradient size != parameter size");
        let mut touched = 0;
        for r in 0..self.t.len() {
            let span = r * w..(r + 1) * w;
            if g[span.clone()].iter().all(|&x| x == 0.0) {
                continue;
            }
            touched += 1;
            self.t[r] += 1;
            let time = self.t[r] as i32;
            let bias_correction2_sqrt = (1.0 - BETA_2.powi(time)).sqrt();
            let combined_factor = bias_correction2_sqrt / (1.0 - BETA_1.powi(time));
            for i in span {
                self.m[i] = self.m[i] * BETA_1 + g[i] * (1.0 - BETA_1);
                self.v[i] = self.v[i] * BETA_2 + (g[i] * g[i]) * (1.0 - BETA_2);
                let update =
                    (self.m[i] * combined_factor) / (self.v[i].sqrt() + EPSILON * bias_correction2_sqrt);
                p[i] -= update * lr;
            }
        }
        touched
    }
}

/// Row-sparse Adam over a [`DamParams`] module. See the module docs.
#[derive(Clone, Debug, PartialEq)]
pub struct LazyAdam {
    /// State of `θ` (`None` without a rule curve).
    pub theta: Option<RowAdamState>,
    /// State of `δ` (`None` without per-dam `T0`).
    pub delta: Option<RowAdamState>,
}

/// Rows each parameter updated in one [`LazyAdam::step`].
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct RowsTouched {
    pub theta: usize,
    pub delta: usize,
}

impl LazyAdam {
    /// Zero state for `params` (all rows untouched).
    pub fn new<B: burn::tensor::backend::Backend>(params: &DamParams<B>) -> Self {
        let n = params.n_dams();
        Self {
            theta: params.theta.as_ref().map(|_| RowAdamState::zeros(n, 4)),
            delta: params.delta.as_ref().map(|_| RowAdamState::zeros(n, 1)),
        }
    }

    /// One step at learning rate `lr`: every row with a nonzero gradient in
    /// `grads` updates; every other row, and its moments and counter, is
    /// left exactly as it was.
    pub fn step<B: AutodiffBackend>(
        &mut self,
        lr: f32,
        mut params: DamParams<B>,
        grads: &GradientsParams,
    ) -> (DamParams<B>, RowsTouched) {
        let mut touched = RowsTouched::default();
        if let (Some(th), Some(st)) = (params.theta.take(), self.theta.as_mut()) {
            let (th, k) = step_param::<B, 2>(th, st, lr, grads);
            params.theta = Some(th);
            touched.theta = k;
        }
        if let (Some(de), Some(st)) = (params.delta.take(), self.delta.as_mut()) {
            let (de, k) = step_param::<B, 1>(de, st, lr, grads);
            params.delta = Some(de);
            touched.delta = k;
        }
        (params, touched)
    }

    /// Write the state as JSON, every f32 as its bit pattern.
    pub fn save(&self, path: &Path) -> Result<()> {
        let io = |e: String| DataError::Io {
            path: path.to_path_buf(),
            source: std::io::Error::new(std::io::ErrorKind::Other, e),
        };
        let json = serde_json::to_string(&LazyAdamRecord::from(self)).map_err(|e| io(e.to_string()))?;
        std::fs::write(path, json).map_err(|source| DataError::Io { path: path.to_path_buf(), source })
    }

    /// Read a state written by [`Self::save`]. Errors when its shape does not
    /// match `self` (a different dam table or option set).
    pub fn load(self, path: &Path) -> Result<Self> {
        let err = |e: String| DataError::Io {
            path: path.to_path_buf(),
            source: std::io::Error::new(std::io::ErrorKind::Other, e),
        };
        let json = std::fs::read_to_string(path).map_err(|source| DataError::Io { path: path.to_path_buf(), source })?;
        let rec: LazyAdamRecord = serde_json::from_str(&json).map_err(|e| err(e.to_string()))?;
        let loaded = LazyAdam::from(rec);
        let shape = |s: &Option<RowAdamState>| s.as_ref().map(|s| (s.width, s.t.len()));
        if shape(&loaded.theta) != shape(&self.theta) || shape(&loaded.delta) != shape(&self.delta) {
            return Err(err(format!(
                "per-dam optimizer state has shapes theta {:?} / delta {:?}, the run expects {:?} / {:?}",
                shape(&loaded.theta),
                shape(&loaded.delta),
                shape(&self.theta),
                shape(&self.delta)
            )));
        }
        Ok(loaded)
    }
}

/// Step one parameter tensor: host copy, row-sparse update, back to a tensor
/// with the same id and `require_grad`.
fn step_param<B: AutodiffBackend, const D: usize>(
    param: Param<Tensor<B, D>>,
    state: &mut RowAdamState,
    lr: f32,
    grads: &GradientsParams,
) -> (Param<Tensor<B, D>>, usize) {
    let Some(g) = grads.get::<B::InnerBackend, D>(param.id) else {
        return (param, 0);
    };
    let g: Vec<f32> = g.into_data().convert::<f32>().to_vec().expect("f32 gradient");
    let mut touched = 0;
    let param = param.map(|t| {
        let require_grad = t.is_require_grad();
        let shape = t.shape();
        let device = t.device();
        let mut p: Vec<f32> = t.into_data().convert::<f32>().to_vec().expect("f32 parameter");
        touched = state.step(lr, &mut p, &g);
        let inner = Tensor::<B::InnerBackend, D>::from_data(TensorData::new(p, shape), &device);
        let out = Tensor::<B, D>::from_inner(inner);
        if require_grad {
            out.require_grad()
        } else {
            out
        }
    });
    (param, touched)
}

/// On-disk form: f32 as bit patterns, so JSON round-trips bitwise.
#[derive(Serialize, Deserialize)]
struct RowAdamRecord {
    width: usize,
    m: Vec<u32>,
    v: Vec<u32>,
    t: Vec<u32>,
}

#[derive(Serialize, Deserialize)]
struct LazyAdamRecord {
    /// `(β1, β2, ε)` as bits, for the reader's eye; the constants are fixed.
    betas_eps: [u32; 3],
    theta: Option<RowAdamRecord>,
    delta: Option<RowAdamRecord>,
}

impl From<&LazyAdam> for LazyAdamRecord {
    fn from(a: &LazyAdam) -> Self {
        let bits = |v: &[f32]| v.iter().map(|x| x.to_bits()).collect();
        let row = |s: &RowAdamState| RowAdamRecord { width: s.width, m: bits(&s.m), v: bits(&s.v), t: s.t.clone() };
        Self {
            betas_eps: [BETA_1.to_bits(), BETA_2.to_bits(), EPSILON.to_bits()],
            theta: a.theta.as_ref().map(row),
            delta: a.delta.as_ref().map(row),
        }
    }
}

impl From<LazyAdamRecord> for LazyAdam {
    fn from(r: LazyAdamRecord) -> Self {
        let floats = |v: Vec<u32>| v.into_iter().map(f32::from_bits).collect();
        let row = |s: RowAdamRecord| RowAdamState { width: s.width, m: floats(s.m), v: floats(s.v), t: s.t };
        Self { theta: r.theta.map(row), delta: r.delta.map(row) }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use burn::backend::{Autodiff, NdArray};
    use burn::optim::Optimizer;

    type I = NdArray<f32>;
    type AB = Autodiff<I>;

    fn host2(t: Tensor<AB, 2>) -> Vec<f32> {
        t.into_data().to_vec().unwrap()
    }

    /// Gradients for `params` with the given host values (`θ` row-major, `δ`).
    fn grads_for(params: &DamParams<AB>, theta_g: &[f32], delta_g: &[f32]) -> GradientsParams {
        let d = Default::default();
        let n = params.n_dams();
        let mut g = GradientsParams::new();
        if let Some(t) = params.theta.as_ref() {
            g.register::<I, 2>(t.id, Tensor::<I, 1>::from_floats(theta_g, &d).reshape([n, 4]));
        }
        if let Some(t) = params.delta.as_ref() {
            g.register::<I, 1>(t.id, Tensor::<I, 1>::from_floats(delta_g, &d));
        }
        g
    }

    /// A deterministic, sign-varying gradient for step `k`, entry `i`.
    fn gval(k: usize, i: usize) -> f32 {
        ((k * 7 + i * 3) as f32 * 0.37).sin() * (1.0 + 0.5 * i as f32)
    }

    #[test]
    fn untouched_rows_stay_bitwise_at_init() {
        let d = Default::default();
        let mut params = DamParams::<AB>::zeros(3, true, true, &d);
        let mut opt = LazyAdam::new(&params);
        for k in 0..6 {
            // Rows 0 and 2 get gradients; row 1 never does. Row 2 skips odd steps.
            let mut th = vec![0.0_f32; 12];
            let mut de = vec![0.0_f32; 3];
            for j in 0..4 {
                th[j] = gval(k, j);
                if k % 2 == 0 {
                    th[8 + j] = gval(k, 8 + j);
                }
            }
            de[0] = gval(k, 20);
            let before_row2: Vec<f32> = host2(params.theta.as_ref().unwrap().val())[8..12].to_vec();
            let g = grads_for(&params, &th, &de);
            let (p, touched) = opt.step(0.05, params, &g);
            params = p;
            assert_eq!(touched.theta, if k % 2 == 0 { 2 } else { 1 }, "step {k}");
            assert_eq!(touched.delta, 1, "step {k}");
            let theta = host2(params.theta.as_ref().unwrap().val());
            for j in 4..8 {
                assert_eq!(theta[j].to_bits(), 0.0_f32.to_bits(), "step {k}: untouched theta row 1 moved");
            }
            if k % 2 == 1 {
                assert_eq!(&theta[8..12], before_row2.as_slice(), "step {k}: row 2 moved with no gradient");
            }
            let delta: Vec<f32> = params.delta.as_ref().unwrap().val().into_data().to_vec().unwrap();
            assert_eq!((delta[1].to_bits(), delta[2].to_bits()), (0, 0), "step {k}: untouched delta rows moved");
            assert!(
                theta[0..4].iter().any(|&x| x != 0.0) && delta[0] != 0.0,
                "step {k}: touched rows did not move"
            );
        }
        let st = opt.theta.as_ref().unwrap();
        assert_eq!(st.t, vec![6, 0, 3]);
        assert!(st.m[4..8].iter().chain(&st.v[4..8]).all(|&x| x.to_bits() == 0));
        assert_eq!(opt.delta.as_ref().unwrap().t, vec![6, 0, 0]);
    }

    #[test]
    fn a_row_with_a_gradient_every_step_follows_dense_adam() {
        // One row, gradient every step: lazy Adam must reproduce Burn's
        // dense Adam as `build_head_optimizer` builds it.
        let d = Default::default();
        let mut lazy_p = DamParams::<AB>::zeros(1, true, true, &d);
        let mut dense_p = lazy_p.clone();
        let mut lazy = LazyAdam::new(&lazy_p);
        let mut dense = crate::training::optimizer::build_head_optimizer::<DamParams<AB>, AB>(
            crate::config::OptimizerKind::Adam,
        );
        let mut max_rel = 0.0_f64;
        let mut n_diff_bits = 0;
        for k in 0..25 {
            let th: Vec<f32> = (0..4).map(|j| gval(k, j)).collect();
            let de = [gval(k, 9)];
            let g = grads_for(&lazy_p, &th, &de);
            lazy_p = lazy.step(0.05, lazy_p, &g).0;
            dense_p = dense.step(0.05, dense_p.clone(), grads_for(&dense_p, &th, &de));
            let a = host2(lazy_p.theta.as_ref().unwrap().val());
            let b = host2(dense_p.theta.as_ref().unwrap().val());
            let da: Vec<f32> = lazy_p.delta.as_ref().unwrap().val().into_data().to_vec().unwrap();
            let db: Vec<f32> = dense_p.delta.as_ref().unwrap().val().into_data().to_vec().unwrap();
            for (x, y) in a.iter().chain(&da).zip(b.iter().chain(&db)) {
                n_diff_bits += (x.to_bits() != y.to_bits()) as usize;
                max_rel = max_rel.max(((x - y).abs() / x.abs().max(y.abs()).max(1e-12)) as f64);
            }
        }
        println!("lazy vs dense Adam, 25 steps: {n_diff_bits} of 125 values differ bitwise, max rel {max_rel:.3e}");
        // Same f32 ops in the same order as Burn's AdaptiveMomentum: bitwise.
        assert_eq!(n_diff_bits, 0, "lazy Adam diverged from dense Adam on an always-touched row: {max_rel:.3e}");
    }

    #[test]
    fn state_saves_and_restores_bitwise() {
        let d = Default::default();
        let mut params = DamParams::<AB>::zeros(4, true, true, &d);
        let mut opt = LazyAdam::new(&params);
        for k in 0..5 {
            let th: Vec<f32> = (0..16).map(|i| if i / 4 == 2 { 0.0 } else { gval(k, i) }).collect();
            let de: Vec<f32> = (0..4).map(|i| if i == 3 { 0.0 } else { gval(k, 30 + i) }).collect();
            let g = grads_for(&params, &th, &de);
            params = opt.step(0.02, params, &g).0;
        }
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("release_dams_optim.json");
        opt.save(&path).unwrap();
        let restored = LazyAdam::new(&params).load(&path).unwrap();
        let bits = |s: &RowAdamState| {
            (s.m.iter().map(|x| x.to_bits()).collect::<Vec<_>>(), s.v.iter().map(|x| x.to_bits()).collect::<Vec<_>>(), s.t.clone())
        };
        assert_eq!(bits(opt.theta.as_ref().unwrap()), bits(restored.theta.as_ref().unwrap()));
        assert_eq!(bits(opt.delta.as_ref().unwrap()), bits(restored.delta.as_ref().unwrap()));
        assert_eq!(opt, restored);

        // One more step from each: identical parameters, bit for bit.
        let th: Vec<f32> = (0..16).map(|i| gval(9, i)).collect();
        let de: Vec<f32> = (0..4).map(|i| gval(9, 40 + i)).collect();
        let g = grads_for(&params, &th, &de);
        let (a, _) = opt.clone().step(0.02, params.clone(), &g);
        let (b, _) = restored.clone().step(0.02, params.clone(), &g);
        let (a, b) = (host2(a.theta.unwrap().val()), host2(b.theta.unwrap().val()));
        assert!(a.iter().zip(&b).all(|(x, y)| x.to_bits() == y.to_bits()));

        // A state for a different dam count is refused.
        let other = DamParams::<AB>::zeros(5, true, true, &d);
        assert!(LazyAdam::new(&other).load(&path).is_err());
    }
}

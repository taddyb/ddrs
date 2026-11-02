//! Fused multiply-add as two small autodiff ops, for `StageBlock::step_reference`
//! (ruling F, `fma-ruling.md` in the SDD workspace).
//!
//! Burn has no fma tensor op. The fused stage op and the CUDA kernels fuse every
//! `a*b + c` site (rule F1); the reference has to round the same way to stay
//! bitwise equal. `fma3` is `a*b + c` with one rounding per element, and
//! `spmv_fma` is the row sum `out[row] = fma(x[col], w, out[row])` over the CSR
//! slots in slot order. Both run the forward on host data with `f32::mul_add`
//! (IEEE fused, the same bits as CUDA `fma`). The backward is the plain
//! derivative: `g*b`, `g*a`, `g` for `fma3`; the scatter and gather of `g*w`
//! and `g*x` for `spmv_fma`. This is the CsrSolveOp recipe of
//! `docs/book/reference/burn-autograd.md`. The reference runs on NdArray in the
//! tests, so the host round trip costs nothing that matters.

use std::sync::Arc;

use burn::backend::autodiff::checkpoint::base::Checkpointer;
use burn::backend::autodiff::checkpoint::strategy::NoCheckpointing;
use burn::backend::autodiff::grads::Gradients;
use burn::backend::autodiff::ops::{Backward, Ops, OpsKind};
use burn::backend::Autodiff;
use burn::tensor::backend::Backend;
use burn::tensor::{Tensor, TensorData, TensorPrimitive};

use crate::sparse::{primitive_to_vec, CsrPattern};

fn float<I: Backend>(t: Tensor<Autodiff<I>, 1>) -> <Autodiff<I> as burn::tensor::backend::BackendTypes>::FloatTensorPrimitive {
    match t.into_primitive() {
        TensorPrimitive::Float(a) => a,
        TensorPrimitive::QFloat(_) => panic!("expected float tensor"),
    }
}

fn from_host<I: Backend>(v: Vec<f32>, like: &I::FloatTensorPrimitive) -> I::FloatTensorPrimitive {
    let device = I::float_device(like);
    let n = v.len();
    I::float_from_data(TensorData::new(v, [n]), &device)
}

// ------------------------------------------------------------------- fma3

#[derive(Debug)]
struct FmaOp;

impl<B: Backend + 'static> Backward<B, 3> for FmaOp
where
    B::FloatTensorPrimitive: 'static,
{
    type State = (B::FloatTensorPrimitive, B::FloatTensorPrimitive);

    fn backward(self, ops: Ops<Self::State, 3>, grads: &mut Gradients, _checkpointer: &mut Checkpointer) {
        let (a, b) = ops.state;
        let [pa, pb, pc] = ops.parents;
        let g = grads.consume::<B>(&ops.node);
        if let Some(p) = pa {
            grads.register::<B>(p.id, B::float_mul(g.clone(), b));
        }
        if let Some(p) = pb {
            grads.register::<B>(p.id, B::float_mul(g.clone(), a));
        }
        if let Some(p) = pc {
            grads.register::<B>(p.id, g);
        }
    }
}

/// `a*b + c` per element with one rounding. All three tensors have the same length.
pub(crate) fn fma3<I: Backend + 'static>(
    a: Tensor<Autodiff<I>, 1>,
    b: Tensor<Autodiff<I>, 1>,
    c: Tensor<Autodiff<I>, 1>,
) -> Tensor<Autodiff<I>, 1>
where
    I::FloatTensorPrimitive: 'static,
{
    let (a, b, c) = (float::<I>(a), float::<I>(b), float::<I>(c));
    let (va, vb, vc) = (
        primitive_to_vec::<I>(a.primitive.clone()),
        primitive_to_vec::<I>(b.primitive.clone()),
        primitive_to_vec::<I>(c.primitive.clone()),
    );
    assert!(va.len() == vb.len() && vb.len() == vc.len(), "fma3: operand lengths differ");
    let out: Vec<f32> = va.iter().zip(&vb).zip(&vc).map(|((x, y), z)| x.mul_add(*y, *z)).collect();
    let out = from_host::<I>(out, &a.primitive);
    let result = match FmaOp
        .prepare::<NoCheckpointing>([a.node.clone(), b.node.clone(), c.node.clone()])
        .compute_bound()
        .stateful()
    {
        OpsKind::Tracked(prep) => prep.finish((a.primitive.clone(), b.primitive.clone()), out),
        OpsKind::UnTracked(prep) => prep.finish(out),
    };
    Tensor::from_primitive(TensorPrimitive::Float(result))
}

/// `a*s + c` with a constant `s`.
pub(crate) fn fma_s<I: Backend + 'static>(a: Tensor<Autodiff<I>, 1>, s: f32, c: Tensor<Autodiff<I>, 1>) -> Tensor<Autodiff<I>, 1>
where
    I::FloatTensorPrimitive: 'static,
{
    let b = Tensor::<Autodiff<I>, 1>::full([a.dims()[0]], s, &a.device());
    fma3::<I>(a, b, c)
}

// --------------------------------------------------------------- spmv_fma

#[derive(Debug)]
struct SpmvFmaOp;

#[derive(Clone, Debug)]
struct SpmvState {
    x: Vec<f32>,
    w: Vec<f32>,
    pattern: Arc<CsrPattern>,
}

impl<B: Backend + 'static> Backward<B, 2> for SpmvFmaOp
where
    B::FloatTensorPrimitive: 'static,
{
    type State = SpmvState;

    fn backward(self, ops: Ops<Self::State, 2>, grads: &mut Gradients, _checkpointer: &mut Checkpointer) {
        let SpmvState { x, w, pattern } = ops.state;
        let [px, pw] = ops.parents;
        let grad_out = grads.consume::<B>(&ops.node);
        let device = B::float_device(&grad_out);
        let g = primitive_to_vec::<B>(grad_out);
        if let Some(p) = px {
            let mut gx = vec![0.0f32; x.len()];
            for k in 0..pattern.nnz() {
                gx[pattern.col[k] as usize] += g[pattern.row_for_nnz[k] as usize] * w[k];
            }
            let n = gx.len();
            grads.register::<B>(p.id, B::float_from_data(TensorData::new(gx, [n]), &device));
        }
        if let Some(p) = pw {
            let gw: Vec<f32> =
                (0..pattern.nnz()).map(|k| g[pattern.row_for_nnz[k] as usize] * x[pattern.col[k] as usize]).collect();
            let n = gw.len();
            grads.register::<B>(p.id, B::float_from_data(TensorData::new(gw, [n]), &device));
        }
    }
}

/// `out[row[k]] = fma(x[col[k]], w[k], out[row[k]])` over the slots in slot
/// order, from zero. `x` has length `n`, `w` length `nnz`.
pub(crate) fn spmv_fma<I: Backend + 'static>(
    pattern: &Arc<CsrPattern>,
    x: Tensor<Autodiff<I>, 1>,
    w: Tensor<Autodiff<I>, 1>,
) -> Tensor<Autodiff<I>, 1>
where
    I::FloatTensorPrimitive: 'static,
{
    let (x, w) = (float::<I>(x), float::<I>(w));
    let (vx, vw) = (primitive_to_vec::<I>(x.primitive.clone()), primitive_to_vec::<I>(w.primitive.clone()));
    let mut out = vec![0.0f32; pattern.n];
    for k in 0..pattern.nnz() {
        let r = pattern.row_for_nnz[k] as usize;
        out[r] = vx[pattern.col[k] as usize].mul_add(vw[k], out[r]);
    }
    let out = from_host::<I>(out, &x.primitive);
    let result = match SpmvFmaOp.prepare::<NoCheckpointing>([x.node.clone(), w.node.clone()]).compute_bound().stateful() {
        OpsKind::Tracked(prep) => prep.finish(SpmvState { x: vx, w: vw, pattern: pattern.clone() }, out),
        OpsKind::UnTracked(prep) => prep.finish(out),
    };
    Tensor::from_primitive(TensorPrimitive::Float(result))
}

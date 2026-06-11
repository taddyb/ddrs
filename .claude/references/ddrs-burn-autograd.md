---
name: ddrs-burn-autograd
description: BURN 0.21 recipe for registering a custom Backward op from a downstream crate. Visibility map of public/private autograd types; canonical Backward<B, N> pattern.
output: reference/burn-autograd.md
sources:
  - src/sparse/mod.rs
  - src/routing/mmc_op.rs
---

# BURN 0.21 — registering a custom Backward op from a downstream crate

## What to know

BURN 0.21 exposes most of the autograd plumbing publicly, but
`NodeRef` is `pub(crate)`. You cannot *name* it, but you can *pass values*
of that type by type-inference (via `AutodiffTensor.node`) — that's enough
to register a custom `Backward<B, N>` from a downstream crate like ddrs.
ddrs uses this for `CsrSolveOp` (in `src/sparse/mod.rs`) and `TimestepOp`
(in `src/routing/mmc_op.rs`).

Verified by the `sparse_gradcheck` and `sp8_gradcheck` integration tests
(see Verification below).

### Visibility map

```
burn::backend::autodiff::
    ops::{Backward, Ops, OpsKind, binary, unary}       ✓ public
    grads::Gradients                                    ✓ public
    checkpoint::base::Checkpointer                      ✓ public
    checkpoint::strategy::NoCheckpointing               ✓ public
    graph::NodeRef                                      ✗ pub(crate) — UNNAMEABLE from outside

burn::backend::Autodiff<B>                              ✓ public
<Autodiff<B> as BackendTypes>::FloatTensorPrimitive
    = AutodiffTensor<B> (pub fields: .primitive, .node) ✓ values reachable
```

You cannot *name* `NodeRef`, but you can *pass values* of that type by inference
(via `AutodiffTensor.node`). That's enough to call `Backward::prepare`.

## Canonical pattern (1 input, 1 output)

```rust
use burn::backend::autodiff::checkpoint::base::Checkpointer;
use burn::backend::autodiff::checkpoint::strategy::NoCheckpointing;
use burn::backend::autodiff::grads::Gradients;
use burn::backend::autodiff::ops::{Backward, Ops, OpsKind};
use burn::backend::{Autodiff, NdArray};
use burn::tensor::backend::Backend;
use burn::tensor::ops::FloatTensorOps;          // <- needed for B::float_* methods
use burn::tensor::{Tensor, TensorPrimitive};

#[derive(Debug)]
struct MyOp;

impl<B: Backend> Backward<B, 1> for MyOp {       // N = number of inputs
    type State = B::FloatTensorPrimitive;        // anything Clone + Send + Debug + 'static

    fn backward(self, ops: Ops<Self::State, 1>, grads: &mut Gradients, _: &mut Checkpointer) {
        let state = ops.state;
        let [parent] = ops.parents;
        if let Some(parent) = parent {
            let grad_out = grads.consume::<B>(&ops.node);
            let grad_in = /* analytical adjoint using state and grad_out, in B::float_* ops */;
            grads.register::<B>(parent.id, grad_in);
        }
    }
}

fn my_op<B: Backend>(x: Tensor<Autodiff<B>, D>) -> Tensor<Autodiff<B>, D> {
    let at = match x.into_primitive() {
        TensorPrimitive::Float(p) => p,
        TensorPrimitive::QFloat(_) => panic!(),
    };
    let output_inner = forward_on_inner_backend(at.primitive.clone());
    let saved_state = at.primitive.clone();              // for backward

    let result = match MyOp
        .prepare::<NoCheckpointing>([at.node.clone()])   // NodeRef passed by inference
        .compute_bound()                                 // or .memory_bound() for recompute
        .stateful()                                      // or .stateless() if State = ()
    {
        OpsKind::Tracked(prep)   => prep.finish(saved_state, output_inner),
        OpsKind::UnTracked(prep) => prep.finish(output_inner),     // grad not needed downstream
    };
    Tensor::from_primitive(TensorPrimitive::Float(result))
}
```

## Gotchas

- **`<NdArray<f32> as Backend>::float_mul` fails.** `float_mul` is on `FloatTensorOps`,
  a supertrait of `Backend`. Use bare `B::float_mul(...)` syntax (works via supertrait
  method dispatch when `FloatTensorOps` is in scope), or cast through the right trait.
- **Scalars: `2.0f32.elem()` is wrong here.** `float_mul_scalar` wants a `Scalar` enum;
  use `(2.0f32).into()` (there's a `From<f32> for Scalar` impl).
- **Method-takes-self moves.** `Tensor::sum`, `into_primitive`, etc. consume the tensor.
  Clone defensively in user code: `y.clone().sum()`.
- **N inputs:** `Backward<B, N>`. `ops.parents: [Option<NodeRef>; N]`. Pass
  `[node1, node2, ...]` to `prepare`. Each `Option` is `Some` iff that input was tracked.
- **Saving multiple tensors:** make `State = (B::FloatTensorPrimitive, ...)` a tuple, or
  use `Checkpointer` if you'd rather recompute than store.

## Implications for ddrs

The CSR triangular-solve port is unblocked. State for the analytical adjoint is
`CsrSolveState { a_values, x, pattern: Arc<CsrPattern>, use_cuda }` — the structural
CSR arrays live behind a shared `Arc<CsrPattern>` (so per-timestep cloning is a
refcount bump, not an O(nnz) copy onto the tape), not loose `Vec<i32>` index arrays
in State. `CsrSolveOp` is therefore `Backward<B, 2>` (two parents: `a_values` and
`b`). The backward will:

1. Build CSR `A` from `a_values` + the `Arc<CsrPattern>` on the inner backend (no autograd).
2. Solve `A^T · gradb = grad_out` via upper-triangular substitution.
3. Compute `gradA_values[k] = -gradb[row(k)] * x[col(k)]` for each non-zero.
4. Register `gradA_values` → `a_values` parent, `gradb` → `b` parent.

This mirrors `~/projects/ddr/src/ddr/routing/utils.py:515` (`TriangularSparseSolver`) 1:1.

## Verification

```bash
cargo test --test sparse_gradcheck
cargo test --test sp8_gradcheck
```

`sparse_gradcheck` validates the hand-written `Backward` for `CsrSolveOp`
against DDR's `TriangularSparseSolver` adjoint fixtures; `sp8_gradcheck`
checks the fused timestep op against central finite differences over all
five tracked parents. Neither test is `#[ignore]`d, so plain `cargo test`
runs them (no `-- --ignored` needed).

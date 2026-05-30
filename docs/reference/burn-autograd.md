# BURN autograd recipe

This chapter is the BURN 0.21 recipe for registering a custom
`Backward` op from a downstream crate like ddrs. It is the load-bearing
plumbing that lets `CsrSolveOp` (in `src/sparse/mod.rs`) and
`TimestepOp` (in `src/routing/mmc_op.rs`) collapse what would otherwise
be O(n²) or O(33-ops-per-step) autograd tape entries into a single
`Backward<I, N>` node each. Without this recipe, the differentiable
side of ddrs would be untrainable at CONUS scale.

The recipe was verified by `spike_backward/visibility_check.rs` (a
toy square op; `dy/dx = 2x`) before being used in production.

## What to know

BURN 0.21 exposes most of the autograd plumbing publicly, but
`NodeRef` is `pub(crate)`. You cannot *name* it, but you can *pass
values* of that type by type-inference (via `AutodiffTensor.node`) —
that's enough to register a custom `Backward<B, N>` from a downstream
crate like ddrs.

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

You cannot *name* `NodeRef`, but you can *pass values* of that type by
inference (via `AutodiffTensor.node`). That's enough to call
`Backward::prepare`.

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

The shape of this code is: convert the autodiff tensor to its inner
primitive, do the forward work on the inner backend (so no autodiff
nodes are added inside), save whatever state the analytical adjoint
will need, then call `MyOp.prepare(...).stateful()` to register the
single backward node. The `OpsKind::Tracked` vs `OpsKind::UnTracked`
branch picks between "downstream needs the gradient" and "the result
will be detached, skip the bookkeeping".

## Gotchas

- **`<NdArray<f32> as Backend>::float_mul` fails.** `float_mul` is on
  `FloatTensorOps`, a supertrait of `Backend`. Use bare
  `B::float_mul(...)` syntax (works via supertrait method dispatch
  when `FloatTensorOps` is in scope), or cast through the right
  trait.
- **Scalars: `2.0f32.elem()` is wrong here.** `float_mul_scalar`
  wants a `Scalar` enum; use `(2.0f32).into()` (there's a
  `From<f32> for Scalar` impl).
- **Method-takes-self moves.** `Tensor::sum`, `into_primitive`, etc.
  consume the tensor. Clone defensively in user code:
  `y.clone().sum()`.
- **N inputs:** `Backward<B, N>`. `ops.parents: [Option<NodeRef>;
  N]`. Pass `[node1, node2, ...]` to `prepare`. Each `Option` is
  `Some` iff that input was tracked.
- **Saving multiple tensors:** make `State =
  (B::FloatTensorPrimitive, ...)` a tuple, or use `Checkpointer` if
  you'd rather recompute than store.

## Implications for ddrs

The CSR triangular-solve port is unblocked. State for the analytical
adjoint is `(A_values_primitive, x_primitive, crow: Vec<i32>, col:
Vec<i32>)` — the index arrays are not tensors, just `Vec`s we clone
into State. The backward will:

1. Build CSR `A` from `(A_values, crow, col)` on the inner backend
   (no autograd).
2. Solve `A^T · gradb = grad_out` via upper-triangular substitution
   (using the pre-built transposed-CSR view on `CsrPattern`).
3. Compute `gradA_values[k] = -gradb[row(k)] * x[col(k)]` for each
   non-zero.
4. Register `gradA_values` → A_values parent, `gradb` → b parent.

This mirrors `~/projects/ddr/src/ddr/routing/utils.py:515`
(`TriangularSparseSolver`) 1:1.

For `TimestepOp` the pattern is similar but with `Backward<I, 5>`
instead of `Backward<I, 1>` — five parents (`n`, `q_spatial`,
`p_spatial`, `q_t`, `q_prime_t`) and a 23-field saved-state struct
holding all the forward intermediates the chain rule needs. See
[Architecture](../architecture.md) and [Algorithm](../algorithm.md)
for what each saved field is and why the backward needs it.

## Why this matters

The point of writing a custom `Backward` from a downstream crate is
**tape size**. The naive approach — letting BURN autodiff trace every
tensor op individually — would push ~33 op nodes per timestep onto the
tape, and the sparse triangular solve alone would push O(n²) entries
(one per back-substitution row). Multiply by O(timesteps) ×
O(batches) and the GPU runs out of memory before the first epoch
finishes.

A custom `Backward<I, N>` collapses an arbitrary forward computation
into a **single** tape node with N parents. The cost is writing the
analytical adjoint by hand, which is exactly what `src/sparse/mod.rs`
and `src/routing/mmc_op.rs` do. The benefit is trainability on the
full CONUS network at f32 precision.

## Verification

```bash
cargo test --test sparse_gradcheck
```

Validates that the hand-written custom `Backward` for `CsrSolveOp`
produces correct gradients via central differences. The
`spike_backward/visibility_check.rs` file also exercises the recipe
directly on a toy `square` op.

For `TimestepOp`:

```bash
cargo test --test sp8_gradcheck -- --ignored
```

Same finite-difference check, but on the full per-timestep chain
with all five parents (`n`, `q_spatial`, `p_spatial`, `q_t`,
`q_prime_t`).

## See also

- [Architecture](../architecture.md) — where `TimestepOp` and
  `CsrSolveOp` sit in the module map.
- [Algorithm](../algorithm.md) — the math the analytical adjoints
  implement.
- [Graph objects](../usage/graph-objects.md) — the `CsrPattern`
  structure both ops reuse.
- [Performance & CUDA Graphs](perf.md) — how the custom Backward
  composes with the SP-10 forward capture path.

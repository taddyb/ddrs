# BURN autograd recipe

This chapter is the BURN 0.21 recipe for registering a custom
`Backward` op from a downstream crate like ddrs. It is the load-bearing
plumbing that lets ddrs's three custom ops — `CsrSolveOp` (in
`src/sparse/mod.rs:417`), `TimestepOp` and `TimestepLeakanceOp` (both in
`src/routing/mmc_op.rs:560` and `:617`) — collapse what would otherwise
be O(n²) or ~33-ops-per-timestep autograd tape entries into a single
`Backward<I, N>` node each. Without this recipe, the differentiable side
of ddrs would be untrainable at CONUS scale.

## What it is

BURN 0.21 exposes most of the autodiff plumbing publicly — enough to
write a custom backward op from outside the `burn-autodiff` crate — with
exactly one gap: the graph node type `NodeRef` is `pub(crate)`. You
cannot *name* that type from ddrs, but you can *pass values* of it by
type inference: every `AutodiffTensor<B>` carries a public `.node`
field, and handing those node values to `Backward::prepare(...)` is all
the registration machinery needs. That one trick is what makes the
custom-backward recipe work from a downstream crate.

### Visibility map

```text
burn::backend::autodiff::
    ops::{Backward, Ops, OpsKind, binary, unary}       ✓ public
    grads::Gradients                                    ✓ public
    checkpoint::base::Checkpointer                      ✓ public
    checkpoint::strategy::NoCheckpointing               ✓ public
    graph::NodeRef                                      ✗ pub(crate) — UNNAMEABLE from outside

burn::backend::Autodiff<B>                              ✓ public
<Autodiff<B> as Backend>::FloatTensorPrimitive
    = AutodiffTensor<B> (pub fields: .primitive, .node) ✓ values reachable
```

The autodiff imports the ddrs ops actually use are at the top of
`src/sparse/mod.rs` (lines 21-27, reproduced verbatim; `src/routing/
mmc_op.rs` carries the same autodiff set with its own tensor imports):

```rust
use burn::backend::Autodiff;
use burn::backend::autodiff::checkpoint::base::Checkpointer;
use burn::backend::autodiff::checkpoint::strategy::NoCheckpointing;
use burn::backend::autodiff::grads::Gradients;
use burn::backend::autodiff::ops::{Backward, Ops, OpsKind};
use burn::tensor::backend::Backend;
use burn::tensor::{IndexingUpdateOp, Int, Tensor, TensorData, TensorPrimitive};
```

The last line's `IndexingUpdateOp`, `Int`, and `TensorData` are not part
of the custom-backward recipe — they belong to the CPU scatter fallback
and the host round-trips in the same module. The recipe itself needs only
`Tensor` and `TensorPrimitive` from `burn::tensor`, as in the minimal
example below.

## How to use it

A custom op is two pieces: a zero-sized marker struct implementing
`Backward<B, N>` (the analytical adjoint), and a forward function that
runs the computation on the inner backend and registers a single node on
the autograd tape. Here is the minimal one-input shape that both ddrs
ops generalise:

```rust
use burn::backend::Autodiff;
use burn::backend::autodiff::checkpoint::base::Checkpointer;
use burn::backend::autodiff::checkpoint::strategy::NoCheckpointing;
use burn::backend::autodiff::grads::Gradients;
use burn::backend::autodiff::ops::{Backward, Ops, OpsKind};
use burn::tensor::backend::Backend;
use burn::tensor::{Tensor, TensorPrimitive};

#[derive(Debug)]
struct MyOp;

impl<B: Backend> Backward<B, 1> for MyOp {       // N = number of inputs
    type State = B::FloatTensorPrimitive;        // any Clone + Send + Debug + 'static

    fn backward(self, ops: Ops<Self::State, 1>, grads: &mut Gradients, _: &mut Checkpointer) {
        let state = ops.state;
        let [parent] = ops.parents;
        if let Some(parent) = parent {
            let grad_out = grads.consume::<B>(&ops.node);
            let grad_in = /* analytical adjoint of state + grad_out, in inner-backend ops */;
            grads.register::<B>(parent.id, grad_in);
        }
    }
}

fn my_op<B: Backend>(x: Tensor<Autodiff<B>, 1>) -> Tensor<Autodiff<B>, 1> {
    let at = match x.into_primitive() {
        TensorPrimitive::Float(p) => p,
        TensorPrimitive::QFloat(_) => panic!("expected float tensor"),
    };
    let output_inner = forward_on_inner_backend(at.primitive.clone());
    let saved_state = at.primitive.clone();              // for backward

    let result = match MyOp
        .prepare::<NoCheckpointing>([at.node.clone()])   // NodeRef passed by inference
        .compute_bound()                                 // or .memory_bound() to recompute
        .stateful()                                      // or .stateless() if State = ()
    {
        OpsKind::Tracked(prep)   => prep.finish(saved_state, output_inner),
        OpsKind::UnTracked(prep) => prep.finish(output_inner),  // grad not needed downstream
    };
    Tensor::from_primitive(TensorPrimitive::Float(result))
}
```

The shape is always the same: peel the `AutodiffTensor` off each input
with `into_primitive()`, do the forward work on the inner backend `B` (so
no autodiff nodes are created inside the op), stash whatever the
analytical adjoint will need into `State`, then call
`MyOp.prepare(...).stateful()` to register one backward node. The
`OpsKind::Tracked` branch saves state and finishes; the
`OpsKind::UnTracked` branch is taken when no input is tracked (the result
will be detached, so the bookkeeping is skipped).

### Scaling to N inputs

`Backward<B, N>` widens the parent array to `[Option<NodeRef>; N]` and
the `prepare` call to an N-element node array. Each `Option` is `Some`
iff that input was being tracked. ddrs uses three widths — 2, 5, and 8:

- **`CsrSolveOp` is `Backward<B, 2>`** (`src/sparse/mod.rs:417`) — the
  two inputs are the assembled matrix values `a_values` and the
  right-hand side `b`. Its forward (`triangular_csr_solve` in
  `src/sparse/mod.rs`) destructures
  `[a_at.node.clone(), b_at.node.clone()]` into the prepare call, and
  its backward destructures `let [parent_a, parent_b] = ops.parents;`.
- **`TimestepOp` is `Backward<I, 5>`** (`src/routing/mmc_op.rs:560`) —
  the five tracked parents are, in fixed order,
  `[n, q_spatial, p_spatial, q_t, q_prime_t]`. The three
  remaining forward inputs (`length`, `slope`, `x_storage`) are network
  constants, not differentiated through, so they are saved in state but
  never registered.
- **`TimestepLeakanceOp` is `Backward<I, 8>`**
  (`src/routing/mmc_op.rs:617`) — the leakance (GW–SW water-loss)
  variant of the fused timestep. It widens `TimestepOp`'s five parents
  with the three leakance parameters:
  `let [p_n, p_qsp, p_psp, p_qt, p_qpt, p_kd, p_dgw, p_fac] =
  ops.parents;` (`:630`). Its `State` is `TimestepLeakanceState`, which
  *composes* rather than duplicates — a `base: TimestepState<I>` plus a
  `leak: LeakanceSaved<I>` holding `area_z`, `k_d`, `d_gw`,
  `leakance_factor`, the `losing_only` flag, and the optional impervious
  mask. Exactly one of the two timestep ops is registered per step:
  `route_timestep` (`src/routing/mmc.rs:377-383`) takes the leakance
  branch when all three of `k_d`, `d_gw`, `leakance_factor` are present,
  and `timestep_forward` otherwise.

### Saved state holds primitives, not autodiff tensors

`State` must be `Clone + Send + Debug + 'static`, and — critically —
must not itself participate in autograd, or the op would defeat its own
purpose. All three ddrs ops therefore store inner-backend
`B::FloatTensorPrimitive` values plus plain Rust scalars. `CsrSolveState`
is small:

```rust
#[derive(Clone, Debug)]
struct CsrSolveState<B: Backend> {
    a_values: B::FloatTensorPrimitive,
    x: SavedX<B>,            // Cpu(Arc<Vec<f32>>) or Cuda(B::FloatTensorPrimitive)
    pattern: Arc<CsrPattern>,
    use_cuda: bool,
}
```

Two details worth copying: the structural CSR arrays live in an
`Arc<CsrPattern>` so per-timestep cloning is a refcount bump rather than
an O(nnz) copy onto the tape, and the forward solve output `x` is kept as
a host `Vec<f32>` (wrapped in `Arc`) on the CPU path because the backward
needs it as one immediately. `TimestepState` is the same idea at larger
scale — 23 saved forward intermediates plus the eight inputs and a
handful of `f32` clamp bounds — every field an inner-backend primitive or
a scalar, none of them autodiff tensors.

### Writing the analytical adjoint

Inside `backward`, the saved primitives are wrapped back into
*inner-backend* `Tensor<I, 1>` values (not `Tensor<Autodiff<I>, 1>`), so
every arithmetic step is a plain backend op with no further tape pushes.
`CsrSolveOp::backward` is the compact case — it consumes the incoming
gradient, solves the transposed system, and registers both parent
gradients:

```rust
fn backward(self, ops: Ops<Self::State, 2>, grads: &mut Gradients, _: &mut Checkpointer) {
    let CsrSolveState { a_values, x, pattern, use_cuda } = ops.state;
    let [parent_a, parent_b] = ops.parents;
    let grad_out = grads.consume::<B>(&ops.node);
    let device = B::float_device(&grad_out);

    // grad_b = (A^T)^{-1} · grad_out  via upper-triangular back-substitution.
    let gradb_prim = crate::sparse::dispatch::backward_solve_primitive::<B>(
        &pattern, &a_values, &grad_out, &device, use_cuda,
    );
    if let Some(p_b) = parent_b {
        grads.register::<B>(p_b.id, gradb_prim.clone());
    }
    if let Some(p_a) = parent_a {
        // grad_a_values[k] = -grad_b[row(k)] · x[col(k)]   (per non-zero scatter)
        let grada_prim = crate::sparse::dispatch::grada_primitive::<B>(
            &pattern, gradb_prim, x, &device, use_cuda,
        );
        grads.register::<B>(p_a.id, grada_prim);
    }
}
```

`TimestepOp::backward` is the same structure with a much longer body: a
reverse walk B28 → B1 through the trapezoidal-geometry and
Muskingum-coefficient chain, accumulating partials on each of the five
parents and registering them at the end. It calls
`backward_solve_primitive`, `assemble_backward_primitive`, and
`spmv_backward_primitive` (the inner-backend adjoints of the sparse
solve, the `A = I − c·N` assembly, and the `N · q` SpMV) at the exact
points where the forward used their counterparts. See
[Algorithm](../algorithm.md) for the math each step implements.

## Reference

### `prepare` builder chain

The `MyOp.prepare(...)` builder selects three independent properties:

| Call | Choices | What it controls |
|---|---|---|
| `.prepare::<C>([nodes])` | `C = NoCheckpointing` in ddrs | Checkpoint strategy; nodes are the parent `NodeRef`s |
| `.compute_bound()` / `.memory_bound()` | ddrs uses `compute_bound()` | Whether the op recomputes its forward (`memory_bound`) or stores state (`compute_bound`) |
| `.stateful()` / `.stateless()` | ddrs uses `.stateful()` | Whether `State` is non-trivial (`.stateful()`) or `()` (`.stateless()`) |

The chain yields an `OpsKind` enum; matching on `Tracked` vs `UnTracked`
is mandatory. `Tracked(prep).finish(state, output)` takes both state and
output; `UnTracked(prep).finish(output)` takes only the output.

### `Ops<State, N>` fields used in `backward`

| Field | Type | Use |
|---|---|---|
| `ops.state` | `Self::State` | The struct saved at forward time |
| `ops.parents` | `[Option<NodeRef>; N]` | One slot per input; `Some` iff tracked |
| `ops.node` | (the op's own node) | Passed to `grads.consume::<B>(&ops.node)` to pull `∂L/∂output` |

`grads.consume::<B>(&ops.node)` returns the incoming gradient as a
`B::FloatTensorPrimitive`; `grads.register::<B>(parent.id, grad)` writes a
parent's gradient back. Both are generic over the inner backend `B`.

### Visibility constraint, restated

`NodeRef` (`burn::backend::autodiff::graph::NodeRef`) is `pub(crate)`, so
ddrs never writes the type. It only ever moves `at.node.clone()` values
around — produced by the public `.node` field on `AutodiffTensor<B>` and
consumed by `prepare`. If a future BURN release makes `NodeRef` public,
nothing in ddrs needs to change; if it tightens `.node`, the whole recipe
breaks and the port would need a different registration path.

### Why a hand-written backward (the O(nnz)-tape rationale)

The point of writing these adjoints by hand is **tape size**. Letting
BURN autodiff trace the forward op-by-op would push ~33 op nodes per
timestep onto the tape, and the sparse triangular solve alone would push
O(n²) entries — one per back-substitution row. Multiplied by
O(timesteps) × O(batches), the autograd graph exhausts GPU memory before
the first epoch finishes.

A custom `Backward<I, N>` collapses an arbitrary forward computation into
a **single** tape node with N parents, holding O(nnz) saved primitives
(via the shared `Arc<CsrPattern>`) instead of O(n²) traced nodes. The
cost is writing the analytical adjoint once; the payoff is trainability
on the full CONUS network at the f32 precision floor. This is exactly why
the project invariants forbid replacing the hand-written sparse backward
with autograd-tape unrolling.

## Verification

The sparse op's backward is checked against DDR's own
`TriangularSparseSolver` adjoint at f32 precision:

```bash
cargo test --test sparse_gradcheck
```

This loads the same `(A, b, grad_output)` fixtures DDR produced, runs the
ddrs solver under `Autodiff<NdArray<f32>>`, backprops `(grad_output ·
x).sum()`, and asserts the parameter gradients match DDR's to f32
tolerance.

The fused timestep op is checked against central finite differences over
all five tracked parents:

```bash
cargo test --test sp8_gradcheck
```

It builds a small synthetic `NdArray` network, runs one
`timestep_forward`, and compares the analytical gradient against central
differences for each of `n`, `q_spatial`, `p_spatial`, `q_t`,
`q_prime_t` (1e-3 relative tolerance, with an absolute-tolerance
allowance for clamp-saturated slots whose analytical gradient is zero).

The 8-wide leakance op has its own finite-difference gradcheck:

```bash
cargo test --test leakance_gradcheck
```

It mirrors `sp8_gradcheck` but drives `timestep_forward_leakance`
end-to-end and adds the three leakance parents to the swept set
(`{n, q_spatial, p_spatial, x_storage, K_D, d_gw, leakance_factor}`),
with the leakance parameters placed in the interior of their ranges
(`K_D = 5e-7`, `d_gw = 0.0`, `leakance_factor = 0.5`) on a losing
configuration so no clamp saturates. Any change to
`src/routing/leakance.rs` or to `TimestepLeakanceOp` must keep this
green — alongside `cargo test --test leakance_off_parity`, which asserts
the no-leakance path stays byte-identical when the term is off.

## See also

- [Architecture](../architecture.md) — where `TimestepOp` and
  `CsrSolveOp` sit in the module map and the per-timestep dataflow.
- [Algorithm](../algorithm.md) — the math the analytical adjoints
  implement, step by step.
- [Graph objects](../usage/graph-objects.md) — the `CsrPattern`
  structure both ops carry in their saved state via `Arc`.
- [Performance & CUDA Graphs](perf.md) — how the custom backward composes
  with the SP-10 forward-capture (`timestep_forward_via_graph`) path.

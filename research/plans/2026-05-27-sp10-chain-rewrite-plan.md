# SP-10 forward-chain rewrite — feasibility study + design plan

Status: design-only, awaiting user sign-off.
Branch context: `sp10-cuda-graphs` (work in `.claude/worktrees/sp10/`).
Problem refs: Task #109 (CUDA_ERROR_ILLEGAL_ADDRESS on replay).

---

## 1. Root cause recap

`try_capture_forward` (`src/sparse/cusparse.rs:2118`) currently:

1. Probe-runs `forward_chain_inner` outside capture, observes src devptrs of
   the 24 declared outputs, drops the probe tensors.
2. Re-runs the chain inside `cuStreamBeginCapture` / `cuStreamEndCapture`.
3. Appends `cuMemcpyDtoDAsync(src → scratch.dst)` for each of the 24
   declared outputs.

The chain has ~30+ **unnamed sub-expression temporaries** (e.g.
`qt_in * n_in * (q_eps + 1.0)` produces 2–3 implicit cubecl handles that BURN
never names). Those temps are allocated from the **dynamic pool**, which
recycles slices to physical addresses after their handles drop. The captured
graph bakes in those transient device addresses; cubecl reissues the same
slot to later work; replay reads/writes freed memory → `ILLEGAL_ADDRESS`.

Pinning the 24 named outputs via D2D copies is necessary but insufficient:
the **kernels' input reads** also touch those transient temp pointers.

## 2. API-surface findings

### cubecl-runtime
- `ComputeServer::launch(kernel, count, KernelArguments { buffers: Vec<Binding>, … }, …)`
  takes `Vec<Binding>` — every buffer slot is a binding on an **existing**
  handle. There is no output-slot type distinction; the kernel's SASS decides
  read vs write. So at the cubecl level, **pre-allocated output buffers are
  natively supported** — just bind an existing handle.
  (`/home/tbindas/projects/cubecl/crates/cubecl-runtime/src/server/base.rs:378`)
- `MemoryAllocationMode::Persistent` exists
  (`crates/cubecl-runtime/src/memory_management/memory_manage.rs:107`). When
  active, `reserve` calls go to `PersistentPool::alloc`, whose `cleanup`
  only `dealloc`s slices when `explicit=true` *and* the slice is currently
  free. The storage page itself is **never invalidated as long as the cache
  keeps the corresponding `ManagedMemoryHandle` alive** (`persistent_pool.rs:149-184`).
- `ComputeServer::allocation_mode(mode, stream_id)` + `MemoryManagement::mode()`
  toggle the mode at runtime (`memory_manage.rs:360`). Toggleable per-call.

### BURN-cubecl
- BURN's high-level tensor ops (`Tensor::mul`, `Tensor::add`, …) always allocate
  a fresh output. They have no public "write-into-this-handle" primitive in
  burn-cubecl 0.21. Output-bound BURN ops do **not** exist; the only knob
  is at the cubecl layer.

### cubecl `#[cube]` macro
- Custom kernels written with `#[cube]` can take any number of `&Tensor<F>` /
  `&mut Tensor<F>` arguments. A single fused kernel covering S1..S28 over a
  1-D `[n_segments]` element axis is straightforward — pure elementwise
  arithmetic + one sparse linear-solve. The solve is already a separate kernel
  (`cusparse_csr_lower_solve_inplace`), so the fused kernel would be S1..S14
  before the solve and S25..S28 after, with the solve called between.

---

## 3. Feasibility of each Option

| Option | Verdict | Why |
|---|---|---|
| 1. Pre-alloc every intermediate + rewrite chain into output-bound ops | **Blocked at BURN layer**; would require bypassing BURN entirely or adding new burn-cubecl primitives upstream. Untenable in a contained patch. |
| 2. Hybrid scratch-as-primitive re-binding | **Blocked** — same BURN limitation; `into_primitive`/`from_primitive` round-trips don't give us write-into-this-buffer semantics. |
| 3. Single fused `#[cube]` kernel | **Viable**; ~400–600 lines of `#[cube]` Rust to replace the 28-step chain. Highest blast radius (re-derive grad math by hand). Best perf. |
| 4. `MemoryAllocationMode::Persistent` toggle + keep-alive of every Handle | **Viable and minimal**. Toggle persistent mode for the duration of capture; collect every Handle the chain produces (named + unnamed temps) and keep them alive on the cache; persistent pool never deallocates pages whose handles are alive → device addresses stay valid forever. |
| 5. Capture-only sub-pool | Folded into Option 4 above; `MemoryAllocationMode::Persistent` IS this. |
| 6. cuMemPool-based isolation | Largest scope. cubecl-cuda would need a new allocator backend; out of scope for SP-10. |

The collision avoidance in Option 4 reduces to a single requirement: **the
cache must own a `Vec<Handle>` of every cubecl handle materialized during
capture, including unnamed sub-expression temps**. As long as those handles
are alive, persistent-pool slices remain reserved and addresses cannot be
recycled. Capturing a wrapper around `client.empty`-like calls is not
straightforward at the call site, but a *post hoc* mechanism is: instrument
`forward_chain_inner` to thread a `Vec<I::FloatTensorPrimitive>` (sink) that
every intermediate tensor's primitive gets cloned into before the
sub-expression scope ends. That keeps every handle alive without rewriting
arithmetic.

---

## 4. Recommended path: Option 4 (Persistent-mode + handle pinning)

One-line summary: **toggle `MemoryAllocationMode::Persistent` around capture,
collect every intermediate tensor's primitive into a cache-owned `Vec`, and
the existing D2D-copy strategy in `try_capture_forward` becomes sound.**

This preserves the SP-10 architecture wholesale (no API redesign, no kernel
rewrite, no `#[cube]` work), keeps V1 ABSOLUTE MATCH safe (kernel math
unchanged), and makes V9 bit-match a meaningful gate. The only new code is
the pinning sink + persistent-mode toggle.

---

## 5. Task breakdown (commit-sized)

### Task A — `chain_with_pinning` variant of `forward_chain_inner`

- File: `src/routing/mmc_op.rs`
- Add a `forward_chain_inner_pinned<I>(…, pin: &mut Vec<I::FloatTensorPrimitive>) -> (q_next, [saved; 23])` that mirrors `forward_chain_inner` line-for-line, but every `let x = a OP b;` is followed by `pin.push(unwrap_clone(x))`. The non-pinned variant calls the pinned variant with a thrown-away sink so production code path is unchanged.
- Scope: ~120 lines added (28 ops × ~3 lines + helpers); ~1 hour.
- Verify: existing `compare_ddr_sandbox` still ABSOLUTE MATCH (forward path is byte-identical when sink is discarded).

### Task B — `MemoryAllocationMode::Persistent` toggle helper

- File: `src/cuda_graph/capture.rs` (new helper) or `src/sparse/cusparse.rs`.
- Wrap `client.allocation_mode(Persistent, stream_id)` in an RAII guard that restores `Auto` on Drop. Exposed as `PersistentModeGuard::new(&client, stream_id)`.
- Scope: ~40 lines; ~30 min.
- Verify: unit test asserts guard correctly toggles and restores.

### Task C — Re-wire `try_capture_forward` to use pinning + persistent mode

- File: `src/sparse/cusparse.rs` (around line 2118).
- Acquire `PersistentModeGuard` before *both* the probe pass and the capture pass.
- Allocate a `pin: Vec<I::FloatTensorPrimitive>` and pass to `forward_chain_inner_pinned` in both passes.
- After capture succeeds, MOVE `pin` onto a new field `cache.pinned_intermediates: Vec<I::FloatTensorPrimitive>` so the handles stay alive for the cache's lifetime. **This is the critical change**: even after `try_capture_forward` returns, the persistent slices stay reserved.
- Drop guard at end of capture so production allocations resume `Auto` mode.
- Scope: ~80 lines changed; ~1.5 hours.
- Verify: V1 ABSOLUTE MATCH on `compare_ddr_sandbox` (with `use_cuda_graphs=true`).

### Task D — Add `pinned_intermediates` field to cache

- File: `src/sparse/cusparse.rs` (cache struct), `src/cuda_graph/mod.rs` if applicable.
- Push the `Vec<FloatTensorPrimitive>` onto `CudaPatternCache`. Make sure it drops AFTER `graph_fwd` in struct field order so the graph is destroyed before its referenced handles.
- Scope: ~20 lines; ~30 min.
- Verify: `cargo build` clean; no leaks via `memory_usage` probe before/after a session.

### Task E — V9 bit-match test (already pending as Task #104)

- Re-enable / write `tests/sp10_graph_bitmatch.rs` per existing plan.
- Verify: 10 consecutive replays bit-identical to 10 consecutive direct-launches.
- Scope: ~150 lines; ~1.5 hours.

### Task F — V7a perf gate update + ARCHITECTURE.md note

- Re-enable Task #106 (V7a) and document the persistent-pinning pattern in
  `.claude/ARCHITECTURE.md`. Note that the pinning sink doubles each
  intermediate's lifetime, adding ~30 × 4 bytes × `n_segments` of "leaked"
  memory per cache (~40 MB for full CONUS — acceptable).
- Scope: ~80 lines + docs; ~1 hour.

**Total: ~6 hours of work, ~500 lines net, 3 files touched in `src/` + 1 new
test.**

---

## 6. Risk assessment

| Risk | Likelihood | Impact | Mitigation |
|---|---|---|---|
| Persistent pool still recycles slice after pin Vec holds the handle | Low | Fatal | Verified in `persistent_pool.rs:83-100`: `try_reserve` only returns slices where `slice.is_free()` (handle dropped); pin Vec prevents that. Unit test in Task B asserts. |
| Pin sink itself triggers extra allocations (cloning a primitive) | Low | Perf-only | `FloatTensorPrimitive` clone is Arc-like in cubecl; no device alloc. Confirmed by inspection of `Handle` clone impl. |
| Persistent mode leaks memory across many MC instances | Medium | Memory pressure | Each `CudaPatternCache` owns its pin Vec; cache drop deallocates. Document max-leak per cache in ARCHITECTURE.md. |
| Probe vs capture pass diverge despite persistent mode | Low | Fatal (silent on replay) | Existing devptr-equality check in `try_capture_forward:2360-2375` catches this and falls back. |
| Cubecl drop-queue runs between passes and host-syncs inside capture | Medium | Capture fails | Already mitigated by `flush_async` and double-flush before capture. Persistent mode does not interact with drop-queue. |

**Top 3 known unknowns:**

1. Does cubecl's CUDA backend honour `MemoryAllocationMode::Persistent` for sub-bucket-size allocations? `MemoryManagement::reserve` short-circuits to `persistent.alloc` (line 469) only if `mode == Persistent` OR `persistent.has_size(size)`. The first condition fires unconditionally inside our guard, so yes.
2. Does the sparse-solve kernel (`cusparse_csr_lower_solve_inplace`) allocate any internal workspace? If yes, those workspaces also need pinning. Inspect `src/sparse/cusparse.rs` for any in-flight `client.empty` in the solve kernel.
3. Does the pin Vec's clone-into-Vec hold the handle by value or by Arc? Need to confirm `FloatTensorPrimitive: Clone` is shallow (Arc-bumping) and that pushing into a Vec increments the refcount rather than producing a fresh allocation. Quick test in Task A.

---

## 7. Verification gates

- **V1 ABSOLUTE MATCH** (max abs diff < 1e-3 m³/s vs DDR) must still hold on `examples/compare_ddr_sandbox` with `use_cuda_graphs=true`. This is the load-bearing gate.
- **V9 bit-match** (Task E): 10 consecutive replays bit-identical to 10 direct-launches.
- **V7a perf** (Task F): per-step kernel launch count drops from ~30 to 1 (CUDA Graph replay = 1 launch).
- **Memory regression**: `memory_usage` before vs after a 1000-step training run grows by ≤ pin-Vec size; no unbounded growth.

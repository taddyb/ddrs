# DDR ↔ DDRS training-step parity testing plan

**Date:** 2026-06-04
**Branch:** `training-step-parity` (sibling to merged `trained-parity` / PR #12)
**Successor to:** `docs/superpowers/specs/2026-06-03-ddr-ddrs-trained-saturation-parity-design.md`
(`§5.1 Empirical verdict`: outcome row 2 — "DDR healthier, DDRS has a
training-loop bug")
**Symptom:** With NaN-gauge filter wired (`30501af`) and `log_space_parameters`
flipped from `[n]` to `[p_spatial]` (`4728dd8`), DDRS's trained Manning's `n`
distribution is still saturated relative to DDR at identical config:

- DDRS median `n` = 0.030, DDR median `n` = 0.074 (KS = 0.69 across 346k reaches)
- DDRS `p_spatial` median = 5.67, DDR median = 8.15 (KS = 0.53)
- Both `n` and `p_spatial` get pulled **systematically lower** in DDRS than in DDR.
- `q_spatial` medians match (0.460 vs 0.463) — only per-reach order is scrambled
  (Spearman = 0.30), consistent with batch-shuffle PRNG STAT-only drift, not a port bug.

The previous parity work proved the KAN head is bit-identical forward + backward
at fixed init params (PR #11 — `kan_head_fixture_{forward,backward}.rs` at
5.96e-8 / 1.91e-6). So the residual divergence is downstream of the head: in
the MC routing forward, the loss-construction pipeline, or the optimizer
step.

This spec localizes the divergence by comparing **a single training mini-batch
end-to-end** between DDR and DDRS, with identical fixture-loaded KAN init
params and identical batch contents. Each layer narrows the cause.

---

## 1. Why this matters / why now

The previous parity scaffold's trained-output comparison (KS on 346k reaches)
is a strong but coarse signal — it tells us the optimization trajectories
diverge but doesn't say where. A 5-epoch trajectory comparison would also be
coarse: too many SGD steps to localize where the trajectories first separate.

The cheapest localization is **one mini-batch at fixed init**:

```
fixture-loaded params + same batch_gauges + same date_window
              │
              ├─ KAN forward   (already proven bit-identical, PR #11 Task 9)
              ├─ MC routing forward
              ├─ tau-trim + daily downsample
              ├─ NaN-gauge filter
              ├─ L1 loss
              ├─ loss.backward()
              ├─ clip_grad_norm
              └─ optimizer.step()
```

Comparing the state at each stage isolates which stage first diverges. The
KAN-head boundary is already nailed down to f32 floor; everything from "MC
routing forward" downward is new ground for parity.

---

## 2. Concerns

| # | Concern | Why it could go wrong |
|---|---------|----------------------|
| C1 | DDR's `scripts/train.py` is monolithic — extracting per-stage state for comparison requires either modifying DDR (invasive) or instrumenting it with logging hooks. | Mitigation: use `torch.utils.hooks.RemovableHandle` or a small wrapper module that captures intermediate tensors via `forward_pre_hook` / `register_hook`. Do not touch DDR's source. |
| C2 | DDR's MC routing forward is GPU-native (torch sparse on CUDA); DDRS's is also GPU-native (BURN + cuSPARSE). The two CUDA paths accumulate atomically in different orders. Bit-parity on GPU is not achievable; we must use CPU forward for both and accept the `~1e-4` cuBLAS gap as already proven (PR #11 Task 9 CUDA branch). | Layer B compares on CPU first (NdArray vs torch-CPU) for tight tolerance, then GPU as a sanity check. |
| C3 | DDR's training script doesn't expose a "run one mini-batch with these params" entry point. We have to call the lower-level functions (`dmc.routing.muskingum_cunge`, the KAN forward, the loss computation) in the right order. | Mitigation: read `scripts/train.py:50-100` carefully and reproduce the call sequence in a fixture-driver script. Cite line numbers in comments so future agents can verify. |
| C4 | The hot-start state for MC routing (initial discharge per reach) is computed once per training run from the streamflow forcing. Whether DDR and DDRS compute identical hot-start is not yet proven. Hot-start differences would shift the entire forward Q. | Layer B explicitly compares the hot-start tensor as its first sub-step. |
| C5 | The per-gauge subgraph construction (filtering the full CONUS adjacency to the upstream-of-gauge subgraph) might differ. DDR uses pre-built per-gauge zarr (`merit_gages_conus_adjacency.zarr`); DDRS reads the same zarr. They SHOULD be identical, but a different filter / sort / topological ordering would silently shift gradients. | Layer A audits the subgraph construction; Layer B's first sub-step verifies tensor equality of the loaded subgraph adjacency for a known gauge. |
| C6 | The Adam optimizer state (first/second moment estimates) starts at zero and is updated each step. After mini-batch 1, the optimizer state is non-zero. Layer D compares post-step-1 params — that requires bit-identical gradient AND bit-identical Adam math. If Adam differs between burn and torch in subtle ways (epsilon application order, bias correction formula), Layer D will flag it. | Acceptable — that's the point of the test. |
| C7 | The user has confirmed `tau`-time-slicing is intentional (Bindas et al. 2025 WRR — timezone offset for midnight-aligned daily aggregation). DDR train.py uses `[13 : -11+tau]`, DDRS loss.rs uses `[13+tau : -11+tau]`. Both are correct given the tau semantics, but they sample slightly different hours. | Document as an acceptable divergence in Layer C. Do not "fix" it. |

---

## 3. Assumptions

| # | Assumption | Justification |
|---|------------|---------------|
| A1 | A single fixture-loaded mini-batch comparison is sufficient to localize the divergence. We do not need to compare multiple batches because the per-batch divergence is what accumulates over 5 epochs into the observed shift. | If the per-batch comparison shows no divergence at any stage, the bug is in batch ordering / sampling — pivot to Layer D+ (multi-batch trajectory). |
| A2 | DDR's `~/projects/ddr/.venv` is still usable from CPU (matches Task 11 + Task 7 of the previous plan). | Already established. |
| A3 | The KAN-head fixture from PR #11 (`tests/fixtures/kan_head_init_seed42.npz`) can be reused as the fixed init for both ports. Loading it into DDR requires writing the transpose-reverse of DDRS's `from_npz` loader. | New code, but ~50 lines of Python under DDR's venv. |
| A4 | A "matched mini-batch" means identical (gauge IDs, training time window start, time window length, hot-start state). We do NOT need bit-identical RNG; the batch is a deterministic set once those four are fixed. | Mirrors how `compare_ddr_sandbox` works — fix the inputs, compare the outputs. |
| A5 | Tolerances: CPU forward ≤ 1e-5, gradients ≤ 1e-4, Adam-stepped params ≤ 1e-4. These are 10–100× the head-only fixture tolerances from PR #11 because the MC routing forward composes many ops. Tighter ranges are nice-to-have but not load-bearing for localization. | Loose tolerances surface only divergences ≥ those bounds; that's exactly what we want at this level. |
| A6 | We use **gauge index 0** from `gages_3000.csv` (whichever it resolves to) as the fixture gauge. Its upstream subgraph is small enough to compute on CPU; results trivially extend to larger gauges if the small-gauge test passes. | One gauge is enough to localize a stage-level divergence; multi-gauge testing only matters if the divergence is gauge-size-dependent. |

---

## 4. Layered test plan

Four layers, ordered cheapest → most decisive. Each layer answers a yes/no
question; failures route to a specific module.

### Layer A — Full training-step config audit (no code; 2 hours)

**Question:** Is every step in DDR's training inner loop (`scripts/train.py:50-130`)
mirrored verbatim by DDRS's (`src/training/driver.rs:50-200`), with the
already-known and documented divergences (tau-slicing, batch-shuffle PRNG)
explicitly listed?

**Procedure:** like the previous Layer 0, but focused on the **inner loop**
rather than the hyperparameter setup. Audit at the granularity of:

| Step | DDR source | DDRS source | Match? |
|------|------------|-------------|--------|
| Per-mini-batch gauge selection | `train.py:50-60` | `data/dataset.rs::shuffle_*` | (audit) |
| Subgraph adjacency load | `dmc.routing.muskingum_cunge` setup | `src/routing/mmc.rs::setup_inputs` | (audit) |
| Hot-start discharge computation | `dmc.routing.compute_hotstart_discharge` | `src/routing/utils.rs::hotstart_discharge` | (audit) |
| KAN forward | `kan.forward(inputs=...)` | `KanHead::forward` | ✓ (PR #11) |
| MC routing forward | `muskingum_cunge(...)` per-timestep loop | `MuskingumCunge::forward` | (audit) |
| Tau-trim + daily downsample | `[13:-11+tau]` then `.reshape(N, 24).mean(2)` | `tau_trim_and_downsample()` in loss.rs | ✓ + STAT-only (intentional tau divergence per spec C7) |
| NaN-gauge filter | `train.py:75-89` (per-gauge mask) | `src/training/driver.rs:115-160` (post 30501af) | ✓ (audit ✗ in previous plan; fixed in PR #12) |
| L1 loss | `F.l1_loss(p_masked, o_masked)` | `(p_filtered - o_filtered).abs().mean()` | ✓ |
| loss.backward() | `loss.backward()` | `loss.backward()` | ✓ (PR #11 Task 10) |
| grad_clip | `torch.nn.utils.clip_grad_norm_(..., max_norm=1.0)` | `clip_grad_norm(..., 1.0)` | (audit) |
| Adam step | `optimizer.step()` (PyTorch Adam) | `optimizer.step()` (burn AdamConfig) | (audit) |

The 5 (audit) rows are the new investigation. Each should be ✓, STAT-only,
or ✗ with concrete source citations.

**Pass criterion:** All rows ✓ or STAT-only (documented). Any unexpected ✗ →
that's the localized cause; skip to the relevant Layer B/C/D substep to
verify and quantify.

**Failure routing:**
- ✗ on subgraph adjacency load → Layer B sub-step 1 will surface it.
- ✗ on hot-start → Layer B sub-step 2.
- ✗ on MC routing forward → Layer B sub-step 3 (the headline test).
- ✗ on grad_clip → Layer C reveals it.
- ✗ on Adam step → Layer D reveals it.

### Layer B — Forward-pipeline parity at a fixed mini-batch (~1 day)

**Question:** Given fixture-loaded KAN params and a fixed gauge / time window,
do DDR and DDRS produce bit-identical daily Q predictions at the post-warmup
slice?

**Procedure:**

1. **Pick the fixture gauge.** Inspect `gages_3000.csv`, pick gauge index 0 (or
   whichever has the smallest upstream subgraph for fastest iteration).
   Record COMID + STAID in the test fixture.

2. **Pick the fixture time window.** Start `1990/01/01`, length = `rho` (90
   days hourly = 2160 hours). Avoid the boundaries of DDR's `start_time`/
   `end_time` to reduce edge-case risk.

3. **Sub-step 1: subgraph adjacency tensor parity.** DDR loads the
   per-gauge subgraph from the same zarr DDRS does. Dump both as
   `(rows, cols, vals)` triplets in topological order, write to
   `tests/fixtures/training_step/subgraph_<COMID>.npz`. Rust + Python
   assertions assert byte-equal triplets.

4. **Sub-step 2: hot-start discharge parity.** Both compute initial Q per
   reach as `streamflow_warmup_window.mean()` (or whatever DDR's
   `compute_hotstart_discharge` does — verify). Dump DDR's Python result and
   DDRS's Rust result to the same `.npz`; assert max abs diff ≤ 1e-6.

5. **Sub-step 3: MC routing forward parity.** Given fixture-loaded KAN params
   denormalized to `(n, q_spatial, p_spatial)` per reach, plus the hot-start
   from sub-step 2, plus streamflow forcing over the rho window, run the MC
   forward in both ports. Dump both result tensors `(n_reaches, rho_hours)`.
   Assert max abs diff ≤ 1e-5 (looser than head forward because MC composes
   ~24×rho ops per reach).

6. **Sub-step 4: post-tau-trim daily Q parity.** Apply both ports' tau-trim
   + daily downsample. Assert max abs diff ≤ 1e-5 (tau-slicing divergence
   is STAT-only — both lose 3 hours but at opposite ends; still expect
   ≥ 86 days of bit-identical post-warmup Q).

**Pass criterion:** All 4 sub-steps pass at their respective tolerances.

**Failure routing:** the first sub-step that fails localizes the bug to that
stage. If sub-step 3 (MC forward) fails, the bug is in the sparse solver or
geometry computation — see `.claude/skills/ddrs-burn-autograd.md`.

### Layer C — Loss + gradient parity at the fixed mini-batch (~half day)

**Question:** Given Layer B's matched daily Q, do DDR and DDRS produce
bit-identical L1 loss + gradients w.r.t. KAN params?

**Procedure:**

1. **L1 loss value.** Both ports apply NaN-gauge filter (already proven ✓
   post `30501af`) then `(pred - obs).abs().mean()`. Compare the scalar
   loss to ≤ 1e-7 (single float).

2. **Per-KAN-param gradients.** Both ports' `loss.backward()` populates
   `kan.input.weight.grad`, `kan.output.weight.grad`, `block_*_coef.grad`,
   etc. Compare each tensor to ≤ 1e-5. (Already proven at the head level by
   PR #11 Task 10 with synthetic loss; this is the same comparison with the
   actual training loss.)

3. **Post-clip gradient norm.** Compute global L2 norm of grads after
   `clip_grad_norm(max_norm=1.0)`. Compare to ≤ 1e-6.

**Pass criterion:** All 3 pass.

**Failure routing:** Loss mismatch → NaN-filter bug or downsample bug
(unlikely given PR #12). Grad mismatch but loss matches → autograd path
through the MC solver differs (likely the sparse backward). Post-clip norm
mismatch → grad-clip implementation differs.

### Layer D — Post-Adam-step parameter parity (~half day)

**Question:** Given matched gradients and identical initial Adam state, do
DDR and DDRS produce bit-identical post-step KAN params?

**Procedure:**

1. Both ports initialize Adam with `lr=0.001, beta_1=0.9, beta_2=0.999, eps=1e-8`.

2. Both call `optimizer.step()` on the loss from Layer C.

3. Dump every KAN parameter's value post-step. Compare to ≤ 1e-5.

4. Also dump Adam's first/second moment estimates for each parameter
   (`optimizer.state[param]['exp_avg']` and `'exp_avg_sq'` in PyTorch;
   the equivalent in burn). Compare to ≤ 1e-5.

**Pass criterion:** All param values + Adam state match at ≤ 1e-5.

**Failure routing:** Param mismatch but Adam state matches → param-update
formula differs (lr applied differently, sign error). Adam state mismatch
→ moment update formula differs (epsilon application, bias correction
ordering).

---

## 5. What success looks like

Three possible outcomes after Layer D:

| Outcome | Meaning | Next step |
|---------|---------|-----------|
| **All 4 layers pass** | Every stage is bit-identical at single-step level. The trained-output divergence is then entirely due to batch-shuffle PRNG ordering accumulating over 5 epochs — STAT-only per spec C5 of the previous plan, not a bug. | Document this and close the localization investigation. The n saturation is the genuine attractor for this loss surface; if we want different n, the next move is hydrology (change the loss, change the prior, regularize). |
| **Layer B fails (MC forward or hot-start)** | The routing solver differs between DDR's PyTorch sparse and DDRS's BURN+cuSPARSE in ways the existing `compare_ddr_sandbox` regression doesn't catch (different gauge subgraph size, different forcing pattern). | New spec to fix the sparse solver divergence. The 5-reach sandbox is too small to catch CONUS-scale bugs. |
| **Layer C / D fails (loss, grads, or Adam)** | The training-loop machinery itself differs — likely Adam implementation details, grad_clip, or NaN-filter edge cases. | Targeted fix in `src/training/`. Could be a one-line correction or a substantial reimplementation depending on the specific divergence. |

---

## 6. Implementation order

1. Layer A audit (2 hours, no code). Surface any ✗ rows from the 5 untested fields.
2. Layer B sub-step 1 (subgraph adjacency) — cheapest test.
3. Layer B sub-step 2 (hot-start discharge).
4. Layer B sub-step 3 (MC forward) — the load-bearing one.
5. Layer B sub-step 4 (tau-trim + downsample).
6. Layer C (loss + gradients).
7. Layer D (Adam step).
8. Spec §5.1 verdict.

Pause after Layer B sub-step 3. If it fails, the bug is localized — don't
build Layers C and D, write a new spec to fix the MC solver instead.

---

## 7. Out of scope

- Fixing the bug. This spec only localizes; the fix gets a new spec.
- Multi-mini-batch trajectory comparison. If single-batch parity holds, the
  divergence is batch-ordering only (STAT, not a bug). No need for multi-batch
  testing.
- Multi-gauge or full-CONUS comparison. Single gauge isolates stage-level
  divergence; full-CONUS would only matter if the bug is size-dependent.
- Changing the loss / model / training schedule. Those would be design
  decisions, not parity work.
- The `q_spatial` parameter. Its medians already match between DDR and DDRS;
  the per-reach scrambling is documented STAT-only batch-shuffle drift.

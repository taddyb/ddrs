# Reach Subdivision (variable Δx) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development
> (recommended) or superpowers:executing-plans to implement this plan task-by-task.
> Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Normalize every MERIT reach to `Δx ≈ c_ref·Δt` at adjacency-build time —
**splitting** reaches that are too long and **clamping the length** of reaches that
are too short — so `Cr ≈ 1` network-wide and the Muskingum coefficients are
non-negative *by construction*, with no runtime clamp and no gradient masking.

**The two-sided rule:**

```
Δx_target = c_ref · Δt
  L > Δx_target  →  split into m = ceil(L/Δx_target) pieces of length L/m, q' → q'/m
  L < Δx_target  →  clamp length up to Δx_target (do NOT merge)
```

Merging short reaches is rejected deliberately: a short reach may have two upstream
tributaries or a parallel tributary joining below it, so collapsing it would destroy
junction structure. Clamping its length achieves the same `Cr` with no topology change.

**Why a static length clamp is safe where the runtime K floor was not.** They are
algebraically related (`L ≥ c·Δt ⟺ K ≥ Δt`), but `enforce_positivity`'s K floor is
applied per timestep to a *learned* celerity — it masks gradients and makes
`X ∝ Cr ∝ 1/n`, which is what drove `n` to its floor on 98 % of reaches. A build-time
length clamp is just a different constant in the `length_m` array: **no gradient
path, no `∂X/∂n` coupling, and zero change to `mmc_op.rs` or any backward.**

**Architecture:** Subdivision is a **static preprocessing step** inside the managed
adjacency builder, between `build_conus_adjacency` and the gauge-subgraph/zarr write.
The runtime sees only a larger network. Piece count `m` is computed once from a
reference flow and **capped at 8** — the graph must be fixed for the CSR pattern and
the hand-written autograd, so `m` can never vary per timestep.

**Tech Stack:** Rust, BURN 0.21, `zarrs`, `blake3` (cache keys), `ndarray`.

## Global Constraints

- **f32 throughout the routing core.** No f64/bf16 casts (invariant 2, `CLAUDE.md`).
- **`examples/compare_ddr_sandbox` must stay an ABSOLUTE MATCH** (max abs diff < 1e-3
  m³/s). It builds its own tiny network and must be unaffected.
- **Adjacency must stay topologically ordered and lower-triangular** (`rows[k] >= cols[k]`),
  asserted at `src/adjacency/build.rs:235-240` via `BuildError::NotLowerTriangular`.
  The forward-substitution solver depends on it (invariant 3).
- **Do not replace the hand-written sparse backward** in `src/sparse/` with
  autograd-tape unrolling (invariant 4).
- **Never run `git add -A`.** Stage only files you changed, by name.
- **Never kill a running process.** Training runs may be active.
- Default OFF: `params.subdivision.enabled: false` reproduces current behaviour
  byte-for-byte.

---

## Measured feasibility — read before designing anything

Computed by mirroring `forward_chain_inner` S1–S17 in NumPy against the real
adjacency store, trained KAN parameters from
`.ddrs/runs/2026-08-03T13-11-00Z-train-and-test/plot/kan_parameters.nc`, and
per-divide median Q' accumulated downstream through the topological order.

**The model was validated against four independently measured in-run anchors:**

| Quantity | model | measured in-run |
|---|---|---|
| median `Cr` | 0.250 | 0.226 |
| `X` p5/p50/p95 | 0.340 / 0.4973 / 0.5000 | 0.330 / 0.4976 / 0.5000 |
| `enforce_positivity` bind rate | 95.5 % | 95.3 % |
| median `X_eff` after cap | 0.0864 | 0.0794 |

### Cap sweep

| cap M | sub-reaches | reach × | edges × | **critical path ×** | median Cr | median `X_eff` |
|---|---|---|---|---|---|---|
| 1 (today) | 346,321 | 1.00 | 1.00 | 1.00 | 0.250 | 0.086 |
| 2 | 633,500 | 1.83 | 1.85 | 1.50 | 0.499 | 0.146 |
| **4** | 1,071,482 | 3.09 | 3.14 | 1.85 | **1.00** | 0.239 |
| **8** | **1,652,965** | **4.77** | 4.86 | **2.18** | 1.10 | **0.321** |
| 16 | 2,326,605 | 6.72 | 6.84 | 2.64 | 1.10 | 0.372 |
| uncapped | 4,586,580 | 13.24 | 13.52 | **9.23** | 1.10 | 0.417 |

**Uncapped is infeasible — and the blocker is latency, not memory.** Per-timestep
state is only 422 MB uncapped, but forward substitution is sequential over
topological levels and chaining sub-reaches lengthens the critical path 9.23×, on a
trainer already forward-pass bound at ~27.5 s/micro-batch.

**Capping also makes the cost estimable.** 43 % of reaches have no Q' in the
configured store and must be imputed; across defensible imputations uncapped `Σm`
swings **2.3 M – 10.5 M** (4.6× uncertainty), but `Σ min(m,4)` stays within ±6 %
and `Σ min(m,8)` within ±12 %. **Any design resting on uncapped `Σm` rests on a
number that cannot be pinned down.**

### CORRECTION — the "restores X's dynamic range" claim is WRONG

An earlier conversational claim (and the reasoning that motivated this work) held
that subdivision would un-saturate the Cunge X. **It does not.** Raw `X_cunge`
median moves only **0.4973 → 0.4815** even uncapped. `Q/(B·S·c·Δx)` is small
primarily because of the `attribute_minimums.slope = 1e-3` floor and large top
width `B` — not because Δx is long. **Do not justify this change on X's dynamic
range.**

The two real justifications are:

1. **Non-negativity becomes automatic.** At `Cr = 1`, `c1` and `c3` both reduce to
   `(1 − 2X)/(1 + 2(1−X))`, which is `≥ 0` for *any* `X ≤ 0.5` — a bound already
   enforced physically. So `enforce_positivity` (and its 6.3× X collapse, and the
   `n`-to-floor degeneracy it causes) becomes unnecessary.
2. **If `enforce_positivity` is kept**, subdivision recovers most of its damage:
   median `X_eff` 0.086 → 0.321 at M=8 (71 % of achievable relief).

### The two-sided rule closes the gap subdivision alone cannot

Subdivision fixes only reaches that are too *long*. Adding the short-reach length
clamp (indicative reference celerity, cap 8):

| | subdivide only | **subdivide + length clamp** |
|---|---|---|
| median Cr | 1.10 | 1.08 |
| frac `Cr > 2` | 17.3 % (unfixable by splitting) | **0.00 %** |
| frac `Cr < 0.5` | 17.7 % | **0.05 %** |

Essentially the whole network lands inside the stable window, which is what makes
"non-negative by construction" an honest claim rather than an aspiration.

### Why the target is `Δx = c·Δt` and NOT the Ponce–Theurer limit

Formulas verified verbatim against [Ponce](https://ponce.sdsu.edu/muskingum_cunge_method_explained.html):
`K = Δx/c`; `X = ½(1 − q/(So·c·Δx))` with `q` the **unit-width** discharge; and
`C0 = (Δt−2KX)/denom`, `C1 = (Δt+2KX)/denom`, `C2 = (2K(1−X)−Δt)/denom` — identical
to ddrs's `c1`, `c2`, `c3` (`mmc_op.rs:1077-1080`). Ponce also confirms the failure
mode: *"for very large values of the space step, there is a tendency for physically
unrealistic negative outflows"*, and *"negative values of C2 are invariably
associated with dips in the rising portion of the outflow hydrograph"* — his `C2` is
our `c3`.

[Ponce & Theurer](https://ponce.sdsu.edu/accuracy_criteria_in_diffusion_routing.html)
give the accuracy criterion `C·D ≥ ξ` (a **product**) and the limit
`Δx ≤ ½(c·Δt + qo/(So·c))`. **That limit was evaluated and rejected for this network:**

| Δx target | Σm (cap 8) | median Cr | frac Cr > 2 | frac `C·D ≥ 0.33` |
|---|---|---|---|---|
| `c·Δt` (this plan) | 918 k | **1.08** | **0.00 %** | 0.3 % |
| Ponce–Theurer `½(c·Δt + qo/So·c)` | 1.42 M | 2.06 | **57.3 %** | 17.7 % |

The cell Reynolds number on MERIT is `D ≈ 0.012` — physical diffusion is ~1–2 % of
advective transport — so `Δx_D/Δx_C ≈ 0.020` and the Ponce–Theurer limit collapses to
`≈ c·Δt/2`, i.e. `C ≈ 2`. That **violates** the non-negativity ceiling `C ≤ 2(1−X)`
(which is `≈ 1` when `X ≈ 0.5`) and puts 57 % of reaches above `Cr = 2` — it makes
negative coefficients *worse*.

**`C·D ≥ ξ` is unsatisfiable here at any Δx that also keeps coefficients
non-negative.** It is an accuracy criterion for *diffusion* routing; as `D → 0` there
is no physical diffusion left to resolve and the criterion degenerates. Confirmed not
to be an artifact of the slope floor: lowering `attribute_minimums.slope` from `1e-3`
to `1e-6` moves median `D` only 0.0112 → 0.0120.

So the target is the Courant length, `Δx = c·Δt` — which is also HEC-HMS's Auto-DX
rule — because at `C = 1` both coefficients reduce to `(1−2X)/(1+2(1−X)) ≥ 0` for any
`X ≤ 0.5`.

### Where the cost lives

`corr(log m, log uparea) = −0.14`; `corr(log m, log c) = −0.75`; `corr(log m, log L) = +0.57`.
The cost is **long slow mid-order reaches**, not big rivers — the 71–169 km² uparea
bin alone is 33.7 % of `Σm`. Large rivers are already near Cr = 1 and mostly need
no split (17.1 % of all reaches need `m = 1`).

### Concerns for the user

- **The length clamp inflates total network length.** Indicative measurement with a
  crude reference celerity: 34.4 % of reaches clamped, median factor 2.0×, p95 12.5×,
  p99 36×, **max 48,597×**, total CONUS channel length +17.1 %. The extreme tail is
  degenerate MERIT geometry (11 reaches shorter than 10 m), so the clamp arguably
  repairs bad data there — but a reach modelled 12× longer than reality has a 12×
  longer travel time, which is a real physical distortion in exchange for numerical
  stability. **Consider a separate cap on the clamp factor** and measure the
  sensitivity in Task 8.
- **The clamped fraction is sensitive to `c_ref`.** The 34.4 % above used a crude
  `Q_ref = 0.01·uparea^0.9` giving median `c = 1.03 m/s`; the *calibrated* reference
  (validated against measured Cr) gives median `c = 0.443 m/s`, hence a smaller
  `Δx_target` and materially fewer clamped reaches. **Recompute with the calibrated
  celerity before trusting any of these figures.**
- **A static `m` is only Cr≈1 at the reference flow.** Median `c` varies ~3× between
  low and high flow, so reaches will be over-split at high flow and under-split at
  low flow. This is unavoidable: the graph must be fixed for autograd.
- **Retraining is required.** Every learned parameter was fit against the un-split
  network's effective diffusion. Checkpoints do not transfer.
- **2.18× critical path is a real training-time cost** on top of an ~11 h run.
- **The KAN sees no new information.** Sub-reaches inherit their parent's attributes,
  so subdivision cannot improve parameter identifiability — only numerics.

### Assumptions

- **Reference flow is config-specified, not checkpoint-derived.** Coupling
  subdivision to a trained checkpoint would make the graph depend on training state.
  The cap is what makes this safe (±6–12 % robustness).
- **Sub-reaches are hydraulically identical to their parent** — same `n`, `p`, `q`,
  slope; length `L/m`. MERIT carries no within-reach variation to do better.
- **Lateral inflow is uniform along the reach**, so each piece gets `q'/m`. This is
  HEC-HMS's own treatment: its lateral term is `C4·(q_L·Δx)` with `q_L` per unit
  length.

---

## File Structure

| Path | Responsibility |
|---|---|
| `src/adjacency/subdivide.rs` | **NEW.** Reference celerity, piece counts, graph expansion. Pure functions over plain `Vec`s — no BURN, no I/O. |
| `src/adjacency/build.rs` | Call subdivision after `build_conus_adjacency`, before gauge subgraphs. |
| `src/adjacency/cache.rs` | Fold subdivision config into the content key; bump `BUILDER_VERSION`. |
| `src/data/store/zarr.rs` | Persist/load `parent_offset` + `parent_order`. |
| `src/config.rs` | `params.subdivision` block + validation. |
| `src/routing/mmc.rs` | Divide `q'` by piece count after the clamp. |
| `src/training/forward.rs` | Gather KAN outputs from parent rows to sub-reach rows. |
| `src/data/collate.rs` | Map each gauge to its parent's **outlet** piece. |
| `tests/subdivide.rs` | **NEW.** Unit tests for celerity, piece counts, expansion invariants. |
| `tests/subdivision_integration.rs` | **NEW.** End-to-end: mass conservation, Cr distribution, non-negative coefficients without the clamp. |

### Data model

`order` gains duplicates (m rows share one COMID), which would break the existing
`IdIndex<Comid>` COMID→position lookup. Resolved by keeping **two** index spaces:

```
parent space (N = 346,321)          sub-reach space (N' = Σ min(m,M))
  parent_order[p] = COMID             order[i]        = COMID of i's parent
  IdIndex<Comid> built HERE           parent_offset[p]..parent_offset[p+1]
                                        = contiguous rows owned by parent p
                                      m_p = parent_offset[p+1] - parent_offset[p]
```

Pieces are contiguous and ordered upstream→downstream, so parent `p`'s **outlet**
is `parent_offset[p+1] - 1`. Because parents are already in topological order and
each chain runs low→high index, the expanded graph stays topologically ordered and
lower-triangular for free.

```
   BEFORE                       AFTER (m=3)
   U ──> P ──> D                U₂ ──> P₀ ──> P₁ ──> P₂ ──> D₀
                                       └─ q'/3   q'/3   q'/3
   len(P) = L                   len(Pᵢ) = L/3,  slope unchanged
   gauge@P → row(P)             gauge@P → P₂  (outlet = last piece)
```

---

### Task 1: Config block `params.subdivision`

**Files:**
- Modify: `src/config.rs`
- Test: `tests/subdivide.rs` (create)

**Interfaces:**
- Produces: `Subdivision { enabled: bool, max_pieces: usize, reference_n: f32,
  reference_discharge_exponent: f32, reference_discharge_coefficient: f32 }`,
  reachable as `cfg.params.subdivision`.

- [ ] **Step 1: Write the failing tests**

Create `tests/subdivide.rs`. Follow the inline-YAML-to-tempfile style already used
in `tests/ddr_match_flag.rs` (read it first; `Config::from_yaml_file` is the loader).

```rust
#[test]
fn subdivision_defaults_to_disabled() {
    let cfg = load_cfg(&yaml_with_params(""));
    assert!(!cfg.params.subdivision.enabled);
    assert_eq!(cfg.params.subdivision.max_pieces, 8);
}

#[test]
fn subdivision_rejects_max_pieces_below_one() {
    let err = try_load_cfg(&yaml_with_params(
        "  subdivision:\n    enabled: true\n    max_pieces: 0\n",
    ))
    .expect_err("must reject");
    assert!(err.to_string().contains("max_pieces"), "got: {err}");
}

#[test]
fn subdivision_enabled_loads() {
    let cfg = load_cfg(&yaml_with_params(
        "  subdivision:\n    enabled: true\n    max_pieces: 4\n",
    ));
    assert!(cfg.params.subdivision.enabled);
    assert_eq!(cfg.params.subdivision.max_pieces, 4);
}
```

- [ ] **Step 2: Run to verify it fails**

`cargo test --test subdivide` → FAIL (`no field subdivision`).

- [ ] **Step 3: Implement**

In `src/config.rs`, mirror how `params.leakance` / `kan_head.disaggregation` declare a
nested optional block with serde defaults (grep for `disaggregation` and copy the
pattern). Add:

```rust
#[derive(Debug, Clone, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Subdivision {
    #[serde(default)]
    pub enabled: bool,
    /// Hard cap on pieces per reach. Uncapped subdivision is infeasible:
    /// 13.2x reaches and 9.2x solver critical path, and Sum(m) cannot be
    /// pinned down (2.3M-10.5M across defensible reference-flow choices).
    /// Capping bounds the cost AND makes it estimable (+/-12% at M=8).
    #[serde(default = "default_max_pieces")]
    pub max_pieces: usize,
    /// Manning's n used ONLY to compute the reference celerity that sets m.
    /// Deliberately NOT taken from a checkpoint: the graph must not depend on
    /// training state. 0.05 is the trained CONUS median.
    #[serde(default = "default_reference_n")]
    pub reference_n: f32,
    /// Reference discharge Q_ref = coefficient * uparea_km2^exponent (m3/s).
    #[serde(default = "default_ref_q_coeff")]
    pub reference_discharge_coefficient: f32,
    #[serde(default = "default_ref_q_exp")]
    pub reference_discharge_exponent: f32,
    /// Short reaches get their length clamped UP to
    /// `min_length_fraction * c_ref * dt`, giving `Cr <= 1/min_length_fraction`
    /// at the reference flow. 1.0 targets Cr = 1; 0.5 targets Cr <= 2 (the
    /// non-negativity bound) with half the length distortion; 0.0 disables the
    /// clamp entirely, leaving short reaches over-Courant.
    ///
    /// This is a BUILD-TIME constant, unlike the runtime K floor in
    /// `enforce_positivity`. It therefore has no gradient path and cannot
    /// create the `X ~ Cr ~ 1/n` coupling that drove n to its floor.
    #[serde(default = "default_min_length_fraction")]
    pub min_length_fraction: f32,
}

fn default_max_pieces() -> usize { 8 }
fn default_reference_n() -> f32 { 0.05 }
fn default_ref_q_coeff() -> f32 { 0.01 }
fn default_ref_q_exp() -> f32 { 0.9 }
fn default_min_length_fraction() -> f32 { 1.0 }

impl Default for Subdivision {
    fn default() -> Self {
        Self {
            enabled: false,
            max_pieces: default_max_pieces(),
            reference_n: default_reference_n(),
            reference_discharge_coefficient: default_ref_q_coeff(),
            reference_discharge_exponent: default_ref_q_exp(),
            min_length_fraction: default_min_length_fraction(),
        }
    }
}
```

Add `#[serde(default)] pub subdivision: Subdivision` to `Params` and to `ParamsRaw`
(follow exactly how `ddr_match` is threaded through both). Add a validator beside
`validate_ddr_match` (`src/config.rs:804`) and call it from the same site
(`config.rs:~808`):

```rust
fn validate_subdivision(cfg: &Config) -> std::result::Result<(), String> {
    let s = &cfg.params.subdivision;
    if s.enabled && s.max_pieces < 1 {
        return Err("params.subdivision: `max_pieces` must be >= 1".to_string());
    }
    if s.enabled && cfg.params.use_cuda_graphs {
        return Err(
            "params.subdivision: `enabled: true` requires `use_cuda_graphs: false` \
             — the captured graph is sized to a fixed reach count."
                .to_string(),
        );
    }
    Ok(())
}
```

- [ ] **Step 4: Verify** — `cargo test --test subdivide` → PASS.

- [ ] **Step 5: Commit**

```bash
git add src/config.rs tests/subdivide.rs
git commit -m "feat(config): params.subdivision block, default off"
```

---

### Task 2: Reference celerity and the two-sided reach plan

**Files:**
- Create: `src/adjacency/subdivide.rs`
- Modify: `src/adjacency/mod.rs` (add `pub mod subdivide;`)
- Test: `tests/subdivide.rs`

**Interfaces:**
- Consumes: `Subdivision` from Task 1.
- Produces:
  ```rust
  pub fn reference_celerity(uparea_km2: f32, slope: f32, cfg: &Subdivision) -> f32;

  /// Both sides of the two-sided rule. `pieces[i]` is how many sub-reaches
  /// parent `i` becomes; `length_m[i]` is its (possibly clamped) total length,
  /// which Task 3 then divides by `pieces[i]`.
  pub struct ReachPlan { pub pieces: Vec<u32>, pub length_m: Vec<f32> }

  pub fn plan_reaches(
      length_m: &[f32], slope: &[f32], uparea_km2: &[f32],
      dt_seconds: f32, cfg: &Subdivision,
  ) -> ReachPlan;
  ```

`reference_celerity` mirrors the solver's own chain rather than inventing one. Use
the wide-channel approximation `c = (5/3)·v` here — **this is deliberate**: it is
only setting `m`, the cap dominates the answer, and pulling in the full trapezoidal
`β` would require `p_spatial`/`q_spatial`, which are learned.

- [ ] **Step 1: Write the failing tests**

```rust
use ddrs::adjacency::subdivide::{plan_reaches, reference_celerity, ReachPlan};
use ddrs::config::Subdivision;

fn cfg(max_pieces: usize) -> Subdivision {
    Subdivision { enabled: true, max_pieces, ..Default::default() }
}

#[test]
fn celerity_rises_with_slope_and_area() {
    let c = cfg(8);
    let lo = reference_celerity(100.0, 1e-3, &c);
    assert!(reference_celerity(100.0, 1e-2, &c) > lo, "steeper must be faster");
    assert!(reference_celerity(10_000.0, 1e-3, &c) > lo, "bigger must be faster");
    assert!(lo > 0.0 && lo < 15.0, "celerity {lo} outside physical range");
}

#[test]
fn long_reaches_split_and_are_capped() {
    let c = cfg(4);
    let p = plan_reaches(&[200_000.0], &[1e-4], &[30.0], 3600.0, &c);
    assert_eq!(p.pieces[0], 4, "must clamp to max_pieces");
    assert_eq!(p.length_m[0], 200_000.0, "long reaches keep their true length");
}

#[test]
fn short_reaches_are_length_clamped_not_split() {
    let c = cfg(8);
    // 50 m reach with a fast celerity: far below dx_target, so it stretches.
    let p = plan_reaches(&[50.0], &[1e-2], &[10_000.0], 3600.0, &c);
    assert_eq!(p.pieces[0], 1, "short reach must not split");
    let dx = reference_celerity(10_000.0, 1e-2, &c) * 3600.0;
    assert!((p.length_m[0] - dx).abs() < 1e-3,
        "expected clamp to dx_target {dx}, got {}", p.length_m[0]);
    assert!(p.length_m[0] > 50.0, "clamp must lengthen, not shorten");
}

#[test]
fn min_length_fraction_zero_disables_the_clamp() {
    let mut c = cfg(8);
    c.min_length_fraction = 0.0;
    let p = plan_reaches(&[50.0], &[1e-2], &[10_000.0], 3600.0, &c);
    assert_eq!(p.length_m[0], 50.0, "clamp must be off");
}

#[test]
fn disabled_is_an_exact_no_op() {
    let mut c = cfg(8);
    c.enabled = false;
    let p = plan_reaches(&[200_000.0, 50.0], &[1e-4, 1e-2], &[30.0, 10_000.0], 3600.0, &c);
    assert_eq!(p.pieces, vec![1, 1]);
    assert_eq!(p.length_m, vec![200_000.0, 50.0], "lengths must be untouched");
}

#[test]
fn degenerate_input_never_yields_zero_pieces_or_zero_length() {
    let c = cfg(8);
    let p = plan_reaches(&[5000.0, 0.0], &[0.0, 0.0], &[0.0, 0.0], 3600.0, &c);
    assert!(p.pieces.iter().all(|&v| v >= 1), "pieces must be >= 1, got {:?}", p.pieces);
    assert!(p.length_m.iter().all(|&v| v > 0.0),
        "length must be > 0 (a 0 m reach gives K = 0 and c1 = 1), got {:?}", p.length_m);
}
```

- [ ] **Step 2: Run to verify it fails** — `cargo test --test subdivide` → FAIL (module missing).

- [ ] **Step 3: Implement**

```rust
//! Static reach subdivision so Cr = c*dt/dx lands near 1.
//!
//! HEC-HMS picks the space step as `dx = c*dt` (Technical Reference Manual,
//! Muskingum-Cunge Model). ddrs historically used the full MERIT reach length,
//! giving median Cr = 0.226 — reaches ~4.4x too long. See
//! `.claude/REACH-SUBDIVISION.md`.

use crate::config::Subdivision;

/// Reference celerity (m/s) used ONLY to choose the piece count.
///
/// Mirrors the solver's S15/S17 chain with the wide-channel ratio c = (5/3)*v
/// instead of the exact trapezoidal beta: this only sets `m`, the cap dominates
/// the result, and beta needs the learned p_spatial/q_spatial.
pub fn reference_celerity(uparea_km2: f32, slope: f32, cfg: &Subdivision) -> f32 {
    // Slope floor mirrors `attribute_minimums.slope` (mmc.rs:208).
    let s = slope.max(1e-3);
    let q_ref = cfg.reference_discharge_coefficient
        * uparea_km2.max(0.0).powf(cfg.reference_discharge_exponent);
    // Hydraulic radius via a wide-channel regime relation; the exponent 0.4 is
    // the Leopold & Maddock downstream depth exponent.
    let r = (q_ref.max(1e-3)).powf(0.4).max(0.01);
    let v = (1.0 / cfg.reference_n) * r.powf(2.0 / 3.0) * s.sqrt();
    (v * (5.0 / 3.0)).clamp(0.01, 15.0)
}

pub struct ReachPlan {
    pub pieces: Vec<u32>,
    pub length_m: Vec<f32>,
}

/// The two-sided rule. Long reaches split; short reaches have their length
/// clamped UP to `dx_target`. Merging short reaches is deliberately not done:
/// a short reach can carry two upstream tributaries or have a parallel
/// tributary joining below it, so collapsing it would destroy junctions.
pub fn plan_reaches(
    length_m: &[f32],
    slope: &[f32],
    uparea_km2: &[f32],
    dt_seconds: f32,
    cfg: &Subdivision,
) -> ReachPlan {
    if !cfg.enabled {
        return ReachPlan { pieces: vec![1; length_m.len()], length_m: length_m.to_vec() };
    }
    let mut pieces = Vec::with_capacity(length_m.len());
    let mut lengths = Vec::with_capacity(length_m.len());
    for ((&l, &s), &a) in length_m.iter().zip(slope).zip(uparea_km2) {
        let dx_target = reference_celerity(a, s, cfg) * dt_seconds;
        // Short reach: stretch it so Cr ~ 1 at the reference flow. This is a
        // STATIC constant, unlike the runtime K floor in `enforce_positivity`
        // — no gradient path, so it cannot pull `n` toward its bound.
        let l_eff = (l.max(0.0)).max(dx_target * cfg.min_length_fraction);
        // Long reach: split.
        let m = ((l_eff / dx_target).ceil().max(1.0) as u32).min(cfg.max_pieces as u32);
        pieces.push(m);
        lengths.push(l_eff);
    }
    ReachPlan { pieces, length_m: lengths }
}
```

- [ ] **Step 4: Verify** — `cargo test --test subdivide` → PASS.

- [ ] **Step 5: Commit**

```bash
git add src/adjacency/subdivide.rs src/adjacency/mod.rs tests/subdivide.rs
git commit -m "feat(adjacency): two-sided reach plan — split long, clamp short"
```

---

### Task 3: Graph expansion

**Files:**
- Modify: `src/adjacency/subdivide.rs`
- Test: `tests/subdivide.rs`

**Interfaces:**
- Consumes: `ReachPlan { pieces, length_m }` from Task 2; `ConusAdjacency { order,
  rows, cols, length_m, slope, dropped_comids }` from `src/adjacency/build.rs:63-77`.
- Produces:
  ```rust
  pub struct SubdividedAdjacency {
      pub order: Vec<i32>,          // COMID per sub-reach row (duplicated)
      pub parent_order: Vec<i32>,   // original COMID order, length N
      pub parent_offset: Vec<i32>,  // length N+1; rows [o[p], o[p+1]) belong to p
      pub rows: Vec<i32>,
      pub cols: Vec<i32>,
      pub length_m: Vec<f32>,
      pub slope: Vec<f32>,
  }
  pub fn subdivide(adj: &ConusAdjacency, plan: &ReachPlan) -> SubdividedAdjacency;
  ```

**Topology rules.** Parent `p` owns rows `[off[p], off[p+1])` ordered
upstream→downstream. Internal chain edges connect consecutive pieces. An external
edge `u -> p` (u upstream of p) becomes `outlet(u) -> off[p]`, where
`outlet(u) = off[u+1] - 1`. Each piece gets `plan.length_m[p] / m` — note this is
the **clamped** length from Task 2, not `adj.length_m[p]`. Slope is unchanged.

- [ ] **Step 1: Write the failing tests**

```rust
use ddrs::adjacency::build::ConusAdjacency;
use ddrs::adjacency::subdivide::subdivide;

/// Chain of 3 reaches: 0 -> 1 -> 2 (rows = downstream, cols = upstream,
/// so rows[k] >= cols[k]).
fn chain3() -> ConusAdjacency {
    ConusAdjacency {
        order: vec![100, 200, 300],
        rows: vec![1, 2],
        cols: vec![0, 1],
        length_m: vec![3000.0, 6000.0, 900.0],
        slope: vec![1e-3, 2e-3, 3e-3],
        dropped_comids: vec![],
    }
}

/// Explicit plan so these tests exercise expansion alone, independent of the
/// celerity heuristic in Task 2.
fn plan(pieces: Vec<u32>, length_m: Vec<f32>) -> ReachPlan {
    ReachPlan { pieces, length_m }
}

#[test]
fn expansion_preserves_total_length_and_slope() {
    let s = subdivide(&chain3(), &plan(vec![3, 2, 1], chain3().length_m.clone()));
    assert_eq!(s.length_m.len(), 6);
    for (p, &m) in [3u32, 2, 1].iter().enumerate() {
        let lo = s.parent_offset[p] as usize;
        let hi = s.parent_offset[p + 1] as usize;
        assert_eq!(hi - lo, m as usize, "parent {p} piece count");
        let total: f32 = s.length_m[lo..hi].iter().sum();
        assert!((total - chain3().length_m[p]).abs() < 1e-3,
            "parent {p} length not conserved: {total}");
        assert!(s.slope[lo..hi].iter().all(|&v| v == chain3().slope[p]),
            "slope must be inherited unchanged");
    }
}

#[test]
fn expansion_stays_lower_triangular_and_topological() {
    let s = subdivide(&chain3(), &plan(vec![3, 2, 1], chain3().length_m.clone()));
    for (&r, &c) in s.rows.iter().zip(s.cols.iter()) {
        assert!(r > c, "edge {c}->{r} violates strict lower-triangular ordering");
    }
}

#[test]
fn expansion_edge_count_is_original_plus_internal_links() {
    let s = subdivide(&chain3(), &plan(vec![3, 2, 1], chain3().length_m.clone()));
    // 2 original edges + (3-1) + (2-1) + (1-1) internal = 5
    assert_eq!(s.rows.len(), 5, "rows: {:?} cols: {:?}", s.rows, s.cols);
}

#[test]
fn external_edges_land_on_parent_outlet_and_inlet() {
    let s = subdivide(&chain3(), &plan(vec![3, 2, 1], chain3().length_m.clone()));
    // parent0 rows 0..3, parent1 rows 3..5, parent2 row 5.
    // edge 0->1 becomes outlet(0)=2 -> inlet(1)=3
    assert!(s.rows.iter().zip(&s.cols).any(|(&r, &c)| c == 2 && r == 3),
        "missing 2->3; rows {:?} cols {:?}", s.rows, s.cols);
    // edge 1->2 becomes outlet(1)=4 -> inlet(2)=5
    assert!(s.rows.iter().zip(&s.cols).any(|(&r, &c)| c == 4 && r == 5),
        "missing 4->5; rows {:?} cols {:?}", s.rows, s.cols);
}

#[test]
fn all_ones_is_an_exact_identity() {
    let a = chain3();
    let s = subdivide(&a, &plan(vec![1, 1, 1], a.length_m.clone()));
    assert_eq!(s.order, a.order);
    assert_eq!(s.rows, a.rows);
    assert_eq!(s.cols, a.cols);
    assert_eq!(s.length_m, a.length_m);
    assert_eq!(s.parent_offset, vec![0, 1, 2, 3]);
}

#[test]
fn expansion_uses_the_clamped_length_not_the_raw_one() {
    let a = chain3();                       // reach 2 is only 900 m
    let clamped = vec![3000.0, 6000.0, 4000.0];  // reach 2 stretched to 4 km
    let s = subdivide(&a, &plan(vec![1, 1, 1], clamped));
    assert_eq!(s.length_m[2], 4000.0,
        "must use ReachPlan.length_m, not ConusAdjacency.length_m");
}
```

- [ ] **Step 2: Run to verify it fails** — FAIL (`subdivide` not found).

> **Verify the self-edge assumption before implementing.** The expansion asserts
> the input COO has no `r == c` entries. `build.rs:235-240` only rejects `r < c`,
> so `r == c` is *permitted* by that check. Confirm empirically first:
> ```bash
> /home/tbindas/projects/ddrs/ddrs-py/.venv/bin/python -c "
> import zarr, numpy as np
> g = zarr.open('/home/tbindas/projects/ddr/data/merit_conus_adjacency.zarr', mode='r')
> r, c = np.asarray(g['indices_0'][:]), np.asarray(g['indices_1'][:])
> print('self-edges (r==c):', int((r==c).sum()), 'of', r.size)"
> ```
> If the count is non-zero, STOP and report — the design needs a self-edge rule
> before Task 3 can be correct, and every downstream task inherits the error.

- [ ] **Step 3: Implement**

```rust
use crate::adjacency::build::ConusAdjacency;

pub struct SubdividedAdjacency {
    pub order: Vec<i32>,
    pub parent_order: Vec<i32>,
    pub parent_offset: Vec<i32>,
    pub rows: Vec<i32>,
    pub cols: Vec<i32>,
    pub length_m: Vec<f32>,
    pub slope: Vec<f32>,
}

impl SubdividedAdjacency {
    #[inline]
    pub fn inlet(&self, parent: usize) -> usize { self.parent_offset[parent] as usize }
    #[inline]
    pub fn outlet(&self, parent: usize) -> usize {
        self.parent_offset[parent + 1] as usize - 1
    }
    #[inline]
    pub fn pieces(&self, parent: usize) -> usize {
        (self.parent_offset[parent + 1] - self.parent_offset[parent]) as usize
    }
}

pub fn subdivide(adj: &ConusAdjacency, plan: &ReachPlan) -> SubdividedAdjacency {
    let n = adj.order.len();
    assert_eq!(plan.pieces.len(), n, "pieces must be one per parent reach");
    assert_eq!(plan.length_m.len(), n, "lengths must be one per parent reach");

    let mut parent_offset = Vec::with_capacity(n + 1);
    let mut acc: i32 = 0;
    parent_offset.push(0);
    for &m in &plan.pieces {
        acc += m.max(1) as i32;
        parent_offset.push(acc);
    }
    let n_sub = acc as usize;

    let mut order = Vec::with_capacity(n_sub);
    let mut length_m = Vec::with_capacity(n_sub);
    let mut slope = Vec::with_capacity(n_sub);
    let mut rows = Vec::new();
    let mut cols = Vec::new();

    for p in 0..n {
        let m = plan.pieces[p].max(1) as usize;
        let base = parent_offset[p] as usize;
        // plan.length_m, NOT adj.length_m: short reaches were clamped UP in Task 2.
        let piece_len = plan.length_m[p] / m as f32;
        for k in 0..m {
            order.push(adj.order[p]);
            length_m.push(piece_len);
            slope.push(adj.slope[p]);
            if k > 0 {
                // internal chain link: piece k-1 flows into piece k
                cols.push((base + k - 1) as i32);
                rows.push((base + k) as i32);
            }
        }
    }

    // External edges: upstream parent's OUTLET -> downstream parent's INLET.
    for (&r, &c) in adj.rows.iter().zip(adj.cols.iter()) {
        // A self-edge (r == c) would map to outlet(p) -> inlet(p), i.e.
        // off[p]+m-1 -> off[p], which is UPPER triangular and would break the
        // forward-substitution invariant. The COO from `build_conus_adjacency`
        // should carry only off-diagonal edges (the diagonal is synthesized by
        // `CsrPattern::from_sparse`), so this is an assertion, not a filter.
        assert!(r != c, "self-edge on parent {r}: subdivision cannot expand it");
        let up_outlet = parent_offset[c as usize + 1] - 1;
        let down_inlet = parent_offset[r as usize];
        cols.push(up_outlet);
        rows.push(down_inlet);
    }

    SubdividedAdjacency {
        order,
        parent_order: adj.order.clone(),
        parent_offset,
        rows,
        cols,
        length_m,
        slope,
    }
}
```

- [ ] **Step 4: Verify** — `cargo test --test subdivide` → PASS (all 9 tests).

- [ ] **Step 5: Commit**

```bash
git add src/adjacency/subdivide.rs tests/subdivide.rs
git commit -m "feat(adjacency): graph expansion preserving topological order"
```

---

### Task 4: Persist to zarr and invalidate the cache

**Files:**
- Modify: `src/data/store/zarr.rs`, `src/adjacency/cache.rs`, `src/adjacency/build.rs`
- Test: `tests/subdivide.rs`

**Interfaces:**
- Consumes: `SubdividedAdjacency` from Task 3.
- Produces: two new zarr arrays `/parent_order` (i32, `[N]`) and `/parent_offset`
  (i32, `[N+1]`); `ConusAdjacencyStore` gains `parent_order: Vec<Comid>` and
  `parent_offset: Vec<i32>`.

**Critical:** `ConusAdjacencyStore.index: IdIndex<Comid>` (`zarr.rs:29-99`) must be
built from `parent_order`, **not** `order` — after subdivision `order` contains
duplicates and a COMID→row lookup would be ambiguous.

- [ ] **Step 1: Write the failing tests**

```rust
#[test]
fn cache_key_changes_with_every_subdivision_field() {
    use ddrs::adjacency::cache::content_key_for_test as key;
    use ddrs::config::Subdivision;
    let on = Subdivision { enabled: true, ..Default::default() };
    let base = key("fab", "gag", None, &Subdivision::default());
    assert_ne!(base, key("fab", "gag", None, &on), "enabling must invalidate");

    // Every field that feeds `plan_reaches` must invalidate, or a config edit
    // silently reuses a graph built with different geometry.
    for (name, modified) in [
        ("max_pieces",  Subdivision { max_pieces: 4, ..on.clone() }),
        ("reference_n", Subdivision { reference_n: 0.03, ..on.clone() }),
        ("q_coeff",     Subdivision { reference_discharge_coefficient: 0.02, ..on.clone() }),
        ("q_exp",       Subdivision { reference_discharge_exponent: 0.8, ..on.clone() }),
        ("min_len_fr",  Subdivision { min_length_fraction: 0.5, ..on.clone() }),
    ] {
        assert_ne!(key("fab", "gag", None, &on), key("fab", "gag", None, &modified),
            "changing {name} must invalidate the cache");
    }
}

#[test]
fn store_index_maps_comid_to_parent_not_subreach() {
    // Built with pieces [3,2,1]; COMID 200 is parent 1.
    let store = round_trip_store(&chain3(), &[3, 2, 1]);
    assert_eq!(store.parent_offset, vec![0, 3, 5, 6]);
    assert_eq!(store.n, 6, "sub-reach count");
    assert_eq!(store.parent_order.len(), 3, "parent count");
    let p = store.index.get(&Comid(200)).expect("COMID 200 must resolve");
    assert_eq!(p, 1, "index must return the PARENT position, not a sub-reach row");
}
```

Write `round_trip_store` to build → subdivide → write zarr to a `tempfile::TempDir`
→ reload via `ConusAdjacencyStore::open`. Reuse whatever writer
`src/adjacency/cache.rs` already calls (find it; do not invent a new one).

- [ ] **Step 2: Run to verify it fails.**

- [ ] **Step 3: Implement**

Write both new arrays alongside the existing `/order`, `/length_m`, `/slope`,
`/indices_0`, `/indices_1`. In `ConusAdjacencyStore::open`, read them if present and
otherwise synthesize the identity (`parent_order = order`,
`parent_offset = 0..=n`) so **pre-existing stores keep loading unchanged**.

In `src/adjacency/cache.rs`, extend `content_key` (`cache.rs:316-329`) to hash the
two subdivision fields after `BUILDER_VERSION`, and bump `BUILDER_VERSION` by 1:

**Hash ALL SIX fields, not just `enabled` + `max_pieces`.** Every one of
`reference_n`, `reference_discharge_coefficient`, `reference_discharge_exponent`
and `min_length_fraction` changes the reference celerity → `dx_target` → both
`pieces` and the clamped `length_m`, i.e. it changes the built graph. Hashing only
two of them would silently reuse a stale cached adjacency after an edit to any of
the other four.

```rust
// `content_key` takes `s: &Subdivision` as a new parameter (thread it from the
// caller's `cfg.params.subdivision`), so the test can vary one field at a time.
h.update(BUILDER_VERSION.to_le_bytes().as_ref());
h.update(&[s.enabled as u8]);
h.update((s.max_pieces as u32).to_le_bytes().as_ref());
// f32::to_bits gives a stable byte pattern; NaN is impossible here (validated).
h.update(s.reference_n.to_bits().to_le_bytes().as_ref());
h.update(s.reference_discharge_coefficient.to_bits().to_le_bytes().as_ref());
h.update(s.reference_discharge_exponent.to_bits().to_le_bytes().as_ref());
h.update(s.min_length_fraction.to_bits().to_le_bytes().as_ref());
```

Expose `content_key_for_test` as `#[doc(hidden)] pub` so the test can call it.

In `src/adjacency/build.rs`, call `plan_reaches` + `subdivide` after
`build_conus_adjacency` and **before** gauge-subgraph construction, so subgraphs are
cut from the expanded graph. Drainage area comes from the attributes NetCDF
(`catchsize`); thread it in as a `&[f32]` aligned to `order`.

- [ ] **Step 4: Verify** — `cargo test --test subdivide` → PASS.

- [ ] **Step 5: Commit**

```bash
git add src/data/store/zarr.rs src/adjacency/cache.rs src/adjacency/build.rs tests/subdivide.rs
git commit -m "feat(adjacency): persist parent map, invalidate cache on subdivision"
```

---

### Task 5: Distribute q′ across pieces

**Files:**
- Modify: `src/routing/mmc.rs`
- Test: `tests/subdivision_integration.rs` (create)

**Interfaces:**
- Consumes: `parent_offset` via the adjacency inputs reaching `setup_inputs`.
- Produces: `q_prime` divided by each row's parent piece count.

**Placement is exact:** after the clamp at `src/routing/mmc.rs:453`
(`let q_prime_clamped = q_prime.clamp_min(discharge_lb);`) and **before** the first
slice at `mmc.rs:495`. Dividing before the clamp would let the floor `1e-4` be
applied to an undivided value and silently create mass.

- [ ] **Step 1: Write the failing test**

```rust
/// A 1-reach network with constant q' must reach the SAME steady-state outflow
/// whether or not it is subdivided — the pieces split the inflow m ways but
/// chain in series, so the outlet still carries the whole reach's runoff.
#[test]
fn subdivision_conserves_mass_at_steady_state() {
    let un_split = steady_state_outflow(/*pieces*/ 1);
    let split    = steady_state_outflow(/*pieces*/ 4);
    assert!((split - un_split).abs() / un_split < 1e-3,
        "mass not conserved: 1 piece = {un_split}, 4 pieces = {split}");
}
```

- [ ] **Step 2: Run to verify it fails** — the 4-piece case returns ~4× the correct
  outflow, because each piece receives the full `q'`.

- [ ] **Step 3: Implement**

Build a per-row divisor tensor once in `setup_inputs` (beside `length`/`slope` at
`mmc.rs:204-213`) and store it as `self.pieces_per_row: Option<Tensor<Autodiff<I>, 1>>`:

```rust
// One entry per sub-reach row: the piece count of its parent. Used to split
// lateral inflow, mirroring HEC-HMS's C4*(q_L*dx) with q_L per unit length.
let mut divisor = Vec::with_capacity(n);
for p in 0..inputs.adjacency.parent_offset.len() - 1 {
    let m = inputs.adjacency.parent_offset[p + 1] - inputs.adjacency.parent_offset[p];
    for _ in 0..m { divisor.push(m as f32); }
}
self.pieces_per_row = Some(Tensor::from_floats(divisor.as_slice(), &self.device));
```

Then at `mmc.rs:453`:

```rust
let q_prime_clamped = q_prime.clamp_min(discharge_lb);
// Split lateral inflow evenly along the reach. MUST be after the clamp: the
// 1e-4 floor applied to an undivided value would create mass.
let q_prime_clamped = match self.pieces_per_row.as_ref() {
    Some(d) => q_prime_clamped / d.clone().unsqueeze_dim::<2>(0),
    None => q_prime_clamped,
};
```

- [ ] **Step 4: Verify** — `cargo test --test subdivision_integration` → PASS.

- [ ] **Step 5: Commit**

```bash
git add src/routing/mmc.rs tests/subdivision_integration.rs
git commit -m "feat(routing): split lateral inflow across sub-reaches"
```

---

### Task 6: Gather KAN parameters onto sub-reaches

**Files:**
- Modify: `src/training/forward.rs`
- Test: `tests/subdivision_integration.rs`

**Interfaces:**
- Consumes: KAN outputs `HashMap<String, Tensor<B, 1>>` shaped `[N_parent]`
  (`src/nn/kan_head.rs:215-244`).
- Produces: the same map gathered to `[N_sub]`.

**Run the KAN at parent resolution and gather** — do not duplicate attribute rows.
Same result, 4.77× less KAN compute, and `select`'s backward is a scatter-add, so
each parent correctly receives the summed gradient from all its pieces.

- [ ] **Step 1: Write the failing test**

```rust
#[test]
fn every_piece_inherits_its_parents_parameters() {
    // parent_offset [0,3,5,6]; parent params [0.02, 0.05, 0.10]
    let gathered = gather_for_test(&[0.02, 0.05, 0.10], &[0, 3, 5, 6]);
    assert_eq!(gathered, vec![0.02, 0.02, 0.02, 0.05, 0.05, 0.10]);
}

#[test]
fn gradient_sums_back_to_the_parent() {
    // d(sum of gathered)/d(parent p) must equal that parent's piece count.
    let g = gather_grad_for_test(&[0.02, 0.05, 0.10], &[0, 3, 5, 6]);
    assert_eq!(g, vec![3.0, 2.0, 1.0], "scatter-add must sum piece gradients");
}
```

- [ ] **Step 2: Run to verify it fails.**

- [ ] **Step 3: Implement**

In `src/training/forward.rs` (after `head.forward` at `forward.rs:235-239`,
before `denormalize`), build a row→parent index once and gather:

```rust
// Sub-reaches share their parent's hydraulics: MERIT carries no within-reach
// variation. `select` backward is a scatter-add, so each parent receives the
// summed gradient of all its pieces.
let n_param = n_param.select(0, parent_idx.clone());
let q_param = q_param.select(0, parent_idx.clone());
let p_param = p_param.map(|t| t.select(0, parent_idx.clone()));
```

`parent_idx: Tensor<Int, 1>` is built from `parent_offset` and carried on
`RoutingTensors`. When subdivision is off it is the identity, so make the whole
block conditional on `parent_offset.len() - 1 != n_rows` to keep the disabled path
byte-identical.

- [ ] **Step 4: Verify** — `cargo test --test subdivision_integration` → PASS.

- [ ] **Step 5: Commit**

```bash
git add src/training/forward.rs tests/subdivision_integration.rs
git commit -m "feat(training): gather KAN parameters from parents to sub-reaches"
```

---

### Task 7: Map gauges to their parent's outlet piece

**Files:**
- Modify: `src/data/collate.rs`
- Test: `tests/gauge_mass_conservation.rs`

**Interfaces:**
- Consumes: `compress(unioned, conus_order, ddr_match) -> Result<CompressedAdj>`
  (`src/data/collate.rs:90-207`), `outflow_idx` built at `collate.rs:178-198`.
- Produces: for `ddr_match: false`, `outflow_idx = vec![outlet_row_of(gauge_parent)]`.

A gauge measures everything upstream. With subdivision the parent's whole runoff
arrives at its **last** piece, so that is the row to read. Reading any earlier piece
would drop the downstream fraction of the reach's own lateral inflow — the same
class of bug as `2fe6bee`.

- [ ] **Step 1: Write the failing test**

Add to `tests/gauge_mass_conservation.rs`, reusing its existing 3-reach fixture
(two headwaters into a gauge reach, constant `Q_PRIME = 10.0`, steady state):

```rust
#[test]
fn gauge_conserves_mass_when_its_reach_is_subdivided() {
    // Same topology as `gauge_prediction_conserves_mass_when_not_ddr_match`,
    // but the gauge's own reach is split 4 ways. The answer must not change.
    let un_split = gauge_steady_state(/*pieces*/ 1);
    let split    = gauge_steady_state(/*pieces*/ 4);
    assert!((un_split - 30.0).abs() < 1e-3, "control changed: {un_split}");
    assert!((split - 30.0).abs() < 1e-3,
        "subdivided gauge lost mass: got {split}, expected 30.0");
}
```

- [ ] **Step 2: Run to verify it fails** — expect ~27.5 (the last piece receives
  only 1/4 of the gauge reach's own lateral inflow if the outlet is chosen wrongly).

- [ ] **Step 3: Implement**

In `collate.rs`, thread `parent_offset` into `compress` and replace the
`ddr_match: false` branch (`collate.rs:196-198`):

```rust
// The gauge's whole reach discharges at its LAST piece. Any earlier piece
// omits the downstream part of the reach's own lateral inflow.
gauge_compressed.iter()
    .map(|&g_parent| vec![parent_offset[g_parent + 1] as usize - 1])
    .collect()
```

Leave the `ddr_match: true` branch untouched — it reproduces DDR and is pinned by
`ddr_match_gauge_prediction_omits_the_gauge_reach`.

- [ ] **Step 4: Verify**

```bash
cargo test --test gauge_mass_conservation
```

- [ ] **Step 5: Commit**

```bash
git add src/data/collate.rs tests/gauge_mass_conservation.rs
git commit -m "feat(collate): read gauges at their parent's outlet piece"
```

---

### Task 8: End-to-end validation on real CONUS data

**Files:**
- Modify: `src/bin/probe_courant.rs`, `tests/subdivision_integration.rs`

**Interfaces:**
- Consumes: everything above.
- Produces: measured Cr / X / negative-solve counts with subdivision on vs off.

This is the task that decides whether the change is worth keeping. `probe_courant`
already reports Cr, X, and exact negative-solve counts on the real network — reuse
it rather than writing a new probe.

- [ ] **Step 1: Write the failing test**

```rust
#[test]
fn subdivision_makes_coefficients_non_negative_without_the_clamp() {
    // enforce_positivity OFF, subdivision ON: the whole point is that Cr ~ 1
    // makes c1, c3 >= 0 by construction.
    let r = run_probe(Subdiv::On { max_pieces: 8 }, EnforcePositivity::Off);
    assert!(r.frac_c1_negative < 0.01,
        "c1 negative on {:.2}% of cells", 100.0 * r.frac_c1_negative);
    assert!(r.frac_c3_negative < 0.01,
        "c3 negative on {:.2}% of cells", 100.0 * r.frac_c3_negative);
    assert!(r.median_cr > 0.8 && r.median_cr < 2.0,
        "median Cr = {} is not near 1", r.median_cr);
}

#[test]
fn subdivision_off_is_bit_identical_to_today() {
    let a = run_probe(Subdiv::Off, EnforcePositivity::Off);
    assert_eq!(a.x_sol_bits, GOLDEN_X_SOL_BITS_NO_SUBDIV);
}
```

- [ ] **Step 2: Run to verify it fails.**

- [ ] **Step 3: Add a `--max-pieces` flag to `probe_courant`** and report, for
  subdivision off vs on: sub-reach count, median Cr, `frac Cr > 2`, X percentiles,
  `frac c1 < 0`, `frac c3 < 0`, and exact negative-solve counts.

- [ ] **Step 4: Measure on the real network and record the numbers**

```bash
cargo run --release --bin probe_courant -- \
  --config ddrs.yaml --checkpoint <run>/checkpoints/<epoch> \
  --backend cuda --gauges 1841 --rho 90 --steps 2136 --max-pieces 8
```

**Also report the length-clamp cost**, which is the price of the short-reach branch:
fraction of reaches clamped, the clamp-factor distribution (p50/p95/p99/max), and
total network length inflation. Indicative figures were 34.4 % clamped, median 2.0×,
max 48,597×, +17.1 % total length — but those used a crude reference celerity and
**must be recomputed with the calibrated one**.

**Report honestly.** Subdivision alone cannot fix reaches that are already
`Cr > 2` (17.3 % of the network); only the length clamp reaches them. If
`min_length_fraction` is reduced or disabled, that population returns. Record the
actual residual `frac c1 < 0` / `frac c3 < 0` — do **not** claim a guarantee the
measurement does not support.

- [ ] **Step 5: Run the full gate set**

```bash
cargo test --test subdivide --test subdivision_integration \
           --test gauge_mass_conservation --test adjacency_parity \
           --test positivity_clamp --test cunge_x --test celerity_beta
cargo test --lib
cargo run --release --example compare_ddr_sandbox   # must stay ABSOLUTE MATCH
```

`tests/adjacency_parity.rs` asserts element-for-element equality against an
engine-built store over all 346,321 positions. With subdivision **off** it must
still pass unchanged; if it fails, the disabled path is not a true no-op.

- [ ] **Step 6: Commit**

```bash
git add src/bin/probe_courant.rs tests/subdivision_integration.rs
git commit -m "test(subdivision): real-CONUS Courant and coefficient validation"
```

---

### Task 9: Document

**Files:**
- Create: `.claude/REACH-SUBDIVISION.md`
- Modify: `.claude/PHYSICS-CORRECTIONS.md`, `CLAUDE.md`

- [ ] **Step 1: Write `.claude/REACH-SUBDIVISION.md`** with the ASCII topology
  diagram from this plan's §Data model, the cap-sweep table, the measured
  before/after numbers from Task 8, and the two-index-space explanation.

- [ ] **Step 2: Correct `.claude/PHYSICS-CORRECTIONS.md`.** Its
  §"Why sub-stepping still does not substitute" currently claims subdivision
  "restores dynamic range to the Cunge X". **That is false** — raw X median moves
  only 0.4973 → 0.4815. Replace with the two real justifications (automatic
  non-negativity at Cr≈1; recovery of `X_eff` if `enforce_positivity` is kept) and
  mark it as an erratum, matching the erratum style already in that file.

- [ ] **Step 3: Add the `params.subdivision` block to `CLAUDE.md`'s config notes**,
  including the cap rationale and the retraining requirement.

- [ ] **Step 4: Commit**

```bash
git add .claude/REACH-SUBDIVISION.md .claude/PHYSICS-CORRECTIONS.md CLAUDE.md
git commit -m "docs: reach subdivision design, cap rationale, X erratum"
```

---

## Verification criteria

1. `params.subdivision.enabled: false` is byte-identical to current output —
   proven by `adjacency_parity` and the `GOLDEN_X_SOL_BITS_NO_SUBDIV` test.
2. `compare_ddr_sandbox` still reports ABSOLUTE MATCH.
3. Total `q'` is conserved per parent reach; total length is conserved for
   *split* reaches and increased only for *clamped* ones, by a measured amount.
4. Expanded graph is topologically ordered and strictly lower-triangular.
5. Gauge predictions are unchanged by subdividing the gauge's reach.
6. With subdivision on and `enforce_positivity` **off**, `frac c1 < 0` and
   `frac c3 < 0` both fall below 1 % on the real network, and median Cr ∈ [0.8, 2.0].
7. `frac Cr > 2` and `frac Cr < 0.5` are both measured on the real network and
   reported. The indicative target is ~0 % and ~0.05 % respectively; any material
   residual must be stated, not hidden.
8. The length-clamp distortion (fraction clamped, clamp-factor p95/max, total
   network length inflation) is measured and reported alongside the benefit.

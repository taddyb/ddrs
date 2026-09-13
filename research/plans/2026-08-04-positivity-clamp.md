# Positivity Clamp — Provably Zero Negative Muskingum Solves

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development
> to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax.

**Goal:** Drive `negative solves before clamp` to exactly 0/N by enforcing the
Muskingum non-negativity window `2X ≤ Cr ≤ 2(1−X)` on every reach-timestep,
behind a new `params.enforce_positivity` flag.

**Architecture:** Clamp the *inputs* (K, X), never the coefficients. The identity
`c1+c2+c3 = 1` holds for any `(K, X)`, so clamping inputs preserves mass exactly
while clamping `c3` would not.

**Tech stack:** Rust, BURN 0.21 hand-written `Backward<I,N>` (invariant 4).

---

## The theorem this rests on

With `Cr = Δt/K` and `denom = 2K(1−X) + Δt > 0`:

```
c2 = (2KX + Δt)/denom > 0        always  (K>0, X≥0, Δt>0)
c4 = 2Δt/denom        > 0        always
c1 = (Δt − 2KX)/denom ≥ 0   <=>  Cr ≥ 2X
c3 = (2K(1−X) − Δt)/denom ≥ 0 <=> Cr ≤ 2(1−X)
```

The solve is forward substitution in topological order:
`x[i] = b[i] + c1[i]·Σ_{j∈up(i)} x[j]`, with
`b[i] = c2·(N q_t)[i] + c3·q_t[i] + c4·q'[i]`.

**Claim.** If `c1 ≥ 0` and `c3 ≥ 0` everywhere, then `x ≥ 0` everywhere.
*Proof:* `q_t > 0` (S28 `clamp_min(1e-4)` and the hotstart at `utils.rs:97`),
`q' > 0` (`mmc.rs:453`), so `b ≥ 0`. Induction over the topological order:
headwaters have no upstream so `x = b ≥ 0`; each subsequent `x[i]` is a
non-negative combination of `b[i]` and already-non-negative upstream values. ∎

Verified numerically: `2X ≤ Cr ≤ 2(1−X)` ⟺ `c1,c3 ≥ 0` with 0 mismatches in
200,000 random `(K,X)` draws; 34,364 negatives unclamped → **0** clamped over
800,000 chain solves.

### Why a margin δ is mandatory

At δ=0 the clamp lands exactly on `c1=0` / `c3=0`, and f32 roundoff crosses it:
7,149 `c1<0` and 964 `c3<0` in a 400k f32 sweep. Measured minima:

| δ | min c1 | min c3 | negative solves (240k adversarial) |
|---|---|---|---|
| 0 | −3.3e−08 | −6.8e−08 | >0 |
| 1e−4 | +1.8e−06 | +0.0e+00 | 0 |
| 1e−3 | +1.8e−05 | +5.1e−07 | 0 |
| **1e−2** | **+1.8e−04** | **+5.0e−05** | **0** |

**Use δ = 1e−2** — ~400× f32 eps, and it costs almost nothing (X ceiling tightens
by 1%, K floor rises by 1%).

## The clamp (ddr_match=false AND enforce_positivity only)

```
k_floor = Δt·(1+δ)/2                                  scalar constant
k_musk  = max(k_raw, k_floor)          k_raw = length/celerity
cr      = Δt / k_musk                  =>  cr ∈ (0, 2/(1+δ)]
x_hi_a  = (1−δ)·0.5·cr                 enforces c1 ≥ 0
x_hi_b  = (1−δ)·(1 − 0.5·cr)           enforces c3 ≥ 0
x_eff   = min(x_cunge, x_hi_a, x_hi_b)
```

`x_hi_b > 0` is guaranteed by the K floor (`cr < 2`), so no `clamp_min` is
needed — the min of three positives is positive.

### Known cost (do not hide this in the findings)

> **SUPERSEDED — the table below is WRONG.** It evaluates `X_max` *at* each Cr
> percentile, but `X_max(Cr)` is non-monotone (peaks at `Cr = 1`), so that does
> not give the percentiles of X. The tell: the p95 row came out *below* p75.
> Measured truth (`src/bin/probe_courant.rs`, 1,841 gauges): Cr p50 = **0.226**
> (not 1.09), the cap binds on **95.3 %** of reach-timesteps, and median X used
> falls **0.4976 → 0.0794 (6.3×)**, not to 0.45. See
> `.claude/PHYSICS-CORRECTIONS.md` §`enforce_positivity` for the corrected
> tables. Kept here only so the error is traceable.

At real CONUS Courant percentiles the cap binds **everywhere**:

| | Cr raw | Cr after floor | X_max | vs Cunge 0.49 |
|---|---|---|---|---|
| p5 | 0.19 | 0.190 | 0.094 | cap binds |
| p25 | 0.54 | 0.540 | 0.267 | cap binds |
| p50 | 1.09 | 1.090 | 0.450 | cap binds |
| p75 | 2.46 | 1.980 | 0.010 | cap binds |
| p95 | 10.20 | 1.980 | 0.010 | cap binds |

So this **partially overrides the Cunge X** (commit `54ec215`): X becomes
stability-dictated rather than diffusion-matched, except near Cr≈1. The fully
capped limit is benign, not degenerate — `c1=0.495, c2=0.505, c3=0.000`, i.e.
`Q_out(t+1) ≈ ½I(t+1) + ½I(t)`, a mass-conserving 2-step lag for a sub-grid reach.

The K floor inflates travel time only where `Cr > 2` (≈ the fastest quartile;
4.2× at p5). A reach with `K < Δt/2` is sub-grid — the timestep cannot resolve
its transit, and the unclamped scheme expressed that as oscillation that S28
clamped to 1e-4 anyway.

**Sub-stepping does not substitute for this.** The Task-5 abandonment analysed
landing *inside* `[0.98,1.02]` at fixed X≈0.49. Here the K constraint is
one-sided (`K ≥ Δt/2n`), which `n_sub=8` satisfies even at p5=425 s — but
shrinking Δt drives already-slow reaches *further* from Cr=1 (p95 would go
Cr 0.19→0.024, X_max→0.012). Only per-reach Δx puts every reach near Cr≈1.
Record this; do not re-attempt global sub-stepping.

---

## File structure

- `src/config.rs` — add `params.enforce_positivity: bool` (default **false**),
  plumb through `ParamsRaw`, validate it requires `ddr_match: false`.
- `src/routing/mmc_op.rs` — forward S18′/S19′; backward B18′/B19′.
- `tests/positivity_clamp.rs` — new: positive control, zero-guarantee,
  partition identity, off-parity, falsifiable gradcheck.
- `.claude/PHYSICS-CORRECTIONS.md` — new section + blast-radius row.

---

### Task 1: Config flag `enforce_positivity`

**Files:** Modify `src/config.rs`; Test: `tests/ddr_match_flag.rs`

- [ ] **Step 1: Write the failing tests**

Append to `tests/ddr_match_flag.rs`:

```rust
#[test]
fn enforce_positivity_defaults_to_false() {
    let cfg = load_mock_config();
    assert!(!cfg.params.enforce_positivity,
        "enforce_positivity must default to false so existing runs are unchanged");
}

#[test]
fn enforce_positivity_requires_corrected_physics() {
    // ddr_match: true + enforce_positivity: true must be rejected at load:
    // the clamp changes K and X, which would break compare_ddr_sandbox.
    let yaml = mock_config_yaml_with("ddr_match: true\n  enforce_positivity: true\n");
    let err = load_config_str(&yaml).expect_err("must reject");
    assert!(err.to_string().contains("enforce_positivity"),
        "error must name the offending key, got: {err}");
}
```

Follow the exact helper names already used in `tests/ddr_match_flag.rs`; if they
differ, adapt the test to the existing helpers rather than inventing new ones.

- [ ] **Step 2: Run to verify it fails**

`cargo test --test ddr_match_flag` → FAIL (`no field enforce_positivity`).

- [ ] **Step 3: Implement**

In `src/config.rs`, mirror exactly how `ddr_match` is declared, defaulted, and
plumbed through `ParamsRaw` (search for `ddr_match` and copy the pattern).
Default `false`. Extend the existing `validate_ddr_match` (≈line 871) — or add a
sibling validator called from the same place — to reject
`enforce_positivity && ddr_match`.

- [ ] **Step 4: Verify** — `cargo test --test ddr_match_flag` → PASS.

- [ ] **Step 5: Commit**

```bash
git add src/config.rs tests/ddr_match_flag.rs
git commit -m "feat(config): enforce_positivity flag, requires ddr_match: false"
```

---

### Task 2: Forward — K floor and X stability cap

**Files:** Modify `src/routing/mmc_op.rs`; Test: `tests/positivity_clamp.rs`

- [ ] **Step 1: Write the failing tests**

Create `tests/positivity_clamp.rs`. Build a stress network whose reaches span
`Cr` from ≈0.02 to ≈10 (short/fast reaches with `Cr > 2` are the ones that
produce negatives). Use `reset_negative_solve_stats` /
`negative_solve_stats` from `src/routing/mmc_op.rs` and
`enable_negative_discharge_tracking`.

```rust
#[test]
fn positive_control_negatives_exist_without_the_clamp() {
    let (neg, tot) = run_stress_network(/* enforce_positivity */ false);
    assert!(neg > 0, "fixture must actually produce negatives, got {neg}/{tot}");
}

#[test]
fn clamp_drives_negatives_to_exactly_zero() {
    let (neg, tot) = run_stress_network(/* enforce_positivity */ true);
    assert_eq!(neg, 0, "expected exactly zero negative solves, got {neg}/{tot}");
    assert!(tot > 0, "fixture must have solved something");
}

#[test]
fn partition_identity_survives_the_clamp() {
    // c1 + c2 + c3 == 1 to f32 tolerance on every reach, clamp on.
    let (c1, c2, c3) = coefficients_from_stress_network(true);
    for i in 0..c1.len() {
        assert!((c1[i] + c2[i] + c3[i] - 1.0).abs() < 1e-5,
            "reach {i}: c1+c2+c3 = {}", c1[i] + c2[i] + c3[i]);
    }
}

#[test]
fn coefficients_are_non_negative_with_margin() {
    let (c1, _, c3) = coefficients_from_stress_network(true);
    assert!(c1.iter().all(|&v| v >= 0.0), "min c1 = {:?}", c1.iter().cloned().fold(f32::INFINITY, f32::min));
    assert!(c3.iter().all(|&v| v >= 0.0), "min c3 = {:?}", c3.iter().cloned().fold(f32::INFINITY, f32::min));
}

#[test]
fn off_parity_byte_identical_when_disabled() {
    // enforce_positivity: false must reproduce current ddr_match:false output exactly.
    let a = route_stress_network(false);
    let b = route_stress_network_baseline(); // pre-change code path
    assert_eq!(a, b, "disabled clamp must be a byte-identical no-op");
}
```

- [ ] **Step 2: Run to verify they fail**

`cargo test --test positivity_clamp` → the positive control passes, the
zero-guarantee test FAILS with a non-zero count.

- [ ] **Step 3: Implement the forward**

In `forward_chain_inner` (`src/routing/mmc_op.rs`), replace the current
`k_muskingum` / `x_eff` block. Keep `ddr_match: true` and
`enforce_positivity: false` byte-identical — guard with a bool that is only true
when BOTH `!ddr_match` and `enforce_positivity`.

```rust
const POSITIVITY_DELTA: f32 = 1e-2;

let k_raw = length_in.clone() / celerity.clone();
let k_muskingum = if enforce_pos {
    // A reach with K < dt/2 is sub-grid: the timestep cannot resolve its
    // transit, and the unclamped scheme expressed that as oscillation that
    // S28 clamped to 1e-4 anyway. Flooring K makes the coarse-graining
    // explicit and puts Cr in (0, 2/(1+d)], the feasible region for X.
    k_raw.clone().clamp_min(dt * (1.0 + POSITIVITY_DELTA) / 2.0)
} else {
    k_raw.clone()
};
```

then after `x_cunge` is formed (the existing `((-w + 1.0) * 0.5).clamp(0.0, 0.5)`):

```rust
let x_eff = if enforce_pos {
    let cr = k_muskingum.clone().recip() * dt;
    let hi_a = cr.clone() * (0.5 * (1.0 - POSITIVITY_DELTA));          // c1 >= 0
    let hi_b = (-cr.clone() * 0.5 + 1.0) * (1.0 - POSITIVITY_DELTA);   // c3 >= 0
    x_cunge.clone().min_pair(hi_a).min_pair(hi_b)
} else {
    x_cunge.clone()
};
```

Use whatever elementwise-min helper BURN 0.21 exposes in this codebase (check
how `clamp`/`min` are already called nearby); do not add a dependency.

- [ ] **Step 4: Verify** — `cargo test --test positivity_clamp` → all PASS.

- [ ] **Step 5: Commit**

```bash
git add src/routing/mmc_op.rs tests/positivity_clamp.rs
git commit -m "feat(routing): K floor + X stability cap under enforce_positivity"
```

---

### Task 3: Backward — B18′ K-floor mask and B19′ three-way min

**Files:** Modify `src/routing/mmc_op.rs`; Test: `tests/positivity_clamp.rs`

The forward introduced two new gradient facts:

1. `k_musk = max(k_raw, k_floor)` — gradient is masked where floored.
2. `x_eff = min(x_cunge, hi_a, hi_b)` — gradient goes to exactly one branch, and
   the `hi_a`/`hi_b` branches open a **new path** `x_eff → cr → k_musk → celerity`
   that did not exist before.

Derivatives:

```
d(hi_a)/d(cr) = +0.5·(1−δ)
d(hi_b)/d(cr) = −0.5·(1−δ)
d(cr)/d(k_musk) = −Δt / k_musk²
d(k_musk)/d(k_raw) = 1 where k_raw > k_floor, else 0
```

Accumulation order (critical — mirrors the existing B18/B19 ordering note):

```
gk_musk  = (existing g_2k_total · 2)                    from c1..c4
         + gx_eff·[mask_a]·(+0.5(1−δ)) ·(−Δt/k_musk²)   NEW
         + gx_eff·[mask_b]·(−0.5(1−δ)) ·(−Δt/k_musk²)   NEW
gk_raw   = gk_musk · mask(k_raw > k_floor)              NEW
gcelerity = −gk_raw · length / celerity²                (existing B18)
          + XGrads.g_celerity                           (existing B19, now
                                                         masked by mask_cunge)
```

The existing `XGrads` path (`∂X/∂Q`, `∂X/∂B`, `∂X/∂c`) must additionally be
masked by `mask_cunge` — where `hi_a` or `hi_b` won the min, `x_cunge` is not
the active branch and receives nothing.

- [ ] **Step 1: Write the failing gradcheck**

Add to `tests/positivity_clamp.rs`, following the pattern in `tests/cunge_x.rs`:

```rust
#[test]
fn positivity_clamp_gradcheck() {
    // Analytical vs central finite difference on n, p_spatial, q_spatial.
    // Fixture MUST exercise all three min branches and both sides of the
    // K floor - assert that before comparing gradients.
    let f = stress_fixture();
    assert!(f.frac_branch_cunge() > 0.05 && f.frac_branch_hi_a() > 0.05
         && f.frac_branch_hi_b() > 0.05,
        "fixture is vacuous: branch mix {:?}", f.branch_mix());
    assert!(f.frac_k_floored() > 0.05 && f.frac_k_floored() < 0.95,
        "fixture must straddle the K floor, got {}", f.frac_k_floored());
    let (analytic, fd) = f.grads();
    let rel = rel_err(&analytic, &fd);
    assert!(rel < 1e-2, "gradcheck rel err {rel:.3e}");
}
```

**The fixture-mix assertions are not optional.** The Cunge-X gradcheck was
initially vacuous because at 1000 m reaches `W ≈ 1.6` saturated the clamp on
every reach and all four tests passed with the backward terms deleted. Prove
falsifiability the same way it was proved there.

- [ ] **Step 2: Run to verify it fails** — non-trivial `rel` before the backward exists.

- [ ] **Step 3: Implement the backward** per the accumulation order above.

- [ ] **Step 4: Verify falsifiability**

Temporarily delete the two new `gk_musk` terms, re-run, confirm the gradcheck
FAILS, then restore. Record both numbers in the commit message.

- [ ] **Step 5: Run the full gate set**

```bash
cargo test --test positivity_clamp --test cunge_x --test celerity_beta \
           --test sparse_gradcheck --test mmc --test leakance_gradcheck
cargo run --release --example compare_ddr_sandbox   # must still be ABSOLUTE MATCH
```

- [ ] **Step 6: Commit**

```bash
git add src/routing/mmc_op.rs tests/positivity_clamp.rs
git commit -m "feat(routing): exact backward for the positivity clamp"
```

---

### Task 4: Document

**Files:** Modify `.claude/PHYSICS-CORRECTIONS.md`

- [ ] **Step 1:** Add an S18′/S19′ branch to the ASCII dataflow, a section
  carrying the theorem, the δ table, and the CONUS Cr/X_max cost table above.
- [ ] **Step 2:** Add the blast-radius row:
  `| Positivity clamp | S18′,S19′ | gk_musk mask + cr path | yes (flag-gated) | positivity_clamp |`
- [ ] **Step 3:** Correct the Task-5 note to record *why* sub-stepping still
  does not substitute (one-sided vs two-sided constraint; slow reaches move away
  from Cr≈1).
- [ ] **Step 4: Commit**

```bash
git add .claude/PHYSICS-CORRECTIONS.md
git commit -m "docs: positivity clamp theorem, delta margin, and its cost"
```

---

## Verification criteria

1. `cargo test --test positivity_clamp` — all pass, gradcheck proven falsifiable.
2. `compare_ddr_sandbox` still ABSOLUTE MATCH (untouched: needs `ddr_match: true`).
3. `enforce_positivity: false` byte-identical to current output.
4. A short real run logs `negative solves before clamp: 0/N (0.000%)`.

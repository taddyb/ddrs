# Reach subdivision (`params.subdivision`, variable Δx)

> ## STATUS: **NO-GO for "non-negative by construction"** (2026-08-05)
>
> The premise was wrong. Subdivision was built to drive `Cr = Δt/K` to ≈ 1
> network-wide so that `c1` and `c3` would both be non-negative without the
> runtime `enforce_positivity` clamp. **Measured on the real network it makes
> `frac c1 < 0` WORSE — 93.0 % → 98.8 % at `max_pieces: 8`** — while costing
> 2.05× the rows, 1.5× the step time and +23.9 % total channel length.
>
> The code is correct, gated off by default, and **stays in-tree**: it is the
> only measurement apparatus for this question, it does fix the `Cr > 2` /
> `c3 < 0` population, and the hot-start division it added is a real bug fix.
> Do not re-open the "Cr ≈ 1 ⇒ non-negative coefficients" argument without
> reading §Why it fails.

Config: `params.subdivision` (`src/config.rs:491-576`), default `enabled: false`.
Implementation: `src/adjacency/subdivide.rs`, wired in `src/adjacency/cache.rs`.
Plan of record: `docs/superpowers/plans/2026-08-05-reach-subdivision.md`
(Tasks 1-8, commits `741e475` … `6cb66bf`).

---

## Why it fails — the window-width argument

The plan's load-bearing claim: at `Cr = 1` both coefficients reduce to
`(1−2X)/(1+2(1−X)) ≥ 0` for **any** `X ≤ 0.5`, so `Cr ≈ 1` makes non-negativity
automatic.

That is true only at `Cr` **exactly** 1. In general
`c1 ≥ 0 ⟺ Cr ≥ 2X` and `c3 ≥ 0 ⟺ Cr ≤ 2(1−X)`, so both hold only inside the
window `[2X, 2(1−X)]`, whose **width is `2(1−2X)`** and collapses as `X → 0.5`:

| X | window `[2X, 2(1−X)]` | width |
|---|---|---|
| 0.30 (the `ddr_match: true` constant) | [0.600, 1.400] | 80 % |
| 0.45 | [0.900, 1.100] | 20 % |
| **0.4923** (measured, `max_pieces: 8`) | [0.9846, 1.0154] | **3.1 %** |
| **0.4966** (measured, subdivision off) | [0.9932, 1.0068] | **1.4 %** |

A **static** piece count fixes Δx from a *reference* flow at build time, while
`Cr` tracks the **routed** celerity, which varies severalfold within a single
storm. Holding `Cr` inside a 1.4 % window with a fixed graph is structurally
impossible — not a tuning problem, and no choice of `max_pieces`,
`reference_n` or `min_length_fraction` changes it.

Gate that records this: `tests/subdivision_integration.rs::
both_coefficients_are_non_negative_only_inside_a_window_that_collapses_at_x_half`.
The plan's sketched `subdivision_makes_coefficients_non_negative_without_the_clamp`
(asserting `frac c1 < 0` under 1 %) was deliberately **not** added — the
measurement refutes it.

### Root cause: `X → 0.5` because MERIT is advection-dominated

The Cunge `X = ½(1 − q/(So·c·Δx))` saturates at 0.5 because the cell Reynolds
number `D = q/(So·c·Δx) ≈ 0.012` — physical diffusion is ~1 % of advective
transport, so Cunge correctly returns near-pure translation. Ponce & Theurer's
`C·D ≥ ξ` accuracy criterion is likewise unsatisfiable on this network at any
Δx that also keeps coefficients non-negative (see the plan, §"Why the target is
`Δx = c·Δt`").

**This retroactively vindicates DDR's constant `X = 0.3`.** It is not an
oversight — it is a deliberate stability trade buying an 80 %-wide window at
the cost of hydraulically-wrong numerical diffusion. Any future "correct the X"
work must state what it does about the window it destroys.

---

## Measured on the real network

`src/bin/probe_courant.rs`, 2026-08-05, trained head, 1,841 CONUS gauges,
2,135 hourly steps, `enforce_positivity` **OFF** (the whole point is whether
subdivision removes the need for it):

| arm | rows | Cr p50 | Cr > 2 | **c1 < 0** | **c3 < 0** | both ≥ 0 | neg solves | ms/step |
|---|---|---|---|---|---|---|---|---|
| off | 92,488 | 0.096 | 2.10 % | **93.0 %** | 3.93 % | 3.1 % | 0.1356 % | 1.90 |
| cap 4 | 171,381 | 0.115 | 0.18 % | **98.75 %** | 0.33 % | 0.9 % | 0.0945 % | 2.73 |
| cap 8 | 184,676 | 0.123 | 0.16 % | **98.79 %** | 0.31 % | 0.9 % | 0.0876 % | 2.87 |
| cap 8, `reference_n 0.13` | 344,262 | 0.254 | 0.14 % | 90.90 % | 0.45 % | 8.7 % | 0.0401 % | 4.75 |

Negative *solves* fall only 35 % (0.1356 % → 0.0876 %) for a 2.05× network and
1.5× step time. `both ≥ 0` — the quantity the design targeted — goes
**3.1 % → 0.9 %**, i.e. the wrong direction.

Cap 16 at `reference_n 0.13` **OOMs a 16 GB RTX 4080**.

### What subdivision DOES fix

`frac Cr > 2` falls **2.10 % → 0.16 %**, which essentially eliminates `c3 < 0`
(**3.93 % → 0.31 %**). A `min_length_fraction: 0` control arm shows this is the
**short-reach length clamp**, not the splitting: with the clamp off, `Cr > 2`
returns to 1.08 % and `c3 < 0` to 2.00 %.

But `c3 < 0` is the far smaller population, and the clamp costs **+23.9 % total
channel length** — a reach modelled longer than reality has a proportionally
longer travel time. That is a real physical distortion bought for a numerical
gain on 3.6 % of cells.

### Graph cost (cap 8, shipped defaults)

| quantity | measured | plan predicted |
|---|---|---|
| Σm (sub-reaches) | 709,974 (**2.05×**) | ~1.65 M (4.77×) |
| reaches clamped | **51.0 %** | 34.4 % |
| pinned at `max_clamp_factor = 4.0` | **14.1 %** | — |
| total channel length | **+23.9 %** | +17.1 % |

**The plan's cap sweep is materially wrong for the shipped config** and should
not be quoted. Cause: `reference_n: 0.05` makes the reference celerity ~5× the
routed celerity — the trained CONUS median `n` is **0.130**, and `ddr_match:
false` uses `c = v·β` with β ≈ 1.33, not the wide-rectangular 5/3. A too-fast
`c_ref` gives a too-long `dx_target`, so fewer reaches split and far more get
length-clamped. Reproduce the cost table without routing:

```bash
cargo run --release --bin probe_courant -- \
  --config ddrs.yaml --fabric <merit fabric> --max-pieces 8 --clamp-report
```

### The X dynamic range does NOT recover

Raw `X_cunge` median moves only **0.4973 → 0.4815** even uncapped. Subdivision
cannot un-saturate X, because `D = q/(So·c·Δx)` is small primarily from the
`attribute_minimums.slope = 1e-3` floor and large top width `B`, not from Δx.
Do not justify this feature on X's dynamic range (see the erratum in
`.claude/PHYSICS-CORRECTIONS.md`).

---

## Hot start — a real bug the campaign found and fixed

`setup_inputs` cold-starts with `(I − N)·Q₀ = q'₀`. The `q'/m` split lives in
`forward`, so an undivided cold start feeds the full `q'₀` into a chain of `m`
pieces and every parent outlet begins at `m ×` its true steady state.

Measured (cap 8, 184,676 rows): **2.94× the correct total network discharge**,
still **41.7 % off at t = 120** — the *configured* `warmup` (5 days) — reaching
<10 % only at t = 221 and <5 % at t = 282.

So **`MuskingumCunge::divide_hotstart_by_pieces` defaults `true`**
(`src/routing/mmc.rs:144,214`). It is an exact no-op without subdivision (no
divisor exists), so `compare_ddr_sandbox` stays an ABSOLUTE MATCH (1.53e-5 m³/s)
and `adjacency_parity` still matches element-for-element.
`probe_courant --divide-hotstart` A/B's it.

---

## What the code does (all of this is correct and stays)

### The two-sided rule

```
Δx_target = c_ref · Δt          c_ref from reference_n + Q_ref = coeff·uparea^exp
  L > Δx_target  →  split into m = min(ceil(L/Δx_target), max_pieces) pieces
                    of length L/m; q' → q'/m
  L < Δx_target  →  clamp the length UP to min_length_fraction·Δx_target,
                    bounded by original_length · max_clamp_factor  (do NOT merge)
```

Merging short reaches was rejected: a short reach may carry two upstream
tributaries or a junction below it, so collapsing it destroys topology.
Clamping its length achieves the same `Cr` with no topology change.
`max_clamp_factor` (default 4.0, added in `2a06c0f`) bounds the distortion —
unbounded, measured clamp factors ran to p99 = 36× and **max 48,597×**, because
`reference_celerity` uses a depth relation with no slope dependence while `v`
scales as `√S`, so steep small catchments get big-river depth *and*
steep-slope velocity (~8.9 m/s → a 32 km `dx_target`).

### Topology

```
   BEFORE                       AFTER (m=3)
   U ──> P ──> D                U₂ ──> P₀ ──> P₁ ──> P₂ ──> D₀
                                       └─ q'/3   q'/3   q'/3
   len(P) = L                   len(Pᵢ) = L/3,  slope/n/p/q unchanged
   gauge@P → row(P)             gauge@P → P₂  (outlet = last piece)
```

Sub-reaches are hydraulically identical to their parent (MERIT carries no
within-reach variation), pieces are contiguous and ordered upstream→downstream,
and parents are already topologically ordered — so the expanded graph stays
topologically ordered and **lower-triangular for free** (invariant 3 holds).
The upstream parent's *outlet* piece connects to the downstream parent's
*inlet* piece (`subdivide.rs:327-328`).

### Two index spaces

`order` gains duplicates (m rows share one COMID), which would break
`IdIndex<Comid>`. Resolved by keeping both:

```
parent space (N = 346,321)          sub-reach space (N' = Σ min(m, M))
  parent_order[p] = COMID             order[i] = COMID of i's parent
  IdIndex<Comid> built HERE           parent_offset[p]..parent_offset[p+1]
  ← attributes, q', KAN outputs         = contiguous rows owned by parent p
                                      m_p = parent_offset[p+1] − parent_offset[p]
                                      outlet(p) = parent_offset[p+1] − 1
```

`IdIndex` is built from **`parent_order`**, never from `order`
(`src/data/store/zarr.rs:121-122`). A store without the map synthesizes the
identity (`parent_order == order`, `parent_offset == 0..=n`), so every un-split
store keeps working unchanged.

### Where each piece of plumbing lives

| Concern | Site |
|---|---|
| Config + validation | `src/config.rs:491-576`, `validate_subdivision` `:1034` |
| Reference celerity, reach plan, expansion, `plan_stats` | `src/adjacency/subdivide.rs` |
| Upstream-area accumulation, sequencing, cache key | `src/adjacency/cache.rs` (`upstream_area_km2`, `reach_plan`, `resolve_or_build`) |
| Zarr persist/load of `parent_order` + `parent_offset` | `src/data/store/zarr.rs` |
| `q'/m` after the clamp; hot-start divisor | `src/routing/mmc.rs:267-273,340,539` |
| KAN parent→sub-reach gather | `src/training/forward.rs::gather_params_to_subreaches` |
| Gauge read at the parent's outlet piece; compressed-space `parent_offset` | `src/data/collate.rs` |
| Tests | `tests/subdivide.rs`, `tests/subdivision_integration.rs` (12) |

### Gotchas worth keeping

- **`catchsize` is NOT drainage area.** It is the *local* divide area (median
  36.7 km², max ~612 km² even for continental rivers). `reference_celerity`
  needs accumulated upstream area, so `cache.rs::upstream_area_km2` accumulates
  it downstream over the topological order (validated against the fabric's own
  `10^log10_uparea`: ratio p5/p50/p95 = 1.000/1.000/1.000 over all 346,321
  reaches). `log10_uparea` cannot be read directly — it is NaN on 88 % of
  `merit_global_attributes_v2.nc`.
- **All seven `Subdivision` fields are hashed into the adjacency content key**
  (`cache.rs::content_key`), because every one of them moves `dx_target` and
  hence the built graph. Hash the whole struct, not just `enabled` +
  `max_pieces`.
- **`enabled: true` + explicit `conus_adjacency`/`gages_adjacency` is a config
  error**, not a warning. Subdivision runs *inside* the managed builder, which
  those keys bypass, so the flag would be silently inert and the manifest would
  lie. `adjacency::validate::store_is_subdivided` reads only zarr metadata
  (`n_parent < n`) and allows the one legitimate case: an explicit path to a
  store that was already built subdivided.
- **`enabled: true` requires `use_cuda_graphs: false`** — a captured graph is
  sized to a fixed reach count.
- **Retraining is mandatory.** Every learned parameter was fit against the
  un-split network's effective diffusion; checkpoints do not transfer.
- **The KAN sees no new information.** Sub-reaches inherit the parent's
  attributes, so subdivision can only change numerics, never identifiability.

---

## Reproducing the measurement

```bash
# un-split control
cargo run --release --bin probe_courant -- --config ddrs.yaml \
  --checkpoint <run>/checkpoints/<epoch> --backend cuda \
  --gauges 1841 --rho 90 --steps 2136

# subdivided arm (--fabric switches to the managed build; caps cache separately)
cargo run --release --bin probe_courant -- --config ddrs.yaml \
  --checkpoint <run>/checkpoints/<epoch> --backend cuda \
  --gauges 1841 --rho 90 --steps 2136 \
  --fabric <merit fabric> --max-pieces 8 --reference-n 0.13 --divide-hotstart
```

Gate set after any change under `src/adjacency/subdivide.rs`, `src/routing/mmc.rs`
piece handling, or the parent map:

```bash
cargo test --test subdivide --test subdivision_integration \
           --test adjacency_parity --test gauge_mass_conservation
cargo test --lib
cargo run --release --example compare_ddr_sandbox   # must stay ABSOLUTE MATCH
```

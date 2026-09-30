# ddrs model equations: complete reference as implemented

**Date:** 2026-09-17
**Worktree:** `.claude/worktrees/agent-a10b11fb99c152d22`, branch `leakance-gate`,
HEAD `289163b` ("feat(leakance): temperature-annealed gate replaces the continuous factor").
**Audience:** an analyst judging whether the learned parameters are physically
meaningful and what this work contributes to distributed hydrological modelling.

## 0. How this document was produced, and how to read it

Every equation below was transcribed from source read in this worktree, not from
any prior document. The files read in full or in the relevant part were:
`src/nn/kan_head.rs`, `src/routing/utils.rs`, `src/training/gate.rs`,
`src/geometry.rs`, `src/routing/mmc.rs`, `src/routing/mmc_op.rs`,
`src/routing/leakance.rs`, `src/sparse/mod.rs`, `src/training/loss.rs`,
`src/training/forward.rs`, `src/training/metrics.rs`, `src/training/eval.rs`,
`src/training/driver.rs`, `src/dump_parameters.rs`, `src/config.rs`,
`src/data/dataset.rs`, `tests/stage_roughness.rs`, and the pinned
`rskan` v0.1.3 `KanLayer::forward`. Where a docstring and the code disagree the
code is authoritative here and the disagreement is listed in §12.

**Symbol-status column.** Every symbol table carries a `status` column with one
of five values:

| status | meaning |
|---|---|
| **learned** | a per-reach KAN output, trained by gradient descent |
| **config** | prescribed by YAML (`params.*`, `experiment.*`) |
| **data** | read from a store (attributes, adjacency, forcing, observations) |
| **derived** | computed from the above inside the forward pass |
| **literal** | a numeric constant hard-coded in Rust, not configurable |

**Clamps.** Every clamp is named and numbered, because a saturated clamp
sets the incoming gradient to exactly zero for that reach-timestep (the
backward implements each clamp as a boolean mask fill, see §9). A clamp is
therefore both a physical floor and a gradient sink, and a parameter whose
clamp binds on most reach-timesteps is not being trained by that path.

**Backend precision.** The entire routing core is f32
(`CLAUDE.md` invariant 2). `Δt = 3600 s` is hard-coded
(`src/routing/mmc.rs::DT_SECONDS`); there is no sub-hourly routing step.

---

## 1. Pipeline overview

```
 attributes (N x F, z-scored)
        |
        v  §2  KAN head, ends in sigmoid
   u in (0,1)^P   normalized parameters
        |
        +---> §3  leakance_factor only: temperature-annealed gate  g(u; tau)
        |
        v  §2.3  denormalize (linear or log-space) per parameter box
   physical parameters:  n_0, q_spatial, p_spatial, [gamma], [K_D, d_gw, f]
        |
        v  §4/§5 per reach, per timestep, from Q_t
   depth -> top width -> side slope -> bottom width -> A, P_w, R -> v -> celerity c
        |
        v  §6
   K = L/c ;  X (constant, or Cunge-derived) ;  c1..c4
        |
        +---> §8  zeta = f * area_z * K_D * (d - d_gw)   subtracted from the RHS
        |
        v  §7  (I - c1 .* N) Q_{t+1} = c2 (N Q_t) + c3 Q_t + c4 q'_t - zeta
   sparse lower-triangular forward substitution + hand-written adjoint
        |
        v  Q_{t+1} = max(x_sol, 1e-4)
   gauge scatter-add -> tau shift + daily area pooling -> §10 loss
```

---

## 2. The parameter head

### 2.1 Input

`src/data/dataset.rs::MeritGagesDataset::finalize_attrs` builds the head input.
Missing COMIDs are filled with the attribute's row mean, then each attribute
column is z-scored against precomputed statistics:

$$
X_{i,\,\text{fi}} \;=\; \frac{a_{\text{fi},i} - \mu_{\text{fi}}}{\sigma_{\text{fi}}}
\tag{1}
$$

and the array is transposed to `(N, F)`.

| symbol | meaning | units | status |
|---|---|---|---|
| $a_{\text{fi},i}$ | raw attribute `fi` at reach `i` | attribute-specific | data |
| $\mu_{\text{fi}}, \sigma_{\text{fi}}$ | per-attribute mean / std | same | data (statistics JSON) |
| $N$ | reaches in the batch's network (parent resolution) | - | data |
| $F$ | number of attributes, `kan_head.input_var_names` | - | config |

The shipped $F = 10$ attribute set is `SoilGrids1km_clay, aridity,
meanelevation, meanP, NDVI, meanslope, log10_uparea, SoilGrids1km_sand,
ETPOT_Hargr, Porosity`. **The head sees no discharge, no state, and no time.**
It is a static map from catchment attributes to reach parameters; all
time-dependence in the model enters through the geometry's dependence on $Q_t$.

### 2.2 Layer stack

`src/nn/kan_head.rs::KanHead::forward` composes, per parameter group (default:
one group over all outputs):

$$
h^{(0)} = \mathrm{Linear}_{F \to H}(X), \qquad
h^{(\ell)} = \mathrm{KanLayer}_{H \to H}\!\left(h^{(\ell-1)}\right), \;\; \ell = 1 \ldots L,
\tag{2}
$$

$$
z = \mathrm{Linear}_{H \to P}\!\left(h^{(L)}\right), \qquad u = \sigma(z) = \frac{1}{1 + e^{-z}} \in (0,1)^{N \times P}
\tag{3}
$$

There is **no activation between KAN blocks** (DDR `kan.py` parity, enforced by
`CLAUDE.md` invariant 5). `u` is transposed to `[P, N]` and split row-by-row
into a `HashMap` keyed by `kan_head.learnable_parameters` in order.

One `rskan::KanLayer` (pinned tag `v0.1.3`, `rskan/src/layer.rs::KanLayer::forward`)
computes, for input $x \in \mathbb{R}^{B \times I}$ and output index $o$:

$$
y_{b,o} \;=\; \sum_{i=1}^{I} m_{i,o}\left[\, s^{\mathrm{b}}_{i,o}\,\mathrm{SiLU}(x_{b,i})
\;+\; s^{\mathrm{sp}}_{i,o} \sum_{j} c_{i,o,j}\, B_{j,k}\!\left(x_{b,i}\right) \right]
\tag{4}
$$

$$
\mathrm{SiLU}(x) = x\,\sigma(x)
\tag{5}
$$

with $B_{j,k}$ the order-$k$ B-spline basis on the layer's grid. So every edge
$(i,o)$ carries its own learned univariate function: a fixed SiLU shape scaled
by $s^{\mathrm{b}}$ plus a free spline scaled by $s^{\mathrm{sp}}$.

| symbol | meaning | units | status |
|---|---|---|---|
| $H$ | `kan_head.hidden_size`, default 21 | - | config |
| $L$ | `kan_head.num_hidden_layers`, default 2 | - | config |
| $P$ | number of emitted parameters | - | config |
| $k$ | B-spline order, default 3 (current arm: 2) | - | config |
| grid | B-spline intervals, default 5 (current arm: 50) | - | config |
| $c_{i,o,j}, s^{\mathrm{b}}, s^{\mathrm{sp}}$ | spline coefficients and scales | - | learned (head weights) |
| $m_{i,o}$ | edge mask | - | learned/fixed (init to 1) |
| $u$ | normalized parameter | dimensionless in $(0,1)$ | derived |

Initialization (`src/nn/init.rs`, called from `KanHeadConfig::init_trunk`):
input `Linear` weight is Kaiming-normal, $\mathrm{std} = \sqrt{2}/\sqrt{F}$;
output `Linear` weight is Xavier-normal, $\mathrm{std} = 0.1\sqrt{2/(H+P)}$;
both biases zero; **every** inner `KanLayer` receives the SAME seed (a preserved
DDR quirk, invariant 5).

Three optional architecture switches, all default off and all breaking DDR
parity by construction:
`input_layer_kan` replaces $\mathrm{Linear}_{F\to H}$ with $\mathrm{KanLayer}_{F\to H}$;
`output_layer_kan` replaces $\mathrm{Linear}_{H\to P}$ with $\mathrm{KanLayer}_{H\to P}$;
`parameter_groups` partitions the outputs into independently-weighted trunks,
group 0 seeded with `seed` and group $i>0$ with
`seed + i * 0x9E3779B97F4A7C15` (wrapping).

**Structural note the analyst needs.** Under the default (all-`Linear`
read-out, one shared trunk), every output is an affine functional of the same
latent vector: $z_{\cdot,o} = w_o \cdot h^{(L)} + b_o$. If the informative part
of $h^{(L)}$ is effectively one direction, then any two outputs are *exactly*
affinely related in logit space. The code's own docstring on
`KanHeadConfig::output_layer_kan` records the measurement:
$\mathrm{logit}(q) = 3.979\,\mathrm{logit}(n) + 2.752$ with $R^2 = 0.987$.
Two "different" learned parameters can therefore be one parameter relabelled.
That is a property of the head topology, not of the physics.

### 2.3 Normalized to physical: `denormalize`

`src/routing/utils.rs::denormalize`, with box $[\theta_{\mathrm{lo}}, \theta_{\mathrm{hi}}]$
from `params.parameter_ranges`:

Linear branch:
$$
\theta = u\,(\theta_{\mathrm{hi}} - \theta_{\mathrm{lo}}) + \theta_{\mathrm{lo}}
\tag{6}
$$

Log-space branch (parameter name listed in `params.log_space_parameters`):
$$
\theta = \exp\!\Big[\, u\big(\ln \theta_{\mathrm{hi}} - \Lambda(\theta_{\mathrm{lo}})\big) + \Lambda(\theta_{\mathrm{lo}}) \Big]
\tag{7}
$$

where $\Lambda$ is `src/routing/utils.rs::log_space_lower`:

$$
\Lambda(\theta_{\mathrm{lo}}) \;=\;
\begin{cases}
\ln \theta_{\mathrm{lo}}, & \theta_{\mathrm{lo}} > 0\\[2pt]
\ln\!\left(\theta_{\mathrm{lo}} + 10^{-6}\right), & \theta_{\mathrm{lo}} \le 0
\end{cases}
\tag{8}
$$

**Why $\Lambda$ exists.** DDR's Python applies `bounds[0] + 1e-6`
unconditionally. For a box whose lower bound is small next to $10^{-6}$ that
inverts the map. Worked example from the docstring, for the historical
$K_D \in [10^{-8}, 10^{-6}]$: unconditional nudging gives
$\ln(1.01\times10^{-6}) = -13.8065$ as the lower end and
$\ln(10^{-6}) = -13.8155$ as the upper, so
$\ln\theta_{\mathrm{hi}} - \Lambda < 0$; the map runs backwards and spans about
1% instead of two decades. The consequence is recorded in the current
experiment config: "before it, $K_D$'s declared box $[10^{-8}, 10^{-6}]$
collapsed to a 1% band and $K_D$ was frozen, so every leakance run in the July
campaign learned only (`leakance_factor`, `d_gw`)." $\Lambda$ uses
$\ln\theta_{\mathrm{lo}}$ whenever the bound is already positive, and keeps the
old behaviour only for $\theta_{\mathrm{lo}} \le 0$ so that zero-floored boxes
map exactly as before.

Inverse (`src/training/forward.rs::physical_to_normalized`), sharing the same
$\Lambda$ so the two directions cannot drift:

$$
u = \frac{\theta - \theta_{\mathrm{lo}}}{\theta_{\mathrm{hi}} - \theta_{\mathrm{lo}}}
\quad\text{(linear)}, \qquad
u = \frac{\ln \theta - \Lambda(\theta_{\mathrm{lo}})}{\ln \theta_{\mathrm{hi}} - \Lambda(\theta_{\mathrm{lo}})}
\quad\text{(log)}
\tag{9}
$$

This inverse is what holds a non-emitted parameter at a prescribed constant:
`src/training/forward.rs::fixed_output_normalized` maps `params.defaults[name]`
through (9) so that (6)/(7) maps it back exactly.

### 2.4 The parameter boxes

`src/config.rs::ParameterRanges::default` and
`src/config.rs::Params::default`:

| parameter | box | log-space by default | units | status |
|---|---|---|---|---|
| `n` (Manning $n_0$) | $[0.015,\,0.25]$ | no | $\mathrm{s\,m^{-1/3}}$ | learned |
| `q_spatial` ($q$, width **exponent**) | $[0,\,1]$ | no | dimensionless | learned or fixed |
| `p_spatial` ($p$, width **coefficient**) | $[1,\,200]$ | **yes** | $\mathrm{m}^{1-q}$ | learned or fixed (21.0) |
| `x_storage` ($X$) | $[0,\,0.5]$ | no | dimensionless | see §12.2 |
| `K_D` | $[10^{-8},\,10^{-6}]$ default; $[10^{-8},\,10^{-4}]$ in the current arm | no by default; **yes** in the current arm | $\mathrm{s^{-1}}$ | learned (leakance only) |
| `d_gw` | $[-2,\,2]$ | no | m | learned (leakance only) |
| `leakance_factor` ($f$) | $[0,\,1]$ | no | dimensionless | learned (leakance only) |
| `gamma` ($\gamma$) | $[0,\,0.5]$ | no | dimensionless | learned, or global config constant |

`params.log_space_parameters` defaults to `["p_spatial"]` only. Because
$\theta_{\mathrm{lo}} = 1$ there, (7) reduces to $p = 200^{u}$, i.e. $u$ is
$\log_{200} p$.

`params.defaults` defaults to `{p_spatial: 21.0}`. The current experiment arm
(`config/experiments/sr_n0_leakance_gated.yaml`) learns
`[n, K_D, d_gw, leakance_factor]` and fixes `p_spatial = 21.0`,
`q_spatial = 0.65`, with `log_space_parameters: [p_spatial, K_D]`.

**What each parameter can and cannot represent.**
- $n_0$ absorbs everything that scales travel time. Since $K = L/c$ and
  $c \propto 1/n$, any smooth systematic bias in the celerity (§5, and the
  2.3% beta approximation) is absorbed almost entirely by $n_0$. A learned
  $n_0$ is therefore an *effective* travel-time parameter, not a
  measurable channel roughness, unless the celerity convention is
  independently validated.
- $(p, q)$ are the at-a-station Leopold and Maddock width relation
  $T = p\,d^{q}$. They are falsifiable against remotely sensed river width,
  which is the stated reason for learning them as a pair.
- $X$ sets the split between translation and attenuation. Under the default
  corrected physics it is **not** learned; it is derived from the flow (§6.3).
- $K_D$, $d_gw$, $f$ are the GW-SW exchange triple (§8), with the identifiability
  caveat in §8.6.

---

## 3. The leakance gate

`src/training/gate.rs::leakance_gate`. Applied to the **normalized** head
output $u$ for `leakance_factor` *before* denormalization, at every reader:
training (`src/training/forward.rs::forward`), eval
(`src/training/forward.rs`, `src/training/eval.rs`), the probe
(`src/training/probe.rs`) and `src/dump_parameters.rs`.

$$
g \;=\; \sigma\!\left(\frac{\mathrm{logit}\big(\mathrm{clamp}(u,\,\varepsilon_g,\,1-\varepsilon_g)\big)}{\tau}\right),
\qquad \mathrm{logit}(x) = \ln x - \ln(1-x)
\tag{10}
$$

with $\varepsilon_g = 10^{-6}$ (`src/training/gate.rs::GATE_EPS`, a Rust
`const`, not configurable).

**$\tau = 1$ is the identity.** $\sigma \circ \mathrm{logit} = \mathrm{id}$, so
$g = u$ analytically. The code special-cases `tau == 1.0` and returns `u`
untouched, so the identity is *bit*-exact rather than round-tripped through
`ln`/`exp`. This is what makes a $\tau$ schedule starting at 1.0 begin from the
historical ungated model exactly.

As $\tau \downarrow 0$, (10) sharpens toward a step at $u = 0.5$. It is
differentiable for every $\tau > 0$ (no straight-through estimator):

$$
\frac{\partial g}{\partial u} \;=\; \frac{g(1-g)}{\tau\,u\,(1-u)}
\quad\text{inside the clamp}, \qquad 0 \ \text{ outside}
\tag{11}
$$

**Epsilon justification, from the `GATE_EPS` docstring.** The head ends in a
sigmoid, so $u$ can be exactly $0.0$ or $1.0$ in f32 once the pre-activation
passes about $\pm 17$ ($\sigma(17) = 1.0$ in f32). An unclamped logit there is
$\pm\infty$ and its backward $1/(u(1-u))$ is $\infty$, giving NaN gradients.
$10^{-6}$ is chosen because $1-10^{-6}$ is about 17 ulps below $1.0$ in f32
(ulp $= 2^{-24} \approx 6\times10^{-8}$), so $1-u$ resolves to about 3%
relative accuracy; $10^{-7}$ would sit 2 ulps away and carry about 20% error.
$\mathrm{logit}(10^{-6}) \approx -13.8$, well inside f32 range, and the clamp
only alters values the head has already saturated past
$|z| > 13.8$. No f64 is introduced (invariant 2).

**Temperature schedule.** `src/config.rs::LeakanceGate` holds a
`BTreeMap<usize, f32>` keyed by 1-indexed epoch.

$$
\tau(\text{epoch}) = \text{value at the largest key} \le \text{epoch};
\quad\text{if no such key, the first value}
\tag{12}
$$

$$
\tau_{\text{final}} = \text{the last scheduled value}
\tag{13}
$$

`LeakanceGate::resolve(epoch)` implements (12) and is what the training driver
passes per epoch (`src/training/driver.rs`, alongside the learning rate).
`LeakanceGate::final_temperature` implements (13) and is what **eval, the
probe, and `dump_parameters` use**, so the scored and reported model is the one
training ended on. Config load (`src/config.rs::validate_leakance_gate`)
requires `use_leakance: true`, a non-empty schedule, and every temperature
positive and finite.

The current arm anneals `{1: 1.0, 11: 0.5, 21: 0.25, 31: 0.1, 41: 0.05}`.

**Why the gate exists** (module docstring of `src/training/gate.rs`):
$\zeta$ (equation 45) multiplies $f$ and $K_D$, so only their product reaches
the physics and the two are exactly degenerate; the measured correlation over
CONUS on a trained arm was $\rho = +0.9986$. Physically a reach either sits
above a losing aquifer or it does not, so $f$ should be a $0/1$ **selector** of
*where* leakance acts rather than a continuous multiplier of *how much*. A gate
and a conductance are not degenerate with each other.

| symbol | meaning | units | status |
|---|---|---|---|
| $u$ | normalized `leakance_factor` head output | dimensionless | learned |
| $g$ | gated factor, still normalized, fed to (6) | dimensionless | derived |
| $\tau$ | gate temperature for this epoch | dimensionless | config schedule |
| $\varepsilon_g$ | clamp margin, $10^{-6}$ | dimensionless | literal |

---

## 4. Channel geometry

Implemented twice, identically: `src/geometry.rs::compute_trapezoidal_geometry_gamma`
(the reference, autograd-by-BURN) and steps S1-S16 of
`src/routing/mmc_op.rs::forward_chain_inner` (the production path, which runs
at backend-primitive level with the hand-written backward of §9). The equations
below are read from `forward_chain_inner`; `src/geometry.rs` agrees term for
term.

### 4.1 The width relation, stated unambiguously

$$
T \;=\; p \cdot d^{\,q_\varepsilon}
\tag{14}
$$

**$p$ is the coefficient. $q$ is the exponent.** $p$ is `p_spatial`, box
$[1, 200]$, log-space; $q$ is `q_spatial`, box $[0, 1]$, linear. Dimensionally
$[p] = \mathrm{m}^{1-q}$, which is why $p$ is not directly comparable across
reaches with different $q$.

### 4.2 The chain, step by step

$$
q_\varepsilon = q + 10^{-6}
\tag{S1, 15}
$$

$$
\mathcal{N} = Q_t \cdot n \cdot (q_\varepsilon + 1) \cdot d_{\mathrm{ref}}^{\,\gamma}
\tag{S2, 16}
$$

$$
\mathcal{D} = p\sqrt{s} + 10^{-8}
\tag{S3, 17}
$$

$$
\mathcal{R} = \mathcal{N} / \mathcal{D}
\tag{S4, 18}
$$

$$
e \;=\; \frac{3}{\,5 + 3 q_\varepsilon + 3\gamma\,}
\tag{S5, 19}
$$

$$
d \;=\; \max\!\left(\mathcal{R}^{\,e},\; d_{\mathrm{lb}}\right)
\tag{S6, 20}
$$

$$
T = p\,d^{\,q_\varepsilon}
\tag{S7, 21}
$$

$$
z_{\mathrm{raw}} = \frac{T\,q_\varepsilon}{2d},
\qquad
z = \mathrm{clamp}(z_{\mathrm{raw}},\, 0.5,\, 50)
\tag{S8-S9, 22}
$$

$$
b_{\mathrm{raw}} = T - 2 z d,
\qquad
b = \max(b_{\mathrm{raw}},\, b_{\mathrm{lb}})
\tag{S10-S11, 23}
$$

$$
A = \frac{(T + b)\,d}{2}
\tag{S12, 24}
$$

$$
P_w = b + 2 d\sqrt{1 + z^2}
\tag{S13, 25}
$$

$$
R = A / P_w
\tag{S14, 26}
$$

$$
v_{\mathrm{un}} = \frac{1}{n}\left(\frac{d}{d_{\mathrm{ref}}}\right)^{\!\gamma} R^{2/3}\sqrt{s}
\tag{S15, 27}
$$

$$
v = \mathrm{clamp}(v_{\mathrm{un}},\, v_{\mathrm{lb}},\, 15)
\tag{S16, 28}
$$

At $\gamma = 0$, equations 16, 19 and 27 reduce to
$\mathcal{N} = Q_t n (q_\varepsilon+1)$, $e = 3/(5+3q_\varepsilon)$ and
$v_{\mathrm{un}} = n^{-1} R^{2/3}\sqrt{s}$ exactly; the code branches on
`gamma != 0.0` rather than relying on `powf(0.0) == 1.0`, so the historical
path is bit-identical.

| symbol | meaning | units | status |
|---|---|---|---|
| $Q_t$ | discharge at the start of the step | m³/s | derived (state) |
| $n$ | Manning $n_0$ | s m$^{-1/3}$ | learned |
| $p$ | L&M width coefficient | m$^{1-q}$ | learned or config default |
| $q$ | L&M width exponent | - | learned or config default |
| $s$ | reach slope, floored at `attribute_minimums.slope` | m/m | data (adjacency zarr) |
| $L$ | reach length | m | data (adjacency zarr) |
| $d$ | flow depth | m | derived |
| $T$ | top width | m | derived |
| $z$ | side slope, horizontal per vertical | - | derived |
| $b$ | bottom width | m | derived |
| $A$ | cross-sectional area | m² | derived |
| $P_w$ | wetted perimeter | m | derived |
| $R$ | hydraulic radius | m | derived |
| $v$ | Manning velocity | m/s | derived |
| $\gamma, d_{\mathrm{ref}}$ | stage-roughness pair (§5) | -, m | learned / config |

### 4.3 Where the depth closed form comes from, and what it assumes

Equation 20 is *not* an inversion of the trapezoid that equations 24-26 then
build. Substituting $R \approx d$ (wide-channel limit) and the power-law area
$A_{\mathrm{pl}} = \int_0^d T\,\mathrm{d}d' = p\,d^{q+1}/(q+1) = T d/(q+1)$ into
$Q = vA$ with $v = n^{-1}R^{2/3}\sqrt{s}$ gives

$$
Q = \frac{p\sqrt{s}}{n\,(q+1)}\, d^{\,q + 5/3}
\;\Longrightarrow\;
d = \left[\frac{Q\,n\,(q+1)}{p\sqrt{s}}\right]^{3/(5+3q)}
\tag{29}
$$

which is exactly equations 16-20 at $\gamma = 0$. So the exponent $3/(5+3q)$
encodes the **power-law section with $R = d$**, while $A$, $P_w$, $R$, $v$ and
the celerity are then evaluated on the **trapezoid**. The two sections do not
agree. Substituting $z = Tq/(2d)$ into equation 23 gives $b = T(1-q)$, hence

$$
A_{\mathrm{tz}} = \frac{T(2-q)\,d}{2}, \qquad
\frac{A_{\mathrm{tz}}}{A_{\mathrm{pl}}} = \frac{(2-q)(q+1)}{2}
\tag{30}
$$

which equals 1 only at $q = 0$ and $q = 1$, and is 1.114 at $q = 0.65$ (the
value the current arm prescribes). This is the same $(2-q)(q+1)/2$ factor
recorded in `tests/stage_roughness.rs` as the "up to 11%" gap between the true
$\mathrm{d}A/\mathrm{d}d$ and the $T$ the celerity convention uses. It is
inherited from DDR and out of scope of any fix in this tree, but it means the
depth reported by the model is the depth of a power-law channel, not of the
trapezoid whose conveyance is then computed.

### 4.4 Every clamp, and what it does to the gradient

Each row names the clamp, its bound, the config key, and the backward label
(§9) that masks it. A saturated clamp zeroes the gradient for that
reach-timestep through that path.

| # | clamp | bound | source | backward | gradient when saturated |
|---|---|---|---|---|---|
| C1 | slope floor | $s \ge 10^{-3}$ | `attribute_minimums.slope`, applied once in `mmc.rs::setup_inputs` | n/a | $s$ is not a parameter; no gradient path |
| C2 | lateral inflow floor | $q' \ge 10^{-4}$ | `attribute_minimums.discharge`, once per window in `mmc.rs::forward` | n/a | $q'$ is data (a parent only in the adjoint study) |
| C3 | depth floor | $d \ge 10^{-2}$ m | `attribute_minimums.depth` | B6 | zero into $\mathcal{R}$, $e$, hence into $n$, $q$, $p$, $Q_t$ through the depth chain |
| C4 | side-slope band | $z \in [0.5, 50]$ | **literal**, not configurable | B9 | zero into $T$, $q_\varepsilon$, $d$ through $z$ |
| C5 | bottom-width floor | $b \ge 10^{-2}$ m | `attribute_minimums.bottom_width` | B11 | zero into $T$, $z$, $d$ through $b$ |
| C6 | velocity band | $v \in [0.01, 15]$ m/s | lower from `attribute_minimums.velocity`, **upper literal 15** | B16 | zero into all of $R$, $n$, $d$ through the velocity |
| C7 | Muskingum $K$ floor | $K \ge \Delta t (1+\delta)/2$ | `enforce_positivity` only, $\delta = 10^{-2}$ literal | B18' | zero into celerity, hence into all geometry |
| C8 | Cunge $X$ band | $X \in [0, 0.5]$ | literal | B19 | zero into $Q_t$, $T$, $c$ through $X$ |
| C9 | $X$ stability cap | $X \le \min(\tfrac{Cr}{2}, 1-\tfrac{Cr}{2})(1-\delta)$ | `enforce_positivity` only | B19' | routes to one branch only; the Cunge terms are masked off where a cap wins |
| C10 | solve output floor | $Q_{t+1} \ge 10^{-4}$ | `attribute_minimums.discharge` | B28 | zero into everything for that reach-timestep |
| C11 | gate clamp | $u \in [10^{-6}, 1-10^{-6}]$ | literal `GATE_EPS` | autograd | zero into the head's `leakance_factor` logit |
| C12 | losing-only clamp | $\max(0, d - d_{gw})$ | `leakance_losing_only`, default **true** | `zeta_backward` gate | zero into *all six* leakance partials on gaining reaches |
| C13 | KGE per-gauge cap | weighted component sum $\le 10$ | `loss.kge_clamp` | autograd | zero for that gauge's KGE term |

C10 is the consequential one for mass: see §6.5.

---

## 5. Stage-dependent roughness

`params.stage_roughness` (a global constant $\gamma$) or `gamma` in
`kan_head.learnable_parameters` (per-reach learned). Config load
(`src/config.rs::validate_stage_roughness`, `::validate_learned_gamma`) rejects
setting both, requires $\gamma \in [0,1]$, $d_{\mathrm{ref}} > 0$,
`ddr_match: false`, `use_cuda_graphs: false`, and, for the learned form,
`use_leakance: false` (the eight-parent leakance op has no $\gamma$ parent, so
$\gamma$ would silently receive no gradient). With a *global* $\gamma$,
$d_{\mathrm{ref}}$ must be exactly 1.0, because $d_{\mathrm{ref}}^{\gamma}$ is
then a constant on the Manning numerator and is fully absorbed by the learned
$n$ field.

### 5.1 The roughness law

$$
n(d) \;=\; n_0 \left(\frac{d}{d_{\mathrm{ref}}}\right)^{\!-\gamma}
\tag{31}
$$

$\gamma \ge 0$ means channels get **smoother** as they fill. $n_0$ is the
roughness **at** $d = d_{\mathrm{ref}}$, so changing $d_{\mathrm{ref}}$
rescales $n_0$ and makes published $n$ values incomparable
(`src/config.rs::StageRoughnessSection` docstring).

$d_{\mathrm{ref}}$ comes from `params.stage_roughness.d_ref`, default
`default_d_ref()`; it is **required to be 1.0 m** whenever $\gamma$ is global,
and only free when $\gamma$ is a per-reach KAN output. So in every run to date,
$d_{\mathrm{ref}} = 1\ \mathrm{m}$ and $d_{\mathrm{ref}}^{\gamma} = 1$ exactly.

### 5.2 The modified depth exponent

Substituting (31) into the Manning inversion of §4.3 moves one power of depth
across, shifting the exponent denominator and putting a constant
$d_{\mathrm{ref}}^{\gamma}$ on the numerator:

$$
e \;=\; \frac{3}{\,5 + 3q_\varepsilon + 3\gamma\,},
\qquad
d = \left[\frac{Q_t\,n_0\,(q_\varepsilon+1)\,d_{\mathrm{ref}}^{\gamma}}{p\sqrt{s} + 10^{-8}}\right]^{e}
\tag{32}
$$

The velocity must carry the same factor, because
$1/n(d) = (1/n_0)(d/d_{\mathrm{ref}})^{\gamma}$:

$$
v_{\mathrm{un}} = \frac{1}{n_0}\left(\frac{d}{d_{\mathrm{ref}}}\right)^{\!\gamma} R^{2/3}\sqrt{s}
\tag{33}
$$

The `src/geometry.rs` comment records why this is called out explicitly:
applying the factor to only the depth inversion and not the velocity "leaves
the velocity exponent at Manning's 2/3 while the depth exponent moves, which is
silently wrong rather than loudly wrong", and `tests/stage_roughness.rs` caught
exactly that on the first implementation.

### 5.3 The celerity and its correction term

`ddr_match: true` (deprecated):

$$
c = \tfrac{5}{3}\,v
\tag{S17a, 34}
$$

This is the wide-rectangular Kleitz-Seddon limit. The in-code comment states it
is WRONG for the trapezoid S7-S13 builds: $\kappa = b/y \approx 0.7$-$1.8$
there, so the true ratio is $\approx 1.30$-$1.36$, making 5/3 some 22-27% high.

`ddr_match: false` (the default since 2026-08-19), from
$c = \mathrm{d}Q/\mathrm{d}A$ with $\mathrm{d}A/\mathrm{d}d = T$:

$$
c \;=\; v \cdot \beta,
\qquad
\beta \;=\; \underbrace{\frac{5}{3} - \frac{4}{3}\,\frac{A\sqrt{1+z^2}}{T\,P_w}}_{\beta_{\text{trapezoid}}}
\;+\; \underbrace{\gamma\,\frac{A}{T\,d}}_{\text{stage-roughness term}}
\tag{S17b, 35}
$$

$\beta_{\text{trapezoid}} \to 5/3$ as $b/y \to \infty$ and $\to 4/3$ as
$b \to 0$. The **trapezoid factor** is $\sqrt{1+z^2}$, which is
$\mathrm{d}P_w/\mathrm{d}d$ divided by 2; it is what makes $\beta$ depend on
the side slope at all. Sanity check recorded in the code: for a pure power-law
section $A/(Td) = 1/(q+1)$, so

$$
\beta \;=\; \frac{5 + 3q + 3\gamma}{3(q+1)}
\tag{36}
$$

which is $5/3$ at $q = \gamma = 0$, and is the reciprocal of $3e/(q+1)$ with
$e$ from (19).

### 5.4 The documented approximation in the celerity convention, and its size

Two approximations, both pinned by tests.

**(a) The stage-roughness increment is exact.**
`tests/stage_roughness.rs::stage_roughness_celerity_increment_is_exact`
verifies, against central differences at $q \in \{0.084, 0.65\}$,
$d \in \{0.25, 1, 3\}$ m, $\gamma \in \{0.1, 0.183, 0.4\}$:

$$
c(d,\gamma) \;=\; \left(\frac{d}{d_{\mathrm{ref}}}\right)^{\!\gamma} c(d, 0)
\;+\; \gamma\, v(d,\gamma)\,\frac{A}{T\,d}
\tag{37}
$$

to relative tolerance $10^{-4}$, *under the convention the solver uses*.

**(b) The convention itself is approximate, by up to 2.3%.** The solver
computes $c = (\mathrm{d}Q/\mathrm{d}d)/T$, i.e. it assumes
$\mathrm{d}A/\mathrm{d}d = T$ and $\mathrm{d}P_w/\mathrm{d}d = 2\sqrt{1+z^2}$,
i.e. a trapezoid of **fixed shape** being filled. This geometry reshapes as it
fills: $b = T(1-q)$ and $z = (pq/2)\,d^{\,q-1}$ both move with depth. The exact
derivatives, from
`tests/stage_roughness.rs::documents_the_preexisting_beta_approximation`:

$$
T' = \frac{qT}{d}, \qquad
z' = \frac{z(q-1)}{d}, \qquad
b' = T' - 2\left(z + d z'\right)
\tag{38}
$$

$$
A' = \frac{(T' + b')\,d + (T + b)}{2},
\qquad
P_w' = b' + 2\sqrt{1+z^2} + 2d\,\frac{z}{\sqrt{1+z^2}}\,z'
\tag{39}
$$

$$
\beta_{\mathrm{exact}} \;=\; \frac{5}{3} - \frac{2}{3}\,\frac{A}{P_w}\,\frac{P_w'}{A'}
\tag{40}
$$

(The $z'$ and $b'$ terms are set to zero where the C4 / C5 clamps bind, so the
measurement respects the same clamp masks the code applies.)

**Measured size, from the test's own assertions and docstring:** the worst
relative gap over the grid
$q \in \{0.05, 0.084, 0.2, 0.35, 0.5, 0.65, 0.85, 1.0\}$,
$d \in \{0.1, 0.25, 1, 3\}$ m was **about 2.3%**, and about **1.2% at the
trained $q \approx 0.084$**. The test asserts `worst < 0.05` (a 5% ceiling, so
a regression is caught) and `worst > 1e-4` (so the test cannot pass vacuously
if $\beta$ is ever made exact). The true $\mathrm{d}A/\mathrm{d}d$ is
$T(2-q)(q+1)/2$, up to **11%** away from the $T$ the convention uses.

**Why it was documented rather than fixed**, verbatim from the test: the gap
changes sign with depth so it partly averages out, and since $K = L/c$ with
$c \propto 1/n$, a smooth systematic celerity bias is absorbed almost entirely
by the learned roughness. **Only the $q$- and depth-dependent structure of the
error survives training.** This is the single most important caveat for
interpreting a learned $n_0$ as a physical roughness.

---

## 6. Muskingum-Cunge routing

`src/routing/mmc_op.rs::forward_chain_inner` steps S18-S28. A second,
API-level copy of the coefficients lives at
`src/routing/mmc.rs::MuskingumCunge::calculate_muskingum_coefficients`; it is
term-for-term identical but is **not** on the production per-timestep path.

### 6.1 Storage time, Courant number, cell Reynolds number

$$
K \;=\; \frac{L}{c}
\tag{S18, 41}
$$

$$
Cr \;=\; \frac{\Delta t}{K} \;=\; \frac{c\,\Delta t}{L}
\tag{42}
$$

$$
D_{\mathrm{cell}} \;=\; \frac{Q_t}{T\,s\,c\,L}
\;=\; \frac{Q_t/T}{s\,c\,L}
\tag{43}
$$

Equation 42 is the Courant number: cells are reaches, $\Delta x = L$.
Equation 43 is the cell (grid) Reynolds number of the Muskingum-Cunge
literature, $D = q_{\text{unit}}/(s_0 c \Delta x)$ with $q_{\text{unit}} = Q/T$
the unit-width discharge. In code it appears as the local `w` inside S19, with
a $+10^{-12}$ guard on the denominator:
`w = q_t / (top_width * slope * celerity * length + 1e-12)`.

| symbol | meaning | units | status |
|---|---|---|---|
| $K$ | Muskingum storage time | s | derived |
| $\Delta t$ | routing step, $3600$ | s | literal (`DT_SECONDS`) |
| $Cr$ | Courant number | - | derived |
| $D_{\mathrm{cell}}$ | cell Reynolds number | - | derived |

Measured context from `config/experiments/sr_n0_leakance_gated.yaml`: the
median CONUS $Cr$ is **0.226**, i.e. the typical MERIT reach is about 4.4x too
long for an hourly step.

### 6.2 The four coefficients

$$
\mathrm{denom} \;=\; 2K(1-X) + \Delta t
\tag{S20-S23, 44}
$$

$$
c_1 = \frac{-2KX + \Delta t}{\mathrm{denom}}, \quad
c_2 = \frac{2KX + \Delta t}{\mathrm{denom}}, \quad
c_3 = \frac{2K(1-X) - \Delta t}{\mathrm{denom}}, \quad
c_4 = \frac{2\Delta t}{\mathrm{denom}}
\tag{45}
$$

All four share the single denominator (44). The identity

$$
c_1 + c_2 + c_3 \;=\; \frac{\Delta t + 2K(1-X)}{\mathrm{denom}} \;=\; 1
\tag{46}
$$

holds for **any** $(K, X)$, exactly (up to f32 round-off). $c_2, c_4 > 0$
unconditionally; $c_1 \ge 0 \iff X \le Cr/2$ and $c_3 \ge 0 \iff X \le 1 - Cr/2$,
which is the window

$$
2X \;\le\; Cr \;\le\; 2(1-X)
\tag{47}
$$

### 6.3 The storage parameter $X$

`ddr_match: true` (deprecated): $X$ is the caller's constant. Every non-learned
path sets it to **0.3** (`src/training/forward.rs`, `Tensor::full([n], 0.3)`).
The in-code comment records that this severs the link between the scheme's
numerical diffusion and the channel's physical hydraulic diffusivity, giving a
median **28x over-diffusion** on CONUS, and is exact only on the measure-zero
locus $D_{\mathrm{cell}} = 0.4$.

`ddr_match: false` (default), Cunge:

$$
X_{\mathrm{cunge}} \;=\; \mathrm{clamp}\!\left(\tfrac{1}{2}\left(1 - D_{\mathrm{cell}}\right),\, 0,\, 0.5\right)
\tag{S19, 48}
$$

This is the choice that matches numerical to physical diffusivity:

$$
D_{\mathrm{num}} = c L\left(\tfrac12 - X\right) \;=\; \frac{Q_t}{2\,T\,s} \;=\; D_{\mathrm{phys}}
\tag{49}
$$

Measured $X \approx 0.49$ on CONUS, which narrows the non-negativity window
(47) from $[0.6, 1.4]$ (at $X = 0.3$) to about $[0.98, 1.02]$.

### 6.4 `enforce_positivity`: what it does and when it fires

`params.enforce_positivity`, default **false**, requires `ddr_match: false`.
With $\delta =$ `POSITIVITY_DELTA` $= 10^{-2}$ (a Rust `const` in
`src/routing/mmc_op.rs`):

$$
K \;\leftarrow\; \max\!\left(\frac{L}{c},\; \frac{\Delta t\,(1+\delta)}{2}\right)
\tag{S18', 50}
$$

$$
X \;\leftarrow\; \min\!\Big(X_{\mathrm{cunge}},\;
\underbrace{\tfrac{Cr}{2}(1-\delta)}_{\text{hi}_a},\;
\underbrace{\left(1 - \tfrac{Cr}{2}\right)(1-\delta)}_{\text{hi}_b}\Big)
\tag{S19', 51}
$$

(50) puts $Cr \in (0,\, 2/(1+\delta)]$, which is what guarantees
$\mathrm{hi}_b > 0$ so no extra `clamp_min` is needed on the three-way `min`.
(51) forces $c_1, c_3 \ge 0$. With $b \ge 0$ and forward substitution
$x_i = b_i + c_{1,i}\sum_{\text{up}} x_j$ in topological order, induction gives
every $x_i \ge 0$, so no negative solve can reach S28.

**$\delta$ is mandatory, not cosmetic.** At $\delta = 0$ the cap lands exactly
on $c_1 = 0$ / $c_3 = 0$ and f32 round-off crosses it: measured minima
$c_1 = -3.3\times10^{-8}$, $c_3 = -6.8\times10^{-8}$ over a 400k-draw sweep.
$\delta = 10^{-2}$ (about 400x f32 eps) moves those minima to
$+1.8\times10^{-4}$ and $+5.0\times10^{-5}$, costing a 1% tightening of the $X$
ceiling and a 1% rise in the $K$ floor.

**When it fires and what it costs.** From the current config's own commentary:
the $X$ cap binds on **95.3%** of reach-timesteps and median $X$ falls
$0.4976 \to 0.0794$ (6.3x). It therefore largely *replaces* the Cunge $X$
rather than shading it: numerical diffusion becomes stability-set instead of
matched to hydraulic diffusivity. It delivered 0 negative solves out of
197,461,880, but because the cap is applied at runtime to a learned celerity it
made $X \sim Cr \sim 1/n$, handing the optimizer a lever it rode to the
roughness floor: at the same learning rate, `n_at_floor` went
$35\% \to 97.9\%$ with the flag on. **It is off in the current arm.**

### 6.5 Mass conservation: what is and is not conserved

**No coefficient is ever clamped.** A grep over `src/routing/` confirms there
is no `c1.clamp`, `c3.clamp`, or equivalent. This is deliberate: because (46)
holds for any $(K, X)$, clamping the **inputs** $K$ and $X$ preserves
$c_1+c_2+c_3 = 1$ exactly, whereas clamping $c_3$ would not. The code comment
on `params.enforce_positivity` states this reasoning explicitly.

**What does break mass is C10**, the post-solve floor at S28:

$$
Q_{t+1} \;=\; \max\!\left(x_{\mathrm{sol}},\; 10^{-4}\right)
\tag{S28, 52}
$$

applied in `src/routing/mmc_op.rs::forward_chain_inner`. The
`NEG_SOLVES`/`TOTAL_SOLVES` docstring in that file states the case plainly:
measured on CONUS at mean flow with $X = 0.3$, **69.8%** of reaches sit outside
window (47) (28.4% give $c_1 < 0$ and 41.4% give $c_3 < 0$), so negative
discharge is *expected*, and the clamp "both CREATES MASS and removes the only
symptom". Nothing measured this before 2026-08-02. The counters are diagnostic
only (host readback gated on `track_negative_discharge`, enabled in training,
off in eval, and structurally unavailable on the CUDA-graph path).

So the accurate statement for the analysis that follows this document is:
*ddrs does not clamp Muskingum coefficients; it clamps $K$ and $X$ (optionally)
and the solve output (always). The first preserves $c_1+c_2+c_3=1$; the second
creates mass whenever the solve goes negative, at a measured rate of
$5.5\times10^{4}$ per $2\times10^{8}$ solves on a trained head, or 69.8% of
reaches by window-violation count at $X = 0.3$.*

### 6.6 The per-timestep linear system

$$
\left(I - c_1 \odot N\right) Q_{t+1}
\;=\;
c_2 \odot (N Q_t) \;+\; c_3 \odot Q_t \;+\; c_4 \odot q'_t \;-\; \zeta_t
\tag{S24-S27, 53}
$$

with $\odot$ a row-wise (per-reach) scaling, $N$ the sparse adjacency
(`indices_0`/`indices_1` from the topologically ordered adjacency store,
$N_{ij} = 1$ when $j$ flows directly into $i$), and $\zeta_t$ present only when
leakance is active (§8). Written out:

$$
Q_{t+1,i} \;=\; c_{1,i}\!\!\sum_{j \in \mathrm{up}(i)}\!\! Q_{t+1,j}
\;+\; c_{2,i}\!\!\sum_{j \in \mathrm{up}(i)}\!\! Q_{t,j}
\;+\; c_{3,i} Q_{t,i} \;+\; c_{4,i} q'_{t,i} \;-\; \zeta_{t,i}
\tag{54}
$$

This is the standard Muskingum update $O_{j+1} = C_1 I_{j+1} + C_2 I_j + C_3 O_j$
with the inflow $I$ taken as the sum of immediate upstream outflows, which is
what makes the $t+1$ term implicit and requires the solve.

Intermediates, in code order:
$i_t = N Q_t$ (S24, sparse matrix-vector),
$b = c_2 i_t + c_3 Q_t + c_4 q'_t - \zeta$ (S25),
$A_{\text{values}}$ assembled from $c_1$ (S26),
$x_{\mathrm{sol}}$ from the triangular solve (S27),
$Q_{t+1}$ from (52) (S28).

### 6.7 Cold start

`src/routing/utils.rs::compute_hotstart_discharge` / the equivalent CSR path in
`mmc.rs::setup_inputs` with $c \equiv 1$:

$$
(I - N)\,Q_0 = q'_0,
\qquad
Q_0 \leftarrow \max(Q_0,\, 10^{-4})
\tag{55}
$$

On a topologically ordered DAG, $I - N$ is lower triangular with unit diagonal,
so this is the exact upstream accumulation of lateral inflow (a cumulative sum
on a linear chain). An externally supplied `initial_state` replaces it when a
state cache is configured.

### 6.8 Sub-reach subdivision (off by default)

When `parent_offset` marks a parent reach as split into $m_p$ pieces, the
lateral inflow is divided so total $q'$ per parent is conserved:

$$
q'_{\text{row}} = \frac{q'_{\text{parent}}}{m_p}
\tag{56}
$$

applied **after** the C2 clamp (`mmc.rs::forward`), because dividing first would
floor each of the $m$ pieces independently and inject $m \cdot 10^{-4}$ instead
of $10^{-4}$. The same divisor is applied to the cold start by default
(`divide_hotstart_by_pieces: true`). Subdivision is a NO-GO and gated off; see
`.claude/REACH-SUBDIVISION.md`.

---

## 7. The sparse solve and its adjoint

`src/sparse/mod.rs`. `CsrPattern` is built once per network at `setup_inputs`
and reused for every timestep (`Arc`, so per-step cost is a refcount bump).
$A$ is lower triangular because the adjacency is topologically ordered with
$\text{rows}[k] \ge \text{cols}[k]$ (invariant 3).

### 7.1 Assembly

`src/sparse/mod.rs::assemble_primitive`, for each non-zero slot $k$ with
row $r(k)$:

$$
A_{\text{values}}[k] \;=\; \mathrm{diag}[k] \;-\; c_{1}[r(k)] \cdot \mathrm{adj}[k]
\tag{57}
$$

where `diag[k] = 1` at diagonal slots and 0 elsewhere, and `adj[k] = 0` at
diagonal slots. So $A_{ii} = 1$ and $A_{ij} = -c_{1,i} N_{ij}$ for $j \ne i$.

### 7.2 Forward substitution

`src/sparse/mod.rs::forward_sub_lower` (a pure-Rust port of
`scipy.sparse.linalg.spsolve_triangular(A, b, lower=True)`):

$$
x_i \;=\; \frac{b_i - \sum_{k \in \mathrm{row}(i),\, \mathrm{col}(k) \ne i} A_{\text{values}}[k]\; x_{\mathrm{col}(k)}}{A_{ii}}
\tag{58}
$$

evaluated for $i = 0, 1, \ldots, n-1$ in topological order. The CUDA path
(`params.sparse_solver: cuda`) substitutes `cusparseSpSV`; the two are bit-matched
by `tests/cuda_backward_parity.rs`.

### 7.3 The hand-written adjoint

`src/sparse/mod.rs::CsrSolveOp impl Backward<B, 2>` with parents
`[a_values, b]`. From $Ax = b$:

$$
\mathrm{d}x = A^{-1}\left(\mathrm{d}b - \mathrm{d}A\, x\right)
\tag{59}
$$

so with $\bar{x} \equiv \partial L/\partial x$ the incoming gradient:

$$
\bar{b} \;=\; A^{-\mathsf{T}}\,\bar{x}
\tag{60}
$$

$$
\bar{A}_{ij} \;=\; -\,\bar{b}_i\, x_j
\qquad\text{i.e.}\qquad
\bar{A}_{\text{values}}[k] \;=\; -\,\bar{b}\big[r(k)\big]\cdot x\big[\mathrm{col}(k)\big]
\tag{61}
$$

Equation 60 is evaluated by back-substitution on $A^{\mathsf{T}}$, which is
upper triangular, using the cached transposed pattern
(`src/sparse/mod.rs::back_sub_upper_transposed`; `trans_crow`, `trans_col`,
`trans_to_orig` let it read the original `A_values` without rebuilding
structure):

$$
y_i \;=\; \frac{\bar{x}_i - \sum_{k \in \mathrm{row}^{\mathsf{T}}(i),\, \mathrm{col}^{\mathsf{T}}(k) \ne i} A_{\text{values}}\big[\mathrm{trans\_to\_orig}[k]\big]\, y_{\mathrm{col}^{\mathsf{T}}(k)}}{A^{\mathsf{T}}_{ii}},
\quad i = n-1 \ldots 0
\tag{62}
$$

Equation 61 is evaluated as a per-nnz gather-multiply (host loop on CPU,
`select+mul+neg` on GPU). This is the whole point of invariant 4: **$O(nnz)$
tape entries per timestep, not $O(n^2)$.** Do not replace it with autograd-tape
unrolling.

Chaining (61) back through (57):

$$
\bar{c}_1[i] \;=\; -\sum_{k:\, r(k) = i} \bar{A}_{\text{values}}[k]\cdot \mathrm{adj}[k]
\tag{63}
$$

(`src/sparse/mod.rs::assemble_backward_primitive` / `::cpu_assemble_backward`,
`cusparseSpMV` on the GPU with $\alpha = -1$ folding in the negation), and the
SpMV adjoint:

$$
\bar{Q}_t \;=\; N^{\mathsf{T}}\,\bar{i}_t
\qquad\text{i.e.}\qquad
\bar{Q}_t\big[\mathrm{col}(k)\big] \mathrel{+}= \mathrm{adj}[k]\cdot \bar{i}_t\big[r(k)\big]
\tag{64}
$$

(`src/sparse/mod.rs::spmv_backward_primitive` / `::cpu_spmv_backward`).

---

## 8. Leakance: the GW-SW exchange term

`src/routing/leakance.rs`, ported from DDR `_compute_zeta` (`mmc.py`, commit
`c2bd0f9`). `params.use_leakance`, default **false**. Status: CLOSED and NOT
PROMOTABLE (see `CLAUDE.md` and §8.6), but code-complete and gradient-exact,
and enabled in the current experiment arm with the gate of §3.

### 8.1 The flux

`src/routing/leakance.rs::zeta_forward`:

$$
w_z \;=\; \left(p\,d\right)^{q_\varepsilon}
\tag{65}
$$

$$
\mathcal{A}_z \;=\; w_z \cdot L
\tag{66}
$$

$$
h \;=\;
\begin{cases}
\max\!\left(0,\; d - d_{gw}\right), & \texttt{leakance\_losing\_only = true (default)}\\[2pt]
d - d_{gw}, & \texttt{false (back-compat)}
\end{cases}
\tag{67}
$$

$$
\zeta \;=\; f \cdot \mathcal{A}_z \cdot K_D \cdot h \cdot \mathbb{1}_{\text{mask}}
\tag{68}
$$

$$
b \;\leftarrow\; b - \zeta
\qquad\text{(S25; positive } \zeta \text{ = losing reach)}
\tag{69}
$$

$d$ and $q_\varepsilon$ are the **shared** S6 depth and S1 exponent, not
recomputed. `zeta_forward` returns $(w_z, \mathcal{A}_z, \zeta)$; only
$\mathcal{A}_z$ is saved for the backward.

| symbol | meaning | units | status |
|---|---|---|---|
| $f$ | `leakance_factor`, gated per §3 | dimensionless | learned |
| $K_D$ | leakance = conductivity / streambed thickness | s$^{-1}$ | learned |
| $d_{gw}$ | groundwater depth parameter, box $[-2, 2]$ | m | learned |
| $\mathcal{A}_z$ | plan-view wetted area | see §8.3 | derived |
| $h$ | driving head | m | derived |
| $\zeta$ | exchange flux, positive = losing | m³/s | derived |
| $\mathbb{1}_{\text{mask}}$ | impervious mask, $0$ or $1$ | - | data (constant, no gradient) |

### 8.2 The impervious mask

`src/data/dataset.rs::build_mask_from_raw_col`, from the
`corridor_impervious` attribute column with threshold
`params.leakance_impervious_threshold` (default 0.7, i.e. 70% impervious
$\approx$ a concrete-lined channel):

$$
\mathbb{1}_{\text{mask}}[i] \;=\;
\begin{cases}
1, & \texttt{corridor\_impervious}[i] \le \text{threshold} \ \ \textbf{or NaN}\\
0, & \text{otherwise}
\end{cases}
\tag{70}
$$

NaN maps to 1 on purpose: no StreamCat coverage means *absence of imperviousness
data*, which is not the same as impervious concrete, so leakance is allowed
where data is missing. The mask is a plain inner-backend tensor, never
autograd-tracked, and it is only built when `use_leakance` is true **and**
`corridor_impervious` is among the configured attributes. Where
$\mathbb{1}_{\text{mask}} = 0$, both $\zeta$ and all six leakance partials are
exactly zero.

### 8.3 The plan-view area, and a dimensional problem

Equation 65 is $(p\,d)^{q_\varepsilon}$, **not** $p\,d^{q_\varepsilon}$. It is
therefore not the top width $T$ of equation 21. With $[p] = \mathrm{m}^{1-q}$:

$$
[w_z] = \left(\mathrm{m}^{1-q}\cdot\mathrm{m}\right)^{q} = \mathrm{m}^{(2-q)q},
\qquad
[\mathcal{A}_z] = \mathrm{m}^{(2-q)q + 1}
\tag{71}
$$

$[\mathcal{A}_z] = \mathrm{m}^2$ only when $(2-q)q = 1$, i.e. only at $q = 1$.
At the prescribed $q = 0.65$ the exponent is $1.878$, so $\mathcal{A}_z$ has
units $\mathrm{m}^{1.878}$ and $\zeta$ is not in m³/s. The intended quantity
is presumably $T \cdot L = p\,d^{q}L$ (a genuine streambed plan area). This is
inherited from DDR verbatim, is not flagged anywhere in the source, and is
listed in §12 as a discrepancy. **Numerically it means $K_D$ absorbs a
$q$- and $p$-dependent unit conversion, so a learned $K_D$ is not directly
comparable to a measured streambed leakance.** The reported `area_z_mean`
diagnostic is labelled "m^2" in the NetCDF attributes and is not.

### 8.4 The relation to MODFLOW's river package

MODFLOW RIV computes $Q_{\text{riv}} = C_{\text{riv}}(h_{\text{riv}} - h_{\text{aq}})$
with $C_{\text{riv}} = K_v W L / M$ for streambed vertical conductivity $K_v$,
width $W$, length $L$ and thickness $M$. Equation 68 is that form with

$$
C \;=\; f \cdot \mathcal{A}_z \cdot K_D,
\qquad
K_D \;\equiv\; \frac{K_v}{M} \quad [\mathrm{s^{-1}}]
\tag{72}
$$

so **$K_D$ is leakance proper** (conductivity over streambed thickness, inverse
seconds), $\mathcal{A}_z$ plays the role of $WL$, and $f$ is an extra
dimensionless area-fraction-or-selector that MODFLOW does not have.

Two departures from MODFLOW worth stating. First, the driving head in
equation 67 is $d - d_{gw}$: a *flow depth* minus a "groundwater depth
parameter" with box $[-2, 2]$ m, not a stage minus an aquifer head. It equals a
true head difference only under the reading that $d_{gw}$ is the water-table
depth below the channel bed. Second, with `leakance_losing_only: true` (the
default) the term is one-directional: gaining reaches produce exactly zero, so
the model cannot represent baseflow gain at all.

### 8.5 The backward: all six partials

`src/routing/leakance.rs::zeta_backward`. Because equation 69 subtracts
$\zeta$, the incoming gradient is negated:

$$
\bar{\zeta} \;=\; -\bar{b} \cdot \mathbb{1}_{\text{mask}} \cdot \mathbb{1}\!\left[d > d_{gw}\right]
\tag{73}
$$

where the $\mathbb{1}[d > d_{gw}]$ gate is applied only when
`losing_only = true`, with the subgradient at the kink $d = d_{gw}$ defined as
**0** (measure-zero, finite-difference safe). Gating $\bar{\zeta}$ once zeroes
all six outputs at a stroke.

With $m \equiv d - d_{gw}$ (recomputed **unclamped** in the backward, which is
safe precisely because $\bar{\zeta}$ is already gated):

$$
\frac{\partial L}{\partial f} \;=\; \bar{\zeta}\,\mathcal{A}_z\,K_D\,m
\tag{74}
$$

$$
\frac{\partial L}{\partial K_D} \;=\; \bar{\zeta}\,f\,\mathcal{A}_z\,m
\tag{75}
$$

$$
\frac{\partial L}{\partial d_{gw}} \;=\; -\,\bar{\zeta}\,f\,\mathcal{A}_z\,K_D
\tag{76}
$$

$$
\frac{\partial L}{\partial p}\bigg|_{\zeta} \;=\; \bar{\zeta}\,f\,K_D\,m\,\mathcal{A}_z\,\frac{q_\varepsilon}{p}
\tag{77}
$$

$$
\frac{\partial L}{\partial q_\varepsilon}\bigg|_{\zeta} \;=\; \bar{\zeta}\,f\,K_D\,m\,\mathcal{A}_z\,\ln\!\left(p\,d\right)
\tag{78}
$$

$$
\frac{\partial L}{\partial d}\bigg|_{\zeta} \;=\; \bar{\zeta}\,f\,K_D\,\mathcal{A}_z
\;+\; \bar{\zeta}\,f\,K_D\,m\,\mathcal{A}_z\,\frac{q_\varepsilon}{d}
\tag{79}
$$

Equations 74-76 are registered directly on the three leakance parents.
Equations 77-79 are the *geometry-side* contributions and are folded into the
base operator's existing accumulators (`gp_total`, `gq_spatial`, `gd_total`)
through the `zeta_hook` closure, which is called the moment $\bar{b}$ becomes
available at B27. Verified against central differences by
`src/routing/leakance.rs::grad_tests::zeta_grads_match_central_differences`
(relative tolerance $10^{-3}$ to $10^{-2}$ per partial) and by
`tests/leakance_gradcheck.rs`.

Derivation check for equation 79: $\zeta = fK_DmL(pd)^{q}$, so
$\partial\zeta/\partial d = fK_DL[(pd)^q + m\,q\,(pd)^{q-1}p]
= fK_D[\mathcal{A}_z + m\,\mathcal{A}_z q/d]$, matching term for term.

### 8.6 The identifiability verdict

Recorded in `CLAUDE.md` and
`research/findings/2026-07-06-leakance-nogo-scientific-summary.md` §3, and
reproduced here because it directly bears on "are the learned parameters
physically meaningful": **a gauge measures the SUM of $\zeta$ over its upstream
network, and that sum does not determine the per-reach distribution.** Training
therefore constrains the aggregate loss while carrying zero information about
per-reach flux. Every rival explanation (gradient starvation, objective noise,
uninformative inputs, sign ambiguity) was individually refuted. The gate of §3
is an attempt to change *what* is being identified (a selector rather than a
magnitude), not a refutation of that verdict.

---

## 9. Gradient paths

`src/routing/mmc_op.rs`. One autograd node per timestep instead of about 33
BURN ops. The forward chain is labelled S1-S28 (with S18', S19' for the
positivity clamps); the backward mirrors it as B28-B1 (with B18', B19') inside
the single shared function
`src/routing/mmc_op.rs::timestep_backward_core`.

### 9.1 Backward labels, in execution order

| label | differentiates | key expression |
|---|---|---|
| **B28** | S28 output clamp (C10) | $\bar{x}_{\mathrm{sol}} = \bar{y}\cdot\mathbb{1}[x_{\mathrm{sol}} > 10^{-4}]$ |
| **B27** | the triangular solve | equations 60-61; calls `dispatch::backward_solve_primitive`, then the per-nnz $-\bar{b}[r]x[c]$ |
| *zeta hook* | equations 73-79 | called with $\bar{b}$ the moment B27 produces it; returns the three geometry grads |
| **B26** | S26 assembly | equation 63, $\bar{c}_1$ from $\bar{A}_{\text{values}}$ |
| **B25** | S25 RHS | $\bar{c}_2 = \bar{b}\,i_t$; $\bar{c}_3 = \bar{b}\,Q_t$; $\bar{c}_4 = \bar{b}\,q'_t$; $\bar{i}_t = c_2\bar{b}$; $\bar{Q}_t \mathrel{+}= c_3\bar{b}$; $\bar{q}'_t = c_4\bar{b}$ |
| **B24** | S24 SpMV | equation 64; **the one $O(nnz)$ piece owned by a single parent**, skipped entirely when $Q_t$ is untracked |
| **B23-B20** | $c_1..c_4$ and the shared denominator | $\overline{\mathrm{denom}}\big|_{c_i} = -\bar{c}_i\,\mathrm{num}_i/\mathrm{denom}^2$; $\overline{\mathrm{num}_i} = \bar{c}_i/\mathrm{denom}$; accumulated into $\overline{2KX}$ and $\overline{2K(1-X)}$ |
| **B19** | the Cunge $X$ (`ddr_match: false` only) | $\bar{X} = 2K(\overline{2KX} - \overline{2K(1-X)})$, then $\partial X/\partial Q_t = -\tfrac{1}{2}/(TscL)$, $\partial X/\partial T = +\tfrac{1}{2}D_{\mathrm{cell}}/T$, $\partial X/\partial c = +\tfrac{1}{2}D_{\mathrm{cell}}/c$, masked by C8 |
| **B19'** | the S19' three-way `min` (`enforce_positivity`) | partitions $\bar{X}$ across branches Cunge > hi$_a$ > hi$_b$ by a `<=` cascade; opens the NEW path $X \to Cr \to K$, accumulated as $\bar{K}\big|_{X\text{-cap}} = -\bar{Cr}\,\Delta t/K^2$ |
| **B18'** | the S18' $K$ floor (C7) | $\bar{K}_{\mathrm{raw}} = \bar{K}\cdot\mathbb{1}[L/c > \Delta t(1+\delta)/2]$ |
| **B18** | $K = L/c$ | $\bar{c} = -\bar{K}_{\mathrm{raw}}L/c^2$, plus B19's $\partial X/\partial c$ term |
| **B17** | the celerity | `ddr_match`: $\bar{v} = \bar{c}\cdot 5/3$. Else with $G = \tfrac43 A\sqrt{1+z^2}/(TP_w)$ and $H = \gamma A/(Td)$: $\bar{v} = \bar{c}\beta$, $\bar\beta = \bar{c}v$, $\partial\beta/\partial A = -G/A + H/A$, $\partial\beta/\partial T = +G/T - H/T$, $\partial\beta/\partial P_w = +G/P_w$, $\partial\beta/\partial z = -Gz/(1+z^2)$, $\partial\beta/\partial d = -H/d$, $\partial c/\partial\gamma = vA/(Td)$ |
| **B16** | S16 velocity clamp (C6) | mask $v_{\mathrm{lb}} < v_{\mathrm{un}} < 15$ |
| **B15** | Manning velocity | $\partial v/\partial n = -v/n$; $\partial v/\partial R = \tfrac23 v/R$; and with stage roughness $\partial v/\partial d\big|_{\text{explicit}} = \gamma v/d$, $\partial v/\partial\gamma = v\ln(d/d_{\mathrm{ref}})$ |
| **B14** | $R = A/P_w$ | $\partial R/\partial A = 1/P_w$; $\partial R/\partial P_w = -R/P_w$ |
| **B13** | wetted perimeter | $\partial P_w/\partial b = 1$; $\partial P_w/\partial d = 2\sqrt{1+z^2}$; $\partial P_w/\partial z = 2dz/\sqrt{1+z^2}$ |
| **B12** | area | $\partial A/\partial T = \partial A/\partial b = d/2$; $\partial A/\partial d = (T+b)/2$ |
| **B11** | bottom-width floor (C5) | mask $b_{\mathrm{raw}} > b_{\mathrm{lb}}$ |
| **B10** | $b_{\mathrm{raw}} = T - 2zd$ | $\partial/\partial T = 1$; $\partial/\partial z = -2d$; $\partial/\partial d = -2z$ |
| **B9** | side-slope clamp (C4) | mask $0.5 < z_{\mathrm{raw}} < 50$ |
| **B8** | $z_{\mathrm{raw}} = Tq_\varepsilon/(2d)$ | $\partial/\partial T = q_\varepsilon/(2d)$; $\partial/\partial q_\varepsilon = T/(2d)$; $\partial/\partial d = -Tq_\varepsilon/(2d^2)$ |
| **B7** | $T = p\,d^{q_\varepsilon}$ | $\partial T/\partial p = T/p$; $\partial T/\partial d = Tq_\varepsilon/d$; $\partial T/\partial q_\varepsilon = T\ln d$ |
| **B6** | depth floor (C3) and $d = \mathcal{R}^e$ | mask $d > d_{\mathrm{lb}}$; $\partial d/\partial\mathcal{R} = e\,d/\mathcal{R}$; $\partial d/\partial e = d\ln\mathcal{R}$ |
| **B5** | the exponent | $\partial e/\partial q_\varepsilon = \partial e/\partial\gamma = -9/(5+3q_\varepsilon+3\gamma)^2$ |
| **B4** | $\mathcal{R} = \mathcal{N}/\mathcal{D}$ | $\partial\mathcal{R}/\partial\mathcal{N} = 1/\mathcal{D}$; $\partial\mathcal{R}/\partial\mathcal{D} = -\mathcal{R}/\mathcal{D}$ |
| **B3** | $\mathcal{D} = p\sqrt{s} + 10^{-8}$ | $\partial\mathcal{D}/\partial p = \sqrt{s}$ (slope is constant) |
| **B2** | the numerator | $\partial\mathcal{N}/\partial Q_t = n(q_\varepsilon+1)d_{\mathrm{ref}}^{\gamma}$; $\partial/\partial n = Q_t(q_\varepsilon+1)d_{\mathrm{ref}}^{\gamma}$; $\partial/\partial q_\varepsilon = Q_t n\,d_{\mathrm{ref}}^{\gamma}$ |
| **B1** | $q_\varepsilon = q + 10^{-6}$ | identity; sums the four $q_\varepsilon$ contributions (B8, B7, B5, B2) plus the zeta one |

Two implementation facts that matter for reading the code: `state.c1` and
`state.x_storage` are deliberately **never read** in the backward
(`gc1` comes from the assembled A-values at B26, and $X$ comes from the saved
`x_effective`, which is what $c_1..c_4$ were actually built from); and the
$X$-cap term $\bar{K}\big|_{X\text{-cap}}$ must join $\bar{K}$ *after* the
$c_1..c_4$ contribution but *before* the B18' floor mask, because both paths
reach the celerity through the same `clamp_min`.

### 9.2 How many gradient paths reach each quantity

- **$Q_t$** has up to **four** paths: the S25 RHS ($c_3 Q_t$), the S24 SpMV
  ($N Q_t$), the S2 depth chain, and, under `ddr_match: false`, the Cunge $X$
  at S19. This is why the routing is recurrent in the parameters and not just
  in the state.
- **$T$** has up to five contributors before B7 consumes it: B12, B10, B8,
  B17's $\partial\beta/\partial T$, and B19's $\partial X/\partial T$.
- **$d$** has up to seven: B13, B12, B10, B8, B7, plus B17's $-H/d$ and B15's
  $\gamma v/d$ under stage roughness, plus the zeta term (79).
- **$\gamma$**, when learned, accumulates exactly three: B5 (depth exponent),
  B15 (the explicit $(d/d_{\mathrm{ref}})^{\gamma}$ on the velocity), and B17
  (the celerity's $\gamma A/(Td)$). The code asserts all three are `Some`
  together or `None` together.

### 9.3 The three operators and `ParentMask`

| operator | arity | parents, in fixed order |
|---|---|---|
| `TimestepOp` | `Backward<I, 5>` | `n`, `q_spatial`, `p_spatial`, `q_t`, `q_prime_t` |
| `TimestepGammaOp` | `Backward<I, 6>` | the five above **+ `gamma`** |
| `TimestepLeakanceOp` | `Backward<I, 8>` | the five above **+ `K_D`, `d_gw`, `leakance_factor`** |

All three share `timestep_backward_core`; only parent bookkeeping differs.
`TimestepGammaOp` exists because a tensor that is not a parent receives no
gradient at all, and widening `TimestepOp` would have given every existing run
a sixth parent it does not use. `TimestepLeakanceOp` has **no `gamma` parent**,
which is why config load rejects learned `gamma` together with `use_leakance`.

`src/routing/mmc_op.rs::ParentMask` is a six-field `bool` struct built from
`ops.parents` by each `Backward` impl (`ids[i].is_some()`). It gates only:

1. each parent's **final assembly** (`mask.n.then(|| ...)` and friends), and
2. the one $O(nnz)$ piece that belongs to a single parent, namely B24's
   $N^{\mathsf{T}}\bar{i}_t$ for `q_t`.

The shared chain through the solve and the geometry is needed by every parent
and is never skipped. `register_parent` **panics** if a tracked parent arrives
without a gradient, so a mask that disagrees with `ops.parents` is a loud bug
rather than a silent zero. Typical untracked parents: `p_spatial` when it is a
fixed constant, `q_t` at the first timestep (the hotstart state), and
`q_prime_t` in ordinary training (the lateral inflow is data; only the adjoint
study lifts it to a leaf).

---

## 10. Loss functions

`src/training/loss.rs`, selected by `experiment.loss.kind`
(`src/config.rs::LossKind`). There are **five** kinds, not four.

### 10.1 The `tau` output shift and daily pooling

`src/training/loss.rs::tau_trim_and_downsample`, applied to the hourly routed
prediction $(G, T_{\mathrm{hours}})$ before any loss or metric.

$$
\text{slice} \;=\; \big[\,\tau,\; T_{\mathrm{hours}} - (24 - \tau)\,\big),
\qquad
T_{\mathrm{days}} = \frac{T_{\mathrm{hours}} - 24}{24}
\tag{80}
$$

$$
\bar{Q}_{g,i} \;=\; \sum_{j} W_{i,j}\,Q^{\mathrm{hourly}}_{g,\,\tau + j},
\qquad
W_{i,j} = \frac{\big|\,[\,is,\,(i+1)s\,) \cap [\,j,\,j+1\,)\,\big|}{s},
\quad s = \frac{L}{M}
\tag{81}
$$

Equation 81 is `torch.nn.functional.interpolate(mode="area")` for the 1D case,
implemented as an explicit weight matrix (`::area_pool_weights`); each row sums
to 1, and because the trimmed length is an exact multiple of 24, $s = 24$ and
the pooling is an exact block mean.

**The sign convention, and the trap.** Under the convention in force since
2026-08-08, pooled day $i$ covers hours $[\tau + 24i,\ \tau + 24(i+1))$ and is
scored against **observation day $i$**. So $\tau$ is the number of hours the
routed output is **advanced** before scoring (a translation-only
inverse-routing shift, same sign and magnitude as dMC-Juniata's tau):
$\tau = 0$ is exactly day-aligned; $\tau = 9$ pairs obs day $i$ with routed
hours $[24i + 9,\ 24i + 33)$.

The pre-2026-08-08 slice was $[13+\tau,\ -11+\tau)$ with pooled day $i$ scored
against obs day $i+1$, so

$$
\tau_{\mathrm{new}} \;=\; \tau_{\mathrm{old}} - 11
\tag{82}
$$

The old shipped value 3 is $\tau_{\mathrm{new}} = -8$; the old measured optimum
20 is $\tau_{\mathrm{new}} = 9$. **`params.tau` is a `u32` asserted to lie in
$[0, 24)$, so $\tau_{\mathrm{new}} = -8$ is unrepresentable.** The default is
still `tau: 3` (`src/config.rs::Params::default`), and under the new
convention that denotes a different shift than the legacy `tau: 3` did. Any config carrying
`tau: 3` inherited from before 2026-08-08 is therefore silently reinterpreted;
`config/experiments/gridded_conus.yaml` carries the migrated `tau: 9`.
DDR-Python's `compute_daily_runoff` still uses the legacy form. Total trim is
24 h under both conventions, so $T_{\mathrm{days}}$ is unchanged and a
$\tau$ sweep scores an identical day sample at every shift.

### 10.2 Masking rule for missing observations

Two-stage, and the same rule in both stages.

**Stage 1, per gauge, in the driver** (`src/training/driver.rs`). After the
post-warmup slice, a gauge is **dropped entirely** if its window contains *any*
NaN:

$$
\text{keep}(g) \iff \forall t \in [\text{warmup},\, T_{\mathrm{days}}):\ \neg\,\mathrm{isNaN}\!\left(o_{g,t}\right)
\tag{83}
$$

Kept gauges are selected with `Tensor::select` so autograd stays alive on the
predictions. If no gauge survives, the micro-batch is skipped. The driver's
comment records why this matters: without it, a NaN-to-0.0 substitution biases
the head toward predicting near-zero flow and saturates Manning's $n$ at the
lower bound.

Stage 1 therefore means **every loss below sees NaN-free observations**. The
per-element masking that follows only ever matters for the derivative term.

**Stage 2, per adjacent pair, inside `obs_diff_term`.** A pair $(j, j+1)$
counts for gauge $g$ only when both days are finite, so a gap **breaks the
chain**: $N$ valid days with a hole give strictly fewer than $N-1$ pairs.
Implemented as a dense $(G, T-1)$ weight tensor that is zero at invalid pairs,
with the observed-difference tensor also zero there, so no gather is needed,
the term stays differentiable in $p$, and a NaN observation can never reach the
arithmetic.

`src/training/metrics.rs::Metrics::compute` masks differently again: it drops
individual non-finite *pairs* rather than whole gauges, and emits NaN for a
gauge with no finite pair at all.

### 10.3 `l1` (the default)

$$
L_{\mathrm{L1}} \;=\; \frac{1}{G\,T}\sum_{g=1}^{G}\sum_{t=1}^{T}\left|p_{g,t} - o_{g,t}\right|
\tag{84}
$$

A mean over **elements**. `src/training/loss.rs::l1_loss_post_warmup` is the
`ndarray` twin used for logging.

### 10.4 `nse-batch` (dHBV's `NSELossBatch`)

$$
L_{\mathrm{nse}} \;=\; \frac{1}{G\,T}\sum_{g}\sum_{t}
\frac{\left(p_{g,t} - o_{g,t}\right)^2}{\left(\sigma_g + \epsilon\right)^2}
\tag{85}
$$

$\epsilon$ is `loss.eps`, default **0.1** (matching DDR `hydrograph_loss`), and
it sits **inside the square, added to $\sigma$**, not added to $\sigma^2$.
$\sigma_g$ is the gauge's observed standard deviation over the **whole training
period**, a fixed vector (`MeritGagesDataset::gauge_obs_std`), *not* recomputed
per window; recomputing per window would make the objective drift between
micro-batches and break accumulation exactness. Pairs with
`experiment.optimizer: adadelta` by design (scale-free), though the current arm
uses Adam.

### 10.5 `nse-batch-deriv`

$$
L \;=\; L_{\mathrm{nse}} \;+\; \lambda \cdot L_{\mathrm{deriv}}
\tag{86}
$$

$$
L_{\mathrm{deriv}} \;=\; \frac{1}{N_{\mathrm{pairs}}}\sum_{(g,j)\ \mathrm{valid}}
\frac{\Big[\left(p_{g,j+1} - p_{g,j}\right) - \left(o_{g,j+1} - o_{g,j}\right)\Big]^2}
{\left(\sigma^{\mathrm{d}}_g + \epsilon\right)^2}
\tag{87}
$$

$\lambda$ is `loss.deriv_weight`, default 0.5. $N_{\mathrm{pairs}}$ is the
count of valid adjacent pairs across the whole batch, so (87) is one global
mean, exactly as (85)'s `.mean()` is one global mean.
$\sigma^{\mathrm{d}}_g$ is the gauge's observed **consecutive-day difference**
standard deviation over the training period
(`MeritGagesDataset::gauge_obs_diff_std`, returning 0.0 for a degenerate
gauge, where the $+\epsilon$ is what keeps the term finite). Returns exactly 0
when $T < 2$ or every pair is broken, so the composite degrades to the level
term rather than to NaN. At $\lambda = 0$ it is numerically identical to (85).

Rationale from the docstring: the landscape study's curvature probe found the
loss curvature in Manning's $n$ to be governed by the mean square of the
hydrograph's **time derivative**, so a level-only objective leaves the $n$
valley nearly flat and the aggregate gradient vanishes long before $n$ is
identified. Scoring the derivative directly deepens that valley (predicted
about 3.7x at $\lambda = 0.5$). This mirrors
`experiment::landscape::objective::deriv_window_loss` term for term.

### 10.6 `nnse-kge`

Per gauge, over the time axis, with **population** moments (divide by $T$):

$$
\mu_p = \overline{p_g}, \quad \mu_o = \overline{o_g}, \quad
\sigma_p = \sqrt{\overline{(p-\mu_p)^2} + \epsilon}, \quad
\sigma_o = \sqrt{\overline{(o-\mu_o)^2} + \epsilon}
\tag{88}
$$

$$
r = \frac{\overline{(p-\mu_p)(o-\mu_o)}}{\sigma_p\,\sigma_o},
\qquad
\alpha = \frac{\sigma_p}{\sigma_o},
\qquad
\beta = \frac{\mu_p}{\mu_o + \epsilon}
\tag{89}
$$

$$
1 - \mathrm{KGE} \;=\; \sqrt{(r-1)^2 + (\alpha-1)^2 + (\beta-1)^2}
\tag{90}
$$

$$
\mathrm{NSE} = 1 - \frac{\sum_t (p - o)^2}{\sum_t (o - \mu_o)^2 + \epsilon},
\qquad
\mathrm{NNSE} = \frac{1}{2 - \mathrm{NSE}},
\qquad
1 - \mathrm{NNSE}
\tag{91}
$$

$$
L_{\mathrm{nnse\text{-}kge}} \;=\; \lambda_{\mathrm{nnse}}\,\Big\langle 1 - \mathrm{NNSE}\Big\rangle_g
\;+\; \lambda_{\mathrm{kge}}\,\Big\langle 1 - \mathrm{KGE}\Big\rangle_g
\tag{92}
$$

$\lambda$'s are `loss.nnse_weight` / `loss.kge_weight`, both default 1.0.
$\langle\cdot\rangle_g$ is a mean over **gauges**, so large basins do not
dominate. **Epsilon placement**, exactly as coded: $+\epsilon$ is added to the
*variance* before the square root in (88) (so it stabilizes $\sigma$, not
$\sigma^2$); $+\epsilon$ is added to $\mu_o$ in the $\beta$ denominator; and
$+\epsilon$ is added to the *sum* of squared observed deviations (SSO) in (91),
not to its mean. The $r$ and $\alpha$ ratios are invariant to the population-vs-
sample choice; NSE's SSE/SSO is too.

Rationale: L1 and NSE are both maximized at a simulated variance *below*
observed (NSE's optimum sits at $\alpha = r < 1$), so they reward the
Muskingum-Cunge routing for over-attenuating flood peaks. This was the
diagnosed cause of KGE regressing against the summed-Q' baseline: median KGE
$0.723 \to 0.701$ while NSE *improved* $0.639 \to 0.684$, with the whole drop in
$\alpha = \sigma_{\mathrm{sim}}/\sigma_{\mathrm{obs}}$ falling $0.93 \to 0.85$.
KGE's $(\alpha-1)^2$ supplies the restoring gradient; NNSE guards correlation
and volume.

### 10.7 `kge` (component-weighted)

$$
L_{\mathrm{kge}} \;=\; \Big\langle \min\!\big(\,w_r(r-1)^2 + w_\alpha(\alpha-1)^2 + w_\beta(\beta-1)^2,\; \kappa\,\big)\Big\rangle_g
\;+\; \lambda_{\mathrm{nnse}}\Big\langle 1 - \mathrm{NNSE}\Big\rangle_g
\tag{93}
$$

with $r, \alpha, \beta$ from (89) and $\epsilon$ placed identically.
$w_r, w_\alpha, w_\beta$ default 1.0; $\kappa$ is `loss.kge_clamp`, default 10
(clamp C13). When $\lambda_{\mathrm{nnse}} = 0$ the function returns after the
first term, skipping the NNSE computation entirely.

Two reasons for the squared, unrooted form over (90), from the docstring:
$\sqrt{\cdot}$ has an infinite-slope cusp as the prediction approaches perfect
KGE (the argument goes to 0), whereas the squared form is smooth there; and
$w_\alpha$ can independently up-weight the variance-ratio term, the direct
counter-pressure to MC over-attenuation. $\kappa$ exists because a gauge with
near-constant observed flow has a collapsing $\sigma_o$ / $\mu_o$ denominator
and $(\alpha-1)^2$ / $(\beta-1)^2$ can explode: a single gauge drove batch loss
to about $10^4$ in testing.

### 10.8 Gradient-accumulation weights

`src/training/loss.rs::loss_denominator` returns the exact-recombination
weight, which differs by objective because L1 and the NSE forms average over
**elements** while the KGE composites average over **gauges**:

$$
n_i \;=\;
\begin{cases}
G^{\mathrm{kept}}_i \cdot T^{\mathrm{post}}_i, & \texttt{l1},\ \texttt{nse-batch},\ \texttt{nse-batch-deriv}\\
G^{\mathrm{kept}}_i, & \texttt{nnse-kge},\ \texttt{kge}
\end{cases}
\tag{94}
$$

$$
L_{\text{pooled}} \;=\; \frac{\sum_i L_i\, n_i}{\sum_i n_i}
\tag{95}
$$

A naive $1/N$ average is wrong whenever micro-batches differ in surviving-gauge
count, which the NaN filter (83) makes the common case. For
`nse-batch-deriv`, recombination is exact only to $O(1/T^{\mathrm{post}})$
because the derivative term's pair count is $G(T-1)$ rather than $GT$; the
docstring recommends single-batch training with that objective if
micro-batches differ in gauge count.

| symbol | meaning | units | status |
|---|---|---|---|
| $p_{g,t}$ | daily routed prediction | m³/s | derived |
| $o_{g,t}$ | daily USGS observation | m³/s | data |
| $\sigma_g, \sigma^{\mathrm{d}}_g$ | training-period per-gauge std of obs / obs differences | m³/s | data (fixed per run) |
| $\epsilon$ | `loss.eps`, default 0.1 | m³/s (it is added to a std) | config |
| $\tau$ | output advance in hours, $[0,24)$ | h | config |
| warmup | days dropped from the head of each window | d | config |

---

## 11. Diagnostics used for validation

### 11.1 The KGE decomposition

`src/training/metrics.rs::Metrics::compute`, per gauge, on the post-warmup
daily arrays, dropping individual non-finite pairs. **Sample-free population
moments** (divide by $n$, the number of surviving pairs) and **no epsilon**:
the guards are explicit zero tests that emit NaN instead:

$$
r \;=\; \frac{\frac1n\sum(p-\mu_p)(o-\mu_o)}{\sigma_p\,\sigma_o}
\quad\text{(NaN if }\sigma_p = 0\text{ or }\sigma_o = 0)
\tag{96}
$$

$$
\alpha \;=\; \frac{\sigma_p}{\sigma_o}
\quad\text{(NaN if }\sigma_o = 0)
\tag{97}
$$

$$
\beta \;=\; \frac{\mu_p}{\mu_o}
\quad\text{(NaN if }\mu_o = 0)
\tag{98}
$$

$$
\mathrm{KGE} \;=\; 1 - \sqrt{(r-1)^2 + (\alpha-1)^2 + (\beta-1)^2}
\tag{99}
$$

Precisely: **$r$ is the Pearson correlation of the daily series**,
**$\alpha$ is the variability ratio $\sigma_{\mathrm{sim}}/\sigma_{\mathrm{obs}}$
of the daily series** (not of the flow-duration curve, and not a
coefficient-of-variation ratio), and **$\beta$ is the bias ratio
$\mu_{\mathrm{sim}}/\mu_{\mathrm{obs}}$**. Note the distinction from the loss:
equation 99 has **no $\epsilon$**, whereas (88)-(89) do, so a reported KGE and
the KGE term inside `nnse-kge` are not the same number on a low-variance gauge.

The companion metrics, also per gauge:

$$
\mathrm{NSE} = 1 - \frac{\sum(p-o)^2}{\sum(o-\mu_o)^2}
\quad\text{(NaN if SSO} = 0),
\qquad
\mathrm{RMSE} = \sqrt{\frac1n\sum(p-o)^2}
\tag{100}
$$

$$
\mathrm{bias} \;=\; \frac1n\sum_t\left(p_t - o_t\right)
\qquad [\mathrm{m^3/s}]
\tag{101}
$$

$$
\mathrm{FHV} = 100\,\frac{\sum_{\mathrm{top\,2\%}}(p^{\uparrow} - o^{\uparrow})}{\sum_{\mathrm{top\,2\%}} o^{\uparrow}},
\qquad
\mathrm{FLV} = 100\,\frac{\sum_{\mathrm{bottom\,30\%}}(p^{\uparrow} - o^{\uparrow})}{\sum_{\mathrm{bottom\,30\%}} o^{\uparrow}}
\tag{102}
$$

$p^{\uparrow}$ and $o^{\uparrow}$ are each series **sorted independently**, so
FHV and FLV are flow-duration-curve volume biases and are **not
timestep-paired**. Indices are `round(0.98 n)` and `round(0.30 n)`. `bias` is a
signed mean difference in m³/s, not a percentage or a ratio.

### 11.2 The eval-time zeta flux accumulation

Turned on by `MuskingumCunge::enable_zeta_accumulation` (eval only; the
training path never enables it). Per timestep, `route_timestep` adds to five
inner-backend running sums with no autograd tape
(`src/routing/mmc.rs`, `ZetaSumTensors`):

$$
S^{\mathrm{abs}}_i = \sum_{t} \left|\zeta_{t,i}\right|, \quad
S^{\mathrm{net}}_i = \sum_{t} \zeta_{t,i}, \quad
S^{d}_i = \sum_{t} d_{t,i}, \quad
S^{\mathcal{A}}_i = \sum_{t} \mathcal{A}_{z,t,i}, \quad
S^{Q}_i = \sum_{t} Q_{t+1,i}
\tag{103}
$$

Across chunked eval calls these merge in `src/training/forward.rs::ZetaSums`
(eval builds a fresh engine per chunk), and `src/training/eval.rs` divides by
the accumulated step count:

$$
\overline{X}_i \;=\; \frac{S^{X}_i}{n_{\mathrm{steps}}}
\tag{104}
$$

`src/dump_parameters.rs::write_zeta_netcdf` writes these onto a `COMID_eval`
dimension in `<run_dir>/kan_parameters.nc`. **Precisely what each reported
number is:**

| NetCDF variable | quantity | units | caveat |
|---|---|---|---|
| `zeta` | eval-window mean of $|\zeta|$ per reach | m³/s | **magnitude**, so a reach that alternates sign does not cancel; this is the $|\zeta| > 0.01$ m³/s GO/NO-GO bar |
| `zeta_net` | eval-window mean of signed $\zeta$ | m³/s | positive = losing reach; equals `zeta` exactly under `leakance_losing_only: true` |
| `depth_mean` | eval-window mean routed depth | m | **start-of-step**, computed from $Q_t$ |
| `area_z_mean` | eval-window mean $\mathcal{A}_z$ | labelled m², actually $\mathrm{m}^{(2-q)q+1}$ | see §8.3 |
| `q_mean` | eval-window mean routed discharge | m³/s | **end-of-step** $Q_{t+1}$, so offset one timestep from `depth_mean` |
| `COMID_eval` | reach ids | - | the **eval network** (gauge-subgraph union), NOT full CONUS |

The step offset between `depth_mean`/`area_z_mean` (start-of-step) and `q_mean`
(end-of-step) is documented in `write_zeta_netcdf`'s docstring and matters if
the analyst ratios them. The `zeta` value written is recomputed from the *saved*
primitives with the same `losing_only` flag and the same impervious mask the
forward used, so it is exactly what was subtracted from $b$, not an
independent estimate.

Two other diagnostics worth naming because they are easy to misread:

- **`negative solves`** (`src/routing/mmc_op.rs::negative_solve_stats`): the
  count of solve outputs that were negative *before* clamp C10 rewrote them to
  $+10^{-4}$, over all reaches and timesteps in one `forward`. A host readback,
  gated on `track_negative_discharge` (on in training, off in eval). On the
  CUDA-graph path the counters stay at zero because
  `forward_chain_inner` is never entered, so `forward` prints
  `UNAVAILABLE` rather than a zero, deliberately, so silence cannot be
  misread as a measurement.
- **The summed-Q' baseline** (`src/baseline/`): per-gauge sum of upstream
  divide $Q_r$ over the eval window against USGS daily observations. No
  routing, no learned parameters. It scores the **same** gauge population
  training does: `valid_gauges` skips gauges with no subgraph, single-divide
  headwater gauges, and gauges with no observation series. The headwater skip
  moved the baseline median NSE from 0.142 to 0.315; never compare a trained
  median against a baseline computed over a different gauge population.

---

## 12. Code versus documentation discrepancies found

Ordered by how much they could distort a physical-meaningfulness analysis.

### 12.1 `area_z` is dimensionally not an area (code-versus-code, undocumented)

`src/routing/leakance.rs::zeta_forward` computes
$w_z = (p\,d)^{q_\varepsilon}$, whereas the top width everywhere else is
$T = p\,d^{q_\varepsilon}$. Since $[p] = \mathrm{m}^{1-q}$, the plan-view
"area" $\mathcal{A}_z = w_z L$ carries units
$\mathrm{m}^{(2-q)q + 1}$, which is m² only at $q = 1$ (see equation 71); at
the prescribed $q = 0.65$ it is $\mathrm{m}^{1.878}$. Consequently $\zeta$ is
not in m³/s and a learned $K_D$ absorbs a $q$- and $p$-dependent unit
conversion. `src/dump_parameters.rs::write_zeta_netcdf` nevertheless labels
`area_z_mean` with `units = "m^2"`. Nothing in the source flags this; it is
ported verbatim from DDR `_compute_zeta`. **A learned $K_D$ should not be
compared numerically against a measured streambed leakance without redoing
this algebra.**

### 12.2 A learnable `x_storage` receives no gradient, but the code says it does

`src/training/forward.rs::forward` denormalizes an `x_storage` head output when
the KAN emits it, with the comment "gradient already flows via the custom
sparse backward in mmc_op.rs". It does not.
`src/routing/mmc_op.rs::timestep_forward` registers exactly five parents
(`n`, `q_spatial`, `p_spatial`, `q_t`, `q_prime_t`), six with `gamma`, eight
with leakance; `x_storage` is **not** among them in any variant. Its node is
extracted (`xst_aut`) and its primitive saved into `TimestepState::x_storage`,
but the backward's own comment states `state.x_storage` "is deliberately never
read here". Doubly inert under the default physics, because with
`ddr_match: false` the S19 Cunge $X$ ignores `xst_in` entirely. Nothing in
`src/config.rs` validates or rejects `x_storage` in
`kan_head.learnable_parameters`, so such a config trains a dead output
silently. Contrast this with the learned-`gamma` case, which is explicitly
rejected alongside `use_leakance` precisely because "gamma would silently
receive no gradient".

### 12.3 `.claude/ARCHITECTURE.md` describes a solver that no longer exists

Three claims in the "Per-timestep dataflow" and "Why dense forward
substitution, not sparse CSR + custom autograd" sections are now false:

1. It states the solve is a **dense** `triangular_solve_lower` with autograd
   falling out of BURN. The production path is sparse CSR with a hand-written
   `Backward` (`src/sparse/mod.rs::CsrSolveOp`), which `CLAUDE.md` invariant 4
   protects by name. `src/routing/utils.rs::triangular_solve_lower` still
   exists but is only reachable from the reference/test path.
2. It states `celerity c = clamp(v, v_lb, 15) * 5/3` unconditionally. That is
   the deprecated `ddr_match: true` branch; the default since 2026-08-19 is the
   trapezoidal $\beta$ of equation 35.
3. Its dataflow diagram shows `x = 0.3` implicitly via `x_storage`, with no
   mention of the Cunge $X$ (equation 48), the positivity clamps (50)-(51), or
   stage-dependent roughness (31)-(33).

The module-map table and the SP-8/9/10 sections are still accurate.

### 12.4 `docs/book/algorithm.md` presents only the deprecated physics

Read in this worktree, `docs/book/algorithm.md` states the Leopold and Maddock
coefficient and exponent **correctly** (line 9: "width **exponent**
`q_spatial` and width **coefficient** `p_spatial`", with
`top_width = p * depth^q`), as do `docs/book/architecture.md` and
`docs/book/usage/inputs-formatting.md`. The prompt for this document reported
that page as having them backwards; that is not the case at HEAD `289163b`, so
either it was already fixed or the error is elsewhere. What that page **does**
get wrong is the physics branch: it presents $c = \tfrac53 v$ as *the* celerity
and $X \in [0, 0.5]$ as a supplied constant, with no Cunge derivation, no
positivity clamp, and no stage roughness. Its depth formula (equation 29 here)
and its $c_1..c_4$ block are correct for $\gamma = 0$.

### 12.5 The prompt's framing of "coefficient clamping" does not match the code

There is **no** clamping of $c_1, c_2, c_3, c_4$ anywhere in `src/routing/`.
`params.enforce_positivity` clamps the *inputs* $K$ and $X$ (equations 50-51),
and its own config docstring explains why: because $c_1+c_2+c_3 = 1$ holds for
any $(K, X)$, clamping inputs preserves mass exactly while clamping $c_3$ would
not. The clamp that **does** break mass is the post-solve floor
$Q_{t+1} = \max(x_{\mathrm{sol}}, 10^{-4})$ in
`src/routing/mmc_op.rs::forward_chain_inner` (equation 52), which the
`NEG_SOLVES` docstring describes as one that "both CREATES MASS and removes the
only symptom". See §6.5 for the accurate statement and the measured rates.

### 12.6 `src/geometry.rs` says "invert Manning's equation for a trapezoidal section"

The closed form it then computes (equations 16-20) is the inversion of the
**power-law** section with $R \approx d$, not of the trapezoid that equations
24-26 subsequently build; the two areas differ by $(2-q)(q+1)/2$, which is
1.114 at $q = 0.65$ (equation 30). `tests/stage_roughness.rs` measures the
related $\mathrm{d}A/\mathrm{d}d$ gap and calls it "up to 11 %", so the fact is
known in the test suite but the geometry docstring still reads as if the
inversion were self-consistent with the trapezoid.

### 12.7 `params.tau` default is stale relative to its own convention

`src/config.rs::Params::default` sets `tau: 3`. Under the 2026-08-08 convention
documented on `src/training/loss.rs::tau_trim_and_downsample`, the legacy
shipped value 3 maps to $\tau_{\mathrm{new}} = -8$ (equation 82), which a `u32`
asserted into $[0, 24)$ cannot express. So the default is not the migrated
legacy behaviour, and it is not the measured optimum either (legacy 20, new 9).
Only `config/experiments/gridded_conus.yaml` carries the migrated `tau: 9`;
`config/merit_training.yaml` sets no `tau` and therefore inherits 3.

### 12.8 `experiment.learning_rate` is documented as inert for AdaDelta but is applied

Recorded in the current experiment config's own commentary, and worth repeating
because it invalidated a whole run:
`src/config.rs::OptimizerKind::Adadelta` documents the key as ignored
("Ignores the `learning_rate` schedule by design"), but the driver passes `lr`
to `optimizer.step` and `src/training/adadelta.rs` multiplies delta by it. The
2026-07-31 AdaDelta run therefore ran every update about 1000x scaled down and
moved the head by $L_2(\Delta w) = 1.9\times10^{-4}$, i.e. it never left
initialization while reporting plausible losses. dHBV intends AdaDelta at
$lr = 1.0$. **Verified for this document:**
`src/training/driver.rs` passes `lr as f64` into `optimizer.step` on both the
single-batch and accumulation paths, and
`src/training/adadelta.rs::AdaDelta` applies
`tensor - delta.mul_scalar(lr as f32)`. So the schedule is applied, the
docstring is wrong, and an AdaDelta run must set
`learning_rate: {1: 1.0}` explicitly.

### 12.9 `src/routing/mmc.rs::calculate_muskingum_coefficients` is a second, unused copy

It is term-for-term identical to S20-S23 in `forward_chain_inner` and is what
`docs/book/algorithm.md` cites, but the production per-timestep path never
calls it (`route_timestep` dispatches straight into `mmc_op`). A reader who
edits one and not the other gets no test failure from the
`compare_ddr_sandbox` gate, which exercises the `mmc_op` path.

### 12.10 Minor: the loss family has five members, not four

`src/config.rs::LossKind` is `L1 | NnseKge | Kge | NseBatch | NseBatchDeriv`.
`CLAUDE.md`'s "Training objective" section lists four (`l1`, `nnse-kge`,
`kge`, `nse-batch`) and omits `nse-batch-deriv`, which is the objective the
landscape-curvature work introduced and which carries its own
`loss.deriv_weight` and a second per-gauge statistic (`gauge_obs_diff_std`).

---

## 13. What I could not determine from the code

Labelled gaps, not guesses.

1. **Whether the learned $n_0$ is being read at a consistent reference depth
   across published runs.** Equation 31 makes $n_0$ the roughness *at*
   $d = d_{\mathrm{ref}}$, and `validate_stage_roughness` forces
   $d_{\mathrm{ref}} = 1.0$ m for a global $\gamma$. For a *learned* per-reach
   $\gamma$ the restriction lifts, and I did not find where (or whether)
   `d_ref` is then recorded in the run manifest. If two runs used different
   `d_ref`, their `n` fields are not comparable, and I could not confirm the
   manifest captures it.

2. **The `mask` field of `rskan::KanLayer`.** The fixture loader reads
   `block_{b}_mask` and the forward multiplies by it (equation 4), but I did not
   read `rskan`'s init or its `Module` derive closely enough to say whether
   `mask` is a trained `Param` or a frozen buffer initialized to ones. It is
   listed as "learned/fixed" in the §2.2 table for that reason. This matters
   only if an analyst wants the head's exact trainable-parameter count.

3. **The B-spline basis and grid update.** Equation 4 cites
   $B_{j,k}(x)$ abstractly. I read `rskan/src/layer.rs::forward` and
   the call signature of its `coef2curve` helper, but not
   `rskan/src/spline.rs`, so I cannot state
   the knot placement, the extension rule outside `grid_range`, or whether the
   grid is ever re-fit during training. The `kan_grid_range` docstring in
   `src/nn/kan_head.rs` says that outside the grid "the spline term flattens
   and only the `scale_base * SiLU` path survives", which implies a fixed grid
   with flat extrapolation, but I did not verify it in `spline.rs`.

4. **Which `params.stage_roughness` / learned-`gamma` configuration produced
   any specific published number.** The current arm
   (`config/experiments/sr_n0_leakance_gated.yaml`) learns
   `[n, K_D, d_gw, leakance_factor]` with no `stage_roughness` block, so
   $\gamma = 0$ in it despite the filename prefix `sr_`. I did not audit
   `.ddrs/runs/*/config.yaml` snapshots, so I cannot say which recorded results
   had stage roughness active.

5. **The disaggregation head's equations.** `src/nn/disagg_head.rs` implements
   the precip-conditioned mass-preserving daily-to-hourly forcing head and is a
   learned component of the model, but it was outside the ten requested topics
   and I did not read it. In the current arm it is present with
   `enabled: false`, so the forcing is flat repeat-24; anyone analysing an arm
   with `disaggregation.enabled: true` needs that file documented separately.
   The one relevant fact I can state from `src/training/forward.rs` is that
   when the head is attached it replaces `tensors.q_prime` wholesale and its
   output feeds $q'_t$ in equation 53.

6. **The measured CONUS distributions of $d$, $T$, $z$, $b$, and how often each
   clamp in §4.4 binds.** Only three such numbers are recorded in source or
   tests: the $X$ cap binds on 95.3% of reach-timesteps under
   `enforce_positivity`, 69.8% of reaches violate window (47) at $X = 0.3$, and
   `n_at_floor` reached 47.3% at $lr = 0.01$. I found no in-tree measurement of
   how often C3 (depth floor), C4 (side-slope band), C5 (bottom-width floor) or
   C6 (velocity band) saturate. Since each saturation is a gradient sink, that
   is the measurement most worth taking before claiming any learned parameter
   is identified.

7. **Whether `q_prime` is ever a tracked parent outside the adjoint study.**
   `ParentMask::q_prime_t` exists and B25 computes $\bar{q}'_t = c_4\bar{b}$
   behind it, and the code comments say "only the adjoint study lifts it" as a
   leaf. I did not read `src/experiment/adjoint/` to confirm what it does with
   that gradient, so the adjoint influence map's exact definition is outside
   this document.

8. **The DDR-side reference for `_compute_zeta`.** I cite
   `src/routing/leakance.rs`'s own header
   (`~/projects/ddr/src/ddr/routing/mmc.py:146-197`, commit `c2bd0f9`) for the
   claim that equation 65's $(p\,d)^{q}$ form is inherited verbatim. I did not
   open the DDR tree in this worktree, so the port fidelity of that one
   expression is asserted on the ddrs docstring, not independently checked.

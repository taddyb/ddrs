# Zeta gradient-sensitivity probe — experiment report (2026-07-03)

Spec: `docs/superpowers/specs/2026-07-02-zeta-gradient-probe-design.md`
Plan: `docs/superpowers/plans/2026-07-02-zeta-gradient-probe.md`
Prior experiment: `docs/2026-07-02-leakance-diagnosis-findings.md` (whose §5
item 1 this executes, reshaped from a twin experiment into two no-training
instruments after the recoverability discussion).
Code: `src/bin/probe_zeta_gradient.rs`, `src/training/probe.rs`,
`scripts/zeta_probe_sites.py`, `scripts/zeta_gradient_analysis.py`,
`scripts/notebooks/gradient_maps.ipynb`.

**One-line answer: the leakance gradient is alive everywhere (starvation
REFUTED — gauged/ungauged |g| ratio only 1.5–2.9 vs the ≥10 bar), and the
reason learned zeta stays small is now measured, not inferred: a
literature-magnitude reach loss (0.01 m³/s) arrives at its measurement gauge
at ~95% fidelity but is 53× smaller than the median reference gauge's 5%
discharge-uncertainty band — detectability NO-GO (4.2% of Ref probes
detectable). Gauge-only discharge supervision cannot see real-world leakance;
auxiliary spatial supervision is the only viable path.**

---

## 1. Hypotheses

The low-zeta diagnosis (2026-07-02) attributed the tiny learned leakance flux
(median |zeta| 6.4e-4 m³/s) partly to "H4: gauge bias / gradient starvation" —
but its evidence was correlational (zeta is small where gauges aren't). Two
rival mechanisms fit that correlation and demand different remedies, plus a
third hypothesis every gauge-trained rescue silently depends on:

| # | Hypothesis | Prediction if true | Remedy if true |
|---|---|---|---|
| P1 | **Starvation** — `∂Loss/∂zeta` is effectively dead away from gauges; the optimizer never receives a signal there | median \|∂L/∂factor\| on ungauged (and arid) reaches ≥ 1–2 orders of magnitude below gauged, at BOTH parameter points | auxiliary supervision fills a genuine gap |
| P2 | **Rejection** — the gradient is alive off-gauge but consistently points toward smaller zeta; the objective actively dislikes leakance there | off-gauge magnitudes comparable, but > 67% of dry-tercile signed grads push zeta down at the trained point | the objective (or physics) is mis-specified; auxiliary supervision would fight the loss |
| P3 | **Detectability** — a real-magnitude reach loss produces a gauge signal large enough for ANY discharge objective to reward | ≥ 10% of probes planted on reference-quality basins at δ = 0.01 m³/s clear both the rerun-noise floor and the 5% observational band | if false, no gauge-only objective can learn leakance and auxiliary supervision is *forced*, not optional |

A recover-planted-zeta twin experiment was considered and rejected: the
parameter triple `(K_D, d_gw, factor)` is internally degenerate (only the
flux is identifiable), routing parameters absorb planted losses (the
diagnosis's H5 finding), and training stochasticity smears the endpoint. The
hypotheses above are instead testable with **zero training** — pre-registered
thresholds, spec §Analysis.

## 2. What was changed to test them

All changes are read-only with respect to training; no autograd Backward
impl was touched (repo invariant 4), and the DDR-parity regression stayed at
ABSOLUTE MATCH throughout.

1. **`src/training/probe.rs`** (new): `lift_leaf` — detach a KAN head output
   from its graph and re-register it as a `require_grad` autograd leaf
   (values bit-identical, tape topology only); `probe_forward` — a faithful
   mirror of the training `forward` whose ONLY difference is that the three
   per-reach leakance vectors pass through `lift_leaf`, so
   `loss.backward()` yields exact per-reach gradients via the existing
   analytical `TimestepLeakanceOp` backward; `GradAccum` — COMID-keyed
   accumulation across batches (batches route different gauge-subgraph
   unions).
2. **`src/bin/probe_zeta_gradient.rs`** (new binary), two modes:
   - `--mode grad`: replicates the training mini-batch loop exactly
     (sampler, rho-windows, obs alignment, NaN gauge filter, L1 loss) minus
     the optimizer step; accumulates per-reach `Σ|g|`/`Σg`/coverage; writes
     `write_grad_netcdf` (new, `src/dump_parameters.rs`). Fail-fast guards:
     non-finite-gradient abort, COMID key-set alignment asserts, retry cap.
   - `--mode perturb`: forward-only chunked eval (replica of `evaluate`)
     with `+δ` m³/s added to the lateral-inflow forcing (`q_prime` AND
     `q_prime_daily`, so the disaggregation head carries it) at planned
     reaches; validates the plan against the network up front; writes
     per-round daily gauge predictions.
   - `--backend {cpu,cuda}` (cpu default, forces `sparse_solver: cpu`) —
     the GPU was occupied by another training job, so the entire experiment
     ran on CPU (`NdArray<f32>`), which is deterministic: the two
     unperturbed baselines came out byte-identical (`max|b1−b2| = 0`),
     reducing detection to the observational band alone.
3. **`scripts/zeta_probe_sites.py`** (new): GAGES-II `CLASS` join (the
   dam/lake control — probes measured only at least-disturbed `Ref` gauges;
   `Non-ref` kept as a labeled contrast), stratified sampling
   (upstream-area × aridity × stage-1-reachability terciles), and round
   packing. **Methods evolution, recorded:** the initial "no two probes
   share any gauge" constraint stalled at 64 rounds (median 1 probe/round);
   the fix was (a) conflicts = *nearest-gauge contamination only* (the
   analysis measures each probe solely at its nearest gauge) and (b) a cap
   of 2 reaches per measurement gauge (per-gauge count × deltas is a hard
   round floor) → **8 rounds**.
4. **`scripts/zeta_gradient_analysis.py`** (new): the pre-registered
   verdicts; **`scripts/notebooks/gradient_maps.ipynb`** (new): CONUS
   per-gauge (4-panel trained/cold × magnitude/sign) and per-reach maps per
   the `ddrs-eval-plots` conventions.
5. **Tests** (`tests/zeta_gradient_probe.rs`): leaf-lifting leaves the
   routed forward byte-identical; leaf grads finite and nonzero on a losing
   chain; COMID accumulation exact. Guard suites (`leakance_gradcheck` 8/8,
   `leakance_off_parity` 3/3, `zeta_accum` 6/6, lib 172) green throughout.

## 3. The experiment

**Stage 1 — adjoint reachability map (tests P1, P2).** 96 training-style
windows (seed 42; a timing gate measured 23 s/window on CPU, so N was raised
from the plan's 32 after the first pass left median coverage at 1 window),
covering the **full 64,892-reach eval network** with median coverage 4.
Run twice with identical windows: the trained hourly-ON checkpoint
(`2026-07-01T13-43-32Z…/epoch_5_mb_9`) and the cold seed-42 head init — the
pair separates "converged-so-flat" from "never-saw-a-signal". Gradients are
w.r.t. the NORMALIZED [0,1] parameters (denormalization happens inside
`setup_inputs`), hence dimensionless and spatially comparable.

**Stage 2 — detectability bound (tests P3).** 292 probes = 146 reaches
(96 Ref-only) × 2 deltas on 104 measurement gauges, 8 rounds + 2 baselines,
each a forward-only eval over the first 3 years (`--eval-days 1095`) of the
eval window. Detection criterion per probe, at its nearest downstream gauge:
`|mean ΔQ|` > 99th-pct rerun noise (= 0 on CPU) AND > 5% of the gauge's mean
flow (the differential-gauging detectability band, McCallum 2012).

## 4. Did the tests pass or fail?

| # | Hypothesis | Pre-registered bar | Measured | Verdict |
|---|---|---|---|---|
| P1 | Starvation | gauged/ungauged \|g\| ≥ 10× at both points | **1.5×** (trained), **2.9×** (cold) | **FAILED (refuted)** |
| P2 | Rejection (trained point) | >67% dry-tercile push-down | **52.5%** (≈ neutral) | **FAILED (refuted)** |
| P3 | Detectability | ≥10% of Ref δ=0.01 probes detectable | **4.2%** (4/96); δ=0.1: 16.7%; Non-ref: 0.0%/2.1% | **FAILED (NO-GO)** |
| — | Cross-check (stage-1 ↔ stage-2, ρ > 0.3) | — | ρ = **−0.02** | SUSPECT — explained below |

**P1 failed — the gradient reaches everywhere.** The eval network's gauge
coverage is dense enough that autograd carries loss signal to essentially
every reach; ungauged reaches see gradients only 1.5–2.9× weaker. The
"gradient starvation" reading of the diagnosis's H4 is dead.

**P2 failed at the trained point but revealed the discovered mechanism
(post-hoc, labeled as such): initial-signal rejection.** At the cold point,
**80.5% of ungauged and 66.2% of dry-tercile gradients push zeta DOWN**,
with a **15.9× wet/dry gradient-magnitude asymmetry** (trained: 1.4×);
per-basin the median cold sign-fraction is **0.98** (the CONUS map's cold
panels are near-uniformly "push down"). Early training actively suppressed
leakance almost everywhere before converging to a balanced field. The
**trained** sign map is regionally coherent: "wants MORE leakance"
(∂L/∂factor < 0) clusters in the interior West / High Plains — physically
the right losing-stream country — with "wants less" persisting along the
East/Gulf. The term differentiated regions correctly; it simply cannot be
rewarded further.

**P3 failed — and the failure decomposes cleanly.** The planted flux is NOT
lost in routing: the median Ref probe's `|mean ΔQ|/δ = 0.946` — the gauge
receives ~95% of the planted loss. But the median Ref measurement gauge's 5%
band is **0.53 m³/s — 53× the 0.01 m³/s literature-magnitude loss**. The 4
detections all sat on tiny gauges (median band 7e-4 m³/s). Even δ = 0.1 (an
extreme upper-literature loss) clears the band at only 16.7% of Ref gauges;
regulated (Non-ref) gauges are worse (0–2%) because they sit on larger
rivers. **Detection fails on dilution, not transmission.**

**Cross-check post-mortem (ρ = −0.02).** Explainable and methodological:
`|∂L/∂factor|` embeds the reach's local flux capacity
(`area_z·K_D·(depth−d_gw)`), whereas detectability of a FIXED δ is pure
hydraulic dilution at the measurement gauge. The two instruments answer
different questions by construction; a comparable check would use
capacity-normalized `∂L/∂zeta` against band-normalized ΔQ. Recorded as a
design improvement, not rerun — the NO-GO does not depend on it. Also
noted: 4/292 probes had no matching gauge row (their nearest gauge fell to
the eval network's DA_VALID/headwater filters) — excluded, immaterial.

CONUS maps: `output/zeta_probe/plots/gradient_gauge_map.png`,
`gradient_reach_map.png` (method notebook committed at
`scripts/notebooks/gradient_maps.ipynb`).

## 5. Conclusions

1. **Gauge-only discharge supervision cannot learn real-world leakance —
   measured, not argued.** Real-magnitude losses transmit to gauges nearly
   intact but land 1–2 orders of magnitude below the observational
   detectability band at >95% of even reference-quality gauges. No objective
   computed from gauged discharge alone (L1, NSE, KGE, or otherwise) can
   reward what it cannot distinguish from measurement uncertainty.
2. **The diagnosis's H4 is re-mechanized.** Not gradient starvation
   (gradients reach everywhere) but signal starvation at the sensor — plus
   an initial-training regime in which the objective pushed leakance down in
   ~98% of basins before equilibrating.
3. **The physics term is healthy.** At convergence the gradient field wants
   more leakance exactly where losing streams live. The parameterization is
   not the obstacle; the supervision is.
4. **The dam/lake filter mattered less than expected for this question**
   (Non-ref probes were undetectable for river-size reasons before
   regulation noise even enters) but remains mandatory for any future
   real-data differential-gauging validation.

## 6. Next steps

1. **Auxiliary-constraint experiment** (top follow-up, now empirically
   forced rather than literature-recommended): regularize `zeta_net` (or
   `d_gw`) toward an independent losing-potential signal (Jasechko-style
   well-vs-stream levels, water-table-depth attributes); hourly forcing;
   evaluate whether the 2×2's marginal losing-subset gain (+0.0018 KGE)
   grows while zeta magnitudes move toward literature values.
2. **Keep leakance hourly-gated and experimental** — nothing here justifies
   promotion; the NO-GO sharpens the documentation of why.
3. Optional methodological cleanup if the probe is reused:
   capacity-normalized reachability (`∂L/∂zeta`) for a meaningful
   stage-1↔stage-2 cross-check; band-normalized detectability scores; a
   `band10` column alongside `band5` in `detectability_rows.csv` (the spec
   listed both; only the 5% band feeds the pre-registered verdict).

## 7. Raw verdict output

```
aridity-vs-meanP spearman -0.84 → aridity is a DRYNESS index; dry tercile n=21415

========================================================================
Stage 1 — |dL/dfactor| by stratum, trained vs cold
========================================================================
trained  gauged=6.038e-05 ungauged=3.997e-05 (ratio 1.5) | dry=3.238e-05 wet=4.494e-05 (wet/dry 1.4)
  trained/gauged: frac pushing zeta DOWN = 30.1%
  trained/ungauged: frac pushing zeta DOWN = 52.1%
  trained/dry: frac pushing zeta DOWN = 52.5%
cold     gauged=3.924e-04 ungauged=1.364e-04 (ratio 2.9) | dry=1.836e-05 wet=2.921e-04 (wet/dry 15.9)
  cold/gauged: frac pushing zeta DOWN = 55.9%
  cold/ungauged: frac pushing zeta DOWN = 80.5%
  cold/dry: frac pushing zeta DOWN = 66.2%

========================================================================
Stage 2 — planted-delta detectability at nearest gauges
========================================================================
baseline determinism: max |b1-b2| = 0.0
WARNING: 4 probes had no matching gauge row — investigate
Non-ref  delta=0.01: detectable 0.0% of 48
Non-ref  delta=0.1: detectable 2.1% of 48
Ref      delta=0.01: detectable 4.2% of 96
Ref      delta=0.1: detectable 16.7% of 96

cross-check: spearman(reachability, detected) = -0.02
per-probe rows → /home/tbindas/projects/ddrs/output/zeta_probe/detectability_rows.csv

========================================================================
VERDICTS
========================================================================
  [REFUTED] H4-starvation: gauged/ungauged |g| ratio trained=1.5, cold=2.9 (bar: >=10 at both points)
  [REFUTED] H4-rejection: 52.5% of dry-tercile grads push zeta down
  [NO-GO] Detectability: 4.2% of Ref probes at delta=0.01 detectable (NO-GO bar: <10%)
  [SUSPECT] Cross-check: rank corr -0.02 (bar: > 0.3)
```

Supporting stats (recomputed from `detectability_rows.csv` by the final
review): Ref median 5%-band 0.531 m³/s; Ref median |mean ΔQ|/δ at δ=0.01 =
0.946; the 4 detections' median band 7e-4 m³/s. Timing figures: 23 s/window
is the timing-gate measurement (`--windows 1`); full N=96 runs averaged
~20 s/window.

## 8. Reproduce

```bash
# Stage 1 (CPU, ~35 min per run; identical seeds ⇒ identical windows)
cd /home/tbindas/projects/ddrs
WT=<checkout with the probe binary>
nice -n 10 $WT/target/release/probe_zeta_gradient \
  --config config/experiments/leakance_hourly_on.yaml \
  --checkpoint .ddrs/runs/2026-07-01T13-43-32Z-train-and-test/checkpoints/epoch_5_mb_9 \
  --windows 96 --seed 42 --output output/zeta_probe/grad_trained.nc
nice -n 10 $WT/target/release/probe_zeta_gradient \
  --config config/experiments/leakance_hourly_on.yaml \
  --windows 96 --seed 42 --output output/zeta_probe/grad_cold.nc

# Sites + packing (ddrs-py venv)
cd ddrs-py && uv run python ../scripts/zeta_probe_sites.py

# Stage 2 (CPU, 8 rounds + 2 baselines, ~overnight)
nice -n 10 $WT/target/release/probe_zeta_gradient \
  --config config/experiments/leakance_hourly_on.yaml \
  --checkpoint .ddrs/runs/2026-07-01T13-43-32Z-train-and-test/checkpoints/epoch_5_mb_9 \
  --mode perturb --probe-plan output/zeta_probe/probe_plan.csv \
  --eval-days 1095 --output output/zeta_probe/perturb

# Verdicts + maps
cd ~/projects/ddr && uv run python <ddrs>/scripts/zeta_gradient_analysis.py
# notebook: scripts/notebooks/gradient_maps.ipynb (execute from ddrs-py)
```

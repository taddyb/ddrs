# Zeta gradient-sensitivity probe — experiment report (2026-07-03)

Spec: `docs/superpowers/specs/2026-07-02-zeta-gradient-probe-design.md`
Plan: `docs/superpowers/plans/2026-07-02-zeta-gradient-probe.md`
Prior experiment: `docs/2026-07-02-leakance-diagnosis-findings.md` (whose §5
item 1 this executes, reshaped from a twin experiment into two no-training
instruments after the recoverability discussion).
Scripts: `src/bin/probe_zeta_gradient.rs`, `scripts/zeta_probe_sites.py`,
`scripts/zeta_gradient_analysis.py`; maps notebook
`scripts/notebooks/gradient_maps.ipynb`.

**One-line answer: the leakance gradient is alive everywhere (H4-starvation
REFUTED — gauged/ungauged |g| ratio 1.5–2.9 vs the ≥10 bar), and the reason
learned zeta stays small is now measured, not inferred: a literature-magnitude
reach loss (0.01 m³/s) arrives at its measurement gauge at ~95% fidelity but
is 53× smaller than the median reference gauge's 5% discharge-uncertainty
band — detectability NO-GO (4.2% of Ref probes detectable). Gauge-only
discharge supervision cannot see real-world leakance; auxiliary spatial
supervision is the only viable path.**

---

## 1. Motivating question and the two rival mechanisms

The low-zeta diagnosis (2026-07-02) attributed the tiny learned leakance flux
to driving-head throttling plus a gauge-shaped training signal — but its H4
evidence was correlational (zeta is small where gauges aren't). Two rival
mechanisms fit that pattern and demand different remedies:

- **Starvation** — `∂Loss/∂zeta` is effectively dead away from gauges; the
  optimizer never receives a signal there. Remedy: auxiliary supervision.
- **Rejection** — the gradient is alive off-gauge but points toward smaller
  zeta; the discharge objective actively dislikes leakance there. Remedy: the
  objective (or physics) is mis-specified; auxiliary supervision would fight
  the loss.

A recover-planted-zeta twin experiment cannot separate these (parameter
degeneracy, routing-parameter absorption, training stochasticity), so the
probe uses two **no-training** instruments with pre-registered verdicts.

## 2. Methods

**Stage 1 — adjoint reachability map.** A new binary
(`probe_zeta_gradient --mode grad`) replicates the training mini-batch
distribution exactly (64-gauge batches, rho=90-day windows, the run's L1
objective, seed-fixed sampler) but takes no optimizer step: the KAN head's
three per-reach leakance outputs are detached and re-lifted as autograd
leaves (`lift_leaf`; the analytical `TimestepLeakanceOp` backward supplies
exact gradients — no Backward impl touched, guard suites + DDR-sandbox
ABSOLUTE MATCH all green), and per-batch leaf gradients are accumulated by
COMID. N=96 windows covered the **full 64,892-reach eval network** (median
coverage 4 windows/reach; first pass at N=32 had median 1 and was re-run —
each window costs only 23 s on CPU). Run twice with identical windows:
the **trained** hourly-ON checkpoint (`epoch_5_mb_9`) and the **cold**
seed-42 initialization — the pair separates "converged-so-flat" from
"never-saw-a-signal". Gradients are w.r.t. the NORMALIZED [0,1] parameters
(denormalization happens inside `setup_inputs`), so magnitudes are
dimensionless and spatially comparable within a parameter.

**Stage 2 — detectability bound.** Forward-only (no autograd): plant a
constant `+δ` m³/s of lateral inflow (`q_prime` + `q_prime_daily`, so the
disaggregation head carries it) at selected reaches and measure ΔQ at each
probe's nearest downstream gauge over the first 3 years of the eval window
(`--eval-days 1095`), δ ∈ {0.01, 0.1} (the Shanafield & Cook
transmission-loss range). **Site selection with the dam/lake control**:
gauges joined to GAGES-II `CLASS` (point-shapefile DBF); the primary
population is reaches whose *every* containing gauge is `Ref`
(least-disturbed); Non-ref kept as a labeled contrast. 146 reaches
(96 Ref-only) on 104 measurement gauges, stratified by upstream-area ×
aridity × stage-1-reachability terciles, capped at 2 reaches per measurement
gauge. **Packing lesson (methods evolution, recorded):** the initial
"no two probes share any gauge" constraint stalled at 64 rounds (median 1
probe/round) because probes sharing a measurement gauge can never share a
round; relaxing conflicts to *nearest-gauge contamination only* and capping
per-gauge probes collapsed the plan to **8 rounds** (292 probes) + 2
baselines. Everything ran on **CPU** (GPU occupied by another training job;
`NdArray<f32>`, `sparse_solver` forced to `cpu`): the two unperturbed
baselines came out **byte-identical** (`max|b1−b2| = 0`), so the noise floor
is exactly zero and detection reduces to the observational-uncertainty band —
detection criterion: `|mean ΔQ|` > 99th-pct rerun noise (=0) AND > 5% of the
gauge's mean flow (the differential-gauging detectability band, McCallum
2012).

## 3. Results

| Verdict | Criterion (pre-registered) | Measured |
|---|---|---|
| **H4-starvation REFUTED** | gauged/ungauged median \|∂L/∂factor\| ≥ 10× at both points | ratio **1.5** (trained), **2.9** (cold) |
| **H4-rejection REFUTED** (trained point) | >67% of dry-tercile signed grads push zeta down with comparable magnitudes | **52.5%** (≈ converged neutrality) |
| **Detectability NO-GO** | <10% of Ref probes at δ=0.01 clear noise + 5% band | **4.2%** (4/96); δ=0.1: 16.7%; Non-ref: 0.0%/2.1% |
| **Cross-check SUSPECT** | stage-1 reachability rank-predicts detectability (ρ > 0.3) | ρ = **−0.02** (explained below) |

**The gradient reaches everywhere.** The eval network's gauge coverage is
dense enough that autograd carries loss signal to essentially every reach:
ungauged reaches see gradients only 1.5–2.9× weaker than gauged ones. The
diagnosis's "gradient starvation" reading of H4 is dead — zeta's
gauge-proximity pattern is not caused by missing gradients.

**Discovered mechanism (post-hoc, labeled as such): initial-signal
rejection.** At the cold point, **80.5% of ungauged and 66.2% of dry-tercile
gradients push zeta DOWN**, with a **15.9× wet/dry gradient-magnitude
asymmetry** (trained: 1.4×) — per-basin, the median sign-fraction is 0.98
(CONUS map, cold panels ~uniformly "push down"). Early training actively
suppressed leakance almost everywhere — hardest in wet, well-observed
basins — before converging to a balanced field. The **trained** sign map has
coherent regional structure: "wants MORE leakance" (∂L/∂factor < 0) clusters
in the interior West / High Plains — physically the right losing-stream
country — while "wants less" persists along the East/Gulf. The term is not
misbehaving at convergence; it differentiated regions correctly but at
magnitudes the objective cannot reward further.

**Why: the detectability floor (the headline number).** The planted flux is
NOT lost in routing — the median Ref probe's `|mean ΔQ|/δ = 0.946`: the
gauge receives ~95% of the planted loss. But the median Ref measurement
gauge's 5% band is **0.53 m³/s — 53× the 0.01 m³/s literature-magnitude
loss**. The 4 detected probes all sat on tiny gauges (median band
7e-4 m³/s). Even δ=0.1 (an extreme, upper-literature loss) clears the band
at only 16.7% of Ref gauges. Non-ref (regulated) gauges are worse (0–2%) —
they sit on larger rivers. **Detection fails on dilution, not transmission**:
the discharge objective is signal-starved not in gradient topology but in
signal-to-uncertainty at the gauges themselves.

**Cross-check post-mortem (ρ = −0.02, SUSPECT as pre-registered).** The
failure is explainable and methodological, not evidence against either
stage: `|∂L/∂factor|` embeds the reach's local flux capacity
(`area_z·K_D·(depth−d_gw)`), whereas detectability of a FIXED δ is pure
hydraulic dilution at the measurement gauge (band ∝ gauge mean flow). The
two instruments answer different questions by construction; a comparable
cross-check would use `∂L/∂zeta` (capacity-normalized) against a
band-normalized ΔQ. Recorded as a design improvement, not rerun (the NO-GO
does not depend on it). Also noted: 4/292 probes had no matching gauge row
in the eval outputs (their nearest gauge fell to the eval network's
DA_VALID/headwater filters) — excluded, immaterial at this n.

CONUS maps: `output/zeta_probe/plots/gradient_gauge_map.png` (4 panels:
trained/cold × log10|g|/sign-fraction), `gradient_reach_map.png` (per-reach
log10|∂L/∂factor| on MERIT flowlines). Notebook committed at
`scripts/notebooks/gradient_maps.ipynb`.

## 4. Conclusions

1. **Gauge-only discharge supervision cannot learn real-world leakance —
   measured, not argued.** Real-magnitude losses transmit to gauges nearly
   intact but land 1–2 orders of magnitude below the observational
   detectability band at >95% of even reference-quality gauges. No objective
   computed from gauged discharge alone (L1, NSE, KGE, or otherwise) can
   reward what it cannot distinguish from measurement uncertainty.
2. **The diagnosis's H4 is re-mechanized.** Not gradient starvation
   (gradients reach everywhere) but signal starvation at the sensor: the
   optimizer keeps zeta small because, within the detectable band, small is
   optimal — and at initialization the objective actively pushed leakance
   down nearly everywhere (0.98 of basins).
3. **The term itself is healthy.** At convergence the gradient field wants
   more leakance exactly in the High Plains / interior West and less in the
   humid East — the physically correct spatial reading. The physics
   parameterization is not the obstacle; the supervision is.
4. **The auxiliary-supervision remedy is now the only path** (diagnosis
   findings §5 item 2), upgraded from literature-recommended to empirically
   forced. Regularizing `d_gw`/`zeta_net` against an independent
   losing-potential signal (Jasechko-style well-vs-stream levels,
   water-table-depth attributes) supplies exactly the constraint the gauges
   cannot.
5. **The dam/lake filter mattered less than expected for this question**
   (Non-ref probes were undetectable for size reasons before regulation
   noise even enters), but remains mandatory for any future real-data
   differential-gauging validation.

## 5. Next steps

1. **Auxiliary-constraint experiment** (now the top follow-up): add a
   spatial regularizer on `zeta_net` (or `d_gw`) toward a losing-potential
   map; hourly forcing; evaluate whether the losing-subset skill gain from
   the 2×2 grows beyond its marginal +0.0018 KGE while zeta magnitudes move
   toward literature values.
2. **Keep leakance hourly-gated and experimental** — unchanged from the
   diagnosis; nothing here justifies promotion, and the NO-GO sharpens the
   documentation of *why*.
3. Optional methodological cleanup if the probe is reused: capacity-normalized
   reachability (`∂L/∂zeta`) for a meaningful stage-1↔stage-2 cross-check;
   band-normalized detectability scores.

## 6. Raw verdict output

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

Supporting stats (from `detectability_rows.csv`): Ref median 5%-band
0.531 m³/s; Ref median |mean ΔQ|/δ at δ=0.01 = 0.946; the 4 detections'
median band 7e-4 m³/s.

## 7. Reproduce

```bash
# Stage 1 (CPU, ~37 min per run; identical seeds ⇒ identical windows)
cd /home/tbindas/projects/ddrs
WT=<worktree or checkout with the probe binary>
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

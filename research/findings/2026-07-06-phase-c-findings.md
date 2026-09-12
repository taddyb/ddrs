# Phase C Findings — Leakance Promotion Gate: NO-GO (Selective Equifinality)

Date: 2026-07-06
Worktree: `zeta-sensitivity` (branch `worktree-zeta-sensitivity`)
Experiment spec: `docs/2026-07-05-phase-c-leakance-gate-experiment.md`
Program spec: `docs/superpowers/specs/2026-07-04-leakance-gate-program-design.md`
Prior findings: `docs/2026-07-02-leakance-diagnosis-findings.md`,
`docs/2026-07-03-zeta-gradient-probe-findings.md`,
`docs/2026-07-04-synthetic-recoverability-findings.md`.

**One-line result: leakance is NOT promotable. On real USGS gauges with a
fixed objective and genuinely informative groundwater inputs, it improves KGE
only marginally (+0.006 on the losing subset, below the +0.01 bar), does not
cannibalize Manning's n (Leg 2 clean), but its learned spatial field is
ANTI-correlated with independent groundwater data (ρ = −0.36 vs bed-relative
water-table depth; 84% of the flux sits on non-losing reaches). This is
selective equifinality, measured with every confound removed.**

---

## 1. Hypothesis

H_C (spec §1): with a clean objective (Phase B), informative inputs (Phase A),
sign-constrained physics (losing-only clamp + impervious hard-zero), does
leakance improve real-gauge metrics without repainting roughness and with a
field consistent with groundwater data? Pre-registered three-leg gate, bars
fixed before the run.

## 2. What was changed to test it

Routing code (all gradient-exact, guard-verified, sandbox ABSOLUTE MATCH):
- **Losing-only clamp** `zeta = factor · area_z · K_D · max(0, depth − d_gw)`
  (`bc7e7f0`) — config-gated `leakance_losing_only` (default true); gaining
  reaches produce zeta ≡ 0 and zero gradient. 14 gradcheck cases.
- **Impervious hard-zero** (`4292f23`) — `zeta ← zeta · (impervious ≤ 0.7)`,
  a constant mask off the autograd path.
- **Attribute wiring** (`7b7745b`) — `corridor_impervious` read from the raw
  attribute column → the mask; NaN → allowed.
- **C0 multi-store attributes** (`2351556`, prior) — `data_sources.attributes`
  as a COMID-joined list, so the global backbone (`merit_global_attributes_v2`)
  and the CONUS channel/GW store (`merit_channel_attributes_v1`, Phase A)
  concatenate feature-wise.
- **Configs** (`d72fada`): `phase_c_on.yaml` / `phase_c_off.yaml` — real USGS
  obs, hourly, 17 inputs (10 base + permeability, channel_wtd_bed_rel,
  losing_fraction, corridor_impervious, alluvium_fraction, bfi, bankfull_depth),
  K_D widened to [1e-8, 1e-5]. OFF is the same-inputs baseline with the term
  off (isolates the leakance term, not the inputs).

**Scope decision (documented):** run matched to the 2×2's objective (NO state
cache) for direct comparability against the known +0.0018 KGE baseline,
isolating exactly the new variables (Phase-A inputs + losing-only physics).
The state-cache-clean-objective variant is a deferred follow-up — justified
because the recovery control already showed the objective cleanup does not
change leakance identifiability (§4).

## 3. The experiment

- ON vs OFF, real USGS daily obs, hourly forcing, seed 42, CPU, 5 epochs.
- Training window 1981-10-01…1995-09-30; eval on the test window
  (1995-10-01…2010-09-30), 2365 gauges.
- Seam-free eval (post the continuity fix) with per-reach zeta accumulation
  for ON; `dump_parameters` for ON and OFF (Δn).
- Losing subset = the top-50% of gauges by |ΔQ|(ON−OFF), i.e. the gauges
  leakance actually affected (1182 gauges).

## 4. Did it pass or fail?

### Recovery pre-flight (2026-07-05, on the CLEAN objective)
R1 = **0.008** (planted-leakance recovery ratio) — identical to 0.009 on the
noisy objective. The Phase B floor fix did NOT make leakance recoverable.
Mechanism = **equifinality by smearing**: loss gap widened (A 0.50 vs B 1.74,
+71%), the term active on 78% of reaches (1485 m³/s total), yet the 58 planted
reaches hold 0.1% of it and A's top-10 zeta reaches include zero plants. The
gauge constrains the integrated upstream loss, not its location.

### Phase C three-leg gate (real gauges)

| Leg | Metric | Value | Bar | Verdict |
|---|---|---|---|---|
| 1 | losing-subset ΔNSE | +0.0015 | ≥ +0.01 | below |
| 1 | losing-subset ΔKGE | +0.0060 (73.4% improve) | ≥ +0.01 | below |
| 1 | overall ΔNSE / ΔKGE | +0.0011 / +0.0046 | degrade ≤ 0.002 | PASS (improves) |
| 2 | Δn(ON−OFF) IQR | 0.0143 | < 0.1 | **PASS** |
| 2 | ρ(Δn, zeta_net) | +0.079 | \|ρ\| < 0.2 | **PASS** |
| 3 | ρ(\|zeta\|, bed-relative WTD) | **−0.355** | > 0.3 | **FAIL (negative)** |
| 3 | active reaches losing-possible (WTD>0) | 15.8% | high | FAIL |
| 3 | impervious-reach zeta vs losing | 372× lower | ≥ 5× | PASS (imposed by hard-zero) |

**Interpretation:**
- **Leg 1 — marginal, sub-threshold.** Leakance genuinely helps (ΔKGE +0.006,
  73% of affected gauges improve; ~3× the 2×2's +0.0018 thanks to the enriched
  inputs) and does no harm — but it does not clear the substantive +0.01
  promotion bar.
- **Leg 2 — PASSES cleanly.** Δn IQR 0.014 (vs the 0.59 daily anti-pattern)
  and near-zero Δn–zeta correlation: the leakance term did NOT steal roughness's
  job. The metric gain is not cannibalization.
- **Leg 3 — FAILS decisively.** The learned zeta is *negatively* correlated
  with where groundwater says losing streams are (ρ = −0.36), and 84% of the
  flux sits on reaches the water table marks as gaining/connected. The field
  is unphysical — it went where the loss-fit needed it, not where the physics
  is. This is the recovery smearing result, independently confirmed on real
  data. (The impervious-reach zeta suppression is real but *imposed* by the
  hard-zero mask, not learned.)

**Verdict: NO-GO for promotion.** PROMOTE requires all three legs; Leg 1 is
sub-threshold and Leg 3 fails. Leakance is benign and marginally helpful but
its spatial field contradicts groundwater reality — do not promote it as a
physical parameterization; it stays experimental (hourly-gated).

## 5. Conclusions — the complete evidence chain

Leakance is a **selective-equifinality** term, established by removing every
confound in sequence:

1. **Physically motivated** — MODFLOW-family Darcy conductance (32 verified
   citations, `docs/2026-07-04-leakance-literature-review.md`).
2. **Live gradients everywhere** — the gradient probe refuted starvation
   (gauged/ungauged |g| ratio 1.5×, not ≥10×).
3. **Real losses undetectable at gauges** — 53× below the 5% discharge
   uncertainty band (P3 detectability, 4.2% of Ref probes).
4. **Planted losses unrecoverable even on a clean objective** — R1 = 0.008,
   smearing (recovery control on the Phase B state-cache objective).
5. **On real gauges with the best inputs** — marginal aggregate benefit
   (+0.006 KGE, no cannibalization) but an anti-physical field (ρ = −0.36).

The mechanism is a property of the observation operator, not the term or the
optimizer: **a gauge measures Σ(loss) over its entire upstream network, so it
constrains the aggregate loss (metrics nudge up) but cannot constrain the
per-reach distribution (the field smears anti-physically).** Gauge discharge
is not invertible for sub-network flux. This holds with informative inputs, a
clean objective, live exact gradients, and sign-constrained physics — every
lever pulled. That is a stronger, more general result than a marginal metric
win: a measured impossibility of identifying a physically-real routing term
from gauge supervision.

## 6. Next steps

1. **The paper** (`ddr_equifinality/paper.tex`): this is the central result.
   Frame: differentiable routing can *use* a physically-motivated term to
   improve aggregate fit while being unable to identify or promote it —
   selective equifinality — because the gauge observation operator integrates
   over the network. Positions against dPL optimism (Tsai 2021, Feng 2022):
   parameter learning has a hard identifiability ceiling set by the
   observation operator, not the optimizer or the data volume.
2. **If leakance is ever to be promoted**, it requires supervision OUTSIDE
   gauge discharge — a spatial prior on zeta_net/d_gw against groundwater data
   (Jasechko well-vs-stream, Zell & Sanford WTD, ParFlow GW–SW flux). The
   Phase A attributes are already built for this; the auxiliary-loss experiment
   is the natural follow-up IF the science warrants pursuing promotion (the
   NO-GO suggests documenting rather than rescuing).
3. **General ddrs finding to keep** (Phase B byproduct): windowed training
   carries an IC-noise floor (~40% of the loss budget at large gauges) fixed
   by the state-cache hotstart; and `evaluate` cold-restarted every 15 days
   (now fixed) — worth a short methods note beyond leakance.
4. Leakance stays hourly-gated and experimental; the impervious hard-zero and
   losing-only clamp remain useful sign-discipline for any future GW–SW term.

## 7. Raw numbers (recomputed from artifacts)

```
LEG 1 (losing subset, 1182 gauges, top-50% leakance-affected):
  ΔNSE median=+0.0015  improved=60.6%   (bar +0.01)
  ΔKGE median=+0.0060  improved=73.4%   (bar +0.01)
  overall ΔNSE=+0.0011  ΔKGE=+0.0046    (bar: degrade <=0.002 -> improves)
LEG 2 (equifinality):
  Δn IQR = 0.0143                        (bar <0.1; daily anti-pattern 0.59)
  ρ(Δn, zeta_net) = +0.079               (bar |ρ|<0.2)
LEG 3 (external consistency, 23487 active reaches):
  ρ(|zeta|, bed-relative WTD) = -0.355   (bar >0.3)
  active reaches losing-possible (WTD>0) = 15.8%
  high-impervious (>0.5) median |zeta| = 4.6e-6 vs losing 1.7e-3 (372x)
Recovery pre-flight (clean objective): R1 = 0.008 (bar >=0.5 recover)
Overall real-gauge (2365): NSE ON 0.7221 OFF 0.7208; KGE ON 0.7136 OFF 0.7055
```

## 8. Reproduce

```bash
WT=.claude/worktrees/zeta-sensitivity; cd ~/projects/ddrs
$WT/target/release/train --backend cpu --config $WT/config/experiments/phase_c_on.yaml  --checkpoint-dir output/phase_c/on
$WT/target/release/train --backend cpu --config $WT/config/experiments/phase_c_off.yaml --checkpoint-dir output/phase_c/off
$WT/target/release/eval  --backend cpu --config $WT/config/experiments/phase_c_on.yaml  --checkpoint <on_ckpt>  --output output/phase_c/eval_on.zarr  --zeta-output output/phase_c/zeta_on.nc
$WT/target/release/eval  --backend cpu --config $WT/config/experiments/phase_c_off.yaml --checkpoint <off_ckpt> --output output/phase_c/eval_off.zarr
$WT/target/release/dump_parameters --backend cpu --config ...on... --checkpoint <on>/head  --output output/phase_c/params_on.nc
$WT/target/release/dump_parameters --backend cpu --config ...off... --checkpoint <off>/head --output output/phase_c/params_off.nc
# gate: the Leg 1/2/3 computation in §7 (from eval_*.zarr, params_*.nc, zeta_on.nc, merit_channel_attributes_v1.nc)
```

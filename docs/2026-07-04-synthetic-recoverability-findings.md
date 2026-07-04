# Synthetic losing-reach recoverability (positive control) — experiment report (2026-07-04)

Spec: `docs/superpowers/specs/2026-07-03-synthetic-recoverability-design.md`
Plan: `docs/superpowers/plans/2026-07-03-synthetic-recoverability.md`
Prior instruments:
`docs/2026-07-03-zeta-gradient-probe-findings.md` (P1/P2 refuted, P3 NO-GO),
`docs/2026-07-02-leakance-diagnosis-findings.md` (H1–H7 diagnosis).
Code: `src/training/forward.rs` (`LeakanceOverride`), `src/data/store/obs_writer.rs`,
`src/bin/probe_zeta_gradient.rs` (`--mode teacher`), `src/bin/{train,eval,dump_parameters}.rs`
(`--backend cpu`), `scripts/recoverability_sites.py`,
`scripts/recoverability_analysis.py`, `scripts/notebooks/recovery_maps.ipynb`.

**One-line answer: the positive control FAILED (R1 median recovery 0.009 vs
the ≥0.5 bar) — but decomposition shows the planted signal was never visible
to the windowed training objective: with IDENTICAL teacher weights and
identical obs, the continuous residual is 0.0076 mean L1 (exactly the planted
signal) while the rho-90/warmup-5 windowed training loss is 1.017 — a ~130×
hotstart-transient noise floor that re-buries the plants inside a noise-free
world. Gauge-loss training cannot see reach-scale flux signals below its own
initial-condition noise, independent of observation uncertainty (P3) and
initialization (P2-cold).**

---

## 1. Hypothesis

The gradient probe closed two links of the attribution chain (P1 starvation
REFUTED; P2 rejection REFUTED; P3 detectability NO-GO), establishing that a
real-magnitude reach loss arrives at its measurement gauge at ~95% fidelity but
is 53× smaller than the median reference gauge's 5% discharge-uncertainty band.
One link remained unmeasured:

> **When the gauge signal IS visible — the planted flux is detectable by
> construction — does training attribute the missing water to the leakance
> term, or absorb it into routing parameters (Manning's n, q_spatial; the
> diagnosis's H5 mechanism)?**

This experiment plants detectable-scale losses in a synthetic teacher world
with a known answer key and measures where the optimizer puts the water. It is a
**positive control**: if recovery fails here, no gauge-supervised rescue
(including the auxiliary-constraint experiment) can assume the recovery
machinery works, and the auxiliary design must also constrain routing
parameters.

Two design commitments addressed the original objection to a naive twin
experiment:

1. **Recovery target is the flux field** (per-reach `zeta_net`), never the
   internally degenerate triple `(K_D, d_gw, leakance_factor)`.
2. **Warm-start attribution**: the student initializes from the *same weights
   that generated the observations*, so its step-0 residual equals exactly the
   planted signal routed to gauges — every gradient attributable to the plant,
   no cold-start stochasticity.

Pre-registered verdicts (spec §3):

| # | Metric | Bar |
|---|---|---|
| R1 | Recovery ratio: median over planted reaches of (A `zeta_net` / answer-key `zeta_net`) | ≥ 0.5 RECOVERED; ≤ 0.1 FAILED; else PARTIAL |
| R2 | Spatial precision: A `zeta_net` on non-planted reaches vs unmodified-checkpoint baseline | < 2× = PRECISE; else SMEARED |
| R3 | Absorption gap: final-epoch mean training loss, A vs B, same obs + seed | A < B by > 5% rel ⇒ leakance term needed; A ≈ B ⇒ H5 absorption confirmed |
| R4 | Absorption map: per-reach Δn (B minus checkpoint) around planted basins | descriptive — no bar |
| R5 | Cold emergence: run-C `zeta_net` at planted reaches vs C's own non-planted median | > 3× EMERGES |

**Headline: PASS iff R1 ≥ 0.5 AND R3 shows A < B.**

---

## 2. What was changed to test it

All changes are confined to the eval path and tooling; no autograd `Backward`
impl was touched (repo invariant 4), and the DDR-parity regression stayed at
ABSOLUTE MATCH throughout.

1. **`LeakanceOverride` eval-path seam** (2e078e8 + guards 1fa4c3f) — new
   struct in `src/training/forward.rs` injecting per-reach normalized overrides
   of K_D/d_gw/factor between `head.forward` and denormalization inside
   `forward_eval`; training `forward` is untouched. The override is a dense
   mask-select over the network's reach columns (not a side-channel into
   autograd): `param * (1 − mask) + vals * mask`. Re-exported from
   `src/training/mod.rs`. Guard suites green throughout; `compare_ddr_sandbox`
   ABSOLUTE MATCH (override path passes `None` at all existing call sites).

2. **Zarr-v2 synthetic-obs writer** `src/data/store/obs_writer.rs` (19ac965)
   — `write_obs_zarr_v2(dir, staids, epoch, day0, daily)` writes one uncompressed
   f64 array per STAID, NaN-padded from the implicit 1980-01-01 epoch. Roundtrip
   test goes through the dispatching `ObservationsStore::open` and asserts every
   STAID resolves and values + pad read back exactly.

3. **`--backend cpu` dispatch** for `train`/`eval`/`dump_parameters` (2c4415b +
   31af6ab) — copies the probe binary's generic-`fn run<I: Backend>` pattern
   into each, with `default_value = "cuda"` so all existing callers are
   unaffected. The cpu arm forces `sparse_solver: cpu` and
   `use_cuda_graphs: false`. Entire experiment ran on CPU (`NdArray<f32>`,
   deterministic; GPU was occupied by another job).

4. **`--mode teacher` in the probe binary** (9003012 + pre-flight checks
   6952543) — chunked forward with `LeakanceOverride` + zeta accumulation;
   writes synthetic obs via `obs_writer` and the answer key via the existing
   `write_zeta_netcdf`. The obs window is the teacher's tau-trimmed daily
   predictions, minus the last day (matching the training axis exactly).
   Answer-key identity proven by `tests/zeta_accum.rs` (same accumulator path
   the backward reads).

5. **Experiment configs** (9777dfc, headers cleaned 33da10c) — five YAML files
   in `config/experiments/recoverability_*.yaml`, generated from
   `leakance_hourly_on.yaml`: teacher/measure use the `testing:` window
   1981-09-30..1995-10-01 (one day wider each side so tau-trim + last-day drop
   produce exactly the student training axis); A/B warm-start from
   `output/recoverability/init_head/` (head.mpk only — existing
   `bootstrap_head_and_state` fallback requires zero new code); B has
   `use_leakance: false` with the ON-head architecture (config validation only
   rejects the `use_cuda_graphs` combination, not this pair). All five configs
   set `sparse_solver: cpu` and widen `K_D` to `[1e-8, 1e-5]` (see below).

6. **Plant-site selection** `scripts/recoverability_sites.py` (82447b6) —
   joins the gradient probe's Ref reaches with `band5` from
   `detectability_rows.csv` and `depth_mean`/`area_z_mean` from the diagnosis
   run's `kan_parameters.nc`; computes `ceiling_flux = area_z_mean · K_D_max ·
   (depth_mean + 2)`; target = `min(2·band5, 0.5·ceiling)`; drops reaches where
   `ceiling < 0.25·band5` (unrepresentable). **Mid-experiment amendment
   (user decision 2026-07-03): K_D range widened [1e-8, 1e-6] → [1e-8, 1e-5]
   in all five configs** after the original ceiling left only 23/96 sites
   expressible. The widened range applies identically to teacher and all three
   students, so expressibility-by-construction and step-0 attribution hold. This
   is the second time the K_D ceiling proved binding (first: learned K_D pinning
   at 1e-6 in the 2×2). Result: 58/96 sites kept, 38 dropped.

7. **Analysis script and notebook** — `scripts/recoverability_analysis.py`
   (fe055a0, a218546) implements R1–R5 exactly per spec §3; prints the
   VERDICTS block to stdout (tee'd to `logs/verdicts.log`). Notebook
   `scripts/notebooks/recovery_maps.ipynb` (0cdfc5f) generates the two CONUS
   maps.

---

## 3. The experiment

**Teacher world.** Hourly-ON checkpoint
(`.ddrs/runs/2026-07-01T13-43-32Z-train-and-test/checkpoints/epoch_5_mb_9`)
run in eval-mode over the testing window 1981-09-30..1995-10-01 (5115 days,
341 15-day chunks) with 58 planted reaches (k_d_norm = 1.0, d_gw_norm = 0.0,
factor_norm = target/ceiling; factor_norm p50 = 0.5). Wall-time ~2.6 h on CPU
(solo projection from the timing gate was 71 min; shared-CPU slowdown accounts
for the gap). Outputs: a 2,365-gauge zarr-v2 obs store (5,113 data days from
1981-10-01, plus a 639-day NaN pad to 1980-01-01 epoch) and the answer key
over 64,892 eval reaches.

**Teacher validation.** Answer-key / target-flux ratio: p10 = 0.370, p50 =
0.600, p90 = 1.018 (scatter around 1 expected — the ceiling used mean depth,
sub-daily depth variation is real). All 58 planted reaches have positive
`zeta_net`. Planted median 7.213e-2 m³/s vs non-planted 1.752e-3 m³/s (41×
background). STAID resolution exact: 2,365/2,365 gauges matched, 0 missing.

**Operational note — trailing-NaN patch.** The measurement eval needed
observations one day past the teacher's last predicted day (the eval window's
start is 1981-09-30, one day before the first synthetic obs on 1981-10-01). Fix:
append a single NaN f64 to all 2,365 gauge arrays and bump `.zarray`
`"shape"` from 5,752 → 5,753. One-off Python snippet (run once, before
measurement passes):

```python
import json, struct
from pathlib import Path

obs = Path("/home/tbindas/projects/ddrs/output/recoverability/synthetic_obs")
nan_bytes = struct.pack("<d", float("nan"))
for zarr_dir in obs.iterdir():
    if not zarr_dir.is_dir(): continue
    zarray = zarr_dir / ".zarray"
    meta = json.loads(zarray.read_text())
    meta["shape"] = [meta["shape"][0] + 1]
    meta["chunks"] = [meta["chunks"][0] + 1]
    zarray.write_text(json.dumps(meta))
    chunk = zarr_dir / "0"
    chunk.write_bytes(chunk.read_bytes() + nan_bytes)
```

Obs values do not enter routing or zeta; the answer-key window identity is
preserved. The patch was verified clean by the lag test (§4 below).

**Students.** Run in parallel on CPU with `nice -n 10`, ~85 min each. All three
share the identical recipe: 5 epochs × 36 mini-batches (batch 64, rho 90,
warmup 5, seed 42, L1 loss, lr 1e-3 → 5e-4 at epoch 3, hourly
precip-disaggregation head).

| Run | Init | Leakance | Step-0 loss |
|---|---|---|---|
| A | `epoch_5_mb_9` head weights only, fresh Adam | ON | 1.017 |
| B | same head weights only, fresh Adam | OFF | 2.323 |
| C | cold seed-42 init | ON | 10.312 |

Boot lines verified in each log: A and B contain `warm start: loaded KAN head`
+ `no …/optim.mpk — Adam starts cold` + `restarting at epoch 1 with a fresh
shuffle`; C contains none. Step-0 ordering (A < B < C) is the predicted
sequence and confirms the world is internally consistent.

**Lag test (alignment check).** Continuous eval of student A's final checkpoint
against the synthetic obs: lag-0 mean L1 = 0.4431; ±1-day lag ≈ 4.9 — a 10×
jump, confirming no day-offset bug in the obs alignment.

---

## 4. Did the test pass or fail?

| # | Metric | Measured | Verdict |
|---|---|---|---|
| R1 | Recovery ratio median (n=58) | 0.009 (p10 = −0.073, p90 = 0.199) | **FAILED** (bar: ≥ 0.5) |
| R2 | Non-planted \|zeta_net\| A/baseline | 1.11 (A = 1.942e-3, base = 1.752e-3) | **PRECISE** (bar: < 2) |
| R3 | Final-epoch loss A vs B | A = 1.339 (n=36), B = 2.317 (n=36), gap +42.2% | **A < B (leakance needed)** — but CONFOUNDED |
| R4 | Δn absorption map | median Δn planted = −0.019; all = −0.019; p90\|Δn\| planted = 0.028 | descriptive |
| R5 | Cold emergence ratio | 1.20 | **SUPPRESSED** (bar: > 3) |

**HEADLINE: FAIL.**

---

**R1 FAILED.** Student A left planted-reach `zeta_net` at baseline levels; the
median recovery ratio 0.009 is indistinguishable from the non-planted
background (non-planted median |zeta_net| = 1.942e-3, planted answer-key median
= 7.213e-2). The worst-10 sites all have NEGATIVE recovered flux — Adam moved
zeta in the wrong direction at some planted reaches. Nothing recovered.

**R2 PRECISE — but for the wrong reason.** Non-planted |zeta_net| hardly moved
(1.11× baseline): the model didn't smear, because it didn't move at all. R2 is
trivially satisfied when R1 fails.

**R3 CONFOUNDED — report honestly.** The 42.2% loss gap (A 1.339 vs B 2.317)
says the leakance BASE field is load-bearing for fit. But the spec's §6.2
caveat — that B's step-0 handicap (base zeta field absent) would be
second-order, since planted magnitudes are ~2 orders larger than the background
— proved first-order: B's step-0 handicap (loss 2.323 vs A's 1.017, ratio ≈
1.3) accounts for approximately its final gap (0.98). The base field aggregates
over thousands of upstream reaches per gauge → O(1) m³/s effect at gauges. R4
corroborates: B's Δn is global (median Δn planted ≈ median Δn everywhere,
|Δn| max 0.043) — a global re-equilibration against the missing base field, not
plant-localized absorption. R3 measures that the leakance TERM matters
aggregate; it cannot say whether individual planted fluxes were recoverable.

**R5 SUPPRESSED.** Cold student C also failed to place elevated zeta at planted
reaches (ratio 1.20 vs >3 bar), consistent with the gradient probe's cold
push-down finding (80.5% of ungauged gradients push zeta down at the cold
point).

---

**The discovered mechanism (post-hoc, labeled as such): the windowed
objective's hotstart-transient noise floor.** The decomposition that reveals
why R1 fails even in the best possible world:

| Quantity | Value | Source |
|---|---|---|
| Continuous residual (teacher weights + teacher obs, full-window eval) | 0.00759 mean L1 (median 0.0; 30 gauges > 0.01) | recomputed from eval_a.zarr + /tmp/recov_baseline_eval.zarr |
| Step-0 windowed training loss (run A) | 1.017 | student_a.log |
| Ratio | ~130× | — |
| Run A's continuous residual after 5 epochs of training | 0.4431 | lag-test eval |

The training objective samples rho-90 windows hot-started from heuristic
initial conditions; the synthetic obs were generated by *continuous* routing
with fully developed storage. A warmup of 5 days trims far too little — big
rivers carry memory of tens to hundreds of days, so the initial-condition
mismatch dominates every training window. The planted signal (0.0076 mean L1)
is 0.8% of the training loss (1.017) — invisible. Crucially, Adam then actively
degrades the model: after 5 epochs the continuous residual has grown from 0.0076
to 0.4431 (58× worse than not training) as the optimizer chases irreducible
initial-condition noise. The final training losses confirm convergence to the
same floor from all three starting points (A 1.339, B 2.317, C 1.370: C, cold,
lands at the same floor as A in the same number of epochs).

This is a third, independent masking layer beyond P3 (observational uncertainty
in the real world) and P2-cold (initial-training suppression): even in a
noise-free world with perfectly warm-started weights, the windowed objective's
initial-condition transient swamps the signal before any plant-attributable
gradient can accumulate.

---

## 5. Conclusions

1. **Gauge-loss training cannot reward reach-scale leakance even in the best
   possible world.** Signal that is detectable at the gauge (by construction),
   zero obs noise, expressible via the head, and warm-started from the answer
   is still invisible to training because the windowed objective's own noise
   floor is ~130× larger. This is the third independent masking layer, layered
   on top of P3 (real-world detectability) and P2-cold (early suppression),
   and it operates before either of those comes into play.

2. **The R3/R4 pair shows the leakance base field is load-bearing for fit in
   aggregate, even while individual reach fluxes are unlearnable.** Removing
   the routing term costs 42% loss and forces a global n re-equilibration.
   The term matters; the individual contributions are unreachable by the
   current training objective.

3. **The auxiliary-supervision path is now triply forced, and it must inject
   signal outside the gauge-discharge loss.** Any gauge-mediated reward is
   subject to the ~130× noise floor — so the spatial prior on zeta_net/d_gw
   must act directly on head outputs, not through routed discharge. This also
   implies that staged training (leakance-OFF convergence → leakance-ON +
   auxiliary term) may be necessary to avoid P2-cold suppression before the
   aux constraint can counter it.

4. **A general ddrs training finding beyond leakance: warmup = 5 under-trims
   hotstart transients by ~2 orders of magnitude.** The windowed training
   objective is not a fixed point of continuous behavior — fine-tuning a
   converged model on its own outputs makes it WORSE. Longer warmup, persistent
   state training windows, or transient-weighted loss are potential mitigations;
   this deserves its own experiment (cheap — floor vs warmup is forward-only).

5. **K_D ceiling bound twice.** The 2×2 learned K_D pinned at the original
   1e-6 ceiling; the recoverability sites script needed the widened [1e-8, 1e-5]
   range to achieve adequate expressibility (58/96 sites vs 23/96). The widened
   range should be considered the default for any future leakance work.

---

## 6. Next steps

1. **Warmup/transient experiment (new, general):** quantify the floor vs warmup
   length — compute teacher-weights windowed loss at warmup ∈ {5, 15, 30, 60}
   days; consider persistent-state training windows. This is forward-only, cheap
   (no training required for the floor curve itself), and separates the
   "objective floor" problem from any remaining attribution failure.

2. **Auxiliary-constraint experiment (unchanged top leakance follow-up),** now
   with two added design requirements: (a) the auxiliary loss must act directly
   on head outputs (zeta_net/d_gw) without passing through routed discharge;
   (b) staged training (leakance-OFF convergence → ON + auxiliary) should be
   evaluated to dodge P2-cold suppression before the auxiliary term can open
   the gradient path.

3. **Optional: re-run this recoverability control after a warmup fix** to test
   whether floor removal alone enables recovery. If yes, the aux-constraint
   design can work through the discharge loss (cheaper architecture). If no,
   direct head-output constraints are confirmed necessary.

4. **Leakance stays hourly-gated and experimental.** Nothing here justifies
   promotion; the compounding masking layers sharpen the documentation of why
   gauge-supervised leakance is marginal.

---

## 7. Raw verdict output

```
========================================================================
VERDICTS (bars pre-registered in the spec)
========================================================================
  [R1 FAILED] recovery ratio median=0.009 (p10=-0.073 p90=0.199, n=58; bar: >=0.5 recovered, <=0.1 failed)
  [R2 PRECISE] non-planted |zeta_net| A/baseline = 1.11 (A=1.942e-03, base=1.752e-03; bar: <2)
  [R3 A<B (leakance needed)] final-epoch mean loss A=1.33933 (n=36) B=2.31692 (n=36) rel gap=+42.2% (bar: 5%)
  [R4] median dn planted=-1.898e-02 all=-1.893e-02 p90|dn| planted=2.757e-02
  [R5 SUPPRESSED] cold planted/non-planted |zeta_net| = 1.20 (bar: >3)

  HEADLINE: positive control FAIL (requires R1>=0.5 AND A beats B)

per-reach rows -> /home/tbindas/projects/ddrs/output/recoverability/recovery_rows.csv
```

Supporting decomposition (recomputed from eval_a.zarr + /tmp/recov_baseline_eval.zarr):

| Quantity | Value | Derivation |
|---|---|---|
| Continuous residual (teacher weights, teacher obs, full eval window) | 0.00759 mean L1 | baseline continuous eval with A's final checkpoint; median 0.0 because most gauges have no planted upstream reach; 30 gauges > 0.01 |
| Step-0 windowed training loss (run A) | 1.017 | first mini-batch of student_a.log, epoch 1 |
| Ratio (noise floor / signal) | ~130× | 1.017 / 0.00759 |
| Run A continuous residual after training | 0.4431 | lag-test eval (lag 0); ±1-day lag ≈ 4.9 (confirms no alignment bug) |

Lag test row: lag = 0: mean L1 = 0.4431; lag = +1 day: mean L1 ≈ 4.9; lag = −1 day: mean L1 ≈ 4.9. The ~10× jump at ±1 day confirms zero day-offset in the synthetic-obs alignment.

---

## 8. Reproduce

All commands use cwd `/home/tbindas/projects/ddrs` (main tree — data and
`.ddrs/` live there); `WT=/home/tbindas/projects/ddrs/.claude/worktrees/zeta-sensitivity`.

```bash
# 0. Build the worktree binaries (once, after any src/ change)
cd $WT && cargo build --release --bins && cd /home/tbindas/projects/ddrs

# 1. Plant-site selection (ddrs-py uv venv)
cd /home/tbindas/projects/ddrs/ddrs-py && uv run python \
  $WT/scripts/recoverability_sites.py
# outputs: output/recoverability/plants.csv, output/recoverability/sites_report.txt
cd /home/tbindas/projects/ddrs

# 2. Teacher run + baseline zeta eval (parallel; ~2.6 h each on CPU)
nice -n 10 $WT/target/release/probe_zeta_gradient \
  --mode teacher --backend cpu \
  --config $WT/config/experiments/recoverability_teacher.yaml \
  --checkpoint .ddrs/runs/2026-07-01T13-43-32Z-train-and-test/checkpoints/epoch_5_mb_9 \
  --plant-file output/recoverability/plants.csv \
  --eval-days 999999 \
  --obs-output output/recoverability/synthetic_obs \
  --zeta-output output/recoverability/answer_key.nc \
  2>&1 | tee output/recoverability/logs/teacher.log &

nice -n 10 $WT/target/release/eval --backend cpu \
  --config $WT/config/experiments/recoverability_teacher.yaml \
  --checkpoint .ddrs/runs/2026-07-01T13-43-32Z-train-and-test/checkpoints/epoch_5_mb_9 \
  --output /tmp/recov_baseline_eval.zarr \
  --zeta-output output/recoverability/baseline_zeta.nc \
  2>&1 | tee output/recoverability/logs/baseline_zeta.log &

wait

# 3. Trailing-NaN patch (one-off; needed before measurement passes)
#    Appends one NaN f64 to every gauge array in the synthetic obs store and
#    bumps .zarray shape 5752→5753 so the eval window's one-day look-ahead resolves.
python3 - <<'EOF'
import json, struct
from pathlib import Path
obs = Path("/home/tbindas/projects/ddrs/output/recoverability/synthetic_obs")
nan_bytes = struct.pack("<d", float("nan"))
for zarr_dir in obs.iterdir():
    if not zarr_dir.is_dir(): continue
    zarray = zarr_dir / ".zarray"
    meta = json.loads(zarray.read_text())
    meta["shape"] = [meta["shape"][0] + 1]
    meta["chunks"] = [meta["chunks"][0] + 1]
    zarray.write_text(json.dumps(meta))
    chunk = zarr_dir / "0"
    chunk.write_bytes(chunk.read_bytes() + nan_bytes)
print("patched", sum(1 for d in obs.iterdir() if d.is_dir()), "gauges")
EOF

# 4. Create warm-start dir (head.mpk only — bootstrap_head_and_state
#    detects missing optim.mpk/state.json and starts Adam cold at epoch 1)
mkdir -p output/recoverability/init_head
cp .ddrs/runs/2026-07-01T13-43-32Z-train-and-test/checkpoints/epoch_5_mb_9/head.mpk \
   output/recoverability/init_head/head.mpk

# 5. Student training runs A, B, C (parallel; ~85 min each)
for s in a b c; do
  mkdir -p output/recoverability/students/$s
  nice -n 10 $WT/target/release/train --backend cpu \
    --config $WT/config/experiments/recoverability_student_$s.yaml \
    --checkpoint-dir output/recoverability/students/$s \
    > output/recoverability/logs/student_$s.log 2>&1 &
done
wait

# 6. Measurement eval passes (A and C; B has no zeta by construction)
for s in a c; do
  CKPT=$(ls -d output/recoverability/students/$s/epoch_5_mb_* | sort -V | tail -1)
  nice -n 10 $WT/target/release/eval --backend cpu \
    --config $WT/config/experiments/recoverability_measure.yaml \
    --checkpoint "$CKPT" \
    --output output/recoverability/eval_$s.zarr \
    --zeta-output output/recoverability/zeta_$s.nc \
    2>&1 | tee output/recoverability/logs/measure_$s.log &
done
wait

# 7. dump_parameters for R4 (B's Manning's n vs the original head)
CKPT_B=$(ls -d output/recoverability/students/b/epoch_5_mb_* | sort -V | tail -1)
nice -n 10 $WT/target/release/dump_parameters --backend cpu \
  --config $WT/config/experiments/recoverability_student_b.yaml \
  --checkpoint "$CKPT_B/head" \
  --output output/recoverability/params_b.nc
nice -n 10 $WT/target/release/dump_parameters --backend cpu \
  --config $WT/config/experiments/recoverability_student_b.yaml \
  --checkpoint output/recoverability/init_head/head \
  --output output/recoverability/params_orig.nc

# 8. Verdicts
cd /home/tbindas/projects/ddrs/ddrs-py && uv run python \
  $WT/scripts/recoverability_analysis.py 2>&1 | tee \
  /home/tbindas/projects/ddrs/output/recoverability/logs/verdicts.log

# 9. Maps notebook
uv run jupyter nbconvert --to notebook --execute \
  $WT/scripts/notebooks/recovery_maps.ipynb --output recovery_maps \
  --output-dir /home/tbindas/projects/ddrs/output/recoverability/plots
```

Maps at:
- `output/recoverability/plots/recovery_ratio_map.png`
- `output/recoverability/plots/absorption_dn_map.png`

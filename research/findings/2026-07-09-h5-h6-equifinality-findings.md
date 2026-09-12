# H5/H6 parameter-transfer and loss-landscape — findings (2026-07-09)

Spec: `docs/superpowers/specs/2026-07-08-landscape-hypotheses-h5-h6-draft.md`
Prior findings: `docs/2026-07-07-lstm-equifinality-v2-findings.md` (H1–H4)

> **Superseded analysis (2026-07-09):**
> `docs/2026-07-09-h5-h6-equifinality-v2-findings.md` re-analyzes these same
> raw CSVs with paired statistics and corrects §3.1's noise comparison, the
> missing split-half floors (computable from the per-window CSVs), the
> unreported R1↔R2 control comparison, and §3.2's surface characterization.
> Verdicts unchanged (both INCONCLUSIVE); read the v2 doc for interpretation.
Skill reference (plots): `.claude/skills/ddrs-eval-plots/references/parameter_swap.md`, `references/loss_landscape_h6.md`

**One-line verdict:** Both H5 and H6 are **INCONCLUSIVE** under the registered bars — H5 because the transfer penalties (P_n, P_geo) are tiny relative to window-to-window noise and no split-half noise floor has been computed to certify them either way; H6 because the loss surface is a sharp, roughly isotropic bowl (not a degenerate valley) at the tested resolution, so the SUPPORTED path never opens even though the minimum's location moves substantially (1.90 log2-units) between forcings.

---

## 1. Motivating observation and hypotheses

The LSTM equifinality campaign (H1–H4) found Manning's n diverges ~40% in level across independently-forced arms while realized channel geometry converges, but could not distinguish whether n is *sloppy* (poorly constrained, one shared optimum) or a genuine *compensator* (a forcing-indexed family of optima) — every existing analysis level (raw parameters, realized geometry, routing skill, gradient alignment) is blind to that distinction. H5 and H6 are the next two pre-registered hypotheses, testing it directly: H5 at the endpoint level (does swapping a parameter class between arms hurt the loss?), H6 at the landscape level (does the loss surface's valley floor itself move with the forcing source?).

| # | Hypothesis | Mechanism proposed |
|---|---|---|
| H5 | Forcing-bound roughness | Learned n is bound to its training forcing — swapping n between independently-forced arms degrades the loss substantially more than swapping geometry (q_spatial, p_spatial), under either arm's own Q′ |
| H6 | Forcing-indexed valley | The training objective under each Q′ source has a degenerate n–geometry valley whose floor location shifts with the forcing source — cross-arm n divergence is movement along a forcing-indexed family of minima, not noise around one shared minimum |

---

## 2. Methods — how the experiment was done

Both hypotheses reuse the existing R1 (daily-LSTM flat, `2026-07-07T03-55-53Z-train-and-test`), R2 (daily-LSTM disagg, `2026-07-07T04-49-19Z-train-and-test`), and R3 (hourly-MTS-LSTM, `2026-07-07T06-50-28Z-train-and-test`) checkpoints (`checkpoints/epoch_5_mb_35`) and their `dump_parameters` NetCDF dumps (`output/equif/R{1,2,3}_kan_parameters.nc`) from the H1–H4 campaign — **no retraining**. Both are new modes on the `probe_zeta_gradient` binary (`src/bin/probe_zeta_gradient.rs`), sharing one seeded-plan-then-replay determinism pattern.

### 2.1 Shared infrastructure: sample the window plan once, replay it everywhere

Both modes must guarantee that every parameter composition (H5) or grid/barrier point (H6) is evaluated against the *identical* rho-day windows and gauge batches — otherwise a loss difference is confounded with different training data, not attributable to the parameter change. This is done by sampling the deterministic window/gauge plan **exactly once**, using only observations (never anything parameter-dependent) to decide which windows survive:

```rust
// src/bin/probe_zeta_gradient.rs — sample_window_plan (shared by H5 and H6)
let mut rng = ChaCha12Rng::seed_from_u64(seed);
let mut sampler = BatchSource::Shuffle(RandomSampler::new(dataset.len(), batch_size, true));
sampler.reshuffle(&mut rng);

while processed < windows {
    let idx = match sampler.next_batch() {
        Some(idx) => idx,
        None => { sampler.reshuffle(&mut rng); continue; }
    };
    let staids: Vec<Staid> = idx.iter().map(|&i| dataset.staids()[i].clone()).collect();
    let window = dataset.time_axis().sample_rho_window(&mut rng, rho);
    let batch = dataset.collate(&staids, &window)?;

    // Validity depends ONLY on observations — never on any parameter or
    // composition — so the plan cannot diverge between compositions/grid points.
    let surviving = (0..batch.gauge_staids.len()).any(|gi| {
        (warmup..t_days).all(|ti| !batch.observations[(ti + 1, gi)].is_nan())
    });
    if surviving { plan.push((staids, window)); processed += 1; }
}
```

Each window is then collated **once** and its `RoutingTensors` reused (borrowed, not consumed) across every composition or grid point — `forward_eval` takes `&RoutingTensors`, so re-evaluating with a different parameter override costs one forward pass, not one re-collate.

### 2.2 The override mechanism (`RoutingParamOverride`)

Both H5 and H6 inject per-reach `n`/`q_spatial`/`p_spatial` values in NORMALIZED `[0,1]` space, applied immediately after the KAN head's own forward pass and before the routing solve — the same seam the existing `LeakanceOverride` uses for the leakance trio:

```rust
// src/training/forward.rs
pub struct RoutingParamOverride {
    pub n: Option<Vec<f32>>,
    pub q_spatial: Option<Vec<f32>>,
    pub p_spatial: Option<Vec<f32>>,
}

// Inside forward_eval_core, right after head.forward():
let (n_param, q_param, p_param) = if let Some(po) = param_overrides {
    let n_param = RoutingParamOverride::replace_or_passthrough(n_param, &po.n, device);
    let q_param = RoutingParamOverride::replace_or_passthrough(q_param, &po.q_spatial, device);
    let p_param = match (p_param, &po.p_spatial) {
        (Some(p), ov) => Some(RoutingParamOverride::replace_or_passthrough(p, ov, device)),
        (None, Some(_)) => panic!("p_spatial override requested but head has none"),
        (None, None) => None,
    };
    (n_param, q_param, p_param)
} else { (n_param, q_param, p_param) };
```

This is orthogonal to the disaggregation head (`head.disagg`, computed independently from attributes/precip) and to leakance (off for every equif arm) — overriding `n`/`q_spatial`/`p_spatial` never touches either.

### 2.3 H5 — parameter swap (`--mode eval-loss`)

Four compositions per window: `own` (no override, the checkpoint's native baseline), `n-swap` (donor's `n`, own geometry), `geo-swap` (donor's `q_spatial`+`p_spatial`, own `n`), `full-swap` (all three from the donor). The donor's physical-unit values are read once per window (`load_comid_field` + `gather_by_comid`, COMID-keyed, hard-error on a missing COMID) and converted to normalized space (`physical_to_normalized`, the exact inverse of `denormalize`):

```rust
Composition::NSwap => Some(RoutingParamOverride {
    n: donor_n.clone(), q_spatial: None, p_spatial: None,
}),
Composition::GeoSwap => Some(RoutingParamOverride {
    n: None, q_spatial: donor_q.clone(), p_spatial: donor_p.clone(),
}),
```

Transfer penalties: `P_n = L_X(n-swap) − L_X(own)`, `P_geo = L_X(geo-swap) − L_X(own)`, attribution fraction `f_n = P_n / (P_n + P_geo)`.

**Registered invocation** (96 windows, seed 42, both forcings and the low-disagreement control pair):

```bash
cargo run --release --bin probe_zeta_gradient -- \
    --mode eval-loss --backend cpu \
    --config .ddrs/runs/<forcing-arm-run-id>/config.yaml \
    --checkpoint .ddrs/runs/<forcing-arm-run-id>/checkpoints/epoch_5_mb_35 \
    --donor-params-nc output/equif/<donor-arm>_kan_parameters.nc \
    --compositions own,n-swap,geo-swap,full-swap \
    --windows 96 --seed 42 \
    --loss-output output/equif/h5/registered/<label>.csv \
    --per-gauge-output output/equif/h5/registered/<label>_per_gauge.csv
```

### 2.4 H6 — loss-landscape overlay (`--mode landscape`)

Anchor field θ̄ = the arm-mean of R1's and R3's own dumps, per-COMID, geometric mean for config-flagged log-space fields (only `p_spatial` in these configs) and arithmetic mean otherwise:

```rust
fn arm_mean_field(a: &HashMap<i64,f32>, b: &HashMap<i64,f32>, log_space: bool) -> HashMap<i64,f32> {
    a.iter().filter_map(|(comid, &va)| b.get(comid).map(|&vb| {
        let mean = if log_space { ((va.ln() + vb.ln()) / 2.0).exp() } else { (va + vb) / 2.0 };
        (*comid, mean)
    })).collect()
}
```

11×11 grid over `(log2 α, log2 β) ∈ [-1.5, 1.5]²`; at each grid point `n_grid = n̄ · 2^log2_α`, `p_grid = p̄ · 2^log2_β`, `q_grid = q̄` (q_spatial is never scaled — only 2 axes are scanned):

```rust
let n_phys = scale_by_log2(&anchor_n_gathered, log2_alpha); // v * 2^log2_alpha
let p_phys = scale_by_log2(&anchor_p_gathered, log2_beta);
let ov = RoutingParamOverride {
    n: Some(physical_to_normalized(&n_phys, ranges.n, log_space("n"))),
    q_spatial: Some(anchor_q_normalized.clone()),               // held fixed
    p_spatial: Some(physical_to_normalized(&p_phys, ranges.p_spatial, log_space("p_spatial"))),
};
```

Linear barrier: 21 points `t ∈ {0, 0.05, …, 1}` interpolating in LOG space between each arm's *own* field (not the anchor) — `θ(t) = exp((1−t)·ln θ_R1 + t·ln θ_R3)` for all three parameters, per the registered formula, with `t=0`/`t=1` special-cased to reproduce the endpoint fields bit-exactly:

```rust
fn log_interp(a: &[f32], b: &[f32], t: f32) -> Vec<f32> {
    if t == 0.0 { return a.to_vec(); }
    if t == 1.0 { return b.to_vec(); }
    a.iter().zip(b).map(|(&va, &vb)| ((1.0 - t) * va.ln() + t * vb.ln()).exp()).collect()
}
```

**Registered invocation** (16-window fixed subset, 11×11 grid, 21-point barrier, one invocation per forcing arm):

```bash
cargo run --release --bin probe_zeta_gradient -- \
    --mode landscape --backend cpu \
    --config .ddrs/runs/<forcing-arm-run-id>/config.yaml \
    --checkpoint .ddrs/runs/<forcing-arm-run-id>/checkpoints/epoch_5_mb_35 \
    --params-nc-a output/equif/R1_kan_parameters.nc \
    --params-nc-b output/equif/R3_kan_parameters.nc \
    --windows 16 --seed 42 \
    --surface-output output/equif/h6/<arm>_surface.csv \
    --barrier-output output/equif/h6/<arm>_barrier.csv
```

Derived scalars (minima, sublevel-set aspect ratio, minima displacement, barrier statistic) are computed in Python from the raw CSVs, not in Rust — matching the H1–H4 campaign's Rust-computes-raw/Python-computes-verdict convention.

### 2.5 Binary/provenance note

Built and run on this session's `probe_zeta_gradient` (rebuilt via `cargo build --release --bin probe_zeta_gradient` immediately before the registered runs — checkpoints are directories, `epoch_5_mb_35/head.mpk`, confirming the current checkpoint format, not the stale flat-file layout). CPU backend (`--backend cpu`, forces `sparse_solver=cpu` for determinism), `nice -n 10` (a `neuralhydrology` training job was concurrently running on the same host).

---

## 3. Results — how it was resolved

| # | Hypothesis | Verdict | Key number |
|---|---|---|---|
| H5 | Forcing-bound roughness | **INCONCLUSIVE** | f_n = 1.066 (R1 forcing) / 1.485 (R3 forcing) — both nominally clear the ≥2/3 bar, but P_n (+0.095 / −0.028 m³/s) and P_geo (−0.006 / +0.009 m³/s) are both tiny relative to the ~4.2–4.5 m³/s window-to-window std, and no split-half noise floor exists to certify the SUPPORTED bar's required 3× check |
| H6 | Forcing-indexed valley | **INCONCLUSIVE** | 5%-sublevel aspect ratio 1.00 (R1 forcing) / 0.90 (R3 forcing) — fails the ≥3:1 degeneracy bar, so SUPPORTED cannot trigger regardless of the minima displacement (1.90 log2-units, R1 at (0.3, 0.9) vs R3 at (−0.3, −0.9)) |

### 3.1 H5 detail

Registered protocol: 96 windows, seed 42, `probe_zeta_gradient --mode eval-loss`, both directions of the primary pair (R1↔R3) and the low-disagreement control pair (R1↔R2):

| Run | own | n-swap | geo-swap | full-swap | P_n | P_geo | f_n |
|---|---|---|---|---|---|---|---|
| R1 under R1 forcing (donor R3) | 9.9443 | 10.0396 | 9.9384 | 10.0351 | +0.0953 | −0.0059 | 1.0664 |
| R3 under R3 forcing (donor R1) | 10.7495 | 10.7215 | 10.7586 | 10.7356 | −0.0279 | +0.0091 | 1.4849 |
| R1 under R1 forcing (donor R2, control) | 9.9443 | 10.0694 | 9.9347 | 10.0761 | +0.1251 | −0.0095 | 1.0826 |
| R2 under R2 forcing (donor R1, control) | 9.5547 | 9.6508 | 9.5482 | 9.6595 | +0.0961 | −0.0065 | 1.0728 |

(mean L1 loss in m³/s, 96 windows, 2,365 training gauges' batches, `output/equif/h5/registered/*.csv`; per-gauge companion CSVs at `*_per_gauge.csv`, 23,676 rows each, not yet analyzed in this pass.)

Applying the registered bars literally: none of the four `f_n` values are ≤ 1/2, so H5 does **not** hit the REFUTED threshold. All four nominally clear the ≥2/3 SUPPORTED threshold on `f_n` alone — but SUPPORTED additionally requires "P_n exceeds the split-half noise floor by ≥ 3×," and that split-half comparison (a second 96-window pass at seed 123) has **not been run**. Absent that evidence, SUPPORTED cannot be certified. Two further observations argue against reading these numbers as a clean compensator signal even informally: (1) P_n's **sign flips** between forcings — positive under R1/control forcing, negative under R3 forcing — which is not the pattern a consistent forcing-specific compensator would produce (it would inflict a similarly-signed penalty regardless of which arm receives the donor's n); (2) both P_n and P_geo are two orders of magnitude smaller than the ~4.2–4.5 m³/s window-to-window standard deviation within any single composition, i.e. the transfer penalties are small relative to the experiment's own sampling noise even before a formal split-half comparison. The honest read is that at the registered sample size, neither swap moves the loss by a distinguishable amount in either direction.

### 3.2 H6 detail

Registered protocol: 16-window fixed subset, seed 42, 11×11 grid (`log2 α, log2 β ∈ [-1.5, 1.5]`), 21-point linear barrier, both forcings:

| Forcing | Grid minimum | Loss at min | 5%-sublevel aspect ratio | Barrier B |
|---|---|---|---|---|
| R1 (daily flat) | (log2 α=0.3, log2 β=0.9) | 9.8523 | 1.00 | 0.0000 |
| R3 (hourly native) | (log2 α=−0.3, log2 β=−0.9) | 10.6732 | 0.90 | 0.0000 |

Minima displacement (R1 vs R3 forcing): **1.8974** log-coord units, out of a 3.0-unit axis span — a substantial fraction of the tested range. Barrier statistic `B_X = max_t L_X(θ(t)) − max(L_X(0), L_X(1))` is exactly 0.0000 under both forcings, meaning the linear path between R1's own field and R3's own field never exceeds the higher endpoint's loss — the two arms' own optima are linearly connected (same basin, no barrier), consistent with Frankle et al.'s "linear mode connectivity" reading.

Applying the registered decision table: the degeneracy criterion (5%-sublevel aspect ratio ≥ 3:1) **fails outright** for both forcings (aspect ≈ 0.9–1.0, i.e. the sublevel set is roughly as wide as it is tall — an isotropic bowl, not an elongated valley). Since SUPPORTED requires degeneracy **and** displacement jointly, it cannot trigger regardless of the (substantial) displacement. REFUTED requires the minima to coincide within noise — they clearly do not (displacement is 63% of the total axis range) — so REFUTED does not trigger either. The formal verdict is therefore INCONCLUSIVE, independent of the still-missing split-half noise floor (unlike H5, this call does not hinge on that missing evidence — degeneracy fails on its own terms).

The spec's registered joint-interpretation table (not itself a SUPPORTED/REFUTED bar, but a pre-registered qualitative reading) maps a **sharp basin whose floor moves with Q′** to "forcing-specific identification (structural input error)" — distinct from both "n identifiable" (sharp, pinned) and "n is a compensator" (flat, moving). That is the pattern observed here: the optimum is comparatively sharp (not a sloppy valley) but its location depends on which arm's forcing is active. This is offered as a descriptive reading per the spec's own pre-registered table, not a third verdict category — the formal verdict remains INCONCLUSIVE.

---

## 4. Conclusions

1. **H5 and H6 are both INCONCLUSIVE under the registered bars.** Neither SUPPORTED nor REFUTED can be certified from the data collected this pass.
2. **The missing split-half noise floor is the single biggest gap for H5.** The raw `f_n` numbers superficially clear the SUPPORTED threshold, but the required noise-floor comparison was never run, and the transfer penalties are small enough relative to within-composition window variance that the result plausibly would not survive that comparison.
3. **H6's INCONCLUSIVE call does not depend on the missing noise floor** — the degeneracy criterion fails cleanly at the tested 11×11/`[-1.5,1.5]` resolution regardless. The substantial minima displacement (1.90 log2-units) is real and reproducible but insufficient on its own to satisfy the registered SUPPORTED bar.
4. **This does not resolve, support, or refute the paper's selective-equifinality thesis** (`/home/tbindas/projects/ddr_equifinality/paper.tex`) any further than the H1–H4 campaign already did. Do not cite H5/H6 as evidence for either direction of the n-identifiability question.
5. **No src/ invariants were touched.** `compare_ddr_sandbox` remains ABSOLUTE MATCH throughout this work; the override mechanism is additive and eval-path-only.

---

## 5. Next steps

1. **Split-half noise floor (both H5 and H6)** — re-run the same registered protocols at `--seed 123` (same window counts) and diff against the seed-42 results. This is the single most consequential missing measurement: it could either certify H5's SUPPORTED bar (if P_n clears 3× the floor) or close the question definitively.
2. **H5 per-gauge / DA-stratified analysis** — the `*_per_gauge.csv` files (23,676 rows each) already exist but haven't been analyzed; joining to drainage area (per `.claude/skills/ddrs-eval-plots/references/parameter_swap.md`'s documented join) could reveal a DA-conditional signal masked by the network-wide mean.
3. **H6 at finer/wider grid resolution** — the observed minima sit near the grid edges in places (β* = ±0.9, within the [−1.5, 1.5] range but not central); a wider β range or finer spacing near the observed optima would sharpen the degeneracy/displacement read.
4. **H6 single-point 96-window re-check** — `--single-point` at each forcing's grid-minimum, `--windows 96`, to confirm the minima aren't an artifact of the 16-window fixed subsample.
5. **dHBV2 cross-family arms** — still the actual bottleneck for the whole campaign (unchanged from H1–H4); no configs exist yet.
6. Dropped this pass — **per-parameter (rather than global-scale) H6 landscape**: the registered design scans global n/p multipliers only; a per-reach landscape was noted in the spec as a possible follow-up "after the global version reads out," which it now has (INCONCLUSIVE) — worth reconsidering once the noise floor exists.

---

## 6. Raw script output

```
=== H5 (96 windows, seed 42) ===
                                    P_n     P_geo     f_n  n_windows
R1 under R1 (donor R3)           0.0953   -0.0059  1.0664         96
R3 under R3 (donor R1)          -0.0279    0.0091  1.4849         96
R1 under R1 (donor R2, control)  0.1251   -0.0095  1.0826         96
R2 under R2 (donor R1, control)  0.0961   -0.0065  1.0728         96

=== H6 minima (16 windows, seed 42, 11x11 grid) ===
R1 forcing: min=9.8523 at (log2_alpha=0.3, log2_beta=0.9), 5%-sublevel n=100, aspect~1.00
R3 forcing: min=10.6732 at (log2_alpha=-0.3, log2_beta=-0.9), 5%-sublevel n=105, aspect~0.90

minima displacement (R1 vs R3 forcing): 1.8974 (log-coord units)

=== H6 barrier (21 points, seed 42) ===
R1 forcing: L(t=0)=9.8541 L(t=1)=9.9451 max=9.9451 B=0.0000
R3 forcing: L(t=0)=10.7072 L(t=1)=10.6850 max=10.7072 B=0.0000
```

---

## 7. Reproduce

```bash
cd /home/tbindas/projects/ddrs

# H5 — registered protocol (4 invocations; swap config/checkpoint/donor per row in §3.1)
./target/release/probe_zeta_gradient \
  --mode eval-loss --backend cpu \
  --config .ddrs/runs/2026-07-07T03-55-53Z-train-and-test/config.yaml \
  --checkpoint .ddrs/runs/2026-07-07T03-55-53Z-train-and-test/checkpoints/epoch_5_mb_35 \
  --donor-params-nc output/equif/R3_kan_parameters.nc \
  --compositions own,n-swap,geo-swap,full-swap \
  --windows 96 --seed 42 \
  --loss-output output/equif/h5/registered/r1_under_r1_donor_r3.csv \
  --per-gauge-output output/equif/h5/registered/r1_under_r1_donor_r3_per_gauge.csv

# H6 — registered protocol (one invocation per forcing arm)
./target/release/probe_zeta_gradient \
  --mode landscape --backend cpu \
  --config .ddrs/runs/2026-07-07T03-55-53Z-train-and-test/config.yaml \
  --checkpoint .ddrs/runs/2026-07-07T03-55-53Z-train-and-test/checkpoints/epoch_5_mb_35 \
  --params-nc-a output/equif/R1_kan_parameters.nc \
  --params-nc-b output/equif/R3_kan_parameters.nc \
  --windows 16 --seed 42 \
  --surface-output output/equif/h6/r1_surface.csv \
  --barrier-output output/equif/h6/r1_barrier.csv

# Analysis (ddrs-py venv)
cd ddrs-py && uv run python3 -c "
import pandas as pd
df = pd.read_csv('../output/equif/h5/registered/r1_under_r1_donor_r3.csv')
print(df.groupby('composition')['mean_loss'].agg(['mean','std']))
"
```

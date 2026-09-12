# Phase B: Objective Floor Fix Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Measure the hotstart-transient noise floor of ddrs's windowed training objective as a function of warmup/rho, pick and validate the cheapest fix that brings the floor to ≤ 0.25 mean L1 (from ~1.0 today), and document the fixed recipe for Phase C.

**Architecture:** One new `--mode floor` in the probe binary samples training-style windows with the TEACHER weights against the SELF-GENERATED synthetic obs (both already on disk from the recoverability experiment) and dumps per-day per-gauge absolute residuals — since warmup is only a loss-trim, ONE run per rho measures the floor at EVERY warmup value post-hoc. An analysis script produces the floor curves + decay-by-basin-size stratification and applies the pre-registered decision rule. If a config-only fix (longer warmup/rho) reaches the bar, it is validated here; if not, this plan STOPS at a documented decision gate and the state-cache alternative gets its own plan informed by the measured decay curves.

**Tech Stack:** Rust (probe binary extension, CPU/NdArray), Python under `ddrs-py` uv venv (xarray, netCDF4, numpy, matplotlib) for analysis.

**Spec:** `docs/superpowers/specs/2026-07-04-leakance-gate-program-design.md` §4 (Phase B).

**Worktree:** `/home/tbindas/projects/ddrs/.claude/worktrees/zeta-sensitivity` (branch `worktree-zeta-sensitivity`). Heavy runs: cwd `/home/tbindas/projects/ddrs` (main tree), ABSOLUTE worktree binary paths, `nice -n 10`, logs under `/home/tbindas/projects/ddrs/output/floor_fix/logs/`. Guard suites after every code task: `cargo test --test leakance_gradcheck --test leakance_off_parity --test zeta_accum` + `cargo run --release --example compare_ddr_sandbox` (ABSOLUTE MATCH).

**Assets already on disk (from the recoverability experiment — do NOT regenerate):**
- Synthetic obs: `/home/tbindas/projects/ddrs/output/recoverability/synthetic_obs` (zarr-v2, 2365 gauges, 1981-10-01..1995-09-30 + trailing NaN pad day)
- Teacher weights: `/home/tbindas/projects/ddrs/output/recoverability/init_head/head.mpk`
- Config with synthetic obs + cpu solver: `config/experiments/recoverability_student_a.yaml` (its `experiment.checkpoint` warm-start conveniently loads the teacher head)
- Reference numbers: teacher-weights continuous residual vs these obs = **0.00759 mean L1**; windowed (rho 90, warmup 5) ≈ **1.017** (the ~130× floor this plan attacks)

**Why teacher-weights + synthetic obs is the right instrument:** with the true weights and noise-free self-generated obs, EVERY residual is initial-condition transient (plus the 58 planted signals, a 0.0076 background measured and subtractable). No real-obs noise, no model error — the floor in isolation.

**Key existing code (read before Task 1):** `src/bin/probe_zeta_gradient.rs` (`run` grad-mode: sampler replica, window collation, obs alignment `obs_arr[(ti+1, gi)]`, NaN gauge filter — the floor mode is this loop minus autograd minus loss, plus residual dumping), `src/training/forward.rs` (`forward_eval`), `src/data/dataset.rs` (`collate`, `sample_rho_window`).

---

### Task 1: `--mode floor` in the probe binary

**Files:**
- Modify: `src/bin/probe_zeta_gradient.rs` (new mode + `write_floor_netcdf` helper + unit test in the existing test mod)

- [ ] **Step 1: Failing unit test** for the writer (append to the binary's existing `#[cfg(test)] mod tests`):

```rust
    #[test]
    fn floor_netcdf_roundtrip() {
        let dir = std::env::temp_dir().join("probe_floor_nc_test");
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("floor_test.nc");
        let _ = std::fs::remove_file(&path);

        // 2 windows x 2 gauges x 3 days of |pred - obs|; NaN = filtered gauge-day.
        let resid = vec![
            0.1f32, 0.2, 0.3, 0.4, 0.5, 0.6, // window 0: gauge rows
            0.7, f32::NAN, 0.9, 1.0, 1.1, 1.2, // window 1
        ];
        let staids = vec![
            "01010000".into(), "02020000".into(),
            "01010000".into(), "03030000".into(),
        ];
        let uparea = vec![100.0f32, 2000.0, 100.0, 55000.0];
        write_floor_netcdf(&path, 2, 2, 3, &resid, &staids, &uparea, 42, "ckpt").unwrap();

        let f = netcdf::open(&path).unwrap();
        let v = f.variable("abs_residual").unwrap();
        assert_eq!(
            v.dimensions().iter().map(|d| d.len()).collect::<Vec<_>>(),
            vec![2, 2, 3]
        );
        let vals: Vec<f32> = v.get_values(..).unwrap();
        assert_eq!(vals[0], 0.1);
        assert!(vals[7].is_nan());
        let ua = f.variable("uparea").unwrap();
        let uv: Vec<f32> = ua.get_values(..).unwrap();
        assert_eq!(uv[3], 55000.0);
    }
```

Run: `cargo test --bin probe_zeta_gradient floor_netcdf_roundtrip` → FAIL (function missing).

- [ ] **Step 2: Implement the writer** (near `write_round_netcdf`, same style):

```rust
/// Floor-mode output: per-window, per-gauge-slot, per-post-trim-day |pred-obs|.
/// dims (window, gauge_slot, day); `gauge_staid` (window, gauge_slot) NC_STRING
/// identifies which gauge occupied each slot (batches differ per window);
/// `uparea` (window, gauge_slot) f32 carries the gauge's drainage area for
/// decay-by-basin-size stratification. NaN = gauge filtered (NaN obs) that
/// window-day.
#[allow(clippy::too_many_arguments)]
fn write_floor_netcdf(
    path: &Path,
    n_windows: usize,
    n_slots: usize,
    n_days: usize,
    abs_residual: &[f32],
    gauge_staids: &[String],
    uparea: &[f32],
    seed: u64,
    checkpoint_label: &str,
) -> Result<(), Box<dyn std::error::Error>> {
    debug_assert_eq!(abs_residual.len(), n_windows * n_slots * n_days);
    debug_assert_eq!(gauge_staids.len(), n_windows * n_slots);
    debug_assert_eq!(uparea.len(), n_windows * n_slots);
    let mut file = netcdf::create(path)?;
    file.add_dimension("window", n_windows)?;
    file.add_dimension("gauge_slot", n_slots)?;
    file.add_dimension("day", n_days)?;
    file.add_attribute("seed", seed as i64)?;
    file.add_attribute("checkpoint", checkpoint_label)?;
    file.add_attribute(
        "note",
        "teacher-weights windowed |pred-obs| vs self-generated obs; day 0 = first \
         post-trim day; floor(warmup) = nanmean over days >= warmup",
    )?;
    {
        let mut v = file.add_variable::<f32>("abs_residual", &["window", "gauge_slot", "day"])?;
        v.put_values(abs_residual, ..)?;
        v.put_attribute("units", "m^3/s")?;
    }
    {
        let mut v = file.add_string_variable("gauge_staid", &["window", "gauge_slot"])?;
        for (i, s) in gauge_staids.iter().enumerate() {
            v.put_string(s, (i / n_slots, i % n_slots))?;
        }
    }
    {
        let mut v = file.add_variable::<f32>("uparea", &["window", "gauge_slot"])?;
        v.put_values(uparea, ..)?;
        v.put_attribute("units", "km^2")?;
    }
    Ok(())
}
```

(Adapt `put_string` indexing to the netcdf crate's actual 2-D string API — `write_round_netcdf` used flat 1-D; if 2-D strings are awkward, store `gauge_staid` as a flat `(window*gauge_slot)` 1-D string variable with a `layout` attribute saying `window-major` — the analysis script reshapes. Choose whichever compiles cleanly; record the choice.)

Test passes.

- [ ] **Step 3: Add the mode.** `Mode::Floor` parsed from `"floor"`, `cfg_mode = ConfigMode::Training` (training-style windows), routed to `run_floor::<I>` in both backend arms. Implementation — the grad-mode loop with autograd stripped:

```rust
/// Floor mode: teacher-weights windowed residuals vs self-generated obs.
/// Mirrors `run` (grad mode) sampling exactly — LOCAL ChaCha12 rng from
/// --seed, same batch/window draws — but forward-only on the inner backend,
/// no loss, no backward. Saves |pred - obs| for every post-trim day so the
/// floor at ANY warmup is computable post-hoc: floor(w) = nanmean over
/// days >= w. The head is loaded from --checkpoint (REQUIRED: floor is
/// defined at the teacher point).
fn run_floor<I: Backend>(
    cfg: Config,
    cli: Cli,
    device: I::Device,
) -> Result<(), Box<dyn std::error::Error>> {
    let output = cli.output.as_ref().ok_or("--output is required in floor mode")?;
    let checkpoint = cli.checkpoint.as_ref().ok_or("--checkpoint required in floor mode")?;

    let head_section = cfg.kan_head.as_ref().expect("kan_head config required");
    let head_cfg = kan_config(head_section, cfg.seed);
    let head_template: KanHead<I> = head_cfg.init::<I>(&device);
    let head = load_kan_head::<I>(&head_base(checkpoint), head_template, &device)?;
    eprintln!("loaded checkpoint: {}", head_base(checkpoint).display());

    let dataset = MeritGagesDataset::open(&cfg)?;
    let exp = cfg.experiment.as_ref().expect("experiment section required");
    let rho = exp.rho.expect("floor mode requires experiment.rho");

    let mut rng = ChaCha12Rng::seed_from_u64(cli.seed);
    let mut sampler =
        BatchSource::Shuffle(RandomSampler::new(dataset.len(), exp.batch_size, true));
    sampler.reshuffle(&mut rng);

    // Gauge uparea lookup for stratification (staid -> uparea via the gauge's
    // outlet comid). Use whatever the dataset exposes; if there is no direct
    // accessor, read uparea per gauge from the gages CSV column (check
    // dataset/gauge metadata for the available field and record the source).
    let uparea_of = build_gauge_uparea_lookup(&dataset)?;

    let (mut all_resid, mut all_staids, mut all_uparea) = (Vec::new(), Vec::new(), Vec::new());
    let mut n_days_expected: Option<usize> = None;
    let mut processed = 0usize;
    let mut total = 0usize;
    while processed < cli.windows {
        if total > 10 * cli.windows {
            return Err("retry cap exceeded — check dataset NaN coverage".into());
        }
        let idx = match sampler.next_batch() {
            Some(idx) => idx,
            None => { sampler.reshuffle(&mut rng); continue; }
        };
        total += 1;
        let staids: Vec<_> = idx.iter().map(|&i| dataset.staids()[i].clone()).collect();
        let window = dataset.time_axis().sample_rho_window(&mut rng, rho);
        let batch = dataset.collate(&staids, &window)?;
        let obs_arr = batch.observations.clone();
        let batch_staids: Vec<String> =
            batch.gauge_staids.iter().map(|s| s.as_str().to_string()).collect();
        let tensors = batch.to_tensors::<I>(&device);
        let pred_hourly = forward_eval::<I>(&cfg, &tensors, &head, &device, false, None, None);
        let daily = tau_trim_and_downsample(pred_hourly, cfg.params.tau);
        let dims = daily.dims();
        let (g, t_days) = (dims[0], dims[1]);
        *n_days_expected.get_or_insert(t_days) == t_days
            || return Err("t_days varies across windows".into());
        let pred: Vec<f32> = daily.into_data().into_vec().unwrap();

        // |pred - obs| with the SAME obs alignment as training (grad mode's
        // obs_arr[(ti + 1, gi)]); obs NaN propagates -> NaN residual (the
        // analysis nanmeans). No gauge filtering: NaN carries the information.
        for gi in 0..g {
            for ti in 0..t_days {
                let o = obs_arr[(ti + 1, gi)];
                all_resid.push((pred[gi * t_days + ti] - o).abs());
            }
        }
        all_staids.extend(batch_staids.iter().cloned());
        all_uparea.extend(batch_staids.iter().map(|s| uparea_of(s)));
        eprintln!("window {}/{} (rho {rho}, {g} gauges)", processed + 1, cli.windows);
        processed += 1;
    }

    let n_slots = all_staids.len() / processed;
    write_floor_netcdf(
        output,
        processed,
        n_slots,
        n_days_expected.unwrap(),
        &all_resid,
        &all_staids,
        &all_uparea,
        cli.seed,
        &checkpoint.display().to_string(),
    )?;
    println!(
        "wrote {} ({} windows x {} slots x {} days)",
        output.display(), processed, n_slots, n_days_expected.unwrap()
    );
    Ok(())
}
```

Adaptation notes (same drill as previous probe modes): `forward_eval` on the inner backend returns GAUGE-aggregated hourly predictions `(G, T_hours)` (it ends in `scatter_add_by_group`) — confirm and adapt the daily indexing; batch sizes vary only if the last sampler batch is short (RandomSampler drop-last=true per training — the `n_slots` division assumes constant 64; assert it). `build_gauge_uparea_lookup`: implement from whatever gauge metadata the dataset exposes (`gages` CSV has a drainage-area column — check its name; fall back to NaN uparea with a warning rather than blocking, stratification then uses obs mean flow instead and the analysis script notes it). Update the module doc-comment usage block with a floor-mode example.

- [ ] **Step 4:** `cargo build --release --bin probe_zeta_gradient && cargo test --bin probe_zeta_gradient` → clean + tests pass. Guard sweep: `cargo test --test leakance_gradcheck --test leakance_off_parity --test zeta_accum && cargo run --release --example compare_ddr_sandbox` → green + ABSOLUTE MATCH.

- [ ] **Step 5: Commit** `feat(probe): --mode floor — per-day windowed residuals for transient-floor curves`.

---

### Task 2: rho-180 config + the two floor runs

**Files:**
- Create: `config/experiments/floor_rho90.yaml`, `config/experiments/floor_rho180.yaml` (derived from `recoverability_student_a.yaml`)

- [ ] **Step 1: Generate the two configs** from the worktree root (same single-source patch pattern as the recoverability configs; base already has synthetic obs + cpu + the teacher warm-start checkpoint):

```bash
python3 - <<'EOF'
import re, pathlib
base = pathlib.Path("config/experiments/recoverability_student_a.yaml").read_text()
hdr = lambda n, note: (f"# {n} — Phase B objective-floor measurement.\n"
                       f"# GENERATED from recoverability_student_a.yaml — {note}\n"
                       f"# Spec: docs/superpowers/specs/2026-07-04-leakance-gate-program-design.md §4\n")
def patch(text, subs):
    for pat, rep in subs:
        text, n = re.subn(pat, rep, text, count=1, flags=re.M)
        assert n == 1, f"pattern not found: {pat}"
    return text
pathlib.Path("config/experiments/floor_rho90.yaml").write_text(
    hdr("floor_rho90.yaml", "rho 90 (training default); teacher head via checkpoint warm start") + base)
pathlib.Path("config/experiments/floor_rho180.yaml").write_text(
    hdr("floor_rho180.yaml", "rho 180 arm — does doubling the window shrink the post-warmup floor?")
    + patch(base, [(r"^  rho: 90$", "  rho: 180")]))
print("wrote floor_rho90.yaml floor_rho180.yaml")
EOF
```

Verify both parse (`python3 -c "import yaml,sys; yaml.safe_load(open(sys.argv[1]))" ...`). NOTE: floor mode loads the head from `--checkpoint` (CLI), not from `experiment.checkpoint` — the config's checkpoint line is inert in this mode; leave it (harmless) and note in the header if desired.

- [ ] **Step 2: Timing gate** (2 windows, ~1 min): 

```bash
cd /home/tbindas/projects/ddrs
WT=/home/tbindas/projects/ddrs/.claude/worktrees/zeta-sensitivity
mkdir -p output/floor_fix/logs
nice -n 10 $WT/target/release/probe_zeta_gradient \
  --mode floor --backend cpu \
  --config $WT/config/experiments/floor_rho90.yaml \
  --checkpoint output/recoverability/init_head \
  --windows 2 --seed 42 --output /tmp/floor_gate.nc \
  2>&1 | tee output/floor_fix/logs/gate.log
```

Expected ~20 s/window (grad mode measured 20–25 s; floor mode is cheaper — no backward). Project the full runs; if > 3× expected, stop and investigate.

- [ ] **Step 3: Full runs** (background, parallel — they're independent):

```bash
nice -n 10 $WT/target/release/probe_zeta_gradient \
  --mode floor --backend cpu \
  --config $WT/config/experiments/floor_rho90.yaml \
  --checkpoint output/recoverability/init_head \
  --windows 96 --seed 42 --output output/floor_fix/floor_rho90.nc \
  > output/floor_fix/logs/floor_rho90.log 2>&1 &
nice -n 10 $WT/target/release/probe_zeta_gradient \
  --mode floor --backend cpu \
  --config $WT/config/experiments/floor_rho180.yaml \
  --checkpoint output/recoverability/init_head \
  --windows 96 --seed 42 --output output/floor_fix/floor_rho180.nc \
  > output/floor_fix/logs/floor_rho180.log 2>&1 &
wait
```

(~35 min and ~70 min respectively at gate rates.) Verify: both .nc files exist; no error lines in logs; `abs_residual` shapes `(96, 64, 88)` and `(96, 64, 178)`.

- [ ] **Step 4: Consistency anchor:** compute mean over days ≥ 5 of the rho-90 file (one-liner below) — must reproduce ≈ 1.0 (the recoverability step-0 measurement; same instrument, so agreement within ~20% given different window draws):

```bash
cd /home/tbindas/projects/ddrs/ddrs-py && uv run python -c "
import xarray as xr, numpy as np
ds = xr.open_dataset('/home/tbindas/projects/ddrs/output/floor_fix/floor_rho90.nc')
r = ds['abs_residual'].values
print('floor(warmup=5)  =', np.nanmean(r[:, :, 5:]))
print('floor(warmup=60) =', np.nanmean(r[:, :, 60:]))"
```

If floor(warmup=5) is NOT ≈ 1.0 (say, outside [0.5, 2.0]), STOP — the instrument disagrees with the recoverability measurement; diagnose before analysis.

- [ ] **Step 5: Commit** the configs: `config: floor-measurement rho90/rho180 configs (Phase B)`.

---

### Task 3: Floor-curve analysis + decision

**Files:**
- Create: `scripts/floor_analysis.py`

- [ ] **Step 1: Write the script:**

```python
#!/usr/bin/env python3
"""Phase B floor curves: transient noise floor vs warmup, by rho and basin size.

Pre-registered (spec §4): the fix must reach floor <= 0.25 mean L1.
Decision rule: if any (warmup <= 60, rho <= 180) cell reaches the bar ->
option A (config-only). Else -> STOP; option B (state-cache hotstart) gets
its own plan informed by the decay curves printed here.

The 58 planted-reach signals contribute ~0.0076 background (measured
continuously); irrelevant at the 0.25 bar but printed for honesty.
"""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import xarray as xr

OUT = Path("/home/tbindas/projects/ddrs/output/floor_fix")
BAR = 0.25
WARMUPS = [0, 5, 10, 15, 20, 30, 45, 60, 90, 120, 150]

def curves(path):
    ds = xr.open_dataset(path)
    r = ds["abs_residual"].values          # (win, slot, day)
    ua = ds["uparea"].values               # (win, slot)
    n_days = r.shape[2]
    rows = []
    # uparea terciles over finite entries (falls back gracefully if uparea all-NaN)
    finite_ua = ua[np.isfinite(ua)]
    edges = (np.percentile(finite_ua, [33, 66]) if finite_ua.size else [np.inf, np.inf])
    strata = {
        "all": np.ones_like(ua, dtype=bool),
        "small": ua <= edges[0],
        "mid": (ua > edges[0]) & (ua <= edges[1]),
        "large": ua > edges[1],
    }
    for w in WARMUPS:
        if w >= n_days - 5:
            continue
        for name, m in strata.items():
            sel = r[:, :, w:][m[:, :, None] & np.ones((1, 1, n_days - w), bool)]
            rows.append((n_days, w, name, float(np.nanmean(sel)),
                         float(np.nanmedian(sel))))
    return rows

rows = []
for f in ["floor_rho90.nc", "floor_rho180.nc"]:
    rows += curves(OUT / f)

print(f"{'rho_days':>8} {'warmup':>6} {'stratum':>7} {'mean_L1':>9} {'median_L1':>9}")
passing = []
for n_days, w, s, mean, med in rows:
    flag = " <-- PASSES BAR" if (s == "all" and mean <= BAR) else ""
    print(f"{n_days:>8} {w:>6} {s:>7} {mean:>9.4f} {med:>9.4f}{flag}")
    if s == "all" and mean <= BAR:
        passing.append((n_days, w, mean))

# Per-day decay curve (mean over windows/slots) -> transient e-folding estimate.
fig, ax = plt.subplots(figsize=(9, 5))
for f, label in [("floor_rho90.nc", "rho 90"), ("floor_rho180.nc", "rho 180")]:
    r = xr.open_dataset(OUT / f)["abs_residual"].values
    day_mean = np.nanmean(r, axis=(0, 1))
    ax.semilogy(day_mean, label=label)
ax.axhline(BAR, color="r", ls="--", label=f"bar {BAR}")
ax.axhline(0.0076, color="g", ls=":", label="continuous residual (plants)")
ax.set(xlabel="post-trim day in window", ylabel="mean |residual| m^3/s",
       title="Hotstart transient decay (teacher weights, self-generated obs)")
ax.legend()
fig.savefig(OUT / "floor_decay.png", dpi=200, bbox_inches="tight")

print("\n" + "=" * 60)
if passing:
    n_days, w, mean = min(passing, key=lambda t: (t[0], t[1]))
    # effective loss-days per window after trim:
    print(f"DECISION: OPTION A — rho yielding {n_days} post-trim days with "
          f"warmup {w} reaches floor {mean:.4f} <= {BAR}")
    print(f"loss-days per window: {n_days - w} (sample-efficiency note for Phase C)")
else:
    best = min((x for x in rows if x[2] == "all"), key=lambda t: t[3])
    print(f"DECISION: OPTION B REQUIRED — best config-only floor is "
          f"{best[3]:.4f} (rho-days {best[0]}, warmup {best[1]}) > {BAR}. "
          f"STOP: write the state-cache plan using the decay curves above.")
print(f"plot -> {OUT / 'floor_decay.png'}")
```

- [ ] **Step 2: Run** (`cd /home/tbindas/projects/ddrs/ddrs-py && uv run python <worktree>/scripts/floor_analysis.py 2>&1 | tee /home/tbindas/projects/ddrs/output/floor_fix/logs/analysis.log`). Read the DECISION line.

- [ ] **Step 3: Commit** `feat(scripts): Phase B floor-curve analysis (pre-registered 0.25 bar + decision rule)`.

---

### Task 4: DECISION GATE

- [ ] **If DECISION = OPTION A:** proceed to Task 5.
- [ ] **If DECISION = OPTION B REQUIRED:** STOP this plan here. Deliverables so far (floor curves, decay plot, stratification) ARE the Phase-B1 output; write `docs/2026-07-XX-floor-curve-findings.md` (Task 6's structure, verdict = "config-only insufficient"), commit, and surface to the human with the measured decay lengths — the state-cache design (spec §4 option B: continuous-run state cache, ~1.3 GB, dataset plumbing, staleness measurement) gets its own brainstorm+plan informed by these numbers. Do not improvise the state cache from this plan.

---

### Task 5: Validate the option-A fixed recipe (only on OPTION A)

- [ ] **Step 1:** Take the chosen (rho, warmup) from Task 3's DECISION. Generate `config/experiments/floor_validated.yaml` from `floor_rho90.yaml` with the chosen `rho:` and `warmup:` patched in (same python patch pattern as Task 2, asserting one substitution each).

- [ ] **Step 2: Independent validation run** — different seed (send `--seed 1042`), 48 windows, same teacher point:

```bash
nice -n 10 $WT/target/release/probe_zeta_gradient \
  --mode floor --backend cpu \
  --config $WT/config/experiments/floor_validated.yaml \
  --checkpoint output/recoverability/init_head \
  --windows 48 --seed 1042 --output output/floor_fix/floor_validated.nc \
  2>&1 | tee output/floor_fix/logs/floor_validated.log
```

Then compute the floor at the CHOSEN warmup only (one-liner as in Task 2 Step 4). PASS iff ≤ 0.25 on the fresh seed. If it fails on the fresh seed but passed in Task 3, the window sample was lucky — report both numbers, pick the next-larger (rho, warmup) cell that passed, and re-validate once; two failures ⇒ treat as OPTION B (Task 4's stop).

- [ ] **Step 3:** Also record the **real-obs floor context**: the fix will be used with USGS obs in Phase C, where obs noise adds to the budget. No new run needed — state in the findings that the transient component is now ≤ 0.25 and obs noise is irreducible/shared by both Phase-C cells.

- [ ] **Step 4: Commit** `config: validated floor-fixed objective recipe (rho X, warmup Y)`.

---

### Task 6: Findings report

**Files:**
- Create: `docs/2026-07-XX-floor-fix-findings.md` (XX = run date)

- [ ] **Step 1: Write it** in the established experiment-report structure (mirror `docs/2026-07-04-synthetic-recoverability-findings.md`): §1 Hypothesis (the floor is warmup-governed and a config-level trim can reach 0.25; pre-registered bar + decision rule), §2 What was changed (floor mode + configs + analysis script, commit SHAs; no Backward/training-path changes; guards green), §3 The experiment (2 runs × 96 windows, teacher point, synthetic obs; the 0.0076/1.017 anchors), §4 Did it pass (the floor table verbatim from analysis.log, the decay plot, the DECISION line, validation-seed result), §5 Conclusions (chosen recipe + sample-efficiency cost: loss-days per window; decay length by basin size — the generalizable ddrs finding), §6 Next steps (Phase C consumes the recipe; CLAUDE.md warmup guidance update; state-cache only if B was triggered), §7 Raw output, §8 Reproduce.

- [ ] **Step 2:** Final guard sweep (`cargo test` + sandbox ABSOLUTE MATCH). Commit `docs(findings): objective floor fix — curves, decision, validated recipe`.

---

## Self-review (done at write time)

- **Spec coverage:** §4 B1 (curve at warmup {5,15,30,60}@rho90 + {90}@rho180, uparea stratification) → Tasks 1–3 (the post-hoc trim trick supersedes per-warmup runs — one run per rho measures ALL warmups, strictly better than the spec's enumeration); B2 decision rule → Tasks 3–4; option A → Task 5; option B → explicit STOP gate (Task 4) rather than speculative code, honoring "each phase produces working software" (B1's curves are a complete deliverable on the B path); B3 bar (≤ 0.25, rerun-noise 0 on CPU) → Tasks 3/5; "explicitly general finding" → Task 6 §5.
- **Placeholder scan:** `docs/2026-07-XX-...` is the run-date convention used by prior findings docs (named at write time), not a content placeholder. Known unknowns flagged with resolution paths: netcdf 2-D string API (Task 1 Step 2), uparea lookup source (Task 1 Step 3, NaN fallback defined), grad-mode t_days consistency assert.
- **Type consistency:** `write_floor_netcdf(path, n_windows, n_slots, n_days, abs_residual, gauge_staids, uparea, seed, label)` matches between the test (Step 1) and impl (Step 2); `--mode floor` flags match between Tasks 1–2–5; file names `floor_rho90.nc`/`floor_rho180.nc` consistent across Tasks 2–3.

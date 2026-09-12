# Phase B2: State-Cache Hotstart Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Initialize every training window from a cached continuous-run routing state instead of the synthetic hotstart, eliminating the large-river initial-condition plateau (~4 m³/s, non-decaying) that Phase B1 proved no warmup can cure — validated by the pre-registered floor bar (≤ 0.25 mean L1 at the teacher point).

**Architecture:** A cache-writer mode in the probe binary runs one continuous forward over the training window and saves the per-reach discharge state at every day boundary (~1.3 GB f32). A new optional config key `experiment.state_cache` makes the dataset load the cache and `collate` attach the window-start state; the training/probe forward paths seed the router from it instead of calling the hotstart heuristic. With the key absent, behavior is BYTE-IDENTICAL to today (parity-tested) — the cache is opt-in per experiment.

**Why this design (from B1's data):** the floor is entirely large-river IC error (median gauge 0.075; large-uparea stratum plateaus at 3.7–4.1 m³/s through day 87 and is still ~2.7 at day 178). Routing memory on mainstems exceeds any feasible window, so the ONLY fix is starting windows from realistic in-sequence states. Staleness (cache generated under one head vs the head evolving during training) is bounded empirically by Task 6's validation — if the teacher-point floor passes the bar with a teacher-generated cache, staleness at nearby heads is second-order (states are dominated by Q′ forcing, not head params); Phase C's findings must re-verify on its own runs' loss curves.

**Tech Stack:** Rust (ddrs worktree `/home/tbindas/projects/ddrs/.claude/worktrees/zeta-sensitivity`, branch `worktree-zeta-sensitivity`), CPU/NdArray for all runs. Python (ddrs-py venv) for validation analysis.

**Spec:** `docs/superpowers/specs/2026-07-04-leakance-gate-program-design.md` §4 option B; decision evidence `output/floor_fix/logs/analysis.log` + `docs` findings from Phase B1 (B3 commit `ff7c59a`).

**Hard invariants (this touches the training path — maximum care):**
- With `experiment.state_cache` UNSET: every code path byte-identical to master behavior. Proven by a new parity test + the existing guard suites (`leakance_gradcheck`, `leakance_off_parity`, `zeta_accum`) + `compare_ddr_sandbox` ABSOLUTE MATCH after every task.
- No autograd `Backward` impl changes. The injected initial state is a CONSTANT tensor (no grad path to the head) — it replaces the hotstart values, which were also head-independent. Autograd topology unchanged.
- f32 throughout.

**Key existing code (read before Task 1):** `src/routing/utils.rs` (the `hotstart` function — what exactly it produces and where it's consumed), `src/routing/mmc.rs::setup_inputs` (how the initial state enters the router), `src/training/forward.rs` (`forward` + `probe_forward` + `forward_eval` — which call hotstart where), `src/data/dataset.rs::collate` (+ `RoutingTensors`), `src/bin/probe_zeta_gradient.rs` (`run_teacher`'s chunked continuous loop with `carry_state` — the cache writer mirrors it).

**What the cached state IS:** the Muskingum-Cunge router's carried state is the previous timestep's per-reach discharge vector `Q_{t-1}`. A training window starting at day `d` (hour 0) needs `Q` at day-`d`-hour-0 from a continuous run. Cache layout: netCDF, dims `(day, COMID)` f32, day 0 = the training axis start date (attr `day0`), COMIDs = the eval-union network order (the same `divide_comids` order every batch subsets from).

---

### Task 1: Cache writer — `--mode state-cache` in the probe binary

**Files:**
- Modify: `src/bin/probe_zeta_gradient.rs`

- [ ] **Step 1: Failing writer test** (existing test mod):

```rust
    #[test]
    fn state_cache_netcdf_roundtrip() {
        let dir = std::env::temp_dir().join("probe_state_cache_test");
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("cache_test.nc");
        let _ = std::fs::remove_file(&path);

        // 3 days x 2 reaches of day-boundary discharge states.
        let q = vec![1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0];
        let comids = vec![10i64, 20];
        write_state_cache_netcdf(&path, 3, &comids, &q, "1981-10-01", "ckpt").unwrap();

        let f = netcdf::open(&path).unwrap();
        let v = f.variable("q_state").unwrap();
        assert_eq!(
            v.dimensions().iter().map(|d| d.len()).collect::<Vec<_>>(),
            vec![3, 2]
        );
        let vals: Vec<f32> = v.get_values(..).unwrap();
        assert_eq!(vals, q);
        let c = f.variable("COMID").unwrap();
        let cv: Vec<i64> = c.get_values(..).unwrap();
        assert_eq!(cv, comids);
        let d0: String = f.attribute("day0").unwrap().value().unwrap().try_into().unwrap();
        assert_eq!(d0, "1981-10-01");
    }
```

(Adapt the attribute-read idiom to the netcdf crate's actual API — the file's other tests show the pattern.) Run → FAIL.

- [ ] **Step 2: Implement the writer:**

```rust
/// State cache: per-reach discharge at each day boundary of a continuous
/// run. `q_state[(d, r)]` = Q entering day d (hour 0) at reach column r.
/// Day 0 = `day0` (the training axis start). COMID order = the continuous
/// run's network order; consumers index by COMID, never by position.
fn write_state_cache_netcdf(
    path: &Path,
    n_days: usize,
    comids: &[i64],
    q_state: &[f32],
    day0: &str,
    checkpoint_label: &str,
) -> Result<(), Box<dyn std::error::Error>> {
    debug_assert_eq!(q_state.len(), n_days * comids.len());
    let mut file = netcdf::create(path)?;
    file.add_dimension("day", n_days)?;
    file.add_dimension("COMID", comids.len())?;
    file.add_attribute("day0", day0)?;
    file.add_attribute("checkpoint", checkpoint_label)?;
    {
        let mut v = file.add_variable::<i64>("COMID", &["COMID"])?;
        v.put_values(comids, ..)?;
    }
    {
        let mut v = file.add_variable::<f32>("q_state", &["day", "COMID"])?;
        v.put_values(q_state, ..)?;
        v.put_attribute("units", "m^3/s")?;
    }
    Ok(())
}
```

- [ ] **Step 3: Add `Mode::StateCache`** (`"state-cache"`), `ConfigMode::Testing` (continuous window like teacher mode), routed in both backend arms to `run_state_cache::<I>`, which is `run_teacher`'s chunk loop MINUS plants/obs/zeta PLUS state capture: after each chunk's `forward_eval`, extract the LAST hourly column of each day in the chunk... **adaptation duty:** the clean way is to capture the day-boundary state = the routed discharge at the last hourly step before each day boundary. Read how `run_perturb` reconstructs `(G, T_hours)` — but the cache needs PER-REACH discharge, not gauge-aggregated. `forward_eval` returns gauge-aggregated output, so the cache writer CANNOT use it directly. Instead mirror `run_teacher`'s loop but call the engine path that exposes per-reach discharge: check `MuskingumCunge::forward`'s return — it returns the routed per-reach matrix BEFORE `scatter_add_by_group` (see `src/training/forward.rs:395-403`: `runoff` is per-reach `(n_reaches, T)`; the gauge aggregation happens after). Add a small `pub fn forward_eval_reaches<I>(...) -> Tensor<I, 2>` in `src/training/forward.rs` that is `forward_eval` minus the final scatter (share the body via a private helper returning the pre-scatter tensor; `forward_eval` becomes helper + scatter — behavior-identical, verified by the parity guards). The cache writer slices hourly column `24*d - 1` (state entering day d is the discharge after the last step of day d-1; day 0's state row = the hotstart values used at window start — write the run's initial hotstart values for row 0).
- [ ] **Step 4:** CLI: reuses `--config` (a floor/teacher-style testing-window config), `--checkpoint`, `--output`, `--eval-days`. Pre-flight: refuse to overwrite an existing non-empty output file. Build + unit tests + guard suites + sandbox ABSOLUTE MATCH.
- [ ] **Step 5: Commit** `feat(probe): --mode state-cache — continuous-run day-boundary discharge cache + forward_eval_reaches split`.

---

### Task 2: Config key + dataset plumbing

**Files:**
- Modify: `src/config.rs` (optional `experiment.state_cache: Option<PathBuf>`)
- Modify: `src/data/dataset.rs` (load cache lazily; `collate` attaches the window-start state row for the batch's comids)
- Modify: `src/data/dataset.rs::RoutingTensors` (new field `initial_state: Option<Tensor<I, 1>>` — per-reach Q at window start, batch-network order)
- Test: `tests/state_cache_dataset.rs` (new)

- [ ] **Step 1: Failing test** — synthetic cache netCDF (write with the Task-1 writer via a tiny fixture, or hand-write with the netcdf crate in the test) + a config with `state_cache` set: `collate` must return a batch whose `initial_state` matches the cache row for the window's start day, reordered to the batch's `divide_comids`; with the key unset, `initial_state` is `None` and every other field byte-identical (assert equality of `q_prime` etc. between a with-key and without-key dataset on the same window).
- [ ] **Step 2: Implement.** Cache loading: open once (netcdf), read `COMID` + `day0`; per collate, compute `day_idx = (window.start - day0).num_days()` (hard error if out of range — a window before/after the cache is a config bug, never silent), read the row, map COMID→value, then emit values in the batch's `divide_comids` order (missing comid → hard error listing the first few — the cache must cover the training network). `to_tensors` lifts it to a `Tensor<I,1>`.
- [ ] **Step 3:** Tests pass. Config-parse test: `state_cache` absent → `None` (add to the config test mod).
- [ ] **Step 4: Commit** `feat(data): experiment.state_cache — window-start routing state attached at collate`.

---

### Task 3: Forward-path consumption

**Files:**
- Modify: `src/routing/mmc.rs` (accept an optional initial state in `setup_inputs` — exact seam depends on how hotstart currently enters; read first)
- Modify: `src/training/forward.rs` (`forward`, `probe_forward`, `forward_eval` thread `tensors.initial_state` through; `None` → existing hotstart, unchanged)
- Test: `tests/state_cache_forward.rs` (new)

- [ ] **Step 1: Failing tests** (use `tests/common` mock harness like `zeta_accum` does):
  1. `initial_state_replaces_hotstart`: a 5-reach chain routed with an explicit initial state differs from the hotstart route at t=1 exactly as the state difference implies (first output column reflects the injected Q, not the hotstart heuristic).
  2. `no_state_is_byte_identical`: `initial_state: None` routes byte-identically to current master behavior (route the mock chain twice — once via the old signature path if kept, once via None — assert exact equality).
  3. `gradients_unaffected_by_constant_state`: on a losing chain with leakance, `loss.backward()` gradients w.r.t. the head-derived leaves are FINITE and the injected state tensor requires no grad (assert `initial_state` is constructed without `require_grad`). Guard: `cargo test --test leakance_gradcheck` must stay green (the analytical backward never sees a new node).
- [ ] **Step 2: Implement.** The state is data, not a parameter: wrap to the autograd backend with `Tensor::from_inner` (no `require_grad`), exactly how `q_prime` enters. KEEP the hotstart function untouched — selection happens at the call site.
- [ ] **Step 3:** All three tests pass + full guard sweep + sandbox ABSOLUTE MATCH (the sandbox path passes `None`).
- [ ] **Step 4: Commit** `feat(routing): optional window-start state injection (hotstart fallback unchanged)`.

---

### Task 4: Generate the cache (teacher point)

- [ ] **Step 1:** `WT=<worktree>`; cwd main tree; the cache config = `config/experiments/floor_rho90.yaml`'s data sources but a TESTING-window config covering the training axis — reuse `config/experiments/recoverability_measure.yaml` (testing window 1981-09-30..1995-10-01, synthetic obs, cpu):

```bash
nice -n 10 $WT/target/release/probe_zeta_gradient \
  --mode state-cache --backend cpu \
  --config $WT/config/experiments/recoverability_measure.yaml \
  --checkpoint output/recoverability/init_head \
  --eval-days 999999 \
  --output output/floor_fix/state_cache_teacher.nc \
  2>&1 | tee output/floor_fix/logs/state_cache.log
```

(~2.5 h at teacher-run rates; background + poll.) Verify: dims (5115ish, 64892), no NaN/negative rows (`q_state.min() >= 0`), spot-check a mainstem reach's state series against the teacher eval's routed discharge at matching days (values should track — same weights, same forcing).

- [ ] **Step 2:** No commit (runtime artifact); record numbers.

---

### Task 5: Floor validation with the cache (the pre-registered bar)

- [ ] **Step 1:** Generate `config/experiments/floor_rho90_cached.yaml` = `floor_rho90.yaml` + `state_cache: /home/tbindas/projects/ddrs/output/floor_fix/state_cache_teacher.nc` under `experiment:` (same python patch pattern; assert one substitution).
- [ ] **Step 2:** Floor run, fresh seed:

```bash
nice -n 10 $WT/target/release/probe_zeta_gradient \
  --mode floor --backend cpu \
  --config $WT/config/experiments/floor_rho90_cached.yaml \
  --checkpoint output/recoverability/init_head \
  --windows 96 --seed 1042 --output output/floor_fix/floor_cached.nc \
  2>&1 | tee output/floor_fix/logs/floor_cached.log
```

- [ ] **Step 3:** Analysis: extend `scripts/floor_analysis.py` with an optional third input (or a small companion snippet) printing the cached-floor table (all strata, warmup {0,5,15,30}) next to the uncached rho-90 numbers. **PASS: all-strata mean ≤ 0.25 at warmup ≤ 5.** Expected if the theory is right: the LARGE stratum collapses from ~3.8 to near the small-stratum level; warmup becomes nearly irrelevant. If it fails: per-stratum numbers say whether the residual is cache staleness (uniform floor) or a day0-indexing bug (day-0-spike pattern) — diagnose before any retry; report either way.
- [ ] **Step 4: Commit** config + analysis extension: `feat(floor): cached-hotstart validation (pre-registered 0.25 bar)`.

---

### Task 6: Findings + CLAUDE.md note

- [ ] **Step 1:** Write `docs/2026-07-XX-floor-fix-findings.md` (run date) — the Phase B report covering BOTH halves: B1 curves (the plateau discovery, the OPTION B decision verbatim) and B2 (state cache design, validation table, PASS/FAIL vs the bar). Structure: hypothesis / what was changed / experiment / pass-fail / conclusions (incl. "every prior ddrs training run carried this floor" and the staleness caveat for Phase C) / next steps (Phase C consumes `state_cache`; refresh policy only if Phase C loss curves demand it) / raw output / reproduce.
- [ ] **Step 2:** Add a short CLAUDE.md subsection under the training notes: what `experiment.state_cache` is, when to regenerate it (config's forcing/network changed), and that warmup alone cannot fix large-river IC error (pointer to the findings).
- [ ] **Step 3:** Final guard sweep + sandbox. Commit docs.

---

## Self-review (done at write time)

- **Spec coverage:** spec §4 option B fully covered (cache generation Task 4, dataset plumbing Task 2, staleness measured via Task 5's teacher-point validation + Phase-C follow-up note in Task 6); pre-registered bar unchanged (0.25); B1's STOP-gate deliverables already committed (`ff7c59a`).
- **Invariant protection:** parity with key-unset is tested at three layers (dataset Task 2, forward Task 3, sandbox every task); no Backward impls; state enters as a constant like `q_prime`.
- **Known unknowns flagged with resolution paths:** exact hotstart seam in `mmc.rs` (Task 3 reads first), netcdf attr-read idiom (Task 1), `forward_eval` pre-scatter split (Task 1 Step 3 — behavior-identical refactor guarded by the full suite), cache day indexing convention (day-boundary definition stated; Task 4 spot-check verifies against the teacher eval).
- **Type consistency:** `write_state_cache_netcdf(path, n_days, comids, q_state, day0, label)` matches test/impl; `initial_state: Option<Tensor<I,1>>` named identically in Tasks 2–3; config key `experiment.state_cache` throughout.

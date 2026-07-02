# Leakance × hourly-disaggregation experiment — session handoff (2026-07-01)

Purpose: hand another session everything needed to continue the leakance feasibility
experiment. Read this top-to-bottom; the **"What's left"** section is the action list.

Spec:  `docs/superpowers/specs/2026-06-29-leakance-hourly-feasibility-design.md`
Plan:  `docs/superpowers/plans/2026-06-29-leakance-hourly-feasibility.md`
Branch: `hourly-forcings`   HEAD at handoff: `2cdd341`

---

## 0. RESOLUTION (2026-07-01, later session) — the "disagg no-op" was a STALE BINARY

**Root cause of §5c is NOT in the forcing plumbing.** The July runs invoked the
**installed** `~/.cargo/bin/ddrs`, which was dated **June 3** — *before* the
disaggregation feature (June 19, `04aef30`) and *before* leakance (June 29,
`496c8b1`). So the hourly-ON cell silently:
- ignored the `disaggregation:` block (serde skips unknown fields) → flat repeat-24,
- ignored `use_leakance` → no leakance,
- wrote **flat** `.mpk` checkpoints (the pre-resume format).

That makes hourly-ON == daily-ON byte-identical (both flat-daily, both no leakance).
The manifest's `git.sha = 2cdd341` was stamped from `.git` at runtime, **not** the
binary — which is exactly why §5c looked like a code bug at HEAD.

**Three independent confirmations:**
1. `~/.cargo/bin/ddrs` mtime = **2026-06-03**; disagg landed 2026-06-19.
2. July checkpoints are **flat files** (`epoch_5_mb_35.mpk`); current code
   (`driver::train`) writes **directories** (`epoch_5_mb_9/head.mpk`, as the
   June-23 working-disagg run did).
3. July head = **103459 B** (no disagg); June disagg head = 107178 B. After the
   fix, a fresh head = **107320 B** = disagg + 6 leakance params (June's 107178
   + 3 extra output cols for K_D/d_gw/leakance_factor). All accounted for.

**The fix:** refresh the installed binary — `cargo install --path .` (or
`cp target/release/ddrs ~/.cargo/bin/ddrs`). Documented as the "STALE-BINARY
TRAP" in `CLAUDE.md` (§`ddrs` CLI). The config files (`leakance_hourly_on.yaml`
/ `leakance_daily_on.yaml`) were **correct all along**.

**Re-run DONE** — `2026-07-01T13-43-32Z-train-and-test` (current binary, precip
verified: `AORC precip store: 290878 catchments`, directory checkpoints,
head 107320 B with disagg). Valid paired comparison vs the June leakance-OFF
control `2026-06-23T02-49-12Z-conus-hourly` (same seed 42, same eval window
1995/10–2010/09, same 2365 gauges, **both with active disagg**):

| | NSE (med) | KGE (med) | NSE (mean) |
|---|---|---|---|
| leakance OFF (June) | 0.7153 | 0.7104 | 0.5761 |
| leakance ON  (new)  | 0.7145 | 0.7150 | 0.5773 |
| Δ (ON − OFF)        | **−0.0008** | **+0.0046** | +0.0012 |

**NSE does not improve** (flat, within noise); **KGE +0.0046** — the volume/
variance-correction signature leakance is meant to give. And leakance is
**identifiable & active, not collapsed** (`dump_parameters` on the new head):
`K_D` median 1.003e-6 **frac@ceil=100%** (wants more exchange), `leakance_factor`
median 0.327 (interior, nowhere near 0), `d_gw` median 0.294 m (spatially
varying −0.02…0.78). So the §5a "identifiable" picture reproduces on a *valid*
run, and the KGE gain is backed by real leakance — but the all-gauge NSE verdict
is neutral. Next: re-run **daily-ON** (also stale-binary-tainted) for the true
2×2 interaction, then the losing-stream subset GO/NO-GO (§7 steps 3–6).

**Tooling note:** `dump_parameters --checkpoint` appends `.mpk`, so for the
directory checkpoint format pass the **head base** (`…/epoch_E_mb_M/head`), not
the directory (`…/epoch_E_mb_M`).

---

## 1. The hypothesis we tried to test

DDR added a **leakance** term (groundwater–surface-water water loss) to routing, then
**reverted it** because on **daily-scale forcing** it was *unidentifiable* and *didn't help
metrics* — the learned exchange collapsed to **sub-0.01 m³/s** (physically negligible values
that added nothing).

```
zeta   = leakance_factor · area_z · K_D · (depth − d_gw)      # subtracted from routing RHS b
area_z = (p · depth)^q_eps · length                          # plan-view; depth shared w/ geometry
```

**Hypothesis:** the **hourly disaggregation signal** (precip-driven, mass-preserving
daily→hourly forcing) restores the sub-daily **depth** dynamic range that `zeta ∝ (depth − d_gw)`
needs, making leakance **identifiable and helpful** where daily forcing made it neither.

**Decisive test (per spec):** a 2×2 — forcing (hourly disagg vs flat daily) × leakance (on/off)
— evaluated on the **losing-stream subset** (gauges where summed-Q′ baseline over-predicts,
ratio > 1). GO = NSE-or-KGE gain on the subset **AND** learned leakance clears the ~0.01 floor
(non-collapse) **AND** the effect is present under hourly but absent/weaker under daily
(the interaction).

---

## 2. What was built (complete, verified, committed)

The flag-gated, gradient-exact leakance **testbed** — 13 TDD tasks, `1a2e75d..2cdd341`:

- `src/config.rs` — `params.use_leakance` flag + `K_D`/`d_gw`/`leakance_factor` ranges;
  rejects `use_cuda_graphs && use_leakance`.
- `src/routing/leakance.rs` — `zeta_forward` + `zeta_backward` (analytical grads,
  finite-diff verified).
- `src/routing/mmc_op.rs` — `TimestepLeakanceOp: Backward<I,8>` extending the fused MC
  timestep backward with zeta; `forward_chain_inner` gated (`None` = byte-identical).
- `src/routing/mmc.rs`, `src/training/forward.rs` — optional leakance params on
  `SpatialParameters`, dispatch in `route_timestep`, threaded from the KAN head HashMap.
- `src/dump_parameters.rs` — now exports learned `K_D`/`d_gw`/`leakance_factor` per COMID +
  prints a floor/ceiling-collapse summary (`2cdd341`).

**Test gate — ALL PASS** (re-run: `bash /tmp/leakance_test_gate.sh` is gone; regenerate below):
```
cargo test --lib                                   # 172 passed
cargo test --test leakance_gradcheck               # 8/8 (incl. K_D, d_gw, leakance_factor)
cargo test --test sp8_gradcheck                    # 5/5 (fused-op extraction didn't regress)
cargo test --test leakance_off_parity              # 3/3 (byte-identical when off)
cargo run --release --example compare_ddr_sandbox  # ABSOLUTE MATCH (1.5e-5 m³/s)
```
The leakance **implementation is sound and gradient-exact**. The problem below is in the
**forcing plumbing**, not leakance.

---

## 3. How the experiment was run (gotchas the next session needs)

**Permissions.** `ddrs run` is gated by the auto-mode classifier. The user added
`.claude/settings.local.json` → `permissions.allow: ["Bash(ddrs:*)"]`. An agent CANNOT write
that file itself (self-modification denial) — the user must.

**Workspace flag gotcha.** `--workspace` takes the path to the **`.ddrs` directory itself**
(not its parent), and `--config <path>` otherwise derives the workspace as
`<config_dir>/.ddrs`. Experiment configs live in `config/experiments/`, so you MUST pass the
root `.ddrs` explicitly or it looks for `config/experiments/.ddrs`:

```bash
cd ~/projects/ddrs
ddrs run --config config/experiments/leakance_hourly_on.yaml \
         --workspace /home/tbindas/projects/ddrs/.ddrs \
         --workflow train-and-test
```

Baseline + GPU-probe caches in the root `.ddrs` are reused (same data_sources as the June runs).

---

## 4. The runs (inventory)

| role | run id | forcing (intended) | leakance | eval-pred md5 |
|---|---|---|---|---|
| hourly-ON (mine) | `2026-07-01T01-07-17Z-train-and-test` | hourly disagg + precip | ON | `52ec721…` |
| daily-ON (mine)  | `2026-07-01T01-58-03Z-train-and-test` | flat repeat-24 | ON | `52ec721…` |
| hourly-OFF (June)| `2026-06-23T02-49-12Z-conus-hourly-train-and-test` | hourly disagg + precip | OFF | `c644d93…` |
| daily-OFF        | `2026-06-05T01-41-16Z-train-and-test` | flat daily | OFF | — |

Re-derive the hashes / params:
```bash
cd ~/projects/ddrs
for r in 2026-07-01T01-07-17Z-train-and-test 2026-07-01T01-58-03Z-train-and-test \
         2026-06-23T02-49-12Z-conus-hourly-train-and-test; do
  echo -n "$r  "; find .ddrs/runs/$r/eval/predictions.zarr/predictions -type f ! -name '*.json' \
    -exec cat {} + 2>/dev/null | md5sum
done
# learned leakance params (both ON runs identical):
target/release/dump_parameters --config config/experiments/leakance_hourly_on.yaml \
  --checkpoint .ddrs/runs/2026-07-01T01-07-17Z-train-and-test/checkpoints/epoch_5_mb_35 \
  --output /tmp/kp.nc 2>&1 | grep learned
```

---

## 5. Key findings

### 5a. Leakance IS identifiable — the DDR revert failure did NOT reproduce  ✅
Learned params over 346,321 CONUS reaches (identical in both ON runs):

| param | median | pinning | read |
|---|---|---|---|
| `K_D` (1/s) | 1.005e-6 | **frac@ceil = 100%** | saturated at the *upper* bound — wants MORE loss |
| `leakance_factor` | 0.48 | interior | gate ~half-open everywhere, nowhere near 0 |
| `d_gw` (m) | 0.085 | varies −0.02…0.38 | spatially non-trivial |

This is the **inverse** of DDR's "sub-0.01 collapse to nothing." Even under (effectively)
flat-daily forcing, leakance trained to **active, non-trivial** values.

### 5b. Leakance changes predictions and lifts KGE  ✅ (partial)
- hourly-ON eval (`52ec721`) **differs** from June precip+L1 (`c644d93`) → leakance is doing
  something real.
- All-gauge median: **NSE 0.6990, KGE 0.7234** vs June precip+L1 **NSE 0.7152, KGE 0.7106**.
  i.e. adding leakance moved **KGE up (+0.013)**, NSE down (−0.016) — the volume-correction
  (KGE-β) signature leakance is meant to give. (Losing-stream *subset* skill not yet computed.)

### 5c. BUG — the hourly-vs-daily forcing contrast is INVALID  ❌
**hourly-ON and daily-ON produced byte-identical eval predictions (`52ec721`).** Two runs whose
*forcing* differs cannot coincide byte-for-byte → the **disaggregation head was a no-op**;
hourly-ON effectively ran flat-daily. So the 2×2's forcing axis never varied.

Ruled out (so the bug is NOT here):
- Config parses correctly: `kan_head.disaggregation.is_some() = true`, `use_precip = true`,
  no stale `experiment.checkpoint`.
- Head build correct: `kan_config(section).init()` yields `head.disagg = Some` (tested in isolation).
- `forward.rs:199–208` applies disagg when `head.disagg` is `Some` — intact; the T9 leakance
  threading sits *after* it.
- Checkpoints for both ON runs are the **same size** (103459 B) — the saved head carries no
  disagg weight, despite the head being built with disagg. Contradiction ⇒ the disagg head is
  built but its effect never reaches the saved/evaluated forward.

Remaining suspects (need runtime instrumentation): the **dataset** not populating
`tensors.precip_hourly` / `tensors.q_prime_daily` with real data (so disagg conditions on
nothing and returns ≈flat), or the disagg output collapsing to uniform softmax at runtime.
Note the June precip run (pre-leakance, same branch) used disagg fine, so this may be a
regression introduced somewhere in `1a2e75d..2cdd341` OR a config/dataset-path difference for
`config/experiments/*.yaml` vs the June `.ddrs/runs/.../config.yaml`.

---

## 6. Was the hypothesis valid?

**Partially answered; the decisive part is still open.**

- **"Leakance is unidentifiable" (the DDR failure): REFUTED.** ✅ It trained to active,
  non-trivial values (K_D at ceiling, factor ≈ 0.48) — even under flat-daily forcing. On the
  identifiability axis the result is positive and robust.
- **"Hourly forcing unlocks leakance where daily doesn't" (the interaction): UNTESTED.** ❌
  The disagg no-op means hourly forcing never actually differed from daily, so the interaction
  cannot be evaluated from these runs.
- **"Leakance helps skill" (losing-stream subset): INCONCLUSIVE.** All-gauge KGE rose, but the
  subset GO/NO-GO was deliberately NOT computed on invalid-forcing data (would be misleading).

Bottom line: leakance is worth continuing — it is identifiable and volume-correcting — but the
**central hourly-vs-daily claim requires fixing the forcing bug and re-running.**

---

## 7. What's left (action list, in order)

1. **Root-cause the disagg no-op.**
   - Instrument `src/training/forward.rs` (print `head.disagg.is_some()` at runtime; assert
     `q_prime_hourly != tensors.q_prime` when disagg is on) and/or `src/data/dataset.rs`
     (is `precip_hourly` non-empty / non-zero? is `q_prime_daily` populated?).
   - Fast repro (no full train): 
     `ddrs run --config config/experiments/leakance_hourly_on.yaml --workspace /home/tbindas/projects/ddrs/.ddrs --workflow train --max-mini-batches 2`
   - Bisect against June: `git diff <June-commit> HEAD -- src/data/dataset.rs src/training/forward.rs`
     to spot any regression in the precip/disagg forcing path.
2. **Verify the fix:** a short hourly run whose eval predictions **differ** from a flat-daily run
   (byte-compare as in §4).
3. **Re-run the 2×2** with corrected forcing:
   `leakance_hourly_on.yaml` and `leakance_daily_on.yaml` (both use `--workspace …/.ddrs`).
4. ~~**(Recommended) add an eval-time per-reach `zeta` accumulator**~~ **DONE (2026-07-01,
   later session).** `evaluate` now accumulates per-reach mean `|zeta|` + signed `zeta_net`
   when leakance is active and writes them to `<run_dir>/kan_parameters.nc` (the exact file
   `maybe_load_zeta` reads). Automatic in `ddrs run --workflow train-and-test`; for existing
   checkpoints: `target/release/eval --config <cfg> --checkpoint <ckpt_dir> --output <zarr>
   --zeta-output <run_dir>/kan_parameters.nc`. Correctness: `tests/zeta_accum.rs` (the
   accumulated zeta equals the headwater `q_next` difference exactly). See CLAUDE.md
   §Leakance "Eval-time zeta diagnostic".
5. **Run the subset analysis** (under DDR's uv venv):
   ```bash
   cd ~/projects/ddr && uv run python ~/projects/ddrs/scripts/leakance_subset_analysis.py \
     --hourly-on  <new hourly-ON id> --daily-on <new daily-ON id> \
     --hourly-off 2026-06-23T02-49-12Z-conus-hourly-train-and-test \
     --daily-off  2026-06-05T01-41-16Z-train-and-test
   ```
6. **Write the findings doc** (`docs/2026-XX-XX-leakance-hourly-findings.md`), mirroring
   `docs/2026-06-23-precip-disaggregation-findings.md`: the 2×2 table, subset NSE/KGE/β,
   `|zeta|` distribution, and the GO/NO-GO verdict.

### Interim result available WITHOUT the fix
If you want a caveated number now: both ON runs are effectively "daily + leakance," so you can
compare **leakance-ON vs daily-OFF (`2026-06-05…`)** on the losing-stream subset to ask "does
leakance help under *daily* forcing" (the regime DDR reverted it in). Clearly label it as the
daily arm only — it does NOT test the hourly hypothesis.

---

## 8. One-glance retrieval

```bash
cd ~/projects/ddrs
git log --oneline 1a2e75d..2cdd341            # the leakance testbed commits
ls .ddrs/runs/2026-07-01T01-07-17Z-train-and-test   # my hourly-ON run
cat docs/superpowers/specs/2026-06-29-leakance-hourly-feasibility-design.md   # the hypothesis + GO/NO-GO
sed -n '/## Leakance/,/## Baseline/p' CLAUDE.md    # how to enable leakance
```

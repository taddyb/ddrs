<!-- agent af2511434de2d88e2 | Review dam-release-head diff | handback | 2026-09-27T07:38:29.630Z -->
Review of `dam-release-head` vs `origin/reservoir-options` (9 commits, routing core at 0ac6f2e). **Nothing blocking found.** Tests run in a separate target dir: `reservoir_override` 8/8, `reservoir_release` 7/7, `reservoir_release_gradcheck` 7/7, `cargo test --lib` 437 passed / 3 ignored, all green.

## 1. Math and gradients (derived, confirmed)

Dam row, `K := T_{t+1}`, `X := 0`, `denom = 2T_{t+1}+dt`. From the trapezoid `T_{t+1}Q_{t+1} − T_tQ_t = dt[(I_t+I_{t+1})/2 + q' − (Q_t+Q_{t+1})/2]`:
`Q_{t+1} = [(2T_t−dt)Q_t + dt·I_t + dt·I_{t+1} + 2dt·q']/(2T_{t+1}+dt)`, i.e. c1=c2=dt/denom, c3=(2T_t−dt)/denom, c4=2dt/denom. `forward_chain_inner` S19''' produces exactly this: K/X override before `denom`, `c3` overridden by `mask_where` only when `t_prev_seconds` is `Some`. q' uses the full-dt (rectangle) convention, same as the Muskingum row.

Partials: `∂L/∂T_{t+1} = 2·Σ_i gc_i·∂c_i/∂denom = 2·gdenom_total` with `num_c3 = 2T_t−dt` in `gdenom_from_c3`; `∂L/∂T_t = 2·gc3/denom`. `timestep_backward_core` does this: `num_c3` overridden on dam rows, `g_2k1mx_from_c3` split off into `g_2t_prev` (so it never reaches K), `g_2k_from_2kx = g_2kx_total·x_eff = 0` on dam rows, `g_t_release = gk_muskingum` read after the `gk_from_x_cap` fold (zero on dam rows since `gx` is masked before the S19' branch split) and before the B19'' K mask and B18' floor mask (which belong to the channel `k_raw`). `g_t_release_prev = 2·g_2t_prev`. Both gathered by `select(0, rows)`; `register` accumulates when one node fills both slots (constant-T learned path). `state.c3`/`state.denom`/`state.k_muskingum` saved post-override, so `gq_t_from_s25` uses the real c3.

No leakage to the dam reach's channel parameters: with X=0 the only K path is `denom`, masked at B19'' before B18/B18'; the X half is masked before the Cunge/cap split; so celerity, n, q, p, gamma, top_width, and the S2 depth chain on the dam row get exactly 0 (test asserts `gn[dam]==0`).

Clamp: `(T0·exp(a sin+b cos)).clamp_min(1/24)·86400` in ordinary autodiff; bound step passes zero (test pins). `2·3600−3600` is exact in f32. f32 throughout; phase computed in f64 and cast once.

Window/eval starts: `forward` loop `t in 1..n_rows` uses phase rows `t` (end) and `t−1` (start); `seasonal_phase(window_start, n_hours)` row 0 = hour 0 of the window's first day; `RhoWindow.window_start` is the first routed day (warmup is inside rho), and eval chunks carry their own `window_start`, so `S = T·Q` is continuous across chunk boundaries with `carry_state` (same closed form at the same hour).

## 2. Invariants (confirmed by reading)

No-dam and option C: `reservoir_t_prev: None` (`set_reservoir_rows` sets `t_prev_seconds: None`), `c3` match falls through, backward matches on `(None, _)`, `mask.t_release=false` on every historical op, and the `TimestepReleaseOp` branch is entered only when `release.is_some()`. Routing head untouched (`release_head.rs` builds a second `KanHead`). Sparse backward extended (two new parents in `timestep_backward_core`), not replaced. I did not re-run `compare_ddr_sandbox` (not requested; findings report ABSOLUTE MATCH).

## 3. Gradcheck coverage

Adequate. T0/a/b at T0 = 0.1, 1.5, 20 d with a,b ≠ 0, gamma sibling op, one head read-out weight end to end, T_t and T_{t+1} separately at 2 h/3 h and 5 d/30 d with T_t ≠ T_{t+1}, shared-leaf sum check, clamp = exact 0. A dropped or mis-scaled parent would show O(1) error against REL_TOL 5e-3; `ABS_TOL 1e-4` cannot mask anything at these gradient magnitudes and the `a != 0` guard blocks vacuous passes. Two soft spots, not defects: the end-to-end weight touches only the T0 read-out column (a/b head columns rely on plain autodiff), and the large-T a/b step of 0.1 (14 % of a) means a sub-percent error there would pass.

## 4. Training integration

Joint backward: `GradientsParams::from_module` extracts each head from one `loss.backward()`; per-head `clip_grad_norm`; same `lr`; accumulation path scales both by `1/total_n`. Batches with no dams are safe: the norm/scale visitors and Adam skip absent params. Checkpoints write `release_head.mpk`/`release_optim.mpk` next to `head.mpk`; bootstrap restores both, starts cold (logged) from a no-release checkpoint; `state.json` untouched so resume position is unchanged. Test phase resolves once before any `collate_window` in all three entry points (`cli/run.rs`, `train_and_test.rs`, `eval.rs`; `eval.rs --checkpoint` is a directory, consistent with `release_head_base`). Same-model claim: same closed form and clamp on both paths, pinned bitwise (seasonal and constant) on NdArray.

## 5. Config and readers

Guards are sound: learned requires `use_reservoirs`, `release_head`, `kan_head`, non-empty inputs, `T0` floor ≥ 1/24; `release_head:` without `learned` rejected; `deny_unknown_fields` on the block; option C rejections retained. Readers: feature columns by name in config order, missing column / non-finite / duplicate COMID / empty rejected; fixed table needs both `a` and `b` or neither; dams sharing a COMID are aggregated by the build script and rejected as duplicates by the reader and by `set_dam_release`.

## Findings, ranked (none blocking)

1. **Plausible only, low.** Test-phase equality with training is bitwise only under NdArray small shapes. Training runs the head on each batch's dam subset; the test phase runs all 1,024 dams in one matmul on the inner backend. On CUDA (cubecl tile selection by shape) ulp-level `T0` differences are plausible. Harmless (~1e-7 rel), but the "bitwise the training path" wording should say NdArray-pinned.
2. **Confirmed, low.** `params.parameter_ranges` is an untyped `HashMap`, so a typo (`reservoir_t0` for `reservoir_T0`) is silently ignored: the run trains on the default `[1/24, 365]` box and init bias with no error. Pre-existing pattern (`K_D` etc.), widened by three new case-sensitive keys. A key whitelist in `From<ParamsRaw>` would close it.
3. **Confirmed, informational.** `apply_reservoir_rows` checks `release_head.expect(..)` before `rows.rows.is_empty()`, so any head-less forward on a learned config panics even with zero dams (`probe_forward`, `eval --frozen` without `--checkpoint`). Intended refusal; just note the frozen path is unusable with `learned`.
4. **Confirmed, performance only.** `t_seconds_at` recomputes each hour's T twice (as end of step t, start of step t+1), doubling the per-step T tape; `release_t0_stats` runs the head a second time per micro-batch.

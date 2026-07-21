# LSTM Q′ wave 2 + cross-wave comparison, findings (2026-07-16)

Plan: `.claude/plans/glowing-bubbling-sprout.md` (agent-authored, not checked in)
Wave 1: `docs/2026-07-16-aorc2f-wave1-findings.md`
Configs: `config/experiments/lstm_daily_frozen_chunk1.yaml`,
         `config/experiments/lstm_hourly_native.yaml`
Plots: `output/2026-07-16-wave-comparison/` (script: `scripts/wave_comparison_plots.py`)

## 1. What this tests

Two NH-LSTM Q′ forecast stores routed through the CONUS KAN head, same
constants as wave 1 (seed 42, 5 epochs, rho 90, warmup 5, L1 loss, no
leakance, train 1981/10/01–1995/09/30, eval 1995/10/01–2010/09/30):

| Arm | Q′ store | Resolution | Disagg head |
|---|---|---|---|
| daily-lstm | `daily_lstm_merit_unit_catchments.ic` (288,421 divides) | daily | frozen capacity-boosted (same as wave 1) |
| hourly-lstm | `hourly_lstm_merit_unit_catchments.ic` (197,088 divides) | hourly-native | none — hourly-native + disagg head is a hard config error |

daily-lstm updates the older `equif_daily_lstm_disagg.yaml` arm (2026-07-07
equifinality campaign), which pre-dates the 2026-07-10/12 frozen
capacity-boosted disagg-head fixes and used a plain, jointly-trained
`hidden_size: 16` head.

## 2. Execution — this time, GPU alone (no contention) also OOM'd

Lesson from wave 1 was "don't co-schedule two `--backend cuda` trainings" —
so wave 2 launched **one arm per backend from the start**: daily-lstm on
GPU alone, hourly-lstm on CPU alone. hourly-lstm ran clean end-to-end on CPU
(median NSE 0.5543, KGE 0.4852, 2365/2365 finite).

**daily-lstm hit the exact same silent-corruption incident as wave 1's
distributed arm — with NO other process on the GPU.** Training completed
cleanly (5 epochs, all checkpoints written). Phase 2 (testing) then hit the
same cubecl-cuda `DSD-0-0` worker-thread OOM panic (`can't allocate buffer of
size: 4178264064`) that never propagates to the caller; the process kept
looping through chunks (confirmed to chunk 6/366) at 900%+ CPU. **This
confirms the failure is NOT contention-specific** — Phase 1 alone leaves the
16 GB card with too little headroom for Phase 2's ~4.2 GB per-chunk buffer,
even running solo. Recovery: killed, then ran the standalone `eval` binary
against the epoch-5 checkpoint with `--backend cpu` (same recovery pattern as
wave 1).

**Practical recommendation going forward:** on this GPU, either (a) always
use `--backend cpu` for `train-and-test`/eval workflows at the default
`testing.batch_size: 15` (days/chunk), or (b) reduce `testing.batch_size` to
shrink the per-chunk buffer (e.g. 3-5 days) and re-test whether Phase 2 then
fits on GPU — not yet tried. GPU is fine for training only (`--workflow
train`) at these settings.

### Bug found and fixed mid-campaign

The corruption-detection gate added after wave 1's incident
(`corrupted_chunk_reason` — all-zero/non-finite value check) did **not**
catch this second occurrence: the corrupted chunk's buffer held stale GPU
memory that happened to look like plausible finite, non-zero predictions.
Strengthened with a process-global panic-hook detector
(`WORKER_PANICKED`/`ensure_panic_hook_installed`/`take_worker_panicked` in
`src/training/eval.rs`) that catches ANY background-thread panic during a
chunk or the post-loop tensor readback, independent of what the output
values look like. See `docs/2026-07-16-aorc2f-wave1-findings.md` and the
commit introducing `DataError::CorruptedEvalChunk` for the first (partial)
fix; 6 unit tests cover both detectors (`cargo test --lib training::eval`).

## 3. Results

| Arm | Trained median NSE | Trained median KGE | Own baseline NSE | Own baseline KGE |
|---|---|---|---|---|
| daily-lstm | 0.5674 | 0.6169 | 0.437 | 0.616 |
| hourly-lstm | 0.5543 | 0.4852 | 0.532 | 0.547 |

(2365/2365 gauges finite NSE in both arms.)

daily-lstm is the best-performing arm in this entire campaign on NSE (0.5674)
and its KGE (0.6169) roughly matches its own baseline (+0.001) — routing
doesn't hurt here. hourly-lstm's routed KGE (0.4852) is noticeably *worse*
than its own raw-Q′ baseline (0.547, Δ −0.062) despite an NSE gain (+0.022) —
the routing model is trading KGE's variance/bias components for
correlation/NSE here, worth a follow-up if the hourly-native arm is pursued
further.

## 4. Cross-wave comparison (all 4 arms)

| Arm | Trained NSE | Trained KGE | Own baseline NSE | Own baseline KGE | Δ vs 0.7152/0.7106 benchmark (NSE/KGE) |
|---|---|---|---|---|---|
| AORC2F distributed | 0.3437 | 0.3256 | 0.233 | 0.327 | −0.372 / −0.385 |
| AORC2F lumped | 0.5259 | 0.5175 | −0.105 | 0.393 | −0.189 / −0.193 |
| daily-lstm | 0.5674 | 0.6169 | 0.437 | 0.616 | −0.148 / −0.094 |
| hourly-lstm | 0.5543 | 0.4852 | 0.532 | 0.547 | −0.161 / −0.225 |

**Ranking by trained NSE:** daily-lstm (0.567) > lumped (0.526) > hourly-lstm
(0.554, close 3rd) > distributed (0.344). By trained KGE: daily-lstm (0.617)
> lumped (0.518) > hourly-lstm (0.485) > distributed (0.326).

**None of the four new Q′ stores match the standing CONUS benchmark**
(median NSE 0.7152 / KGE 0.7106, `merit_dhbv2_UH_retrospective.ic`) under
this exact routing configuration — daily-lstm comes closest (−0.148 NSE,
−0.094 KGE) but all four arms lose meaningful skill relative to it. The
routing model consistently helps most on arms with weak/negative raw-Q′
baselines (lumped: baseline NSE −0.105 → trained 0.526; distributed:
baseline 0.233 → trained 0.344) and helps least — or even hurts on KGE — on
arms whose raw Q′ is already strong (hourly-lstm, daily-lstm).

Full distribution comparison (8 series: 4 trained + 4 own baselines):
`output/2026-07-16-wave-comparison/wave_comparison_boxplot.png`,
`wave_comparison_nse_cdf.png`, `wave_comparison_kge_cdf.png`. Note:
`ddr.validation.plot_cdf`'s legend hardcodes the label `NSE=<median>`
regardless of which metric was passed in — the KGE CDF's legend text says
"NSE=" but is plotting KGE medians (matches the title/axis, which are
correct). Cosmetic upstream-DDR quirk, not a data error.

## 5. Open follow-ups

- **2026-07-21 correction:** wave 1's AORC2F-distributed-vs-lumped
  backend-mismatch dismissal was based on an inapplicable test tolerance;
  see the corrected §3 discussion in `docs/2026-07-16-aorc2f-wave1-findings.md`
  for the actual (mb=0 loss ordering) evidence and the leading untested
  hypothesis (the distributed store may already be UH-routed, causing
  double-routing). An adversarial review of this whole campaign's controls
  is at `/tmp/experiment-handoff-aorc2f-lstm-routing.md` — no replicate
  seeds, no backend-consistent control run, and no per-gauge/spatial
  diagnostics were done for any of the 4 arms.
- Try `testing.batch_size` < 15 days to see if Phase 2 becomes GPU-viable
  (would meaningfully speed up future campaigns — each of these runs took
  2–8.5 hours, dominated by the 366-chunk CPU testing phase).
- hourly-lstm's KGE regression under routing (routed 0.485 vs baseline 0.547)
  is unexplained — worth a per-gauge diff against distributed/lumped/daily
  to see if it's concentrated in a subset (e.g. high-variance flashy basins).
- None of these four Q′ stores currently beat the standard
  `merit_dhbv2_UH_retrospective.ic` benchmark; before concluding routing
  can't help them further, check whether their coverage/fill-rate or
  training-window alignment differs materially from the benchmark store.

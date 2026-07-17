# AORC2F Q′ wave 1 — frozen capacity-boosted disagg head, findings (2026-07-16)

Plan: `.claude/plans/glowing-bubbling-sprout.md` (agent-authored, not checked in)
Configs: `config/experiments/aorc2f_distributed_frozen_chunk1.yaml`,
         `config/experiments/aorc2f_lumped_frozen_chunk1.yaml`
Prior benchmark: `docs/2026-06-23-precip-disaggregation-findings.md` (median NSE 0.7152 / KGE 0.7106)

## 1. What this tests

Two new dHBV rainfall-runoff Q′ forecast stores (distinct from the standard
`merit_dhbv2_UH_retrospective.ic` used in all prior CONUS benchmarks), routed
through the same CONUS KAN head + the 2026-07-10/12 capacity-boosted frozen
disagg head (`hidden 16, 2 layers, grid 20, k 3, chunk_days 1`, warm-started
from `output/disagg_pretrain/capacity_chunk1.mpk`, frozen). Everything else —
network, attributes, gauges, seed 42, 5 epochs, rho 90, warmup 5, L1 loss, no
leakance, train 1981/10/01–1995/09/30, eval 1995/10/01–2010/09/30 — matches
`kan_disagg_conus_frozen_chunk1.yaml`.

| Arm | Model | Q′ store | Upstream checkpoint | Divides |
|---|---|---|---|---|
| Distributed | Distributed + UH routing (dHBV AORC2F v2) | `daily_dhbv_aorc2f_merit_unit_catchments.ic` | `CONUS2717_AORC2F_v2` ep69 | 197,088 |
| Lumped | Lumped (AORC) dHBV | `daily_dhbv2_merit_unit_catchments.ic` | `CONUS2717_AORC2F_LUMPED` ep63 | 197,088 |

The "checkpoint" column is provenance only — these are DDR/PyTorch `.pt`
checkpoints of the upstream rainfall-runoff model that produced each store's
Q′ values; `ddrs` reads the resulting streamflow directly and never loads
those files.

## 2. Execution notes (both arms hit a real GPU OOM — see below)

Both arms were launched in parallel with `--backend cuda` (KAN head + router
both on GPU, per instruction). The card is a single 16 GB RTX 4080 Super.

- **Lumped** hit a CUDA OOM ~30s into epoch 1 (`can't allocate buffer of size:
  16321536`, the *distributed* arm having claimed ~9.9 GB first). Relaunched
  whole with `--backend cpu` (forces `sparse_solver=cpu` for both KAN and
  router, not a YAML edit) — completed cleanly end-to-end on CPU. Total wall
  time 02:23:20Z→10:49:20Z ≈ **8h 26min** (train ~1h 46min + test ~6h 40min).
- **Distributed** trained cleanly on GPU for all 5 epochs, but Phase 2
  (testing) hit a *silent* failure mode: a cubecl-cuda background worker
  thread panicked on OOM (`can't allocate buffer of size: 4178264064` — Phase
  2's per-chunk buffer is ~260x a training minibatch's) without the panic
  propagating to the main thread, which kept looping through eval chunks
  indefinitely at 900%+ CPU instead of failing. Killed and recovered by
  running the (deprecated) standalone `eval` binary directly against the
  epoch-5 checkpoint with `--backend cpu` — this is why the distributed arm
  has no `manifest.json` (the killed `ddrs run` process never wrote one) but
  does have `checkpoints/`, `config.yaml`, and `eval/predictions.zarr`.
- This silent-corruption failure mode is now hard-gated in `src/training/eval.rs`
  (`DataError::CorruptedEvalChunk` — see commit introducing
  `corrupted_chunk_reason`); a future run in the same situation will fail
  loudly at the first corrupted chunk instead of producing bad output.
- **Lesson for future waves:** don't co-launch two `--backend cuda` CONUS
  trainings on one 16 GB card — Phase 1 alone uses ~9.9 GB, leaving no
  headroom for a second process, and Phase 2's larger buffers make it worse.
  Wave 2 launches one GPU + one CPU arm from the start instead of
  crash-and-retry.

## 3. Results

| Arm | Trained median NSE | Trained median KGE | Own summed-Q′ baseline NSE | Own baseline KGE | Δ NSE vs own baseline | Δ KGE vs own baseline |
|---|---|---|---|---|---|---|
| Distributed | 0.3437 | 0.3256 | 0.233 | 0.327 | +0.111 | −0.001 |
| Lumped | 0.5259 | 0.5175 | −0.105 | 0.393 | +0.631 | +0.124 |

(2365/2365 gauges finite NSE in both arms.)

Both are well below the standing CONUS benchmark (median NSE 0.7152 / KGE
0.7106, `merit_dhbv2_UH_retrospective.ic` streamflow, same disagg head) —
**Δ vs benchmark: distributed −0.372 NSE / −0.385 KGE; lumped −0.189 NSE /
−0.193 KGE.**

Lumped clearly outperforms distributed on this routing setup (+0.182 NSE,
+0.192 KGE) and its raw Q′ is dramatically improved by routing (own baseline
NSE −0.105 → trained 0.526), whereas distributed's raw Q′ was already
directionally reasonable (baseline NSE 0.233) and routing helps far less in
relative terms.

**Backend caveat:** distributed trained on GPU, tested on CPU (via the
recovery path); lumped ran entirely on CPU. Per prior parity testing
(`tests/sparse_cusparse_v5.rs`), CPU/CUDA numerical drift is ULP-scale
(~1e-4 rel tolerance) — far smaller than the 0.18 NSE gap observed here, so
the lumped > distributed ranking is not a backend artifact.

## 4. Interpretation

Neither AORC2F Q′ store is a drop-in replacement for the standard
`merit_dhbv2_UH_retrospective.ic` under this routing configuration — both
lose substantial skill relative to the standing benchmark. Of the two, the
lumped (AORC) dHBV variant is the better Q′ source for this KAN+disagg
routing setup. Follow-up (not yet done): check whether the AORC2F stores'
lower coverage assumptions, different training windows, or different
input-forcing pipeline (vs the retrospective store) explain the gap before
concluding this is a routing-model limitation.

# Learned dam release: five reviews and the plan that follows from them

**Date:** 2026-09-27
**Asked (user):** Fable reviewers to look for opportunities to improve reservoirs, leakance and metrics with the new code,
and whether the learned T0 correlates with dam size.
**Inputs reviewed:** the learned dam release on branch `dam-release-head` (61a4a50); full-population arms
`2026-09-27T07-29-47Z` / `07-29-55Z` (seed 42, off / learned) and `10-31-30Z` / `10-31-50Z` (seed 43); smoke arms
`04-29-33Z` (off) and `07-30-13Z` (learned, fixed dam row); results page https://claude.ai/artifact/HhcubFbHpBdmpKykN7PeUT.
**Evidence:** scripts, tables and figures of each review in `experiments/reservoir/review_2026-09-27/`
(`opps_release`, `opps_training`, `opps_leakance`, `opps_metrics`, `t0_size`); reports where the agent could write one.
All reviews read existing outputs only; no training.

## 1. What the reviews agree on

- **The law is a sub-weekly smoother.** NSE of daily differences rises ten times more than NSE at dammed gauges
  (+0.014 / +0.027 vs +0.0014 / +0.0039); the NSE of the monthly climatology does not move; the KGE loss is entirely
  alpha (dammed alpha -0.0086, 82 % of gauges down; r slightly up; beta flat). Amplitude metrics worsen: FHV, peak ratio
  0.84 -> 0.78, flashiness 0.77 -> 0.68; 69 % -> 80 % of dammed predictions are smoother than observed. This is the
  NSE objective's optimum at alpha = r < 1, not a physics error.
- **The effect is real and uneven.** Per-gauge changes correlate across seeds at 0.63 at dammed gauges (0.74 on the
  gauge reach) and -0.15 at undammed gauges. 32-38 % of dammed gauges gain more than 0.01, 13-15 % lose more than 0.01;
  gains concentrate where the learned T0 exceeds 3 d (+0.041 / +0.043) and in the worst-fit tercile.
- **The undammed loss is the co-trained routing head wandering, not a dam effect.** Roughness does not change more
  above dams than elsewhere; seed 43's learned arm has n 0.68x its off arm over all of CONUS; gamma is 0.067 / 0.168 /
  0.272 / 0.043 across the four arms. Every paired number inherits this.
- **The seasonal terms are not identified.** a and b are functions of T0 within a seed (Spearman -0.99 / -0.93), the
  phase differs between seeds (b sign agrees at 43 %), and seasonality adds nothing even offline (+0.0013 per gauge
  over the plain bucket; seasonal minus linear -0.0011 in another fit).
- **T0 tracks absolute storage, not residence time.** Spearman with max storage 0.78 / 0.72, normal storage 0.67 /
  0.60, residence time 0.36 / 0.42, drainage area 0.10 / 0.01; a surrogate from the 19 head inputs explains log T0 at
  R2 0.95 / 0.93 with max storage (importance 0.53 / 0.58) and the flood-control flag (0.19 / 0.24). T0 is about 0.6 %
  of residence time. It agrees at rank level with offline fits on the same routed inflow (0.46 / 0.50) and not with
  fits from the dams' own records (ResOpsUS, 0.12 / 0.03, those run 7-8x longer).
- **The ceiling is lost mostly in which parameters are learned.** On-reach smoke dams: per-gauge offline fit +0.021;
  a feature-limited version of it +0.017; the ddrs-learned parameters simulated offline +0.008 / +0.010; in the engine
  +0.006. 73 % of the offline gain sits at fitted T0 > 2 d, where learned T0 is a factor 3.6 short.
- **A dam row replaces its reach's channel routing.** So T = 1 h is faster than no dam (smoke check 1b: 188 gauges
  below NSE 0.999, min 0.485, Spearman 0.55 with dam-reach length), the head cannot opt out, and the offline ceiling
  double-counts the dam reach (its floor corresponds to T of about K_reach, 4 h, in the engine).

## 2. Where the reviews disagreed, and how it resolves

- **Do the 90-day training windows cause the short T0?** The release-law review said yes (warmup 5 d scores the
  storage transient). The training review tested it directly: refitting each on-reach dam's bucket with the training
  objective (random 90-day windows, storage started at T x I(t0), scored from day 5) gives the same T0 as a continuous
  fit (median 1.75 d, per-dam ratio 1.00, Spearman 0.98); warmup 30, rho 180 / 365 and an oracle start agree. The
  direct test wins: windows matter only for T0 above about 10 d (20 % of active dams), which is a tail fix. The cause
  of the short T0 is a flat objective (the 0.02-NSE band is a median 0.96 decades wide), the loss population (same
  seed: full population 0.35 d, smoke set 0.59 d) and the co-trained routing head.
- **A learned evaporation / withdrawal loss at dams?** The leakance review proposed one (high-DOR gauges carry
  +6.0 % more volume than matched controls at DOR 1-2, +2.6 % above 2). The release-law review found no volume excess
  at on-reach dams against matched controls (median log ratio -0.006) and that a learned loss absorbs Q' volume bias
  (Spearman 0.51 with pass-through beta; controls with beta > 1.1 want it 23 / 3). Resolution: a learned loss only
  after a volume-scalar null (§3, item 3) and only with an external check (loss correlating with open-water
  evaporation x surface area); a prescribed evaporation sink is the defensible form.

## 3. Plan, cheapest decisive step first

Every test pairs on one gauge CSV (trap T19). Protocol for all of them: difference-in-differences against undammed
gauges (not a test against zero), seed-consistency of per-gauge changes, and the amplitude metrics next to NSE.

**Tier 0: no training (hours)**
1. Analysis protocol: DiD vs undammed, seed-consistency Spearman, NSE of daily differences, r / alpha / beta, FHV,
   peak ratio, Hodges-Lehmann and the share of gauges beyond +/-0.01, NSE clipped at -1 before any mean. Fix
   `paired_full_run.py::paired` (null), the manifest's `mean_nse_finite` (set by two gauges below -16), f64
   accumulation in `src/training/metrics.rs::Metrics::compute` (f32 error up to 3e-4 NSE), and a log-form low-flow
   metric instead of FLV. Update the results page: bimodal effect, seasonal cycle unchanged, peaks smoother.
2. Test-phase decomposition of the existing learned runs (test passes only): dams off with the learned routing head;
   a = b = 0; T0 x2 / x4 / x8 through the fixed seasonal table. Go: the release term carries >= 80 % of the dammed
   gain; a = b = 0 within +/-0.001 at dammed gauges (then seasonality goes); x2-x4 raises the dammed change by > 0.002
   with the interval clear of zero (training under-shoots T0).
3. Volume-scalar null: a global (and per-HUC2) scale on Q' fitted on training years. On the test years a fixed 0.95
   already lifts median NSE 0.7391 -> 0.7438 (0.93: 0.7442; seed 43 0.7341 -> 0.7390), because predicted/observed
   volume is 1.045 at the median and above 1 at 65 % of gauges. Any loss term (leakance, dam loss) must beat it, paired.

**Tier 1: one change each, smoke arms (~1 h CPU)**
4. Freeze the routing head at the off checkpoint and train the release head alone (per-head learning rate or freeze in
   `training/driver.rs::ReleaseTrainer`). Undammed gauges become bitwise identical, so no seed confound. Adopt a two-stage
   schedule (release on frozen routing, then joint at low lr) if the frozen arm keeps >= 80 % of the joint dam gain.
5. `release_head.seasonal: false` (config), folded into 4.
6. Additive dam row: `K := K_reach + T`, `X := X_reach K_reach / (K_reach + T)` (S19'' / B19'' edit, gradcheck). T -> 0
   is the no-dam row exactly, the head can opt out, check 1b becomes a real gate, and T0 reads as added storage.
7. Loss for the release phase: `nse-batch-deriv` (scores day-to-day changes, penalises smoothing past observed
   flashiness) or `nnse-kge`. Go: dammed NSE not below the nse-batch arm, alpha change >= -0.003, FHV not worse.

**Tier 2**
8. Release-head features: add log DOR (storage over mean Q' at the dam reach; 0.38 with fitted T0) and upstream
   large-dam storage (chains gain +0.010 vs +0.006); collapse the collinear storage / drainage / storage-per-area trio;
   ablate max storage and the flood flag. Per-dam free T0 as the ceiling of any head.
9. Release head: own constant learning rate (seed 43's T0 was still rising at lr 0.001) and train until T0 is flat
   over the last 20 epochs in two seeds.
10. Inflow-driven T, `T(t+1) = T_seas (I_t / I_ref)^c`, after 5: best of the law family offline (114 / 61 on-reach,
    the only law raising KGE, +0.0015), linear in the solve; the test phase's fixed table needs a per-dam c.
11. Longer windows (warmup 30, rho 180 or 365) only for the T0 > 10 d tail; shrink the T0 box top to about 60 d.
12. Dam loss term: prescribed open-water evaporation, or learned with the external check, only after 3.

**Tier 3: leakance**
13. The latest two-way leakance arm (2026-09-17T16-38-16Z) is a near-uniform 2 % sink: K_D, factor and d_gw sit at
    their initial values, it lowered volume at 2,235 of 2,365 gauges (450 of 531 already short), and it scores below
    the 0.95 scalar (0.7300 vs 0.7438). It did not localise on dams, so reservoirs are not a hidden confound in the
    July NO-GO. Next: sparse leakance (gate off at init or an L1 on sum |zeta|) on the 2026-07-04 planted-reach
    harness with K_D free; go if recovery >= 0.5 and planted reaches hold > 50 % of flux, else close the
    parameterisation question. Only then a smoke experiment {leakance off, on} x {dams off, learned} x two seeds,
    read out as the share of gauges where leakance moves volume toward observed, and the plumbing for leakance and
    reservoirs together (dam mask through `TimestepLeakanceOp` / `TimestepLeakanceGammaOp`, zeta zero on dam rows).
14. The routing head itself is unconverged and underdetermined (gamma wanders 4x, n shifts 0.68x between arms). Fixing
    it (gamma box or regularisation, longer training, the curvature instrument) helps every comparison above.

## 4. No-go, with the evidence

Release cap as a fraction of NID max discharge (on-reach 48 / 37, median 0; spillway design flow far above operating
flow); free cap (controls 24 / 72, overfits, as before); flood-pool spill and two-bucket laws (64 / 44, null);
storage-driven T (behind inflow-driven, collinear); Nash cascades of buckets (worse NSE); uniform T0 scaling
(x3 costs KGE 0.009); learned Muskingum X on dam rows (c1 >= 0 needs X <= dt / 2K, about 0.02). No recommended law
needs an identity row: all are linear in the solve.

## 5. Corrections to what this session said earlier

- "Training can't see long storage" was given as a main cause of the short T0. It is not, except for dams with T0
  above about 10 d (§2).
- Smoke check 1 ("T0 = 1 h reproduces the no-dam run") was wrong twice: first as 1e-4 m3/s, then as NSE > 0.999
  everywhere. A dam row replaces the channel reach; item 6 makes the check meaningful.
- Smoke check 5's "controls within 0.01" is ten times too loose for the co-training confound (-0.0007 to -0.0011).
- The offline ceiling of record double-counts the dam reach, so check 4's "stays low" floor is about K_reach (4 h),
  not 1 h.
- The results page's single dammed number hides a bimodal effect, and it does not say that the seasonal cycle is
  unchanged and that peaks got smoother.

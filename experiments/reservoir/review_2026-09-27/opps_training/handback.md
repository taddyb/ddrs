<!-- agent aec6f5ca48c5bab5a | Fable: reservoir training and identifiability | handback | 2026-09-27T13:51:04.322Z -->
> The agent's final handback message, verbatim; the full report is `report.md` beside it. Saved 2026-09-27.

Full report: /home/tbindas/.claude/jobs/dacd6d8c/tmp/opps_training/report.md (scripts, CSVs and outputs in the same directory; all light Python on existing run outputs, no training).

## Headline finding (new evidence)

The 90-day hotstarted window is NOT why T0 comes out short. I refit the linear bucket per on-reach smoke dam gauge with exactly the training objective (300 random 90-day windows in WY1983-1995, S0 = T x I(t0), scored from day 5): the argmin T0 matches the continuous offline fit (median 1.75 d both; per-dam ratio 1.00, IQR 0.78-1.00; Spearman 0.98). Warmup 30, rho 180/365, and an oracle initial state (a converged state cache) give the same answer. The window only truncates T0 above about 10 d (20 % of active dams; even a 365-day window caps at ~50 d). So state_cache buys nothing and longer rho is a tail fix.

Instead: the objective is flat (the offline 0.02-NSE band is a median 0.96 decades wide); 53-56 % of learned T0 sit inside the band, 39 % below it, and the below-band dams are the week-scale ones (fit 8-9 d, learned 0.8-0.9 d, in all three learned arms). The release head is near-linear (R² 0.91 on the 19 features) and rank-1: within a seed a and b are functions of T0 (Spearman −0.99, −0.93 in seed 42), phase per dam differs between seeds by a median −38° with IQR −89° to +76°, and the offline fit itself gains nothing from seasonality (seasonal minus linear −0.0011 NSE). The loss population moves T0 (same seed: full population 0.35 d, smoke 0.59 d, per-dam ratio 1.38). Δlog n above dams versus controls between arms (−0.012 / −0.060 in seeds 42 / 43) is within the same contrast between two no-dam seeds (−0.034), so n compensation is not local; it is global (seed 43's learned arm has n 0.68x its off arm over all CONUS, gamma 0.27 → 0.04), which is where the undammed cost comes from.

## Top 5 opportunities, ranked by (gain x confidence) / cost

1. Test-phase decomposition of the existing runs, no training (5 test passes, ~18 min smoke / 35 min full each). Re-run each learned run's test phase with (a) dams off at the learned routing head, (b) a = b = 0, (c) T0 x2 / x4 / x8 via the existing fixed seasonal CSV path. Evidence: global n shift and undammed cost (§1.6); rank-1 seasonality; 39 % of dams below the flat band. Go/no-go: (a) release-only term carries ≥ 80 % of the dammed gain and undammed gauges are bitwise unchanged; (b) a = b = 0 within ±0.001 at dammed gauges with KGE recovered → ship seasonality off; (c) x2-x4 raises dammed ΔNSE by > 0.002 with CI clear of zero → training under-shoots T0 and items 2-4 are worth running; flat → a parameter-level, not skill-level, deficiency.

2. Freeze the routing head; train the release head alone (per-head lr knob, ~40 lines in driver.rs; 2 smoke arms, 2 h). The in-engine analogue of the offline fit with a shared head; separates joint adaptation with (n, gamma) from head smoothing. Evidence: seed 43's global n shift; loss median flat after epoch 10. Go/no-go: on-reach T0 in-band ≥ 75 % and median log10(learned/fit) > −0.1 → adopt two-stage (release on frozen routing, then joint at low lr); else the head or population is the limit.

3. Per-dam free T0 embedding (ceiling of any head; ~1 day Rust, 2 smoke arms) plus a CSV-only feature revision: add log DOR = storage / (mean Q' at the dam reach x 1 yr) from the forcing store, collapse the collinear storage / drainage / storage-per-area trio to storage_max + DOR, keep purpose, height, surface, drop year (its role is the completion-year activation). Evidence: offline T0 correlates 0.38 with DOR vs 0.14 with storage/area, 0.02 with year; seed-split coefficients on the collinear trio; cascade unsupported. Go/no-go: free-T0 arm reaches ≥ 90 % of the offline ceiling (+0.0068 at dam gauges) and in-band ≥ 75 %; features raise Spearman(learned, fit T0) from 0.47-0.50 above 0.6 with coefficients reproducing across seeds within 0.1.

4. Loss population / weighting toward regulated gauges (no code if paired with 2; 2 smoke arms). Train the release head with the routing head frozen on on-reach dam gauges only vs the full smoke set. Evidence: §1.5. Go/no-go: on-reach T0 in-band ≥ 75 % where the full set is not, and on-reach gain up ≥ 0.002; production form is a per-gauge weight 1 + DOR in batch_loss, release-only.

5. Own constant lr (0.01) for the release head, per-epoch release-table logging, one 100-epoch run (config only, 2 smoke arms). Evidence: seed 43's batch-median T0 still rising over epochs 41-50 at lr 0.001 (0.57 → 0.73 d); seed 42 plateaued by epoch 10; between-seed per-dam T0 ratio 1.33. Go/no-go: T0 flat over the last 20 epochs in both seeds and ratio within 1.0 ± 0.2; a reproduced plateau means the budget is not the limit, the flatness is.

Lower: seasonality off by default (folded into 1b); rho 365 only for the T > 10 d tail (4x cost) and shrink the T0 box top to ~60 d (40 % of the log box is unlearnable by construction); 4 seeds/arm on smoke after the protocol is fixed (dammed effect seed SD ~0.0018 is 6x the bootstrap SE; the undammed cost needs no seeds after 1a); cap per-gauge loss (single micro-batches hit 2e6-2e9 in the off arms, routing blow-ups the dam rows suppress in the same batch); the adjoint/landscape studies refuse reservoir arms, so use the Δn-above-dams proxy until T0 is stable.

## What looks wrong now

- The routing head's (n, gamma) is degenerate and unconverged (median gamma 0.067 / 0.168 / 0.272 / 0.043 across the four arms; n 0.68x between seed-43 arms); every paired number inherits this wander and the undammed cost is its footprint.
- Seasonal a, b are trained but unidentified (rank-1 with T0, phase flips, no offline gain, KGE cost).
- One lr schedule for two heads at different stages; the release head starts from a zero read-out and is still moving in one seed when lr has decayed.
- T0 box [1 h, 365 d]: nothing above ~50 d is learnable with any window up to a year.
- Page CIs are within-seed bootstrap only; seed SD is 6x larger. Quote both.
- Loss blow-ups of 1e6-1e9 in single micro-batches reach the optimizer (clipped, step wasted).
- Outside my angle but affects check 4's reading: the dam row replaces the reach's channel routing (K ~4 h), so learned T0 includes the replaced travel time and cannot stay at 4.5 h where no storage exists, and it differs from the offline (bucket-on-top) T0 by ~K_reach before any learning question arises.

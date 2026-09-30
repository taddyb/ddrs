# Learned dam release: how it is learned. Review and ranked opportunities

Reviewer angle: training setup, identifiability, data and features. Date 2026-09-27.
Inputs read: the design brief, the implementation findings (`2026-09-27-learned-dam-release-findings.md`), the smoke
README, the option C benchmark findings, `research-status.md` (§Reservoirs, §Open, §Training convergence), traps T13/T17,
the release head and driver source at commit `61a4a50`, the four full-population run logs and `release_params.csv`,
`kan_parameters.nc` of all four arms, the smoke offline fit, and the results page.
Scripts and outputs (all under `/home/tbindas/.claude/jobs/dacd6d8c/tmp/opps_training/`): `parse_logs.py`,
`parse_logs2.py` (log trajectories, spikes), `release_params.py` (per-dam parameters vs seeds, features, offline fit),
`n_above_dams.py` (+ `n_above_dams.csv`), `windowed_fit.py` (+ `windowed_fit.csv`, `.out`), `flatness.py`,
`seed_power.py`. All run under `~/projects/ddr/.venv/bin/python`, nothing heavier than a few minutes single core.

## 1. The question: why short T0 and no stable seasonality

### 1.1 The 90-day hotstarted window is NOT the cause for T0 up to about 10 days (new evidence)

`windowed_fit.py` refits the linear bucket per on-reach smoke dam gauge (214 gauges, 158 with an active fit) on the
no-dam routed flow, with the exact objective ddrs training sees: 300 random 90-day windows in WY1983-1995, storage
initialised at the hotstart analogue `S0 = T * I(t0)`, scored from day 5, SSE summed over windows. Against the
continuous WY1983-1995 fit (which reproduces the CSV's `lin_T0`, Spearman 0.999):

| objective | median T0 (d) | ratio to continuous, median [IQR] | Spearman |
|---|---:|---|---:|
| continuous (offline fit) | 1.75 | | |
| 90-day windows, warmup 5 (ddrs training) | 1.75 | 1.00 [0.78, 1.00] | 0.98 |
| 90-day windows, oracle initial state (a converged state cache) | 1.75 | 1.00 [1.00, 1.29] | 0.98 |
| 90-day windows, warmup 30 | 1.75 | 1.00 [0.78, 1.00] | 0.98 |
| 180-day windows, warmup 30 | 1.75 | 1.00 [1.00, 1.22] | 0.98 |
| 365-day windows, warmup 30 | 1.75 | 1.00 [1.00, 1.00] | 0.99 |

By continuous-fit T0 bin (medians, d): 0.23 / 1.05 / 3.75 / 17.2 / 101.7 continuous; 0.30 / 1.05 / 3.75 / 13.3 / 17.2
under 90-day windows; 0.23 / 1.05 / 4.83 / 17.2 / 47.5 under 365-day windows. So the window truncates T0 only above
about 10 d (32 of 158 active on-reach dams, 20 %), and even a 365-day window caps at about 50 d. The design brief's
§6 caveat ("T0 near or above rho is poorly learned") is true only for that tail. A state cache buys nothing: the oracle
initial state gives the same answer as the hotstart, because the bucket's transient from `S0 = T * I(t0)` is short
compared with 85 scored days whenever T is below about 10 d.

Consequence: the learned median T0 of 0.35 d (full population) / 0.59 d (smoke) against the fit's 1.75 d, and the
miss at the 8-9 d dams (below), cannot be attributed to rho, warmup or the hotstart. `experiment.state_cache` and
longer rho are tail fixes, not the fix.

### 1.2 The objective is flat in T0, and the learned T0 sits inside or below the flat band

`flatness.py`: the offline fit's band of T0 within 0.02 NSE of its optimum is a median 0.96 decades wide (IQR 0.56 to
1.35; 39 % of bands reach the 0.05 d floor). Of the learned T0 at the 158 active on-reach dams: 53-56 % lie inside the
band, 39 % below it, 6-11 % above. The below-band dams are the ones with storage: fit T0 median 8-9 d, learned 0.8-0.9
d, half a decade below the band's lower edge, in all three learned arms (seed 42, seed 43, smoke). The median
log10(learned / fit) is −0.28 to −0.35 (a factor 2 to 2.2). These 8-9 d dams are exactly the ones the offline windowed
refit recovers without bias, so the miss is on the model side of training, not the data side.

### 1.3 What the training trajectory says (`parse_logs.py`, `parse_logs2.py`)

- Batch-median T0, seed 42: 0.195 d at epoch 1, 0.42 by epoch 10, then a plateau at 0.37-0.53 d for 40 epochs through
  both learning-rate drops. Seed 43: 0.35 at epoch 10, 0.49 at 20, 0.50 at 30, 0.58 at 40, 0.62-0.73 over epochs 44-50,
  still rising at lr 0.001. One of two seeds is unconverged in T0; the seed-43/seed-42 T0 ratio is a median 1.33 per
  dam (IQR 1.00 to 1.89).
- Micro-batch loss median by 10-epoch block: off 42: 0.257, 0.245, 0.231, 0.243, 0.249; learned 42: 0.254, 0.243,
  0.226, 0.243, 0.244. The training objective barely falls after epoch 10 in any arm. The learned arm's median
  micro-loss is lower than its off arm by 0.002-0.013 in every block (both seeds), consistent with a small dam gain.
- Manning n (batch median): 0.128 at epoch 1, 0.05-0.08 from epoch 5 on, wandering; in learned 43 it keeps falling to
  0.039 at epoch 50 with gamma 0.043 (the n0-gamma degeneracy of research-status §Stage-dependent roughness, rho 0.90).
- Loss spikes: single micro-batches at 2.4e6 (off 42, epoch 6), 2.7e3 (learned 42, same batch), 1.6e9 (off 43, epoch
  30), 110 (off 43, epoch 43); learned 43 has none above 2.4. These are routing blow-ups (the negative-coefficient
  oscillation of check 1b), and the dam rows (all coefficients non-negative) suppress them in the same batch. With
  clipping at norm 1 the step is not destroyed, but it is spent on one gauge.

### 1.4 The release head is near-linear and rank-1 (`release_params.py`)

- Linear R² of log T0 on the 19 features: 0.91 (seed 42), 0.90 (seed 43). The head is a smooth, near-linear map.
- Within a seed, a and b are functions of T0: Spearman(log T0, a) = −0.99, (log T0, b) = −0.93 in seed 42; −0.61 and
  −0.87 in seed 43; Spearman(a, b) = 0.95 in seed 42. The seasonal terms are not independent outputs; they ride the
  storage latent. Across seeds T0 ranks agree (0.82) but the phase difference per dam is a median −38° with IQR −89° to
  +76°, and the amplitude rank agreement is 0.46. The seasonal terms are not identified.
- Learned a, b have no relation to the offline fit's (Spearman 0.05-0.18), and the offline fit itself gains nothing from
  seasonality: median (seasonal gain − linear gain) = −0.0011 NSE at active dams, with amplitudes at the ±2 box edge
  (median 1.5, 70 % ≥ 1) and phases spread over the whole circle. Seasonality is unsupported at both ends.
- Which features carry T0 (standardised linear coefficients, seed 42 / 43): log storage_max +0.47 / +0.43, flood
  purpose +0.30 / +0.28, surface +0.16 / +0.18, year +0.15 / +0.11, height +0.11 / +0.10, hydro −0.12 / −0.11,
  recreation −0.19 / −0.13, surface_missing −0.17 / −0.21. Seed-split (collinear) inputs: log storage +0.20 / +0.01,
  log drainage −0.20 / 0.00, storage_per_area −0.02 / +0.21, max_discharge +0.02 / −0.21. Storage, drainage and
  storage-per-area are one quantity in log space; the head splits them arbitrarily.
- Offline support for the features (Spearman with the offline seas_T0 at active smoke dams): NID degree of regulation
  0.38, storage 0.30, height 0.19, drainage area 0.19, max discharge −0.19, completion year 0.02, cascade no
  difference (median 1.75 d either way). Purpose: flood 81 % active, irrigation 95 %, hydro 67 %, supply 57 %.
  So DOR (storage over mean annual flow) is the best single predictor and it is not in the feature table
  (storage/area is a poor proxy: 0.14). Year is used by the head (+0.15) but has no offline support; its role belongs
  in the activation-by-completion-year mechanism the other agent is adding, not in the feature vector.

### 1.5 The loss population moves the learned T0

Same seed, same code: full population T0 median 0.35 d, smoke set 0.59 d, per-dam ratio 1.38, Spearman 0.77;
seasonal amplitude 0.13 vs 0.31. The smoke set's loss is half dam gauges within 3x of their dam's area; the full
population's dammed gauges are mostly far below their dams (540 of 917 not on the dam's reach). The release head's
only gradient comes from gauges below dams, and at diluted gauges a dam bucket is mostly cost, so they pull T0 down.

### 1.6 Compensation with the routing head is global, not local (`n_above_dams.py`)

Per-gauge mean Δlog n (learned − off) over the gauge's subgraph, smoke set:

| | on-reach dam (214) | any dam (458) | control (458) | dam − matched control |
|---|---:|---:|---:|---:|
| seed 42 | −0.100 | −0.081 | −0.070 | −0.012 (on-reach −0.015) |
| seed 43 | −0.396 | −0.383 | −0.323 | −0.060 (on-reach −0.055) |
| off 43 − off 42 (seed noise) | −0.161 | −0.122 | −0.088 | −0.034 (on-reach −0.061) |

The dam-versus-control contrast in Δn between arms is no larger than the same contrast between two no-dam seeds. No
localised n compensation above dams is detectable. What IS large is the global shift: in seed 43 the learned arm's n
is 0.68x the off arm's over all of CONUS (median 0.041 vs 0.061) with gamma 0.043 vs 0.272; in seed 42 n is 1.04x.
The sign pattern (seed 42: rougher channels, T0 0.35 d; seed 43: smoother channels, T0 0.53 d) is what a global
n-versus-storage trade looks like, but two seeds cannot establish it. The undammed-gauge cost (−0.0007 / −0.0004) has
to be this global routing-head re-solution; a 10-attribute shared head cannot aim at dams.

### 1.7 Seeds (`seed_power.py`)

Per-gauge paired ΔNSE at dammed gauges: SD 0.32 (heavy tails), IQR −0.004 to +0.016, bootstrap SE of the median
0.0003. The two seeds' medians (+0.0014, +0.0039) imply a seed SD of about 0.0018, six times the bootstrap SE; the
page's intervals are within-seed only. For a two-arm difference with n seeds per arm, t = 1.5 / 2.1 / 3.0 at n = 2 /
4 / 8 on the full population. The undammed cost (seed SD about 0.0002-0.0006) needs 3-4 seeds for t > 3. On the smoke
set the dammed effect is 4x larger, so 3-4 seeds suffice there. But see opportunity 1: a test-phase decomposition
makes seeds unnecessary for attribution.

## 2. Ranked opportunities

Ranking is (expected gain x confidence) / cost. Costs are CPU, smoke set: about 1 h per training arm, 18 min per test
pass; full population 2.4 h and 35 min.

### 1. Test-phase decomposition of the existing runs (no training; 5 test passes)

Re-run the test phase of each learned run (both seeds, smoke and full) with (a) the learned routing head and dams OFF;
(b) dams on with a = b = 0 (fixed seasonal table from `release_params.csv`); (c) T0 x 2, x 4, x 8 in the same table.
Everything is the existing `fixed` path, config plus CSV.
- (a) splits the paired effect into "release" (learned-with-dams minus learned-without-dams, same routing head, undammed
  gauges bitwise identical) and "routing head change" (learned-without-dams minus off arm). This answers the undammed
  cost and the dammed attribution with zero seed noise. Evidence: §1.6.
- (b) tests whether seasonality earns anything in test. Evidence: §1.4 (rank-1, unreproducible phase, no offline gain,
  KGE falls at dammed gauges −0.0009 / −0.0014).
- (c) measures, in the engine, whether the short T0 costs skill: the offline band says the objective is flat within a
  decade, and 39 % of dams sit below the band. Evidence: §1.2.
Go/no-go: if (c) at x2 or x4 raises dammed-gauge ΔNSE by > 0.002 paired with a CI clear of zero and no control loss,
training under-shoots T0 and it matters (run opportunities 2-4); if flat, the short T0 is a parameter-level, not a
skill-level, deficiency and the paper should say so. For (b): if a = b = 0 is within ±0.001 at dammed gauges and KGE
recovers, ship with seasonality off. For (a): if the release-only term carries ≥ 80 % of the dammed gain, the
co-training cost is separable and the two-stage protocol (opportunity 2) is the production recipe.

### 2. Freeze the routing head; train the release head alone (one small code change, 2 smoke arms)

Add a per-head learning rate (`release_head.learning_rate`, with the routing head's lr allowed to be 0). Train the
release head on the smoke set from the off arm's `epoch_50_mb_9` checkpoint with the routing head frozen, seed 42 and
43. This is the in-engine analogue of the offline fit with a shared head, and it separates the two remaining causes:
joint adaptation with (n, gamma) versus the shared head's smoothing and features. Controls are bitwise unchanged, so
check 5's control criterion is met by construction.
Go/no-go: on-reach learned T0 inside the offline 0.02-NSE band at ≥ 75 % of active dams and median log10(learned/fit)
> −0.1 means joint adaptation was absorbing the storage: adopt a two-stage protocol (release head on the frozen
routing head, then joint at the release head's decayed lr) and re-run the paired check. If T0 stays at 0.4-0.8 d with
the routing head frozen, the head or the loss population is the limit: go to 3 and 4.
Cost: about 40 lines in `src/training/driver.rs` (optimizer step skipped when lr is 0) plus config, 2 h CPU.
Risk: none to invariants (forward and backward unchanged); the routing head's frozen state must be the same checkpoint
in both arms.

### 3. Per-dam free T0 as the head's ceiling, then the feature revision (code + 2 smoke arms, then CSV-only arms)

(i) A per-dam learnable log T0 (an embedding, no features), same init, same optimizer, trained jointly on the smoke
set. It bounds what any release head can deliver on this objective and network, and its gap to the offline ceiling
(+0.0068 at dam gauges, +0.0085 dam minus control) is the cost of the shared-head assumption. (ii) Feature revision,
CSV-only (`build_dam_features.py`): add log10 DOR = storage / (mean Q' at the dam reach x 1 yr), computed from the
forcing store (model-internal, not observed dam data); collapse storage / drainage / storage-per-area to storage_max
plus DOR; keep purpose one-hots, height, surface and the missing flags; remove year from the features once the
completion-year activation exists. Evidence: §1.4 (DOR 0.38 vs storage/area 0.14 offline; seed-split coefficients on
the collinear trio; year unsupported offline).
Go/no-go for (i): dammed-gauge gain ≥ 90 % of the offline ceiling and T0 in-band ≥ 75 %. If the free-T0 arm also stops
at 0.6 d, the gap is the joint objective (opportunity 2 / 4), not the head. For (ii): with the frozen routing head of
opportunity 2, the revised features must raise Spearman(learned T0, offline T0) above 0.6 (now 0.47-0.50) and
reproduce coefficients across two seeds within 0.1.
Cost: embedding is about a day of Rust (a `[n_dams]` parameter with its own optimizer next to the head; checkpoint,
resume, the resolved test table); features are an afternoon. Risk: a per-dam embedding needs every table dam in
training batches often enough (about 6 % of dams per micro-batch now).

### 4. Loss population and weighting toward regulated gauges (no code if paired with 2; 2 smoke arms)

Evidence: the same head learns T0 1.4x larger and seasonality 2.4x larger on the smoke population than on the full
population (§1.5); the release head's gradient comes only from gauges below dams and most of those are diluted. With
the routing head frozen (opportunity 2), train the release head on (a) the on-reach dam gauges only (214) and (b) the
full smoke set; compare learned T0 and the paired gain at the same 214 gauges. If weighting is needed in production, a
per-gauge loss weight `1 + DOR` (or oversampling gauges by regulation) in `batch_loss` is a small change.
Go/no-go: (a) reaches T0 in-band ≥ 75 % where (b) does not, with the on-reach gain up by ≥ 0.002.
Risk: with the routing head unfrozen, reweighting changes the routing solution everywhere; keep it release-only.

### 5. Release-head learning rate, schedule and step budget (config only, 2 smoke arms)

Evidence: seed 43's T0 is still rising at epoch 41-50 under lr 0.001; seed 42 plateaued by epoch 10; the release head
starts from a zero read-out and shares the routing head's decaying schedule; the per-step Adam displacement bound
(0.005 x 200 + 0.002 x 200 + 0.001 x 100 = 1.5 logit units per weight) is of the order of the travel the tail dams
need (u from 0.17 to 0.5 is 1.6 logits on the T0 bias alone). Give the release head its own constant lr (0.01) with
the routing head's schedule unchanged, and log the resolved release table every epoch (cheap: one head forward over
1,024 rows) so per-dam trajectories exist; run 100 epochs once.
Go/no-go: batch-median T0 flat over the last 20 epochs in both seeds and the between-seed per-dam T0 ratio within
1.0 ± 0.2 (now 1.33). If the plateau of seed 42 is reproduced at the higher lr, the budget is not the limit and the
flatness in §1.2 is.
Risk: oscillation in a, b early; mitigated by opportunity 1(b) turning seasonality off.

### Lower-ranked

6. Seasonality off by default (a = b = 0) until a multi-month objective exists: no offline gain, rank-1 with T0,
   phase unreproducible. Folded into 1(b); zero cost.
7. Longer rho for the long-T tail: 20 % of active on-reach dams have fit T0 > 10 d, which 90-day windows truncate to
   13-17 d and 365-day windows recover to about 50 d (§1.1). A rho 365 / warmup 30 arm costs 4x per epoch (about 4 h
   on smoke). `experiment.state_cache` does not help (oracle initial state gives the same T0 as the hotstart) and a
   no-dam teacher's state is wrong at dam rows anyway. Do this after 1-3, and shrink the T0 box's upper bound to about
   60 d so the sigmoid's live range is not 40 % dead: even a 365-day window cannot see beyond that.
8. Seeds: 4 seeds per arm on the smoke set (8 h) once the protocol from 1-2 is fixed; report the seed SD next to the
   bootstrap interval. Do not spend seeds on the undammed cost; opportunity 1(a) settles it bitwise.
9. Loss blow-ups: cap the per-gauge normalised loss (or drop a gauge whose micro-batch term exceeds, say, 100x the
   batch median) so a routing oscillation at one gauge does not own a step. One to two steps of 500 per run; small.
10. Equifinality instrument: the paper studies refuse `use_reservoirs` arms (`src/experiment/adjoint/influence.rs`),
    so the curvature instrument cannot yet see T0. The cheap proxy above (Δlog n above dams vs controls vs seed noise,
    `n_above_dams.py`) is already decisive at the resolution two seeds allow; extending the landscape objective to
    (n, T0) at the on-reach gauges is worth it only after the two-stage protocol shows a stable T0.

## 3. What looks wrong in the current setup

1. The routing head's (n, gamma) solution is degenerate and unconverged (median gamma 0.067 / 0.168 / 0.272 / 0.043
   across the four arms; seed 43's learned arm has n 0.68x its off arm everywhere). Every paired dam-versus-no-dam
   number inherits this wander, and the undammed-gauge cost is its footprint. Freezing the routing head for the
   release stage, or at least reading the release effect at a fixed routing head (opportunity 1a), is required
   before any further paired claim.
2. Seasonal terms are trained but not identified: rank-1 with T0 within a seed, phase not reproducible across seeds,
   no gain in the offline fit that motivated them, and a KGE cost at dammed gauges.
3. One learning-rate schedule for two heads at different stages of training; the release head starts from a zero
   read-out and in seed 43 is still moving when the schedule has decayed to 0.001.
4. The T0 box [1 h, 365 d] with 90-day windows: nothing above about 50 d is learnable by construction (§1.1), so the
   upper 40 % of the log box is dead weight on the sigmoid; document or shrink it.
5. Bootstrap intervals on the page are within-seed; the seed SD of the dammed median (0.0018) is 6x the bootstrap SE
   (0.0003). Quote both.
6. Single micro-batch losses of 1e6-1e9 (routing blow-ups) reach the optimizer; clipped, but the step is wasted.
7. Not this reviewer's angle, but it affects the reading of check 4: a dam row replaces its reach's channel routing
   (K about 4 h at the median), so the learned T0 includes the replaced travel time and cannot stay at 4.5 h where no
   storage exists; and the offline fit adds its bucket on top of the routed flow, so learned and fitted T0 differ by
   about K_reach before any learning question arises.

## 4. Answers to the specific prompts

- 90-day windows vs long T (state_cache, longer rho, warmup): refuted as the cause below 10 d; tail-only fix; state
  cache useless for this; see §1.1 and item 7.
- Separate lr / schedule / init for the release head: warranted by seed 43's unconverged T0 (item 5); init is fine.
- Curriculum: freeze-routing-first is the decisive diagnostic and likely the production protocol (item 2); pretraining
  the release head is the same thing with the joint phase appended.
- Loss weighting toward regulated gauges: evidence exists that the population moves T0 by 1.4x (item 4).
- Compensation between roughness and storage: not local (Δn above dams is within seed noise), global and inside the
  (n, gamma) degeneracy; measure with the frozen-head contrast, not the adjoint (which refuses reservoir arms).
- Features: storage_max and flood purpose carry the signal; add DOR from mean Q'; drop the collinear trio; year belongs
  to the activation mechanism; cascade unsupported; max discharge weak and seed-split (item 3ii).
- Multi-seed protocol: 4 seeds/arm on smoke for the dammed effect; none needed for the undammed cost after item 1a.
- Convergence: the objective is flat after epoch 10; T0 unconverged in one seed; log per-epoch release tables.

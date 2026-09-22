# Regional leakance-frame curvature census: ten gauges per HUC2

**Date:** 2026-09-22
**Runs:** `.ddrs/experiments/landscape-huc-leakance/2026-09-22T01-56-55Z-shard-{0..15}-of-16`
(16 shards, 180 gauges, all `success`, 4 h wall on CPU, median 22 min per gauge)
**Bundle:** `experiments/landscape-huc-leakance/` (gauge list in `huc_selection.csv`)
**Arm:** `twoway-ep30` = `2026-09-17T16-38-16Z-train-and-test` @ `epoch_30_mb_9`
(two-way leakance + learned n(d), CONUS, NSE 0.7300 / KGE 0.7567)
**Outputs:** `output/huc_leakance/` (per-gauge CSV, tables, summary figures, four curvature
landscapes under `landscapes/`)
**Prior reads this extends:** the Juniata leakance frame
(`experiments/landscape-jrb-leakance`, journal 2026-09-19) and the curvature landscapes of
2026-09-21 on three Juniata gauges, where roughness was 100 to 300x stiffer than either
leakance parameter at the trained point and the loss was a plateau in `K_D` and `d_gw`.

## 1. Question

Does the daily hydrograph constrain the leakance parameters, the conductance-over-thickness
ratio `K_D` and the water-table offset `d_gw` in
`zeta = alpha * A * K_D * (d - d_gw)`, differently in different parts of CONUS? The Juniata
read said the loss does not see them over a factor of ten either way. The user's question was
whether arid regions, where losing streams are common, look different.

Predicted before the run (2026-09-21 session): flatter still at the trained point in arid
regions, because the head parks the flux even lower there; the corner of the `K_D x d_gw` plane
where the loss finally bends should arrive earlier; intermittent gauges may not produce a
clean surface.

## 2. Design

- Ten gauges in each of the 18 HUC2 regions (HUC02 from the GAGES-II shapefile, 10U and 10L
  merged), drawn from the 2,365-gauge population that produced a WY2000 landscape in the
  `landscape-p21-all` census, restricted to 3 to 300 reaches, evenly spaced in network-size
  rank within each region. Every region had at least 21 eligible gauges.
- Axes `[n, K_D, d_gw]`, log multipliers, box +-ln 10, planes `n-kd`, `n-dgw`, `kd-dgw`
  through the **trained** point, grid 25^2, `fd_step 0.025`, damped Newton 12 iterations,
  objective `nse-batch`, WY2000, one window. Identical to the Juniata leakance bundle.
- Read-out per gauge: the fine-step Hessian at the trained point (`hess0`), the fraction of
  the `K_D x d_gw` grid within 5 % of the trained loss ("plateau"), the best NSE reachable on
  that plane with `n` held at its trained field ("leak-only gain"), where that best cell sits,
  and the Newton optimum.

## 3. Results

### 3.1 The trained fit is broken at 55 of 180 gauges, all in the West and Gulf

| region | HUC2 | trained NSE < 0 | trained NSE < 0.3 |
|---|---|---:|---:|
| Souris-Red-Rainy | 09 | 6 | 8 |
| Missouri | 10 | 5 | 8 |
| Texas-Gulf | 12 | 6 | 6 |
| Rio Grande | 13 | 9 | 10 |
| Lower Colorado | 15 | 8 | 9 |
| Lower Mississippi, Arkansas-White-Red, Upper Colorado, Great Basin | 08, 11, 14, 16 | 3 to 5 each | 5 to 7 each |
| the nine humid regions | 01 to 07, 17, 18 | 0 to 2 each | 1 to 5 each |

At these gauges the observation variance is tiny (mean observed flow at the worst eight is
under 0.4 m^3/s, one is 3e-5), the normalised loss is enormous (trained NSE down to -2e5),
and any move that removes water "improves" NSE by tens to thousands of units. The
leak-only gain correlates with the trained NSE at Spearman -0.88 across all 180: the gain is a
recovery from a broken fit, not information about leakance. Everything below is therefore
read on the 105 gauges with trained NSE >= 0.3, with the full-population table kept for
reference (`HUC_LEAKANCE_TABLE_all.md`).

### 3.2 Well-fit gauges: the plateau holds in the humid regions and breaks in the plains

Medians per region, trained NSE >= 0.3 (`HUC_LEAKANCE_TABLE_wellfit.md`):

| HUC2 | region | n | trained NSE | \|H_n\| / \|H_KD\| | \|H_n\| / \|H_dgw\| | H_n < 0 | plateau | leak-only dNSE | K_D at wall |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 01 | New England | 9 | 0.83 | 7.8 | 27 | 1 | 0.73 | +0.008 | 2 |
| 02 | Mid-Atlantic | 8 | 0.67 | 58 | 254 | 1 | 0.74 | +0.027 | 6 |
| 03 | South Atlantic-Gulf | 7 | 0.72 | 31 | 56 | 3 | 0.78 | +0.013 | 3 |
| 04 | Great Lakes | 8 | 0.71 | 6.7 | 20 | 3 | 0.67 | +0.041 | 4 |
| 05 | Ohio | 10 | 0.79 | 81 | 300 | 3 | 0.93 | +0.012 | 5 |
| 06 | Tennessee | 9 | 0.68 | 97 | 384 | 2 | 0.92 | +0.005 | 1 |
| 07 | Upper Mississippi | 7 | 0.71 | 62 | 201 | 1 | **0.44** | +0.032 | 3 |
| 08 | Lower Mississippi | 6 | 0.58 | 58 | 110 | 3 | 0.80 | +0.063 | 1 |
| 09 | Souris-Red-Rainy | 2 | 0.56 | 24 | 41 | 0 | **0.25** | +0.145 | 1 |
| 10 | Missouri | 5 | 0.49 | 66 | 26 | 2 | **0.30** | +0.124 | 2 |
| 11 | Arkansas-White-Red | 5 | 0.56 | 62 | 135 | 2 | 0.84 | +0.021 | 1 |
| 12 | Texas-Gulf | 4 | 0.69 | 1100 | 5900 | 3 | 0.88 | +0.012 | 2 |
| 13 | Rio Grande | 1 | 0.34 | 2.1 | 4.7 | 1 | 0.84 | +0.083 | 0 |
| 14 | Upper Colorado | 5 | 0.70 | 2.5 | 7.0 | 2 | 0.99 | +0.002 | 1 |
| 15 | Lower Colorado | 1 | 0.85 | 0.4 | 0.2 | 1 | 0.77 | +0.011 | 0 |
| 16 | Great Basin | 5 | 0.69 | 7.2 | 6.3 | 1 | 0.84 | +0.033 | 2 |
| 17 | Pacific Northwest | 7 | 0.85 | 7.7 | 17 | 3 | 0.86 | +0.009 | 1 |
| 18 | California | 6 | 0.86 | 69 | 210 | 2 | 0.96 | +0.002 | 1 |

CONUS over the 105: plateau median 0.84, leak-only gain median +0.015 NSE, `H_n < 0` at 34,
`K_D` at the sweep wall at 36.

Three regimes:

1. **Humid East, Ohio, Tennessee, California, Pacific Northwest** (01 to 06, 17, 18): the
   Juniata picture generalises. Roughness is 10 to 400x stiffer than either leakance
   parameter, over 85 % of the `K_D x d_gw` box sits within 5 % of the trained loss, and the
   best reachable NSE by moving leakance alone is +0.002 to +0.04. The Scioto below
   O'Shaughnessy Dam (`landscapes/curvature3d_huc_03221000.png`, 53 reaches, NSE 0.80) is the
   type case: the loss is a cylinder in the `n` planes and a plateau in the leakance plane,
   bending only in the corner where both parameters compound.
2. **Plains and prairie: Upper Mississippi, Missouri, Souris-Red-Rainy** (07, 09, 10): the
   plateau collapses to 25 to 45 % of the box and leakance alone reaches +0.03 to +0.15 NSE at
   well-fit gauges. The Little Sioux at Linn Grove, Iowa
   (`landscapes/curvature3d_huc_06605850.png`, 79 reaches, NSE 0.45 to 0.80 on the leakance
   plane alone) is the type case: the loss slopes across most of the `K_D x d_gw` box, so the
   gauge does see the leakance parameters there, with `n` still 26 to 66x stiffer.
3. **Interior West** (13 to 16): too few well-fit gauges to say anything per region (1 to 5),
   and at those `n` itself is nearly flat (ratios of 0.2 to 7, `H_n < 0` at most). The
   arid-region question is not answered by this census because the arm does not fit the arid
   gauges; see 3.1.

### 3.3 Where the optimiser sends leakance

Where any gain exists, the best cell is the same corner at 19 of 105 well-fit gauges and at
most of the broken ones: `K_D` at the sweep ceiling and `d_gw` at the floor, which under the
log-multiplier sweep is the maximal *losing* flux. In the humid regions the best `d_gw` cell
is instead above the trained value (medians +0.4 to +0.9 log units), which reduces the losing
flux. So the sign of what the gauge wants from leakance is regional: less loss in the East,
more loss in the plains and West. Whether the plains signal is groundwater exchange or a
volume bias in the inflow product cannot be separated by this read; the July diagnosis found
the learned flux tracks river size rather than dryness, which points at the latter.

### 3.4 Negative roughness curvature at the trained point

`H_n < 0` at 34 of 105 well-fit gauges and 81 of 180 overall, concentrated in the West. This
repeats the strat400 finding (137 of 400 over an 1826-day window). At those gauges the
trained `n` field sits on a crest of the one-year loss, not in a bowl, so "n is the stiff
direction" is a statement about magnitude, not about the trained point being a minimum.

## 4. Caveats

- One arm, one seed, one water year, ten gauges per region, network size capped at 300
  reaches (the largest mainstem sites are excluded everywhere).
- The 5 % plateau criterion and the leak-only gain are both relative to the trained loss, so
  they are scale-free, but the Hessian magnitudes are not comparable across gauges with very
  different observation variance. Only the ratios and the plateau are compared across regions.
- `d_gw` is swept as a log multiplier, so `d - d_gw` never changes sign in the box; the
  gaining-reach regime is untested, as in the Juniata read.
- Grid cells with more than 5 % of reaches clamped are blank; in the `n` planes this is
  about half the box at large gauges.
- HUC02 comes from the GAGES-II shapefile. It is not the USGS station-number prefix.

## 5. What this changes

- The Juniata "leakance is flat" read is a **humid-region** statement, now supported at 8 of
  18 regions with 6 to 10 well-fit gauges each. It is **refuted in the plains** (07, 09, 10),
  where the loss does respond to `K_D` and `d_gw` at well-fit gauges.
- The arid-region question is **not answered**: the arm's trained fit is negative at 6 to 9 of
  10 gauges in each arid region, and the census cannot read curvature on a broken fit.
- Any further leakance arm should be trained or evaluated with those two facts in front of
  it: the term has leverage in the plains, and it has no meaningful gauge signal in the arid
  West under this loss and this Q' product.

## 6. Follow-ups, in order

1. Plains only: re-run the ten Missouri and Upper Mississippi gauges with an **additive**
   `d_gw` sweep that crosses the water table, to see whether the slope is toward gaining or
   losing once the sign can change.
2. Arid regions: the census needs an arm and an objective that fit them first. Either
   the hourly-lstm inflow arm or a loss that does not divide by a near-zero variance
   (KGE, or NSE with a flow floor).
3. Separate leakance leverage from inflow-volume bias in the plains: compare the leak-only
   gain against the summed-Q' baseline's volume bias at the same gauges.

## Reproduction

```bash
cargo build --release --bin ddrs
for i in $(seq 0 15); do
  target/release/ddrs --workspace .ddrs experiment landscape-huc-leakance --backend cpu --shard $i/16 &
done; wait
~/projects/ddr/.venv/bin/python experiments/landscape/huc_leakance_summary.py \
    ".ddrs/experiments/landscape-huc-leakance/<ts>-shard-*-of-16" \
    --sel experiments/landscape-huc-leakance/huc_selection.csv --out output/huc_leakance --min-nse0 0.3
~/projects/ddr/.venv/bin/python experiments/landscape/curvature3d.py <gauge.nc> --out <dir> --tag huc
```

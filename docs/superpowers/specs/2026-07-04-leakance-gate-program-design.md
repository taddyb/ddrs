# Leakance promotion-gate program — design

Date: 2026-07-04
Worktree: `zeta-sensitivity` (branch `worktree-zeta-sensitivity`); extraction
work lives in `/home/tbindas/projects/extractrs`.
Prior instruments: `docs/2026-07-02-leakance-diagnosis-findings.md`,
`docs/2026-07-03-zeta-gradient-probe-findings.md`,
`docs/2026-07-04-synthetic-recoverability-findings.md`,
`docs/2026-07-04-leakance-literature-review.md`.

## 1. Question

Under a sound training objective and with genuinely informative inputs, does a
sign-constrained leakance term improve real-observation metrics (NSE, KGE,
high-flow and low-flow bias) **without repainting the roughness/geometry
solution**, and is its learned field consistent with independent groundwater
data? Leakance is enabled everywhere in routing ONLY if it passes a
pre-registered three-leg gate; otherwise it stays experimental (documented
NO-GO).

### Decisions driving this design (user-settled, 2026-07-04)

1. **Floor first.** The recoverability control found the windowed training
   objective has a hotstart-transient noise floor of ~1.0 mean L1 (~130× the
   reach-scale signal; ~40% of a converged run's loss budget). Every prior
   leakance result was measured under it. Fix the objective before judging the
   term.
2. **Promotion requires attribution, not just metrics.** Adding learned fields
   adds equifinality — an NSE gain that comes from leakance cannibalizing
   Manning's n is a loss, not a win. The gate has an explicit
   parameter-stability leg.
3. **Losing-only, everywhere, via a clamp — not a spatial mask.** Streams gain
   or lose with the water table (continuous physics), but in ddrs the gaining
   branch duplicates Q′-baseflow (dHBV2 already delivers groundwater gains at
   the divides) — a second baseflow knob is manufactured equifinality. So
   `zeta = factor · area_z · K_D · max(0, depth − d_gw)`: sign discipline is
   structural; WHERE losing occurs is learned.
4. **The KAN route, properly informed.** `d_gw` was never learnable from the
   current attributes — they contain no groundwater information (empirically:
   the diagnosis's d_gw≈depth collapse; physically: Condon & Maxwell 2015 —
   water-table depth is not a function of topography/soils; and `permeability`
   exists in the attributes file but was never in `input_var_names`). The fix
   is better inputs, not a prescribed field.
5. **Channel-corridor data, not basin averages.** Wells are not in channels;
   basins average away the channel corridor. All bed-process attributes are
   sampled ALONG THE REACH POLYLINE (buffered corridor), including the
   water-table field (channel pixels of physics-interpolated grids, converted
   to bed-relative head) and bed-material indicators.
6. **Impermeable channels get a structural zero.** Concrete-lined reaches
   (LA-River class) have bed K ≈ 0 regardless of head; near-channel
   imperviousness above a high threshold sets `zeta ≡ 0`. This is an
   engineering fact, not a hydrogeologic inference — no equifinality cost.
7. **Extraction is first-class work in `extractrs`** — scripts and any library
   changes needed for corridor extraction live in
   `/home/tbindas/projects/extractrs`.

## 2. Architecture — three phases, three plans

Phases A and B are independent (different repos, different skills) and can run
in parallel; Phase C consumes both. Each phase gets its own implementation
plan.

```
PHASE A (extractrs)                    PHASE B (ddrs)
Channel-corridor + GW attributes       Objective floor fix
────────────────────────────          ────────────────────────────
A0 network crosswalks                 B1 floor-vs-warmup curve
   (MERIT↔SWORD published;               (forward-only, teacher weights
    NHDPlus→MERIT built once)             on self-generated obs;
A1 precomputed per-reach transfer         warmup ∈ {5,15,30,60} @ rho 90,
   (StreamCat imperviousness,             {90} @ rho 180; decay stratified
    Zarrabi bankfull geometry,            by upstream area)
    SWORD widths, confinement, BFI)    B2 pick + implement the fix
A2 corridor extraction                    A: longer warmup/rho (config-only)
   (channel WTD bed-relative +            B: state-cache hotstart (windows
    losing_fraction — the only            init from a saved continuous run's
    targets with NO per-reach             reach states)
    product; two-tier buffers)         B3 validate: fixed-objective floor
A3 basin-scale additions                  ≤ 0.25 mean L1 (≤10% of converged
   (permeability unlock,                  loss, vs ~40% today)
    drainage density from MERIT)
A4 per-COMID netCDF in the exact
   global.nc schema (dim COMID,
   f64 vars, NaN outside CONUS)
   + normalization statistics
        │                                   │
        └───────────────┬───────────────────┘
                        ▼
PHASE C (ddrs) — the gate experiment
C1 losing-only clamp + impervious hard-zero in src/routing/leakance.rs
   (max(0, ·) on the head term; static zero mask from A's imperviousness)
C2 retrain the pair: hourly, cold, seed 42, fixed objective,
   enriched inputs (BOTH cells — pair stays fair)
     OFF: no leakance          ON: losing-only leakance
C3 three-leg gate (pre-registered below) → PROMOTE / KILL / REVISE
```

### The enriched input architecture (two-scale, matching the physics)

```
zeta = factor · area_z · K_D · max(0, depth − d_gw)
                         │               │
   bed-K axis (CORRIDOR):          head axis (CORRIDOR):
   corridor_impervious (NLCD)      channel_wtd_bed_rel (Zell&Sanford
   alluvium_fraction (Soller)        primary, Fan cross-check; channel
   corridor lithology (GLiM,         pixels, minus per-reach bankfull
     backstop)                       depth estimate)
                         │         losing_fraction (fraction of channel
   aquifer-K axis (BASIN):           pixels with WT below bed —
   permeability (in attributes       sub-reach mixing)
     file, currently UNUSED —
     add to input_var_names)
   sand/clay/Porosity (existing)
   BFI, drainage_density (new)
```

Per-reach `d_gw` is defined as the conductance-weighted mean groundwater head
beneath the wetted channel relative to the bed, static (climatological). The
existing `d_gw ∈ [−2, 2] m` bound doubles as a crude disconnection cap
(Brunner 2009: flux saturates below a critical water-table depth), so driving
head never exceeds `depth + 2 m` — retained deliberately.

## 3. Phase A — attribute extraction (extractrs)

**Sourcing principle (2026-07-04 literature pass, see
`docs/2026-07-04-leakance-literature-review.md` §6): precomputed per-reach
products first; raw-raster corridor extraction ONLY where no per-reach product
exists.** The channel-characteristics search found most of our targets already
published per-reach on NHDPlusV2 or SWORD — the genuinely novel extraction is
the channel water-table field.

Deliverable: a per-COMID attribute netCDF **in the exact `global.nc` schema**
— single dimension `COMID` (int64 ids), one float64 variable per attribute, no
extra dims — so ddrs's existing `AttributesStore` reads it unchanged
(verified: `merit_global_attributes_v2.nc` = dim `COMID` (2,939,404), f64
vars). CONUS-only coverage is NaN-filled outside CONUS on the full global
COMID vector (copied from `global.nc`), keeping index compatibility. Plus
normalization statistics (ddrs stats-JSON convention) for every new attribute.

### A0 — network crosswalks (one-time infrastructure)

**ID-space clarification (do not skip).** Within MERIT-Basins, flowline =
unit catchment = COMID, strictly 1:1 — no crosswalk is ever needed inside
MERIT. Crosswalks exist only because the attribute PRODUCTS are keyed to
FOREIGN networks. ⚠ **Name collision:** NHDPlusV2's reach identifier is ALSO
called "COMID" but is a completely different ID space (~2.7M reaches at
1:100k scale vs MERIT's ~346k CONUS reaches from the 90 m DEM; MERIT-Basins
borrowed the name). A StreamCat/Zarrabi "COMID" is an NHDPlus COMID, never a
MERIT COMID.

A crosswalk here is a reach-ID → reach-ID lookup table with along-channel
overlap lengths as weights (the two networks segment the same rivers at
different breakpoints, so one MERIT reach maps to PORTIONS of several foreign
reaches). Attribute transfer = length-weighted average over the matched
foreign reaches. Nothing geometric happens at use time; geometry is consumed
once, when the table is built.

| Crosswalk | Status | Method |
|---|---|---|
| MERIT ↔ SWORD | **published** — Wade et al. 2025 (WRR, 10.1029/2024WR038633; Zenodo 10.5281/zenodo.13152826): per-pfaf-2 NetCDF tables, `sword_1..sword_40` ranked matches + `part_len_1..part_len_40` intersection lengths (m), bidirectional | download + apply length-weighted join |
| NHDPlusV2 → MERIT | not published | build via buffered-geometry intersection with length-weighted matching (mirroring Wade et al.'s method) in extractrs — reusable artifact for StreamCat/Zarrabi/McManamay transfer; emit the same `(comid, nhd_1..nhd_k, part_len_1..part_len_k)` table shape + quality flags |

### A1 — precomputed per-reach products (transfer, don't rasterize)

| Attribute | Source (per-reach, verified downloadable) | Transfer |
|---|---|---|
| `corridor_impervious` | EPA StreamCat `PctImp2019Rp100Cat` (Hill et al. 2016) — NLCD imperviousness in a 100 m riparian buffer, precomputed for 2.65M NHDPlus reaches | NHDPlus→MERIT crosswalk, length-weighted |
| `bankfull_depth`, `bankfull_width` | Zarrabi et al. 2025 (WRR 10.1029/2024WR037997; Zenodo 13883263) — ML bankfull geometry for 2.7M NHDPlus reaches (R²=0.85 width / 0.69 depth; supersedes Bieger 2015 curves AND our uparea power-law placeholder) | NHDPlus→MERIT crosswalk |
| `channel_width_obs` | SWORD v2 (Altenau et al. 2021) GRWL-derived Landsat widths | MERIT-SWORD (published) — rivers ≥30 m; NaN below |
| `valley_confinement` | McManamay & DeRolph 2019 stream classification (confined/mod/unconfined — alluvial-setting proxy) | NHDPlus→MERIT crosswalk |
| `bfi` | USGS/Wolock gridded BFI | basin zonal mean (existing extractrs path) |
| `drainage_density` | computed from MERIT fabric itself | channel length / catchment area (no raster) |

### A2 — corridor extraction (the genuinely novel part)

No precomputed per-reach product exists for the channel water-table field —
this is ours to build:

| Attribute | Source raster | Extraction |
|---|---|---|
| `channel_wtd_bed_rel` | Zell & Sanford 2020 (ScienceBase 10.5066/P91LFFN1) primary; Fan 2013 (THREDDS) cross-check | channel-cell WTD − Zarrabi `bankfull_depth` (per-reach, replaces the uparea power-law) |
| `losing_fraction` | same | fraction of channel cells with WT below bed (threshold in xarray, then corridor mean) |
| `alluvium_fraction` | USGS Principal Aquifers polygons (10.5066/P9Y2HOUJ) + Soller surficial materials | polygon/corridor overlay fraction in alluvial classes |

**Buffer strategy (literature-grounded — MERIT flowlines are NOT positionally
exact):** Amatulli et al. 2022 benchmarked 90 m-DEM networks against NHDPlus
HR and found <50% of cells within 100 m of the reference — typical MERIT
lateral error is 100–300 m, worst in flat valleys (Yamazaki et al. 2019's
stated worst case). Two-tier design, matching published practice:
- **Fine rasters (30 m NLCD-class, if ever needed raw):** fixed **100 m
  half-width** corridor — the StreamCat precedent (Hill et al. 2016) — widened
  to **200 m** where GFPLAIN250m (Nardi et al. 2019) marks a broad floodplain
  (flat-valley/braided error tail).
- **Coarse grids (Zell & Sanford / Fan WTD at ~500 m–1 km):** NO fine buffer —
  the cell already spans the positional error; take the nearest-channel-cell
  value along the polyline (RiverATLAS-style native-grid association, Linke
  et al. 2019), with the 100 m-buffer variant computed once as a sensitivity
  column.

Mechanics:
- Corridors = MERIT reach polylines buffered (GeoDataFrame) → existing
  `ds.extrs.zonal_stats(corridors, id_col="COMID")`. Overlap at confluences is
  fine (per-feature exact stats).
- Library changes IN extractrs only where a stat is missing (categorical
  fraction; fraction-below-threshold prefers xarray preprocessing + mean).
- Validation: spot-check ~10 known reaches (LA River `corridor_impervious`
  ≈ 100; an Ogallala losing reach with deep channel WTD; an Appalachian
  gaining reach with WT above bed); Zell&Sanford-vs-Fan channel-WTD
  correlation (expect positive, not identical; both retained); MERIT-SWORD
  width vs Zarrabi width consistency on large rivers.
- Roundtrip test: the produced netCDF opens through ddrs's `AttributesStore`
  and the new names resolve in `input_var_names`.

## 4. Phase B — objective floor fix (ddrs)

- B1 uses the recoverability experiment's assets: the synthetic teacher obs +
  init head give the exact floor measurement (teacher weights, windowed loss =
  pure transient; continuous residual = 0.0076 reference). Forward-only, CPU,
  no training. Output: floor(warmup, rho) curve + transient-decay-vs-uparea
  analysis.
- B2 decision rule: if warmup ≤ 60 @ rho ≤ 180 reaches the bar → option A
  (config-only). Else option B: cache a continuous run's daily reach states
  over the training window (~64,892 × 5,113 f32 ≈ 1.3 GB) and initialize each
  training window from the cached state at its start date. Staleness (cached
  states from one head vs the training head) is measured as the residual
  floor; periodic refresh is a follow-up if needed.
- B3 bar: fixed-objective floor ≤ 0.25 mean L1. Also rerun-noise check on CPU
  (should stay 0).
- Explicitly general: this fixes the objective for ALL ddrs training, not just
  leakance. Findings feed CLAUDE.md guidance on `warmup`.

## 5. Phase C — the gate experiment (ddrs)

- C1 code: `max(0, depth − d_gw)` clamp in the leakance flux (differentiable
  relu on the head term; gradcheck extended); static impervious hard-zero mask
  (from `corridor_impervious` > threshold, e.g. 0.7) applied like the
  losing-only clamp — multiplication into the flux, off the autograd-sensitive
  path. Both changes must pass `leakance_gradcheck`, `leakance_off_parity`,
  `zeta_accum`, and `compare_ddr_sandbox` ABSOLUTE MATCH.
- C2 training pair: hourly forcing, cold init, seed 42, identical recipe,
  fixed objective from B, enriched inputs from A in BOTH cells. K_D range
  stays `[1e-8, 1e-5]`.
- C3 gate (pre-registered here, before any C2 run):

| Leg | Test | Bar |
|---|---|---|
| 1 metrics | losing-subset median ΔNSE and ΔKGE (ON − OFF) | both ≥ +0.01 |
| 1 metrics | overall median NSE, KGE | degrade ≤ 0.002 |
| 1 metrics | median \|FHV\|, \|FLV\| (Yilmaz 2008 defs) | not worse on either gauge set |
| 2 equifinality | Δn(ON−OFF) per-reach distribution | IQR < 0.1 (daily anti-pattern was 0.59) |
| 2 equifinality | spearman ρ(Δn, zeta_net) on nonzero-zeta reaches | \|ρ\| < 0.2 |
| 3 external consistency | ρ(learned zeta magnitude, continuous bed-relative WTD) on nonzero-zeta reaches | > 0.3 (mask/threshold info not consumed by training targets — non-circular) |
| 3 external consistency | zeta ≈ 0 on lined-urban deep-WTD reaches (the LA-River falsification set) | median \|zeta\| below the losing-reach median by ≥ 5× |
| 3 external consistency | magnitudes within literature transmission-loss ranges (Shanafield & Cook 2014) | qualitative check, reported |

Losing-subset definition: reuse the 2×2's subset script for continuity; also
report stratified by `losing_fraction` terciles (descriptive).

Decision: **PROMOTE** (leakance default-on in hourly configs; docs updated)
iff all three legs pass. **KILL** (documented NO-GO; term stays experimental)
if leg 1 or leg 2 fails. **REVISE** if only leg 3 fails — the term helps and
doesn't cannibalize, but isn't physically aligned; reparameterize (Rushton
2007 direction: separate bed/aquifer conductance inputs) before any retry; no
promotion on a REVISE.

## 6. Concerns

1. **Gridded WTD products are model-interpolated, not pure measurement.**
   Wells aren't in channels; the grids do the well→valley interpolation with
   physics (Zell & Sanford calibrate on perennial-stream positions — the
   connection pattern itself). Mitigated by channel-pixel sampling,
   bed-relative conversion, dual-product cross-check, and using the data as
   INPUT (KAN can down-weight it) rather than as a training target.
2. **The +0.01 subset bar is ~5× the 2×2's effect.** Deliberate ("improves
   substantially"); KILL is a live outcome and the gate is pre-registered so
   it cannot be rationalized post hoc.
3. **Enriched inputs change both cells**, so no result is comparable to any
   prior run except descriptively. All gate decisions are within the new pair.
4. **Floor-fix option B staleness** (cached states vs moving head): measured,
   not assumed; refresh is a follow-up.
5. **extractrs may need new stat kinds** (categorical mode, threshold
   fraction). Preference order: express in xarray preprocessing; only extend
   the library if performance demands. Library changes reviewed in that repo's
   own conventions.
6. **Bankfull-depth estimate enters `channel_wtd_bed_rel`.** It comes from the
   same trapezoidal geometry the model uses (consistent), but it is itself
   uncertain; `losing_fraction` at two depth assumptions bounds the
   sensitivity.
7. **NLCD imperviousness is a proxy for lining**, not a lining map — high
   corridor imperviousness can occur without a concrete bed. The hard-zero
   threshold is set high (≥ 0.7) to keep the structural zero conservative;
   misclassified reaches simply fall back to learned (probably small) zeta.
8. **Scope**: three phases across two repos is a program, not one plan. Each
   phase gets its own implementation plan; A and B run in parallel; C blocks
   on both.

## 7. Assumptions

- Zell & Sanford 2020 (ScienceBase), Fan 2013 (THREDDS), StreamCat (EPA
  API/FTP), Zarrabi 2025 (Zenodo 13883263), SWORD v2 + MERIT-SWORD (Zenodo
  10013982 / 13152826), and a gridded BFI are downloadable (all verified
  2026-07-04 by the channel-characteristics literature pass); Phase A
  re-verifies at execution time.
- The NHDPlus→MERIT crosswalk is buildable by buffered-geometry
  length-weighted matching at acceptable quality on the eval network
  (headwater divergence between the networks is the known weak spot —
  crosswalk quality flags are a Phase A deliverable).
- CPU-only remains workable (~85 min per training run measured); GPU used if
  free.
- Gains remain Q′-baseflow's job; regional-groundwater channel gains stay
  unmodeled (status quo; a signed-zeta follow-up would need its own
  anti-double-counting design).
- Static (climatological) `d_gw`; no new dynamic states.

## 8. Out of scope

- Auxiliary LOSS terms on zeta/d_gw (the enriched-input KAN route was chosen
  instead; a soft-supervision variant is a follow-up only if C returns REVISE).
- The gaining branch / regional-GW channel gains.
- Global (non-CONUS) extraction (corridor pipeline is CONUS-first; the
  mechanism generalizes).
- Any DDR (Python) backport.
- Promotion of leakance under daily forcing (the 2×2 already showed daily
  hurts; the pair is hourly).

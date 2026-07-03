# Literature review — why is the learned leakance (zeta) so small?

Date: 2026-07-01. Companion to
`docs/superpowers/specs/2026-07-01-leakance-low-zeta-diagnosis-design.md`.

Context: the leakance term `zeta = leakance_factor · area_z · K_D · (depth − d_gw)`
learned per reach by the KAN head came out tiny in the 2×2 experiment
(`docs/2026-07-01-leakance-hourly-findings.md`): median |zeta| 6.4e-4 m³/s
(hourly-ON), |zeta| > 0.01 m³/s on only 10.4% of eval reaches, `K_D` pinned at
its `1e-6 s⁻¹` ceiling on 100% of reaches. Two questions to the literature:
(A) what magnitudes are physically expected, and (B) why might a term trained
only on gauged discharge learn near-zero exchange?

Dimensional note: `area_z (m²) · K_D (1/s) · head (m) = m³/s`, so `K_D` is
exactly a MODFLOW-style **streambed leakance** `K_v/b′` (vertical hydraulic
conductivity over streambed thickness), and `K_D · area_z` is the RIV/SFR
**conductance**. That makes the literature directly comparable.

---

## A. Physical magnitudes

### A1. Streambed leakance (K_v/b′)

Riverbed vertical hydraulic conductivity spans ~1e-9 to >1e-3 m/s across 41
pooled investigations — silt/clay ~1e-7 m/s, clean sand ~1e-3 m/s
(Calver 2001). With typical streambed (clogging-layer) thickness b′ = 0.1–1 m:

| regime | K_v (m/s) | b′ (m) | leakance K_v/b′ (1/s) |
|---|---|---|---|
| clogged silt/clay bed | 1e-7 | 1 | **1e-7** |
| silty sand | 1e-5 | 1 | **1e-5** |
| clean sand | 1e-4 | 1 | **1e-4** |
| clean sand, thin bed | 1e-4–1e-3 | 0.1–0.5 | **~1e-3–1e-2** |

**Implication: the ddrs range `K_D ∈ [1e-8, 1e-6]` covers only clogged
silt/clay beds. Sand/gravel streambeds — precisely the losing/ephemeral case
the term targets — lie 2–4 orders of magnitude above the ceiling.** The
observed 100% ceiling-pinning is the expected signature of a physically
too-tight box. Literature-supported widening: upper bound 1e-4, arguably 1e-3.
The 1e-8 floor already covers well-clogged beds; no need to lower it.

- Calver, A. (2001). Riverbed permeabilities: information from pooled data.
  *Ground Water* 39(4), 546–553.
- Abimbola et al. (2020). Streambed hydraulic conductivity across stream
  orders (Frenchman Creek). PMC7048843.
- Song et al. (2017). Factors influencing streambed hydraulic conductivity.
  *Environ Sci Pollut Res* 24.
- MODFLOW-6 GWF-RIV docs (leakance/conductance definitions);
  SFR2 input instructions (USGS).
- Rosenberry & Pitlick (2009): K_v differs by seepage direction (gaining > losing).

### A2. Transmission-loss rates of losing/ephemeral streams

- Streambed infiltration velocities 0.1–1 m/day (up to several) for permeable
  sandy ephemeral beds (Shanafield & Cook 2014 review); 0.043–0.127 m/day in a
  reach-scale field experiment (Batlle-Aguilar & Cook 2012).
- 0.1 m/day over a 10 m × 1 km reach ≈ **0.012 m³/s per km** — the same order
  as measured alluvial Rio Grande channel losses (0–0.37 cfs/mile).
- Arid-zone event losses are commonly tens of percent of flow, up to complete
  loss for small floods (Lane 1983; SCS NEH ch. 19).

So where losing streams are real, per-reach losses of order 1e-2 m³/s are
typical — ~20× our learned median. The 0.01 m³/s GO-bar was well chosen.

- Shanafield, M. & Cook, P.G. (2014). Transmission losses, infiltration and
  groundwater recharge through ephemeral and intermittent streambeds.
  *J. Hydrology* 511, 518–529.
- Lane, L.J. (1983). Transmission losses. SCS National Engineering Handbook §4 ch. 19.
- Batlle-Aguilar, J. & Cook, P.G. (2012). Transient infiltration from
  ephemeral streams. *WRR* 48.

### A3. Driving head and disconnection

The linear `(depth − d_gw)` law is a **connected-regime** Darcy law. Brunner,
Cook & Simmons (2009, *WRR*; 2011, *Ground Water*) show:

- Streams disconnect from the water table at a critical depth of order
  **1–5 m** (proportional to stream depth and K_bed/b′, inverse in aquifer K).
- Once disconnected, infiltration **saturates** at ≈ (K_v/b′)·(b′ + h_stream)
  and becomes independent of further water-table decline.
- Water tables under losing reaches in the arid West are frequently **tens of
  meters** deep — fully disconnected.

**Implication: `d_gw ∈ [−2, 2] m` is defensible for connected reaches but the
linear head law is the wrong model form for the strongest-losing (disconnected)
population.** Deepening `d_gw` alone would over-predict (flux grows linearly
where physics saturates); the faithful fix is a flux cap at the disconnected
maximum, not a wider head window.

### A4. Prevalence of losing streams

- 64% of 4.2M CONUS wells sit below the adjacent stream surface — widespread
  *potential* for streamflow loss, concentrated in drier, flatter, pumped
  regions (Jasechko et al. 2021, *Nature* 591, 391–395).
- >50% of global river length is non-perennial (Messager et al. 2021,
  *Nature* 594) — the reach class where transmission loss dominates.

---

## B. Identifiability — why discharge-only training learns ~zero exchange

Ranked by literature support and fit to our setup (2365 CONUS gauges,
NSE/KGE-family loss, per-reach params from static attributes):

1. **Gauge-placement bias (strongest).** Gauge networks sit disproportionately
   on large perennial rivers (Krabbenhoft et al. 2022, *Nat. Sustain.* 5,
   586–592); losing/ephemeral reaches are systematically ungauged (Messager
   2021; Zimmer et al., zero-flow gage readings). The training loss barely
   samples reaches where zeta should be non-zero — a data-distribution ceiling
   that range-widening cannot fix.
2. **Equifinality / no restoring gradient.** Bindas et al. (2024, *WRR* 60 —
   the dMC paper this repo ports) found channel-geometry parameters
   unidentifiable and Manning's n only moderately identifiable from downstream
   hydrographs. Leakance shapes the same observable (peak attenuation/volume)
   as n and storage, so the base model absorbs the signal and `∂Loss/∂zeta ≈ 0`
   (Beven 2006 equifinality; Kirchner 2006 "right answers for the right
   reasons"; Nijzink et al. 2018 multi-source constraints as the remedy).
3. **Sub-detection-limit signal.** Differential gauging cannot resolve
   exchange below the 5–10% streamflow-uncertainty band; in one paired-gauge
   study the flow difference exceeded error on only 3/19 upstream occasions
   (McCallum et al. 2012, *J. Hydrology* 416–417; Kiang et al. 2018 *WRR* on
   rating uncertainty). Losses that small are invisible in the hydrograph.
4. **Genuine physics on the sampled population.** Gauged perennial CONUS
   reaches are frequently neutral-to-gaining; near-zero zeta may be *correct*
   there — and the term isn't fully dead (interior leakance_factor ≈ 0.33,
   53.7% net-losing).
5. **Parameter-range clipping (real but secondary).** The `1e-6` ceiling clips
   the *maximum* zeta, not the pull toward zero. Prediction: widening `K_D`
   raises the losing-tail magnitude but moves the median little as long as
   1–3 dominate. It is still the correct *first* experiment because it is the
   only limiter we fully control and its removal cleanly discriminates
   "clipped" from "unidentifiable".

Discriminating tests suggested by the literature: stratify learned zeta by
reach class (arid vs humid, gauged vs ungauged); compare leakance-param
spatial variance against routing-param variance from the same head; check
whether routing params shift between paired leakance-ON/OFF runs
(bias-compensation signature); widen `K_D` as the controlled H1 test.

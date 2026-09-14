# Scouting: an end-to-end (coupled) arm for the Beven-challenges paper (2026-09-13)

Two read-only surveys, both prompted by findings §38: the routing absorbs each runoff product's timing lead
into roughness, and the causal control is a model that can fix the timing at its source. Two candidate
sources were named: an hourly dHBV2.1 runoff product on AORC, and a differentiable precipitation mapper
(the corduroy idea). Neither exists yet; this records what does, what it would take, and what to do first.

## 1. dHBV2.1 hourly on AORC over CONUS unit catchments (`~/projects/water_loss`)

**What exists.** The daily AORC2f products the arms used were made by the water_loss pipeline
(`dPLHBVrelease-master/hydroDL-dev/example/water_loss_aorc/`): `prepare_conus_aorc_inputs.py` →
`CONUS_water_loss_v6_1_v18_1_AORC_2forcing_gradaccum.py` (dHBV2.0 multiscale, LSTM(64)+ANN(4096)+HBV+gamma UH,
2,717 gauges, checkpoint `CONUS2717_AORC2F_v3_gradaccum` ep100, NSE 0.749) → `forward_conus_divides.py --mode
export --uh-routed` (197,088 divides, 1980–2020, chunks (200, n_time), hence trap T15). "aorc2f" = two AORC
variables (P, T), PET by Hargreaves. Lumped counterpart `CONUS2717_AORC2F_LUMPED` ep63, NSE 0.587. Runoff and
routing are coupled only by files (the Q′ store contract); no joint training exists.

**What does not.** `hbv_2_1_hourly.py` (hydrodl2 PR #45) is on no machine here: the local fork
`~/projects/hydrodl2` (`6a7fc3c`, PR #35) has `hbv_2_hourly.py` (infiltration excess already, `dt = 1/24`,
needs hourly prcp/tmean/pet) and `hbv_2_mts.py`; installed hydrodl2 builds are 1.3.4 and 0.1.dev89. No
hourly or MTS CONUS checkpoint exists; the MTS BMI wrapper in `~/projects/dhbv2` was demoed on three CAMELS
catchments. The AORC catchment store carries P and T for 290,878 catchments (ID reconciliation needed against
the 197,088 fabric) and **no PET or radiation**, so an hourly PET needs a radiation source or a
temperature-only approximation.

**Cost.** (1) get PR #45 or port the 2.1 infiltration physics from `hbv_2_hourly.py`: 0.5–2 days once
available; (2) hourly forcing with PET: 1–3 days; (3) **train a CONUS hourly model from scratch**: the daily
grad-accum run took ~90 min/epoch × 100 epochs ≈ 6.4 GPU-days on the RTX 4080; hourly is 24× the timesteps
and may not fit at all: **multi-week, uncertain**; (4) the CONUS hourly export: GPU-days; (5) **disk**: the
existing hourly LSTM store is 257 GB on disk (effectively uncompressed) and `/mnt/ssd1` has ~290 GB free, so a
second hourly product does not fit without deleting one. Verdict: not a this-paper arm. It is the follow-up's
runoff model, and steps (1)–(2) are worth doing now so the follow-up can start.

## 2. Corduroy as a differentiable precipitation mapper (`~/projects/corduroy`, `bb36dd9`)

**What exists.** An anonymous ARCO-ERA5 zarr v3 fetch (`src/data.rs:69-134`, `total_precipitation`, hourly,
0.25°), a well-tested conservative grid-to-grid regridder (`src/regrid.rs:172-370`), PNG rendering, optional
pyo3 bindings, and, separately, a complete 2D diffusive-wave overland + channel solver (`src/hydro/`, CASC2D
style, unit and end-to-end tests). **No adaptive mesh in code** (README wording only), no grid-to-catchment
mapping, no autograd (no burn, no tch). `.claude/memories/diff_router_architecture.md` sketches a
differentiable cell→reach aggregation and a Muskingum–Cunge port, self-described as aspirational.

**The right zonal tool is extractrs** (`~/projects/extractrs`): a Rust port of exactextract's exact
fractional coverage, Python-exposed; this is how AORC was put onto the unit catchments and how ERA5 cells →
catchments weights should be built. Corduroy's regridder is grid-to-grid and does not apply.

**Minimal differentiable mapper (BURN, inside ddrs).** A fixed sparse exact-coverage matrix `W`
(cells → catchments, built once with extractrs on the fabric ddrs already reads, stored like `CsrPattern`)
applied per hour, composed with a learnable per-cell gain `g` and bias `b`:
`P_catch[t] = W · (g ⊙ P_grid[t]) + b`, all as tensor ops so a gauge loss reaches the forcing. It must
reproduce `AorcPrecipStore::read_window_hourly`'s `(n_hourly, N)` contract (`src/data/store/zarr_aorc.rs:168`).
Definition of done for the first version: perturb one `g[cell]` and assert a nonzero gradient.

**The runoff link is the decision.** (a) an HBV-type runoff model reimplemented in BURN inside ddrs: days,
no language boundary, the whole chain differentiable in one graph, risk of drifting from hydrodl2's physics
(mitigated by a parity fixture against `hbv_2_hourly.py` on a few catchments); (b) a PyTorch bridge around
dHBV2 (tch-rs / pyo3 / a Python loop calling ddrs): reuses the model as-is, but autograd across the FFI
boundary is hand-maintained and fragile; the scout recommends against; (c) a BURN surrogate distilled from
dHBV2 outputs: fastest to wire, adds its own error between gauge and physics. Recommendation: (a), starting
from the HBV core of `hbv_2_hourly.py`, which is small.

**Blockers.** CONUS hourly grids over a 14-year training window do not fit in memory: cache to a local zarr
(as AORC is) and read windows; GCS egress for repeated epochs; the Rust/PyTorch boundary only if (b).

## 3. What to do first, in order

1. **The learned timing head in ddrs (this paper's coupled control).** A two-parameter unit hydrograph or
   lag per unit catchment predicted from attributes, at the same seam as the disaggregation head, trained
   jointly with n_0 + gamma on the lumped and distributed dHBV2 products. Registered prediction: on the
   lumped product the timing head absorbs the 1–2 day lead and n_0 returns from the ceiling to the
   distributed-family field. Days of work, no new data, keeps every other variable identical to the five
   arms. This is the experiment that turns §38 from a description into a causal statement.
2. **Phase 0 of the mapper**: build `W` with extractrs for ERA5 (and AORC) cells → MERIT unit catchments,
   verify it reproduces the existing AORC catchment store on a sample; one to two days.
3. **Phase 1**: `src/nn/precip_map.rs` in ddrs with the gradient unit test; two to three days.
4. **dHBV2.1 prerequisites** in parallel and cheap: pull PR #45 into the fork, add hourly PET to the AORC
   catchment store (radiation source decision), reconcile the 290,878 vs 197,088 catchment IDs.
5. **Then** the runoff link (option a) and the end-to-end arm, as the follow-up paper.

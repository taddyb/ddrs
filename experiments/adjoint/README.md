# Adjoint influence map — `ddrs experiment adjoint`

**Question.** How does an error in lateral inflow at reach *i* on day *d*
reach a gauge, and does that answer depend on which inflow source the routing
model was trained on? Per gauge and per trained arm, the study differentiates
routed gauge discharge with respect to the hourly lateral inflow at every
upstream reach (the adjoint of the Muskingum–Cunge network operator, read
from the existing routing backward) and reduces it to three quantities:

| Functional | What it measures |
|---|---|
| **kernel** | `dQ_g(t0)/dq'(reach, hour)` at high- and low-flow anchor days: the bias-propagation kernel by reach and lag. |
| **volume** | `d[Σ_t Q_g]/dq'`: fraction of a reach's inflow volume the gauge sees. ≈1 upstream by mass conservation; the sanity gate. |
| **residual** | `d[mean(Q̄_g − obs)]/dq'` over four seasonal windows: which reaches the gauge's bias is attributed to. |

Spec: `docs/superpowers/specs/2026-09-03-ddrs-experiment-adjoint-design.md`.
Leakance is out of scope.

## Run

From the repo root (the bundle path defaults to `experiments/adjoint`):

```bash
cargo build --release --bin ddrs
target/release/ddrs --workspace .ddrs experiment adjoint --backend cpu
# smoke: one arm, one gauge
target/release/ddrs --workspace .ddrs experiment adjoint --arms daily-lstm --max-gauges 1
```

Arms are **run ids** under `.ddrs/runs/`; each arm's `config.yaml` snapshot
and latest `checkpoints/epoch_E_mb_M/` are what gets analyzed. Flat `.mpk`
checkpoints are refused (stale binary). The first anchor of the first
gauge/arm runs the finite-difference gate (+5 % inflow at three reaches,
rel. error < 5 %); failure aborts with exit 1 and the numbers land in the
manifest.

## Output

```
.ddrs/experiments/adjoint/<UTC ts>/
├── manifest.json          spec copy, resolved arms, git, validation result, notes
├── experiment.yaml
├── run.log
├── gauges.csv             staid, role (upstream|downstream), pair, upstream_staids
├── <arm>/gauges/<staid>.nc   dims reach, anchor, lag_day, lag_hour, window, hour
├── <arm>/summary.csv      one row per (gauge, reach): volume_sens, residual_attr,
│                          kernel_mass_{high,low}, kernel_mean_lag_{high,low}_days
└── figures/               written by plots.py
```

## Figures

```bash
~/projects/ddr/.venv/bin/python experiments/adjoint/plots.py .ddrs/experiments/adjoint/<ts>
```

1. `kernel_by_lag.png` — reach-summed hourly kernel vs lag, high vs low anchors, one line per arm.
2. `kernel_vs_distance.png` — per-reach kernel mass and mean lag vs along-channel distance.
3. `influence_map_<staid>.png` — MERIT catchment polygons colored by residual attribution and volume sensitivity, per arm.
4. `inherited_vs_local_<pair>.png` — downstream bias split into the part routed from the upstream gauge and the intervening-area remainder, per window and arm.
5. `volume_sens.png` — volume-sensitivity histogram per arm against the 1.0 line, and vs distance.

## Scope of the proof of concept (2026-09-03)

One nested pair in the Juniata: 01563500 Mapleton Depot (5,262 km²) inside
01567000 Newport (8,657 km²). The full study replaces `gauges.pairs` with the
GAGES-II reference-class nested selection described in the spec.

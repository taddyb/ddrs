# Reference: H5 parameter-swap (transfer-penalty) plots

Visualizes the selective-equifinality campaign's H5 hypothesis test
(`docs/superpowers/specs/2026-07-08-landscape-hypotheses-h5-h6-draft.md`):
does exchanging Manning's `n` between two independently-forced checkpoints
degrade the training loss more than exchanging channel geometry (`q_spatial`,
`p_spatial`)? A self-contained pandas/matplotlib notebook — no `ddr.validation`
dependency, unlike the hydrograph/parameter_map/metrics families, since the
data is a plain CSV, not a zarr or NetCDF store.

## Inputs

### eval-loss CSV (from `probe_zeta_gradient --mode eval-loss`)

One CSV per (forcing arm, donor arm) pair, tidy long format:

```
composition,window,mean_loss
own,0,7.130814
n-swap,0,7.3130217
geo-swap,0,7.1397943
full-swap,0,7.346762
own,1,6.5884113
...
```

- `composition` ∈ `{own, n-swap, geo-swap, full-swap}` (see the spec for what
  each swaps).
- `window` — integer index into the deterministic rho-window plan (same
  `--seed`/`--windows` ⇒ same windows across compositions AND across the two
  arms in a pair, since both share the training time axis and gauge set).
- `mean_loss` — training L1 loss (mean `|pred - obs|` in m³/s) for that
  window under that composition, evaluated under the RUN's own Q′ forcing
  (the `--config`/`--checkpoint` passed to the binary — the donor supplies
  parameters only, never forcing).

Naming convention used by this campaign (not enforced by the binary, but
keep it so notebooks don't have to guess): `<forcing_arm>_under_<forcing_arm>_donor_<donor_arm>.csv`,
e.g. `r1_under_r1_donor_r3.csv` = R1's checkpoint/config, R3's dump as donor.
A full H5 primary-pair run needs BOTH directions (`r1_..._donor_r3.csv` AND
`r3_..._donor_r1.csv`) plus, if reporting the low-disagreement control pair,
`r1_..._donor_r2.csv` and `r2_..._donor_r1.csv`.

### Optional per-gauge CSV (from `--per-gauge-output`)

`--mode eval-loss` optionally writes a SECOND CSV alongside the required
`--loss-output` one, when `--per-gauge-output <path>` is passed. It has a
different, wider schema — one row per surviving gauge per window per
composition, not one row per window:

```
composition,window,staid,gauge_loss
own,0,0000000A,3.5198908
own,0,07336200,23.219501
n-swap,0,0000000A,3.7...
...
```

- `composition`, `window` — same meaning as the aggregate CSV.
- `staid` — the gauge's `Staid` (zero-padded to 8 characters, e.g.
  `USGS`-prefixed IDs keep their prefix; plain numeric IDs get zero-padded —
  `Staid::new`, `src/data/ids.rs:29`). This is the SAME identifier used by
  `gage_ids` in the predictions zarr and by `STAID` in the gauges CSV
  (`metrics.md`'s drainage-area join), so it joins directly against those.
- `gauge_loss` — that gauge's own mean `|pred - obs|` over the window's
  post-warmup days (column-wise mean, not averaged across gauges like
  `mean_loss` in the aggregate CSV — `per_gauge_l1`,
  `src/bin/probe_zeta_gradient.rs`).

Row count is `n_windows * n_compositions * n_kept_gauges_in_that_window` —
kept-gauge count VARIES per window (gauges with any NaN in that window's
post-warmup obs are dropped by `filter_nan_gauges` before the loss is
computed), so don't assume a fixed row count per window when parsing.

**Sketch: DA-stratified median plot (not yet built — build when needed).**
Join this CSV's `staid` to a gauges CSV's `STAID` → `DRAIN_SQKM`, exactly the
pattern `metrics.md`'s drainage-area boxplot already uses:

```python
gauges_df = pd.read_csv(GAUGES_CSV)               # data_sources.gages path
gauges_df["STAID"] = gauges_df["STAID"].astype(str).str.zfill(8)
per_gauge = pd.read_csv("r1_under_r1_donor_r3_per_gauge.csv")
per_gauge["staid"] = per_gauge["staid"].astype(str).str.zfill(8)
merged = per_gauge.merge(
    gauges_df[["STAID", "DRAIN_SQKM"]], left_on="staid", right_on="STAID", how="inner"
)
DRAINAGE_BINS = np.array([0, 1000, 5000, 10000, 30000, 50000])
merged["da_bin"] = pd.cut(merged["DRAIN_SQKM"], DRAINAGE_BINS)
# Per-bin, per-composition median transfer penalty (e.g. n-swap gauge_loss -
# own gauge_loss, joined on staid+window first) — the DA-stratified statistic
# the spec's H5 Design section calls for.
medians = merged.groupby(["da_bin", "composition"])["gauge_loss"].median().unstack()
```

Watch the same pitfalls `metrics.md` documents for its own DA join: only
gauges present in BOTH files should be kept (`how="inner"`), and any
per-gauge transfer-penalty computation (`n-swap` minus `own`) must join on
`(staid, window)` first — `gauge_loss` values across compositions are NOT
guaranteed to appear in the same row order, only the same `(staid, window)`
KEY.

### Locating the CSVs

No fixed ddrs convention yet (H5 is new as of 2026-07-08) — ask the user or
look under `output/equif/h5/*.csv`. Load every CSV the user points at; each
becomes one "run" in the plots (label = filename stem, or ask the user for a
friendlier label like `"R1 under R1 (donor R3)"`).

## Notebook template

```python
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# --- USER INPUTS ---------------------------------------------------------
RUNS = {
    "R1 under R1 (donor R3)": Path("/home/tbindas/projects/ddrs/output/equif/h5/r1_under_r1_donor_r3.csv"),
    "R3 under R3 (donor R1)": Path("/home/tbindas/projects/ddrs/output/equif/h5/r3_under_r3_donor_r1.csv"),
    "R1 under R1 (donor R2, control)": Path("/home/tbindas/projects/ddrs/output/equif/h5/r1_under_r1_donor_r2.csv"),
    "R2 under R2 (donor R1, control)": Path("/home/tbindas/projects/ddrs/output/equif/h5/r2_under_r2_donor_r1.csv"),
}
PLOT_DIR = Path("/home/tbindas/projects/ddrs/output/equif/h5/plots")
# -------------------------------------------------------------------------

PLOT_DIR.mkdir(parents=True, exist_ok=True)
COMPOSITIONS = ["own", "n-swap", "geo-swap", "full-swap"]

dfs = {label: pd.read_csv(path) for label, path in RUNS.items() if path.exists()}
missing = [label for label, path in RUNS.items() if not path.exists()]
if missing:
    print(f"skipping (file not found): {missing}")

# --- Per-composition summary + transfer penalties + f_n -------------------
summary_rows = []
for label, df in dfs.items():
    means = df.groupby("composition")["mean_loss"].mean().reindex(COMPOSITIONS)
    stds = df.groupby("composition")["mean_loss"].std().reindex(COMPOSITIONS)
    p_n = means["n-swap"] - means["own"]
    p_geo = means["geo-swap"] - means["own"]
    # f_n undefined (0/0) if both penalties are ~0 — report NaN rather than
    # a misleading value; a near-zero denominator means neither swap moved
    # the loss, which is itself the finding (sloppy-in-both, or a bug).
    denom = p_n + p_geo
    f_n = p_n / denom if abs(denom) > 1e-9 else float("nan")
    summary_rows.append(
        {
            "run": label,
            "own_mean": means["own"],
            "n_swap_mean": means["n-swap"],
            "geo_swap_mean": means["geo-swap"],
            "full_swap_mean": means["full-swap"],
            "P_n": p_n,
            "P_geo": p_geo,
            "f_n": f_n,
            "n_windows": df["window"].nunique(),
        }
    )
summary = pd.DataFrame(summary_rows).set_index("run")
print(summary.round(4))
summary.to_csv(PLOT_DIR / "h5_summary.csv")

# --- Bar chart: mean loss +/- std per composition, one panel per run ------
n_runs = len(dfs)
fig, axes = plt.subplots(1, n_runs, figsize=(5 * n_runs, 5), sharey=False)
if n_runs == 1:
    axes = [axes]
colors = {"own": "#4C72B0", "n-swap": "#C44E52", "geo-swap": "#55A868", "full-swap": "#8172B2"}
for ax, (label, df) in zip(axes, dfs.items()):
    means = df.groupby("composition")["mean_loss"].mean().reindex(COMPOSITIONS)
    stds = df.groupby("composition")["mean_loss"].std().reindex(COMPOSITIONS)
    ax.bar(COMPOSITIONS, means.values, yerr=stds.values, capsize=4,
           color=[colors[c] for c in COMPOSITIONS])
    ax.set_title(label, fontsize=11)
    ax.set_ylabel("mean L1 loss (m³/s)")
    ax.tick_params(axis="x", rotation=20)
fig.suptitle("H5: transfer-penalty per composition (bar = window mean, error = window std)", fontsize=13)
fig.tight_layout()
fig.savefig(PLOT_DIR / "h5_composition_bars.png", dpi=300, bbox_inches="tight", facecolor="white")

# --- Per-window paired lines: does every window agree on direction? -------
fig, axes = plt.subplots(1, n_runs, figsize=(5 * n_runs, 5), sharey=False)
if n_runs == 1:
    axes = [axes]
for ax, (label, df) in zip(axes, dfs.items()):
    pivot = df.pivot(index="window", columns="composition", values="mean_loss")[COMPOSITIONS]
    for _, row in pivot.iterrows():
        ax.plot(COMPOSITIONS, row.values, color="gray", alpha=0.3, linewidth=0.8)
    ax.plot(COMPOSITIONS, pivot.mean().values, color="black", linewidth=2.5, marker="o", label="mean")
    ax.set_title(label, fontsize=11)
    ax.set_ylabel("mean L1 loss (m³/s)")
    ax.tick_params(axis="x", rotation=20)
    ax.legend()
fig.suptitle("H5: per-window loss across compositions (gray = one window, black = mean)", fontsize=13)
fig.tight_layout()
fig.savefig(PLOT_DIR / "h5_per_window_lines.png", dpi=300, bbox_inches="tight", facecolor="white")

# --- f_n summary bar (the primary H5 statistic) ---------------------------
fig, ax = plt.subplots(figsize=(1.5 * len(summary) + 2, 5))
bars = ax.bar(summary.index, summary["f_n"], color="#4C72B0")
ax.axhline(2 / 3, color="green", linestyle="--", label="SUPPORTED bar (f_n >= 2/3)")
ax.axhline(1 / 2, color="red", linestyle="--", label="REFUTED bar (f_n <= 1/2)")
ax.set_ylabel(r"$f_n = P_n / (P_n + P_{geo})$")
ax.set_title("H5 attribution fraction per run")
ax.tick_params(axis="x", rotation=20)
ax.legend()
fig.tight_layout()
fig.savefig(PLOT_DIR / "h5_fn_summary.png", dpi=300, bbox_inches="tight", facecolor="white")

print(f"saved H5 plots to {PLOT_DIR}")
```

## Notes

- **No `ddr.validation` dependency.** Unlike the other three families, H5's
  data is a plain CSV the Rust binary wrote — plotting is pure
  pandas/matplotlib. This also means the "host where `uv` can't build
  `ddrs-py`" workaround in `SKILL.md` doesn't apply here: `pandas` +
  `matplotlib` are always available even when `ddr`/`torch` aren't.
- **`f_n` is undefined, not zero, when both penalties vanish.** `P_n = P_geo
  = 0` means neither swap moved the loss (could mean both parameter classes
  are sloppy, or that the override didn't actually change anything — check
  the per-window line plot before trusting a NaN `f_n`).
- **The window-std error bars are NOT the pre-registered noise floor.**
  The spec's noise floor is a split-half comparison across two DISJOINT
  window sets (e.g. seed 42 vs seed 123, non-overlapping windows) — the
  within-run window std plotted here is a weaker, descriptive spread, useful
  for a first look but not a substitute for the registered split-half gate.
  If split-half CSVs exist (same composition, disjoint window sets), extend
  the bar chart to show both a run's std AND its split-half displacement
  side by side.
- **Per-window line plot is the sanity check for H5's central validity
  claim** (docs/superpowers/specs/2026-07-08-landscape-hypotheses-h5-h6-draft.md
  §"H5 — Forcing-bound roughness"): if some windows show n-swap increasing
  loss and others show it decreasing loss by comparable magnitude, the mean
  P_n could be near zero while masking real per-window variance — the line
  plot makes this visible where the bar chart alone would not.
- **Composition order is fixed** (`own, n-swap, geo-swap, full-swap`) so bar
  colors and x-axis order are consistent across every run/panel — don't let
  `groupby`/`pivot`'s default alphabetical sort silently reorder them
  (`full-swap` would sort before `geo-swap` before `n-swap` before `own`
  alphabetically, which is misleading reading order).

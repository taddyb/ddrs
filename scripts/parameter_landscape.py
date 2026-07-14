"""Parameter landscapes: learned routing parameters as surfaces over attribute space.

For each learned parameter (n, q_spatial, p_spatial), renders the H6-style
landscape treatment with the PARAMETER as the height field instead of the loss:
median learned value per (log10 drainage area x log10 mean slope) bin, one
panel per arm (R1/R2/R3, seed-42 checkpoints), plus a fourth panel showing the
cross-arm range per reach (max - min across arms, median per bin).

p_spatial is learned in log space over [1, 200], so it is plotted and ranged
in log10 units; n and q_spatial are linear.

Outputs: output/equif/figs/{n,q_spatial,p_spatial}_parameter_landscape.png

Run: cd ~/projects/ddrs/ddrs-py && uv run python ../scripts/parameter_landscape.py
"""

import numpy as np
import xarray as xr
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import binned_statistic_2d, spearmanr

EQUIF = "/home/tbindas/projects/ddrs/output/equif"
ATTRS = "/home/tbindas/projects/ddr/data/merit_global_attributes_v2.nc"
ARMS = ["R1", "R2", "R3"]
TITLES = {
    "R1": "R1 — daily LSTM, flat repeat-24",
    "R2": "R2 — daily LSTM + disagg head",
    "R3": "R3 — hourly MTS-LSTM native",
}
# param -> (nc variable, plotted transform, axis label)
PARAMS = {
    "n": ("n", None, "median learned n"),
    "q_spatial": ("q_spatial", None, "median learned q_spatial"),
    "p_spatial": ("p_spatial", np.log10, "median learned log10 p_spatial"),
}
BINS = 36
MIN_COUNT = 15

sa = np.load(f"{EQUIF}/stage_a.npz", allow_pickle=True)
eval_comids = sa["eval_comids"]

attrs = xr.open_dataset(ATTRS)
da_all = attrs["log10_uparea"].to_series().reindex(eval_comids).to_numpy()
sl_all = np.log10(np.clip(attrs["meanslope"].to_series().reindex(eval_comids).to_numpy(), 1e-3, None))

dumps = {arm: xr.open_dataset(f"{EQUIF}/{arm}_kan_parameters.nc") for arm in ARMS}

x_lo, x_hi = np.percentile(da_all, [0.5, 99.5])
y_lo, y_hi = np.percentile(sl_all, [0.5, 99.5])
x_edges = np.linspace(x_lo, x_hi, BINS + 1)
y_edges = np.linspace(y_lo, y_hi, BINS + 1)
x_mid = 0.5 * (x_edges[:-1] + x_edges[1:])
y_mid = 0.5 * (y_edges[:-1] + y_edges[1:])
extent = [x_edges[0], x_edges[-1], y_edges[0], y_edges[-1]]


def binned(x, y, z):
    med, _, _, _ = binned_statistic_2d(x, y, z, statistic="median", bins=[x_edges, y_edges])
    cnt, _, _, _ = binned_statistic_2d(x, y, z, statistic="count", bins=[x_edges, y_edges])
    med[cnt < MIN_COUNT] = np.nan
    return med.T  # rows = y


for param, (var, transform, cbar_label) in PARAMS.items():
    vals = np.vstack([dumps[arm][var].to_series().reindex(eval_comids).to_numpy() for arm in ARMS])
    if transform is not None:
        vals = transform(vals)
    valid = np.isfinite(da_all) & np.isfinite(sl_all) & np.all(np.isfinite(vals), axis=0)
    da_v, sl_v, vals_v = da_all[valid], sl_all[valid], vals[:, valid]
    rng = vals_v.max(axis=0) - vals_v.min(axis=0)

    fields = {arm: binned(da_v, sl_v, vals_v[i]) for i, arm in enumerate(ARMS)}
    rng_field = binned(da_v, sl_v, rng)
    vmin = np.nanpercentile(list(fields.values()), 2)
    vmax = np.nanpercentile(list(fields.values()), 98)

    fig, axes = plt.subplots(1, 4, figsize=(21, 5.2), sharey=True)
    for ax, arm in zip(axes[:3], ARMS):
        im = ax.imshow(fields[arm], origin="lower", extent=extent, aspect="auto",
                       cmap="viridis", vmin=vmin, vmax=vmax)
        cs = ax.contour(x_mid, y_mid, fields[arm], levels=6,
                        colors="white", linewidths=0.6, alpha=0.7)
        ax.clabel(cs, fontsize=7, fmt="%.3f")
        ax.set_title(f"{TITLES[arm]}\n{cbar_label}", fontsize=11)
        ax.set_xlabel("log10 drainage area [km²]")
    axes[0].set_ylabel("log10 mean slope [m/m]")
    fig.colorbar(im, ax=axes[:3], shrink=0.85, pad=0.01, label=cbar_label)

    im2 = axes[3].imshow(rng_field, origin="lower", extent=extent, aspect="auto", cmap="magma")
    cs2 = axes[3].contour(x_mid, y_mid, rng_field, levels=5,
                          colors="cyan", linewidths=0.6, alpha=0.8)
    axes[3].clabel(cs2, fontsize=7, fmt="%.3f")
    unit = "log10 units" if transform is not None else "range"
    axes[3].set_title(f"cross-arm RANGE ({unit})\nmax(R1,R2,R3) − min(R1,R2,R3), median per bin",
                      fontsize=11)
    axes[3].set_xlabel("log10 drainage area [km²]")
    fig.colorbar(im2, ax=axes[3], shrink=0.85, pad=0.02, label=f"{param} range across arms")

    fig.suptitle(f"Parameter landscape: learned {param} over attribute space "
                 f"({valid.sum():,} eval reaches, seed-42 checkpoints)", fontsize=13, y=1.02)
    out = f"{EQUIF}/figs/{param}_parameter_landscape.png"
    fig.savefig(out, dpi=110, bbox_inches="tight")
    plt.close(fig)

    print(f"\n=== {param} === saved {out}")
    for i, arm in enumerate(ARMS):
        print(f"  {arm}: median {np.median(vals_v[i]):.4f}  p5-p95 {np.percentile(vals_v[i], 5):.4f}"
              f"-{np.percentile(vals_v[i], 95):.4f}  rho(v,log10DA)={spearmanr(vals_v[i], da_v)[0]:+.3f}"
              f"  rho(v,log10slope)={spearmanr(vals_v[i], sl_v)[0]:+.3f}")
    print(f"  cross-arm range: median {np.median(rng):.4f}  p90 {np.percentile(rng, 90):.4f}"
          f"  rho(range,log10DA)={spearmanr(rng, da_v)[0]:+.3f}"
          f"  rho(range,log10slope)={spearmanr(rng, sl_v)[0]:+.3f}")

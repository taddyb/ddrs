#!/usr/bin/env python
"""Per-reach loss-gradient map for the alpha-landscape study
(src/experiment/landscape/output.rs).

For each gauge that has the optional `reach_grad` fields (written only when
`landscape.reach_grad: true`), plots how the gauge's loss gradient w.r.t.
each reach's log-parameter (dL/d ln n, dL/d ln p, dL/d ln q) is distributed
over the upstream network, at both the trained point (alpha = 0) and the
per-gauge optimum (alpha*). The sum over reaches of a per-reach gradient
field equals the corresponding basin-uniform gradient component
(`grad0`/`grad_star`, see src/experiment/landscape/mod.rs:368-387) -- this
script prints both as a cross-check in each panel title.

Reads only <run_dir>/manifest.json, <run_dir>/gauges.csv, and
<run_dir>/<arm>/gauges/<staid>.nc. Writes, to <out>:
  - reachgrad_<arm>_<staid>.png  per gauge per arm (2 rows x 4 cols); the arm
    is in the filename because a run can have more than one arm, and
    <staid>.nc alone is not unique across arms.
  - REACHGRAD.md                 summary table across all gauges/arms
  - reachgrad_per_reach.csv      per-reach dump across all gauges/arms

Usage:
    ~/projects/ddr/.venv/bin/python experiments/landscape/reachgrad.py <run_dir> \\
        [--out <run_dir>/figures]
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import xarray as xr  # noqa: E402

ALPHA_INDEX = {"n": 0, "p": 1, "q": 2}
COMPONENTS = ["n", "p", "q"]
TOP_K = 10


# ----------------------------------------------------------------------------- io
def load(run_dir: Path):
    manifest = json.loads((run_dir / "manifest.json").read_text())
    arms = [a["name"] for a in manifest["arms"]]
    gauges = pd.read_csv(run_dir / "gauges.csv", dtype={"staid": str}, keep_default_na=False)
    data: dict[tuple[str, str], xr.Dataset] = {}
    input_files = ["manifest.json", "gauges.csv"]
    for arm in arms:
        for staid in gauges["staid"]:
            p = run_dir / arm / "gauges" / f"{staid}.nc"
            if p.exists():
                data[(arm, staid)] = xr.open_dataset(p, decode_timedelta=False)
                input_files.append(str(p.relative_to(run_dir)))
    return manifest, arms, gauges, data, input_files


def has_reach_grad(ds: xr.Dataset) -> bool:
    return "reach_grad0_n" in ds.variables and "reach_grad_star_n" in ds.variables


def active_mask(ds: xr.Dataset) -> np.ndarray:
    """Bool (n, p, q) mask of which alpha components are learned model
    parameters vs fixed at a constant default. Defaults to all-active for
    netCDFs written before the active/active_params schema existed."""
    if "active" in ds.variables:
        return ds["active"].values.astype(bool)
    return np.array([True, True, True])


# ------------------------------------------------------------------------- stats
def cumulative_share(dist_km: np.ndarray, abs_g: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return (sorted_dist_km, cumulative |g| share) sorted by ascending distance."""
    order = np.argsort(dist_km)
    d_sorted = dist_km[order]
    csum = np.cumsum(abs_g[order])
    total = csum[-1] if len(csum) else 0.0
    share = csum / total if total > 0 else np.full_like(csum, np.nan)
    return d_sorted, share


def distance_at_share(d_sorted: np.ndarray, share: np.ndarray, target: float) -> float:
    if len(share) == 0 or not np.isfinite(share[-1]):
        return float("nan")
    idx = int(np.searchsorted(share, target))
    if idx >= len(d_sorted):
        idx = len(d_sorted) - 1
    return float(d_sorted[idx])


def per_reach_stats(g0: np.ndarray, gs: np.ndarray, dist_km: np.ndarray) -> dict:
    abs_g0, abs_gs = np.abs(g0), np.abs(gs)
    d0_sorted, share0 = cumulative_share(dist_km, abs_g0)
    ds_sorted, shares = cumulative_share(dist_km, abs_gs)

    def top_k_frac(abs_g: np.ndarray) -> float:
        total = abs_g.sum()
        if total <= 0:
            return float("nan")
        k = min(TOP_K, len(abs_g))
        top = np.sort(abs_g)[::-1][:k].sum()
        return float(top / total)

    sign_agree = float(np.mean(np.sign(g0) == np.sign(gs))) if len(g0) else float("nan")

    return {
        "alpha0": dict(
            sum_g=float(g0.sum()), sum_abs_g=float(abs_g0.sum()),
            frac_top10=top_k_frac(abs_g0),
            dist50_km=distance_at_share(d0_sorted, share0, 0.5),
            dist90_km=distance_at_share(d0_sorted, share0, 0.9),
            frac_zero=float(np.mean(g0 == 0.0)) if len(g0) else float("nan"),
            sign_agreement=sign_agree,
        ),
        "alpha_star": dict(
            sum_g=float(gs.sum()), sum_abs_g=float(abs_gs.sum()),
            frac_top10=top_k_frac(abs_gs),
            dist50_km=distance_at_share(ds_sorted, shares, 0.5),
            dist90_km=distance_at_share(ds_sorted, shares, 0.9),
            frac_zero=float(np.mean(gs == 0.0)) if len(gs) else float("nan"),
            sign_agreement=sign_agree,
        ),
    }


# ----------------------------------------------------------------------- figure
def marker_sizes(abs_g: np.ndarray) -> np.ndarray:
    m = float(np.nanmax(abs_g)) if len(abs_g) and np.nanmax(abs_g) > 0 else 1.0
    return 12.0 + 180.0 * np.sqrt(np.clip(abs_g / m, 0.0, None))


def scatter_panel(ax, dist_km: np.ndarray, g: np.ndarray, sum_g: float, netcdf_val: float, label: str):
    colors = np.where(g >= 0, "crimson", "steelblue")
    sizes = marker_sizes(np.abs(g))
    ax.scatter(dist_km, g, c=colors, s=sizes, alpha=0.75, edgecolor="k", linewidth=0.2)
    ax.axhline(0.0, color="0.4", lw=0.7)
    finite_nonzero = np.abs(g[g != 0])
    linthresh = float(np.nanmin(finite_nonzero)) if len(finite_nonzero) else 1e-8
    ax.set_yscale("symlog", linthresh=max(linthresh, 1e-12))
    ax.set_xlabel("distance to gauge [km]")
    ax.set_title(f"{label}: sum g = {sum_g:.4g} (netcdf {netcdf_val:.4g})", fontsize=8)


def cumulative_panel(ax, dist_km: np.ndarray, grads: dict[str, np.ndarray], label: str,
                      active_comps: list[str] | None = None):
    if active_comps is None:
        active_comps = COMPONENTS
    for name in active_comps:
        d_sorted, share = cumulative_share(dist_km, np.abs(grads[name]))
        ax.plot(d_sorted, share, label=name)
    ax.set_xlabel("distance to gauge [km]")
    ax.set_ylabel("cumulative |g| share")
    ax.set_ylim(0, 1.02)
    ax.legend(fontsize=7)
    ax.set_title(f"{label}: cumulative |g| vs distance", fontsize=8)


def plot_gauge(ds: xr.Dataset, arm: str, staid: str, out_png: Path, active_comps: list[str] | None = None):
    if active_comps is None:
        active_comps = COMPONENTS
    dist_km = ds["dist_to_gauge_m"].values.astype(np.float64) / 1000.0
    grad0 = {c: ds[f"reach_grad0_{c}"].values.astype(np.float64) for c in COMPONENTS}
    grad_star = {c: ds[f"reach_grad_star_{c}"].values.astype(np.float64) for c in COMPONENTS}
    nc_grad0 = ds["grad0"].values
    nc_grad_star = ds["grad_star"].values
    nse0 = float(ds["nse0"].values)
    nse_star = float(ds["nse_star"].values)

    fig, axes = plt.subplots(2, 4, figsize=(20, 9))
    for row, (grads, nc_vals, alpha_label) in enumerate(
        [(grad0, nc_grad0, "alpha=0"), (grad_star, nc_grad_star, "alpha*")]
    ):
        for col, comp in enumerate(COMPONENTS):
            ax = axes[row, col]
            if comp not in active_comps:
                # Fixed parameter: reach_grad*_{comp} is a real tensor in the
                # file but not a meaningful sensitivity (comp isn't a
                # learned parameter for this arm) -- don't present it as one.
                ax.axis("off")
                ax.text(0.5, 0.5, f"{comp}: fixed, not shown", ha="center", va="center",
                        fontsize=9, transform=ax.transAxes)
                continue
            g = grads[comp]
            scatter_panel(
                ax, dist_km, g,
                sum_g=float(g.sum()), netcdf_val=float(nc_vals[ALPHA_INDEX[comp]]),
                label=f"{alpha_label} {comp}",
            )
            if col == 0:
                ax.set_ylabel("dL/d ln x (symlog)")
        cumulative_panel(axes[row, 3], dist_km, grads, alpha_label, active_comps)

    fig.suptitle(
        f"{staid}  arm={arm}  reach-gradient map  (NSE {nse0:.3f} -> {nse_star:.3f}, n_reach={ds.attrs['n_reach']})",
        fontsize=12,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(out_png, dpi=150)
    plt.close(fig)


# -------------------------------------------------------------------------- report
def md_table(rows: list[dict], cols: list[str]) -> str:
    lines = ["| " + " | ".join(cols) + " |", "|" + "|".join(["---"] * len(cols)) + "|"]
    for row in rows:
        cells = []
        for c in cols:
            v = row[c]
            if isinstance(v, (float, np.floating)):
                cells.append(f"{v:.4g}" if np.isfinite(v) else "nan")
            else:
                cells.append(str(v))
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def write_report(out: Path, summary_rows: list[dict], arms: list[str], gauges: pd.DataFrame,
                  processed: set, skipped: list[str], input_files: list[str]):
    lines = ["# Per-reach loss-gradient map", ""]
    lines.append(
        f"Arms: {arms}. Gauges with reach_grad data: {sorted(processed)}. "
        f"Gauges skipped (no netCDF yet, or written without landscape.reach_grad): {skipped}."
    )
    lines.append("")
    lines.append(
        "For each gauge, arm, and parameter (n, p, q), at the trained point (alpha = 0) and the "
        "per-gauge optimum (alpha*): sum of the per-reach gradient (should match the netCDF's "
        "basin-uniform grad0/grad_star component), sum of |gradient|, the fraction of total "
        "|gradient| carried by the 10 reaches with the largest |gradient|, the distance from the "
        "gauge within which 50%/90% of total |gradient| lies, the fraction of reaches with an "
        "exactly-zero gradient (clamped parameter or no flow), and the sign agreement between "
        "alpha = 0 and alpha* (fraction of reaches whose gradient sign matches at both points; "
        "identical for both rows of a given gauge/parameter since it compares the two)."
    )
    lines.append("")

    cols = ["arm", "staid", "param", "alpha", "sum_g", "sum_abs_g", "frac_top10",
            "dist50_km", "dist90_km", "frac_zero", "sign_agreement"]
    if summary_rows:
        lines.append(md_table(summary_rows, cols))
    else:
        lines.append("(no gauges with reach_grad data available yet)")
    lines.append("")

    lines.append("## Input files")
    lines.append("")
    for f in input_files:
        lines.append(f"- {f}")
    lines.append("")
    (out / "REACHGRAD.md").write_text("\n".join(lines))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("run_dir", type=Path)
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()

    run_dir = args.run_dir
    out = args.out if args.out is not None else run_dir / "figures"
    out.mkdir(parents=True, exist_ok=True)

    manifest, arms, gauges, data, input_files = load(run_dir)
    print(f"arms: {arms}; gauges: {list(gauges['staid'])}; datasets found: {len(data)}")

    summary_rows: list[dict] = []
    per_reach_rows: list[dict] = []
    processed: set = set()
    skipped: list[str] = []
    written = []

    for arm in arms:
        for staid in gauges["staid"]:
            ds = data.get((arm, staid))
            if ds is None:
                skipped.append(f"{arm}/{staid} (no netCDF)")
                continue
            if not has_reach_grad(ds):
                skipped.append(f"{arm}/{staid} (no reach_grad fields)")
                continue
            processed.add(staid)

            dist_km = ds["dist_to_gauge_m"].values.astype(np.float64) / 1000.0
            comid = ds["comid"].values
            n0, p0, q0 = ds["n0"].values, ds["p0"].values, ds["q0"].values

            # Fixed parameters are dropped from the summary table -- comp
            # isn't a learned model parameter for this arm, so its
            # reach-gradient stats aren't a real sensitivity measurement.
            active_comps = [c for c, a in zip(COMPONENTS, active_mask(ds)) if a]
            for comp in active_comps:
                g0 = ds[f"reach_grad0_{comp}"].values.astype(np.float64)
                gs = ds[f"reach_grad_star_{comp}"].values.astype(np.float64)
                stats = per_reach_stats(g0, gs, dist_km)
                for alpha_label, key in [("0", "alpha0"), ("star", "alpha_star")]:
                    row = dict(arm=arm, staid=staid, param=comp, alpha=alpha_label)
                    row.update(stats[key])
                    summary_rows.append(row)

            for i in range(ds.attrs["n_reach"]):
                per_reach_rows.append(dict(
                    arm=arm, staid=staid, comid=int(comid[i]), dist_to_gauge_m=float(dist_km[i] * 1000.0),
                    n0=float(n0[i]), p0=float(p0[i]), q0=float(q0[i]),
                    reach_grad0_n=float(ds["reach_grad0_n"].values[i]),
                    reach_grad0_p=float(ds["reach_grad0_p"].values[i]),
                    reach_grad0_q=float(ds["reach_grad0_q"].values[i]),
                    reach_grad_star_n=float(ds["reach_grad_star_n"].values[i]),
                    reach_grad_star_p=float(ds["reach_grad_star_p"].values[i]),
                    reach_grad_star_q=float(ds["reach_grad_star_q"].values[i]),
                ))

            out_png = out / f"reachgrad_{arm}_{staid}.png"
            plot_gauge(ds, arm, staid, out_png, active_comps)
            written.append(out_png)
            print(f"wrote {out_png}")

    if per_reach_rows:
        df = pd.DataFrame(per_reach_rows)
        csv_path = out / "reachgrad_per_reach.csv"
        df.to_csv(csv_path, index=False)
        written.append(csv_path)
        print(f"wrote {csv_path}")

    write_report(out, summary_rows, arms, gauges, processed, skipped, input_files)
    written.append(out / "REACHGRAD.md")

    for p in written:
        print("wrote", p)


if __name__ == "__main__":
    main()

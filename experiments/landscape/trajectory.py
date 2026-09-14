#!/usr/bin/env python
"""Training-trajectory figures for the alpha-landscape study (spec section 7.2,
research/specs/2026-09-07-adjoint-landscape-design.md).

Arms are checkpoints of ONE run, named ep-init, ep-01, ep-05, ep-10, ep-20,
ep-30 (the epoch is parsed from the arm name; ep-init = 0). The checkpoint
with the highest epoch (ep-30, by convention) is the reference physical
frame: for every gauge and every checkpoint,

    delta_alpha(ep) = mean over reaches common to ep and the final checkpoint
                      (paired by comid) of ln(field_ep / field_final),
                      per component (n0, p0, q0)

expresses that checkpoint's trained fields as a log-multiplier offset from
the final checkpoint's trained fields, then

    c_k(ep) = eigvec_final[:, k] . (delta_alpha(ep) - alpha_star_final)

re-expresses that offset in the final checkpoint's own eigenbasis at its own
optimum alpha_star_final -- c_k = 0 means the checkpoint's trained point
coincides with the final checkpoint's own per-gauge optimum along axis k.
(For ep = final itself, delta_alpha = 0 by construction and c_k reduces
exactly to coord_trained_final[k], stored in the final checkpoint's netCDF --
a useful self-check.)

Reads only <run_dir>/manifest.json, <run_dir>/gauges.csv, and
<run_dir>/<arm>/gauges/<staid>.nc. Writes <out>/TRAJECTORY.md,
<out>/trajectory_per_gauge.csv, and <out>/trajectory.png.

Usage:
    ~/projects/ddr/.venv/bin/python experiments/landscape/trajectory.py <run_dir> \\
        [--out <run_dir>/figures] [--tol 0.05]
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import xarray as xr  # noqa: E402

FIELD_VARS = ["n0", "p0", "q0"]
COMPONENTS = ["n", "p", "q"]
EPOCH_RE = re.compile(r"^ep-(\d+)$")


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


def parse_epoch(arm: str) -> int:
    if arm == "ep-init":
        return 0
    m = EPOCH_RE.match(arm)
    if not m:
        raise ValueError(f"cannot parse an epoch number from arm name {arm!r} (expected ep-init or ep-<int>)")
    return int(m.group(1))


def align_by_comid(ds_a: xr.Dataset, ds_b: xr.Dataset):
    comid_a = ds_a["comid"].values
    comid_b = ds_b["comid"].values
    common, idx_a, idx_b = np.intersect1d(comid_a, comid_b, return_indices=True)
    return common, idx_a, idx_b


def tol_index(ds: xr.Dataset, tol_target: float) -> int:
    tolerances = ds["tolerances"].values
    return int(np.argmin(np.abs(tolerances - tol_target)))


def active_mask(ds: xr.Dataset) -> np.ndarray:
    """Bool (n, p, q) mask of which alpha components are learned model
    parameters vs fixed at a constant default. Defaults to all-active for
    netCDFs written before the active/active_params schema existed."""
    if "active" in ds.variables:
        return ds["active"].values.astype(bool)
    return np.array([True, True, True])


def gauge_colors(staids):
    cmap = plt.get_cmap("tab10")
    return {s: cmap(i % 10) for i, s in enumerate(staids)}


# --------------------------------------------------------------------- per-gauge
def checkpoint_row(arm: str, epoch: int, staid: str, ds_ep: xr.Dataset, ds_final: xr.Dataset) -> dict:
    common, idx_ep, idx_final = align_by_comid(ds_ep, ds_final)
    n_common = int(len(common))
    d_per_reach = {}
    for var in FIELD_VARS:
        v_ep = ds_ep[var].values[idx_ep]
        v_final = ds_final[var].values[idx_final]
        d_per_reach[var] = np.log(v_ep / v_final)
    delta_alpha = np.array([d_per_reach[v].mean() for v in FIELD_VARS])
    delta_alpha_std = np.array([d_per_reach[v].std() for v in FIELD_VARS])

    alpha_star_final = ds_final["alpha_star"].values.astype(float)
    eigvec_final = ds_final["eigvec_star"].values  # [component, k]
    x = delta_alpha - alpha_star_final
    c = np.einsum("ik,i->k", eigvec_final, x)
    # eigvec_final's dropped eigen-slot (when a component is fixed) is a
    # placeholder basis vector, not a real eigenvector -- the projection onto
    # it is a real-looking but meaningless number. Force it to NaN so the
    # final checkpoint's self-check (c_k == coord_trained_final[k]) holds
    # and c doesn't look like a genuine displacement along a sloppy axis
    # that doesn't exist.
    n_active_final = int(active_mask(ds_final).sum())
    if n_active_final < 3:
        c[n_active_final:] = np.nan

    loss0 = float(ds_ep["loss0"].values)
    nse0 = float(ds_ep["nse0"].values)
    nse_star = float(ds_ep["nse_star"].values)

    row = {
        "arm": arm,
        "epoch": epoch,
        "staid": staid,
        "n_reach_ep": int(ds_ep.attrs["n_reach"]),
        "n_reach_final": int(ds_final.attrs["n_reach"]),
        "n_common": n_common,
        "loss0": loss0,
        "nse0": nse0,
        "nse_star": nse_star,
        "gain": nse_star - nse0,
    }
    for k, comp in enumerate(COMPONENTS):
        row[f"delta_alpha_{comp}"] = float(delta_alpha[k])
        row[f"delta_alpha_std_{comp}"] = float(delta_alpha_std[k])
    # Restrict the combined spread to active components: a fixed parameter's
    # field is a constant default, not a displacement in alpha space, and
    # shouldn't dilute or pad this norm.
    active_final = active_mask(ds_final)
    row["delta_alpha_std_norm"] = float(np.linalg.norm(delta_alpha_std[active_final]))
    for k in range(3):
        row[f"c{k + 1}"] = float(c[k])
    return row


def build_df(arms, gauges, data, final_arm: str) -> pd.DataFrame:
    rows = []
    for arm in arms:
        epoch = parse_epoch(arm)
        for staid in gauges["staid"]:
            ds_ep = data.get((arm, staid))
            ds_final = data.get((final_arm, staid))
            if ds_ep is None or ds_final is None:
                continue
            rows.append(checkpoint_row(arm, epoch, staid, ds_ep, ds_final))
    df = pd.DataFrame(rows)
    if not df.empty:
        df = df.sort_values(["staid", "epoch"]).reset_index(drop=True)
    return df


def final_gauge_stats(gauges, data, final_arm: str, tol_target: float) -> dict:
    """Per gauge: the final checkpoint's own stiff half-width and nse_star."""
    out = {}
    for staid in gauges["staid"]:
        ds = data.get((final_arm, staid))
        if ds is None:
            continue
        ti = tol_index(ds, tol_target)
        hw1 = float(ds["half_width"].values[ti, 0])
        out[staid] = dict(hw1=hw1, tol_used=float(ds["tolerances"].values[ti]),
                           nse_star_final=float(ds["nse_star"].values))
    return out


# ----------------------------------------------------------------------- figure
def fig_trajectory(out: Path, df: pd.DataFrame, final_stats: dict, final_arm: str) -> Path:
    staids = sorted(df["staid"].unique())
    colors = gauge_colors(staids)

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    ax_c1, ax_c23, ax_nse, ax_std = axes[0, 0], axes[0, 1], axes[1, 0], axes[1, 1]

    for staid in staids:
        sub = df[df["staid"] == staid].sort_values("epoch")
        color = colors[staid]
        ax_c1.plot(sub["epoch"], sub["c1"], marker="o", color=color, label=staid)
        fs = final_stats.get(staid)
        if fs is not None and np.isfinite(fs["hw1"]):
            ax_c1.axhline(fs["hw1"], color=color, ls=":", lw=0.9)
            ax_c1.axhline(-fs["hw1"], color=color, ls=":", lw=0.9)

        ax_c23.plot(sub["epoch"], sub["c2"], marker="o", color=color, ls="-", label=f"{staid} c2")
        ax_c23.plot(sub["epoch"], sub["c3"], marker="s", color=color, ls="--", label=f"{staid} c3")

        ax_nse.plot(sub["epoch"], sub["nse0"], marker="o", color=color, label=staid)
        if fs is not None:
            ax_nse.axhline(fs["nse_star_final"], color=color, ls="--", lw=1.0)

        ax_std.plot(sub["epoch"], sub["delta_alpha_std_norm"], marker="o", color=color, label=staid)

    ax_c1.axhline(0.0, color="k", lw=0.6)
    ax_c1.set_xlabel("epoch")
    ax_c1.set_ylabel("c1 (stiff)")
    ax_c1.set_title(f"(a) stiff coordinate vs epoch, {final_arm} half-width band (dotted)")
    ax_c1.legend(fontsize=7)

    ax_c23.axhline(0.0, color="k", lw=0.6)
    ax_c23.set_xlabel("epoch")
    ax_c23.set_ylabel("c2, c3 (sloppy)")
    ax_c23.set_title("(b) sloppy coordinates vs epoch (solid c2, dashed c3)")
    ax_c23.legend(fontsize=6, ncol=2)

    ax_nse.set_xlabel("epoch")
    ax_nse.set_ylabel("NSE at trained point (nse0)")
    ax_nse.set_title(f"(c) trained-point NSE vs epoch, dashed = {final_arm} nse_star")
    ax_nse.legend(fontsize=7)

    ax_std.set_xlabel("epoch")
    ax_std.set_ylabel("|delta_alpha_std| (L2 over n, p, q)")
    ax_std.set_title("(d) across-reach spread of the per-checkpoint field change")
    ax_std.legend(fontsize=7)

    fig.suptitle(f"Training trajectory in the {final_arm} physical frame and eigenbasis", fontsize=10)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    p = out / "trajectory.png"
    fig.savefig(p, dpi=150)
    plt.close(fig)
    return p


# -------------------------------------------------------------------------- report
def md_table(df: pd.DataFrame, cols: list[str]) -> str:
    lines = ["| " + " | ".join(cols) + " |", "|" + "|".join(["---"] * len(cols)) + "|"]
    for _, row in df.iterrows():
        cells = []
        for c in cols:
            v = row[c]
            if isinstance(v, (float, np.floating)):
                cells.append(f"{v:.4g}" if np.isfinite(v) else "nan")
            else:
                cells.append(str(v))
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def write_report(out: Path, df: pd.DataFrame, arms, final_arm, gauges, input_files, tol_target, is_smoke: bool):
    lines = ["# Landscape training trajectory (spec section 7.2)", ""]
    if is_smoke:
        lines.append(
            "**This run is a synthetic smoke test, not a real training trajectory.** It exercises "
            "the pipeline end to end; the numbers have no scientific meaning."
        )
        lines.append("")
    present = set(df["staid"]) if not df.empty else set()
    missing = [s for s in gauges["staid"] if s not in present]
    lines.append(f"Arms (checkpoints): {arms}. Reference (final) checkpoint: {final_arm}.")
    lines.append(f"Gauges with data: {sorted(present)}. Gauges not yet available: {missing}.")
    lines.append("")

    lines.append("## Table 1: per-checkpoint per-gauge trajectory")
    lines.append("")
    t1_cols = [
        "arm", "epoch", "staid", "n_common", "n_reach_ep", "n_reach_final",
        "delta_alpha_n", "delta_alpha_p", "delta_alpha_q",
        "delta_alpha_std_n", "delta_alpha_std_p", "delta_alpha_std_q",
        "c1", "c2", "c3", "loss0", "nse0", "nse_star", "gain",
    ]
    if df.empty:
        lines.append("(no checkpoint netCDFs available yet)")
    else:
        lines.append(md_table(df, t1_cols))
    lines.append("")
    lines.append(
        f"delta_alpha(ep) is the basin-uniform mean, over reaches paired by comid with the {final_arm} "
        "checkpoint, of ln(field_ep / field_final) for n, p, q; delta_alpha_std is the across-reach "
        f"standard deviation of that same per-reach log ratio. c1, c2, c3 express delta_alpha(ep) minus "
        f"alpha_star_{final_arm}, projected onto {final_arm}'s own Hessian eigenvectors at its optimum -- "
        "c_k = 0 means the checkpoint's trained point coincides with the final checkpoint's own "
        f"per-gauge optimum along that axis. loss0/nse0/nse_star/gain are the checkpoint's OWN "
        "landscape diagnostics (not projected)."
    )
    lines.append("")
    lines.append("## Input files")
    lines.append("")
    for f in input_files:
        lines.append(f"- {f}")
    lines.append("")
    (out / "TRAJECTORY.md").write_text("\n".join(lines))


# ------------------------------------------------------------------------------ main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("run_dir", type=Path)
    ap.add_argument("--out", type=Path, default=None)
    ap.add_argument("--tol", type=float, default=0.05)
    ap.add_argument("--smoke-test", action="store_true",
                     help="label the report as a synthetic smoke test (no scientific meaning)")
    args = ap.parse_args()

    run_dir = args.run_dir
    out = args.out if args.out is not None else run_dir / "figures"
    out.mkdir(parents=True, exist_ok=True)

    manifest, arms, gauges, data, input_files = load(run_dir)
    epochs = {a: parse_epoch(a) for a in arms}
    final_arm = max(arms, key=lambda a: epochs[a])
    print(f"arms: {arms}; epochs: {epochs}; final (reference) checkpoint: {final_arm}; "
          f"gauges: {list(gauges['staid'])}; datasets found: {len(data)}")

    df = build_df(arms, gauges, data, final_arm)
    final_stats = final_gauge_stats(gauges, data, final_arm, args.tol)

    written = []
    if not df.empty:
        df.to_csv(out / "trajectory_per_gauge.csv", index=False)
        written.append(out / "trajectory_per_gauge.csv")
        written.append(fig_trajectory(out, df, final_stats, final_arm))

    write_report(out, df, arms, final_arm, gauges, input_files, args.tol, args.smoke_test)
    written.append(out / "TRAJECTORY.md")

    if not df.empty:
        print(md_table(df, ["arm", "epoch", "staid", "c1", "c2", "c3", "nse0", "gain", "n_common"]))
    else:
        print("no checkpoints available yet")

    for p in written:
        print("wrote", p)


if __name__ == "__main__":
    main()

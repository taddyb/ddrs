"""Per-gauge tau sweep on a pre-trim hourly eval dump.

Spec: docs/superpowers/specs/2026-08-05-per-gauge-tau-sweep-design.md

Reads the DDRS_HOURLY_DUMP raw f32 (n_gauges, n_hours) + `.json` sidecar and
the eval run's predictions.zarr, reconstructs daily predictions for each
tau in 0..23 (block mean of hours [13+tau+24i, 13+tau+24(i+1)), scored
against obs day i+1 -- reproduces `tau_trim_and_downsample` exactly at the
shipped tau), and reports per-gauge NSE(tau) restricted to a pilot window.

Run:
    uv run --with "zarr>=3" --with numpy --with pandas --with matplotlib \
        python scripts/tau_sweep.py \
        --dump output/tau_sweep/hourly_full.f32 \
        --zarr output/tau_sweep/eval_full.zarr \
        --baseline-dir .ddrs/runs/<id>/baseline \
        --gages-csv ~/projects/ddr/references/gage_info/gages_2000_area_balanced.csv \
        --out-dir output/tau_sweep \
        --pilot-start 1995-10-01 --pilot-end 1996-09-30
"""

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import zarr

TAUS = np.arange(24)
AREA_BINS = [0, 1000, 5000, 10000, 30000, 50000]
AREA_LABELS = ["0~1000", "1000~5000", "5000~10000", "10000~30000", "30000~50000"]
MIN_VALID_DAYS = 100


def nse(pred: np.ndarray, obs: np.ndarray) -> np.ndarray:
    """Per-gauge NSE over axis 1, NaN-masked. Returns (G,) with NaN where undefined."""
    valid = np.isfinite(obs) & np.isfinite(pred)
    n_valid = valid.sum(axis=1)
    o = np.where(valid, obs, 0.0)
    p = np.where(valid, pred, 0.0)
    o_mean = o.sum(axis=1) / np.maximum(n_valid, 1)
    sse = (((p - o) * valid) ** 2).sum(axis=1)
    svar = (((o - o_mean[:, None]) * valid) ** 2).sum(axis=1)
    out = 1.0 - sse / np.where(svar > 0, svar, np.nan)
    out[n_valid < MIN_VALID_DAYS] = np.nan
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dump", required=True)
    ap.add_argument("--zarr", required=True)
    ap.add_argument("--baseline-dir", required=True)
    ap.add_argument("--gages-csv", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--pilot-start", default="1995-10-01")
    ap.add_argument("--pilot-end", default="1996-09-30")
    args = ap.parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    meta = json.loads(Path(args.dump + ".json").read_text())
    g_n, h_n, tau_ship = meta["n_gauges"], meta["n_hours"], meta["tau_shipped"]
    hourly = np.memmap(args.dump, dtype=np.float32, mode="r", shape=(g_n, h_n))

    zg = zarr.open_group(args.zarr, mode="r")
    zpred = zg["predictions"][:]  # (G, D)
    zobs = zg["observations"][:]
    ztime = pd.to_datetime(zg["time"][:])  # ns since epoch
    gage_ids = np.array([bytes(r).decode().strip("\x00") for r in zg["gage_ids"][:]])
    d_n = zpred.shape[1]
    assert zpred.shape[0] == g_n, f"gauge mismatch: dump {g_n} vs zarr {zpred.shape[0]}"

    # --- Method-validation gate item 1: reproduce shipped tau exactly. ---
    start = 13 + tau_ship
    recon = np.asarray(hourly[:, start : start + 24 * d_n], dtype=np.float64)
    recon = recon.reshape(g_n, d_n, 24).mean(axis=2)
    denom = np.maximum(np.abs(zpred), 1e-6)
    rel = np.abs(recon - zpred) / denom
    print(f"[gate 1] tau={tau_ship} reconstruction: max rel {np.nanmax(rel):.2e}, "
          f"median rel {np.nanmedian(rel):.2e} (PASS if max < 1e-3)")
    gate1 = np.nanmax(rel) < 1e-3

    # --- Gate item 2: full-window NSE at shipped tau matches the eval's own daily output. ---
    nse_ship_full = nse(zpred, zobs)
    nse_recon_full = nse(recon, zobs)
    d_med = abs(np.nanmedian(nse_ship_full) - np.nanmedian(nse_recon_full))
    print(f"[gate 2] full-window median NSE shipped {np.nanmedian(nse_ship_full):.4f} "
          f"vs recon {np.nanmedian(nse_recon_full):.4f} (PASS if |d| < 0.001)")
    gate2 = d_med < 0.001

    # --- Sweep: common day count so every tau scores the same days. ---
    # Highest start is 13+23; need 13+23+24*D <= h_n.
    d_common = min(d_n, (h_n - (13 + int(TAUS.max()))) // 24)
    pilot = (ztime[:d_common] >= pd.Timestamp(args.pilot_start)) & (
        ztime[:d_common] <= pd.Timestamp(args.pilot_end)
    )
    pilot = np.asarray(pilot)
    print(f"pilot window {args.pilot_start}..{args.pilot_end}: {pilot.sum()} days")
    obs_pilot = zobs[:, :d_common][:, pilot]

    nse_by_tau = np.full((g_n, len(TAUS)), np.nan)
    for tau in TAUS:
        s = 13 + int(tau)
        daily = np.asarray(hourly[:, s : s + 24 * d_common], dtype=np.float64)
        daily = daily.reshape(g_n, d_common, 24).mean(axis=2)
        nse_by_tau[:, tau] = nse(daily[:, pilot], obs_pilot)
    gate3_frac = np.isfinite(nse_by_tau).all(axis=1).mean()
    print(f"[gate 3] gauges with full NSE(tau) curves: {gate3_frac:.1%} (PASS if >= 95%)")
    gate3 = gate3_frac >= 0.95

    # --- Baseline (same population), pilot window. ---
    bdir = Path(args.baseline_dir)
    bman = json.loads((bdir / "manifest.json").read_text())
    bg, bd = bman["n_gauges"], bman["n_days"]
    btime = pd.to_datetime(bman["time_range_daily"])
    bpred = np.fromfile(bdir / "predictions.f32", dtype=np.float32).reshape(bg, bd)
    bobs = np.fromfile(bdir / "observations.f32", dtype=np.float32).reshape(bg, bd)
    border = pd.Index(bman["gage_ids"]).get_indexer(gage_ids)
    covered = border >= 0
    if not covered.all():
        print(f"WARNING: baseline covers {covered.sum()}/{g_n} eval gauges; rest -> NaN")
    bmask = np.asarray(
        (btime >= pd.Timestamp(args.pilot_start)) & (btime <= pd.Timestamp(args.pilot_end))
    )
    nse_base = np.full(g_n, np.nan)
    nse_base[covered] = nse(
        bpred[border[covered]][:, bmask], bobs[border[covered]][:, bmask]
    )

    # --- Covariates + summary table. ---
    gcsv = pd.read_csv(args.gages_csv, dtype={"STAID": str})
    gcsv["STAID"] = gcsv["STAID"].str.zfill(8)
    cov = gcsv.set_index("STAID").reindex(gage_ids)
    has_curve = np.isfinite(nse_by_tau).any(axis=1)
    best_tau = np.zeros(g_n, dtype=int)
    best_tau[has_curve] = np.nanargmax(nse_by_tau[has_curve], axis=1)
    best_tau = np.where(has_curve, best_tau, tau_ship)
    df = pd.DataFrame(
        {
            "gage_id": gage_ids,
            "drain_sqkm": cov["DRAIN_SQKM"].to_numpy(),
            "lng_gage": cov["LNG_GAGE"].to_numpy(),
            "has_curve": has_curve,
            "best_tau": best_tau,
            "nse_tau_shipped": nse_by_tau[:, tau_ship],
            "nse_tau_best": nse_by_tau[np.arange(g_n), best_tau],
            "nse_baseline": nse_base,
        }
    )
    df["area_bin"] = pd.cut(df["drain_sqkm"], AREA_BINS, labels=AREA_LABELS)
    df.to_csv(out_dir / "best_tau_wy1996.csv", index=False)
    pd.DataFrame(nse_by_tau, index=gage_ids, columns=[f"tau_{t}" for t in TAUS]).to_csv(
        out_dir / "nse_by_tau_wy1996.csv"
    )

    med = df.groupby("area_bin", observed=True)[
        ["nse_tau_shipped", "nse_tau_best", "nse_baseline"]
    ].median()
    med["n"] = df.groupby("area_bin", observed=True).size()
    improved = df["nse_tau_best"] - df["nse_tau_shipped"] > 0.01
    rho = df.loc[improved, ["best_tau", "lng_gage"]].corr(method="spearman").iloc[0, 1]

    lines = [
        "# tau sweep pilot (WY1996) summary",
        "",
        f"- gate 1 (recon match): {'PASS' if gate1 else 'FAIL'}",
        f"- gate 2 (NSE match):   {'PASS' if gate2 else 'FAIL'}",
        f"- gate 3 (curve cover): {'PASS' if gate3 else 'FAIL'} ({gate3_frac:.1%})",
        "",
        "Median NSE by drainage-area bin (pilot window, in-sample best tau):",
        "",
        med.to_markdown(),
        "",
        f"- gauges improved > 0.01 by best tau: {improved.sum()} / {g_n}",
        f"- spearman(best_tau, longitude) on improved gauges: {rho:.3f}",
        f"- best-tau distribution (has_curve only): "
        f"{np.bincount(best_tau[has_curve], minlength=24).tolist()}",
    ]
    (out_dir / "summary_wy1996.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))

    # --- Plots. ---
    fig, ax = plt.subplots(figsize=(8, 5))
    for label in AREA_LABELS:
        sel = (df["area_bin"] == label).to_numpy()
        if sel.any():
            ax.plot(TAUS, np.nanmedian(nse_by_tau[sel], axis=0), marker="o", label=f"{label} (n={sel.sum()})")
    ax.axvline(tau_ship, color="k", ls="--", lw=1, label=f"shipped tau={tau_ship}")
    ax.set_xlabel("tau (h)"), ax.set_ylabel("median NSE"), ax.legend(fontsize=8)
    ax.set_title("Median NSE vs tau by drainage area (WY1996)")
    fig.savefig(out_dir / "nse_vs_tau_by_area.png", dpi=150, bbox_inches="tight")

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.hist(df.loc[improved, "best_tau"], bins=np.arange(25) - 0.5)
    ax.set_xlabel("best tau (h)"), ax.set_ylabel("gauges (improved > 0.01)")
    fig.savefig(out_dir / "best_tau_hist.png", dpi=150, bbox_inches="tight")

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.scatter(df.loc[improved, "lng_gage"], df.loc[improved, "best_tau"], s=8, alpha=0.5)
    ax.set_xlabel("longitude"), ax.set_ylabel("best tau (h)")
    ax.set_title(f"spearman rho = {rho:.3f} (improved gauges)")
    fig.savefig(out_dir / "best_tau_vs_longitude.png", dpi=150, bbox_inches="tight")


if __name__ == "__main__":
    main()

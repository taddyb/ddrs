#!/usr/bin/env python
"""Stage 1 channel-geometry tables and figures from the merged (n, p, q) census.

Reads <merged>/uh-free-channel/gauges/*.nc (one per gauge) and writes
<out>/stage1_per_gauge.csv, <out>/stage1_tables.json, and three figures.
Definitions are stated in research/findings/*-stage1-census-findings.md.

Usage:
    ~/projects/ddr/.venv/bin/python experiments/landscape/stage1_channel_geometry.py <merged> [--out DIR]
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

LN2 = float(np.log(2.0))
ALPHA_MAX = 2.3025851
NAMES = ["n", "p", "q"]


def load(merged: Path) -> pd.DataFrame:
    rows = []
    for f in sorted((merged / "uh-free-channel" / "gauges").glob("*.nc")):
        d = xr.open_dataset(f, decode_timedelta=False)
        r = dict(staid=d.attrs["staid"], n_reach=int(d.attrs["n_reach"]),
                 obs_mean=float(d.attrs["obs_mean_q_m3s"]),
                 loss0=float(d.loss0), loss_star=float(d.loss_star),
                 nse0=float(d.nse0), kge0=float(d.kge0), nse_star=float(d.nse_star),
                 kge_star=float(d.kge_star), clamped=float(d.attrs["clamped_frac_star"]),
                 hit_bound=int(d.attrs["hit_range_bound"]),
                 grad_fallback=int(d.attrs["used_gradient_fallback"]))
        g0 = d.grad0.values.astype(float)
        H0 = d.hess0.values.astype(float)
        Hs = d.hess_star.values.astype(float)
        a = d.alpha_star.values.astype(float)
        ev = d.eigval_star.values.astype(float)
        V = d.eigvec_star.values.astype(float)
        for i, nm in enumerate(NAMES):
            r[f"alpha_{nm}"] = a[i]
            r[f"g0_{nm}"] = g0[i]
            r[f"H0_{nm}"] = H0[i, i]
            r[f"Hs_{nm}"] = Hs[i, i]
            # loss change for a factor-two move, trained point (quadratic form
            # |g| ln2 + 0.5 |H| ln2^2), relative to the loss there
            r[f"dL2_{nm}"] = (abs(g0[i]) * LN2 + 0.5 * abs(H0[i, i]) * LN2**2) / r["loss0"]
            r[f"edge_{nm}"] = int(abs(a[i]) >= ALPHA_MAX - 1e-3)
        for k in range(3):
            r[f"lam{k+1}"] = ev[k]
        # sign-normalised stiff eigenvector (v1_n >= 0) and flat eigenvector
        v1 = V[:, 0] * (1 if V[0, 0] >= 0 else -1)
        v3 = V[:, 2]
        for i, nm in enumerate(NAMES):
            r[f"v1_{nm}"] = v1[i]
            r[f"v3_{nm}"] = v3[i]
        # curvature along the depth-invariant direction (1,1,0)/sqrt2 and its
        # orthogonal partner (1,-1,0)/sqrt2, at the optimum and the trained point
        u_same = np.array([1, 1, 0]) / np.sqrt(2)
        u_opp = np.array([1, -1, 0]) / np.sqrt(2)
        for tag, H in (("s", Hs), ("0", H0)):
            r[f"Huu_np_same_{tag}"] = float(u_same @ H @ u_same)
            r[f"Huu_np_opp_{tag}"] = float(u_opp @ H @ u_opp)
        r["indef_star"] = int(ev.min() <= 0)
        r["indef_trained"] = int(np.linalg.eigvalsh(H0).min() <= 0)
        r["cos_v1_same"] = abs(float(v1 @ u_same))
        r["cos_v3_same"] = abs(float(v3 @ u_same))
        rows.append(r)
    return pd.DataFrame(rows)


def bands(df: pd.DataFrame):
    # KGE bands on the trained-point KGE over the five-year window
    edges = [-np.inf, 0.0, 0.3, 0.6, np.inf]
    labels = ["KGE < 0", "0 to 0.3", "0.3 to 0.6", ">= 0.6"]
    return pd.cut(df["kge0"], edges, labels=labels, right=False)


def summarize(sub: pd.DataFrame) -> dict:
    out = {"n_gauges": int(len(sub))}
    if len(sub) == 0:
        return out
    for nm in NAMES:
        out[f"flat_{nm}_1pct"] = float((sub[f"dL2_{nm}"] < 0.01).mean())
        out[f"H0_concave_{nm}"] = float((sub[f"H0_{nm}"] < 0).mean())
        out[f"Hs_concave_{nm}"] = float((sub[f"Hs_{nm}"] < 0).mean())
        out[f"edge_{nm}"] = float(sub[f"edge_{nm}"].mean())
    out["flat_q_0.5pct"] = float((sub.dL2_q < 0.005).mean())
    out["flat_q_2pct"] = float((sub.dL2_q < 0.02).mean())
    out["q_flat_and_n_curved"] = float(((sub.dL2_q < 0.01) & (sub.dL2_n >= 0.01)).mean())
    out["n_flat_and_q_curved"] = float(((sub.dL2_n < 0.01) & (sub.dL2_q >= 0.01)).mean())
    for tag in ("0", "s"):
        for nm in ("p", "q"):
            # ratio of absolute diagonal curvature to n's; medians over gauges
            # with a nonzero n curvature
            r = (sub[f"H{tag}_{nm}"].abs() / sub[f"H{tag}_n"].abs()).replace([np.inf], np.nan)
            out[f"ratio_{nm}_to_n_{'trained' if tag=='0' else 'optimum'}_median"] = float(r.median())
            out[f"ratio_{nm}_to_n_{'trained' if tag=='0' else 'optimum'}_q25_q75"] = [float(r.quantile(.25)), float(r.quantile(.75))]
    out["indef_star"] = float(sub.indef_star.mean())
    out["indef_trained"] = float(sub.indef_trained.mean())
    out["lam3_le0_star"] = float((sub.lam3 <= 0).mean())
    out["lam1_le0_star"] = float((sub.lam1 <= 0).mean())
    for nm in NAMES:
        out[f"v1_{nm}_median"] = float(sub[f"v1_{nm}"].median())
        out[f"v1_{nm}_q25_q75"] = [float(sub[f"v1_{nm}"].quantile(.25)), float(sub[f"v1_{nm}"].quantile(.75))]
    # exponents of the stiff combination n^a p^b q^c scaled so a = 1 and a = 0.8 norm
    good = sub[sub.v1_n.abs() > 0.2]
    out["stiff_b_over_a_median"] = float((good.v1_p / good.v1_n).median())
    out["stiff_c_over_a_median"] = float((good.v1_q / good.v1_n).median())
    out["stiff_n_dominant_share"] = float((sub[["v1_n", "v1_p", "v1_q"]].abs().idxmax(axis=1) == "v1_n").mean())
    out["cos_v1_with_n_equals_p_median"] = float(sub.cos_v1_same.median())
    out["cos_v3_with_n_equals_p_median"] = float(sub.cos_v3_same.median())
    out["Huu_same_over_opp_star_median"] = float((sub.Huu_np_same_s / sub.Huu_np_opp_s).median())
    out["Huu_same_over_Hnn_star_median"] = float((sub.Huu_np_same_s / sub.Hs_n).median())
    out["Huu_opp_over_Hnn_star_median"] = float((sub.Huu_np_opp_s / sub.Hs_n).median())
    out["hit_bound"] = float(sub.hit_bound.mean())
    out["kge0_median"] = float(sub.kge0.median())
    out["nse0_median"] = float(sub.nse0.median())
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("merged", type=Path)
    ap.add_argument("--out", type=Path, default=None)
    a = ap.parse_args()
    out = a.out or a.merged / "figures" / "stage1"
    out.mkdir(parents=True, exist_ok=True)
    df = load(a.merged)
    df["band"] = bands(df)
    df.to_csv(out / "stage1_per_gauge.csv", index=False)

    tables = {"all": summarize(df), "nse0_gt_0.3": summarize(df[df.nse0 > 0.3])}
    for b in df.band.cat.categories:
        tables[f"kge0 {b}"] = summarize(df[df.band == b])
    tables["kge0 > 0.3 (well fit, KGE)"] = summarize(df[df.kge0 > 0.3])
    (out / "stage1_tables.json").write_text(json.dumps(tables, indent=1))

    wf = df[df.nse0 > 0.3]
    # --- Figure 1: curvature ratios of p and q to n at the optimum, by KGE band
    fig, ax = plt.subplots(1, 3, figsize=(15, 4.4))
    cols = {"KGE < 0": "#b2182b", "0 to 0.3": "#ef8a62", "0.3 to 0.6": "#67a9cf", ">= 0.6": "#2166ac"}
    for nm, axx in zip(("p", "q"), ax[:2]):
        for b, c in cols.items():
            s = df[df.band == b]
            r = (s[f"Hs_{nm}"].abs() / s.Hs_n.abs()).replace([np.inf], np.nan).dropna()
            r = np.sort(r.values)
            axx.plot(r, np.arange(1, len(r) + 1) / len(r), color=c, label=f"{b} (n={len(s)})")
        axx.set_xscale("log"); axx.axvline(1, color="k", lw=.6, ls=":")
        axx.set_xlabel(f"|H_{nm}{nm}| / |H_nn| at the optimum"); axx.set_ylabel("cumulative share of gauges")
        axx.set_title(f"curvature of {nm} as a fraction of n's")
    ax[0].legend(fontsize=8)
    # dL2 for q
    for b, c in cols.items():
        s = df[df.band == b]
        r = np.sort(s.dL2_q.values)
        ax[2].plot(r, np.arange(1, len(r) + 1) / len(r), color=c)
    ax[2].set_xscale("log"); ax[2].axvline(0.01, color="k", lw=.8, ls="--")
    ax[2].set_xlabel("relative loss change for a factor-two move in q (trained point)")
    ax[2].set_ylabel("cumulative share of gauges"); ax[2].set_title("how flat is q (dashed: 1 %)")
    fig.tight_layout(); fig.savefig(out / "fig1_curvature_ratios.png", dpi=130); plt.close(fig)

    # --- Figure 2: stiff eigenvector components
    fig, ax = plt.subplots(1, 2, figsize=(11, 4.6))
    s = df[df.v1_n.abs() > 0.0]
    sc = ax[0].scatter(s.v1_p, s.v1_n, c=np.log10(s.lam1.clip(lower=1e-6)), s=7, cmap="viridis", alpha=.6)
    ax[0].set_xlabel("stiff eigenvector, p component"); ax[0].set_ylabel("stiff eigenvector, n component")
    ax[0].axhline(0, color="k", lw=.4); ax[0].axvline(0, color="k", lw=.4)
    fig.colorbar(sc, ax=ax[0], label="log10 stiff eigenvalue")
    ax[0].set_title("stiff direction (sign: n component >= 0)")
    ax[1].hist(np.degrees(np.arctan2(s.v1_p, s.v1_n)), bins=72, color="#2166ac")
    ax[1].set_xlabel("angle of stiff direction in the (n, p) plane (deg; 0 = pure n)")
    ax[1].set_ylabel("gauges"); ax[1].set_title("direction of the stiff axis")
    fig.tight_layout(); fig.savefig(out / "fig2_stiff_eigenvector.png", dpi=130); plt.close(fig)

    # --- Figure 3: share of gauges whose parameter is curved enough, vs bar
    bars = np.logspace(-3, 0, 60)
    fig, ax = plt.subplots(figsize=(6.5, 4.4))
    for sub, ls, lab in ((wf, "-", "NSE at trained point > 0.3"), (df, "--", "all gauges")):
        for nm, c in (("p", "#1b7837"), ("q", "#762a83")):
            ratio = sub[f"Hs_{nm}"] / sub.Hs_n.abs()
            share = [float(((sub[f"Hs_{nm}"] > 0) & (ratio >= b)).mean()) for b in bars]
            ax.plot(bars, share, color=c, ls=ls, label=f"{nm}, {lab}")
    ax.set_xscale("log"); ax.set_xlabel("bar: H_ii / |H_nn| at the optimum, with H_ii > 0")
    ax.set_ylabel("share of gauges at or above the bar")
    ax.set_title("share of gauges where p and q are curved enough"); ax.legend(fontsize=7)
    fig.tight_layout(); fig.savefig(out / "fig3_share_vs_bar.png", dpi=130); plt.close(fig)
    print(json.dumps({k: v["n_gauges"] for k, v in tables.items()}))


if __name__ == "__main__":
    main()

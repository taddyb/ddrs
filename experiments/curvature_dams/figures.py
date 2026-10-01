"""Figures for the dam-parameter curvature study (PNG, light mode, reference palette slots in fixed order)."""
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
BLUE, ORANGE, AQUA, YELLOW, MAGENTA, GREEN, VIOLET, RED = ("#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4",
                                                           "#008300", "#4a3aa7", "#e34948")
INK, INK2, MUTED, GRID = "#0b0b0b", "#52514e", "#8a8984", "#e4e3df"
LN10 = np.log(10)
plt.rcParams.update({"font.size": 9, "axes.edgecolor": MUTED, "axes.labelcolor": INK2, "xtick.color": INK2,
                     "ytick.color": INK2, "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6,
                     "axes.spines.top": False, "axes.spines.right": False, "figure.facecolor": "white",
                     "axes.titlesize": 9, "axes.titlecolor": INK, "legend.frameon": False})

S = np.load(os.path.join(HERE, "slices.npz"))
PL = np.load(os.path.join(HERE, "planes.npz"))
df = pd.read_csv(os.path.join(HERE, "per_gauge_slices_derived.csv"), dtype={"STAID": str})
PICK = open(os.path.join(HERE, "picked_gauges.txt")).read().split()
import landscapes as L  # noqa: E402

NAMES = {s: L.DAM.loc[s, "dam_name"] for s in PICK}
INIT = dict(T0=L.T0_HEAD, amp=0.0, z=0.05, kc=3.0, phi=0.5)


# ------------------------------------------------------------------ fig 1: 1-D landscapes at three gauges
def fig1():
    cols = [("L4", "T0", "T0 (d), bucket + rule curve"), ("L4", "amp", "rule-curve amplitude s"),
            ("FA", "z", "pool size z (d of mean inflow)"), ("FA", "kc", "release target kc (x mean inflow)"),
            ("FA", "phi", "capture share phi")]
    fig, ax = plt.subplots(3, 5, figsize=(14, 7.6), sharey=True)
    for i, s in enumerate(PICK):
        for j, (law, name, lab) in enumerate(cols):
            a = ax[i, j]
            k = f"{s}|{law}|{name}"
            v, tr, te = S[k + "|vals"], S[k + "|tr"], S[k + "|te"]
            a.axhspan(-0.005, 0.0, color=BLUE, alpha=0.08, lw=0)
            a.axhspan(-0.02, -0.005, color=BLUE, alpha=0.04, lw=0)
            a.plot(v, tr - tr.max(), color=BLUE, lw=2, label="train WY1983-1995")
            a.plot(v, te - te.max(), color=ORANGE, lw=2, ls="--", label="test WY1996-2010")
            r = df[(df.STAID == s) & (df.law == law) & (df.param == name)].iloc[0]
            a.axvline(INIT[name], color=MUTED, lw=1, ls=":")
            a.axvline(r.v_opt, color=INK, lw=0.8)
            if name in ("T0", "z", "kc"):
                a.set_xscale("log")
            a.set_ylim(-0.08, 0.006)
            if i == 0:
                a.set_title(lab)
            if j == 0:
                a.set_ylabel(f"{s} {NAMES[s]}\nNSE minus its max")
            if i == 0 and j == 0:
                a.legend(loc="lower left", fontsize=7)
            if i == 0 and j == 2:
                a.text(INIT["z"] * 1.2, -0.075, "engine init", color=MUTED, fontsize=7)
    fig.suptitle("Figure 1. One-dimensional training (solid) and test (dashed) NSE landscapes at three flood-control "
                 "dams (other parameters at their fitted values)\nshaded: 0.005 and 0.02 NSE bands; dotted: engine "
                 "init; black: training optimum", fontsize=10, color=INK)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(os.path.join(HERE, "fig1_landscapes_3gauges.png"), dpi=150)
    plt.close(fig)


# ------------------------------------------------------------------ fig 2: distributions per parameter
def fig2():
    rt = pd.read_csv(L.ROUTING if hasattr(L, "ROUTING") else
                     "/home/tbindas/projects/ddrs/.ddrs/experiments/landscape-p21-all-5yr/merged/p21-conus/summary.csv",
                     dtype={"staid": str})
    rt = rt[rt.nse0 > 0.3]
    Hnn = rt.lambda1 * rt.v1_n ** 2 + rt.lambda2 * rt.v1_q ** 2
    Hqq = rt.lambda1 * rt.v1_q ** 2 + rt.lambda2 * rt.v1_n ** 2
    jt = pd.read_csv(os.path.join(HERE, "joint_init.csv"), dtype={"STAID": str})
    ji, jm = jt[jt.point == "init"].set_index("STAID"), jt[jt.point == "mid"].set_index("STAID")
    groups = [("T0\n(L4)", df[(df.law == "L4") & (df.param == "T0")]), ("rule\ncurve", df[(df.law == "L4") & (df.param == "amp")]),
              ("pool\nz", df[(df.law == "FA") & (df.param == "z")]), ("pool\nkc", df[(df.law == "FA") & (df.param == "kc")]),
              ("pool phi\n(logit)", df[(df.law == "FA") & (df.param == "phi")])]
    colors = [BLUE, ORANGE, AQUA, YELLOW, MAGENTA, INK2, MUTED]
    fig, ax = plt.subplots(1, 4, figsize=(17, 4.8))

    def strip(a, vals, labels, log=True, hline=None, hlab=None):
        rng = np.random.default_rng(0)
        for i, (v, c) in enumerate(zip(vals, colors)):
            v = np.asarray(v, float)
            v = v[np.isfinite(v) & ((v > 0) if log else True)]
            y = np.log10(v) if log else v
            a.scatter(i + rng.uniform(-0.25, 0.25, len(y)), y, s=8, color=c, alpha=0.45, lw=0)
            q = np.percentile(y, [25, 50, 75])
            a.plot([i - 0.32, i + 0.32], [q[1]] * 2, color=INK, lw=2)
            a.plot([i, i], [q[0], q[2]], color=INK, lw=1)
        a.set_xticks(range(len(labels)))
        a.set_xticklabels(labels, fontsize=7.5)
        if hline is not None:
            a.axhline(np.log10(hline) if log else hline, color=RED, lw=1, ls="--")
            if hlab:
                a.text(-0.4, (np.log10(hline) if log else hline) - 0.15, hlab, color=RED, fontsize=7,
                       ha="left", va="top")

    labs = [g[0] for g in groups] + ["routing\nn", "routing\nq"]
    strip(ax[0], [g[1].H_q for g in groups] + [Hnn, Hqq], labs)
    ax[0].set_ylim(-7.5, 3.5)
    ax[0].set_ylabel("log10 curvature at optimum, d2L/dx2 (x in ln units)")
    ax[0].set_title("(a) curvature at the training optimum")
    f2n = 0.5 * Hnn * np.log(2) ** 2 / rt.loss_star
    f2q = 0.5 * Hqq * np.log(2) ** 2 / rt.loss_star
    strip(ax[1], [g[1].f2_rel for g in groups] + [f2n, f2q], labs, hline=0.01, hlab="flat below (routing census rule)")
    ax[1].set_ylabel("log10 loss rise for a factor-2 move / L*")
    ax[1].set_title("(b) factor-2 flatness")
    b = [groups[0][1]["band_tr_0.02_u"], groups[2][1]["band_tr_0.02_u"], groups[3][1]["band_tr_0.02_u"],
         2 * np.sqrt(2 * 0.02 / np.maximum(Hnn, 1e-12)) / LN10, 2 * np.sqrt(2 * 0.02 / np.maximum(Hqq, 1e-12)) / LN10]
    colors_b = [BLUE, AQUA, YELLOW, INK2, MUTED]
    rng = np.random.default_rng(1)
    for i, (v, c) in enumerate(zip(b, colors_b)):
        v = np.minimum(np.asarray(v, float), 6.0)
        ax[2].scatter(i + rng.uniform(-0.25, 0.25, len(v)), v, s=8, color=c, alpha=0.45, lw=0)
        q = np.percentile(v, [25, 50, 75])
        ax[2].plot([i - 0.32, i + 0.32], [q[1]] * 2, color=INK, lw=2)
        ax[2].plot([i, i], [q[0], q[2]], color=INK, lw=1)
    ax[2].set_xticks(range(5))
    ax[2].set_xticklabels(["T0\n(L4)", "pool z", "pool kc", "routing n\n(quadratic)", "routing q\n(quadratic)"],
                          fontsize=7.5)
    ax[2].set_ylabel("width of the 0.02 NSE band (decades; capped at 6)")
    ax[2].set_title("(c) 0.02 band width, training years")
    ax[2].text(0.02, 0.97, "dam bands measured on the grid:\nT0 4 decades, z 3.4 decades wide", transform=ax[2].transAxes,
               fontsize=7, color=INK2, va="top")
    t0 = groups[0][1]
    amp = groups[1][1]
    rat = [t0.g_ratio, amp.g_ratio, np.abs(ji.pg_z) / np.abs(jm.pg_z), np.abs(ji.pg_kc) / np.abs(jm.pg_kc),
           np.abs(ji.pg_phi) / np.abs(jm.pg_phi)]
    strip(ax[3], rat, [g[0] for g in groups], hline=1.0)
    ax[3].set_ylim(-3.2, 2.0)
    ax[3].axhline(np.log10(0.25), color=MUTED, lw=1, ls=":")
    ax[3].text(4.5, np.log10(0.25), "plateau below 0.25", color=INK2, fontsize=7, ha="right", va="top")
    ax[3].set_ylabel("log10 |gradient at engine init| / |gradient halfway|")
    ax[3].set_title("(d) gradient at the init vs halfway to the optimum")
    fig.suptitle("Figure 2. Dam parameters and routing parameters on one scale (dams: 214 on-reach gauges, pool: 69 "
                 "flood-control; routing: 2,124 well-fit gauges, 5-yr census)", fontsize=10, color=INK)
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    fig.savefig(os.path.join(HERE, "fig2_curvature_bands_by_parameter.png"), dpi=150)
    plt.close(fig)


# ------------------------------------------------------------------ fig 3: the pool plateau, offline and engine
def fig3():
    zp = pd.read_csv(os.path.join(HERE, "zprofile.csv"), dtype={"STAID": str})
    fig, ax = plt.subplots(1, 3, figsize=(15, 4.4))
    # (a) dNSE vs z relative to offline L2 (the fair twin), medians and IQR, + engine points
    a = ax[0]
    l2 = {s: L.RCF.loc[s, "L2_nse"] for s in L.FA69}
    l2tr = {s: L.RCF.loc[s, "L2_train_nse"] for s in L.FA69}
    f = zp[zp["at"] == "fitted"].copy()
    f["dte"] = f.teNSE - f.STAID.map(l2)
    f["dtr"] = f.trNSE - f.STAID.map(l2tr)
    for col, c, lab, ls in (("dtr", BLUE, "offline, train", "-"), ("dte", ORANGE, "offline, test", "--")):
        g = f.groupby("z")[col]
        m, lo, hi = g.median(), g.quantile(0.25), g.quantile(0.75)
        a.plot(m.index, m.values, color=c, lw=2, ls=ls, label=lab)
        a.fill_between(m.index, lo.values, hi.values, color=c, alpha=0.10, lw=0)
    e = pd.read_csv(os.path.join(HERE, "engine_z.csv"), dtype={"STAID": str}).set_index("STAID")
    xs, ys, lo_, hi_ = [], [], [], []
    for zt, col in ((0.05, "z0.05"), (1.0, "z1"), (5.0, "z5"), (20.0, "z20")):
        if col in e:
            d = e[col] - e.twin
            xs.append(zt)
            ys.append(d.median())
            lo_.append(d.quantile(0.25))
            hi_.append(d.quantile(0.75))
    if xs:
        a.errorbar(xs, ys, yerr=[np.array(ys) - lo_, np.array(hi_) - ys], fmt="o", ms=7, color=VIOLET, lw=1.2,
                   capsize=3, label="engine replay, test (median, IQR)")
    if "fitted" in e:
        d = e.fitted - e.twin
        a.axhline(d.median(), color=VIOLET, lw=1, ls=":")
        a.text(0.06, d.median(), "engine, fitted z (median z 17 d)", color=VIOLET, fontsize=7, va="bottom")
    a.axhline(0, color=MUTED, lw=0.8)
    a.set_xscale("log")
    a.set_xlabel("pool size z (days of mean inflow); kc, phi, T0 at the FA fit")
    a.set_ylabel("NSE minus the no-pool bucket (offline L2 / engine twin)")
    a.set_title("(a) skill vs pool size, 69 flood-control dams")
    a.legend(loc="upper left", fontsize=7)
    # (b) pathwise gradient magnitudes vs z
    a = ax[1]
    for at, ls in (("fitted", "-"), ("init", "--")):
        t = zp[zp["at"] == at]
        for col, c, lab in (("pg_z", AQUA, "|dL/d ln z|"), ("pg_kc", YELLOW, "|dL/d ln kc|"),
                            ("pg_phi", MAGENTA, "|dL/d logit phi|")):
            m = t.groupby("z")[col].apply(lambda v: np.median(np.abs(v)))
            a.plot(m.index, np.maximum(m.values, 1e-9), color=c, lw=2, ls=ls,
                   label=f"{lab}, kc/phi {'fitted' if at == 'fitted' else 'at init (3, 0.5)'}")
    zz = np.logspace(np.log10(0.05), np.log10(3), 10)
    a.plot(zz, 2e-3 * zz / 0.05 * 0.1, color=MUTED, lw=1, ls=":")
    a.text(0.3, 2e-3 * 0.3 / 0.05 * 0.1 * 1.5, "slope 1 (gradient ~ z)", color=INK2, fontsize=7, rotation=28)
    a.set_xscale("log")
    a.set_yscale("log")
    a.set_ylim(1e-7, 1e-1)
    a.set_xlabel("pool size z (days of mean inflow)")
    a.set_ylabel("median |pathwise gradient| of the training loss")
    a.set_title("(b) what autodiff sees along z (69 dams)")
    a.legend(fontsize=6.5, loc="lower right")
    # (c) capacity-limited share
    a = ax[2]
    t = zp.copy()
    pr = pd.read_csv(os.path.join(HERE, "pool_regime.csv"), dtype={"STAID": str})
    for at, c, lab in (("fitted_kc_phi", BLUE, "kc, phi fitted"), ("init_kc_phi", ORANGE, "kc 3, phi 0.5 (init)")):
        u = pr[(pr["at"] == at) & pr.z.round(3).isin([0.05, 0.2, 1.0, 5.0, 20.0])]
        g = u.groupby(u.z.round(3))
        a.plot(g.pool_full_share.median().index, g.pool_full_share.median().values, "o-", color=c, lw=2,
               label=f"capture capacity-limited, {lab}")
        a.plot(g.evac_pool_share.median().index, g.evac_pool_share.median().values, "s--", color=c, lw=1.2,
               label=f"evacuation pool-limited, {lab}")
    a.set_xscale("log")
    a.set_ylim(0, 1.02)
    a.set_xlabel("pool size z (days of mean inflow)")
    a.set_ylabel("share of training flood / evacuation days (median)")
    a.set_title("(c) at z = 0.05 d the pool is full on ~all flood days")
    a.legend(fontsize=6.5, loc="upper right")
    fig.suptitle("Figure 3. The flood-pool plateau: skill rises only once z reaches days, while the gradient at z = 0.05 d"
                 " is ~15x weaker and kc, phi carry almost none", fontsize=10, color=INK)
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    fig.savefig(os.path.join(HERE, "fig3_pool_plateau_offline_engine.png"), dpi=150)
    plt.close(fig)


# ------------------------------------------------------------------ fig 4: 2-D planes
def fig4():
    fig, ax = plt.subplots(2, 3, figsize=(13, 8))
    for j, s in enumerate(PICK):
        for i, (w, la, lb) in enumerate((("T0_z", "T0 (d)", "pool z (d)"), ("z_kc", "pool z (d)", "release target kc"))):
            a = ax[i, j]
            k = f"{s}|{w}"
            A, B, tr, te = PL[k + "|a"], PL[k + "|b"], PL[k + "|tr"], PL[k + "|te"]
            Z = tr - tr.max()
            cs = a.contourf(A, B, Z.T, levels=[-1, -0.1, -0.05, -0.02, -0.005, 0.001],
                            colors=["#f4f3f0", "#dbe7f6", "#a9c8ee", "#6ea3e3", "#2a78d6"])
            a.contour(A, B, (te - te.max()).T, levels=[-0.02, -0.005], colors=[ORANGE], linewidths=[0.8, 1.6],
                      linestyles=["--", "-"])
            it, jt_ = np.unravel_index(np.argmax(tr), tr.shape)
            ie, je = np.unravel_index(np.argmax(te), te.shape)
            a.plot(A[it], B[jt_], "x", color=INK, ms=9, mew=2, label="train optimum")
            a.plot(A[ie], B[je], "+", color=ORANGE, ms=11, mew=2, label="test optimum")
            if w == "T0_z":
                a.plot(L.T0_HEAD, 0.05, "o", color=MUTED, ms=7, mfc="none", mew=1.5, label="engine init", clip_on=False, zorder=5)
            else:
                a.plot(0.05, 3.0, "o", color=MUTED, ms=7, mfc="none", mew=1.5, label="engine init", clip_on=False, zorder=5)
            a.set_xscale("log")
            a.set_yscale("log")
            a.set_xlabel(la)
            a.set_ylabel(lb)
            if i == 0:
                a.set_title(f"{s} {NAMES[s]}")
            if i == 0 and j == 0:
                a.legend(fontsize=7, loc="lower left")
    cb = fig.colorbar(cs, ax=ax, shrink=0.6, pad=0.01)
    cb.set_label("training NSE minus its max")
    fig.suptitle("Figure 4. Two-dimensional training landscapes at the three gauges (blue fill); orange contours: test "
                 "NSE 0.005 (solid) and 0.02 (dashed) below its max", fontsize=10, color=INK)
    fig.savefig(os.path.join(HERE, "fig4_planes_T0z_zkc.png"), dpi=150, bbox_inches="tight")
    plt.close(fig)


# ------------------------------------------------------------------ fig 5: why the pool does not open from off
def fig5():
    from read_dams_mpk import tensors
    feat = pd.read_csv(os.path.join(L.AG, "experiments/reservoir/release_head/dam_features.csv"), usecols=["COMID"])
    fa = pd.read_csv(os.path.join(L.AG, "experiments/reservoir/smoke/fixed_FA.csv"), dtype={"STAID": str})
    row = {c: i for i, c in enumerate(feat.COMID)}
    q = 0.05 / 119.95

    def z_of(path):
        rz = tensors(path)["pool"][:, 2]
        return {s: 120 / (1 + np.exp(-(4 * rz[row[c]] + np.log(q)))) for s, c in zip(fa.STAID, fa.COMID)}

    runs = os.path.join(L.AG, ".ddrs/runs")
    z0 = z_of(os.path.join(runs, "2026-09-30T02-46-23Z-train-and-test/checkpoints/epoch_50_mb_3/release_dams.mpk"))
    z3 = z_of(os.path.join(runs, "2026-09-30T01-53-15Z-train-and-test/checkpoints/epoch_13_mb_0/release_dams.mpk"))
    zfit = dict(zip(fa.STAID, fa.z_offline_days))
    yg = pd.read_csv(os.path.join(HERE, "year_grad.csv"), dtype={"STAID": str})
    share = yg[yg.point == "init"].groupby("STAID").pg_z.agg(lambda v: float((v < 0).mean()))
    jt = pd.read_csv(os.path.join(HERE, "joint_init.csv"), dtype={"STAID": str})
    fig, ax = plt.subplots(1, 3, figsize=(15, 4.4))
    a = ax[0]
    for d, c, lab, ls in ((zfit, INK2, "offline training optimum (fitted z)", "-"),
                          (z0, AQUA, "engine-trained, per-dam L2 0, 200 steps", "-"),
                          (z3, RED, "engine-trained, per-dam L2 1e-3 (epoch 13)", "--")):
        v = np.sort(np.asarray(list(d.values()), float))
        a.step(v, np.arange(1, len(v) + 1) / len(v), where="post", color=c, lw=2, ls=ls, label=lab)
    a.axvline(0.05, color=MUTED, lw=1, ls=":")
    a.text(0.055, 0.03, "init 0.05 d", color=INK2, fontsize=7)
    a.set_xscale("log")
    a.set_xlabel("pool size z (days of mean inflow)")
    a.set_ylabel("cumulative share of the 69 flood-control dams")
    a.set_title("(a) where z ends up")
    a.legend(fontsize=7, loc="center right")
    a = ax[1]
    st = [s for s in fa.STAID if s in share.index]
    x = share.loc[st].values + np.random.default_rng(0).uniform(-0.015, 0.015, len(st))
    y = np.log10(np.array([z0[s] for s in st]) / 0.05)
    a.scatter(x, y, s=16, color=AQUA, alpha=0.7, lw=0)
    rho = pd.Series(share.loc[st].values).corr(pd.Series(y), method="spearman")
    a.axhline(0, color=MUTED, lw=0.8)
    a.set_xlabel("share of training years whose init gradient wants a larger pool (offline)")
    a.set_ylabel("log10(engine-trained z / 0.05 d), per-dam L2 0")
    a.set_title(f"(b) the engine moved z where the sign was consistent (Spearman {rho:.2f})")
    a = ax[2]
    G = 250.0
    lam = np.logspace(-7, -2, 60)
    ji = jt[jt.point == "init"]
    jm = jt[jt.point == "mid"]
    for gsrc, c, lab in ((ji, ORANGE, "data gradient at the init (z 0.05 d)"),
                         (jm, BLUE, "data gradient halfway (z ~ 1 d)")):
        g = np.median(np.abs(gsrc.pg_z))
        r_eq = g / (2 * lam * G)
        zeq = 120 / (1 + np.exp(-(4 * r_eq + np.log(q))))
        a.plot(lam, zeq, color=c, lw=2, label=lab)
    a.axvline(1e-3, color=RED, lw=1, ls="--")
    a.text(1.1e-3, 30, "per_dam_l2\n= 1e-3", color=RED, fontsize=7)
    a.axhline(np.median(fa.z_offline_days), color=INK2, lw=1, ls=":")
    a.text(1.2e-7, np.median(fa.z_offline_days) * 1.1, "offline median z* 17 d", color=INK2, fontsize=7)
    a.set_xscale("log")
    a.set_yscale("log")
    a.set_ylim(0.03, 200)
    a.set_xlabel("per-dam L2 weight lambda (G = 250 gauges per step)")
    a.set_ylabel("equilibrium z under L2 (d)")
    a.set_title("(c) L2 balance: z* = 17 d needs lambda < 2e-6 to 3e-5")
    a.legend(fontsize=7, loc="lower left")
    fig.suptitle("Figure 5. Why the pool did not open in training: a weak, sign-noisy gradient on the shelf, a short "
                 "step budget, and an L2 anchor far stronger than the data", fontsize=10, color=INK)
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    fig.savefig(os.path.join(HERE, "fig5_why_pool_stays_shut.png"), dpi=150)
    plt.close(fig)
    return {"z_trained_l2_0_median": float(np.median(list(z0.values()))),
            "z_trained_l2_1e-3_median": float(np.median(list(z3.values()))),
            "z_trained_l2_1e-3_max": float(np.max(list(z3.values()))), "spearman_share_vs_z": float(rho)}


if __name__ == "__main__":
    fig1()
    fig2()
    fig3()
    fig4()
    print(fig5())
    print("figures written")

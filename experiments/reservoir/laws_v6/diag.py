"""Task 1 diagnostics: what the 30-120 d error looks like at flood-control dams (event composites) and what the
seasonal error looks like at irrigation dams (monthly climatology), before designing any law.

Events are picked on the model inflow I (the routed no-dam flow, a model input), never on observations:
independent peaks of I (>= 60 d apart, above the gauge's 95th percentile, at most 2 per water year), WY1983-2010.
Composite window -15..+150 d, flows divided by the gauge's training-year mean inflow Ibar. Stored volume is the
cumulative sum of (I - O) - mean(I - O) from day -15 (days of mean flow), i.e. with the gauge's long-run volume
bias removed, so it measures what the dam holds back during the event and gives back afterwards.
Outputs: diag_events.csv (per gauge), diag.txt, fig1_flood_composite.png, fig2_flood_residual.png,
fig3_irrigation_season.png.
"""
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import v6

D = v6.load()
sg, dam, ctl = v6.gauges()
rcf = pd.read_csv(v6.RCFITS, dtype={"STAID": str}).set_index("STAID")
W0, W1 = -15, 150
LAGS = np.arange(W0, W1 + 1)
ana = (D["wy"] >= 1983) & (D["wy"] <= 2010)


def series(s):
    i = D["ids"].get_loc(s)
    g = v6.prep(D["P"][i], D["O"][i], D["doy"], D["train"], D["test"])
    p = rcf.loc[s]
    q2 = v6.sim(g, T=float(p.L2_p_T0), want=True)["Q"]
    c = [p.L4_p_c1s, p.L4_p_c1c, p.L4_p_c2s, p.L4_p_c2c]
    q4 = v6.sim(g, T=float(p.L4_p_T0), r=v6.l4_flux(g, c), want=True)["Q"]
    return g, q2, q4


def events(I, wy):
    thr = np.nanpercentile(I[ana], 95)
    order = np.argsort(-np.where(ana, I, -np.inf))
    kept, per_wy = [], {}
    for t in order:
        if not ana[t] or I[t] < thr:
            break
        if t + W0 < 0 or t + W1 >= len(I):
            continue
        if any(abs(t - k) < 60 for k in kept) or per_wy.get(wy[t], 0) >= 2:
            continue
        kept.append(t)
        per_wy[wy[t]] = per_wy.get(wy[t], 0) + 1
    return kept


def gauge_events(s):
    g, q2, q4 = series(s)
    I, O, Ib = g["I"], g["O"], g["Ibar"]
    ev = events(I, D["wy"])
    if len(ev) < 5:
        return None
    bias = np.nanmean((I - O)[ana])
    bias2 = np.nanmean((I - q2)[ana])
    mats = {k: [] for k in ("I", "O", "L2", "L4", "stO", "stL2", "res2")}
    pk = []
    for t in ev:
        sl = slice(t + W0, t + W1 + 1)
        o = O[sl]
        if np.isfinite(o).mean() < 0.9:
            continue
        o = pd.Series(o).interpolate(limit_direction="both").values
        mats["I"].append(I[sl] / Ib)
        mats["O"].append(o / Ib)
        mats["L2"].append(q2[sl] / Ib)
        mats["L4"].append(q4[sl] / Ib)
        mats["stO"].append(np.cumsum(I[sl] - o - bias) / Ib)
        mats["stL2"].append(np.cumsum(I[sl] - q2[sl] - bias2) / Ib)
        mats["res2"].append((o - q2[sl]) / Ib)
        w = slice(-W0 - 5, -W0 + 6)
        pk.append((I[sl][w].max(), o[w].max(), q2[sl][w].max(), int(np.argmax(o[w])) - 5))
    if len(pk) < 5:
        return None
    m = {k: np.mean(v, axis=0) for k, v in mats.items()}
    pk = np.array(pk)
    stO = m["stO"]
    imax = int(np.argmax(stO))
    half = np.flatnonzero(stO[imax:] <= 0.5 * stO[imax])
    stL = m["stL2"]
    jmax = int(np.argmax(stL))
    halfL = np.flatnonzero(stL[jmax:] <= 0.5 * stL[jmax])
    return dict(STAID=s, n_ev=len(pk), peak_ratio_obs=float(np.median(pk[:, 1] / pk[:, 0])),
                peak_ratio_L2=float(np.median(pk[:, 2] / pk[:, 0])), peak_lag_obs=float(np.median(pk[:, 3])),
                stored_max_obs=float(stO[imax]), stored_day_obs=int(LAGS[imax]),
                evac_half_obs=float(half[0]) if len(half) else np.nan,
                stored_max_L2=float(stL[jmax]), stored_day_L2=int(LAGS[jmax]),
                evac_half_L2=float(halfL[0]) if len(halfL) else np.nan,
                res_post_15_60=float(m["res2"][-W0 + 15:-W0 + 61].mean()),
                res_post_60_120=float(m["res2"][-W0 + 60:-W0 + 121].mean()),
                Ibar=g["Ibar"]), m


def group_rows(ids):
    rows, comp = [], []
    for s in ids:
        r = gauge_events(s)
        if r is None:
            continue
        rows.append(r[0])
        comp.append(r[1])
    return pd.DataFrame(rows).set_index("STAID"), comp


fc = dam[(dam.dam_purpose == "Flood Risk Reduction")]
sets = {
    "flood, on-reach DOR>0.5": fc[fc.on_reach & (fc.nid_dor > 0.5)].index,
    "flood, all DOR>0.5": fc[fc.nid_dor > 0.5].index,
}
ctl_of = ctl.reset_index().set_index("control_for").STAID
out, comps = {}, {}
lines = []
for lab, ids in sets.items():
    a, ca = group_rows(ids)
    b, cb = group_rows([ctl_of[s] for s in ids])
    out[lab], comps[lab] = (a, b), (ca, cb)
    lines.append(f"\n== {lab}: dams n={len(a)}, controls n={len(b)} (medians over gauges of per-gauge event means)")
    for col in ["n_ev", "peak_ratio_obs", "peak_ratio_L2", "peak_lag_obs", "stored_max_obs", "stored_day_obs",
                "evac_half_obs", "stored_max_L2", "stored_day_L2", "evac_half_L2", "res_post_15_60", "res_post_60_120"]:
        lines.append(f"  {col:18s} dam {a[col].median():+8.3f}   control {b[col].median():+8.3f}")

pd.concat({k: pd.concat({"dam": v[0], "control": v[1]}) for k, v in out.items()}).to_csv(v6.HERE + "/diag_events.csv")


# ---------- figure 1: composite hydrographs and stored volume
def med(comp, k):
    return np.median(np.stack([c[k] for c in comp]), axis=0)


def q(comp, k, p):
    return np.percentile(np.stack([c[k] for c in comp]), p, axis=0)


lab = "flood, on-reach DOR>0.5"
ca, cb = comps[lab]
fig, ax = plt.subplots(1, 3, figsize=(15, 4.6))
cols = {"I": "#8a959b", "O": "#1b1b1b", "L2": "#2a78d6", "L4": "#b8651b"}
names = {"I": "no dam (model inflow)", "O": "observed", "L2": "per-dam bucket (L2)", "L4": "rule curve (L4)"}
for j, (comp, ttl) in enumerate([(ca, f"Flood-control dams, on-reach, DOR > 0.5 (n={len(ca)})"),
                                 (cb, f"Their matched controls (n={len(cb)})")]):
    for k in ["I", "O", "L2", "L4"]:
        ax[j].plot(LAGS, med(comp, k), color=cols[k], lw=1.8 if k == "O" else 1.3, label=names[k])
    ax[j].set_title(ttl, fontsize=10, loc="left")
    ax[j].set_xlabel("days from inflow peak")
    ax[j].set_yscale("log")
    ax[j].axvspan(30, 120, color="#f2e6d8", zorder=0)
ax[0].set_ylabel("flow / mean inflow (median over gauges)")
ax[0].legend(fontsize=8, frameon=False)
for comp, c, ls, lb in [(ca, "#1b1b1b", "-", "dams: observed"), (ca, "#2a78d6", "-", "dams: bucket L2"),
                        (cb, "#1b1b1b", "--", "controls: observed"), (cb, "#2a78d6", "--", "controls: bucket L2")]:
    k = "stO" if "observed" in lb else "stL2"
    ax[2].plot(LAGS, med(comp, k), color=c, ls=ls, lw=1.5, label=lb)
ax[2].fill_between(LAGS, q(ca, "stO", 25), q(ca, "stO", 75), color="#1b1b1b", alpha=0.08, lw=0)
ax[2].axhline(0, color="#aaa", lw=0.8)
ax[2].axvspan(30, 120, color="#f2e6d8", zorder=0)
ax[2].set_title("Volume held back (cumulative I - Q, bias removed)", fontsize=10, loc="left")
ax[2].set_xlabel("days from inflow peak")
ax[2].set_ylabel("days of mean inflow")
ax[2].legend(fontsize=8, frameon=False)
fig.tight_layout()
fig.savefig(v6.HERE + "/fig1_flood_composite.png", dpi=120)

# ---------- figure 2: residual after the event
fig, ax = plt.subplots(1, 2, figsize=(11, 4.2))
for j, lab in enumerate(sets):
    ca, cb = comps[lab]
    for comp, c, lb in [(ca, "#b8651b", "dams"), (cb, "#2a78d6", "controls")]:
        ax[j].plot(LAGS, med(comp, "res2"), color=c, lw=1.6, label=f"{lb}: observed - bucket L2 (median)")
        ax[j].fill_between(LAGS, q(comp, "res2", 25), q(comp, "res2", 75), color=c, alpha=0.12, lw=0)
    ax[j].axhline(0, color="#888", lw=0.8)
    ax[j].axvspan(30, 120, color="#f2e6d8", zorder=0)
    ax[j].set_title(f"{lab}: n={len(ca)}", fontsize=10, loc="left")
    ax[j].set_xlabel("days from inflow peak")
    ax[j].set_ylim(-1.5, 1.0)
ax[0].set_ylabel("(observed - L2) / mean inflow")
ax[0].legend(fontsize=8, frameon=False)
fig.tight_layout()
fig.savefig(v6.HERE + "/fig2_flood_residual.png", dpi=120)

# ---------- figure 3: irrigation season
mon = D["t"].month.values
te = D["test"]


def clim_rows(ids):
    rows = []
    for s in ids:
        g, q2, _ = series(s)
        m = te & np.isfinite(g["O"])
        Ib = g["Ibar"]
        f = lambda x: pd.Series(x[m]).groupby(mon[m]).mean().reindex(range(1, 13)).values / Ib
        rows.append(dict(STAID=s, I=f(g["I"]), O=f(g["O"]), L2=f(q2)))
    return rows


groups = {"Irrigation": dam[dam.dam_purpose == "Irrigation"].index,
          "Water Supply": dam[dam.dam_purpose == "Water Supply"].index}
fig, ax = plt.subplots(1, 3, figsize=(15, 4.4))
mlab = list("ONDJFMAMJJAS")
order = [10, 11, 12, 1, 2, 3, 4, 5, 6, 7, 8, 9]
for j, (lab, ids) in enumerate(groups.items()):
    rd, rc_ = clim_rows(ids), clim_rows([ctl_of[s] for s in ids])
    for rows, ls, who in [(rd, "-", "dams"), (rc_, "--", "controls")]:
        for k, c in [("I", "#8a959b"), ("O", "#1b1b1b")]:
            v = np.median(np.stack([r[k] for r in rows]), axis=0)
            ax[j].plot(range(12), v[np.array(order) - 1], color=c, ls=ls, lw=1.5, label=f"{who}: {'no dam' if k == 'I' else 'observed'}")
    ax[j].set_xticks(range(12)); ax[j].set_xticklabels(mlab)
    ax[j].set_title(f"{lab} dams (n={len(rd)}) and matched controls, test years", fontsize=10, loc="left")
    ax[j].set_ylabel("monthly mean / mean inflow (median over gauges)")
    ax[j].legend(fontsize=8, frameon=False)
    # missing volume by month: (I - O)/Ibar, per gauge, median
    miss_d = np.stack([r["I"] - r["O"] for r in rd])
    miss_c = np.stack([r["I"] - r["O"] for r in rc_])
    ax[2].plot(range(12), np.median(miss_d, axis=0)[np.array(order) - 1], lw=1.6, ls="-",
               color="#b8651b" if j == 0 else "#2a78d6", label=f"{lab} dams")
    ax[2].plot(range(12), np.median(miss_c, axis=0)[np.array(order) - 1], lw=1.2, ls="--",
               color="#b8651b" if j == 0 else "#2a78d6", label=f"{lab} controls")
    tot_d = np.median(miss_d.mean(axis=1))
    tot_c = np.median(miss_c.mean(axis=1))
    js = np.array([4, 5, 6, 7, 8]) + 1  # May-Sep month numbers 5..9
    sea_d = np.median(miss_d[:, js - 1].mean(axis=1))
    sea_c = np.median(miss_c[:, js - 1].mean(axis=1))
    lines.append(f"\n== {lab} dams n={len(rd)}: missing volume (I - O)/Ibar, test years, median over gauges: "
                 f"annual {tot_d:+.3f} (controls {tot_c:+.3f}); May-Sep mean {sea_d:+.3f} (controls {sea_c:+.3f})")
    vo = np.median([np.nansum(r["O"]) / np.nansum(r["I"]) for r in rd])
    vc = np.median([np.nansum(r["O"]) / np.nansum(r["I"]) for r in rc_])
    lines.append(f"   observed/no-dam volume, median: dams {vo:.3f}, controls {vc:.3f}")
    for mm in range(12):
        m_ = order[mm]
        lines.append(f"   {mlab[mm]} dam I {np.median([r['I'][m_-1] for r in rd]):.3f} O {np.median([r['O'][m_-1] for r in rd]):.3f}"
                     f" L2 {np.median([r['L2'][m_-1] for r in rd]):.3f} | ctl I {np.median([r['I'][m_-1] for r in rc_]):.3f}"
                     f" O {np.median([r['O'][m_-1] for r in rc_]):.3f}")
ax[2].axhline(0, color="#888", lw=0.8)
ax[2].set_xticks(range(12)); ax[2].set_xticklabels(mlab)
ax[2].set_title("Volume missing at the gauge: (no dam - observed) / mean inflow", fontsize=10, loc="left")
ax[2].legend(fontsize=8, frameon=False)
fig.tight_layout()
fig.savefig(v6.HERE + "/fig3_irrigation_season.png", dpi=120)

open(v6.HERE + "/diag.txt", "w").write("\n".join(lines) + "\n")
print("\n".join(lines))

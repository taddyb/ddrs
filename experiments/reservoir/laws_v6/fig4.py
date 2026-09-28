"""Figure 4: does the flood-pool law reproduce the observed post-flood hold-and-release?
Left: event-composite stored volume (as fig1, flood-control on-reach DOR>0.5 dams): observed, bucket L2, rule curve
L4, flood pool + rule curve FA4. Right: one flood-control dam's test water year with the largest share of flood-pool
use, at the gauge whose FA-vs-L2 test gain is at the group's 75th percentile (a typical helped dam, not the best)."""
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import v6
from diag import events, D, dam, LAGS, W0, W1, ana

rcf = pd.read_csv(v6.RCFITS, dtype={"STAID": str}).set_index("STAID")
F = pd.read_csv(v6.HERE + "/laws_by_gauge.csv", dtype={"STAID": str}).set_index("STAID")
ids = dam[(dam.dam_purpose == "Flood Risk Reduction") & dam.on_reach & (dam.nid_dor > 0.5)].index


def runs(s):
    i = D["ids"].get_loc(s)
    g = v6.prep(D["P"][i], D["O"][i], D["doy"], D["train"], D["test"])
    p, f = rcf.loc[s], F.loc[s]
    r4 = v6.l4_flux(g, [p.L4_p_c1s, p.L4_p_c1c, p.L4_p_c2s, p.L4_p_c2c])
    out = {"L2": v6.sim(g, T=float(p.L2_p_T0), want=True), "L4": v6.sim(g, T=float(p.L4_p_T0), r=r4, want=True)}
    for law, rr in [("FA", None), ("FA4", r4)]:
        kw = v6.flood_kw(g, 0, np.log(f[f"{law}_p_T0"]), np.log(f[f"{law}_p_kc"]), f[f"{law}_p_phi"], np.log(f[f"{law}_p_z"]))
        out[law] = v6.sim(g, r=rr, want=True, **kw)
    return g, out


curves = {k: [] for k in ["O", "L2", "L4", "FA4"]}
for s in ids:
    g, out = runs(s)
    I, O, Ib = g["I"], g["O"], g["Ibar"]
    ev = events(I, D["wy"])
    if len(ev) < 5:
        continue
    acc = {k: [] for k in curves}
    for t in ev:
        sl = slice(t + W0, t + W1 + 1)
        o = O[sl]
        if np.isfinite(o).mean() < 0.9:
            continue
        o = pd.Series(o).interpolate(limit_direction="both").values
        for k in curves:
            q = o if k == "O" else out[k]["Q"][sl]
            full = O if k == "O" else out[k]["Q"]
            bias = np.nanmean((I - full)[ana])
            acc[k].append(np.cumsum(I[sl] - q - bias) / Ib)
    if len(acc["O"]) >= 5:
        for k in curves:
            curves[k].append(np.mean(acc[k], axis=0))

fig, ax = plt.subplots(1, 2, figsize=(14, 4.8), gridspec_kw=dict(width_ratios=[1, 1.5]))
col = {"O": "#1b1b1b", "L2": "#2a78d6", "L4": "#8a959b", "FA4": "#b8651b"}
lab = {"O": "observed", "L2": "per-dam bucket L2", "L4": "rule curve L4", "FA4": "flood pool + rule curve FA4"}
for k in curves:
    ax[0].plot(LAGS, np.median(np.stack(curves[k]), axis=0), color=col[k], lw=1.8 if k in ("O", "FA4") else 1.2, label=lab[k])
ax[0].axhline(0, color="#aaa", lw=0.8)
ax[0].axvspan(30, 120, color="#f2e6d8", zorder=0)
ax[0].set_title(f"Volume held back after inflow peaks, flood-control on-reach DOR>0.5 (n={len(curves['O'])})", fontsize=10, loc="left")
ax[0].set_xlabel("days from inflow peak")
ax[0].set_ylabel("days of mean inflow (median over dams)")
ax[0].legend(fontsize=8, frameon=False)

gain = (F.loc[ids, "FA_nse"] - F.loc[ids, "L2_nse"])
s = (gain - gain.quantile(0.75)).abs().idxmin()
g, out = runs(s)
wy = D["wy"]
use = pd.Series(out["FA"]["C"]).groupby(wy).sum()
use = use[(use.index >= 1996) & (use.index <= 2010)]
y = int(use.idxmax())
m = wy == y
x = D["t"][m]
ax[1].plot(x, g["I"][m], color="#8a959b", lw=1.0, label="no dam")
ax[1].plot(x, g["O"][m], color="#1b1b1b", lw=1.6, label="observed")
ax[1].plot(x, out["L2"]["Q"][m], color="#2a78d6", lw=1.1, label=f"bucket L2 (T0 {rcf.loc[s, 'L2_p_T0']:.2f} d)")
ax[1].plot(x, out["FA"]["Q"][m], color="#b8651b", lw=1.4,
           label=f"flood pool FA (Qc {F.loc[s, 'FA_p_kc']:.2f} x mean, phi {F.loc[s, 'FA_p_phi']:.2f}, Fmax {F.loc[s, 'FA_p_z']:.0f} d)")
ax2 = ax[1].twinx()
ax2.fill_between(x, 0, out["FA"]["F"][m] / g["Ibar"], color="#b8651b", alpha=0.12, lw=0)
ax2.set_ylabel("flood pool F (days of mean inflow)", color="#b8651b")
ax[1].set_ylabel("m3/s")
nm_ = dam.loc[s, "dam_name"]
ax[1].set_title(f"{s} below {nm_} (DOR {dam.loc[s, 'nid_dor']:.2f}), WY{y}; test NSE L2 {F.loc[s, 'L2_nse']:.3f}, "
                f"FA {F.loc[s, 'FA_nse']:.3f}", fontsize=10, loc="left")
ax[1].legend(fontsize=7, frameon=True, framealpha=0.9, loc="upper left")
fig.tight_layout()
fig.savefig(v6.HERE + "/fig4_flood_law.png", dpi=120)
print("example gauge", s, nm_, "WY", y, "gain FA-L2", round(gain[s], 4))

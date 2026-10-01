"""Per-gauge classification of each dam parameter, and the T0 split by its optimum.

  sloppy          a factor-2 move from the training optimum costs < 1 % of L* (the routing census's flatness rule)
  plateau-at-init not sloppy, and the gradient at the engine init is < 25 % of the gradient halfway to the optimum
                  (pool parameters: the JOINT engine init, all pool parameters off at once; T0 and amplitude: 1-D)
  stiff           neither
Also: 'path ratio' = |g(init)| / max |g| on the grid between the init and the optimum (robustness check), and a
window-stability flag (optimum moves between WY1983-89 and WY1990-95 by more than half the 0.005 band).
Writes classification.csv and classification.txt."""
import os

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
S = np.load(os.path.join(HERE, "slices.npz"))
df = pd.read_csv(os.path.join(HERE, "per_gauge_slices_derived.csv"), dtype={"STAID": str})
jt = pd.read_csv(os.path.join(HERE, "joint_init.csv"), dtype={"STAID": str})
ji, jm = jt[jt.point == "init"].set_index("STAID"), jt[jt.point == "mid"].set_index("STAID")


def path_ratio(r):
    k = f"{r.STAID}|{r.law}|{r.param}"
    x, Lg = S[k + "|x"], S[k + "|L"]
    g = np.gradient(Lg, x)
    lo, hi = sorted([r.x_init, r.x_opt])
    m = (x >= lo) & (x <= hi)
    if m.sum() < 2:
        return np.nan
    gi = np.interp(np.clip(r.x_init, x[0], x[-1]), x, g)
    return abs(gi) / np.abs(g[m]).max()


df["path_ratio"] = df.apply(path_ratio, axis=1)
jr = {}
for p, c in (("z", "pg_z"), ("kc", "pg_kc"), ("phi", "pg_phi")):
    jr[p] = (np.abs(ji[c]) / np.abs(jm[c])).to_dict()
df["init_ratio"] = [jr[r.param][r.STAID] if (r.law == "FA" and r.param in jr) else r.g_ratio for r in df.itertuples()]
df["cls"] = np.where(df.f2_rel < 0.01, "sloppy", np.where(df.init_ratio < 0.25, "plateau-at-init", "stiff"))
df.to_csv(os.path.join(HERE, "classification.csv"), index=False)
lines = ["class shares per parameter (sloppy / plateau-at-init / stiff), window-unstable share, path ratio median"]
for k in ["L2:T0", "L4:T0", "L4:amp", "FA:T0", "FA:z", "FA:kc", "FA:phi"]:
    s = df[df.key == k]
    c = s.cls.value_counts(normalize=True)
    lines.append(f"  {k:7s} n={len(s):3d}  sloppy {c.get('sloppy', 0):.0%}  plateau {c.get('plateau-at-init', 0):.0%}  "
                 f"stiff {c.get('stiff', 0):.0%}   window-unstable {s.half_unstable.mean():.0%}   path ratio "
                 f"{s.path_ratio.median():.3f}; init ratio {s.init_ratio.median():.3f}")
lines.append("\nT0 (L4) split by its training optimum T0*")
t = df[df.key == "L4:T0"].copy()
t["bin"] = pd.cut(t.v_opt, [0, 0.5, 2, 1e9], labels=["T0* < 0.5 d", "0.5-2 d", "> 2 d"])
for b, u in t.groupby("bin", observed=True):
    c = u.cls.value_counts(normalize=True)
    lines.append(f"  {b:12s} n={len(u):3d}  H {u.H_q.median():.4f}  factor-2 dL/L* {u.f2_rel.median():.4f}  0.02 band "
                 f"{u['band_tr_0.02_u'].median():.2f} dec (open {u['band_tr_0.02_open'].mean():.0%})  sloppy "
                 f"{c.get('sloppy', 0):.0%} plateau {c.get('plateau-at-init', 0):.0%} stiff {c.get('stiff', 0):.0%}  "
                 f"init in 0.005 band {u['init_in_tr_0.005'].mean():.0%}  path ratio {u.path_ratio.median():.3f}  "
                 f"flood-control {(u.dam_purpose == 'Flood Risk Reduction').mean():.0%}  median DOR {u.nid_dor.median():.2f}")
lines.append("\nclass shares by DOR bin and purpose (L4:T0, L4:amp, FA:z, FA:kc)")
df["dor_bin"] = pd.cut(df.nid_dor, [0, 0.5, 1e9], labels=["DOR<0.5", "DOR>0.5"])
for k in ["L4:T0", "L4:amp", "FA:z", "FA:kc"]:
    for col in ("dor_bin", "fc"):
        for lev, u in df[df.key == k].groupby(col, observed=True):
            c = u.cls.value_counts(normalize=True)
            lines.append(f"  {k:7s} {str(lev):8s} n={len(u):3d} sloppy {c.get('sloppy', 0):.0%} plateau "
                         f"{c.get('plateau-at-init', 0):.0%} stiff {c.get('stiff', 0):.0%}")
txt = "\n".join(lines)
print(txt)
open(os.path.join(HERE, "classification.txt"), "w").write(txt + "\n")

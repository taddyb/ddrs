"""Task 4 (i)/(ii): the combined per-dam law set on the smoke set, paired, with controls receiving the same law as
their dam (flood pool + rule curve + bucket where the dam is flood-control, rule curve + bucket elsewhere).
Also: a per-gauge runoff scalar fitted on training years at EVERY population gauge (smoke gauges direct; the other
1,449 gauges imputed from smoke gauges of the same dammed/undammed class and DOR bin), as the strongest allowed
form of runoff rescaling. Writes lawset.txt."""
import numpy as np
import pandas as pd

import v6
from summarise import boot, fmt, GROUPS, ctl_of, dam, F

rng = np.random.default_rng(2)
isfc_dam = dam.dam_purpose == "Flood Risk Reduction"
fc = {**isfc_dam.to_dict(), **{ctl_of[s]: v for s, v in isfc_dam.items()}}
for m in ["nse", "kge", "r", "alpha", "beta"] + [f"band_{b}" for b in v6.BAND_LAB]:
    F[f"SF_{m}"] = np.where(F.index.map(fc).astype(bool), F[f"FA_{m}"], F[f"L2_{m}"])
    F[f"SF4_{m}"] = np.where(F.index.map(fc).astype(bool), F[f"FA4_{m}"], F[f"L4_{m}"])
    F[f"SC4_{m}"] = np.where(F.index.map(fc).astype(bool), F[f"FC4_{m}"], F[f"L4_{m}"])

lines = []
for gname in ["all on-reach DOR>0.5", "flood-control on-reach DOR>0.5", "flood-control, all", "all 458 dams"]:
    ids = [s for s in GROUPS[gname] if s in F.index]
    c = [ctl_of[s] for s in ids]
    lines.append(f"\n== {gname} (n={len(ids)})")
    for law, ref in [("L2", "L0"), ("L4", "L0"), ("SF", "L0"), ("SF4", "L0"), ("SC4", "L0"), ("SF", "L2"), ("SF4", "L2"),
                     ("SF4", "L4")]:
        a = F.loc[ids, f"{law}_nse"].values - F.loc[ids, f"{ref}_nse"].values
        b = F.loc[c, f"{law}_nse"].values - F.loc[c, f"{ref}_nse"].values
        k = F.loc[ids, f"{law}_kge"].values - F.loc[ids, f"{ref}_kge"].values
        dd = {m: np.median(F.loc[ids, f"{law}_{m}"].values - F.loc[ids, f"{ref}_{m}"].values) for m in ["r", "alpha", "beta"]}
        cm = (np.maximum(F.loc[ids, f"{law}_nse"], -1) - np.maximum(F.loc[ids, f"{ref}_nse"], -1)).mean()
        lines.append(f"  {law:4s} vs {ref:3s} {fmt(boot(a))} {int((a > 0).sum())}/{int((a < 0).sum())} cmean {cm:+.4f} | "
                     f"ctl {fmt(boot(b))} | DiD {fmt(boot(a - b))} | dKGE {fmt(boot(k))} dr {dd['r']:+.4f} "
                     f"dalpha {dd['alpha']:+.4f} dbeta {dd['beta']:+.4f}")
    lines.append("  mean band MSE (lt7 7-30 30-120 120-400 gt400) dam | ctl:")
    for law in ["L0", "L2", "L4", "SF", "SF4", "SC4"]:
        lines.append(f"    {law:4s} " + " ".join(f"{F.loc[ids, f'{law}_band_{b}'].mean():.3f}" for b in v6.BAND_LAB) + " | " +
                     " ".join(f"{F.loc[c, f'{law}_band_{b}'].mean():.3f}" for b in v6.BAND_LAB))
    lines.append(f"  median NSE: dam L0 {F.loc[ids, 'L0_nse'].median():.3f} SF4 {F.loc[ids, 'SF4_nse'].median():.3f}; "
                 f"ctl L0 {F.loc[c, 'L0_nse'].median():.3f} SF4 {F.loc[c, 'SF4_nse'].median():.3f}")

# per-gauge scalar (training-year fit) at every population gauge
pop = pd.read_csv(v6.POP, dtype={"STAID": str}).set_index("STAID")
base = pop.nse_off
BM = base.median()
gk = (F.Kg_nse - F.L0_nse)
dammed = pop.n_nid_ge10mcm > 0
dor = pop.nid_dor.fillna(0)
bins = pd.cut(dor, [-1, 0.1, 0.5, 1, 2, 1e9], labels=False)
sm = gk.index
cls_sm = dammed[sm]
new_med, draws = base.copy(), []
new_med[sm] += gk
rest = pop.index[~pop.index.isin(sm)]
for s in rest:
    pool = gk[(cls_sm == dammed[s]) & (bins[sm] == bins[s])] if dammed[s] else gk[~cls_sm]
    new_med[s] += pool.median()
for _ in range(200):
    nd = base.copy()
    nd[sm] += gk
    for cl in [True, False]:
        for b in range(5):
            tgt = rest[(dammed[rest] == cl) & ((bins[rest] == b) if cl else True)]
            if cl:
                pool = gk[(cls_sm == cl) & (bins[sm] == b)].values
            else:
                pool = gk[~cls_sm].values
                if b > 0:
                    continue
            nd[tgt] += pool[rng.integers(0, len(pool), len(tgt))]
    draws.append(nd.median() - BM)
lines.append(f"\n== per-gauge runoff scalar fitted on training years at every gauge (smoke 916 direct, other 1,449 imputed)")
lines.append(f"  smoke gains Kg - L0 median: dam gauges {gk[cls_sm].median():+.4f}, undammed controls {gk[~cls_sm].median():+.4f}")
lines.append(f"  population median change: cell-median imputation {new_med.median() - BM:+.4f}; random draws "
             f"{np.mean(draws):+.4f} [{np.percentile(draws, 2.5):+.4f},{np.percentile(draws, 97.5):+.4f}]")
open(v6.HERE + "/lawset.txt", "w").write("\n".join(lines) + "\n")
print("\n".join(lines))

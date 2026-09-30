"""Lever 1: runoff volume multipliers, fitted on TRAINING years (WY1982-1995), scored on TEST years (WY1996-2010).

Usage: volume.py <seed> <replay run id>
  training-year predictions: the zero-step replay of the seed's no-dam checkpoint (testing window 1981-10-01..1996-09-30);
  test-year predictions: the source run's own test phase (exactly what the run scored).
Checks the replay against the source run on the WY1996 overlap first.

Multipliers are applied to the routed discharge (post hoc). In the engine a Q' multiplier would also shift celerity a
little (routing is nonlinear in Q through depth), so these are first-order estimates.

Writes volume_s<seed>.txt / .json and volume_by_gauge_s<seed>.csv.
"""
import json
import sys

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import RidgeCV
from sklearn.model_selection import GroupKFold, KFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

import common as C

seed, replay = int(sys.argv[1]), sys.argv[2]
src = {42: C.S42, 43: C.S43}[seed]
out, res = [], {}
say = out.append

ids, tt, PT, OT = C.load_preds(src)
rid, tr, PR, OR = C.load_preds(replay)
PR, OR = C.align(rid, PR, ids), C.align(rid, OR, ids)
wy_t, wy_r = C.water_year(tt), C.water_year(tr)

# ---- replay check on the WY1996 overlap (skip October 1995: the source run cold-started on 1995-10-01) ----
ov = pd.Timestamp("1995-11-01")
tpos = {d: k for k, d in enumerate(tr)}
i_t = np.array([k for k in np.where((wy_t == 1996) & (tt >= ov))[0] if tt[k] in tpos])
i_r = np.array([tpos[tt[k]] for k in i_t])
a, b = PT[:, i_t], PR[:, i_r]
rel = np.nanmax(np.abs(a - b) / np.maximum(np.abs(a), 1e-3), axis=1)
nse_ov_t = C.metrics(a, OT[:, i_t], min_days=200).nse
nse_ov_r = C.metrics(b, OR[:, i_r], min_days=200).nse
say(f"== seed {seed}: source {src}, replay {replay}")
say(f"replay check, WY1996 from 1995-11-01 ({len(i_t)} days): per-gauge max relative difference median {np.nanmedian(rel):.2e}, "
    f"p99 {np.nanpercentile(rel, 99):.2e}, max {np.nanmax(rel):.2e}; |NSE diff| max {np.nanmax(np.abs(nse_ov_t - nse_ov_r)):.2e}; "
    f"obs identical {np.array_equal(np.nan_to_num(OT[:, i_t], nan=-9), np.nan_to_num(OR[:, i_r], nan=-9))}")
res["replay_check"] = dict(rel_median=float(np.nanmedian(rel)), rel_p99=float(np.nanpercentile(rel, 99)),
                           rel_max=float(np.nanmax(rel)), nse_absdiff_max=float(np.nanmax(np.abs(nse_ov_t - nse_ov_r))))

# ---- windows ----
import os
TWY = [int(x) for x in os.environ.get("LEVERS_TRAIN_WY", "1982,1995").split(",")]  # smoke-test override only
trn = (wy_r >= TWY[0]) & (wy_r <= TWY[1]) & (tr >= pd.Timestamp("1981-11-01"))  # drop the replay's cold-start month
tst = (wy_t >= 1996) & (wy_t <= 2010)  # the source run's whole test phase
Ptr, Otr = PR[:, trn], OR[:, trn]
Pte, Ote = PT[:, tst], OT[:, tst]
say(f"training window {tr[trn][0].date()}..{tr[trn][-1].date()} ({trn.sum()} d); test {tt[tst][0].date()}..{tt[tst][-1].date()} "
    f"({tst.sum()} d)")

Mtr = C.metrics(Ptr, Otr)
Mte = C.metrics(Pte, Ote)
base_n, base_k = Mte.nse.values, Mte.kge.values
BASE_N, BASE_K = np.nanmedian(base_n), np.nanmedian(base_k)
say(f"test-year median NSE {BASE_N:.4f} KGE {BASE_K:.4f} (n={np.isfinite(base_n).sum()}); training-year median NSE "
    f"{np.nanmedian(Mtr.nse):.4f} KGE {np.nanmedian(Mtr.kge):.4f} (n={np.isfinite(Mtr.nse).sum()} with >= 365 valid days)")

ktr = C.ls_scale(Ptr, Otr)          # NSE-optimal per-gauge scalar, training years
vtr = C.vol_ratio(Ptr, Otr)         # observed / predicted volume, training years
kte = C.ls_scale(Pte, Ote)
vte = C.vol_ratio(Pte, Ote)
ok_tr = np.isfinite(Mtr.nse.values) & np.isfinite(ktr) & (ktr > 0)
say(f"predicted/observed volume (1/v): training median {np.nanmedian(1 / vtr[ok_tr]):.4f} (share > 1: "
    f"{np.mean(1 / vtr[ok_tr] > 1):.3f}); test median {np.nanmedian(1 / vte):.4f} (share > 1: {np.nanmean(1 / vte > 1):.3f})")
say(f"per-gauge LS scalar k*: training median {np.nanmedian(ktr[ok_tr]):.4f}, test median {np.nanmedian(kte):.4f}; "
    f"Spearman(train k*, test k*) {pd.Series(ktr).corr(pd.Series(kte), method='spearman'):.3f}; "
    f"Spearman(train vol ratio, test vol ratio) {pd.Series(vtr).corr(pd.Series(vte), method='spearman'):.3f}")

G = pd.read_csv(C.HERE / "gauges_table.csv", dtype={"STAID": str, "HUC02": str}).set_index("STAID").reindex(ids)
huc = G.HUC02.values


def score(mult, name):
    """Apply per-gauge multipliers (array or scalar) to test-year predictions; summarize NSE and KGE."""
    m = np.broadcast_to(np.asarray(mult, dtype=float), (len(ids),)).copy()
    m[~np.isfinite(m)] = 1.0
    M = C.metrics(Pte * m[:, None], Ote)
    s = C.summarize(name, M.nse.values, base_n)
    sk = C.summarize(name, M.kge.values, base_k, "KGE")
    say(C.fmt(s))
    say(C.fmt(sk))
    res[name] = dict(nse=s, kge=sk, mult_median=float(np.median(m)), mult_p10=float(np.percentile(m, 10)),
                     mult_p90=float(np.percentile(m, 90)))
    return M


GRID = np.round(np.arange(0.80, 1.1001, 0.0025), 4)


def best_median_scalar(P, O, rows):
    """Scalar maximizing the median training-year NSE over `rows` (grid)."""
    best, arg = -np.inf, 1.0
    for k in GRID:
        v = np.nanmedian(C.metrics(P[rows] * k, O[rows]).nse.values)
        if v > best:
            best, arg = v, k
    return arg, best


say("\n-- global scalars (fit on training years)")
k_med, v_med = best_median_scalar(Ptr, Otr, np.arange(len(ids)))
say(f"   scalar maximizing training median NSE: {k_med} (training median {v_med:.4f} vs {np.nanmedian(Mtr.nse):.4f})")
score(k_med, f"global scalar {k_med} (max training median NSE)")
k_ls = float(np.nanmedian(ktr[ok_tr]))
score(k_ls, f"global scalar {k_ls:.4f} (median per-gauge LS k*)")
k_v = float(np.nanmedian(vtr[ok_tr]))
score(k_v, f"global scalar {k_v:.4f} (median volume ratio)")
score(0.95, "fixed 0.95 (reference; chosen on test years earlier)")

say("\n-- per-HUC2 scalars (fit on training years within each HUC2)")
m_h_med, m_h_ls, m_h_v = np.ones(len(ids)), np.ones(len(ids)), np.ones(len(ids))
hk = {}
for h in sorted(set(huc)):
    rows = np.where(huc == h)[0]
    kk, _ = best_median_scalar(Ptr, Otr, rows)
    r_ok = rows[ok_tr[rows]]
    m_h_med[rows] = kk
    m_h_ls[rows] = np.nanmedian(ktr[r_ok]) if len(r_ok) else 1.0
    m_h_v[rows] = np.nanmedian(vtr[r_ok]) if len(r_ok) else 1.0
    hk[h] = dict(n=len(rows), k_med=float(kk), k_ls=float(m_h_ls[rows][0]), k_v=float(m_h_v[rows][0]))
say("   " + " ".join(f"{h}:{v['k_ls']:.3f}(n{v['n']})" for h, v in hk.items()))
res["huc2_scalars"] = hk
score(m_h_med, "per-HUC2 scalar (max training median NSE in the HUC2)")
score(m_h_ls, "per-HUC2 scalar (median per-gauge LS k* in the HUC2)")
score(m_h_v, "per-HUC2 scalar (median volume ratio in the HUC2)")

say("\n-- attribute-conditioned multiplier: predict log training k* from catchment attributes, out-of-fold by gauge")
FEAT = ["up_SoilGrids1km_clay", "up_aridity", "up_meanelevation", "up_meanP", "up_NDVI", "up_meanslope", "out_log10_uparea",
        "up_SoilGrids1km_sand", "up_ETPOT_Hargr", "up_Porosity", "up_snow_fraction", "up_snowfall_fraction",
        "up_seasonality_P", "up_meanTa", "up_permeability", "LAT_GAGE", "LNG_GAGE"]
X = G[FEAT].values.astype(float)
X = np.where(np.isfinite(X), X, np.nanmedian(X, axis=0))
fit_ok = ok_tr & np.isfinite(ktr) & (ktr > 0.2) & (ktr < 5)
say(f"   gauges usable for fitting: {fit_ok.sum()} (training NSE finite, 0.2 < k* < 5); features: {len(FEAT)}")


def oof(target, model_fn, splitter, groups=None, clip=(0.7, 1.3)):
    """Out-of-fold predictions for every gauge; fit rows exclude the held-out fold. Gauges not usable for fitting still
    receive a prediction from the fold that holds them out."""
    y = np.log(target)
    pred = np.full(len(ids), np.nan)
    idx = np.arange(len(ids))
    for tr_i, te_i in splitter.split(idx, groups=groups):
        fit = tr_i[fit_ok[tr_i]]
        mdl = model_fn()
        mdl.fit(X[fit], y[fit])
        pred[te_i] = mdl.predict(X[te_i])
    r2 = 1 - np.nanmean((pred[fit_ok] - y[fit_ok]) ** 2) / np.nanvar(y[fit_ok])
    return np.clip(np.exp(pred), *clip), r2


ridge = lambda: make_pipeline(StandardScaler(), RidgeCV(alphas=np.logspace(-2, 4, 25)))
gbm = lambda: HistGradientBoostingRegressor(max_iter=300, learning_rate=0.05, max_leaf_nodes=15, min_samples_leaf=40,
                                            l2_regularization=1.0, random_state=0)
kf = KFold(10, shuffle=True, random_state=0)
gk = GroupKFold(n_splits=len(set(huc)))
for tname, target in [("LS k*", ktr), ("volume ratio", vtr)]:
    for mname, fn in [("ridge", ridge), ("GBM", gbm)]:
        for sname, sp, grp in [("10-fold by gauge", kf, None), ("leave-one-HUC2-out", gk, huc)]:
            mult, r2 = oof(target, fn, sp, grp)
            say(f"   target {tname}, {mname}, {sname}: out-of-fold R2 of log target {r2:.3f}")
            score(mult, f"attr {mname} on log {tname}, {sname}")
            if tname == "LS k*" and mname == "GBM" and sname == "10-fold by gauge":
                res["_mult_attr_gbm"] = mult.tolist()

say("\n-- per-gauge scalars (gauged calibration; the ceiling of any regionalized multiplier)")
m_g = np.where(fit_ok, ktr, 1.0)
score(m_g, "per-gauge LS k* fitted on training years")
m_gv = np.where(fit_ok, vtr, 1.0)
score(m_gv, "per-gauge volume ratio fitted on training years")
# shrink the per-gauge scalar halfway to 1 (a common hedge against training-year noise)
score(np.sqrt(m_g), "per-gauge LS k*, shrunk halfway in log space")
score(np.where(np.isfinite(kte), kte, 1.0), "per-gauge LS k* fitted ON TEST YEARS (in-sample oracle)")

# which gauges move the median, for the best honest regional scheme
T = pd.DataFrame(dict(nse_test=base_n, kge_test=base_k, nse_train=Mtr.nse.values, k_train=ktr, v_train=vtr, k_test=kte,
                      v_test=vte), index=ids)
T.to_csv(C.HERE / f"volume_by_gauge_s{seed}.csv")
open(C.HERE / f"volume_s{seed}.txt", "w").write("\n".join(out) + "\n")
json.dump({k: v for k, v in res.items() if not k.startswith("_")}, open(C.HERE / f"volume_s{seed}.json", "w"), indent=1,
          default=float)
np.save(C.HERE / f"mult_attr_gbm_s{seed}.npy", np.array(res.get("_mult_attr_gbm", [])))
print("\n".join(out))

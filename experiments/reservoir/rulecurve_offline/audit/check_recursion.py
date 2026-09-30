"""Item 2: C (shipped libbucket.so and a fresh compile of bucket.c) vs my pure-Python loop, and re-scoring
their fitted parameters with my simulator to reproduce fits_by_gauge.csv NSEs."""
import ctypes
import numpy as np
import pandas as pd
import audit_impl as A

R = '/home/tbindas/.claude/jobs/dacd6d8c/tmp/rulecurve'
libs = {}
for lab, path in [("shipped", f"{R}/libbucket.so"), ("fresh", f"{A.OUT}/libbucket_audit.so")]:
    L = ctypes.CDLL(path)
    L.bucket.restype = ctypes.c_double
    L.bucket.argtypes = [ctypes.c_void_p] * 2 + [ctypes.c_int] * 2 + [ctypes.c_void_p] * 4
    libs[lab] = L


def cb(L, x, T, mode):
    x = np.ascontiguousarray(x, float); T = np.ascontiguousarray(np.broadcast_to(T, x.shape), float)
    Q = np.empty(len(x)); fl = np.empty(len(x), np.uint8)
    L.bucket(x.ctypes.data, T.ctypes.data, len(x), mode, None, None, Q.ctypes.data, fl.ctypes.data)
    return Q, fl.astype(bool)


def bucket_ec_py(x, T):
    """Engine-clamp variant written from its description: floored day keeps S = T*Q = 0."""
    S = T * x[0]; Q = np.empty(len(x)); fl = np.zeros(len(x), bool)
    for t in range(len(x)):
        avail = S + x[t]; q = avail / (T + 1.0)
        if q < 0: q = 0.0; fl[t] = True; S = 0.0
        else: S = avail - q
        Q[t] = q
    return Q, fl


A.load()
f = pd.read_csv(f'{R}/fits_by_gauge.csv', dtype={'STAID': str}).set_index('STAID')
v = pd.read_csv(f'{R}/fits_variants.csv', dtype={'STAID': str}).set_index('STAID')
sel = pd.read_csv(f'{A.OUT}/sel117.csv', dtype=str).iloc[:, 0].tolist()
pairs = pd.read_csv(f'{A.OUT}/pairs117.csv', dtype=str)
rng = np.random.default_rng(1)
# floored-heavy gauges first, then random, plus two controls
heavy = f.loc[sel].L4_floor_test.sort_values(ascending=False).index[:3].tolist()
pick = heavy + list(rng.choice([s for s in sel if s not in heavy], 5, replace=False)) + pairs.ctl.iloc[:2].tolist()
rows = []
for s in pick:
    k = A.G["ids"][s]; I, O = A.G["P"][k], A.G["O"][k]
    Ibar = float(I[A.G["train"]].mean())
    c = [f.loc[s, f"L4_p_{kk}"] for kk in ["c1s", "c1c", "c2s", "c2c"]]
    T0 = f.loc[s, "L4_p_T0"]
    x = I - A.flux(Ibar, A.G["H"], c)
    Qp, flp = A.bucket_py(x, T0, True)
    out = dict(STAID=s, floor_share_test=float(flp[A.G["test"]].mean()))
    for lab, L in libs.items():
        Qc, flc = cb(L, x, T0, 1)
        out[f"maxabs_py_vs_{lab}_floor"] = float(np.abs(Qp - Qc).max())
        out[f"flags_equal_{lab}"] = bool((flp == flc).all())
    Ql = A.lin_resp(x, T0); Qlp, _ = A.bucket_py(x, T0, False)
    Qlc, _ = cb(libs["shipped"], x, T0, 0)
    out["maxabs_lfilter_vs_loop"] = float(np.abs(Ql - Qlp).max()); out["maxabs_loop_vs_C_nofloor"] = float(np.abs(Qlp - Qlc).max())
    Qe, _ = bucket_ec_py(x, T0); Qec, _ = cb(libs["shipped"], x, T0, 2)
    out["maxabs_ec_py_vs_C"] = float(np.abs(Qe - Qec).max())
    # re-score their parameters with my simulator + my metrics
    out["their_L4_nse"] = f.loc[s, "L4_nse"]; out["my_L4_nse_their_params"] = A.nse(Qp, O, A.G["test"])
    q1 = A.L1_series(I, A.G["w"], f.loc[s, "L1_p_T0"], f.loc[s, "L1_p_a"], f.loc[s, "L1_p_b"], True)
    out["their_L1_nse"] = f.loc[s, "L1_nse"]; out["my_L1_nse_their_params"] = A.nse(q1, O, A.G["test"])
    out["their_L0_nse"] = f.loc[s, "L0_nse"]; out["my_L0_nse"] = A.nse(I, O, A.G["test"])
    # sign check: the flux is subtracted from inflow (x = I - r). Positive r = dam storing water.
    rows.append(out)
df = pd.DataFrame(rows).set_index("STAID")
pd.set_option("display.width", 250); pd.set_option("display.max_columns", 30)
print(df.T.to_string())
df.to_csv(f"{A.OUT}/check_recursion.csv")

# All 916 gauges: re-score their L4 and L1 parameters with my simulator (fast path check of the NSEs)
allrows = []
for s in f.index:
    k = A.G["ids"][s]; I, O = A.G["P"][k], A.G["O"][k]
    Ibar = float(I[A.G["train"]].mean())
    c = [f.loc[s, f"L4_p_{kk}"] for kk in ["c1s", "c1c", "c2s", "c2c"]]
    Qp, _ = A.bucket_py(I - A.flux(Ibar, A.G["H"], c), f.loc[s, "L4_p_T0"], True)
    q1 = A.L1_series(I, A.G["w"], f.loc[s, "L1_p_T0"], f.loc[s, "L1_p_a"], f.loc[s, "L1_p_b"], True)
    allrows.append(dict(STAID=s, dL4=A.nse(Qp, O, A.G["test"]) - f.loc[s, "L4_nse"],
                        dL1=A.nse(q1, O, A.G["test"]) - f.loc[s, "L1_nse"], dL0=A.nse(I, O, A.G["test"]) - f.loc[s, "L0_nse"],
                        dIbar=Ibar - f.loc[s, "Ibar"]))
a = pd.DataFrame(allrows).set_index("STAID")
print("all 916: max |my - their| NSE  L0 %.2e  L1 %.2e  L4 %.2e  Ibar %.2e" % tuple(a.abs().max()))

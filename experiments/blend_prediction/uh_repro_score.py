"""Score the UH fixed-channel reproduction retrain against the existing UH model (2026-09-27T07-29-47Z).
Usage: python uh_repro_score.py <readout dir> ; reads dump/uh_repro_kan_parameters.nc and fields_repro/."""
import sys
from pathlib import Path
import netCDF4
import numpy as np
from scipy.stats import spearmanr

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "levers"))
sys.path.insert(0, "/home/tbindas/projects/ddrs/.claude/skills/ddrs-eval-plots/scripts")
import common as C  # noqa: E402

ro = Path(sys.argv[1])
REPRO = "2026-10-07T22-05-06Z-train-and-test"
a = netCDF4.Dataset(ro / "dump" / "uh_repro_kan_parameters.nc")
b = netCDF4.Dataset("/home/tbindas/projects/ddrs/.ddrs/runs/2026-09-27T07-29-47Z-train-and-test/plot/kan_parameters_epoch50.nc")
ca = np.asarray(a["COMID"][:], dtype=np.int64); cb = np.asarray(b["COMID"][:], dtype=np.int64)
print("COMID order identical:", (ca == cb).all(), len(ca))
m = np.isin(ca, np.load(ro / "fields_repro" / "comids_full_upstream_cover.npy"))
print("set (b) reaches:", m.sum())
for v in ("n", "gamma"):
    x = np.asarray(a[v][:], float); y = np.asarray(b[v][:], float)
    d = np.abs(x - y)
    print(f"{v}: set(b) repro median {np.median(x[m]):.6f} p25 {np.percentile(x[m],25):.6f} p75 {np.percentile(x[m],75):.6f} | "
          f"UH median {np.median(y[m]):.6f} p25 {np.percentile(y[m],25):.6f} p75 {np.percentile(y[m],75):.6f}")
    print(f"   per-reach |diff| on (b): max {d[m].max():.3e} median {np.median(d[m]):.3e} p99 {np.percentile(d[m],99):.3e}; "
          f"all reaches max {d.max():.3e}; Spearman(b) {spearmanr(x[m], y[m])[0]:.6f}; median diff of medians {np.median(x[m])-np.median(y[m]):+.2e}")
ids, tt, P, O = C.load_preds(REPRO)
i0, t0, P0, O0 = C.load_preds(C.S42)
assert (t0 == tt).all()
P0 = C.align(i0, P0, ids)
print("eval gauges", len(ids), "NaN preds", int(np.isnan(P).sum()))
M = C.metrics(P, O); M0 = C.metrics(P0, O)
for k, mm in (("retrain", M), ("existing UH", M0)):
    print(f"{k:12s} NSE median {mm.nse.median():.4f} KGE median {mm.kge.median():.4f}")
dn = (M.nse - M0.nse).abs(); print("per-gauge |dNSE| max", float(dn.max()), "median", float(dn.median()))
print("max |pred diff| m3/s:", float(np.nanmax(np.abs(P - P0))))

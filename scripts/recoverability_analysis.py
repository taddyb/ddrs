#!/usr/bin/env python3
"""Pre-registered verdicts R1-R5 for the synthetic recoverability control.

Spec section 3 of 2026-07-03-synthetic-recoverability-design.md. Bars are
FIXED there; this script only reports.
"""
import re
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr

ROOT = Path("/home/tbindas/projects/ddrs")
OUT = ROOT / "output/recoverability"

plants = pd.read_csv(OUT / "plants.csv")
planted = set(plants["comid"])

def zeta_frame(path, prefix):
    ds = xr.open_dataset(path)
    return pd.DataFrame({
        "comid": ds["COMID_eval"].values,
        f"{prefix}_net": ds["zeta_net"].values,
        f"{prefix}_abs": ds["zeta"].values,
    })

key = zeta_frame(OUT / "answer_key.nc", "key")
base = zeta_frame(OUT / "baseline_zeta.nc", "base")
za = zeta_frame(OUT / "zeta_a.nc", "a")
zc = zeta_frame(OUT / "zeta_c.nc", "c")
df = key.merge(base, on="comid").merge(za, on="comid").merge(zc, on="comid")
df["planted"] = df["comid"].isin(planted)
n_p = int(df["planted"].sum())
assert n_p == len(planted), f"planted rows {n_p} != plan {len(planted)}"

# R1 recovery ratio (planted reaches, run A vs answer key)
p = df[df["planted"]]
ratio = (p["a_net"] / p["key_net"]).values
r1 = float(np.median(ratio))
r1_verdict = "RECOVERED" if r1 >= 0.5 else ("FAILED" if r1 <= 0.1 else "PARTIAL")

# R2 spatial precision (non-planted |zeta_net|, A vs baseline field)
np_a = float(np.median(np.abs(df.loc[~df["planted"], "a_net"])))
np_b = float(np.median(np.abs(df.loc[~df["planted"], "base_net"])))
r2_ratio = np_a / np_b if np_b > 0 else float("inf")
r2_verdict = "PRECISE" if r2_ratio < 2.0 else "SMEARED"

# R3 absorption gap: mean final-epoch loss from student logs
def final_epoch_mean_loss(log):
    text = Path(log).read_text()
    epochs = re.findall(r"^epoch (\d+) ", text, re.M)
    assert epochs, f"no epoch lines in {log}"
    last = epochs[-1]
    seg = text.split(f"epoch {last} ")[-1]
    losses = [float(m) for m in re.findall(r"mb=\d+ loss=([0-9.eE+-]+)", seg)]
    assert losses, f"no losses parsed from {log}"
    return float(np.mean(losses)), len(losses)

loss_a, na = final_epoch_mean_loss(OUT / "logs/student_a.log")
loss_b, nb = final_epoch_mean_loss(OUT / "logs/student_b.log")
rel_gap = (loss_b - loss_a) / loss_b if loss_b else 0.0
r3_verdict = ("A<B (leakance needed)" if rel_gap > 0.05
              else ("A~B (H5 absorption confirmed)" if abs(rel_gap) <= 0.05
                    else "B<A (INVESTIGATE)"))

# R4 absorption map (descriptive): where did B move Manning's n?
po = xr.open_dataset(OUT / "params_orig.nc")
pb = xr.open_dataset(OUT / "params_b.nc")
dn = pd.DataFrame({"comid": po["COMID"].values,
                   "dn": pb["n"].values - po["n"].values})
dn_planted = dn[dn["comid"].isin(planted)]["dn"]
r4 = (f"median dn planted={np.median(dn_planted):.3e} "
      f"all={np.median(dn['dn']):.3e} "
      f"p90|dn| planted={np.percentile(np.abs(dn_planted), 90):.3e}")

# R5 cold emergence
c_p = float(np.median(np.abs(p["c_net"])))
c_np = float(np.median(np.abs(df.loc[~df["planted"], "c_net"])))
r5_ratio = c_p / c_np if c_np > 0 else float("inf")
r5_verdict = "EMERGES" if r5_ratio > 3.0 else "SUPPRESSED"

df.to_csv(OUT / "recovery_rows.csv", index=False)

print("=" * 72)
print("VERDICTS (bars pre-registered in the spec)")
print("=" * 72)
print(f"  [R1 {r1_verdict}] recovery ratio median={r1:.3f} "
      f"(p10={np.percentile(ratio,10):.3f} p90={np.percentile(ratio,90):.3f}, "
      f"n={n_p}; bar: >=0.5 recovered, <=0.1 failed)")
print(f"  [R2 {r2_verdict}] non-planted |zeta_net| A/baseline = {r2_ratio:.2f} "
      f"(A={np_a:.3e}, base={np_b:.3e}; bar: <2)")
print(f"  [R3 {r3_verdict}] final-epoch mean loss A={loss_a:.5f} (n={na}) "
      f"B={loss_b:.5f} (n={nb}) rel gap={rel_gap:+.1%} (bar: 5%)")
print(f"  [R4] {r4}")
print(f"  [R5 {r5_verdict}] cold planted/non-planted |zeta_net| = {r5_ratio:.2f} (bar: >3)")
headline = "PASS" if (r1 >= 0.5 and rel_gap > 0.05) else "FAIL"
print(f"\n  HEADLINE: positive control {headline} "
      f"(requires R1>=0.5 AND A beats B)")
print(f"\nper-reach rows -> {OUT / 'recovery_rows.csv'}")

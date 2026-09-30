"""Fixed dam table that replays the offline flood-pool fits (law FA, laws_v6) in the engine.

laws_v6 (``experiments/reservoir/laws_v6``, report §2 and §5) fitted, per smoke gauge, the flood pool FA on top of
the per-dam bucket, T0 refitted jointly, on the routed no-dam flow at the gauge (daily, WY1983-1995, scored
WY1996-2010): ``laws_by_gauge.csv`` columns ``FA_p_T0`` (days), ``FA_p_kc``, ``FA_p_phi``, ``FA_p_z`` (days) and
``Ibar`` (training-period mean of the routed flow at the gauge, m3/s). This script writes one
``reservoir_release: fixed`` table routed twice with the routing head frozen:

  fixed_FA.csv   COMID, T_days = FA_p_T0, kc, phi, z, inflow_mean_m3s, year_completed (+ provenance columns)

  config/experiments/dam_release_smoke_replay_FA_pool.yaml    params.reservoir_flood_pool: true  (T0 + pool)
  config/experiments/dam_release_smoke_replay_FA_nopool.yaml  params.reservoir_flood_pool: false (same T0, no pool)

so the two arms differ by the pool alone (the no-pool twin routes the same T_days on the same additive row with
the dam-row positivity cap).

Which dams: the flood-control dams ON their gauge's own reach (``expected_release_fit.csv``: role == dam and
on_reach true; ``smoke_gauges.csv``: dam_purpose == "Flood Risk Reduction"). Every other dam is absent from the
table, i.e. an ordinary channel. The dam's COMID is ``smoke_gauges.csv`` ``dam_COMID``; where one dam COMID sits on
the reach of two gauges, the fit of the gauge whose drainage area is closest to the dam reach's upstream area
(min |log(area_km2 / dam_reach_uparea_km2)|) is kept, as ``replay_offline_fits.py`` does. ``year_completed`` and
``inflow_mean_m3s`` come from ``experiments/reservoir/release_head/dam_features.csv``.

Conventions, offline vs engine (``laws_v6/laws.c``, ``src/routing/mmc.rs`` ``FloodPool``):

- Law. Offline, daily (dt = 1 d): c = min(phi (I - Qc)+, Fmax - F), p = I - c, e = min(F, (Qc - p)+),
  F += c - e, and the bucket receives p + e. Engine, hourly: Vc = min(phi (I - Qc)+ dt, Fmax - F),
  Ve = min(F, (Qc - p)+ dt), the dam row's q' gains (Ve - Vc)/dt. Same law, per step; the engine's I is the dam's
  step-start inflow (routed upstream plus its own q'), the offline I the same-day routed flow at the gauge.
- Scale. Offline Qc = kc * Ibar_off and Fmax = z * Ibar_off (m3/s * d) with Ibar_off the gauge's training mean.
  The engine scales by the dam's ``inflow_mean_m3s`` (Ibar_eng). The release target in m3/s and the pool in m3
  are kept: kc_eng = kc_off * Ibar_off / Ibar_eng, z_eng = z_off * Ibar_off / Ibar_eng; phi is unchanged. This is
  replay_offline_fits.py's rule-curve rescaling (the flux in m3/s is kept).
- Bucket. T_days = FA_p_T0 as fitted (days, >= 0.05 d, above the fixed reader's one-hour floor), routed on the
  additive dam row (the offline daily bucket sits on the routed gauge series; the engine's hourly additive row
  keeps the reach's channel storage, the replay_offline_fits.py caveat).
- The fixed table carries z as fitted, including values above the learned transform's 120-day bound; the fixed
  path does not bound them. The summary line reports how many exceed the learned boxes (kc in [0.5, 20],
  z <= 120 d).

Run from the repo root (any python3 with the standard library):

  python3 experiments/reservoir/smoke/replay_flood_pool.py
"""
import argparse
import csv
import math
import os
from collections import defaultdict

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, "..", "..", ".."))
FLOOD = "Flood Risk Reduction"


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--laws", default=os.path.join(REPO, "experiments/reservoir/laws_v6/laws_by_gauge.csv"))
    ap.add_argument("--fit-table", default=os.path.join(HERE, "expected_release_fit.csv"))
    ap.add_argument("--gauges", default=os.path.join(HERE, "smoke_gauges.csv"))
    ap.add_argument("--dam-features", default=os.path.join(REPO, "experiments/reservoir/release_head/dam_features.csv"))
    ap.add_argument("--out", default=os.path.join(HERE, "fixed_FA.csv"))
    args = ap.parse_args()

    fit = {r["STAID"]: r for r in csv.DictReader(open(args.fit_table))}
    gauges = {r["STAID"]: r for r in csv.DictReader(open(args.gauges))}
    laws = {r["STAID"]: r for r in csv.DictReader(open(args.laws))}
    feats = {r["COMID"]: r for r in csv.DictReader(open(args.dam_features))}

    chosen = [
        s for s, r in fit.items()
        if r["role"] == "dam" and r["on_reach"] == "True" and gauges[s]["dam_purpose"] == FLOOD
    ]
    by_dam = defaultdict(list)
    for s in chosen:
        by_dam[str(int(float(gauges[s]["dam_COMID"])))].append(s)

    def closeness(s):
        g = gauges[s]
        return abs(math.log(float(g["area_km2"]) / float(g["dam_reach_uparea_km2"])))

    rows, shared, missing = [], [], []
    for comid, staids in sorted(by_dam.items(), key=lambda kv: int(kv[0])):
        staid = min(staids, key=closeness)
        if len(staids) > 1:
            shared.append((comid, staids, staid))
        if staid not in laws or comid not in feats:
            missing.append((comid, staid))
            continue
        f = laws[staid]
        d = feats[comid]
        ibar_off = float(f["Ibar"])
        ibar_eng = float(d["inflow_mean_m3s"])
        scale = ibar_off / ibar_eng
        kc_off, phi, z_off = float(f["FA_p_kc"]), float(f["FA_p_phi"]), float(f["FA_p_z"])
        rows.append(dict(
            COMID=comid,
            STAID=staid,
            T_days=float(f["FA_p_T0"]),
            kc=kc_off * scale,
            phi=min(max(phi, 0.0), 1.0),
            z=z_off * scale,
            inflow_mean_m3s=ibar_eng,
            year_completed=d["year_completed"],
            Ibar_offline_m3s=ibar_off,
            kc_offline=kc_off,
            z_offline_days=z_off,
            L2_T0_days=float(f["L2_p_T0"]),
        ))

    cols = ["COMID", "T_days", "kc", "phi", "z", "inflow_mean_m3s", "year_completed", "STAID", "Ibar_offline_m3s",
            "kc_offline", "z_offline_days", "L2_T0_days"]
    with open(args.out, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(cols)
        for r in rows:
            w.writerow([r[c] if isinstance(r[c], str) else repr(r[c]) for c in cols])

    med = lambda v: sorted(v)[len(v) // 2]
    kc = [r["kc"] for r in rows]
    z = [r["z"] for r in rows]
    ratio = [r["Ibar_offline_m3s"] / r["inflow_mean_m3s"] for r in rows]
    print(f"{len(chosen)} on-reach flood-control dam gauges -> {len(rows)} dams "
          f"({len(shared)} dams shared by two gauges; kept the gauge closest in area; {len(missing)} without a fit or "
          f"feature row: {missing})")
    for comid, staids, kept in shared:
        print(f"  COMID {comid}: gauges {staids}, kept {kept}")
    print(f"Ibar_offline / inflow_mean_m3s: median {med(ratio):.3f}, range [{min(ratio):.3f}, {max(ratio):.3f}]")
    print(f"engine kc: median {med(kc):.3f} [{min(kc):.3f}, {max(kc):.3f}], {sum(not 0.5 <= v <= 20 for v in kc)} "
          f"outside the learned box [0.5, 20]")
    print(f"engine z (days of inflow_mean_m3s): median {med(z):.2f} [{min(z):.2f}, {max(z):.2f}], "
          f"{sum(v > 120 for v in z)} above the learned bound 120 d")
    print(f"T_days: median {med([r['T_days'] for r in rows]):.3f}")
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()

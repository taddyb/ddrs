#!/usr/bin/env python3
"""Zeta gradient probe — verdicts.

H4-starvation: median |dL/dfactor| ungauged (and arid) >= 1-2 OOM below gauged,
               at BOTH parameter points.
H4-rejection:  magnitudes comparable off-gauge but signed grad pushes zeta down
               (dL/dfactor > 0 dominant) on arid/ungauged reaches.
Detectability NO-GO: <10% of Ref probes at delta=0.01 clear noise floor AND the
               5% obs band.
Cross-check:   stage-1 reachability rank-predicts stage-2 detectability.

Run: cd ~/projects/ddr && uv run python <ddrs>/scripts/zeta_gradient_analysis.py
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr

OUT = Path("/home/tbindas/projects/ddrs/output/zeta_probe")
GAGES_CSV = Path("/home/tbindas/projects/ddr/references/gage_info/gages_3000.csv")
ATTRS_NC = Path("/home/tbindas/projects/ddr/data/merit_global_attributes_v2.nc")


def sec(t: str) -> None:
    print(f"\n{'=' * 72}\n{t}\n{'=' * 72}")


def attach(attrs_path: Path, comids: np.ndarray, names: list[str]) -> dict[str, np.ndarray]:
    ds = xr.open_dataset(attrs_path)
    acom = ds["COMID"].values.astype(np.int64)
    order = np.argsort(acom)
    pos = order[np.clip(np.searchsorted(acom, comids, sorter=order), 0, len(acom) - 1)]
    ok = acom[pos] == comids
    out = {}
    for n in names:
        v = ds[n].values.astype(float)[pos]
        v[~ok] = np.nan
        out[n] = v
    return out


def load_preds(path: Path) -> tuple[np.ndarray, list[str]]:
    ds = xr.open_dataset(path)
    preds = ds["predictions"].values
    if "gauge_staid" in ds:
        gids = [str(x) for x in ds["gauge_staid"].values]
    elif "gage_ids" in ds:
        gids = [str(x) for x in ds["gage_ids"].values]
    else:
        gids = (path.parent / (path.stem + ".gauges.txt")).read_text().split()
    return preds, gids


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--probe-dir", type=Path, default=OUT)
    ap.add_argument("--skip-stage2", action="store_true",
                    help="only stage-1 verdicts (perturb outputs not present yet)")
    args = ap.parse_args()
    verdicts = []

    # ---------- Stage 1: reachability ----------
    tr = xr.open_dataset(args.probe_dir / "grad_trained.nc")
    co = xr.open_dataset(args.probe_dir / "grad_cold.nc")
    comids = tr["COMID_probe"].values.astype(np.int64)
    assert (comids == co["COMID_probe"].values).all(), "trained/cold COMID sets differ"

    gages = pd.read_csv(GAGES_CSV)
    gauged = np.isin(comids, gages["COMID"].values.astype(np.int64))
    attrs = attach(ATTRS_NC, comids, ["aridity", "meanP", "log10_uparea"])
    from scipy.stats import spearmanr
    r = spearmanr(attrs["aridity"], attrs["meanP"], nan_policy="omit").statistic
    dry_ix = (attrs["aridity"] >= np.nanpercentile(attrs["aridity"], 67)) if r < 0 \
        else (attrs["aridity"] <= np.nanpercentile(attrs["aridity"], 33))
    print(f"aridity-vs-meanP spearman {r:+.2f} → aridity is a "
          f"{'DRYNESS' if r < 0 else 'WETNESS'} index; dry tercile n={int(dry_ix.sum())}")

    sec("Stage 1 — |dL/dfactor| by stratum, trained vs cold")
    ratios = {}
    for label, ds in (("trained", tr), ("cold", co)):
        g = ds["grad_factor_abs"].values
        m_g, m_u = np.median(g[gauged]), np.median(g[~gauged])
        m_dry = np.median(g[dry_ix])
        wet_ix = ~dry_ix & np.isfinite(attrs["aridity"])
        m_wet = np.median(g[wet_ix])
        ratios[label] = (m_g / max(m_u, 1e-300), m_wet / max(m_dry, 1e-300))
        print(f"{label:8s} gauged={m_g:.3e} ungauged={m_u:.3e} (ratio {ratios[label][0]:.1f})"
              f" | dry={m_dry:.3e} wet={m_wet:.3e} (wet/dry {ratios[label][1]:.1f})")
        net = ds["grad_factor_net"].values
        # dL/dfactor > 0 means the loss wants LESS leakance.
        for sl, m in (("gauged", gauged), ("ungauged", ~gauged), ("dry", dry_ix)):
            frac_down = float((net[m] > 0).mean())
            print(f"  {label}/{sl}: frac pushing zeta DOWN = {frac_down * 100:.1f}%")

    starv = all(rt[0] >= 10 for rt in ratios.values())
    verdicts.append(("H4-starvation",
                     "SUPPORTED" if starv else "REFUTED",
                     f"gauged/ungauged |g| ratio trained={ratios['trained'][0]:.1f}, "
                     f"cold={ratios['cold'][0]:.1f} (bar: >=10 at both points)"))

    tr_net_dry = tr["grad_factor_net"].values[dry_ix]
    reject = (not starv) and float((tr_net_dry > 0).mean()) > 0.67
    verdicts.append(("H4-rejection",
                     "SUPPORTED" if reject else ("N/A (starvation holds)" if starv else "REFUTED"),
                     f"{float((tr_net_dry > 0).mean()) * 100:.1f}% of dry-tercile grads push zeta down"))

    # ---------- Stage 2: detectability ----------
    if args.skip_stage2:
        sec("VERDICTS (stage 1 only)")
        for name, v, detail in verdicts:
            print(f"  [{v}] {name}: {detail}")
        return

    sec("Stage 2 — planted-delta detectability at nearest gauges")
    plan = pd.read_csv(args.probe_dir / "probe_plan.csv", dtype={"staid_nearest": str})
    b1, gids = load_preds(args.probe_dir / "perturb/baseline_1.nc")
    b2, _ = load_preds(args.probe_dir / "perturb/baseline_2.nc")
    gid_ix = {g.lstrip("0"): i for i, g in enumerate(gids)}
    # CPU backend is deterministic: noise should be exactly 0 and detection
    # reduces to the obs-uncertainty band. The noise term stays in the
    # criterion so the same script scores CUDA-produced runs unchanged.
    noise = np.abs(b1 - b2)
    print("baseline determinism: max |b1-b2| =", float(noise.max()))

    rows = []
    n_unmatched = 0
    for rnd, grp in plan.groupby("round"):
        pr, _ = load_preds(args.probe_dir / f"perturb/round_{rnd}.nc")
        dq = pr - b1
        for _, p in grp.iterrows():
            i = gid_ix.get(str(p["staid_nearest"]).lstrip("0"))
            if i is None:
                n_unmatched += 1
                continue
            mean_dq = float(np.nanmean(dq[i]))
            peak_dq = float(np.nanmax(np.abs(dq[i])))
            nf = float(np.nanpercentile(noise[i], 99))
            band5 = 0.05 * float(np.nanmean(b1[i]))
            rows.append(dict(comid=p["comid"], delta=p["delta"], cls=p["class"],
                             mean_dq=mean_dq, peak_dq=peak_dq, noise=nf, band5=band5,
                             detect=(abs(mean_dq) > nf) and (abs(mean_dq) > band5),
                             s_re=p["stratum_reach"]))
    if n_unmatched:
        print(f"WARNING: {n_unmatched} probes had no matching gauge row — investigate")
    det = pd.DataFrame(rows)
    for (cls, delta), grp in det.groupby(["cls", "delta"]):
        print(f"{cls:8s} delta={delta}: detectable {grp['detect'].mean() * 100:.1f}% of {len(grp)}")
    ref001 = det[(det["cls"] == "Ref") & (det["delta"] == 0.01)]
    nogo = float(ref001["detect"].mean()) < 0.10
    verdicts.append(("Detectability",
                     "NO-GO" if nogo else "GO",
                     f"{float(ref001['detect'].mean()) * 100:.1f}% of Ref probes at delta=0.01 detectable "
                     "(NO-GO bar: <10%)"))

    grad_map = dict(zip(comids, tr["grad_factor_abs"].values))
    det["reach_abs"] = det["comid"].map(grad_map)
    rc = spearmanr(det["reach_abs"], det["detect"].astype(float), nan_policy="omit").statistic
    print(f"\ncross-check: spearman(reachability, detected) = {rc:+.2f}")
    verdicts.append(("Cross-check", "PASS" if rc > 0.3 else "SUSPECT",
                     f"rank corr {rc:+.2f} (bar: > 0.3)"))

    det.to_csv(args.probe_dir / "detectability_rows.csv", index=False)
    print("per-probe rows →", args.probe_dir / "detectability_rows.csv")

    sec("VERDICTS")
    for name, v, detail in verdicts:
        print(f"  [{v}] {name}: {detail}")


if __name__ == "__main__":
    main()

#!/usr/bin/env python
"""How often do the geometry clamps saturate? Each saturated clamp is a gradient
sink: the parameter behind it receives zero gradient at that reach, so no amount
of training can identify it there. The equations reference flags this as the
measurement the repo has never taken.

Reconstructs the geometry chain from a parameter dump plus the eval diagnostic's
mean depth, and reports the saturated fraction for each clamp in
params.attribute_minimums and for the side-slope bounds.

Caveat stated up front: this uses each reach's MEAN eval depth, so it is a
lower bound on saturation. A clamp that fires only at low flow is invisible here.

usage: clamp_saturation.py <params.nc> <kan_parameters.nc> [--config C]
"""
import argparse, json
import numpy as np, netCDF4 as nc, pandas as pd

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("params"); ap.add_argument("zeta")
    ap.add_argument("--depth-lb", type=float, default=0.01)
    ap.add_argument("--velocity-lb", type=float, default=0.01)
    ap.add_argument("--bottom-width-lb", type=float, default=0.01)
    ap.add_argument("--side-slope-bounds", type=float, nargs=2, default=(0.5, 50.0))
    a = ap.parse_args()

    P = nc.Dataset(a.params); Z = nc.Dataset(a.zeta)
    pc = np.asarray(P["COMID"][:], dtype=np.int64)
    zc = np.asarray(Z["COMID_eval"][:], dtype=np.int64)
    pos = pd.Index(pc).get_indexer(zc); ok = pos >= 0
    g = lambda d, v: np.asarray(d[v][:], dtype=float)
    depth = g(Z, "depth_mean")[ok]
    q = g(P, "q_spatial")[pos[ok]] if "q_spatial" in P.variables else np.full(ok.sum(), 0.65)
    p = g(P, "p_spatial")[pos[ok]] if "p_spatial" in P.variables else np.full(ok.sum(), 21.0)
    n = g(P, "n")[pos[ok]]

    # geometry chain, mirroring src/geometry.rs
    top_width   = p * depth**q
    side_slope  = top_width * q / (2.0 * depth)
    side_clamped = np.clip(side_slope, *a.side_slope_bounds)
    bottom_width = top_width - 2.0 * side_clamped * depth

    rep = {
        "n_eval_reaches": int(ok.sum()),
        "caveat": "computed from each reach's MEAN eval depth, so every figure is a LOWER bound on saturation",
        "depth at its floor": float((depth <= a.depth_lb * 1.001).mean()),
        "side slope at its lower bound 0.5": float((side_slope <= a.side_slope_bounds[0]).mean()),
        "side slope at its upper bound 50": float((side_slope >= a.side_slope_bounds[1]).mean()),
        "side slope clamped either way": float(((side_slope <= a.side_slope_bounds[0]) | (side_slope >= a.side_slope_bounds[1])).mean()),
        "bottom width at or below its floor": float((bottom_width <= a.bottom_width_lb).mean()),
        "roughness at its range floor 0.015": float((n <= 0.015 * 1.001).mean()),
        "roughness at its range ceiling 0.25": float((n >= 0.25 * 0.999).mean()),
    }
    # the side-slope clamp is the interesting one: it is a pure function of q
    rep["note_side_slope"] = ("side_slope = p*d^q * q / (2d) = (q/2)*p*d^(q-1); with q fixed at 0.65 "
                              "it saturates the 0.5 lower bound whenever d exceeds (q*p/(2*0.5))^(1/(1-q))")
    if np.isfinite(q).all() and np.allclose(q, q[0]):
        qq, pp = q[0], p[0]
        d_crit = (qq * pp / (2.0 * a.side_slope_bounds[0]))**(1.0 / (1.0 - qq))
        rep["depth above which the side slope pins at 0.5 (m)"] = float(d_crit)
        rep["fraction of reaches above that depth"] = float((depth > d_crit).mean())
    print(json.dumps(rep, indent=2))

if __name__ == "__main__":
    main()

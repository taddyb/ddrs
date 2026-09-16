"""Export DDR's historical `_compute_zeta` reference flux to a committed fixture.

Leakance was added to DDR in commit `c2bd0f9` ("feat: add leakance (GW-SW
exchange) to routing (#130)", 2026-02-13) and later reverted on DDR master, so
there is no importable reference on a current checkout. This script pulls the
function's exact bytes out of git history, executes them, and writes the inputs
plus the reference flux to `fixtures/leakance/ddr_reference_zeta.csv`, which
`tests/leakance_reference_match.rs` reads.

The reference chain (verbatim from `_compute_zeta`):

    numerator   = q_t * n * (q_spatial + 1)
    denominator = p_spatial * sqrt(s0)
    depth       = (numerator / (denominator + 1e-8)) ** (3 / (5 + 3*q_spatial))
    width       = (p_spatial * depth) ** q_spatial
    area        = width * length
    zeta        = leakance_factor * area * K_D * (depth - d_gw)

Note that the reference applies NO lower bound to `depth` (unlike its own
`_get_trapezoid_velocity`, which clamps at `depth_lb`), and is sign-symmetric:
`depth < d_gw` is a gaining reach and yields a negative flux.

Run under DDR's venv (the fixture only needs torch + numpy):

    cd ~/projects/ddr && uv run python ~/projects/ddrs/scripts/export_ddr_leakance_reference.py
"""

from __future__ import annotations

import ast
import hashlib
import subprocess
from pathlib import Path

import numpy as np
import torch

DDR_REPO = Path.home() / "projects" / "ddr"
DDR_COMMIT = "c2bd0f9"
DDR_PATH = "src/ddr/routing/mmc.py"
OUT_DIR = Path(__file__).resolve().parent.parent / "fixtures" / "leakance"

# ddrs `attribute_minimums.depth` (config/merit_training.yaml). Recorded in the
# fixture header so the Rust test can flag the reaches where ddrs's shared
# depth saturates and the reference's unclamped depth does not.
DEPTH_LB = 0.01


def load_reference_compute_zeta() -> tuple[object, str]:
    """Return DDR's `_compute_zeta` from commit `c2bd0f9`, plus a sha256 of its source.

    The function is executed from the historical source text rather than
    imported, because `ddr.routing.mmc` on DDR master no longer defines it.
    """
    blob = subprocess.run(
        ["git", "-C", str(DDR_REPO), "show", f"{DDR_COMMIT}:{DDR_PATH}"],
        capture_output=True,
        check=True,
        text=True,
    ).stdout
    tree = ast.parse(blob)
    for node in tree.body:
        if isinstance(node, ast.FunctionDef) and node.name == "_compute_zeta":
            src = ast.get_source_segment(blob, node)
            break
    else:  # pragma: no cover - a missing function means the SHA is wrong
        raise SystemExit(f"_compute_zeta not found in {DDR_COMMIT}:{DDR_PATH}")
    digest = hashlib.sha256(src.encode()).hexdigest()
    namespace: dict[str, object] = {"torch": torch}
    exec(compile(ast.parse(src), filename=f"{DDR_COMMIT}:{DDR_PATH}", mode="exec"), namespace)
    return namespace["_compute_zeta"], digest


# Hand-chosen reaches. Column order matches CASE_COLUMNS below.
#   q_t [m3/s], n [-], q_spatial [-], s0 [m/m], p_spatial [-], length [m],
#   K_D [1/s], d_gw [m], leakance_factor [-]
#
# `K_D` box is [1e-8, 1e-6] and `d_gw` is [-2, 2] (`src/config.rs`
# `ParameterRanges::default`), so "floor"/"mid"/"ceiling" below refer to those.
BASE = dict(n=0.035, q_spatial=0.5, s0=1.0e-3, p_spatial=21.0, length=1000.0,
            K_D=1.0e-7, d_gw=0.0, leakance_factor=0.5)

CASES: list[tuple[str, dict[str, float]]] = []


def case(label: str, **overrides: float) -> None:
    CASES.append((label, {**BASE, **overrides}))


# Discharge over six decades. q_t = 1e-4 sits below ddrs's depth floor.
case("q_1e-4_at_depth_floor", q_t=1.0e-4)
case("q_1e-3", q_t=1.0e-3)
case("q_1e-2", q_t=1.0e-2)
case("q_1e-1", q_t=1.0e-1)
case("q_1e0", q_t=1.0)
case("q_1e1", q_t=1.0e1)
case("q_1e2", q_t=1.0e2)
case("q_1e3", q_t=1.0e3)
case("q_1e4", q_t=1.0e4)

# Groundwater offset: losing (depth > d_gw) and gaining (depth < d_gw).
# depth(q_t=10) is about 0.90 m, so d_gw = 2 is a gaining reach.
case("d_gw_-2_strongly_losing", q_t=1.0e1, d_gw=-2.0)
case("d_gw_0_losing", q_t=1.0e1, d_gw=0.0)
case("d_gw_0.5_weakly_losing", q_t=1.0e1, d_gw=0.5)
case("d_gw_2_gaining", q_t=1.0e1, d_gw=2.0)
case("d_gw_2_strongly_gaining_low_q", q_t=1.0e-1, d_gw=2.0)

# Exchange rate across its box.
case("K_D_floor_1e-8", q_t=1.0e2, K_D=1.0e-8)
case("K_D_logmid_1e-7", q_t=1.0e2, K_D=1.0e-7)
case("K_D_ceiling_1e-6", q_t=1.0e2, K_D=1.0e-6)

# Leakance weight at both ends of [0, 1].
case("factor_0_inactive", q_t=1.0e2, leakance_factor=0.0)
case("factor_1_full", q_t=1.0e2, leakance_factor=1.0)

# Channel-shape and roughness extremes.
case("q_spatial_0_rectangular", q_t=1.0e1, q_spatial=0.0)
case("q_spatial_1_triangular", q_t=1.0e1, q_spatial=1.0)
case("p_spatial_1", q_t=1.0e1, p_spatial=1.0)
case("p_spatial_50", q_t=1.0e1, p_spatial=50.0)
case("n_0.01_smooth", q_t=1.0e1, n=0.01)
case("n_0.3_rough", q_t=1.0e1, n=0.3)

# Slope and reach length extremes.
case("slope_1e-5_flat", q_t=1.0e1, s0=1.0e-5)
case("slope_1e-1_steep", q_t=1.0e1, s0=1.0e-1)
case("length_100", q_t=1.0e1, length=100.0)
case("length_50000", q_t=1.0e1, length=50000.0)

CASE_COLUMNS = ["q_t", "n", "q_spatial", "s0", "p_spatial", "length",
                "K_D", "d_gw", "leakance_factor"]


def run(compute_zeta, dtype: torch.dtype) -> tuple[np.ndarray, np.ndarray]:
    """Return `(depth, zeta)` for every case, evaluated in `dtype`."""
    kwargs = {
        name: torch.tensor([c[name] for _, c in CASES], dtype=dtype)
        for name in CASE_COLUMNS
    }
    zeta = compute_zeta(**kwargs)
    # `_compute_zeta` returns only zeta; recover its depth by the same spelling
    # so the fixture can separate a depth-inversion mismatch from a tail-chain one.
    numerator = kwargs["q_t"] * kwargs["n"] * (kwargs["q_spatial"] + 1)
    denominator = kwargs["p_spatial"] * torch.pow(kwargs["s0"], 0.5)
    depth = torch.pow(
        torch.div(numerator, denominator + 1e-8),
        torch.div(3.0, 5.0 + 3.0 * kwargs["q_spatial"]),
    )
    return depth.numpy(), zeta.numpy()


def main() -> None:
    compute_zeta, digest = load_reference_compute_zeta()
    depth32, zeta32 = run(compute_zeta, torch.float32)
    depth64, zeta64 = run(compute_zeta, torch.float64)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out = OUT_DIR / "ddr_reference_zeta.csv"
    with out.open("w") as fh:
        fh.write(f"# DDR reference `_compute_zeta`, commit {DDR_COMMIT} ({DDR_PATH})\n")
        fh.write(f"# sha256(extracted function source) = {digest}\n")
        fh.write(f"# torch {torch.__version__}; ref_* columns are the reference's own\n")
        fh.write("# unclamped depth and flux, evaluated in float32 and float64.\n")
        fh.write(f"# ddrs_depth_lb = {DEPTH_LB}\n")
        fh.write("# Regenerate: cd ~/projects/ddr && uv run python \\\n")
        fh.write("#   ~/projects/ddrs/scripts/export_ddr_leakance_reference.py\n")
        fh.write("case," + ",".join(CASE_COLUMNS)
                 + ",ref_depth_f32,ref_zeta_f32,ref_depth_f64,ref_zeta_f64\n")
        for i, (label, c) in enumerate(CASES):
            vals = ",".join(f"{c[name]!r}" for name in CASE_COLUMNS)
            fh.write(
                f"{label},{vals},"
                f"{float(depth32[i]):.9e},{float(zeta32[i]):.9e},"
                f"{float(depth64[i]):.17e},{float(zeta64[i]):.17e}\n"
            )

    n_floor = int((depth64 < DEPTH_LB).sum())
    n_gaining = int((zeta64 < 0).sum())
    print(f"Wrote {out} ({len(CASES)} reaches)")
    print(f"  sha256(_compute_zeta @ {DDR_COMMIT}) = {digest}")
    print(f"  depth range        : {depth64.min():.6e} .. {depth64.max():.6e} m")
    print(f"  |zeta| range       : {np.abs(zeta64[zeta64 != 0]).min():.6e} .. "
          f"{np.abs(zeta64).max():.6e} m3/s")
    print(f"  reaches below ddrs depth floor ({DEPTH_LB} m): {n_floor}")
    print(f"  gaining reaches (zeta < 0)                   : {n_gaining}")


if __name__ == "__main__":
    main()

"""Run DDR's sandbox benchmark and dump inputs + outputs to CSV.

These CSVs are read by the Rust comparison example to produce a bit-comparable
replay of DDR's Muskingum-Cunge routing.

Outputs (written to `../fixtures/sandbox/`):
    qprime_topo.csv         (T, N) lateral inflow in topological order
    adjacency_topo.csv      (N, N) dense adjacency in topological order
    topo_order.csv          (N,)   reach IDs in topological order
    rapid2_order.csv        (N,)   reach IDs in RAPID2 order = [10,20,30,40,50]
    ddr_discharge_rapid2.csv (N, T) DDR routed discharge in RAPID2 order
    config.csv              key=value pairs: length, slope, x_storage, params, etc.
"""

from pathlib import Path

import numpy as np
import torch

import sys

THIS = Path(__file__).resolve()
DDR_REPO = Path.home() / "projects" / "ddr"
sys.path.insert(0, str(DDR_REPO))
sys.path.insert(0, str(DDR_REPO / "engine"))
sys.path.insert(0, str(DDR_REPO / "benchmarks"))

from tests.benchmarks.conftest import (  # noqa: E402
    RAPID2_REACH_IDS,
    create_ddr_config,
    create_routing_dataclass,
    MockKAN,
    MockStreamflow,
)

# ---------- recreate fixture chain in one shot ----------
import pandas as pd
import geopandas as gpd
import xarray as xr
from shapely.geometry import LineString
from scipy.interpolate import interp1d
from ddr_engine.merit import build_merit_adjacency, coo_from_zarr
from ddr import dmc


SANDBOX_DIR = DDR_REPO / "tests" / "input" / "Sandbox"
OUT_DIR = Path(__file__).resolve().parent.parent / "fixtures" / "sandbox"
OUT_DIR.mkdir(parents=True, exist_ok=True)


def _sandbox_to_merit_format(sandbox_df: pd.DataFrame) -> gpd.GeoDataFrame:
    sandbox_df = sandbox_df.copy()
    sandbox_df.columns = ["COMID", "NextDownID"]
    upstream_lookup: dict[int, list[int]] = {}
    for _, row in sandbox_df.iterrows():
        comid = int(row["COMID"])
        next_down = int(row["NextDownID"])
        if next_down != 0:
            upstream_lookup.setdefault(next_down, []).append(comid)
    records = []
    for _, row in sandbox_df.iterrows():
        comid = int(row["COMID"])
        next_down = int(row["NextDownID"])
        upstreams = upstream_lookup.get(comid, [])
        records.append(
            {
                "COMID": comid,
                "NextDownID": next_down,
                "up1": upstreams[0] if len(upstreams) > 0 else 0,
                "up2": upstreams[1] if len(upstreams) > 1 else 0,
                "up3": upstreams[2] if len(upstreams) > 2 else 0,
                "up4": upstreams[3] if len(upstreams) > 3 else 0,
            }
        )
    df = pd.DataFrame(records)
    df["geometry"] = [LineString([(0, i), (1, i)]) for i in range(len(df))]
    return gpd.GeoDataFrame(df, geometry="geometry", crs="EPSG:4326")


def main() -> None:
    # ---- build sandbox zarr in a temp dir, just like the fixture does ----
    import tempfile

    sandbox_connections = pd.read_csv(
        SANDBOX_DIR / "rapid_connect_Sandbox.csv", header=None
    )
    merit_fp = _sandbox_to_merit_format(sandbox_connections)
    with tempfile.TemporaryDirectory() as tmp:
        zarr_path = Path(tmp) / "sandbox_adjacency.zarr"
        build_merit_adjacency(merit_fp, zarr_path)

        # ---- load Qext, interpolate to hourly (exactly like the fixture) ----
        ds = xr.open_dataset(SANDBOX_DIR / "Qext_Sandbox_19700101_19700110.nc4")
        qext = ds["Qext"].values  # (80, 5) in RAPID2 order
        ds.close()
        n_original = qext.shape[0]
        n_reaches = qext.shape[1]
        t_original = np.arange(n_original) * 3
        t_hourly = np.arange(t_original[-1] + 1)
        qprime_rapid2 = np.zeros((len(t_hourly), n_reaches), dtype=np.float32)
        for i in range(n_reaches):
            f = interp1d(t_original, qext[:, i], kind="linear")
            qprime_rapid2[:, i] = f(t_hourly)

        # ---- recover topological order from zarr ----
        _, ts_order = coo_from_zarr(zarr_path)
        topo_idx = [RAPID2_REACH_IDS.index(rid) for rid in ts_order]
        qprime_topo = qprime_rapid2[:, topo_idx]  # (T, N) topo-ordered

        # ---- build routing dataclass + run DDR ----
        cfg = create_ddr_config()
        routing_dataclass, _ = create_routing_dataclass(zarr_path, num_reaches=5)
        adj_dense = routing_dataclass.adjacency_matrix.to_dense().numpy()  # (N,N) topo

        learnable = ["n", "q_spatial", "top_width", "side_slope"]
        kan = MockKAN(num_reaches=5, learnable_params=learnable)
        spatial_params = kan()
        streamflow_tensor = torch.from_numpy(qprime_topo).float()

        model = dmc(cfg=cfg, device="cpu")
        model.set_progress_info(epoch=0, mini_batch=0)
        out = model(
            routing_dataclass=routing_dataclass,
            spatial_parameters=spatial_params,
            streamflow=streamflow_tensor,
        )
        # ddr_output is (reaches, timesteps) in topological order
        ddr_topo = out["runoff"].detach().numpy()
        # Reorder topo -> RAPID2
        reorder_idx = [ts_order.index(rid) for rid in RAPID2_REACH_IDS]
        ddr_rapid2 = ddr_topo[reorder_idx, :]

        # ---- write CSVs ----
        np.savetxt(OUT_DIR / "qprime_topo.csv", qprime_topo, delimiter=",", fmt="%.6f")
        np.savetxt(OUT_DIR / "adjacency_topo.csv", adj_dense, delimiter=",", fmt="%.1f")
        np.savetxt(OUT_DIR / "topo_order.csv", np.asarray(ts_order, dtype=int), fmt="%d")
        np.savetxt(OUT_DIR / "rapid2_order.csv", np.asarray(RAPID2_REACH_IDS, dtype=int), fmt="%d")
        np.savetxt(OUT_DIR / "ddr_discharge_rapid2.csv", ddr_rapid2, delimiter=",", fmt="%.8f")

        # config snapshot
        with open(OUT_DIR / "config.csv", "w") as fh:
            fh.write("# n_reaches,5\n")
            fh.write(f"# n_timesteps,{qprime_topo.shape[0]}\n")
            fh.write("length_m,5000.0\n")
            fh.write("slope,0.001\n")
            fh.write("x_storage,0.25\n")
            fh.write("dt_seconds,3600.0\n")
            fh.write("p_spatial_default,21.0\n")
            # DDR parameter ranges
            for name, lo, hi in [
                ("n", 0.015, 0.25),
                ("q_spatial", 0.0, 1.0),
            ]:
                fh.write(f"range_{name},{lo},{hi}\n")
            fh.write("log_space_parameters,n\n")  # DDR's benchmark uses n log-space

        print(f"Wrote fixtures to {OUT_DIR}")
        print(f"  topological order: {ts_order}")
        print(f"  qprime_topo shape: {qprime_topo.shape}")
        print(f"  ddr_rapid2 shape:  {ddr_rapid2.shape}")
        print(f"  ddr min/mean/max:  {ddr_rapid2.min():.4f} / {ddr_rapid2.mean():.4f} / {ddr_rapid2.max():.4f}")


if __name__ == "__main__":
    main()

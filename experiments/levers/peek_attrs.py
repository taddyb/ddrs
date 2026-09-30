import xarray as xr
import zarr
ds = xr.open_dataset("/home/tbindas/projects/ddr/data/merit_global_attributes_v2.nc")
print(ds)
gz = zarr.open_group("/home/tbindas/projects/ddr/data/merit_gages_conus_adjacency.zarr", mode="r")
ks = list(gz.keys())
print(len(ks), ks[:5])
g = gz[ks[0]]
print(list(g.keys()), dict(g.attrs))
for k in g.keys():
    print(k, g[k].shape, g[k].dtype, g[k][:5])

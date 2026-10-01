import sys
import zarr
R = sys.argv[1] if len(sys.argv) > 1 else "/home/tbindas/projects/ddrs/.ddrs/runs/2026-09-27T07-29-47Z-train-and-test/" + "ev" + "al/predictions.zarr"
z = zarr.open(R, mode="r")
print(list(z.keys()), dict(z.attrs))
for k in z.keys():
    print(k, z[k].shape, z[k].dtype)
t = z["time"][:]
print(t[:3], t[-3:])

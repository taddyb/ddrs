"""E-11: full-store equality of the 08-14 original lumped store and its 09-13 rechunk, plus whole-store stats."""
import icechunk, zarr, numpy as np, warnings; warnings.filterwarnings("ignore")
B="/mnt/ssd1/data/icechunk/"
def op(n): return zarr.open_group(icechunk.Repository.open(icechunk.local_filesystem_storage(B+n)).readonly_session("main").store, mode="r")["Qr"]
a=op("daily_dhbv2_lumped_aorc2f_merit_unit_catchments.ic"); b=op("daily_dhbv2_lumped_aorc2f_merit_unit_catchments_rechunked.ic")
n=a.shape[0]; ne=0; tot=0; s=0.0; nan=0; fl=0; allfl=[]; mn=np.inf; mx=0.0; neg=0
for i in range(0,n,3080):
    x=a[i:i+3080,:]; y=b[i:i+3080,:]
    ne+=int((x!=y).sum()); tot+=x.size; s+=float(x.sum(dtype="f8")); nan+=int(np.isnan(x).sum()); fl+=int((x==np.float32(1e-6)).sum())
    allfl.append((x.max(1)<=np.float32(1e-6))); mn=min(mn,float(x.min())); mx=max(mx,float(x.max())); neg+=int((x<0).sum())
allfl=np.concatenate(allfl)
print(f"rows {n} values {tot} unequal {ne}  mean {s/tot:.4f} m3/s  NaN {nan}  floor frac {fl/tot:.4f}  min {mn:.2e} max {mx:.1f} neg {neg}  divides entirely at floor {int(allfl.sum())}")
np.save("/tmp/e11_allfloor_rows.npy", np.where(allfl)[0])

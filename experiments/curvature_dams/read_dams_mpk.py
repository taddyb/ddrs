"""Read the per-dam parameters (theta, delta, pool) from a release_dams.mpk checkpoint (burn NamedMpkFileRecorder,
plain MessagePack, f32) with a minimal MessagePack decoder, and test the per-dam L2 balance prediction:
under per_dam_l2 = lambda the equilibrium of a raw parameter is r_eq = |dL_gauge/dr| / (2 lambda G), G ~ 250 gauges
per optimizer step. Usage: python read_dams_mpk.py <release_dams.mpk> [...]"""
import struct
import sys

import numpy as np


def unpack(b, i=0):
    t = b[i]
    if t <= 0x7F:
        return t, i + 1
    if 0x80 <= t <= 0x8F or t in (0xDE, 0xDF):
        if t <= 0x8F:
            n, i = t & 0x0F, i + 1
        elif t == 0xDE:
            n, i = struct.unpack(">H", b[i + 1:i + 3])[0], i + 3
        else:
            n, i = struct.unpack(">I", b[i + 1:i + 5])[0], i + 5
        d = {}
        for _ in range(n):
            k, i = unpack(b, i)
            v, i = unpack(b, i)
            d[k] = v
        return d, i
    if 0x90 <= t <= 0x9F or t in (0xDC, 0xDD):
        if t <= 0x9F:
            n, i = t & 0x0F, i + 1
        elif t == 0xDC:
            n, i = struct.unpack(">H", b[i + 1:i + 3])[0], i + 3
        else:
            n, i = struct.unpack(">I", b[i + 1:i + 5])[0], i + 5
        out = []
        for _ in range(n):
            v, i = unpack(b, i)
            out.append(v)
        return out, i
    if 0xA0 <= t <= 0xBF or t in (0xD9, 0xDA, 0xDB):
        if t <= 0xBF:
            n, i = t & 0x1F, i + 1
        elif t == 0xD9:
            n, i = b[i + 1], i + 2
        elif t == 0xDA:
            n, i = struct.unpack(">H", b[i + 1:i + 3])[0], i + 3
        else:
            n, i = struct.unpack(">I", b[i + 1:i + 5])[0], i + 5
        return b[i:i + n].decode(), i + n
    if t in (0xC4, 0xC5, 0xC6):
        w = {0xC4: 1, 0xC5: 2, 0xC6: 4}[t]
        n = int.from_bytes(b[i + 1:i + 1 + w], "big")
        i += 1 + w
        return bytes(b[i:i + n]), i + n
    if t == 0xC0:
        return None, i + 1
    if t in (0xC2, 0xC3):
        return t == 0xC3, i + 1
    if t == 0xCA:
        return struct.unpack(">f", b[i + 1:i + 5])[0], i + 5
    if t == 0xCB:
        return struct.unpack(">d", b[i + 1:i + 9])[0], i + 9
    fmt = {0xCC: ">B", 0xCD: ">H", 0xCE: ">I", 0xCF: ">Q", 0xD0: ">b", 0xD1: ">h", 0xD2: ">i", 0xD3: ">q"}
    if t in fmt:
        w = struct.calcsize(fmt[t])
        return struct.unpack(fmt[t], b[i + 1:i + 1 + w])[0], i + 1 + w
    if t >= 0xE0:
        return t - 256, i + 1
    raise ValueError(hex(t))


def tensors(path):
    d, _ = unpack(open(path, "rb").read())
    item = d["item"]
    out = {}
    for k, v in item.items():
        if isinstance(v, dict) and "param" in v:
            p = v["param"]
            raw = p["bytes"]
            arr = np.frombuffer(raw if isinstance(raw, bytes) else bytes(raw), dtype="<f4")
            out[k] = arr.reshape(p["shape"])
    return out


if __name__ == "__main__":
    for path in sys.argv[1:]:
        t = tensors(path)
        print(path)
        for k, v in t.items():
            a = np.abs(v)
            print(f"  {k}: shape {v.shape}; nonzero rows {int((a.reshape(len(v), -1).max(1) > 0).sum())}; "
                  f"|value| median over nonzero {np.median(a[a > 0]) if (a > 0).any() else 0:.4f}, p90 "
                  f"{np.percentile(a[a > 0], 90) if (a > 0).any() else 0:.4f}, max {a.max():.4f}")
            if k == "pool":
                rz = v[:, 2]
                nz = rz[rz != 0]
                if len(nz):
                    z = 120 / (1 + np.exp(-(4 * nz + np.log(0.05 / 119.95))))
                    print(f"    pool z (d) over moved dams: median {np.median(z):.3f}, p90 {np.percentile(z, 90):.3f}, "
                          f"max {z.max():.3f}")

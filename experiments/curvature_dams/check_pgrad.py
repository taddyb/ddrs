"""Check: the pathwise gradient (lawgrad.c) against tiny-step central differences of laws.c, and the loss value."""
import numpy as np

import landscapes as L

worst = 0.0
for s in L.FA69[:8]:
    g = L.ctx(s)
    f = L.fitted(s, "FA")
    base = {k: f[k] for k in ("T0", "kc", "phi", "z")}
    for pt in (base, dict(T0=L.T0_HEAD, kc=3.0, phi=0.5, z=0.05), dict(base, z=float(np.sqrt(0.05 * base["z"])))):
        Lp, gr = L.pgrad(g, pt)
        assert abs(L.trloss(g, L.simQ(g, "FA", pt)) - Lp) < 1e-12
        fd = []
        for name in ("T0", "kc", "phi", "z"):
            h = 1e-6
            x = L.to_x(name, pt[name])
            lp = L.trloss(g, L.simQ(g, "FA", dict(pt, **{name: float(L.from_x(name, x + h))})))
            lm = L.trloss(g, L.simQ(g, "FA", dict(pt, **{name: float(L.from_x(name, x - h))})))
            fd.append((lp - lm) / (2 * h))
        fd = np.array(fd)
        rel = np.abs(gr - fd) / np.maximum(np.abs(fd), 1e-6)
        worst = max(worst, rel.max())
        print(s, np.round(gr, 6), np.round(fd, 6))
print("worst relative difference", worst)

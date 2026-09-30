"""Print the effective (parsed YAML) differences between run configs, pairwise against the first run."""
import sys

import yaml

import common as C


def flat(d, p=""):
    out = {}
    if isinstance(d, dict):
        for k, v in d.items():
            out.update(flat(v, f"{p}{k}."))
    else:
        out[p[:-1]] = d
    return out


runs = sys.argv[1:]
ref = flat(yaml.safe_load(open(C.RUNS / runs[0] / "config.yaml")))
for r in runs[1:]:
    c = flat(yaml.safe_load(open(C.RUNS / r / "config.yaml")))
    ks = sorted(set(ref) | set(c))
    diff = {k: (ref.get(k), c.get(k)) for k in ks if ref.get(k) != c.get(k)}
    print(f"== {runs[0][:20]} vs {r[:20]}")
    for k, (a, b) in diff.items():
        print(f"   {k}: {a} -> {b}")

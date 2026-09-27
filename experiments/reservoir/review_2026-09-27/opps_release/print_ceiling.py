import json
r = json.load(open("/home/tbindas/.claude/jobs/dacd6d8c/tmp/opps_release/ceiling.json"))


def p(x):
    return None if x is None else (x['median'], x['ci'], f"{x['n_up']}/{x['n_down']}")


print("n", r["n_dam"], r["n_control"], "cv", r["cv_feature_model"])
for name in ["fit", "cv", "cv_T0_only", "learned42", "learned43", "learned42_x3", "learned43_x3", "nash1", "nash2", "nash3"]:
    if name in r:
        x = r[name]
        print(f"{name:14s} dNSE {p(x['dnse_vs_pass'])} dKGE {p(x['dkge_vs_pass'])[:2]} dalpha {x['dalpha_vs_pass']['median']} dr {x['dr_vs_pass']['median']} median {x['median_nse']}")
for N in (2, 3):
    x = r[f"nash{N}_vs_nash1"]
    print(f"nash{N} vs nash1: dNSE {p(x['dnse'])} dKGE {p(x['dkge'])[:2]} dalpha {x['dalpha']['median']} dr {x['dr']['median']} ctl {p(x['control_dnse'])} T0 ratio {x['median_T0_ratio']}")
print("gain by fitted T0 class")
for k, v in r["fit_gain_by_fitted_T0"].items():
    print(f"  {k:8s} n {v['n']:3d} fit {p(v['fit'])} learned42 {p(v['learned42'])}")
print("share of summed fit gain", r["share_of_summed_fit_gain_by_T0_class"])

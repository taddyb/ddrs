import json
r = json.load(open('/home/tbindas/.claude/jobs/dacd6d8c/tmp/t0_size/results.json'))
print(json.dumps(r['partial'], indent=0))
print(r['seed_rank_agreement'])
print(r['seed_log10_ratio_43_over_42'])
for s in ('s42', 's43'):
    print(s, {k: (round(v['rho'], 3), v['n']) for k, v in r['disagreement_drivers'][s].items()})
    o = r['surrogate'][s]
    print(s, {k: round(v, 3) for k, v in o.items() if isinstance(v, float)})
    print(' ridge:', {k: round(v, 3) for k, v in list(o['ridge_coef'].items())[:8]})
    print(' gbm:', {k: round(v, 3) for k, v in list(o['gbm_perm_importance'].items())[:8]})

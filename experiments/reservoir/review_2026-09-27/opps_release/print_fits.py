import json
r = json.load(open("/home/tbindas/.claude/jobs/dacd6d8c/tmp/opps_release/fits.json"))


def p(x):
    return None if x is None else (x['median'], x['ci'], f"{x['n_up']}/{x['n_down']}", x.get('sign_p'))


for law, e in r['laws'].items():
    print('==', law)
    for role in ['dam', 'control']:
        if role in e and e[role]['dnse_vs_seas'] is not None:
            x = e[role]
            print(' ', role, 'dNSE vs seas', p(x['dnse_vs_seas']), '| dKGE', p(x['dkge_vs_seas'])[:2], '| dalpha', x['dalpha_vs_seas']['median'],
                  '| dbeta', x['dbeta_vs_seas']['median'], '| theta pct', x['theta_pct'], '| neutral', x['frac_theta_neutral'],
                  '| >0.01', x['frac_dnse_gt_0_01'], '<-0.01', x['frac_dnse_lt_m0_01'], '| train gain', x['train_gain_vs_seas_median'],
                  '| median nse', x['median_nse'])
    if e.get('dam_minus_matched_control_vs_seas'):
        print('  dam-ctl', p(e['dam_minus_matched_control_vs_seas']))
print('by_dor')
for law, d in r['by_dor'].items():
    print(' ', law, {k: (v['median'], v['n'], f"{v['n_up']}/{v['n_down']}") for k, v in d.items() if v})
print('by_purpose')
for law, d in r['by_purpose'].items():
    print(' ', law, {k: (v['median'], v['n'], f"{v['n_up']}/{v['n_down']}") for k, v in d.items() if v})
print('loss_vs_beta', {k: (p(v) if isinstance(v, dict) else v) for k, v in r['loss_vs_beta'].items()})
print('train_selected', p(r['train_selected_best_law']['dnse_vs_seas']), r['train_selected_best_law']['counts'])

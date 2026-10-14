"""Exploratory, added after the extension runs (not pre-registered): each gated condition minus vanilla,
paired by seed, on Pass@1, Pass@32 and boundary exits. t-intervals over seeds, uncorrected.

    python scripts/exploratory_vs_vanilla.py --artifact_root ../results_0.5B
"""
import argparse
import json
from pathlib import Path

import numpy as np
from scipy import stats

STEM = 'math500_n32_p500'
CONDS = ['o_self', 'meg', 'cb_grpo', 'h_cb_grpo', 'heg_grpo']


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--artifact_root', required=True)
    ap.add_argument('--model_size', default='0.5B')
    a = ap.parse_args()
    root = Path(a.artifact_root)
    load = lambda p: np.load(p)['correctness'].astype(int)
    base = load(root / 'baseline' / a.model_size / 'eval' / f'{STEM}.npz')
    solved = base.sum(1) > 0

    def runs(cond):
        return {int(p.parents[1].name[4:]): load(p) for p in root.glob(f'runs/{cond}/{a.model_size}/seed*/eval/{STEM}.npz')}

    van = runs('vanilla')
    out = {}
    for cond in CONDS:
        r = runs(cond)
        seeds = sorted(set(r) & set(van))
        rows = {'pass1_pp': [100 * (r[s].mean() - van[s].mean()) for s in seeds],
                'pass32_pp': [100 * ((r[s].sum(1) > 0).mean() - (van[s].sum(1) > 0).mean()) for s in seeds],
                'exits': [float((solved & (r[s].sum(1) == 0)).sum() - (solved & (van[s].sum(1) == 0)).sum())
                          for s in seeds]}
        out[cond] = {'seeds': seeds}
        for k, d in rows.items():
            d = np.asarray(d)
            h = stats.t.ppf(0.975, len(d) - 1) * d.std(ddof=1) / np.sqrt(len(d))
            out[cond][k] = {'mean': float(d.mean()), 'ci95': [float(d.mean() - h), float(d.mean() + h)],
                            'per_seed': d.tolist()}
            print(f'{cond:10s} {k:9s} {d.mean():+6.2f}  [{d.mean() - h:+.2f}, {d.mean() + h:+.2f}]  n={len(d)}')
    # Holm across all comparisons run here (conditions x metrics)
    tests = []
    for cond, v in out.items():
        for k in ('pass1_pp', 'pass32_pp', 'exits'):
            d = np.asarray(v[k]['per_seed'])
            p = float(2 * stats.t.sf(abs(d.mean() / (d.std(ddof=1) / np.sqrt(len(d)))), len(d) - 1))
            v[k]['p'] = p
            tests.append((p, cond, k))
    tests.sort()
    running = 0.0
    for i, (p, cond, k) in enumerate(tests):
        running = max(running, min(1.0, (len(tests) - i) * p))
        out[cond][k]['p_holm'] = running
    print('smallest Holm-adjusted p:', round(tests and out[tests[0][1]][tests[0][2]]['p_holm'], 3))
    path = root / 'analysis' / f'exploratory_vs_vanilla_{a.model_size}.json'
    path.write_text(json.dumps(out, indent=1))
    print('wrote', path)


if __name__ == '__main__':
    main()

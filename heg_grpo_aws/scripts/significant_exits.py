"""Per-problem significance of boundary entries and exits (added for reviewer comment R2.3; descriptive).

An exit counts as *significant* when the base model's successes on that problem (c of 32) differ from the trained
model's 0 of 32 under a two-sided Fisher exact test at p < 0.05; entries likewise in reverse. Under no real change
almost no problem passes (the test is conservative), so these counts separate real losses and gains from sampling
turnover.

    python scripts/significant_exits.py --artifact_root ../results_0.5B
"""
import argparse
import json
from pathlib import Path

import numpy as np
from scipy.stats import fisher_exact

STEM = 'math500_n32_p500'
CONDS = ['vanilla', 'o_self', 'meg', 'cb_grpo', 'h_cb_grpo', 'heg_grpo']


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--artifact_root', required=True)
    ap.add_argument('--model_size', default='0.5B')
    ap.add_argument('--alpha', type=float, default=0.05)
    a = ap.parse_args()
    root = Path(a.artifact_root)
    load = lambda p: np.load(p)['correctness'].astype(int)
    base = load(root / 'baseline' / a.model_size / 'eval' / f'{STEM}.npz')
    n = base.shape[1]
    bc = base.sum(1)
    # smallest base count whose drop to 0/n is significant (and largest trained count for an entry from 0/n)
    thr = next(c for c in range(1, n + 1) if fisher_exact([[c, n - c], [0, n]])[1] < a.alpha)
    out = {'n_samples': n, 'alpha': a.alpha, 'min_count_for_significance': thr,
           'base_problems_with_count_ge_thr': int((bc >= thr).sum())}
    # sampling-noise check: base vs base on two halves cannot be tested at n=32, so report the test's
    # false-positive behaviour by splitting the base samples and testing halves (n=16 each)
    h1, h2 = base[:, :16].sum(1), base[:, 16:].sum(1)
    fp = sum(1 for x, y in zip(h1, h2) if (x == 0) != (y == 0) and fisher_exact([[x, 16 - x], [y, 16 - y]])[1] < a.alpha)
    out['base_half_vs_half_significant_flips'] = fp
    for cond in CONDS:
        ex, en, ex_all, en_all = [], [], [], []
        for p in sorted(root.glob(f'runs/{cond}/{a.model_size}/seed*/eval/{STEM}.npz')):
            rc = load(p).sum(1)
            e_all = (bc > 0) & (rc == 0); n_all = (bc == 0) & (rc > 0)
            ex_all.append(int(e_all.sum())); en_all.append(int(n_all.sum()))
            ex.append(int((e_all & (bc >= thr)).sum()))
            en.append(int((n_all & (rc >= thr)).sum()))
        out[cond] = {'exits_all': float(np.mean(ex_all)), 'exits_significant': float(np.mean(ex)),
                     'entries_all': float(np.mean(en_all)), 'entries_significant': float(np.mean(en)),
                     'per_seed_exits_significant': ex, 'per_seed_entries_significant': en}
    print(f"significance needs >= {thr}/{n} successes on one side and 0/{n} on the other (Fisher p < {a.alpha})")
    print(f"base half-vs-half (n=16) significant flips: {fp}")
    for cond in CONDS:
        v = out[cond]
        print(f"{cond:10s} exits {v['exits_all']:5.1f} -> significant {v['exits_significant']:4.1f} | "
              f"entries {v['entries_all']:5.1f} -> significant {v['entries_significant']:4.1f}")
    path = root / 'analysis' / f'significant_exits_{a.model_size}.json'
    path.write_text(json.dumps(out, indent=1))
    print('wrote', path)


if __name__ == '__main__':
    main()

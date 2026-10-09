"""Planning numbers for the seed count: given the between-seed spread we actually observed, what contrast
can n seeds detect, and how wide would a 95% interval be?

This is a design calculation, not a prediction of any result: it says nothing about what a contrast will be,
only how small a difference a given number of seeds could separate from zero.

    python scripts/power.py --artifact_root ../results_0.5B

For each contrast, the per-seed value d_s (a linear combination of the seeds' shrinkage slopes) has standard
deviation sd across the seeds that the contrast's conditions share. With n seeds the paired t-test detects
|true effect| >= (t_{1-a/2,n-1} + t_{1-b,n-1}) * sd / sqrt(n)  (alpha = 0.05 two-sided, power 1-b = 0.80),
and the 95% t-interval has half-width t_{0.975,n-1} * sd / sqrt(n). Slope units are pp per ln k; one unit of
slope = ln(32) = 3.47 points of gain from k=1 to k=32.

Caveats printed with the table: sd comes from only 2-3 seeds (itself uncertain), and the bootstrap interval in
analyze.py also counts problem sampling, so it is the more conservative of the two.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np
from scipy import stats

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from analyze import CONTRASTS  # noqa: E402
from rlvr.config import DEFAULT_ARTIFACT_ROOT  # noqa: E402

LN32 = math.log(32)


def mde(sd: float, n: int, alpha: float = 0.05, power: float = 0.80) -> float:
    return float((stats.t.ppf(1 - alpha / 2, n - 1) + stats.t.ppf(power, n - 1)) * sd / math.sqrt(n))


def half_width(sd: float, n: int, alpha: float = 0.05) -> float:
    return float(stats.t.ppf(1 - alpha / 2, n - 1) * sd / math.sqrt(n))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--artifact_root', default=DEFAULT_ARTIFACT_ROOT)
    ap.add_argument('--model_size', default='0.5B')
    ap.add_argument('--benchmark', default='math500')
    ap.add_argument('--n_seeds', default='3,5', help='comma list of seed counts to tabulate')
    args = ap.parse_args()
    root = Path(args.artifact_root)
    summ = json.loads((root / 'analysis' / f'{args.benchmark}_{args.model_size}.json').read_text())
    slopes = {c: dict(zip(v['seeds'], v['slopes'])) for c, v in summ['conditions'].items()}
    ns = [int(x) for x in args.n_seeds.split(',')]

    out = [f'# Seed planning from the observed between-seed spread ({args.benchmark}, {args.model_size})', '',
           'Effects in pp per ln k (x ln 32 = points of gain from k=1 to k=32). MDE = minimum detectable effect at '
           '80% power, two-sided alpha 0.05; half-width = half of the 95% t-interval.', '',
           '| contrast | seeds used | sd of per-seed value | ' + ' | '.join(
               f'MDE n={n} (pp/ln k | Pass@32 pts) | half-width n={n}' for n in ns) + ' |',
           '|---|---|---|' + '---|' * (2 * len(ns))]
    res = {}
    for name, (coefs, role) in CONTRASTS.items():
        if not all(c in slopes for c in coefs):
            continue
        seeds = sorted(set.intersection(*(set(slopes[c]) for c in coefs)))
        if len(seeds) < 2:
            continue
        d = np.array([sum(w * slopes[c][s] for c, w in coefs.items()) for s in seeds]) * 100  # pp per ln k
        sd = float(d.std(ddof=1))
        cells = []
        res[name] = {'role': role, 'n_seeds_observed': len(seeds), 'sd_pp_per_lnk': sd}
        for n in ns:
            m, h = mde(sd, n), half_width(sd, n)
            res[name][f'n{n}'] = {'mde_pp_per_lnk': m, 'mde_pass32_points': m * LN32, 'half_width_pp_per_lnk': h}
            cells += [f'{m:.2f} \\| {m * LN32:.1f}', f'{h:.2f}']
        out.append(f'| {name} | {len(seeds)} | {sd:.2f} | ' + ' | '.join(cells) + ' |')
    out += ['', 'Interactions (S1, S2) combine four conditions, so their per-seed sd is the largest and they gain '
            'most from extra seeds. sd rests on 2-3 seeds, so every number here is itself uncertain.']
    out_dir = root / 'analysis'
    (out_dir / f'power_{args.model_size}.md').write_text('\n'.join(out) + '\n', encoding='utf-8')
    (out_dir / f'power_{args.model_size}.json').write_text(json.dumps(res, indent=2))
    print('\n'.join(out))


if __name__ == '__main__':
    main()

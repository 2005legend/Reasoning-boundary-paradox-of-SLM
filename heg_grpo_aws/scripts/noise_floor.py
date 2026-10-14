"""Sampling-noise floor for boundary entries/exits (added after the extension runs; descriptive).

Splits the base model's 32 samples per problem into two halves of 16 and counts how many problems one half solves
and the other does not (pure sampling turnover, same policy). Trained runs are compared with each base half using
their first 16 samples, so both numbers use n=16.

    python scripts/noise_floor.py --artifact_root ../results_0.5B
"""
import argparse
import json
from pathlib import Path

import numpy as np

STEM = 'math500_n32_p500'
CONDS = ['vanilla', 'o_self', 'meg', 'cb_grpo', 'h_cb_grpo', 'heg_grpo']


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--artifact_root', required=True)
    ap.add_argument('--model_size', default='0.5B')
    a = ap.parse_args()
    root = Path(a.artifact_root)
    load = lambda p: np.load(p)['correctness'].astype(int)
    base = load(root / 'baseline' / a.model_size / 'eval' / f'{STEM}.npz')
    halves = [base[:, :16].sum(1) > 0, base[:, 16:].sum(1) > 0]
    out = {'base_vs_base': {'exits': int((halves[0] & ~halves[1]).sum()), 'entries': int((~halves[0] & halves[1]).sum())}}
    for cond in CONDS:
        ex, en = [], []
        for p in sorted(root.glob(f'runs/{cond}/{a.model_size}/seed*/eval/{STEM}.npz')):
            run = load(p)[:, :16].sum(1) > 0
            for h in halves:
                ex.append(int((h & ~run).sum()))
                en.append(int((~h & run).sum()))
        out[cond] = {'exits': float(np.mean(ex)), 'entries': float(np.mean(en)), 'n_runs': len(ex) // 2}
    for k, v in out.items():
        print(f'{k:13s} exits {v["exits"]:5.1f}  entries {v["entries"]:5.1f}')
    path = root / 'analysis' / f'noise_floor_{a.model_size}.json'
    path.write_text(json.dumps(out, indent=1))
    print('wrote', path)


if __name__ == '__main__':
    main()

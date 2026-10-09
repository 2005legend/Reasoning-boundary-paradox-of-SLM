"""Aggregate final evals into the paper's tables (Req 16, 42; PREREGISTRATION.md).

    python analyze.py --model_size 0.5B --benchmark math500

Outcome: the shrinkage slope of (run Pass@k - base Pass@k) against log k (higher = less shrinkage).
Contrasts are linear combinations of conditions, paired by seed (seed s shares data order and LoRA
init across conditions). Two CIs per contrast: a t-interval over seeds, and a hierarchical
bootstrap that resamples seeds and problems jointly (the same problem indices for every condition
and the base model). Primary contrasts P1/P2 get Holm-adjusted bootstrap p-values.
"""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy import stats
from scipy.special import comb

from rlvr.config import DEFAULT_ARTIFACT_ROOT, EvalConfig, baseline_dir_for

# name -> ({condition: coefficient}, role). Fixed before any main run (PREREGISTRATION.md).
CONTRASTS = {
    'P1  heg_grpo - h_cb_grpo': ({'heg_grpo': 1, 'h_cb_grpo': -1}, 'primary'),
    'P2  heg_grpo - meg': ({'heg_grpo': 1, 'meg': -1}, 'primary'),
    'S1  CBxMEG interaction': ({'heg_grpo': 1, 'cb_grpo': -1, 'meg': -1, 'vanilla': 1}, 'secondary'),
    'S2  CBxSELF interaction': ({'h_cb_grpo': 1, 'cb_grpo': -1, 'o_self': -1, 'vanilla': 1}, 'secondary'),
    'S3  CB main effect': ({'cb_grpo': 1 / 3, 'h_cb_grpo': 1 / 3, 'heg_grpo': 1 / 3,
                            'vanilla': -1 / 3, 'o_self': -1 / 3, 'meg': -1 / 3}, 'secondary'),
    'E1  heg_grpo - vanilla': ({'heg_grpo': 1, 'vanilla': -1}, 'exploratory'),
    'E2  heg_grpo - heg_grpo__randpart': ({'heg_grpo': 1, 'heg_grpo__randpart': -1}, 'exploratory'),
    'E3  heg_grpo - bbg': ({'heg_grpo': 1, 'bbg': -1}, 'exploratory'),
}


def eval_stem(benchmark: str) -> str:
    s = EvalConfig().settings_for(benchmark)
    return f"{benchmark}_n{s['n_samples']}_p{s['n_problems']}"


def per_problem_pass_at_k(correctness: np.ndarray, ks: list) -> np.ndarray:
    """[P, len(ks)] unbiased per-problem Pass@k, so any problem resample is a row mean."""
    n = correctness.shape[1]
    c = correctness.sum(axis=1).astype(int)
    out = np.zeros((len(c), len(ks)))
    for j, k in enumerate(ks):
        denom = comb(n, k, exact=True)
        out[:, j] = [1.0 if n - ci < k else 1.0 - comb(n - ci, k, exact=True) / denom for ci in c]
    return out


def slopes_from_rows(run_rows: np.ndarray, base_rows: np.ndarray, log_k: np.ndarray) -> float:
    delta = run_rows.mean(axis=0) - base_rows.mean(axis=0)
    lk = log_k - log_k.mean()
    return float((lk * (delta - delta.mean())).sum() / (lk ** 2).sum())


def subject_labels(benchmark: str, prompt_ids):
    """MATH-500 subject of each evaluated problem ('math500_<row>'), for the per-topic breakdown."""
    if benchmark != 'math500':
        return None
    try:
        from datasets import load_dataset
        rows = load_dataset('HuggingFaceH4/MATH-500', split='test')
        return np.array([rows[int(str(pid).rsplit('_', 1)[1])]['subject'] for pid in prompt_ids])
    except Exception as e:  # the breakdown is optional; never fail the main analysis over it
        print(f'(no per-subject breakdown: {type(e).__name__}: {e})')
        return None


def mean_from_log(run_dir: Path, key: str):
    path = run_dir / 'train_log.jsonl'
    if not path.exists():
        return None
    vals = [json.loads(l).get(key) for l in path.read_text().splitlines() if l.strip()]
    vals = [v for v in vals if v is not None]
    return float(np.mean(vals)) if vals else None


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--model_size', default='0.5B')
    p.add_argument('--benchmark', default='math500')
    p.add_argument('--artifact_root', default=DEFAULT_ARTIFACT_ROOT)
    p.add_argument('--n_boot', type=int, default=2000)
    p.add_argument('--seed', type=int, default=0)
    args = p.parse_args()

    root = Path(args.artifact_root)
    stem = eval_stem(args.benchmark)
    base_dir = baseline_dir_for(args.artifact_root, args.model_size) / 'eval'
    if not (base_dir / f'{stem}.json').exists():
        raise SystemExit(f'Missing base-model eval {base_dir / stem}.json; run evaluate.py --base first.')
    base = json.loads((base_dir / f'{stem}.json').read_text())
    ks = [int(k) for k in base['settings']['k_values']]
    log_k = np.log(np.array(ks, dtype=np.float64))
    base_npz = np.load(base_dir / f'{stem}.npz')
    base_rows = per_problem_pass_at_k(base_npz['correctness'], ks)
    base_ids = base_npz['prompt_ids']

    rows, slope, pass_k, info = defaultdict(dict), defaultdict(dict), defaultdict(dict), {}
    for run_json in sorted((root / 'runs').glob(f'*/{args.model_size}/seed*/eval/{stem}.json')):
        cond, seed = run_json.parents[3].name, int(run_json.parents[1].name[4:])
        res = json.loads(run_json.read_text())
        npz = np.load(run_json.with_suffix('.npz'))
        if res['settings'] != base['settings'] or not np.array_equal(npz['prompt_ids'], base_ids):
            print(f'SKIP {cond}/seed{seed}: eval settings or problem set differ from the base-model eval')
            continue
        rows[cond][seed] = per_problem_pass_at_k(npz['correctness'], ks)
        slope[cond][seed] = slopes_from_rows(rows[cond][seed], base_rows, log_k)
        pass_k[cond][seed] = rows[cond][seed].mean(axis=0)
        run_dir = run_json.parents[1]
        done = json.loads((run_dir / 'DONE.json').read_text()) if (run_dir / 'DONE.json').exists() else {}
        c_base, c_run = base_npz['correctness'].sum(1), npz['correctness'].sum(1)
        info.setdefault(cond, defaultdict(list))
        info[cond]['entered'].append(int(((c_base == 0) & (c_run > 0)).sum()))
        info[cond]['exited'].append(int(((c_base > 0) & (c_run == 0)).sum()))
        info[cond]['pos_w'].append(mean_from_log(run_dir, 'mean_pos_weight'))
        info[cond]['entropy'].append(mean_from_log(run_dir, 'mean_group_mode_entropy'))
        info[cond]['rollouts_per_problem'].append(done.get('rollouts_per_training_problem'))
    if not slope:
        raise SystemExit('No finished run evals found.')

    def fmt(x, spec='.3f'):
        vals = [v for v in (x if isinstance(x, list) else [x]) if v is not None]
        return format(float(np.mean(vals)), spec) if vals else '-'

    n_samples = base['settings']['n_samples']
    out = [f'# {args.benchmark} @ {args.model_size}  (n = {n_samples} samples/problem, '
           f'{len(base_ids)} problems)', '',
           '| condition | seeds | ' + ' | '.join(f'Pass@{k}' for k in ks)
           + ' | slope | entered / exited | pos. weight | mode entropy | rollouts/problem |',
           '|---|---|' + '---|' * len(ks) + '---|---|---|---|---|',
           '| base | - | ' + ' | '.join(f'{v:.3f}' for v in base_rows.mean(axis=0)) + ' | - | - | - | - | - |']
    summary = {'base_pass_at_k': dict(zip(ks, base_rows.mean(axis=0).tolist())), 'conditions': {}, 'contrasts': {}}
    for cond in sorted(slope):
        seeds = sorted(slope[cond])
        mpk = np.mean([pass_k[cond][s] for s in seeds], axis=0)
        ent, ext = info[cond]['entered'], info[cond]['exited']
        out.append(f'| {cond} | {len(seeds)} | ' + ' | '.join(f'{v:.3f}' for v in mpk)
                   + f" | {np.mean([slope[cond][s] for s in seeds]):+.4f} | {np.mean(ent):.1f} / {np.mean(ext):.1f}"
                   f" | {fmt(info[cond]['pos_w'])} | {fmt(info[cond]['entropy'])}"
                   f" | {fmt(info[cond]['rollouts_per_problem'], '.1f')} |")
        summary['conditions'][cond] = {'seeds': seeds, 'slopes': [slope[cond][s] for s in seeds],
                                       'mean_pass_at_k': dict(zip(ks, mpk.tolist())),
                                       'pass_at_k_by_seed': {s: pass_k[cond][s].tolist() for s in seeds},
                                       'entered': ent, 'exited': ext,
                                       'pos_weight': info[cond]['pos_w'], 'mode_entropy': info[cond]['entropy'],
                                       'rollouts_per_problem': info[cond]['rollouts_per_problem']}

    subj = subject_labels(args.benchmark, base_ids)
    if subj is not None:  # does shrinkage concentrate in some topics? (the macro gate acts on topics)
        names = sorted(set(subj.tolist()))
        ends = {1: ks.index(1), ks[-1]: len(ks) - 1}
        bys = {'subjects': names, 'n': [int((subj == n).sum()) for n in names], 'delta': {}}
        out += ['', f'## Change vs base by MATH subject (pp), Pass@1 / Pass@{ks[-1]}', '',
                '| condition | ' + ' | '.join(names) + ' |', '|---|' + '---|' * len(names)]
        for cond in sorted(rows):
            d = {}
            for k, j in ends.items():
                per_seed = [[float((rows[cond][s][subj == n, j] - base_rows[subj == n, j]).mean()) for n in names]
                            for s in sorted(rows[cond])]
                d[f'pass@{k}'] = np.mean(per_seed, axis=0).tolist()
            bys['delta'][cond] = d
            out.append(f'| {cond} | ' + ' | '.join(f"{100 * a:+.1f} / {100 * b:+.1f}" for a, b in
                                                  zip(d['pass@1'], d[f'pass@{ks[-1]}'])) + ' |')
        summary['by_subject'] = bys

    rng = np.random.default_rng(args.seed)
    n_prob = len(base_ids)
    out += ['', '## Contrasts on the shrinkage slope (paired by seed)', '',
            '| contrast | role | seeds | estimate | 95% CI (seed t) | 95% CI (seeds x problems) | p (Holm for P) |',
            '|---|---|---|---|---|---|---|']
    primary_p = {}
    for name, (coefs, role) in CONTRASTS.items():
        if not all(c in slope for c in coefs):
            continue
        seeds = sorted(set.intersection(*(set(slope[c]) for c in coefs)))
        if len(seeds) < 2:
            out.append(f'| {name} | {role} | {len(seeds)} | needs >= 2 shared seeds | | | |')
            continue
        d = np.array([sum(w * slope[c][s] for c, w in coefs.items()) for s in seeds])
        half = stats.t.ppf(0.975, len(d) - 1) * d.std(ddof=1) / np.sqrt(len(d))
        boot = np.empty(args.n_boot)
        for b in range(args.n_boot):
            ss = rng.choice(seeds, size=len(seeds), replace=True)
            pi = rng.integers(0, n_prob, size=n_prob)
            br = base_rows[pi]
            boot[b] = np.mean([sum(w * slopes_from_rows(rows[c][s][pi], br, log_k) for c, w in coefs.items())
                               for s in ss])
        lo, hi = np.percentile(boot, [2.5, 97.5])
        p_boot = float(min(1.0, 2 * min((boot <= 0).mean(), (boot >= 0).mean())))
        entry = {'role': role, 'seeds': seeds, 'estimate': float(d.mean()),
                 't_ci': [float(d.mean() - half), float(d.mean() + half)],
                 'boot_ci': [float(lo), float(hi)], 'p_boot': p_boot}
        summary['contrasts'][name] = entry
        if role == 'primary':
            primary_p[name] = p_boot
    # Holm over the primaries
    for i, (name, p_raw) in enumerate(sorted(primary_p.items(), key=lambda kv: kv[1])):
        summary['contrasts'][name]['p_holm'] = min(1.0, p_raw * (len(primary_p) - i))
    running = 0.0
    for name, _ in sorted(primary_p.items(), key=lambda kv: kv[1]):  # enforce monotonicity
        running = max(running, summary['contrasts'][name]['p_holm'])
        summary['contrasts'][name]['p_holm'] = running
    for name, e in summary['contrasts'].items():
        p_txt = f"{e['p_holm']:.3f}" if 'p_holm' in e else f"{e['p_boot']:.3f} (unadj.)"
        if e['role'] == 'primary' and e['p_holm'] < 0.05:
            p_txt += ' **'
        out.append(f"| {name} | {e['role']} | {len(e['seeds'])} | {e['estimate']:+.4f} | "
                   f"[{e['t_ci'][0]:+.4f}, {e['t_ci'][1]:+.4f}] | [{e['boot_ci'][0]:+.4f}, {e['boot_ci'][1]:+.4f}] | {p_txt} |")
    out += ['', 'Positive estimate = the first condition shrinks less. Interactions are reported as '
            'bounds (secondary), not verdicts. ** = primary contrast significant after Holm.']

    out_dir = root / 'analysis'
    out_dir.mkdir(parents=True, exist_ok=True)
    fname = f'{args.benchmark}_{args.model_size}'
    (out_dir / f'{fname}.md').write_text('\n'.join(out) + '\n')
    (out_dir / f'{fname}.json').write_text(json.dumps(summary, indent=2))
    print('\n'.join(out))

    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(7, 4.5))
        base_mean = base_rows.mean(axis=0)
        for cond in sorted(pass_k):
            deltas = np.array([pass_k[cond][s] - base_mean for s in sorted(pass_k[cond])])
            ax.errorbar(ks, deltas.mean(0), yerr=deltas.std(0) if len(deltas) > 1 else None,
                        marker='o', capsize=3, label=f'{cond} (n={len(deltas)})')
        ax.axhline(0, color='grey', lw=0.8)
        ax.set_xscale('log', base=2)
        ax.set_xlabel('k')
        ax.set_ylabel('Pass@k - base Pass@k')
        ax.set_title(f'{args.benchmark} @ {args.model_size}: change vs base model')
        ax.legend(fontsize=7)
        fig.tight_layout()
        fig.savefig(out_dir / f'{fname}.png', dpi=150)
        print(f'\nWrote {out_dir / fname}.md/.json/.png')
    except ImportError:
        print(f'\nWrote {out_dir / fname}.md/.json (matplotlib missing: no plot)')


if __name__ == '__main__':
    main()

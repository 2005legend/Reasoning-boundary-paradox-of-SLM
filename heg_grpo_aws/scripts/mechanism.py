"""Mechanism analysis (descriptive, not a pre-registered test): does diversity of correct reasoning track
what each run loses?

For every model (base and each run) it partitions the CORRECT final-eval completions of each MATH-500
problem into solution modes with the calibrated partitioner, then asks, across runs:

  diversity of correct solutions  ->  boundary exits / entries  ->  shrinkage slope

and, per topic, whether the subjects that absorbed the most positive credit during training (CB runs,
where the spend EMA is saved) lost the most Pass@32.

    python scripts/mechanism.py --model_size 0.5B
writes analysis/mechanism_<size>.json and .md. Reads only saved files; safe to run while training.
"""
from __future__ import annotations

import argparse
import gzip
import json
import sys
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from analyze import per_problem_pass_at_k, slopes_from_rows, subject_labels  # noqa: E402
from rlvr.config import DEFAULT_ARTIFACT_ROOT, baseline_dir_for  # noqa: E402
from rlvr.gates import compute_group_partitions  # noqa: E402

KS = [1, 2, 4, 8, 16, 32]
STEM = 'math500_n32_p500'


def read_completions(path: Path):
    rows = [json.loads(l) for l in gzip.open(path, 'rt', encoding='utf-8') if l.strip()]
    return [r['prompt_id'] for r in rows], rows


def mode_stats(rows, threshold: float) -> dict:
    """Per problem: number of correct solution modes, normalized mode entropy and the share of the
    largest mode among correct samples (NaN when fewer than 2 samples are correct)."""
    n = len(rows)
    modes, ent, top = (np.full(n, np.nan) for _ in range(3))
    for i, r in enumerate(rows):
        correct = np.asarray(r['correct'], dtype=bool)
        if correct.sum() < 2:
            continue
        labels, _ = compute_group_partitions(r['completions'], correct, len(correct), method='bigram',
                                             bigram_threshold=threshold)
        counts = np.bincount(labels[labels >= 0])
        counts = counts[counts > 0]
        p = counts / counts.sum()
        modes[i] = len(counts)
        ent[i] = float(-(p * np.log(p)).sum() / np.log(correct.sum()))
        top[i] = float(p.max())
    return {'modes': modes, 'entropy': ent, 'top_share': top}


def train_summary(run_dir: Path) -> dict:
    path = run_dir / 'train_log.jsonl'
    if not path.exists():
        return {}
    recs = [json.loads(l) for l in path.read_text().splitlines() if l.strip()]
    n = max(1, len(recs) // 5)

    def mean(rs, k):
        v = [r[k] for r in rs if r.get(k) is not None]
        return float(np.mean(v)) if v else None
    return {'train_entropy_first': mean(recs[:n], 'mean_group_mode_entropy'),
            'train_entropy_last': mean(recs[-n:], 'mean_group_mode_entropy'),
            'train_modes_last': mean(recs[-n:], 'mean_correct_modes'),
            'kl_last': mean(recs[-n:], 'approx_kl'),
            'pos_weight': mean(recs, 'mean_pos_weight')}


def spend_share(run_dir: Path):
    """CB runs only: final per-subject spend EMA (subjects in sorted order, as train.py assigns ids)."""
    path = run_dir / 'checkpoint' / 'run_state.json'
    if not path.exists():
        return None
    g = json.loads(path.read_text()).get('gate_state', {})
    s = g.get('spend_ema') or g.get('macro', {}).get('spend_ema')
    if not s:
        return None
    s = np.asarray(s, dtype=float)
    return s / s.sum()


def corr(x, y) -> dict:
    x, y = np.asarray(x, float), np.asarray(y, float)
    ok = np.isfinite(x) & np.isfinite(y)
    if ok.sum() < 4:
        return {'n': int(ok.sum()), 'rho': None, 'p': None}
    r = spearmanr(x[ok], y[ok])
    return {'n': int(ok.sum()), 'rho': float(r.statistic), 'p': float(r.pvalue)}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--model_size', default='0.5B')
    ap.add_argument('--artifact_root', default=DEFAULT_ARTIFACT_ROOT)
    args = ap.parse_args()
    root = Path(args.artifact_root)
    chosen = json.loads((root / 'calibration' / args.model_size / 'calibration.json').read_text())['chosen']
    thr = float(chosen.get('meg_bigram_threshold', 0.4))

    base_dir = baseline_dir_for(args.artifact_root, args.model_size) / 'eval'
    base_ids, base_rows_c = read_completions(base_dir / f'{STEM}_completions.jsonl.gz')
    base_corr = np.load(base_dir / f'{STEM}.npz')['correctness']
    base_pk = per_problem_pass_at_k(base_corr, KS)
    log_k = np.log(np.array(KS, float))
    subj = subject_labels('math500', base_ids)
    subjects = sorted(set(subj)) if subj is not None else []
    print(f'base: partitioning {len(base_rows_c)} problems (bigram, threshold {thr})', flush=True)
    base_ms = mode_stats(base_rows_c, thr)

    runs = []
    for ev in sorted(root.glob(f'runs/*/{args.model_size}/seed*/eval/{STEM}.json')):
        run_dir, cond, seed = ev.parents[1], ev.parents[3].name, int(ev.parents[1].name[4:])
        if '__' in cond:  # pilot / smoke runs
            continue
        comp = ev.parent / f'{STEM}_completions.jsonl.gz'
        if not comp.exists():
            continue
        ids, rows_c = read_completions(comp)
        if ids != base_ids:
            print(f'SKIP {cond}/seed{seed}: problem order differs from the base eval')
            continue
        print(f'{cond}/seed{seed}: partitioning', flush=True)
        c = np.load(ev.parent / f'{STEM}.npz')['correctness']
        pk = per_problem_pass_at_k(c, KS)
        ms = mode_stats(rows_c, thr)
        both = np.isfinite(ms['modes']) & np.isfinite(base_ms['modes'])
        base_solved, run_solved = base_corr.sum(1) > 0, c.sum(1) > 0
        rec = {'condition': cond, 'seed': seed,
               'slope': slopes_from_rows(pk, base_pk, log_k),
               'pass1_delta': float(pk[:, 0].mean() - base_pk[:, 0].mean()),
               'pass32_delta': float(pk[:, -1].mean() - base_pk[:, -1].mean()),
               'exits': int((base_solved & ~run_solved).sum()),
               'entries': int((~base_solved & run_solved).sum()),
               'modes': float(np.nanmean(ms['modes'])), 'entropy': float(np.nanmean(ms['entropy'])),
               'top_share': float(np.nanmean(ms['top_share'])),
               # paired with the base model on problems where both have >= 2 correct samples
               'modes_delta': float((ms['modes'] - base_ms['modes'])[both].mean()),
               'top_share_delta': float((ms['top_share'] - base_ms['top_share'])[both].mean()),
               'n_paired': int(both.sum())}
        rec.update(train_summary(run_dir))
        if subj is not None:
            rec['subject_pass32_delta'] = {s: float(pk[subj == s, -1].mean() - base_pk[subj == s, -1].mean())
                                           for s in subjects}
        share = spend_share(run_dir)
        if share is not None and len(share) == len(subjects):
            rec['subject_spend_share'] = dict(zip(subjects, share.tolist()))
        runs.append(rec)

    base_summary = {'modes': float(np.nanmean(base_ms['modes'])), 'entropy': float(np.nanmean(base_ms['entropy'])),
                    'top_share': float(np.nanmean(base_ms['top_share']))}

    def col(k):
        return [r.get(k, np.nan) if r.get(k) is not None else np.nan for r in runs]
    links = {
        'diversity_change -> exits (modes_delta vs exits)': corr(col('modes_delta'), col('exits')),
        'concentration_change -> exits (top_share_delta vs exits)': corr(col('top_share_delta'), col('exits')),
        'exits -> slope': corr(col('exits'), col('slope')),
        'diversity_change -> slope (modes_delta vs slope)': corr(col('modes_delta'), col('slope')),
        'training entropy (last 20%) -> slope': corr(col('train_entropy_last'), col('slope')),
        'policy movement (KL, last 20%) -> slope': corr(col('kl_last'), col('slope')),
    }
    xs, ys = [], []
    for r in runs:
        if 'subject_spend_share' in r and 'subject_pass32_delta' in r:
            for s in subjects:
                xs.append(r['subject_spend_share'][s])
                ys.append(r['subject_pass32_delta'][s])
    links['topic spend share -> topic Pass@32 change (CB runs, pooled)'] = corr(xs, ys)

    by_cond = {}
    for r in runs:
        by_cond.setdefault(r['condition'], []).append(r)
    cond_means = {c: {k: float(np.mean([r[k] for r in rs if r.get(k) is not None]))
                      for k in ('slope', 'exits', 'entries', 'modes', 'modes_delta', 'top_share',
                                'top_share_delta', 'train_entropy_last', 'kl_last')
                      if any(r.get(k) is not None for r in rs)} | {'seeds': sorted(r['seed'] for r in rs)}
                  for c, rs in sorted(by_cond.items())}

    out_dir = root / 'analysis'
    out_dir.mkdir(parents=True, exist_ok=True)
    res = {'partitioner': {'method': 'bigram', 'threshold': thr}, 'base': base_summary,
           'conditions': cond_means, 'links': links, 'runs': runs,
           'note': 'Descriptive, not pre-registered. Spearman correlations across runs; not causal evidence.'}
    (out_dir / f'mechanism_{args.model_size}.json').write_text(json.dumps(res, indent=2))

    md = [f'# Mechanism analysis ({args.model_size}, MATH-500, descriptive)', '',
          f"Base model: {base_summary['modes']:.2f} correct modes per problem, top-mode share "
          f"{base_summary['top_share']:.2f}, normalized entropy {base_summary['entropy']:.2f}.", '',
          '| condition | seeds | slope | exits | entries | modes | d modes vs base | top share | d top share | train ent (last) | KL (last) |',
          '|---|---|---|---|---|---|---|---|---|---|---|']
    for c, m in cond_means.items():
        g = lambda k, f='{:.3f}': f.format(m[k]) if k in m else '-'  # noqa: E731
        md.append(f"| {c} | {len(m['seeds'])} | {g('slope', '{:+.4f}')} | {g('exits', '{:.1f}')} | "
                  f"{g('entries', '{:.1f}')} | {g('modes', '{:.2f}')} | {g('modes_delta', '{:+.3f}')} | "
                  f"{g('top_share')} | {g('top_share_delta', '{:+.3f}')} | {g('train_entropy_last')} | "
                  f"{g('kl_last', '{:.4f}')} |")
    md += ['', '## Links across runs (Spearman)', '', '| link | n | rho | p |', '|---|---|---|---|']
    for k, v in links.items():
        md.append(f"| {k} | {v['n']} | " + ('-' if v['rho'] is None else f"{v['rho']:+.2f}") + ' | '
                  + ('-' if v['p'] is None else f"{v['p']:.3f}") + ' |')
    (out_dir / f'mechanism_{args.model_size}.md').write_text('\n'.join(md) + '\n')
    print('\n'.join(md))


if __name__ == '__main__':
    main()

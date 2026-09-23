"""Blinded human audit of what the automatic measurements claim (PREREGISTRATION.md, "Human evaluation").

    python scripts/human_eval.py sheets                # after the main queue: writes ~/artifacts/human_eval/
    #  -> give annotator_A/ and annotator_B/ to two people (HUMAN_EVAL_GUIDE.md); never share key.json
    python scripts/human_eval.py score                 # after both sheets are filled in

H1  Are the partitioner's "modes" strategies? Pairs of correct base-model solutions to the same MATH
    training problem (from calibration_groups.jsonl), half of them put in the same mode by the
    partitioner and half in different modes. Annotators answer SAME or DIFFERENT approach without seeing
    the partitioner. Reported: Cohen's kappa; accuracy and balanced accuracy of the partitioner against the
    annotators' consensus.
H2  What does RLVR lose? MATH-500 problems the base model solves (>= 1 of 32 samples) but vanilla GRPO never
    solves in a majority of seeds ("exited"), each matched to a retained problem with the closest base
    solve count. One random correct base-model solution per problem; annotators label it VALID or FLAWED
    (right answer, invalid reasoning) without knowing its group. Reported: valid-reasoning rate per group
    with Wilson 95% CIs, two-sided Fisher exact test, Cohen's kappa.
"""
from __future__ import annotations

import argparse
import csv
import gzip
import json
import math
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from rlvr.config import DEFAULT_ARTIFACT_ROOT, EvalConfig, baseline_dir_for, calibration_path  # noqa: E402

H1_LABELS, H2_LABELS = ('SAME', 'DIFFERENT'), ('VALID', 'FLAWED')
UNSURE = 'UNSURE'


# ---------- statistics ----------------------------------------------------------------------------
def cohen_kappa(a: list, b: list) -> float | None:
    """Cohen's kappa for two raters over the same items (labels already filtered to sure answers)."""
    if not a:
        return None
    cats = sorted(set(a) | set(b))
    n = len(a)
    po = sum(x == y for x, y in zip(a, b)) / n
    pe = sum((a.count(c) / n) * (b.count(c) / n) for c in cats)
    return 1.0 if pe == 1 else (po - pe) / (1 - pe)


def wilson(k: int, n: int, z: float = 1.96) -> tuple:
    if n == 0:
        return (None, None)
    p = k / n
    centre = (p + z * z / (2 * n)) / (1 + z * z / n)
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    return (max(0.0, centre - half), min(1.0, centre + half))


def fisher_exact(a: int, b: int, c: int, d: int) -> float:
    """Two-sided Fisher exact p for [[a, b], [c, d]] (sum of tables no more likely than the observed one)."""
    r1, c1, n = a + b, a + c, a + b + c + d

    def prob(x):
        return math.comb(c1, x) * math.comb(n - c1, r1 - x) / math.comb(n, r1)
    p_obs = prob(a)
    lo, hi = max(0, r1 + c1 - n), min(r1, c1)
    return min(1.0, sum(prob(x) for x in range(lo, hi + 1) if prob(x) <= p_obs * (1 + 1e-9)))


# ---------- building the sheets -------------------------------------------------------------------
def read_gz_jsonl(path: Path) -> dict:
    with gzip.open(path, 'rt', encoding='utf-8') as f:
        return {r['prompt_id']: r for r in map(json.loads, f)}


def h1_items(cal_groups: Path, n_pairs: int, rng) -> list:
    same, diff = [], []
    for g in map(json.loads, cal_groups.read_text(encoding='utf-8').splitlines()):
        lab = g['labels_chosen']
        idx = [i for i, ok in enumerate(g['is_correct']) if ok and lab[i] >= 0]
        pairs = [(i, j) for a, i in enumerate(idx) for j in idx[a + 1:]]
        rng.shuffle(pairs)
        for kind, bucket in (('same', same), ('different', diff)):
            chosen = [p for p in pairs if (lab[p[0]] == lab[p[1]]) == (kind == 'same')][:2]  # <= 2 per group
            bucket += [(g, i, j, kind) for i, j in chosen]
    rng.shuffle(same)
    rng.shuffle(diff)
    half = n_pairs // 2
    take = same[:half] + diff[:half]
    rest = same[half:] + diff[half:]
    take += rest[:max(0, n_pairs - len(take))]       # top up from the other kind if one is short
    items = []
    for g, i, j, kind in take:
        a, b = (i, j) if rng.random() < 0.5 else (j, i)
        items.append({'problem': g.get('problem', f"(training problem {g['prompt_id']})"),
                      'solution_A': g['completions'][a], 'solution_B': g['completions'][b],
                      '_key': {'partitioner': kind, 'prompt_id': g['prompt_id'], 'rollouts': [a, b]}})
    return items


def h2_items(root: Path, size: str, n_exit: int, rng) -> tuple:
    s = EvalConfig().settings_for('math500')
    stem = f"math500_n{s['n_samples']}_p{s['n_problems']}"
    base_dir = baseline_dir_for(str(root), size) / 'eval'
    base = np.load(base_dir / f'{stem}.npz')
    ids, base_c = base['prompt_ids'], base['correctness'].sum(1)
    texts = read_gz_jsonl(base_dir / f'{stem}_completions.jsonl.gz')
    van = []
    for npz in sorted((root / 'runs' / 'vanilla' / size).glob(f'seed*/eval/{stem}.npz')):
        d = np.load(npz)
        assert np.array_equal(d['prompt_ids'], ids), f'{npz}: problem order differs from the base eval'
        van.append(d['correctness'].sum(1))
    if not van:
        raise SystemExit('No finished vanilla evals found (runs/vanilla/<size>/seed*/eval).')
    van = np.array(van)
    exited = np.flatnonzero((base_c > 0) & ((van == 0).sum(0) > len(van) / 2))
    retained = list(np.flatnonzero((base_c > 0) & (van > 0).all(0)))
    exited = rng.permutation(exited)[:n_exit]
    items, pairs = [], []
    for e in exited:  # match each exited problem to the unused retained problem with the closest base count
        if not retained:
            break
        gaps = np.array([abs(int(base_c[r]) - int(base_c[e])) for r in retained])
        best = [r for r, g in zip(retained, gaps) if g == gaps.min()]
        r = best[int(rng.integers(len(best)))]
        retained.remove(r)
        pairs.append((int(e), int(r)))
    for e, r in pairs:
        for idx, group in ((e, 'exited'), (r, 'retained')):
            rec = texts[str(ids[idx])]
            ok = [i for i, c in enumerate(rec['correct']) if c]
            pick = ok[int(rng.integers(len(ok)))]
            items.append({'problem': rec['problem'], 'reference_answer': rec['ground_truth'],
                          'solution': rec['completions'][pick],
                          '_key': {'group': group, 'prompt_id': str(ids[idx]), 'base_correct': int(base_c[idx]),
                                   'vanilla_correct_by_seed': van[:, idx].tolist(), 'sample': pick}})
    return items, {'n_exited_total': int(((base_c > 0) & ((van == 0).sum(0) > len(van) / 2)).sum()),
                   'n_vanilla_seeds': len(van)}


def write_sheet(path: Path, items: list, cols: list, label_help: str):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, 'w', newline='', encoding='utf-8-sig') as f:  # utf-8-sig: opens cleanly in Excel
        w = csv.writer(f)
        w.writerow(['item_id'] + cols + ['label', 'notes'])
        for it in items:
            w.writerow([it['item_id']] + [it[c] for c in cols] + ['', ''])
    (path.parent / 'LABELS.txt').write_text(label_help, encoding='utf-8')


def cmd_sheets(a):
    root, out = Path(a.artifact_root), Path(a.out or Path(a.artifact_root) / 'human_eval')
    rng = np.random.default_rng(a.seed)
    h1 = h1_items(calibration_path(str(root), a.model_size).parent / 'calibration_groups.jsonl', a.n_pairs, rng)
    h2, h2_info = h2_items(root, a.model_size, a.n_exit, rng)
    key = {'h1': {}, 'h2': {}, 'h2_info': h2_info, 'seed': a.seed}
    for prefix, items, name in (('P', h1, 'h1'), ('S', h2, 'h2')):
        order = rng.permutation(len(items))
        for n, i in enumerate(order, 1):
            items[i]['item_id'] = f'{prefix}{n:03d}'
            key[name][items[i]['item_id']] = items[i].pop('_key')
        items.sort(key=lambda it: it['item_id'])
    help_text = ('h1_pairs.csv: label = SAME or DIFFERENT (approach), UNSURE if you cannot decide.\n'
                 'h2_solutions.csv: label = VALID or FLAWED (reasoning), UNSURE if you cannot decide.\n'
                 'Definitions and examples: HUMAN_EVAL_GUIDE.md. Work alone; do not discuss items.\n')
    for ann in a.annotators.split(','):
        write_sheet(out / f'annotator_{ann}' / 'h1_pairs.csv', h1, ['problem', 'solution_A', 'solution_B'], help_text)
        write_sheet(out / f'annotator_{ann}' / 'h2_solutions.csv', h2, ['problem', 'reference_answer', 'solution'],
                    help_text)
    (out / 'key.json').write_text(json.dumps(key, indent=2))
    n_ex = sum(v['group'] == 'exited' for v in key['h2'].values())
    print(f'H1: {len(h1)} solution pairs ({sum(v["partitioner"] == "same" for v in key["h1"].values())} '
          f'same-mode by the partitioner). H2: {n_ex} exited + {len(h2) - n_ex} matched retained problems '
          f'({h2_info["n_exited_total"]} exited in total over {h2_info["n_vanilla_seeds"]} vanilla seeds).')
    print(f'Sheets for annotators {a.annotators} in {out}. Keep key.json away from the annotators.')


# ---------- scoring -------------------------------------------------------------------------------
def read_labels(path: Path, allowed: tuple) -> dict:
    out = {}
    with open(path, newline='', encoding='utf-8-sig') as f:
        for row in csv.DictReader(f):
            lab = (row.get('label') or '').strip().upper()
            if lab in allowed or lab == UNSURE:
                out[row['item_id']] = lab
    return out


def consensus(labels_by_ann: list, adjudicated: dict, allowed: tuple) -> dict:
    """Item -> label all annotators agree on (UNSURE never counts); an adjudication file overrides."""
    out = {}
    for item in set().union(*labels_by_ann):
        labs = [la.get(item) for la in labels_by_ann]
        if all(x in allowed for x in labs) and len(set(labs)) == 1:
            out[item] = labs[0]
    out.update({k: v for k, v in adjudicated.items() if v in allowed})
    return out


def pairwise_kappa(labels_by_ann: list, allowed: tuple):
    ks, n = [], 0
    for i in range(len(labels_by_ann)):
        for j in range(i + 1, len(labels_by_ann)):
            common = [k for k in labels_by_ann[i] if labels_by_ann[i][k] in allowed
                      and labels_by_ann[j].get(k) in allowed]
            n = max(n, len(common))
            kv = cohen_kappa([labels_by_ann[i][k] for k in common], [labels_by_ann[j][k] for k in common])
            if kv is not None:
                ks.append(kv)
    return (float(np.mean(ks)) if ks else None), n


def cmd_score(a):
    root = Path(a.artifact_root)
    sheets = Path(a.sheets or root / 'human_eval')
    key = json.loads((sheets / 'key.json').read_text())
    anns = sorted(p for p in sheets.glob('annotator_*') if p.is_dir())
    if len(anns) < 2:
        raise SystemExit(f'Need at least two annotator folders in {sheets}.')
    res = {'annotators': len(anns)}
    for name, fname, allowed in (('h1', 'h1_pairs.csv', H1_LABELS), ('h2', 'h2_solutions.csv', H2_LABELS)):
        labs = [read_labels(p / fname, allowed) for p in anns]
        adj_path = sheets / f'adjudication_{name}.csv'
        adj = read_labels(adj_path, allowed) if adj_path.exists() else {}
        cons = consensus(labs, adj, allowed)
        kappa, n_common = pairwise_kappa(labs, allowed)
        r = {'items': len(key[name]), 'kappa': kappa, 'n_rated_by_all': n_common, 'n_consensus': len(cons),
             'raw_agreement': (float(np.mean([len({la.get(i) for la in labs}) == 1 for i in key[name]]))
                               if key[name] else None)}
        if name == 'h1':
            truth = {i: ('SAME' if key['h1'][i]['partitioner'] == 'same' else 'DIFFERENT') for i in cons}
            hits = [truth[i] == cons[i] for i in cons]
            recall = {}
            for lab in H1_LABELS:
                sel = [i for i in cons if cons[i] == lab]
                recall[lab] = float(np.mean([truth[i] == lab for i in sel])) if sel else None
            r.update({'partitioner_accuracy': float(np.mean(hits)) if hits else None,
                      'partitioner_balanced_accuracy': (float(np.mean([v for v in recall.values() if v is not None]))
                                                        if any(v is not None for v in recall.values()) else None),
                      'human_different_given_partitioner_different': (
                          float(np.mean([cons[i] == 'DIFFERENT' for i in cons if truth[i] == 'DIFFERENT']))
                          if any(truth[i] == 'DIFFERENT' for i in cons) else None),
                      'human_different_given_partitioner_same': (
                          float(np.mean([cons[i] == 'DIFFERENT' for i in cons if truth[i] == 'SAME']))
                          if any(truth[i] == 'SAME' for i in cons) else None)})
        else:
            groups = {}
            for g in ('exited', 'retained'):
                sel = [i for i in cons if key['h2'][i]['group'] == g]
                k = sum(cons[i] == 'VALID' for i in sel)
                groups[g] = {'n': len(sel), 'valid': k, 'rate': (k / len(sel)) if sel else None,
                             'ci': wilson(k, len(sel))}
            e, t = groups['exited'], groups['retained']
            r.update({'groups': groups, 'fisher_p': fisher_exact(e['valid'], e['n'] - e['valid'],
                                                                 t['valid'], t['n'] - t['valid'])
                      if e['n'] and t['n'] else None})
        res[name] = r
    out = root / 'analysis'
    out.mkdir(parents=True, exist_ok=True)
    (out / 'human_eval.json').write_text(json.dumps(res, indent=2))
    h1, h2 = res['h1'], res['h2']
    fmt = (lambda x: '-' if x is None else f'{x:.2f}')
    print(f"H1 partition validity: kappa={fmt(h1['kappa'])}, consensus on {h1['n_consensus']}/{h1['items']} "
          f"pairs, partitioner accuracy={fmt(h1['partitioner_accuracy'])} "
          f"(balanced {fmt(h1['partitioner_balanced_accuracy'])})")
    g = h2['groups']
    print(f"H2 what is lost: kappa={fmt(h2['kappa'])}; valid reasoning exited {g['exited']['valid']}/"
          f"{g['exited']['n']} vs retained {g['retained']['valid']}/{g['retained']['n']}, "
          f"Fisher p={fmt(h2['fisher_p'])}")
    print(f'Wrote {out / "human_eval.json"} (read by paper/fill_numbers.py)')


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest='cmd', required=True)
    s = sub.add_parser('sheets', help='build blinded annotation sheets + key.json')
    s.add_argument('--artifact_root', default=DEFAULT_ARTIFACT_ROOT)
    s.add_argument('--model_size', default='0.5B')
    s.add_argument('--out', default=None, help='default: <artifact_root>/human_eval')
    s.add_argument('--n_pairs', type=int, default=60, help='H1 solution pairs (half same-mode, half different)')
    s.add_argument('--n_exit', type=int, default=40, help='H2 exited problems (each gets a matched retained one)')
    s.add_argument('--annotators', default='A,B')
    s.add_argument('--seed', type=int, default=0)
    c = sub.add_parser('score', help='agreement, partitioner validity, valid-reasoning rates')
    c.add_argument('--artifact_root', default=DEFAULT_ARTIFACT_ROOT)
    c.add_argument('--sheets', default=None, help='default: <artifact_root>/human_eval')
    a = p.parse_args()
    cmd_sheets(a) if a.cmd == 'sheets' else cmd_score(a)


if __name__ == '__main__':
    main()

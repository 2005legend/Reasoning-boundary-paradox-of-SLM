import csv, glob, json, re
from math import comb
import numpy as np
import pymupdf
from scipy import stats
from scipy.stats import fisher_exact

import os, sys
R = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..') + os.sep
PDF = sys.argv[1] if len(sys.argv) > 1 else os.path.join(R, 'paper', 'main.pdf')
NREF = int(sys.argv[2]) if len(sys.argv) > 2 else 25
pdf = pymupdf.open(PDF)
T = ' '.join(' '.join(p.get_text().split()) for p in pdf).replace('ﬁ', 'fi').replace('ﬂ', 'fl')
fails, oks = [], 0
def has(s, what):
    global oks
    if s in T: oks += 1
    else: fails.append(f'{what}: expected "{s}"')

ks = [1, 2, 4, 8, 16, 32]
def load(p): return np.load(p)['correctness'].astype(int)
def passk(c):
    n = c.shape[1]; s = c.sum(1)
    return np.array([[1 - comb(n - x, k) / comb(n, k) if n - x >= k else 1.0 for x in s] for k in ks]).mean(1)
base = load(os.path.join(R, r'results_0.5B\baseline\0.5B\eval\math500_n32_p500.npz')); pb = passk(base); bs = base.sum(1) > 0
lk = np.log(ks); lkc = lk - lk.mean()
def slope(p): d = 100 * (p - pb); return (lkc * (d - d.mean())).sum() / (lkc ** 2).sum()
names = {'vanilla': 'VANILLA', 'o_self': 'SELF', 'meg': 'MEG', 'cb_grpo': 'CB', 'h_cb_grpo': 'CB×SELF', 'heg_grpo': 'CB×MEG'}
S, runs = {}, {}
for c in names:
    fs = sorted(glob.glob(R + rf'results_0.5B\runs\{c}\0.5B\seed*\eval\math500_n32_p500.npz'))
    rr = {int(re.search(r'seed(\d)', f).group(1)): load(f) for f in fs}; runs[c] = rr
    pk = np.mean([passk(r) for r in rr.values()], 0)
    sl = np.mean([slope(passk(r)) for r in rr.values()]); S[c] = {s: slope(passk(r)) for s, r in rr.items()}
    ent = np.mean([((r.sum(1) > 0) & ~bs).sum() for r in rr.values()]); ex = np.mean([(bs & (r.sum(1) == 0)).sum() for r in rr.values()])
    row = f'{names[c]} {100*pk[0]:.1f} {100*pk[2]:.1f} {100*pk[4]:.1f} {100*pk[5]:.1f} {sl:+.2f} {ent:.1f} / {ex:.1f}'
    has(row, f'Table IV row {c}')
has(f'Base model {100*pb[0]:.1f} {100*pb[2]:.1f} {100*pb[4]:.1f} {100*pb[5]:.1f}', 'Table IV base row')
for name, a, b in (('P1', 'heg_grpo', 'h_cb_grpo'), ('P2', 'heg_grpo', 'meg'), ('E1', 'heg_grpo', 'vanilla')):
    common = sorted(set(S[a]) & set(S[b])); est = np.mean([S[a][s] - S[b][s] for s in common])
    has(f'{est:+.2f}'.replace('-', '−'), f'{name} estimate')
# abstract / V-A numbers
has(f'Pass@1 moves from {100*pb[0]:.1f}% to {100*np.mean([passk(r)[0] for r in runs["vanilla"].values()]):.1f}%', 'abstract Pass@1')
has(f'27 models', 'model count'); has('432,000', 'solution count')
assert sum(len(v) for v in runs.values()) == 27 and 27 * 500 * 32 == 432000
# noise floor
A, B = base[:, :16].sum(1) > 0, base[:, 16:].sum(1) > 0
has(f'differ by {(A & ~B).sum()} exits and {(~A & B).sum()} entries', 'noise floor base')
# exploratory
for c, txt in (('cb_grpo', 'CB (95% CI'), ('h_cb_grpo', 'CB×SELF ([')):
    d = np.array([100 * (runs[c][s].mean() - runs['vanilla'][s].mean()) for s in sorted(set(runs[c]) & set(runs['vanilla']))])
    h = stats.t.ppf(.975, len(d) - 1) * d.std(ddof=1) / np.sqrt(len(d))
    has(f'[{d.mean()-h:+.1f}, {d.mean()+h:+.1f}]', f'exploratory CI {c}')
# human audit
he = os.path.join(R, 'results_0.5B', 'human_eval') + os.sep
lab = lambda p: {r['item_id']: r['label'].strip().upper() for r in csv.DictReader(open(p, encoding='utf-8-sig'))}
key = json.load(open(he + 'key.json'))
A1, B1 = lab(he + r'labelled\annotator_A\h1_pairs.csv'), lab(he + r'labelled\annotator_B\h1_pairs.csv')
agree = [i for i in A1 if A1[i] == B1[i]]
has(f'called {sum(v=="SAME" for v in A1.values())} and {sum(v=="SAME" for v in B1.values())} of the 60 pairs', 'H1 counts')
has(f'They agreed on {len(agree)} pairs ({round(100*len(agree)/60)}%)', 'H1 agreement')
part = lambda i: 'SAME' if key['h1'][i]['partitioner'] == 'same' else 'DIFFERENT'
has(f'partitioner scores {round(100*np.mean([part(i)==A1[i] for i in agree]))}%', 'H1 partitioner accuracy')
A2, B2, C2 = lab(he + r'labelled\annotator_A\h2_solutions.csv'), lab(he + r'labelled\annotator_B\h2_solutions.csv'), lab(he + r'labelled\adjudication_h2.csv')
fin = {i: (A2[i] if A2[i] == B2[i] else C2[i]) for i in A2}
tab = [[sum(fin[i] == 'VALID' for i in fin if key['h2'][i]['group'] == g), sum(fin[i] == 'FLAWED' for i in fin if key['h2'][i]['group'] == g)] for g in ('exited', 'retained')]
has(f'{round(100*tab[0][0]/20)}% of exited solutions were valid', 'H2 exited rate')
has(f'{round(100*tab[1][0]/20)}% of retained ones', 'H2 retained rate')
has(f'Fisher p = {fisher_exact(tab)[1]:.3f}', 'H2 Fisher p')
has(f'sided with A on {sum(C2[i]==A2[i] for i in C2)}', 'third annotator agreement')
# authors, acknowledgment, link
for s in ['Sidaarth Krishnakanth', 'Shivam Sunderam', 'Ms. T. Bhanu Shree, M.E., (Ph.D.)', 'Shruti K, BBA', 'Sri Prathin', 'Mukunth',
          'We thank the three annotators', 'An AI assistant (Claude, Anthropic)', 'https://github.com/2005legend/',
          'Index Terms—reinforcement learning with verifiable rewards, GRPO, Pass@k, small language models, human evaluation']:
    has(s, 'front/back matter')
# references
refs = T[T.rfind('REFERENCES'):]
nums = sorted(set(int(x) for x in re.findall(r'(?:^|\s)\[(\d{1,2})\]\s+[A-Z]', refs)))
body = T[:T.rfind('REFERENCES')]
cited = {int(x) for x in re.findall(r'\[(\d{1,2})\]', body) if int(x) <= NREF}
for a, b in re.findall(r'\[(\d{1,2})\]\s*[–-]\s*\[(\d{1,2})\]', body): cited |= set(range(int(a), int(b) + 1))
if nums != list(range(1, NREF + 1)): fails.append(f'reference numbering {nums}')
if set(range(1, NREF + 1)) - cited: fails.append(f'uncited refs {set(range(1,26))-cited}')
for bad in ['[?]', '??', 'TODO', 'Compose?A']:
    if bad in T: fails.append(f'found {bad}')
# format
fonts = {f[3] for p in pdf for f in p.get_fonts()}
nonemb = [f for p in pdf for f in p.get_fonts() if f[1] == 'n/a' or not f[1]]
print('pages', len(pdf), '| page size', tuple(round(x) for x in pdf[0].rect[2:]), '| fonts', len(fonts), '| non-embedded', len(nonemb))
print('checks passed:', oks, '| failed:', len(fails))
for f in fails: print('FAIL', f)

"""Generate numbers.tex and figdata/*.dat for main.tex. Every result number in the paper comes from here.

    python fill_numbers.py --expected                        # pre-registered expectations (red + watermark)
    python fill_numbers.py --artifacts ../aws_artifacts --gpu_hours 160 --usd 165   # real results

--artifacts is the artifacts folder copied back from AWS (heg_grpo_aws/README.md Part I) after running
`python analyze.py --benchmark math500` and `--benchmark gsm8k` there. It must contain
  analysis/math500_0.5B.json, analysis/gsm8k_0.5B.json            (from analyze.py)
  runs/<condition>/0.5B/seed*/{config.json, DONE.json, train_log.jsonl}
  calibration/0.5B/calibration.json
Plain Python 3, no packages needed. Never edit numbers.tex by hand: re-run this script.
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import math
import statistics as st
from pathlib import Path

HERE = Path(__file__).resolve().parent
KS = [1, 2, 4, 8, 16, 32]
# paper key -> condition name used by train.py / analyze.py
CONDS = {'vanilla': 'vanilla', 'self': 'o_self', 'meg': 'meg', 'cb': 'cb_grpo', 'hcb': 'h_cb_grpo',
         'heg': 'heg_grpo', 'rand': 'heg_grpo__randpart', 'bbg': 'bbg'}
CONTRASTS = {  # same definitions as heg_grpo_aws/analyze.py (pre-registered)
    'P1': {'heg': 1, 'hcb': -1}, 'P2': {'heg': 1, 'meg': -1},
    'S1': {'heg': 1, 'cb': -1, 'meg': -1, 'vanilla': 1}, 'S2': {'hcb': 1, 'cb': -1, 'self': -1, 'vanilla': 1},
    'S3': {'cb': 1 / 3, 'hcb': 1 / 3, 'heg': 1 / 3, 'vanilla': -1 / 3, 'self': -1 / 3, 'meg': -1 / 3},
    'E1': {'heg': 1, 'vanilla': -1}, 'E2': {'heg': 1, 'rand': -1}, 'E3': {'heg': 1, 'bbg': -1},
}
MISSING = '--'


# ---------- formatting ----------------------------------------------------------------------------
def pct(x):            # probability -> "30.0"
    return MISSING if x is None else f'{100 * x:.1f}'


def signed(x, nd=2):   # already in display units -> "\ensuremath{-3.02}"
    return MISSING if x is None else '\\ensuremath{' + f'{x:+.{nd}f}'.replace('-', '-') + '}'


def num(x, nd=1):
    return MISSING if x is None else f'{x:.{nd}f}'


def pval(p):
    if p is None:
        return MISSING
    return '\\ensuremath{<}0.001' if p < 0.001 else f'{p:.3f}'


def crossover_k(deltas):
    """k where a Delta-Pass@k curve first crosses zero (log-linear interpolation), as text."""
    for (k1, d1), (k2, d2) in zip(zip(KS, deltas), zip(KS[1:], deltas[1:])):
        if d1 > 0 >= d2:
            return f'{math.exp(math.log(k1) + d1 / (d1 - d2) * (math.log(k2) - math.log(k1))):.0f}'
    return 'none' if deltas[0] <= 0 else f'>{KS[-1]}'


def slope(run, base):
    """OLS slope of (run - base) on ln k, in pp per unit ln k (x100). Same as analyze.py."""
    lk = [math.log(k) for k in KS]
    d = [r - b for r, b in zip(run, base)]
    mk, md = st.mean(lk), st.mean(d)
    return 100 * sum((x - mk) * (y - md) for x, y in zip(lk, d)) / sum((x - mk) ** 2 for x in lk)


# ---------- expected (pre-registered) scenario ----------------------------------------------------
def beta_pass_at_k(a, b, k):
    """Pass@k when per-problem solve probability p ~ Beta(a, b): 1 - B(a, b+k) / B(a, b)."""
    return 1 - math.exp(math.lgamma(b + k) + math.lgamma(a + b) - math.lgamma(b) - math.lgamma(a + b + k))


def curve(p1, p32):
    """Beta(a, b) curve with Pass@1 = p1 and Pass@32 = p32 (bisection on a)."""
    lo, hi = 1e-3, 50.0
    for _ in range(200):
        a = math.sqrt(lo * hi)
        if beta_pass_at_k(a, a * (1 - p1) / p1, 32) < p32:
            lo = a
        else:
            hi = a
    return [beta_pass_at_k(a, a * (1 - p1) / p1, k) for k in KS]


# (Pass@1, Pass@32) anchors. Provenance: claude science/expected_results_forecast.md (Pass@1 +5-7 pp,
# Pass@32 -4 to -8 pp after vanilla GRPO), rescaled to MATH-500 using the Qwen2.5-0.5B-Instruct MATH
# score (~34%, Qwen2.5 report) and our 1024-token, T=0.6 eval. Ordering = the pre-registered hypothesis.
EXPECTED_MATH = {'base': (0.300, 0.715), 'vanilla': (0.365, 0.665), 'self': (0.354, 0.676),
                 'meg': (0.358, 0.689), 'cb': (0.362, 0.670), 'hcb': (0.352, 0.681),
                 'heg': (0.356, 0.701), 'rand': (0.361, 0.672), 'bbg': (0.350, 0.684)}
EXPECTED_GSM = {'base': (0.420, 0.890), 'vanilla': (0.455, 0.872), 'self': (0.450, 0.878),
                'meg': (0.452, 0.882), 'cb': (0.454, 0.874), 'hcb': (0.449, 0.880),
                'heg': (0.452, 0.886), 'rand': (0.454, 0.875), 'bbg': (0.447, 0.881)}
EXPECTED_DIAG = {  # entered, exited (of 500), mean positive gate weight, mean normalized mode entropy
    'vanilla': (8.3, 24.7, 1.00, 0.31), 'self': (9.3, 20.7, 0.55, 0.32), 'meg': (10.0, 18.3, 1.00, 0.42),
    'cb': (8.7, 22.3, 1.00, 0.32), 'hcb': (9.0, 19.7, 0.54, 0.33), 'heg': (10.7, 15.3, 1.00, 0.43),
    'rand': (8.7, 22.0, 1.00, 0.33), 'bbg': (9.7, 19.0, 0.71, 0.33)}
EXPECTED_HALF = {'t': 0.80, 'boot': 0.45}   # placeholder 95% CI half-widths (slope units)
# MATH-500 subjects (counts from HuggingFaceH4/MATH-500) and the code used as a pgfplots coordinate
SUBJECTS = [('Algebra', 'alg', 124), ('Counting & Probability', 'cp', 38), ('Geometry', 'geo', 41),
            ('Intermediate Algebra', 'ia', 97), ('Number Theory', 'nt', 62), ('Prealgebra', 'pa', 82),
            ('Precalculus', 'pc', 56)]
SUBJECT_CONDS = ['vanilla', 'cb', 'meg', 'heg']
EXPECTED_SUBJECT_SHAPE = [0.5, 1.3, 1.5, 1.3, 0.8, 0.4, 1.4]   # relative Pass@32 loss per subject (vanilla)
# calibration candidates: (method, threshold, median modes, multi-mode share)
EXPECTED_CALIB = [('embedding', 0.80, 1.0, 0.05), ('embedding', 0.85, 1.0, 0.10), ('embedding', 0.90, 1.5, 0.35),
                  ('embedding', 0.95, 2.5, 0.70), ('bigram', 0.3, 3.5, 0.92), ('bigram', 0.4, 3.0, 0.78),
                  ('bigram', 0.5, 2.0, 0.55), ('bigram', 0.6, 1.5, 0.35), ('bigram', 0.7, 1.0, 0.15)]


# Human audit (heg_grpo_aws/scripts/human_eval.py). Expected scenario = "mixed": exited problems were
# solved by valid reasoning less often than matched retained ones, but more than half the time.
EXPECTED_HUMAN = {'h1': {'items': 60, 'n_consensus': 51, 'kappa': 0.62, 'partitioner_accuracy': 0.78,
                         'partitioner_balanced_accuracy': 0.77},
                  'h2': {'items': 80, 'kappa': 0.68, 'exited': (23, 40), 'retained': (34, 40), 'n_exited_total': 61}}


def wilson(k, n, z=1.96):
    if not n:
        return (None, None)
    p = k / n
    c, h = (p + z * z / (2 * n)) / (1 + z * z / n), z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    return (max(0.0, c - h), min(1.0, c + h))


def fisher_two_sided(a, b, c, d):
    r1, c1, n = a + b, a + c, a + b + c + d
    prob = (lambda x: math.comb(c1, x) * math.comb(n - c1, r1 - x) / math.comb(n, r1))
    p0 = prob(a)
    return min(1.0, sum(prob(x) for x in range(max(0, r1 + c1 - n), min(r1, c1) + 1) if prob(x) <= p0 * (1 + 1e-9)))


def human_keys(R, h1, h2_groups, h2, total_exited=None):
    """h1/h2: dicts shaped like human_eval.json; h2_groups: {'exited': (valid, n), 'retained': (valid, n)}."""
    pc = (lambda x: MISSING if x is None else f'{100 * x:.0f}')
    R['human.h1.items'] = str(h1.get('items', MISSING))
    R['human.h1.consensus'] = str(h1.get('n_consensus', MISSING))
    R['human.h1.kappa'] = num(h1.get('kappa'), 2)
    R['human.h1.acc'] = pc(h1.get('partitioner_accuracy'))
    R['human.h1.bacc'] = pc(h1.get('partitioner_balanced_accuracy'))
    R['human.h2.items'] = str(h2.get('items', MISSING))
    R['human.h2.kappa'] = num(h2.get('kappa'), 2)
    R['human.h2.exited.total'] = str(total_exited) if total_exited is not None else MISSING
    for g, (k, n) in h2_groups.items():
        lo, hi = wilson(k, n)
        R[f'human.h2.{g}.n'] = str(n)
        R[f'human.h2.{g}.rate'] = pc(k / n if n else None)
        R[f'human.h2.{g}.ci'] = f'[{pc(lo)}, {pc(hi)}]' if n else MISSING
    (ke, ne), (kr, nr) = h2_groups['exited'], h2_groups['retained']
    R['human.h2.p'] = pval(fisher_two_sided(ke, ne - ke, kr, nr - kr)) if ne and nr else MISSING
    # descriptive robustness checks written by scripts/human_eval.py (per annotator, raw agreement)
    cc = h1.get('consensus_counts') or {}
    R['human.h1.cons.same'] = str(cc.get('SAME', MISSING))
    R['human.h1.cons.diff'] = str(cc.get('DIFFERENT', MISSING))
    R['human.h1.rawagree'] = pc(h1.get('raw_agreement'))
    R['human.h2.rawagree'] = pc(h2.get('raw_agreement'))
    # prevalence-adjusted bias-adjusted kappa (binary labels): 2 * raw agreement - 1
    for tag, h in (('h1', h1), ('h2', h2)):
        ra = h.get('raw_agreement')
        R[f'human.{tag}.pabak'] = signed(2 * ra - 1) if ra is not None else MISSING
    # accuracy of a trivial always-SAME labeller against the H1 consensus
    if cc and sum(cc.values()):
        R['human.h1.alwayssame'] = pc(cc.get('SAME', 0) / sum(cc.values()))
    R['human.h2.consensus'] = str(h2.get('n_consensus', MISSING))
    for tag, ann in (('A', 'annotator_A'), ('B', 'annotator_B')):
        a1 = (h1.get('per_annotator') or {}).get(ann)
        if a1:
            R[f'human.h1.{tag}.same'] = str(a1['counts'].get('SAME', MISSING))
            R[f'human.h1.{tag}.acc'] = pc(a1.get('accuracy'))
        a2 = (h2.get('per_annotator') or {}).get(ann)
        if a2:
            R[f'human.h2.{tag}.valid'] = str(a2['counts'].get('VALID', MISSING))
            gr = a2['groups']
            for g in ('exited', 'retained'):
                R[f'human.h2.{tag}.{g}'] = f"{gr[g]['valid']}/{gr[g]['n']}"
            R[f'human.h2.{tag}.p'] = pval(gr['fisher_p']) if gr.get('fisher_p') is not None else MISSING


def calib_keys(R, cands, chosen=None):
    for method, thr, modes, multi in cands:
        tag = f'calib.{"emb" if method == "embedding" else "bi"}.{thr:.2f}'
        R[f'{tag}.modes'] = num(modes) if modes is not None else MISSING
        R[f'{tag}.multi'] = f'{100 * multi:.0f}' if multi is not None else MISSING
        R[f'{tag}.mark'] = '$\\star$' if chosen == (method, round(thr, 2)) else ''


def expected():
    R, fig = {}, {}
    for bench, table in (('math', EXPECTED_MATH), ('gsm', EXPECTED_GSM)):
        curves = {c: curve(*v) for c, v in table.items()}
        base = curves['base']
        slopes = {c: slope(v, base) for c, v in curves.items() if c != 'base'}
        for c, v in curves.items():
            for k, x in zip(KS, v):
                R[f'{bench}.{c}.p{k}'] = pct(x)
        for c, s in slopes.items():
            R[f'{bench}.{c}.slope'] = signed(s)
            R[f'{bench}.{c}.seeds'] = '3'
        ps = {}
        for name, coefs in CONTRASTS.items():
            est = sum(w * slopes[c] for c, w in coefs.items())
            se = EXPECTED_HALF['boot'] / 1.96
            ps[name] = 2 * (1 - 0.5 * (1 + math.erf(abs(est) / se / math.sqrt(2))))
            put_contrast(R, bench, name, est, (est - EXPECTED_HALF['t'], est + EXPECTED_HALF['t']),
                         (est - EXPECTED_HALF['boot'], est + EXPECTED_HALF['boot']), 3)
        holm(R, bench, ps)
        fig[bench] = {c: [(x - b, 0.004) for x, b in zip(v, base)] for c, v in curves.items() if c != 'base'}
    # per-subject change in Pass@32: vanilla follows the shape; CB flattens it (it acts on topics); scaled so
    # the count-weighted mean equals each condition's overall change
    n_tot = sum(n for _, _, n in SUBJECTS)
    subj = {}
    for c in SUBJECT_CONDS:
        flat = {'vanilla': 0.0, 'meg': 0.0, 'cb': 0.4, 'heg': 0.4}[c]
        shape = [(1 - flat) * s + flat for s in EXPECTED_SUBJECT_SHAPE]
        overall = 100 * (curve(*EXPECTED_MATH[c])[-1] - curve(*EXPECTED_MATH['base'])[-1])
        scale = overall / (sum(s * n for s, (_, _, n) in zip(shape, SUBJECTS)) / n_tot)
        subj[c] = [s * scale for s in shape]
    fig['subject'] = subj
    calib_keys(R, EXPECTED_CALIB, chosen=('bigram', 0.4))
    eh = EXPECTED_HUMAN
    human_keys(R, eh['h1'], {'exited': eh['h2']['exited'], 'retained': eh['h2']['retained']}, eh['h2'],
               eh['h2']['n_exited_total'])
    for c, (ent, ext, pw, me) in EXPECTED_DIAG.items():
        R[f'math.{c}.entered'], R[f'math.{c}.exited'] = num(ent), num(ext)
        R[f'math.{c}.posw'], R[f'math.{c}.entropy'] = num(pw, 2), num(me, 2)
    steps = list(range(0, 121, 10))
    decay = {'vanilla': 0.24, 'self': 0.26, 'cb': 0.25, 'hcb': 0.27, 'meg': 0.40, 'heg': 0.41}
    entropy = {c: [0.46 - (0.46 - end) * (1 - math.exp(-s / 36)) / (1 - math.exp(-120 / 36)) for s in steps]
               for c, end in decay.items()}
    R.update({
        'meta.steps': '120', 'meta.lr': '\\ensuremath{1\\times10^{-5}}', 'meta.levels': '1--5',
        'meta.partitioner': 'word-bigram Jaccard, threshold 0.4', 'meta.calib.modes': '3.0',
        'meta.calib.multishare': '90', 'meta.calib.basecor': '38', 'meta.rpp': '2.0', 'meta.epochs': '0.26',
        'meta.selrate': '62', 'meta.trainh': '55', 'meta.gpuh': '95', 'meta.usd': '96', 'meta.nseeds': '3',
        'meta.entropy0': num(entropy['vanilla'][0], 2),
        'meta.crossk': crossover_k([d for d, _ in fig['math']['vanilla']]),
    })
    return R, fig, (steps, entropy), True


# ---------- real results --------------------------------------------------------------------------
def put_contrast(R, bench, name, est, t_ci, b_ci, n):
    R[f'{bench}.{name}.est'] = signed(est)
    R[f'{bench}.{name}.tci'] = f'[{signed(t_ci[0])}, {signed(t_ci[1])}]'
    R[f'{bench}.{name}.bci'] = f'[{signed(b_ci[0])}, {signed(b_ci[1])}]'
    R[f'{bench}.{name}.n'] = str(n)


def holm(R, bench, raw):
    """Holm over P1/P2; other contrasts report the raw bootstrap p-value."""
    prim = sorted(((p, n) for n, p in raw.items() if n.startswith('P')))
    running = 0.0
    for i, (p, n) in enumerate(prim):
        running = max(running, min(1.0, p * (len(prim) - i)))
        R[f'{bench}.{n}.p'] = pval(running)
        R[f'{bench}.{n}.sig'] = 'yes' if running < 0.05 else 'no'
    for n, p in raw.items():
        if not n.startswith('P'):
            R[f'{bench}.{n}.p'] = pval(p)


def load_jsonl(path):
    return [json.loads(l) for l in path.read_text().splitlines() if l.strip()] if path.exists() else []


def real(art: Path, gpu_hours=None, usd=None):
    R, fig = {}, {}
    for bench, fname in (('math', 'math500_0.5B.json'), ('gsm', 'gsm8k_0.5B.json')):
        path = art / 'analysis' / fname
        if not path.exists():
            print(f'WARNING: {path} missing (run analyze.py --benchmark {fname.split("_")[0]}); {bench} left as {MISSING}')
            continue
        a = json.loads(path.read_text())
        base = [a['base_pass_at_k'][str(k)] for k in KS]
        for k, x in zip(KS, base):
            R[f'{bench}.base.p{k}'] = pct(x)
        fig[bench] = {}
        for key, cond in CONDS.items():
            c = a['conditions'].get(cond)
            if not c:
                continue
            for k in KS:
                R[f'{bench}.{key}.p{k}'] = pct(c['mean_pass_at_k'][str(k)])
            R[f'{bench}.{key}.slope'] = signed(100 * st.mean(c['slopes']))
            R[f'{bench}.{key}.seeds'] = str(len(c['seeds']))
            if bench == 'math':
                R[f'math.{key}.entered'] = num(st.mean(c['entered']))
                R[f'math.{key}.exited'] = num(st.mean(c['exited']))
                for fld, out, nd in (('pos_weight', 'posw', 2), ('mode_entropy', 'entropy', 2)):
                    vals = [v for v in c.get(fld, []) if v is not None]
                    R[f'math.{key}.{out}'] = num(st.mean(vals), nd) if vals else MISSING
            by_seed = c.get('pass_at_k_by_seed', {})
            deltas = [[x - b for x, b in zip(v, base)] for v in by_seed.values()]
            if deltas:
                fig[bench][key] = [(st.mean(col), st.pstdev(col) if len(col) > 1 else 0.0) for col in zip(*deltas)]
            if bench == 'math' and key == 'vanilla':
                R['meta.crossk'] = crossover_k([c['mean_pass_at_k'][str(k)] - b for k, b in zip(KS, base)])
        bys = a.get('by_subject')
        if bench == 'math' and bys:
            kmax = f'pass@{KS[-1]}'
            idx = {name: i for i, name in enumerate(bys['subjects'])}
            fig['subject'] = {key: [100 * bys['delta'][CONDS[key]][kmax][idx[s]] for s, _, _ in SUBJECTS]
                              for key in SUBJECT_CONDS if CONDS[key] in bys['delta']}
        raw = {}
        for name, e in a['contrasts'].items():
            short = name.split()[0]
            put_contrast(R, bench, short, 100 * e['estimate'], [100 * x for x in e['t_ci']],
                         [100 * x for x in e['boot_ci']], len(e['seeds']))
            raw[short] = e.get('p_boot')
            # the same bootstrap interval as a change in the k=1 -> k=32 gain (slope x ln 32), in points
            g = [100 * x * math.log(KS[-1]) for x in e['boot_ci']]
            R[f'{bench}.{short}.g32'] = f'[{signed(g[0], 1)}, {signed(g[1], 1)}]'
            R[f'{bench}.{short}.equiv'] = 'yes' if e.get('equivalent_to_zero') else 'no'
            if 'sesoi' in e:
                R['meta.sesoi'] = num(100 * e['sesoi'], 1)
                R['meta.sesoi32'] = num(100 * e['sesoi'] * math.log(KS[-1]), 1)
        holm(R, bench, raw)
        if bench == 'math':  # between-seed spread of the slope (descriptive; n = 2-3 seeds)
            for key, cond in CONDS.items():
                c = a['conditions'].get(cond)
                if c and len(c['slopes']) > 1:
                    R[f'math.{key}.slopesd'] = num(100 * st.stdev(c['slopes']), 2)

    hpath = art / 'analysis' / 'human_eval.json'
    if hpath.exists():
        h = json.loads(hpath.read_text())
        g = h['h2']['groups']
        key_path = art / 'human_eval' / 'key.json'
        total = json.loads(key_path.read_text())['h2_info']['n_exited_total'] if key_path.exists() else None
        human_keys(R, h['h1'], {k: (g[k]['valid'], g[k]['n']) for k in ('exited', 'retained')}, h['h2'], total)
    else:
        print(f'NOTE: {hpath} missing (run scripts/human_eval.py score); human-audit numbers left as {MISSING}')

    # training metadata, diagnostics over training, compute
    runs = art / 'runs'
    cfg_files = sorted(runs.glob('heg_grpo/0.5B/seed*/config.json')) or sorted(runs.glob('*/0.5B/seed*/config.json'))
    if cfg_files:
        cfg = json.loads(cfg_files[0].read_text())
        R['meta.steps'] = str(cfg.get('steps_override') or cfg.get('training_steps', MISSING))
        lr = cfg.get('optim', {}).get('learning_rate')
        if lr:
            m, e = f'{lr:.0e}'.split('e')
            R['meta.lr'] = f'\\ensuremath{{{int(m)}\\times10^{{{int(e)}}}}}'
        lv = cfg.get('math_levels') or []
        R['meta.levels'] = f'{min(lv)}--{max(lv)}' if lv else MISSING
    cal = art / 'calibration' / '0.5B' / 'calibration.json'
    if cal.exists():
        c = json.loads(cal.read_text())
        ch = c['chosen']
        R['meta.partitioner'] = (f"word-bigram Jaccard, threshold {ch['meg_bigram_threshold']}"
                                 if ch['meg_partition'] == 'bigram'
                                 else f"MiniLM cosine, threshold {ch.get('meg_tau_mode')}")
        best = next((x for x in c['candidates'] if x['method'] == ch['meg_partition']
                     and x.get('meg_bigram_threshold', x.get('meg_tau_mode')) ==
                     ch.get('meg_bigram_threshold', ch.get('meg_tau_mode'))), None)
        if best:
            R['meta.calib.modes'] = num(best['median_modes'])
            R['meta.calib.multishare'] = f"{100 * best['frac_multi_mode']:.0f}"
        R['meta.calib.basecor'] = f"{100 * c['base_correctness']:.0f}"
        calib_keys(R, [(x['method'], x.get('meg_bigram_threshold', x.get('meg_tau_mode')), x['median_modes'],
                        x['frac_multi_mode']) for x in c['candidates']],
                   chosen=(ch['meg_partition'], round(ch.get('meg_bigram_threshold', ch.get('meg_tau_mode')), 2)))
    # tagged runs (e.g. vanilla__s240 from the registered extension) are not part of the main study's averages
    done = [json.loads(p.read_text()) for p in runs.glob('*/0.5B/seed*/DONE.json') if '__' not in p.parts[-4]]
    if done:
        def mean_of(key):
            vals = [d[key] for d in done if d.get(key) is not None]
            return st.mean(vals) if vals else None
        R['meta.rpp'] = num(mean_of('rollouts_per_training_problem'))
        R['meta.epochs'] = num(mean_of('epochs'), 2)
        train_h = sum((d.get('steps') or 0) * (d.get('median_step_seconds') or 0) for d in done) / 3600
        R['meta.trainh'] = f'{train_h:.0f}' if train_h else MISSING
    sel = [r['frac_prompts_selected'] for p in runs.glob('o_self/0.5B/seed*/train_log.jsonl')
           for r in load_jsonl(p) if r.get('frac_prompts_selected') is not None]
    if sel:
        R['meta.selrate'] = f'{100 * st.mean(sel):.0f}'
    R['meta.nseeds'] = str(max((int(v) for k, v in R.items() if k.endswith('.seeds')), default=0))
    # Total GPU-hours (training + all evals) and spend are not in the logs: pass them from the AWS bill.
    R['meta.gpuh'] = f'{gpu_hours:.0f}' if gpu_hours else MISSING
    R['meta.usd'] = f'{usd:.0f}' if usd else MISSING
    entropy, steps = {}, None
    for key in ('vanilla', 'self', 'cb', 'hcb', 'meg', 'heg'):
        per_seed = []
        for p in sorted(runs.glob(f'{CONDS[key]}/0.5B/seed*/train_log.jsonl')):
            pts = {r['step']: r['mean_group_mode_entropy'] for r in load_jsonl(p)
                   if r.get('mean_group_mode_entropy') is not None and 'step' in r}
            if pts:
                per_seed.append(pts)
        if per_seed:
            common = sorted(set.intersection(*(set(s) for s in per_seed)))
            stride = max(1, len(common) // 40)
            common = common[::stride]
            entropy[key] = [st.mean(s[x] for s in per_seed) for x in common]
            steps = common if steps is None or len(common) < len(steps) else steps
    if entropy and steps:
        entropy = {k: v[:len(steps)] for k, v in entropy.items() if len(v) >= len(steps)}
        R['meta.entropy0'] = num(st.mean(v[0] for v in entropy.values()), 2)

    # seed planning (scripts/power.py): what the observed between-seed spread lets 3 or 5 seeds detect
    ppath = art / 'analysis' / 'power_0.5B.json'
    if ppath.exists():
        pw = json.loads(ppath.read_text())
        for short, v in pw.items():
            tag = short.split()[0]
            R[f'pow.{tag}.sd'] = num(v['sd_pp_per_lnk'], 2)
            for n in (3, 5):
                if f'n{n}' in v:
                    R[f'pow.{tag}.mde{n}'] = num(v[f'n{n}']['mde_pp_per_lnk'], 1)
                    R[f'pow.{tag}.mde{n}pts'] = num(v[f'n{n}']['mde_pass32_points'], 1)
                    R[f'pow.{tag}.hw{n}'] = num(v[f'n{n}']['half_width_pp_per_lnk'], 2)
    else:
        print(f'NOTE: {ppath} missing (run scripts/power.py); power numbers left as {MISSING}')

    # descriptive mechanism analysis (scripts/mechanism.py): correct-solution diversity at evaluation
    mpath = art / 'analysis' / 'mechanism_0.5B.json'
    if mpath.exists():
        m = json.loads(mpath.read_text())
        R['mech.base.modes'] = num(m['base']['modes'], 2)
        R['mech.base.top'] = num(100 * m['base']['top_share'], 0)
        for key, cond in CONDS.items():
            c = m['conditions'].get(cond)
            if c:
                R[f'mech.{key}.modes'] = num(c['modes'], 2)
                R[f'mech.{key}.top'] = num(100 * c['top_share'], 0)
        names = {'div_exits': 'diversity_change -> exits', 'conc_exits': 'concentration_change -> exits',
                 'exits_slope': 'exits -> slope', 'div_slope': 'diversity_change -> slope',
                 'ent_slope': 'training entropy', 'kl_slope': 'policy movement', 'topic': 'topic spend share'}
        # Benjamini-Hochberg over all the descriptive links reported by mechanism.py
        items = [(k, v['p']) for k, v in m['links'].items() if v.get('p') is not None]
        order = sorted(range(len(items)), key=lambda i: items[i][1])
        adj, running = {}, 1.0
        for rank in range(len(items) - 1, -1, -1):
            k, p = items[order[rank]]
            running = min(running, p * len(items) / (rank + 1))
            adj[k] = running
        for short, prefix in names.items():
            key = next((k for k in m['links'] if k.startswith(prefix)), None)
            link = m['links'].get(key) if key else None
            if link and link.get('rho') is not None:
                R[f'mech.{short}.rho'] = signed(link['rho'])
                R[f'mech.{short}.p'] = f"{link['p']:.2f}"
                R[f'mech.{short}.padj'] = f"{adj[key]:.2f}"
                R[f'mech.{short}.n'] = str(link['n'])
        R['mech.nlinks'] = str(len(items))
    else:
        print(f'NOTE: {mpath} missing (run scripts/mechanism.py); mechanism numbers left as {MISSING}')
    return R, fig, (steps or [], entropy), False


# ---------- writers -------------------------------------------------------------------------------
def write_all(R, fig, ent, placeholder, source):
    subj = fig.pop('subject', None)
    for c in SUBJECT_CONDS:  # per-subject values are also text keys, e.g. \R{subj.geo.vanilla}
        for i, (_, code, _) in enumerate(SUBJECTS):
            R[f'subj.{code}.{c}'] = signed(subj[c][i], 1) if subj and c in subj else MISSING
    lines = [f'% AUTO-GENERATED by fill_numbers.py ({source}) on {dt.date.today()}. Do not edit by hand.',
             f'\\placeholderdata{"true" if placeholder else "false"}']
    lines += [f'\\setR{{{k}}}{{{v}}}' for k, v in sorted(R.items())]
    (HERE / 'numbers.tex').write_text('\n'.join(lines) + '\n', encoding='utf-8')
    out = HERE / 'figdata'
    out.mkdir(exist_ok=True)
    order = ['vanilla', 'self', 'meg', 'cb', 'hcb', 'heg']
    if subj:
        cols = [c for c in SUBJECT_CONDS if c in subj]
        rows = ['subj n ' + ' '.join(cols)]
        rows += [f'{code} {n} ' + ' '.join(f'{subj[c][i]:.3f}' for c in cols)
                 for i, (_, code, n) in enumerate(SUBJECTS)]
        (out / 'subject_math.dat').write_text('\n'.join(rows) + '\n')
    for bench, data in fig.items():
        cols = [c for c in order if c in data]
        rows = ['k ' + ' '.join(f'{c} {c}_err' for c in cols)]
        for i, k in enumerate(KS):
            rows.append(f'{k} ' + ' '.join(f'{100 * data[c][i][0]:.3f} {100 * data[c][i][1]:.3f}' for c in cols))
        (out / f'delta_{bench}.dat').write_text('\n'.join(rows) + '\n')
    steps, entropy = ent
    if steps and entropy:
        cols = [c for c in order if c in entropy]
        rows = ['step ' + ' '.join(cols)]
        rows += [f'{s} ' + ' '.join(f'{entropy[c][i]:.4f}' for c in cols) for i, s in enumerate(steps)]
        (out / 'entropy.dat').write_text('\n'.join(rows) + '\n')
    print(f'Wrote numbers.tex ({len(R)} values, placeholder={placeholder}) and figdata/ for: '
          f'{", ".join(fig)}{", entropy" if steps else ""}')


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    g = p.add_mutually_exclusive_group(required=True)
    g.add_argument('--expected', action='store_true', help='write the pre-registered expected scenario')
    g.add_argument('--artifacts', type=Path, help='artifacts folder copied back from AWS')
    p.add_argument('--gpu_hours', type=float, help='total A10G hours, training + evals (AWS Cost Explorer)')
    p.add_argument('--usd', type=float, help='total spend in USD (AWS Billing), for the acknowledgment')
    a = p.parse_args()
    if a.expected:
        write_all(*expected(), source='--expected')
    else:
        R, fig, ent, ph = real(a.artifacts.resolve(), a.gpu_hours, a.usd)
        exp_keys = set(expected()[0])
        missing = sorted(exp_keys - set(R))
        for k in missing:
            R[k] = '' if k.endswith('.mark') else MISSING
        if missing:
            print(f'NOTE: {len(missing)} values not found in the artifacts are set to "{MISSING}" '
                  f'(e.g. extras not run): {", ".join(missing[:12])}{" ..." if len(missing) > 12 else ""}')
        write_all(R, fig, ent, ph, source=f'--artifacts {a.artifacts}')


if __name__ == '__main__':
    main()

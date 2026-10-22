"""Upgraded ICITIIT 2027 submission (39 references) with the improvements made for the IDCIoT-144 camera-ready.

    python tools/make_icitiit_v2.py   -> main_icitiit_v2.tex (+ main_icitiit_v2_blind.tex). Run from paper/.
ICITIIT limit: 6 pages including references."""
import os
D = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..') + os.sep
t = open(D + 'main_icitiit.tex', encoding='utf-8').read()

EDITS = [
# space: drop the three most peripheral background references (36 remain)
('replace', r""" Other work changes the optimizer, the rollout search or the advantage
function~\cite{ba2026es,yu2026datpo,decugis2026fade}. At SLM scale, GRPO dynamics~\cite{ghosh2026dynamics},
difficulty scaling~\cite{yadav2026difficulty} and reward granularity~\cite{palandye2026granularity} have been
studied, but not shrinkage.""", r""""""),
('after', r"""ProRL~\cite{liu2025prorl} reports expansion after prolonged, KL-controlled training""", r""" of a 1.5B model"""),
('after', r"""reshape advantages using diversity or structure within the group.""",
 r""" Cue-GRPO, ReCo and DRA-GRPO were evaluated on
1.5B--8B models; whether they help at 0.5B is open."""),
('silent', r"""\subsection{Shrinkage at 0.5B}""", r"""\subsection{Shrinkage at 0.5B}\label{sec:shrink}"""),
('after', r"""Exits are at the noise level and
entries exceed it by about five problems.""",
 r""" Tested per problem (two-sided Fisher exact
test, $p<0.05$, which needs at least 6 of 32 successes against none), at most one entry or exit per run is
significant (0.0--0.4 exits and 0.0--1.0 entries per condition on average)."""),
('after', r"""of its first three seeds (the audit was drawn before the extension runs).""",
 r""" The base model had solved each of them in
only 1--3 of 32 samples, too few for any single loss to be significant."""),
('replace', r"""\textbf{Limitations.} One model at one size, LoRA, one RL algorithm and short training (\R{meta.steps} steps, cut from
a planned 240), evaluated only on MATH-500. Four or five seeds per cell leave the interactions open, and each run has one full evaluation, so we show no
trajectory over training. SELF omits its
forward-KL term and, by skipping solved prompts, takes fewer effective updates than vanilla. No random-partition
control for MEG or random-weight control for CB was run. The annotators differed in strictness on H2, and the
majority follows the stricter two. The correlations above are descriptive and were added after the
pre-registration.""",
 r"""\textbf{Limitations and future work.} One model, LoRA, one RL algorithm, \R{meta.steps} training steps and one
benchmark leave other model sizes and datasets untested; a larger model, GSM8K and AIME come first. The training-intensity reading is untested and needs the registered
240-step runs or more rollouts per problem. Our SELF omits the forward-KL term and takes fewer effective updates than
vanilla. Without random-weight (CB) and random-partition (MEG) controls, the Pass@1 gain cannot be attributed to
topic-specific allocation, and MEG needs an LLM-judge or reasoning-step partitioner to be judged fairly. Four or five
seeds leave the interactions open, one evaluation per run shows no trajectory, and at 32 samples boundary changes are
almost all noise (Sec.~\ref{sec:shrink}); 128 or 256 samples would resolve single problems. The annotators differed in
strictness on H2, and the correlations above are descriptive and post hoc."""),
('replace', r"""AWS credits (\R{meta.gpuh} A10G GPU-hours, about
\$\R{meta.usd}) came from the Amazon ML Challenge 2026.""",
 r"""AWS credits for placing among the top 500 teams in the Amazon ML Challenge 2026 paid for
\R{meta.gpuh} A10G GPU-hours (about \$172 in all, with pilots and one failed instance)."""),
]
SPLITS = [
("agreed pairs to share one approach; the word-overlap partitioner", "agreed pairs to share one approach. The word-overlap partitioner"),
("low training intensity; we did not vary it.", "low training intensity, which we did not vary."),
("tested together; ReCo combines", "tested together. ReCo combines"),
("restores the mean; we adopt its rule", "restores the mean. We adopt its rule"),
("token-level policy gradient; there is no KL penalty,", "token-level policy gradient. There is no KL penalty,"),
("in past steps; the micro gate sees", "in past steps. The micro gate sees"),
("not how much; an earlier", "not how much. An earlier"),
("$\\ln k$; a negative slope", "$\\ln k$. A negative slope"),
("main effect (S3); E1 is", "main effect (S3). E1 is"),
("definition of~\\cite{lee2026strategy}; modes count", "definition of~\\cite{lee2026strategy}. Modes count"),
("dated entries; it was first committed", "dated entries. It was first committed"),
("(16 runs); we then added", "(16 runs). We then added"),
("grow with $k$; the budgeted conditions start higher.", "grow with $k$, while the budgeted conditions start higher."),
("after seeing the data; the interactions do not meet it.", "after seeing the data. The interactions do not meet it."),
("its topic signal; a random-weight control", "its topic signal. A random-weight control"),
("budget reaches; we varied neither", "budget reaches. We varied neither"),
("could not afford it; we predicted", "could not afford it. We predicted"),
("parts of the text; the authors checked", "parts of the text. The authors checked"),
]
for kind, a, b in EDITS:
    assert t.count(a) == 1, (kind, t.count(a), a[:70])
    t = t.replace(a, a + b if kind == 'after' else b)
for a, b in (SPLITS if os.environ.get('SPLITS') == '1' else []):   # cosmetic; off by default to fit 6 pages
    if t.count(a) == 1:
        t = t.replace(a, b)
    else:
        print('split skipped:', a[:50])
open(D + 'main_icitiit_v2.tex', 'w', encoding='utf-8').write(t)
open(D + 'main_icitiit_v2_blind.tex', 'w', encoding='utf-8').write('\\def\\BLINDBUILD{}\n\\input{main_icitiit_v2.tex}\n')
print('ok:', len(EDITS), 'edits')

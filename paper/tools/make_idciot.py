"""Build the IDCIoT-144 camera-ready (main_idciot.tex) from the accepted main.tex: reviewer-driven text changes plus the
sentence splits from the style revision. Run from paper/:  python tools/make_idciot.py"""
import os
D = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..') + os.sep
t = open(D + 'main.tex', encoding='utf-8').read()

REVIEW = [
# R1.4: prior methods' model sizes and budgets (from their abstracts)
(r"""ProRL~\cite{liu2025prorl} reports expansion after prolonged, KL-controlled training, Wen et""",
 r"""ProRL~\cite{liu2025prorl} reports expansion after prolonged, KL-controlled training of a 1.5B model, Wen et"""),
(r"""DRA-GRPO~\cite{chen2025dragrpo} adjusts rewards by the semantic density of the group.""",
 r"""DRA-GRPO~\cite{chen2025dragrpo} adjusts rewards by the semantic density of the group. Their reported evaluations
use 1.5B--8B models (Qwen2.5-Math, Llama-3.1-8B, DeepSeek-R1-Distill-Qwen-1.5B), so whether they help a 0.5B model,
whose correct solutions may rarely differ in approach, is open."""),
# R1.1, R1.3, R1.5, R2.1-R2.5: limitations and future work, one item per comment
(r"""\textbf{Limitations.} One model at one size, LoRA, one RL algorithm and short training (\R{meta.steps} steps, cut from
a planned 240), evaluated only on MATH-500. Four or five seeds per cell leave the interactions open, and each run has one full evaluation, so we show no
trajectory over training. SELF omits its
forward-KL term and, by skipping solved prompts, takes fewer effective updates than vanilla. No random-partition
control for MEG or random-weight control for CB was run. The annotators differed in strictness on H2, and the
majority follows the stricter two. The correlations above are descriptive and were added after the
pre-registration.""",
 r"""\textbf{Limitations and future work.} We study one model (Qwen2.5-0.5B-Instruct), with LoRA, one RL algorithm,
short training (\R{meta.steps} steps, cut from a planned 240) and one benchmark (MATH-500). The results therefore say
nothing yet about other model sizes or datasets; repeating the design on a larger model and on GSM8K and AIME is the
first extension. Four or five seeds per cell leave the interactions open, and each run has one full evaluation, so we
show no trajectory over training. The training-intensity explanation is untested. The registered 240-step runs, or
training on a small subset to raise rollouts per problem, would test it directly. Our SELF is the selection rule
alone: it omits the forward-KL term and, by skipping solved prompts, takes fewer effective updates than vanilla, so
the full objective may behave differently. Boundary entries and exits at 32 samples are almost all sampling noise
(Sec.~\ref{sec:shrink}); larger budgets (128 or 256 samples) are needed to detect real changes on single problems. Both partitioners we tried fail to find
approaches, so MEG-style redistribution needs an LLM-judge or reasoning-step partitioner before it can be judged
fairly. We ran no random-weight control for CB and no random-partition control for MEG, so the Pass@1 gain cannot be
attributed to topic-specific allocation. The annotators differed in strictness on H2, and the majority follows the
stricter two. The correlations above are descriptive and were added after the pre-registration."""),
(r"""\subsection{Shrinkage at 0.5B}""", r"""\subsection{Shrinkage at 0.5B}\label{sec:shrink}"""),
# compact reference formatting (same entries; refs_icitiit.bib = refs.bib with short venue names)
(r"""\bibliography{IEEEabrv,refs}""", r"""\bibliography{IEEEabrv,refs_icitiit}"""),
# R2.3: per-problem significance of entries and exits (scripts/significant_exits.py)
(r"""Exits are at the noise level and
entries exceed it by about five problems.""",
 r"""Exits are at the noise level and
entries exceed it by about five problems. Tested one problem at a time (two-sided Fisher exact test, $p<0.05$, which
needs at least 6 of 32 successes on one side and none on the other), at most one entry or exit per run is
significant: 0.0--0.4 exits and 0.0--1.0 entries per condition on average."""),
(r"""of its first three seeds (the audit was drawn before the extension runs).""",
 r"""of its first three seeds (the audit was drawn before the extension runs). The base model had solved each of them in
only 1--3 of 32 samples, too few for any single loss to be significant."""),
# R1.2: no claim that the topic weighting itself helps
(r"""People judged the
partitioner's modes to be wording variants of one approach.""",
 r"""The topic budget's small Pass@1 gain is not shown to come from its topic weighting. People judged the
partitioner's modes to be wording variants of one approach."""),
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
("could not afford it; our prediction", "could not afford it. Our prediction"),
("parts of the text; the authors checked", "parts of the text. The authors checked"),
]
for a, b in REVIEW:
    n = t.count(a)
    assert n == 1, ('review', n, a[:80])
    t = t.replace(a, b)
done = 0
for a, b in SPLITS:
    if t.count(a) == 1:
        t = t.replace(a, b); done += 1
    else:
        print('split not applied (text differs in this version):', a[:50])
open(D + 'main_idciot.tex', 'w', encoding='utf-8').write(t)
print('ok:', len(REVIEW), 'review edits,', done, 'sentence splits')

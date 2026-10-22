"""Build the IDCIoT-144 camera-ready from the accepted main.tex.

    python tools/make_idciot.py      -> main_idciot.tex (camera-ready) and main_idciot_highlighted.tex
                                        (every reviewer-driven change in blue). Run from paper/.

Edit kinds: 'after' appends text after an anchor, 'replace' swaps a passage (the new text is highlighted whole),
'silent' is a technical change that is not highlighted (labels, reference formatting, sentence splits)."""
import os
D = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..') + os.sep
SRC = open(D + 'main.tex', encoding='utf-8').read()

EDITS = [
# --- Reviewer 1.4: model sizes and budgets of prior methods (from their arXiv abstracts) -------------------------
('after', r"""ProRL~\cite{liu2025prorl} reports expansion after prolonged, KL-controlled training""",
 r""" of a 1.5B model"""),
('after', r"""DRA-GRPO~\cite{chen2025dragrpo} adjusts rewards by the semantic density of the group.""",
 r""" These three methods were evaluated
on 1.5B--8B models (Qwen2.5-Math, Llama-3.1-8B, DeepSeek-R1-Distill-Qwen-1.5B); whether they help a 0.5B model,
whose correct solutions may rarely differ in approach, is open."""),
# --- Reviewer 2.3: sampling noise of entries and exits (scripts/significant_exits.py) ------------------------------
('silent', r"""\subsection{Shrinkage at 0.5B}""", r"""\subsection{Shrinkage at 0.5B}\label{sec:shrink}"""),
('after', r"""Exits are at the noise level and
entries exceed it by about five problems.""",
 r""" That small excess is spread thinly. Tested one problem at a time (two-sided
Fisher exact test, $p<0.05$, which needs at least 6 of 32 successes on one side and none on the other), at most one
entry or exit per run is significant: 0.0--0.4 exits and 0.0--1.0 entries per condition on average."""),
('after', r"""of its first three seeds (the audit was drawn before the extension runs).""",
 r""" The base model had solved each of them in
only 1--3 of 32 samples, too few for any single loss to be significant."""),
('after', r"""gave 5 of 6 against 2 of 9, $p=0.041$.)""",
 r""" Together with Sec.~\ref{sec:shrink},
this suggests the lost set consists mostly of rarely solved problems that sampling moves in and out of the boundary."""),
# --- Reviewers 1.1, 1.3, 1.5, 2.1-2.5: limitations and future work, grouped by theme -------------------------------
('replace', r"""\textbf{Limitations.} One model at one size, LoRA, one RL algorithm and short training (\R{meta.steps} steps, cut from
a planned 240), evaluated only on MATH-500. Four or five seeds per cell leave the interactions open, and each run has one full evaluation, so we show no
trajectory over training. SELF omits its
forward-KL term and, by skipping solved prompts, takes fewer effective updates than vanilla. No random-partition
control for MEG or random-weight control for CB was run. The annotators differed in strictness on H2, and the
majority follows the stricter two. The correlations above are descriptive and were added after the
pre-registration.""",
 r"""\textbf{Limitations and future work.} The study is narrow in scope. It uses one model (Qwen2.5-0.5B-Instruct),
LoRA, one RL algorithm, short training (\R{meta.steps} steps, cut from a planned 240) and one benchmark (MATH-500), so it
says nothing yet about other model sizes or datasets. The first extension is the same design on a larger model and on
GSM8K and AIME. The study also cannot test its own explanation: the training-intensity reading needs the registered
240-step runs, or training on a small subset to raise rollouts per problem. Three gaps limit what the gate results mean. Our SELF is the selection rule alone, without the forward-KL term and
with fewer effective updates than vanilla, so the full objective may behave differently. We ran no random-weight
control for CB and no random-partition control for MEG, so the Pass@1 gain cannot be attributed to topic-specific
allocation. Both partitioners we tried fail to find approaches, so MEG-style redistribution needs an LLM-judge or
reasoning-step partitioner before it can be judged fairly. The measurements are also coarse. Four or five seeds per cell leave the interactions open, and one evaluation per
run shows no trajectory over training. At 32 samples, boundary entries and exits are almost all sampling noise
(Sec.~\ref{sec:shrink}); detecting real changes on single problems needs 128 or 256 samples. The annotators differed in
strictness on H2, and the majority follows the stricter two. The correlations above are descriptive and were added
after the pre-registration."""),
# --- Reviewer 1.2: no claim that the topic weighting itself helps ---------------------------------------------------
('after', r"""their combination moves the shrinkage slope by more than about one point per unit $\ln k$}.""",
 r""" The topic budget's small Pass@1 gain is not shown to
come from its topic weighting."""),
# --- Acknowledgment: credit source and the true total cost -----------------------------------------------------------
('replace', r"""AWS credits came from the Amazon ML Challenge 2026. AWS compute: \R{meta.gpuh}
A10G GPU-hours, about \$\R{meta.usd}.""",
 r"""This work used AWS credits awarded to our team for placing among the top 500 teams in the
Amazon ML Challenge 2026. The reported runs took \R{meta.gpuh} A10G GPU-hours; total AWS spend, including pilots and one
failed instance, was about \$172."""),
# --- formatting only ---------------------------------------------------------------------------------------------------
('silent', r"""\bibliography{IEEEabrv,refs}""", r"""\bibliography{IEEEabrv,refs_icitiit}"""),
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


def build(highlight):
    t = SRC
    for kind, a, b in EDITS:
        assert t.count(a) == 1, (kind, t.count(a), a[:70])
        if kind == 'after':
            t = t.replace(a, a + (r'\textcolor{blue}{' + b + '}' if highlight else b))
        elif kind == 'replace':
            t = t.replace(a, (r'{\color{blue}' + b + '}' if highlight else b))
        else:
            t = t.replace(a, b)
    for a, b in SPLITS:
        if t.count(a) == 1:
            t = t.replace(a, b)
        else:
            print('split skipped:', a[:50])
    return t


open(D + 'main_idciot.tex', 'w', encoding='utf-8').write(build(False))
open(D + 'main_idciot_highlighted.tex', 'w', encoding='utf-8').write(build(True))
print('ok:', len(EDITS), 'edits,', len(SPLITS), 'splits; wrote main_idciot.tex and main_idciot_highlighted.tex')

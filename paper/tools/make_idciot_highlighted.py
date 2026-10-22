"""Highlighted copy of the IDCIoT-144 camera-ready: reviewer-driven additions in blue. Run from paper/."""
import os, re
D = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..') + os.sep
t = open(D + 'main_idciot.tex', encoding='utf-8').read()
B = lambda s: r'\textcolor{blue}{' + s + '}'
R = [
(r"""prolonged, KL-controlled training of a 1.5B model, Wen et""",
 r"""prolonged, KL-controlled training """ + B('of a 1.5B model') + r""", Wen et"""),
(r"""Their reported evaluations
use 1.5B--8B models (Qwen2.5-Math, Llama-3.1-8B, DeepSeek-R1-Distill-Qwen-1.5B), so whether they help a 0.5B model,
whose correct solutions may rarely differ in approach, is open.""",
 B(r"""Their reported evaluations
use 1.5B--8B models (Qwen2.5-Math, Llama-3.1-8B, DeepSeek-R1-Distill-Qwen-1.5B), so whether they help a 0.5B model,
whose correct solutions may rarely differ in approach, is open.""")),
(r"""Tested one problem at a time (two-sided Fisher exact test, $p<0.05$, which
needs at least 6 of 32 successes on one side and none on the other), at most one entry or exit per run is
significant: 0.0--0.4 exits and 0.0--1.0 entries per condition on average.""",
 B(r"""Tested one problem at a time (two-sided Fisher exact test, $p<0.05$, which
needs at least 6 of 32 successes on one side and none on the other), at most one entry or exit per run is
significant: 0.0--0.4 exits and 0.0--1.0 entries per condition on average.""")),
(r"""The base model had solved each of them in
only 1--3 of 32 samples, too few for any single loss to be significant.""",
 B(r"""The base model had solved each of them in
only 1--3 of 32 samples, too few for any single loss to be significant.""")),
(r"""The topic budget's small Pass@1 gain is not shown to come from its topic weighting.""",
 B(r"""The topic budget's small Pass@1 gain is not shown to come from its topic weighting.""")),
]
for a, b in R:
    assert t.count(a) == 1, a[:60]
    t = t.replace(a, b)
# whole rewritten Limitations paragraph in blue
i = t.index(r'\textbf{Limitations and future work.}')
j = t.index(r'pre-registration.', i) + len(r'pre-registration.')
t = t[:i] + r'{\color{blue}' + t[i:j] + '}' + t[j:]
open(D + 'main_idciot_highlighted.tex', 'w', encoding='utf-8').write(t)
print('ok')

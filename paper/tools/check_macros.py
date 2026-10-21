import re
import os, sys
P = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..') + os.sep
tex = open(P + (sys.argv[1] if len(sys.argv) > 1 else 'main.tex'), encoding='utf-8').read()
num = open(P + 'numbers.tex', encoding='utf-8').read()
used = sorted(set(re.findall(r'\\R\{([^}]*)\}', tex)) - {'key'})
vals = {}
for line in num.splitlines():
    m = re.match(r'\\setR\{([^}]*)\}\{(.*)\}\s*$', line)
    if m:
        vals[m.group(1)] = m.group(2)
missing = [u for u in used if u not in vals]
dash = [u for u in used if vals.get(u) == '--']
print('macros used:', len(used), '| defined:', len(vals), '| undefined:', missing, '| "--" placeholders used:', dash)

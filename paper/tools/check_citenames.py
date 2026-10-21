import re, sys
import pymupdf
f = sys.argv[1]
d = pymupdf.open(f)
T = ' '.join(' '.join(p.get_text().split()) for p in d).replace('ﬁ', 'fi')
body, refs = T[:T.rfind('REFERENCES')], T[T.rfind('REFERENCES'):]
ref = {}
for m in re.finditer(r'(?:^|\s)\[(\d{1,2})\]\s+(.*?)(?=\s\[\d{1,2}\]\s+[A-Z]|$)', refs):
    ref[int(m.group(1))] = m.group(2)
# names/methods used in the text -> a string that must appear in that reference entry
expect = {'Yue': 'Yue', 'Nguyen': 'Nguyen', 'Wu': 'Wu', 'Zhou and Li': 'Q. Zhou', 'Zhou': 'T. Zhou', 'ProRL': 'ProRL',
          'Wen': 'Wen', 'Yao': 'Yao', 'Yuan': 'Yuan', 'Dragoi': 'Dragoi', 'Lee': 'Lee', 'Yang': 'Yang',
          'Taşaltı': 'Hüdaverdi', 'Cue-GRPO': 'Rarity', 'ReCo': 'ReCo', 'GCPO': 'Breaking winner', 'DRA-GRPO': 'DRA-GRPO',
          'EDAS': 'error diversity', 'TreeAdv': 'TreeAdv', 'MT-GRPO': 'Multi-task GRPO', 'CurveRL': 'CurveRL',
          'GRPO': 'DeepSeekMath', 'Math-Verify': 'Math-Verify', 'MiniLM (all-MiniLM-L6-v2)': 'MiniLM'}
bad, n = [], 0
for name, must in expect.items():
    pat = re.escape(name).replace('Taşaltı', r'Ta\S*alt\S*') + r'(?: et al\.)?\s*\[(\d{1,2})\]'
    for m in re.finditer(pat, body):
        k = int(m.group(1)); n += 1
        if must.lower() not in ref.get(k, '').lower():
            bad.append(f'{name} [{k}] -> {ref.get(k, "MISSING")[:60]}')
print(f'{f.split(chr(92))[-1]}: {len(ref)} reference entries, {n} named citations checked, mismatches: {bad or "none"}')

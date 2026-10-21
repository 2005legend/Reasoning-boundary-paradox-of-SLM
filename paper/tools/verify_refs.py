import re, time, urllib.request, difflib, xml.etree.ElementTree as ET
import os, sys
P = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..') + os.sep
tex = open(P + (sys.argv[1] if len(sys.argv) > 1 else 'main.tex'), encoding='utf-8').read()
cited = set()
for m in re.finditer(r'\\cite\{([^}]*)\}', tex):
    cited.update(k.strip() for k in m.group(1).split(','))
bib = open(P + (sys.argv[2] if len(sys.argv) > 2 else 'refs.bib'), encoding='utf-8').read()
entries = {}
for m in re.finditer(r'@(\w+)\{([^,]+),(.*?)\n\}', bib, re.S):
    key, body = m.group(2).strip(), m.group(3)
    f = {k.lower(): v for k, v in re.findall(r'(\w+)\s*=\s*\{(.*?)\},?\s*$', body, re.M)}
    entries[key] = f
def clean(s): return ' '.join(re.sub(r'[^a-z0-9 ]', ' ', s.replace('\\', '').replace('{', '').replace('}', '').lower()).split())
ns = {'a': 'http://www.w3.org/2005/Atom'}
arx = {k: re.search(r'arXiv:([\d.]+)', e.get('journal', '')).group(1) for k, e in entries.items()
       if k in cited and re.search(r'arXiv:([\d.]+)', e.get('journal', ''))}
got = {}
ids = list(arx.values())
for i in range(0, len(ids), 15):
    url = 'http://export.arxiv.org/api/query?max_results=50&id_list=' + ','.join(ids[i:i + 15])
    root = ET.fromstring(urllib.request.urlopen(url, timeout=60).read())
    for en in root.findall('a:entry', ns):
        aid = re.sub(r'v\d+$', '', en.find('a:id', ns).text.rsplit('/', 1)[1])
        got[aid] = (' '.join(en.find('a:title', ns).text.split()),
                    [a.find('a:name', ns).text for a in en.findall('a:author', ns)],
                    en.find('a:published', ns).text[:4])
    time.sleep(3)
print(f'{len(cited)} cited keys; {len(arx)} with arXiv IDs\n')
for k in sorted(cited):
    e = entries.get(k)
    if e is None:
        print('MISSING IN BIB', k); continue
    if k not in arx:
        print(f'[non-arXiv] {k}: {e.get("title")} | {e.get("journal", e.get("howpublished", ""))}'); continue
    aid = arx[k]; r = got.get(aid)
    if not r:
        print('NOT FOUND ON ARXIV', k, aid); continue
    sim = difflib.SequenceMatcher(None, clean(e['title']), clean(r[0])).ratio()
    first_bib = clean(e['author'].split(' and ')[0]).split()
    first_arx = clean(r[1][0]).split()
    au_ok = bool(set(first_bib) & set(first_arx)) or 'qwen' in first_bib
    yr_ok = abs(int(e['year']) - int(r[2])) <= 1
    status = 'OK ' if sim > 0.9 and au_ok and yr_ok else 'CHECK'
    print(f'{status} {aid}  title-match={sim:.2f}  first-author={r[1][0]}  year={r[2]}  | {r[0][:70]}')

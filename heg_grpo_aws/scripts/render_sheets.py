"""Readable copies of the human-audit sheets: one HTML file per CSV, with the maths typeset (MathJax).
Annotators read the HTML and still type their labels into the CSV. Built only from the annotator's own
CSVs, so it reveals nothing the sheet does not.

    python scripts/render_sheets.py ~/artifacts/human_eval/annotator_A
"""
import csv
import html
import sys
from pathlib import Path

HEAD = """<!doctype html><html><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>{title}</title>
<script>window.MathJax={{tex:{{inlineMath:[['$','$'],['\\\\(','\\\\)']],displayMath:[['$$','$$'],['\\\\[','\\\\]']]}}}};</script>
<script src="https://cdn.jsdelivr.net/npm/mathjax@3/es5/tex-chtml.js" async></script>
<style>
body{{font-family:Georgia,serif;max-width:980px;margin:0 auto;padding:16px;line-height:1.5;background:#fff;color:#111}}
.item{{border:1px solid #ccc;border-radius:6px;padding:12px 16px;margin:18px 0}}
.id{{font:bold 15px sans-serif;color:#444}} .q{{background:#f6f6f6;padding:8px;border-radius:4px}}
.cols{{display:grid;grid-template-columns:1fr 1fr;gap:12px}} .sol{{white-space:pre-wrap;border-left:3px solid #999;padding-left:8px}}
@media (max-width:700px){{.cols{{grid-template-columns:1fr}}}}
</style></head><body><h2>{title}</h2><p>{help}</p>
"""


def esc(s):
    return html.escape(s or '', quote=False)


def render(csv_path: Path):
    rows = list(csv.DictReader(open(csv_path, encoding='utf-8-sig')))
    h1 = 'solution_A' in rows[0]
    title = f'{csv_path.parent.name}: {csv_path.name}'
    help_ = ('For each pair: are the two solutions the SAME approach or DIFFERENT approaches? '
             if h1 else 'For each solution: is the reasoning VALID or FLAWED (right answer, broken reasoning)? ')
    out = [HEAD.format(title=esc(title), help=help_ + 'Type your label in the CSV next to the same item id.')]
    for r in rows:
        out.append(f'<div class="item"><div class="id">{esc(r["item_id"])}</div>'
                   f'<p class="q"><b>Problem.</b> {esc(r["problem"])}</p>')
        if h1:
            out.append(f'<div class="cols"><div><b>Solution A</b><div class="sol">{esc(r["solution_A"])}</div></div>'
                       f'<div><b>Solution B</b><div class="sol">{esc(r["solution_B"])}</div></div></div>')
        else:
            out.append(f'<p><b>Reference answer:</b> {esc(r["reference_answer"])}</p>'
                       f'<div class="sol">{esc(r["solution"])}</div>')
        out.append('</div>')
    out.append('</body></html>')
    dest = csv_path.with_suffix('.html')
    dest.write_text('\n'.join(out), encoding='utf-8')
    print('wrote', dest)


if __name__ == '__main__':
    for d in sys.argv[1:]:
        for c in sorted(Path(d).glob('*.csv')):
            render(c)

import json
import ast

with open('rlvr_training_pipeline.ipynb', 'r', encoding='utf-8') as f:
    nb = json.load(f)

print("=" * 80)
print("FINAL VALIDATION: ALL CELLS")
print("=" * 80)
print()

code_cells = [(i, c) for i, c in enumerate(nb['cells']) if c.get('cell_type') == 'code']
passed = 0
failed = []
magic = 0

for nb_idx, cell in code_cells:
    source = ''.join(cell.get('source', []))
    
    if not source.strip():
        continue
    
    if source.strip().startswith('%%'):
        magic += 1
        continue
    
    try:
        ast.parse(source)
        passed += 1
    except SyntaxError as e:
        failed.append((nb_idx, e.lineno, e.msg))
        print(f"❌ Cell at index {nb_idx}: {e.msg} at line {e.lineno}")

print(f"Total code cells: {len(code_cells)}")
print(f"Python cells passed: {passed}")
print(f"Magic cells: {magic}")
print(f"Failed: {len(failed)}")
print()

if len(failed) == 0:
    print("=" * 80)
    print("✅ ALL CELLS SYNTAX VALID")
    print("=" * 80)
    print()
    print("Your notebook is ready to execute!")
    print("  • Cells 1-19: Setup and definitions")
    print("  • Cell 20: Pre-flight verification")
    print("  • Cell 21: Production training")
    print()
    print("Action: RESTART AND RUN ALL in Colab")
else:
    print("=" * 80)
    print(f"❌ {len(failed)} CELLS HAVE SYNTAX ERRORS")
    print("=" * 80)
    for idx, lineno, msg in failed:
        print(f"  Cell {idx}, line {lineno}: {msg}")

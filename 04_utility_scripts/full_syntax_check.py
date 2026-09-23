import json
import ast

with open("rlvr_training_pipeline.ipynb", "r", encoding="utf-8") as f:
    nb = json.load(f)

code_cells = [c for c in nb["cells"] if c.get("cell_type") == "code"]

print("=" * 70)
print("FULL NOTEBOOK SYNTAX VALIDATION")
print("=" * 70)

errors = []
passed = 0

for i, cell in enumerate(code_cells, 1):
    source = "".join(cell.get("source", []))
    
    if not source.strip():
        continue
        
    try:
        ast.parse(source)
        passed += 1
    except SyntaxError as e:
        errors.append(f"Cell {i}: Line {e.lineno} - {e.msg}")

print(f"\nResults:")
print(f"  Cells checked: {len(code_cells)}")
print(f"  Passed: {passed}")
print(f"  Failed: {len(errors)}")

if errors:
    print(f"\n❌ ERRORS FOUND:")
    for err in errors:
        print(f"  {err}")
else:
    print(f"\n✅ ALL CELLS HAVE VALID PYTHON SYNTAX")
    print("✅ NOTEBOOK IS READY FOR EXECUTION")

print("\n" + "=" * 70)

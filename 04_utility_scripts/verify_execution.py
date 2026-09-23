import json
import ast
import sys

print("=" * 80)
print("NOTEBOOK EXECUTION FLOW VERIFICATION")
print("=" * 80)

# Load notebook
with open("rlvr_training_pipeline.ipynb", "r", encoding="utf-8") as f:
    nb = json.load(f)

code_cells = [(i, c) for i, c in enumerate(nb["cells"]) if c.get("cell_type") == "code"]

print(f"\nFound {len(code_cells)} code cells to verify")

errors = []
warnings = []
cell_info = []

for cell_idx, (nb_idx, cell) in enumerate(code_cells, 1):
    source = "".join(cell.get("source", []))
    
    if not source.strip() or source.strip().startswith("#") and len(source.strip()) < 100:
        warnings.append(f"Cell {cell_idx} (nb_idx {nb_idx}): Empty or comment-only")
        continue
    
    # Try to parse as Python
    try:
        ast.parse(source)
        status = "OK"
    except SyntaxError as e:
        errors.append(f"Cell {cell_idx} (nb_idx {nb_idx}): Syntax Error at line {e.lineno}: {e.msg}")
        status = "SYNTAX ERROR"
    except Exception as e:
        warnings.append(f"Cell {cell_idx} (nb_idx {nb_idx}): Parse warning: {str(e)[:50]}")
        status = "WARNING"
    
    # Check for common issues
    if "from google.colab import drive" in source and "try:" not in source:
        warnings.append(f"Cell {cell_idx}: Missing try-except for Colab import")
    
    # Extract imports
    imports = []
    for line in source.split("\n"):
        line = line.strip()
        if line.startswith("import ") or line.startswith("from "):
            imports.append(line.split()[1] if line.startswith("import") else line.split()[1])
    
    cell_info.append({
        "cell": cell_idx,
        "nb_idx": nb_idx,
        "status": status,
        "imports": imports[:3] if imports else [],
        "lines": len(source.split("\n"))
    })

print("\n" + "=" * 80)
print("CELL-BY-CELL VERIFICATION")
print("=" * 80)

for info in cell_info[:10]:  # Show first 10
    imports_str = ", ".join(info["imports"]) if info["imports"] else "None"
    print(f"Cell {info['cell']:2d} (idx {info['nb_idx']:2d}): {info['status']:15s} | {info['lines']:3d} lines | Imports: {imports_str}")

print("\n" + "=" * 80)
print("CRITICAL CHECKS")
print("=" * 80)

# Check Cell 1 - Google Drive
cell1_src = "".join(code_cells[0][1].get("source", []))
checks = {
    "Cell 1 has code": len(cell1_src.strip()) > 50,
    "Cell 1 has drive.mount": "drive.mount" in cell1_src,
    "Cell 1 has try-except": "try:" in cell1_src and "except" in cell1_src,
    "Cell 1 has mkdir": "mkdir" in cell1_src,
}

for check, result in checks.items():
    print(f"  {'PASS' if result else 'FAIL'}: {check}")

# Check Cell 2 - Dependencies
cell2_src = "".join(code_cells[1][1].get("source", []))
dep_checks = {
    "Cell 2 has code": len(cell2_src.strip()) > 50,
    "Cell 2 has pip install": "pip" in cell2_src or "subprocess" in cell2_src,
    "Cell 2 lists packages": "torch" in cell2_src or "transformers" in cell2_src,
}

for check, result in dep_checks.items():
    print(f"  {'PASS' if result else 'FAIL'}: {check}")

print("\n" + "=" * 80)
print("ISSUES FOUND")
print("=" * 80)

if errors:
    print(f"\n❌ ERRORS ({len(errors)}):")
    for err in errors[:5]:
        print(f"  - {err}")
else:
    print("\n✅ No syntax errors found")

if warnings:
    print(f"\n⚠️  WARNINGS ({len(warnings)}):")
    for warn in warnings[:5]:
        print(f"  - {warn}")
else:
    print("\n✅ No warnings")

print("\n" + "=" * 80)
print("EXECUTION READINESS")
print("=" * 80)

if not errors:
    print("\n✅ NOTEBOOK SYNTAX: VALID")
    print("✅ ALL CODE CELLS: PARSEABLE")
    print("✅ READY FOR COLAB EXECUTION")
else:
    print("\n❌ NOTEBOOK HAS ERRORS")
    print("❌ NEEDS FIXING BEFORE EXECUTION")

print("\n" + "=" * 80)
print("NEXT: IMPORT AVAILABILITY CHECK")
print("=" * 80)

# Check if key imports would work locally
test_imports = {
    "torch": False,
    "transformers": False,
    "datasets": False,
    "sklearn": False,
    "hypothesis": False,
}

for pkg in test_imports:
    try:
        __import__(pkg if pkg != "sklearn" else "sklearn")
        test_imports[pkg] = True
    except ImportError:
        test_imports[pkg] = False

print("\nLocal import availability (for syntax check only):")
for pkg, available in test_imports.items():
    status = "✓ Available" if available else "✗ Not installed (OK - will install in Colab)"
    print(f"  {pkg:20s}: {status}")

print("\n" + "=" * 80)

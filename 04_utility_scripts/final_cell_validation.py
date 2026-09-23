import json
import ast
import re

print("=" * 80)
print("COMPREHENSIVE CELL-BY-CELL VALIDATION")
print("=" * 80)

with open("rlvr_training_pipeline.ipynb", "r", encoding="utf-8") as f:
    nb = json.load(f)

all_cells = nb["cells"]
code_cells = [(i, c) for i, c in enumerate(all_cells) if c.get("cell_type") == "code"]

print(f"\nTotal cells: {len(all_cells)}")
print(f"Code cells: {len(code_cells)}")
print(f"Markdown cells: {len(all_cells) - len(code_cells)}")

print("\n" + "=" * 80)
print("DETAILED CELL-BY-CELL CHECK")
print("=" * 80)

results = []
critical_checks = {
    "drive_mount": False,
    "pip_install": False,
    "gpu_check": False,
    "config_class": False,
    "dataset_load": False,
    "clustering": False,
    "model_load": False,
    "xml_parser": False,
    "gates": False,
    "pass_at_k": False,
    "checkpoint": False,
}

for cell_num, (nb_idx, cell) in enumerate(code_cells, 1):
    source = "".join(cell.get("source", []))
    
    result = {
        "cell_num": cell_num,
        "nb_idx": nb_idx,
        "lines": len(source.split("\n")),
        "chars": len(source),
        "status": "UNKNOWN",
        "issues": [],
        "imports": [],
        "key_features": []
    }
    
    # Skip empty cells
    if not source.strip():
        result["status"] = "EMPTY"
        results.append(result)
        continue
    
    # Check for Jupyter magic
    if source.strip().startswith("%%"):
        result["status"] = "MAGIC"
        result["key_features"].append("Jupyter magic command")
        results.append(result)
        continue
    
    # Syntax check
    try:
        ast.parse(source)
        result["status"] = "PASS"
    except SyntaxError as e:
        result["status"] = "FAIL"
        result["issues"].append(f"Syntax error at line {e.lineno}: {e.msg}")
        results.append(result)
        continue
    
    # Extract imports
    for line in source.split("\n"):
        line_stripped = line.strip()
        if line_stripped.startswith("import ") or line_stripped.startswith("from "):
            pkg = line_stripped.split()[1].split(".")[0]
            if pkg not in result["imports"]:
                result["imports"].append(pkg)
    
    # Check for key functionality
    if "drive.mount" in source:
        result["key_features"].append("Google Drive mounting")
        critical_checks["drive_mount"] = True
    
    if "pip" in source and ("install" in source or "subprocess" in source):
        result["key_features"].append("Package installation")
        critical_checks["pip_install"] = True
    
    if "torch.cuda" in source or "cuda.is_available" in source:
        result["key_features"].append("GPU verification")
        critical_checks["gpu_check"] = True
    
    if "ExperimentConfig" in source and "class" in source:
        result["key_features"].append("Configuration system")
        critical_checks["config_class"] = True
    
    if "load_dataset" in source or "openai/gsm8k" in source:
        result["key_features"].append("Dataset loading")
        critical_checks["dataset_load"] = True
    
    if "KMeans" in source or "cluster" in source.lower():
        result["key_features"].append("Clustering")
        critical_checks["clustering"] = True
    
    if "AutoModelForCausalLM" in source or "BitsAndBytesConfig" in source:
        result["key_features"].append("Model loading")
        critical_checks["model_load"] = True
    
    if "parse" in source and "xml" in source.lower():
        result["key_features"].append("XML parser")
        critical_checks["xml_parser"] = True
    
    if "Gate" in source and ("class" in source or "def" in source):
        result["key_features"].append("Gate implementation")
        critical_checks["gates"] = True
    
    if "pass_at_k" in source.lower() or "Pass@k" in source:
        result["key_features"].append("Pass@k evaluation")
        critical_checks["pass_at_k"] = True
    
    if "checkpoint" in source.lower() and ("save" in source or "load" in source):
        result["key_features"].append("Checkpointing")
        critical_checks["checkpoint"] = True
    
    # Check for common issues
    if len(source) < 50:
        result["issues"].append("Very short cell (< 50 chars)")
    
    if "TODO" in source:
        result["issues"].append("Contains TODO marker")
    
    if source.count("def ") > 10:
        result["key_features"].append(f"{source.count('def ')} functions")
    
    if source.count("class ") > 3:
        result["key_features"].append(f"{source.count('class ')} classes")
    
    results.append(result)

# Print detailed results
for r in results:
    status_symbol = {
        "PASS": "PASS",
        "MAGIC": "MAGIC",
        "EMPTY": "EMPTY",
        "FAIL": "FAIL"
    }.get(r["status"], "?")
    
    features = ", ".join(r["key_features"][:2]) if r["key_features"] else "General code"
    print(f"Cell {r['cell_num']:2d} [{status_symbol:5s}] {r['lines']:4d} lines | {features}")
    
    if r["imports"]:
        print(f"         Imports: {', '.join(r['imports'][:5])}")
    
    if r["issues"]:
        for issue in r["issues"]:
            print(f"         WARNING: {issue}")

# Summary
print("\n" + "=" * 80)
print("VALIDATION SUMMARY")
print("=" * 80)

total = len(results)
passed = sum(1 for r in results if r["status"] == "PASS")
magic = sum(1 for r in results if r["status"] == "MAGIC")
empty = sum(1 for r in results if r["status"] == "EMPTY")
failed = sum(1 for r in results if r["status"] == "FAIL")

print(f"\nCell Status:")
print(f"  PASS:   {passed:2d}/{total}")
print(f"  MAGIC:  {magic:2d}/{total}")
print(f"  EMPTY:  {empty:2d}/{total}")
print(f"  FAIL:   {failed:2d}/{total}")

print(f"\nCritical Components:")
for component, present in critical_checks.items():
    status = "FOUND" if present else "MISSING"
    symbol = "OK" if present else "!!"
    print(f"  [{symbol}] {component:20s}: {status}")

all_critical_present = all(critical_checks.values())

print("\n" + "=" * 80)
if failed == 0 and all_critical_present:
    print("STATUS: ALL CHECKS PASSED")
    print("NOTEBOOK IS READY FOR EXECUTION")
elif failed > 0:
    print("STATUS: SYNTAX ERRORS FOUND")
    print("FIX REQUIRED BEFORE EXECUTION")
else:
    print("STATUS: MISSING CRITICAL COMPONENTS")
    print("REVIEW REQUIRED")
print("=" * 80)

# Export detailed report
with open("cell_validation_report.txt", "w", encoding="utf-8") as f:
    f.write("DETAILED CELL VALIDATION REPORT\n")
    f.write("=" * 80 + "\n\n")
    for r in results:
        f.write(f"Cell {r['cell_num']} (notebook index {r['nb_idx']}):\n")
        f.write(f"  Status: {r['status']}\n")
        f.write(f"  Lines: {r['lines']}\n")
        f.write(f"  Characters: {r['chars']}\n")
        if r['imports']:
            f.write(f"  Imports: {', '.join(r['imports'])}\n")
        if r['key_features']:
            f.write(f"  Features: {', '.join(r['key_features'])}\n")
        if r['issues']:
            f.write(f"  Issues: {', '.join(r['issues'])}\n")
        f.write("\n")

print("\nDetailed report saved to: cell_validation_report.txt")

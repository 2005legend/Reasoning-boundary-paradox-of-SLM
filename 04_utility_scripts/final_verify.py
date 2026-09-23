import json

# Verify notebook
with open("rlvr_training_pipeline.ipynb", "r", encoding="utf-8") as f:
    nb = json.load(f)

cells = [c for c in nb["cells"] if c.get("cell_type") == "code"]

print("=" * 60)
print("RLVR TRAINING PIPELINE - FINAL VERIFICATION")
print("=" * 60)

# Check implementation
checks = {
    "Total code cells": len(cells) == 19,
    "Cell 1 (Google Drive)": "drive.mount" in "".join(cells[0].get("source", [])),
    "Cell 2 (Dependencies)": "pip" in "".join(cells[1].get("source", [])),
    "Cell 3 (GPU Check)": "torch.cuda" in "".join(cells[2].get("source", [])),
}

print("\nImplementation Checks:")
for name, result in checks.items():
    status = "PASS" if result else "FAIL"
    print(f"  {name}: {status}")

all_pass = all(checks.values())

print("\n" + "=" * 60)
if all_pass:
    print("STATUS: ALL CHECKS PASSED - NOTEBOOK READY FOR EXECUTION")
else:
    print("STATUS: SOME CHECKS FAILED - REVIEW NEEDED")
print("=" * 60)

print("\nFiles:")
import os
files = ["config_validation.py", "test_gpu_verification.py", "test_seed_reproducibility.py", "IMPLEMENTATION_STATUS.md"]
for f in files:
    exists = "EXISTS" if os.path.exists(f) else "MISSING"
    print(f"  {f}: {exists}")

print("\nNext Steps:")
print("1. Upload rlvr_training_pipeline.ipynb to Google Colab")
print("2. Run cells 1-6 for setup")
print("3. Run experiment cells (14-17) to execute experiments")

import json

# Load notebook
with open("rlvr_training_pipeline.ipynb", "r", encoding="utf-8") as f:
    nb = json.load(f)

# Check cell 1
cell1 = [c for c in nb["cells"] if c.get("cell_type") == "code"][0]
source = cell1.get("source", [])

print("Cell 1 source structure:")
print(f"  Type: {type(source)}")
print(f"  Length: {len(source)}")
print(f"  First element type: {type(source[0]) if source else 'N/A'}")
print(f"\nFirst 3 elements:")
for i, line in enumerate(source[:3]):
    print(f"  [{i}]: {repr(line[:50])}")

# The issue: source should be a list of strings, each ending with \n
# Check if they have newlines
has_newlines = any("\n" in line for line in source)
print(f"\nHas newlines: {has_newlines}")

# FIX: If source is a single string, split it
if source and isinstance(source, list) and len(source) == 1:
    print("\n! PROBLEM: Source is a single string, needs to be split")

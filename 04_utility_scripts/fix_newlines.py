import json

with open("rlvr_training_pipeline.ipynb", "r", encoding="utf-8") as f:
    nb = json.load(f)

code_cells_indices = [i for i, c in enumerate(nb["cells"]) if c.get("cell_type") == "code"]

# Fix cells 1 and 2 (indices 2 and 4 in notebook)
for cell_idx in code_cells_indices[:2]:  # First 2 code cells
    source = nb["cells"][cell_idx].get("source", [])
    if source and not any(line.endswith("\n") for line in source[:-1]):
        # Add newlines to all but last line
        nb["cells"][cell_idx]["source"] = [
            line + "\n" if i < len(source) - 1 else line
            for i, line in enumerate(source)
        ]
        print(f"Fixed cell at index {cell_idx}")

with open("rlvr_training_pipeline.ipynb", "w", encoding="utf-8") as f:
    json.dump(nb, f, indent=1, ensure_ascii=False)

print("Notebook fixed - newlines added")

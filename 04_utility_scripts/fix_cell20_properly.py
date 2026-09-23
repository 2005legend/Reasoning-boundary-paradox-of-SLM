import json

with open('rlvr_training_pipeline.ipynb', 'r', encoding='utf-8') as f:
    nb = json.load(f)

# Find Cell 20
cell_20_idx = None
for i, cell in enumerate(nb['cells']):
    if cell.get('cell_type') == 'code':
        source = ''.join(cell.get('source', []))
        if 'PRE-FLIGHT VERIFICATION SYSTEM' in source:
            cell_20_idx = i
            break

if not cell_20_idx:
    print("❌ Could not find Cell 20")
    exit(1)

print(f"Found Cell 20 at index {cell_20_idx}")

# Get current source
source_lines = nb['cells'][cell_20_idx]['source']

# Find the problematic section and rebuild properly
new_lines = []
skip_next = 0

for i, line in enumerate(source_lines):
    if skip_next > 0:
        skip_next -= 1
        continue
    
    # Find the warnings section and replace it completely
    if 'if results[\'warnings\']:' in line and i < len(source_lines) - 4:
        # Replace the entire warnings block
        new_lines.append('    if results["warnings"]:\n')
        new_lines.append('        print()\n')
        new_lines.append('        print(f"⚠️  Warnings: {len(results[\'warnings\'])}")\n')
        new_lines.append('        print()\n')
        new_lines.append('        for i, warning in enumerate(results["warnings"], 1):\n')
        new_lines.append('            print(f"  {i}. {warning}")\n')
        # Skip the old broken lines
        skip_next = 5
    else:
        new_lines.append(line)

# Update cell
nb['cells'][cell_20_idx]['source'] = new_lines

# Save
with open('rlvr_training_pipeline.ipynb', 'w', encoding='utf-8') as f:
    json.dump(nb, f, indent=1)

print("✅ Cell 20 properly fixed")
print("   Replaced entire warnings section with clean code")

import json

with open('rlvr_training_pipeline.ipynb', 'r', encoding='utf-8') as f:
    nb = json.load(f)

# Find Cell 20
cell_20_code_idx = None
for i, cell in enumerate(nb['cells']):
    if cell.get('cell_type') == 'code':
        source = ''.join(cell.get('source', []))
        if 'PRE-FLIGHT VERIFICATION SYSTEM' in source:
            cell_20_code_idx = i
            break

if cell_20_code_idx:
    print(f'Found Cell 20 at index {cell_20_code_idx}')
    
    # Get the source
    source_lines = nb['cells'][cell_20_code_idx]['source']
    
    # Find and fix the problematic line (around line 255)
    fixed_lines = []
    for line in source_lines:
        # Fix the specific line with unclosed string
        if 'Warnings ({}):' in line and line.strip().startswith('print'):
            # Replace with proper f-string
            fixed_line = '    if results[\'warnings\']:\n'
            fixed_lines.append(fixed_line)
            fixed_lines.append('        print()\n')
            fixed_lines.append('        print(f"⚠️  Warnings ({len(results[\'warnings\'])}):")\n')
            fixed_lines.append('        print()\n')
        else:
            fixed_lines.append(line)
    
    # Update the cell
    nb['cells'][cell_20_code_idx]['source'] = fixed_lines
    
    # Save
    with open('rlvr_training_pipeline.ipynb', 'w', encoding='utf-8') as f:
        json.dump(nb, f, indent=1)
    
    print('✅ Fixed syntax error in Cell 20')
    print('   Replaced problematic line with proper f-string')
else:
    print('❌ Could not find Cell 20')

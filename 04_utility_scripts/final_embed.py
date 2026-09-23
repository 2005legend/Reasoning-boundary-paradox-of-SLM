import json

# Load the production training cell
with open('CELL_21_PRODUCTION_TRAINING.py', 'r', encoding='utf-8') as f:
    training_code = f.read()

# Load notebook
with open('rlvr_training_pipeline.ipynb', 'r', encoding='utf-8') as f:
    nb = json.load(f)

# Add markdown for Cell 21
markdown_21 = {
    'cell_type': 'markdown',
    'metadata': {},
    'source': [
        '## Cell 21: Production Training Execution\n',
        '\n',
        '**Full TRL GRPO implementation with production-grade error handling.**\n',
        '\n',
        'This cell implements:\n',
        '- Complete training pipeline with TRL GRPOTrainer\n',
        '- Custom reward function integration (XML parsing + correctness)\n',
        '- Automatic checkpoint management and resumption\n',
        '- Production logging and metrics\n',
        '- Memory management and cleanup\n',
        '\n',
        '**Select experiment and run to start training.**'
    ]
}

# Add code cell
code_21 = {
    'cell_type': 'code',
    'execution_count': None,
    'metadata': {},
    'outputs': [],
    'source': [line + '\n' if i < len(training_code.split('\n')) - 1 else line 
               for i, line in enumerate(training_code.split('\n'))]
}

nb['cells'].append(markdown_21)
nb['cells'].append(code_21)

# Save
with open('rlvr_training_pipeline.ipynb', 'w', encoding='utf-8') as f:
    json.dump(nb, f, indent=1)

print('=' * 80)
print('✅ PRODUCTION NOTEBOOK COMPLETE')
print('=' * 80)
print(f'Total cells: {len(nb["cells"])}')
print(f'Training code: {len(training_code.split(chr(10)))} lines')
print()
print('Final structure:')
print('  Cells 1-19: Setup and function definitions')
print('  Cell 20: Pre-flight verification system')
print('  Cell 21: Production training execution')
print()
print('=' * 80)
print('STATUS: PRODUCTION-READY FOR RESEARCH EXECUTION')
print('=' * 80)
print()
print('Next steps:')
print('  1. Upload to Google Colab')
print('  2. Select GPU runtime')
print('  3. Run cells 1-19')
print('  4. Run cell 20 (pre-flight)')
print('  5. Run cell 21 (training)')

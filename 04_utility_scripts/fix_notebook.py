import json

# Read the notebook
with open('rlvr_training_pipeline.ipynb', 'r', encoding='utf-8') as f:
    nb = json.load(f)

# Cell 1: Google Drive Mounting
cell1_code = '''# Task 1.1: Google Drive Mounting and Directory Structure Creation
import os
from pathlib import Path

print('=' * 60)
print('MOUNTING GOOGLE DRIVE')
print('=' * 60)

try:
    from google.colab import drive
    drive.mount('/content/drive')
    print('\\nGoogle Drive mounted successfully')
    
    base_dir = Path('/content/drive/MyDrive/RLVR_Research')
    directories = {
        'checkpoints': base_dir / 'checkpoints',
        'results': base_dir / 'results',
        'figures': base_dir / 'figures',
        'logs': base_dir / 'logs'
    }
    
    print('\\nCreating directory structure...')
    for name, path in directories.items():
        path.mkdir(parents=True, exist_ok=True)
        print(f'Created: {path}')
    
    test_file = base_dir / 'test.txt'
    test_file.write_text('test')
    test_file.unlink()
    print('\\nWrite permissions verified')
    print('=' * 60)
    
except ImportError:
    print('Not in Colab, using local directories')
    base_dir = Path('./RLVR_Research')
    directories = {
        'checkpoints': base_dir / 'checkpoints',
        'results': base_dir / 'results',
        'figures': base_dir / 'figures',
        'logs': base_dir / 'logs'
    }
    for name, path in directories.items():
        path.mkdir(parents=True, exist_ok=True)
'''

# Cell 2: Dependencies
cell2_code = '''# Task 1.2: Dependency Installation
import sys
import subprocess

print('=' * 60)
print('INSTALLING DEPENDENCIES')
print('=' * 60)

packages = [
    'torch', 'transformers', 'trl', 'peft', 'bitsandbytes',
    'datasets', 'sentence-transformers', 'scikit-learn',
    'sympy', 'plotly', 'matplotlib', 'hypothesis'
]

for pkg in packages:
    print(f'Installing {pkg}...')
    subprocess.run([sys.executable, '-m', 'pip', 'install', '-q', pkg], check=False)

print('\\nDependencies installed')
print('=' * 60)
'''

# Update the cells
nb['cells'][1]['source'] = [cell1_code]
nb['cells'][3]['source'] = [cell2_code]

# Save
with open('rlvr_training_pipeline_fixed.ipynb', 'w', encoding='utf-8') as f:
    json.dump(nb, f, indent=1)

print('Notebook fixed and saved as rlvr_training_pipeline_fixed.ipynb')

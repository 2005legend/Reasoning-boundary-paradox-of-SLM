import json

with open("rlvr_training_pipeline.ipynb", "r", encoding="utf-8") as f:
    nb = json.load(f)

cell1_source = """import os
from pathlib import Path

print("=" * 60)
print("TASK 1.1: GOOGLE DRIVE MOUNTING")
print("=" * 60)

try:
    from google.colab import drive
    drive.mount("/content/drive")
    print("\\nGoogle Drive mounted successfully!")
    
    base_dir = Path("/content/drive/MyDrive/RLVR_Research")
    dirs = ["checkpoints", "results", "figures", "logs"]
    
    print("\\nCreating directory structure...")
    for d in dirs:
        (base_dir / d).mkdir(parents=True, exist_ok=True)
        print(f"  Created: {base_dir / d}")
    
    test_file = base_dir / ".test_write"
    test_file.write_text("OK")
    test_file.unlink()
    print("\\nWrite permissions verified!")
    print("=" * 60)
    
except ImportError:
    print("\\nNot in Colab - using local directories")
    base_dir = Path("./RLVR_Research")
    for d in ["checkpoints", "results", "figures", "logs"]:
        (base_dir / d).mkdir(parents=True, exist_ok=True)
    print("Local setup complete")
"""

cell2_source = """import subprocess
import sys

print("=" * 60)
print("TASK 1.2: INSTALLING DEPENDENCIES")
print("=" * 60)

packages = [
    "torch", "transformers", "trl", "peft", "bitsandbytes",
    "accelerate", "datasets", "sentence-transformers", 
    "scikit-learn", "sympy", "plotly", "matplotlib", "hypothesis"
]

print("\\nInstalling packages (this may take a few minutes)...")
for pkg in packages:
    print(f"  Installing {pkg}...")
    result = subprocess.run(
        [sys.executable, "-m", "pip", "install", "-q", pkg],
        capture_output=True
    )
    if result.returncode == 0:
        print(f"    OK")
    else:
        print(f"    Warning: {pkg} had issues")

print("\\nDependencies installed!")
print("=" * 60)
"""

# Update code cells (indices 2 and 4)
nb["cells"][2]["source"] = cell1_source.split("\n")
nb["cells"][4]["source"] = cell2_source.split("\n")

with open("rlvr_training_pipeline.ipynb", "w", encoding="utf-8") as f:
    json.dump(nb, f, indent=1, ensure_ascii=False)

print("Notebook fixed successfully!")
import json

with open("rlvr_training_pipeline.ipynb", "r", encoding="utf-8") as f:
    nb = json.load(f)

cell1 = """import os
from pathlib import Path
from google.colab import drive
drive.mount(\"/content/drive\")
base_dir = Path(\"/content/drive/MyDrive/RLVR_Research\")
for d in [\"checkpoints\", \"results\", \"figures\", \"logs\"]:
    (base_dir / d).mkdir(parents=True, exist_ok=True)
print(\"Setup complete\")"""

cell2 = """import subprocess, sys
pkgs = [\"torch\", \"transformers\", \"trl\", \"peft\", \"datasets\"]
for p in pkgs: subprocess.run([sys.executable, \"-m\", \"pip\", \"install\", \"-q\", p])
print(\"Dependencies installed\")"""

nb["cells"][1]["source"] = [cell1]
nb["cells"][3]["source"] = [cell2]

with open("rlvr_training_pipeline.ipynb", "w", encoding="utf-8") as f:
    json.dump(nb, f, indent=1)

print("Fixed")

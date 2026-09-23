# RLVR Training - Shared Notebook Template

## BEFORE RUNNING:
1. Check experiment tracker: https://docs.google.com/spreadsheets/d/YOUR_SHEET_ID
2. Claim your experiment (change status to "🔄 RUNNING")
3. Fill in YOUR_NAME and START_TIME

---

# ============================================================================
# CELL 1: CONFIGURATION - EDIT THIS CELL ONLY
# ============================================================================

import torch
from pathlib import Path
from datetime import datetime

# ============================================================================
# EXPERIMENT CONFIGURATION - EDIT THESE 3 LINES ONLY
# ============================================================================

EXPERIMENT_NAME = "exp2_n1_vanilla_0.5B"  # ← Change this
MODEL_SIZE = "0.5B"                        # ← "0.5B" or "1.5B"
ALGORITHM = "vanilla"                      # ← "vanilla" or "cbgrpo"

# ============================================================================
# DO NOT EDIT BELOW THIS LINE
# ============================================================================

# Fixed configuration (same for all experiments)
RANDOM_SEED = 42
USE_SMOKE_TIER = False  # Set True for quick validation

# Metadata
YOUR_NAME = "YOUR_NAME_HERE"  # ← Put your name
START_TIME = datetime.now().strftime('%Y-%m-%d %H:%M:%S')

print("=" * 80)
print("🎯 EXPERIMENT CONFIGURATION")
print("=" * 80)
print(f"Experiment: {EXPERIMENT_NAME}")
print(f"Model: {MODEL_SIZE}")
print(f"Algorithm: {ALGORITHM}")
print(f"Seed: {RANDOM_SEED}")
print(f"Runner: {YOUR_NAME}")
print(f"Start: {START_TIME}")
print("=" * 80)
print()

# Verify GPU
if not torch.cuda.is_available():
    raise RuntimeError("❌ No GPU! Change runtime: Runtime → Change runtime type → T4 GPU")

print(f"✅ GPU Available: {torch.cuda.get_device_name(0)}")
print(f"✅ VRAM: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
print()

# ============================================================================
# CELL 2: MOUNT DRIVE & SETUP
# ============================================================================

print("⏳ Mounting Google Drive...")

from google.colab import drive
drive.mount("/content/drive", force_remount=False)

# Create output directories
base_dir = Path("/content/drive/MyDrive/RLVR_Research")
checkpoint_dir = base_dir / "checkpoints" / EXPERIMENT_NAME
results_dir = base_dir / "results" / EXPERIMENT_NAME

for d in [checkpoint_dir, results_dir]:
    d.mkdir(parents=True, exist_ok=True)

print(f"✅ Drive mounted")
print(f"✅ Checkpoints: {checkpoint_dir}")
print(f"✅ Results: {results_dir}")
print()

# ============================================================================
# CELL 3: RUN TRAINING (DO NOT EDIT)
# ============================================================================

# All training logic is in the main notebook
# This cell just verifies configuration

print("=" * 80)
print("✅ CONFIGURATION VERIFIED")
print("=" * 80)
print()
print("Next steps:")
print("1. Run all cells below")
print("2. Training will auto-start")
print("3. Checkpoints save to Drive automatically")
print("4. Update tracker when done: ✅ COMPLETE")
print()
print("If anything breaks:")
print("1. Stop immediately")
print("2. Update tracker: ❌ FAILED + error message")
print("3. Contact experiment owner")
print()
print("=" * 80)

# ============================================================================
# AFTER TRAINING COMPLETE:
# ============================================================================
# 1. Update tracker sheet:
#    - Status: ✅ COMPLETE
#    - End time: YYYY-MM-DD HH:MM
#    - Final loss: X.XXX
#    - Checkpoint path: /content/drive/MyDrive/RLVR_Research/checkpoints/...
#
# 2. Verify checkpoint saved:
#    - Check Drive folder
#    - Verify final_model/ exists
#    - Note file sizes
#
# 3. DO NOT run evaluation (centralized eval only)
# ============================================================================

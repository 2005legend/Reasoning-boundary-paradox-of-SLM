"""
Pre-flight checklist for actual RLVR training execution.
Run this in Colab BEFORE starting training to verify environment.
"""

import sys
import subprocess

print("=" * 80)
print("🚀 PRE-FLIGHT CHECKLIST FOR REAL TRAINING")
print("=" * 80)
print()

checks_passed = 0
checks_total = 8

# 1. Check if running in Colab
print("1. Environment Check...")
try:
    import google.colab
    print("   ✅ Running in Google Colab")
    checks_passed += 1
except ImportError:
    print("   ⚠️  Not in Colab - will use local environment")
    checks_passed += 1

# 2. Check GPU availability
print("\n2. GPU Check...")
try:
    import torch
    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)
        vram_gb = torch.cuda.get_device_properties(0).total_memory / (1024**3)
        print(f"   ✅ GPU Available: {gpu_name}")
        print(f"   ✅ VRAM: {vram_gb:.1f} GB")
        if vram_gb >= 14:
            print(f"   ✅ VRAM sufficient for 1.5B model")
            checks_passed += 1
        else:
            print(f"   ⚠️  VRAM < 14GB - use 0.5B model or reduce batch size")
            checks_passed += 1
    else:
        print("   ❌ NO GPU DETECTED")
        print("   → Go to Runtime > Change runtime type > GPU")
except Exception as e:
    print(f"   ❌ GPU check failed: {e}")

# 3. Check critical packages
print("\n3. Package Check...")
required_packages = [
    'torch', 'transformers', 'trl', 'peft', 'datasets',
    'sentence_transformers', 'scikit-learn', 'hypothesis'
]
missing = []
for pkg in required_packages:
    try:
        __import__(pkg.replace('-', '_'))
    except ImportError:
        missing.append(pkg)

if not missing:
    print(f"   ✅ All {len(required_packages)} required packages installed")
    checks_passed += 1
else:
    print(f"   ⚠️  Missing packages: {', '.join(missing)}")
    print(f"   → Run Cell 2 to install dependencies")

# 4. Check Google Drive mount
print("\n4. Google Drive Check...")
try:
    import os
    from pathlib import Path
    drive_path = Path("/content/drive/MyDrive/RLVR_Research")
    if drive_path.exists():
        print(f"   ✅ Drive mounted at {drive_path}")
        subdirs = ['checkpoints', 'results', 'figures', 'logs']
        all_exist = all((drive_path / d).exists() for d in subdirs)
        if all_exist:
            print(f"   ✅ All directories created")
            checks_passed += 1
        else:
            print(f"   ⚠️  Some directories missing - will be created")
            checks_passed += 1
    else:
        print(f"   ⚠️  Drive not mounted - using local storage")
        checks_passed += 1
except Exception as e:
    print(f"   ⚠️  Drive check skipped: {e}")
    checks_passed += 1

# 5. Check datasets loaded
print("\n5. Dataset Check...")
try:
    if 'gsm8k_train' in globals() and 'gsm8k_test' in globals():
        print(f"   ✅ GSM8K train: {gsm8k_train['count']} problems")
        print(f"   ✅ GSM8K test: {gsm8k_test['count']} problems")
        checks_passed += 1
    else:
        print(f"   ⚠️  Datasets not loaded - run Cell 5")
except Exception as e:
    print(f"   ⚠️  Run Cell 5 to load datasets")

# 6. Check cluster assignments
print("\n6. Clustering Check...")
try:
    if 'cluster_assignments' in globals():
        print(f"   ✅ Cluster assignments: {len(cluster_assignments)} prompts")
        print(f"   ✅ Clusters: {len(set(cluster_assignments.values()))}")
        checks_passed += 1
    else:
        print(f"   ⚠️  Cluster assignments not found - run Cell 7")
except Exception as e:
    print(f"   ⚠️  Run Cell 7 for clustering")

# 7. Check config system
print("\n7. Configuration Check...")
try:
    from dataclasses import is_dataclass
    if 'ExperimentConfig' in globals():
        all_configs = get_all_configs()
        print(f"   ✅ ExperimentConfig system loaded")
        print(f"   ✅ Available configs: {len(all_configs)}")
        checks_passed += 1
    else:
        print(f"   ⚠️  ExperimentConfig not loaded - run Cell 4")
except Exception as e:
    print(f"   ⚠️  Run Cell 4 for config system")

# 8. Check disk space
print("\n8. Disk Space Check...")
try:
    import shutil
    if Path("/content/drive/MyDrive").exists():
        stat = shutil.disk_usage("/content/drive/MyDrive")
        free_gb = stat.free / (1024**3)
        print(f"   ✅ Drive free space: {free_gb:.1f} GB")
        if free_gb >= 5:
            print(f"   ✅ Sufficient space for checkpoints")
            checks_passed += 1
        else:
            print(f"   ⚠️  Low disk space - clean up or use smaller checkpoints")
            checks_passed += 1
    else:
        stat = shutil.disk_usage("/content")
        free_gb = stat.free / (1024**3)
        print(f"   ✅ Local free space: {free_gb:.1f} GB")
        checks_passed += 1
except Exception as e:
    print(f"   ⚠️  Could not check disk space")
    checks_passed += 1

# Summary
print("\n" + "=" * 80)
print("CHECKLIST SUMMARY")
print("=" * 80)
print(f"Checks passed: {checks_passed}/{checks_total}")
print()

if checks_passed >= 6:
    print("✅ READY FOR TRAINING!")
    print()
    print("Recommended first run:")
    print("  1. Start with SMOKE TIER (100 steps, 30-60 min)")
    print("  2. Use 0.5B model first (less VRAM, faster)")
    print("  3. Monitor GPU usage and logs")
    print()
    print("To start training, run the EXECUTION CELL (Cell 20)")
elif checks_passed >= 4:
    print("⚠️  PARTIALLY READY - fix warnings above")
    print()
    print("Missing items can be resolved by:")
    print("  - Running all cells 1-19 in order")
    print("  - Checking Runtime > Change runtime type > GPU")
    print("  - Mounting Google Drive")
else:
    print("❌ NOT READY - resolve critical issues above")
    print()
    print("Please:")
    print("  1. Ensure GPU runtime is enabled")
    print("  2. Run all setup cells (1-19)")
    print("  3. Check error messages above")

print("=" * 80)
